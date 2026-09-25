#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.h"
#include "bwc_device_arena.hpp"
#include "bwc_dev_pieces.h"
#include "matmul_top_comm_plugin.hpp"
#include "select_mpi.h"

#include "comm_multimem.h"

/* A bwc communication plugin (comm_impl=nvls) that runs the two vector
 * exchange legs with NVLink SHARP. The multimem PTX instructions
 * reduce and broadcast in the switch (comm_multimem.cu): one kernel
 * per leg replaces the NCCL ring of comm_impl=nccl (bwccomm_nccl.cpp,
 * which stays as it is). The selection is explicit through comm_impl,
 * and the plugin never falls back to a ring: when a device has no
 * multicast or fabric handle support, or a multicast call fails, the
 * plugin reports the failed call and aborts. The plugin covers one
 * NVLink domain (one node, or a multi-node NVLink domain); ranks outside the domain make the group setup fail.
 *
 * Like the nccl plugin, the exchange runs on SPMV_GF2_NGPU lanes; each
 * lane is one local device and moves one slice of every block. Per
 * lane and communicator the plugin binds one multicast group over the
 * ranks. The local replica holds njobs block slots, one extra slot,
 * and the barrier counter, with two mappings: uc, the unicast address
 * of the replica, for the ordinary copies in and out, and mc, the
 * multicast address of the group, for the multimem kernels.
 *
 * The reduce-scatter stages the slice of every block into its slot;
 * one multimem.ld_reduce.xor kernel over slot rank then leaves the
 * XOR of that block across the ranks in the extra slot, and the
 * result goes out to block 0 of the buffer. The allgather stages the
 * own block into the extra slot; one multimem.st kernel writes it to
 * slot rank of every rank, and the slots go out to all blocks of the
 * buffer. Arrival barriers over the group (comm_multimem_barrier)
 * order the legs: one before the staging of either leg, so the
 * multimem reads and writes of the leg before it are complete on
 * every rank and the slots are free; one between the staging and the
 * kernel of the reduce-scatter, so every rank has staged what its
 * peers read; and one after the kernel of the allgather, so every
 * slot is current before the copies out. The allgather needs no
 * barrier between its staging and its kernel, because the kernel
 * reads the extra slot, which no other rank touches. The barrier
 * kernel carries the proxy fences that order the multimem accesses
 * against the ordinary accesses of the same memory.
 *
 * The arena side follows the nccl plugin without its overlap
 * machinery: the lane stream waits on the ready and band events of a
 * mirror it reads, stages from the output segments of the product or
 * from the mirror, and writes results back on the lane stream. The
 * reduce-scatter writes the mirror only, so its write counts as a
 * plain write and clears the segment record. The allgather writes
 * each block to the mirror and to every source replica of the next
 * product, marks the replicas current, and publishes one band per
 * block, so the product reads its replicas instead of a refresh from
 * the mirror (bwc_dev_pieces.h; a mirror without bands sends the
 * product down the whole-vector path, src_bands in spmv_gf2.cu).
 * One event on a publication stream becomes the new ready marker of a
 * written mirror. Host buffers stage through the same slots with host
 * copies and one host wait per call. */

#define CUDA_CHECK(call) do {                                           \
    cudaError_t const e = (call);                                       \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                cudaGetErrorString(e));                                 \
        abort();                                                        \
    }                                                                   \
} while (0)

#define CU_CHECK(call) do {                                             \
    CUresult const e = (call);                                          \
    if (e != CUDA_SUCCESS) {                                            \
        const char * s_ = nullptr;                                      \
        cuGetErrorString(e, &s_);                                       \
        fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,   \
                s_ ? s_ : "?");                                         \
        abort();                                                        \
    }                                                                   \
} while (0)

namespace {

/* Band publication for the allgather: one event per block. The store
 * and its events live for the process, which covers the arena
 * lifetime that the band contract asks for (bwc_device_arena.hpp).
 * Each arena has its own store, so a mirror of one interleaved
 * sequence keeps its bands while the other sequence exchanges. */
struct band_store {
    std::vector<cudaEvent_t> ev;
    std::vector<bwc_device_band> bands;
};

/* One lane: a device, its stream, and its multicast group over the
 * ranks of the communicator. */
struct lane {
    int device;
    cudaStream_t stream;
    cudaEvent_t done;           /* the lane completed a leg */
    cudaEvent_t rd;             /* the lane read output segments */
    std::vector<cudaEvent_t> blk;   /* one per block, the copies out */
    CUmemGenericAllocationHandle group, mem;
    CUdeviceptr uc, mc;         /* the replica; the group */
    size_t size = 0;            /* bound bytes */
    size_t asz;                 /* local allocation bytes */
    size_t slice_alloc = 0;
    size_t flag_off;            /* the barrier counter in the replica */
    unsigned long long gen = 0; /* barriers so far */
};

/* The lanes of one MPI communicator, made on first use. bwc uses one
 * communicator per grid direction, so the list stays small. */
struct comm_state {
    MPI_Comm mpi;
    std::vector<lane> lanes;
    cudaStream_t pub;           /* on the mirror device: the ready events */
    cudaEvent_t ev;             /* ready marker for the written mirrors */
    int rank;
    int njobs;
    std::map<bwc_device_arena *, band_store> stores;
};

/* The lane count and the first lane device follow the matmul backend
 * (env_ngpu and pick_first_device in spmv_gf2.cu), so that the lanes
 * are the devices that hold the matrix. */
int env_ngpu()
{
    const char * s = getenv("SPMV_GF2_NGPU");
    int n = s ? atoi(s) : 1;
    return n > 0 ? n : 1;
}

int pick_first_device(int ngpu, int ndev)
{
    const char * lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!lr) lr = getenv("SLURM_LOCALID");
    return (ngpu * (lr ? atoi(lr) : 0)) % ndev;
}

comm_state & get_state(MPI_Comm comm)
{
    static std::mutex mm;
    static std::vector<comm_state *> states;
    std::lock_guard<std::mutex> const dummy(mm);
    for (comm_state * s : states)
        if (s->mpi == comm)
            return *s;
    CU_CHECK(cuInit(0));
    auto * s = new comm_state;
    s->mpi = comm;
    MPI_Comm_rank(comm, &s->rank);
    MPI_Comm_size(comm, &s->njobs);
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int const ngpu = env_ngpu() < ndev ? env_ngpu() : ndev;
    int const first = pick_first_device(ngpu, ndev);
    s->lanes.resize(ngpu);
    for (int d = 0; d < ngpu; d++) {
        lane & l = s->lanes[d];
        l.device = (first + d) % ndev;
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaFree(nullptr));  /* make the context */
        CUdevice cd;
        CU_CHECK(cuDeviceGet(&cd, l.device));
        int mcast = 0, fabric = 0;
        CU_CHECK(cuDeviceGetAttribute(&mcast,
                    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cd));
        CU_CHECK(cuDeviceGetAttribute(&fabric,
                    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cd));
        if (!mcast || !fabric) {
            fprintf(stderr, "comm_impl=nvls: device %d reports"
                    " multicast=%d fabric=%d; the plugin needs both"
                    " and does not fall back to a ring\n",
                    l.device, mcast, fabric);
            abort();
        }
        CUDA_CHECK(cudaStreamCreate(&l.stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&l.done,
                    cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&l.rd,
                    cudaEventDisableTiming));
        for (int b = 0; b < s->njobs; b++) {
            cudaEvent_t bev;
            CUDA_CHECK(cudaEventCreateWithFlags(&bev,
                        cudaEventDisableTiming));
            l.blk.push_back(bev);
        }
    }
    CUDA_CHECK(cudaSetDevice(s->lanes[0].device));
    CUDA_CHECK(cudaStreamCreate(&s->pub));
    CUDA_CHECK(cudaEventCreateWithFlags(&s->ev, cudaEventDisableTiming));
    if (s->rank == 0)
        fprintf(stderr, "comm_impl=nvls: multimem exchange on %d jobs,"
                " %d lanes per job\n", s->njobs, (int) s->lanes.size());
    states.push_back(s);
    return *s;
}

band_store & store_of(comm_state & s, bwc_device_arena * a)
{
    band_store & bs = s.stores[a];
    CUDA_CHECK(cudaSetDevice(s.lanes[0].device));
    while ((int) bs.ev.size() < s.njobs) {
        cudaEvent_t bev;
        CUDA_CHECK(cudaEventCreateWithFlags(&bev, cudaEventDisableTiming));
        bs.ev.push_back(bev);
    }
    bs.bands.resize(s.njobs);
    return bs;
}

/* The slice of lane d inside a block: whole words, and the same size
 * on every lane but the last one. */
struct slice {
    size_t off;
    size_t bytes;
};

slice slice_of(comm_state const & s, int d, size_t block_bytes)
{
    size_t const words = block_bytes / sizeof(uint64_t);
    size_t const per = (words + s.lanes.size() - 1) / s.lanes.size();
    size_t const lo = std::min(words, per * d);
    size_t const hi = std::min(words, per * (d + 1));
    return { lo * sizeof(uint64_t), (hi - lo) * sizeof(uint64_t) };
}

/* The lanes that carry a slice of the blocks. */
std::vector<size_t> active_lanes(comm_state & s, size_t block_bytes)
{
    std::vector<size_t> ds;
    for (size_t d = 0; d < s.lanes.size(); d++)
        if (slice_of(s, d, block_bytes).bytes)
            ds.push_back(d);
    return ds;
}

/* Unmap and release the group and the replica of the lane. The caller
 * has made sure that no rank still reads the memory. */
void mc_release(lane & l)
{
    CUdevice cd;
    CU_CHECK(cuDeviceGet(&cd, l.device));
    CU_CHECK(cuMemUnmap(l.mc, l.size));
    CU_CHECK(cuMemAddressFree(l.mc, l.size));
    CU_CHECK(cuMemUnmap(l.uc, l.asz));
    CU_CHECK(cuMemAddressFree(l.uc, l.asz));
    CU_CHECK(cuMulticastUnbind(l.group, cd, 0, l.size));
    CU_CHECK(cuMemRelease(l.mem));
    CU_CHECK(cuMemRelease(l.group));
    l.size = 0;
}

/* Bind the multicast group of the lane for slices of slice_bytes. The
 * sizes are the same on every rank, so the ranks reach the collective
 * steps together. */
void reserve(comm_state & s, lane & l, size_t slice_bytes)
{
    if (l.slice_alloc >= slice_bytes)
        return;
    if (l.size) {
        /* no rank may still read the old memory through the group */
        CUDA_CHECK(cudaStreamSynchronize(l.stream));
        MPI_Barrier(s.mpi);
        mc_release(l);
    }
    l.slice_alloc = slice_bytes;
    l.flag_off = (size_t) (s.njobs + 1) * slice_bytes;
    l.gen = 0;

    CUmulticastObjectProp mp = {};
    mp.numDevices = s.njobs;
    mp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t gran = 0;
    CU_CHECK(cuMulticastGetGranularity(&gran, &mp,
                CU_MULTICAST_GRANULARITY_RECOMMENDED));
    l.size = (l.flag_off + sizeof(unsigned long long) + gran - 1)
        / gran * gran;
    mp.size = l.size;

    CUmemFabricHandle fh;
    if (s.rank == 0) {
        CU_CHECK(cuMulticastCreate(&l.group, &mp));
        CU_CHECK(cuMemExportToShareableHandle(&fh, l.group,
                    CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    MPI_Bcast(&fh, sizeof fh, MPI_BYTE, 0, s.mpi);
    if (s.rank != 0)
        CU_CHECK(cuMemImportFromShareableHandle(&l.group, &fh,
                    CU_MEM_HANDLE_TYPE_FABRIC));
    CUdevice cd;
    CU_CHECK(cuDeviceGet(&cd, l.device));
    CU_CHECK(cuMulticastAddDevice(l.group, cd));
    /* every device joined before any bind */
    MPI_Barrier(s.mpi);

    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = l.device;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t agran = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&agran, &ap,
                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    l.asz = (l.size + agran - 1) / agran * agran;
    CU_CHECK(cuMemCreate(&l.mem, l.asz, &ap, 0));
    CU_CHECK(cuMulticastBindMem(l.group, 0, l.mem, 0, l.size, 0));

    CUmemAccessDesc ad = {};
    ad.location = ap.location;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemAddressReserve(&l.uc, l.asz, 0, 0, 0));
    CU_CHECK(cuMemMap(l.uc, l.asz, 0, l.mem, 0));
    CU_CHECK(cuMemSetAccess(l.uc, l.asz, &ad, 1));
    CU_CHECK(cuMemAddressReserve(&l.mc, l.size, 0, 0, 0));
    CU_CHECK(cuMemMap(l.mc, l.size, 0, l.group, 0));
    CU_CHECK(cuMemSetAccess(l.mc, l.size, &ad, 1));

    /* each rank zeroes its replica of the barrier counter */
    CU_CHECK(cuMemsetD8(l.uc + l.flag_off, 0,
                sizeof(unsigned long long)));
    /* every replica bound and zeroed before any use of the group */
    MPI_Barrier(s.mpi);
}

/* One arrival barrier over the ranks, on the lane stream. The counter
 * only grows, so a barrier never waits for a value a later barrier
 * overwrites. */
void barrier(comm_state & s, lane & l)
{
    l.gen++;
    comm_multimem_barrier((unsigned long long *) (l.mc + l.flag_off),
            (const unsigned long long *) (l.uc + l.flag_off),
            l.gen * (unsigned long long) s.njobs, l.stream);
}

/* Device address of a buffer inside its arena mirror, when the mirror
 * holds the current data. */
struct dev_view {
    char * p = nullptr;
    int device = 0;
    bwc_device_arena * a = nullptr;
    size_t off = 0;             /* buffer offset inside the arena */
};

dev_view dev_of(void * buf)
{
    dev_view v;
    bwc_device_arena * const a = bwc_dev_find(buf);
    if (!a || a->state != BWC_ARENA_DEV)
        return v;
    v.off = (size_t) ((char *) buf - (char *) a->host);
    v.p = (char *) a->dev + v.off;
    v.device = a->device;
    v.a = a;
    return v;
}

/* Make the stream wait until the mirror range [off, off + bytes) is
 * complete. The band events cover the range when the product published
 * them; the ready event covers the whole mirror otherwise. */
void wait_range(cudaStream_t st, bwc_device_arena * a, size_t off,
        size_t bytes)
{
    size_t covered = 0;         /* the bands are contiguous from zero */
    for (int i = 0; i < a->nbands; i++) {
        bwc_device_band const & b = a->bands[i];
        if (b.off < off + bytes && off < b.off + b.bytes)
            CUDA_CHECK(cudaStreamWaitEvent(st, (cudaEvent_t) b.ready, 0));
        covered = b.off + b.bytes;
        if (covered >= off + bytes)
            return;
    }
    if (a->ready)
        CUDA_CHECK(cudaStreamWaitEvent(st, (cudaEvent_t) a->ready, 0));
}

bwc_dev_pieces * pieces_of(bwc_device_arena * a)
{
    auto * p = (bwc_dev_pieces *) a->owner;
    return p && p->magic == BWC_DEV_PIECES_MAGIC ? p : nullptr;
}

/* Copy the mirror range [off, off + bytes) of the arena to dst on the
 * lane device, behind the band or ready events of the mirror. */
void mirror_in(int device, cudaStream_t st, bwc_device_arena * a,
        size_t off, size_t bytes, uint8_t * dst)
{
    wait_range(st, a, off, bytes);
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, device, (char *) a->dev + off,
                a->device, bytes, st));
}

/* Copy the range [off, off + bytes) of the arena data to dst on the
 * lane device: each part that an output segment covers comes from the
 * device that holds the segment, the rest from the mirror. */
void arena_in(int device, cudaStream_t st, bwc_device_arena * a,
        size_t off, size_t bytes, uint8_t * dst)
{
    bwc_dev_pieces const * const p = pieces_of(a);
    size_t const end = off + bytes;
    size_t at = off;
    for (int i = 0; p && i < p->nout; i++) {
        bwc_dev_piece const & o = p->out[i];
        size_t const lo = std::max(o.off, at);
        size_t const hi = std::min(o.off + o.bytes, end);
        if (lo >= hi)
            continue;
        if (at < lo)
            mirror_in(device, st, a, at, lo - at, dst + (at - off));
        CUDA_CHECK(cudaStreamWaitEvent(st, o.ready, 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(dst + (lo - off), device,
                    o.p + (lo - o.off), o.device, hi - lo, st));
        at = hi;
    }
    if (at < end)
        mirror_in(device, st, a, at, end - at, dst + (at - off));
}

/* Copy the slice of block blk of the buffer into the given slot of the
 * lane replica: from the arena data, or from the host. */
void stage_block(lane & l, slice const & sl, dev_view const & v,
        const void * buf, int blk, int slot, size_t block_bytes)
{
    size_t const off = (size_t) blk * block_bytes + sl.off;
    uint8_t * const dst = (uint8_t *) l.uc + (size_t) slot * sl.bytes;
    if (v.p) {
        arena_in(l.device, l.stream, v.a, v.off + off, sl.bytes, dst);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dst, (const char *) buf + off,
                    sl.bytes, cudaMemcpyHostToDevice, l.stream));
    }
}

void reader(bwc_dev_pieces * p, cudaEvent_t ev)
{
    if (std::find(p->rd, p->rd + p->nrd, ev) != p->rd + p->nrd)
        return;
    ASSERT_ALWAYS(p->nrd < BWC_DEV_PIECES_MAX);
    p->rd[p->nrd++] = ev;
}

/* Record that stream st read the output segments of the arena, on the
 * record of each arena whose memory holds a segment, so the backend
 * waits for the read before it writes that memory again. ev is the
 * event of the lane: a re-record moves it later on the same stream,
 * which still covers every read of the stream. */
void segments_read(cudaEvent_t ev, cudaStream_t st, bwc_device_arena * a)
{
    bwc_dev_pieces const * const p = pieces_of(a);
    if (!p || !p->nout)
        return;
    CUDA_CHECK(cudaEventRecord(ev, st));
    for (int i = 0; i < p->nout; i++)
        reader(p->out[i].owner, ev);
}

/* The replica records of the arena, when the buffer of the exchange
 * covers the whole replica range, so the exchange fills the replicas
 * completely; nullptr otherwise. */
bwc_dev_pieces * replicas_of(dev_view const & v, size_t bytes)
{
    bwc_dev_pieces * const p = v.p ? pieces_of(v.a) : nullptr;
    if (!p || !p->nrep)
        return nullptr;
    bwc_dev_piece const & r = p->rep[0];
    return v.off <= r.off && r.off + r.bytes <= v.off + bytes ? p : nullptr;
}

/* Write the part of the buffer range [off, off + bytes) inside the
 * replica range, from src on the lane, to every replica, on the lane
 * stream. */
void replicas_out(lane & l, dev_view const & v, bwc_dev_pieces const * reps,
        size_t off, size_t bytes, const uint8_t * src)
{
    for (int k = 0; reps && k < reps->nrep; k++) {
        bwc_dev_piece const & r = reps->rep[k];
        size_t const lo = std::max(v.off + off, r.off);
        size_t const hi = std::min(v.off + off + bytes, r.off + r.bytes);
        if (lo >= hi)
            continue;
        CUDA_CHECK(cudaMemcpyPeerAsync(r.p + (lo - r.off), r.device,
                    src + (lo - v.off - off), l.device, hi - lo, l.stream));
    }
}

/* Record the plugin as the last writer of the mirror: the output
 * segments no longer match, and the replicas match when the plugin
 * filled them. */
void written(dev_view const & v, bool reps)
{
    bwc_dev_pieces * const p = pieces_of(v.a);
    if (!p)
        return;
    p->nout = 0;
    if (reps)
        p->rep_wgen = p->wgen;
    else
        p->wgen++;
}

/* Store one event as the new ready marker of a written mirror, on the
 * publication stream, behind the completion of every lane and behind
 * the old marker. The event object lives for the process, which covers
 * every arena lifetime; a later re-record only moves the fire point
 * later on the same stream. */
void publish(comm_state & s, bwc_device_arena * a)
{
    for (lane & l : s.lanes) {
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaEventRecord(l.done, l.stream));
    }
    CUDA_CHECK(cudaSetDevice(s.lanes[0].device));
    for (lane & l : s.lanes)
        CUDA_CHECK(cudaStreamWaitEvent(s.pub, l.done, 0));
    if (a->ready)
        CUDA_CHECK(cudaStreamWaitEvent(s.pub, (cudaEvent_t) a->ready, 0));
    CUDA_CHECK(cudaEventRecord(s.ev, s.pub));
    a->ready = s.ev;
    a->bands = nullptr;
    a->nbands = 0;
}

/* The band event of block blk fires on the publication stream when
 * every active lane has written its slice of the block. */
void band_ready(comm_state & s, std::vector<size_t> const & ds,
        band_store & bs, int blk)
{
    CUDA_CHECK(cudaSetDevice(s.lanes[0].device));
    for (size_t d : ds)
        CUDA_CHECK(cudaStreamWaitEvent(s.pub, s.lanes[d].blk[blk], 0));
    CUDA_CHECK(cudaEventRecord(bs.ev[blk], s.pub));
}

/* Wait on the host for every lane stream. */
void sync_lanes(comm_state & s)
{
    for (lane & l : s.lanes) {
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaStreamSynchronize(l.stream));
    }
}

/* Reduce-scatter: after the kernel, the extra slot holds the XOR of
 * block rank of every rank, and it goes out to block 0 of the buffer. */
void reduce_scatter_xor(void * const * bufs, int nbufs,
        size_t block_bytes, MPI_Comm comm)
{
    comm_state & s = get_state(comm);
    /* the XOR over one job is the identity */
    if (s.njobs == 1)
        return;
    ASSERT_ALWAYS(block_bytes % sizeof(uint64_t) == 0);
    int const n = s.njobs;
    int host_sync = 0;
    std::vector<size_t> const ds = active_lanes(s, block_bytes);
    for (int w = 0; w < nbufs; w++) {
        dev_view const v = dev_of(bufs[w]);
        for (size_t d : ds) {
            lane & l = s.lanes[d];
            slice const sl = slice_of(s, d, block_bytes);
            CUDA_CHECK(cudaSetDevice(l.device));
            reserve(s, l, sl.bytes);
            /* the slots are free once every rank passes the reads of
             * the leg before this one */
            barrier(s, l);
            for (int b = 0; b < n; b++)
                stage_block(l, sl, v, bufs[w], b, b, block_bytes);
            if (v.p)
                segments_read(l.rd, l.stream, v.a);
            /* every rank staged what its peers read */
            barrier(s, l);
            uint8_t * const res = (uint8_t *) l.uc + (size_t) n * sl.bytes;
            comm_multimem_ldxor((uint64_t *) res,
                    (const uint64_t *) (l.mc + (size_t) s.rank * sl.bytes),
                    sl.bytes / sizeof(uint64_t), l.stream);
            if (v.p) {
                /* the write must follow every queued access of its
                 * range of the mirror */
                wait_range(l.stream, v.a, v.off + sl.off, sl.bytes);
                CUDA_CHECK(cudaMemcpyPeerAsync(v.p + sl.off, v.device,
                            res, l.device, sl.bytes, l.stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync((char *) bufs[w] + sl.off,
                            res, sl.bytes, cudaMemcpyDeviceToHost,
                            l.stream));
                host_sync = 1;
            }
        }
        if (v.p) {
            publish(s, v.a);
            written(v, false);
        }
    }
    /* a host buffer must be settled at return; a mirror carries its
     * completion in the ready marker */
    if (host_sync)
        sync_lanes(s);
}

/* Allgather: the kernel writes the own block from the extra slot to
 * slot rank of every rank, and the slots go out to all blocks of the
 * buffer. */
void allgather(void * const * bufs, int nbufs, size_t block_bytes,
        MPI_Comm comm)
{
    comm_state & s = get_state(comm);
    /* the allgather over one job is the identity */
    if (s.njobs == 1)
        return;
    ASSERT_ALWAYS(block_bytes % sizeof(uint64_t) == 0);
    int const n = s.njobs;
    int host_sync = 0;
    std::vector<size_t> const ds = active_lanes(s, block_bytes);
    for (int w = 0; w < nbufs; w++) {
        dev_view const v = dev_of(bufs[w]);
        band_store * const bs = v.p ? &store_of(s, v.a) : nullptr;
        bwc_dev_pieces const * const reps =
            replicas_of(v, (size_t) n * block_bytes);
        for (size_t d : ds) {
            lane & l = s.lanes[d];
            slice const sl = slice_of(s, d, block_bytes);
            CUDA_CHECK(cudaSetDevice(l.device));
            reserve(s, l, sl.bytes);
            /* the slots are free once every rank passes the reads of
             * the leg before this one */
            barrier(s, l);
            stage_block(l, sl, v, bufs[w], s.rank, n, block_bytes);
            if (v.p)
                segments_read(l.rd, l.stream, v.a);
            comm_multimem_bcast(
                    (uint64_t *) (l.mc + (size_t) s.rank * sl.bytes),
                    (const uint64_t *) (l.uc + (size_t) n * sl.bytes),
                    sl.bytes / sizeof(uint64_t), l.stream);
            /* every slot is current before the copies out */
            barrier(s, l);
            if (v.p) {
                /* the writes must follow every queued access of the
                 * mirror and of the replicas */
                wait_range(l.stream, v.a, v.off,
                        (size_t) n * block_bytes);
                for (int k = 0; reps && k < reps->nrep; k++)
                    CUDA_CHECK(cudaStreamWaitEvent(l.stream,
                                reps->rep[k].ready, 0));
            }
            for (int b = 0; b < n; b++) {
                size_t const off = (size_t) b * block_bytes + sl.off;
                uint8_t * const got =
                    (uint8_t *) l.uc + (size_t) b * sl.bytes;
                if (v.p) {
                    CUDA_CHECK(cudaMemcpyPeerAsync(v.p + off, v.device,
                                got, l.device, sl.bytes, l.stream));
                    replicas_out(l, v, reps, off, sl.bytes, got);
                    CUDA_CHECK(cudaEventRecord(l.blk[b], l.stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync((uint8_t *) bufs[w] + off,
                                got, sl.bytes, cudaMemcpyDeviceToHost,
                                l.stream));
                    host_sync = 1;
                }
            }
        }
        if (v.p) {
            /* the band events must also cover the accesses the old
             * ready marker holds */
            if (v.a->ready)
                CUDA_CHECK(cudaStreamWaitEvent(s.pub,
                            (cudaEvent_t) v.a->ready, 0));
            for (int b = 0; b < n; b++)
                band_ready(s, ds, *bs, b);
            publish(s, v.a);
            written(v, reps != nullptr);
            /* the band contract wants bands that start at offset zero */
            if (v.off == 0) {
                for (int b = 0; b < n; b++)
                    bs->bands[b] = { (size_t) b * block_bytes,
                            block_bytes, bs->ev[b] };
                v.a->bands = bs->bands.data();
                v.a->nbands = n;
            }
        }
    }
    if (host_sync)
        sync_lanes(s);
}

bwc_comm_plugin_api const api = { reduce_scatter_xor, allgather, 1 };

} // namespace

extern "C" bwc_comm_plugin_api const * bwc_comm_solib_reach_api()
{
    return &api;
}
