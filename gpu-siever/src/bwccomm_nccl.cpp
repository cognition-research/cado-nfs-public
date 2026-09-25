#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "macros.h"
#include "bwc_device_arena.hpp"
#include "bwc_dev_pieces.h"
#include "matmul_top_comm_plugin.hpp"
#include "select_mpi.h"

#include "comm_xor.h"
#define TL_K0_ENV "BWC_NCCL_TIMELINE_K0"
#include "timeline.h"

static tl_state tl = { {}, 0, nullptr, {}, "nccl" };
static unsigned long tl_k = 0;

/* A bwc communication plugin (comm_impl=nccl) that runs the two vector
 * exchange legs over NCCL. Both legs are steps of NCCL send and
 * receive pairs: NCCL has no XOR reduction for the reduce-scatter,
 * and the allgather steps give one completion event per block. The
 * steps form a ring (n-1 steps of one pair, the default) or, with
 * BWC_NCCL_DIRECT=rs,ag, one direct step of n-1 pairs to every peer
 * at once. A buffer whose device mirror is current moves through
 * device copies only; the other buffers stage through host memory.
 *
 * The exchange runs on SPMV_GF2_NGPU lanes. Each lane is one local
 * device with its own NCCL communicator, stream and staging area, and
 * it moves one slice of every block. The lanes run the same ring at
 * the same time, so the exchange uses the network paths of all the
 * lane devices and not the path of one device. The lane devices are
 * the devices of the matmul backend (spmv_gf2.cu), so the mirror data
 * moves to and from the lanes with peer copies.
 *
 * The plugin overlaps the exchange with the product. The lane streams
 * wait on the ready and band events of each mirror they read instead
 * of a host wait, and each mirror the plugin writes gets one event of
 * lane 0 as its new ready marker (bwc_device_arena.hpp). Lane 0
 * records that event behind the completion of every lane. The host
 * only waits for the buffers without a mirror.
 *
 * The matmul backend keeps the product output in one segment per
 * device and reads the source from one replica per device
 * (bwc_dev_pieces.h). The reduce-scatter stages each slice from the
 * device that holds it, and the allgather writes each received slice
 * to the mirror and to every replica, so no vector data waits for a
 * copy chain through the mirror device. The lane stream carries the
 * network steps only. The reduce-scatter stages its slices on a copy
 * stream (lane::copy) and the allgather writes the received slices
 * out on a fan-out stream (lane::fan), each behind the step of the
 * lane stream it depends on, so the copies of one ring step overlap
 * the network of the next step. The staging area is double: the
 * reduce-scatter of one exchange stages into the half that the
 * fan-out of the exchange before the last one read, so the staging
 * never waits for the fan-out still in flight. The events that cover
 * every lane (the band events and the ready marker) are recorded on
 * a publication stream of the mirror device (comm_state::pub), so
 * the lane streams never wait for one another.
 *
 * BWC_NCCL_HOLDER=1 replaces the equal lane slices of a device buffer
 * with the pieces the devices hold. The ranks swap the layouts of
 * their product output (the output segments cut at the block
 * boundaries, and the mirror for the rest) over MPI at each
 * reduce-scatter, and one NCCL communicator spans every lane of every
 * rank. Each overlap of two layouts goes from the device that holds
 * it on one rank to the device that holds it on the other, which
 * receives into its accumulation area (lane::acc) and XORs its own
 * segment in, so no slice stages onto a lane device. The allgather
 * sends the result pieces from their holders and receives each piece
 * straight into the replica (or the mirror) on the device of the same
 * lane index, which copies it on to the mirror and the other replicas.
 * A device buffer without output segments or a lane without a replica
 * or the mirror aborts; host buffers keep the host staging. */

#define CUDA_CHECK(call) do {                                           \
    cudaError_t const e = (call);                                       \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                cudaGetErrorString(e));                                 \
        abort();                                                        \
    }                                                                   \
} while (0)

#define NCCL_CHECK(call) do {                                           \
    ncclResult_t const e = (call);                                      \
    if (e != ncclSuccess) {                                             \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                ncclGetErrorString(e));                                 \
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
    std::vector<std::vector<cudaEvent_t>> wbev; /* per lane: the write records */
};

/* One lane: a device, its NCCL communicator over the MPI
 * communicator, and the staging area for its slice of the blocks. */
struct lane {
    ncclComm_t nccl;
    cudaStream_t stream;
    cudaStream_t copy;          /* reduce-scatter staging reads */
    cudaStream_t fan;           /* joins the fan-out streams per block */
    std::vector<cudaStream_t> fans; /* one per destination: mirror, replicas */
    std::vector<cudaEvent_t> fanev; /* one per fan-out stream */
    cudaEvent_t done;           /* fires when the lane completes a leg */
    cudaEvent_t mir;            /* the mirror copy of the result, on copy */
    cudaEvent_t fork;           /* a side stream waits here on the lane stream */
    cudaEvent_t rd[2];          /* the copy stream, the lane stream read segments */
    cudaEvent_t fanned[2];      /* the fan-out read the staging half */
    std::vector<cudaEvent_t> blk;   /* one per block, allgather arrival */
    int device;
    uint8_t * dev[2] = { nullptr, nullptr };    /* njobs slices each */
    bwc_dev_pieces stg[2] = {}; /* the readers of each half (rd only) */
    uint8_t * tmp = nullptr;    /* one slice per receive of a step */
    size_t slice_alloc = 0;
    int cur = 0;                /* the staging half of this exchange */
    ncclComm_t hnccl = nullptr; /* BWC_NCCL_HOLDER: every lane of every rank */
    uint8_t * acc[2] = { nullptr, nullptr };    /* the pieces this lane holds */
    size_t acc_alloc[2] = { 0, 0 };
    uint8_t * htmp = nullptr;   /* reduce-scatter receives past the first */
    size_t htmp_alloc = 0;
    uint8_t * agtmp = nullptr;  /* allgather receives without replicas */
    size_t agtmp_alloc = 0;
};

/* A layout piece as the ranks swap it: a range of the buffer and the
 * lane index of the device that holds it. */
#define HOLDER_MAX 256
struct hwire {
    uint64_t off, bytes;
    int64_t lane;
};

struct hlayout {
    int64_t n;
    hwire w[HOLDER_MAX];
};

/* The lanes of one MPI communicator, made on first use. bwc uses one
 * communicator per grid direction, so the list stays small. */
struct comm_state {
    MPI_Comm mpi;
    std::vector<lane> lanes;
    cudaStream_t pub;           /* on the mirror device: the events over all lanes */
    cudaEvent_t ev;             /* ready marker for the written mirrors */
    int rank;
    int njobs;
    bool direct_rs, direct_ag;  /* BWC_NCCL_DIRECT names the leg */
    bool stg_streams;           /* BWC_NCCL_STG_STREAMS=1: one staging stream per block */
    bool holder;                /* BWC_NCCL_HOLDER=1 */
    bool nomirror;              /* BWC_NCCL_HOLDER_NOMIRROR=1 */
    bool wbands;                /* BWC_NCCL_HOLDER_WBANDS=1 */
    bool holder_started = false; /* a device buffer took the holder exchange */
    std::map<bwc_device_arena *, band_store> stores;
};

/* One send and receive pair of an exchange step: the block slice this
 * rank sends and to whom, the block slice it receives and from whom. */
struct hop {
    int sblk, to, rblk, from;
};

/* The direct schedule: one step, a pair with every peer. */
template<typename F>
std::vector<std::vector<hop>> direct_steps(int n, int rank, F hop_to)
{
    std::vector<hop> step;
    for (int j = 1; j < n; j++)
        step.push_back(hop_to((rank + j) % n));
    return { step };
}

/* The ring schedule: n-1 steps of one pair with the two neighbours. */
template<typename F>
std::vector<std::vector<hop>> ring_steps(int n, F hop_at)
{
    std::vector<std::vector<hop>> steps;
    for (int i = 0; i < n - 1; i++)
        steps.push_back({ hop_at(i) });
    return steps;
}

/* Reduce-scatter: a hop sends a staged block and receives one to
 * reduce into another. The ring reduces along the way (step i sends
 * block rank-1-i and reduces into block rank-2-i, so the own block
 * carries the full XOR after n-1 steps); the direct step sends block j
 * to rank j and reduces every receipt into the own block. */
std::vector<std::vector<hop>> rs_steps(comm_state const & s)
{
    int const n = s.njobs, r = s.rank;
    if (s.direct_rs)
        return direct_steps(n, r, [=](int j) { return hop { j, j, r, j }; });
    return ring_steps(n, [=](int i) {
        return hop { (r - 1 - i + 2 * n) % n, (r + 1) % n,
                     (r - 2 - i + 2 * n) % n, (r - 1 + n) % n }; });
}

/* Allgather: a hop sends a block and receives another. The ring
 * forwards one block per step (the block of rank r-1-i arrives at step
 * i); the direct step sends the own block to every peer. */
std::vector<std::vector<hop>> ag_steps(comm_state const & s)
{
    int const n = s.njobs, r = s.rank;
    if (s.direct_ag)
        return direct_steps(n, r, [=](int j) { return hop { r, j, j, j }; });
    return ring_steps(n, [=](int i) {
        return hop { (r - i + n) % n, (r + 1) % n,
                     (r - 1 - i + 2 * n) % n, (r - 1 + n) % n }; });
}

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

/* BWC_NCCL_COPY_PRIO=1 gives the copy streams of the lane the greatest
 * priority, so that a copy queued while the product kernel fills the
 * device runs as its blocks retire instead of after it; =2 gives the
 * lane stream the same, so the NCCL and XOR kernels also run beside
 * the product instead of after it. */
int prio_level()
{
    static int const level = getenv("BWC_NCCL_COPY_PRIO")
        ? atoi(getenv("BWC_NCCL_COPY_PRIO")) : 0;
    return level;
}

void prio_stream(cudaStream_t * st, int level)
{
    if (prio_level() < level) {
        CUDA_CHECK(cudaStreamCreate(st));
        return;
    }
    int least, greatest;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(st, cudaStreamDefault, greatest));
}

comm_state & get_state(MPI_Comm comm)
{
    static std::mutex mm;
    static std::vector<comm_state *> states;
    std::lock_guard<std::mutex> const dummy(mm);
    for (comm_state * s : states)
        if (s->mpi == comm)
            return *s;
    auto * s = new comm_state;
    s->mpi = comm;
    MPI_Comm_rank(comm, &s->rank);
    MPI_Comm_size(comm, &s->njobs);
    const char * const direct = getenv("BWC_NCCL_DIRECT");
    s->direct_rs = direct && strstr(direct, "rs");
    s->direct_ag = direct && strstr(direct, "ag");
    const char * const stg = getenv("BWC_NCCL_STG_STREAMS");
    s->stg_streams = stg && atoi(stg) == 1;
    const char * const holder = getenv("BWC_NCCL_HOLDER");
    s->holder = holder && atoi(holder) == 1;
    const char * const nomirror = getenv("BWC_NCCL_HOLDER_NOMIRROR");
    s->nomirror = nomirror && atoi(nomirror) == 1;
    if (s->nomirror && !s->holder) {
        fprintf(stderr, "bwccomm_nccl: BWC_NCCL_HOLDER_NOMIRROR needs"
                " BWC_NCCL_HOLDER=1\n");
        abort();
    }
    const char * const wbands = getenv("BWC_NCCL_HOLDER_WBANDS");
    s->wbands = wbands && atoi(wbands) == 1;
    if (s->wbands && !s->holder) {
        fprintf(stderr, "bwccomm_nccl: BWC_NCCL_HOLDER_WBANDS needs"
                " BWC_NCCL_HOLDER=1\n");
        abort();
    }
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int const ngpu = env_ngpu() < ndev ? env_ngpu() : ndev;
    int const first = pick_first_device(ngpu, ndev);
    s->lanes.resize(ngpu);
    for (int d = 0; d < ngpu; d++) {
        lane & l = s->lanes[d];
        l.device = (first + d) % ndev;
        CUDA_CHECK(cudaSetDevice(l.device));
        ncclUniqueId id;
        if (s->rank == 0)
            NCCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
        NCCL_CHECK(ncclCommInitRank(&l.nccl, s->njobs, id, s->rank));
        prio_stream(&l.stream, 2);
        prio_stream(&l.copy, 1);
        prio_stream(&l.fan, 1);
        l.fans.resize(ndev + 1);
        l.fanev.resize(ndev + 1);
        for (size_t k = 0; k < l.fans.size(); k++) {
            prio_stream(&l.fans[k], 1);
            CUDA_CHECK(cudaEventCreateWithFlags(&l.fanev[k],
                        cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&l.done,
                    cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&l.mir,
                    cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&l.fork,
                    cudaEventDisableTiming));
        for (cudaEvent_t & r : l.rd)
            CUDA_CHECK(cudaEventCreateWithFlags(&r, cudaEventDisableTiming));
        for (cudaEvent_t & f : l.fanned)
            CUDA_CHECK(cudaEventCreateWithFlags(&f, cudaEventDisableTiming));
        for (int b = 0; b < s->njobs; b++) {
            cudaEvent_t bev;
            CUDA_CHECK(cudaEventCreateWithFlags(&bev,
                        cudaEventDisableTiming));
            l.blk.push_back(bev);
        }
    }
    if (s->holder && s->njobs > 1) {
        int lo = ngpu, hi = ngpu;
        MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT, MPI_MAX, comm);
        if (lo != hi) {
            fprintf(stderr, "bwccomm_nccl: BWC_NCCL_HOLDER needs the same"
                    " lane count on every rank (%d to %d)\n", lo, hi);
            abort();
        }
        ncclUniqueId id;
        if (s->rank == 0)
            NCCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
        NCCL_CHECK(ncclGroupStart());
        for (int d = 0; d < ngpu; d++) {
            lane & l = s->lanes[d];
            CUDA_CHECK(cudaSetDevice(l.device));
            NCCL_CHECK(ncclCommInitRank(&l.hnccl, s->njobs * ngpu, id,
                        s->rank * ngpu + d));
        }
        NCCL_CHECK(ncclGroupEnd());
    }
    CUDA_CHECK(cudaSetDevice(s->lanes[0].device));
    CUDA_CHECK(cudaStreamCreate(&s->pub));
    CUDA_CHECK(cudaEventCreateWithFlags(&s->ev, cudaEventDisableTiming));
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

void reserve(comm_state & s, lane & l, size_t slice_bytes)
{
    if (l.slice_alloc >= slice_bytes)
        return;
    for (uint8_t * & d : l.dev) {
        CUDA_CHECK(cudaFree(d));
        CUDA_CHECK(cudaMalloc(&d, (size_t) s.njobs * slice_bytes));
    }
    CUDA_CHECK(cudaFree(l.tmp));
    CUDA_CHECK(cudaMalloc(&l.tmp, (size_t) s.njobs * slice_bytes));
    l.slice_alloc = slice_bytes;
}

/* The staging area of the lane for this exchange. */
uint8_t * staging(lane & l)
{
    return l.dev[l.cur];
}

/* Switch the lane to its other staging half, behind the fan-out that
 * last read it and the readers of the result segments it held. */
void fresh_half(lane & l)
{
    l.cur ^= 1;
    bwc_dev_pieces & r = l.stg[l.cur];
    for (cudaStream_t st : { l.stream, l.copy }) {
        CUDA_CHECK(cudaStreamWaitEvent(st, l.fanned[l.cur], 0));
        for (int i = 0; i < r.nrd; i++)
            CUDA_CHECK(cudaStreamWaitEvent(st, r.rd[i], 0));
    }
    r.nrd = 0;
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
 * lane, on stream st, from the mirror behind its band or ready events. */
void mirror_in(lane & l, cudaStream_t st, bwc_device_arena * a, size_t off,
        size_t bytes, uint8_t * dst)
{
    wait_range(st, a, off, bytes);
    tl_mark(tl, st, l.device, "in mir", tl_k);
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, l.device, (char *) a->dev + off,
                a->device, bytes, st));
}

/* Copy the range [off, off + bytes) of the arena data to dst on the
 * lane, on stream st: each part that an output segment covers comes
 * from the device that holds the segment, the rest from the mirror. */
void arena_in(lane & l, cudaStream_t st, bwc_device_arena * a, size_t off,
        size_t bytes, uint8_t * dst)
{
    bwc_dev_pieces const * const p = pieces_of(a);
    size_t const end = off + bytes;
    size_t at = off;
    for (int i = 0; p && i < p->nout; i++) {
        bwc_dev_piece const & o = p->out[i];
        if (tl_on() && tl_k == TL_K0 && off == 0)
            fprintf(stderr, "TP %d dev%d %zu %zu\n", i, o.device, o.off, o.bytes);
        size_t const lo = std::max(o.off, at);
        size_t const hi = std::min(o.off + o.bytes, end);
        if (lo >= hi)
            continue;
        if (at < lo)
            mirror_in(l, st, a, at, lo - at, dst + (at - off));
        CUDA_CHECK(cudaStreamWaitEvent(st, o.ready, 0));
        tl_mark(tl, st, l.device, tl_name("in w", i), tl_k);
        CUDA_CHECK(cudaMemcpyPeerAsync(dst + (lo - off), l.device,
                    o.p + (lo - o.off), o.device, hi - lo, st));
        tl_mark(tl, st, l.device, tl_name("in c", i), tl_k);
        at = hi;
    }
    if (at < end)
        mirror_in(l, st, a, at, end - at, dst + (at - off));
}

/* Copy the slice of block blk of the buffer to the staging area of the
 * lane, on stream st: from the arena data, or from the host. */
void stage_block(lane & l, cudaStream_t st, slice const & sl,
        dev_view const & v, const void * buf, int blk, size_t block_bytes)
{
    size_t const off = (size_t) blk * block_bytes + sl.off;
    uint8_t * const dst = staging(l) + (size_t) blk * sl.bytes;
    if (v.p) {
        arena_in(l, st, v.a, v.off + off, sl.bytes, dst);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dst, (const char *) buf + off,
                    sl.bytes, cudaMemcpyHostToDevice, st));
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
 * event of the lane for this stream: a re-record moves it later on the
 * same stream, which still covers every read of the stream. */
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

/* Move a side stream of the lane behind the work queued on the lane
 * stream up to now. */
void fork(lane & l, cudaStream_t side)
{
    CUDA_CHECK(cudaEventRecord(l.fork, l.stream));
    CUDA_CHECK(cudaStreamWaitEvent(side, l.fork, 0));
}

/* Move every fan-out stream of the lane behind the work queued on the
 * lane stream up to now. */
void fork_fans(lane & l)
{
    CUDA_CHECK(cudaEventRecord(l.fork, l.stream));
    for (cudaStream_t f : l.fans)
        CUDA_CHECK(cudaStreamWaitEvent(f, l.fork, 0));
}

/* Make the stream wait for the copies queued on every fan-out stream
 * of the lane up to now. */
void join_fans(lane & l, cudaStream_t into)
{
    for (size_t k = 0; k < l.fans.size(); k++) {
        CUDA_CHECK(cudaEventRecord(l.fanev[k], l.fans[k]));
        CUDA_CHECK(cudaStreamWaitEvent(into, l.fanev[k], 0));
    }
}

/* The stream that stages block i of the reduce-scatter: the copy
 * stream, or with stg_streams one fan-out stream per block, so that a
 * block whose segments are ready does not queue behind the copies of a
 * block whose product still runs. The fan-out streams follow the copy
 * stream into the staging, and the copy stream joins them after it. */
cudaStream_t stage_stream(comm_state const & s, lane & l, int i)
{
    return s.stg_streams ? l.fans[i % l.fans.size()] : l.copy;
}

void fork_stage_streams(comm_state const & s, lane & l)
{
    if (!s.stg_streams)
        return;
    CUDA_CHECK(cudaEventRecord(l.fork, l.copy));
    for (cudaStream_t f : l.fans)
        CUDA_CHECK(cudaStreamWaitEvent(f, l.fork, 0));
}

void join_stage_streams(comm_state const & s, lane & l)
{
    if (s.stg_streams)
        join_fans(l, l.copy);
}

/* Write the part of the buffer range [off, off + bytes) inside the
 * replica range, from src on the lane, to every replica, one fan-out
 * stream per replica so the copies run side by side. */
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
                    src + (lo - v.off - off), l.device, hi - lo,
                    l.fans[k + 1]));
    }
}

/* Write the range [off, off + bytes) of the buffer, from src on the
 * lane, to the mirror and to every replica, on the fan-out streams. */
void arena_out(lane & l, dev_view const & v, bwc_dev_pieces const * reps,
        size_t off, size_t bytes, const uint8_t * src)
{
    CUDA_CHECK(cudaMemcpyPeerAsync(v.p + off, v.device, src, l.device,
                bytes, l.fans[0]));
    replicas_out(l, v, reps, off, bytes, src);
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

/* The result slices of the lanes become the output segments of the
 * arena, each in the staging half of its lane behind the done event
 * of the lane, so the allgather takes its own block from the lane that
 * wrote it, on the device that holds it, and does not wait for the
 * other lanes. A reader records its event in the staging record of
 * the half, which fresh_half waits for before the half is written
 * again. The caller has recorded the done events. */
void results_out(comm_state & s, std::vector<size_t> const & ds,
        dev_view const & v, size_t block_bytes)
{
    bwc_dev_pieces * const p = pieces_of(v.a);
    if (!p)
        return;
    ASSERT_ALWAYS(ds.size() <= BWC_DEV_PIECES_MAX);
    p->nout = 0;
    for (size_t d : ds) {
        lane & l = s.lanes[d];
        slice const sl = slice_of(s, d, block_bytes);
        p->out[p->nout++] = { l.device,
            (char *) staging(l) + (size_t) s.rank * sl.bytes,
            v.off + sl.off, sl.bytes, l.done, &l.stg[l.cur] };
    }
    /* the lanes waited for the ready marker before they wrote */
    p->out_settled = true;
    p->out_before = nullptr;
    p->out_end = v.off + block_bytes;
    p->wgen++;
}

/* Make the stream follow every access of the mirror that precedes a
 * write by the lane that took the range: through the settled segments
 * that cover the range, or else through the band and ready events. */
void wait_before(cudaStream_t st, bwc_device_arena * a, size_t off,
        size_t bytes)
{
    bwc_dev_pieces const * const p = pieces_of(a);
    bool const settled = p && p->nout && p->out_settled
        && p->out[0].off <= off && off + bytes <= p->out_end;
    if (!settled) {
        wait_range(st, a, off, bytes);
        return;
    }
    if (p->out_before)
        CUDA_CHECK(cudaStreamWaitEvent(st, p->out_before, 0));
}

/* Make the publication stream wait for the completion of every lane,
 * so that one event on it covers the whole exchange. */
void join_lanes(comm_state & s)
{
    for (lane & l : s.lanes) {
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaEventRecord(l.done, l.stream));
        CUDA_CHECK(cudaStreamWaitEvent(s.pub, l.done, 0));
    }
}

/* Store the plugin's event as the new ready marker of a written
 * mirror. The event object lives for the process, which covers every
 * arena lifetime; a later re-record only moves the fire point later on
 * the same stream. The caller has joined the lanes on the publication
 * stream. */
void publish(comm_state & s, bwc_device_arena * a)
{
    CUDA_CHECK(cudaSetDevice(s.lanes[0].device));
    CUDA_CHECK(cudaEventRecord(s.ev, s.pub));
    a->ready = s.ev;
    a->bands = nullptr;
    a->nbands = 0;
}

/* Wait on the host for every lane stream. */
void sync_lanes(comm_state & s)
{
    for (lane & l : s.lanes) {
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaStreamSynchronize(l.stream));
    }
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

/* Ask the product backend to compute the next product into the arena
 * block by block in the order the ring consumes them, so the staging
 * of a block starts while the product computes the later ones. */
template<typename F>
void request_order(dev_view const & v, int n, size_t block_bytes, F blk_at)
{
    bwc_dev_pieces * const p = pieces_of(v.a);
    if (!p || v.off)
        return;
    ASSERT_ALWAYS(n <= BWC_DEV_PIECES_MAX);
    p->blk_bytes = block_bytes;
    p->nblk = n;
    for (int i = 0; i < n; i++)
        p->blk[i] = blk_at(i);
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

/* A piece of the local layout: the range [off, off + bytes) of the
 * buffer on lane, read from src behind ready (the band and ready events
 * of the mirror when ready is null), with owner as the record of the
 * memory, and acc as its offset in the accumulation area of the lane. */
struct hpiece {
    size_t off, bytes;
    int lane;
    char * src;
    cudaEvent_t ready;
    bwc_dev_pieces * owner;
    size_t acc;
};

/* An overlap of two layouts: indices a and c of the pieces that hold
 * it. */
struct hcut {
    size_t off, bytes;
    int a, c;
};

[[noreturn]] void holder_fail(const char * what)
{
    fprintf(stderr, "bwccomm_nccl: BWC_NCCL_HOLDER: %s\n", what);
    abort();
}

int lane_of(comm_state const & s, int device)
{
    for (size_t d = 0; d < s.lanes.size(); d++)
        if (s.lanes[d].device == device)
            return (int) d;
    holder_fail("a piece is on a device outside the lanes");
}

void grow(uint8_t * & p, size_t & alloc, size_t want)
{
    if (alloc >= want)
        return;
    CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaMalloc(&p, want));
    alloc = want;
}

void mirror_piece(std::vector<hpiece> & whole, bwc_dev_pieces const * p,
        dev_view const & v, size_t at, size_t bytes, int ml)
{
    if (bwc_dev_stale_hits(p, v.off + at, bytes))
        holder_fail("a mirror piece holds rows the product left to the plugin");
    whole.push_back({ at, bytes, ml, v.p + at, nullptr, nullptr, 0 });
}

/* The layout of the buffer range [lo, hi): the output segments (only
 * settled ones when settled is set), the mirror between and after
 * them, every piece cut at the block boundaries; the accumulation
 * offsets pack the pieces of each lane. A buffer without segments is
 * one mirror piece per block. */
std::vector<hpiece> holder_layout(comm_state const & s, dev_view const & v,
        size_t lo, size_t hi, size_t block_bytes, bool settled)
{
    bwc_dev_pieces const * const p = pieces_of(v.a);
    if (!p)
        holder_fail("the buffer arena has no piece records");
    std::vector<bwc_dev_piece> segs;
    if (!settled || p->out_settled)
        segs.assign(p->out, p->out + p->nout);
    std::sort(segs.begin(), segs.end(),
            [](bwc_dev_piece const & x, bwc_dev_piece const & y) {
                return x.off < y.off; });
    int const ml = lane_of(s, v.device);
    std::vector<hpiece> whole;
    size_t at = lo;
    for (bwc_dev_piece const & o : segs) {
        if (o.off < v.off)
            holder_fail("an output segment starts before the buffer");
        size_t const base = o.off - v.off;
        size_t const a = std::max(base, lo);
        size_t const b = std::min(base + o.bytes, hi);
        if (a >= b)
            continue;
        if (a < at)
            holder_fail("the output segments overlap");
        if (at < a)
            mirror_piece(whole, p, v, at, a - at, ml);
        whole.push_back({ a, b - a, lane_of(s, o.device), o.p + (a - base),
                o.ready, o.owner, 0 });
        at = b;
    }
    if (at < hi)
        mirror_piece(whole, p, v, at, hi - at, ml);
    std::vector<hpiece> L;
    std::vector<size_t> packed(s.lanes.size(), 0);
    for (hpiece q : whole) {
        while (q.bytes) {
            size_t const end = (q.off / block_bytes + 1) * block_bytes;
            size_t const len = std::min(q.bytes, end - q.off);
            hpiece x = q;
            x.bytes = len;
            x.acc = packed[x.lane];
            packed[x.lane] += len;
            L.push_back(x);
            q.off += len;
            q.src += len;
            q.bytes -= len;
        }
    }
    if (L.size() > HOLDER_MAX)
        holder_fail("the layout passes HOLDER_MAX pieces");
    return L;
}

hlayout wire_of(std::vector<hpiece> const & L)
{
    hlayout w {};
    w.n = (int64_t) L.size();
    for (size_t i = 0; i < L.size(); i++)
        w.w[i] = { L[i].off, L[i].bytes, L[i].lane };
    return w;
}

/* The overlaps of layouts A and C inside [lo, hi), in offset order. */
std::vector<hcut> intersect(hlayout const & A, hlayout const & C,
        size_t lo, size_t hi)
{
    std::vector<hcut> out;
    int i = 0, j = 0;
    while (i < A.n && j < C.n) {
        size_t const a1 = A.w[i].off + A.w[i].bytes;
        size_t const c1 = C.w[j].off + C.w[j].bytes;
        size_t const x0 = std::max({ (size_t) A.w[i].off,
                (size_t) C.w[j].off, lo });
        size_t const x1 = std::min({ a1, c1, hi });
        if (x0 < x1)
            out.push_back({ x0, x1 - x0, i, j });
        if (a1 < c1)
            i++;
        else
            j++;
    }
    return out;
}

void wait_piece(cudaStream_t st, dev_view const & v, hpiece const & q)
{
    if (q.ready)
        CUDA_CHECK(cudaStreamWaitEvent(st, q.ready, 0));
    else
        wait_range(st, v.a, v.off + q.off, q.bytes);
}

/* Free a lane area behind every read of it on the device. */
void grow_behind(lane & l, uint8_t * & p, size_t & alloc, size_t want)
{
    if (alloc >= want)
        return;
    for (cudaStream_t st : { l.stream, l.copy, l.fan })
        CUDA_CHECK(cudaStreamSynchronize(st));
    for (cudaStream_t st : l.fans)
        CUDA_CHECK(cudaStreamSynchronize(st));
    for (bwc_dev_pieces const & r : l.stg)
        for (int i = 0; i < r.nrd; i++)
            CUDA_CHECK(cudaEventSynchronize(r.rd[i]));
    grow(p, alloc, want);
}

/* The reduce-scatter of a device buffer along rs_steps, from the
 * holders of the pieces. A rank sends its block pieces to the holders
 * of the same ranges on the peer, and receives into the accumulation
 * area of its own holders: the first receipt of a block into the area,
 * XOR the own segment; a later receipt into htmp, XOR into the area.
 * Then the result pieces of the own block go to block 0 of the mirror
 * and become the output segments of the buffer. */
void holder_rs(comm_state & s, dev_view const & v, size_t block_bytes)
{
    int const n = s.njobs, r = s.rank;
    int const g = (int) s.lanes.size();
    std::vector<hpiece> const L = holder_layout(s, v, 0,
            (size_t) n * block_bytes, block_bytes, false);
    if (s.nomirror)
        pieces_of(v.a)->out_only = true;
    hlayout const mine = wire_of(L);
    std::vector<hlayout> lay(n);
    MPI_Allgather(&mine, sizeof(hlayout), MPI_BYTE, lay.data(),
            sizeof(hlayout), MPI_BYTE, s.mpi);
    std::vector<std::vector<hop>> const steps = rs_steps(s);
    size_t nhops = 0;
    for (auto const & step : steps)
        nhops = std::max(nhops, step.size());
    std::vector<size_t> accb(g, 0);
    for (hpiece const & q : L)
        accb[q.lane] = std::max(accb[q.lane], q.acc + q.bytes);
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        fork(l, l.copy);
        fresh_half(l);
        grow_behind(l, l.acc[l.cur], l.acc_alloc[l.cur], accb[d]);
        if (nhops > 1 || steps.size() > 1)
            grow_behind(l, l.htmp, l.htmp_alloc, nhops * accb[d]);
        tl_mark(tl, l.stream, l.device, "hr free", tl_k);
    }
    auto const acc_at = [&](hpiece const & q, size_t off) {
        return (char *) s.lanes[q.lane].acc[s.lanes[q.lane].cur]
            + q.acc + (off - q.off);
    };
    std::vector<char> accd(n, 0);
    for (size_t i = 0; i < steps.size(); i++) {
        std::vector<hop> const & step = steps[i];
        for (hop const & h : step) {
            if (accd[h.sblk])
                continue;
            for (hcut const & x : intersect(mine, lay[h.to],
                        (size_t) h.sblk * block_bytes,
                        (size_t) (h.sblk + 1) * block_bytes)) {
                hpiece const & q = L[x.a];
                lane & l = s.lanes[q.lane];
                CUDA_CHECK(cudaSetDevice(l.device));
                wait_piece(l.stream, v, q);
            }
        }
        for (int d = 0; d < g; d++) {
            lane & l = s.lanes[d];
            CUDA_CHECK(cudaSetDevice(l.device));
            tl_mark(tl, l.stream, l.device, tl_name("hr rdy", i), tl_k);
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t k = 0; k < step.size(); k++) {
            hop const & h = step[k];
            for (hcut const & x : intersect(mine, lay[h.to],
                        (size_t) h.sblk * block_bytes,
                        (size_t) (h.sblk + 1) * block_bytes)) {
                hpiece const & q = L[x.a];
                lane & l = s.lanes[q.lane];
                CUDA_CHECK(cudaSetDevice(l.device));
                char * const src = accd[h.sblk] ? acc_at(q, x.off)
                    : q.src + (x.off - q.off);
                NCCL_CHECK(ncclSend(src, x.bytes, ncclUint8,
                            h.to * g + (int) lay[h.to].w[x.c].lane,
                            l.hnccl, l.stream));
            }
            for (hcut const & x : intersect(lay[h.from], mine,
                        (size_t) h.rblk * block_bytes,
                        (size_t) (h.rblk + 1) * block_bytes)) {
                hpiece const & q = L[x.c];
                lane & l = s.lanes[q.lane];
                CUDA_CHECK(cudaSetDevice(l.device));
                char * const dst = !accd[h.rblk] && k == 0
                    ? acc_at(q, x.off)
                    : (char *) l.htmp + k * accb[q.lane] + q.acc
                        + (x.off - q.off);
                NCCL_CHECK(ncclRecv(dst, x.bytes, ncclUint8,
                            h.from * g + (int) lay[h.from].w[x.a].lane,
                            l.hnccl, l.stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        for (int d = 0; d < g; d++) {
            lane & l = s.lanes[d];
            CUDA_CHECK(cudaSetDevice(l.device));
            tl_mark(tl, l.stream, l.device, tl_name("hr net", i), tl_k);
        }
        for (size_t k = 0; k < step.size(); k++) {
            hop const & h = step[k];
            size_t const lo = (size_t) h.rblk * block_bytes;
            for (hpiece const & q : L) {
                if (q.off < lo || q.off >= lo + block_bytes)
                    continue;
                lane & l = s.lanes[q.lane];
                CUDA_CHECK(cudaSetDevice(l.device));
                uint64_t * const a = (uint64_t *) acc_at(q, q.off);
                if (!accd[h.rblk] && k == 0) {
                    wait_piece(l.stream, v, q);
                    comm_xor_gf2(a, (const uint64_t *) q.src,
                            q.bytes / sizeof(uint64_t), l.stream);
                } else {
                    comm_xor_gf2(a, (const uint64_t *) ((char *) l.htmp
                                + k * accb[q.lane] + q.acc),
                            q.bytes / sizeof(uint64_t), l.stream);
                }
            }
        }
        for (hop const & h : step)
            accd[h.rblk] = 1;
        for (int d = 0; d < g; d++) {
            lane & l = s.lanes[d];
            CUDA_CHECK(cudaSetDevice(l.device));
            tl_mark(tl, l.stream, l.device, tl_name("hr xor", i), tl_k);
        }
    }
    int const ml = lane_of(s, v.device);
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaEventRecord(l.done, l.stream));
        CUDA_CHECK(cudaEventRecord(l.rd[0], l.stream));
    }
    for (hpiece const & q : L)
        if (q.owner)
            reader(q.owner, s.lanes[q.lane].rd[0]);
    /* the mirror writes of block 0 follow the mirror reads of the
     * mirror lane and the XOR of their own lane */
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        fork(l, l.copy);
        CUDA_CHECK(cudaStreamWaitEvent(l.copy, s.lanes[ml].done, 0));
    }
    size_t const own = (size_t) r * block_bytes;
    for (hpiece const & q : L) {
        if (q.off < own || q.off >= own + block_bytes)
            continue;
        lane & l = s.lanes[q.lane];
        CUDA_CHECK(cudaSetDevice(l.device));
        size_t const x = q.off - own;
        wait_range(l.copy, v.a, v.off + x, q.bytes);
    }
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        tl_mark(tl, l.copy, l.device, "hr mir0", tl_k);
    }
    for (hpiece const & q : L) {
        if (q.off < own || q.off >= own + block_bytes)
            continue;
        lane & l = s.lanes[q.lane];
        CUDA_CHECK(cudaSetDevice(l.device));
        size_t const x = q.off - own;
        CUDA_CHECK(cudaMemcpyPeerAsync(v.p + x, v.device, acc_at(q, q.off),
                    l.device, q.bytes, l.copy));
    }
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaEventRecord(l.mir, l.copy));
        tl_mark(tl, l.copy, l.device, "hr out", tl_k);
    }
    bwc_dev_stale_clear(pieces_of(v.a), v.off, block_bytes);
    join_lanes(s);
    for (lane & l : s.lanes)
        CUDA_CHECK(cudaStreamWaitEvent(s.pub, l.mir, 0));
    publish(s, v.a);
    bwc_dev_pieces * const p = pieces_of(v.a);
    p->nout = 0;
    for (hpiece const & q : L) {
        if (q.off < own || q.off >= own + block_bytes)
            continue;
        if (p->nout == BWC_DEV_PIECES_MAX)
            holder_fail("the result pieces pass BWC_DEV_PIECES_MAX");
        lane & l = s.lanes[q.lane];
        p->out[p->nout++] = { l.device, acc_at(q, q.off),
            v.off + (q.off - own), q.bytes, l.done, &l.stg[l.cur] };
    }
    p->out_settled = true;
    p->out_before = nullptr;
    p->out_end = v.off + block_bytes;
    p->wgen++;
    tl_mark(tl, s.pub, s.lanes[0].device, "hr pub", tl_k);
}

/* The destination of an allgather piece on lane d: the replica on the
 * lane device, or the mirror when the device holds it; without
 * replicas, the mirror on its own lane and agtmp on the others. */
char * holder_dst(comm_state const & s, dev_view const & v,
        bwc_dev_pieces const * reps, int d, size_t off, size_t bytes,
        size_t agoff)
{
    lane const & l = s.lanes[d];
    if (reps) {
        for (int k = 0; k < reps->nrep; k++) {
            bwc_dev_piece const & rp = reps->rep[k];
            if (rp.device != l.device)
                continue;
            if (v.off + off < rp.off || rp.off + rp.bytes < v.off + off + bytes)
                holder_fail("a replica does not cover an allgather piece");
            return rp.p + (v.off + off - rp.off);
        }
    }
    if (l.device == v.device)
        return v.p + off;
    if (reps)
        holder_fail("a lane device holds neither a replica nor the mirror");
    return (char *) l.agtmp + agoff;
}

/* The fan-out stream of a lane that writes the copy on a device: the
 * stream of its replica, or of the mirror when the device holds none. */
int fan_of(bwc_dev_pieces const * reps, int device)
{
    for (int k = 0; reps && k < reps->nrep; k++)
        if (reps->rep[k].device == device)
            return k + 1;
    return 0;
}

/* A piece of an allgather block, cut at the quarter boundaries of the
 * block: the holder lane of the piece on the rank of the block, the
 * quarter lane, the index of the piece in the block layout, and its
 * offset in the agtmp area of the quarter lane. */
struct hsub {
    size_t off, bytes;
    int from, to;
    size_t k;
    size_t agoff;
};

/* A write of the allgather on fan-out stream fan of lane: the arena
 * range [off, off + bytes) of the replica on device (-1: the mirror). */
struct hwrite {
    int lane;
    size_t fan;
    int device;
    size_t off, bytes;
};

/* The device of the memory holder_dst gives on lane d, -1 for the
 * mirror, -2 for the lane scratch. */
int dst_device(comm_state const & s, dev_view const & v,
        bwc_dev_pieces const * reps, int d)
{
    int const device = s.lanes[d].device;
    if (fan_of(reps, device))
        return device;
    return device == v.device ? -1 : -2;
}

cudaEvent_t wb_event(band_store & bs, size_t d, size_t i)
{
    if (bs.wbev.size() <= d)
        bs.wbev.resize(d + 1);
    while (bs.wbev[d].size() <= i) {
        cudaEvent_t wev;
        CUDA_CHECK(cudaEventCreateWithFlags(&wev, cudaEventDisableTiming));
        bs.wbev[d].push_back(wev);
    }
    return bs.wbev[d][i];
}

/* The allgather of a device buffer along ag_steps. The ranks swap the
 * layouts of their own blocks (the settled output segments, which after
 * a holder reduce-scatter are its result pieces, and the mirror for the
 * rest), and every piece is cut at the g quarters of its block. The
 * holder sends quarter j of its block to lane j of the peer, and a lane
 * forwards the quarters it holds on the same lane. A received quarter
 * lands in the replica or the mirror of its lane device, and the lane
 * copies it to the mirror and the other replicas. The own block takes
 * two hops: each holder copies quarter j to the replica of lane j, and
 * lane j copies it on to the other replicas. So every lane fans out
 * about one quarter, whichever lane holds the rows. */
void holder_ag(comm_state & s, dev_view const & v, size_t block_bytes,
        band_store & bs)
{
    int const n = s.njobs, r = s.rank;
    int const g = (int) s.lanes.size();
    bwc_dev_pieces const * const reps =
        replicas_of(v, (size_t) n * block_bytes);
    size_t const own_off = (size_t) r * block_bytes;
    std::vector<hpiece> const own = holder_layout(s, v, own_off,
            own_off + block_bytes, block_bytes, true);
    hlayout const mine = wire_of(own);
    std::vector<hlayout> lay(n);
    MPI_Allgather(&mine, sizeof(hlayout), MPI_BYTE, lay.data(),
            sizeof(hlayout), MPI_BYTE, s.mpi);
    size_t const qb = block_bytes / (size_t) g / sizeof(uint64_t)
        * sizeof(uint64_t);
    std::vector<std::vector<hsub>> sub(n);
    std::vector<size_t> agb(g, 0);
    for (int b = 0; b < n; b++) {
        size_t const b0 = (size_t) b * block_bytes;
        for (int64_t i = 0; i < lay[b].n; i++) {
            hwire const & w = lay[b].w[i];
            if (w.off / block_bytes != (uint64_t) b || w.lane < 0 || w.lane >= g)
                holder_fail("an allgather layout piece is outside its block or lanes");
            for (int j = 0; j < g; j++) {
                size_t const lo = std::max((size_t) w.off, b0 + (size_t) j * qb);
                size_t const hi = std::min((size_t) (w.off + w.bytes),
                        j == g - 1 ? b0 + block_bytes : b0 + (size_t) (j + 1) * qb);
                if (lo >= hi)
                    continue;
                sub[b].push_back({ lo, hi - lo, (int) w.lane, j, (size_t) i, agb[j] });
                if (b != r)
                    agb[j] += hi - lo;
            }
        }
    }
    /* the lane that fans out quarter x of the own block: the quarter
     * lane when its device holds a replica, else the holder */
    auto const route = [&](hsub const & x) {
        return reps && fan_of(reps, s.lanes[x.to].device) ? x.to : x.from;
    };
    bool const wbands = s.wbands && reps;
    std::vector<hwrite> pend;
    std::vector<bwc_dev_piece> wb;
    std::vector<size_t> wbi(g, 0);
    auto const note = [&](int d, size_t fan, int device, size_t off,
            size_t bytes) {
        if (wbands && device != -2)
            pend.push_back({ d, fan, device, v.off + off, bytes });
    };
    /* the copies of [off, off + bytes) from got on lane d to the replicas */
    auto const note_reps = [&](int d, size_t off, size_t bytes,
            const char * got) {
        for (int q = 0; wbands && q < reps->nrep; q++) {
            bwc_dev_piece const & rp = reps->rep[q];
            size_t const lo = std::max(v.off + off, rp.off);
            size_t const hi = std::min(v.off + off + bytes, rp.off + rp.bytes);
            if (lo < hi && (!got || rp.p + (lo - rp.off) != got + (lo - v.off - off)))
                note(d, (size_t) q + 1, rp.device, lo - v.off, hi - lo);
        }
    };
    /* one event per fan-out stream of lane d behind the writes noted
     * on it since the last records of the lane */
    auto const record = [&](int d) {
        lane & l = s.lanes[d];
        for (size_t k = 0; wbands && k < l.fans.size(); k++) {
            cudaEvent_t wev = nullptr;
            for (hwrite const & w : pend) {
                if (w.lane != d || w.fan != k)
                    continue;
                if (!wev) {
                    wev = wb_event(bs, (size_t) d, wbi[d]++);
                    CUDA_CHECK(cudaEventRecord(wev, l.fans[k]));
                }
                wb.push_back({ w.device, nullptr, w.off, w.bytes, wev, nullptr });
            }
        }
        pend.erase(std::remove_if(pend.begin(), pend.end(),
                    [d](hwrite const & w) { return w.lane == d; }), pend.end());
    };
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        if (!reps) {
            CUDA_CHECK(cudaStreamWaitEvent(l.stream, l.fanned[0], 0));
            CUDA_CHECK(cudaStreamWaitEvent(l.stream, l.fanned[1], 0));
            grow_behind(l, l.agtmp, l.agtmp_alloc, agb[d]);
        }
        wait_before(l.stream, v.a, v.off + own_off, block_bytes);
        for (int k = 0; reps && k < reps->nrep; k++)
            CUDA_CHECK(cudaStreamWaitEvent(l.stream, reps->rep[k].ready, 0));
        for (hpiece const & q : own)
            if (q.lane == d)
                wait_piece(l.stream, v, q);
        fork_fans(l);
        for (size_t k = 0; k < l.fans.size(); k++)
            tl_mark(tl, l.fans[k], l.device, tl_name("ha o1s f", (int) k), tl_k);
        for (hsub const & x : sub[r]) {
            if (x.from != d)
                continue;
            hpiece const & q = own[x.k];
            const uint8_t * const src = (const uint8_t *) q.src + (x.off - q.off);
            if (route(x) == d) {
                replicas_out(l, v, reps, x.off, x.bytes, src);
                note_reps(d, x.off, x.bytes, nullptr);
                continue;
            }
            lane const & to = s.lanes[x.to];
            CUDA_CHECK(cudaMemcpyPeerAsync(
                        holder_dst(s, v, reps, x.to, x.off, x.bytes, 0),
                        to.device, src, l.device, x.bytes,
                        l.fans[fan_of(reps, to.device)]));
            note(d, (size_t) fan_of(reps, to.device),
                    dst_device(s, v, reps, x.to), x.off, x.bytes);
        }
        record(d);
        for (size_t k = 0; k < l.fans.size(); k++) {
            CUDA_CHECK(cudaEventRecord(l.fanev[k], l.fans[k]));
            tl_mark(tl, l.fans[k], l.device, tl_name("ha o1e f", (int) k), tl_k);
        }
    }
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        std::vector<char> from(g, 0);
        for (hsub const & x : sub[r])
            if (x.to == d && x.from != d && route(x) == d)
                from[x.from] = 1;
        for (int e = 0; e < g; e++) {
            if (!from[e])
                continue;
            cudaEvent_t const ev = s.lanes[e].fanev[fan_of(reps, l.device)];
            for (cudaStream_t f : l.fans)
                CUDA_CHECK(cudaStreamWaitEvent(f, ev, 0));
        }
        for (size_t k = 0; k < l.fans.size(); k++)
            tl_mark(tl, l.fans[k], l.device, tl_name("ha o2s f", (int) k), tl_k);
    }
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        for (hsub const & x : sub[r]) {
            if (x.to != d || x.from == d || route(x) != d)
                continue;
            char * const got = holder_dst(s, v, reps, d, x.off, x.bytes, 0);
            for (int q = 0; q < reps->nrep; q++) {
                bwc_dev_piece const & rp = reps->rep[q];
                char * const to = rp.p + (v.off + x.off - rp.off);
                if (to == got)
                    continue;
                CUDA_CHECK(cudaMemcpyPeerAsync(to, rp.device, got,
                            l.device, x.bytes, l.fans[q + 1]));
            }
            note_reps(d, x.off, x.bytes, got);
        }
        record(d);
        for (size_t k = 0; k < l.fans.size(); k++)
            tl_mark(tl, l.fans[k], l.device, tl_name("ha o2e f", (int) k), tl_k);
        join_fans(l, l.fan);
        CUDA_CHECK(cudaEventRecord(l.blk[r], l.fan));
        tl_mark(tl, l.fan, l.device, "ha own", tl_k);
    }
    std::vector<size_t> ds(g);
    for (int d = 0; d < g; d++)
        ds[d] = (size_t) d;
    if (v.a->ready)
        CUDA_CHECK(cudaStreamWaitEvent(s.pub, (cudaEvent_t) v.a->ready, 0));
    band_ready(s, ds, bs, r);
    std::vector<std::vector<hop>> const steps = ag_steps(s);
    for (size_t i = 0; i < steps.size(); i++) {
        std::vector<hop> const & step = steps[i];
        NCCL_CHECK(ncclGroupStart());
        for (hop const & h : step) {
            for (hsub const & x : sub[h.sblk]) {
                bool const mine_blk = h.sblk == r;
                lane & l = s.lanes[mine_blk ? x.from : x.to];
                CUDA_CHECK(cudaSetDevice(l.device));
                char * const src = mine_blk
                    ? own[x.k].src + (x.off - own[x.k].off)
                    : holder_dst(s, v, reps, x.to, x.off, x.bytes, x.agoff);
                NCCL_CHECK(ncclSend(src, x.bytes, ncclUint8,
                            h.to * g + x.to, l.hnccl, l.stream));
            }
            for (hsub const & x : sub[h.rblk]) {
                lane & l = s.lanes[x.to];
                CUDA_CHECK(cudaSetDevice(l.device));
                int const peer = h.from == h.rblk ? x.from : x.to;
                NCCL_CHECK(ncclRecv(holder_dst(s, v, reps, x.to,
                                x.off, x.bytes, x.agoff),
                            x.bytes, ncclUint8, h.from * g + peer,
                            l.hnccl, l.stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        for (int d = 0; d < g; d++) {
            lane & l = s.lanes[d];
            CUDA_CHECK(cudaSetDevice(l.device));
            tl_mark(tl, l.stream, l.device, tl_name("ha net", i), tl_k);
            fork_fans(l);
            for (size_t k = 0; k < l.fans.size(); k++)
                tl_mark(tl, l.fans[k], l.device, tl_name(
                            ("ha b" + std::to_string(i) + "s f").c_str(),
                            (int) k), tl_k);
            for (hop const & h : step) {
                for (hsub const & x : sub[h.rblk]) {
                    if (x.to != d)
                        continue;
                    const uint8_t * const got = (const uint8_t *) holder_dst(
                            s, v, reps, d, x.off, x.bytes, x.agoff);
                    note(d, (size_t) fan_of(reps, l.device),
                            dst_device(s, v, reps, d), x.off, x.bytes);
                    if (got != (const uint8_t *) v.p + x.off) {
                        CUDA_CHECK(cudaMemcpyPeerAsync(v.p + x.off, v.device,
                                    got, l.device, x.bytes, l.fans[0]));
                        note(d, 0, -1, x.off, x.bytes);
                    }
                    for (int q = 0; reps && q < reps->nrep; q++) {
                        bwc_dev_piece const & rp = reps->rep[q];
                        char * const to = rp.p + (v.off + x.off - rp.off);
                        if ((const uint8_t *) to == got)
                            continue;
                        CUDA_CHECK(cudaMemcpyPeerAsync(to, rp.device, got,
                                    l.device, x.bytes, l.fans[q + 1]));
                    }
                    if (reps)
                        note_reps(d, x.off, x.bytes, (const char *) got);
                }
            }
            record(d);
            for (size_t k = 0; k < l.fans.size(); k++)
                tl_mark(tl, l.fans[k], l.device, tl_name(
                            ("ha b" + std::to_string(i) + "e f").c_str(),
                            (int) k), tl_k);
            join_fans(l, l.fan);
            for (hop const & h : step) {
                CUDA_CHECK(cudaEventRecord(l.blk[h.rblk], l.fan));
                tl_mark(tl, l.fan, l.device, tl_name("ha blk", h.rblk), tl_k);
            }
        }
        for (hop const & h : step)
            band_ready(s, ds, bs, h.rblk);
    }
    for (int b = 0; b < n; b++)
        if (b != r)
            bwc_dev_stale_clear(pieces_of(v.a),
                    v.off + (size_t) b * block_bytes, block_bytes);
    for (int d = 0; d < g; d++) {
        lane & l = s.lanes[d];
        CUDA_CHECK(cudaSetDevice(l.device));
        CUDA_CHECK(cudaEventRecord(l.rd[1], l.stream));
        CUDA_CHECK(cudaEventRecord(l.fanned[l.cur], l.fan));
    }
    for (hpiece const & q : own)
        if (q.owner)
            reader(q.owner, s.lanes[q.lane].rd[1]);
    publish(s, v.a);
    written(v, reps != nullptr);
    if (wbands) {
        bwc_dev_pieces * const p = pieces_of(v.a);
        if (wb.size() > BWC_DEV_WBANDS_MAX)
            holder_fail("the allgather writes more pieces than the write records hold");
        std::copy(wb.begin(), wb.end(), p->wb);
        p->nwb = (int) wb.size();
        p->wb_wgen = p->wgen;
    }
    tl_mark(tl, s.pub, s.lanes[0].device, "ha pub", tl_k);
    if (v.off == 0) {
        for (int b = 0; b < n; b++)
            bs.bands[b] = { (size_t) b * block_bytes, block_bytes, bs.ev[b] };
        v.a->bands = bs.bands.data();
        v.a->nbands = n;
    }
}

/* Under BWC_NCCL_HOLDER, the host buffers of the program start (the
 * communication benchmark and the broadcasts of the loaded vectors)
 * take the host staging, and the plugin says so once per leg; a host
 * buffer after the first holder exchange aborts. */
void holder_host(comm_state const & s, const char * leg)
{
    static bool said[2];
    int const i = leg[0] == 'a';
    if (!s.holder)
        return;
    if (s.holder_started) {
        fprintf(stderr, "bwccomm_nccl: BWC_NCCL_HOLDER: a host buffer of the"
                " %s after the holder exchange started\n", leg);
        abort();
    }
    if (said[i])
        return;
    said[i] = true;
    fprintf(stderr, "bwccomm_nccl: BWC_NCCL_HOLDER: a host buffer of the %s"
            " takes the host staging\n", leg);
}

/* Reduce-scatter over the lanes, along rs_steps. The lane stages its
 * slice of the blocks on its copy stream, in the order the ring
 * consumes them, with one block event each; the lane stream waits for
 * the block events right before the step that sends the blocks. So
 * the first send goes out when its block of the product is complete,
 * while the product still computes the other blocks. The result of
 * every lane goes back to the mirror at the end. Every copy that
 * touches the mirror sits before the NCCL steps or after them, so no
 * copy of one lane queues behind the steps of another lane. */
void reduce_scatter_xor(void * const * bufs, int nbufs,
        size_t block_bytes, MPI_Comm comm)
{
    comm_state & s = get_state(comm);
    /* the XOR over one job is the identity; the staging round-trip
     * below would move every buffer to the staging area and back */
    if (s.njobs == 1)
        return;
    ASSERT_ALWAYS(block_bytes % sizeof(uint64_t) == 0);
    int const n = s.njobs;
    int host_sync = 0;
    std::vector<size_t> const ds = active_lanes(s, block_bytes);
    /* the staging order is the send order of the ring, rank - 1,
     * rank - 2, ... */
    auto const blk_at = [&](int i) { return (s.rank - 1 - i + 2 * n) % n; };
    std::vector<std::vector<hop>> const steps = rs_steps(s);
    for (int w = 0; w < nbufs; w++) {
        dev_view const v = dev_of(bufs[w]);
        if (s.holder && v.p) {
            s.holder_started = true;
            holder_rs(s, v, block_bytes);
            continue;
        }
        if (!v.p)
            holder_host(s, "reduce-scatter");
        if (v.p)
            request_order(v, n, block_bytes, blk_at);
        for (size_t d : ds) {
            lane & l = s.lanes[d];
            slice const sl = slice_of(s, d, block_bytes);
            CUDA_CHECK(cudaSetDevice(l.device));
            reserve(s, l, sl.bytes);
            /* the staging half is free once the ring before this one
             * and the fan-out that read the half are done with it */
            fork(l, l.copy);
            fresh_half(l);
            tl_mark(tl, l.copy, l.device, "rs free", tl_k);
            fork_stage_streams(s, l);
            for (int i = 0; i < n; i++) {
                cudaStream_t const st = stage_stream(s, l, i);
                stage_block(l, st, sl, v, bufs[w], blk_at(i), block_bytes);
                CUDA_CHECK(cudaEventRecord(l.blk[blk_at(i)], st));
                tl_mark(tl, st, l.device, tl_name("rs stg", i), tl_k);
            }
            join_stage_streams(s, l);
            if (v.p)
                segments_read(l.rd[0], l.copy, v.a);
        }
        for (size_t i = 0; i < steps.size(); i++) {
            std::vector<hop> const & step = steps[i];
            for (size_t d : ds) {
                lane & l = s.lanes[d];
                CUDA_CHECK(cudaSetDevice(l.device));
                for (hop const & h : step)
                    CUDA_CHECK(cudaStreamWaitEvent(l.stream, l.blk[h.sblk], 0));
            }
            NCCL_CHECK(ncclGroupStart());
            for (size_t d : ds) {
                lane & l = s.lanes[d];
                slice const sl = slice_of(s, d, block_bytes);
                CUDA_CHECK(cudaSetDevice(l.device));
                for (size_t k = 0; k < step.size(); k++) {
                    hop const & h = step[k];
                    NCCL_CHECK(ncclSend(staging(l) + (size_t) h.sblk * sl.bytes,
                                sl.bytes, ncclUint8, h.to, l.nccl, l.stream));
                    NCCL_CHECK(ncclRecv(l.tmp + k * sl.bytes, sl.bytes,
                                ncclUint8, h.from, l.nccl, l.stream));
                }
            }
            NCCL_CHECK(ncclGroupEnd());
            for (size_t d : ds) {
                lane & l = s.lanes[d];
                slice const sl = slice_of(s, d, block_bytes);
                CUDA_CHECK(cudaSetDevice(l.device));
                tl_mark(tl, l.stream, l.device, tl_name("rs net", i), tl_k);
                for (size_t k = 0; k < step.size(); k++) {
                    hop const & h = step[k];
                    CUDA_CHECK(cudaStreamWaitEvent(l.stream, l.blk[h.rblk], 0));
                    comm_xor_gf2((uint64_t *) (staging(l) + (size_t) h.rblk * sl.bytes),
                            (const uint64_t *) (l.tmp + k * sl.bytes),
                            sl.bytes / sizeof(uint64_t), l.stream);
                }
                tl_mark(tl, l.stream, l.device, tl_name("rs xor", i), tl_k);
            }
        }
        for (size_t d : ds) {
            lane & l = s.lanes[d];
            slice const sl = slice_of(s, d, block_bytes);
            CUDA_CHECK(cudaSetDevice(l.device));
            /* the result of the reduce-scatter is the first block of
             * the buffer */
            size_t const off = sl.off;
            uint8_t * const res = staging(l) + (size_t) s.rank * sl.bytes;
            if (v.p) {
                /* the mirror copy runs on the copy stream, so the lane
                 * stream (and the allgather, which reads the result
                 * from the staging half) does not wait for it; the
                 * publication waits for it through mir. The write must
                 * follow every queued access of its range of the
                 * mirror; the band events of the product carry the
                 * accesses before the product */
                fork(l, l.copy);
                wait_range(l.copy, v.a, v.off + off, sl.bytes);
                CUDA_CHECK(cudaMemcpyPeerAsync(v.p + off, v.device, res,
                            l.device, sl.bytes, l.copy));
                CUDA_CHECK(cudaEventRecord(l.mir, l.copy));
                tl_mark(tl, l.copy, l.device, "rs out", tl_k);
            } else {
                CUDA_CHECK(cudaMemcpyAsync((char *) bufs[w] + off, res,
                            sl.bytes, cudaMemcpyDeviceToHost, l.stream));
                host_sync = 1;
            }
        }
        if (v.p) {
            join_lanes(s);
            for (size_t d : ds)
                CUDA_CHECK(cudaStreamWaitEvent(s.pub, s.lanes[d].mir, 0));
            publish(s, v.a);
            results_out(s, ds, v, block_bytes);
            tl_mark(tl, s.pub, s.lanes[0].device, "rs pub", tl_k);
        }
    }
    if (tl_on() && tl_k >= TL_K0 && tl_k < TL_K0 + TL_N)
        fprintf(stderr, "TH nccl host rs-return %lu %.6f\n", tl_k, tl_now());
    /* a host buffer must be settled at return; a mirror carries its
     * completion in the ready marker */
    if (host_sync)
        sync_lanes(s);
}

/* Allgather along ag_steps, with one completion event per block. A
 * buffer with a current mirror gets each received block with a device
 * copy when the block arrives, and its arena publishes one band per
 * block, so the next product pulls a block while the ring still
 * delivers the later blocks. A host buffer gets per-block host copies
 * and one host wait at the end. Every rank runs the same NCCL call
 * sequence whatever the mirror states are. The lanes run each step
 * together. The copies of a received block run on the fan-out stream
 * of the lane, so the lane stream goes on to the next step at once;
 * the block event of the lane fires on the fan-out stream, and the
 * fan-out stream marks the staging half read after the last block,
 * for the exchange that takes the half next. */
void allgather(void * const * bufs, int nbufs,
        size_t block_bytes, MPI_Comm comm)
{
    comm_state & s = get_state(comm);
    /* the allgather over one job is the identity; the publish below
     * would still replace the piece bands of the buffer with one
     * whole-buffer event, and the next product would wait for the
     * whole write instead of its own source ranges */
    if (s.njobs == 1)
        return;
    ASSERT_ALWAYS(block_bytes % sizeof(uint64_t) == 0);
    int const n = s.njobs;
    int host_sync = 0;
    std::vector<size_t> const ds = active_lanes(s, block_bytes);
    std::vector<std::vector<hop>> const steps = ag_steps(s);
    for (int w = 0; w < nbufs; w++) {
        dev_view const v = dev_of(bufs[w]);
        band_store * const bs = v.p ? &store_of(s, v.a) : nullptr;
        if (s.holder && v.p) {
            s.holder_started = true;
            holder_ag(s, v, block_bytes, *bs);
            continue;
        }
        if (!v.p)
            holder_host(s, "allgather");
        bwc_dev_pieces const * const reps =
            replicas_of(v, (size_t) n * block_bytes);
        for (size_t d : ds) {
            lane & l = s.lanes[d];
            slice const sl = slice_of(s, d, block_bytes);
            CUDA_CHECK(cudaSetDevice(l.device));
            reserve(s, l, sl.bytes);
            fresh_half(l);
            tl_mark(tl, l.stream, l.device, "ag free", tl_k);
            uint8_t * const own = staging(l) + (size_t) s.rank * sl.bytes;
            size_t const own_off = (size_t) s.rank * block_bytes + sl.off;
            if (v.p) {
                arena_in(l, l.stream, v.a, v.off + own_off, sl.bytes, own);
                segments_read(l.rd[1], l.stream, v.a);
                /* the block writes below must follow every queued
                 * mirror and replica access */
                wait_before(l.stream, v.a, v.off + own_off, sl.bytes);
                for (int k = 0; reps && k < reps->nrep; k++)
                    CUDA_CHECK(cudaStreamWaitEvent(l.stream,
                                reps->rep[k].ready, 0));
                fork_fans(l);
                replicas_out(l, v, reps, own_off, sl.bytes, own);
                join_fans(l, l.fan);
                CUDA_CHECK(cudaEventRecord(l.blk[s.rank], l.fan));
                tl_mark(tl, l.fan, l.device, "ag own", tl_k);
                tl_mark(tl, l.stream, l.device, "ag stg", tl_k);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(own,
                            (const uint8_t *) bufs[w] + own_off,
                            sl.bytes, cudaMemcpyHostToDevice, l.stream));
            }
        }
        if (v.p) {
            /* the publication events must also cover the accesses the
             * old ready marker holds */
            if (v.a->ready)
                CUDA_CHECK(cudaStreamWaitEvent(s.pub,
                            (cudaEvent_t) v.a->ready, 0));
            band_ready(s, ds, *bs, s.rank);
        }
        for (size_t i = 0; i < steps.size(); i++) {
            std::vector<hop> const & step = steps[i];
            NCCL_CHECK(ncclGroupStart());
            for (size_t d : ds) {
                lane & l = s.lanes[d];
                slice const sl = slice_of(s, d, block_bytes);
                CUDA_CHECK(cudaSetDevice(l.device));
                for (hop const & h : step) {
                    NCCL_CHECK(ncclSend(staging(l) + (size_t) h.sblk * sl.bytes,
                                sl.bytes, ncclUint8, h.to, l.nccl, l.stream));
                    NCCL_CHECK(ncclRecv(staging(l) + (size_t) h.rblk * sl.bytes,
                                sl.bytes, ncclUint8, h.from, l.nccl, l.stream));
                }
            }
            NCCL_CHECK(ncclGroupEnd());
            for (size_t d : ds) {
                lane & l = s.lanes[d];
                slice const sl = slice_of(s, d, block_bytes);
                CUDA_CHECK(cudaSetDevice(l.device));
                if (v.p) {
                    fork_fans(l);
                    tl_mark(tl, l.stream, l.device, tl_name("ag net", i), tl_k);
                }
                for (hop const & h : step) {
                    size_t const off = (size_t) h.rblk * block_bytes + sl.off;
                    uint8_t * const got = staging(l) + (size_t) h.rblk * sl.bytes;
                    if (v.p) {
                        arena_out(l, v, reps, off, sl.bytes, got);
                        join_fans(l, l.fan);
                        CUDA_CHECK(cudaEventRecord(l.blk[h.rblk], l.fan));
                        tl_mark(tl, l.fan, l.device, tl_name("ag blk", h.rblk), tl_k);
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) bufs[w] + off,
                                    got, sl.bytes, cudaMemcpyDeviceToHost,
                                    l.stream));
                        host_sync = 1;
                    }
                }
            }
            for (hop const & h : step)
                if (v.p)
                    band_ready(s, ds, *bs, h.rblk);
        }
        if (v.p) {
            for (size_t d : ds) {
                lane & l = s.lanes[d];
                CUDA_CHECK(cudaSetDevice(l.device));
                CUDA_CHECK(cudaEventRecord(l.fanned[l.cur], l.fan));
            }
            /* the publication stream has waited for every band, so
             * its event covers the whole exchange */
            publish(s, v.a);
            written(v, reps != nullptr);
            tl_mark(tl, s.pub, s.lanes[0].device, "ag pub", tl_k);
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
    if (tl_on() && tl_k >= TL_K0 && tl_k < TL_K0 + TL_N)
        fprintf(stderr, "TH nccl host ag-return %lu %.6f\n", tl_k, tl_now());
    tl_flush(tl, ++tl_k);
    if (host_sync)
        sync_lanes(s);
}

bwc_comm_plugin_api const api = { reduce_scatter_xor, allgather, 1 };

} // namespace

extern "C" bwc_comm_plugin_api const * bwc_comm_solib_reach_api()
{
    return &api;
}
