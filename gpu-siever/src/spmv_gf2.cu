#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <unistd.h>

#include <algorithm>
#include <functional>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "bwc_device_arena.hpp"
#include "bwc_dev_pieces.h"
#include "spmv_gf2.h"
#include "timeline.h"

static tl_state tlg = { {}, 0, nullptr, {}, "spmv" };

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                cudaGetErrorString(err_));                              \
        abort();                                                        \
    }                                                                   \
} while (0)


/* One instance may spread its rows over SPMV_GF2_NGPU devices.  Each
 * device holds a contiguous slice of the rows, split so that every
 * slice carries about the same estimated gather work: coefficients
 * plus SPMV_GF2_CHUNK_COST per gather chunk.  Timed products refine
 * the split: trials at the setup (SPMV_GF2_SPLIT_ROUNDS), then the
 * kernel times of the real products before call SPMV_GF2_RESPLIT_AT.
 *
 * The columns split into bands, so that the vector segment one gather
 * pass reads stays in the L2 cache.  Each device stores one CSR block
 * per band: the entries of band b in row order, with the start of row
 * i at rowptr[b * (nrows + 1) + i].  The rowptr values index the
 * shared cols array.  SPMV_GF2_BAND_MB sets the band size; a value of
 * 0 or less gives one band, which is the unsplit layout. */

/* one gather work item: len entries of row, at start in cols; 16-byte
 * aligned so a warp reads one descriptor with one vector load */
struct __align__(16) gather_chunk {
    uint64_t start;
    uint32_t len;
    uint32_t row;
};

/* The rows of one output block on a device: [r0, r1) of the device
 * rows, the chunk range of each band, and the event after its gather. */
struct gather_seg {
    int blk;
    unsigned int r0, r1;
    std::vector<std::pair<uint64_t, uint64_t>> c;   /* per band */
    cudaEvent_t ev;
};

struct spmv_dev {
    int device;
    unsigned int row0;
    unsigned int nrows;
    uint64_t * rowptr;      /* device, nbands * (nrows + 1) */
    uint32_t * cols;        /* device, one entry per coefficient */
    /* the gather chunk list; chunk0[b] is the first chunk of band b */
    gather_chunk * chunks;  /* device, one descriptor per chunk */
    std::vector<uint64_t> chunk0;   /* host, nbands + 1 */
    std::vector<uint32_t> chunk_row;   /* host, the device row of each chunk */
    /* host copies for SPMV_GF2_CHUNK_ORDER=col: the chunks in row
     * order and the first column of each */
    std::vector<gather_chunk> chunk_h;
    std::vector<uint32_t> chunk_col;
    /* the device rows cut at the block boundaries of seg_bytes, in row
     * order; one segment when seg_bytes is zero */
    std::vector<gather_seg> segs;
    size_t seg_bytes = SIZE_MAX;
    uint64_t * vtmp;        /* device, max_dim words, multi-GPU only */
    cudaStream_t stream;
    cudaStream_t cstream;   /* source pieces, under the kernels */
    cudaEvent_t ev;         /* last product step on this stream */
    cudaEvent_t cev;        /* last source piece on cstream */
    std::vector<cudaEvent_t> pe; /* timing events around the probed kernels */
    size_t npe = 0;         /* events recorded in the probed call */
    double busy = 0;        /* kernel time of the probed calls, seconds */
    int l2cap = 0;          /* max access policy window bytes (SPMV_GF2_L2_MB) */
};

/* The raw row stream: row i is its length, then its columns, at word
 * rowptr[i] + i; rowptr[i] counts the columns of the rows before it. */
struct row_stream {
    uint32_t * rows;
    std::vector<uint64_t> rowptr;
    uint32_t * cols(unsigned int i) const
    { return rows + rowptr[i] + i + 1; }
    uint64_t len(unsigned int i) const { return rowptr[i + 1] - rowptr[i]; }
};

struct spmv_gf2 {
    std::vector<spmv_dev> devs;
    unsigned int nrows;
    /* the row stream and the estimated work, kept until the split on
     * the kernel times of the probed calls [probe0, probe1) */
    std::vector<uint32_t> rows;
    row_stream rs;
    std::vector<uint64_t> work;
    unsigned long probe0 = 0, probe1 = 0;
    unsigned int nbands;
    unsigned int bandcols;  /* source columns per band */
    unsigned int words;     /* 64-bit words per vector element */
    /* host driving time per phase; with a device arena destination the
     * product only enqueues, and the device work settles behind the
     * ready event of the arena */
    double t_h2d = 0, t_kernel = 0, t_d2h = 0;
    unsigned long calls = 0;
    unsigned long rep_hits = 0;  /* products that read the source replicas */
    unsigned int max_dim;
    /* The per-device scratch of each vector arena (nullptr: the host
     * path), max_dim words per device: vsrc holds the source replicas
     * of the arena, vdst the output rows of a product into it.  One
     * scratch per arena, so that a product never writes the memory
     * that an exchange of another arena still reads. */
    std::map<const bwc_device_arena *, std::vector<uint64_t *>> vsrc, vdst;
};

static uint64_t ** scratch(spmv_gf2 * m,
        std::map<const bwc_device_arena *, std::vector<uint64_t *>> & role,
        const bwc_device_arena * a)
{
    std::vector<uint64_t *> & v = role[a];
    if (v.empty()) {
        v.resize(m->devs.size());
        for (size_t d = 0; d < v.size(); d++) {
            CUDA_CHECK(cudaSetDevice(m->devs[d].device));
            CUDA_CHECK(cudaMalloc(&v[d],
                        (size_t) m->max_dim * m->words * sizeof(uint64_t)));
        }
    }
    return v.data();
}

/* Device mirrors for the bwc vector arenas.  When a communication
 * plugin accepts device arenas, each vector that reaches the product
 * functions gets a mirror on the first device of the context, and the
 * products read and write the mirrors instead of staging the vectors
 * through the host on every call.  The registry in the bwc core tracks
 * which copy is current.
 *
 * The bwc threads that share one source vector reach the product
 * functions concurrently, each through its own instance.  Each arena
 * carries its own lock, which serializes its state changes; an
 * operation on two arenas takes both locks at once.  One process lock
 * serializes mirror creation and growth.  The instance that creates a
 * mirror places it on its own first device, so another instance can
 * see a mirror on a device it does not drive; it then reads the
 * mirror with a peer copy and never writes it. */

static std::mutex arena_mutex;

/* mirror() page-locks the host range of each arena so the copies in
 * this file run at full link speed, and arena_release drops the lock.
 * The lock lifetime equals the arena lifetime, so a vector that bwc
 * frees leaves no lock behind on pages that a later allocation may
 * reuse.  When the platform refuses the lock, the copies stay
 * pageable.  Returns the locked size: bytes, or zero on refusal. */
static size_t lock_host(void * p, size_t bytes)
{
    cudaError_t const e = cudaHostRegister(p, bytes,
            cudaHostRegisterPortable);
    if (e == cudaSuccess)
        return bytes;
    if (e != cudaErrorHostMemoryAlreadyRegistered)
        fprintf(stderr, "spmv_gf2: cudaHostRegister: %s\n",
                cudaGetErrorString(e));
    cudaGetLastError();
    return 0;
}

/* Page-lock the host vectors on the path without arenas, so the
 * per-iteration copies run at full link speed.  The map records one
 * entry per registration: the key is the first page, the value the
 * end page and the locked size (zero when the platform refused, so
 * one span gets at most one attempt).  The driver rejects a copy
 * whose host range straddles two registrations, so a pin that
 * overlaps existing spans replaces them with one registration that
 * covers the union.  The no-arena copy sections hold pin_copy_mutex
 * shared until they return; the replacement holds it exclusive and
 * drains the devices the instances drive (pin_devices), so the old
 * spans release with no copy in flight and no new copy enqueues
 * before the union is registered.  The arena paths hold their own
 * registrations, which pin() never touches, and every arena runs on
 * an instance device, so pin_devices covers the arena streams too. */
static std::mutex pin_mutex;
static std::shared_mutex pin_copy_mutex;
static std::set<int> pin_devices;
struct pin_span { uintptr_t end; size_t locked; };
static std::map<uintptr_t, pin_span> pin_spans;

static void pin_quiesce()
{
    int prev = 0;
    CUDA_CHECK(cudaGetDevice(&prev));
    for (int const d : pin_devices) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(prev));
}

static void pin(const void * p, size_t bytes)
{
    static size_t const page = sysconf(_SC_PAGESIZE);
    uintptr_t lo = (uintptr_t) p / page * page;
    uintptr_t hi = ((uintptr_t) p + bytes + page - 1) / page * page;

    std::lock_guard<std::mutex> const lock(pin_mutex);
    auto it = pin_spans.lower_bound(lo);
    if (it != pin_spans.begin() && std::prev(it)->second.end > lo)
        --it;
    if (it != pin_spans.end() && it->first <= lo && hi <= it->second.end)
        return;
    std::unique_lock<std::shared_mutex> copies(pin_copy_mutex,
            std::defer_lock);
    while (it != pin_spans.end() && it->first < hi) {
        lo = std::min(lo, it->first);
        hi = std::max(hi, it->second.end);
        if (it->second.locked) {
            if (!copies.owns_lock()) {
                copies.lock();
                pin_quiesce();
            }
            CUDA_CHECK(cudaHostUnregister((void *) it->first));
        }
        it = pin_spans.erase(it);
    }
    pin_spans[lo] = { hi, lock_host((void *) lo, hi - lo) };
}

/* A copy of a mirror on another device, current while the mirror
 * keeps the write generation the copy was taken at. */
struct replica {
    uint64_t * p = nullptr;
    size_t bytes = 0;
    uint64_t wgen = UINT64_MAX;
};

struct arena_ctx {
    bwc_dev_pieces pieces {};   /* first: the plugin finds it at owner */
    std::mutex mtx;
    std::atomic<size_t> ready_bytes { 0 }; /* mirror size, set after growth */
    int device;
    cudaStream_t stream;
    size_t locked = 0;          /* host bytes under the page lock */
    uint64_t * doff = nullptr;  /* device, word gather offsets */
    uint64_t * dval = nullptr;  /* device, gathered words */
    uint64_t * hval = nullptr;  /* host, gathered words */
    size_t nwords = 0;
    uint64_t * xbuf = nullptr;  /* device, staged xor source */
    size_t xbytes = 0;
    uint64_t * qval = nullptr;  /* device, queued word captures */
    uint64_t * qoff = nullptr;  /* device, the capture offsets */
    uint64_t * qsrc = nullptr;  /* host copy the offset uploads read */
    size_t qwords = 0;
    std::map<int, replica> reps; /* per device, a copy for the combine */
    uint64_t * obuf = nullptr;  /* device, combine result and coefficients */
    size_t owords = 0;
    cudaEvent_t ev;             /* the ready event of the arena */
    cudaStream_t cstream;       /* peer piece copies, under the xors */
    cudaEvent_t cev;            /* the last piece copy on cstream */
    cudaEvent_t bev = nullptr;  /* band event, product output tail */
    cudaEvent_t pre = nullptr;  /* the accesses before a copy in */
    std::map<int, cudaEvent_t> pev;     /* per device, product output piece */
    std::vector<bwc_device_band> bands; /* published band descriptors */
    std::vector<cudaEvent_t> wbev;      /* piece events, dev_copy/xor */
    std::vector<bwc_device_band> wbands; /* written pieces, sorted */
};

/* An operation on two arenas takes both locks at once.  The two
 * operands of a copy or xor can live in one arena, which then has one
 * lock. */
struct arena_lock2 {
    std::unique_lock<std::mutex> d, s;
    arena_lock2(arena_ctx * cd, arena_ctx * cs)
        : d(cd->mtx, std::defer_lock), s(cs->mtx, std::defer_lock)
    {
        if (cd == cs)
            d.lock();
        else
            std::lock(d, s);
    }
};

/* An operation on one destination and several sources takes every
 * distinct lock once, in address order. */
struct arena_lockn {
    std::vector<std::unique_lock<std::mutex>> locks;
    arena_lockn(std::vector<arena_ctx *> ctxs)
    {
        std::sort(ctxs.begin(), ctxs.end());
        ctxs.erase(std::unique(ctxs.begin(), ctxs.end()), ctxs.end());
        for (arena_ctx * c : ctxs)
            locks.emplace_back(c->mtx);
    }
};

/* The band size in bytes; the default stays under a 126 MB device L2
 * cache.  The gather splits the source vector into bands of
 * this size, and the arena copy and xor operations split their work
 * into pieces of the same size, so the published bands match the
 * ranges the gather consumes. */
static size_t env_band_bytes()
{
    const char * s = getenv("SPMV_GF2_BAND_MB");
    long const mb = s ? atol(s) : 96;
    return mb > 0 ? (size_t) mb << 20 : SIZE_MAX;
}

/* SPMV_GF2_ADDMUL64=nibble selects the nibble-table combine kernel */
static bool env_addmul64_nibble()
{
    const char * s = getenv("SPMV_GF2_ADDMUL64");
    return s && !strcmp(s, "nibble");
}

/* The ready field of an arena holds the event that fires when every
 * queued device access to the mirror is complete (see
 * bwc_device_arena.hpp).  This backend stores only the arena's own
 * event there, so the handle stays valid for the arena lifetime; the
 * communication plugin stores its own process-lifetime event.  Each
 * event is always recorded on one stream, so a re-record only moves
 * the fire point later.  The callers hold the arena lock. */
static void ready_wait(bwc_device_arena * a, cudaStream_t s)
{
    if (a->ready)
        CUDA_CHECK(cudaStreamWaitEvent(s, (cudaEvent_t) a->ready, 0));
}

/* drop the published bands and the piece store behind them */
static void bands_drop(bwc_device_arena * a)
{
    ((arena_ctx *) a->owner)->wbands.clear();
    a->bands = nullptr;
    a->nbands = 0;
}

/* store a fresh ready event for the accesses queued on the arena
 * stream; the bands stay valid, for a read or for a write that
 * published its own bands (see the band contract) */
static void ready_mark(bwc_device_arena * a)
{
    auto * c = (arena_ctx *) a->owner;
    CUDA_CHECK(cudaEventRecord(c->ev, c->stream));
    a->ready = c->ev;
}

/* Count a write of the mirror data: the output segments and the
 * source replicas of the arena no longer match it. */
static void mirror_written(bwc_device_arena * a)
{
    auto * c = (arena_ctx *) a->owner;
    c->pieces.nout = 0;
    c->pieces.wgen++;
}

/* publish the accesses queued on the arena stream after a write
 * without bands */
static void ready_record(bwc_device_arena * a)
{
    ready_mark(a);
    mirror_written(a);
    bands_drop(a);
}

/* Fold the accesses that b->ready covers into a->ready.  A product
 * that reads mirror a and writes mirror b calls this at the end, so
 * that a later write to a waits for the queued reads.  The new event
 * also waits the old one, so the accesses it covered stay covered
 * when the reader waited only on band events.  A read keeps the bands
 * of a: each band event still covers its range. */
static void ready_follow(bwc_device_arena * a, bwc_device_arena * b)
{
    auto * c = (arena_ctx *) a->owner;
    ready_wait(a, c->stream);
    ready_wait(b, c->stream);
    ready_mark(a);
}

/* host wait until the mirror is settled */
static void ready_sync(bwc_device_arena * a)
{
    if (!a->ready)
        return;
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t) a->ready));
    a->ready = nullptr;
    bands_drop(a);
}

/* Make the stream wait until the mirror range [off, off + bytes) is
 * complete: each band event that covers a part of the range, and the
 * ready event when the bands stop short of the range end.  The caller
 * holds the arena lock. */
static void ready_wait_range(bwc_device_arena * a, cudaStream_t s,
        size_t off, size_t bytes)
{
    size_t covered = 0;         /* the bands are contiguous from zero */
    for (int i = 0; i < a->nbands; i++) {
        bwc_device_band const & b = a->bands[i];
        if (b.ready && b.off < off + bytes && off < b.off + b.bytes)
            CUDA_CHECK(cudaStreamWaitEvent(s, (cudaEvent_t) b.ready, 0));
        covered = b.off + b.bytes;
    }
    if (covered < off + bytes)
        ready_wait(a, s);
}

/* Begin a banded write.  A foreign ready event means another module
 * wrote the mirror, so the stored pieces no longer cover the last
 * write of their ranges and drop. */
static void band_begin(bwc_device_arena * a)
{
    auto * c = (arena_ctx *) a->owner;
    if (a->ready != (void *) c->ev)
        bands_drop(a);
}

/* Publish the write of the mirror range [off, off + bytes), queued on
 * the arena stream, as a band: a consumer then starts on the range
 * when its write completes instead of behind the whole ready event.
 * The pieces of concurrent writers arrive in any order, so the store
 * keeps them sorted and the arena publishes the contiguous prefix
 * from offset zero, as the band contract asks; ready covers the
 * rest.  A piece that overlaps a stored piece begins a new write
 * generation.  The caller holds the arena lock. */
static void band_publish(bwc_device_arena * a, size_t off, size_t bytes)
{
    auto * c = (arena_ctx *) a->owner;
    auto & w = c->wbands;
    for (bwc_device_band const & b : w)
        if (b.off < off + bytes && off < b.off + b.bytes) {
            w.clear();
            break;
        }
    if (c->wbev.size() == w.size()) {
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        c->wbev.push_back(e);
    }
    cudaEvent_t const e = c->wbev[w.size()];
    CUDA_CHECK(cudaEventRecord(e, c->stream));
    auto const at = std::lower_bound(w.begin(), w.end(), off,
            [](bwc_device_band const & b, size_t o) { return b.off < o; });
    w.insert(at, { off, bytes, e });
    size_t end = 0;
    int n = 0;
    for (bwc_device_band const & b : w) {
        if (b.off != end)
            break;
        end = b.off + b.bytes;
        n++;
    }
    a->bands = n ? w.data() : nullptr;
    a->nbands = n;
}

/* the first device writes the mirror in place; its stream waits for
 * the queued accesses before the kernels start */
static void ready_wait_locked(bwc_device_arena * a, cudaStream_t s)
{
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    ready_wait(a, s);
}

__global__ void xor_kernel(uint64_t * dst, const uint64_t * src,
                           unsigned int n);
__global__ void permute_kernel(uint64_t * dst, const uint32_t * di,
                               const uint64_t * src, const uint32_t * si,
                               size_t total, size_t words);
__global__ void dotprod64_kernel(const uint64_t * y, size_t words,
                                 const uint64_t * c, size_t n,
                                 uint64_t * out);
#define ADDMUL64_MAX 4
struct addmul64_srcs { const uint64_t * p[ADDMUL64_MAX]; };
__global__ void addmul64_kernel(uint64_t * dst, addmul64_srcs s,
                                const uint64_t * f, unsigned int nsrc,
                                unsigned int n);
__global__ void addmul64_nibble_kernel(uint64_t * dst, addmul64_srcs s,
                                       const uint64_t * f, unsigned int nsrc,
                                       unsigned int n);

static bwc_device_arena * mirror_on(int device, const void * host,
        size_t bytes);
static void arena_make_dev(bwc_device_arena * a, size_t filled);

static void arena_to_host(bwc_device_arena * a, size_t off, size_t bytes)
{
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    ready_sync(a);
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaMemcpy((char *) a->host + off, (char *) a->dev + off,
                bytes, cudaMemcpyDeviceToHost));
}

static void arena_to_dev(bwc_device_arena * a, size_t off, size_t bytes)
{
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    ready_sync(a);
    mirror_written(a);
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaMemcpy((char *) a->dev + off, (char *) a->host + off,
                bytes, cudaMemcpyHostToDevice));
}

__global__ void gather_words_kernel(const char * base, const uint64_t * off,
                                    uint64_t * out, unsigned int n)
{
    unsigned int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = *(const uint64_t *) (base + off[i]);
}

/* Size the gather scratch buffers for n words.
 * The caller holds the arena lock. */
static void ensure_words(arena_ctx * c, size_t n)
{
    if (c->nwords >= n)
        return;
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaFree(c->doff));
    CUDA_CHECK(cudaFree(c->dval));
    delete[] c->hval;
    CUDA_CHECK(cudaMalloc(&c->doff, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&c->dval, n * sizeof(uint64_t)));
    c->hval = new uint64_t[n];
    c->nwords = n;
}

/* Gather n mirror words at the given offsets into a device buffer.
 * The caller holds the arena lock and has called ensure_words. */
static void gather_words(arena_ctx * c, bwc_device_arena * a,
        const uint64_t * off, size_t n, uint64_t * out)
{
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaMemcpyAsync(c->doff, off, n * sizeof(uint64_t),
                cudaMemcpyHostToDevice, c->stream));
    unsigned int const blocks = (n + 255) / 256;
    gather_words_kernel<<<blocks, 256, 0, c->stream>>>(
            (const char *) a->dev, c->doff, out, n);
    CUDA_CHECK(cudaGetLastError());
}

static void arena_words_to_host(bwc_device_arena * a,
        const uint64_t * off, size_t n)
{
    /* the threads that share one vector gather concurrently */
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    ensure_words(c, n);
    ready_wait(a, c->stream);
    gather_words(c, a, off, n, c->dval);
    CUDA_CHECK(cudaMemcpyAsync(c->hval, c->dval, n * sizeof(uint64_t),
                cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    for (size_t i = 0; i < n; i++)
        *(uint64_t *) ((char *) a->host + off[i]) = c->hval[i];
}

/* The capture only enqueues the offset upload and the gather; the
 * host reads the values at flush time.  The staging row in qsrc stays
 * unchanged until the flush settles the stream, so the upload can
 * complete late.  The recorded ready event makes a later mirror write
 * wait for the read; a read keeps the bands. */
static void arena_words_queue(bwc_device_arena * a,
        const uint64_t * off, size_t n, size_t slot)
{
    if (!n)
        return;
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    CUDA_CHECK(cudaSetDevice(c->device));
    if (c->qwords < (slot + 1) * n) {
        size_t want = 2 * c->qwords;
        if (want < (slot + 1) * n)
            want = (slot + 1) * n;
        uint64_t * grown;
        CUDA_CHECK(cudaMalloc(&grown, want * sizeof(uint64_t)));
        /* the queued captures settle before they move */
        CUDA_CHECK(cudaStreamSynchronize(c->stream));
        if (slot)
            CUDA_CHECK(cudaMemcpy(grown, c->qval,
                        slot * n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(c->qval));
        c->qval = grown;
        CUDA_CHECK(cudaFree(c->qoff));
        CUDA_CHECK(cudaMalloc(&c->qoff, want * sizeof(uint64_t)));
        delete[] c->qsrc;
        c->qsrc = new uint64_t[want];
        c->qwords = want;
    }
    memcpy(c->qsrc + slot * n, off, n * sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpyAsync(c->qoff + slot * n, c->qsrc + slot * n,
                n * sizeof(uint64_t), cudaMemcpyHostToDevice, c->stream));
    ready_wait(a, c->stream);
    unsigned int const blocks = (n + 255) / 256;
    gather_words_kernel<<<blocks, 256, 0, c->stream>>>(
            (const char *) a->dev, c->qoff + slot * n, c->qval + slot * n, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(c->ev, c->stream));
    a->ready = c->ev;
}

static void arena_words_flush(bwc_device_arena * a, uint64_t * out,
        size_t n, size_t nslots)
{
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    CUDA_CHECK(cudaMemcpy(out, c->qval, nslots * n * sizeof(uint64_t),
                cudaMemcpyDeviceToHost));
}

/* A copy of settled output segments hands them on: a reader of the
 * destination range takes the source segments behind their events,
 * and pre covers the accesses of the destination before the copy. */
static void forward_segments(bwc_device_arena * dst, size_t doff,
        bwc_device_arena * src, size_t soff, size_t bytes)
{
    auto * cd = (arena_ctx *) dst->owner;
    bwc_dev_pieces const & s = ((arena_ctx *) src->owner)->pieces;
    bwc_dev_pieces & d = cd->pieces;
    if (!s.nout || !s.out_settled || soff < s.out[0].off
            || s.out_end < soff + bytes)
        return;
    d.nout = 0;
    for (int i = 0; i < s.nout; i++) {
        bwc_dev_piece const & p = s.out[i];
        size_t const lo = std::max(p.off, soff);
        size_t const hi = std::min(p.off + p.bytes, soff + bytes);
        if (lo >= hi)
            continue;
        d.out[d.nout++] = { p.device, p.p + (lo - p.off),
            doff + (lo - soff), hi - lo, p.ready, p.owner };
    }
    d.out_settled = true;
    d.out_before = cd->pre;
    d.out_end = doff + bytes;
}

static void arena_dev_copy(bwc_device_arena * dst, size_t doff,
        bwc_device_arena * src, size_t soff, size_t bytes)
{
    /* the threads that import from one leader vector copy concurrently */
    auto * cd = (arena_ctx *) dst->owner;
    auto * cs = (arena_ctx *) src->owner;
    arena_lock2 const lock(cd, cs);
    /* band_publish creates its events in the current device context */
    CUDA_CHECK(cudaSetDevice(cd->device));
    band_begin(dst);
    ready_wait(dst, cd->stream);
    if (!cd->pre)
        CUDA_CHECK(cudaEventCreateWithFlags(&cd->pre,
                    cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(cd->pre, cd->stream));
    size_t const piece = env_band_bytes();
    for (size_t p = 0; p < bytes; ) {
        size_t const len = std::min(piece, bytes - p);
        ready_wait_range(src, cd->stream, soff + p, len);
        CUDA_CHECK(cudaMemcpyPeerAsync((char *) dst->dev + doff + p,
                    cd->device, (const char *) src->dev + soff + p,
                    cs->device, len, cd->stream));
        band_publish(dst, doff + p, len);
        p += len;
    }
    ready_mark(dst);
    mirror_written(dst);
    forward_segments(dst, doff, src, soff, bytes);
    ready_follow(src, dst);
}

static void arena_dev_xor(bwc_device_arena * dst, size_t doff,
        bwc_device_arena * src, size_t soff, size_t bytes)
{
    /* the threads that reduce into one vector xor concurrently; the
     * peer pieces arrive on the copy stream, so the copy of a piece
     * runs under the xor of the piece before it */
    auto * cd = (arena_ctx *) dst->owner;
    auto * cs = (arena_ctx *) src->owner;
    arena_lock2 const lock(cd, cs);
    CUDA_CHECK(cudaSetDevice(cd->device));
    bool const peer = cs->device != cd->device;
    if (peer && cd->xbytes < bytes) {
        CUDA_CHECK(cudaFree(cd->xbuf));
        CUDA_CHECK(cudaMalloc(&cd->xbuf, bytes));
        cd->xbytes = bytes;
    }
    band_begin(dst);
    ready_wait(dst, cd->stream);
    if (peer)
        ready_wait(dst, cd->cstream);
    size_t const piece = env_band_bytes();
    for (size_t p = 0; p < bytes; ) {
        size_t const len = std::min(piece, bytes - p);
        ready_wait_range(src, peer ? cd->cstream : cd->stream,
                soff + p, len);
        auto * s = (const uint64_t *) ((const char *) src->dev + soff + p);
        if (peer) {
            CUDA_CHECK(cudaMemcpyPeerAsync((char *) cd->xbuf + p,
                        cd->device, s, cs->device, len, cd->cstream));
            CUDA_CHECK(cudaEventRecord(cd->cev, cd->cstream));
            CUDA_CHECK(cudaStreamWaitEvent(cd->stream, cd->cev, 0));
            s = (const uint64_t *) ((char *) cd->xbuf + p);
        }
        unsigned int const n = len / sizeof(uint64_t);
        xor_kernel<<<(n + 255) / 256, 256, 0, cd->stream>>>(
                (uint64_t *) ((char *) dst->dev + doff + p), s, n);
        CUDA_CHECK(cudaGetLastError());
        band_publish(dst, doff + p, len);
        p += len;
    }
    ready_mark(dst);
    mirror_written(dst);
    ready_follow(src, dst);
}

static void arena_release(bwc_device_arena * a)
{
    auto * c = (arena_ctx *) a->owner;
    {
        /* the registry already dropped the arena, so no new operation
         * can reach it; the lock only drains an operation in flight */
        std::lock_guard<std::mutex> const lock(c->mtx);
        ready_sync(a);
        for (int i = 0; i < c->pieces.nrd; i++)
            CUDA_CHECK(cudaEventSynchronize(c->pieces.rd[i]));
        if (c->locked)
            CUDA_CHECK(cudaHostUnregister(a->host));
        CUDA_CHECK(cudaSetDevice(c->device));
        CUDA_CHECK(cudaFree(a->dev));
        CUDA_CHECK(cudaFree(c->doff));
        CUDA_CHECK(cudaFree(c->dval));
        delete[] c->hval;
        CUDA_CHECK(cudaFree(c->xbuf));
        CUDA_CHECK(cudaFree(c->qval));
        CUDA_CHECK(cudaFree(c->qoff));
        delete[] c->qsrc;
        for (auto const & r : c->reps) {
            CUDA_CHECK(cudaSetDevice(r.first));
            CUDA_CHECK(cudaFree(r.second.p));
        }
        CUDA_CHECK(cudaSetDevice(c->device));
        CUDA_CHECK(cudaFree(c->obuf));
        if (c->bev)
            CUDA_CHECK(cudaEventDestroy(c->bev));
        for (auto const & e : c->pev)
            CUDA_CHECK(cudaEventDestroy(e.second));
        for (cudaEvent_t e : c->wbev)
            CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaEventDestroy(c->ev));
        CUDA_CHECK(cudaEventDestroy(c->cev));
        CUDA_CHECK(cudaStreamDestroy(c->stream));
        CUDA_CHECK(cudaStreamDestroy(c->cstream));
    }
    delete c;
}

/* The combine stream of one device of the process, with the
 * coefficient store of its kernel launches. */
struct combine_dev {
    cudaStream_t stream;
    cudaEvent_t ev;
    uint64_t * fbuf;            /* device, the combine matrices */
    int nsms;
};
static std::mutex combine_mutex;
static std::map<int, combine_dev> combine_devs;

static combine_dev & combine_on(int device)
{
    std::lock_guard<std::mutex> const lock(combine_mutex);
    auto const it = combine_devs.find(device);
    if (it != combine_devs.end())
        return it->second;
    CUDA_CHECK(cudaSetDevice(device));
    combine_dev & c = combine_devs[device];
    CUDA_CHECK(cudaStreamCreate(&c.stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&c.fbuf, ADDMUL64_MAX * 64 * sizeof(uint64_t)));
    CUDA_CHECK(cudaDeviceGetAttribute(&c.nsms,
                cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaFuncSetAttribute(addmul64_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                ADDMUL64_MAX * 2048 * sizeof(uint64_t)));
    return c;
}

/* The devices that combine a slice each: the devices the instances of
 * the process drive that reach the destination device directly, the
 * destination device first. */
static std::vector<int> combine_devices(int device)
{
    std::vector<int> devs { device };
    std::lock_guard<std::mutex> const lock(pin_mutex);
    for (int const d : pin_devices) {
        int can = 0;
        if (d != device)
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, d, device));
        if (can)
            devs.push_back(d);
    }
    return devs;
}

/* The mirror of a as device reads it: the mirror itself on its own
 * device, and a copy on any other, taken again on stream s when the
 * mirror was written since the last copy.  The caller holds the arena
 * lock, and the mirror is current. */
static const uint64_t * replica_on(bwc_device_arena * a, int device,
        cudaStream_t s)
{
    auto * c = (arena_ctx *) a->owner;
    if (c->device == device) {
        ready_wait(a, s);
        return (const uint64_t *) a->dev;
    }
    replica & r = c->reps[device];
    if (r.wgen == c->pieces.wgen && r.bytes == a->bytes)
        return r.p;
    if (r.bytes < a->bytes) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaFree(r.p));
        CUDA_CHECK(cudaMalloc(&r.p, a->bytes));
        r.bytes = a->bytes;
    }
    ready_wait(a, s);
    CUDA_CHECK(cudaMemcpyPeerAsync(r.p, device, a->dev, c->device,
                a->bytes, s));
    r.wgen = c->pieces.wgen;
    return r.p;
}

/* This creates a mirror for each source vector that has none, and
 * moves it to the device state.  After that, a host write to a source
 * vector must go through bwc_dev_push or bwc_dev_invalidate, or the
 * write is lost.
 *
 * The destination range splits into one slice per device of the
 * process.  Each device reads its slice of the sources from its own
 * copy of them (the sources of mksol change once per block, so the
 * copies are rare) and reads and writes its slice of the destination
 * in place through peer access.  One kernel per slice reads up to
 * ADDMUL64_MAX sources; more sources take further passes.  The
 * destination stream waits for every slice before it publishes. */
static void arena_dev_addmul64(bwc_device_arena * dst, size_t doff,
        size_t bytes, unsigned int nsrc, const void * const * v,
        const void * const * v_base, const size_t * v_bytes,
        const uint64_t * const * f)
{
    auto * cd = (arena_ctx *) dst->owner;
    std::vector<bwc_device_arena *> src(nsrc);
    std::vector<arena_ctx *> ctxs { cd };
    for (unsigned int j = 0; j < nsrc; j++) {
        src[j] = mirror_on(cd->device, v_base[j], v_bytes[j]);
        ctxs.push_back((arena_ctx *) src[j]->owner);
    }
    arena_lockn const lock(ctxs);
    for (unsigned int j = 0; j < nsrc; j++)
        arena_make_dev(src[j], 0);
    std::vector<int> const devs = combine_devices(cd->device);
    size_t const n = bytes / sizeof(uint64_t);
    size_t const slice = (n + devs.size() - 1) / devs.size();
    auto * const d = (uint64_t *) ((char *) dst->dev + doff);
    for (size_t i = 0; i < devs.size(); i++) {
        size_t const w0 = std::min(n, i * slice);
        unsigned int const nw = std::min(n, w0 + slice) - w0;
        if (!nw)
            break;
        combine_dev & c = combine_on(devs[i]);
        CUDA_CHECK(cudaSetDevice(devs[i]));
        ready_wait(dst, c.stream);
        for (unsigned int j0 = 0; j0 < nsrc; j0 += ADDMUL64_MAX) {
            unsigned int const k = std::min(nsrc - j0,
                    (unsigned) ADDMUL64_MAX);
            addmul64_srcs s {};
            for (unsigned int j = 0; j < k; j++) {
                bwc_device_arena * a = src[j0 + j];
                size_t const soff = (const char *) v[j0 + j]
                    - (const char *) a->host;
                s.p[j] = (const uint64_t *) ((const char *)
                        replica_on(a, devs[i], c.stream) + soff) + w0;
                CUDA_CHECK(cudaMemcpyAsync(c.fbuf + 64 * j, f[j0 + j],
                            64 * sizeof(uint64_t), cudaMemcpyHostToDevice,
                            c.stream));
            }
            unsigned int const blocks = std::min<unsigned int>(
                    (nw + 255) / 256, 8 * c.nsms);
            if (env_addmul64_nibble())
                addmul64_nibble_kernel<<<blocks, 256,
                    k * 256 * sizeof(uint64_t), c.stream>>>(
                            d + w0, s, c.fbuf, k, nw);
            else
                addmul64_kernel<<<blocks, 256, k * 2048 * sizeof(uint64_t),
                    c.stream>>>(d + w0, s, c.fbuf, k, nw);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(c.ev, c.stream));
        CUDA_CHECK(cudaStreamWaitEvent(cd->stream, c.ev, 0));
    }
    CUDA_CHECK(cudaSetDevice(cd->device));
    ready_record(dst);
    for (unsigned int j = 0; j < nsrc; j++)
        ready_follow(src[j], dst);
}

/* The coefficients and the result cross the host on each call: the
 * combine runs once per checkpoint over the thread's share of the
 * vector, so the copies are a small part of the epoch. */
static void arena_dev_dotprod64(bwc_device_arena * src, size_t soff,
        size_t bytes, size_t stride, const uint64_t * c, uint64_t * out)
{
    auto * cs = (arena_ctx *) src->owner;
    std::lock_guard<std::mutex> const lock(cs->mtx);
    CUDA_CHECK(cudaSetDevice(cs->device));
    size_t const words = stride / sizeof(uint64_t);
    size_t const n = bytes / stride;
    size_t const need = 64 * words + n;
    if (cs->owords < need) {
        CUDA_CHECK(cudaFree(cs->obuf));
        CUDA_CHECK(cudaMalloc(&cs->obuf, need * sizeof(uint64_t)));
        cs->owords = need;
    }
    uint64_t * const dc = cs->obuf + 64 * words;
    ready_wait(src, cs->stream);
    CUDA_CHECK(cudaMemsetAsync(cs->obuf, 0, 64 * words * sizeof(uint64_t),
                cs->stream));
    CUDA_CHECK(cudaMemcpyAsync(dc, c, n * sizeof(uint64_t),
                cudaMemcpyHostToDevice, cs->stream));
    dotprod64_kernel<<<dim3(256, words), 256, 0, cs->stream>>>(
            (const uint64_t *) ((const char *) src->dev + soff), words,
            dc, n, cs->obuf);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint64_t> h(64 * words);
    CUDA_CHECK(cudaMemcpyAsync(h.data(), cs->obuf, h.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost, cs->stream));
    CUDA_CHECK(cudaStreamSynchronize(cs->stream));
    for (size_t i = 0; i < h.size(); i++)
        out[i] ^= h[i];
}

/* all four mirrors are current and on one device (see the ops); the
 * index mirrors are read-only after their upload */
static void arena_dev_scatter(bwc_device_arena * dst, bwc_device_arena * di,
        bwc_device_arena * src, bwc_device_arena * si, size_t n,
        size_t stride)
{
    auto * cd = (arena_ctx *) dst->owner;
    auto * cs = (arena_ctx *) src->owner;
    arena_lock2 const lock(cd, cs);
    CUDA_CHECK(cudaSetDevice(cd->device));
    ready_wait(dst, cd->stream);
    ready_wait(src, cd->stream);
    ready_wait(di, cd->stream);
    ready_wait(si, cd->stream);
    size_t const words = stride / sizeof(uint64_t);
    size_t const total = n * words;
    permute_kernel<<<(total + 255) / 256, 256, 0, cd->stream>>>(
            (uint64_t *) dst->dev, (const uint32_t *) di->dev,
            (const uint64_t *) src->dev, (const uint32_t *) si->dev,
            total, words);
    CUDA_CHECK(cudaGetLastError());
    ready_record(dst);
    ready_follow(src, dst);
}

static void arena_mirror(bwc_device_arena * like, void * host, size_t bytes)
{
    mirror_on(((arena_ctx *) like->owner)->device, host, bytes);
}

static void arena_set_zero(bwc_device_arena * a, size_t off, size_t bytes)
{
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    CUDA_CHECK(cudaSetDevice(c->device));
    ready_wait(a, c->stream);
    CUDA_CHECK(cudaMemsetAsync((char *) a->dev + off, 0, bytes, c->stream));
    ready_record(a);
}

static bwc_device_arena_ops const arena_ops = {
    arena_to_host, arena_to_dev, arena_words_to_host, arena_dev_copy,
    arena_release, arena_dev_xor, arena_words_queue, arena_words_flush,
    arena_dev_addmul64, arena_dev_dotprod64, arena_dev_scatter,
    arena_mirror, arena_set_zero,
};

/* The same vector base can arrive as the source of one product and the
 * destination of the transposed product with a different size, so an
 * existing mirror grows to the largest observed range.  The host holds
 * the current data for the bytes past the old range.
 *
 * The steady-state lookup takes no lock: growth publishes the new size
 * through the atomic ready_bytes after the device copy settles, so a
 * lookup that sees a sufficient size sees a complete mirror.  The
 * process lock covers creation and growth, so two threads never create
 * or grow one mirror at the same time. */
static bwc_device_arena * mirror_on(int device, const void * host,
        size_t bytes)
{
    if (!bwc_dev_allowed()) return nullptr;
    {
        bwc_device_arena * const a = bwc_dev_find(host);
        if (a) {
            auto * c = (arena_ctx *) a->owner;
            if (c->ready_bytes.load(std::memory_order_acquire) >= bytes)
                return a;
        }
    }
    std::lock_guard<std::mutex> const reg(arena_mutex);
    bwc_device_arena * const a = bwc_dev_find(host);
    if (a) {
        if (a->bytes < bytes) {
            auto * c = (arena_ctx *) a->owner;
            std::lock_guard<std::mutex> const lock(c->mtx);
            /* close the lock-free lookup while the device buffer moves */
            c->ready_bytes.store(0, std::memory_order_release);
            ready_sync(a);
            if (c->locked)
                CUDA_CHECK(cudaHostUnregister(a->host));
            c->locked = lock_host(a->host, bytes);
            CUDA_CHECK(cudaSetDevice(c->device));
            void * grown;
            CUDA_CHECK(cudaMalloc(&grown, bytes));
            if (a->state == BWC_ARENA_DEV) {
                CUDA_CHECK(cudaMemcpy(grown, a->dev, a->bytes,
                            cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy((char *) grown + a->bytes,
                            (const char *) a->host + a->bytes,
                            bytes - a->bytes, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaFree(a->dev));
            a->dev = grown;
            a->bytes = bytes;
            mirror_written(a);
            c->ready_bytes.store(bytes, std::memory_order_release);
        }
        return a;
    }
    CUDA_CHECK(cudaSetDevice(device));
    auto * c = new arena_ctx;
    c->pieces.magic = BWC_DEV_PIECES_MAGIC;
    c->pieces.wgen = 1;
    c->device = device;
    c->locked = lock_host(const_cast<void *>(host), bytes);
    CUDA_CHECK(cudaStreamCreate(&c->stream));
    CUDA_CHECK(cudaStreamCreate(&c->cstream));
    CUDA_CHECK(cudaEventCreateWithFlags(&c->ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&c->cev, cudaEventDisableTiming));
    void * dev;
    CUDA_CHECK(cudaMalloc(&dev, bytes));
    c->ready_bytes.store(bytes, std::memory_order_release);
    return bwc_dev_register({ const_cast<void *>(host), bytes, dev,
            device, BWC_ARENA_HOST, c, &arena_ops, nullptr, nullptr, 0 });
}

static bwc_device_arena * mirror(spmv_gf2 * m, const void * host,
        size_t bytes)
{
    return mirror_on(m->devs[0].device, host, bytes);
}

/* Mark the mirror current.  The caller filled the mirror range
 * [0, filled) on the device; the rest of the arena comes from the host
 * copy, queued on the arena stream behind the ready event.  The
 * caller holds the arena lock. */
static void arena_make_dev(bwc_device_arena * a, size_t filled)
{
    auto * c = (arena_ctx *) a->owner;
    if (a->state == BWC_ARENA_HOST && filled < a->bytes) {
        ready_wait(a, c->stream);
        CUDA_CHECK(cudaMemcpyAsync((char *) a->dev + filled,
                    (char *) a->host + filled, a->bytes - filled,
                    cudaMemcpyHostToDevice, c->stream));
        ready_record(a);
    }
    a->state = BWC_ARENA_DEV;
}

/* Publish the mirror at the end of a product: the destination range
 * [0, filled) is queued on the arena stream, the tail comes from the
 * host, and the source reads fold into the source's ready event.  The
 * caller holds both arena locks. */
static void product_publish(bwc_device_arena * adst,
        bwc_device_arena * asrc, size_t filled)
{
    arena_make_dev(adst, filled);
    ready_record(adst);
    if (asrc)
        ready_follow(asrc, adst);
}

/* Return a pointer on the first device of the instance that holds the
 * current source data, with buf the source scratch of that device.  The
 * data reaches that device with one copy: from the host when no mirror
 * exists or the host copy is current, and with a peer copy when the
 * current mirror lives on a device of another instance. */
static const uint64_t * src_on_dev(spmv_gf2 * m, bwc_device_arena * a,
        const uint64_t * src, size_t bytes, uint64_t * buf)
{
    spmv_dev & r = m->devs[0];
    CUDA_CHECK(cudaSetDevice(r.device));
    if (!a) {
        CUDA_CHECK(cudaMemcpyAsync(buf, src, bytes,
                    cudaMemcpyHostToDevice, r.stream));
        return buf;
    }
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    arena_make_dev(a, 0);
    ready_wait(a, r.stream);
    if (a->device == r.device)
        return (const uint64_t *) a->dev;
    CUDA_CHECK(cudaMemcpyPeerAsync(buf, r.device, a->dev, a->device,
                bytes, r.stream));
    return buf;
}

/* Snapshot the bands of a source mirror, with a tail range behind the
 * ready event when the bands stop short of the mirror end. The
 * producer keeps the band events alive for the arena lifetime
 * (bwc_device_arena.hpp), so the copies can enqueue after the lock
 * drops. Empty when the mirror is not current or has no bands, and
 * the caller takes the whole-vector path. */
static std::vector<bwc_device_band> src_bands(bwc_device_arena * a)
{
    if (!a)
        return {};
    std::lock_guard<std::mutex> const lock(arena_mutex);
    if (a->state != BWC_ARENA_DEV || !a->nbands)
        return {};
    std::vector<bwc_device_band> sb(a->bands, a->bands + a->nbands);
    size_t const end = sb.back().off + sb.back().bytes;
    if (end < a->bytes)
        sb.push_back({ end, a->bytes - end, a->ready });
    return sb;
}

/* Make the stream wait until the mirror range [off, off + bytes) is
 * complete, on the band events that cover the range. */
static void wait_bands(std::vector<bwc_device_band> const & sb,
        cudaStream_t s, size_t off, size_t bytes)
{
    for (bwc_device_band const & b : sb)
        if (b.ready && b.off < off + bytes && off < b.off + b.bytes)
            CUDA_CHECK(cudaStreamWaitEvent(s, (cudaEvent_t) b.ready, 0));
}

/* Copy the mirror range [off, off + bytes) to dst on the device, one
 * piece per covering band, each behind its band event, so a piece
 * moves as soon as its range is complete.  The pieces go on the copy
 * stream: the compute stream waits for them, and the kernels it
 * already holds run under them.  The wait on ev keeps the pieces
 * behind the queued kernels that still read the destination. */
static void copy_bands(bwc_device_arena * a,
        std::vector<bwc_device_band> const & sb, spmv_dev & dev,
        char * dst, size_t off, size_t bytes)
{
    CUDA_CHECK(cudaSetDevice(dev.device));
    CUDA_CHECK(cudaStreamWaitEvent(dev.cstream, dev.ev, 0));
    for (bwc_device_band const & b : sb) {
        size_t const lo = std::max(b.off, off);
        size_t const hi = std::min(b.off + b.bytes, off + bytes);
        if (lo >= hi)
            continue;
        if (b.ready)
            CUDA_CHECK(cudaStreamWaitEvent(dev.cstream,
                        (cudaEvent_t) b.ready, 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(dst + (lo - off), dev.device,
                    (char *) a->dev + lo, a->device, hi - lo, dev.cstream));
    }
    CUDA_CHECK(cudaEventRecord(dev.cev, dev.cstream));
    CUDA_CHECK(cudaStreamWaitEvent(dev.stream, dev.cev, 0));
}

static double now()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int env_ngpu()
{
    const char * s = getenv("SPMV_GF2_NGPU");
    int n = s ? atoi(s) : 1;
    return n > 0 ? n : 1;
}

/* The number of column bands for a source vector of vec_bytes.  A
 * band of at most env_band_bytes() keeps the source segment of one
 * gather in L2, but every band also cuts each row into one more
 * segment, and the gather pays one chunk per segment.  When the rows
 * are too sparse for the bands, that overhead outweighs the locality:
 * at a sparse shape (50 entries per local row, 66 bands of 96 MB, 0.8
 * entries per segment) one band runs the iteration twice as fast,
 * while shapes with 6 or more entries per segment run 40 to 57
 * percent faster with their bands, and a shape at 1.3 still gains 13
 * percent.
 * SPMV_GF2_BAND_MIN is the least mean segment length that keeps the
 * bands; 0 keeps them always. */
static unsigned int env_nbands(size_t vec_bytes, uint64_t entries_per_row)
{
    size_t const band = env_band_bytes();
    size_t const n = vec_bytes / band + (vec_bytes % band != 0);
    const char * s = getenv("SPMV_GF2_BAND_MIN");
    uint64_t const min_seg = s ? atol(s) : 4;
    if (n <= 1 || entries_per_row < min_seg * n)
        return 1;
    return n;
}

/* The estimated cost of one gather chunk beyond its coefficients, in
 * coefficient units.  A chunk binds one subwarp for at least one
 * full pass plus the reduction and the atomic, so a slice of short
 * rows runs longer than a slice of heavy rows with the same
 * coefficient total (measured skew up to 6x on the c189 matrix).
 * The default comes from a sweep on the c189 matrix with the subwarp
 * kernel: 32 gives the best krylov wall time, and larger values move
 * the bottleneck to the chunk-heavy slices. */
static uint64_t env_chunk_cost()
{
    const char * s = getenv("SPMV_GF2_CHUNK_COST");
    long const c = s ? atol(s) : 32;
    return c > 0 ? c : 0;
}

/* The maximum number of entries one gather subwarp processes.  A
 * device that holds few heavy rows launches one subwarp per chunk, so
 * a smaller chunk gives it more subwarps in flight.  The default comes
 * from a sweep of 8192, 2048, 1024 and 512 on several device
 * types: 1024 has the best krylov wall time on all but one, where it
 * loses 1.5%. */
static uint64_t env_gather_chunk()
{
    const char * s = getenv("SPMV_GF2_CHUNK");
    long const c = s ? atol(s) : 1024;
    return c > 0 ? c : 1024;
}

/* SPMV_GF2_GATHER_DEPTH: the loads each gather lane keeps in flight
 * (1, 2, 4 or 8); more hides more of the scattered-read latency, but
 * costs registers */
static int env_gather_depth()
{
    const char * s = getenv("SPMV_GF2_GATHER_DEPTH");
    return s ? atoi(s) : 4;
}

/* SPMV_GF2_GATHER_WARPS: warps per gather block; smaller blocks give
 * more blocks per SM, so a stalled block leaves eligible warps in the
 * others */
static int env_gather_warps()
{
    const char * s = getenv("SPMV_GF2_GATHER_WARPS");
    return s ? atoi(s) : 4;
}

/* SPMV_GF2_CHUNK_ORDER=col sorts the gather chunks of each band
 * segment by their first column, so chunks near each other in launch
 * order read nearby source words; the default is row order.  Xor is
 * commutative, so the products stay bit-identical. */
static bool env_chunk_col_order()
{
    const char * s = getenv("SPMV_GF2_CHUNK_ORDER");
    return s && !strcmp(s, "col");
}

/* SPMV_GF2_L2_MB: mark the source band of each gather as L2-persisting
 * through an access policy window of at most this many MB; 0 (the
 * default) leaves the cache policy alone */
static size_t env_l2_window()
{
    const char * s = getenv("SPMV_GF2_L2_MB");
    return s ? (size_t) atol(s) << 20 : 0;
}

/* set to 0 to keep the plugin on the mirror path: no segment records,
 * no source replicas (measurement switch) */
static bool env_pieces()
{
    const char * s = getenv("SPMV_GF2_PIECES");
    return !s || atoi(s);
}

/* SPMV_GF2_TIMING=n prints a host stamp every n product calls (1024
 * for n < 2); 0 when unset */
static unsigned long timing_period()
{
    const char * s = getenv("SPMV_GF2_TIMING");
    if (!s)
        return 0;
    unsigned long const v = strtoul(s, nullptr, 10);
    return v < 2 ? 1024 : v;
}

/* the number of timed rounds that rebalance the row split */
static int env_split_rounds()
{
    const char * s = getenv("SPMV_GF2_SPLIT_ROUNDS");
    return s ? atoi(s) : 2;
}

/* the product call that splits the rows again on the kernel times of
 * the SPMV_GF2_PROBE_CALLS calls before it; 0 for no such split.  Off
 * by default: at the selected configuration the split is
 * worth nothing outside the run-to-run noise. */
static unsigned long env_resplit_at()
{
    const char * s = getenv("SPMV_GF2_RESPLIT_AT");
    return s ? strtoul(s, nullptr, 10) : 0;
}

static unsigned long env_probe_calls()
{
    const char * s = getenv("SPMV_GF2_PROBE_CALLS");
    return s ? strtoul(s, nullptr, 10) : 32;
}

/* Run f(t) for t in [0, nthr) on nthr threads. */
template<class F>
static void for_threads(unsigned int nthr, F const & f)
{
    std::vector<std::thread> pool;
    for (unsigned int t = 0; t < nthr; t++)
        pool.emplace_back([&f, t]() { f(t); });
    for (std::thread & th : pool)
        th.join();
}

/* Run f(i0, i1) on nthr threads, which cover the rows [row0, row1) in
 * contiguous ranges of about equal entry counts.  The heaviest rows
 * hold millions of entries and the lightest a few, so ranges of equal
 * row counts would leave one thread with most of the work. */
template<class F>
static void for_rows(row_stream const & rs, unsigned int row0,
        unsigned int row1, unsigned int nthr, F const & f)
{
    uint64_t const w0 = rs.rowptr[row0], w1 = rs.rowptr[row1];
    std::vector<unsigned int> cut(nthr + 1, row1);
    cut[0] = row0;
    for (unsigned int t = 1; t < nthr; t++)
        cut[t] = std::upper_bound(rs.rowptr.begin() + cut[t - 1],
                rs.rowptr.begin() + row1, w0 + (w1 - w0) / nthr * t)
            - rs.rowptr.begin();
    for_threads(nthr, [&](unsigned int t) { f(cut[t], cut[t + 1]); });
}

/* Sort the n values of a, all below max_val, on nthr threads: the
 * values go to buckets by their high bits, then each bucket is sorted. */
static void sort_parallel(uint32_t * a, size_t n, uint32_t max_val,
        unsigned int nthr)
{
    unsigned int const shift = 16;
    size_t const nb = ((size_t) max_val >> shift) + 1;
    auto piece = [n, nthr](unsigned int t) {
        return std::pair<size_t, size_t>(n * t / nthr, n * (t + 1) / nthr);
    };
    /* cnt[t][b]: the count, then the next slot, of bucket b in piece t */
    std::vector<size_t> cnt((size_t) nthr * nb);
    for_threads(nthr, [&](unsigned int t) {
        auto const [p0, p1] = piece(t);
        for (size_t k = p0; k < p1; k++)
            cnt[(size_t) t * nb + (a[k] >> shift)]++;
    });
    std::vector<size_t> bstart(nb + 1);
    size_t off = 0;
    for (size_t b = 0; b < nb; b++) {
        bstart[b] = off;
        for (unsigned int t = 0; t < nthr; t++) {
            size_t const c = cnt[(size_t) t * nb + b];
            cnt[(size_t) t * nb + b] = off;
            off += c;
        }
    }
    bstart[nb] = off;
    std::vector<uint32_t> tmp(n);
    for_threads(nthr, [&](unsigned int t) {
        auto const [p0, p1] = piece(t);
        for (size_t k = p0; k < p1; k++)
            tmp[cnt[(size_t) t * nb + (a[k] >> shift)]++] = a[k];
    });
    for_threads(nthr, [&](unsigned int t) {
        size_t const b0 = nb * t / nthr, b1 = nb * (t + 1) / nthr;
        for (size_t b = b0; b < b1; b++)
            std::sort(tmp.begin() + bstart[b], tmp.begin() + bstart[b + 1]);
        std::copy(tmp.begin() + bstart[b0], tmp.begin() + bstart[b1],
                a + bstart[b0]);
    });
}

/* Sort the columns of each row of the stream, in place.  The fill of
 * the banded blocks keeps the order of a row, so every row-band
 * segment comes out sorted and the scattered src reads of one gather
 * warp land closer together.  Xor is commutative, so the products stay
 * bit-identical.  A row that holds more than a small share of a
 * thread's entries is sorted by all threads together, so that no thread
 * ends long after the others; a row under a million entries never is,
 * one thread sorts it in well under a second. */
static void sort_rows(row_stream const & rs, unsigned int nrows,
        uint32_t max_val, unsigned int nthr)
{
    uint64_t const long_row = std::max<uint64_t>(rs.rowptr[nrows] / nthr / 16,
            1 << 20);
    for_rows(rs, 0, nrows, nthr, [&](unsigned int i0, unsigned int i1) {
        for (unsigned int i = i0; i < i1; i++) {
            if (rs.len(i) > long_row) continue;
            std::sort(rs.cols(i), rs.cols(i) + rs.len(i));
        }
    });
    for (unsigned int i = 0; i < nrows; i++) {
        if (rs.len(i) <= long_row) continue;
        sort_parallel(rs.cols(i), rs.len(i), max_val, nthr);
    }
}

/* the host threads the one-time setup of this instance may use */
static unsigned int setup_threads()
{
    return std::max(1U, std::thread::hardware_concurrency());
}

static void split_rows(spmv_gf2 * m, std::vector<uint64_t> const & work,
        unsigned int nrows);
static void resplit(spmv_gf2 * m, std::vector<uint64_t> const & work,
        row_stream const & rs, std::vector<double> const & t,
        const char * what);
static void pieces_release(spmv_gf2 * m, const bwc_device_arena * a);
static void build_dev(spmv_gf2 * m, spmv_dev & dev, row_stream const & rs);
static void build_all(spmv_gf2 * m, row_stream const & rs);
static void free_dev(spmv_dev & dev);
static void sync_all(spmv_gf2 * m);
static std::vector<double> trial_gather(spmv_gf2 * m, unsigned int max_dim);
static void launch_gather(spmv_gf2 * m, spmv_dev & dev,
        bwc_device_arena * asrc, std::vector<bwc_device_band> const & sb,
        const uint64_t * src, uint64_t * dst, size_t src_bytes, bool copy,
        std::vector<gather_seg *> const & segs);

/* Distinct instances get distinct device ranges.  Instance number i of
 * local rank r starts at ngpu * (r + nlocal * i): the first instances of
 * the nlocal ranks on a node take consecutive ranges, then the second
 * instances take the next ranges, and so on.  Ranges only overlap once
 * total demand exceeds the device count and the modulo wraps. */
static int device_range_start(int ngpu, int instance)
{
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    const char * lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!lr) lr = getenv("SLURM_LOCALID");
    int const base = lr ? atoi(lr) : 0;
    /* SLURM_TASKS_PER_NODE is always set and looks like "4(x2)";
     * atoi reads the leading count. */
    const char * ls = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (!ls) ls = getenv("SLURM_TASKS_PER_NODE");
    int const nlocal = ls && atoi(ls) > 0 ? atoi(ls) : 1;
    return (ngpu * (base + nlocal * instance)) % ndev;
}

static int pick_first_device(int ngpu)
{
    static std::atomic<int> next(0);
    return device_range_start(ngpu, next++);
}

__global__ static void random_length_kernel(unsigned long n, double mean,
        unsigned long long seed, uint32_t * len)
{
    unsigned long const i = blockIdx.x * (unsigned long) blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 2 * i, 0, &st);
    len[i] = curand_poisson(&st, mean);
}

/* soff[i] is the stream position of stored row i: one length word per
 * earlier row plus their lengths. */
__global__ static void random_entry_kernel(unsigned long n,
        unsigned long range, unsigned long long seed,
        const uint64_t * soff, uint32_t * out)
{
    unsigned long const i = blockIdx.x * (unsigned long) blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 2 * i + 1, 0, &st);
    uint32_t * p = out + soff[i];
    uint32_t const k = soff[i + 1] - soff[i] - 1;
    *p++ = k;
    for (uint32_t j = 0; j < k; j++)
        p[j] = ((uint64_t) curand(&st) * range) >> 32;
}

/* see spmv_gf2.h; call i binds to the first device of instance i's
 * range, the same numbering that spmv_gf2_new uses, so the concurrent
 * contexts of one rank generate on distinct devices */
extern "C" std::vector<uint32_t> random_matrix_stream_solib(
        unsigned long n, unsigned long range, double mean,
        unsigned long seed)
{
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int const ngpu = std::min({ env_ngpu(), ndev, BWC_DEV_PIECES_MAX });
    static std::atomic<int> call(0);
    CUDA_CHECK(cudaSetDevice(device_range_start(ngpu, call++)));

    unsigned int const B = 256;
    unsigned int const G = (n + B - 1) / B;

    uint32_t * len;
    CUDA_CHECK(cudaMalloc(&len, n * sizeof(uint32_t)));
    random_length_kernel<<<G, B>>>(n, mean, seed, len);
    std::vector<uint32_t> hlen(n);
    CUDA_CHECK(cudaMemcpy(hlen.data(), len,
            n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(len));

    std::vector<uint64_t> hsoff(n + 1);
    hsoff[0] = 0;
    for (unsigned long i = 0; i < n; i++)
        hsoff[i + 1] = hsoff[i] + 1 + hlen[i];

    uint64_t * soff;
    CUDA_CHECK(cudaMalloc(&soff, (n + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(soff, hsoff.data(),
            (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    uint32_t * out;
    CUDA_CHECK(cudaMalloc(&out, hsoff[n] * sizeof(uint32_t)));
    random_entry_kernel<<<G, B>>>(n, range, seed, soff, out);

    std::vector<uint32_t> stream(hsoff[n]);
    CUDA_CHECK(cudaMemcpy(stream.data(), out,
            hsoff[n] * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(soff));
    CUDA_CHECK(cudaFree(out));
    return stream;
}

spmv_gf2 * spmv_gf2_new(unsigned int nrows, uint64_t nwords,
                        uint32_t * rows, unsigned int max_dim,
                        unsigned int words)
{
    /* the end of each setup phase, on SPMV_GF2_TIMING */
    double const t_start = now();
    auto phase = [t_start](const char * what) {
        if (getenv("SPMV_GF2_TIMING"))
            fprintf(stderr, "spmv_gf2 setup: %s at %.1f s\n", what,
                    now() - t_start);
    };

    row_stream rs { rows, std::vector<uint64_t>(nrows + 1) };
    for (unsigned int i = 0; i < nrows; i++)
        rs.rowptr[i + 1] = rs.rowptr[i] + rows[rs.rowptr[i] + i];
    if (rs.rowptr[nrows] + nrows != nwords) {
        fprintf(stderr, "spmv_gf2_new: row stream length mismatch\n");
        abort();
    }
    phase("row pointers");
    sort_rows(rs, nrows, max_dim, setup_threads());
    phase("row sort");

    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int const ngpu = std::min({ env_ngpu(), ndev, BWC_DEV_PIECES_MAX });
    int const first = pick_first_device(ngpu);

    spmv_gf2 * m = new spmv_gf2;
    m->nrows = nrows;
    m->words = words;
    m->max_dim = max_dim;
    m->devs.resize(ngpu);
    size_t const esz = (size_t) words * sizeof(uint64_t);
    unsigned int const nbands = env_nbands((size_t) max_dim * esz,
            (nwords - nrows) / nrows);
    unsigned int const bandcols = (max_dim + nbands - 1) / nbands;
    m->nbands = nbands;
    m->bandcols = bandcols;

    /* the estimated gather work up to each row: coefficients plus a
     * fixed cost per chunk, one chunk per gather chunk of entries of a
     * row-band segment */
    std::vector<uint64_t> work;
    if (ngpu > 1) {
        uint64_t const ccost = env_chunk_cost();
        uint64_t const chunk = env_gather_chunk();
        work.resize(nrows + 1);
        for_rows(rs, 0, nrows, setup_threads(), [&](unsigned int i0,
                    unsigned int i1) {
            std::vector<uint32_t> stamp(nbands, 0), cnt(nbands, 0);
            for (unsigned int i = i0; i < i1; i++) {
                uint64_t w = rs.len(i);
                const uint32_t * cols = rs.cols(i);
                for (uint64_t k = 0; k < rs.len(i); k++) {
                    unsigned int const b = cols[k] / bandcols;
                    if (stamp[b] != i + 1) {
                        stamp[b] = i + 1;
                        cnt[b] = 0;
                    }
                    if (cnt[b]++ % chunk == 0)
                        w += ccost;
                }
                work[i + 1] = w;
            }
        });
        for (unsigned int i = 0; i < nrows; i++)
            work[i + 1] += work[i];
    }
    phase("work estimate");

    for (int d = 0; d < ngpu; d++) {
        spmv_dev & dev = m->devs[d];
        dev.device = (first + d) % ndev;
        CUDA_CHECK(cudaSetDevice(dev.device));
        CUDA_CHECK(cudaStreamCreate(&dev.stream));
        CUDA_CHECK(cudaStreamCreate(&dev.cstream));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev.cev, cudaEventDisableTiming));
        if (env_l2_window()) {
            int persist = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&dev.l2cap,
                        cudaDevAttrMaxAccessPolicyWindowSize, dev.device));
            CUDA_CHECK(cudaDeviceGetAttribute(&persist,
                        cudaDevAttrMaxPersistingL2CacheSize, dev.device));
            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                        (size_t) persist));
        }
        dev.vtmp = nullptr;
        if (ngpu > 1)
            CUDA_CHECK(cudaMalloc(&dev.vtmp, max_dim * esz));
    }

    /* Peer access is a speed optimization: without it, the peer copies
     * below stage through the host.  The vector mirrors of the other
     * instances in the process can sit on any device, so every pair
     * counts, not only the pairs inside this instance. */
    static std::once_flag peer_once;
    std::call_once(peer_once, [ndev]() {
        for (int a = 0; a < ndev; a++)
            for (int b = 0; b < ndev; b++) {
                if (a == b) continue;
                int can;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can, a, b));
                if (!can) continue;
                CUDA_CHECK(cudaSetDevice(a));
                cudaError_t const e = cudaDeviceEnablePeerAccess(b, 0);
                if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
                cudaGetLastError();
            }
    });

    split_rows(m, work, nrows);
    build_all(m, rs);
    phase("first build");

    /* The estimated work misses the cost that depends on the row
     * range: the source locality of a row differs with its density.
     * Each round times a product with all devices at work at once,
     * scales the estimated work of each device to its measured time,
     * splits again on the scaled work, rebuilds, and times again.  The
     * product time is the slowest device; a split that does not lower
     * it goes back to the split before it. */
    int const rounds = ngpu > 1 ? env_split_rounds() : 0;
    std::vector<double> t = rounds > 0 ? trial_gather(m, max_dim)
                                       : std::vector<double>();
    phase("first trial");
    for (int round = 0; round < rounds; round++) {
        std::vector<std::pair<unsigned int, unsigned int>> prev(ngpu);
        for (int d = 0; d < ngpu; d++)
            prev[d] = { m->devs[d].row0, m->devs[d].nrows };
        double const before = *std::max_element(t.begin(), t.end());
        resplit(m, work, rs, t, "split round");
        phase("rebuild");
        t = trial_gather(m, max_dim);
        phase("trial");
        double const after = *std::max_element(t.begin(), t.end());
        if (getenv("SPMV_GF2_TIMING"))
            fprintf(stderr, "spmv_gf2 split round %d: slowest device"
                    " %.3f ms before, %.3f ms after%s\n", round,
                    before * 1e3, after * 1e3,
                    after < before ? "" : "; kept the split before");
        if (after < before)
            continue;
        for (int d = 0; d < ngpu; d++) {
            free_dev(m->devs[d]);
            m->devs[d].row0 = prev[d].first;
            m->devs[d].nrows = prev[d].second;
        }
        build_all(m, rs);
        phase("rebuild of the split before");
        break;
    }

    {
        std::lock_guard<std::mutex> const lock(pin_mutex);
        for (spmv_dev const & dev : m->devs)
            pin_devices.insert(dev.device);
    }
    unsigned long const at = ngpu > 1 ? env_resplit_at() : 0;
    if (at) {
        m->rows.assign(rows, rows + nwords);
        m->rs = { m->rows.data(), std::move(rs.rowptr) };
        m->work = std::move(work);
        m->probe1 = at;
        m->probe0 = at - std::min(at, env_probe_calls());
    }
    return m;
}

/* Split the rows again on the measured time t of each device: the
 * estimated work of its rows scales to its time, so the new split
 * gives every device the same measured time, and the devices
 * rebuild. */
static void resplit(spmv_gf2 * m, std::vector<uint64_t> const & work,
        row_stream const & rs, std::vector<double> const & t,
        const char * what)
{
    std::vector<uint64_t> scaled(m->nrows + 1);
    for (size_t d = 0; d < m->devs.size(); d++) {
        spmv_dev & dev = m->devs[d];
        unsigned int const r1 = dev.row0 + dev.nrows;
        uint64_t const w = work[r1] - work[dev.row0];
        /* measured time per unit of estimated work on this device,
         * scaled so that the sum stays an integer count */
        double const rate = w ? t[d] * 1e15 / (double) w : 0;
        for (unsigned int i = dev.row0; i < r1; i++)
            scaled[i + 1] = scaled[i]
                + (uint64_t) ((double) (work[i + 1] - work[i]) * rate);
        if (getenv("SPMV_GF2_TIMING"))
            fprintf(stderr, "spmv_gf2 %s dev%d: rows %u work %lu"
                    " time %.3f ms\n", what, (int) d, dev.nrows,
                    (unsigned long) w, t[d] * 1e3);
        free_dev(dev);
    }
    split_rows(m, scaled, m->nrows);
    build_all(m, rs);
}

/* The probed calls record a timing event before and after each gather
 * kernel; the call after a probed one adds the kernel times up. */
static void probe_mark(spmv_dev & dev)
{
    if (dev.pe.size() == dev.npe) {
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreate(&e));
        dev.pe.push_back(e);
    }
    CUDA_CHECK(cudaEventRecord(dev.pe[dev.npe++], dev.stream));
}

static void probe_collect(spmv_gf2 * m)
{
    for (spmv_dev & dev : m->devs) {
        if (!dev.npe)
            continue;
        CUDA_CHECK(cudaSetDevice(dev.device));
        CUDA_CHECK(cudaEventSynchronize(dev.pe[dev.npe - 1]));
        for (size_t i = 0; i < dev.npe; i += 2) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, dev.pe[i], dev.pe[i + 1]));
            dev.busy += ms * 1e-3;
        }
        dev.npe = 0;
    }
}

/* At call probe1: the split on the kernel time of the probed calls,
 * under the source arrival and the exchange work of the real
 * iteration, which the trials of the setup do not see.  The plugin
 * holds no output segment of a product any more once every device
 * stream is idle and the segment records are released. */
/* the row stream, its pointers and the work counts that the probed
 * split needs, which no later call reads */
static void probe_release(spmv_gf2 * m)
{
    m->rows = std::vector<uint32_t>();
    m->rs = {};
    m->work = std::vector<uint64_t>();
    m->probe1 = 0;
}

static void resplit_probed(spmv_gf2 * m)
{
    probe_collect(m);
    if (m->calls != m->probe1)
        return;
    for (auto const & v : m->vdst)
        pieces_release(m, v.first);
    sync_all(m);
    std::vector<double> t;
    for (spmv_dev & dev : m->devs)
        t.push_back(dev.busy / (double) (m->probe1 - m->probe0));
    resplit(m, m->work, m->rs, t, "probed split");
    probe_release(m);
}

/* Give each device a contiguous row range with about the same share
 * of the cumulative work; with one device, all the rows. */
static void split_rows(spmv_gf2 * m, std::vector<uint64_t> const & work,
        unsigned int nrows)
{
    int const ngpu = (int) m->devs.size();
    unsigned int row0 = 0;
    for (int d = 0; d < ngpu; d++) {
        spmv_dev & dev = m->devs[d];
        dev.row0 = row0;
        unsigned int row1 = nrows;
        if (d + 1 < ngpu) {
            uint64_t const target = work[row0]
                + (work[nrows] - work[row0]) / (ngpu - d);
            row1 = std::lower_bound(work.begin() + row0 + 1, work.end(),
                    target) - work.begin();
        }
        dev.nrows = row1 - row0;
        row0 = row1;
    }
}

/* Build the matrix data of every device, one thread per device. */
static void build_all(spmv_gf2 * m, row_stream const & rs)
{
    std::vector<std::thread> builders;
    for (spmv_dev & dev : m->devs)
        builders.emplace_back(build_dev, m, std::ref(dev), std::cref(rs));
    for (std::thread & t : builders)
        t.join();
}

static void segs_release(spmv_dev & dev)
{
    CUDA_CHECK(cudaSetDevice(dev.device));
    for (gather_seg & s : dev.segs)
        CUDA_CHECK(cudaEventDestroy(s.ev));
    dev.segs.clear();
    dev.seg_bytes = SIZE_MAX;
}

/* Reupload the chunks with each band range of each segment sorted by
 * first column (SPMV_GF2_CHUNK_ORDER=col).  The row-order chunk_row
 * and chunk0 arrays keep cutting the segments; only the launch order
 * inside each range changes, which xor commutativity allows. */
static void chunks_col_order(spmv_dev & dev)
{
    if (dev.chunk_col.empty())
        return;
    std::vector<uint32_t> idx(dev.chunk_h.size());
    for (uint32_t k = 0; k < idx.size(); k++)
        idx[k] = k;
    for (gather_seg const & s : dev.segs)
        for (auto const & r : s.c)
            std::sort(idx.begin() + r.first, idx.begin() + r.second,
                    [&](uint32_t a, uint32_t b)
                    { return dev.chunk_col[a] < dev.chunk_col[b]; });
    std::vector<gather_chunk> out(idx.size());
    for (size_t k = 0; k < idx.size(); k++)
        out[k] = dev.chunk_h[idx[k]];
    CUDA_CHECK(cudaMemcpy(dev.chunks, out.data(),
                out.size() * sizeof(gather_chunk), cudaMemcpyHostToDevice));
}

/* Cut the device rows at the block boundaries of blk_bytes (one
 * segment when zero), with the chunk range of each band: the chunks
 * of a band are in row order. */
static void segs_cut(spmv_gf2 * m, spmv_dev & dev, size_t blk_bytes)
{
    if (dev.seg_bytes == blk_bytes)
        return;
    segs_release(dev);
    dev.seg_bytes = blk_bytes;
    size_t const esz = (size_t) m->words * sizeof(uint64_t);
    if (blk_bytes % esz) {
        fprintf(stderr, "spmv_gf2: block size is not a multiple of the element\n");
        abort();
    }
    uint64_t const bw = blk_bytes ? blk_bytes / esz : UINT64_MAX;
    unsigned int const row1 = dev.row0 + dev.nrows;
    for (unsigned int r0 = dev.row0; r0 < row1; ) {
        int const blk = (int) (r0 / bw);
        auto const r1 = (unsigned int) std::min<uint64_t>(row1,
                (blk + 1) * bw);
        gather_seg s { blk, r0, r1, {}, nullptr };
        for (unsigned int b = 0; b < m->nbands; b++) {
            auto const lo = dev.chunk_row.begin() + dev.chunk0[b];
            auto const hi = dev.chunk_row.begin() + dev.chunk0[b + 1];
            s.c.emplace_back(
                    std::lower_bound(lo, hi, r0 - dev.row0) - dev.chunk_row.begin(),
                    std::lower_bound(lo, hi, r1 - dev.row0) - dev.chunk_row.begin());
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
        dev.segs.push_back(std::move(s));
        r0 = r1;
    }
    chunks_col_order(dev);
}

/* The segments of the device in the consumption order of the plugin
 * (bwc_dev_pieces::blk), or in row order without a request. */
static std::vector<gather_seg *> segs_ordered(spmv_gf2 * m, spmv_dev & dev,
        bwc_dev_pieces const * p)
{
    segs_cut(m, dev, p && p->nblk ? p->blk_bytes : 0);
    std::vector<gather_seg *> out;
    if (!p || !p->nblk) {
        for (gather_seg & s : dev.segs)
            out.push_back(&s);
        return out;
    }
    for (int i = 0; i < p->nblk; i++)
        for (gather_seg & s : dev.segs)
            if (s.blk == p->blk[i])
                out.push_back(&s);
    if (out.size() != dev.segs.size()) {
        fprintf(stderr, "spmv_gf2: the block order misses a block\n");
        abort();
    }
    return out;
}

/* Build the banded CSR blocks and the chunk list of the row range of
 * the device, and upload them. */
static void build_dev(spmv_gf2 * m, spmv_dev & dev, row_stream const & rs)
{
    unsigned int const nbands = m->nbands;
    unsigned int const bandcols = m->bandcols;
    unsigned int const row0 = dev.row0;
    unsigned int const row1 = row0 + dev.nrows;
    unsigned int const nthr = std::max<unsigned int>(1,
            setup_threads() / m->devs.size());
    {
        /* the banded CSR blocks: count, then scan, then fill; the
         * count and fill passes write the cells of one row only, so
         * row ranges run on separate threads */
        size_t const N1 = dev.nrows + 1;
        uint64_t const nloc = rs.rowptr[row1] - rs.rowptr[row0];
        std::vector<uint64_t> local((size_t) nbands * N1);
        for_rows(rs, row0, row1, nthr, [&](unsigned int i0,
                    unsigned int i1) {
            for (unsigned int i = i0; i < i1; i++) {
                const uint32_t * cols = rs.cols(i);
                for (uint64_t k = 0; k < rs.len(i); k++)
                    local[(size_t) (cols[k] / bandcols) * N1 + i - row0]++;
            }
        });
        uint64_t off = 0;
        for (uint64_t & x : local) {
            uint64_t const count = x;
            x = off;
            off += count;
        }
        std::vector<uint64_t> next(local);
        std::vector<uint32_t> bc(nloc);
        for_rows(rs, row0, row1, nthr, [&](unsigned int i0,
                    unsigned int i1) {
            for (unsigned int i = i0; i < i1; i++) {
                const uint32_t * cols = rs.cols(i);
                for (uint64_t k = 0; k < rs.len(i); k++)
                    bc[next[(size_t) (cols[k] / bandcols) * N1 + i - row0]++]
                        = cols[k];
            }
        });

        /* the gather chunk list; an empty row has no chunk and keeps
         * the zero that launch_gather writes */
        std::vector<gather_chunk> chunks;
        uint64_t const chunk = env_gather_chunk();
        dev.chunk0.resize(nbands + 1);
        for (unsigned int b = 0; b < nbands; b++) {
            dev.chunk0[b] = chunks.size();
            for (unsigned int i = 0; i < dev.nrows; i++) {
                uint64_t const rend = local[(size_t) b * N1 + i + 1];
                for (uint64_t s = local[(size_t) b * N1 + i]; s < rend;
                        s += chunk) {
                    uint64_t const e = std::min(rend, s + chunk);
                    chunks.push_back({ s, (uint32_t) (e - s), i });
                }
            }
        }
        dev.chunk0[nbands] = chunks.size();
        dev.chunk_row.resize(chunks.size());
        for (size_t k = 0; k < chunks.size(); k++)
            dev.chunk_row[k] = chunks[k].row;
        if (env_chunk_col_order()) {
            dev.chunk_col.resize(chunks.size());
            for (size_t k = 0; k < chunks.size(); k++)
                dev.chunk_col[k] = bc[chunks[k].start];
            dev.chunk_h = chunks;
        }
        segs_release(dev);

        CUDA_CHECK(cudaSetDevice(dev.device));
        CUDA_CHECK(cudaMalloc(&dev.rowptr, local.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dev.cols, nloc * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(dev.rowptr, local.data(),
                    local.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.cols, bc.data(),
                    nloc * sizeof(uint32_t), cudaMemcpyHostToDevice));
        size_t const nchunks = chunks.size();
        CUDA_CHECK(cudaMalloc(&dev.chunks, nchunks * sizeof(gather_chunk)));
        CUDA_CHECK(cudaMemcpy(dev.chunks, chunks.data(),
                    nchunks * sizeof(gather_chunk), cudaMemcpyHostToDevice));
    }
}

/* Drop the matrix data of the device; the scratch and the streams stay. */
static void free_dev(spmv_dev & dev)
{
    segs_release(dev);
    CUDA_CHECK(cudaSetDevice(dev.device));
    CUDA_CHECK(cudaFree(dev.rowptr));
    CUDA_CHECK(cudaFree(dev.cols));
    CUDA_CHECK(cudaFree(dev.chunks));
    dev.rowptr = nullptr;
    dev.cols = nullptr;
    dev.chunks = nullptr;
}

/* The product time of each device on a full-width source of set bits,
 * under the conditions of the real product with a source mirror on
 * the first device: all devices at work at once, the first device
 * reads the source in place, each other device copies the source
 * from the first device band by band under its gather (copy_bands
 * from launch_gather, on its copy stream), and copies its output rows
 * to the first device after its gather.  The source copies of all
 * devices read one device, so their time depends on how many devices
 * are still at work, and the window has to hold them.  With one band
 * no copy runs under a gather: every device waits for its whole
 * source before its kernel starts, the copy costs the same for every
 * split, and the window holds the kernel only (on a sparse shape the
 * copy put 2 ms per device index on the measured time, and the split
 * on it made the slowest device 10 ms slower).  The median of a few
 * timed rounds after a warm-up; the gather cost does not depend on
 * the vector values. */
static std::vector<double> trial_gather(spmv_gf2 * m, unsigned int max_dim)
{
    size_t const esz = (size_t) m->words * sizeof(uint64_t);
    size_t const bytes = max_dim * esz;
    int const reps = 5;
    size_t const ndev = m->devs.size();
    spmv_dev & r = m->devs[0];
    uint64_t ** const vsrc = scratch(m, m->vsrc, nullptr);
    uint64_t ** const vdst = scratch(m, m->vdst, nullptr);
    /* a source arena on the first device with one settled band, so
     * that copy_bands takes the pieces from vsrc[0] with no event wait */
    bwc_device_arena asrc;
    asrc.host = nullptr;
    asrc.bytes = bytes;
    asrc.dev = vsrc[0];
    asrc.device = r.device;
    asrc.state = BWC_ARENA_DEV;
    asrc.owner = nullptr;
    asrc.ops = nullptr;
    std::vector<bwc_device_band> const sb = { { 0, bytes, nullptr } };
    std::vector<cudaEvent_t> e0(ndev), e1(ndev);
    std::vector<std::vector<float>> ms(ndev);
    for (size_t d = 0; d < ndev; d++) {
        spmv_dev & dev = m->devs[d];
        CUDA_CHECK(cudaSetDevice(dev.device));
        if (!d)
            CUDA_CHECK(cudaMemsetAsync(vsrc[0], 0xa5, bytes, dev.stream));
        CUDA_CHECK(cudaEventCreate(&e0[d]));
        CUDA_CHECK(cudaEventCreate(&e1[d]));
    }
    for (int rep = 0; rep < reps + 1; rep++) {
        sync_all(m);
        for (size_t d = 0; d < ndev; d++) {
            spmv_dev & dev = m->devs[d];
            bool const own = &dev == &r;
            /* the source copies run under the gather, band by band */
            bool const under = !own && m->nbands > 1;
            CUDA_CHECK(cudaSetDevice(dev.device));
            if (!own && !under)
                copy_bands(&asrc, sb, dev, (char *) vsrc[d], 0, bytes);
            CUDA_CHECK(cudaEventRecord(e0[d], dev.stream));
            /* the copy stream of copy_bands waits on dev.ev, as it does
             * behind the product step before; this puts the copies in
             * the window */
            CUDA_CHECK(cudaEventRecord(dev.ev, dev.stream));
            launch_gather(m, dev, &asrc, sb, vsrc[d], vdst[d], bytes, under,
                    segs_ordered(m, dev, nullptr));
            if (d)
                CUDA_CHECK(cudaMemcpyPeerAsync(
                            vdst[0] + (size_t) dev.row0 * m->words, r.device,
                            vdst[d], dev.device, dev.nrows * esz,
                            dev.stream));
            CUDA_CHECK(cudaEventRecord(e1[d], dev.stream));
        }
        sync_all(m);
        for (size_t d = 0; d < ndev && rep; d++) {
            float x;
            CUDA_CHECK(cudaEventElapsedTime(&x, e0[d], e1[d]));
            ms[d].push_back(x);
        }
    }
    std::vector<double> t(ndev);
    for (size_t d = 0; d < ndev; d++) {
        std::sort(ms[d].begin(), ms[d].end());
        t[d] = ms[d][reps / 2] * 1e-3;
        CUDA_CHECK(cudaSetDevice(m->devs[d].device));
        CUDA_CHECK(cudaEventDestroy(e0[d]));
        CUDA_CHECK(cudaEventDestroy(e1[d]));
    }
    return t;
}

/* the number of lanes that share one gather chunk */
#ifndef GATHER_LANES
#define GATHER_LANES 8
#endif

/* One chunk of at most env_gather_chunk() entries per subwarp of
 * GATHER_LANES lanes.  The stored matrix is the transpose: its heavy
 * rows reach millions of entries, which starves the device when one
 * warp walks a whole row, and its banded row segments hold only tens
 * of entries, which idles most lanes when a full warp takes a whole
 * chunk.  For W=4 a lane pair shares each entry, so the two 16-byte
 * loads of one 32-byte element land in one L1 sector access.  Each
 * lane keeps four gathers in flight, which hides part of the latency
 * of the scattered src reads.  The cols entries stream through the
 * cache with no reuse, so they load with the evict-first hint.  The
 * partial results of one row meet in dst through atomicXor, which
 * starts from the zero that launch_gather writes; xor is associative
 * and commutative, so the result does not depend on the atomic
 * order. */
template <int W, int D>
__global__ void gather_kernel(const gather_chunk * chunks,
                              const uint32_t * cols, const uint64_t * src,
                              uint64_t * dst, unsigned int nchunks)
{
    constexpr int G = GATHER_LANES;
    constexpr int P = W == 4 ? 2 : 1;   /* lanes per entry */
    constexpr int A = W / P;            /* words per lane */
    constexpr int S = G / P;            /* entries per step */
    unsigned int const c = (blockIdx.x * blockDim.x + threadIdx.x) / G;
    if (c >= nchunks) return;
    unsigned int const lane = threadIdx.x % G;
    unsigned int const mask =
        (unsigned int) ((1ull << G) - 1) << (threadIdx.x % 32 / G * G);
    gather_chunk const g = chunks[c];
    uint64_t const end = g.start + g.len;
    uint64_t acc[A] = {};
    for (uint64_t k = g.start + lane / P; k < end; k += D * S) {
        uint32_t cc[D];
#pragma unroll
        for (int u = 0; u < D; u++)
            if (k + S * u < end)
                cc[u] = __ldcs(cols + k + S * u);
        uint64_t v[D][A] = {};
#pragma unroll
        for (int u = 0; u < D; u++) {
            if (k + S * u >= end) continue;
            if constexpr (A % 2 == 0) {
                /* src comes from cudaMalloc, so the 16-byte half of a
                 * lane sits on a 16-byte boundary */
                ulonglong2 const x = ((const ulonglong2 *) src)
                    [(size_t) cc[u] * (W / 2) + lane % P];
                v[u][0] = x.x;
                v[u][1] = x.y;
            } else {
                v[u][0] = src[(size_t) cc[u]];
            }
        }
        for (int u = 0; u < D; u++)
            for (int w = 0; w < A; w++)
                acc[w] ^= v[u][w];
    }
    for (int w = 0; w < A; w++) {
        for (int off = G / 2; off >= P; off >>= 1)
            acc[w] ^= __shfl_xor_sync(mask, acc[w], off);
        if (lane < P)
            atomicXor((unsigned long long *)
                    &dst[(size_t) g.row * W + lane * A + w],
                    (unsigned long long) acc[w]);
    }
}

template <int W>
__global__ void scatter_kernel(const uint64_t * rowptr, const uint32_t * cols,
                               const uint64_t * src, uint64_t * dst,
                               unsigned int nrows)
{
    unsigned int const row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= nrows) return;
    unsigned int const lane = threadIdx.x & 31;
    uint64_t const end = rowptr[row + 1];
    uint64_t val[W];
    for (int w = 0; w < W; w++) val[w] = src[(size_t) row * W + w];
    for (uint64_t k = rowptr[row] + lane; k < end; k += 32)
        for (int w = 0; w < W; w++)
            atomicXor((unsigned long long *) &dst[(size_t) cols[k] * W + w],
                      (unsigned long long) val[w]);
}

__global__ void xor_kernel(uint64_t * dst, const uint64_t * src,
                           unsigned int n)
{
    unsigned int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] ^= src[i];
}

/* word w of dst item di[k] = word w of src item si[k] */
__global__ void permute_kernel(uint64_t * dst, const uint32_t * di,
                               const uint64_t * src, const uint32_t * si,
                               size_t total, size_t words)
{
    size_t const t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (t >= total) return;
    size_t const k = t / words;
    size_t const w = t - k * words;
    dst[di[k] * words + w] = src[si[k] * words + w];
}

/* out[64w + b] ^= c[i] over the items i whose word w has bit b set;
 * block row w handles word w, each thread folds a stride of items,
 * the warp folds its lanes, and one atomic per warp folds into out */
__global__ void dotprod64_kernel(const uint64_t * y, size_t words,
                                 const uint64_t * c, size_t n,
                                 uint64_t * out)
{
    size_t const w = blockIdx.y;
    uint64_t acc[64];
    for (int b = 0; b < 64; b++) acc[b] = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n;
            i += (size_t) gridDim.x * blockDim.x) {
        uint64_t const a = y[i * words + w];
        uint64_t const ci = c[i];
#pragma unroll
        for (int b = 0; b < 64; b++)
            acc[b] ^= ci & -((a >> b) & 1);
    }
#pragma unroll
    for (int b = 0; b < 64; b++) {
        uint64_t v = acc[b];
        for (int o = 16; o > 0; o >>= 1)
            v ^= __shfl_xor_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0 && v)
            atomicXor((unsigned long long *) &out[w * 64 + b],
                    (unsigned long long) v);
    }
}

/* dst[i] ^= xor over the sources j and the set bits k of s.p[j][i] of
 * f[64 j + k]: one byte-indexed table of 256 words per source and
 * byte, built in shared memory by each block, then eight lookups per
 * source word */
__global__ void addmul64_kernel(uint64_t * dst, addmul64_srcs s,
                                const uint64_t * f, unsigned int nsrc,
                                unsigned int n)
{
    extern __shared__ uint64_t tab[];
    for (unsigned int e = threadIdx.x; e < nsrc * 2048; e += blockDim.x) {
        unsigned int const j = e >> 11, b = (e >> 8) & 7, x = e & 255;
        const uint64_t * fb = f + j * 64 + b * 8;
        uint64_t t = 0;
        for (int k = 0; k < 8; k++)
            t ^= fb[k] & (uint64_t) -(int64_t) ((x >> k) & 1);
        tab[e] = t;
    }
    __syncthreads();
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += gridDim.x * blockDim.x) {
        uint64_t r = dst[i];
        for (unsigned int j = 0; j < nsrc; j++) {
            uint64_t const a = s.p[j][i];
            const uint64_t * t = tab + j * 2048;
#pragma unroll
            for (int b = 0; b < 8; b++)
                r ^= t[b * 256 + ((a >> (8 * b)) & 255)];
        }
        dst[i] = r;
    }
}

/* the nibble variant of addmul64_kernel: 16 tables of 16 words per
 * source (2 KB instead of 16 KB of shared memory), at sixteen lookups
 * per source word instead of eight (SPMV_GF2_ADDMUL64=nibble) */
__global__ void addmul64_nibble_kernel(uint64_t * dst, addmul64_srcs s,
                                       const uint64_t * f, unsigned int nsrc,
                                       unsigned int n)
{
    extern __shared__ uint64_t tab[];
    for (unsigned int e = threadIdx.x; e < nsrc * 256; e += blockDim.x) {
        unsigned int const j = e >> 8, b = (e >> 4) & 15, x = e & 15;
        const uint64_t * fb = f + j * 64 + b * 4;
        uint64_t t = 0;
        for (int k = 0; k < 4; k++)
            t ^= fb[k] & (uint64_t) -(int64_t) ((x >> k) & 1);
        tab[e] = t;
    }
    __syncthreads();
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += gridDim.x * blockDim.x) {
        uint64_t r = dst[i];
        for (unsigned int j = 0; j < nsrc; j++) {
            uint64_t const a = s.p[j][i];
            const uint64_t * t = tab + j * 256;
#pragma unroll
            for (int b = 0; b < 16; b++)
                r ^= t[b * 16 + ((a >> (4 * b)) & 15)];
        }
        dst[i] = r;
    }
}

#define WARPS_PER_BLOCK 8

/* Mark bytes at base as L2-persisting for the launches that follow on
 * the device stream (SPMV_GF2_L2_MB); the window moves to each new
 * gather band. */
static void l2_window(spmv_dev & dev, const void * base, size_t bytes)
{
    cudaStreamAttrValue attr {};
    attr.accessPolicyWindow.base_ptr = const_cast<void *>(base);
    attr.accessPolicyWindow.num_bytes = std::min(bytes, (size_t) dev.l2cap);
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CUDA_CHECK(cudaStreamSetAttribute(dev.stream,
                cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void sync_all(spmv_gf2 * m)
{
    for (spmv_dev & dev : m->devs) {
        CUDA_CHECK(cudaSetDevice(dev.device));
        CUDA_CHECK(cudaStreamSynchronize(dev.stream));
        CUDA_CHECK(cudaStreamSynchronize(dev.cstream));
    }
}

typedef void gather_kern(const gather_chunk *, const uint32_t *,
                         const uint64_t *, uint64_t *, unsigned int);
typedef void scatter_kern(const uint64_t *, const uint32_t *,
                          const uint64_t *, uint64_t *, unsigned int);

template <int W>
static gather_kern * gather_depth()
{
    switch (env_gather_depth()) {
        case 1: return gather_kernel<W, 1>;
        case 2: return gather_kernel<W, 2>;
        case 4: return gather_kernel<W, 4>;
        case 8: return gather_kernel<W, 8>;
    }
    fprintf(stderr, "spmv_gf2: unsupported gather depth\n");
    abort();
}

static gather_kern * gather_for(unsigned int words)
{
    switch (words) {
        case 1: return gather_depth<1>();
        case 2: return gather_depth<2>();
        case 4: return gather_depth<4>();
    }
    fprintf(stderr, "spmv_gf2: unsupported element width\n");
    abort();
}

static scatter_kern * scatter_for(unsigned int words)
{
    switch (words) {
        case 1: return scatter_kernel<1>;
        case 2: return scatter_kernel<2>;
        case 4: return scatter_kernel<4>;
    }
    fprintf(stderr, "spmv_gf2: unsupported element width\n");
    abort();
}

/* Gather the rows of the device, one kernel launch per column band.
 * With source bands, the launch of band b enqueues behind only the
 * source ranges of band b: a copy into src when copy is set, a wait
 * when the device reads the mirror in place or reads a replica the
 * plugin filled.  The later ranges then arrive under the compute of
 * the earlier bands. */
static void launch_gather(spmv_gf2 * m, spmv_dev & dev,
        bwc_device_arena * asrc, std::vector<bwc_device_band> const & sb,
        const uint64_t * src, uint64_t * dst, size_t src_bytes, bool copy,
        std::vector<gather_seg *> const & segs)
{
    if (!dev.nrows) return;
    CUDA_CHECK(cudaMemsetAsync(dst, 0,
                (size_t) dev.nrows * m->words * sizeof(uint64_t),
                dev.stream));
    gather_kern * const kern = gather_for(m->words);
    bool const probe = m->calls >= m->probe0 && m->calls < m->probe1;
    unsigned int const warps = env_gather_warps();
    unsigned int const cpb = 32 * warps / GATHER_LANES;
    size_t const l2win = env_l2_window();
    size_t const bb = (size_t) m->bandcols * m->words * sizeof(uint64_t);
    std::vector<bool> in(m->nbands, false);
    for (gather_seg * s : segs) {
        for (unsigned int b = 0; b < m->nbands; b++) {
            auto const n = (unsigned int) (s->c[b].second - s->c[b].first);
            if (!n) continue;
            if (!in[b]) {
                in[b] = true;
                size_t const lo = std::min(b * bb, src_bytes);
                size_t const hi = std::min((b + 1) * bb, src_bytes);
                if (copy)
                    copy_bands(asrc, sb, dev, (char *) src + lo, lo, hi - lo);
                else if (!sb.empty())
                    wait_bands(sb, dev.stream, lo, hi - lo);
                if (l2win)
                    l2_window(dev, (const char *) src + lo,
                            std::min(l2win, hi - lo));
                tl_mark(tlg, dev.stream, dev.device, tl_name("g in", b), m->calls);
            }
            unsigned int const blocks = (n + cpb - 1) / cpb;
            if (probe)
                probe_mark(dev);
            kern<<<blocks, 32 * warps, 0, dev.stream>>>(
                    dev.chunks + s->c[b].first, dev.cols, src, dst, n);
            CUDA_CHECK(cudaGetLastError());
            if (probe)
                probe_mark(dev);
        }
        CUDA_CHECK(cudaEventRecord(s->ev, dev.stream));
        tl_mark(tlg, dev.stream, dev.device, tl_name("g out", s->blk), m->calls);
    }
}

static void launch_scatter(spmv_gf2 * m, spmv_dev & dev,
        const uint64_t * src, uint64_t * dst)
{
    if (!dev.nrows) return;
    unsigned int const blocks =
        (dev.nrows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    scatter_kern * const kern = scatter_for(m->words);
    for (unsigned int b = 0; b < m->nbands; b++) {
        kern<<<blocks, 32 * WARPS_PER_BLOCK, 0, dev.stream>>>(
                dev.rowptr + (size_t) b * (dev.nrows + 1), dev.cols,
                src, dst, dev.nrows);
        CUDA_CHECK(cudaGetLastError());
    }
}

/* Before a product writes the destination arena and its scratch: wait
 * on every device stream for the plugin reads of the output segments
 * of the last product into it, then drop the segment record. */
static void pieces_release(spmv_gf2 * m, const bwc_device_arena * a)
{
    if (!a)
        return;
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    for (spmv_dev & dev : m->devs) {
        CUDA_CHECK(cudaSetDevice(dev.device));
        for (int i = 0; i < c->pieces.nrd; i++)
            CUDA_CHECK(cudaStreamWaitEvent(dev.stream, c->pieces.rd[i], 0));
    }
    c->pieces.nout = 0;
    c->pieces.nrd = 0;
}

/* Before a product writes the source scratch of an arena itself: wait
 * on every device stream for the plugin writes of the replicas the
 * last product from it asked for, and drop the replica record. */
static void replicas_release(spmv_gf2 * m, bwc_device_arena * a)
{
    if (!a)
        return;
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    for (spmv_dev & dev : m->devs)
        ready_wait(a, dev.stream);
    if (c->pieces.rep_owner == m) {
        c->pieces.nrep = 0;
        c->pieces.rep_owner = nullptr;
    }
}

/* True when the plugin filled the replicas of the source vector since
 * the last product asked for them, and nothing wrote the mirror since. */
static bool replicas_current(spmv_gf2 * m, bwc_device_arena * a,
        size_t off, size_t bytes)
{
    auto * c = (arena_ctx *) a->owner;
    std::lock_guard<std::mutex> const lock(c->mtx);
    bwc_dev_pieces const & p = c->pieces;
    return p.rep_wgen == p.wgen && p.rep_owner == m && p.nrep
        && p.rep[0].off == off && p.rep[0].bytes == bytes;
}

/* Ask the plugin to fill the source scratch vsrc of every device that
 * does not read the mirror in place at its next write of the source
 * range.  The caller holds the arena lock. */
static void replicas_publish(spmv_gf2 * m, bwc_device_arena * a,
        uint64_t ** vsrc, size_t off, size_t bytes, bool first_in_place)
{
    bwc_dev_pieces & p = ((arena_ctx *) a->owner)->pieces;
    p.nrep = 0;
    for (size_t d = first_in_place; d < m->devs.size(); d++)
        p.rep[p.nrep++] = { m->devs[d].device, (char *) vsrc[d], off, bytes,
                m->devs[d].ev, &p };
    p.rep_owner = m;
    p.rep_wgen = 0;
}

void spmv_gf2_gather(spmv_gf2 * m, uint64_t * dst, const uint64_t * src,
                     unsigned int ndst, unsigned int nsrc)
{
    size_t const esz = (size_t) m->words * sizeof(uint64_t);
    bwc_device_arena * const asrc = mirror(m, src, nsrc * esz);
    bwc_device_arena * const adst = mirror(m, dst, ndst * esz);
    if (!asrc)
        pin(src, nsrc * esz);
    if (!adst)
        pin(dst, ndst * esz);
    std::shared_lock<std::shared_mutex> copies(pin_copy_mutex,
            std::defer_lock);
    if (!asrc || !adst)
        copies.lock();
    if (m->calls > m->probe0 && m->calls <= m->probe1)
        resplit_probed(m);
    double const t0 = now();
    spmv_dev & r = m->devs[0];
    uint64_t ** const vsrc = scratch(m, m->vsrc, asrc);
    uint64_t ** const vdst = scratch(m, m->vdst, adst);
    pieces_release(m, adst);
    std::vector<bwc_device_band> const sb = src_bands(asrc);
    bool const in_place = asrc && asrc->device == r.device;
    bool const rep = !sb.empty()
        && replicas_current(m, asrc, 0, nsrc * esz);
    if (!rep)
        replicas_release(m, asrc);
    const uint64_t * rsrc;
    if (sb.empty()) {
        rsrc = src_on_dev(m, asrc, src, nsrc * esz, vsrc[0]);
        CUDA_CHECK(cudaSetDevice(r.device));
        CUDA_CHECK(cudaEventRecord(r.ev, r.stream));
        for (size_t d = 1; d < m->devs.size(); d++) {
            spmv_dev & dev = m->devs[d];
            CUDA_CHECK(cudaSetDevice(dev.device));
            CUDA_CHECK(cudaStreamWaitEvent(dev.stream, r.ev, 0));
            CUDA_CHECK(cudaMemcpyPeerAsync(vsrc[d], dev.device, rsrc,
                        r.device, nsrc * esz, dev.stream));
        }
    } else {
        /* launch_gather delivers the source per range, so a gather
         * band starts as soon as its own ranges are complete; the
         * first device reads a mirror on its own device in place */
        rsrc = in_place ? (const uint64_t *) asrc->dev : vsrc[0];
    }
    double const t1 = now();
    uint64_t * const mdst = adst ? (uint64_t *) adst->dev : nullptr;
    uint64_t * const rdst =
        adst && adst->device == r.device ? mdst : vdst[0];
    if (adst && rdst == mdst)
        ready_wait_locked(adst, r.stream);
    /* the block order the plugin asked for at its last exchange of
     * this destination; the plugin runs on this thread before the call */
    bwc_dev_pieces const * const want =
        adst ? &((arena_ctx *) adst->owner)->pieces : nullptr;
    for (size_t d = 0; d < m->devs.size(); d++) {
        spmv_dev & dev = m->devs[d];
        CUDA_CHECK(cudaSetDevice(dev.device));
        tl_mark(tlg, dev.stream, dev.device, "g free", m->calls);
        launch_gather(m, dev, asrc, sb, d ? vsrc[d] : rsrc,
                d ? vdst[d] : rdst, nsrc * esz,
                !sb.empty() && !rep && !(!d && in_place),
                segs_ordered(m, dev, want));
        CUDA_CHECK(cudaEventRecord(dev.ev, dev.stream));
    }
    if (tl_on() && m->calls >= TL_K0 && m->calls < TL_K0 + TL_N)
        fprintf(stderr, "TH spmv host gather-enqueued %lu %.6f\n", m->calls, tl_now());
    double const t2 = now();
    if (adst) {
        /* Each device copies its row band into the mirror on its own
         * stream, and the arena publishes one band event per range and
         * one segment record per device, so the communication plugin
         * starts on a range while later bands still compute, and reads
         * a segment from the device that holds it. The host does not
         * wait. */
        auto * const c = (arena_ctx *) adst->owner;
        auto * const csrc = (arena_ctx *) asrc->owner;
        arena_lock2 const lock(c, csrc);
        c->bands.resize(m->devs.size() + 1);
        bwc_dev_pieces & p = c->pieces;
        size_t nbands = 0;
        int nout = 0;
        for (size_t d = 0; d < m->devs.size(); d++) {
            spmv_dev & dev = m->devs[d];
            uint64_t * const part = d ? vdst[d] : rdst;
            CUDA_CHECK(cudaSetDevice(dev.device));
            cudaEvent_t & e = c->pev[dev.device];
            if (!e)
                CUDA_CHECK(cudaEventCreateWithFlags(&e,
                            cudaEventDisableTiming));
            if (part != mdst) {
                ready_wait(adst, dev.stream);
                CUDA_CHECK(cudaMemcpyPeerAsync(
                            mdst + (size_t) dev.row0 * m->words, adst->device,
                            part, dev.device, dev.nrows * esz, dev.stream));
            }
            CUDA_CHECK(cudaEventRecord(e, dev.stream));
            tl_mark(tlg, dev.stream, dev.device, "g fan", m->calls);
            CUDA_CHECK(cudaStreamWaitEvent(c->stream, e, 0));
            c->bands[nbands] = { dev.row0 * esz, dev.nrows * esz, e };
            nbands++;
            /* one segment per block of the device, in row order */
            if (nout + dev.segs.size() > BWC_DEV_PIECES_MAX) {
                fprintf(stderr, "spmv_gf2: %zu output segments of %d devices"
                        " pass BWC_DEV_PIECES_MAX %d\n",
                        nout + dev.segs.size(), (int) m->devs.size(),
                        BWC_DEV_PIECES_MAX);
                abort();
            }
            for (gather_seg const & s : dev.segs)
                p.out[nout++] = { dev.device,
                    (char *) (part + (size_t) (s.r0 - dev.row0) * m->words),
                    s.r0 * esz, (s.r1 - s.r0) * esz, s.ev, &p };
        }
        CUDA_CHECK(cudaSetDevice(c->device));
        if (!c->bev)
            CUDA_CHECK(cudaEventCreateWithFlags(&c->bev,
                        cudaEventDisableTiming));
        if (ndst > m->nrows) {
            ready_wait(adst, c->stream);
            CUDA_CHECK(cudaMemsetAsync(mdst + (size_t) m->nrows * m->words, 0,
                        (ndst - m->nrows) * esz, c->stream));
            CUDA_CHECK(cudaEventRecord(c->bev, c->stream));
            c->bands[nbands] = { m->nrows * esz, (ndst - m->nrows) * esz,
                    c->bev };
            nbands++;
        }
        product_publish(adst, asrc, ndst * esz);
        adst->bands = c->bands.data();
        adst->nbands = (int) nbands;
        static bool const pieces = env_pieces();
        p.nout = pieces ? nout : 0;
        p.out_settled = false;
        p.out_end = m->nrows * esz;
        p.nrd = 0;
        if (pieces)
            replicas_publish(m, asrc, vsrc, 0, nsrc * esz, in_place);
        tl_flush(tlg, m->calls + 1);
    } else {
        for (size_t d = 0; d < m->devs.size(); d++) {
            spmv_dev & dev = m->devs[d];
            CUDA_CHECK(cudaSetDevice(dev.device));
            CUDA_CHECK(cudaMemcpyAsync(dst + (size_t) dev.row0 * m->words,
                        d ? vdst[d] : rdst, dev.nrows * esz,
                        cudaMemcpyDeviceToHost, dev.stream));
        }
        sync_all(m);
        if (ndst > m->nrows)
            memset(dst + (size_t) m->nrows * m->words, 0,
                    (ndst - m->nrows) * esz);
    }
    double const t3 = now();
    m->t_h2d += t1 - t0; m->t_kernel += t2 - t1; m->t_d2h += t3 - t2;
    m->calls++;
    m->rep_hits += rep;
    static unsigned long const period = timing_period();
    if (period && m->calls % period == 0)
        fprintf(stderr, "spmv_gf2 dev %d: call %lu at %.3f s\n",
                m->devs[0].device, m->calls, t3);
}

void spmv_gf2_scatter(spmv_gf2 * m, uint64_t * dst, const uint64_t * src,
                      unsigned int ndst, unsigned int nsrc)
{
    /* the probed split is timed on the gather kernels, so this
     * instance will not use its host row stream */
    probe_release(m);
    size_t const esz = (size_t) m->words * sizeof(uint64_t);
    bwc_device_arena * const asrc = mirror(m, src, nsrc * esz);
    bwc_device_arena * const adst = mirror(m, dst, ndst * esz);
    if (!asrc)
        pin(src, nsrc * esz);
    if (!adst)
        pin(dst, ndst * esz);
    std::shared_lock<std::shared_mutex> copies(pin_copy_mutex,
            std::defer_lock);
    if (!asrc || !adst)
        copies.lock();
    spmv_dev & r = m->devs[0];
    uint64_t ** const vsrc = scratch(m, m->vsrc, asrc);
    uint64_t ** const vdst = scratch(m, m->vdst, adst);
    pieces_release(m, adst);
    replicas_release(m, asrc);
    std::vector<bwc_device_band> const sb = src_bands(asrc);
    const uint64_t * rsrc = nullptr;
    if (asrc && sb.empty()) {
        rsrc = src_on_dev(m, asrc, src, nsrc * esz, vsrc[0]);
        CUDA_CHECK(cudaSetDevice(r.device));
        CUDA_CHECK(cudaEventRecord(r.ev, r.stream));
    }
    uint64_t * const mdst = adst ? (uint64_t *) adst->dev : nullptr;
    uint64_t * const racc =
        adst && adst->device == r.device ? mdst : vdst[0];
    if (adst && racc == mdst)
        ready_wait_locked(adst, r.stream);
    for (size_t d = 0; d < m->devs.size(); d++) {
        spmv_dev & dev = m->devs[d];
        CUDA_CHECK(cudaSetDevice(dev.device));
        const uint64_t * dsrc = vsrc[d];
        if (!asrc) {
            CUDA_CHECK(cudaMemcpyAsync(vsrc[d],
                        src + (size_t) dev.row0 * m->words,
                        dev.nrows * esz, cudaMemcpyHostToDevice, dev.stream));
        } else if (!sb.empty()) {
            /* the device pulls only its row slice, each range behind
             * its band event */
            if (!d && asrc->device == r.device) {
                wait_bands(sb, r.stream, 0, r.nrows * esz);
                dsrc = (const uint64_t *) asrc->dev;
            } else {
                copy_bands(asrc, sb, dev, (char *) vsrc[d],
                            (size_t) dev.row0 * esz, dev.nrows * esz);
            }
        } else if (!d) {
            dsrc = rsrc;
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(dev.stream, r.ev, 0));
            CUDA_CHECK(cudaMemcpyPeerAsync(vsrc[d], dev.device,
                        rsrc + (size_t) dev.row0 * m->words, r.device,
                        dev.nrows * esz, dev.stream));
        }
        uint64_t * const dacc = d ? vdst[d] : racc;
        CUDA_CHECK(cudaMemsetAsync(dacc, 0, ndst * esz, dev.stream));
        launch_scatter(m, dev, dsrc, dacc);
        /* r.ev must keep marking "source ready" until every device has
         * enqueued its wait; the reduction tree never waits on r */
        if (d)
            CUDA_CHECK(cudaEventRecord(dev.ev, dev.stream));
    }
    for (size_t stride = 1; stride < m->devs.size(); stride *= 2) {
        for (size_t i = 0; i + stride < m->devs.size(); i += 2 * stride) {
            spmv_dev & a = m->devs[i];
            spmv_dev & b = m->devs[i + stride];
            CUDA_CHECK(cudaSetDevice(a.device));
            CUDA_CHECK(cudaStreamWaitEvent(a.stream, b.ev, 0));
            CUDA_CHECK(cudaMemcpyPeerAsync(a.vtmp, a.device,
                        vdst[i + stride], b.device, ndst * esz, a.stream));
            unsigned int const nw = ndst * m->words;
            unsigned int const blocks = (nw + 255) / 256;
            uint64_t * const dacc = i == 0 ? racc : vdst[i];
            xor_kernel<<<blocks, 256, 0, a.stream>>>(dacc, a.vtmp, nw);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(a.ev, a.stream));
        }
    }
    /* every device chains into racc through the events, so the stream
     * of the first device carries the whole product */
    CUDA_CHECK(cudaSetDevice(r.device));
    if (adst) {
        /* every wait on r.ev is enqueued by now, so the re-record only
         * hands the settled product to the arena stream */
        CUDA_CHECK(cudaEventRecord(r.ev, r.stream));
        auto * const c = (arena_ctx *) adst->owner;
        auto * const csrc = (arena_ctx *) asrc->owner;
        arena_lock2 const lock(c, csrc);
        CUDA_CHECK(cudaStreamWaitEvent(c->stream, r.ev, 0));
        if (racc != mdst) {
            ready_wait(adst, c->stream);
            CUDA_CHECK(cudaMemcpyPeerAsync(mdst, adst->device, racc,
                        r.device, ndst * esz, c->stream));
        }
        product_publish(adst, asrc, ndst * esz);
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(r.stream));
    CUDA_CHECK(cudaMemcpy(dst, vdst[0], ndst * esz,
                cudaMemcpyDeviceToHost));
}

void spmv_gf2_free(spmv_gf2 * m)
{
    if (!m) return;
    if (getenv("SPMV_GF2_TIMING") && m->calls)
        fprintf(stderr, "spmv_gf2 dev %d+%zu: %lu calls, %lu replica hits,"
                " h2d %.1f us/call, kernel %.1f us/call, d2h %.1f us/call\n",
                m->devs[0].device, m->devs.size(), m->calls, m->rep_hits,
                1e6 * m->t_h2d / m->calls,
                1e6 * m->t_kernel / m->calls, 1e6 * m->t_d2h / m->calls);
    for (spmv_dev & dev : m->devs) {
        CUDA_CHECK(cudaSetDevice(dev.device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (auto * role : { &m->vsrc, &m->vdst })
        for (auto const & kv : *role)
            for (size_t d = 0; d < kv.second.size(); d++) {
                CUDA_CHECK(cudaSetDevice(m->devs[d].device));
                CUDA_CHECK(cudaFree(kv.second[d]));
            }
    for (spmv_dev & dev : m->devs) {
        CUDA_CHECK(cudaSetDevice(dev.device));
        CUDA_CHECK(cudaFree(dev.rowptr));
        CUDA_CHECK(cudaFree(dev.cols));
        CUDA_CHECK(cudaFree(dev.chunks));
        if (dev.vtmp) CUDA_CHECK(cudaFree(dev.vtmp));
        for (cudaEvent_t e : dev.pe)
            CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaEventDestroy(dev.ev));
        CUDA_CHECK(cudaEventDestroy(dev.cev));
        CUDA_CHECK(cudaStreamDestroy(dev.stream));
        CUDA_CHECK(cudaStreamDestroy(dev.cstream));
    }
    delete m;
}
