/* CUDA sieve core.  Same algorithms as the CPU core, arranged as four
 * kernels per special-q:
 *
 *   1. transform_kernel: factor-base roots to the q-lattice
 *   2. fill_kernel:      large entries walk their (i,j) lattice and
 *                        append hits to per-region buckets
 *   3. region_kernel:    one block per 2^15-position region; computes
 *                        the norms in shared memory, applies the small
 *                        entries and the buckets, then tests the
 *                        survivor bound (side 1 pass, then side 0 pass
 *                        which intersects, emits survivors, and rescans
 *                        both bucket lists for the large resieve hits)
 *   4. resieve_kernel:   small prime entries report hits on survivor
 *                        positions
 *
 * The sieve region never touches global memory as a whole array: only
 * bucket hits, survivor bitmasks and result lists do.
 *
 * Several special-q are in flight at the same time: each host thread
 * takes one of N_SLOTS slots, and each slot owns a stream plus all the
 * device and pinned buffers for one special-q.
 */

#include "sieve_core.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <gmp.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t const err_ = (call);                                \
        if (err_ != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                    cudaGetErrorString(err_), __FILE__, __LINE__);      \
            abort();                                                    \
        }                                                               \
    } while (0)

namespace {

constexpr int MAX_SLOTS = 8;                  /* special-q in flight */
constexpr uint32_t FCAND_MAX = 1u << 15;      /* assembled candidates per chunk */
constexpr int SREG_LOG = 15;                  /* positions per region */
constexpr uint32_t SREG = 1u << SREG_LOG;
constexpr uint32_t SKIP_ROOT = UINT32_MAX;

/* filter meta word of a kept survivor: bit 31 assembled, bits 29-30
 * the leftover of side 0/1 waits for its primality test, bits 16-28
 * nq, bits 8-15 nf1, bits 0-7 nf0.  FMETA_REJECT marks a record the
 * primality test dropped; the pack step leaves it out of the chunk. */
constexpr uint32_t FMETA_REJECT = 1u << 30;

/* per-entry walk cursor sentinels */
constexpr uint64_t WALK_SKIP = UINT64_MAX;    /* entry never hits */
constexpr uint64_t WALK_PROJ = 0;             /* projective: row indexed */

/* GLAS_FILL_CARRY: the walk state of one large entry (index >= n_band)
 * in one 24-byte record.  A companion byte per entry holds the chunk
 * of its next hit; the fill kernel walks only the entries due in the
 * current chunk, so an entry with no hit in a chunk costs one byte
 * read. */
struct carry_state {
    uint2 L0, L1;      /* reduced FK basis (uint4 would pad to 32 bytes) */
    uint64_t x;        /* walk cursor, or WALK_SKIP / WALK_PROJ */
};
static_assert(sizeof(carry_state) == 24, "carry_state is 24 bytes");
constexpr uint8_t CARRY_DONE = 255;   /* no hit in any later chunk */

/* Sieve cells are 16-bit: BIAS keeps them positive, NORM_CAP keeps
 * BIAS + norm + the survivor test inside 16 bits. */
constexpr uint32_t BIAS = 1u << 12;
constexpr uint32_t NORM_CAP = 0x3fff;

/* bucket entry: offset in region (15) | weight (13) | resieve (1) |
 * q (32) | unused (3) */
constexpr int BK_W_SHIFT = 15;
constexpr int BK_RESIEVE_SHIFT = 28;
constexpr int BK_Q_SHIFT = 29;
constexpr uint32_t BK_W_MASK = (1u << (BK_RESIEVE_SHIFT - BK_W_SHIFT)) - 1;
static_assert(BK_Q_SHIFT + 32 <= 64);

constexpr int FILL_THREADS = 256;
/* blocks per SM the fill kernel is compiled to fit (register cap
 * 65536 / (256 * N)); 0 leaves the register count to ptxas */
#ifndef GLAS_FILL_MINBLOCKS
#define GLAS_FILL_MINBLOCKS 0
#endif
#if GLAS_FILL_MINBLOCKS > 0
#define FILL_BOUNDS __launch_bounds__(FILL_THREADS, GLAS_FILL_MINBLOCKS)
#else
#define FILL_BOUNDS
#endif
/* GLAS_FILL_STAGE: threads per block of the exchange-staged fill */
constexpr int FILL_STAGE_THREADS = 256;

/* Each region's bucket counter gets its own 128-byte L2 line.  The L2
 * serializes atomics that share a line, so packed counters queue each
 * fill atomic behind atomics for 31 unrelated regions, which inflates
 * the latency of the returned slot index that the bucket store needs. */
constexpr uint32_t BK_CNT_STRIDE = 32;

struct gpu_fb_entry {
    uint32_t q;
    uint32_t r;
    uint32_t weight;
    uint32_t resieve;
};

/* what the region and resieve kernels read of a small entry; one
 * 16-byte load */
struct __align__(16) gpu_small_entry {
    uint32_t q;
    uint32_t weight;
    uint64_t magic;           /* (2^64-1)/q */
};

/* One chunk of a special-q: rows [j0, j0 + rows).  Chunk positions are
 * 32-bit offsets from j0 << logI; the host adds the base back. */
struct gpu_task {
    uint64_t q;
    uint64_t rho;
    int sq_side;
    int64_t a0, b0, a1, b1;
    int logI;
    uint32_t J;
    uint32_t j0;
    uint32_t rows;
    uint32_t area;      /* rows << logI, at most 2^31 */
    uint32_t nregions;
    int degree[2];
    double fijd[2][GLAS_MAX_DEGREE + 1];
    int32_t bound[2];
    int skip;   /* probe bits: 1 norms, 2 small walk, 4 bucket apply,
                   8 bucket stores, 64 index-derived stores */
};

enum { PH_TRANSFORM, PH_FILL, PH_REGION, PH_TINY, PH_RESIEVE, N_PHASES };
/* stream events per chunk: the start, then one per phase boundary (the
 * GLAS_FILL_GATE order has two fill and two region phases) */
constexpr int N_EVENTS = 8;

/* tiny primes and the norm coefficients mod each, from the tiny_table */
constexpr uint32_t MAX_TINY = 64;
__constant__ uint32_t c_tiny_np;
__constant__ uint32_t c_tiny_p[MAX_TINY];
__constant__ uint64_t c_tiny_M[MAX_TINY];
__constant__ uint32_t c_tiny_c[2][MAX_TINY * (GLAS_MAX_DEGREE + 1)];

/* GLAS_BUCKET_SLAB: the bucket table's stride in regions; 0 selects the
 * plain [region][bucket_cap] layout. */
__constant__ uint32_t c_bucket_slab;

/* GLAS_NORM_REG: the region norm samples evaluate the row polynomial
 * from register coefficients and share the next span's samples across the
 * warp; 0 selects the shared-memory coefficients and the lane-31 sample. */
__constant__ uint32_t c_norm_reg;

/* GLAS_NORM_STRIDE: the number of consecutive cells a lane owns in the
 * norm phase of the register-coefficient path.  The warp evaluates the
 * row polynomial at the first cell of each lane run and interpolates the
 * log units across the rest.  The default 4 reproduces main; 8, 16 and
 * 32 trade survivor accuracy for fewer evaluations and change the
 * relation set. */
__constant__ uint32_t c_norm_stride;

/* GLAS_FILL_COPRIME: the fill also drops the hits where 3 divides both
 * i and j (the hits where 2 divides both are always dropped). */
__constant__ uint32_t c_fill_coprime;

/* GLAS_FILL_GATE_SHIFT: the side-0 fill gate has one bit per
 * 2^c_gate_shift positions; 0 is the survivor mask itself. */
__constant__ uint32_t c_gate_shift;

/* True when the hit at row j and offset off = i + I/2 can give a
 * relation.  The survivor test removes the others anyway. */
__device__ inline bool dev_fill_keeps(gpu_task const & t, uint32_t j,
                                      uint32_t off)
{
    uint32_t const h = 1u << (t.logI - 1);
    if (!((j | off) & 1))
        return false;
    return !c_fill_coprime || j % 3 || off % 3 != h % 3;
}

/* Index of slot b of the region's bucket.  The slab layout puts piece k
 * (64 entries) of every region into slab k, so the write fronts of all
 * regions share a few pages and the address translation hits. */
__device__ __forceinline__ uint64_t dev_bucket_idx(uint32_t region,
                                                   uint32_t b,
                                                   uint32_t bucket_cap)
{
    if (c_bucket_slab == 0)
        return (uint64_t)region * bucket_cap + b;
    return ((uint64_t)(b >> 6) * c_bucket_slab + region) * 64 + (b & 63);
}

/* Device + pinned host buffers for one in-flight special-q.  The
 * chunk output buffers exist twice, indexed by chunk parity: the
 * sieve stream writes chunk k+1 into one set while the out stream
 * filters and copies chunk k from the other, so the sieve stream
 * never waits for the CPU between chunks.  The filter kernel writes
 * the rows it keeps straight into the mapped pinned host buffers
 * (h_survivors, h_tiny, h_hits, h_fmeta, h_ffac, h_flo); without the
 * filter the out stream copies the whole chunk output into them. */
struct gpu_slot {
    cudaStream_t stream;      /* chunk kernels */
    cudaStream_t out_stream;  /* filter kernel and D2H copies */
    uint32_t * roots[2];
    uint64_t * walk_x[2];     /* FK walk resume position per entry */
    uint4 * lattice[2];       /* reduced FK basis per large entry */
    carry_state * carry[2];   /* GLAS_FILL_CARRY: [entry - n_band] */
    uint8_t * nextc[2];       /* GLAS_FILL_CARRY: next hit chunk per entry */
    uint32_t * active[2];     /* GLAS_FILL_CARRY: entries due this chunk */
    uint32_t * n_active;      /* [side] */
    uint64_t * buckets[2];    /* [region][bucket_cap] packed hits */
    uint32_t * bucket_cnt[2]; /* [region * BK_CNT_STRIDE] */
    uint32_t * bitmask;       /* survivor bits, 1 bit per position */
    uint32_t * cgate;         /* GLAS_FILL_GATE_SHIFT: folded mask */
    uint32_t * survivors[2];  /* [parity] */
    uint32_t * rs_idx[2];     /* small entries to resieve this task */
    uint32_t * rs_cnt;        /* [side] */
    uint64_t * tiny[2];       /* [parity][survivor][side] tiny masks */
    uint64_t * hits[2];       /* [parity] packed resieve hits */
    uint64_t * hits_r[2];     /* GLAS_RHITS: [parity][region][slot] */
    uint32_t * hit_rcnt[2];   /* GLAS_RHITS: [parity][region] lengths */
    uint64_t * hits_ov[2];    /* GLAS_RHITS: [parity] overflow tail */
    uint32_t * counters[2];   /* [parity] survivor_cnt, hit_cnt, overflow,
                                 overflow-tail count */
    uint64_t * hits_sorted;   /* hits ordered by position, filter input */
    void * sort_temp;
    void * select_temp;       /* GLAS_FILL_CARRY: cub::DeviceSelect scratch */
    uint32_t * fcnt;          /* filter output: kept survivors, host hits */
    cudaEvent_t fev[2];
    uint32_t * h_counters;
    uint32_t * h_survivors;
    uint64_t * h_tiny;
    uint64_t * h_hits;        /* with the filter: hits of the kept
                                 survivors the host assembles */
    uint32_t * h_fmeta;       /* per kept survivor: factor counts + flag */
    uint32_t * h_ffac;        /* [fcand_cap][GPU_FAC_CAP] divided-out primes */
    uint64_t * h_flo;         /* [fcand_cap][side][2 limbs] leftover norms */
    uint32_t * h_bucket;      /* scratch for the bucket peak stat */
    cudaEvent_t kdone[2];     /* [parity] chunk kernels done */
    cudaEvent_t odone[2];     /* [parity] chunk output copies done */
    cudaEvent_t ev[2][N_EVENTS];
    int ev_phase[2][N_EVENTS];  /* the phase that ends at each event */
    int n_ev[2];
};

struct gpu_context {
    gpu_fb_entry * fb[2];
    gpu_small_entry * small[2];
    std::vector<gpu_fb_entry> h_fb[2];        /* gpu_prepare builds, gpu_init uploads */
    std::vector<gpu_small_entry> h_small[2];
    uint32_t n_fb[2];
    uint32_t n_band[2];       /* entries with q < fill_bands * I come first */
    uint32_t n_owned[2];
    uint32_t n_local[2];
    uint32_t n_small[2];      /* entries with q < I come before those */
    uint32_t n_dense[2];      /* entries with q < I/128 come before those */
    uint32_t n_skip[2];       /* leading entries with q < skip bound, not sieved */
    uint32_t n_pat[2];        /* the pattern entries come first in the dense tier */
    int32_t skip_comp[2];     /* bound compensation for skipped entries */
    uint32_t bucket_cap[2];   /* hits per region per side */
    uint64_t chunk_area;      /* positions sieved per kernel pass */
    uint32_t max_regions;
    uint32_t max_survivors;
    uint32_t max_hits;
    int n_slots;
    uint32_t fill_bands = 2;  /* fill threads per entry in [n_small, n_band) */
    bool carry = false;       /* GLAS_FILL_CARRY */
    bool bucket_slab = true;  /* GLAS_BUCKET_SLAB=0 selects the plain layout */
    bool norm_reg = true;     /* GLAS_NORM_REG=0 selects the shared-memory path */
    uint32_t norm_stride = 4; /* GLAS_NORM_STRIDE: cells per lane run */
    bool fill_coprime = true; /* GLAS_FILL_COPRIME=0 keeps the 3 | gcd(i, j) hits */
    bool fill_gate = true;    /* GLAS_FILL_GATE=0 fills side 0 ungated */
    uint32_t gate_shift = 3;  /* GLAS_FILL_GATE_SHIFT=0 reads the survivor mask */
    bool fill_tail0 = true;   /* GLAS_FILL_TAIL0=0 walks the side-0 tail in the row kernel */
    bool region_cand = true;  /* GLAS_REGION_CAND=0 computes all side-0 cells */
    uint32_t carry_grid = 0;
    bool fill_stage = false;  /* GLAS_FILL_STAGE */
    uint32_t fill_owners = 1;
    uint32_t fill_pair_slots = 0;
    bool fill_pair_fused = true;
    uint32_t fill_row_bands = 0;
    bool fill_row_group = true;
    bool fill_row_fused = true;
    uint32_t fill_row_window = 0;
    int fill_row_window_side = -1;
    bool fill_row_window_lean = false;
    uint32_t fill_row_window_blocks = 2;
    uint32_t fill_row_cluster = 1;
    uint32_t fill_ahead = 0;
    bool fill_owner_pairs = true;
    uint32_t fill_stage_grid = 0;
    uint32_t fill_stage_regions = 0;  /* regions with a slot in shared memory */
    gpu_slot slot[MAX_SLOTS];
    std::mutex lock;
    std::condition_variable cv;
    uint32_t free_mask = 0;
    bool stats = false;
    bool pipeline = false;
    bool filter = false;
    bool assemble = false;    /* filter kernel also assembles candidates */
    bool rhits = false;       /* resieve hits land in per-region lists */
    bool host_mr = false;     /* the host runs the deferred primality tests */
    uint32_t fcand_cap = 0;   /* fmeta/ffac/flo rows per slot */
    /* launch grids of the grid-striding kernels: one wave of resident
     * blocks on this device */
    uint32_t filter_grid = 0, tiny_grid = 0, rs_grid = 0;
    size_t sort_temp_bytes = 0;
    size_t select_temp_bytes = 0;
    double filter_ms = 0;
    double ecm_ms = 0;        /* cofactor batches, classify to last round */
    int skip = 0;
    int xform_repeat = 1;   /* transform launches per task, timing probe */
    double phase_ms[N_PHASES] = {};
    cudaEvent_t ref;
    std::vector<std::pair<float, float>> spans[MAX_SLOTS];
    uint32_t bucket_max[2] = {};
    uint64_t bucket_sum[2] = {};   /* bucket entries over all chunks */
    FILE * bucket_dump = nullptr;  /* GLAS_BUCKET_DUMP: counts per chunk */
};

char const * const PHASE_NAMES[N_PHASES] = {"transform", "fill", "region",
                                            "tiny", "resieve_s"};

gpu_context ctx;

int acquire_slot()
{
    trace_scope const tr(TR_ACQUIRE);
    std::unique_lock<std::mutex> lk(ctx.lock);
    ctx.cv.wait(lk, [] { return ctx.free_mask != 0; });
    int const k = __builtin_ctz(ctx.free_mask);
    ctx.free_mask &= ~(1u << k);
    return k;
}

void release_slot(int k)
{
    ctx.lock.lock();
    ctx.free_mask |= 1u << k;
    ctx.lock.unlock();
    ctx.cv.notify_one();
}

/* One chunk's raw GPU output, copied out of the slot's pinned buffers
 * so the slot can start the next chunk; a worker unpacks it later. */
struct raw_chunk {
    uint64_t xbeg;
    uint32_t n_out, nc, n_hits;
    std::vector<uint32_t> survivors;   /* [n_out] */
    std::vector<uint64_t> tiny;        /* [2 * n_out] */
    std::vector<uint64_t> hits;        /* [n_hits] */
    std::vector<uint32_t> fmeta;       /* [nc] */
    std::vector<uint32_t> ffac;        /* [nc * GPU_FAC_CAP] */
    std::vector<uint64_t> flo;         /* [nc * 4] */
};

/* Host side of one chunk: unpack the raw output into the sieve_result
 * the workers consume. */
static sieve_result assemble_chunk(raw_chunk const & rc)
{
    trace_scope const tr(TR_ASSEMBLE);
    sieve_result res;
    if (rc.nc)
        res.cands.reserve(rc.n_out);
    else
        res.survivors.reserve(rc.n_out);
    for (uint32_t k = 0; k < rc.n_out; k++) {
        uint32_t const meta = k < rc.nc ? rc.fmeta[k] : 0;
        if (!(meta >> 31)) {
            res.survivors.push_back(
                {rc.xbeg + rc.survivors[k],
                 {rc.tiny[2 * k], rc.tiny[2 * k + 1]}});
            continue;
        }
        gpu_cand c;
        c.x = rc.xbeg + rc.survivors[k];
        c.nf[0] = meta & 0xff;
        c.nf[1] = meta >> 8 & 0xff;
        c.nq = meta >> 16 & 0x1fff;
        c.pend = meta >> 29 & 3;
        memcpy(c.leftover, rc.flo.data() + 4 * (uint64_t)k,
               sizeof c.leftover);
        memcpy(c.fac, rc.ffac.data() + (uint64_t)k * GPU_FAC_CAP,
               (c.nf[0] + c.nf[1]) * 4);
        res.cands.push_back(c);
    }
    res.hits.reserve(rc.n_hits);
    for (uint32_t k = 0; k < rc.n_hits; k++) {
        uint64_t const h = rc.hits[k];
        res.hits.push_back({rc.xbeg + (h >> 32),
                            (uint32_t)(h & 0xffffffff) >> 1,
                            (uint8_t)(h & 1)});
    }
    std::sort(res.survivors.begin(), res.survivors.end(),
              [](survivor const & u, survivor const & v) {
                  return u.x < v.x;
              });
    std::sort(res.cands.begin(), res.cands.end(),
              [](gpu_cand const & u, gpu_cand const & v) {
                  return u.x < v.x;
              });
    sort_hits(res.hits);
    return res;
}

/* Copies the n_out kept rows of a slot's pinned buffers into the chunk
 * and leaves out the FMETA_REJECT rows.  The first min(n_out, fcand_cap)
 * rows carry assembled records; the rows after them have none. */
static void pack_rows(gpu_slot const & s, uint32_t n_out, raw_chunk & rc)
{
    uint32_t const nkeep = std::min(n_out, ctx.fcand_cap);
    uint32_t const nc =
        nkeep - std::count(s.h_fmeta, s.h_fmeta + nkeep, FMETA_REJECT);
    rc.n_out = n_out - (nkeep - nc);
    rc.nc = nc;
    rc.survivors.reserve(rc.n_out);
    rc.tiny.reserve(2 * (uint64_t)rc.n_out);
    rc.fmeta.reserve(nc);
    rc.ffac.reserve((uint64_t)nc * GPU_FAC_CAP);
    rc.flo.reserve(4 * (uint64_t)nc);
    for (uint32_t k = 0; k < nkeep; k++) {
        if (s.h_fmeta[k] == FMETA_REJECT)
            continue;
        rc.survivors.push_back(s.h_survivors[k]);
        rc.tiny.insert(rc.tiny.end(), s.h_tiny + 2 * (uint64_t)k,
                       s.h_tiny + 2 * (uint64_t)k + 2);
        rc.fmeta.push_back(s.h_fmeta[k]);
        rc.ffac.insert(rc.ffac.end(), s.h_ffac + (uint64_t)k * GPU_FAC_CAP,
                       s.h_ffac + (uint64_t)(k + 1) * GPU_FAC_CAP);
        rc.flo.insert(rc.flo.end(), s.h_flo + 4 * (uint64_t)k,
                      s.h_flo + 4 * (uint64_t)k + 4);
    }
    rc.survivors.insert(rc.survivors.end(), s.h_survivors + nkeep,
                        s.h_survivors + n_out);
    rc.tiny.insert(rc.tiny.end(), s.h_tiny + 2 * (uint64_t)nkeep,
                   s.h_tiny + 2 * (uint64_t)n_out);
}

/* What each traced thread did during the holes, next to its share of
   the whole span: a state whose hole share is far above its span
   share is what stalls the GPU. */
static void report_hole_states(
    std::vector<std::pair<float, float>> const & holes, float span0,
    float span1)
{
    std::lock_guard<std::mutex> const lk(trace_lock);
    if (holes.empty() || trace_logs.empty())
        return;
    std::vector<std::array<double, TR_N>> per_hole(holes.size());
    double span_ms[2][TR_N] = {}, hole_ms[2][TR_N] = {};
    size_t nthr[2] = {};
    for (auto const & log : trace_logs) {
        int const g = log->worker;
        nthr[g]++;
        auto const & ev = log->ev;
        size_t h = 0;
        for (size_t i = 0; i < ev.size(); i++) {
            double const a = std::max(ev[i].first, span0);
            double const b =
                std::min(i + 1 < ev.size() ? ev[i + 1].first : span1, span1);
            if (b <= a)
                continue;
            int const st = ev[i].second;
            span_ms[g][st] += b - a;
            while (h < holes.size() && holes[h].second <= a)
                h++;
            for (size_t k = h; k < holes.size() && holes[k].first < b; k++) {
                double const d = std::min(b, (double)holes[k].second) -
                                 std::max(a, (double)holes[k].first);
                hole_ms[g][st] += d;
                if (!g)
                    per_hole[k][st] += d;
            }
        }
    }
    auto const pct = [](double x, double total) {
        return total ? 100 * x / total : 0.0;
    };
    for (int g = 0; g < 2; g++) {
        if (!nthr[g])
            continue;
        double hole_total = 0, span_total = 0;
        for (int s = 0; s < TR_N; s++) {
            hole_total += hole_ms[g][s];
            span_total += span_ms[g][s];
        }
        fprintf(stderr,
                "# GPU %s states, %zu threads (in holes vs span):\n",
                g ? "worker" : "siever", nthr[g]);
        for (int s = 0; s < TR_N; s++)
            fprintf(stderr, "#   %-8s %5.1f%% vs %5.1f%%\n", TRACE_NAMES[s],
                    pct(hole_ms[g][s], hole_total),
                    pct(span_ms[g][s], span_total));
    }
    std::vector<size_t> order(holes.size());
    for (size_t i = 0; i < order.size(); i++)
        order[i] = i;
    auto const dur = [&](size_t i) {
        return holes[i].second - holes[i].first;
    };
    size_t const ntop = std::min<size_t>(10, order.size());
    std::partial_sort(order.begin(), order.begin() + ntop, order.end(),
                      [&](size_t u, size_t v) { return dur(u) > dur(v); });
    fprintf(stderr, "# GPU top hole states (siever-ms per state):\n");
    for (size_t i = 0; i < ntop; i++) {
        size_t const g = order[i];
        fprintf(stderr, "#   %7.3f ms @ %9.1f:", dur(g), holes[g].first);
        for (int s = 0; s < TR_N; s++)
            if (per_hole[g][s] >= 0.05)
                fprintf(stderr, " %s %.1f", TRACE_NAMES[s], per_hole[g][s]);
        fprintf(stderr, "\n");
    }
}

/* Union of all chunk spans across slots. A hole in the union is time
   when no slot had chunk work in flight: launch, sync, or tail idle. */
void report_idle()
{
    std::vector<std::pair<float, float>> all;
    for (int k = 0; k < ctx.n_slots; k++)
        all.insert(all.end(), ctx.spans[k].begin(), ctx.spans[k].end());
    if (all.empty())
        return;
    std::sort(all.begin(), all.end());
    float wall1 = all[0].second;
    for (auto const & sp : all)
        wall1 = std::max(wall1, sp.second);
    double busy = 0;
    std::vector<std::pair<float, float>> holes;
    float cur0 = all[0].first, cur1 = all[0].second;
    for (size_t i = 1; i < all.size(); i++) {
        if (all[i].first > cur1) {
            holes.emplace_back(cur1, all[i].first);
            busy += cur1 - cur0;
            cur0 = all[i].first;
        }
        cur1 = std::max(cur1, all[i].second);
    }
    busy += cur1 - cur0;
    double const wall = wall1 - all[0].first;
    std::vector<float> gaps;
    for (auto const & g : holes)
        gaps.push_back(g.second - g.first);
    std::sort(gaps.rbegin(), gaps.rend());
    double gap_sum = 0;
    for (float g : gaps)
        gap_sum += g;
    fprintf(stderr,
            "# GPU chunk span %.2f s, covered %.2f s, holes %.2f s (%.2f%%)\n",
            wall / 1e3, busy / 1e3, (wall - busy) / 1e3,
            100.0 * (wall - busy) / wall);
    fprintf(stderr, "# GPU holes: %zu, sum %.3f s, top (ms):", gaps.size(),
            gap_sum / 1e3);
    for (size_t i = 0; i < gaps.size() && i < 10; i++)
        fprintf(stderr, " %.3f", gaps[i]);
    fprintf(stderr, "\n# GPU slot tails (ms):");
    for (int k = 0; k < ctx.n_slots; k++) {
        if (ctx.spans[k].empty()) {
            fprintf(stderr, " -");
            continue;
        }
        float last = 0;
        for (auto const & sp : ctx.spans[k])
            last = std::max(last, sp.second);
        fprintf(stderr, " %.1f", wall1 - last);
    }
    fprintf(stderr, "\n");
    report_hole_states(holes, all[0].first, wall1);
}

/* ---------------- device helpers ---------------- */

__device__ uint32_t dev_invmod(uint32_t a, uint32_t m)
{
    int64_t t = 0, newt = 1;
    uint32_t r = m, newr = a % m;
    while (newr != 0) {
        uint32_t const qq = r / newr;
        int64_t const t2 = t - (int64_t)qq * newt;
        t = newt; newt = t2;
        uint32_t const r2 = r - qq * newr;
        r = newr; newr = r2;
    }
    if (r != 1)
        return 0;
    return (uint32_t)(t < 0 ? t + (int64_t)m : t);
}

__device__ uint32_t dev_invmod_po2(uint32_t v)
{
    uint32_t x = v;
    for (int k = 0; k < 4; k++)
        x = x * (2 - v * x);
    return x;
}

__device__ uint32_t dev_mod_i128(__int128 v, uint32_t m)
{
    __int128 const r = v % (__int128)m;
    return (uint32_t)(r < 0 ? r + (__int128)m : r);
}

__device__ uint32_t dev_mod_i64(int64_t v, uint32_t m)
{
    int64_t const r = v % (int64_t)m;
    return (uint32_t)(r < 0 ? r + (int64_t)m : r);
}

__device__ uint64_t dev_iabs(int64_t v)
{
    return v < 0 ? -(uint64_t)v : (uint64_t)v;
}

__device__ uint32_t dev_gcd(uint32_t a, uint32_t b)
{
    while (b) {
        uint32_t const t = a % b;
        a = b;
        b = t;
    }
    return a;
}

__device__ uint32_t dev_transform_root(gpu_task const & t, uint32_t q,
                                       uint32_t r_enc)
{
    if ((q & 1) == 0) {
        bool const proj = r_enc >= q;
        uint32_t const R = proj ? r_enc - q : r_enc;
        uint32_t u, v;
        if (!proj) {
            u = (uint32_t)((int64_t)R * (t.b1 % (int64_t)q) - t.a1);
            v = (uint32_t)(t.a0 - (int64_t)R * (t.b0 % (int64_t)q));
        } else {
            u = (uint32_t)(t.b1 - (int64_t)R * (t.a1 % (int64_t)q));
            v = (uint32_t)((int64_t)R * (t.a0 % (int64_t)q) - t.b0);
        }
        if (v & 1)
            return (u * dev_invmod_po2(v)) & (q - 1);
        if (!(u & 1))
            return SKIP_ROOT;
        return q + ((dev_invmod_po2(u) * v) & (q - 1));
    }
    bool const proj = r_enc >= q;
    int64_t const R = proj ? r_enc - q : r_enc;
    uint32_t u, v;
    /* R < 2^32, so with all coefficients below 2^30 the products fit
     * in int64. */
    uint64_t const cmax = dev_iabs(t.a0) | dev_iabs(t.a1) |
                          dev_iabs(t.b0) | dev_iabs(t.b1);
    if (cmax < (1u << 30)) {
        int64_t const num = proj ? t.b1 - R * t.a1 : R * t.b1 - t.a1;
        int64_t const den = proj ? R * t.a0 - t.b0 : t.a0 - R * t.b0;
        u = dev_mod_i64(num, q);
        v = dev_mod_i64(den, q);
    } else {
        __int128 const num = proj ? (__int128)t.b1 - (__int128)R * t.a1
                                  : (__int128)R * t.b1 - t.a1;
        __int128 const den = proj ? (__int128)R * t.a0 - t.b0
                                  : (__int128)t.a0 - (__int128)R * t.b0;
        u = dev_mod_i128(num, q);
        v = dev_mod_i128(den, q);
    }
    uint32_t const vinv = dev_invmod(v, q);
    if (vinv)
        return (uint32_t)((uint64_t)u * vinv % q);
    uint32_t const uinv = dev_invmod(u, q);
    if (!uinv)
        return SKIP_ROOT;
    return q + (uint32_t)((uint64_t)v * uinv % q);
}

/* Franke-Kleinjung reduction, identical to the CPU core. */
struct dev_plattice {
    uint32_t mi0, j0, i1, j1;

    __device__ dev_plattice(uint32_t q, uint32_t r, uint32_t I)
    {
        mi0 = q; j0 = 0; i1 = r; j1 = 1;
        if (i1 == 0) {
            reduce_with_vertical_vector(I);
            return;
        }
        for (;;) {
            if (i1 < I) {
                if (i1 == 0) {
                    j0 = j1 - j0;
                    reduce_with_vertical_vector(I);
                    return;
                }
                uint32_t const a = (mi0 + i1 - I) / i1;
                mi0 -= a * i1;
                j0 += a * j1;
                return;
            }
            { uint32_t const k = mi0 / i1; mi0 -= k * i1; j0 += k * j1; }
            if (mi0 < I) {
                if (mi0 == 0) {
                    mi0 = i1;
                    i1 = j0; j0 = j1; j1 = i1;
                    i1 = 0;
                    reduce_with_vertical_vector(I);
                    return;
                }
                uint32_t const a = (mi0 + i1 - I) / mi0;
                i1 -= a * mi0;
                j1 += a * j0;
                return;
            }
            { uint32_t const k = i1 / mi0; i1 -= k * mi0; j1 += k * j0; }
        }
    }

    __device__ void reduce_with_vertical_vector(uint32_t I)
    {
        uint32_t const a = (I - 1) % mi0;
        uint32_t mi2 = (I - 1) - a - i1;
        uint32_t const j2 = j1;
        if (mi0 + mi2 < I)
            mi2 += mi0;
        i1 = mi0;
        j1 = j0;
        mi0 = mi2;
        j0 = j2;
    }
};

/* Walks a projective root U on q: rows j = n*g, i = n * t0 (mod qq);
 * calls emit(x - chunk base) for every hit in the rows [jb, je). */
template <typename F>
__device__ void dev_proj_walk(gpu_task const & t, uint32_t q, uint32_t U,
                              uint32_t jb, uint32_t je, F emit)
{
    uint32_t const I = 1u << t.logI;
    uint64_t const xbeg = (uint64_t)t.j0 << t.logI;
    uint32_t const g = U ? dev_gcd(U, q) : q;
    uint32_t const qq = q / g;
    uint64_t const t0 = U ? dev_invmod(U / g, qq) : 0;
    uint64_t const n0 = max((uint64_t)1, ((uint64_t)jb + g - 1) / g);
    for (uint64_t j = n0 * g, n = n0; j < je; j += g, n++) {
        uint64_t const row = (j << t.logI) - xbeg;
        uint64_t const start = qq > 1 ? (t0 * n + I / 2) % qq : 0;
        for (uint64_t off = start; off < I; off += qq)
            if (dev_fill_keeps(t, (uint32_t)j, (uint32_t)off))
                emit((uint32_t)(row + off));
    }
}

/* FK walk of one affine entry from the lattice point x up to xend, with
 * the reduced basis L = (mi0, j0, i1, j1) cached by the transform
 * kernel.  Emits chunk offsets and returns the first lattice point at
 * or past xend: the cursor for the next chunk, so a multi-chunk area
 * costs no repeated walk work. */
template <typename F>
__device__ uint64_t dev_affine_walk(gpu_task const & t, uint4 L, uint64_t x,
                                    uint64_t xend, F emit)
{
    uint32_t const I = 1u << t.logI;
    uint64_t const xbeg = (uint64_t)t.j0 << t.logI;
    uint64_t const inc_warp = ((uint64_t)L.y << t.logI) - L.x;
    uint64_t const inc_step = ((uint64_t)L.w << t.logI) + L.z;
    uint32_t const bound_warp = I - L.z;
    uint32_t const bound_step = L.x;
    uint32_t const maskI = I - 1;
    while (x < xend) {
        if (x >= xbeg && dev_fill_keeps(t, (uint32_t)(x >> t.logI), x & maskI))
            emit((uint32_t)(x - xbeg));
        uint32_t const i = x & maskI;
        uint64_t inc = 0;
        if (i >= bound_warp)
            inc += inc_warp;
        if (i < bound_step)
            inc += inc_step;
        x += inc;
        if (inc == 0)
            break;
    }
    return x;
}

/* A non-null gate is the other side's survivor mask: a hit on a
 * cleared bit cannot touch a survivor or a resieve hit, so the fill
 * does not store it. */
__device__ inline bool dev_gate_drops(uint32_t const * gate, uint32_t x)
{
    return gate &&
           !(gate[x >> (5 + c_gate_shift)] >> (x >> c_gate_shift & 31) & 1);
}

/* GLAS_FILL_GATE_SHIFT: bit b of gate word w is set when a survivor bit
 * is set in the positions [(w << (5 + shift)) + (b << shift),
 * + 2^shift).  The gate is a superset of the mask, so the fill stores
 * more side-0 hits but the survivors and hits do not change. */
__global__ void gate_fold_kernel(uint32_t const * mask, uint32_t * gate,
                                 uint32_t shift, uint32_t words)
{
    uint32_t const w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= words)
        return;
    uint32_t const span = 1u << shift;
    uint32_t const per = 32 >> shift;   /* gate bits per mask word */
    uint64_t const low = (1ull << span) - 1;
    uint32_t g = 0;
    for (uint32_t k = 0; k < span; k++) {
        uint32_t const word = mask[((uint64_t)w << shift) + k];
        for (uint32_t c = 0; c < per; c++)
            g |= (uint32_t)((word >> (c << shift) & low) != 0) << (k * per + c);
    }
    gate[w] = g;
}

/* First lattice point i = r*j (mod q) of the strip in the rows [jb, je),
 * for an affine root r on q >= I: a row holds at most one strip point.
 * Returns the row je start when no such point exists. */
__device__ uint64_t dev_band_start(gpu_task const & t, uint32_t q, uint32_t r,
                                   uint32_t jb, uint32_t je)
{
    uint32_t const I = 1u << t.logI;
    uint32_t off = ((uint64_t)r * jb + I / 2) % q;
    for (uint32_t j = jb; j < je; j++) {
        if (off < I)
            return ((uint64_t)j << t.logI) + off;
        off += r;
        if (off >= q)
            off -= q;
    }
    return (uint64_t)je << t.logI;
}

/* m mod q with the magic multiplier M = (2^64-1)/q; needs m*q < 2^64. */
__device__ uint32_t dev_mod_small(uint64_t m, uint32_t q, uint64_t M)
{
    uint32_t r = (uint32_t)(m - __umul64hi(m, M) * q);
    if (r >= q)
        r -= q;
    return r;
}

/* Offset from w0 of the first hit of the affine root at or after
 * position w0 of row j: (root * j + I/2 - w0) mod q, with q * I added
 * to keep the argument positive. */
__device__ uint32_t dev_small_start(gpu_task const & t, uint32_t q,
                                    uint32_t root, uint64_t M, uint32_t j,
                                    uint32_t w0)
{
    uint32_t const I = 1u << t.logI;
    return dev_mod_small((uint64_t)root * j + I / 2 +
                             ((uint64_t)q << t.logI) - w0, q, M);
}

/* The first hit of one small entry in the window [w0, w0 + min(I, SREG))
 * of row j, as the offset from w0, and the hit stride qq (q, or q / g
 * for a projective root) as the return value; 0 when the row has no
 * hit.  The offset is below q, so the walk arithmetic stays 32-bit.  M
 * is the magic multiplier of q. */
__device__ uint32_t dev_small_walk_start(gpu_task const & t, uint32_t q,
                                         uint32_t root, uint64_t M,
                                         uint32_t j, uint32_t w0,
                                         uint32_t & off)
{
    uint32_t const I = 1u << t.logI;
    if (j == 0)
        return 0;
    if (root < q) {
        off = dev_small_start(t, q, root, M, j, w0);
        return q;
    }
    uint32_t const U = root - q;
    uint32_t const g = U ? dev_gcd(U, q) : q;
    if (j % g)
        return 0;
    uint32_t const qq = q / g;
    uint32_t const t0 = U ? dev_invmod(U / g, qq) : 0;
    uint32_t const s0 = qq > 1 ? ((uint64_t)t0 * (j / g) + I / 2) % qq : 0;
    off = s0 < w0 ? (qq - (w0 - s0) % qq) % qq : s0 - w0;
    return qq;
}

/* ---------------- kernels ---------------- */

__device__ uint64_t dev_pack_entry(gpu_fb_entry const & e)
{
    return (uint64_t)e.q << BK_Q_SHIFT | (uint64_t)e.resieve << BK_RESIEVE_SHIFT |
           e.weight << BK_W_SHIFT;
}

/* With carry (GLAS_FILL_CARRY) the entries from n_band on get a
 * carry_state record and a nextc byte instead of walk_x and lattice. */
__global__ void transform_kernel(gpu_task t, gpu_fb_entry const * fb,
                                 uint32_t n, uint32_t * roots,
                                 uint64_t * walk_x, uint4 * lattice, int side,
                                 uint32_t n_band, carry_state * carry,
                                 uint8_t * nextc)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    gpu_fb_entry const e = fb[idx];
    uint32_t const root = (side == t.sq_side && e.q == t.q && e.r == t.rho)
                              ? SKIP_ROOT
                              : dev_transform_root(t, e.q, e.r);
    roots[idx] = root;
    uint32_t const I = 1u << t.logI;
    uint64_t x = WALK_SKIP;
    uint4 L = make_uint4(0, 0, 0, 0);
    if (root != SKIP_ROOT && root >= e.q) {
        x = WALK_PROJ;
    } else if (root != SKIP_ROOT) {
        dev_plattice const P(e.q, root, I);
        if (!(P.i1 >> t.logI)) {
            L = make_uint4(P.mi0, P.j0, P.i1, P.j1);
            x = I / 2;
        }
    }
    if (carry && idx >= n_band) {
        carry[idx - n_band] = {{L.x, L.y}, {L.z, L.w}, x};
        nextc[idx - n_band] = x == WALK_SKIP ? CARRY_DONE : 0;
        return;
    }
    walk_x[idx] = x;
    if (x != WALK_SKIP && x != WALK_PROJ)
        lattice[idx] = L;
}

/* Appends entry to the region's bucket: one global atomic for the slot
 * and one 8-byte store.  An entry past the capacity is dropped and sets
 * overflow; the host then sieves the special-q on the CPU. */
__device__ void dev_bucket_append(uint32_t region, uint64_t entry,
                                  uint32_t bucket_cap, uint64_t * buckets,
                                  uint32_t * bucket_cnt, uint32_t * overflow)
{
    uint32_t const slot = atomicAdd(&bucket_cnt[region * BK_CNT_STRIDE], 1);
    if (slot >= bucket_cap) {
        *overflow = 1;
        return;
    }
    buckets[dev_bucket_idx(region, slot, bucket_cap)] = entry;
}

/* GLAS_RHITS: a resieve hit goes to the hit list of its own region
 * instead of the flat list the out stream sorts for the filter, which
 * needs a survivor's hits grouped by position.  A hit does not outlive
 * its chunk, so each region keeps a bounded list; a list that fills
 * spills to the chunk's overflow tail, and a tail that fills sets the
 * chunk's overflow flag and the host resieves the special-q on the
 * CPU.  hit_cnt still counts every hit for the host's bound check, and
 * hit_cnt + 2 (the fourth counter word) counts the tail. */
constexpr uint32_t RHITS_CAP = 256;   /* hits kept in a region's list */
constexpr uint32_t RHITS_OV = 1 << 16; /* overflow hits a chunk keeps */
__device__ void dev_hit_emit(uint32_t x, uint64_t rec, uint32_t max_hits,
                             uint64_t * hits, uint32_t * hit_cnt,
                             uint64_t * hits_r, uint32_t * hit_rcnt,
                             uint64_t * hits_ov)
{
    uint32_t const slot = atomicAdd(hit_cnt, 1);
    if (slot >= max_hits)
        return;
    if (!hits_r) {
        hits[slot] = rec;
        return;
    }
    uint32_t const k = atomicAdd(hit_rcnt + (x >> SREG_LOG), 1);
    if (k < RHITS_CAP) {
        hits_r[(uint64_t)(x >> SREG_LOG) * RHITS_CAP + k] = rec;
        return;
    }
    uint32_t const o = atomicAdd(hit_cnt + 2, 1);
    if (o < RHITS_OV) {
        hits_ov[o] = rec;
        return;
    }
    hit_cnt[1] = 1;   /* the chunk's overflow flag */
}

/* Runs walk(emit) with the emit the probe bits in t.skip select: 8
 * walks without stores, 64 stores without the counter atomics; the
 * default appends the gated packed | offset to the region's bucket. */
template <typename W>
__device__ void dev_fill_emit(gpu_task const & t, W walk, uint64_t packed,
                              uint32_t const * gate,
                              uint32_t bucket_cap, uint64_t * buckets,
                              uint32_t * bucket_cnt, uint32_t * overflow)
{
    if (t.skip & 8) {
        /* walk without stores; measures the enumeration cost alone */
        uint32_t acc = 0;
        walk([&](uint32_t x) { acc ^= x; });
        if (acc == 0xdeadbeef)
            *overflow = 2;
        return;
    }
    if (t.skip & 64) {
        /* stores without the counter atomics; wrong results, timing only */
        walk([&](uint32_t x) {
            uint32_t const region = x >> SREG_LOG;
            uint32_t const q = (uint32_t)(packed >> BK_Q_SHIFT);
            buckets[dev_bucket_idx(region, q & (bucket_cap - 1), bucket_cap)] =
                packed | (x & (SREG - 1));
        });
        return;
    }
    walk([&](uint32_t x) {
        if (dev_gate_drops(gate, x))
            return;
        dev_bucket_append(x >> SREG_LOG, packed | (x & (SREG - 1)),
                          bucket_cap, buckets, bucket_cnt, overflow);
    });
}

/* One thread walks one large entry over the chunk, except the entries
 * [n_small, n_band): each of those gets nbands threads, and thread k
 * walks its own band of chunk rows.  A band thread starts from the
 * first strip point of its band instead of the walk cursor, so the
 * cursor stays at the transform kernel's value for those entries. */
__global__ void FILL_BOUNDS fill_kernel(gpu_task t, gpu_fb_entry const * fb,
                            uint32_t n_small, uint32_t n_band,
                            uint32_t nbands, uint32_t n,
                            uint32_t const * roots, uint64_t * walk_x,
                            uint4 const * lattice, uint32_t bucket_cap,
                            uint64_t * buckets, uint32_t * bucket_cnt,
                            uint32_t * overflow, uint32_t const * gate)
{
    uint32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const nbt = (n_band - n_small) * nbands;
    bool const banded = tid < nbt;
    uint32_t const nb = banded ? nbands : 1;
    uint32_t const idx = banded ? n_small + tid / nb : n_band + tid - nbt;
    if (idx >= n)
        return;
    uint64_t const x0 = walk_x[idx];
    if (x0 == WALK_SKIP)
        return;
    uint32_t const band = tid % nb;
    uint32_t const jb = t.j0 + band * t.rows / nb;
    uint32_t const je = t.j0 + (band + 1) * t.rows / nb;
    uint64_t const xend = (uint64_t)je << t.logI;
    if (x0 >= xend)
        return;
    auto walk = [&](auto emit) {
        if (x0 == WALK_PROJ)
            dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, jb, je, emit);
        else if (banded)
            dev_affine_walk(t, lattice[idx],
                            dev_band_start(t, fb[idx].q, roots[idx], jb, je),
                            xend, emit);
        else
            walk_x[idx] = dev_affine_walk(t, lattice[idx], x0, xend, emit);
    };
    dev_fill_emit(t, walk, dev_pack_entry(fb[idx]), gate, bucket_cap, buckets,
                  bucket_cnt, overflow);
}

/* With GATED (GLAS_FILL_TAIL0) the kernel walks the side-0 tail of one
 * chunk and stores only the hits under the gate. */
template <bool GATED>
__global__ void FILL_BOUNDS
fill_ahead_kernel(gpu_task t, gpu_fb_entry const * fb, uint32_t first,
                  uint32_t n, uint32_t const * roots, uint64_t * walk_x,
                  uint4 const * lattice, uint32_t regions, uint32_t cap,
                  uint64_t * buckets, uint32_t * cnt, uint32_t * overflow,
                  uint32_t const * gate)
{
    uint32_t const idx = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    uint64_t const x0 = walk_x[idx];
    uint32_t const je = t.j0 + t.rows;
    uint64_t const xend = (uint64_t)je << t.logI;
    if (x0 == WALK_SKIP || x0 >= xend)
        return;
    uint64_t const packed = dev_pack_entry(fb[idx]);
    auto emit = [&](uint32_t x) {
        if constexpr (GATED)
            if (dev_gate_drops(gate, x))
                return;
        uint32_t const region = x >> SREG_LOG;
        uint32_t const bank = region & ~(regions - 1);
        dev_bucket_append(region - bank, packed | (x & (SREG - 1)), cap,
            buckets + (uint64_t)bank * cap, cnt + bank * BK_CNT_STRIDE,
            overflow);
    };
    if (x0 == WALK_PROJ)
        dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, t.j0, je, emit);
    else
        walk_x[idx] = dev_affine_walk(t, lattice[idx], x0, xend, emit);
}

__device__ void dev_bucket_append_pair(uint32_t region, uint64_t a, uint64_t b,
                                       uint32_t cap, uint64_t * buckets,
                                       uint32_t * cnt, uint32_t * overflow)
{
    uint32_t const slot = atomicAdd(&cnt[region * BK_CNT_STRIDE], 2);
    if (slot >= cap || cap - slot < 2) {
        *overflow = 1;
        return;
    }
    if (slot & 1) {
        buckets[dev_bucket_idx(region, slot, cap)] = a;
        buckets[dev_bucket_idx(region, slot + 1, cap)] = b;
    } else {
        *(ulonglong2 *)(buckets + dev_bucket_idx(region, slot, cap)) =
            make_ulonglong2(a, b);
    }
}

template <int LOG_SLOTS>
__global__ void __launch_bounds__(FILL_THREADS, 6)
fill_local_pair_kernel(gpu_task t, gpu_fb_entry const * fb, uint32_t n_small,
                       uint32_t n_band, uint32_t nbands, uint32_t n,
                       uint32_t n_local,
                       uint32_t const * roots, uint64_t * walk_x,
                       uint4 const * lattice, uint32_t cap, uint64_t * buckets,
                       uint32_t * cnt, uint32_t * overflow,
                       uint32_t const * gate)
{
    constexpr uint32_t SLOTS = 1u << LOG_SLOTS;
    constexpr uint32_t OWNER_SHIFT = 31 - LOG_SLOTS;
    constexpr uint32_t POS_MASK = (1u << OWNER_SHIFT) - 1;
    constexpr uint32_t EMPTY = ~0u;
    static_assert(LOG_SLOTS >= 10 && FILL_THREADS == 256);
    __shared__ uint32_t pending[SLOTS];
    __shared__ uint64_t packed[FILL_THREADS];
    uint32_t const tid = blockIdx.x * FILL_THREADS + threadIdx.x;
    uint32_t const nbt = (n_band - n_small) * nbands;
    bool const banded = tid < nbt;
    uint32_t const nb = banded ? nbands : 1;
    uint32_t const idx = banded ? n_small + tid / nb : n_band + tid - nbt;
    bool const cache_block = blockIdx.x * FILL_THREADS <
        n_local - n_small + (n_band - n_small) * (nbands - 1);
    if (cache_block) {
        packed[threadIdx.x] = idx < n ? dev_pack_entry(fb[idx]) : 0;
        for (uint32_t k = threadIdx.x; k < SLOTS; k += FILL_THREADS)
            pending[k] = EMPTY;
        __syncthreads();
    }
    auto position = [&](uint32_t held, uint32_t k) {
        return (((held & POS_MASK) >> SREG_LOG) << (SREG_LOG + LOG_SLOTS)) |
               (k << SREG_LOG) | (held & (SREG - 1));
    };
    auto emit = [&](uint32_t x) {
        uint32_t const region = x >> SREG_LOG;
        uint32_t const k = region & (SLOTS - 1);
        uint32_t const token = (threadIdx.x << OWNER_SHIFT) |
            ((x >> (SREG_LOG + LOG_SLOTS)) << SREG_LOG) | (x & (SREG - 1));
        cuda::atomic_ref<uint32_t, cuda::thread_scope_block> cell(pending[k]);
        uint32_t held = cell.load(cuda::memory_order_relaxed);
        bool pair;
        for (;;) {
            pair = held != EMPTY &&
                   ((held ^ token) & (POS_MASK & ~(SREG - 1))) == 0;
            uint32_t const seen = atomicCAS(&pending[k], held, pair ? EMPTY : token);
            if (seen == held)
                break;
            held = seen;
        }
        if (held == EMPTY)
            return;
        uint32_t const old_x = position(held, k);
        uint64_t const old = packed[held >> OWNER_SHIFT] | (old_x & (SREG - 1));
        if (pair)
            dev_bucket_append_pair(region, old, packed[threadIdx.x] |
                (x & (SREG - 1)), cap, buckets, cnt, overflow);
        else
            dev_bucket_append(old_x >> SREG_LOG, old, cap, buckets, cnt, overflow);
    };
    auto walk = [&](auto store) {
        if (idx >= n)
            return;
        auto hit = [&](uint32_t x) {
            if (dev_gate_drops(gate, x))
                return;
            store(x);
        };
        uint64_t const x0 = walk_x[idx];
        uint32_t const band = tid % nb;
        uint32_t const jb = t.j0 + band * t.rows / nb;
        uint32_t const je = t.j0 + (band + 1) * t.rows / nb;
        uint64_t const xend = (uint64_t)je << t.logI;
        if (x0 != WALK_SKIP && x0 < xend) {
            if (x0 == WALK_PROJ)
                dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, jb, je, hit);
            else if (banded)
                dev_affine_walk(t, lattice[idx],
                    dev_band_start(t, fb[idx].q, roots[idx], jb, je), xend, hit);
            else
                walk_x[idx] = dev_affine_walk(t, lattice[idx], x0, xend, hit);
        }
    };
    if (idx < n_local) {
        walk(emit);
    } else {
        uint64_t const entry = idx < n ? dev_pack_entry(fb[idx]) : 0;
        walk([&](uint32_t x) {
            dev_bucket_append(x >> SREG_LOG, entry | (x & (SREG - 1)),
                              cap, buckets, cnt, overflow);
        });
    }
    if (!cache_block)
        return;
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < SLOTS; k += FILL_THREADS) {
        uint32_t const held = pending[k];
        if (held != EMPTY) {
            uint32_t const x = position(held, k);
            dev_bucket_append(x >> SREG_LOG, packed[held >> OWNER_SHIFT] |
                (x & (SREG - 1)), cap, buckets, cnt, overflow);
        }
    }
}

template <bool GROUP, uint32_t ROWS = 0, uint32_t C = 1, bool LEAN = false,
          uint32_t BLOCKS = 2>
__global__ void __launch_bounds__(FILL_THREADS, ROWS ? BLOCKS : 6)
fill_row_kernel(gpu_task t, gpu_fb_entry const * fb, uint32_t n_small,
                uint32_t n, uint32_t n_all, uint32_t bands, uint32_t const * roots,
                uint64_t * walk_x, uint4 const * lattice, uint32_t cap, uint64_t * buckets,
                uint32_t * cnt, uint32_t * overflow, uint32_t const * gate)
{
    uint32_t const tid = blockIdx.x * FILL_THREADS + threadIdx.x;
    constexpr uint32_t WIDTH = ROWS ? FILL_THREADS * C : 32;
    uint32_t const row_threads =
        (((n - n_small + WIDTH - 1) / WIDTH * WIDTH * bands + FILL_THREADS - 1) / FILL_THREADS) *
        FILL_THREADS;
    if (tid >= row_threads) {
        uint32_t const idx = n + tid - row_threads;
        if (idx >= n_all)
            return;
        uint64_t const x0 = walk_x[idx];
        uint32_t const je = t.j0 + t.rows;
        uint64_t const xend = (uint64_t)je << t.logI;
        if (x0 == WALK_SKIP || x0 >= xend)
            return;
        uint64_t const packed = dev_pack_entry(fb[idx]);
        auto hit = [&](uint32_t x) {
            if (dev_gate_drops(gate, x))
                return;
            dev_bucket_append(x >> SREG_LOG, packed | (x & (SREG - 1)),
                              cap, buckets, cnt, overflow);
        };
        if (x0 == WALK_PROJ)
            dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, t.j0, je, hit);
        else
            walk_x[idx] = dev_affine_walk(t, lattice[idx], x0, xend, hit);
        return;
    }
    uint32_t const lane = threadIdx.x & 31;
    uint32_t const band = (tid / WIDTH) % bands;
    uint32_t const idx = n_small + (tid / WIDTH / bands) * WIDTH + tid % WIDTH;
    uint32_t const jb = t.j0 + band * t.rows / bands;
    uint32_t const je = t.j0 + (band + 1) * t.rows / bands;
    uint64_t const x0 = idx < n ? walk_x[idx] : WALK_SKIP;
    bool const active = x0 != WALK_SKIP && x0 != WALK_PROJ;
    uint32_t const q = active ? fb[idx].q : 1;
    uint32_t const r = active ? roots[idx] : 0;
    uint64_t const packed = active ? dev_pack_entry(fb[idx]) : 0;
    uint32_t const I = 1u << t.logI;
    uint32_t off = ((uint64_t)r * jb + I / 2) % q;
    if constexpr (ROWS) {
        static_assert(FILL_THREADS * ROWS < (1u << (32 - 17)));
        auto const cluster = cooperative_groups::this_cluster();
        auto sync = [&] {
            if constexpr (C == 1) __syncthreads();
            else cluster.sync();
        };
        auto map = [&](uint32_t * p, uint32_t rank) {
            if constexpr (C == 1) return p;
            else return cluster.map_shared_rank(p, rank);
        };
        uint32_t const rank = blockIdx.x % C;
        uint32_t const regions = ROWS << max((int)t.logI - SREG_LOG, 0);
        extern __shared__ uint32_t row_window_storage[];
        uint32_t * used = row_window_storage, * base = used + regions;
        for (uint32_t row = jb; row < je; row += ROWS) {
            for (uint32_t k = threadIdx.x; k < regions; k += FILL_THREADS)
                used[k] = 0;
            __syncthreads();
            uint32_t const first = ((row - t.j0) << t.logI) >> SREG_LOG;
            uint32_t held[LEAN ? 1 : ROWS];
            uint32_t hits = 0, o = off;
            #pragma unroll
            for (uint32_t k = 0; k < ROWS; k++) {
                uint32_t const x = ((row + k - t.j0) << t.logI) + o;
                bool const hit = active && row + k < je && o < I &&
                                 dev_fill_keeps(t, row + k, o) &&
                                 !dev_gate_drops(gate, x);
                if constexpr (LEAN) {
                    if (hit) {
                        atomicAdd(used + ((x >> SREG_LOG) - first), 1);
                        hits |= 1u << k;
                    }
                } else {
                    held[k] = 0;
                    if (hit) {
                        uint32_t const ticket = atomicAdd(used + ((x >> SREG_LOG) - first), 1);
                        held[k] = ((ticket + 1) << t.logI) | o;
                    }
                }
                uint64_t const next = (uint64_t)o + r;
                o = next >= q ? next - q : next;
            }
            if constexpr (!LEAN)
                off = o;
            sync();
            for (uint32_t k = threadIdx.x * C + rank; k < regions; k += FILL_THREADS * C) {
                uint32_t sizes[C], total = 0;
                #pragma unroll
                for (uint32_t c = 0; c < C; c++) {
                    sizes[c] = map(used, c)[k];
                    total += sizes[c];
                }
                uint32_t offset = total ? atomicAdd(cnt + (first + k) * BK_CNT_STRIDE, total) : 0;
                #pragma unroll
                for (uint32_t c = 0; c < C; c++) {
                    map(base, c)[k] = offset;
                    offset += sizes[c];
                }
            }
            sync();
            if constexpr (LEAN) {
                #pragma unroll
                for (uint32_t k = 0; k < ROWS; k++) {
                    if (hits >> k & 1) {
                        uint32_t const x = ((row + k - t.j0) << t.logI) + off;
                        uint32_t const region = x >> SREG_LOG;
                        uint32_t const slot = atomicAdd(base + (region - first), 1);
                        if (slot < cap)
                            buckets[dev_bucket_idx(region, slot, cap)] = packed | (x & (SREG - 1));
                        else
                            *overflow = 1;
                    }
                    uint64_t const next = (uint64_t)off + r;
                    off = next >= q ? next - q : next;
                }
            } else {
                #pragma unroll
                for (uint32_t k = 0; k < ROWS; k++) {
                    uint32_t const ticket = held[k] >> t.logI;
                    if (ticket) {
                        uint32_t const x = ((row + k - t.j0) << t.logI) + (held[k] & (I - 1));
                        uint32_t const region = x >> SREG_LOG;
                        uint32_t const slot = base[region - first] + ticket - 1;
                        if (slot < cap)
                            buckets[dev_bucket_idx(region, slot, cap)] = packed | (x & (SREG - 1));
                        else
                            *overflow = 1;
                    }
                }
            }
        }
        sync();
    } else for (uint32_t j = jb; j < je; j++) {
        uint32_t const x = ((j - t.j0) << t.logI) + off;
        bool const hit = active && off < I && dev_fill_keeps(t, j, off) &&
                         !dev_gate_drops(gate, x);
        uint32_t const mask = __ballot_sync(~0u, hit);
        if (hit) {
            uint32_t const region = x >> SREG_LOG;
            uint64_t const entry = packed | (x & (SREG - 1));
            if (GROUP) {
                uint32_t const peers = __match_any_sync(mask, region);
                uint32_t const leader = __ffs(peers) - 1;
                uint32_t slot = 0;
                if (lane == leader)
                    slot = atomicAdd(&cnt[region * BK_CNT_STRIDE], __popc(peers));
                slot = __shfl_sync(peers, slot, leader) +
                       __popc(peers & ((1u << lane) - 1));
                if (slot < cap)
                    buckets[dev_bucket_idx(region, slot, cap)] = entry;
                else
                    *overflow = 1;
            } else {
                dev_bucket_append(region, entry, cap, buckets, cnt, overflow);
            }
        }
        uint64_t const next = (uint64_t)off + r;
        off = next >= q ? next - q : next;
    }
    if (idx < n && x0 == WALK_PROJ)
        dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, jb, je,
            [&](uint32_t x) {
                if (dev_gate_drops(gate, x))
                    return;
                dev_bucket_append(x >> SREG_LOG, dev_pack_entry(fb[idx]) |
                    (x & (SREG - 1)), cap, buckets, cnt, overflow);
            });
}

/* GLAS_FILL_STAGE: the staged fill.  One resident block per SM keeps
 * one 8-byte slot per region of the chunk in shared memory, and its
 * threads take the fill items of fill_kernel in a static stride.  A hit
 * puts its entry into the empty slot of its region, or takes the entry
 * the slot holds and appends both to the bucket with one reservation
 * and one 16-byte store, so the global operations per hit halve.  Put
 * and take are one compare-and-swap each; a swap that loses a race
 * sees the new slot value and tries again.  After the walk the block
 * flushes the slots that still hold an entry.  The bucket contents
 * equal the direct fill's in another order inside each bucket.  A
 * packed entry is never 0 (q is not 0), so 0 marks an empty slot.
 * The first n_staged regions have a slot; a hit in a later region
 * takes the direct path. */
__global__ void __launch_bounds__(FILL_STAGE_THREADS, 1)
fill_stage_kernel(gpu_task t, gpu_fb_entry const * fb, uint32_t n_small,
                  uint32_t n_band, uint32_t nbands, uint32_t n,
                  uint32_t const * roots, uint64_t * walk_x,
                  uint4 const * lattice, uint32_t n_staged,
                  uint32_t bucket_cap, uint64_t * buckets,
                  uint32_t * bucket_cnt, uint32_t * overflow,
                  uint32_t owners, uint32_t const * gate)
{
    bool const owned = owners > 1;
    uint32_t const unit = 1u << max((int)t.logI - SREG_LOG, 0);
    uint32_t const units = t.nregions / unit;
    uint32_t const rank = blockIdx.x % owners;
    uint32_t const first = units * rank / owners * unit;
    uint32_t const last = units * (rank + 1) / owners * unit;
    extern __shared__ unsigned long long slot[];   /* [n_staged] */
    for (uint32_t r = threadIdx.x; r < n_staged; r += FILL_STAGE_THREADS)
        slot[r] = 0;
    __syncthreads();
    auto hit = [&](uint32_t x, uint64_t packed) {
        if (dev_gate_drops(gate, x))
            return;
        uint32_t const region = x >> SREG_LOG;
        uint32_t const local = region - first;
        uint64_t const entry = packed | (x & (SREG - 1));
        if (local >= n_staged) {
            dev_bucket_append(region, entry, bucket_cap, buckets, bucket_cnt,
                              overflow);
            return;
        }
        unsigned long long held = slot[local];
        for (;;) {
            unsigned long long const seen =
                atomicCAS(&slot[local], held, held ? 0 : entry);
            if (seen == held)
                break;
            held = seen;
        }
        if (held == 0)
            return;
        uint32_t const b = atomicAdd(&bucket_cnt[region * BK_CNT_STRIDE], 2);
        if (b + 2 > bucket_cap) {
            *overflow = 1;
            return;
        }
        /* the count is odd after another block flushed a single entry;
         * slots b and b + 1 are then not adjacent in the slab layout */
        if (b & 1) {
            buckets[dev_bucket_idx(region, b, bucket_cap)] = held;
            buckets[dev_bucket_idx(region, b + 1, bucket_cap)] = entry;
            return;
        }
        *(ulonglong2 *)(buckets + dev_bucket_idx(region, b, bucket_cap)) =
            make_ulonglong2(held, entry);
    };
    uint32_t const nbt = (n_band - n_small) * nbands;
    uint32_t const n_item = nbt + n - n_band;
    /* the loop is warp-uniform and the warp converges before each item,
     * so the item loads coalesce and the walks of one item run in one
     * warp pass as in fill_kernel */
    for (uint32_t base = blockIdx.x / owners * FILL_STAGE_THREADS +
                             (threadIdx.x & ~31u);
         base < n_item; base += gridDim.x / owners * FILL_STAGE_THREADS) {
        __syncwarp();
        uint32_t const tid = base + (threadIdx.x & 31);
        if (tid >= n_item)
            continue;
        /* the item decode of fill_kernel */
        bool const banded = tid < nbt;
        uint32_t const nb = banded ? nbands : 1;
        uint32_t const idx = banded ? n_small + tid / nb : n_band + tid - nbt;
        uint64_t const x0 = walk_x[idx];
        if (x0 == WALK_SKIP)
            continue;
        uint32_t const band = tid % nb;
        uint32_t const jb = t.j0 + (owned ? (first << SREG_LOG) >> t.logI
                                        : band * t.rows / nb);
        uint32_t const je = t.j0 + (owned ? (last << SREG_LOG) >> t.logI
                                        : (band + 1) * t.rows / nb);
        uint64_t const xend = (uint64_t)je << t.logI;
        if (x0 >= xend)
            continue;
        uint64_t const packed = dev_pack_entry(fb[idx]);
        auto emit = [&](uint32_t x) { hit(x, packed); };
        if (x0 == WALK_PROJ)
            dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, jb, je, emit);
        else if (banded || owned)
            dev_affine_walk(t, lattice[idx],
                            dev_band_start(t, fb[idx].q, roots[idx], jb, je),
                            xend, emit);
        else
            walk_x[idx] = dev_affine_walk(t, lattice[idx], x0, xend, emit);
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < n_staged; r += FILL_STAGE_THREADS)
        if (slot[r])
            dev_bucket_append(first + r, slot[r], bucket_cap, buckets, bucket_cnt,
                              overflow);
}

/* GLAS_FILL_CARRY: cub::DeviceSelect::If predicate; selects the large
 * entries whose next hit is in chunk cur. */
struct carry_due {
    uint8_t const * nextc;
    uint32_t cur;
    __device__ bool operator()(uint32_t li) const { return nextc[li] == cur; }
};

/* GLAS_FILL_CARRY: walks the listed large entries over the whole
 * chunk, one thread per entry, and files each under the chunk of its
 * next hit.  A projective entry is due in every chunk. */
__global__ void fill_carry_kernel(gpu_task t, uint32_t chunk_log,
                                  gpu_fb_entry const * fb,
                                  uint32_t const * roots, uint32_t n_band,
                                  uint32_t const * active,
                                  uint32_t const * n_active,
                                  carry_state * carry, uint8_t * nextc,
                                  uint32_t bucket_cap, uint64_t * buckets,
                                  uint32_t * bucket_cnt, uint32_t * overflow)
{
    uint32_t const je = t.j0 + t.rows;
    uint64_t const xend = (uint64_t)je << t.logI;
    uint32_t const n = *n_active;
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += gridDim.x * blockDim.x) {
        uint32_t const li = active[k];
        uint32_t const idx = n_band + li;
        carry_state const s = carry[li];
        auto walk = [&](auto emit) {
            uint64_t x = xend;
            if (s.x == WALK_PROJ) {
                dev_proj_walk(t, fb[idx].q, roots[idx] - fb[idx].q, t.j0, je,
                              emit);
            } else {
                uint4 const L = make_uint4(s.L0.x, s.L0.y, s.L1.x, s.L1.y);
                x = dev_affine_walk(t, L, s.x, xend, emit);
                carry[li].x = x;
            }
            nextc[li] = (uint8_t)min(x >> chunk_log, (uint64_t)CARRY_DONE);
        };
        dev_fill_emit(t, walk, dev_pack_entry(fb[idx]), nullptr, bucket_cap,
                      buckets, bucket_cnt, overflow);
    }
}

/* log2 units of a norm scaled by 2^-e.  Values at or below the scale
 * floor clamp to zero, like the +1.0 in the CPU core's log2_units. */
__device__ float dev_norm_logf(float v, float e)
{
    return fmaxf((e + __log2f(fabsf(v))) * (float)LOG_SCALE, 0.0f);
}

/* One 16-bit sieve cell inside the shared uint32 array. */
__device__ void dev_cell_sub(uint32_t * S, uint32_t p, uint32_t w)
{
    atomicSub(&S[p >> 1], w << ((p & 1) << 4));
}

/* Subtracts w from the cells p, p + stride, ... below end.  end and
 * stride are even, so the cell parity, the shifted weight and the word
 * stride are loop constants. */
__device__ void dev_cell_sub_run(uint32_t * S, uint32_t p, uint32_t end,
                                 uint32_t stride, uint32_t w)
{
    uint32_t const ws = w << ((p & 1) << 4);
    for (uint32_t wi = p >> 1; wi < end >> 1; wi += stride >> 1)
        atomicSub(&S[wi], ws);
}

/* Pattern tier: the dense entries whose q divides PAT_LEN accumulate
 * into one shared pattern of PAT_LEN 16-bit cells, and the norm store
 * subtracts the pattern word.  Subtraction on the packed word is
 * associative modulo 2^32, so the cells match the per-hit atomic
 * path exactly. */
constexpr uint32_t PAT_LEN = 5040;   /* 2^4 * 3^2 * 5 * 7 */

__device__ bool dev_pattern_entry(uint32_t q, uint32_t root)
{
    return q <= 16 && PAT_LEN % q == 0 && root < q;
}

/* With CAND (GLAS_REGION_CAND, side 0 only) the kernel computes the norm
 * of a cell only at the positions the side-1 mask keeps.  The other cells
 * hold undefined values, and no step reads them. */
template <bool CAND>
__global__ void __launch_bounds__(1024, 2)
region_kernel(gpu_task t, gpu_small_entry const * small,
              uint32_t n_small, uint32_t n_dense, uint32_t n_skip,
              uint32_t n_pat, uint32_t const * roots,
              uint32_t bucket_cap, uint64_t const * buckets,
              uint32_t const * bucket_cnt, int side, int first,
              uint32_t * bitmask,
              uint32_t max_survivors, uint32_t * survivors,
              uint32_t * survivor_cnt,
              uint32_t obucket_cap, uint64_t const * obuckets,
              uint32_t const * obucket_cnt, uint32_t max_hits,
              uint64_t * hits, uint32_t * hit_cnt, uint64_t * hits_r,
              uint32_t * hit_rcnt, uint64_t * hits_ov)
{
    extern __shared__ __align__(8) uint32_t S[];   /* SREG 16-bit cells */
    uint32_t * const P = S + SREG / 2;   /* PAT_LEN 16-bit pattern cells */
    __shared__ uint32_t block_surv;
    __shared__ double cj[SREG >> 12][GLAS_MAX_DEGREE + 1];
    __shared__ float cf[SREG >> 12][GLAS_MAX_DEGREE + 1];
    __shared__ float ce[SREG >> 12];
    if (threadIdx.x == 0)
        block_surv = 0;
    uint32_t const region = blockIdx.x;
    uint32_t const base = region << SREG_LOG;
    uint32_t const I = 1u << t.logI;
    int const d = t.degree[side];

    /* rows in this region (one window of a row when I > SREG) */
    uint32_t const segs_log = t.logI < SREG_LOG ? SREG_LOG - t.logI : 0;
    uint32_t const segs = 1u << segs_log;
    uint32_t const j0 = t.j0 + (base >> t.logI);
    uint32_t const w0 = base & (I - 1);

    /* the pattern is row-specific, so single-row regions only */
    bool const use_pat = segs == 1 && j0 != 0 &&
                         !(t.skip & 2) && !(t.skip & 16);
    if (use_pat)
        for (uint32_t k = threadIdx.x; k < PAT_LEN / 2; k += blockDim.x)
            P[k] = 0;

    /* row-constant coefficients: cj[s][k] = fijd[k] * j^(d-k) */
    if (threadIdx.x < segs * (uint32_t)(d + 1)) {
        uint32_t const s = threadIdx.x / (d + 1);
        uint32_t const k = threadIdx.x % (d + 1);
        double const dj = (double)(j0 + s);
        double jp = 1.0;
        for (int m = 0; m < d - (int)k; m++)
            jp *= dj;
        cj[s][k] = t.fijd[side][k] * jp;
    }
    __syncthreads();

    /* float row coefficients, scaled down by the largest term exponent
     * so the float Horner sum stays in range */
    if (threadIdx.x < segs) {
        double const * const c = cj[threadIdx.x];
        int emax = INT_MIN;
        for (int k = 0; k <= d; k++)
            if (c[k] != 0.0)
                emax = max(emax, ilogb(c[k]) + k * (t.logI - 1));
        ce[threadIdx.x] = (float)emax;
        for (int k = 0; k <= d; k++)
            cf[threadIdx.x][k] = (float)scalbn(c[k], -emax);
    }

    if (use_pat)
        /* skipped entries stay out: the bound compensates them */
        for (uint32_t k = n_skip * 32 + threadIdx.x; k < n_pat * 32;
             k += blockDim.x) {
            uint32_t const e = k >> 5;
            gpu_small_entry const s = small[e];
            uint32_t const root = roots[e];
            if (!dev_pattern_entry(s.q, root))
                continue;
            uint32_t const start =
                dev_small_start(t, s.q, root, s.magic, j0, w0);
            for (uint32_t c = start + (k & 31) * s.q; c < PAT_LEN;
                 c += 32 * s.q)
                atomicAdd(&P[c >> 1], s.weight << ((c & 1) << 4));
        }
    __syncthreads();

    /* norms: each warp owns one run of WP positions inside one row.  The
     * warp evaluates the exact norm at every 4th position, lane l at
     * position 4 * (32 * s + l) of the run, and interpolates the log
     * units in between; the right sample of a span comes from the next
     * lane.  A sign change between two samples puts a root of fij inside
     * the span, so that span is evaluated exactly.  Lane l stores its 4
     * positions as one 8-byte word, so a warp store covers 32 consecutive
     * words: no bank conflicts. */
    constexpr uint32_t WP = SREG / 32;     /* positions per warp */
    constexpr uint32_t NS = WP / 128;      /* spans per lane */
    static_assert(WP <= 1u << 12, "a warp run stays inside one row: logI >= 12");
    uint32_t const lane = threadIdx.x & 31;
    uint32_t const wp0 = (threadIdx.x >> 5) * WP;
    uint32_t const j = t.j0 + ((base + wp0) >> t.logI);   /* the run's row */
    auto store4 = [&](uint32_t p, uint32_t lo, uint32_t hi) {
        uint32_t const cw = p >> 1;
        uint2 pat = make_uint2(0, 0);
        if (use_pat)
            pat = *(uint2 const *)&P[cw % (PAT_LEN / 2)];
        *(uint2 *)&S[cw] = make_uint2(lo - pat.x, hi - pat.y);
    };
    if (j == 0 || (t.skip & 1)) {
        for (uint32_t s = 0; s < NS; s++)
            store4(wp0 + 4 * (32 * s + lane), (BIAS + NORM_CAP) * 0x10001u,
                   (BIAS + NORM_CAP) * 0x10001u);
    } else if (c_norm_reg == 0) {
        float const * const c = cf[wp0 >> t.logI];
        float const e = ce[wp0 >> t.logI];
        float const di0 =
            (float)((int32_t)((base + wp0) & (I - 1)) - (int32_t)(I / 2));
        auto eval = [&](uint32_t off) {
            float const di = di0 + (float)off;
            float v = c[d];
            for (int k = d - 1; k >= 0; k--)
                v = v * di + c[k];
            return v;
        };
        uint32_t off = 4 * lane;
        float v0 = eval(off);
        float u0 = dev_norm_logf(v0, e);
        for (uint32_t s = 0; s < NS; s++, off += 128) {
            float v1 = __shfl_down_sync(0xffffffffu, v0, 1);
            float u1 = __shfl_down_sync(0xffffffffu, u0, 1);
            if (lane == 31) {
                v1 = eval(off + 4);
                u1 = dev_norm_logf(v1, e);
            }
            uint32_t cell[4];
            for (uint32_t r = 0; r < 4; r++) {
                float uu = u0;
                if (r && (v0 < 0) != (v1 < 0))
                    uu = dev_norm_logf(eval(off + r), e);
                else if (r)
                    uu = u0 + (u1 - u0) * (float)r * 0.25f;
                cell[r] = BIAS + min((uint32_t)lrintf(uu), NORM_CAP);
            }
            store4(wp0 + off, cell[0] | cell[1] << 16, cell[2] | cell[3] << 16);
            if (s + 1 < NS) {
                v0 = eval(off + 128);
                u0 = dev_norm_logf(v0, e);
            }
        }
    } else {
        /* the coefficients stay in registers: the Horner sum runs over
         * all GLAS_MAX_DEGREE + 1 slots, the slots above d hold zero and
         * leave the sum unchanged (0 * di + 0 = 0, then 0 * di + c[d] =
         * c[d] exactly) */
        float const * const cs = cf[wp0 >> t.logI];
        float c[GLAS_MAX_DEGREE + 1];
#pragma unroll
        for (int k = 0; k <= GLAS_MAX_DEGREE; k++)
            c[k] = k <= d ? cs[k] : 0.0f;
        float const e = ce[wp0 >> t.logI];
        float const di0 =
            (float)((int32_t)((base + wp0) & (I - 1)) - (int32_t)(I / 2));
        auto eval = [&](uint32_t off) {
            float const di = di0 + (float)off;
            float v = c[GLAS_MAX_DEGREE];
#pragma unroll
            for (int k = GLAS_MAX_DEGREE - 1; k >= 0; k--)
                v = v * di + c[k];
            return v;
        };
        /* the right sample of lane 31 is the left sample of lane 0 in the
         * next span (the position after the run at the last span), so the
         * warp evaluates the next span's samples before the current one
         * and lane 0 passes its next sample to lane 31 */
        uint32_t const G = c_norm_stride;   /* cells per lane run */
        float const invG = 1.0f / (float)G;
        if (CAND) {
            /* thread w owns mask word w.  Cell b of the word gets the
             * value of the lane runs below; a cell outside the mask gets
             * the cap.  The thread stores the shared words that hold a
             * masked cell. */
            uint32_t const mw = bitmask[(base >> 5) + threadIdx.x];
            auto cell = [&](uint32_t b) {
                if (!(mw >> b & 1))
                    return BIAS + NORM_CAP;
                uint32_t const x = 32 * lane + b;   /* offset in the run */
                uint32_t const rr = x % G;
                float const v0 = eval(x - rr);
                float const v1 = eval(x - rr + G);
                float const u0 = dev_norm_logf(v0, e);
                float uu = u0;
                if (rr && (v0 < 0) != (v1 < 0))
                    uu = dev_norm_logf(eval(x), e);
                else if (rr)
                    uu = u0 + (dev_norm_logf(v1, e) - u0) * (float)rr * invG;
                return BIAS + min((uint32_t)lrintf(uu), NORM_CAP);
            };
            uint32_t pairs = (mw | mw >> 1) & 0x55555555u;
            while (pairs) {
                uint32_t const b = __ffs(pairs) - 1;
                pairs &= pairs - 1;
                uint32_t const cw = (threadIdx.x * 32 + b) >> 1;
                S[cw] = (cell(b) | cell(b + 1) << 16) -
                        (use_pat ? P[cw % (PAT_LEN / 2)] : 0);
            }
        } else {
            uint32_t off = G * lane;
            float v0 = eval(off);
            float u0 = dev_norm_logf(v0, e);
            for (; off < WP; off += 32 * G) {
                float const vn = eval(off + 32 * G);
                float const un = dev_norm_logf(vn, e);
                uint32_t const src = (lane + 1) & 31;
                float const v1 = __shfl_sync(0xffffffffu, lane ? v0 : vn, src);
                float const u1 = __shfl_sync(0xffffffffu, lane ? u0 : un, src);
                for (uint32_t w = 0; w < G; w += 4) {
                    uint32_t cell[4];
                    for (uint32_t r = 0; r < 4; r++) {
                        uint32_t const rr = w + r;
                        float uu = u0;
                        if (rr && (v0 < 0) != (v1 < 0))
                            uu = dev_norm_logf(eval(off + rr), e);
                        else if (rr)
                            /* at G = 4 this is the original r * 0.25f,
                             * since 1.0f / G is exact for every stride */
                            uu = u0 + (u1 - u0) * (float)rr * invG;
                        cell[r] = BIAS + min((uint32_t)lrintf(uu), NORM_CAP);
                    }
                    store4(wp0 + off + w,
                           cell[0] | cell[1] << 16, cell[2] | cell[3] << 16);
                }
                v0 = vn;
                u0 = un;
            }
        }
    }
    __syncthreads();

    /* small entries: the warp group g = warp >> segs_log walks the dense
     * entries g, g + 32 / segs, ..., each warp of the group one row
     * segment; one thread per sparse walk */
    if (t.skip & 2)
        n_small = n_dense = n_pat = n_skip = 0;
    uint32_t const dense_end = (t.skip & 16) ? 0 : n_dense;
    uint32_t const sparse_n = (t.skip & 32) ? n_dense : n_small;
    uint32_t const wend = min(I, (uint32_t)SREG);
    uint32_t const grp = (threadIdx.x >> 5) >> segs_log;
    uint32_t const jr = (threadIdx.x >> 5) & (segs - 1);
    uint32_t const row = jr << t.logI;
    /* the warp takes its dense entries 32 at a time: lane l sets up the
     * walk of entry l of the batch (stride 0: no walk), then the warp
     * walks the batch entry by entry from the shuffled setups */
    uint32_t const estep_log = 5 - segs_log;
    for (uint32_t eb = n_skip + grp; eb < dense_end; eb += 32 << estep_log) {
        uint32_t const e = eb + (lane << estep_log);
        uint32_t off = 0, qq = 0, w = 0;
        if (e < dense_end) {
            uint32_t const root = roots[e];
            gpu_small_entry const s = small[e];
            w = s.weight;
            if (root != SKIP_ROOT && !(use_pat && dev_pattern_entry(s.q, root)))
                qq = dev_small_walk_start(t, s.q, root, s.magic, j0 + jr, w0, off);
        }
        uint32_t const n =
            min(32u, (dense_end - eb + (1u << estep_log) - 1) >> estep_log);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t const o = __shfl_sync(0xffffffffu, off, i);
            uint32_t const q = __shfl_sync(0xffffffffu, qq, i);
            uint32_t const ws = __shfl_sync(0xffffffffu, w, i);
            if (q)
                dev_cell_sub_run(S, row + o + lane * q, row + wend, 32 * q, ws);
        }
    }
    /* the thread loads its next entry before it walks the current one */
    uint32_t const sparse_items = (sparse_n - n_dense) * segs;
    uint32_t next_root = 0;
    gpu_small_entry next_s = {};
    auto load = [&](uint32_t k) {
        next_root = roots[n_dense + (k >> segs_log)];
        next_s = small[n_dense + (k >> segs_log)];
    };
    if (threadIdx.x < sparse_items)
        load(threadIdx.x);
    for (uint32_t k = threadIdx.x; k < sparse_items; k += blockDim.x) {
        uint32_t const root = next_root;
        gpu_small_entry const s = next_s;
        if (k + blockDim.x < sparse_items)
            load(k + blockDim.x);
        if (root == SKIP_ROOT)
            continue;
        uint32_t const js = k & (segs - 1);
        uint32_t const rs = js << t.logI;
        uint32_t off;
        uint32_t const qq =
            dev_small_walk_start(t, s.q, root, s.magic, j0 + js, w0, off);
        if (!qq)
            continue;
        for (uint32_t p = rs + off; p < rs + wend; p += qq)
            dev_cell_sub(S, p, s.weight);
    }

    /* bucket hits, four loads in flight per thread */
    uint32_t const cnt =
        (t.skip & 4) ? 0 : min(bucket_cnt[region * BK_CNT_STRIDE], bucket_cap);
    auto rb = [&](uint32_t k) {
        return buckets[dev_bucket_idx(region, k, bucket_cap)];
    };
    auto apply = [&](uint64_t h) {
        dev_cell_sub(S, h & (SREG - 1), (uint32_t)(h >> BK_W_SHIFT) & BK_W_MASK);
    };
    uint32_t kb = threadIdx.x;
    for (; kb + 3 * blockDim.x < cnt; kb += 4 * blockDim.x) {
        uint64_t const h0 = rb(kb);
        uint64_t const h1 = rb(kb + blockDim.x);
        uint64_t const h2 = rb(kb + 2 * blockDim.x);
        uint64_t const h3 = rb(kb + 3 * blockDim.x);
        apply(h0);
        apply(h1);
        apply(h2);
        apply(h3);
    }
    for (; kb < cnt; kb += blockDim.x)
        apply(rb(kb));

    __syncthreads();


    /* survivor bound test; one thread owns one 32-bit mask word.  The
     * warp builds its 32 words one at a time: the lanes test the 32
     * consecutive cells of a word (16 consecutive shared words, no bank
     * conflicts) and a ballot forms the word.  One extra slack unit covers
     * the fast-log rounding differences against the CPU core.  Row 0 has
     * no survivors, and an even row none at even i, where gcd(i, j) > 1. */
    static_assert(SREG / 32 == 1024, "one mask word per thread");
    uint32_t const limit =
        BIAS + (uint32_t)(t.bound[side] + SURVIVOR_SLACK + 1);
    uint32_t const row_mask = j == 0 ? 0 : j & 1 ? ~0u : 0xaaaaaaaau;
    uint16_t const * const cells = (uint16_t const *)S + wp0 + lane;
    uint32_t const wd = threadIdx.x;
    uint32_t word = CAND ? bitmask[(base >> 5) + wd] : 0;
    if (!CAND) {
        for (uint32_t b = 0; b < 32; b++) {
            uint32_t const bal =
                __ballot_sync(0xffffffffu, cells[b * 32] <= limit);
            if (lane == b)
                word = bal;
        }
        word &= row_mask;
        uint32_t const widx = (base >> 5) + wd;
        if (first) {
            bitmask[widx] = word;
            return;
        }
        word &= bitmask[widx];
    }
    uint32_t keep = word;
    while (word) {
        int const b = __ffs(word) - 1;
        word &= word - 1;
        uint32_t const p = wd * 32 + b;
        int32_t const i = (int32_t)((base + p) & (I - 1)) - (int32_t)(I / 2);
        if ((CAND && ((uint16_t const *)S)[p] > limit) ||
            dev_gcd(i < 0 ? (uint32_t)-i : (uint32_t)i, j) != 1) {
            keep &= ~(1u << b);
            continue;
        }
        uint32_t const slot = atomicAdd(survivor_cnt, 1);
        if (slot < max_survivors)
            survivors[slot] = base + p;
    }
    P[wd] = keep;
    if (keep)
        atomicAdd(&block_surv, __popc(keep));
    __syncthreads();
    if (!block_surv)
        return;

    /* resieve the buckets of both sides against the final mask, which
     * the survivor loop left in P: report the resieve entries that hit
     * a survivor position */
    for (int hs = 0; hs < 2; hs++) {
        uint64_t const * const bk = hs ? obuckets : buckets;
        uint32_t const cap = hs ? obucket_cap : bucket_cap;
        uint32_t const bn =
            min((hs ? obucket_cnt : bucket_cnt)[region * BK_CNT_STRIDE], cap);
        for (uint32_t k = threadIdx.x; k < bn; k += blockDim.x) {
            uint64_t const h = bk[dev_bucket_idx(region, k, cap)];
            uint32_t const p = (uint32_t)h & (SREG - 1);
            if (!(P[p >> 5] >> (p & 31) & 1))
                continue;
            if (!(h >> BK_RESIEVE_SHIFT & 1))
                continue;
            dev_hit_emit(base + p,
                         (uint64_t)(base + p) << 32 |
                             (h >> BK_Q_SHIFT) << 1 |
                             (uint32_t)(hs ? side ^ 1 : side),
                         max_hits, hits, hit_cnt, hits_r, hit_rcnt,
                         hits_ov);
        }
    }
}

constexpr uint32_t REGION_SMEM = SREG * 2 + PAT_LEN * 2;

/* Small resieve entries: trial-divide each (survivor, entry) pair
 * instead of re-walking every region.  A block loads a tile of
 * survivors into shared memory; each thread checks its entries
 * against the tile.  The entry list is compacted once per task so
 * the pair loop touches resieve entries only. */
constexpr uint32_t RS_TILE = 1024;
constexpr uint32_t RS_EGRP = 512;   /* entries per block group */

__global__ void resieve_compact_kernel(gpu_fb_entry const * fb,
                                       uint32_t n_small,
                                       uint32_t const * roots,
                                       uint32_t * rs_idx, uint32_t * rs_cnt)
{
    uint32_t const e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_small || !fb[e].resieve || roots[e] == SKIP_ROOT)
        return;
    rs_idx[atomicAdd(rs_cnt, 1)] = e;
}

__global__ void resieve_small_kernel(gpu_task t,
                                     gpu_small_entry const * small,
                                     uint32_t const * rs_idx,
                                     uint32_t const * rs_cnt,
                                     uint32_t const * roots, int side,
                                     uint32_t const * survivors,
                                     uint32_t const * survivor_cnt,
                                     uint32_t max_survivors,
                                     uint32_t max_hits, uint64_t * hits,
                                     uint32_t * hit_cnt, uint64_t * hits_r,
                                     uint32_t * hit_rcnt,
                                     uint64_t * hits_ov)
{
    __shared__ uint32_t sx[RS_TILE];
    __shared__ uint32_t sj[RS_TILE];
    __shared__ uint32_t si[RS_TILE];
    uint32_t const I = 1u << t.logI;
    uint32_t const n_surv = min(*survivor_cnt, max_survivors);
    uint32_t const n_rs = *rs_cnt;
    /* 2D work split: blocks cover (survivor tile, entry group) pairs so
     * the grid stays full when few tiles exist */
    uint32_t const n_tiles = (n_surv + RS_TILE - 1) / RS_TILE;
    uint32_t const n_eg = (n_rs + RS_EGRP - 1) / RS_EGRP;
    for (uint32_t blk = blockIdx.x; blk < n_tiles * n_eg;
         blk += gridDim.x) {
        uint32_t const tile = (blk / n_eg) * RS_TILE;
        uint32_t const e0 = (blk % n_eg) * RS_EGRP;
        uint32_t const e1 = min(e0 + RS_EGRP, n_rs);
        uint32_t const n_tile = min(RS_TILE, n_surv - tile);
        __syncthreads();
        for (uint32_t k = threadIdx.x; k < n_tile; k += blockDim.x) {
            uint32_t const x = survivors[tile + k];
            sx[k] = x;
            sj[k] = t.j0 + (x >> t.logI);
            si[k] = x & (I - 1);
        }
        __syncthreads();
        for (uint32_t w = e0 + threadIdx.x; w < e1; w += blockDim.x) {
            uint32_t const e = rs_idx[w];
            uint32_t const root = roots[e];
            uint32_t const q = small[e].q;
            uint64_t const M = small[e].magic;
            auto emit = [&](uint32_t x) {
                dev_hit_emit(x, (uint64_t)x << 32 | (uint64_t)q << 1 | side,
                             max_hits, hits, hit_cnt, hits_r, hit_rcnt,
                             hits_ov);
            };
            if (root < q) {
                /* hit: xi = root * j + I/2 (mod q), tested as one
                 * reduction of (q - root) * j + xi + pad with
                 * pad = -I/2 (mod q) */
                uint32_t const nr = q - root;
                uint32_t const pad = q - dev_mod_small(I / 2, q, M);
#pragma unroll 4
                for (uint32_t k = 0; k < n_tile; k++)
                    if (dev_mod_small((uint64_t)nr * sj[k] + si[k] + pad,
                                      q, M) == 0)
                        emit(sx[k]);
                continue;
            }
            /* projective root U = root - q: row j hits when g | j and
             * (U/g) * (xi - I/2) = j/g (mod q/g) */
            uint32_t const U = root - q;
            uint32_t const g = U ? dev_gcd(U, q) : q;
            uint32_t const qq = q / g;
            uint32_t const Ug = U / g % max(qq, 1u);
            uint64_t const ppad =
                ((uint64_t)I / 2 + t.J + qq) / max(qq, 1u) * qq;
            for (uint32_t k = 0; k < n_tile; k++) {
                uint32_t const j = sj[k];
                if (j % g)
                    continue;
                uint64_t const lhs =
                    (uint64_t)Ug * (si[k] + ppad - I / 2) + ppad - j / g;
                if (qq < 2 || lhs % qq == 0)
                    emit(sx[k]);
            }
        }
    }
}

/* Tiny-prime masks: evaluate both side norms F(a, b) mod each tiny
 * prime and flag the divisors.  |a|, |b| < 2^48, so the magic
 * reduction stays exact. */
__device__ uint32_t dev_mod_signed(int64_t v, uint32_t p, uint64_t M)
{
    uint32_t const r = dev_mod_small((uint64_t)(v < 0 ? -v : v), p, M);
    return v < 0 && r ? p - r : r;
}

__global__ void tiny_mask_kernel(gpu_task t, uint32_t const * survivors,
                                 uint32_t const * survivor_cnt,
                                 uint32_t max_survivors, uint64_t * tiny)
{
    uint32_t const I = 1u << t.logI;
    uint32_t const n = min(*survivor_cnt, max_survivors);
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x) {
        uint32_t const x = survivors[idx];
        int64_t const i = (int64_t)(x & (I - 1)) - I / 2;
        int64_t const j = (int64_t)t.j0 + (x >> t.logI);
        int64_t const a = i * t.a0 + j * t.a1;
        int64_t const b = i * t.b0 + j * t.b1;
        for (int side = 0; side < 2; side++) {
            uint64_t m = 0;
            for (uint32_t k = 0; k < c_tiny_np; k++) {
                uint32_t const p = c_tiny_p[k];
                uint64_t const M = c_tiny_M[k];
                uint32_t const am = dev_mod_signed(a, p, M);
                uint32_t const bm = dev_mod_signed(b, p, M);
                uint32_t const * c =
                    c_tiny_c[side] + k * (GLAS_MAX_DEGREE + 1);
                uint32_t r = c[t.degree[side]];
                uint32_t bp = 1;
                for (int d = t.degree[side] - 1; d >= 0; d--) {
                    bp = dev_mod_small((uint64_t)bp * bm, p, M);
                    r = dev_mod_small((uint64_t)r * am + (uint64_t)c[d] * bp,
                                      p, M);
                }
                m |= (uint64_t)(r == 0) << k;
            }
            tiny[2 * (uint64_t)idx + side] = m;
        }
    }
}

/* ---------------- cofactorization ---------------- */

/* Montgomery arithmetic + Miller-Rabin + batched ECM (the shape of
 * eprint 2014/397).  A leftover norm is odd, fits in 128 bits, and its
 * prime factors all exceed the factor-base bound.  Values below 2^63
 * use the one-limb path, values below 2^96 use the three-limb path,
 * and larger values use the two-limb path. */

using u128 = unsigned __int128;

__device__ uint64_t dev_ninv64(uint64_t n)   /* -n^-1 mod 2^64, n odd */
{
    uint64_t x = n;
    for (int i = 0; i < 5; i++)
        x *= 2 - n * x;
    return -x;
}

__device__ uint64_t dev_mm64(uint64_t a, uint64_t b, uint64_t n, uint64_t ni)
{
    u128 const t = (u128)a * b;
    uint64_t const m = (uint64_t)t * ni;
    uint64_t r = (uint64_t)(t >> 64) + __umul64hi(m, n) + ((uint64_t)t != 0);
    return r >= n ? r - n : r;
}

__device__ uint64_t dev_gcd64(uint64_t a, uint64_t b)
{
    while (a) {
        b %= a;
        uint64_t const t = a;
        a = b;
        b = t;
    }
    return b;
}

struct mont64 {
    using t = uint64_t;
    uint64_t n, ni, one, r2;   /* one = 2^64 mod n, r2 = 2^128 mod n */

    __device__ explicit mont64(uint64_t n_) : n(n_), ni(dev_ninv64(n_))
    {
        one = (-n) % n;
        r2 = one;
        for (int i = 0; i < 64; i++) {
            r2 += r2;
            if (r2 >= n)
                r2 -= n;
        }
    }
    __device__ uint64_t mul(uint64_t a, uint64_t b) const
    {
        return dev_mm64(a, b, n, ni);
    }
    __device__ uint64_t in(uint64_t a) const { return mul(a, r2); }
    __device__ uint64_t add(uint64_t a, uint64_t b) const
    {
        uint64_t const r = a + b;
        return r >= n ? r - n : r;
    }
    __device__ uint64_t sub(uint64_t a, uint64_t b) const
    {
        return a >= b ? a - b : a + n - b;
    }
    __device__ uint64_t pow(uint64_t a, uint64_t e) const
    {
        uint64_t r = one;
        for (; e; e >>= 1, a = mul(a, a))
            if (e & 1)
                r = mul(r, a);
        return r;
    }
};

__constant__ uint64_t MR_BASES[16] = {2,  3,  5,  7,  11, 13, 17, 19,
                                      23, 29, 31, 37, 41, 43, 47, 53};

/* deterministic for every n < 2^64 (Sinclair's 7-base set) */
__constant__ uint64_t MR_BASES64[7] = {2,      325,     9375,      28178,
                                       450775, 9780504, 1795265022};

__device__ int dev_ctz(uint64_t v) { return __ffsll(v) - 1; }

/* ---- two-limb path (2^63 <= n < 2^126) ---- */

__device__ void dev_mul128(u128 a, u128 b, u128 & lo, u128 & hi)
{
    uint64_t const a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64);
    uint64_t const b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    u128 const p00 = (u128)a0 * b0;
    u128 mid = (u128)a0 * b1;
    u128 const p10 = (u128)a1 * b0;
    mid += p10;
    u128 const wrap = (u128)(mid < p10) << 64;
    lo = p00 + (mid << 64);
    hi = (u128)a1 * b1 + (mid >> 64) + wrap + (lo < p00);
}

__device__ int dev_ctz(u128 v)
{
    uint64_t const l = (uint64_t)v;
    return l ? __ffsll(l) - 1 : 63 + __ffsll((uint64_t)(v >> 64));
}

__device__ u128 dev_div128(u128 n, u128 d)
{
    int sh = 0;
    while (!(d >> 127) && (d << 1) <= n) {
        d <<= 1;
        sh++;
    }
    u128 q = 0;
    for (; sh >= 0; sh--, d >>= 1) {
        q <<= 1;
        if (n >= d) {
            n -= d;
            q |= 1;
        }
    }
    return q;
}

__device__ u128 dev_mod128(u128 a, u128 d)
{
    if (a < d)
        return a;
    int sh = 0;
    while (!(d >> 127) && (d << 1) <= a) {
        d <<= 1;
        sh++;
    }
    for (; sh >= 0; sh--, d >>= 1)
        if (a >= d)
            a -= d;
    return a;
}

struct mont128 {
    using t = u128;
    u128 n, ni, one, r2;

    __device__ explicit mont128(u128 n_) : n(n_)
    {
        u128 x = n;
        for (int i = 0; i < 6; i++)
            x *= 2 - n * x;
        ni = -x;
        one = dev_mod128(-n, n);   /* 2^128 mod n */
        r2 = one;
        for (int i = 0; i < 128; i++) {
            r2 += r2;
            if (r2 >= n)
                r2 -= n;
        }
    }
    __device__ u128 mul(u128 a, u128 b) const
    {
        u128 lo, hi;
        dev_mul128(a, b, lo, hi);
        u128 const m = lo * ni;
        u128 mn_lo, mn_hi;
        dev_mul128(m, n, mn_lo, mn_hi);
        u128 const r = hi + mn_hi + (lo != 0);
        return r >= n ? r - n : r;
    }
    __device__ u128 in(u128 a) const { return mul(a, r2); }
    __device__ u128 add(u128 a, u128 b) const
    {
        u128 const r = a + b;
        return r >= n ? r - n : r;
    }
    __device__ u128 sub(u128 a, u128 b) const
    {
        return a >= b ? a - b : a + n - b;
    }
    __device__ u128 pow(u128 a, u128 e) const
    {
        u128 r = one;
        for (; e; e >>= 1, a = mul(a, a))
            if (e & 1)
                r = mul(r, a);
        return r;
    }
};

/* ---- three-limb path (2^63 <= n < 2^96): the top limb of every
 * product stays below 2^32, so the reduction needs half the multiply
 * work of the two-limb path.  mfb1 = 90 workloads never leave it. ---- */

constexpr u128 MASK96 = ((u128)1 << 96) - 1;

__device__ void dev_mul96(u128 a, u128 b, u128 & lo, uint64_t & hi)
{
    uint64_t const a0 = (uint64_t)a, b0 = (uint64_t)b;
    uint64_t const a1 = (uint64_t)(a >> 64), b1 = (uint64_t)(b >> 64);
    u128 const p00 = (u128)a0 * b0;
    u128 const mid = (u128)a0 * b1 + (u128)a1 * b0;
    lo = p00 + (mid << 64);
    hi = a1 * b1 + (uint64_t)(mid >> 64) + (lo < p00);
}

struct mont96 {
    using t = u128;
    u128 n, ni, one, r2;

    __device__ explicit mont96(u128 n_) : n(n_)
    {
        u128 x = n;
        for (int i = 0; i < 6; i++)
            x *= 2 - n * x;
        ni = -x & MASK96;
        one = dev_mod128((u128)1 << 96, n);
        r2 = one;
        for (int i = 0; i < 96; i++) {
            r2 += r2;
            if (r2 >= n)
                r2 -= n;
        }
    }
    __device__ u128 mul(u128 a, u128 b) const
    {
        u128 lo;
        uint64_t hi;
        dev_mul96(a, b, lo, hi);
        u128 const m = lo * ni & MASK96;
        u128 mn_lo;
        uint64_t mn_hi;
        dev_mul96(m, n, mn_lo, mn_hi);
        u128 const r = ((u128)hi << 32) + ((u128)mn_hi << 32) +
                       (lo >> 96) + (mn_lo >> 96) + ((lo & MASK96) != 0);
        return r >= n ? r - n : r;
    }
    __device__ u128 in(u128 a) const { return mul(a, r2); }
    __device__ u128 add(u128 a, u128 b) const
    {
        u128 const r = a + b;
        return r >= n ? r - n : r;
    }
    __device__ u128 sub(u128 a, u128 b) const
    {
        return a >= b ? a - b : a + n - b;
    }
    __device__ u128 pow(u128 a, u128 e) const
    {
        u128 r = one;
        for (; e; e >>= 1, a = mul(a, a))
            if (e & 1)
                r = mul(r, a);
        return r;
    }
};

template <class M>
__device__ bool dev_prime(typename M::t n, uint64_t const * bases, int nb)
{
    M const Mm(n);
    int const s = dev_ctz(n - 1);
    typename M::t const d = (n - 1) >> s;
    for (int k = 0; k < nb; k++) {
        typename M::t x = Mm.pow(Mm.in(bases[k]), d);
        if (x == 0 || x == Mm.one || x == n - Mm.one)
            continue;
        bool composite = true;
        for (int r = 1; composite && r < s; r++) {
            x = Mm.mul(x, x);
            composite = x != n - Mm.one;
        }
        if (composite)
            return false;
    }
    return true;
}

__device__ bool dev_prime64(uint64_t n)
{
    return dev_prime<mont64>(n, MR_BASES64, 7);
}
__device__ bool dev_prime96(u128 n) { return dev_prime<mont96>(n, MR_BASES, 16); }
__device__ bool dev_prime128(u128 n)
{
    return dev_prime<mont128>(n, MR_BASES, 16);
}

/* ---------------- exact-norm survivor filter ----------------
 *
 * Reproduces the host pass-1 rejection on the device: recompute the
 * exact norms of a survivor, divide out the special q, the resieve
 * hits and the flagged tiny primes, and test the leftover against
 * leftover_ok.  Only passing survivors reach the host. */

__constant__ uint64_t c_flt_c[2][GLAS_MAX_DEGREE + 1][NORM_LIMBS];
__constant__ uint32_t c_flt_neg[2][GLAS_MAX_DEGREE + 1];
__constant__ uint64_t c_flt_lim[2];
__constant__ uint32_t c_flt_lpb[2];
__constant__ uint32_t c_flt_mfb[2];
/* The filter leaves the primality test of a leftover to
   filter_prime_kernel (GLAS_FLT_DEFER_MR) or to the host
   (GLAS_FLT_HOST_MR). */
__constant__ uint32_t c_flt_defer_mr;

struct wide {
    uint64_t v[NORM_LIMBS];
};

__device__ int wide_cmp(wide const & a, wide const & b)
{
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        if (a.v[i] != b.v[i])
            return a.v[i] < b.v[i] ? -1 : 1;
    return 0;
}

__device__ void wide_add(wide & a, wide const & b)
{
    uint64_t carry = 0;
    for (int i = 0; i < NORM_LIMBS; i++) {
        u128 const s = (u128)a.v[i] + b.v[i] + carry;
        a.v[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
}

__device__ void wide_sub(wide & a, wide const & b)   /* needs a >= b */
{
    uint64_t borrow = 0;
    for (int i = 0; i < NORM_LIMBS; i++) {
        u128 const s = (u128)a.v[i] - b.v[i] - borrow;
        a.v[i] = (uint64_t)s;
        borrow = (uint64_t)(s >> 64) & 1;
    }
}

__device__ void wide_mul_u64(wide & a, uint64_t m)
{
    uint64_t carry = 0;
    for (int i = 0; i < NORM_LIMBS; i++) {
        u128 const p = (u128)a.v[i] * m + carry;
        a.v[i] = (uint64_t)p;
        carry = (uint64_t)(p >> 64);
    }
}

__device__ uint64_t wide_divrem_u64(wide & a, uint64_t p)
{
    uint64_t rem = 0;
    for (int i = NORM_LIMBS - 1; i >= 0; i--) {
        u128 const cur = (u128)rem << 64 | a.v[i];
        a.v[i] = (uint64_t)(cur / p);
        rem = (uint64_t)(cur % p);
    }
    return rem;
}

__device__ int wide_divide_out(wide & n, uint64_t p)
{
    int k = 0;
    for (;;) {
        wide q = n;
        if (wide_divrem_u64(q, p))
            return k;
        n = q;
        k++;
    }
}

/* Live limbs: index of the top nonzero limb plus one. */
__device__ int wide_live(wide const & a)
{
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        if (a.v[i])
            return i + 1;
    return 0;
}

/* Divisor below 2^32 with 1 / p as a double: the quotient of a value
 * below p 2^32 is the truncated product or one off either way. */
struct div32 {
    uint32_t p;
    double inv;
};

__device__ div32 make_div32(uint32_t p)
{
    return {p, 1.0 / p};
}

/* Returns cur / p and leaves cur % p in cur; cur < p 2^32. */
__device__ uint64_t div32_step(uint64_t & cur, div32 d)
{
    uint64_t q = __double2ull_rz(__ull2double_rz(cur) * d.inv);
    int64_t r = (int64_t)(cur - q * d.p);
    if (r < 0) {
        q--;
        r += d.p;
    } else if (r >= (int64_t)d.p) {
        q++;
        r -= d.p;
    }
    cur = (uint64_t)r;
    return q;
}

/* q = n / p over the nl live limbs, in 64-by-32 steps; returns n mod p. */
__device__ uint32_t wide_divrem_u32(wide const & n, int nl, wide & q,
                                    div32 d)
{
    uint64_t rem = 0;
    for (int i = nl - 1; i >= 0; i--) {
        rem = rem << 32 | n.v[i] >> 32;
        uint64_t const qh = div32_step(rem, d);
        rem = rem << 32 | (uint32_t)n.v[i];
        uint64_t const ql = div32_step(rem, d);
        q.v[i] = qh << 32 | ql;
    }
    return (uint32_t)rem;
}

__device__ int wide_divide_out_u32(wide & n, int & nl, uint32_t p)
{
    div32 const d = make_div32(p);
    int k = 0;
    for (;;) {
        wide q = n;
        if (!nl || wide_divrem_u32(n, nl, q, d))
            return k;
        n = q;
        if (n.v[nl - 1] == 0)
            nl--;
        k++;
    }
}

__device__ int wide_bits(wide const & a)
{
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        if (a.v[i])
            return 64 * (i + 1) - __clzll((long long)a.v[i]);
    return 0;
}

__device__ double wide_to_double(wide const & a)
{
    double v = 0.0;
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        v = v * 18446744073709551616.0 + (double)a.v[i];
    return v;
}

/* |F_side(a, b)| with b >= 0: signed sum of c_k a^k b^(d-k) */
__device__ void dev_wide_norm(int side, int d, int64_t a, uint64_t b,
                              wide & out)
{
    uint64_t const aa = a < 0 ? (uint64_t)-a : (uint64_t)a;
    wide pos = {}, neg = {};
    for (int k = 0; k <= d; k++) {
        wide term;
        for (int i = 0; i < NORM_LIMBS; i++)
            term.v[i] = c_flt_c[side][k][i];
        for (int m = 0; m < k; m++)
            wide_mul_u64(term, aa);
        for (int m = k; m < d; m++)
            wide_mul_u64(term, b);
        bool const tneg = c_flt_neg[side][k] ^ (a < 0 && (k & 1));
        wide_add(tneg ? neg : pos, term);
    }
    if (wide_cmp(pos, neg) >= 0) {
        wide_sub(pos, neg);
        out = pos;
    } else {
        wide_sub(neg, pos);
        out = neg;
    }
}

__device__ bool dev_prime_any(u128 v)
{
    return v >> 96 ? dev_prime128(v)
         : v >> 63 ? dev_prime96(v)
                   : dev_prime64((uint64_t)v);
}

__device__ bool dev_prime_lo(uint64_t const * lo)
{
    return dev_prime_any((u128)lo[1] << 64 | lo[0]);
}

/* las check_leftover_norm, like the host leftover_ok.  LEFT_PENDING:
   the leftover passes the size checks and still owes the primality
   test (c_flt_defer_mr). */
enum leftover_state { LEFT_BAD, LEFT_OK, LEFT_PENDING };
__device__ leftover_state dev_leftover(wide const & n, int side)
{
    int const s = wide_bits(n);
    if (s == 0 || s > (int)c_flt_mfb[side])
        return LEFT_BAD;
    if (s <= (int)c_flt_lpb[side])
        return LEFT_OK;
    double const nd = wide_to_double(n);
    double const B = (double)c_flt_lim[side];
    double kB = B * B;
    for (uint32_t klpb = c_flt_lpb[side]; klpb < (uint32_t)s;
         klpb += c_flt_lpb[side], kB *= B)
        if (nd < kB)
            return LEFT_BAD;
    if (c_flt_defer_mr)
        return LEFT_PENDING;
    return dev_prime_lo(n.v) ? LEFT_BAD : LEFT_OK;   /* s <= mfb <= 120 */
}

__device__ uint32_t dev_hits_lower_bound(uint64_t const * hits, uint32_t n,
                                         uint64_t key)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t const mid = (lo + hi) / 2;
        if (hits[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__device__ u128 dev_gcd128(u128 a, u128 b);

/* The kernel also assembles the candidate for each kept survivor: the
 * primes it divides out go to out_fac and the leftover norms go to
 * out_lo.  Output slots at fcap and above, non-coprime (i, j) pairs
 * and factor lists past GPU_FAC_CAP get no assembled record; the
 * host assembles those survivors itself from their resieve hits,
 * which the kernel appends to out_hits.  out_cnt[0] counts the kept
 * survivors and out_cnt[1] the hits in out_hits. */
__global__ void filter_kernel(gpu_task t, uint32_t const * survivors,
                              uint32_t n_surv, uint64_t const * tiny,
                              uint64_t const * hits, uint32_t n_hits,
                              uint64_t const * hits_r,
                              uint32_t const * hit_rcnt,
                              uint64_t const * hits_ov,
                              uint32_t const * ov_cnt,
                              uint32_t * out_surv, uint64_t * out_tiny,
                              uint32_t * out_cnt, uint32_t * out_meta,
                              uint32_t * out_fac, uint64_t * out_lo,
                              uint32_t fcap, uint64_t * out_hits)
{
    uint32_t const I = 1u << t.logI;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_surv;
         idx += gridDim.x * blockDim.x) {
        uint32_t const x = survivors[idx];
        int64_t const i = (int64_t)(x & (I - 1)) - I / 2;
        int64_t const j = (int64_t)t.j0 + (x >> t.logI);
        int64_t a = i * t.a0 + j * t.a1;
        int64_t b = i * t.b0 + j * t.b1;
        if (b < 0) {
            a = -a;
            b = -b;
        }
        if (b == 0)
            continue;
        /* The q-lattice is the (a, b) with a == rho*b (mod q), so
         * q | b implies q | gcd(a, b): not a valid relation even
         * with gcd(i, j) == 1 (las gcd_coprime_with_q). */
        if ((uint64_t)b % t.q == 0)
            continue;
        /* The survivor's hits.  With region hits they sit in the
         * region's list, and in the shared tail when the list
         * overflowed; without them, the out stream sorted the flat
         * list by position and a binary search finds them. */
        uint32_t h0 = 0, h1 = 0;
        uint32_t const reg = x >> SREG_LOG;
        uint32_t rcnt = 0, onov = 0;
        if (hits_r) {
            rcnt = hit_rcnt[reg];
            if (rcnt > RHITS_CAP)
                onov = min(*ov_cnt, RHITS_OV);
        } else {
            h0 = dev_hits_lower_bound(hits, n_hits, (uint64_t)x << 32);
            h1 = h0;
            while (h1 < n_hits && (hits[h1] >> 32) == x)
                h1++;
        }
        auto each_hit = [&](auto f) {
            if (hits_r) {
                for (uint32_t k = 0, n = min(rcnt, RHITS_CAP); k < n; k++) {
                    uint64_t const h = hits_r[(uint64_t)reg * RHITS_CAP + k];
                    if (h >> 32 == x)
                        f(h);
                }
                for (uint32_t k = 0; k < onov; k++) {
                    uint64_t const h = hits_ov[k];
                    if (h >> 32 == x)
                        f(h);
                }
                return;
            }
            for (uint32_t h = h0; h < h1; h++)
                f(hits[h]);
        };
        uint64_t const ia = i < 0 ? (uint64_t)-i : (uint64_t)i;
        bool assemble = dev_gcd128(ia, (u128)(uint64_t)j) == 1;
        uint32_t lf[GPU_FAC_CAP];
        uint32_t nf = 0, nf0 = 0, nq = 0, pend = 0;
        uint64_t lo[2][2];
        auto record = [&](uint32_t p, int e) {
            for (; e; e--) {
                if (nf == GPU_FAC_CAP) {
                    assemble = false;
                    return;
                }
                lf[nf++] = p;
            }
        };
        bool ok = true;
        for (int side = 0; side < 2 && ok; side++) {
            wide n;
            dev_wide_norm(side, t.degree[side], a, (uint64_t)b, n);
            int nl = wide_live(n);
            if (side == t.sq_side) {
                nq = t.q >> 32 ? wide_divide_out(n, t.q)
                               : wide_divide_out_u32(n, nl, (uint32_t)t.q);
                if (!nq) {
                    ok = false;
                    break;
                }
                nl = wide_live(n);
            }
            each_hit([&](uint64_t h) {
                if ((h & 1) != (uint64_t)side)
                    return;
                uint32_t const p = (uint32_t)h >> 1;
                if (side == t.sq_side && p == t.q)
                    return;
                record(p, wide_divide_out_u32(n, nl, p));
            });
            for (uint64_t m = tiny[2 * (uint64_t)idx + side]; m; m &= m - 1) {
                uint32_t const p = c_tiny_p[__ffsll((long long)m) - 1];
                record(p, wide_divide_out_u32(n, nl, p));
            }
            leftover_state const l = dev_leftover(n, side);
            ok = l != LEFT_BAD;
            pend |= (uint32_t)(l == LEFT_PENDING) << side;
            if (side == 0)
                nf0 = nf;
            lo[side][0] = n.v[0];
            lo[side][1] = n.v[1];
            for (int w = 2; w < NORM_LIMBS; w++)
                if (n.v[w])
                    assemble = false;
        }
        if (!ok)
            continue;
        uint32_t const slot = atomicAdd(out_cnt, 1);
        out_surv[slot] = x;
        out_tiny[2 * (uint64_t)slot] = tiny[2 * (uint64_t)idx];
        out_tiny[2 * (uint64_t)slot + 1] = tiny[2 * (uint64_t)idx + 1];
        assemble = assemble && slot < fcap;
        if (slot < fcap)
            out_meta[slot] = assemble ? 1u << 31 | pend << 29 | nq << 16 |
                                            (nf - nf0) << 8 | nf0
                                      : 0;
        if (!assemble) {
            uint32_t nh = h1 - h0;
            if (hits_r) {
                nh = 0;
                each_hit([&](uint64_t) { nh++; });
            }
            uint32_t const hslot = atomicAdd(out_cnt + 1, nh);
            uint32_t k = 0;
            each_hit([&](uint64_t h) { out_hits[hslot + k++] = h; });
            continue;
        }
        uint32_t * const fac = out_fac + (uint64_t)slot * GPU_FAC_CAP;
        for (uint32_t k = 0; k < nf; k++)
            fac[k] = lf[k];
        uint64_t * const l = out_lo + 4 * (uint64_t)slot;
        l[0] = lo[0][0];
        l[1] = lo[0][1];
        l[2] = lo[1][0];
        l[3] = lo[1][1];
    }
}

/* The primality tests the filter deferred, over the assembled records
   only: the host repeats its own leftover check on the survivors it
   assembles itself.  A record with a prime leftover becomes
   FMETA_REJECT.  With GLAS_FLT_HOST_MR the kernel does not run and the
   pending bits reach the host in gpu_cand::pend. */
__global__ void filter_prime_kernel(uint32_t const * cnt, uint32_t fcap,
                                    uint32_t * meta, uint64_t const * lo)
{
    uint32_t const n = min(*cnt, fcap);
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += gridDim.x * blockDim.x) {
        uint32_t const m = meta[k];
        uint32_t const pend = m >> 29 & 3;
        if (!(m >> 31) || !pend)
            continue;
        bool prime = false;
        for (int side = 0; side < 2; side++)
            if (pend >> side & 1)
                prime |= dev_prime_lo(lo + 4 * (uint64_t)k + 2 * side);
        meta[k] = prime ? FMETA_REJECT : m & ~(3u << 29);
    }
}

__device__ u128 dev_gcd128(u128 a, u128 b)
{
    if (!a)
        return b;
    int const sh = dev_ctz(a | b);
    a >>= dev_ctz(a);
    while (b) {
        b >>= dev_ctz(b);
        if (a > b) {
            u128 const t = a;
            a = b;
            b = t;
        }
        b -= a;
    }
    return a << sh;
}

/* ---- batched ECM ----
 *
 * Montgomery curves By^2 = x^3 + Ax^2 + x with a24 = (A+2)/4 a small
 * per-curve constant from the Suyama parametrization, so the group
 * order is divisible by 12.  Stage 1 is a Montgomery ladder over a
 * fixed powersmooth exponent; stage 2 is a baby-step giant-step
 * continuation modulo 210 with the baby steps normalized to z = 1.
 * The exponent bits and the giant-step unit masks are precomputed on
 * the host and shared by every thread, so control flow is uniform
 * across a warp.
 *
 * The work runs in rounds.  A round runs one curve on every composite
 * that is still open (cofactor_ecm_kernel, one thread per composite,
 * one launch per limb class), then the split kernel sorts the pieces:
 * a prime piece goes to the factor list of its side, a composite piece
 * goes back to the list of its class with a fresh curve budget, and a
 * composite whose curve budget is spent leaves its side ECM_PENDING.
 * The lists are compacted every round, so a warp never idles on lanes
 * that finished early, and the curve kernel carries no split state. */

constexpr int ECM_UNITS = 24;    /* units mod 210 below 105 */

struct ecm_plan {
    uint32_t const * s1;    /* stage-1 exponent bits, LSB first */
    uint32_t s1_bits;
    uint32_t const * s2;    /* per-giant-step unit masks */
    uint32_t m0, nm;        /* giant steps m0 .. m0+nm-1 */
    uint32_t curves;        /* curve budget per composite */
};

__constant__ int8_t c_ecm_uidx[53];   /* odd u < 105 -> unit index */

/* Per-side ECM output, indexed by pair * 2 + side and merged into
 * cofac_result on the host.  The status is the maximum over the
 * pieces of the side, so one prime above lpb marks the side not
 * smooth even when another piece is still open. */
enum { ECM_SMOOTH = 0, ECM_PENDING = 1, ECM_NOT_SMOOTH = 2 };
struct ecm_side_result {
    uint64_t f[COFAC_MAX_FACTORS];
    uint32_t nf;
    uint32_t status;
};

/* One open composite of a side, and the index of its next curve. */
struct ecm_item {
    uint64_t n[2];
    uint32_t ent;      /* pair * 2 + side */
    uint32_t curve;
};

/* The open composites by limb class: below 2^63, below 2^96, wider. */
struct ecm_lists {
    ecm_item * items[3];
    uint32_t * cnt;
    uint32_t cap;
};

__device__ int ecm_class(u128 v) { return v >> 96 ? 2 : v >> 63 ? 1 : 0; }

/* Appends a composite to the list of its class.  A full list leaves
 * the side to the host. */
__device__ void ecm_push(ecm_lists const & L, u128 v, uint32_t ent,
                         uint32_t curve, ecm_side_result & r)
{
    int const c = ecm_class(v);
    uint32_t const j = atomicAdd(L.cnt + c, 1);
    if (j >= L.cap) {
        atomicMax(&r.status, (uint32_t)ECM_PENDING);
        return;
    }
    L.items[c][j] = {{(uint64_t)v, (uint64_t)(v >> 64)}, ent, curve};
}

__device__ uint64_t dev_gcd_any(uint64_t a, uint64_t b)
{
    return dev_gcd64(a, b);
}
__device__ u128 dev_gcd_any(u128 a, u128 b) { return dev_gcd128(a, b); }

template <class M>
__device__ void dev_xdbl(M const & Mm, typename M::t a24, typename M::t X,
                         typename M::t Z, typename M::t & X2,
                         typename M::t & Z2)
{
    typename M::t const a = Mm.add(X, Z), b = Mm.sub(X, Z);
    typename M::t const aa = Mm.mul(a, a), bb = Mm.mul(b, b);
    typename M::t const c = Mm.sub(aa, bb);
    X2 = Mm.mul(aa, bb);
    Z2 = Mm.mul(c, Mm.add(bb, Mm.mul(a24, c)));
}

/* x-only P1+P2 given the difference P1-P2 */
template <class M>
__device__ void dev_xadd(M const & Mm, typename M::t X1, typename M::t Z1,
                         typename M::t X2, typename M::t Z2,
                         typename M::t XD, typename M::t ZD,
                         typename M::t & Xo, typename M::t & Zo)
{
    typename M::t const u = Mm.mul(Mm.sub(X1, Z1), Mm.add(X2, Z2));
    typename M::t const v = Mm.mul(Mm.add(X1, Z1), Mm.sub(X2, Z2));
    typename M::t const s = Mm.add(u, v), d = Mm.sub(u, v);
    Xo = Mm.mul(ZD, Mm.mul(s, s));
    Zo = Mm.mul(XD, Mm.mul(d, d));
}

/* Inverse of the Montgomery-form value am by binary extended gcd.
 * Returns 1 and sets inv (also Montgomery form), or returns
 * gcd(am, n) > 1. */
template <class M>
__device__ typename M::t dev_minv(M const & Mm, typename M::t am,
                                  typename M::t & inv)
{
    using T = typename M::t;
    T u = Mm.mul(am, 1), v = Mm.n, x1 = 1, x2 = 0;
    while (u) {
        while (!(u & 1)) {
            u >>= 1;
            x1 = (x1 & 1 ? x1 + Mm.n : x1) >> 1;
        }
        if (u < v) {
            T t = u;
            u = v;
            v = t;
            t = x1;
            x1 = x2;
            x2 = t;
        }
        u -= v;
        x1 = Mm.sub(x1, x2);
    }
    if (v != 1)
        return v;
    inv = Mm.in(x2);
    return 1;
}

/* Suyama curve for sigma: start point (x0:z0) and a24.  Returns 1
 * when the curve is ready, or gcd > 1 when a divisor of n makes it
 * degenerate. */
template <class M>
__device__ typename M::t dev_curve(M const & Mm, uint32_t sigma,
                                   typename M::t & x0, typename M::t & z0,
                                   typename M::t & a24)
{
    using T = typename M::t;
    T const s = Mm.in(sigma);
    T const u = Mm.sub(Mm.mul(s, s), Mm.in(5));
    T const v = Mm.add(Mm.add(s, s), Mm.add(s, s));
    T const u3 = Mm.mul(Mm.mul(u, u), u);
    T const v3 = Mm.mul(Mm.mul(v, v), v);
    T const w = Mm.sub(v, u);
    T const num = Mm.mul(Mm.mul(Mm.mul(w, w), w),
                         Mm.add(Mm.add(Mm.add(u, u), u), v));
    T const den = Mm.mul(Mm.mul(u3, v), Mm.in(16));
    T dinv;
    T const g = dev_minv(Mm, den, dinv);
    if (g != 1)
        return g;
    x0 = u3;
    z0 = v3;
    a24 = Mm.mul(num, dinv);
    return 1;
}

/* One curve, stage 1 + stage 2; returns a proper factor of n or 0. */
template <class M>
__device__ typename M::t dev_ecm_curve(M const & Mm, ecm_plan const & pl,
                                       uint32_t curve)
{
    using T = typename M::t;
    T px, pz, a24;
    T g = dev_curve(Mm, 6 + curve, px, pz, a24);
    if (g != 1)
        return g != Mm.n ? g : (T)0;

    /* stage 1: (X0,Z0) = [s]P by a ladder with fixed difference P */
    T X0 = px, Z0 = pz, X1, Z1;
    dev_xdbl(Mm, a24, X0, Z0, X1, Z1);
    for (int i = (int)pl.s1_bits - 2; i >= 0; i--) {
        T Xa, Za, Xd, Zd;
        dev_xadd(Mm, X0, Z0, X1, Z1, px, pz, Xa, Za);
        if (pl.s1[i >> 5] >> (i & 31) & 1) {
            dev_xdbl(Mm, a24, X1, Z1, Xd, Zd);
            X0 = Xa, Z0 = Za, X1 = Xd, Z1 = Zd;
        } else {
            dev_xdbl(Mm, a24, X0, Z0, Xd, Zd);
            X0 = Xd, Z0 = Zd, X1 = Xa, Z1 = Za;
        }
    }
    g = dev_gcd_any(Z0, Mm.n);
    if (g != 1)
        return g != Mm.n ? g : (T)0;
    if (!pl.nm)
        return 0;

    /* stage 2: units mod 210.  Baby steps [u]Q for the 24 units below
     * 105, giant steps [m*210]Q; a prime p = 210m +- u in (B1,B2]
     * contributes the cross term X_m Z_u - X_u Z_m to the product. */
    T ux[ECM_UNITS], uz[ECM_UNITS];
    T X2, Z2;
    dev_xdbl(Mm, a24, X0, Z0, X2, Z2);          /* [2]Q */
    ux[0] = X0, uz[0] = Z0;                     /* [1]Q */
    T Xp = X0, Zp = Z0, Xc, Zc;                 /* [j-2]Q, [j]Q */
    dev_xadd(Mm, X2, Z2, X0, Z0, X0, Z0, Xc, Zc);   /* [3]Q */
    for (int j = 3; j <= 105; j += 2) {
        int const k = c_ecm_uidx[j >> 1];
        if (k >= 0)
            ux[k] = Xc, uz[k] = Zc;
        T Xn, Zn;
        dev_xadd(Mm, Xc, Zc, X2, Z2, Xp, Zp, Xn, Zn);
        Xp = Xc, Zp = Zc, Xc = Xn, Zc = Zn;
    }
    /* normalize the baby steps to z = 1 by one batched inversion */
    T c[ECM_UNITS];
    c[0] = uz[0];
    for (int k = 1; k < ECM_UNITS; k++)
        c[k] = Mm.mul(c[k - 1], uz[k]);
    T inv;
    g = dev_minv(Mm, c[ECM_UNITS - 1], inv);
    if (g != 1)
        return g != Mm.n ? g : (T)0;
    for (int k = ECM_UNITS - 1; k > 0; k--) {
        ux[k] = Mm.mul(ux[k], Mm.mul(inv, c[k - 1]));
        inv = Mm.mul(inv, uz[k]);
    }
    ux[0] = Mm.mul(ux[0], inv);

    /* Xp,Zp = [105]Q; W = [210]Q */
    T WX, WZ;
    dev_xdbl(Mm, a24, Xp, Zp, WX, WZ);
    /* (A,B) = ([m0]W, [m0+1]W) by a ladder with fixed difference W */
    T AX = WX, AZ = WZ, BX, BZ;
    dev_xdbl(Mm, a24, WX, WZ, BX, BZ);
    for (int i = 30 - __clz(pl.m0); i >= 0; i--) {
        T Xa, Za, Xd, Zd;
        dev_xadd(Mm, AX, AZ, BX, BZ, WX, WZ, Xa, Za);
        if (pl.m0 >> i & 1) {
            dev_xdbl(Mm, a24, BX, BZ, Xd, Zd);
            AX = Xa, AZ = Za, BX = Xd, BZ = Zd;
        } else {
            dev_xdbl(Mm, a24, AX, AZ, Xd, Zd);
            AX = Xd, AZ = Zd, BX = Xa, BZ = Za;
        }
    }
    T acc = Mm.one;
    for (uint32_t m = 0; m < pl.nm; m++) {
        for (uint32_t msk = pl.s2[m]; msk; msk &= msk - 1) {
            int const k = __ffs(msk) - 1;
            T const cross = Mm.sub(AX, Mm.mul(ux[k], AZ));
            acc = Mm.mul(acc, cross);
        }
        T Xn, Zn;
        dev_xadd(Mm, BX, BZ, WX, WZ, AX, AZ, Xn, Zn);
        AX = BX, AZ = BZ, BX = Xn, BZ = Zn;
    }
    g = dev_gcd_any(acc, Mm.n);
    return g != 1 && g != Mm.n ? g : (T)0;
}

/* Pass 1: primality-test both sides, record the prime ones, and list
 * the composite ones of the pairs that are still smooth. */
__global__ void cofactor_classify_kernel(cofac_pair const * in,
                                         cofac_result * out, uint32_t n,
                                         uint64_t lpb0, uint64_t lpb1,
                                         ecm_side_result * res,
                                         ecm_lists open)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    uint64_t const lpb_max[2] = {lpb0, lpb1};
    cofac_result r = {};
    r.smooth = 1;
    for (int side = 0; side < 2 && r.smooth; side++) {
        u128 const v = (u128)in[idx].n[side][1] << 64 | in[idx].n[side][0];
        if (v == 1)
            continue;
        if (!dev_prime_any(v))
            r.pending |= 1 << side;
        else if (v > lpb_max[side])
            r.smooth = 0;
        else
            r.f[side][r.nf[side]++] = (uint64_t)v;
    }
    out[idx] = r;
    if (!r.smooth)
        return;
    for (int side = 0; side < 2; side++) {
        if (!(r.pending & (1 << side)))
            continue;
        u128 const v = (u128)in[idx].n[side][1] << 64 | in[idx].n[side][0];
        uint32_t const ent = idx << 1 | side;
        res[ent] = {};
        ecm_push(open, v, ent, 0, res[ent]);
    }
}

/* Pass 2, one round: one curve on every open composite of one class.
 * A composite of a pair that another side already failed skips the
 * curve; the split kernel drops it. */
template <class M>
__global__ void cofactor_ecm_kernel(ecm_item const * items, uint32_t n,
                                    cofac_result const * out, u128 * g,
                                    ecm_plan pl)
{
    uint32_t const j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
        return;
    ecm_item const it = items[j];
    if (!out[it.ent >> 1].smooth) {
        g[j] = 0;
        return;
    }
    M const Mm((typename M::t)((u128)it.n[1] << 64 | it.n[0]));
    g[j] = dev_ecm_curve(Mm, pl, it.curve);
}

/* Records a prime piece of a side, or reopens a composite one. */
__device__ void ecm_piece(u128 v, uint32_t ent, uint64_t lpb_max,
                          ecm_lists const & next, cofac_result & pair,
                          ecm_side_result & r)
{
    if (v == 1)
        return;
    if (!dev_prime_any(v)) {
        ecm_push(next, v, ent, 0, r);
        return;
    }
    uint32_t const m = atomicAdd(&r.nf, 1);
    if (v > lpb_max || m >= COFAC_MAX_FACTORS) {
        atomicMax(&r.status, (uint32_t)ECM_NOT_SMOOTH);
        pair.smooth = 0;
        return;
    }
    r.f[m] = (uint64_t)v;
}

/* Pass 2, one round: after the curve kernels, sort the outcome of every
 * open composite into the lists of the next round. */
__global__ void cofactor_split_kernel(ecm_item const * items, uint32_t n,
                                      u128 const * g, uint64_t lpb0,
                                      uint64_t lpb1, uint32_t curves,
                                      cofac_result * out,
                                      ecm_side_result * res, ecm_lists next)
{
    uint32_t const j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
        return;
    ecm_item const it = items[j];
    cofac_result & pair = out[it.ent >> 1];
    if (!pair.smooth)
        return;
    u128 const v = (u128)it.n[1] << 64 | it.n[0];
    ecm_side_result & r = res[it.ent];
    if (!g[j]) {
        if (it.curve + 1 < curves)
            ecm_push(next, v, it.ent, it.curve + 1, r);
        else
            atomicMax(&r.status, (uint32_t)ECM_PENDING);
        return;
    }
    uint64_t const lpb_max = it.ent & 1 ? lpb1 : lpb0;
    ecm_piece(g[j], it.ent, lpb_max, next, pair, r);
    ecm_piece(dev_div128(v, g[j]), it.ent, lpb_max, next, pair, r);
}

}  // namespace

/* ---------------- host API ---------------- */

namespace {

/* Build the shared ECM schedule: the stage-1 powersmooth exponent
 * s = prod p^max(e : p^e <= B1), the stage-2 giant-step unit masks,
 * and the unit tables.  B1/B2/curve budget come from GLAS_ECM_B1,
 * GLAS_ECM_B2 and GLAS_ECM_CURVES. */
ecm_plan make_ecm_plan()
{
    uint32_t const B1 =
        getenv("GLAS_ECM_B1") ? atoi(getenv("GLAS_ECM_B1")) : 1000;
    uint32_t const B2 =
        getenv("GLAS_ECM_B2") ? atoi(getenv("GLAS_ECM_B2")) : 50 * B1;
    uint32_t const curves =
        getenv("GLAS_ECM_CURVES") ? atoi(getenv("GLAS_ECM_CURVES")) : 8;
    assert(B1 >= 105);

    uint32_t const lim = std::max(B1, B2) + 1;
    std::vector<bool> sieve(lim, true);
    for (uint32_t p = 2; (uint64_t)p * p < lim; p++)
        if (sieve[p])
            for (uint32_t k = p * p; k < lim; k += p)
                sieve[k] = false;

    /* stage-1 exponent bits */
    mpz_t s;
    mpz_init_set_ui(s, 1);
    for (uint32_t p = 2; p <= B1; p++) {
        if (!sieve[p])
            continue;
        uint64_t pp = p;
        while (pp * p <= B1)
            pp *= p;
        mpz_mul_ui(s, s, pp);
    }
    uint32_t const nbits = mpz_sizeinbase(s, 2);
    std::vector<uint32_t> bits((nbits + 31) / 32, 0);
    for (uint32_t i = 0; i < nbits; i++)
        if (mpz_tstbit(s, i))
            bits[i >> 5] |= 1u << (i & 31);
    mpz_clear(s);

    /* units mod 210 below 105 */
    int8_t uidx[53];
    memset(uidx, -1, sizeof uidx);
    int nu = 0;
    for (int u = 1; u < 105; u += 2)
        if (u % 3 && u % 5 && u % 7)
            uidx[u >> 1] = nu++;
    assert(nu == ECM_UNITS);

    /* stage-2 masks: prime p in (B1,B2] -> (m, u) with p = 210m +- u */
    uint32_t const m0 = std::max(1u, (B1 + 106) / 210);
    uint32_t const m1 = B2 > B1 ? (B2 + 105) / 210 : 0;
    uint32_t const nm = m1 >= m0 ? m1 - m0 + 1 : 0;
    std::vector<uint32_t> masks(std::max(nm, 1u), 0);
    for (uint32_t p = B1 + 1; p <= B2 && p < lim; p++) {
        if (!sieve[p])
            continue;
        uint32_t const m = (p + 105) / 210;
        assert(m >= m0 && m <= m1);
        uint32_t const u = p > 210 * m ? p - 210 * m : 210 * m - p;
        assert(u < 105 && uidx[u >> 1] >= 0);
        masks[m - m0] |= 1u << uidx[u >> 1];
    }

    ecm_plan pl;
    uint32_t * d_s1, * d_s2;
    CUDA_CHECK(cudaMalloc(&d_s1, bits.size() * 4));
    CUDA_CHECK(cudaMemcpy(d_s1, bits.data(), bits.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_s2, masks.size() * 4));
    CUDA_CHECK(cudaMemcpy(d_s2, masks.data(), masks.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ecm_uidx, uidx, sizeof uidx));
    pl.s1 = d_s1;
    pl.s1_bits = nbits;
    pl.s2 = d_s2;
    pl.m0 = m0;
    pl.nm = nm;
    pl.curves = curves;
    return pl;
}

}  // namespace

/* Each worker thread keeps its own stream and buffers, so batches from
 * different special-q overlap on the device. */
void gpu_cofactor(std::vector<cofac_pair> const & in,
                  std::vector<cofac_result> & out, int const lpb[2],
                  bool hard_on_gpu)
{
    /* One launch takes up to CAP pairs; GLAS_COFAC_BATCH raises the
     * flush bound in glas.cpp, so the buffers follow it. */
    static uint32_t const CAP = [] {
        uint32_t cap = 1u << 16;
        if (auto const * v = getenv("GLAS_COFAC_BATCH"))
            cap = std::max(cap, (uint32_t)strtoul(v, nullptr, 10));
        return cap;
    }();
    static std::once_flag plan_once;
    static ecm_plan plan;
    static thread_local cudaStream_t st;
    static thread_local cofac_pair * d_in;
    static thread_local cofac_result * d_out;
    static thread_local ecm_side_result * d_res;
    static thread_local ecm_lists lists[2];   /* this round, next round */
    static thread_local u128 * d_g;
    static thread_local cofac_pair * h_in;
    static thread_local cofac_result * h_out;
    static thread_local ecm_side_result * h_res;
    static thread_local uint32_t * h_cnt;
    static thread_local cudaEvent_t ev[2];

    out.resize(in.size());
    if (in.empty())
        return;
    std::call_once(plan_once, [] { plan = make_ecm_plan(); });
    if (!d_in) {
        int least, greatest;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking,
                                                least));
        CUDA_CHECK(cudaEventCreate(&ev[0]));
        CUDA_CHECK(cudaEventCreate(&ev[1]));
        CUDA_CHECK(cudaMalloc(&d_in, CAP * sizeof(cofac_pair)));
        CUDA_CHECK(cudaMalloc(&d_out, CAP * sizeof(cofac_result)));
        CUDA_CHECK(cudaMalloc(&d_res, 2 * CAP * sizeof(ecm_side_result)));
        CUDA_CHECK(cudaMalloc(&d_g, 2 * CAP * sizeof(u128)));
        for (ecm_lists & L : lists) {
            L.cap = 2 * CAP;
            CUDA_CHECK(cudaMalloc(&L.cnt, 3 * sizeof(uint32_t)));
            for (ecm_item *& items : L.items)
                CUDA_CHECK(cudaMalloc(&items, L.cap * sizeof(ecm_item)));
        }
        CUDA_CHECK(cudaMallocHost(&h_in, CAP * sizeof(cofac_pair)));
        CUDA_CHECK(cudaMallocHost(&h_out, CAP * sizeof(cofac_result)));
        CUDA_CHECK(cudaMallocHost(&h_res, 2 * CAP * sizeof(ecm_side_result)));
        CUDA_CHECK(cudaMallocHost(&h_cnt, 3 * sizeof(uint32_t)));
    }
    uint64_t const lpb0 = UINT64_C(1) << lpb[0];
    uint64_t const lpb1 = UINT64_C(1) << lpb[1];
    auto count_open = [&](ecm_lists const & L) {
        CUDA_CHECK(cudaMemcpyAsync(h_cnt, L.cnt, 3 * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        return h_cnt[0] + h_cnt[1] + h_cnt[2];
    };
    for (size_t base = 0; base < in.size(); base += CAP) {
        uint32_t const n = std::min((size_t)CAP, in.size() - base);
        memcpy(h_in, in.data() + base, n * sizeof(cofac_pair));
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, n * sizeof(cofac_pair),
                                   cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaEventRecord(ev[0], st));
        CUDA_CHECK(cudaMemsetAsync(lists[0].cnt, 0, 3 * sizeof(uint32_t), st));
        cofactor_classify_kernel<<<(n + 127) / 128, 128, 0, st>>>(
            d_in, d_out, n, lpb0, lpb1, d_res, lists[0]);
        for (int p = 0; count_open(lists[p]); p ^= 1) {
            ecm_lists const & cur = lists[p], & next = lists[p ^ 1];
            CUDA_CHECK(cudaMemsetAsync(next.cnt, 0, 3 * sizeof(uint32_t), st));
            for (int c = 0; c < 3; c++) {
                uint32_t const m = std::min(h_cnt[c], cur.cap);
                if (!m)
                    continue;
                uint32_t const grid = (m + 127) / 128;
                if (c == 0)
                    cofactor_ecm_kernel<mont64><<<grid, 128, 0, st>>>(
                        cur.items[c], m, d_out, d_g, plan);
                else if (c == 1)
                    cofactor_ecm_kernel<mont96><<<grid, 128, 0, st>>>(
                        cur.items[c], m, d_out, d_g, plan);
                else
                    cofactor_ecm_kernel<mont128><<<grid, 128, 0, st>>>(
                        cur.items[c], m, d_out, d_g, plan);
                cofactor_split_kernel<<<grid, 128, 0, st>>>(
                    cur.items[c], m, d_g, lpb0, lpb1, plan.curves, d_out,
                    d_res, next);
            }
        }
        CUDA_CHECK(cudaEventRecord(ev[1], st));
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, n * sizeof(cofac_result),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res,
                                   2 * n * sizeof(ecm_side_result),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        if (ctx.stats) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
            std::unique_lock<std::mutex> const lk(ctx.lock);
            ctx.ecm_ms += ms;
        }

        /* merge the per-side ECM results */
        for (uint32_t idx = 0; idx < n; idx++) {
            cofac_result & r = h_out[idx];
            for (int side = 0; r.smooth && side < 2; side++) {
                if (!(r.pending & (1 << side)))
                    continue;
                ecm_side_result const & e = h_res[idx << 1 | side];
                if (e.status == ECM_SMOOTH) {
                    r.nf[side] = e.nf;
                    for (uint32_t m = 0; m < e.nf; m++)
                        r.f[side][m] = e.f[m];
                    r.pending &= ~(1 << side);
                } else if (e.status == ECM_NOT_SMOOTH || hard_on_gpu) {
                    r.smooth = 0;
                }
            }
        }
        memcpy(out.data() + base, h_out, n * sizeof(cofac_result));
    }
}

/* Blocks of `block` threads and `smem` bytes of dynamic shared memory
 * that fill the device with one wave of `kernel`: SM count times the
 * resident blocks per SM. */
template <typename K>
static uint32_t full_grid(K kernel, int block, size_t smem)
{
    int sm = 0, per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                             block, smem));
    return sm * per_sm;
}

/* Creates the CUDA context, which takes seconds when several
 * processes start at once.  main runs it on a thread while the CPU
 * reads the factor base.  A thread that waits for the device sleeps
 * on an OS primitive instead of spinning (GLAS_GPU_BLOCK, 0 restores
 * the spin), so the CPU of a waiting siever is free for the workers.
 * Errors are ignored here; gpu_init reports a missing device. */
void gpu_warmup()
{
    if (env_flag("GLAS_GPU_BLOCK", true))
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaFree(nullptr);
}

void gpu_prepare(std::vector<fb_entry> const * fb, int logI, unsigned threads)
{
    auto const t0 = std::chrono::steady_clock::now();
    uint32_t const I = 1u << logI;
    if (char const * e = getenv("GLAS_REGION_SKIP"))
        ctx.skip = atoi(e);
    if (char const * e = getenv("GLAS_FILL_BANDS"))
        ctx.fill_bands = std::max(1, atoi(e));
    ctx.carry = env_flag("GLAS_FILL_CARRY", false);
    if (char const * e = getenv("GLAS_FILL_OWNERS"))
        ctx.fill_owners = atoi(e);
    assert(ctx.fill_owners >= 1 && ctx.fill_owners <= 16 &&
           !(ctx.fill_owners & (ctx.fill_owners - 1)));
    assert(ctx.fill_owners == 1 || !ctx.carry);
    ctx.fill_owner_pairs = env_flag("GLAS_FILL_OWNER_PAIRS", true);
    if (char const * e = getenv("GLAS_FILL_PAIR_SLOTS"))
        ctx.fill_pair_slots = strtoul(e, nullptr, 10);
    ctx.fill_pair_fused = env_flag("GLAS_FILL_PAIR_FUSED", true);
    ctx.fill_stage = ctx.fill_owners > 1 || env_flag("GLAS_FILL_STAGE", false);
    ctx.fill_row_bands = !ctx.fill_pair_slots && !ctx.fill_stage &&
                        !ctx.carry && ctx.skip == 0 ? 32 : 0;
    if (char const * e = getenv("GLAS_FILL_ROW_BANDS"))
        ctx.fill_row_bands = strtoul(e, nullptr, 10);
    ctx.fill_row_group = env_flag("GLAS_FILL_ROW_GROUP", true);
    ctx.fill_row_fused = env_flag("GLAS_FILL_ROW_FUSED", true);
    if (char const * e = getenv("GLAS_FILL_ROW_WINDOW"))
        ctx.fill_row_window = strtoul(e, nullptr, 10);
    if (char const * e = getenv("GLAS_FILL_ROW_WINDOW_SIDE"))
        ctx.fill_row_window_side = atoi(e);
    ctx.fill_row_window_lean = env_flag("GLAS_FILL_ROW_WINDOW_LEAN", false);
    if (char const * e = getenv("GLAS_FILL_ROW_WINDOW_BLOCKS"))
        ctx.fill_row_window_blocks = strtoul(e, nullptr, 10);
    if (char const * e = getenv("GLAS_FILL_ROW_CLUSTER"))
        ctx.fill_row_cluster = strtoul(e, nullptr, 10);
    ctx.fill_ahead = ctx.fill_row_bands ? 1 : 0;
    if (char const * e = getenv("GLAS_FILL_AHEAD"))
        ctx.fill_ahead = strtoul(e, nullptr, 10);
    assert(ctx.fill_ahead <= 2);
    assert(ctx.fill_row_window_side >= -1 && ctx.fill_row_window_side <= 1);
    assert(!ctx.fill_ahead || ctx.fill_row_bands);
    assert(ctx.fill_row_window == 0 || ctx.fill_row_window == 8 ||
           ctx.fill_row_window == 16 || ctx.fill_row_window == 32);
    assert(ctx.fill_row_cluster == 1 || (ctx.fill_row_cluster == 2 && ctx.fill_row_window));
    assert(ctx.fill_row_window_blocks == 2 || ctx.fill_row_window_blocks == 6);
    assert((!ctx.fill_row_window_lean && ctx.fill_row_window_blocks == 2) ||
           (ctx.fill_row_window == 8 && ctx.fill_row_cluster == 1));
    assert(!ctx.fill_row_window || (ctx.fill_row_bands && ctx.fill_row_group));
    assert(ctx.fill_pair_slots == 0 || ctx.fill_pair_slots == 1024 ||
           ctx.fill_pair_slots == 4096);
    bool const local_fill = ctx.fill_pair_slots || ctx.fill_row_bands;
    assert(!(ctx.fill_pair_slots && ctx.fill_row_bands));
    assert(!local_fill || (!ctx.fill_stage && !ctx.carry && ctx.skip == 0));
    uint32_t local_bound = ctx.fill_row_bands ?
        std::max(4u, ctx.fill_bands) * I : 1u << 22;
    if (char const * e = getenv("GLAS_FILL_LOCAL_BOUND"))
        local_bound = strtoul(e, nullptr, 10);
    assert(!local_fill || local_bound >= ctx.fill_bands * I);
    uint32_t owner_bound = 1u << 20;
    if (char const * e = getenv("GLAS_FILL_OWNER_BOUND"))
        owner_bound = strtoul(e, nullptr, 10);
    assert(owner_bound > 0);
    ctx.bucket_slab = env_flag("GLAS_BUCKET_SLAB", true);
    ctx.norm_reg = env_flag("GLAS_NORM_REG", true);
    ctx.fill_gate = env_flag("GLAS_FILL_GATE", true);
    ctx.gate_shift = ctx.fill_gate ? 3 : 0;
    if (char const * e = getenv("GLAS_FILL_GATE_SHIFT"))
        ctx.gate_shift = strtoul(e, nullptr, 10);
    assert(ctx.gate_shift <= 5 && (ctx.fill_gate || !ctx.gate_shift));
    ctx.fill_tail0 = env_flag("GLAS_FILL_TAIL0",
                              ctx.fill_gate && ctx.fill_row_bands);
    assert(!ctx.fill_tail0 || (ctx.fill_gate && ctx.fill_row_bands));
    bool const cand_ok = logI >= SREG_LOG && ctx.norm_reg &&
                         ctx.skip == 0 && !ctx.fill_stage;
    ctx.region_cand = env_flag("GLAS_REGION_CAND", cand_ok);
    assert(!ctx.region_cand || cand_ok);
    /* the lane run must be a multiple of the four cells one packed
     * store covers, and must divide the warp run */
    if (char const * e = getenv("GLAS_NORM_STRIDE"))
        ctx.norm_stride = atoi(e);
    if (ctx.norm_stride < 4 || ctx.norm_stride > 32 ||
        (ctx.norm_stride & 3))
        ctx.norm_stride = 4;
    ctx.fill_coprime = env_flag("GLAS_FILL_COPRIME", true);
    for (int side = 0; side < 2; side++) {
        /* skip tier: entries with q below the bound are not sieved;
         * the survivor bound rises by a multiple of their expected
         * contribution instead.  The host tiny-prime division recovers
         * their factors, so the bound must stay within the tiny-prime
         * table (primes up to TINY_PRIME_BOUND). */
        uint32_t skip_bound = 96;
        if (char const * e = getenv("GLAS_SKIP_BOUND"))
            skip_bound = std::min((uint32_t)atoi(e), TINY_PRIME_BOUND);
        double comp_x = 1.0;
        if (char const * e = getenv("GLAS_SKIP_COMP"))
            comp_x = atof(e);

        /* skip, dense, sparse small, banded, local, then large entries */
        uint32_t const band_bound = ctx.fill_bands * I;
        auto tier = [I, skip_bound, band_bound, local_fill, local_bound](fb_entry const & e) {
            return e.q < skip_bound ? 0 : e.q < I / 128 ? 1 :
                   e.q < I ? 2 : e.q < band_bound ? 3 :
                   local_fill && e.q < local_bound ? 4 : 5;
        };
        /* thread t owns the range [lo(t), lo(t + 1)) of the factor base
         * and its own slots in each tier, so each tier holds its entries
         * in factor-base order.  The pass also sums the expected bucket
         * load: one hit per position per 1/q density.  GLAS_GPU_STATS
         * prints the measured peaks; overflow falls back to the CPU. */
        size_t const n = fb[side].size();
        auto lo = [n, threads](unsigned t) { return n * t / threads; };
        std::vector<std::array<size_t, 6>> part(threads);
        std::vector<double> part_dens(threads);
        run_threads(threads, [&](unsigned t) {
            for (size_t k = lo(t); k < lo(t + 1); k++) {
                fb_entry const & e = fb[side][k];
                part[t][tier(e)]++;
                if (e.q >= I)
                    part_dens[t] += 1.0 / e.q;
            }
        });
        size_t cnt[6] = {};
        std::vector<std::array<size_t, 6>> off(threads);
        for (size_t i = 0, pos = 0; i < 6; i++)
            for (unsigned t = 0; t < threads; t++) {
                off[t][i] = pos;
                pos += part[t][i];
                cnt[i] += part[t][i];
            }
        std::vector<gpu_fb_entry> & h = ctx.h_fb[side];
        h.resize(n);
        run_threads(threads, [&](unsigned t) {
            for (size_t k = lo(t); k < lo(t + 1); k++) {
                fb_entry const & e = fb[side][k];
                h[off[t][tier(e)]++] = {e.q, e.r, e.weight, e.resieve};
            }
        });
        ctx.n_fb[side] = h.size();
        ctx.n_skip[side] = cnt[0];
        ctx.n_dense[side] = cnt[0] + cnt[1];
        ctx.n_small[side] = cnt[0] + cnt[1] + cnt[2];
        ctx.n_band[side] = cnt[0] + cnt[1] + cnt[2] + cnt[3];
        ctx.n_owned[side] = ctx.n_small[side];
        ctx.n_local[side] = local_fill ? ctx.n_band[side] + cnt[4] : ctx.n_small[side];
        if (ctx.fill_owners > 1)
            ctx.n_owned[side] = std::stable_partition(
                h.begin() + ctx.n_band[side], h.end(),
                [owner_bound](gpu_fb_entry const & e) {
                    return e.q < owner_bound;
                }) - h.begin();

        /* dense tier: the pattern entries first, so the pattern build
         * scans only them */
        auto pat_q = [](gpu_fb_entry const & e) {
            return e.q <= 16 && PAT_LEN % e.q == 0;
        };
        ctx.n_pat[side] = std::stable_partition(h.begin() + ctx.n_skip[side],
                                                h.begin() + ctx.n_dense[side],
                                                pat_q) - h.begin();

        double comp = 0.0;
        for (uint32_t k = 0; k < cnt[0]; k++)
            comp += (double)h[k].weight / h[k].q;
        ctx.skip_comp[side] = (int32_t)std::lround(comp_x * comp);

        std::vector<gpu_small_entry> & m = ctx.h_small[side];
        m.resize(ctx.n_small[side]);
        for (uint32_t k = 0; k < ctx.n_small[side]; k++)
            m[k] = {h[k].q, h[k].weight, UINT64_MAX / h[k].q};

        double dens = 0.0;
        for (double d : part_dens)
            dens += d;
        ctx.bucket_cap[side] = ((uint32_t)(SREG * dens * 1.25) & ~31u) + 512;
        /* the slab layout fills whole 64-entry pieces */
        if (ctx.bucket_slab)
            ctx.bucket_cap[side] = (ctx.bucket_cap[side] + 63) & ~63u;
    }
    if (getenv("GLAS_GPU_STATS"))
        fprintf(stderr, "# GPU init tables (%u threads): %.2f s\n", threads,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

bool gpu_init(int logI, uint32_t max_J, tiny_table const & tt,
              filter_params const & fp)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
        return false;
    assert(logI >= 12 && logI <= 17);

    ctx.stats = getenv("GLAS_GPU_STATS") != nullptr;
    if (char const * e = getenv("GLAS_BUCKET_DUMP")) {
        ctx.bucket_dump = fopen(e, "w");
        assert(ctx.bucket_dump);
        ctx.stats = true;
    }
    ctx.filter = env_flag("GLAS_GPU_FILTER", true);
    ctx.assemble = ctx.filter && env_flag("GLAS_GPU_ASSEMBLE", true);
    /* region hits: the resieve emit files each hit under its region and
     * the filter reads the survivor's own region list, so the out
     * stream skips the radix sort of the flat hit list */
    ctx.rhits = ctx.filter && env_flag("GLAS_RHITS", true);
    auto t_init = std::chrono::steady_clock::now();
    auto lap = [&](char const * what) {
        if (!ctx.stats)
            return;
        auto const now = std::chrono::steady_clock::now();
        fprintf(stderr, "# GPU init %s: %.2f s\n", what,
                std::chrono::duration<double>(now - t_init).count());
        t_init = now;
    };

    uint32_t const np = tt.primes.size();
    assert(np <= MAX_TINY);
    std::vector<uint64_t> tm(np);
    for (uint32_t k = 0; k < np; k++)
        tm[k] = UINT64_MAX / tt.primes[k];
    CUDA_CHECK(cudaMemcpyToSymbol(c_tiny_np, &np, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tiny_p, tt.primes.data(), np * 4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tiny_M, tm.data(), np * 8));
    for (int side = 0; side < 2; side++)
        CUDA_CHECK(cudaMemcpyToSymbol(c_tiny_c, tt.cmod[side].data(),
                                      tt.cmod[side].size() * 4,
                                      side * sizeof(c_tiny_c[0])));

    {
        uint64_t fc[2][GLAS_MAX_DEGREE + 1][NORM_LIMBS];
        uint32_t fneg[2][GLAS_MAX_DEGREE + 1];
        for (int side = 0; side < 2; side++)
            for (int k = 0; k <= GLAS_MAX_DEGREE; k++) {
                memcpy(fc[side][k], fp.coeff[side][k].limb,
                       sizeof fc[side][k]);
                fneg[side][k] = fp.coeff[side][k].neg;
            }
        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_c, fc, sizeof fc));
        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_neg, fneg, sizeof fneg));
        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_lim, fp.lim, sizeof fp.lim));
        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_lpb, fp.lpb, sizeof fp.lpb));
        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_mfb, fp.mfb, sizeof fp.mfb));
        ctx.host_mr = env_flag("GLAS_FLT_HOST_MR", false);
        uint32_t const defer =
            ctx.host_mr || env_flag("GLAS_FLT_DEFER_MR", true);
        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_defer_mr, &defer, sizeof defer));
    }

    uint32_t const I = 1u << logI;
    for (int side = 0; side < 2; side++) {
        std::vector<gpu_fb_entry> const h = std::move(ctx.h_fb[side]);
        CUDA_CHECK(cudaMalloc(&ctx.fb[side], h.size() * sizeof(gpu_fb_entry)));
        CUDA_CHECK(cudaMemcpy(ctx.fb[side], h.data(),
                              h.size() * sizeof(gpu_fb_entry),
                              cudaMemcpyHostToDevice));

        std::vector<gpu_small_entry> const m = std::move(ctx.h_small[side]);
        CUDA_CHECK(cudaMalloc(&ctx.small[side],
                              m.size() * sizeof(gpu_small_entry)));
        CUDA_CHECK(cudaMemcpy(ctx.small[side], m.data(),
                              m.size() * sizeof(gpu_small_entry),
                              cudaMemcpyHostToDevice));
    }
    for (int side = 0; ctx.carry && side < 2; side++) {
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceSelect::If(
            nullptr, bytes, thrust::counting_iterator<uint32_t>(0),
            (uint32_t *)nullptr, (uint32_t *)nullptr,
            ctx.n_fb[side] - ctx.n_band[side], carry_due{nullptr, 0}));
        ctx.select_temp_bytes = std::max(ctx.select_temp_bytes, bytes);
    }
    lap("fb");

    /* chunk: the largest power of two up to min(area, 2^31) that leaves
     * room for several concurrent slots.  The walk-cursor arrays cost
     * the same at every chunk size, so a halving that does not add a
     * slot only wastes chunk size; the shrink loop stops there.
     * A larger area is sieved in
     * several chunk passes; the walk cursors make the extra passes
     * cheap, and the slots keep the device busy while the host
     * processes survivors. */
    auto slot_bytes = [](uint64_t area) {
        uint64_t b = area / 8 + 2 * (area / 64 * (4 + 16) + area / 8 * 8);
        if (ctx.filter && !ctx.rhits) {
            b += area / 8 * 8;
            size_t st = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, st,
                                           (uint64_t const *)nullptr,
                                           (uint64_t *)nullptr, area / 8);
            b += st;
        }
        if (ctx.rhits)
            /* a region's hit list and count, plus the overflow tail,
             * per parity */
            b += 2 * ((area >> SREG_LOG) * (RHITS_CAP * 8 + 4) +
                      RHITS_OV * 8);
        b += ctx.select_temp_bytes;
        if (ctx.gate_shift)
            b += area / 8 >> ctx.gate_shift;
        for (int side = 0; side < 2; side++) {
            /* roots for all entries; walk_x + lattice for the walked
             * entries; carry record, next-chunk byte and active-list
             * slot for the carried entries */
            uint64_t const n_large = ctx.n_fb[side] - ctx.n_band[side];
            uint64_t const n_walk = ctx.carry ? ctx.n_band[side] : ctx.n_fb[side];
            b += (uint64_t)ctx.n_fb[side] * 4 + n_walk * 24 +
                 (ctx.carry ? n_large * (sizeof(carry_state) + 5) : 0) +
                 (area >> SREG_LOG) * (side == 1 && ctx.fill_ahead ? ctx.fill_ahead : 1) *
                     (ctx.bucket_cap[side] * 8 + BK_CNT_STRIDE * 4);
        }
        return b;
    };
    size_t mem_free = 0, mem_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
    uint64_t const total_area = (uint64_t)max_J << logI;
    uint64_t const chunk_min = std::max<uint64_t>(SREG, I);
    uint64_t chunk = chunk_min;
    while (chunk < total_area && chunk < (1ull << 31))
        chunk <<= 1;
    uint64_t const chunk_max = chunk;
    uint64_t const budget = mem_free * 8 / 10;
    while (chunk > chunk_min && slot_bytes(chunk) * 6 > budget &&
           (slot_bytes(chunk) > budget ||
            budget / slot_bytes(chunk >> 1) > budget / slot_bytes(chunk)))
        chunk >>= 1;
    /* GLAS_GPU_CHUNK=K sets the chunk to 2^K within [chunk_min, chunk_max];
     * the slot count then follows from the memory budget. */
    if (char const * e = getenv("GLAS_GPU_CHUNK"))
        chunk = std::clamp((uint64_t)1 << atoi(e), chunk_min, chunk_max);
    if (ctx.fill_ahead > 1)
        chunk = std::min(chunk, (uint64_t)1 << 30);
    ctx.chunk_area = chunk;
    ctx.max_regions = chunk >> SREG_LOG;
    uint32_t const slab = ctx.bucket_slab ? ctx.max_regions : 0;
    CUDA_CHECK(cudaMemcpyToSymbol(c_bucket_slab, &slab, 4));
    uint32_t const norm_reg = ctx.norm_reg;
    CUDA_CHECK(cudaMemcpyToSymbol(c_norm_reg, &norm_reg, 4));
    CUDA_CHECK(
        cudaMemcpyToSymbol(c_norm_stride, &ctx.norm_stride, 4));
    uint32_t const fill_coprime = ctx.fill_coprime;
    CUDA_CHECK(cudaMemcpyToSymbol(c_fill_coprime, &fill_coprime, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gate_shift, &ctx.gate_shift, 4));
    ctx.max_survivors = chunk / 64;
    ctx.max_hits = chunk / 8;
    if (ctx.assemble)
        ctx.fcand_cap = std::min(ctx.max_survivors, FCAND_MAX);
    if (ctx.filter) {
        if (!ctx.rhits)
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                nullptr, ctx.sort_temp_bytes, (uint64_t const *)nullptr,
                (uint64_t *)nullptr, ctx.max_hits));
        int uva = 0;
        /* the filter kernel writes through the host pointers of the
         * mapped pinned buffers, which needs unified addressing */
        CUDA_CHECK(cudaDeviceGetAttribute(
            &uva, cudaDevAttrUnifiedAddressing, 0));
        assert(uva);
        ctx.filter_grid = full_grid(filter_kernel, 256, 0);
        if (char const * e = getenv("GLAS_FILTER_GRID"))
            ctx.filter_grid = atoi(e);
    }
    ctx.tiny_grid = full_grid(tiny_mask_kernel, 256, 0);
    ctx.rs_grid = full_grid(resieve_small_kernel, 256, 0);
    if (ctx.carry) {
        ctx.carry_grid = full_grid(fill_carry_kernel, 256, 0);
        assert(total_area / chunk < CARRY_DONE);
    }
    if (ctx.fill_stage) {
        /* the fill probes are fill_kernel paths */
        assert(!(ctx.skip & (8 | 64)));
        /* the 16-byte pair store needs buckets that start 16-byte aligned */
        assert(ctx.bucket_cap[0] % 2 == 0 && ctx.bucket_cap[1] % 2 == 0);
        /* the slot table leaves room for one region block on the SM */
        int smem = 0, reserved = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &reserved, cudaDevAttrReservedSharedMemoryPerBlock, 0));
        uint32_t const unit = 1u << std::max(logI - SREG_LOG, 0);
        uint32_t const owner_regions =
            (ctx.max_regions / unit + ctx.fill_owners - 1) / ctx.fill_owners * unit;
        ctx.fill_stage_regions = std::min<uint32_t>(
            owner_regions, (smem - REGION_SMEM - 2 * reserved) / 8);
        size_t const bytes = (size_t)ctx.fill_stage_regions * 8;
        CUDA_CHECK(cudaFuncSetAttribute(
            fill_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)bytes));
        /* both kernels ask for the largest shared memory carveout, so
         * one fill block and one region block share an SM */
        CUDA_CHECK(cudaFuncSetAttribute(
            fill_stage_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        CUDA_CHECK(cudaFuncSetAttribute(
            region_kernel<0>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        ctx.fill_stage_grid =
            full_grid(fill_stage_kernel, FILL_STAGE_THREADS, bytes);
        ctx.fill_stage_grid =
            ctx.fill_stage_grid / ctx.fill_owners * ctx.fill_owners;
        assert(ctx.fill_stage_grid > 0);
    }

    ctx.n_slots = (int)std::min<uint64_t>(MAX_SLOTS,
                                          budget / slot_bytes(chunk));
    if (char const * e = getenv("GLAS_GPU_SLOTS"))
        ctx.n_slots = std::min(ctx.n_slots, atoi(e));
    if (char const * e = getenv("GLAS_XFORM_REPEAT"))
        ctx.xform_repeat = std::max(1, atoi(e));
    ctx.pipeline = env_flag("GLAS_PIPELINE", true);
    if (ctx.n_slots < 1) {
        fprintf(stderr, "GPU slot needs %lu bytes, budget is %lu\n",
                slot_bytes(chunk), budget);
        abort();
    }

    /* The sieve streams take the highest priority the GPU offers, so
     * the GPU runs a ready sieve kernel before a cofactorization
     * kernel.  A cofactorization stream at default priority competes
     * with the sieve for the same SMs and delays the survivors that
     * the next sieve kernel needs. */
    int prio_least, prio_greatest;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_least,
                                                &prio_greatest));
    for (int k = 0; k < ctx.n_slots; k++) {
        gpu_slot & s = ctx.slot[k];
        CUDA_CHECK(cudaStreamCreateWithPriority(&s.stream,
                                                cudaStreamNonBlocking,
                                                prio_greatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s.out_stream,
                                                cudaStreamNonBlocking,
                                                prio_greatest));
        for (int side = 0; side < 2; side++) {
            uint32_t const n_large = ctx.n_fb[side] - ctx.n_band[side];
            uint32_t const n_walk = ctx.carry ? ctx.n_band[side] : ctx.n_fb[side];
            CUDA_CHECK(cudaMalloc(&s.roots[side], ctx.n_fb[side] * 4));
            CUDA_CHECK(cudaMalloc(&s.walk_x[side], n_walk * 8));
            CUDA_CHECK(cudaMalloc(&s.lattice[side], n_walk * 16));
            if (ctx.carry) {
                CUDA_CHECK(cudaMalloc(&s.carry[side],
                                      n_large * sizeof(carry_state)));
                CUDA_CHECK(cudaMalloc(&s.nextc[side], n_large));
                CUDA_CHECK(cudaMalloc(&s.active[side], n_large * 4));
            }
            uint32_t const regions = ctx.max_regions *
                (side == 1 && ctx.fill_ahead ? ctx.fill_ahead : 1);
            CUDA_CHECK(cudaMalloc(&s.buckets[side], (uint64_t)regions *
                                                        ctx.bucket_cap[side] * 8));
            CUDA_CHECK(cudaMalloc(&s.bucket_cnt[side],
                                  regions * BK_CNT_STRIDE * 4));
            CUDA_CHECK(cudaMalloc(&s.rs_idx[side], ctx.n_small[side] * 4));
        }
        CUDA_CHECK(cudaMalloc(&s.rs_cnt, 2 * 4));
        if (ctx.carry) {
            CUDA_CHECK(cudaMalloc(&s.n_active, 2 * 4));
            CUDA_CHECK(cudaMalloc(&s.select_temp, ctx.select_temp_bytes));
        }
        CUDA_CHECK(cudaMalloc(&s.bitmask, ctx.chunk_area / 8));
        if (ctx.gate_shift)
            CUDA_CHECK(cudaMalloc(&s.cgate, ctx.chunk_area / 8 >> ctx.gate_shift));
        for (int par = 0; par < 2; par++) {
            CUDA_CHECK(cudaMalloc(&s.survivors[par], ctx.max_survivors * 4));
            CUDA_CHECK(cudaMalloc(&s.tiny[par], ctx.max_survivors * 16));
            CUDA_CHECK(cudaMalloc(&s.hits[par], ctx.max_hits * 8));
            CUDA_CHECK(cudaMalloc(&s.counters[par], 4 * 4));
            if (ctx.rhits) {
                CUDA_CHECK(cudaMalloc(&s.hits_r[par],
                                      (uint64_t)ctx.max_regions * RHITS_CAP * 8));
                CUDA_CHECK(cudaMalloc(&s.hit_rcnt[par], ctx.max_regions * 4));
                CUDA_CHECK(cudaMalloc(&s.hits_ov[par], RHITS_OV * 8));
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&s.kdone[par],
                                                cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.odone[par],
                                                cudaEventDisableTiming));
        }
        if (ctx.filter) {
            if (!ctx.rhits) {
                CUDA_CHECK(cudaMalloc(&s.hits_sorted, ctx.max_hits * 8));
                CUDA_CHECK(cudaMalloc(&s.sort_temp, ctx.sort_temp_bytes));
            }
            CUDA_CHECK(cudaMalloc(&s.fcnt, 2 * 4));
        }
    }
    lap("malloc");
    {
        std::vector<std::thread> pin;
        for (int k = 0; k < ctx.n_slots; k++)
            pin.emplace_back([k] {
                gpu_slot & s = ctx.slot[k];
                unsigned const mapped = cudaHostAllocMapped;
                CUDA_CHECK(cudaMallocHost(&s.h_counters, 5 * 4));
                CUDA_CHECK(cudaHostAlloc(&s.h_survivors,
                                         ctx.max_survivors * 4, mapped));
                CUDA_CHECK(cudaHostAlloc(&s.h_tiny, ctx.max_survivors * 16,
                                         mapped));
                CUDA_CHECK(cudaHostAlloc(&s.h_hits, ctx.max_hits * 8,
                                         mapped));
                if (ctx.stats)
                    CUDA_CHECK(cudaMallocHost(
                        &s.h_bucket,
                        4 * ctx.max_regions * BK_CNT_STRIDE * 4));
                if (ctx.fcand_cap) {
                    CUDA_CHECK(cudaHostAlloc(&s.h_fmeta, ctx.fcand_cap * 4,
                                             mapped));
                    CUDA_CHECK(cudaHostAlloc(
                        &s.h_ffac,
                        (uint64_t)ctx.fcand_cap * GPU_FAC_CAP * 4, mapped));
                    CUDA_CHECK(cudaHostAlloc(&s.h_flo, ctx.fcand_cap * 32,
                                             mapped));
                }
            });
        for (std::thread & t : pin)
            t.join();
    }
    lap("pinned");
    CUDA_CHECK(cudaFuncSetAttribute(
        region_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        REGION_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(
        region_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        REGION_SMEM));
    ctx.free_mask = (1u << ctx.n_slots) - 1;

    if (ctx.stats) {
        fprintf(stderr,
                "# GPU slots: %d, chunk 2^%d, %.2f GB per slot, fill stage "
                "%d x %u blocks, %u of %u regions staged, bucket slab %d, "
                "gate shift %u, side-0 tail kernel %d, side-0 candidate "
                "region %d\n",
                ctx.n_slots, 63 - __builtin_clzll(ctx.chunk_area),
                slot_bytes(chunk) / 1073741824.0, ctx.fill_stage,
                ctx.fill_stage_grid, ctx.fill_stage_regions, ctx.max_regions,
                ctx.bucket_slab, ctx.gate_shift, ctx.fill_tail0,
                ctx.region_cand);
        for (int k = 0; k < ctx.n_slots; k++) {
            for (int par = 0; par < 2; par++)
                for (int e = 0; e < N_EVENTS; e++)
                    CUDA_CHECK(cudaEventCreate(&ctx.slot[k].ev[par][e]));
            for (int e = 0; e < 2; e++)
                CUDA_CHECK(cudaEventCreate(&ctx.slot[k].fev[e]));
        }
        CUDA_CHECK(cudaEventCreate(&ctx.ref));
        CUDA_CHECK(cudaEventRecord(ctx.ref));
        CUDA_CHECK(cudaEventSynchronize(ctx.ref));
        trace_t0 = std::chrono::steady_clock::now();
        trace_on = true;
        atexit([] {
            for (int p = 0; p < N_PHASES; p++)
                fprintf(stderr, "# GPU %s: %.2f s\n", PHASE_NAMES[p],
                        ctx.phase_ms[p] / 1e3);
            if (ctx.filter)
                fprintf(stderr, "# GPU filter: %.2f s\n",
                        ctx.filter_ms / 1e3);
            fprintf(stderr, "# GPU ecm: %.2f s\n", ctx.ecm_ms / 1e3);
            for (int side = 0; side < 2; side++)
                fprintf(stderr, "# GPU bucket max %d: %u of %u, entries %lu\n",
                        side, ctx.bucket_max[side], ctx.bucket_cap[side],
                        (unsigned long)ctx.bucket_sum[side]);
            report_idle();
            if (ctx.bucket_dump)
                fclose(ctx.bucket_dump);
        });
    }
    return true;
}

int gpu_slots()
{
    return ctx.n_slots;
}

template <bool CAND>
static void launch_region(gpu_task const & t, gpu_slot const & s,
                          uint64_t * const * buckets,
                          uint32_t * const * bucket_cnt, int par, int side)
{
    region_kernel<CAND><<<t.nregions, 1024, REGION_SMEM, s.stream>>>(
        t, ctx.small[side], ctx.n_small[side], ctx.n_dense[side],
        ctx.n_skip[side], ctx.n_pat[side],
        s.roots[side], ctx.bucket_cap[side],
        buckets[side], bucket_cnt[side], side, side == 1,
        s.bitmask, ctx.max_survivors,
        s.survivors[par], s.counters[par],
        ctx.bucket_cap[side ^ 1], buckets[side ^ 1],
        bucket_cnt[side ^ 1], ctx.max_hits, s.hits[par],
        s.counters[par] + 1,
        ctx.rhits ? s.hits_r[par] : nullptr,
        ctx.rhits ? s.hit_rcnt[par] : nullptr,
        ctx.rhits ? s.hits_ov[par] : nullptr);
}

void sieve_one_gpu(sq_task const & task, std::vector<fb_entry> const * fb,
                   tiny_table const & tt, chunk_sink const & sink)
{
    gpu_task t = {};
    t.q = task.q;
    t.rho = task.rho;
    t.sq_side = task.sq_side;
    t.a0 = task.a0;
    t.b0 = task.b0;
    t.a1 = task.a1;
    t.b1 = task.b1;
    t.logI = task.logI;
    t.J = task.J;
    for (int side = 0; side < 2; side++) {
        t.degree[side] = task.degree[side];
        for (int k = 0; k <= task.degree[side]; k++)
            t.fijd[side][k] = task.fijd[side][k];
        t.bound[side] = task.bound[side] + ctx.skip_comp[side];
    }
    t.skip = ctx.skip;

    trace_scope const tr_feed(TR_FEED);
    uint32_t const chunk_rows = (uint32_t)(ctx.chunk_area >> task.logI);
    int const slot = acquire_slot();
    gpu_slot & s = ctx.slot[slot];
    cudaStream_t const st = s.stream;

    /* Enqueues one chunk's kernels on the sieve stream.  The parity
     * picks the output buffer set; the sieve stream first waits until
     * the out stream has copied the previous chunk of the same parity
     * out. */
    gpu_task tpar[2];
    auto launch = [&](uint32_t j0, int par) {
        trace_scope const tr(TR_LAUNCH);
        t.j0 = j0;
        t.rows = std::min(chunk_rows, task.J - j0);
        t.area = t.rows << task.logI;
        t.nregions = t.area >> SREG_LOG;
        assert(t.nregions >= 1 && t.area % SREG == 0);

        uint32_t const bank = ctx.fill_ahead ? (j0 / chunk_rows) % ctx.fill_ahead : 0;
        uint64_t * buckets[2] = {s.buckets[0],
            s.buckets[1] + (uint64_t)bank * ctx.max_regions * ctx.bucket_cap[1]};
        uint32_t * bucket_cnt[2] = {s.bucket_cnt[0],
            s.bucket_cnt[1] + bank * ctx.max_regions * BK_CNT_STRIDE};
        CUDA_CHECK(cudaStreamWaitEvent(st, s.odone[par]));
        CUDA_CHECK(cudaMemsetAsync(s.counters[par], 0, 4 * 4, st));
        for (int side = 0; side < 2; side++) {
            if (side == 1 && ctx.fill_ahead && bank)
                continue;
            uint32_t const regions = side == 1 && ctx.fill_ahead ?
                ctx.fill_ahead * ctx.max_regions : t.nregions;
            CUDA_CHECK(cudaMemsetAsync(s.bucket_cnt[side], 0,
                                       regions * BK_CNT_STRIDE * 4, st));
        }
        if (ctx.rhits)
            CUDA_CHECK(cudaMemsetAsync(s.hit_rcnt[par], 0, t.nregions * 4,
                                       st));
        int nev = 0;
        auto mark = [&](int phase) {
            if (!ctx.stats)
                return;
            s.ev_phase[par][nev] = phase;
            CUDA_CHECK(cudaEventRecord(s.ev[par][nev++], st));
        };
        mark(PH_TRANSFORM);
        for (int side = 0; j0 == 0 && side < 2; side++) {
            /* roots do not depend on the chunk */
            uint32_t const n = ctx.n_fb[side];
            for (int rep = 0; rep < ctx.xform_repeat; rep++)
                transform_kernel<<<(n + 255) / 256, 256, 0, st>>>(
                    t, ctx.fb[side], n, s.roots[side], s.walk_x[side],
                    s.lattice[side], side, ctx.n_band[side], s.carry[side],
                    s.nextc[side]);
            if (side == 0)
                CUDA_CHECK(cudaMemsetAsync(s.rs_cnt, 0, 2 * 4, st));
            resieve_compact_kernel<<<(ctx.n_small[side] + 255) / 256, 256, 0,
                                     st>>>(ctx.fb[side], ctx.n_small[side],
                                           s.roots[side], s.rs_idx[side],
                                           s.rs_cnt + side);
        }
        mark(PH_TRANSFORM);
        auto fill = [&](int side, uint32_t const * gate) {
            /* with carry, fill_kernel walks the banded entries only */
            bool const ahead = ctx.fill_ahead && side == 1;
            bool const tail0 = ctx.fill_tail0 && side == 0;
            uint32_t const n = ctx.carry ? ctx.n_band[side] :
                ahead || tail0 ? ctx.n_local[side] : ctx.n_fb[side];
            if (ahead && !bank && n < ctx.n_fb[side]) {
                gpu_task next = t;
                next.rows = std::min(ctx.fill_ahead * chunk_rows, task.J - j0);
                next.area = next.rows << task.logI;
                next.nregions = next.area >> SREG_LOG;
                fill_ahead_kernel<false><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                     FILL_THREADS, FILL_THREADS, 0, st>>>(
                    next, ctx.fb[side], n, ctx.n_fb[side], s.roots[side],
                    s.walk_x[side], s.lattice[side], ctx.max_regions,
                    ctx.bucket_cap[side], s.buckets[side], s.bucket_cnt[side],
                    s.counters[par] + 2, nullptr);
            }
            if (tail0 && n < ctx.n_fb[side])
                fill_ahead_kernel<true><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                    FILL_THREADS, FILL_THREADS, 0, st>>>(
                    t, ctx.fb[side], n, ctx.n_fb[side], s.roots[side],
                    s.walk_x[side], s.lattice[side], ctx.max_regions,
                    ctx.bucket_cap[side], buckets[side], bucket_cnt[side],
                    s.counters[par] + 2, gate);
            bool const owned = ctx.fill_owners > 1;
            bool const local = ctx.fill_pair_slots || ctx.fill_row_bands;
            bool const fused = ((ctx.fill_pair_slots && ctx.fill_pair_fused) ||
                                (ctx.fill_row_bands && ctx.fill_row_fused)) &&
                                ctx.n_local[side] > ctx.n_small[side];
            uint32_t const n_small = owned ? ctx.n_owned[side] :
                local ? ctx.n_local[side] : ctx.n_small[side];
            uint32_t const n_band = owned || local ? n_small : ctx.n_band[side];
            uint32_t const n_thr =
                n - n_small + (n_band - n_small) *
                                            (ctx.fill_bands - 1);
            uint32_t const unit = 1u << std::max((int)t.logI - SREG_LOG, 0);
            uint32_t const owner_regions =
                (t.nregions / unit + ctx.fill_owners - 1) / ctx.fill_owners * unit;
            uint32_t const n_staged =
                std::min(owner_regions, ctx.fill_stage_regions);
            if (local && ctx.n_local[side] > ctx.n_small[side]) {
                uint32_t const count = ctx.n_local[side] - ctx.n_small[side];
                if (ctx.fill_pair_slots) {
                    uint32_t const total = ctx.fill_pair_fused ? n : ctx.n_local[side];
                    uint32_t const threads = total - ctx.n_small[side] +
                        (ctx.n_band[side] - ctx.n_small[side]) * (ctx.fill_bands - 1);
                    auto launch = [&](auto kernel) {
                        kernel<<<(threads + FILL_THREADS - 1) / FILL_THREADS,
                                 FILL_THREADS, 0, st>>>(
                            t, ctx.fb[side], ctx.n_small[side], ctx.n_band[side],
                            ctx.fill_bands, total, ctx.n_local[side], s.roots[side],
                            s.walk_x[side], s.lattice[side], ctx.bucket_cap[side],
                            buckets[side], bucket_cnt[side], s.counters[par] + 2,
                            gate);
                    };
                    if (ctx.fill_pair_slots == 1024)
                        launch(fill_local_pair_kernel<10>);
                    else
                        launch(fill_local_pair_kernel<12>);
                } else {
                    uint32_t const bands = std::min(ctx.fill_row_bands, t.rows);
                    uint32_t const total = ctx.fill_row_fused ? n : ctx.n_local[side];
                    uint32_t const window = ctx.fill_row_window_side < 0 ||
                        ctx.fill_row_window_side == side ? ctx.fill_row_window : 0;
                    uint32_t const width = window ?
                        FILL_THREADS * ctx.fill_row_cluster : 32;
                    uint32_t const row_threads =
                        (((count + width - 1) / width * width * bands + FILL_THREADS - 1) /
                         FILL_THREADS) * FILL_THREADS;
                    uint32_t const threads = row_threads + total - ctx.n_local[side];
                    auto launch = [&](auto kernel) {
                        kernel<<<(threads + FILL_THREADS - 1) / FILL_THREADS,
                                 FILL_THREADS, 0, st>>>(
                            t, ctx.fb[side], ctx.n_small[side], ctx.n_local[side], total,
                            bands, s.roots[side], s.walk_x[side], s.lattice[side], ctx.bucket_cap[side],
                            buckets[side], bucket_cnt[side], s.counters[par] + 2,
                            gate);
                    };
                    if (window) {
                        uint32_t const c = ctx.fill_row_cluster;
                        cudaLaunchAttribute attr{};
                        attr.id = cudaLaunchAttributeClusterDimension;
                        attr.val.clusterDim = {c, 1, 1};
                        cudaLaunchConfig_t config{};
                        config.gridDim = {(threads + FILL_THREADS * c - 1) / (FILL_THREADS * c) * c, 1, 1};
                        config.blockDim = {FILL_THREADS, 1, 1};
                        config.dynamicSmemBytes = 2 * window * unit * sizeof(uint32_t);
                        config.stream = st;
                        config.attrs = &attr;
                        config.numAttrs = c > 1 ? 1 : 0;
                        auto reserve = [&](auto kernel) {
                            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, t, ctx.fb[side],
                                ctx.n_small[side], ctx.n_local[side], total, bands, s.roots[side],
                                s.walk_x[side], s.lattice[side], ctx.bucket_cap[side], buckets[side],
                                bucket_cnt[side], s.counters[par] + 2, gate));
                        };
                        bool const lean = ctx.fill_row_window_lean;
                        bool const b6 = ctx.fill_row_window_blocks == 6;
                        if (window == 8) {
                            if (c == 2) reserve(fill_row_kernel<true, 8, 2>);
                            else if (lean && b6) reserve(fill_row_kernel<true, 8, 1, true, 6>);
                            else if (lean) reserve(fill_row_kernel<true, 8, 1, true>);
                            else if (b6) reserve(fill_row_kernel<true, 8, 1, false, 6>);
                            else reserve(fill_row_kernel<true, 8, 1>);
                        } else if (window == 16) {
                            if (c == 1) reserve(fill_row_kernel<true, 16, 1>);
                            else reserve(fill_row_kernel<true, 16, 2>);
                        } else {
                            if (c == 1) reserve(fill_row_kernel<true, 32, 1>);
                            else reserve(fill_row_kernel<true, 32, 2>);
                        }
                    } else if (ctx.fill_row_group)
                        launch(fill_row_kernel<true>);
                    else
                        launch(fill_row_kernel<false>);
                }
            }
            if (ctx.fill_stage && (n_thr || owned))
                fill_stage_kernel<<<ctx.fill_stage_grid, FILL_STAGE_THREADS,
                                    (size_t)n_staged * 8, st>>>(
                    t, ctx.fb[side], ctx.n_small[side],
                    owned ? ctx.n_small[side] : ctx.n_band[side],
                    owned ? 1 : ctx.fill_bands, owned ? ctx.n_owned[side] : n,
                    s.roots[side], s.walk_x[side], s.lattice[side],
                    owned && !ctx.fill_owner_pairs ? 0 : n_staged,
                    ctx.bucket_cap[side], buckets[side],
                    bucket_cnt[side], s.counters[par] + 2, ctx.fill_owners,
                    gate);
            if (n_thr && (!ctx.fill_stage || owned) &&
                !fused)
                fill_kernel<<<(n_thr + FILL_THREADS - 1) / FILL_THREADS,
                              FILL_THREADS, 0, st>>>(
                    t, ctx.fb[side], n_small, n_band,
                    ctx.fill_bands, n,
                    s.roots[side], s.walk_x[side], s.lattice[side],
                    ctx.bucket_cap[side], buckets[side],
                    bucket_cnt[side], s.counters[par] + 2, gate);
            uint32_t const n_large = ctx.n_fb[side] - ctx.n_band[side];
            if (!ctx.carry || !n_large)
                return;
            uint32_t const chunk_log = __builtin_ctzll(ctx.chunk_area);
            uint32_t const cur = (uint32_t)(((uint64_t)j0 << t.logI) >> chunk_log);
            size_t tmp_bytes = ctx.select_temp_bytes;
            CUDA_CHECK(cub::DeviceSelect::If(
                s.select_temp, tmp_bytes, thrust::counting_iterator<uint32_t>(0),
                s.active[side], s.n_active + side, n_large,
                carry_due{s.nextc[side], cur}, st));
            fill_carry_kernel<<<ctx.carry_grid, 256, 0, st>>>(
                t, chunk_log, ctx.fb[side], s.roots[side], ctx.n_band[side],
                s.active[side], s.n_active + side, s.carry[side],
                s.nextc[side], ctx.bucket_cap[side], buckets[side],
                bucket_cnt[side], s.counters[par] + 2);
        };
        /* side 1 fills the bitmask, side 0 intersects, emits survivors
         * and rescans both bucket lists for the resieve hits */
        auto region = [&](int side) {
            if (side == 0 && ctx.region_cand)
                launch_region<1>(t, s, buckets, bucket_cnt, par, side);
            else
                launch_region<0>(t, s, buckets, bucket_cnt, par, side);
        };
        if (ctx.fill_gate) {
            /* GLAS_FILL_GATE: the side-0 fill runs after the side-1
             * region kernel and stores only the hits under its mask */
            fill(1, nullptr);
            mark(PH_FILL);
            region(1);
            mark(PH_REGION);
            if (ctx.gate_shift) {
                uint32_t const words = t.area >> (5 + ctx.gate_shift);
                gate_fold_kernel<<<(words + 255) / 256, 256, 0, st>>>(
                    s.bitmask, s.cgate, ctx.gate_shift, words);
            }
            fill(0, ctx.gate_shift ? s.cgate : s.bitmask);
            mark(PH_FILL);
            region(0);
            mark(PH_REGION);
        } else {
            fill(0, nullptr);
            fill(1, nullptr);
            mark(PH_FILL);
            region(1);
            region(0);
            mark(PH_REGION);
        }
        /* peak read at the chunk's processing step */
        for (int side = 0; ctx.stats && side < 2; side++)
            CUDA_CHECK(cudaMemcpyAsync(
                s.h_bucket + (2 * par + side) * ctx.max_regions * BK_CNT_STRIDE,
                bucket_cnt[side], t.nregions * BK_CNT_STRIDE * 4,
                cudaMemcpyDeviceToHost, st));
        tiny_mask_kernel<<<ctx.tiny_grid, 256, 0, st>>>(t, s.survivors[par],
                                              s.counters[par],
                                              ctx.max_survivors, s.tiny[par]);
        mark(PH_TINY);
        for (int side = 0; side < 2; side++)
            resieve_small_kernel<<<ctx.rs_grid, 256, 0, st>>>(
                t, ctx.small[side], s.rs_idx[side], s.rs_cnt + side,
                s.roots[side], side, s.survivors[par],
                s.counters[par], ctx.max_survivors, ctx.max_hits,
                s.hits[par], s.counters[par] + 1,
                ctx.rhits ? s.hits_r[par] : nullptr,
                ctx.rhits ? s.hit_rcnt[par] : nullptr,
                ctx.rhits ? s.hits_ov[par] : nullptr);
        mark(PH_RESIEVE);
        s.n_ev[par] = nev;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(s.kdone[par], st));
        tpar[par] = t;
    };

    /* GLAS_PIPELINE (on by default, 0 disables) keeps two chunks
     * queued on the sieve stream: chunk k+2 enters the stream once
     * chunk k's copies are queued, before the host waits for them, so
     * the sieve stream has work while the host assembles chunk k's
     * output.  The stream waits on odone[par] before it reuses the
     * parity buffers. */
    uint32_t nextj = 0;
    auto launch_next = [&]() {
        if (nextj >= task.J)
            return;
        launch(nextj, (nextj / chunk_rows) & 1);
        nextj += chunk_rows;
    };
    int par = 0;
    launch_next();
    if (ctx.pipeline)
        launch_next();
    for (uint32_t j0 = 0; j0 < task.J; j0 += chunk_rows, par ^= 1) {
        gpu_task const tc = tpar[par];

        CUDA_CHECK(cudaStreamWaitEvent(s.out_stream, s.kdone[par]));
        CUDA_CHECK(cudaMemcpyAsync(s.h_counters, s.counters[par], 3 * 4,
                                   cudaMemcpyDeviceToHost, s.out_stream));
        {
            trace_scope const tr(TR_SYNC_CTR);
            CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
        }

        if (ctx.stats) {
            trace_scope const tr(TR_STATS);
            std::unique_lock<std::mutex> const lk(ctx.lock);
            int const nev = s.n_ev[par];
            for (int e = 1; e < nev; e++) {
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, s.ev[par][e - 1],
                                                s.ev[par][e]));
                ctx.phase_ms[s.ev_phase[par][e]] += ms;
            }
            float a = 0, b = 0;
            CUDA_CHECK(cudaEventElapsedTime(&a, ctx.ref, s.ev[par][0]));
            CUDA_CHECK(cudaEventElapsedTime(&b, ctx.ref,
                                            s.ev[par][nev - 1]));
            ctx.spans[slot].emplace_back(a, b);
            for (int side = 0; side < 2; side++) {
                uint32_t const * const c =
                    s.h_bucket +
                    (2 * par + side) * ctx.max_regions * BK_CNT_STRIDE;
                uint32_t const m =
                    *std::max_element(c, c + tc.nregions * BK_CNT_STRIDE);
                ctx.bucket_max[side] = std::max(ctx.bucket_max[side], m);
                for (uint32_t r = 0; r < tc.nregions; r++)
                    ctx.bucket_sum[side] += c[r * BK_CNT_STRIDE];
                if (ctx.bucket_dump) {
                    /* one line per chunk and side: q rho j0 side
                     * nregions, then the count of every region */
                    fprintf(ctx.bucket_dump, "%lu %lu %u %d %u",
                            (unsigned long)tc.q, (unsigned long)tc.rho,
                            tc.j0, side, tc.nregions);
                    for (uint32_t r = 0; r < tc.nregions; r++)
                        fprintf(ctx.bucket_dump, " %u",
                                c[r * BK_CNT_STRIDE]);
                    fputc('\n', ctx.bucket_dump);
                }
            }
        }

        uint32_t const n_surv = s.h_counters[0];
        uint32_t const n_hits = s.h_counters[1];
        uint32_t const overflow = s.h_counters[2];
        if (overflow || n_surv > ctx.max_survivors || n_hits > ctx.max_hits) {
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
            release_slot(slot);
            fprintf(stderr, "# GPU overflow at q=%lu; falling back to CPU\n",
                    (unsigned long)task.q);
            /* resieve the whole area, keep only the chunks not yet sunk */
            sieve_result full = sieve_one_cpu(task, fb, tt);
            uint64_t const xcut = (uint64_t)j0 << task.logI;
            sieve_result rest;
            for (survivor const & sv : full.survivors)
                if (sv.x >= xcut)
                    rest.survivors.push_back(sv);
            for (resieve_hit const & h : full.hits)
                if (h.x >= xcut)
                    rest.hits.push_back(h);
            sink([rest = std::move(rest)]() mutable {
                return std::move(rest);
            });
            return;
        }

        /* With the filter, the kernel writes the kept rows and their
         * host hits into the pinned buffers itself and only its two
         * counts cross the bus; the host learns them after the
         * synchronization below.  Without it, the whole chunk output
         * is copied. */
        uint32_t n_out = ctx.filter ? 0 : n_surv;
        uint32_t n_hits_out = ctx.filter ? 0 : n_hits;
        if (ctx.filter && n_surv) {
            if (n_hits && !ctx.rhits) {
                size_t tmp_bytes = ctx.sort_temp_bytes;
                CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                    s.sort_temp, tmp_bytes, s.hits[par], s.hits_sorted,
                    n_hits, 0, 64, s.out_stream));
            }
            CUDA_CHECK(cudaMemsetAsync(s.fcnt, 0, 2 * 4, s.out_stream));
            if (ctx.stats)
                CUDA_CHECK(cudaEventRecord(s.fev[0], s.out_stream));
            uint32_t const fgrid =
                std::min(ctx.filter_grid, (n_surv + 255) / 256);
            filter_kernel<<<fgrid, 256, 0, s.out_stream>>>(
                tc, s.survivors[par], n_surv, s.tiny[par], s.hits_sorted,
                n_hits,
                ctx.rhits ? s.hits_r[par] : nullptr,
                ctx.rhits ? s.hit_rcnt[par] : nullptr,
                ctx.rhits ? s.hits_ov[par] : nullptr,
                s.counters[par] + 3,
                s.h_survivors, s.h_tiny, s.fcnt, s.h_fmeta,
                s.h_ffac, s.h_flo, ctx.fcand_cap, s.h_hits);
            if (ctx.fcand_cap && !ctx.host_mr)
                filter_prime_kernel<<<(ctx.fcand_cap + 255) / 256, 256, 0,
                                      s.out_stream>>>(s.fcnt, ctx.fcand_cap,
                                                      s.h_fmeta, s.h_flo);
            if (ctx.stats)
                CUDA_CHECK(cudaEventRecord(s.fev[1], s.out_stream));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(s.h_counters + 3, s.fcnt, 2 * 4,
                                       cudaMemcpyDeviceToHost,
                                       s.out_stream));
        }
        if (n_out) {
            CUDA_CHECK(cudaMemcpyAsync(s.h_survivors, s.survivors[par],
                                       n_out * 4, cudaMemcpyDeviceToHost,
                                       s.out_stream));
            CUDA_CHECK(cudaMemcpyAsync(s.h_tiny, s.tiny[par], n_out * 16,
                                       cudaMemcpyDeviceToHost,
                                       s.out_stream));
        }
        if (n_hits_out)
            CUDA_CHECK(cudaMemcpyAsync(s.h_hits, s.hits[par],
                                       n_hits_out * 8,
                                       cudaMemcpyDeviceToHost,
                                       s.out_stream));
        CUDA_CHECK(cudaEventRecord(s.odone[par], s.out_stream));
        if (ctx.pipeline)
            launch_next();
        {
            trace_scope const tr(TR_SYNC_OUT);
            CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
        }

        if (ctx.filter && n_surv) {
            n_out = s.h_counters[3];
            n_hits_out = s.h_counters[4];
            if (ctx.stats) {
                trace_scope const tr(TR_STATS);
                float ms = 0, a = 0, b = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, s.fev[0], s.fev[1]));
                CUDA_CHECK(cudaEventElapsedTime(&a, ctx.ref, s.fev[0]));
                CUDA_CHECK(cudaEventElapsedTime(&b, ctx.ref, s.fev[1]));
                std::unique_lock<std::mutex> const lk(ctx.lock);
                ctx.filter_ms += ms;
                ctx.spans[slot].emplace_back(a, b);
            }
        }

        auto rc = std::make_shared<raw_chunk>();
        {
            trace_scope const tr(TR_PACK);
            rc->xbeg = (uint64_t)j0 << task.logI;
            rc->n_hits = n_hits_out;
            rc->hits.assign(s.h_hits, s.h_hits + n_hits_out);
            pack_rows(s, n_out, *rc);
        }
        sink([rc] { return assemble_chunk(*rc); });
        if (!ctx.pipeline)
            launch_next();
    }
    release_slot(slot);
}
