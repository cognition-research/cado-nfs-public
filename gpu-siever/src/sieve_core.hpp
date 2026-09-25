#ifndef GLAS_SIEVE_CORE_HPP
#define GLAS_SIEVE_CORE_HPP

/* Sieve core interface. The host prepares one sq_task per special-q.
 * A backend (CPU reference or CUDA) sieves the (i,j) region and returns
 * the survivor positions plus the factor-base primes that hit them.
 *
 * Position encoding: x = (i + I/2) + (j << logI), with I = 1 << logI,
 * -I/2 <= i < I/2 and 0 <= j < J.  This matches las.  The area
 * J << logI can exceed 2^32, so positions are 64-bit.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

constexpr int GLAS_MAX_DEGREE = 8;

/* Boolean environment flag: unset means dflt, "0" means off, any
 * other value means on. */
inline bool env_flag(char const * name, bool dflt)
{
    char const * v = std::getenv(name);
    return v ? std::strcmp(v, "0") != 0 : dflt;
}

/* GPU-hole diagnosis (active with GLAS_GPU_STATS): each siever thread
 * logs what it does, on the same clock as the GPU chunk spans, and
 * the idle report maps each hole in the GPU coverage to these
 * states. */
enum trace_state {
    TR_OTHER, TR_PREP, TR_FEED, TR_ACQUIRE, TR_LAUNCH, TR_STATS,
    TR_SYNC_CTR, TR_SYNC_OUT, TR_PACK, TR_ASSEMBLE, TR_SINK, TR_N
};

constexpr char const * TRACE_NAMES[TR_N] = {
    "other", "prep",     "feed",     "acquire", "launch",   "stats",
    "sync_ctr", "sync_out", "pack",  "assemble", "sink"};

struct trace_log {
    std::vector<std::pair<float, int>> ev;   /* (ms, state) */
    int cur = TR_OTHER;
    bool worker = false;
};

inline std::mutex trace_lock;
inline std::vector<std::unique_ptr<trace_log>> trace_logs;
inline std::atomic<bool> trace_on{false};
inline std::chrono::steady_clock::time_point trace_t0;

/* A worker thread sets this before its first trace_scope; the idle
 * report then lists its log apart from the siever logs. */
inline thread_local bool trace_worker = false;

inline trace_log & my_trace_log()
{
    thread_local trace_log * const mine = [] {
        std::lock_guard<std::mutex> const lk(trace_lock);
        trace_logs.push_back(std::make_unique<trace_log>());
        trace_logs.back()->worker = trace_worker;
        return trace_logs.back().get();
    }();
    return *mine;
}

inline float trace_now()
{
    return std::chrono::duration<float, std::milli>(
               std::chrono::steady_clock::now() - trace_t0)
        .count();
}

/* Marks the calling thread as in `state` for the scope's lifetime;
 * scopes nest. */
struct trace_scope {
    int prev = -1;
    explicit trace_scope(int state)
    {
        if (!trace_on.load(std::memory_order_relaxed))
            return;
        trace_log & log = my_trace_log();
        prev = log.cur;
        log.cur = state;
        log.ev.emplace_back(trace_now(), state);
    }
    ~trace_scope()
    {
        if (prev < 0)
            return;
        trace_log & log = my_trace_log();
        log.cur = prev;
        log.ev.emplace_back(trace_now(), prev);
    }
};

/* Sieve values are log2 * LOG_SCALE, rounded to integers. */
constexpr int LOG_SCALE = 8;

/* Extra sieve units (log2 * LOG_SCALE) allowed over the survivor bound.
 * Covers rounding error in the norm and in the subtracted prime logs. */
constexpr int SURVIVOR_SLACK = 8;

/* Primes <= this bound are trial-divided on the host and are not
 * resieved. */
constexpr uint32_t TINY_PRIME_BOUND = 256;

struct fb_entry {
    uint32_t q;      /* prime or prime power */
    uint32_t r;      /* root in (a,b) coordinates; r >= q encodes the
                        projective root r - q */
    uint8_t weight;  /* round(LOG_SCALE * (n1-n2) * log2(base prime)) */
    bool resieve;    /* prime entries with q > TINY_PRIME_BOUND */
};

struct sq_task {
    uint64_t q;
    uint64_t rho;    /* affine root of the sq-side polynomial mod q */
    int sq_side;
    int64_t a0, b0, a1, b1;   /* reduced q-lattice basis */
    int logI;
    uint32_t J;
    int degree[2];
    double fijd[2][GLAS_MAX_DEGREE + 1];  /* norm poly in (i,j); sq side
                                             divided by q */
    int32_t bound[2];  /* survivor bound, in sieve units */
};

struct resieve_hit {
    uint64_t x;
    uint32_t q;
    uint8_t side;
};

/* The primes <= TINY_PRIME_BOUND and the norm coefficients mod each;
 * the backends flag the tiny primes that divide each survivor norm.
 * cmod[side][i * (GLAS_MAX_DEGREE + 1) + k] = coefficient k mod
 * primes[i]. */
struct tiny_table {
    std::vector<uint32_t> primes;   /* at most 64 */
    std::vector<uint32_t> cmod[2];
};

struct survivor {
    uint64_t x;
    uint64_t tiny[2];   /* tiny-prime divisor mask of each side norm */
};

/* GPU-assembled candidate (GLAS_GPU_ASSEMBLE, on by default): the
 * filter kernel keeps the values it computes for a kept survivor.
 * fac holds the primes divided out of the side-0 norm, then side 1;
 * a prime repeats once per power.  The special q is not in fac: the
 * sq-side norm contains q^nq.  leftover is what remains of each side
 * norm.  The host builds the relation from these values and does not
 * recompute the norms.  pend (GLAS_FLT_HOST_MR): bit per side whose
 * leftover still owes the primality test; a prime leftover drops the
 * candidate. */
constexpr int GPU_FAC_CAP = 48;

struct gpu_cand {
    uint64_t x;
    uint16_t nf[2];             /* factor counts in fac, per side */
    uint16_t nq;
    uint8_t pend;
    uint64_t leftover[2][2];    /* [side][limb], little-endian */
    uint32_t fac[GPU_FAC_CAP];
};

/* Both backends return survivors, cands and hits sorted by position.
 * survivors still need the pass-1 norm work on the host; cands are
 * GPU-assembled. */
struct sieve_result {
    std::vector<survivor> survivors;
    std::vector<resieve_hit> hits;
    std::vector<gpu_cand> cands;
    size_t nsurv() const { return survivors.size() + cands.size(); }
};

/* GPU cofactorization: each side's leftover norm is a product of
 * primes above the factor-base bound and fits in 128 bits.  The kernel
 * splits it with Miller-Rabin + batched ECM; a side is smooth when
 * every prime is below 2^lpb.  When ECM misses a composite, with
 * hard_on_gpu the side counts as not smooth, otherwise the pending
 * bit for the side stays set and the host must finish it. */
constexpr int COFAC_MAX_FACTORS = 6;

struct cofac_pair {
    uint64_t n[2][2];   /* [side][limb], little-endian; 1 = nothing left */
};

struct cofac_result {
    uint64_t f[2][COFAC_MAX_FACTORS];
    uint8_t nf[2];
    uint8_t smooth;
    uint8_t pending;    /* bit per side; meaningful when !hard_on_gpu */
};

void gpu_cofactor(std::vector<cofac_pair> const & in,
                  std::vector<cofac_result> & out, int const lpb[2],
                  bool hard_on_gpu);

void sort_hits(std::vector<resieve_hit> & hits);

/* Exact-norm survivor filter (GLAS_GPU_FILTER, on by default): the
 * device recomputes each survivor's norms, divides out the special q,
 * the resieve hits and the flagged tiny primes, and keeps only the
 * survivors whose leftovers pass the host leftover_ok test.  Norms are
 * fixed-width NORM_LIMBS x 64-bit magnitudes; leftovers that pass the
 * mfb bound fit in 128 bits. */
constexpr int NORM_LIMBS = 8;

struct filter_params {
    struct {
        uint64_t limb[NORM_LIMBS];   /* |coefficient|, little-endian */
        int32_t neg;
    } coeff[2][GLAS_MAX_DEGREE + 1];
    uint64_t lim[2];
    uint32_t lpb[2];
    uint32_t mfb[2];
};

sieve_result sieve_one_cpu(sq_task const & task,
                           std::vector<fb_entry> const * fb,
                           tiny_table const & tt);

/* CUDA backend (sieve_core_gpu.cu); gpu_init returns false when no
 * device is present.  sieve_one_gpu calls the sink once per chunk, in
 * position order, with a closure that assembles that chunk's results.
 * The sink may run the closure on another thread; the siever returns
 * to the GPU without waiting for the assembly. */
using chunk_result = std::function<sieve_result()>;
using chunk_sink = std::function<void(chunk_result &&)>;
void gpu_warmup();
bool gpu_init(std::vector<fb_entry> const * fb, int logI, uint32_t max_J,
              tiny_table const & tt, filter_params const & fp);
int gpu_slots();
void sieve_one_gpu(sq_task const & task, std::vector<fb_entry> const * fb,
                   tiny_table const & tt, chunk_sink const & sink);

#endif
