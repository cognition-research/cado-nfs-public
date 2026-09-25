/* Exact product of two limb arrays on a GPU: a number-theoretic transform
 * modulo three 62-bit primes p = k 2^32 + 1, pointwise products, inverse
 * transforms, then the coefficients (below 2^160 < p1 p2 p3) are combined
 * by Garner's method and added with their carries (a parallel prefix over
 * generate/propagate bits, so long carry chains cost log time).
 *
 * Built as a shared library (nvcc -shared) that par_mul loads at run time:
 *   int  gpu_mul_devices();
 *   void gpu_mul(uint64_t * rp, const uint64_t * ap, size_t na,
 *                const uint64_t * bp, size_t nb);   // na >= nb >= 1
 * One product at a time per device; devices are used in turn. Operands and
 * result go through pinned staging buffers (a plain cudaMemcpy of pageable
 * memory runs several times slower than staged copies). */
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

typedef uint64_t u64;
typedef unsigned __int128 u128;

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "gpu_mul: %s: %s\n", #x, cudaGetErrorString(e)); abort(); } } while (0)

/* Residues are kept in Montgomery form, x R mod p with R = 2^64 */
struct Prime {
    u64 p, pinv;    /* pinv = -p^-1 mod 2^64 */
    u64 one, r2;    /* R mod p, R^2 mod p */
    u64 w32;        /* a root of unity of order 2^32, Montgomery form */
};

static const int NPRIMES = 3;
static const u64 PRIMES[NPRIMES] = { 4611685941117976577ULL, 4611685692009873409ULL, 4611685606110527489ULL };
static const u64 GENERATORS[NPRIMES] = { 3, 19, 3 };

__host__ __device__ static inline u64 add_mod(u64 a, u64 b, u64 p)
{
    const u64 s = a + b;
    return s >= p ? s - p : s;
}

__host__ __device__ static inline u64 sub_mod(u64 a, u64 b, u64 p)
{
    return a >= b ? a - b : a + p - b;
}

/* a b / R mod p for a b < p R; the result is below p */
__host__ __device__ static inline u64 mont_mul(u64 a, u64 b, Prime const & P)
{
    const u128 t = (u128) a * b;
    const u64 lo = (u64) t, hi = (u64) (t >> 64);
    const u64 m = lo * P.pinv;
    const u64 mp_hi = (u64) (((u128) m * P.p) >> 64);
    const u64 r = hi + mp_hi + (lo != 0);
    return r >= P.p ? r - P.p : r;
}

static u64 host_pow(u64 base, u64 e, Prime const & P)
{
    u64 r = P.one;
    for (; e; e >>= 1, base = mont_mul(base, base, P))
        if (e & 1)
            r = mont_mul(r, base, P);
    return r;
}

static u64 host_inverse(u64 a, Prime const & P)
{
    return host_pow(a, P.p - 2, P);
}

static Prime make_prime(u64 p, u64 g)
{
    Prime P;
    P.p = p;
    u64 inv = 1; /* Newton: inv <- inv (2 - p inv), doubles the correct bits */
    for (int i = 0; i < 6; i++)
        inv *= 2 - p * inv;
    P.pinv = -inv;
    P.one = (u64) (((u128) 1 << 64) % p);
    P.r2 = (u64) (((u128) P.one * P.one) % p);
    P.w32 = host_pow(mont_mul(g, P.r2, P), (p - 1) >> 32, P);
    return P;
}

__constant__ Prime c_primes[NPRIMES];

/* x[i] <- limbs[i] R mod p for i < n, 0 beyond, up to N */
__global__ void load_k(u64 * x, const u64 * limbs, size_t n, size_t N, int k)
{
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < N)
        x[i] = i < n ? mont_mul(limbs[i], c_primes[k].r2, c_primes[k]) : 0;
}

/* W[j] <- w^j for j < half, where w = w32^(2^32/N) has order N = 2 half */
__global__ void twiddles_k(u64 * W, size_t half, int log_n, int k)
{
    const size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (j >= half)
        return;
    Prime const & P = c_primes[k];
    u64 base = P.w32, r = P.one;
    for (int s = 0; s < 32 - log_n; s++)
        base = mont_mul(base, base, P);
    for (size_t e = j; e; e >>= 1, base = mont_mul(base, base, P))
        if (e & 1)
            r = mont_mul(r, base, P);
    W[j] = r;
}

/* Decimation in frequency: natural order in, bit-reversed order out.
 * Stage m pairs x[i] and x[i+m] in groups of 2m; twiddle w_2m^j = W[j N/2m]. */
__global__ void dif_k(u64 * x, const u64 * W, size_t m, size_t stride, size_t half, int k)
{
    const size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (t >= half)
        return;
    Prime const & P = c_primes[k];
    const size_t j = t & (m - 1), i = (t - j) * 2 + j;
    const u64 u = x[i], v = x[i + m];
    x[i] = add_mod(u, v, P.p);
    x[i + m] = mont_mul(sub_mod(u, v, P.p), W[j * stride], P);
}

/* Decimation in time with inverse twiddles: bit-reversed in, natural out,
 * scaled by N. The table W holds w^e for e < thalf, with w of order
 * 2 thalf, and w^-e = -w^(thalf-e) for 0 < e < thalf. */
__global__ void dit_k(u64 * x, const u64 * W, size_t m, size_t stride, size_t half, size_t thalf, int k)
{
    const size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (t >= half)
        return;
    Prime const & P = c_primes[k];
    const size_t j = t & (m - 1), i = (t - j) * 2 + j;
    const size_t e = j * stride;
    const u64 w = e ? P.p - W[thalf - e] : P.one;
    const u64 u = x[i], v = mont_mul(x[i + m], w, P);
    x[i] = add_mod(u, v, P.p);
    x[i + m] = sub_mod(u, v, P.p);
}

/* x[i] <- x[i] y[i] ninv, where x and y are in Montgomery form and ninv is
 * plain, so the result is plain (the transforms work on either form) */
__global__ void pointwise_k(u64 * x, const u64 * y, u64 ninv, size_t N, int k)
{
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < N)
        x[i] = mont_mul(mont_mul(x[i], y[i], c_primes[k]), ninv, c_primes[k]);
}

/* Garner: the coefficient c with c = r_k mod p_k for the three primes,
 * c = t1 + p1 (t2 + p2 t3) < 2^186, written as three limbs into r0, r1, r2.
 * inv12 = p1^-1 mod p2 and inv13 = (p1 p2)^-1 mod p3 are in Montgomery form,
 * so mont_mul(a, inv) is the plain product a inv mod p. */
__global__ void garner_k(u64 * r0, u64 * r1, u64 * r2, u64 inv12, u64 inv13, size_t N)
{
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const u64 p1 = c_primes[0].p, p2 = c_primes[1].p, p3 = c_primes[2].p;
    const u64 t1 = r0[i];
    const u64 t2 = mont_mul(sub_mod(r1[i], t1 % p2, p2), inv12, c_primes[1]);
    const u64 t12 = (u64) (((u128) t1 + (u128) t2 * p1) % p3);
    const u64 t3 = mont_mul(sub_mod(r2[i], t12, p3), inv13, c_primes[2]);
    const u128 in = (u128) p2 * t3 + t2;
    const u128 s0 = (u128) p1 * (u64) in + t1;
    const u128 s1 = (s0 >> 64) + (u128) p1 * (u64) (in >> 64);
    r0[i] = (u64) s0;
    r1[i] = (u64) s1;
    r2[i] = (u64) (s1 >> 64);
}

typedef unsigned char u8;

/* s[i] <- low limb of L[i] + M[i-1] + H[i-2], h[i+1] <- the rest (0 to 2) */
__global__ void sum_k(u64 * s, u8 * h, const u64 * L, const u64 * M, const u64 * H, size_t N)
{
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const u128 t = (u128) L[i] + (i >= 1 ? M[i - 1] : 0) + (i >= 2 ? H[i - 2] : 0);
    s[i] = (u64) t;
    if (i + 1 < N)
        h[i + 1] = (u8) (t >> 64);
}

/* s[i] <- low limb of s[i] + h[i]; g[i] <- the carry out of that sum,
 * p[i] <- whether a carry into limb i would carry out of it */
__global__ void add_k(u64 * s, const u8 * h, u8 * g, u8 * p, size_t N)
{
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const u128 t = (u128) s[i] + h[i];
    s[i] = (u64) t;
    g[i] = (u8) (t >> 64);
    p[i] = (u64) t == ~(u64) 0;
}

/* Carries: c[0] = 0, c[i+1] = g[i] | (p[i] & c[i]). The limbs are cut in
 * chunks; a chunk is one (g, p) pair for the next level up, and the carry
 * into a chunk gives the carries inside it. */
static const int CHUNK = 64;

__global__ void chunk_k(const u8 * g, const u8 * p, size_t n, u8 * G, u8 * P, size_t m)
{
    const size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (j >= m)
        return;
    u8 gg = 0, pp = 1;
    for (size_t i = j * CHUNK; i < (j + 1) * CHUNK && i < n; i++) {
        gg = g[i] | (p[i] & gg);
        pp &= p[i];
    }
    G[j] = gg;
    P[j] = pp;
}

__global__ void expand_k(const u8 * g, const u8 * p, size_t n, const u8 * C, u8 * c, size_t m)
{
    const size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (j >= m)
        return;
    u8 cc = C[j];
    for (size_t i = j * CHUNK; i < (j + 1) * CHUNK && i < n; i++) {
        c[i] = cc;
        cc = g[i] | (p[i] & cc);
    }
}

__global__ void fix_k(u64 * s, const u8 * c, size_t N)
{
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < N)
        s[i] += c[i];
}

static const int BLOCK = 256;

static unsigned int blocks(size_t n)
{
    return (unsigned int) ((n + BLOCK - 1) / BLOCK);
}

static void carries(const u8 * g, const u8 * p, u8 * c, size_t n)
{
    const size_t m = (n + CHUNK - 1) / CHUNK;
    u8 *G, *P, *C;
    CHECK(cudaMalloc(&G, m));
    CHECK(cudaMalloc(&P, m));
    CHECK(cudaMalloc(&C, m));
    chunk_k<<<blocks(m), BLOCK>>>(g, p, n, G, P, m);
    if (m == 1)
        CHECK(cudaMemset(C, 0, 1));
    else
        carries(G, P, C, m);
    expand_k<<<blocks(m), BLOCK>>>(g, p, n, C, c, m);
    CHECK(cudaFree(G));
    CHECK(cudaFree(P));
    CHECK(cudaFree(C));
}

static const size_t STAGE = (size_t) 256 << 20;  /* bytes per staging buffer */
static const int NSTAGE = 4, COPIERS = 8;

struct Device {
    std::mutex lock;
    u64 * W[NPRIMES] = {};  /* twiddle tables of the three primes */
    int log_n = 0;          /* for transforms of length up to 2^log_n */
    bool ready = false;
    cudaStream_t stream;    /* the copies; kernels run on the legacy default
                               stream, which orders itself after them */
    char * stage[NSTAGE];
    cudaEvent_t done[NSTAGE];  /* the last copy out of / into stage[i] */
};

/* memcpy on COPIERS threads (one thread copies at about 10 GB/s) */
static void copy(void * dst, const void * src, size_t n)
{
    const size_t piece = (n + COPIERS - 1) / COPIERS;
    std::vector<std::thread> threads;
    for (size_t off = 0; off < n; off += piece)
        threads.emplace_back([=] {
            memcpy((char *) dst + off, (const char *) src + off, std::min(piece, n - off));
        });
    for (auto & t : threads)
        t.join();
}

static void to_device(Device & d, void * dst, const void * src, size_t n)
{
    for (size_t i = 0, off = 0; off < n; i++, off += STAGE) {
        const size_t len = std::min(STAGE, n - off);
        char * s = d.stage[i % NSTAGE];
        CHECK(cudaEventSynchronize(d.done[i % NSTAGE]));
        copy(s, (const char *) src + off, len);
        CHECK(cudaMemcpyAsync((char *) dst + off, s, len, cudaMemcpyHostToDevice, d.stream));
        CHECK(cudaEventRecord(d.done[i % NSTAGE], d.stream));
    }
}

/* NSTAGE pieces are in flight: piece i is copied out while i+1.. arrive */
static void to_host(Device & d, void * dst, const void * src, size_t n)
{
    const size_t pieces = (n + STAGE - 1) / STAGE;
    auto issue = [&] (size_t i) {
        if (i >= pieces)
            return;
        const size_t off = i * STAGE;
        CHECK(cudaMemcpyAsync(d.stage[i % NSTAGE], (const char *) src + off,
                              std::min(STAGE, n - off), cudaMemcpyDeviceToHost, d.stream));
        CHECK(cudaEventRecord(d.done[i % NSTAGE], d.stream));
    };
    for (size_t i = 0; i < NSTAGE; i++)
        issue(i);
    for (size_t i = 0; i < pieces; i++) {
        const size_t off = i * STAGE;
        CHECK(cudaEventSynchronize(d.done[i % NSTAGE]));
        copy((char *) dst + off, d.stage[i % NSTAGE], std::min(STAGE, n - off));
        issue(i + NSTAGE);
    }
}

static std::unique_ptr<Device[]> devices;
static int ndevices;
static std::atomic<unsigned> next_device;
static Prime primes[NPRIMES];
static u64 inv12, inv13;

static void init()
{
    static std::once_flag once;
    std::call_once(once, [] {
        CHECK(cudaGetDeviceCount(&ndevices));
        devices.reset(new Device[ndevices]);
        for (int k = 0; k < NPRIMES; k++)
            primes[k] = make_prime(PRIMES[k], GENERATORS[k]);
        inv12 = host_inverse(mont_mul(primes[0].p % primes[1].p, primes[1].r2, primes[1]), primes[1]);
        const u64 p12 = (u64) (((u128) primes[0].p * primes[1].p) % primes[2].p);
        inv13 = host_inverse(mont_mul(p12, primes[2].r2, primes[2]), primes[2]);
    });
}

extern "C" int gpu_mul_devices()
{
    init();
    return ndevices;
}

/* The twiddle table of a device serves every length up to 2^log_n; a
 * longer transform rebuilds it. */
static void prepare(Device & d, int log_n)
{
    if (!d.ready) {
        CHECK(cudaMemcpyToSymbol(c_primes, primes, sizeof(primes)));
        CHECK(cudaStreamCreate(&d.stream));
        for (int i = 0; i < NSTAGE; i++) {
            CHECK(cudaMallocHost(&d.stage[i], STAGE));
            CHECK(cudaEventCreateWithFlags(&d.done[i], cudaEventDisableTiming));
        }
        d.ready = true;
    }
    if (log_n <= d.log_n)
        return;
    const size_t half = (size_t) 1 << (log_n - 1);
    for (int k = 0; k < NPRIMES; k++) {
        if (d.W[k])
            CHECK(cudaFree(d.W[k]));
        CHECK(cudaMalloc(&d.W[k], half * sizeof(u64)));
        twiddles_k<<<blocks(half), BLOCK>>>(d.W[k], half, log_n, k);
    }
    d.log_n = log_n;
}

static void ntt(u64 * x, Device const & d, int log_n, int k, bool inverse)
{
    const size_t half = (size_t) 1 << (log_n - 1);
    const size_t thalf = (size_t) 1 << (d.log_n - 1);
    /* the table has the roots of order 2 thalf: those of order 2m sit at
     * multiples of thalf / m */
    for (int l = 0; l < log_n; l++) {
        const int lm = inverse ? l : log_n - 1 - l;
        const size_t m = (size_t) 1 << lm;
        const size_t stride = thalf / m;
        if (inverse)
            dit_k<<<blocks(half), BLOCK>>>(x, d.W[k], m, stride, half, thalf, k);
        else
            dif_k<<<blocks(half), BLOCK>>>(x, d.W[k], m, stride, half, k);
    }
}

extern "C" void gpu_mul(u64 * rp, const u64 * ap, size_t na, const u64 * bp, size_t nb)
{
    init();
    if (na < nb) {
        const u64 * t = ap; ap = bp; bp = t;
        const size_t s = na; na = nb; nb = s;
    }
    if (nb == 0 || na + nb > ((size_t) 1 << 31)) {
        fprintf(stderr, "gpu_mul: sizes %zu x %zu out of range\n", na, nb);
        abort();
    }
    int log_n = 1;
    while (((size_t) 1 << log_n) < na + nb)
        log_n++;
    const size_t N = (size_t) 1 << log_n;

    const int dev = (int) (next_device++ % ndevices);
    Device & d = devices[dev];
    std::lock_guard<std::mutex> guard(d.lock);
    CHECK(cudaSetDevice(dev));
    prepare(d, log_n);

    u64 *da, *db, *r[NPRIMES], *y;
    u8 *h, *g, *p;
    CHECK(cudaMalloc(&da, na * sizeof(u64)));
    CHECK(cudaMalloc(&db, nb * sizeof(u64)));
    CHECK(cudaMalloc(&y, N * sizeof(u64)));
    for (int k = 0; k < NPRIMES; k++)
        CHECK(cudaMalloc(&r[k], N * sizeof(u64)));
    CHECK(cudaMalloc(&h, N));
    CHECK(cudaMalloc(&g, N));
    CHECK(cudaMalloc(&p, N));
    to_device(d, da, ap, na * sizeof(u64));
    to_device(d, db, bp, nb * sizeof(u64));

    for (int k = 0; k < NPRIMES; k++) {
        Prime const & P = primes[k];
        load_k<<<blocks(N), BLOCK>>>(r[k], da, na, N, k);
        load_k<<<blocks(N), BLOCK>>>(y, db, nb, N, k);
        ntt(r[k], d, log_n, k, false);
        ntt(y, d, log_n, k, false);
        /* the inverse transform leaves N c: scale by N^-1 = ((p+1)/2)^log_n,
         * taken out of Montgomery form by a product with 1 */
        const u64 ninv = mont_mul(host_pow(mont_mul((P.p + 1) / 2, P.r2, P), log_n, P), 1, P);
        pointwise_k<<<blocks(N), BLOCK>>>(r[k], y, ninv, N, k);
        ntt(r[k], d, log_n, k, true);
    }
    garner_k<<<blocks(N), BLOCK>>>(r[0], r[1], r[2], inv12, inv13, N);
    CHECK(cudaMemset(h, 0, 1));
    sum_k<<<blocks(N), BLOCK>>>(y, h, r[0], r[1], r[2], N);
    add_k<<<blocks(N), BLOCK>>>(y, h, g, p, N);
    carries(g, p, h, N);
    fix_k<<<blocks(N), BLOCK>>>(y, h, N);
    to_host(d, rp, y, (na + nb) * sizeof(u64));
    CHECK(cudaGetLastError());

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(y));
    for (int k = 0; k < NPRIMES; k++)
        CHECK(cudaFree(r[k]));
    CHECK(cudaFree(h));
    CHECK(cudaFree(g));
    CHECK(cudaFree(p));
}
