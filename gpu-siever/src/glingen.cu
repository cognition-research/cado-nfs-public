#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <linux/mempolicy.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "glingen.h"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                cudaGetErrorString(err_));                              \
        abort();                                                        \
    }                                                                   \
} while (0)

struct __align__(16) K {
    uint64_t lo, hi;
};

/* A matrix of transforms is split by points over the GPUs of the
 * node: piece q holds the points [q*tsize/ndev, (q+1)*tsize/ndev) of
 * every entry, on device q. The product of transforms is pointwise,
 * so each device multiplies its own points; the dft and the ift run
 * on rows of the block, and the points move between the devices
 * before or after. A piece lives point-major: point p of entry e of a
 * matrix of E entries is the two words at (p*E + e)*2, so that the
 * threads of consecutive entries access a point together. The
 * transforms have tsize = 2^k points. */
#define MAX_DEV 16
struct glingen_tmat {
    uint64_t * x[MAX_DEV];
    unsigned int nrows, ncols, ndev;
    size_t tsize;
};

/* the piece of a device, as the kernels see it */
struct dmat {
    uint64_t * x;
    unsigned int nrows, ncols;
    size_t tsize;
};
typedef glingen_tmat tmat;
typedef glingen_sub sub;
typedef glingen_polys polys;
typedef glingen_cantor cantor;

__host__ __device__ __forceinline__ size_t tmat_entries(dmat t)
{
    return (size_t) t.nrows * t.ncols;
}

/* the transform of entry (i,j) of the block R of t: f[p] for point p */
struct tview {
    uint64_t * x;
    size_t stride;
    __device__ K get(int64_t i) const {
        return { x[i * stride], x[i * stride + 1] };
    }
    __device__ void set(int64_t i, K v) const {
        x[i * stride] = v.lo;
        x[i * stride + 1] = v.hi;
    }
    __device__ void add(int64_t i, K v) const {
        x[i * stride] ^= v.lo;
        x[i * stride + 1] ^= v.hi;
    }
};

__device__ __forceinline__ tview entry(dmat t, sub R, unsigned int i, unsigned int j)
{
    size_t e = (size_t) (R.i0 + i) * t.ncols + R.j0 + j;
    return { t.x + 2 * e, 2 * tmat_entries(t) };
}

/* entry (i,j) of the block R of the polynomials a */
__host__ __device__ __forceinline__ uint64_t * entry(polys a, sub R, unsigned int i, unsigned int j)
{
    return a.x + ((size_t) (R.i0 + i) * a.ld + R.j0 + j) * a.words;
}

/* ---- the Cantor transform of gf2x (gf2x-cantor-fft.c), the same
 * steps in the same order as the CPU code ---- */


/* a x^t for t < 128, modulo x^128 + x^7 + x^2 + x + 1, the modulus of
 * gf2x: the 256-bit shift folds by x^128 = x^7 + x^2 + x + 1 twice */
__device__ K kmul_x(K a, unsigned int t)
{
    unsigned int q = t / 64, r = t % 64;
    uint64_t w0 = a.lo << r;
    uint64_t w1 = a.hi << r | (r ? a.lo >> (64 - r) : 0);
    uint64_t w2 = r ? a.hi >> (64 - r) : 0;
    K lo = q ? K { 0, w0 } : K { w0, w1 };
    K hi = q ? K { w1, w2 } : K { w2, 0 };
    uint64_t top = 0;
    for (int s : (const int[]) { 0, 1, 2, 7 }) {
        lo.lo ^= hi.lo << s;
        lo.hi ^= hi.hi << s | (s ? hi.lo >> (64 - s) : 0);
        top ^= s ? hi.hi >> (64 - s) : 0;
    }
    lo.lo ^= top ^ top << 1 ^ top << 2 ^ top << 7;
    return lo;
}

/* The products by a constant beta go through a table in shared
 * memory: for each of the 32 nibbles of an element, the products of
 * beta by the 16 values of the nibble. The block builds it together
 * (the betas of a transform depend on the level and the block, not on
 * the entry), from the powers beta x^t. */
struct ktable {
    K t[32 * 16];
    K pw[128];
};

/* the table of beta, or nullptr when beta is 0; all threads of the
 * block must call it, with the same beta, and none may still read the
 * old table */
__device__ const K * beta_table(K beta, ktable & T)
{
    if (!(beta.lo | beta.hi)) return nullptr;
    for (unsigned int t = threadIdx.x; t < 128; t += blockDim.x)
        T.pw[t] = kmul_x(beta, t);
    __syncthreads();
    for (unsigned int e = threadIdx.x; e < 32 * 16; e += blockDim.x) {
        K s = { 0, 0 };
        for (int j = 0; j < 4; j++) {
            if (!(e >> j & 1)) continue;
            s.lo ^= T.pw[4 * (e / 16) + j].lo;
            s.hi ^= T.pw[4 * (e / 16) + j].hi;
        }
        T.t[e] = s;
    }
    __syncthreads();
    return T.t;
}

__device__ __forceinline__ K kmul_table(K a, const K * T)
{
    K r = { 0, 0 };
#pragma unroll
    for (int p = 0; p < 32; p++) {
        uint64_t w = p < 16 ? a.lo : a.hi;
        K v = T[p * 16 + (w >> 4 * (p % 16) & 15)];
        r.lo ^= v.lo;
        r.hi ^= v.hi;
    }
    return r;
}

/* the tables of gf2x-cantor-fft.c for the field of 128 bits: the
 * basis Betai, and the terms of the polynomials S_k */
#define BETA(x, y, z, t) { (uint64_t) (y) << 32 | (x), (uint64_t) (t) << 32 | (z) }
__constant__ K Betai[32] = {
    BETA(1, 0, 0, 0),
    BETA(2763131656u, 1735044255u, 4101184249u, 693813425u),
    BETA(465473314u, 4279278019u, 3443545289u, 2719171618u),
    BETA(45213768u, 1404590948u, 3790498114u, 2035930892u),
    BETA(1766741572u, 3401373924u, 3912612066u, 150385427u),
    BETA(1642619116u, 1296609638u, 891822488u, 1237922326u),
    BETA(1432407564u, 2705320108u, 2845284892u, 87917909u),
    BETA(3169845528u, 1555107770u, 977112198u, 1297233226u),
    BETA(3766302226u, 3784070584u, 2372619472u, 751494893u),
    BETA(3319219312u, 1485529668u, 1919364198u, 554536894u),
    BETA(2221231524u, 3494370010u, 2246133939u, 3255900204u),
    BETA(4004748792u, 4275287155u, 1207038226u, 273485854u),
    BETA(4134924972u, 3206127284u, 3660212267u, 3465895584u),
    BETA(1810844824u, 432364429u, 3275102095u, 3685546794u),
    BETA(2239846096u, 4115864360u, 3774983060u, 1484196804u),
    BETA(572310756u, 2341975643u, 1205907074u, 1233095387u),
    BETA(374654328u, 2701982826u, 4048512218u, 95057245u),
    BETA(1069419604u, 3135871566u, 603196539u, 2195404101u),
    BETA(2819376738u, 2390789826u, 3206934123u, 3126567229u),
    BETA(2667030844u, 1546355395u, 894286960u, 806061570u),
    BETA(3589162292u, 2284726595u, 3987361705u, 3261030005u),
    BETA(488495394u, 2513641547u, 3724856370u, 956902965u),
    BETA(894705622u, 3412990576u, 2985357841u, 2327012336u),
    BETA(691430644u, 2704377464u, 1647445035u, 3189688948u),
    BETA(3013060094u, 3867471049u, 66167789u, 3014841169u),
    BETA(1101590000u, 1839657742u, 3478054432u, 884001900u),
    BETA(494046998u, 2899707219u, 2156601667u, 2796513881u),
    BETA(3398486548u, 57111011u, 1428951442u, 894796186u),
    BETA(3950556970u, 3334574894u, 3366886047u, 3938361620u),
    BETA(999564750u, 4233655517u, 706258300u, 428364188u),
    BETA(1913007172u, 1894536580u, 871798750u, 1621883580u),
    BETA(2252056148u, 2850320680u, 3455532848u, 234262497u),
};
__constant__ uint32_t ind_S[31][15] = {
    {},
    {1},
    {1},
    {4, 2, 1},
    {1},
    {16, 2, 1},
    {16, 4, 1},
    {64, 32, 16, 8, 4, 2, 1},
    {1},
    {256, 2, 1},
    {256, 4, 1},
    {1024, 512, 256, 8, 4, 2, 1},
    {256, 16, 1},
    {4096, 512, 256, 32, 16, 2, 1},
    {4096, 1024, 256, 64, 16, 4, 1},
    {16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1},
    {1},
    {65536, 2, 1},
    {65536, 4, 1},
    {262144, 131072, 65536, 8, 4, 2, 1},
    {65536, 16, 1},
    {1048576, 131072, 65536, 32, 16, 2, 1},
    {1048576, 262144, 65536, 64, 16, 4, 1},
    {4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 128, 64, 32, 16, 8, 4, 2, 1},
    {65536, 256, 1},
    {16777216, 131072, 65536, 512, 256, 2, 1},
    {16777216, 262144, 65536, 1024, 256, 4, 1},
    {67108864, 33554432, 16777216, 524288, 262144, 131072, 65536, 2048, 1024, 512, 256, 8, 4, 2, 1},
    {16777216, 1048576, 65536, 4096, 256, 16, 1},
    {268435456, 33554432, 16777216, 2097152, 1048576, 131072, 65536, 8192, 4096, 512, 256, 32, 16, 2, 1},
    {268435456, 67108864, 16777216, 4194304, 1048576, 262144, 65536, 16384, 4096, 1024, 256, 64, 16, 4, 1},
};
__constant__ unsigned int ind_number[31] = {
    0, 1, 1, 3, 1, 3, 3, 7, 1, 3, 3, 7, 3, 7, 7, 15,
    1, 3, 3, 7, 3, 7, 7, 15, 3, 7, 7, 15, 7, 15, 15
};

/* the sum of the Betai[j] over the bits j of i */
__device__ K all_betai(size_t i)
{
    K x = { 0, 0 };
    for (unsigned int j = 0; i; i >>= 1, j++) {
        if (!(i & 1)) continue;
        x.lo ^= Betai[j].lo;
        x.hi ^= Betai[j].hi;
    }
    return x;
}

/* The transform kernels run a thread block per tile of E entries (E
 * divides the block size): the threads u0, u0 + du, ... of an entry
 * share its points, so that consecutive threads hold consecutive
 * entries and the accesses of a point coalesce. The block walks the
 * levels and the blocks of the transform together, in the same order
 * as the CPU code, builds the table of each beta together, and does
 * the folds of a block in parallel over the points. A thread past
 * the last entry (on is false) takes part in the tables and the
 * barriers only. */
struct lane {
    tview f;
    bool on;
    unsigned int u0, du;
};

__device__ __forceinline__ tview at(tview f, int64_t off)
{
    return { f.x + off * f.stride, f.stride };
}

__device__ __forceinline__ K kadd(K a, K b)
{
    return { a.lo ^ b.lo, a.hi ^ b.hi };
}

/* The fold of gf2x: source i folds onto i - 2^k + s for the terms s of
 * S_k, and beta i onto i - 2^k. Here the folds of the sources [lo, hi)
 * land on the targets [tlo, thi), gathered by target: the targets hold
 * no source, and the sources hold their values at fold time. T is the
 * table of beta, or null for no beta term. */
__device__ void fold(const lane & w, tview f, unsigned int k, int64_t lo, int64_t hi,
        int64_t tlo, int64_t thi, const K * T)
{
    if (!w.on) return;
    int64_t K1 = (int64_t) 1 << k;
    for (int64_t t = tlo + w.u0; t < thi; t += w.du) {
        K v = { 0, 0 };
        for (unsigned int j = 0; j < ind_number[k]; j++) {
            int64_t i = t + K1 - ind_S[k][j];
            if (i >= lo && i < hi) v = kadd(v, f.get(i));
        }
        if (T && t + K1 >= lo && t + K1 < hi) v = kadd(v, kmul_table(f.get(t + K1), T));
        f.add(t, v);
    }
}

/* f[i + 2^k] += f[i]: the other remainder, modulo S_k + beta + 1 */
__device__ void add_halves(const lane & w, tview f, unsigned int k)
{
    if (!w.on) return;
    int64_t K1 = (int64_t) 1 << k;
    for (int64_t i = w.u0; i < K1; i += w.du) f.add(i + K1, f.get(i));
}

/* quotremSi of gf2x on the L elements at f: the quotient by S_k + beta
 * in [2^k, L), the remainder below. The folds run from the top down,
 * and a source takes the folds of the sources at least 2^k - max S_k
 * above it only: the quotient is final chunk by chunk of that size,
 * from the top, then the remainder takes all the sources at once. */
__device__ void quotrem_si(const lane & w, tview f, unsigned int k, int64_t L, const K * T)
{
    int64_t K1 = (int64_t) 1 << k;
    for (int64_t c = K1 - ind_S[k][0], hi = L - c; hi > K1; hi -= c) {
        fold(w, f, k, hi, L, hi - c > K1 ? hi - c : K1, hi, T);
        __syncthreads();
    }
    fold(w, f, k, K1, L, 0, K1, T);
    __syncthreads();
}

__device__ void reduce_si(const lane & w, tview f, unsigned int k, const K * T)
{
    quotrem_si(w, f, k, (int64_t) 2 << k, T);
    add_halves(w, f, k);
    __syncthreads();
}

/* the folds of the sources [lo, L), from the bottom up: a source reads
 * its value unchanged (its folds come from above), and a chunk of
 * sources lands just below itself */
__device__ void fold_up(const lane & w, tview f, unsigned int k, int64_t lo, int64_t L, const K * T)
{
    int64_t K1 = (int64_t) 1 << k;
    for (int64_t c = K1 - ind_S[k][0]; lo < L; lo += c) {
        fold(w, f, k, lo, lo + c < L ? lo + c : L, lo - K1, lo, T);
        __syncthreads();
    }
}

__device__ void interpolate_si(const lane & w, tview f, unsigned int k, const K * T)
{
    add_halves(w, f, k);
    __syncthreads();
    fold_up(w, f, k, (int64_t) 1 << k, (int64_t) 2 << k, T);
}

__device__ __forceinline__ size_t words_of(size_t bits)
{
    return (bits + 63) / 64;
}

/* the gf2x_cantor_fft_dft of the nF bits at F into f, the k levels of
 * multievaluateKnew_trunc: the blocks at or beyond the n significant
 * points are skipped, and stay unused */
__device__ void dft_entry(const lane & w, const uint64_t * F, size_t nF,
        unsigned int k, size_t n, ktable & T)
{
    size_t Fl = words_of(nF);
    if (w.on)
        for (size_t i = w.u0; i < (size_t) 1 << k; i += w.du)
            w.f.set(i, { i < Fl ? F[i] : 0, 0 });
    __syncthreads();
    for (int l = k - 1; l >= 0; l--)
        for (size_t b = 0; b << (l + 1) < n; b++)
            reduce_si(w, at(w.f, b << (l + 1)), l, beta_table(all_betai(2 * b), T));
}

/* reduceModTrunc of gf2x: after the interpolation of the n points, the
 * reduction modulo the product of the S_i + beta with n roots */
__device__ void reduce_mod_trunc(const lane & w, unsigned int k, size_t n, ktable & T)
{
    size_t len = n, off = 0;
    for (int i = k - 1; i >= 0; i--) {
        if (!(len >> i)) continue;
        const K * t = beta_table(all_betai(off >> i), T);
        quotrem_si(w, at(w.f, off), i, ((int64_t) 1 << k) - off, t);
        off += (size_t) 1 << i;
        len -= (size_t) 1 << i;
    }
    unsigned int i = __ffsll(n) - 1;
    size_t ii = (size_t) 1 << i;
    len = n >> (i + 1);
    for (i++; i < k; i++, len >>= 1) {
        if (!(len & 1)) continue;
        const K * t = beta_table(all_betai((n - ((size_t) 1 << i) - ii) >> i), T);
        fold_up(w, at(w.f, n - ii), i, 0, ii, t);
        ii += (size_t) 1 << i;
    }
}

/* the gf2x_cantor_fft_ift: interpolateK or interpolateK_trunc, then
 * recomposeK_bits into the nH bits at H (recomposeK when shift is 0) */
__device__ void ift_entry(const lane & w, uint64_t * H, size_t nH, unsigned int k,
        size_t n, size_t shift, ktable & T)
{
    if (w.on)
        for (size_t i = n + w.u0; i < (size_t) 1 << k; i += w.du) w.f.set(i, { 0, 0 });
    __syncthreads();
    for (unsigned int i = 0; i + 1 < k; i++)
        for (size_t j = 0; n > (j << (i + 1)) + ((size_t) 1 << i); j++)
            interpolate_si(w, at(w.f, j << (i + 1)), i, beta_table(all_betai(2 * j), T));
    interpolate_si(w, w.f, k - 1, nullptr);
    if (n & (n - 1)) reduce_mod_trunc(w, k, n, T);
    if (!w.on) return;

    size_t Fl = words_of(nH), Q = shift / 64, cnt = shift % 64;
    bool tail = Q + Fl == words_of(nH + shift) - 1;
    auto g = [&](size_t i) -> uint64_t {
        if (i == Fl && !tail) return 0;
        uint64_t v = w.f.get(Q + i).lo;
        return Q + i ? v ^ w.f.get(Q + i - 1).hi : v;
    };
    for (size_t i = w.u0; i < Fl; i += w.du) {
        H[i] = cnt ? g(i) >> cnt | g(i + 1) << (64 - cnt) : g(i);
        if (i == Fl - 1 && nH % 64) H[i] &= ((uint64_t) 1 << nH % 64) - 1;
    }
}

/* the thread's lane: its entry of the tile of E entries of the block
 * (clamped to the last entry when on is false) and its share of the
 * points */
__device__ __forceinline__ lane my_lane(dmat t, sub R, unsigned int E,
        unsigned int & i, unsigned int & j)
{
    size_t n = (size_t) R.nrows * R.ncols;
    size_t e = (size_t) blockIdx.x * E + threadIdx.x % E;
    bool on = e < n;
    e = on ? e : n - 1;
    i = e / R.ncols;
    j = e % R.ncols;
    return { entry(t, R, i, j), on, threadIdx.x / E, blockDim.x / E };
}

__global__ void dft_kernel(dmat t, sub R, polys a, cantor fti, unsigned int E)
{
    __shared__ ktable T;
    unsigned int i, j;
    lane w = my_lane(t, R, E, i, j);
    sub R0 = { 0, 0, R.nrows, R.ncols };
    dft_entry(w, entry(a, R0, i, j), a.bits, fti.k, fti.n, T);
}

__global__ void ift_kernel(polys a, dmat t, sub R, cantor fti, unsigned int E)
{
    __shared__ ktable T;
    unsigned int i, j;
    lane w = my_lane(t, R, E, i, j);
    sub R0 = { 0, 0, R.nrows, R.ncols };
    ift_entry(w, entry(a, R0, i, j), a.bits, fti.k, fti.n, fti.mp_shift, T);
}

/* one thread per (point, entry) of the block R */
__global__ void zero_kernel(dmat t, sub R)
{
    size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t) R.nrows * R.ncols;
    if (x >= n * t.tsize) return;
    entry(t, R, x % n / R.ncols, x % n % R.ncols).set(x / n, { 0, 0 });
}

/* the points of piece q of t */
__host__ __device__ __forceinline__ size_t piece_start(const tmat & t, unsigned int q)
{
    return t.tsize * q / t.ndev;
}

__device__ __forceinline__ unsigned int piece_of(const tmat & t, size_t p)
{
    unsigned int q = p * t.ndev / t.tsize;
    q -= piece_start(t, q) > p;
    q += piece_start(t, q + 1) <= p;
    return q;
}

/* one thread per (point, entry) of the rows Rq: point p of entry (i,j)
 * of the stage st, a whole transform, against the same point in the
 * piece that owns it; the entries of a point are contiguous on both
 * sides */
__global__ void move_kernel(tmat t, dmat st, sub Rq, size_t threads, bool to_pieces)
{
    size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= threads) return;
    size_t p = x / tmat_entries(st), e = x % tmat_entries(st);
    unsigned int q = piece_of(t, p);
    ulonglong2 * s = (ulonglong2 *) st.x + p * tmat_entries(st) + e;
    ulonglong2 * d = (ulonglong2 *) t.x[q]
        + (p - piece_start(t, q)) * ((size_t) t.nrows * t.ncols)
        + (size_t) (Rq.i0 + e / Rq.ncols) * t.ncols + Rq.j0 + e % Rq.ncols;
    if (to_pieces) *d = *s;
    else *s = *d;
}

/* ---- the product of transformed matrices ---- */

/* The product runs bit-sliced over the transform points: a point group
 * is 32 consecutive points, and word w of an element group holds bit w
 * of the 32 field elements, point 32g+l at bit l. A multiplication in
 * GF(2^128) becomes a carry-less 128x128 bit product on 32-bit words,
 * which is AND and XOR only, and the sum over k needs no reduction
 * until the end.
 *
 * The 128 x 128 bit product splits by three levels of Karatsuba into
 * 27 products of 16 x 16 bits: at each level a 2S-bit operand x =
 * lo + hi << S contributes the S-bit pieces lo, hi and lo + hi (sel 0,
 * 1, 2), and the piece products land at offset 0 and S, 2S and S, and
 * S. A piece of the 128-bit operand is thus the sum of the 16-bit
 * eighths e = 4 h + 2 q + r with h, q, r in the sets the three sels
 * pick, and the 16 x 16 product of two pieces at offset o1 + o2 + o3
 * for the offsets the sels allow. The slice kernels write the 27 x 16
 * piece words of each element group, so that the product kernel reads
 * its operands with no preparation; the product comes back as 128 raw
 * words per element group.
 *
 * The product runs over chunks of G point groups; the layouts, with
 * the piece p = 9 s1 + 3 s2 + s3 and its word v, are
 *     A[(((i*nadd + k)*G + g)*27 + p)*16 + v]
 *     B[((((k*27 + p)*4 + v/4)*G + g)*ncp + j)*4 + v%4]
 *     C[((i*128 + w)*G + g)*ncp + j]
 * so that a block of the product kernel copies A in one piece and,
 * with the lanes of a warp over j, B and C in 16-byte granules. ncp is
 * ncols rounded up to a multiple of 32; the pad columns of B are zero. */

/* a range of transform points: the chunk starts at point p0 */
struct pts {
    size_t p0;
    unsigned int n;
};

__host__ __device__ __forceinline__ unsigned int groups(unsigned int n)
{
    return (n + 31) / 32;
}

__device__ __forceinline__ bool sel_has(int sel, int which)
{
    return sel == 2 || sel == which;
}

/* the offsets where the piece product of a level with half-size S
 * lands: lo*lo at 0 and S, hi*hi at 2S and S, the sum piece at S */
__device__ __forceinline__ bool sel_offset(int sel, int which, int S, int & off)
{
    if (which == 1) { off = S; return sel != 2; }
    off = sel == 2 ? S : 2 * S * sel;
    return true;
}

/* the index of word v of piece p of element e = hi*n_lo + lo in group
 * g of a sliced matrix: A has (hi,lo) = (i,k), B has (k,j); lo runs to
 * n_lo, the column count of the source block rounded up for B, and
 * beyond the block (the pad columns of B) the element is zero */
struct slice_map {
    size_t s_hi, s_lo, s_g, s_p, s_v4;
    unsigned int n_lo, n_elts;
};

__device__ __forceinline__ size_t slice_index(slice_map m, unsigned int e,
        unsigned int g, unsigned int p, unsigned int v)
{
    return (e / m.n_lo) * m.s_hi + (e % m.n_lo) * m.s_lo + g * m.s_g
        + p * m.s_p + v / 4 * m.s_v4 + v % 4;
}

/* one warp per (element e, point group g) */
__device__ __forceinline__ void my_warp(unsigned int G, unsigned int & e,
        unsigned int & g, unsigned int & lane)
{
    unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    lane = threadIdx.x & 31;
    g = warp % G;
    e = warp / G;
}

/* one warp per (element, group); lane l loads point 32g+l of the
 * transform of element e, the ballot over the lanes forms the word of
 * bit w, which lands in lane w % 32, and the lanes sum the words into
 * the pieces: lane v < 16 holds the even eighths of word v (its word q
 * is eighth 2q), lane v + 16 the odd ones */
__global__ void slice_kernel(uint32_t * dst, dmat src, sub R, pts P,
        slice_map m)
{
    unsigned int e, g, lane;
    my_warp(groups(P.n), e, g, lane);
    if (e >= m.n_elts) return;
    unsigned int hi = e / m.n_lo, lo = e % m.n_lo;
    unsigned int p = 32 * g + lane;
    uint32_t bits[4] = { 0, 0, 0, 0 };
    if (p < P.n && lo < R.ncols) {
        K v = entry(src, R, hi, lo).get(P.p0 + p);
        bits[0] = v.lo; bits[1] = v.lo >> 32;
        bits[2] = v.hi; bits[3] = v.hi >> 32;
    }
    uint32_t words[4] = { 0, 0, 0, 0 };
#pragma unroll
    for (unsigned int w = 0; w < 128; w++) {
        uint32_t v = __ballot_sync(~0u, (bits[w >> 5] >> (w & 31)) & 1);
        if (lane == (w & 31)) words[w >> 5] = v;
    }
    unsigned int v = lane % 16, r = lane / 16;
#pragma unroll
    for (int s12 = 0; s12 < 9; s12++) {
        uint32_t own = 0;
#pragma unroll
        for (int q = 0; q < 4; q++)
            if (sel_has(s12 / 3, q >> 1) && sel_has(s12 % 3, q & 1))
                own ^= words[q];
        uint32_t other = __shfl_xor_sync(~0u, own, 16);
        dst[slice_index(m, e, g, 3 * s12 + r, v)] = own;
        if (r == 0) dst[slice_index(m, e, g, 3 * s12 + 2, v)] = own ^ other;
    }
}

/* one warp per (i, j, group); lane l gathers bit l of the 128 words of
 * the product and adds the element to c */
__global__ void unslice_kernel(dmat c, sub R, const uint32_t * C,
        unsigned int ncp, unsigned int G, pts P)
{
    unsigned int e, g, lane;
    my_warp(groups(P.n), e, g, lane);
    unsigned int i = e / R.ncols, j = e % R.ncols;
    unsigned int p = 32 * g + lane;
    if (i >= R.nrows || p >= P.n) return;
    uint32_t bits[4] = { 0, 0, 0, 0 };
#pragma unroll
    for (unsigned int w = 0; w < 128; w++) {
        uint32_t v = C[((size_t) i * 128 + w) * G * ncp + (size_t) g * ncp + j];
        bits[w >> 5] |= ((v >> lane) & 1) << (w & 31);
    }
    entry(c, R, i, j).add(P.p0 + p, { bits[0] | (uint64_t) bits[1] << 32,
            bits[2] | (uint64_t) bits[3] << 32 });
}

/* the asynchronous copies of 16-byte granules from global to shared
 * memory, in groups */
__device__ __forceinline__ void cp_async16(void * dst, const void * src)
{
    unsigned int s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(src));
}

__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;");
}

/* waits until at most N groups are in flight */
template<int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

/* The product kernel: one block per row i, 32 columns j and point
 * group, with 32 lanes over j and 27 warps over the pieces. Each
 * thread accumulates its 16 x 16 bit piece product over k in 31
 * registers, with the 16 words of its B piece in registers and the A
 * piece read word by word from shared memory (a broadcast). The block
 * stages the pieces of a k in shared memory with asynchronous copies
 * two steps ahead, in three stages. The epilogue sums the pieces into
 * the 255-bit product in shared memory, reduces it modulo
 * x^128 + x^7 + x^2 + x + 1, and stores it. */
struct gemm_args {
    const uint32_t * A;
    const uint32_t * B;
    uint32_t * C;
    unsigned int nadd, ncp, G;
};

#define PIECES 27
#define GEMM_THREADS (32 * PIECES)
/* B pieces of a k: word v of piece p of lane l at [((4p + v/4)*32 + l)*4 + v%4] */
#define BSTAGE_WORDS (PIECES * 16 * 32)
/* A pieces of a k: word v of piece p at [16p + v] */
#define ASTAGE_WORDS (PIECES * 16)
#define STAGE_WORDS (BSTAGE_WORDS + ASTAGE_WORDS)
#define STAGES 3
#define GEMM_SMEM (STAGES * STAGE_WORDS * sizeof(uint32_t))

__global__ void __launch_bounds__(GEMM_THREADS, 1) gemm_kernel(gemm_args g)
{
    extern __shared__ __align__(16) uint32_t smem[];
    const unsigned int lane = threadIdx.x;
    const unsigned int piece = threadIdx.y;
    const unsigned int tid = piece * 32 + lane;
    const unsigned int i = blockIdx.x;
    const unsigned int j = blockIdx.y * 32 + lane;
    const unsigned int grp = blockIdx.z;
    const int s1 = piece / 9, s2 = piece / 3 % 3, s3 = piece % 3;
    const size_t bstride = (size_t) g.G * g.ncp * 4;

    /* the granules a thread copies per k: the B granules tid + 864 m
     * are the piece rows 4p + v/4 = piece + 27 m of its lane, the A
     * granule is tid */
    const uint32_t * ap = g.A + ((size_t) i * g.nadd * g.G + grp) * ASTAGE_WORDS + tid * 4;
    const uint32_t * bp = g.B + ((size_t) grp * g.ncp + j) * 4 + piece * bstride;
    auto issue = [&](unsigned int k, unsigned int s) {
        uint32_t * bs = smem + s * STAGE_WORDS, * as = bs + BSTAGE_WORDS;
        if (k < g.nadd) {
#pragma unroll
            for (int m = 0; m < 4; m++)
                cp_async16(bs + (tid + m * GEMM_THREADS) * 4,
                        bp + ((size_t) k * 4 * PIECES + PIECES * m) * bstride);
            if (tid < ASTAGE_WORDS / 4)
                cp_async16(as + tid * 4, ap + (size_t) k * g.G * ASTAGE_WORDS);
        }
        cp_commit();
    };

    uint32_t acc[31];
#pragma unroll
    for (int s = 0; s < 31; s++) acc[s] = 0;

    issue(0, 0);
    issue(1, 1);
    for (unsigned int k = 0; k < g.nadd; k++) {
        cp_wait<1>();
        __syncthreads();
        issue(k + 2, (k + 2) % STAGES);
        const uint32_t * bs = smem + (k % STAGES) * STAGE_WORDS;
        const uint32_t * as = bs + BSTAGE_WORDS;
        uint32_t y[16];
#pragma unroll
        for (int v4 = 0; v4 < 4; v4++) {
            uint4 q = *(const uint4 *) (bs + ((piece * 4 + v4) * 32 + lane) * 4);
            y[4 * v4] = q.x; y[4 * v4 + 1] = q.y;
            y[4 * v4 + 2] = q.z; y[4 * v4 + 3] = q.w;
        }
#pragma unroll
        for (int v4 = 0; v4 < 4; v4++) {
            uint4 q = *(const uint4 *) (as + piece * 16 + v4 * 4);
            uint32_t x[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int c = 0; c < 4; c++)
#pragma unroll
                for (int w = 0; w < 16; w++) acc[4 * v4 + c + w] ^= x[c] & y[w];
        }
    }
    __syncthreads();

    /* the product of the epilogue, word t of lane l at [t*32 + l] */
    uint32_t * u = smem;
    for (unsigned int t = tid; t < 255 * 32; t += GEMM_THREADS) u[t] = 0;
    __syncthreads();
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            for (int c = 0; c < 2; c++) {
                int o1, o2, o3;
                if (!sel_offset(s1, a, 64, o1) || !sel_offset(s2, b, 32, o2)
                        || !sel_offset(s3, c, 16, o3))
                    continue;
#pragma unroll
                for (int s = 0; s < 31; s++)
                    atomicXor(&u[(o1 + o2 + o3 + s) * 32 + lane], acc[s]);
            }
        }
    }
    __syncthreads();
    /* x^128 = x^7 + x^2 + x + 1: bit t >= 128 folds to t-128, t-127,
     * t-126, t-121, so the first pass lands up to bit 134 and the
     * second folds those back; the folded words gather into registers
     * first, as the sources of one word are the destinations of others */
    for (int pass = 0; pass < 2; pass++) {
        const int top = pass == 0 ? 255 : 135;
        uint32_t v[5];
#pragma unroll
        for (int n = 0; n < 5; n++) {
            const int s = (tid + n * GEMM_THREADS) / 32;
            v[n] = 0;
            for (int d : (const int[]) { 128, 127, 126, 121 })
                if (s + d >= 128 && s + d < top) v[n] ^= u[(s + d) * 32 + lane];
        }
        __syncthreads();
#pragma unroll
        for (int n = 0; n < 5; n++) {
            const int s = (tid + n * GEMM_THREADS) / 32;
            if (s < 128) u[s * 32 + lane] ^= v[n];
            else u[s * 32 + lane] = v[n];
        }
        __syncthreads();
    }
    uint32_t * cp = g.C + ((size_t) i * 128 * g.G + grp) * g.ncp + j;
    for (unsigned int w = piece; w < 128; w += PIECES)
        cp[(size_t) w * g.G * g.ncp] = u[w * 32 + lane];
}

static unsigned int blocks_for(size_t threads)
{
    return (threads + 255) / 256;
}

/* The entries of a block of the transform kernels: 256 at most, fewer
 * (32 at least, for coalesced accesses) when the entries of the device
 * would not give every SM a few blocks. */
#define TRANSFORM_THREADS 256
static unsigned int transform_tile(size_t entries)
{
    static int sms = [] {
        int n;
        CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0));
        return n;
    }();
    unsigned int E = TRANSFORM_THREADS;
    while (E > 32 && entries / E < (size_t) 4 * sms) E /= 2;
    return E;
}

/* ---- the host side: the devices, the timing, the transfers ---- */

/* the GPUs of the node, with peer access between every pair and one
 * stream each, so that the devices run at the same time */
static cudaStream_t streams[MAX_DEV];

static int devices()
{
    static int n = [] {
        int n;
        CUDA_CHECK(cudaGetDeviceCount(&n));
        n = std::min(n, MAX_DEV);
        for (int q = 0; q < n; q++) {
            CUDA_CHECK(cudaSetDevice(q));
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[q], cudaStreamNonBlocking));
            for (int r = 0; r < n; r++)
                if (r != q) CUDA_CHECK(cudaDeviceEnablePeerAccess(r, 0));
        }
        return n;
    }();
    return n;
}

static void use(int q)
{
    devices();
    CUDA_CHECK(cudaSetDevice(q));
}

/* the stream of the current device */
static cudaStream_t stream()
{
    int q;
    CUDA_CHECK(cudaGetDevice(&q));
    return streams[q];
}

template<typename T>
static T * dev_alloc(size_t n)
{
    T * p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemsetAsync(p, 0, n * sizeof(T), stream()));
    return p;
}

template<typename T>
static T * dev_upload(const T * h, size_t n)
{
    T * d = dev_alloc<T>(n);
    CUDA_CHECK(cudaMemcpyAsync(d, h, n * sizeof(T), cudaMemcpyHostToDevice, stream()));
    return d;
}

template<typename T>
static void dev_download(T * h, const T * d, size_t n)
{
    CUDA_CHECK(cudaMemcpyAsync(h, d, n * sizeof(T), cudaMemcpyDeviceToHost, stream()));
    CUDA_CHECK(cudaStreamSynchronize(stream()));
}

/* f(q) on every device, the current device kept */
template<typename F>
static void on_devices(int n, F f)
{
    int cur;
    CUDA_CHECK(cudaGetDevice(&cur));
    for (int q = 0; q < n; q++) {
        use(q);
        f(q);
    }
    use(cur);
}

static void sync_devices(int n)
{
    on_devices(n, [](int) { CUDA_CHECK(cudaDeviceSynchronize()); });
}

/* One event per step on the current device; the report prints, per
 * device, the time of each step name when GLINGEN_STATS is set (a
 * name repeats when the work runs in chunks). start() opens the chain
 * of a device. */
/* the host time of the transfers: the packing of the rows into the
 * pinned buffer and the waits for a free slot of it */
static double host_ms[2];
enum { PACK, WAIT };
static double now_ms() { return omp_get_wtime() * 1e3; }

struct laps {
    struct ev { int dev; const char * name; cudaEvent_t e; };
    std::vector<ev> evs;
    double t0 = now_ms();
    void start() { lap(""); }
    void lap(const char * name) {
        ev v = { 0, name, {} };
        CUDA_CHECK(cudaGetDevice(&v.dev));
        CUDA_CHECK(cudaEventCreate(&v.e));
        CUDA_CHECK(cudaEventRecord(v.e, stream()));
        evs.push_back(v);
    }
    void report(const char * what) {
        if (getenv("GLINGEN_STATS")) {
            std::vector<std::pair<const char *, float>> sum[MAX_DEV];
            cudaEvent_t last[MAX_DEV];
            for (auto & v : evs) {
                if (*v.name) {
                    float ms;
                    CUDA_CHECK(cudaEventSynchronize(v.e));
                    CUDA_CHECK(cudaEventElapsedTime(&ms, last[v.dev], v.e));
                    auto & s = sum[v.dev];
                    if (s.empty() || strcmp(s.back().first, v.name)) s.push_back({ v.name, 0 });
                    s.back().second += ms;
                }
                last[v.dev] = v.e;
            }
            fprintf(stderr, "glingen: %s:", what);
            bool several = !sum[1].empty();
            for (int q = 0; q < MAX_DEV; q++) {
                if (sum[q].empty()) continue;
                if (several) fprintf(stderr, " [%d]", q);
                for (auto & s : sum[q]) fprintf(stderr, " %s %.0f ms", s.first, s.second);
            }
            fprintf(stderr, " host: pack %.0f ms wait %.0f ms wall %.0f ms\n",
                    host_ms[PACK], host_ms[WAIT], now_ms() - t0);
        }
        host_ms[PACK] = host_ms[WAIT] = 0;
        for (auto & v : evs) CUDA_CHECK(cudaEventDestroy(v.e));
    }
};

static dmat piece(const tmat & t, unsigned int q)
{
    return { t.x[q], t.nrows, t.ncols, piece_start(t, q + 1) - piece_start(t, q) };
}

/* A piece keeps at least this many points: a short transform stays on
 * one device. GLINGEN_GPUS caps the device count. (c190: the top three
 * levels of the recursion have 2^10 to 2^12 points; 8 pieces of 2^9
 * took them from 12.3 s to 5.8 s.) */
#define MIN_PIECE ((size_t) 1 << 9)

/* the devices that share a transform of 2^k points */
static unsigned int devices_for(unsigned int k)
{
    const char * env = getenv("GLINGEN_GPUS");
    unsigned int ndev = std::min<size_t>(devices(), std::max<size_t>(1, ((size_t) 1 << k) / MIN_PIECE));
    if (env) ndev = std::min<unsigned int>(ndev, atoi(env));
    assert(ndev >= 1);
    return ndev;
}

tmat * glingen_alloc(unsigned int nrows, unsigned int ncols, unsigned int k)
{
    tmat * t = new tmat { {}, nrows, ncols, devices_for(k), (size_t) 1 << k };
    on_devices(t->ndev, [&](int q) {
        t->x[q] = dev_alloc<uint64_t>((size_t) nrows * ncols * piece(*t, q).tsize * 2);
    });
    return t;
}

void glingen_free(tmat * t)
{
    on_devices(t->ndev, [&](int q) { CUDA_CHECK(cudaFree(t->x[q])); });
    delete t;
}

void glingen_zero(tmat * t, sub R)
{
    on_devices(t->ndev, [&](int q) {
        zero_kernel<<<blocks_for((size_t) R.nrows * R.ncols * piece(*t, q).tsize), 256, 0, stream()>>>(piece(*t, q), R);
        CUDA_CHECK(cudaGetLastError());
    });
}

/* The polynomials cross the host boundary through two pinned buffers,
 * in chunks of entries: the host threads pack the entries of a block
 * into one buffer while the other is in flight. The buffers grow to the size of
 * an entry when one is larger. The device copy of a block is packed,
 * ncols entries per row. */
#define PIN_BYTES ((size_t) 64 << 20)
/* the threads that pack the entries: a few dozen saturate the memory,
 * and a team of all the hardware threads stalls at its barriers */
#define PACK_THREADS 32
struct pinned {
    char * x[2] = { nullptr, nullptr };
    cudaEvent_t done[MAX_DEV][2];
    size_t cap = 0;
    pinned() {
        on_devices(devices(), [&](int q) {
            for (int s = 0; s < 2; s++) {
                CUDA_CHECK(cudaEventCreate(&done[q][s]));
                CUDA_CHECK(cudaEventRecord(done[q][s], stream()));
            }
        });
    }
    void reserve(size_t bytes) {
        if (bytes <= cap) return;
        cap = std::max(bytes, PIN_BYTES);
        sync_devices(devices());
        for (int s = 0; s < 2; s++) {
            CUDA_CHECK(cudaFreeHost(x[s]));
            CUDA_CHECK(cudaMallocHost(&x[s], cap));
        }
    }
};

static pinned & pin(size_t bytes)
{
    static pinned p;
    p.reserve(bytes);
    return p;
}

/* the chunks of the block: entries [e0, e0 + n) of the nrows x ncols
 * entries, as many as fit the pinned buffer */
struct chunks {
    size_t bytes, per, total;
    chunks(polys a, sub R, size_t cap) : bytes(a.words * sizeof(uint64_t)),
        per(cap / bytes), total((size_t) R.nrows * R.ncols) {}
    size_t count() const { return (total + per - 1) / per; }
    size_t start(size_t c) const { return c * per; }
    size_t size(size_t c) const { return std::min(per, total - start(c)); }
};

/* entry e of the block R of the host polynomials */
static uint64_t * host_entry(polys a, sub R, size_t e)
{
    sub R0 = { 0, 0, R.nrows, R.ncols };
    return entry(a, R0, e / R.ncols, e % R.ncols);
}

/* A memory policy on the pages of the block turns the kernel's NUMA
 * balancing off for them; MPOL_LOCAL is the default placement. With
 * the balancing on, the kernel unmaps the pages that were idle during
 * the CPU work between two products, and the packing threads then
 * take a fault per page (13 GB/s instead of 170). A policy of the
 * thread would not do: the scan runs from any thread of the process.
 * Where the call is not allowed (seccomp), the balancing stays on.
 * The range from the first to the last entry of the block lies in
 * the one allocation of the matrix (the rows between are its other
 * columns), so it is mapped throughout. */
static void keep_mapped(polys a, sub R, chunks C)
{
    uintptr_t page = sysconf(_SC_PAGESIZE);
    uintptr_t lo = (uintptr_t) host_entry(a, R, 0) & ~(page - 1);
    uintptr_t hi = (uintptr_t) host_entry(a, R, C.total - 1) + C.bytes;
    syscall(SYS_mbind, lo, hi - lo, MPOL_LOCAL, nullptr, 0, 0);
}

/* a slot of the pinned buffer is free when every device is done with it */
static void slot_free(pinned & P, size_t c)
{
    double t0 = now_ms();
    for (int r = 0; r < devices(); r++)
        CUDA_CHECK(cudaEventSynchronize(P.done[r][c & 1]));
    host_ms[WAIT] += now_ms() - t0;
}

/* The packing threads live for the whole block (a team per chunk
 * starts slowly when the node has other work); the calling thread,
 * which owns the device, issues the copies between the barriers. */
#define PACK_LOOP(c, C) \
    _Pragma("omp parallel num_threads(PACK_THREADS)") \
    for (size_t c = 0; c < C.count(); c++)
#define PACK_FOR(e, C, c) \
    _Pragma("omp for") \
    for (size_t e = 0; e < C.size(c); e++)

/* the block R of a to the packed device copy d, on the current device */
static void upload(polys d, polys a, sub R)
{
    pinned & P = pin(a.words * sizeof(uint64_t));
    chunks C(a, R, P.cap);
    keep_mapped(a, R, C);
    int q;
    CUDA_CHECK(cudaGetDevice(&q));
    PACK_LOOP(c, C) {
        char * x = P.x[c & 1];
        double t0 = 0;
#pragma omp master
        { slot_free(P, c); t0 = now_ms(); }
#pragma omp barrier
        PACK_FOR(e, C, c)
            memcpy(x + e * C.bytes, host_entry(a, R, C.start(c) + e), C.bytes);
#pragma omp master
        {
            host_ms[PACK] += now_ms() - t0;
            CUDA_CHECK(cudaMemcpyAsync(d.x + C.start(c) * a.words, x,
                        C.size(c) * C.bytes, cudaMemcpyHostToDevice, stream()));
            CUDA_CHECK(cudaEventRecord(P.done[q][c & 1], stream()));
        }
    }
}

static void download(polys a, polys d, sub R)
{
    pinned & P = pin(a.words * sizeof(uint64_t));
    chunks C(a, R, P.cap);
    keep_mapped(a, R, C);
    int q;
    CUDA_CHECK(cudaGetDevice(&q));
    auto fetch = [&](size_t c) {
        if (c >= C.count()) return;
        slot_free(P, c);
        CUDA_CHECK(cudaMemcpyAsync(P.x[c & 1], d.x + C.start(c) * a.words,
                    C.size(c) * C.bytes, cudaMemcpyDeviceToHost, stream()));
        CUDA_CHECK(cudaEventRecord(P.done[q][c & 1], stream()));
    };
    fetch(0);
    PACK_LOOP(c, C) {
        double t0 = 0;
#pragma omp master
        {
            fetch(c + 1);
            double t1 = now_ms();
            CUDA_CHECK(cudaEventSynchronize(P.done[q][c & 1]));
            t0 = now_ms();
            host_ms[WAIT] += t0 - t1;
        }
#pragma omp barrier
        PACK_FOR(e, C, c)
            memcpy(host_entry(a, R, C.start(c) + e), P.x[c & 1] + e * C.bytes, C.bytes);
#pragma omp master
        host_ms[PACK] += now_ms() - t0;
    }
}

/* ---- the transforms ---- */

/* Device q transforms the rows Rq of the block: their polynomials are
 * the packed device copy d (or the polynomials themselves when they
 * live on a device, which the kernels read over the peer links), and
 * the transforms stage on device q in a matrix T of whole transforms
 * (all the points). The points move from the stage to the pieces after
 * the dft, or from the pieces to the stage before the ift. With one
 * device the piece is the stage. */
struct rows {
    sub Rq, RT;
    polys d;
    dmat T;
};

static bool on_device(polys a)
{
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, a.x));
    return attr.type == cudaMemoryTypeDevice;
}

/* the polynomials of the rows Rq of the block R (a holds the block
 * from its row 0) */
static polys block_rows(polys a, sub R, sub Rq)
{
    return { a.x + (size_t) (Rq.i0 - R.i0) * a.ld * a.words, a.ld, a.words, a.bits };
}

static std::vector<rows> rows_of(const tmat & t, sub R, polys a)
{
    bool dev = on_device(a);
    std::vector<rows> W(t.ndev);
    for (unsigned int q = 0; q < t.ndev; q++) {
        use(q);
        rows & w = W[q];
        unsigned int i0 = R.nrows * q / t.ndev, i1 = R.nrows * (q + 1) / t.ndev;
        w.Rq = { R.i0 + i0, R.j0, i1 - i0, R.ncols };
        w.RT = t.ndev == 1 ? w.Rq : sub { 0, 0, w.Rq.nrows, w.Rq.ncols };
        w.d = dev ? block_rows(a, R, w.Rq) : polys {
            dev_alloc<uint64_t>((size_t) w.Rq.nrows * w.Rq.ncols * a.words),
            w.Rq.ncols, a.words, a.bits };
        w.T = t.ndev == 1 ? piece(t, 0) : dmat { dev_alloc<uint64_t>(
                (size_t) w.Rq.nrows * w.Rq.ncols * t.tsize * 2),
                w.Rq.nrows, w.Rq.ncols, t.tsize };
    }
    return W;
}

static void free_rows(const tmat & t, std::vector<rows> & W, polys a)
{
    bool dev = on_device(a);
    on_devices(t.ndev, [&](int q) {
        if (!dev) CUDA_CHECK(cudaFree(W[q].d.x));
        if (t.ndev > 1) CUDA_CHECK(cudaFree(W[q].T.x));
    });
}

/* the n significant points of the rows of w between the stage and the
 * pieces, by a kernel on the device of the stage, which reads or
 * writes the pieces of the other devices over the peer links */
static void move(const tmat & t, const rows & w, size_t n, bool to_pieces)
{
    if (t.ndev == 1) return;
    size_t threads = n * tmat_entries(w.T);
    move_kernel<<<blocks_for(threads), 256, 0, stream()>>>(t, w.T, w.Rq, threads, to_pieces);
    CUDA_CHECK(cudaGetLastError());
}

void glingen_dft(tmat * t, sub R, polys a, cantor fti)
{
    assert(fti.n <= t->tsize);
    laps L;
    bool dev = on_device(a);
    std::vector<rows> W = rows_of(*t, R, a);
    for (unsigned int q = 0; q < t->ndev; q++) {
        rows & w = W[q];
        if (!w.Rq.nrows) continue;
        use(q);
        L.start();
        if (!dev) upload(w.d, block_rows(a, R, w.Rq), w.Rq);
        L.lap("upload");
        size_t n = (size_t) w.Rq.nrows * w.Rq.ncols;
        unsigned int E = transform_tile(n);
        dft_kernel<<<(n + E - 1) / E, TRANSFORM_THREADS, 0, stream()>>>(w.T, w.RT, w.d, fti, E);
        CUDA_CHECK(cudaGetLastError());
        L.lap("dft");
        move(*t, w, fti.n, true);
        L.lap("move");
    }
    sync_devices(t->ndev);
    L.report("dft");
    free_rows(*t, W, a);
}

void glingen_ift(polys a, const tmat * t, sub R, cantor fti)
{
    laps L;
    bool dev = on_device(a);
    std::vector<rows> W = rows_of(*t, R, a);
    for (unsigned int q = 0; q < t->ndev; q++) {
        rows & w = W[q];
        if (!w.Rq.nrows) continue;
        use(q);
        L.start();
        move(*t, w, fti.n, false);
        L.lap("move");
        size_t n = (size_t) w.Rq.nrows * w.Rq.ncols;
        unsigned int E = transform_tile(n);
        ift_kernel<<<(n + E - 1) / E, TRANSFORM_THREADS, 0, stream()>>>(w.d, w.T, w.RT, fti, E);
        CUDA_CHECK(cudaGetLastError());
        L.lap("ift");
    }
    for (unsigned int q = 0; q < t->ndev && !dev; q++) {
        rows & w = W[q];
        if (!w.Rq.nrows) continue;
        use(q);
        download(block_rows(a, R, w.Rq), w.d, w.Rq);
        L.lap("download");
    }
    sync_devices(t->ndev);
    L.report("ift");
    free_rows(*t, W, a);
}

/* ---- the product ---- */

/* the sliced operands and product of a chunk take at most this much */
#define CHUNK_BYTES ((size_t) 4 << 30)

/* the product of the points [0, npts) of the pieces on the current
 * device, in chunks of G point groups */
struct product {
    uint32_t * A, * B, * C;
    gemm_args args;
    slice_map amap, bmap;
    unsigned int nrows, nadd, ncols;

    product(sub Ra, sub Rb, sub Rc, size_t npts)
        : nrows(Ra.nrows), nadd(Ra.ncols), ncols(Rb.ncols)
    {
        assert(Rb.nrows == nadd && Rc.nrows == nrows && Rc.ncols == ncols);
        unsigned int ncp = (ncols + 31) / 32 * 32;
        size_t aw = (size_t) nrows * nadd * ASTAGE_WORDS;
        size_t bw = (size_t) nadd * ncp * ASTAGE_WORDS;
        size_t cw = (size_t) nrows * ncp * 128;
        unsigned int G = std::max<size_t>(1, CHUNK_BYTES / ((aw + bw + cw) * sizeof(uint32_t)));
        G = std::min<size_t>(G, groups(npts));
        A = dev_alloc<uint32_t>(aw * G);
        B = dev_alloc<uint32_t>(bw * G);
        C = dev_alloc<uint32_t>(cw * G);
        args = { A, B, C, nadd, ncp, G };
        amap = { (size_t) nadd * G * ASTAGE_WORDS, (size_t) G * ASTAGE_WORDS,
            ASTAGE_WORDS, 16, 4, nadd, nrows * nadd };
        bmap = { (size_t) 4 * PIECES * G * ncp * 4, 4, (size_t) ncp * 4,
            (size_t) 4 * G * ncp * 4, (size_t) G * ncp * 4, ncp, nadd * ncp };
        CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_SMEM));
    }

    void run(dmat c, sub Rc, dmat a, sub Ra, dmat b, sub Rb, size_t npts, laps & L)
    {
        unsigned int G = args.G, ncp = args.ncp;
        for (size_t p0 = 0; p0 < npts; p0 += 32 * G) {
            pts P = { p0, (unsigned int) std::min<size_t>(32 * G, npts - p0) };
            unsigned int Gc = groups(P.n);
            slice_kernel<<<blocks_for((size_t) nrows * nadd * Gc * 32), 256, 0, stream()>>>(A, a, Ra, P, amap);
            CUDA_CHECK(cudaGetLastError());
            slice_kernel<<<blocks_for((size_t) nadd * ncp * Gc * 32), 256, 0, stream()>>>(B, b, Rb, P, bmap);
            CUDA_CHECK(cudaGetLastError());
            L.lap("slice");
            gemm_kernel<<<dim3(nrows, ncp / 32, Gc), dim3(32, PIECES), GEMM_SMEM, stream()>>>(args);
            CUDA_CHECK(cudaGetLastError());
            L.lap("gemm");
            unslice_kernel<<<blocks_for((size_t) nrows * ncols * Gc * 32), 256, 0, stream()>>>(c, Rc, C, ncp, G, P);
            CUDA_CHECK(cudaGetLastError());
            L.lap("unslice");
        }
    }

    void release()
    {
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B));
        CUDA_CHECK(cudaFree(C));
    }
};

/* the points of piece q below npts */
static size_t piece_points(const tmat & t, unsigned int q, size_t npts)
{
    return std::min(npts, piece_start(t, q + 1)) - std::min(npts, piece_start(t, q));
}

void glingen_addcompose(tmat * c, sub Rc, const tmat * a, sub Ra,
        const tmat * b, sub Rb, size_t npts)
{
    assert(a->ndev == c->ndev && b->ndev == c->ndev && a->tsize == c->tsize
            && b->tsize == c->tsize);
    laps L;
    std::vector<product> W;
    on_devices(c->ndev, [&](int q) {
        W.emplace_back(Ra, Rb, Rc, std::max<size_t>(piece_points(*c, q, npts), 1));
    });
    on_devices(c->ndev, [&](int q) {
        L.start();
        W[q].run(piece(*c, q), Rc, piece(*a, q), Ra, piece(*b, q), Rb,
                piece_points(*c, q, npts), L);
    });
    sync_devices(c->ndev);
    L.report("addcompose");
    on_devices(c->ndev, [&](int q) { W[q].release(); });
}

/* ---- the polynomial matrices of the recursion ----
 *
 * They live in host memory: at the top of a production run E and pi
 * are hundreds of GB each, more than the devices hold, and the
 * transforms of a product are ten times larger again. The products
 * move the block they transform and no more. */

/* the words of an entry of bits coefficients, in whole 64-bit words as
 * the lingen matpoly allocates them */
static size_t words_for(size_t bits)
{
    return std::max<size_t>(1, (bits + 63) / 64);
}

polys glingen_mp_alloc(unsigned int nrows, unsigned int ncols, size_t bits)
{
    size_t words = words_for(bits);
    uint64_t * x = (uint64_t *) calloc((size_t) nrows * ncols * words, sizeof(uint64_t));
    assert(x);
    return { x, ncols, words, bits };
}

void glingen_mp_free(polys a)
{
    free(a.x);
}

/* entry (i, j) onwards of a */
static polys at(polys a, unsigned int i, unsigned int j)
{
    return { a.x + ((size_t) i * a.ld + j) * a.words, a.ld, a.words, a.bits };
}

void glingen_mp_copy(polys dst, polys src, unsigned int nrows)
{
    assert(dst.ld == src.ld);
    size_t n = (size_t) nrows * dst.ld, w = std::min(dst.words, src.words);
#pragma omp parallel for
    for (size_t e = 0; e < n; e++)
        memcpy(dst.x + e * dst.words, src.x + e * src.words, w * sizeof(uint64_t));
}

/* Entry by entry, the words in ascending order: word w reads the source
 * words w + k / 64 and above, which no earlier word has written; so dst
 * may be src. */
void glingen_mp_rshift(polys dst, polys src, unsigned int nrows, size_t k)
{
    assert(dst.ld == src.ld && dst.bits + k <= src.bits);
    size_t n = (size_t) nrows * dst.ld, kq = k / 64, kr = k % 64;
    size_t nw = words_for(dst.bits), sw = words_for(src.bits);
#pragma omp parallel for
    for (size_t e = 0; e < n; e++) {
        const uint64_t * s = src.x + e * src.words;
        uint64_t * d = dst.x + e * dst.words;
        auto src_word = [&](size_t i) { return i < sw ? s[i] : 0; };
        for (size_t w = 0; w < dst.words; w++) {
            uint64_t lo = src_word(w + kq), hi = src_word(w + kq + 1);
            uint64_t v = kr ? lo >> kr | hi << (64 - kr) : lo;
            if (w == nw - 1 && dst.bits % 64) v &= ((uint64_t) 1 << dst.bits % 64) - 1;
            d[w] = w < nw ? v : 0;
        }
    }
}

int glingen_mp_coeff_is_zero(polys a, unsigned int nrows, size_t k)
{
    assert(k < a.bits);
    size_t n = (size_t) nrows * a.ld;
    int nonzero = 0;
#pragma omp parallel for reduction(|:nonzero)
    for (size_t e = 0; e < n; e++)
        nonzero |= a.x[e * a.words + k / 64] >> k % 64 & 1;
    return !nonzero;
}

/* one thread per word: 1 + the index of the last nonzero coefficient of
 * a coefficient-major polynomial of matrices of cw words each */
__global__ void polmat_len_kernel(const uint64_t * x, size_t n, size_t cw, unsigned long long * top)
{
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i]) atomicMax(top, (unsigned long long) i / cw + 1);
}

static size_t polmat_len(const uint64_t * x, size_t n, size_t cw)
{
    unsigned long long * top = dev_alloc<unsigned long long>(1), r;
    polmat_len_kernel<<<blocks_for(n), 256, 0, stream()>>>(x, n, cw, top);
    CUDA_CHECK(cudaGetLastError());
    dev_download(&r, top, 1);
    CUDA_CHECK(cudaFree(top));
    return r;
}

/* The transforms of a product may take this many entries per device:
 * the least free memory of the devices of the transform, less the
 * product scratch, in entries of a piece, half of them, since the dft
 * stages a whole copy of its block and the polynomials of the block
 * ride along. GLINGEN_BUDGET caps the free memory (bytes), for the
 * tests of the blocking. */
static size_t entry_budget(unsigned int k)
{
    unsigned int ndev = devices_for(k);
    size_t avail = SIZE_MAX, total;
    on_devices(ndev, [&](int) {
        size_t f;
        CUDA_CHECK(cudaMemGetInfo(&f, &total));
        avail = std::min(avail, f);
    });
    const char * env = getenv("GLINGEN_BUDGET");
    if (env) avail = std::min<size_t>(avail, atoll(env));
    assert(avail > CHUNK_BYTES);
    size_t piece_bytes = (((size_t) 1 << k) + ndev - 1) / ndev * 2 * sizeof(uint64_t);
    return (avail - CHUNK_BYTES) / piece_bytes / 2;
}

/* c = a * b in blocks: a column block of b is transformed once, and
 * the row blocks of a against it, one at a time, each with its block
 * of c. The three transforms take half of the budget: the column block
 * up to half of that, the row block of a and of c the rest. */
void glingen_mp_product(polys c, polys a, polys b, unsigned int nrows,
        unsigned int nadd, unsigned int ncols, cantor fti)
{
    assert(c.ld == ncols && a.ld == nadd && b.ld == ncols);
    size_t N = entry_budget(fti.k) / 2;
    unsigned int cb = std::max<size_t>(1, std::min<size_t>(ncols, N / 2 / nadd));
    assert((size_t) nadd * cb < N);
    unsigned int rb = std::min<size_t>(nrows, (N - (size_t) nadd * cb) / (nadd + cb));
    assert(rb >= 1);
    if (getenv("GLINGEN_STATS") && (rb < nrows || cb < ncols))
        fprintf(stderr, "glingen: product %ux%ux%u in blocks of %u rows, %u columns\n",
                nrows, nadd, ncols, rb, cb);
    for (unsigned int j0 = 0; j0 < ncols; j0 += cb) {
        unsigned int nc = std::min(cb, ncols - j0);
        tmat * tb = glingen_alloc(nadd, nc, fti.k);
        glingen_dft(tb, { 0, 0, nadd, nc }, at(b, 0, j0), fti);
        for (unsigned int i0 = 0; i0 < nrows; i0 += rb) {
            unsigned int nr = std::min(rb, nrows - i0);
            tmat * ta = glingen_alloc(nr, nadd, fti.k);
            tmat * tc = glingen_alloc(nr, nc, fti.k);
            glingen_dft(ta, { 0, 0, nr, nadd }, at(a, i0, 0), fti);
            glingen_addcompose(tc, { 0, 0, nr, nc }, ta, { 0, 0, nr, nadd },
                    tb, { 0, 0, nadd, nc }, fti.n);
            glingen_free(ta);
            glingen_ift(at(c, i0, j0), tc, { 0, 0, nr, nc }, fti);
            glingen_free(tc);
        }
        glingen_free(tb);
    }
}

/* ---- the binary base case ---- */

/* A polynomial of bit matrices in the bpack layout: coefficient k is
 * the rows x (64 * wc) bit matrix at x + k * rows * wc, word (r / 64 *
 * wc + w) * 64 + r % 64 holding the bits 64 w .. 64 w + 63 of row r.
 * Two buffers: a step reads one and writes the other. */
struct bpoly {
    uint64_t * x[2];
    unsigned int rows, wc, ncoef;
    __host__ __device__ size_t coef_words() const { return (size_t) rows * wc; }
    __host__ __device__ uint64_t * at(int s, unsigned int k, unsigned int r, unsigned int w) const {
        return x[s] + k * coef_words() + ((size_t) (r / 64) * wc + w) * 64 + r % 64;
    }
};

/* what the kernels of the base case share */
struct bc_dev {
    bpoly E, pi;
    uint64_t * Ut, * LL, * zero;     /* b x m, the bpack layout */
    unsigned int * perm, * np;       /* the pivot of the current step */
    unsigned int * delta, * done;    /* done: the step that found the generator, or UINT_MAX */
    int * lucky;
    unsigned int * nlog, * log;      /* the luck reports: 1 + b entries each */
    uint64_t * A, * T;               /* b x m rows of the pivot search, null when in shared memory */
};

static bpoly bpoly_alloc(unsigned int rows, unsigned int wc, unsigned int ncoef)
{
    bpoly P = { {}, rows, wc, ncoef };
    for (int s = 0; s < 2; s++) P.x[s] = dev_alloc<uint64_t>(P.coef_words() * ncoef);
    return P;
}

static void bpoly_free(bpoly & P)
{
    for (int s = 0; s < 2; s++) CUDA_CHECK(cudaFree(P.x[s]));
}

/* coefficient 0 of buffer 0 is the identity */
__global__ void identity_kernel(bpoly P)
{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < P.rows) *P.at(0, 0, r, r / 64) = (uint64_t) 1 << r % 64;
}

/* one thread per word of the polmat: the transpose of the matpoly E
 * (m x b), coefficient k of E_polmat[r][c] = E(c, r)[k] */
__global__ void to_polmat_kernel(bpoly P, int s, polys E)
{
    size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= P.coef_words() * P.ncoef) return;
    unsigned int k = x / P.coef_words(), y = x % P.coef_words();
    unsigned int r = y / (P.wc * 64) * 64 + y % 64, w = y / 64 % P.wc;
    uint64_t v = 0;
    for (unsigned int cb = 0; cb < 64; cb++)
        v |= (E.x[((size_t) (w * 64 + cb) * E.ld + r) * E.words + k / 64] >> k % 64 & 1) << cb;
    *P.at(s, k, r, w) = v;
}

/* one thread per word of the matpoly pi (b x b): pi(i, j)[k] = pi_polmat[k][j][i] */
__global__ void from_polmat_kernel(polys pi, bpoly P, int s)
{
    size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= (size_t) P.rows * pi.ld * pi.words) return;
    unsigned int q = x % pi.words, j = x / pi.words % pi.ld, i = x / pi.words / pi.ld;
    uint64_t v = 0;
    for (unsigned int kb = 0; kb < 64 && q * 64 + kb < pi.bits; kb++)
        v |= (*P.at(s, q * 64 + kb, j, i / 64) >> i % 64 & 1) << kb;
    pi.x[x] = v;
}

__device__ unsigned int block_sum(unsigned int v, unsigned int * red)
{
    for (unsigned int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    v = 0;
    for (unsigned int i = 0; i < blockDim.x / 32; i++) v += red[i];
    return v;
}

__device__ unsigned int warp_min(unsigned int v)
{
    for (unsigned int o = 16; o; o >>= 1) { unsigned int u = __shfl_xor_sync(~0u, v, o); v = u < v ? u : v; }
    return v;
}

__device__ uint64_t warp_xor(uint64_t v)
{
    for (unsigned int o = 16; o; o >>= 1) v ^= __shfl_xor_sync(~0u, v, o);
    return v;
}

template<typename T>
__device__ void dswap(T & a, T & b) { T t = a; a = b; b = t; }

/* bit i of v at bit 2i */
__device__ uint64_t spread(uint64_t v)
{
    v = (v | v << 16) & 0x0000ffff0000ffffull;
    v = (v | v << 8) & 0x00ff00ff00ff00ffull;
    v = (v | v << 4) & 0x0f0f0f0f0f0f0f0full;
    v = (v | v << 2) & 0x3333333333333333ull;
    return (v | v << 1) & 0x5555555555555555ull;
}

/* The priority list of bpack_view::ple (PLE<T>::PLE): the rows sorted
 * by (delta, row), except that a position keeps its own row when the
 * sorted row there has the same delta, and the rows displaced that way
 * fill the remaining positions in sorted order. p2d is the list, d2p
 * its inverse, tmp a scratch of b entries; thread r holds the G rows
 * G r .. G r + G - 1. */
template <unsigned G>
__device__ void ple_priority(const unsigned int * delta, unsigned int * p2d,
        unsigned int * d2p, unsigned int * tmp)
{
    unsigned int b = blockDim.x * G, r0 = threadIdx.x * G, holes = 0;
    for (unsigned int q = 0; q < G; q++) {
        unsigned int r = r0 + q, w = delta[r], rank = 0;
        for (unsigned int i = 0; i < b; i++) rank += delta[i] < w || (delta[i] == w && i < r);
        tmp[rank] = r;
    }
    __syncthreads();
    for (unsigned int q = 0; q < G; q++) {
        unsigned int r = r0 + q;
        bool hole = delta[tmp[r]] != delta[r];
        d2p[r] = hole;
        holes |= (unsigned int) hole << q;
    }
    __syncthreads();
    for (unsigned int q = 0; q < G; q++) {
        unsigned int r = r0 + q, w = delta[r];
        if (!(holes >> q & 1)) {
            p2d[r] = r;
        } else {
            unsigned int k = 0, pos = 0;
            for (unsigned int i = 0; i < b; i++)
                k += d2p[i] && (delta[i] < w || (delta[i] == w && i < r));
            for (unsigned int i = 0, c = 0; i < b; i++)
                if (d2p[i] && c++ == k) pos = i;
            p2d[pos] = r;
        }
    }
    __syncthreads();
    for (unsigned int q = 0; q < G; q++) d2p[p2d[r0 + q]] = r0 + q;
    __syncthreads();
}

/* the bit of row o in column col of the 64 columns of row bits c, held
 * word-major (c[k * 64 + col] has rows 64k..64k+63) so that a warp over
 * the columns reads consecutive words */
__device__ bool col_bit(const uint64_t * c, unsigned int col, unsigned int o)
{
    return c[o / 64 * 64 + col] >> o % 64 & 1;
}

/* The pivot search of step t: generator_found and binary_pivot_search
 * of lingen_qcode_binary, one block of b / G threads, thread r holding
 * the G rows at positions G r .. G r + G - 1 of E_t (A) and of the
 * transform T that takes P E_t to the upper factor, so that at the end
 * A = Ut and T = LL. A block holds at most 1024 threads, so G is 2 for
 * b above 1024. A and T are in shared memory, or in device memory (R)
 * when they do not fit there next to the rest. The same
 * pivot order as bpack_view::ple gives word for word the CPU pi. A
 * step with zero rows in E_t logs t and lucky for the host report.
 *
 * The pivot loop is sequential in the column; it runs on one warp over
 * one 64-column panel at a time, on the panel held as 64 columns of
 * row bits (C) indexed by the row's original position in the panel,
 * which the row swaps of the PLE then leave alone: a swap exchanges
 * the bookkeeping of two data positions (src: the original position
 * of the row there, delta, lucky, perm), and the priority list of
 * PLE<T>::find_pivot is kept as the CPU keeps it. An elimination adds
 * the pivot's column bits to every column the pivot row has a bit in,
 * over the rows that are not pivots yet (pv), and records those rows
 * in the pivot's column of L (Lc). The block then applies the panel to
 * the other words of A and to T at once: the pivot rows of the panel
 * form a unit lower triangular M with final = M^-1 orig, so every row
 * receives the orig pivot rows that its row of L times M^-1 selects. T
 * is kept without its diagonal (the identity at the current position,
 * which the swaps leave alone; a pivot added to a row brings its
 * diagonal bit along) and gets it back at the end. */
template <unsigned G, bool R>
__global__ void __launch_bounds__(1024) ple_kernel(bc_dev D, unsigned int t, unsigned int luck_mini)
{
    extern __shared__ uint64_t sh[];
    unsigned int b = blockDim.x * G, mb = D.E.wc, m = mb * 64, bw = b / 64, r = threadIdx.x, l = r % 32;
    uint64_t * A = R ? D.A : sh, * T = R ? D.T : A + b * mb, * Minv = R ? sh : T + b * mb, * pv = Minv + 64;
    uint64_t * C = pv + bw, * Lc = C + b;
    unsigned int * delta = (unsigned int *) (Lc + b), * lucky = delta + b;
    unsigned int * p2d = lucky + b, * d2p = p2d + b, * perm = d2p + b, * src = perm + b;
    unsigned int * red = src + b, * np = red + 32, * steps = np + 2;
    if (*D.done != UINT_MAX) return;
    unsigned int nzero = 0, nsure = 0;
    for (unsigned int q = 0; q < G; q++) {
        unsigned int ro = r * G + q;
        bool zero = true;
        for (unsigned int w = 0; w < mb; w++) {
            A[ro * mb + w] = *D.E.at(t & 1, t, ro, w);
            T[ro * mb + w] = 0;
            zero &= A[ro * mb + w] == 0;
        }
        delta[ro] = D.delta[ro];
        lucky[ro] = zero ? D.lucky[ro] + 1 : 0;
        perm[ro] = ro;
        D.lucky[ro] = lucky[ro];
        nzero += zero;
        nsure += zero && lucky[ro] >= luck_mini;
    }
    unsigned int newluck = block_sum(nzero, red);
    if (newluck) {
        unsigned int * rec = D.log + (size_t) *D.nlog * (1 + b);
        unsigned int sure = block_sum(nsure, red);
        for (unsigned int q = 0; q < G; q++) rec[1 + r * G + q] = lucky[r * G + q];
        if (r == 0) *rec = t, ++*D.nlog;
        if (newluck == b - m && sure == b - m) {
            if (r == 0) *D.done = t;
            return;
        }
    }
    ple_priority<G>(delta, p2d, d2p, src);
    if (r == 0) np[0] = 0;
    for (unsigned int pw = 0; pw < mb; pw++) {
        unsigned int np0 = np[0];
        if (G == 1) {
            uint64_t aw = A[r * mb + pw];
            for (unsigned int c = 0; c < 64; c++) {
                unsigned int v = __ballot_sync(~0u, aw >> c & 1);
                if (l == 0) ((unsigned int *) C)[(r / 64 * 64 + c) * 2 + r / 32 % 2] = v;
            }
        } else {
            /* the rows of warp k are the bits of word k of the columns */
            uint64_t aw0 = A[2 * r * mb + pw], aw1 = A[(2 * r + 1) * mb + pw];
            for (unsigned int c = 0; c < 64; c++) {
                uint64_t v0 = spread(__ballot_sync(~0u, aw0 >> c & 1));
                uint64_t v1 = spread(__ballot_sync(~0u, aw1 >> c & 1));
                if (l == 0) C[r / 32 * 64 + c] = v0 | v1 << 1;
            }
        }
        for (unsigned int q = 0; q < G; q++) Lc[r * G + q] = 0, src[r * G + q] = r * G + q;
        if (r < bw) pv[r] = np0 >= 64 * (r + 1) ? ~(uint64_t) 0 : np0 <= 64 * r ? 0 : ((uint64_t) 1 << np0 % 64) - 1;
        __syncthreads();
        if (r < 32) {
            unsigned int ii0 = np0;
            for (unsigned int jj = 0; jj < 64 && ii0 < b; jj++) {
                /* PLE<T>::find_pivot: the first position at or after ii0 whose
                 * row has bit jj, and the positions before it where delta
                 * changes (steps), 32 positions per round */
                unsigned int ii = UINT_MAX, wprev = 0;
                for (unsigned int ws = ii0; ws < b && ii == UINT_MAX; ws += 32) {
                    unsigned int p = ws + l, z = p < b ? p2d[p] : 0, wd = delta[z];
                    unsigned int up = __shfl_up_sync(~0u, wd, 1), wp = l ? up : wprev;
                    wprev = __shfl_sync(~0u, wd, 31);
                    unsigned int hit = __ballot_sync(~0u, p < b && col_bit(C, jj, src[z]));
                    unsigned int sm = __ballot_sync(~0u, p < b && p > ii0 && wd != wp);
                    if (hit) ii = ws + __ffs(hit) - 1, sm &= (1u << (ii - ws)) - 1;
                    if (l == 0) steps[(ws - ii0) / 32] = sm;
                }
                if (ii == UINT_MAX) continue;
                if (l == 0) {
                    unsigned int zii = p2d[ii], ii1 = d2p[ii0];
                    p2d[ii] = ii0;
                    p2d[ii1] = zii;
                    d2p[ii0] = ii;
                    d2p[zii] = ii1;
                    dswap(delta[ii0], delta[zii]);
                    dswap(lucky[ii0], lucky[zii]);
                    dswap(perm[ii0], perm[zii]);
                    dswap(src[ii0], src[zii]);
                    unsigned int zii0 = p2d[ii0], hole = ii;
                    p2d[ii0] = ii0;
                    d2p[ii0] = ii0;
                    for (int wn = (ii - ii0) / 32; wn >= 0; wn--)
                        for (unsigned int sm = steps[wn]; sm; sm ^= 1u << (31 - __clz(sm))) {
                            unsigned int ii2 = ii0 + wn * 32 + 31 - __clz(sm), z2 = p2d[ii2];
                            d2p[z2] = hole;
                            p2d[hole] = z2;
                            hole = ii2;
                        }
                    d2p[zii0] = hole;
                    p2d[hole] = zii0;
                    pv[src[ii0] / 64] |= (uint64_t) 1 << src[ii0] % 64;
                }
                __syncwarp();
                /* eliminate: the rows that are not pivots and have bit jj
                 * receive the pivot row, and L records them */
                unsigned int o = src[ii0], cx = ii0 - np0;
                bool hit0 = col_bit(C, l, o), hit1 = col_bit(C, l + 32, o);
                for (unsigned int k = 0; k < bw; k++) {
                    uint64_t e = C[k * 64 + jj] & ~pv[k];
                    __syncwarp();
                    if (hit0) C[k * 64 + l] ^= e;
                    if (hit1) C[k * 64 + l + 32] ^= e;
                    if (l == 0) Lc[k * 64 + cx] = e;
                }
                ii0++;
                __syncwarp();
            }
            /* row c of M^-1: e_c plus the rows c' < c that bit c' of pivot row c selects */
            for (unsigned int c = 0; c < ii0 - np0; c++) {
                unsigned int o = src[np0 + c];
                uint64_t v = 0;
                for (unsigned int i = l; i < c; i += 32) if (col_bit(Lc, i, o)) v ^= Minv[i];
                v = warp_xor(v);
                if (l == 0) Minv[c] = v | (uint64_t) 1 << c;
                __syncwarp();
            }
            if (l == 0) np[1] = ii0;
        }
        __syncthreads();
        unsigned int np1 = np[1], o[G];
        uint64_t v[G], aw[G];
        for (unsigned int q = 0; q < G; q++) {
            uint64_t lp = 0;
            o[q] = src[r * G + q];
            aw[q] = 0;
            v[q] = 0;
            for (unsigned int c = 0; c < 64; c++) {
                lp |= (uint64_t) col_bit(Lc, c, o[q]) << c;
                aw[q] |= (uint64_t) col_bit(C, c, o[q]) << c;
            }
            for (unsigned int c = 0; c < np1 - np0; c++) if (lp >> c & 1) v[q] ^= Minv[c];
        }
        /* four words of A and T at a time over the G rows, to stay
         * within the registers */
        unsigned int ws = 4 / G;
        for (unsigned int w0 = 0; w0 < mb; w0 += ws) {
            uint64_t a[4], tt[4];
            for (unsigned int q = 0; q < G; q++) {
#pragma unroll
                for (unsigned int w = 0; w < ws; w++)
                    if (w0 + w < mb) a[q * ws + w] = A[o[q] * mb + w0 + w], tt[q * ws + w] = T[o[q] * mb + w0 + w];
                for (uint64_t u = v[q]; u; u &= u - 1) {
                    unsigned int c = np0 + __ffsll(u) - 1, oc = src[c];
#pragma unroll
                    for (unsigned int w = 0; w < ws; w++)
                        if (w0 + w < mb) {
                            a[q * ws + w] ^= A[oc * mb + w0 + w];
                            tt[q * ws + w] ^= T[oc * mb + w0 + w] ^ (c / 64 == w0 + w ? (uint64_t) 1 << c % 64 : 0);
                        }
                }
            }
            __syncthreads();
            for (unsigned int q = 0; q < G; q++) {
                unsigned int ro = r * G + q;
#pragma unroll
                for (unsigned int w = 0; w < ws; w++)
                    if (w0 + w < mb) A[ro * mb + w0 + w] = w0 + w == pw ? aw[q] : a[q * ws + w], T[ro * mb + w0 + w] = tt[q * ws + w];
            }
            __syncthreads();
        }
        if (r == 0) np[0] = np1;
        __syncthreads();
    }
    unsigned int ii0 = np[0];
    for (unsigned int q = 0; q < G; q++) {
        unsigned int ro = r * G + q;
        D.delta[ro] = delta[ro] + (ro < ii0);
        D.lucky[ro] = lucky[ro];
        D.perm[ro] = perm[ro];
        if (ro < m) T[ro * mb + ro / 64] ^= (uint64_t) 1 << ro % 64;
        for (unsigned int w = 0; w < mb; w++) {
            size_t i = ((size_t) (ro / 64) * mb + w) * 64 + ro % 64;
            D.Ut[i] = A[ro * mb + w];
            D.LL[i] = T[ro * mb + w];
        }
    }
    if (r == 0) *D.np = ii0;
}

static size_t ple_shmem(unsigned int b, unsigned int mb, bool rows)
{
    return (rows * 2 * (size_t) b * mb + 64 + b / 64 + 2 * b) * sizeof(uint64_t)
        + (6 * b + 34 + b / 32) * sizeof(unsigned int);
}

template <unsigned G, bool R>
static void ple_launch(const bc_dev & D, unsigned int t, unsigned int luck_mini, size_t shmem, bool first)
{
    if (first)
        CUDA_CHECK(cudaFuncSetAttribute(ple_kernel<G, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    ple_kernel<G, R><<<1, D.E.rows / G, shmem, stream()>>>(D, t, luck_mini);
    CUDA_CHECK(cudaGetLastError());
}

/* Y_k = S LL P X_k for the coefficients k0 <= k < k1 of X, into the
 * other buffer: block (k, w) computes word w of every row of Y_k, with
 * the m rows of P X_k that LL multiplies in shared memory; row r of
 * LL P X_k is the sum of the rows c of P X_k with bit c of row r of LL,
 * plus row r of P X_k itself when r >= m. The sum runs over nibbles of
 * the LL row through a table (TB) of the 16 sums of each group of 4
 * rows. S moves the rows below np to coefficient k + 1 (dropped at the
 * end of X) and fills what it leaves: at k0 from U (the rows below np
 * of Y_{k0 - 1}; null when a previous launch wrote them), at k1 with
 * zeros. Thread r holds the G rows G r .. G r + G - 1. */
template <unsigned G>
__global__ void bc_step_kernel(bc_dev D, bpoly X, int s, unsigned int k0, unsigned int k1,
        const uint64_t * U)
{
    extern __shared__ uint64_t PX[];
    if (*D.done != UINT_MAX) return;
    unsigned int k = k0 + blockIdx.x, w = blockIdx.y, r = threadIdx.x, np = *D.np;
    unsigned int m = D.E.wc * 64, mb = D.E.wc;
    uint64_t * TB = PX + m;
    uint64_t x[G];
    for (unsigned int q = 0; q < G; q++) {
        unsigned int ro = r * G + q;
        x[q] = *X.at(s, k, D.perm[ro], w);
        if (ro < m) PX[ro] = x[q];
    }
    __syncthreads();
    for (unsigned int e = r; e < 4 * m; e += blockDim.x) {
        uint64_t v = 0;
        for (unsigned int i = 0; i < 4; i++) if (e >> i & 1) v ^= PX[e / 16 * 4 + i];
        TB[e] = v;
    }
    __syncthreads();
    for (unsigned int q = 0; q < G; q++) {
        unsigned int ro = r * G + q;
        uint64_t y = ro < m ? 0 : x[q];
        for (unsigned int cw = 0; cw < mb; cw++) {
            uint64_t l = D.LL[((size_t) (ro / 64) * mb + cw) * 64 + ro % 64];
            for (unsigned int n = 0; n < 16; n++) y ^= TB[(cw * 16 + n) * 16 + (l >> 4 * n & 15)];
        }
        unsigned int kd = k + (ro < np);
        if (kd < X.ncoef) *X.at(!s, kd, ro, w) = y;
        if (k == k0 && ro < np && U) *X.at(!s, k, ro, w) = U[((size_t) (ro / 64) * X.wc + w) * 64 + ro % 64];
        if (k + 1 == k1 && k1 < X.ncoef && ro >= np) *X.at(!s, k1, ro, w) = 0;
    }
}

static void bc_step(const bc_dev & D, bpoly X, int s, unsigned int k0, unsigned int k1,
        const uint64_t * U)
{
    if (k0 >= k1) return;
    size_t smem = D.E.wc * 64 * 5 * sizeof(uint64_t);
    if (X.rows <= 1024)
        bc_step_kernel<1><<<dim3(k1 - k0, X.wc), X.rows, smem, stream()>>>(D, X, s, k0, k1, U);
    else
        bc_step_kernel<2><<<dim3(k1 - k0, X.wc), X.rows / 2, smem, stream()>>>(D, X, s, k0, k1, U);
    CUDA_CHECK(cudaGetLastError());
}

/* The whole loop is queued at once: step t is the pivot search on E_t,
 * then the update of the coefficients after t and of pi; the kernels
 * after the step that finds the generator do nothing. */
polys glingen_basecase(glingen_bc * bc)
{
    unsigned int m = bc->m, b = bc->b, L = bc->E.bits;
    assert(m % 64 == 0 && b % 64 == 0 && b <= 2048 && bc->E.ld == b);
    use(0);
    unsigned int mb = m / 64, none = UINT_MAX, zero = 0;
    size_t bm = (size_t) b * mb;
    /* pi has at most L + 1 coefficients; one more takes the rows S drops */
    bc_dev D = { bpoly_alloc(b, mb, L), bpoly_alloc(b, b / 64, L + 2),
        dev_alloc<uint64_t>(bm), dev_alloc<uint64_t>(bm), dev_alloc<uint64_t>(bm),
        dev_alloc<unsigned int>(b), dev_alloc<unsigned int>(1),
        dev_upload(bc->delta, b), dev_upload(&none, 1), dev_upload(bc->lucky, b),
        dev_upload(&zero, 1), dev_alloc<unsigned int>((size_t) L * (1 + b)), nullptr, nullptr };
    int smax;
    CUDA_CHECK(cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    /* the rows go to device memory when they do not fit next to the rest */
    if (ple_shmem(b, mb, true) > (size_t) smax)
        D.A = dev_alloc<uint64_t>(bm), D.T = dev_alloc<uint64_t>(bm);
    CUDA_CHECK(cudaMemsetAsync(D.zero, 0, bm * sizeof(uint64_t), stream()));
    polys E = { dev_upload(bc->E.x, (size_t) m * b * bc->E.words), b, bc->E.words, L };
    to_polmat_kernel<<<blocks_for(D.E.coef_words() * L), 256, 0, stream()>>>(D.E, 0, E);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream()));
    CUDA_CHECK(cudaFree(E.x));
    identity_kernel<<<blocks_for(b), 256, 0, stream()>>>(D.pi);
    CUDA_CHECK(cudaGetLastError());
    unsigned int G = b <= 1024 ? 1 : 2;
    bool R = D.A;
    int sdef;
    CUDA_CHECK(cudaDeviceGetAttribute(&sdef, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    if ((size_t) m * 5 * sizeof(uint64_t) > (size_t) sdef) {
        if (G == 1)
            CUDA_CHECK(cudaFuncSetAttribute(bc_step_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, m * 5 * sizeof(uint64_t)));
        else
            CUDA_CHECK(cudaFuncSetAttribute(bc_step_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, m * 5 * sizeof(uint64_t)));
    }
    size_t shmem = ple_shmem(b, mb, !R);
    for (unsigned int t = 0; t < L; t++) {
        int s = t & 1;
        if (G == 1 && !R) ple_launch<1, false>(D, t, bc->luck_mini, shmem, t == 0);
        if (G == 1 && R) ple_launch<1, true>(D, t, bc->luck_mini, shmem, t == 0);
        if (G == 2 && !R) ple_launch<2, false>(D, t, bc->luck_mini, shmem, t == 0);
        if (G == 2 && R) ple_launch<2, true>(D, t, bc->luck_mini, shmem, t == 0);
        bc_step(D, D.E, s, t + 1, L, D.Ut);
        bc_step(D, D.pi, s, 0, t + 2, D.zero);
    }
    dev_download(&bc->t, D.done, 1);
    if (bc->t == UINT_MAX) bc->t = L;
    dev_download(bc->delta, D.delta, b);
    dev_download(bc->lucky, D.lucky, b);
    dev_download(&bc->nlog, D.nlog, 1);
    dev_download(bc->log, D.log, (size_t) bc->nlog * (1 + b));
    int s = bc->t & 1;
    size_t cw = D.pi.coef_words(), n = cw * D.pi.ncoef;
    size_t len = polmat_len(D.pi.x[s], n, cw);
    polys pi = glingen_mp_alloc(b, b, len);
    size_t npi = (size_t) b * b * pi.words;
    polys dpi = { dev_alloc<uint64_t>(npi), b, pi.words, len };
    from_polmat_kernel<<<blocks_for(npi), 256, 0, stream()>>>(dpi, D.pi, s);
    CUDA_CHECK(cudaGetLastError());
    dev_download(pi.x, dpi.x, npi);
    CUDA_CHECK(cudaFree(dpi.x));
    bpoly_free(D.E);
    bpoly_free(D.pi);
    for (void * p : { (void *) D.Ut, (void *) D.LL, (void *) D.zero, (void *) D.perm, (void *) D.np,
            (void *) D.delta, (void *) D.done, (void *) D.lucky, (void *) D.nlog, (void *) D.log,
            (void *) D.A, (void *) D.T })
        CUDA_CHECK(cudaFree(p));
    return pi;
}
