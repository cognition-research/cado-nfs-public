/* gps1_sopt: CUDA port of cado-nfs size optimization (size_optimization.cpp).
 *
 * A prep kernel finds the translation list of each candidate; an item kernel
 * then gives one (candidate, translation) work item to one thread, so one
 * flush offers many more threads than candidates. The port follows the
 * CPU reference step by step: translation search (sopt_find_translations),
 * integer LLL reduction (lll.c), best_norm2, local descent
 * (sopt_local_descent), and the expected rotation gain (auxiliary.cpp,
 * polyselect_alpha.cpp, polyselect_norms.cpp). Integer arithmetic uses a
 * fixed-width sign-magnitude type that mirrors the mpz calls of the
 * reference; an overflow or an iteration cap sets the fail flag of the
 * candidate and the host redoes it on the CPU.
 *
 * The double arithmetic mirrors the CPU sequence of operations. This file
 * must be compiled with -fmad=false so that mul and add do not contract to
 * fma. Device log() can differ from glibc log() by one ulp, so a decision
 * that compares two almost equal lognorms can differ from the CPU; the
 * self-test in gps1.cpp accepts such a result when its lognorm is not worse.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <vector>

#include <omp.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "gps1_sopt.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

/* caps that replace unbounded loops or recursion of the CPU reference; when
 * one is hit the candidate fails and the host falls back to the CPU */
#define SOPT_GPU_MAX_LIST_K 4096
#define SOPT_GPU_MAX_BN2_DEPTH 32
#define SOPT_GPU_MAX_VAL0_DEPTH 12
#define SOPT_GPU_MAX_DOUBLINGS 4096
#define SOPT_GPU_MAX_LLL_ITERS 100000
#define SOPT_GPU_CHUNK 4096            /* pairs per kernel launch */
#define SOPT_GPU_MAX_SLOTS 65536       /* scratch slots (threads) per launch */

/* constants from size_optimization.hpp and auxiliary.hpp */
#define SOPT_DEFAULT_MAX_STEPS 300
#define SOPT_MAX_VALUE_FOR_Q_ROOTS 1e10
#define SOPT_NB_RAT_APPROX_OF_Q_ROOTS 16
#define SOPT_MAX_DEN_IN_RAT_APPROX_OF_Q_ROOTS 100.0
#define SOPT_MAX_DEGREE_ROTATION 7
#define SOPT_LOCAL_DESCENT_GUARD 0.001
#define NORM_MARGIN 0.2

#define ZL SOPT_GPU_LIMBS
#define MAXD SOPT_GPU_MAXDEG
#define ZP_COEFFS (MAXD + 1)
#define DP_COEFFS 10
#define LLL_DIM (MAXD + 2)     /* rows <= MAXD, cols <= MAXD + 1, 1-based */
#define SYL_DIM (2 * MAXD - 1)

struct zpoly { int deg; sopt_z c[ZP_COEFFS]; };
struct dpoly { int deg; double c[DP_COEFFS]; };

/* one frame of the best_norm2 recursion, made explicit */
struct bn2_frame {
    double roots[3];
    int nroots, idx;
    double min_norm;
    sopt_z k, best_k;
};

/* one frame of the special_val0 recursion, made explicit */
struct val0_frame {
    zpoly g, H;
    double v;
    uint32_t roots[MAXD];
    int nroots, idx;
    uint32_t r0;
};

/* per-thread workspace in global memory (too big for the stack).
 * Every k in list_k comes from a rounded double, so it is exactly a
 * double and the sort and the dedup can stay in double arithmetic. */
struct sopt_scratch {
    int fail;
    sopt_z B[LLL_DIM][LLL_DIM + 1];       /* LLL basis, 1-based */
    sopt_z lam[LLL_DIM][LLL_DIM];
    sopt_z D[LLL_DIM];
    sopt_z M[SYL_DIM][SYL_DIM];           /* Sylvester matrix for disc */
    val0_frame val0[SOPT_GPU_MAX_VAL0_DEPTH];
    bn2_frame bn2[SOPT_GPU_MAX_BN2_DEPTH];
    zpoly bn2_f[SOPT_GPU_MAX_BN2_DEPTH], bn2_g[SOPT_GPU_MAX_BN2_DEPTH];
    double list_k[SOPT_GPU_MAX_LIST_K];
    double list_k_opt[SOPT_GPU_MAX_LIST_K];
    zpoly f_raw, g_raw, ft, gt, fld, gld, fbest, gbest, tmpf, bn_ft, bn_gt;
};

/* ------------------------------------------------------------- integers */
/* sign-magnitude, little-endian 32-bit limbs; n < 0 marks an overflow that
 * propagates through every later operation (checked once at the end) */

static __host__ __device__ int z_bad(const sopt_z *a) { return a->n < 0; }

static __host__ __device__ void z_poison(sopt_z *r) { r->n = -1; r->sign = 0; }

static __host__ __device__ void z_normalize(sopt_z *r)
{
    if (r->n < 0)
        return;
    while (r->n > 0 && r->l[r->n - 1] == 0)
        r->n--;
    if (r->n == 0)
        r->sign = 0;
}

static __host__ __device__ void z_set_u32(sopt_z *r, uint32_t v)
{
    r->n = v ? 1 : 0;
    r->sign = v ? 1 : 0;
    r->l[0] = v;
}

static __host__ __device__ void z_set_u64(sopt_z *r, uint64_t v)
{
    r->l[0] = (uint32_t) v;
    r->l[1] = (uint32_t) (v >> 32);
    r->n = r->l[1] ? 2 : (r->l[0] ? 1 : 0);
    r->sign = r->n ? 1 : 0;
}

static __host__ __device__ void z_set_i64(sopt_z *r, int64_t v)
{
    z_set_u64(r, v < 0 ? (uint64_t) (-v) : (uint64_t) v);
    if (v < 0)
        r->sign = -1;
}

static __host__ __device__ void z_set(sopt_z *r, const sopt_z *a)
{
    r->n = a->n;
    r->sign = a->sign;
    for (int i = 0; i < a->n; i++)
        r->l[i] = a->l[i];
}

static __host__ __device__ int mag_cmp(const sopt_z *a, const sopt_z *b)
{
    if (a->n != b->n)
        return a->n < b->n ? -1 : 1;
    for (int i = a->n - 1; i >= 0; i--)
        if (a->l[i] != b->l[i])
            return a->l[i] < b->l[i] ? -1 : 1;
    return 0;
}

static __host__ __device__ int z_cmp(const sopt_z *a, const sopt_z *b)
{
    if (a->sign != b->sign)
        return a->sign < b->sign ? -1 : 1;
    int c = mag_cmp(a, b);
    return a->sign < 0 ? -c : c;
}

static __host__ __device__ void z_neg(sopt_z *r, const sopt_z *a)
{
    z_set(r, a);
    r->sign = -r->sign;
}

/* r = |a| + |b| */
static __host__ __device__ void mag_add(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    const sopt_z *x = a->n >= b->n ? a : b;
    const sopt_z *y = a->n >= b->n ? b : a;
    uint64_t carry = 0;
    int i = 0;
    for (; i < y->n; i++) {
        uint64_t t = (uint64_t) x->l[i] + y->l[i] + carry;
        r->l[i] = (uint32_t) t;
        carry = t >> 32;
    }
    for (; i < x->n; i++) {
        uint64_t t = (uint64_t) x->l[i] + carry;
        r->l[i] = (uint32_t) t;
        carry = t >> 32;
    }
    if (carry) {
        if (i >= ZL) {
            z_poison(r);
            return;
        }
        r->l[i++] = (uint32_t) carry;
    }
    r->n = i;
}

/* r = |a| - |b|, requires |a| >= |b| */
static __host__ __device__ void mag_sub(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    int64_t borrow = 0;
    int i = 0;
    for (; i < b->n; i++) {
        int64_t t = (int64_t) a->l[i] - b->l[i] - borrow;
        borrow = t < 0;
        r->l[i] = (uint32_t) t;
    }
    for (; i < a->n; i++) {
        int64_t t = (int64_t) a->l[i] - borrow;
        borrow = t < 0;
        r->l[i] = (uint32_t) t;
    }
    r->n = a->n;
    z_normalize(r);
}

static __host__ __device__ void z_add(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    if (z_bad(a) || z_bad(b)) { z_poison(r); return; }
    if (a->sign == 0) { z_set(r, b); return; }
    if (b->sign == 0) { z_set(r, a); return; }
    if (a->sign == b->sign) {
        int s = a->sign;
        mag_add(r, a, b);
        if (r->n >= 0)
            r->sign = s;
        return;
    }
    int c = mag_cmp(a, b);
    if (c == 0) { z_set_u32(r, 0); return; }
    int s = c > 0 ? a->sign : b->sign;
    if (c > 0)
        mag_sub(r, a, b);
    else
        mag_sub(r, b, a);
    if (r->n > 0)
        r->sign = s;
}

static __host__ __device__ void z_sub(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    sopt_z nb;
    z_neg(&nb, b);
    z_add(r, a, &nb);
}

static __host__ __device__ void z_mul(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    if (z_bad(a) || z_bad(b)) { z_poison(r); return; }
    if (a->n == 0 || b->n == 0) { z_set_u32(r, 0); return; }
    if (a->n + b->n > ZL) { z_poison(r); return; }
    sopt_z t;
    for (int i = 0; i < a->n + b->n; i++)
        t.l[i] = 0;
    for (int i = 0; i < a->n; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < b->n; j++) {
            uint64_t u = (uint64_t) a->l[i] * b->l[j] + t.l[i + j] + carry;
            t.l[i + j] = (uint32_t) u;
            carry = u >> 32;
        }
        t.l[i + b->n] = (uint32_t) carry;
    }
    t.n = a->n + b->n;
    t.sign = a->sign * b->sign;
    z_normalize(&t);
    z_set(r, &t);
}

static __host__ __device__ void z_addmul(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    sopt_z t;
    z_mul(&t, a, b);
    z_add(r, r, &t);
}

static __host__ __device__ void z_submul(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    sopt_z t;
    z_mul(&t, a, b);
    z_sub(r, r, &t);
}

static __host__ __device__ void z_mul_u32(sopt_z *r, const sopt_z *a, uint32_t v)
{
    sopt_z t;
    z_set_u32(&t, v);
    z_mul(r, a, &t);
}

static __host__ __device__ void z_mul_2exp(sopt_z *r, const sopt_z *a, int e)
{
    if (z_bad(a)) { z_poison(r); return; }
    int limbs = e / 32, bits = e % 32;
    if (a->n == 0) { z_set_u32(r, 0); return; }
    if (a->n + limbs + 1 > ZL) { z_poison(r); return; }
    sopt_z t;
    t.l[a->n + limbs] = 0;
    for (int i = a->n - 1; i >= 0; i--) {
        uint64_t u = (uint64_t) a->l[i] << bits;
        t.l[i + limbs + 1] |= (uint32_t) (u >> 32);
        t.l[i + limbs] = (uint32_t) u;
    }
    for (int i = 0; i < limbs; i++)
        t.l[i] = 0;
    t.n = a->n + limbs + 1;
    t.sign = a->sign;
    z_normalize(&t);
    z_set(r, &t);
}

/* shift right, truncation toward zero of the magnitude */
static __host__ __device__ void mag_shr(sopt_z *r, const sopt_z *a, int e)
{
    int limbs = e / 32, bits = e % 32;
    if (limbs >= a->n) { z_set_u32(r, 0); return; }
    int s = a->sign;
    sopt_z t;
    for (int i = 0; i + limbs < a->n; i++) {
        uint64_t u = a->l[i + limbs];
        if (bits && i + limbs + 1 < a->n)
            u |= (uint64_t) a->l[i + limbs + 1] << 32;
        t.l[i] = (uint32_t) (u >> bits);
    }
    t.n = a->n - limbs;
    t.sign = s;
    z_normalize(&t);
    z_set(r, &t);
}

/* mpz_tdiv_q_2exp: quotient rounded toward zero */
static __host__ __device__ void z_tdiv_q_2exp(sopt_z *r, const sopt_z *a, int e)
{
    if (z_bad(a)) { z_poison(r); return; }
    mag_shr(r, a, e);
}

/* count of leading zero bits of a nonzero 32-bit word */
static __host__ __device__ int clz32(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __clz(x);
#else
    return __builtin_clz(x);
#endif
}

static __host__ __device__ int z_bitlen(const sopt_z *a)
{
    if (a->n == 0)
        return 0;
    return 32 * a->n - clz32(a->l[a->n - 1]);
}

static __host__ __device__ int mag_tstbit(const sopt_z *a, int b)
{
    return (a->l[b / 32] >> (b % 32)) & 1;
}

/* mpz_fdiv_q_2exp (the mpz_div_2exp of the reference): floor */
static __host__ __device__ void z_fdiv_q_2exp(sopt_z *r, const sopt_z *a, int e)
{
    if (z_bad(a)) { z_poison(r); return; }
    int inexact = 0;
    if (a->sign < 0) {
        int lim = e < z_bitlen(a) ? e : z_bitlen(a);
        for (int b = 0; b < lim && !inexact; b++)
            inexact = mag_tstbit(a, b);
    }
    mag_shr(r, a, e);
    if (inexact) {
        sopt_z one;
        z_set_u32(&one, 1);
        z_sub(r, r, &one);
    }
}

/* magnitude division, Knuth algorithm D; q and r may be NULL */
static __host__ __device__ void mag_divrem(sopt_z *q, sopt_z *r, const sopt_z *a,
                                  const sopt_z *b)
{
    int m = a->n, n = b->n;
    if (mag_cmp(a, b) < 0) {
        if (r) {
            sopt_z t;
            z_set(&t, a);
            t.sign = t.n ? 1 : 0;
            z_set(r, &t);
        }
        if (q) z_set_u32(q, 0);
        return;
    }
    if (n == 1) {
        uint32_t d = b->l[0];
        uint64_t rem = 0;
        sopt_z tq;
        for (int i = m - 1; i >= 0; i--) {
            uint64_t cur = (rem << 32) | a->l[i];
            tq.l[i] = (uint32_t) (cur / d);
            rem = cur % d;
        }
        tq.n = m;
        tq.sign = 1;
        z_normalize(&tq);
        if (q) z_set(q, &tq);
        if (r) z_set_u64(r, rem);
        return;
    }
    /* normalize so that the top limb of the divisor has its high bit set */
    int sh = clz32(b->l[n - 1]);
    uint32_t u[ZL + 1], v[ZL];
    uint64_t carry = 0;
    for (int i = 0; i < m; i++) {
        uint64_t t = ((uint64_t) a->l[i] << sh) | carry;
        u[i] = (uint32_t) t;
        carry = t >> 32;
    }
    u[m] = (uint32_t) carry;
    carry = 0;
    for (int i = 0; i < n; i++) {
        uint64_t t = ((uint64_t) b->l[i] << sh) | carry;
        v[i] = (uint32_t) t;
        carry = t >> 32;
    }
    sopt_z tq;
    for (int i = 0; i <= m - n; i++)
        tq.l[i] = 0;
    for (int j = m - n; j >= 0; j--) {
        uint64_t num = ((uint64_t) u[j + n] << 32) | u[j + n - 1];
        uint64_t qhat = num / v[n - 1];
        uint64_t rhat = num % v[n - 1];
        while (qhat >> 32 ||
               qhat * v[n - 2] > ((rhat << 32) | u[j + n - 2])) {
            qhat--;
            rhat += v[n - 1];
            if (rhat >> 32)
                break;
        }
        /* multiply and subtract */
        int64_t borrow = 0;
        uint64_t mc = 0;
        for (int i = 0; i < n; i++) {
            uint64_t p = qhat * v[i] + mc;
            mc = p >> 32;
            int64_t t = (int64_t) u[i + j] - (uint32_t) p - borrow;
            borrow = t < 0;
            u[i + j] = (uint32_t) t;
        }
        int64_t t = (int64_t) u[j + n] - (int64_t) mc - borrow;
        u[j + n] = (uint32_t) t;
        if (t < 0) {
            /* add back */
            qhat--;
            uint64_t c2 = 0;
            for (int i = 0; i < n; i++) {
                uint64_t s = (uint64_t) u[i + j] + v[i] + c2;
                u[i + j] = (uint32_t) s;
                c2 = s >> 32;
            }
            u[j + n] += (uint32_t) c2;
        }
        tq.l[j] = (uint32_t) qhat;
    }
    tq.n = m - n + 1;
    tq.sign = 1;
    z_normalize(&tq);
    if (q) z_set(q, &tq);
    if (r) {
        sopt_z tr;
        carry = 0;
        for (int i = n - 1; i >= 0; i--) {
            uint64_t t = ((uint64_t) carry << 32) | u[i];
            tr.l[i] = (uint32_t) (t >> sh);
            carry = u[i] & ((1u << sh) - 1);
            if (sh == 0)
                carry = 0;
        }
        tr.n = n;
        tr.sign = 1;
        z_normalize(&tr);
        z_set(r, &tr);
    }
}

/* truncated division: q rounded toward zero, r has the sign of a */
static __host__ __device__ void z_tdiv_qr(sopt_z *q, sopt_z *r, const sopt_z *a,
                                 const sopt_z *b)
{
    if (z_bad(a) || z_bad(b) || b->n == 0) {
        if (q) z_poison(q);
        if (r) z_poison(r);
        return;
    }
    int sa = a->sign, sq = a->sign * b->sign;
    sopt_z tq, tr;
    mag_divrem(&tq, &tr, a, b);
    if (tq.n > 0) tq.sign = sq;
    if (tr.n > 0) tr.sign = sa;
    if (q) z_set(q, &tq);
    if (r) z_set(r, &tr);
}

/* floor division: r has the sign of b */
static __host__ __device__ void z_fdiv_qr(sopt_z *q, sopt_z *r, const sopt_z *a,
                                 const sopt_z *b)
{
    sopt_z tq, tr;
    z_tdiv_qr(&tq, &tr, a, b);
    if (!z_bad(&tq) && tr.n > 0 && tr.sign != b->sign) {
        sopt_z one;
        z_set_u32(&one, 1);
        z_sub(&tq, &tq, &one);
        z_add(&tr, &tr, b);
    }
    if (q) z_set(q, &tq);
    if (r) z_set(r, &tr);
}

static __host__ __device__ void z_divexact(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    z_tdiv_qr(r, NULL, a, b);
}

static __host__ __device__ int z_divisible_p(const sopt_z *a, const sopt_z *b)
{
    sopt_z t;
    z_tdiv_qr(NULL, &t, a, b);
    return !z_bad(&t) && t.n == 0;
}

static __host__ __device__ void z_gcd(sopt_z *r, const sopt_z *a, const sopt_z *b)
{
    if (z_bad(a) || z_bad(b)) { z_poison(r); return; }
    sopt_z x, y;
    z_set(&x, a); x.sign = x.n ? 1 : 0;
    z_set(&y, b); y.sign = y.n ? 1 : 0;
    while (y.n > 0) {
        sopt_z t;
        mag_divrem(NULL, &t, &x, &y);
        z_set(&x, &y);
        z_set(&y, &t);
    }
    z_set(r, &x);
}

/* extended gcd, g > 0, g = x*a + y*b */
static __host__ __device__ void z_gcdext(sopt_z *g, sopt_z *x, sopt_z *y,
                                const sopt_z *a, const sopt_z *b)
{
    sopt_z r0, r1, s0, s1, t0, t1, q, t;
    z_set(&r0, a);
    z_set(&r1, b);
    z_set_u32(&s0, 1); z_set_u32(&s1, 0);
    z_set_u32(&t0, 0); z_set_u32(&t1, 1);
    while (r1.n > 0 && !z_bad(&r1)) {
        z_tdiv_qr(&q, &t, &r0, &r1);
        z_set(&r0, &r1); z_set(&r1, &t);
        z_set(&t, &s1); z_mul(&s1, &q, &s1); z_sub(&s1, &s0, &s1); z_set(&s0, &t);
        z_set(&t, &t1); z_mul(&t1, &q, &t1); z_sub(&t1, &t0, &t1); z_set(&t0, &t);
    }
    if (r0.sign < 0) {
        z_neg(&r0, &r0);
        z_neg(&s0, &s0);
        z_neg(&t0, &t0);
    }
    z_set(g, &r0);
    z_set(x, &s0);
    z_set(y, &t0);
}

/* mpz_sizeinbase(z, 2): 1 for zero */
static __host__ __device__ int z_sizeinbase2(const sopt_z *a)
{
    int b = z_bitlen(a);
    return b ? b : 1;
}

/* mpz_get_d: truncation toward zero */
static __host__ __device__ double z_get_d(const sopt_z *a)
{
    if (z_bad(a))
        return nan("");
    if (a->n == 0)
        return 0.0;
    int b = z_bitlen(a);
    /* collect the top 64 bits into t, with the msb at bit 63 */
    uint64_t t = 0;
    int got = 0;
    for (int i = a->n - 1; i >= 0 && got < 64; i--) {
        t = (t << 32) | a->l[i];
        got += 32;
    }
    int tb = ((b - 1) % 32) + (got - 32);   /* index of the msb inside t */
    if (tb < 63)
        t <<= 63 - tb;
    uint64_t mant = t >> 11;                /* 53 bits, truncated */
    double d = ldexp((double) mant, b - 53);
    return a->sign < 0 ? -d : d;
}

/* mpz_get_d_2exp: d in [0.5, 1), truncated mantissa */
static __host__ __device__ double z_get_d_2exp(long *e, const sopt_z *a)
{
    if (a->n == 0) {
        *e = 0;
        return 0.0;
    }
    int b = z_bitlen(a);
    double d = z_get_d(a);
    *e = b;
    return ldexp(d, -b);
}

/* mpz_get<double>(z, scale) of the reference: value * 2^scale */
static __host__ __device__ double z_get_d_scaled(const sopt_z *a, int scale)
{
    long e;
    double d = z_get_d_2exp(&e, a);
    return ldexp(d, (int) e + scale);
}

/* mpz_set_d: truncation toward zero */
static __host__ __device__ void z_set_d(sopt_z *r, double x)
{
    double ax = fabs(x);
    if (!(ax >= 1.0)) {
        z_set_u32(r, 0);
        return;
    }
    if (isinf(ax) || ax >= ldexp(1.0, 32 * ZL)) {
        z_poison(r);
        return;
    }
    int e;
    double m = frexp(ax, &e);               /* ax = m * 2^e, m in [0.5,1) */
    uint64_t mant = (uint64_t) ldexp(m, 53);
    sopt_z t;
    z_set_u64(&t, mant);
    if (e - 53 >= 0)
        z_mul_2exp(r, &t, e - 53);
    else
        z_tdiv_q_2exp(r, &t, 53 - e);
    if (x < 0)
        r->sign = -r->sign;
}

/* ------------------------------------------------------- integer polys */

static __host__ __device__ void zp_set(zpoly *r, const zpoly *a)
{
    r->deg = a->deg;
    for (int i = 0; i <= a->deg; i++)
        z_set(&r->c[i], &a->c[i]);
}

static __host__ __device__ int zp_bad(const zpoly *a)
{
    for (int i = 0; i <= a->deg; i++)
        if (z_bad(&a->c[i]))
            return 1;
    return 0;
}

static __host__ __device__ void zp_cleandeg(zpoly *r, int deg)
{
    while (deg >= 0 && r->c[deg].n == 0)
        deg--;
    r->deg = deg;
}

/* mpz_poly_translation: ft = f(x + k) */
static __host__ __device__ void zp_translation(zpoly *ft, const zpoly *f,
                                      const sopt_z *k)
{
    int d = f->deg;
    zp_set(ft, f);
    for (int i = d - 1; i >= 0; i--)
        for (int j = i; j < d; j++)
            z_addmul(&ft->c[j], &ft->c[j + 1], k);
}

/* mpz_poly_rotation: fr = f + k * x^t * g */
static __host__ __device__ void zp_rotation(zpoly *fr, const zpoly *f, const zpoly *g,
                                   const sopt_z *k, int t)
{
    zp_set(fr, f);
    for (int i = fr->deg + 1; i <= t + g->deg; i++)
        z_set_u32(&fr->c[i], 0);
    for (int i = 0; i <= g->deg; i++)
        z_addmul(&fr->c[i + t], &g->c[i], k);
    int nd = fr->deg > t + g->deg ? fr->deg : t + g->deg;
    zp_cleandeg(fr, nd);
}

static __host__ __device__ void zp_content(sopt_z *c, const zpoly *f)
{
    z_set(c, &f->c[0]);
    for (int i = 1; i <= f->deg; i++)
        z_gcd(c, c, &f->c[i]);
    if (c->sign < 0)
        c->sign = 1;
}

/* ------------------------------------------------------- double polys */
/* port of double_poly.cpp (plain mul and add, no fma; needs -fmad=false) */

#define DP_MAX_ITERS 5000

static __host__ __device__ void dp_cleandeg(dpoly *f, int deg)
{
    while (deg >= 0 && f->c[deg] == 0)
        deg--;
    f->deg = deg;
}

static __host__ __device__ double dp_eval(const dpoly *p, double x)
{
    if (p->deg < 0)
        return 0;
    double r = p->c[p->deg];
    for (int k = p->deg - 1; k >= 0; k--)
        r = r * x + p->c[k];
    return r;
}

static __host__ __device__ void dp_derivative(dpoly *df, const dpoly *f)
{
    for (int n = 1; n <= f->deg; n++)
        df->c[n - 1] = f->c[n] * (double) n;
    df->deg = f->deg - 1;
}

static __host__ __device__ void dp_neg_x(dpoly *r, const dpoly *s)
{
    int i;
    r->deg = s->deg;
    for (i = 0; i < s->deg; i += 2) {
        r->c[i] = s->c[i];
        r->c[i + 1] = -s->c[i + 1];
    }
    if (i == s->deg)
        r->c[i] = s->c[i];
}

static __host__ __device__ double dp_dichotomy(const dpoly *p, double a, double b,
                                      double sa)
{
    double s;
    for (int it = 0; it < DP_MAX_ITERS; it++) {
        s = (a + b) * 0.5;
        if (s == a || s == b)
            return s;
        if (dp_eval(p, s) * sa > 0)
            a = s;
        else
            b = s;
    }
    return s;
}

static __host__ __device__ double dp_falseposition(const dpoly *p, double a, double b,
                                          double pa)
{
    int side = 0;
    const double a0 = a, b0 = b, pa0 = pa;
    double pb = dp_eval(p, b);
    for (int it = 0; it < DP_MAX_ITERS; it++) {
        double s = (a * pb - b * pa) / (pb - pa);
        double middle = (a + b) * 0.5;
        if (s < a || s > b ||
            ((s == a || s == b) && !(middle == a || middle == b)))
            s = middle;
        if (s == a || s == b)
            return s;
        double ps = dp_eval(p, s);
        if (ps * pa > 0) {
            a = s; pa = ps;
            if (side == 1) pb /= 2;
            side = 1;
        } else {
            b = s; pb = ps;
            if (side == -1) pa /= 2;
            side = -1;
        }
        if (isnan(b))
            return dp_dichotomy(p, a0, b0, pa0);
    }
    return dp_dichotomy(p, a0, b0, pa0);
}

static __host__ __device__ unsigned int dp_recurse_roots(const dpoly *poly,
                                                double *roots,
                                                unsigned int sign_changes,
                                                double s)
{
    unsigned int nsc = 0;
    if (poly->deg <= 0) {
    } else if (poly->deg == 1) {
        if (poly->c[0] * dp_eval(poly, s) < 0) {
            nsc = 1;
            roots[0] = -poly->c[0] / poly->c[1];
        }
    } else {
        double a = 0.0;
        double va = poly->c[0];
        for (unsigned int l = 0; l <= sign_changes; l++) {
            double b = (l < sign_changes) ? roots[l] : s;
            double vb = dp_eval(poly, b);
            if (va * vb < 0)
                roots[nsc++] = dp_falseposition(poly, a, b, va);
            a = b;
            va = vb;
        }
    }
    return nsc;
}

static __host__ __device__ double dp_bound_roots(int *fail, const dpoly *p)
{
    int d = p->deg;
    dpoly q;
    double s = p->c[d] > 0 ? 1.0 : -1.0;
    for (int i = 0; i < d; i++)
        q.c[i] = (s * p->c[i] < 0) ? s * p->c[i] : 0.0;
    q.c[d] = fabs(p->c[d]);
    dp_cleandeg(&q, d);
    s = 1.0;
    double c;
    int it = 0;
    while ((c = dp_eval(&q, s)) < 0) {
        s = s + s;
        if (++it > SOPT_GPU_MAX_DOUBLINGS) {
            *fail = 1;
            return s;
        }
    }
    s += (c == 0);
    return s;
}

/* roots in (0, s]; roots must hold p->deg entries */
static __host__ __device__ unsigned int dp_compute_roots(double *roots, const dpoly *p,
                                                double s)
{
    int d = p->deg;
    if (d == 0)
        return 0;
    dpoly dg[DP_COEFFS];
    dg[0] = *p;
    for (int k = 1; k < d; k++)
        dp_derivative(&dg[k], &dg[k - 1]);
    unsigned int sign_changes = 0;
    for (int k = d; k > 0; k--)
        sign_changes = dp_recurse_roots(&dg[k - 1], roots, sign_changes, s);
    return sign_changes;
}

static __host__ __device__ unsigned int dp_all_roots_with_bound(int *fail,
                                                       double *roots,
                                                       const dpoly *p,
                                                       double B)
{
    double bound = dp_bound_roots(fail, p);
    bound = bound < B ? bound : B;
    unsigned int npos = dp_compute_roots(roots, p, bound);
    dpoly t;
    dp_neg_x(&t, p);
    bound = dp_bound_roots(fail, &t);
    bound = bound < B ? bound : B;
    unsigned int nneg = dp_compute_roots(roots + npos, &t, bound);
    for (unsigned int i = 0; i < nneg; i++)
        roots[npos + i] *= -1.;
    if (p->c[0] == 0.0)
        roots[npos + nneg++] = 0.0;
    return npos + nneg;
}

static __host__ __device__ unsigned int dp_all_roots(int *fail, double *roots,
                                            const dpoly *p)
{
    return dp_all_roots_with_bound(fail, roots, p, DBL_MAX);
}

/* ------------------------------------------------- modular arithmetic */

static __host__ __device__ uint32_t z_mod_u32(const sopt_z *a, uint32_t m)
{
    uint64_t r = 0;
    for (int i = a->n - 1; i >= 0; i--)
        r = ((r << 32) | a->l[i]) % m;
    return (uint32_t) r;
}

static __host__ __device__ uint64_t z_mod_u64(const sopt_z *a, uint64_t m)
{
    /* m < 2^32 squared fits; reduce limb by limb in 96-bit steps */
    uint64_t r = 0;
    for (int i = a->n - 1; i >= 0; i--) {
        /* r < m <= 2^27ish here (m = p^2 <= 97^2) so no overflow */
        r = ((r << 32) + a->l[i]) % m;
    }
    return r;
}

/* --------------------------------------------- polynomial<double> path */
/* port of polynomial.hpp real root finding, used by L2_skewness; unlike
 * double_poly.cpp its Horner evaluation uses fma */

static __host__ __device__ double pd_eval(const dpoly *p, double x)
{
    if (p->deg < 0)
        return 0;
    double s = p->c[p->deg];
    for (int k = p->deg - 1; k >= 0; k--)
        s = fma(s, x, p->c[k]);
    return s;
}

static __host__ __device__ double pd_bound_positive_roots(int *fail, const dpoly *p)
{
    int d = p->deg;
    int s = p->c[d] < 0;
    dpoly q;
    for (int i = 0; i < d; i++) {
        double v = s ? -p->c[i] : p->c[i];
        q.c[i] = v < 0 ? v : 0;
    }
    q.c[d] = fabs(p->c[d]);
    q.deg = d;
    double b = 1;
    int it = 0;
    while (pd_eval(&q, b) <= 0) {
        b = b + b;
        if (++it > SOPT_GPU_MAX_DOUBLINGS) {
            *fail = 1;
            return b;
        }
    }
    return b;
}

static __host__ __device__ double pd_findroot_dichotomy(const dpoly *p, double a,
                                               double b, double sa)
{
    double s;
    for (int it = 0; it < DP_MAX_ITERS; it++) {
        s = (a + b) / 2;
        if (s == a || s == b)
            return s;
        if (pd_eval(p, s) * sa > 0)
            a = s;
        else
            b = s;
    }
    return s;
}

static __host__ __device__ double pd_findroot_falseposition(const dpoly *p, double a0,
                                                   double b0, double pa0,
                                                   double pb0)
{
    int side = 0;
    double a = a0, b = b0;
    if (a == b)
        return a;
    double sigma = b - a > 0 ? 1 : -1;
    double pa = pa0, pb = pb0;
    for (int it = 0; it < DP_MAX_ITERS; it++) {
        double s = (a * pb - b * pa) / (pb - pa);
        double middle = (a + b) / 2;
        const bool escapes = sigma * s < sigma * a || sigma * s > sigma * b;
        const bool hits = s == a || s == b;
        const bool middle_nice = !(middle == a || middle == b);
        if (escapes || (hits && middle_nice))
            s = middle;
        if (s == a || s == b)
            return s;
        double ps = pd_eval(p, s);
        if (ps * pa > 0) {
            a = s; pa = ps;
            if (side == 1) pb /= 2;
            side = 1;
        } else {
            b = s; pb = ps;
            if (side == -1) pa /= 2;
            side = -1;
        }
        if (isnan(b))
            return pd_findroot_dichotomy(p, a0, b0, pa0);
    }
    return pd_findroot_dichotomy(p, a0, b0, pa0);
}

static __host__ __device__ int dsgn(double x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

/* positive_roots_from_derivative_sign_changes; v holds nv roots of the
 * derivative on input, the roots of p on output */
static __host__ __device__ void pd_roots_from_dsc(const dpoly *p, double *v, int *nv,
                                         double bound)
{
    if (p->deg == 1) {
        if (p->c[0] != 0 && p->c[0] * pd_eval(p, bound) <= 0) {
            v[0] = -p->c[0] / p->c[1];
            *nv = 1;
        } else
            *nv = 0;
        return;
    }
    double a = 0;
    double va = p->c[0];
    v[*nv] = bound;
    (*nv)++;
    int m = 0;
    int c01 = dsgn(p->c[0]) *
              dsgn(p->c[1] != 0 ? p->c[1] : dsgn(bound) * p->c[2]);
    bool no_chance = c01 * dsgn(bound) > 0 || p->c[0] == 0;
    for (int i = 0; i < *nv; i++) {
        double b = v[i];
        double vb = pd_eval(p, b);
        if (no_chance) {
            no_chance = false;
        } else if (va * vb < 0) {
            v[m++] = pd_findroot_falseposition(p, a, b, va, vb);
        } else {
            no_chance = true;
        }
        a = b;
        va = vb;
    }
    *nv = m;
}

/* positive_roots of polynomial.hpp; roots must hold p->deg entries */
static __host__ __device__ int pd_positive_roots(int *fail, double *roots,
                                        const dpoly *p)
{
    int d = p->deg;
    if (d == 0)
        return 0;
    double bound = pd_bound_positive_roots(fail, p);
    dpoly dg[DP_COEFFS];
    dg[0] = *p;
    for (int k = 1; k < d; k++)
        dp_derivative(&dg[k], &dg[k - 1]);
    int nv = 0;
    for (int k = d; k-- > 0; )
        pd_roots_from_dsc(&dg[k], roots, &nv, bound);
    return nv;
}

/* ------------------------------------------------------------ L2 norms */
/* port of polyselect_norms.cpp (USE_PRESCALING, polynomial<double> code) */

#define L2_LN2_2 0.34657359027997264   /* std::numbers::ln2 / 2 */

#define COEFFS_INTEGRAL_INIT { \
    { 1 }, { 2 }, { 1, 6 }, { 4, 20 }, { 3, 10, 70 }, { 12, 28, 252 }, \
    { 10, 28, 84, 924 }, { 40, 72, 264, 3432 }, \
}
static const uint32_t __attribute__((unused)) h_coeffs_integral[8][4] = COEFFS_INTEGRAL_INIT;
static __constant__ uint32_t d_coeffs_integral[8][4] = COEFFS_INTEGRAL_INIT;
#ifdef __CUDA_ARCH__
#define coeffs_integral d_coeffs_integral
#else
#define coeffs_integral h_coeffs_integral
#endif

/* the templated L2_lognorm; sets fail on nan or inf (the CPU throws) */
static __host__ __device__ double l2_lognorm_pd(int *fail, const dpoly *p, double s)
{
    int d = p->deg;
    const double *a = p->c;
    double invs = 1 / s;
    double s2 = s * s;
    double is2 = invs * invs;
    double tt = 0;
    double u0 = (d & 1) ? invs : 1;
    double u1 = (d & 1) ? s : 1;
    int p0 = d / 2, p1 = d - p0;
    for (int k = 0; k <= d / 2; k++) {
        double t0 = 0, t1 = 0;
        for (int i = 1; i <= d / 2 - k; i++) {
            t0 += a[p0 - k - i] * a[p0 - k + i];
            t1 += a[p1 + k - i] * a[p1 + k + i];
        }
        t0 = (2 * t0 + a[p0 - k] * a[p0 - k]) * u0;
        t1 = (2 * t1 + a[p1 + k] * a[p1 + k]) * u1;
        u0 *= is2;
        u1 *= s2;
        tt += coeffs_integral[d][k] * (t0 + t1);
    }
    tt = ldexp(tt * M_PI / (d + 1), -2 * d);
    if (isnan(tt) || isinf(tt))
        *fail = 1;
    return log(tt) / 2;
}

/* L2_skewness_derivative_numerator (a polynomial in s^2) */
static __host__ __device__ void l2_skew_deriv_num(int *fail, dpoly *dP, const dpoly *p)
{
    int d = p->deg;
    const double *a = p->c;
    for (int k = 0; k <= d; k++) {
        double t0 = 0;
        for (int i = 1; i <= k && i <= d - k; i++)
            t0 += a[k - i] * a[k + i];
        t0 = 2 * t0 + a[k] * a[k];
        int j = (abs(d - 2 * k) - (d & 1)) / 2;
        t0 = t0 * coeffs_integral[d][j] * (2 * k - d);
        dP->c[k] = t0;
        if (isnan(t0) || isinf(t0))
            *fail = 1;
    }
    dp_cleandeg(dP, d);
}

static __host__ __device__ double l2_skewness_pd(int *fail, const dpoly *P)
{
    dpoly dP;
    l2_skew_deriv_num(fail, &dP, P);
    double roots[DP_COEFFS];
    int nroots = pd_positive_roots(fail, roots, &dP);
    if (nroots < 1) {
        *fail = 1;
        return 1.0;
    }
    double best_s = sqrt(roots[0]);
    double l = l2_lognorm_pd(fail, P, best_s);
    for (int i = 1; i < nroots; i++) {
        double s = sqrt(roots[i]);
        double li = l2_lognorm_pd(fail, P, s);
        if (li < l) {
            best_s = s;
            l = li;
        }
    }
    return best_s;
}

/* prescaling of L2_skewness / L2_lognorm (mpz_poly entry points) */
static __host__ __device__ void zp_prescale(dpoly *F, int *bits0, int *scale,
                                   const zpoly *f)
{
    *bits0 = z_sizeinbase2(&f->c[0]);
    int bits1 = z_sizeinbase2(&f->c[f->deg]);
    *scale = (bits1 - *bits0) / f->deg;
    F->deg = f->deg;
    for (int i = 0; i <= f->deg; i++)
        F->c[i] = z_get_d_scaled(&f->c[i], -*bits0 - *scale * i);
}

static __host__ __device__ double l2_skewness_z(int *fail, const zpoly *f)
{
    dpoly F;
    int bits0, scale;
    zp_prescale(&F, &bits0, &scale, f);
    double sx = l2_skewness_pd(fail, &F);
    return ldexp(sx, -scale);
}

static __host__ __device__ double l2_lognorm_z(int *fail, const zpoly *f, double s)
{
    dpoly F;
    int bits0, scale;
    zp_prescale(&F, &bits0, &scale, f);
    double n = l2_lognorm_pd(fail, &F, ldexp(s, scale));
    double fix = 2 * bits0 + f->deg * scale;
    return fma(fix, L2_LN2_2, n);
}

static __host__ __device__ double l2_skew_lognorm_z(int *fail, const zpoly *f)
{
    dpoly F;
    int bits0, scale;
    zp_prescale(&F, &bits0, &scale, f);
    double n = l2_lognorm_pd(fail, &F, l2_skewness_pd(fail, &F));
    double fix = 2 * bits0 + f->deg * scale;
    return fma(fix, L2_LN2_2, n);
}

/* ------------------------------------------------- alpha (projective) */
/* port of polyselect_alpha.cpp for get_alpha_projective(f, 100) */

#define SMALL_PRIMES_INIT { \
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, \
    67, 71, 73, 79, 83, 89, 97, \
}
static const uint32_t __attribute__((unused)) h_small_primes[25] = SMALL_PRIMES_INIT;
static __constant__ uint32_t d_small_primes[25] = SMALL_PRIMES_INIT;
#ifdef __CUDA_ARCH__
#define small_primes d_small_primes
#else
#define small_primes h_small_primes
#endif

/* discriminant of f: Res(f, f') / lc(f), by Bareiss elimination on the
 * Sylvester matrix of f and f' (the CPU uses mpz_poly_resultant; only
 * the exact magnitude matters, the callers test divisibility) */
static __host__ __device__ void zp_discriminant(sopt_scratch *S, sopt_z *disc,
                                       const zpoly *f)
{
    int d = f->deg;
    int n = 2 * d - 1;
    sopt_z (*M)[SYL_DIM] = S->M;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            z_set_u32(&M[i][j], 0);
    for (int i = 0; i < d - 1; i++)             /* rows of f */
        for (int j = 0; j <= d; j++)
            z_set(&M[i][i + j], &f->c[d - j]);
    for (int i = 0; i < d; i++)                 /* rows of f' */
        for (int j = 0; j < d; j++) {
            z_mul_u32(&M[d - 1 + i][i + j], &f->c[d - j], d - j);
        }
    sopt_z prev, t1, t2;
    z_set_u32(&prev, 1);
    int sign = 1;
    for (int k = 0; k < n - 1; k++) {
        int piv = -1;
        for (int i = k; i < n && piv < 0; i++)
            if (M[i][k].n != 0)
                piv = i;
        if (piv < 0) {
            z_set_u32(disc, 0);
            return;
        }
        if (piv != k) {
            sign = -sign;
            for (int j = 0; j < n; j++) {
                sopt_z t = M[k][j];
                M[k][j] = M[piv][j];
                M[piv][j] = t;
            }
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                z_mul(&t1, &M[i][j], &M[k][k]);
                z_mul(&t2, &M[i][k], &M[k][j]);
                z_sub(&t1, &t1, &t2);
                z_divexact(&M[i][j], &t1, &prev);
            }
            z_set_u32(&M[i][k], 0);
        }
        z_set(&prev, &M[k][k]);
        if (z_bad(&prev)) {
            S->fail = 1;
            z_set_u32(disc, 0);
            return;
        }
    }
    z_divexact(disc, &M[n - 1][n - 1], &f->c[d]);
    if (sign < 0)
        disc->sign = -disc->sign;
}

/* distinct roots of f mod p in increasing order; f mod p must be a
 * nonzero polynomial. Returns the root count. */
static __host__ __device__ int zp_roots_mod(uint32_t *roots, const zpoly *f,
                                   uint32_t p)
{
    uint32_t c[ZP_COEFFS];
    for (int i = 0; i <= f->deg; i++) {
        c[i] = z_mod_u32(&f->c[i], p);
        if (f->c[i].sign < 0 && c[i] != 0)
            c[i] = p - c[i];
    }
    int n = 0;
    for (uint32_t r = 0; r < p; r++) {
        uint64_t v = 0;
        for (int i = f->deg; i >= 0; i--)
            v = (v * r + c[i]) % p;
        if (v == 0 && roots)
            roots[n] = r;
        n += v == 0;
    }
    return n;
}

/* f'(r) mod p */
static __host__ __device__ uint32_t zp_eval_diff_mod(const zpoly *f, uint32_t r,
                                            uint32_t p)
{
    uint64_t v = 0;
    for (int i = f->deg; i >= 1; i--) {
        uint32_t c = z_mod_u32(&f->c[i], p);
        if (f->c[i].sign < 0 && c != 0)
            c = p - c;
        v = (v * r + (uint64_t) c * i) % p;
    }
    return (uint32_t) v;
}

/* mpz_poly_shift_divp: H(x) <- H(x + r/p), integer coefficients */
static __host__ __device__ void zp_shift_divp(zpoly *H, uint32_t r, uint32_t p)
{
    sopt_z t, zp;
    z_set_u32(&zp, p);
    for (int i = 1; i <= H->deg; i++)
        for (int k = H->deg - i; k < H->deg; k++) {
            z_divexact(&t, &H->c[k + 1], &zp);
            z_mul_u32(&t, &t, r);
            z_add(&H->c[k], &H->c[k], &t);
        }
}

/* special_val0, made iterative with an explicit frame stack */
static __host__ __device__ double special_val0(sopt_scratch *S, const zpoly *f,
                                      uint32_t p)
{
    val0_frame *st = S->val0;
    int top = 0;
    const zpoly *input = f;
    double ret = 0;

    while (top >= 0) {
        val0_frame *F = &st[top];
        if (input) {                      /* entering a new frame */
            sopt_z c, zp;
            zp_content(&c, input);
            int val = 0;
            z_set_u32(&zp, p);
            while (c.n > 0 && z_divisible_p(&c, &zp)) {
                val++;
                z_divexact(&c, &c, &zp);
            }
            F->v = val;
            if (val != 0) {
                sopt_z pv;
                z_set_u32(&pv, 1);
                for (int i = 0; i < val; i++)
                    z_mul_u32(&pv, &pv, p);
                F->g.deg = input->deg;
                for (int i = 0; i <= input->deg; i++)
                    z_divexact(&F->g.c[i], &input->c[i], &pv);
            } else
                zp_set(&F->g, input);
            F->H.deg = F->g.deg;
            z_set_u32(&c, 1);
            for (int i = 0; i <= F->g.deg; i++) {
                z_mul(&F->H.c[i], &F->g.c[i], &c);
                z_mul_u32(&c, &c, p);
            }
            if (zp_bad(&F->H)) {
                S->fail = 1;
                return 0;
            }
            F->nroots = zp_roots_mod(F->roots, &F->g, p);
            F->idx = 0;
            F->r0 = 0;
            input = NULL;
        } else                            /* child returned ret */
            F->v += ret / (double) p;

        int descend = 0;
        while (F->idx < F->nroots && !descend) {
            uint32_t r = F->roots[F->idx++];
            if (zp_eval_diff_mod(&F->g, r, p) != 0) {
                F->v += 1.0 / (double) (p - 1);
                continue;
            }
            zp_shift_divp(&F->H, r - F->r0, p);
            F->r0 = r;
            if (zp_bad(&F->H) || top + 1 >= SOPT_GPU_MAX_VAL0_DEPTH) {
                S->fail = 1;
                return 0;
            }
            input = &F->H;
            top++;
            descend = 1;
        }
        if (!descend) {
            ret = F->v;
            top--;
        }
    }
    return ret;
}

/* special_valuation; disc is the discriminant of f */
static __host__ __device__ double special_valuation(sopt_scratch *S, const zpoly *f,
                                           uint32_t p, const sopt_z *disc)
{
    double pd = p;
    int d = f->deg;
    uint64_t dm = z_mod_u64(disc, (uint64_t) p * p);
    int pvaluation_disc = disc->n == 0 ? 2
                        : (dm % p == 0) + (dm % p == 0 && dm == 0);
    int p_divides_lc = z_mod_u32(&f->c[d], p) == 0;

    if (pvaluation_disc == 0) {
        unsigned int e = zp_roots_mod(NULL, f, p) + p_divides_lc;
        return (pd * e) / (pd * pd - 1);
    }
    if (pvaluation_disc == 1) {
        unsigned int e = zp_roots_mod(NULL, f, p) + p_divides_lc;
        return (pd * e - 1) / (pd * pd - 1);
    }
    double v = special_val0(S, f, p) * pd;
    if (p_divides_lc) {
        zpoly G;                /* f(1/(px)) * (px)^d */
        sopt_z t;
        z_set_u32(&t, 1);
        for (int i = 0; i <= d; i++) {
            z_mul(&G.c[i], &f->c[d - i], &t);
            z_mul_u32(&t, &t, p);
        }
        zp_cleandeg(&G, d);
        v += special_val0(S, &G, p);
    }
    return v / (pd + 1.0);
}

static __host__ __device__ double special_valuation_affine(sopt_scratch *S,
                                                  const zpoly *f, uint32_t p,
                                                  const sopt_z *disc)
{
    double pd = p;
    uint64_t dm = z_mod_u64(disc, (uint64_t) p * p);
    if (disc->n != 0 && dm % p != 0) {
        unsigned int e = zp_roots_mod(NULL, f, p);
        return (pd * e) / (pd * pd - 1);
    }
    return special_val0(S, f, p) * pd / (pd + 1.0);
}

static __host__ __device__ double get_alpha_projective(sopt_scratch *S,
                                              const zpoly *f)
{
    sopt_z disc;
    zp_discriminant(S, &disc, f);
    if (S->fail)
        return 0;
    double alpha = 0;
    for (int i = 0; i < 25 && !S->fail; i++) {
        uint32_t p = small_primes[i];
        double e = special_valuation(S, f, p, &disc)
                 - special_valuation_affine(S, f, p, &disc);
        alpha += (-e) * log((double) p);
    }
    return alpha;
}

/* expected_alpha of polyselect_alpha.cpp (MU = 0, SIGMA = 0.824) */
static __host__ __device__ double expected_alpha(double logK)
{
    if (logK < 0.999)
        return 0.0;
    return -0.824 * (sqrt(2 * logK) -
                     (log(logK) + 1.3766) / (2 * sqrt(2 * logK)));
}

/* expected_growth of auxiliary.cpp: the largest interval kmin <= k <= kmax
 * such that rotation by k*x^i*g keeps the lognorm below maxlognorm */
static __host__ __device__ void expected_growth(sopt_scratch *S, double *rkmin,
                                       double *rkmax, const zpoly *f0,
                                       const zpoly *g, int i,
                                       double maxlognorm, double s)
{
    zpoly *f = &S->tmpf;
    sopt_z kmin, kmax, k;

    for (int side = 0; side < 2; side++) {
        z_set_i64(&kmin, side == 0 ? -1 : 1);
        for (int it = 0; ; it++) {
            zp_rotation(f, f0, g, &kmin, i);
            if (l2_lognorm_z(&S->fail, f, s) > maxlognorm)
                break;
            z_mul_2exp(&kmin, &kmin, 1);
            if (it > SOPT_GPU_MAX_DOUBLINGS || S->fail || z_bad(&kmin)) {
                S->fail = 1;
                return;
            }
        }
        z_tdiv_q_2exp(&kmax, &kmin, 1);
        for (int it = 0; ; it++) {
            z_add(&k, &kmin, &kmax);
            z_fdiv_q_2exp(&k, &k, 1);
            if (z_cmp(&k, &kmin) == 0 || z_cmp(&k, &kmax) == 0)
                break;
            zp_rotation(f, f0, g, &k, i);
            if (l2_lognorm_z(&S->fail, f, s) > maxlognorm)
                z_set(&kmin, &k);
            else
                z_set(&kmax, &k);
            if (it > SOPT_GPU_MAX_DOUBLINGS || S->fail) {
                S->fail = 1;
                return;
            }
        }
        /* kmin here is the CPU's doubling variable (kmax on the
         * positive side); kmax is the CPU's result variable */
        if (side == 0)
            *rkmin = z_get_d(&kmax);
        else
            *rkmax = z_get_d(&kmax);
    }
}

/* expected_rotation_gain of auxiliary.cpp */
static __host__ __device__ double expected_rotation_gain(sopt_scratch *S,
                                                const zpoly *f,
                                                const zpoly *g)
{
    double S_ = 1.0, incr = 0.0;
    double proj_alpha = get_alpha_projective(S, f);
    double skew = l2_skewness_z(&S->fail, f);
    double n = l2_lognorm_z(&S->fail, f, skew);
    for (int i = 0; 2 * i < f->deg && !S->fail; i++) {
        double kmin, kmax;
        expected_growth(S, &kmin, &kmax, f, g, i, n + NORM_MARGIN, skew);
        double s = kmax - kmin + 1.0;
        S_ *= s;
        if (s >= 2.0)
            incr += NORM_MARGIN / 2.0;
    }
    return proj_alpha + expected_alpha(log(S_)) + incr;
}

/* ----------------------------------------------------------------- LLL */
/* port of utils/lll.c with delta = a/b = 1 and no transformation matrix */

static __host__ __device__ void z_fdiv_q(sopt_z *q, const sopt_z *a, const sopt_z *d)
{
    sopt_z r;
    z_fdiv_qr(q, &r, a, d);
}

static __host__ __device__ void lll_inner(sopt_scratch *S, sopt_z *x, int ra, int rb,
                                 int n)
{
    z_mul(x, &S->B[ra][1], &S->B[rb][1]);
    for (int i = 2; i <= n; i++)
        z_addmul(x, &S->B[ra][i], &S->B[rb][i]);
}

static __host__ __device__ void lll_incremental_gs(sopt_scratch *S, int *P, int *s,
                                          int k, int n)
{
    sopt_z u, t1;
    for (int j = 1; j <= k - 1; j++) {
        int posj = P[j];
        if (posj == 0)
            continue;
        lll_inner(S, &u, k, j, n);
        for (int i = 1; i <= posj - 1; i++) {
            z_mul(&t1, &S->D[i], &u);
            z_submul(&t1, &S->lam[k][i], &S->lam[j][i]);
            z_fdiv_q(&u, &t1, &S->D[i - 1]);
        }
        z_set(&S->lam[k][posj], &u);
    }
    lll_inner(S, &u, k, k, n);
    for (int i = 1; i <= *s; i++) {
        z_mul(&t1, &S->D[i], &u);
        z_submul(&t1, &S->lam[k][i], &S->lam[k][i]);
        z_fdiv_q(&u, &t1, &S->D[i - 1]);
    }
    if (u.n == 0) {
        P[k] = 0;
    } else {
        (*s)++;
        P[k] = *s;
        z_set(&S->D[*s], &u);
    }
}

/* round a/d to the nearest integer, ties toward zero; d > 0 */
static __host__ __device__ void lll_baldiv(sopt_z *q, const sopt_z *a,
                                  const sopt_z *d)
{
    sopt_z r, one;
    z_fdiv_qr(q, &r, a, d);
    z_mul_2exp(&r, &r, 1);
    int cmp = z_cmp(&r, d);
    if (cmp > 0 || (cmp == 0 && q->sign < 0 && q->n != 0)) {
        z_set_u32(&one, 1);
        z_add(q, q, &one);
    }
}

static __host__ __device__ void lll_mulsub_row(sopt_scratch *S, int rk, int rl,
                                      const sopt_z *x, int n)
{
    for (int i = 1; i <= n; i++)
        z_submul(&S->B[rk][i], &S->B[rl][i], x);
}

static __host__ __device__ void lll_reduce(sopt_scratch *S, int k, int l,
                                  const int *P, int n)
{
    if (P[l] == 0)
        return;
    sopt_z t1, r;
    z_mul_2exp(&t1, &S->lam[k][P[l]], 1);
    if (t1.sign < 0)
        t1.sign = 1;
    if (z_cmp(&t1, &S->D[P[l]]) <= 0)
        return;
    lll_baldiv(&r, &S->lam[k][P[l]], &S->D[P[l]]);
    lll_mulsub_row(S, k, l, &r, n);
    for (int j = 1; j <= l - 1; j++)
        if (P[j] != 0)
            z_submul(&S->lam[k][P[j]], &S->lam[l][P[j]], &r);
    z_submul(&S->lam[k][P[l]], &S->D[P[l]], &r);
}

/* test d1^2 > d0*d2 + lam^2 (delta = 1) */
static __host__ __device__ int lll_swaptest(const sopt_z *d0, const sopt_z *d1,
                                   const sopt_z *d2, const sopt_z *lam)
{
    sopt_z t1, t2;
    z_mul(&t1, d0, d2);
    z_addmul(&t1, lam, lam);
    z_mul(&t2, d1, d1);
    return z_cmp(&t2, &t1) > 0;
}

/* (c1, c2) = (x*c1 + y*c2, u*c1 + v*c2) */
static __host__ __device__ void lll_rowtransform(sopt_z *c1, sopt_z *c2,
                                        const sopt_z *x, const sopt_z *y,
                                        const sopt_z *u, const sopt_z *v)
{
    sopt_z t1, t2;
    z_mul(&t1, x, c1);
    z_addmul(&t1, y, c2);
    z_mul(&t2, u, c1);
    z_set(c1, &t1);
    z_mul(&t1, v, c2);
    z_add(c2, &t1, &t2);
}

static __host__ __device__ void lll_swap_rows(sopt_scratch *S, int r1, int r2, int n)
{
    for (int i = 1; i <= n; i++) {
        sopt_z t = S->B[r1][i];
        S->B[r1][i] = S->B[r2][i];
        S->B[r2][i] = t;
    }
}

static __host__ __device__ void lll_swap(sopt_scratch *S, int k, int *P, int m, int n)
{
    sopt_z t1, t2, t3, e, x, y;
    if (P[k] != 0) {
        lll_swap_rows(S, k - 1, k, n);
        for (int j = 1; j <= k - 2; j++)
            if (P[j] != 0) {
                sopt_z t = S->lam[k - 1][P[j]];
                S->lam[k - 1][P[j]] = S->lam[k][P[j]];
                S->lam[k][P[j]] = t;
            }
        for (int i = k + 1; i <= m; i++) {
            /* t1 = (lam[i][Pk-1]*lam[k][Pk-1] + lam[i][Pk]*D[Pk-2])/D[Pk-1] */
            z_mul(&t1, &S->lam[i][P[k] - 1], &S->lam[k][P[k] - 1]);
            z_addmul(&t1, &S->lam[i][P[k]], &S->D[P[k] - 2]);
            z_divexact(&t1, &t1, &S->D[P[k] - 1]);
            /* lam[i][Pk] = (lam[i][Pk-1]*D[Pk] - lam[i][Pk]*lam[k][Pk-1])/D[Pk-1] */
            z_mul(&t2, &S->lam[i][P[k] - 1], &S->D[P[k]]);
            z_submul(&t2, &S->lam[i][P[k]], &S->lam[k][P[k] - 1]);
            z_divexact(&S->lam[i][P[k]], &t2, &S->D[P[k] - 1]);
            z_set(&S->lam[i][P[k] - 1], &t1);
        }
        /* D[Pk-1] = (D[Pk]*D[Pk-2] + lam[k][Pk-1]^2)/D[Pk-1] */
        z_mul(&t1, &S->D[P[k]], &S->D[P[k] - 2]);
        z_addmul(&t1, &S->lam[k][P[k] - 1], &S->lam[k][P[k] - 1]);
        z_divexact(&S->D[P[k] - 1], &t1, &S->D[P[k] - 1]);
    } else if (S->lam[k][P[k - 1]].n != 0) {
        z_gcdext(&e, &x, &y, &S->lam[k][P[k - 1]], &S->D[P[k - 1]]);
        z_divexact(&t1, &S->lam[k][P[k - 1]], &e);
        z_divexact(&t2, &S->D[P[k - 1]], &e);
        z_set(&t3, &t2);
        z_neg(&t2, &t2);
        for (int i = 1; i <= n; i++)
            lll_rowtransform(&S->B[k - 1][i], &S->B[k][i], &t1, &t2, &y, &x);
        for (int j = 1; j <= k - 2; j++)
            if (P[j] != 0)
                lll_rowtransform(&S->lam[k - 1][P[j]], &S->lam[k][P[j]],
                                 &t1, &t2, &y, &x);
        z_mul(&t2, &t2, &t2);
        z_divexact(&S->D[P[k - 1]], &S->D[P[k - 1]], &t2);
        for (int i = k + 1; i <= m; i++)
            if (P[i] != 0) {
                z_divexact(&S->D[P[i]], &S->D[P[i]], &t2);
                for (int j = i + 1; j <= m; j++)
                    z_divexact(&S->lam[j][P[i]], &S->lam[j][P[i]], &t2);
            }
        for (int i = k + 1; i <= m; i++)
            z_divexact(&S->lam[i][P[k - 1]], &S->lam[i][P[k - 1]], &t3);
        int t = P[k - 1]; P[k - 1] = P[k]; P[k] = t;
    } else {
        lll_swap_rows(S, k - 1, k, n);
        for (int j = 1; j <= k - 2; j++)
            if (P[j] != 0) {
                sopt_z t = S->lam[k - 1][P[j]];
                S->lam[k - 1][P[j]] = S->lam[k][P[j]];
                S->lam[k][P[j]] = t;
            }
        int t = P[k - 1]; P[k - 1] = P[k]; P[k] = t;
    }
}

/* LLL-reduce the m x n basis in S->B (1-based rows and columns) */
static __host__ __device__ void lll(sopt_scratch *S, int m, int n)
{
    int P[LLL_DIM];
    for (int j = 0; j <= m; j++)
        P[j] = 0;
    for (int j = 0; j <= m; j++)
        z_set_u32(&S->D[j], j == 0);
    for (int j = 0; j <= m; j++)
        for (int i = 0; i <= m; i++)
            z_set_u32(&S->lam[j][i], 0);

    int s = 0, k = 1, max_k = 0, iters = 0;
    while (k <= m) {
        if (++iters > SOPT_GPU_MAX_LLL_ITERS || S->fail) {
            S->fail = 1;
            return;
        }
        if (k > max_k) {
            lll_incremental_gs(S, P, &s, k, n);
            max_k = k;
        }
        if (k == 1) {
            k++;
            continue;
        }
        lll_reduce(S, k, k - 1, P, n);
        if (P[k - 1] != 0 &&
            (P[k] == 0 || lll_swaptest(&S->D[P[k]], &S->D[P[k] - 1],
                                       &S->D[P[k] - 2],
                                       &S->lam[k][P[k] - 1]))) {
            lll_swap(S, k, P, max_k, n);
            k--;
        } else {
            for (int j = k - 2; j >= 1; j--)
                lll_reduce(S, k, j, P, n);
            k++;
        }
        for (int j = 1; j <= s && !S->fail; j++)
            S->fail |= z_bad(&S->D[j]);
    }
}

/* --------------------------------------------------- size optimization */
/* port of size_optimization.cpp */

/* mpz_set_d rounding to nearest, as rounded_double() does */
static __host__ __device__ double d_round(double e)
{
    return trunc(e > 0 ? e + 0.5 : e - 0.5);
}

static __host__ __device__ uint64_t gcd_u64(uint64_t a, uint64_t b)
{
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static __host__ __device__ unsigned int
compute_rational_approximation(double *Q, double q, unsigned int nb_approx,
                               double bound)
{
    unsigned int n = 0;
    double best_e = 2.0;
    for (double den = 1.0; n < nb_approx && den <= bound; den += 1.0) {
        double num = floor(den * q + 0.5);
        double e = fabs(q - num / den);
        if (e >= best_e)
            continue;
        best_e = e;
        Q[n++] = num / den;
    }
    return n;
}

static __host__ __device__ void list_k_push(sopt_scratch *S, int *nk, double v)
{
    if (*nk >= SOPT_GPU_MAX_LIST_K) {
        S->fail = 1;
        return;
    }
    S->list_k[(*nk)++] = v;
}

/* the next Farey fraction a/b, as in the CPU loops */
static __host__ __device__ void farey_next(int *a, int *b)
{
    if (*a > 0) {
        *a = -*a;
        return;
    }
    *a = -*a + 1;
    while (gcd_u64(*a, *b) != 1)
        (*a)++;
    if (*a > *b) {
        *a = 1;
        (*b)++;
    }
}

static __host__ __device__ void sopt_find_translations_deg6(sopt_scratch *S, int *nk,
                                                   const zpoly *f,
                                                   const zpoly *g,
                                                   int sopt_effort)
{
    const double a6 = z_get_d(&f->c[6]);
    const double a5 = z_get_d(&f->c[5]);
    const double a4 = z_get_d(&f->c[4]);
    const double a3 = z_get_d(&f->c[3]);
    const double a2 = z_get_d(&f->c[2]);
    const double g1 = z_get_d(&g->c[1]);
    const double g0 = z_get_d(&g->c[0]);

    const double kmax = pow(fabs(g0 * g0 / (15.0 * a6 * g1)), 0.25);

    dpoly res;
    double roots_q3[3];
    res.c[3] = a6 * g1 * g1 * g1;
    res.c[2] = 135.0 * a6 * a6 * g0 * g0 - 45.0 * a5 * a6 * g0 * g1
             + 10.0 * a5 * a5 * g1 * g1 - 15.0 * a4 * a6 * g1 * g1;
    res.c[1] = 50.0 * a5 * a5 * a5 * g0 - 180.0 * a4 * a5 * a6 * g0
             + 270.0 * a3 * a6 * a6 * g0 - 10.0 * a4 * a5 * a5 * g1
             + 48.0 * a4 * a4 * a6 * g1 - 45.0 * a3 * a5 * a6 * g1;
    res.c[0] = -20.0 * a4 * a4 * a5 * a5 + 50.0 * a3 * a5 * a5 * a5
             + 64.0 * a4 * a4 * a4 * a6 - 180.0 * a3 * a4 * a5 * a6
             + 135.0 * a3 * a3 * a6 * a6;
    dp_cleandeg(&res, 3);

    unsigned int nb_q3roots = dp_all_roots_with_bound(&S->fail, roots_q3,
                                            &res, SOPT_MAX_VALUE_FOR_Q_ROOTS);

    dpoly C;
    double roots_k[5];
    int a = 1, b = 1;
    for (unsigned int i = 0; i < nb_q3roots + 2 * sopt_effort; i++) {
        unsigned int t;
        double q3_rat_approx[SOPT_NB_RAT_APPROX_OF_Q_ROOTS];
        if (i < nb_q3roots)
            t = compute_rational_approximation(q3_rat_approx, roots_q3[i],
                    SOPT_NB_RAT_APPROX_OF_Q_ROOTS,
                    SOPT_MAX_DEN_IN_RAT_APPROX_OF_Q_ROOTS);
        else
            t = SOPT_NB_RAT_APPROX_OF_Q_ROOTS;

        for (unsigned int j = 0; j < t; j++) {
            double q3_rat = q3_rat_approx[j];
            unsigned int nb;
            if (i >= nb_q3roots) {
                q3_rat = (double) a / (double) b;
                farey_next(&a, &b);
            }

            C.c[2] = 15.0 * a6;
            C.c[1] = 5.0 * a5;
            C.c[0] = g1 * q3_rat + a4;
            dp_cleandeg(&C, 2);
            nb = dp_all_roots_with_bound(&S->fail, roots_k, &C, kmax);
            for (unsigned int l = 0; l < nb; l++)
                list_k_push(S, nk, d_round(roots_k[l]));

            C.c[3] = 20.0 * a6;
            C.c[2] = 10.0 * a5;
            C.c[1] = g1 * q3_rat + 4.0 * a4;
            C.c[0] = q3_rat * g0 + a3;
            dp_cleandeg(&C, 3);
            nb = dp_all_roots_with_bound(&S->fail, roots_k, &C, kmax);
            for (unsigned int l = 0; l < nb; l++)
                list_k_push(S, nk, d_round(roots_k[l]));

            C.c[4] = -5.0 * a6 * g1;
            C.c[3] = -20.0 * a6 * g0;
            C.c[2] = -g1 * g1 * q3_rat - 10.0 * a5 * g0 + 2.0 * g1 * a4;
            C.c[1] = 2.0 * g1 * a3 - 2.0 * g1 * q3_rat * g0 - 4.0 * g0 * a4;
            C.c[0] = -q3_rat * g0 * g0 - g0 * a3 + g1 * a2;
            dp_cleandeg(&C, 4);
            nb = dp_all_roots_with_bound(&S->fail, roots_k, &C, kmax);
            for (unsigned int l = 0; l < nb; l++)
                list_k_push(S, nk, d_round(roots_k[l]));

            dp_derivative(&C, &C);
            nb = dp_all_roots_with_bound(&S->fail, roots_k, &C, kmax);
            for (unsigned int l = 0; l < nb; l++)
                list_k_push(S, nk, d_round(roots_k[l]));
        }
    }
}

static __host__ __device__ void sopt_find_translations_deg5(sopt_scratch *S, int *nk,
                                                   const zpoly *f,
                                                   const zpoly *g,
                                                   int sopt_effort)
{
    const double a5 = z_get_d(&f->c[5]);
    const double a4 = z_get_d(&f->c[4]);
    const double a3 = z_get_d(&f->c[3]);
    const double a2 = z_get_d(&f->c[2]);
    const double g1 = z_get_d(&g->c[1]);
    const double g0 = z_get_d(&g->c[0]);

    dpoly res;
    double roots_q2[3];
    res.c[2] = (5.0 * a5 * g0 - a4 * g1) * (5.0 * a5 * g0 - a4 * g1);
    res.c[1] = 8.0 * a4 * a4 * a4 * g0 - 30.0 * a3 * a4 * a5 * g0
             + 50.0 * a2 * a5 * a5 * g0 - 2.0 * a3 * a4 * a4 * g1
             + 10.0 * a3 * a3 * a5 * g1 - 10.0 * a2 * a4 * a5 * g1;
    res.c[0] = -3.0 * a3 * a3 * a4 * a4 + 8.0 * a2 * a4 * a4 * a4
             + 10.0 * a3 * a3 * a3 * a5 - 30.0 * a2 * a3 * a4 * a5
             + 25.0 * a2 * a2 * a5 * a5;
    dp_cleandeg(&res, 2);

    unsigned int nb_q2roots = dp_all_roots_with_bound(&S->fail, roots_q2,
                                            &res, SOPT_MAX_VALUE_FOR_Q_ROOTS);

    dpoly C;
    double roots_k[5];
    int a = 1, b = 1;
    for (unsigned int i = 0; i < nb_q2roots + sopt_effort; i++) {
        unsigned int t;
        double q2_rat_approx[SOPT_NB_RAT_APPROX_OF_Q_ROOTS];
        if (i < nb_q2roots)
            t = compute_rational_approximation(q2_rat_approx, roots_q2[i],
                    SOPT_NB_RAT_APPROX_OF_Q_ROOTS,
                    SOPT_MAX_DEN_IN_RAT_APPROX_OF_Q_ROOTS);
        else
            t = SOPT_NB_RAT_APPROX_OF_Q_ROOTS;

        for (unsigned int j = 0; j < t; j++) {
            double q2_rat = q2_rat_approx[j];
            if (i >= nb_q2roots) {
                q2_rat = (double) a / (double) b;
                farey_next(&a, &b);
            }

            C.c[2] = 10.0 * a5;
            C.c[1] = 4.0 * a4;
            C.c[0] = g1 * q2_rat + a3;
            dp_cleandeg(&C, 2);
            unsigned int nb = dp_all_roots(&S->fail, roots_k, &C);
            for (unsigned int l = 0; l < nb; l++)
                list_k_push(S, nk, d_round(roots_k[l]));
        }
    }
}

/* best_norm: LLL at translation k; returns the minimum square norm and the
 * corresponding pair in (fopt, gopt) */
static __host__ __device__ double best_norm(sopt_scratch *S, zpoly *fopt, zpoly *gopt,
                                   const zpoly *f_raw, const zpoly *g_raw,
                                   const sopt_z *skew, const sopt_z *k,
                                   int max_rot)
{
    const int d = f_raw->deg;
    const int m = max_rot + 2, n = d + 1;
    double min_norm = DBL_MAX;
    zpoly *ft = &S->bn_ft, *gt = &S->bn_gt;

    zp_translation(ft, f_raw, k);
    zp_translation(gt, g_raw, k);

    /* LLL_set_matrix_from_polys */
    sopt_z tmp;
    for (int r = 1; r <= m; r++)
        for (int c = 1; c <= n; c++)
            z_set_u32(&S->B[r][c], 0);
    z_set_u32(&tmp, 1);
    for (int c = 0; c < n; c++) {
        if (c > 0)
            z_mul(&tmp, &tmp, skew);
        if (c <= m - 2)
            z_set(&S->B[c + 2][c + 1], &tmp);
        if (c <= ft->deg)
            z_mul(&S->B[1][c + 1], &tmp, &ft->c[c]);
    }
    for (int c = 0; c <= m - 2; c++) {
        z_mul(&S->B[c + 2][c + 2], &S->B[c + 2][c + 1], skew);
        z_mul(&S->B[c + 2][c + 1], &S->B[c + 2][c + 1], &gt->c[0]);
        z_mul(&S->B[c + 2][c + 2], &S->B[c + 2][c + 2], &gt->c[1]);
    }
    for (int r = 1; r <= m && !S->fail; r++)
        for (int c = 1; c <= n; c++)
            S->fail |= z_bad(&S->B[r][c]);
    if (S->fail)
        return DBL_MAX;

    lll(S, m, n);
    if (S->fail)
        return DBL_MAX;

    for (int l = 1; l <= m; l++) {
        if (S->B[l][d + 1].n == 0)
            continue;
        z_mul(&tmp, &S->B[l][1], &S->B[l][1]);
        for (int j = 2; j <= n; j++)
            z_addmul(&tmp, &S->B[l][j], &S->B[l][j]);
        if (z_bad(&tmp)) {
            S->fail = 1;
            return DBL_MAX;
        }
        double norm = z_get_d(&tmp);
        if (norm < min_norm) {
            z_set_u32(&tmp, 1);
            for (int j = 0; j <= d; j++) {
                z_divexact(&ft->c[j], &S->B[l][j + 1], &tmp);
                z_mul(&tmp, &tmp, skew);
            }
            ft->deg = d;
            min_norm = norm;
            zp_set(fopt, ft);
            zp_set(gopt, gt);
        }
        if (z_bad(&tmp)) {
            S->fail = 1;
            return DBL_MAX;
        }
    }
    return min_norm;
}

/* best_norm2, made iterative; on entry F->k and F->min_norm are set.
 * Level i writes its pair into S->bn2_f/g[i]; its output pair (fopt) is
 * S->bn2_f/g[i-1], or (fopt0, gopt0) at the top level. k0 is in/out. */
static __host__ __device__ double best_norm2(sopt_scratch *S, zpoly *fopt0,
                                    zpoly *gopt0, const zpoly *f_raw,
                                    const zpoly *g_raw, const sopt_z *skew,
                                    sopt_z *k0, int max_rot)
{
    const int d = f_raw->deg;
    bn2_frame *st = S->bn2;
    int top = 0;
    int entering = 1;
    double ret = 0;

    z_set(&st[0].k, k0);
    st[0].min_norm = DBL_MAX;

    while (top >= 0) {
        bn2_frame *F = &st[top];
        zpoly *ft = &S->bn2_f[top], *gt = &S->bn2_g[top];
        zpoly *fopt = top ? &S->bn2_f[top - 1] : fopt0;
        zpoly *gopt = top ? &S->bn2_g[top - 1] : gopt0;

        if (entering) {
            z_set(&F->best_k, &F->k);
            double norm = best_norm(S, ft, gt, f_raw, g_raw, skew, &F->k,
                                    max_rot);
            if (S->fail)
                return DBL_MAX;
            if (norm >= F->min_norm) {
                F->nroots = 0;
                F->idx = 0;
            } else {
                F->min_norm = norm;
                z_set(&F->best_k, &F->k);
                zp_set(fopt, ft);
                zp_set(gopt, gt);

                dpoly eq;
                eq.c[2] = (double) (d * (d - 1)) / 2.0 * z_get_d(&ft->c[d]);
                eq.c[1] = (double) (d - 1) * z_get_d(&ft->c[d - 1]);
                eq.c[0] = z_get_d(&ft->c[d - 2]);
                dp_cleandeg(&eq, 2);
                F->nroots = dp_all_roots(&S->fail, F->roots, &eq);
                F->idx = 0;
            }
        } else {
            /* the child updated F->k to its best k */
            if (ret < F->min_norm) {
                F->min_norm = ret;
                z_set(&F->best_k, &F->k);
                zp_set(fopt, ft);
                zp_set(gopt, gt);
            }
        }

        if (F->idx < F->nroots && top + 1 < SOPT_GPU_MAX_BN2_DEPTH) {
            double r = F->roots[F->idx++];
            z_set_d(&F->k, d_round(r));
            z_set(&st[top + 1].k, &F->k);
            st[top + 1].min_norm = F->min_norm;
            top++;
            entering = 1;
            continue;
        }
        if (F->idx < F->nroots) {
            S->fail = 1;
            return DBL_MAX;
        }
        z_set(&F->k, &F->best_k);
        ret = F->min_norm;
        top--;
        entering = 0;
        if (top >= 0)
            z_set(&st[top].k, &F->k);
    }
    z_set(k0, &st[0].k);
    return ret;
}

/* sopt_local_descent of size_optimization.cpp */
static __host__ __device__ double sopt_local_descent(sopt_scratch *S, zpoly *f_opt,
                                            zpoly *g_opt, const zpoly *f_raw,
                                            const zpoly *g_raw,
                                            int deg_rotation,
                                            unsigned int max_iter)
{
    sopt_z kt, k[SOPT_MAX_DEGREE_ROTATION + 1], mk;
    int changedt, changed[SOPT_MAX_DEGREE_ROTATION + 1];
    zpoly *ftmp = &S->tmpf;
    double logmu;
    double logmu_opt = l2_skew_lognorm_z(&S->fail, f_raw);

    for (int i = 0; i <= deg_rotation; i++) {
        z_set_u32(&k[i], 1);
        for (int it = 0; ; it++) {
            zp_rotation(ftmp, f_raw, g_raw, &k[i], i);
            logmu = l2_skew_lognorm_z(&S->fail, ftmp);
            if (logmu > logmu_opt + SOPT_LOCAL_DESCENT_GUARD) {
                z_neg(&mk, &k[i]);
                zp_rotation(ftmp, f_raw, g_raw, &mk, i);
                logmu = l2_skew_lognorm_z(&S->fail, ftmp);
                if (logmu > logmu_opt + SOPT_LOCAL_DESCENT_GUARD)
                    break;
            }
            z_mul_2exp(&k[i], &k[i], 1);
            if (it > SOPT_GPU_MAX_DOUBLINGS || S->fail || z_bad(&k[i])) {
                S->fail = 1;
                return 0;
            }
        }
    }
    z_set_u32(&kt, 1);
    for (int it = 0; ; it++) {
        zp_translation(ftmp, f_raw, &kt);
        logmu = l2_skew_lognorm_z(&S->fail, ftmp);
        if (logmu > logmu_opt + SOPT_LOCAL_DESCENT_GUARD) {
            z_neg(&mk, &kt);
            zp_translation(ftmp, f_raw, &mk);
            logmu = l2_skew_lognorm_z(&S->fail, ftmp);
            if (logmu > logmu_opt + SOPT_LOCAL_DESCENT_GUARD)
                break;
        }
        z_mul_2exp(&kt, &kt, 1);
        if (it > SOPT_GPU_MAX_DOUBLINGS || S->fail || z_bad(&kt)) {
            S->fail = 1;
            return 0;
        }
    }

    zp_set(f_opt, f_raw);
    zp_set(g_opt, g_raw);

    sopt_z one;
    z_set_u32(&one, 1);
    for (unsigned int iter = 0; iter < max_iter && !S->fail; iter++) {
        changedt = 0;
        for (int i = 0; i <= deg_rotation; i++)
            changed[i] = 0;

        zp_translation(ftmp, f_opt, &kt);
        logmu = l2_skew_lognorm_z(&S->fail, ftmp);
        if (logmu < logmu_opt) {
            changedt = 1;
            logmu_opt = logmu;
            zp_set(f_opt, ftmp);
            zp_translation(g_opt, g_opt, &kt);
        } else {
            z_neg(&mk, &kt);
            zp_translation(ftmp, f_opt, &mk);
            logmu = l2_skew_lognorm_z(&S->fail, ftmp);
            if (logmu < logmu_opt) {
                changedt = 1;
                logmu_opt = logmu;
                zp_set(f_opt, ftmp);
                zp_translation(g_opt, g_opt, &mk);
            }
        }

        for (int i = 0; i <= deg_rotation; i++) {
            zp_rotation(ftmp, f_opt, g_opt, &k[i], i);
            logmu = l2_skew_lognorm_z(&S->fail, ftmp);
            if (logmu < logmu_opt) {
                changed[i] = 1;
                logmu_opt = logmu;
                zp_set(f_opt, ftmp);
            } else {
                z_neg(&mk, &k[i]);
                zp_rotation(ftmp, f_opt, g_opt, &mk, i);
                logmu = l2_skew_lognorm_z(&S->fail, ftmp);
                if (logmu < logmu_opt) {
                    changed[i] = 1;
                    logmu_opt = logmu;
                    zp_set(f_opt, ftmp);
                }
            }
        }

        int is_finished = changedt == 0 && z_cmp(&kt, &one) == 0;
        for (int i = 0; i <= deg_rotation && is_finished; i++)
            is_finished = changed[i] == 0 && z_cmp(&k[i], &one) == 0;
        if (is_finished)
            break;

        if (changedt == 1)
            z_mul_2exp(&kt, &kt, 1);
        else if (z_cmp(&kt, &one) > 0)
            z_fdiv_q_2exp(&kt, &kt, 1);
        for (int i = 0; i <= deg_rotation; i++) {
            if (changed[i] == 1)
                z_mul_2exp(&k[i], &k[i], 1);
            else if (z_cmp(&k[i], &one) > 0)
                z_fdiv_q_2exp(&k[i], &k[i], 1);
        }
        if (z_bad(&kt))
            S->fail = 1;
        for (int i = 0; i <= deg_rotation; i++)
            if (z_bad(&k[i]))
                S->fail = 1;
    }
    return logmu_opt;
}

static __host__ __device__ void sort_doubles(double *a, int n)
{
    for (int i = 1; i < n; i++) {
        double v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

/* first phase of size_optimization_aux: base lognorm and the sorted,
 * deduplicated translation list in S->list_k */
static __host__ __device__ double sopt_prepare(sopt_scratch *S, int *nk_out,
                                      unsigned int sopt_effort)
{
    zpoly *f_raw = &S->f_raw, *g_raw = &S->g_raw;
    const int d = f_raw->deg;

    double base_lognorm = l2_skew_lognorm_z(&S->fail, f_raw)
                        + expected_rotation_gain(S, f_raw, g_raw);
    if (S->fail)
        return 0;

    int nk = 0;
    if (d == 6)
        sopt_find_translations_deg6(S, &nk, f_raw, g_raw, sopt_effort);
    else if (d == 5)
        sopt_find_translations_deg5(S, &nk, f_raw, g_raw, sopt_effort);
    list_k_push(S, &nk, 0.0);
    if (S->fail)
        return 0;
    sort_doubles(S->list_k, nk);
    int m = 0;
    for (int i = 0; i < nk; i++)
        if (i == 0 || S->list_k[i] != S->list_k[i - 1])
            S->list_k[m++] = S->list_k[i];
    *nk_out = m;
    return base_lognorm;
}

/* second phase for one translation k: best_norm2, local descent, gain.
 * The result pair is in S->fld/gld; kd_out gets the rounded k for the
 * duplicate check. */
static __host__ __device__ double sopt_one_k(sopt_scratch *S, double kd_in,
                                    double skew_seed, double *kd_out)
{
    zpoly *f_raw = &S->f_raw, *g_raw = &S->g_raw;
    const int d = f_raw->deg;

    sopt_z skew, ki;
    z_set_d(&skew, skew_seed);
    z_set_d(&ki, kd_in);
    best_norm2(S, &S->ft, &S->gt, f_raw, g_raw, &skew, &ki, d - 2);
    if (S->fail)
        return 0;

    double kd = z_get_d(&ki);
    if (fabs(kd) >= 9007199254740992.0) {   /* k must be exact in double */
        S->fail = 1;
        return 0;
    }
    *kd_out = kd;

    double lognorm = sopt_local_descent(S, &S->fld, &S->gld, &S->ft, &S->gt,
                                        d - 2, SOPT_DEFAULT_MAX_STEPS);
    lognorm += expected_rotation_gain(S, &S->fld, &S->gld);
    return lognorm;
}

/* size_optimization_aux with max_rot = d - 2; the sequential form, used by
 * the host runner */
static __host__ __device__ void size_optimization_gpu(sopt_scratch *S,
                                             double skew_seed,
                                             unsigned int sopt_effort)
{
    int nk = 0;
    double best_lognorm = sopt_prepare(S, &nk, sopt_effort);
    if (S->fail)
        return;

    zp_set(&S->fbest, &S->f_raw);
    zp_set(&S->gbest, &S->g_raw);

    int nk_opt = 0;
    for (int i = 0; i < nk && !S->fail; i++) {
        double kd;
        double lognorm = sopt_one_k(S, S->list_k[i], skew_seed, &kd);
        if (S->fail)
            return;
        int seen = 0;
        for (int j = 0; j < nk_opt && !seen; j++)
            seen = S->list_k_opt[j] == kd;
        if (seen)
            continue;
        S->list_k_opt[nk_opt++] = kd;
        if (lognorm < best_lognorm) {
            best_lognorm = lognorm;
            zp_set(&S->fbest, &S->fld);
            zp_set(&S->gbest, &S->gld);
        }
    }
}

/* --------------------------------------------------- kernel and runner */

static __host__ __device__ void pair_to_zpoly(zpoly *r, const sopt_z *c, int deg)
{
    r->deg = deg;
    for (int i = 0; i <= deg; i++)
        r->c[i] = c[i];
}

/* per-candidate output of the prep kernel */
struct sopt_prep_t {
    int fail;
    int nk;
    double base_lognorm;
    double list_k[SOPT_GPU_MAX_LIST_K];
};

/* one (candidate, translation) work item and its light result; the winning
 * polynomial is recomputed by the emit kernel, so the result stays small */
struct sopt_item_t {
    uint32_t cand;
    double k;
};

struct sopt_lres_t {
    int fail;
    double lognorm;
};

static __global__ void sopt_prep_kernel(const sopt_pair_t *pairs,
                                        sopt_prep_t *prep, uint32_t *nk_out,
                                        sopt_scratch *scratch,
                                        uint32_t n, unsigned int effort)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    sopt_scratch *S = &scratch[tid];
    for (uint32_t idx = tid; idx < n; idx += stride) {
        const sopt_pair_t *P = &pairs[idx];
        sopt_prep_t *R = &prep[idx];
        S->fail = 0;
        R->fail = 1;
        R->nk = 0;
        nk_out[idx] = 0;
        if (P->deg < 2 || P->deg > SOPT_GPU_MAXDEG)
            continue;
        pair_to_zpoly(&S->f_raw, P->f, P->deg);
        pair_to_zpoly(&S->g_raw, P->g, 1);
        if (zp_bad(&S->f_raw) || zp_bad(&S->g_raw))
            continue;
        int nk = 0;
        double base = sopt_prepare(S, &nk, effort);
        if (S->fail)
            continue;
        R->fail = 0;
        R->nk = nk;
        R->base_lognorm = base;
        for (int i = 0; i < nk; i++)
            R->list_k[i] = S->list_k[i];
        nk_out[idx] = nk;
    }
}

/* write the (candidate, translation) items of every candidate; the item
 * range of candidate c is [off[c], off[c + 1]), in list_k order */
static __global__ void sopt_build_kernel(const sopt_prep_t *prep,
                                         const uint32_t *off,
                                         sopt_item_t *items, double *key,
                                         uint32_t *ord, uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t c = tid; c < n; c += stride)
        for (int i = 0; i < prep[c].nk; i++) {
            items[off[c] + i] = {c, prep[c].list_k[i]};
            key[off[c] + i] = fabs(prep[c].list_k[i]);
            ord[off[c] + i] = off[c] + i;
        }
}

/* ord orders the items by |k|, so the threads of one warp run similar
 * iteration counts; lres stays indexed by the original item index */
static __global__ void sopt_item_kernel(const sopt_pair_t *pairs,
                                        const sopt_item_t *items,
                                        const uint32_t *ord,
                                        sopt_lres_t *lres,
                                        sopt_scratch *scratch, uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    sopt_scratch *S = &scratch[tid];
    for (uint32_t it = tid; it < n; it += stride) {
        uint32_t i = ord[it];
        const sopt_pair_t *P = &pairs[items[i].cand];
        sopt_lres_t *R = &lres[i];
        S->fail = 0;
        R->fail = 1;
        pair_to_zpoly(&S->f_raw, P->f, P->deg);
        pair_to_zpoly(&S->g_raw, P->g, 1);
        double kd;
        double lognorm = sopt_one_k(S, items[i].k, P->skew, &kd);
        if (S->fail || zp_bad(&S->fld) || zp_bad(&S->gld))
            continue;
        R->fail = 0;
        R->lognorm = lognorm;
    }
}

/* pick per candidate the strictly best item, first index on a tie; two
 * items with the same rounded k give the same polynomial and the same
 * lognorm, so this equals the sequential duplicate-k skip. win[c] is the
 * item index, -1 when the raw pair stays the best, -2 on failure. */
static __global__ void sopt_reduce_kernel(const sopt_prep_t *prep,
                                          const uint32_t *off,
                                          const sopt_lres_t *lres,
                                          int32_t *win, uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t c = tid; c < n; c += stride) {
        win[c] = -2;
        if (prep[c].fail)
            continue;
        double best = prep[c].base_lognorm;
        int32_t w = -1;
        int anyfail = 0;
        for (uint32_t j = off[c]; j < off[c + 1] && !anyfail; j++) {
            anyfail = lres[j].fail;
            if (!anyfail && lres[j].lognorm < best) {
                best = lres[j].lognorm;
                w = j;
            }
        }
        if (!anyfail)
            win[c] = w;
    }
}

/* recompute the winning translation of each candidate and write the
 * optimized pair in place */
static __global__ void sopt_emit_kernel(sopt_pair_t *pairs,
                                        const sopt_item_t *items,
                                        const int32_t *win,
                                        sopt_scratch *scratch, uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    sopt_scratch *S = &scratch[tid];
    for (uint32_t c = tid; c < n; c += stride) {
        sopt_pair_t *P = &pairs[c];
        if (win[c] == -1) {
            P->fail = 0;        /* the raw pair stays the best */
            continue;
        }
        P->fail = 1;
        if (win[c] < 0)
            continue;
        S->fail = 0;
        pair_to_zpoly(&S->f_raw, P->f, P->deg);
        pair_to_zpoly(&S->g_raw, P->g, 1);
        double kd;
        sopt_one_k(S, items[win[c]].k, P->skew, &kd);
        if (S->fail || zp_bad(&S->fld) || zp_bad(&S->gld) ||
            S->fld.deg != P->deg)
            continue;
        for (int i = 0; i <= P->deg; i++)
            P->f[i] = S->fld.c[i];
        P->g[0] = S->gld.c[0];
        P->g[1] = S->gld.c[1];
        P->fail = 0;
    }
}

/* device buffers persist across calls, so no cudaMalloc or cudaFree runs
 * in the steady state; both stop the whole device and would stall the
 * raw-search kernels that another host thread launches on the default
 * stream. The flush thread joins before the next flush starts, so no two
 * calls run at the same time. */
static cudaStream_t sopt_stream;
static sopt_pair_t *d_pairs;
static sopt_prep_t *d_prep;
static sopt_scratch *d_scratch;
static uint32_t *d_nk, *d_off;
static int32_t *d_win;
static uint32_t sopt_slots;

static sopt_item_t *d_items;
static sopt_lres_t *d_lres;
static double *d_key, *d_key_alt;
static uint32_t *d_ord, *d_ord_alt;
static void *d_sort_tmp;
static size_t sopt_item_cap, sopt_sort_tmp_cap;

static void sopt_grow(void **p, size_t sz)
{
    if (*p)
        CUDA_CHECK(cudaFree(*p));
    CUDA_CHECK(cudaMalloc(p, sz));
}

/* make the per-item buffers hold ni items; a grow synchronizes the device
 * (cudaFree does), but it happens a few times per process */
static void sopt_grow_items(uint32_t ni)
{
    if (ni <= sopt_item_cap)
        return;
    sopt_grow((void **) &d_items, (size_t) ni * sizeof(sopt_item_t));
    sopt_grow((void **) &d_lres, (size_t) ni * sizeof(sopt_lres_t));
    sopt_grow((void **) &d_key, (size_t) ni * sizeof(double));
    sopt_grow((void **) &d_key_alt, (size_t) ni * sizeof(double));
    sopt_grow((void **) &d_ord, (size_t) ni * sizeof(uint32_t));
    sopt_grow((void **) &d_ord_alt, (size_t) ni * sizeof(uint32_t));
    sopt_item_cap = ni;

    size_t bytes = 0;
    cub::DoubleBuffer<double> dk(NULL, NULL);
    cub::DoubleBuffer<uint32_t> dv(NULL, NULL);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL, bytes, dk, dv, ni));
    if (bytes > sopt_sort_tmp_cap) {
        sopt_grow(&d_sort_tmp, bytes);
        sopt_sort_tmp_cap = bytes;
    }
}

static void sopt_init(void)
{
    if (sopt_stream)
        return;
    /* a non-blocking stream, so this run overlaps the raw-search kernels
     * that another host thread launches on the default stream */
    CUDA_CHECK(cudaStreamCreateWithFlags(&sopt_stream,
                                         cudaStreamNonBlocking));

    size_t freeb, totalb;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    uint32_t slots = (uint32_t) ((freeb / 2) / sizeof(sopt_scratch));
    if (slots > SOPT_GPU_MAX_SLOTS)
        slots = SOPT_GPU_MAX_SLOTS;
    if (slots < 64)
        slots = 64;
    slots &= ~63u;
    sopt_slots = slots;

    CUDA_CHECK(cudaMalloc(&d_pairs, SOPT_GPU_CHUNK * sizeof(sopt_pair_t)));
    CUDA_CHECK(cudaMalloc(&d_prep, SOPT_GPU_CHUNK * sizeof(sopt_prep_t)));
    CUDA_CHECK(cudaMalloc(&d_scratch, (size_t) slots * sizeof(sopt_scratch)));
    CUDA_CHECK(cudaMalloc(&d_nk, SOPT_GPU_CHUNK * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_off, (SOPT_GPU_CHUNK + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_win, SOPT_GPU_CHUNK * sizeof(int32_t)));
}

extern "C" void gps1_sopt_run(sopt_pair_t *pairs, uint32_t n,
                              unsigned int effort)
{
    if (n == 0)
        return;

    int timing = getenv("GPS1_SOPT_TIME") != NULL;
    double t_prep = 0, t_item = 0, t_host = 0, t0;
    uint64_t n_items = 0;

    sopt_init();
    cudaStream_t st = sopt_stream;
    uint32_t slots = sopt_slots;

    uint32_t chunk = n < SOPT_GPU_CHUNK ? n : SOPT_GPU_CHUNK;
    std::vector<uint32_t> nk(chunk), off(chunk + 1);

    for (uint32_t poff = 0; poff < n; poff += chunk) {
        uint32_t cnt = n - poff < chunk ? n - poff : chunk;
        CUDA_CHECK(cudaMemcpyAsync(d_pairs, pairs + poff,
                                   cnt * sizeof(sopt_pair_t),
                                   cudaMemcpyHostToDevice, st));

        t0 = omp_get_wtime();
        uint32_t tp = cnt < slots ? cnt : slots;
        sopt_prep_kernel<<<(tp + 63) / 64, 64, 0, st>>>(d_pairs, d_prep, d_nk,
                                                        d_scratch, cnt, effort);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(nk.data(), d_nk, cnt * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        t_prep += omp_get_wtime() - t0;

        off[0] = 0;
        for (uint32_t c = 0; c < cnt; c++)
            off[c + 1] = off[c] + nk[c];
        uint32_t ni = off[cnt];
        n_items += ni;

        t0 = omp_get_wtime();
        CUDA_CHECK(cudaMemcpyAsync(d_off, off.data(),
                                   (cnt + 1) * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, st));
        if (ni) {
            sopt_grow_items(ni);
            sopt_build_kernel<<<(cnt + 63) / 64, 64, 0, st>>>(d_prep, d_off,
                                                              d_items, d_key,
                                                              d_ord, cnt);
            CUDA_CHECK(cudaGetLastError());
            cub::DoubleBuffer<double> dk(d_key, d_key_alt);
            cub::DoubleBuffer<uint32_t> dv(d_ord, d_ord_alt);
            size_t tmp_bytes = sopt_sort_tmp_cap;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_sort_tmp, tmp_bytes,
                                                       dk, dv, ni, 0, 64,
                                                       st));
            uint32_t ti = ni < slots ? ni : slots;
            sopt_item_kernel<<<(ti + 63) / 64, 64, 0, st>>>(d_pairs, d_items,
                                                            dv.Current(),
                                                            d_lres,
                                                            d_scratch, ni);
            CUDA_CHECK(cudaGetLastError());
        }
        sopt_reduce_kernel<<<(cnt + 63) / 64, 64, 0, st>>>(d_prep, d_off,
                                                           d_lres, d_win, cnt);
        CUDA_CHECK(cudaGetLastError());
        uint32_t te = cnt < slots ? cnt : slots;
        sopt_emit_kernel<<<(te + 63) / 64, 64, 0, st>>>(d_pairs, d_items,
                                                        d_win, d_scratch, cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(st));
        t_item += omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        CUDA_CHECK(cudaMemcpyAsync(pairs + poff, d_pairs,
                                   cnt * sizeof(sopt_pair_t),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        t_host += omp_get_wtime() - t0;
    }
    if (timing)
        fprintf(stderr,
                "# gps1_sopt_run: %u cands %lu items prep %.2fs item %.2fs "
                "host %.2fs\n", n, (unsigned long) n_items, t_prep, t_item,
                t_host);
}

/* the same optimization run on the host CPU; used by -sopt-test to check
 * the port on a machine without a GPU */
extern "C" void gps1_sopt_run_host(sopt_pair_t *pairs, uint32_t n,
                                   unsigned int effort)
{
    sopt_scratch *S = (sopt_scratch *) malloc(sizeof(sopt_scratch));
    if (S == NULL) {
        fprintf(stderr, "gps1_sopt_run_host: out of memory\n");
        exit(1);
    }
    for (uint32_t i = 0; i < n; i++) {
        sopt_pair_t *P = &pairs[i];
        S->fail = 0;
        P->fail = 1;
        if (P->deg < 2 || P->deg > SOPT_GPU_MAXDEG)
            continue;
        pair_to_zpoly(&S->f_raw, P->f, P->deg);
        pair_to_zpoly(&S->g_raw, P->g, 1);
        if (zp_bad(&S->f_raw) || zp_bad(&S->g_raw))
            continue;
        size_optimization_gpu(S, P->skew, effort);
        if (S->fail || zp_bad(&S->fbest) || zp_bad(&S->gbest) ||
            S->fbest.deg != P->deg)
            continue;
        for (int j = 0; j <= P->deg; j++)
            P->f[j] = S->fbest.c[j];
        for (int j = 0; j <= 1; j++)
            P->g[j] = S->gbest.c[j];
        P->fail = 0;
    }
    free(S);
}
