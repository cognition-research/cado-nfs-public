#include "cado.h" // IWYU pragma: keep

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <gmp.h>

#include "macros.h"
#include "omp_proxy.h"
#include "par_mul.hpp"

/* A product is split recursively to this depth at most (3^depth pieces),
 * so that the temporaries of the split take at most about 2 * 1.5^depth
 * operand sizes at a time. PAR_MUL_DEPTH in the environment overrides it. */
static const int par_mul_depth = [] {
    const char * s = getenv("PAR_MUL_DEPTH");
    return s ? atoi(s) : 4;
}();

/* mpz_mul products with both operands smaller than this stay with GMP */
static const mp_size_t par_mul_min = 1 << 20;

/* PAR_MUL_GPU names the shared library built from sqrt/gpu_mul.cu; the
 * products then go to the GPUs unsplit, one product per device at a time,
 * and are split only above gpu_max limbs (the transform length). */
typedef void (*gpu_mul_t)(mp_ptr, mp_srcptr, size_t, mp_srcptr, size_t);
static const gpu_mul_t gpu_mul = [] () -> gpu_mul_t {
    const char * lib = getenv("PAR_MUL_GPU");
    if (!lib)
        return nullptr;
    void * h = dlopen(lib, RTLD_NOW);
    ASSERT_ALWAYS(h);
    typedef int (*devices_t)();
    const devices_t devices = (devices_t) dlsym(h, "gpu_mul_devices");
    ASSERT_ALWAYS(devices);
    const int n = devices();
    fprintf(stderr, "par_mul: %d GPUs\n", n);
    ASSERT_ALWAYS(n > 0);
    return (gpu_mul_t) dlsym(h, "gpu_mul");
}();
static const mp_size_t gpu_max = (mp_size_t) 1 << 31;

static const int all_threads = omp_get_max_threads();

/* The split depth of a product started by the calling thread: the
 * program's threads are shared equally among the threads of the
 * enclosing OpenMP teams, which all multiply at the same time. */
static int split_depth()
{
    int budget = all_threads;
    for (int l = 1; l <= omp_get_level(); l++)
        budget /= omp_get_team_size(l);
    int depth = 0;
    for (; depth < par_mul_depth && budget >= 3; budget /= 3)
        depth++;
    return depth;
}

typedef std::unique_ptr<mp_limb_t[]> limbs;

static limbs new_limbs(mp_size_t n)
{
    return limbs(new mp_limb_t[n]);
}

/* Run the tasks at the same time, the last one on the calling thread */
static void parallel(std::vector<std::function<void()>> const & tasks)
{
    std::vector<std::thread> threads;
    for (size_t i = 0; i + 1 < tasks.size(); i++)
        threads.emplace_back(tasks[i]);
    tasks.back()();
    for (auto & t : threads)
        t.join();
}

static mp_size_t normalize(mp_srcptr p, mp_size_t n)
{
    for (; n > 0 && p[n - 1] == 0; n--)
        ;
    return n;
}

static void par_mul(mp_ptr rp, mp_srcptr ap, mp_size_t na, mp_srcptr bp, mp_size_t nb, int depth)
{
    ASSERT_ALWAYS(na >= nb && nb >= 1);
    /* A split costs 1.5 times the work of the product it replaces (GMP
     * multiplies in about linear time at these sizes), so it pays only
     * when the pieces get their own cores. Below 4 limbs the pieces are
     * not smaller than the input. */
    if (nb < 4 || (depth <= 0 && (!gpu_mul || na + nb <= gpu_max))) {
        if (gpu_mul && nb >= par_mul_min)
            gpu_mul(rp, ap, na, bp, nb);
        else
            mpn_mul(rp, ap, na, bp, nb);
        return;
    }
    depth--;
    const mp_size_t h = na / 2;
    if (nb <= h) {
        /* b reaches at most the low half of a: a1*b and a0*b */
        limbs t = new_limbs(na - h + nb);
        parallel({
            [&] { par_mul(t.get(), ap + h, na - h, bp, nb, depth); },
            [&] { par_mul(rp, ap, h, bp, nb, depth); },
        });
        ASSERT_ALWAYS(mpn_add(rp + h, t.get(), na - h + nb, rp + h, nb) == 0);
        return;
    }
    /* Karatsuba: a = a1 B^h + a0, b = b1 B^h + b0,
     * a*b = a1*b1 B^2h + ((a0+a1)*(b0+b1) - a0*b0 - a1*b1) B^h + a0*b0 */
    const mp_size_t na1 = na - h, nb1 = nb - h;
    const mp_size_t nsa = na1 + 1, nsb = std::max(h, nb1) + 1;
    limbs sa = new_limbs(nsa), sb = new_limbs(nsb), t = new_limbs(nsa + nsb);
    sa[na1] = mpn_add(sa.get(), ap + h, na1, ap, h);
    if (h >= nb1)
        sb[h] = mpn_add(sb.get(), bp, h, bp + h, nb1);
    else
        sb[nb1] = mpn_add(sb.get(), bp + h, nb1, bp, h);
    parallel({
        [&] { par_mul(rp, ap, h, bp, h, depth); },
        [&] { par_mul(rp + 2 * h, ap + h, na1, bp + h, nb1, depth); },
        [&] { par_mul(t.get(), sa.get(), nsa, sb.get(), nsb, depth); },
    });
    sa.reset();
    sb.reset();
    ASSERT_ALWAYS(mpn_sub(t.get(), t.get(), nsa + nsb, rp, 2 * h) == 0);
    ASSERT_ALWAYS(mpn_sub(t.get(), t.get(), nsa + nsb, rp + 2 * h, na1 + nb1) == 0);
    const mp_size_t nt = normalize(t.get(), nsa + nsb);
    ASSERT_ALWAYS(nt <= na + nb - h);
    ASSERT_ALWAYS(mpn_add(rp + h, rp + h, na + nb - h, t.get(), nt) == 0);
}

void par_mul(mp_ptr rp, mp_srcptr ap, mp_size_t na, mp_srcptr bp, mp_size_t nb)
{
    par_mul(rp, ap, na, bp, nb, gpu_mul ? 0 : split_depth());
}

/* mpz_mul, mpz_addmul and mpz_submul, as seen by the whole program (GMP is
 * a shared library, so the definitions here take precedence over its own).
 * Small products go to GMP unchanged. */
typedef void (*mpz_mul_t)(mpz_ptr, mpz_srcptr, mpz_srcptr);

static mpz_mul_t gmp(const char * name)
{
    return (mpz_mul_t) dlsym(RTLD_NEXT, name);
}

static bool small(mpz_srcptr a, mpz_srcptr b)
{
    return std::min(mpz_size(a), mpz_size(b)) < par_mul_min;
}

extern "C" void __gmpz_mul(mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
    static const mpz_mul_t gmp_mul = gmp("__gmpz_mul");
    if (small(a, b)) {
        gmp_mul(r, a, b);
        return;
    }
    mp_size_t na = mpz_size(a), nb = mpz_size(b);
    if (na < nb) {
        std::swap(a, b);
        std::swap(na, nb);
    }
    ASSERT_ALWAYS(na + nb < INT_MAX);
    cxx_mpz t;
    par_mul(mpz_limbs_write(t, na + nb), mpz_limbs_read(a), na, mpz_limbs_read(b), nb);
    mpz_limbs_finish(t, mpz_sgn(a) == mpz_sgn(b) ? na + nb : -(na + nb));
    mpz_swap(r, t);
}

extern "C" void __gmpz_addmul(mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
    static const mpz_mul_t gmp_addmul = gmp("__gmpz_addmul");
    if (small(a, b)) {
        gmp_addmul(r, a, b);
        return;
    }
    cxx_mpz t;
    __gmpz_mul(t, a, b);
    mpz_add(r, r, t);
}

extern "C" void __gmpz_submul(mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
    static const mpz_mul_t gmp_submul = gmp("__gmpz_submul");
    if (small(a, b)) {
        gmp_submul(r, a, b);
        return;
    }
    cxx_mpz t;
    __gmpz_mul(t, a, b);
    mpz_sub(r, r, t);
}

/* Division through a Newton reciprocal built on the parallel products
 * above. GMP divides in one thread. */
typedef void (*mpz_div_t)(mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr);

static mp_bitcnt_t bits(mpz_srcptr a)
{
    return mpz_sizeinbase(a, 2);
}

/* x <- 2^(bits(d)-1+p) / d within a few units, so x has about p bits, by
 * Newton iterations x <- x + x (2^E - d x) / 2^E whose precision doubles at
 * every step. Only the top p+32 bits of d take part. */
static void reciprocal(cxx_mpz & x, mpz_srcptr d, mp_bitcnt_t p)
{
    static const mpz_div_t gmp_tdiv_qr = (mpz_div_t) dlsym(RTLD_NEXT, "__gmpz_tdiv_qr");
    const mp_bitcnt_t s = bits(d) > p + 32 ? bits(d) - (p + 32) : 0;
    const mp_bitcnt_t E = bits(d) - 1 + p - s;
    cxx_mpz b, r;
    mpz_tdiv_q_2exp(b, d, s);
    if (p <= 4096) {
        x = 1;
        mpz_mul_2exp(x, x, E);
        gmp_tdiv_qr(x, r, x, b);
        return;
    }
    const mp_bitcnt_t h = p / 2 + 32;
    reciprocal(x, d, h);
    mpz_mul_2exp(x, x, p - h);
    /* e <- (2^E - b x) / 2^k, then x <- x + x e / 2^(E-k); e has about
     * p/2 bits, x e / 2^(E-k) is within a unit of the exact update */
    const mp_bitcnt_t k = E > p + 32 ? E - (p + 32) : 0;
    cxx_mpz e = 1;
    mpz_mul_2exp(e, e, E);
    mpz_submul(e, b, x);
    mpz_tdiv_q_2exp(e, e, k);
    mpz_mul(e, e, x);
    mpz_tdiv_q_2exp(e, e, E - k);
    mpz_add(x, x, e);
}

void par_tdiv_qr(cxx_mpz & q, cxx_mpz & r, mpz_srcptr a, mpz_srcptr d)
{
    ASSERT_ALWAYS(mpz_sgn(a) > 0 && mpz_sgn(d) > 0 && bits(a) >= bits(d));
    const mp_bitcnt_t p = bits(a) - bits(d) + 33;
    cxx_mpz x;
    reciprocal(x, d, p);
    /* q <- (a / 2^t) x / 2^(bits(d)-1+p-t), within a few units */
    const mp_bitcnt_t t = bits(a) > p + 32 ? bits(a) - (p + 32) : 0;
    mpz_tdiv_q_2exp(q, a, t);
    mpz_mul(q, q, x);
    mpz_tdiv_q_2exp(q, q, bits(d) - 1 + p - t);
    mpz_mul(r, q, d);
    mpz_sub(r, a, r);
    for (; mpz_sgn(r) < 0; mpz_sub_ui(q, q, 1))
        mpz_add(r, r, d);
    for (; mpz_cmp(r, d) >= 0; mpz_add_ui(q, q, 1))
        mpz_sub(r, r, d);
}

/* divisions whose divisor or quotient is small stay with GMP */
static bool small_division(mpz_srcptr a, mpz_srcptr d)
{
    return mpz_size(d) < par_mul_min || mpz_size(a) < mpz_size(d) + par_mul_min;
}

static mpz_srcptr abs_view(mpz_t v, mpz_srcptr a)
{
    return mpz_roinit_n(v, mpz_limbs_read(a), mpz_size(a));
}

extern "C" void __gmpz_tdiv_qr(mpz_ptr q, mpz_ptr r, mpz_srcptr a, mpz_srcptr d)
{
    static const mpz_div_t gmp_tdiv_qr = (mpz_div_t) dlsym(RTLD_NEXT, "__gmpz_tdiv_qr");
    if (small_division(a, d)) {
        gmp_tdiv_qr(q, r, a, d);
        return;
    }
    mpz_t va, vd;
    cxx_mpz tq, tr;
    par_tdiv_qr(tq, tr, abs_view(va, a), abs_view(vd, d));
    if (mpz_sgn(a) != mpz_sgn(d))
        mpz_neg(tq, tq);
    if (mpz_sgn(a) < 0)
        mpz_neg(tr, tr);
    mpz_swap(q, tq);
    mpz_swap(r, tr);
}

extern "C" void __gmpz_tdiv_q(mpz_ptr q, mpz_srcptr a, mpz_srcptr d)
{
    cxx_mpz r;
    __gmpz_tdiv_qr(q, r, a, d);
}

extern "C" void __gmpz_mod(mpz_ptr r, mpz_srcptr a, mpz_srcptr d)
{
    cxx_mpz q, ad;
    mpz_abs(ad, d);
    __gmpz_tdiv_qr(q, r, a, d);
    if (mpz_sgn(r) < 0)
        mpz_add(r, r, ad);
}

/* r <- ({ap, na} * {bp, nb}) / B^drop */
static void mul_top(cxx_mpz & r, mp_srcptr ap, mp_size_t na, mp_srcptr bp, mp_size_t nb, mp_size_t drop)
{
    if (na < nb) {
        std::swap(ap, bp);
        std::swap(na, nb);
    }
    limbs t = new_limbs(na + nb);
    par_mul(t.get(), ap, na, bp, nb);
    ASSERT_ALWAYS(na + nb > drop);
    mpz_t v;
    mpz_set(r, mpz_roinit_n(v, t.get() + drop, na + nb - drop));
}

static void mul_top(cxx_mpz & r, mp_srcptr ap, mp_size_t na, cxx_mpz const & b, mp_size_t drop)
{
    mul_top(r, ap, na, mpz_limbs_read(b), mpz_size(b), drop);
}

/* y <- B^p / sqrt(q) with p limbs of precision, where q = {qp, nq} / B^nq
 * lies in [1/4, 1), by Newton iterations on y <- y (3 - q y^2) / 2 whose
 * precision doubles, less one guard limb, at every step. */
static void inverse_sqrt(cxx_mpz & y, mp_srcptr qp, mp_size_t nq, mp_size_t p)
{
    const mp_size_t base = 1024;
    std::vector<mp_size_t> precisions;
    for (; p > base; p = (p + 2) / 2)
        precisions.push_back(p);
    /* y = B^2p / floor(sqrt(top 2p limbs of q)) */
    mpz_t v;
    cxx_mpz s;
    mpz_sqrt(s, mpz_roinit_n(v, qp + nq - 2 * p, 2 * p));
    y = 1;
    mpz_mul_2exp(y, y, 2 * p * GMP_NUMB_BITS);
    mpz_tdiv_q(y, y, s);
    cxx_mpz y2, t, e;
    for (auto it = precisions.rbegin(); it != precisions.rend(); ++it) {
        const mp_size_t p1 = *it;
        ASSERT_ALWAYS(p1 <= 2 * p - 1 && 2 * p1 <= nq);
        mpz_mul(y2, y, y);
        /* e <- 3 B^(p1+2) - (top 2 p1 limbs of q) * y^2 / B^(2p+p1-2) */
        mul_top(t, qp + nq - 2 * p1, 2 * p1, y2, 2 * p + p1 - 2);
        e = 3;
        mpz_mul_2exp(e, e, (p1 + 2) * GMP_NUMB_BITS);
        mpz_sub(e, e, t);
        /* y <- y e / (2 B^(p+2)) */
        mul_top(y, mpz_limbs_read(e), mpz_size(e), y, p + 2);
        mpz_tdiv_q_2exp(y, y, 1);
        p = p1;
    }
}

cxx_mpz par_sqrtrem(cxx_mpz & root, mp_srcptr pp, mp_size_t np)
{
    ASSERT_ALWAYS(np > 0 && pp[np - 1] != 0);
    /* q = p << shift has an even limb count nq and a top limb >= B/4, so
     * that q / B^nq lies in [1/4, 1); the root of q is the root of p
     * shifted by shift/2 */
    const mp_size_t lo = 2 + (np & 1);
    const mp_size_t nq = np + lo;
    const unsigned int shift = (unsigned int) __builtin_clzl(pp[np - 1]) & ~1U;
    limbs q = new_limbs(nq);
    mpn_zero(q.get(), lo);
    if (shift)
        mpn_lshift(q.get() + lo, pp, np, shift);
    else
        mpn_copyi(q.get() + lo, pp, np);
    const mp_size_t h = nq / 2;

    cxx_mpz y;
    inverse_sqrt(y, q.get(), nq, h);
    /* root <- (top h+2 limbs of q) * y / B^(h+2), then >> shift/2 */
    mul_top(root, q.get() + h - 2, h + 2, y, h + 2);
    mpz_tdiv_q_2exp(root, root, shift / 2 + lo * (GMP_NUMB_BITS / 2));
    q.reset();
    y = 0;

    /* d <- root^2 - p, small: correct root by a few units */
    const mp_size_t nr = mpz_size(root);
    limbs r2 = new_limbs(2 * nr);
    par_mul(r2.get(), mpz_limbs_read(root), nr, mpz_limbs_read(root), nr);
    const mp_size_t n2 = normalize(r2.get(), 2 * nr);
    const int cmp = n2 != np ? (n2 < np ? -1 : 1) : mpn_cmp(r2.get(), pp, np);
    limbs dd = new_limbs(std::max(n2, np));
    if (cmp < 0)
        mpn_sub(dd.get(), pp, np, r2.get(), n2);
    else
        mpn_sub(dd.get(), r2.get(), n2, pp, np);
    r2.reset();
    const mp_size_t nd = normalize(dd.get(), std::max(n2, np));
    ASSERT_ALWAYS(nd <= nr + 1);
    mpz_t v;
    cxx_mpz d;
    mpz_set(d, mpz_roinit_n(v, dd.get(), nd));
    if (cmp < 0)
        mpz_neg(d, d);
    dd.reset();
    for (int i = 0; mpz_sgn(d) > 0; i++) {
        ASSERT_ALWAYS(i < 8);
        /* (root-1)^2 - p = d - 2 root + 1 */
        mpz_submul_ui(d, root, 2);
        mpz_add_ui(d, d, 1);
        mpz_sub_ui(root, root, 1);
    }
    for (int i = 0; ; i++) {
        /* (root+1)^2 - p = d + 2 root + 1 */
        cxx_mpz step;
        mpz_mul_2exp(step, root, 1);
        mpz_add_ui(step, step, 1);
        if (mpz_cmpabs(d, step) < 0)
            break;
        ASSERT_ALWAYS(i < 8);
        mpz_add(d, d, step);
        mpz_add_ui(root, root, 1);
    }
    mpz_neg(d, d);
    return d;
}
