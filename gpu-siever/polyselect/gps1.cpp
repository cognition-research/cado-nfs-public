/* gps1: CUDA port of CADO-NFS polyselect stage 1 (Kleinjung collision search).
 *
 * The host code follows cado-nfs/polyselect exactly: same prime table, same
 * root lifting, same special-q enumeration, same match processing, same
 * output format. The GPU replaces the hash dispatch and collision scan
 * (see gps1_kernels.cu). Every GPU match is verified on the host with GMP
 * before it is used.
 *
 * Usage: gps1 -N <n> -d 5 -P 390000 -admin 5040 -admax 300000 -incr 60 -nq 1296
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <atomic>
#include <thread>
#include <future>

#include <gmp.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "cado.h" // IWYU pragma: keep
#include "mpz_poly.h"
#include "getprime.h"
#include "roots_mod.hpp"
#include "gmp_aux.h"
#include "cxx_mpz.hpp"
#include "gcd.h"
#include "size_optimization.hpp"
#include "auxiliary.hpp"
#include "polyselect_norms.hpp"
#include "polyselect_alpha.h"

#include "gps1.h"
#include "gps1_sopt.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

/* same as cado-nfs polyselect_special_q.c (64-bit list) */
static const unsigned int SPECIAL_Q[] = {
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
  179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
  233, 239, 241, 251, 257, 263, 269, 271, 0 };
#define LEN_SPECIAL_Q (sizeof(SPECIAL_Q)/sizeof(SPECIAL_Q[0]))

struct params_t {
    mpz_t N;
    unsigned long d, P, incr, nq;
    mpz_t admin, admax;
    unsigned long sopt_effort;
    unsigned long flush_at; /* pending matches that start a flush */
    int prep_ahead;     /* 1 (default): compute the roots and items of the
                           next ad on a second thread during the GPU search
                           of this one; 0: compute them before each ad */
    int sopt_gpu;       /* 1: size-optimize on the GPU, CPU redo on fail;
                           off by default, the kernel is slower than the
                           CPU path */
    int sopt_test;      /* 1: also run the CPU reference and compare */
};

struct header_t {
    cxx_mpz ad, Ntilde, m0;
    unsigned long d;
};

static void header_set_ad(header_t *h, mpz_srcptr N, unsigned long d, mpz_srcptr ad)
{
    h->d = d;
    mpz_set(h->ad, ad);
    mpz_set(h->Ntilde, ad);
    mpz_mul_ui(h->Ntilde, h->Ntilde, d);
    mpz_pow_ui(h->Ntilde, h->Ntilde, d - 1);
    mpz_mul_ui(h->Ntilde, h->Ntilde, d);
    mpz_mul(h->Ntilde, h->Ntilde, N);   /* d^d * ad^(d-1) * N */
    mpz_root(h->m0, h->Ntilde, d);
}

static int header_skip(const header_t *h, unsigned long p)
{
    return h->d % p == 0 || mpz_divisible_ui_p(h->ad, p);
}

/* primes in [P, 2P], rearranged as in cado-nfs polyselect_primes_table.c */
static std::vector<uint32_t> init_primes(unsigned long P)
{
    std::vector<uint32_t> v;
    prime_info pi;
    prime_info_init(pi);
    prime_info_seek(pi, P);
    for (unsigned long p; (p = getprime_mt(pi)) <= 2 * P; )
        v.push_back(p);
    prime_info_clear(pi);
    std::vector<uint32_t> w(v.size());
    for (size_t i = 0; i < v.size(); i++)
        w[i] = (i & 1) ? v[v.size() - 1 - i / 2] : v[i / 2];
    return w;
}

/* lift roots of N = x^d (mod p) to roots of N = (m0 + r)^d (mod p^2),
 * same as cado-nfs roots_lift */
static unsigned long lift_roots(uint64_t *r, mpz_srcptr N, unsigned long d,
                                mpz_srcptr m0, unsigned long p, unsigned long n)
{
    uint64_t pp = (uint64_t) p * p;
    mpz_t tmp, lambda;
    mpz_init(tmp);
    mpz_init(lambda);
    unsigned long i = 0;
    for (unsigned long j = 0; j < n; j++) {
        if (r[j] == 0)
            continue;
        mpz_ui_pow_ui(tmp, r[j], d - 1);
        mpz_mul_ui(lambda, tmp, r[j]);
        mpz_sub(lambda, N, lambda);
        mpz_divexact_ui(lambda, lambda, p);
        mpz_mul_ui(tmp, tmp, d);
        unsigned long inv = invert_ul(mpz_fdiv_ui(tmp, p), p);
        mpz_mul_ui(lambda, lambda, inv * p);
        mpz_add_ui(lambda, lambda, r[j]);
        mpz_sub(lambda, lambda, m0);
        r[i++] = mpz_fdiv_ui(lambda, pp);
    }
    mpz_clear(tmp);
    mpz_clear(lambda);
    return i;
}

/* special-q roots, ordered by decreasing root count as in
 * polyselect_qroots_rearrange (selection sort, same tie order) */
struct qroots_t { std::vector<uint32_t> q; std::vector<uint32_t> nr; std::vector<std::vector<uint64_t>> roots; };

static qroots_t comp_sq_roots(const header_t *h, gmp_randstate_ptr rstate)
{
    qroots_t R;
    uint64_t rq[10];
    for (unsigned long i = 1; SPECIAL_Q[i] != 0; i++) {
        unsigned long q = SPECIAL_Q[i];
        if (header_skip(h, q) || mpz_fdiv_ui(h->Ntilde, q) == 0)
            continue;
        unsigned long nrq = roots_mod_uint64(rq, mpz_fdiv_ui(h->Ntilde, q), h->d, q, rstate);
        nrq = lift_roots(rq, h->Ntilde, h->d, h->m0, q, nrq);
        if (nrq == 0)
            continue;
        R.q.push_back(q);
        R.nr.push_back(nrq);
        R.roots.emplace_back(rq, rq + nrq);
    }
    for (size_t i = 0; i < R.q.size(); i++) {
        size_t max = i;
        for (size_t j = i + 1; j < R.q.size(); j++)
            if (R.nr[j] > R.nr[max])
                max = j;
        std::swap(R.q[i], R.q[max]);
        std::swap(R.nr[i], R.nr[max]);
        std::swap(R.roots[i], R.roots[max]);
    }
    return R;
}

static void first_comb(unsigned long k, unsigned long *r)
{
    for (unsigned long i = 0; i < k; i++)
        r[i] = i;
}

static unsigned long next_comb(unsigned long n, unsigned long k, unsigned long *r)
{
    if (r[0] == n - k)
        return k;
    unsigned long j = k - 1;
    if (r[j] < n - 1) {
        r[j]++;
        return j;
    }
    while (r[j] - r[j - 1] == 1)
        j--;
    unsigned long ret = j - 1;
    unsigned long z = ++r[j - 1];
    while (j < k)
        r[j++] = ++z;
    return ret;
}

static unsigned long number_comb(const qroots_t &R, unsigned long k, unsigned long lq)
{
    unsigned long s = 0;
    std::vector<unsigned long> idx(k);
    first_comb(k, idx.data());
    while (1) {
        unsigned long p = 1;
        for (unsigned long j = 0; j < k; j++)
            p *= R.nr[idx[j]];
        s += p;
        if (next_comb(lq, k, idx.data()) == k)
            return s;
    }
}

/* same as cado-nfs find_suitable_lq */
static unsigned long find_suitable_lq(const params_t *pa, const header_t *h,
                                      const qroots_t &R, unsigned long *k)
{
    unsigned long prod = 1;
    double sq = 1.0;
    *k = 0;
    for (unsigned int i = 0; prod < pa->nq && i < R.q.size(); i++) {
        double p4 = pow(2.0 * pa->P, 4);
        if (!(p4 * (sq * R.q[i]) < mpz_get_d(h->m0)))
            break;
        prod *= h->d;
        sq *= (double) R.q[i];
        *k += 1;
    }
    if (*k > 8) *k = 8;
    if (*k < 1) *k = 1;
    unsigned long lq;
    for (lq = *k; number_comb(R, *k, lq) < pa->nq && lq < R.q.size(); lq++);
    return lq;
}

static void crt_sq(mpz_ptr qqz, mpz_ptr r, const unsigned long *q,
                   const unsigned long *rq, unsigned long lq)
{
    mpz_t prod, pprod, mod, inv, sum;
    mpz_init_set_ui(prod, 1);
    mpz_init(pprod);
    mpz_init(mod);
    mpz_init(inv);
    mpz_init_set_ui(sum, 0);
    for (unsigned long i = 0; i < lq; i++)
        mpz_mul_ui(prod, prod, q[i] * q[i]);
    for (unsigned long i = 0; i < lq; i++) {
        mpz_divexact_ui(pprod, prod, q[i] * q[i]);
        mpz_set_ui(mod, q[i] * q[i]);
        mpz_invert(inv, pprod, mod);
        mpz_mul_ui(inv, inv, rq[i]);
        mpz_mul(inv, inv, pprod);
        mpz_add(sum, sum, inv);
    }
    mpz_mod(sum, sum, prod);
    mpz_set(r, sum);
    mpz_set(qqz, prod);
    mpz_clear(prod);
    mpz_clear(pprod);
    mpz_clear(mod);
    mpz_clear(inv);
    mpz_clear(sum);
}

static unsigned int aux_nextcomb(unsigned int *ind, unsigned int len_q,
                                 const unsigned int *len_nr)
{
    for (unsigned int i = len_q - 1;; i--) {
        if (ind[i] < len_nr[i] - 1) {
            ind[i]++;
            return 1;
        }
        if (i == 0)
            return 0;
        ind[i] = 0;
    }
}

/* ------------------------------------------------ match processing (host) */

/* same text as cado-nfs snprintf_poly_info */
static void print_poly_pair(FILE *fp, mpz_srcptr n, mpz_poly_srcptr f,
                            mpz_poly_srcptr g, int raw)
{
    fprintf(fp, "# %s polynomial:\n", raw ? "Raw" : "Size-optimized");
    gmp_fprintf(fp, "%sn: %Zd\n", raw ? "# " : "", n);
    char *str;
    mpz_poly_asprintf_cado_format(&str, g, 'Y', raw ? "# " : "");
    fputs(str, fp);
    free(str);
    mpz_poly_asprintf_cado_format(&str, f, 'c', raw ? "# " : "");
    fputs(str, fp);
    free(str);
    double skew = L2_skewness(f);
    double logmu = L2_lognorm(f, skew);
    double exp_E = logmu + expected_rotation_gain(f, g);
    fprintf(fp, "# %sexp_E %1.2f, lognorm %1.2f, skew %1.2f, %u rroots\n",
            raw ? "raw " : "", exp_E, logmu, skew,
            mpz_poly_number_of_real_roots(f));
}

/* port of polyselect_process_match_async: verify the Kleinjung condition
 * with GMP, build the raw (f, g), apply the raw filters; returns 1 when
 * the candidate survives */
static int build_raw(FILE *out, const params_t *pa, const header_t *h,
                     const std::vector<uint32_t> &primes,
                     unsigned long p1, unsigned long p2, int64_t i,
                     uint64_t q, mpz_srcptr rq,
                     cxx_mpz_poly &f, cxx_mpz_poly &g)
{
    mpz_t l, mtilde, m, adm1, t, k, r;
    int ok = 0, inv;
    if (rq)
        gmp_fprintf(out, "# match ad=%Zd p1=%lu p2=%lu i=%lld q=%llu rq=%Zd\n",
                    (mpz_srcptr) h->ad, p1, p2, (long long) i,
                    (unsigned long long) q, rq);
    else
        gmp_fprintf(out, "# match ad=%Zd p1=%lu p2=%lu i=%lld q=1 rq=0\n",
                    (mpz_srcptr) h->ad, p1, p2, (long long) i);
    mpz_init(l);
    mpz_init(m);
    mpz_init(t);
    mpz_init(k);
    mpz_init(r);
    mpz_init(adm1);
    mpz_init(mtilde);

    mpz_set_ui(l, p1);
    mpz_mul_ui(l, l, p2);
    mpz_mul_ui(l, l, q);
    mpz_set_si(mtilde, i);
    if (rq) {
        mpz_mul_ui(mtilde, mtilde, q);
        mpz_mul_ui(mtilde, mtilde, q);
        mpz_add(mtilde, mtilde, rq);
    }
    mpz_add(mtilde, mtilde, h->m0);

    /* GMP verification of every GPU match: Ntilde = mtilde^d mod p1^2, p2^2, q^2 */
    mpz_pow_ui(t, mtilde, h->d);
    mpz_sub(t, h->Ntilde, t);
    assert(mpz_divisible_ui_p(t, p1 * p1));
    assert(mpz_divisible_ui_p(t, p2 * p2));
    mpz_set_uint64(r, q);
    mpz_mul(r, r, r);
    assert(mpz_divisible_p(t, r));

    unsigned long qmax = 1;
    for (unsigned long j = 0; j < LEN_SPECIAL_Q - 1; j++)
        if (q % SPECIAL_Q[j] == 0)
            qmax = SPECIAL_Q[j];

    mpz_pow_ui(t, mtilde, h->d);
    mpz_sub(t, h->Ntilde, t);
    mpz_divexact(t, t, l);
    mpz_divexact(t, t, l);
    prime_info pi;
    prime_info_init(pi);
    for (unsigned long p = 2; p <= primes.back(); p = getprime_mt(pi)) {
        if (p <= qmax || header_skip(h, p))
            continue;
        while (mpz_divisible_ui_p(t, p * p)) {
            mpz_mul_ui(l, l, p);
            mpz_divexact_ui(t, t, p * p);
        }
    }
    prime_info_clear(pi);

    mpz_mul_ui(m, h->ad, h->d);
    inv = mpz_invert(adm1, l, m);
    assert(inv);
    mpz_mul(adm1, adm1, mtilde);
    mpz_mod(adm1, adm1, m);
    mpz_mul_2exp(t, adm1, 1);
    if (mpz_cmp(t, m) >= 0)
        mpz_sub(adm1, adm1, m);

    mpz_mul(m, adm1, l);
    mpz_sub(m, mtilde, m);
    mpz_divexact_ui(m, m, h->d);
    mpz_divexact(m, m, h->ad);
    mpz_set(mpz_poly_coeff(g, 1), l);
    mpz_neg(mpz_poly_coeff(g, 0), m);
    mpz_set(mpz_poly_coeff(f, h->d), h->ad);
    mpz_pow_ui(t, m, h->d);
    mpz_mul(t, t, h->ad);
    mpz_sub(t, pa->N, t);
    mpz_set(mpz_poly_coeff(f, h->d - 1), adm1);
    mpz_divexact(t, t, l);
    mpz_pow_ui(mtilde, m, h->d - 1);
    mpz_mul(mtilde, mtilde, adm1);
    mpz_sub(t, t, mtilde);
    for (int j = h->d - 2; j > 0; j--) {
        mpz_divexact(t, t, l);
        mpz_pow_ui(mtilde, m, j);
        mpz_fdiv_q(adm1, t, mtilde);
        mpz_invert(k, mtilde, l);
        mpz_mul(k, k, t);
        mpz_sub(k, k, adm1);
        mpz_mod(k, k, l);
        mpz_mul_2exp(k, k, 1);
        int cmp = mpz_cmp(k, l);
        mpz_div_2exp(k, k, 1);
        if (cmp >= 0)
            mpz_sub(k, k, l);
        mpz_add(adm1, adm1, k);
        mpz_set(mpz_poly_coeff(f, j), adm1);
        mpz_submul(t, mtilde, adm1);
    }
    mpz_divexact(t, t, l);
    mpz_set(mpz_poly_coeff(f, 0), t);

    if (mpz_sgn(mpz_poly_coeff_const(f, h->d)) * mpz_sgn(mpz_poly_coeff_const(f, h->d - 2)) > 0)
        goto end;

    mpz_poly_cleandeg(f, h->d);
    assert(mpz_poly_degree(f) == (int) h->d);
    mpz_poly_cleandeg(g, 1);
    assert(mpz_poly_degree(g) == 1);

    if (!mpz_poly_has_trivial_content(f))
        goto end;

    ok = 1;

end:
    mpz_clear(l);
    mpz_clear(m);
    mpz_clear(t);
    mpz_clear(k);
    mpz_clear(r);
    mpz_clear(adm1);
    mpz_clear(mtilde);
    return ok;
}

/* apply the optimized-coefficient filter and print the raw and the
 * optimized pair; returns 1 when the pair is printed */
static unsigned long finish_match(FILE *out, const params_t *pa,
                                  mpz_poly_srcptr f_raw, mpz_poly_srcptr g_raw,
                                  mpz_poly_srcptr f, mpz_poly_srcptr g)
{
    if (mpz_sgn(mpz_poly_coeff_const(f, f->deg - 1)) * mpz_sgn(mpz_poly_coeff_const(f, f->deg - 3)) > 0)
        return 0;
    print_poly_pair(out, pa->N, f_raw, g_raw, 1);
    fputs("#\n", out);
    print_poly_pair(out, pa->N, f, g, 0);
    fputs("\n", out);
    return 1;
}

/* ------------------------------------------- GPU size optimization glue */

/* returns 0 when a does not fit the fixed width */
static int mpz_to_soptz(sopt_z *r, mpz_srcptr a)
{
    if ((mpz_sizeinbase(a, 2) + 31) / 32 > SOPT_GPU_LIMBS)
        return 0;
    r->sign = mpz_sgn(a);
    r->n = 0;
    if (r->sign != 0) {
        size_t cnt;
        mpz_export(r->l, &cnt, -1, 4, 0, 0, a);
        r->n = cnt;
    }
    return 1;
}

static void soptz_to_mpz(mpz_ptr a, const sopt_z *r)
{
    mpz_import(a, r->n, -1, 4, 0, 0, r->l);
    if (r->sign < 0)
        mpz_neg(a, a);
}

/* returns 0 when a coefficient does not fit; skew is the same seed the
 * CPU reference computes in sopt_get_skewness */
static int pair_from_polys(sopt_pair_t *P, mpz_poly_srcptr f, mpz_poly_srcptr g)
{
    if (f->deg > SOPT_GPU_MAXDEG)
        return 0;
    P->deg = f->deg;
    P->fail = 0;
    for (int i = 0; i <= f->deg; i++)
        if (!mpz_to_soptz(&P->f[i], mpz_poly_coeff_const(f, i)))
            return 0;
    for (int i = 0; i <= 1; i++)
        if (!mpz_to_soptz(&P->g[i], mpz_poly_coeff_const(g, i)))
            return 0;
    P->skew = pow(fabs(mpz_get_d(mpz_poly_coeff_const(g, 0)) /
                       mpz_get_d(mpz_poly_coeff_const(f, f->deg))),
                  1.0 / (double) f->deg);
    return 1;
}

static void pair_to_polys(mpz_poly_ptr f, mpz_poly_ptr g, const sopt_pair_t *P)
{
    for (int i = 0; i <= P->deg; i++)
        soptz_to_mpz(mpz_poly_coeff(f, i), &P->f[i]);
    for (int i = 0; i <= 1; i++)
        soptz_to_mpz(mpz_poly_coeff(g, i), &P->g[i]);
    mpz_poly_cleandeg(f, P->deg);
    mpz_poly_cleandeg(g, 1);
}

static unsigned long sopt_test_eq, sopt_test_close, sopt_test_worse,
                     sopt_test_fallback;

/* -sopt-test: compare one GPU result against the CPU reference. Equal
 * coefficients pass. Otherwise the GPU pair passes when its lognorm is
 * not worse than the CPU lognorm plus 1e-6 (device log() rounds
 * differently from glibc, so a near-tie can pick another translation). */
static void sopt_test_compare(mpz_poly_srcptr f_cpu, mpz_poly_srcptr g_cpu,
                              mpz_poly_srcptr f_gpu, mpz_poly_srcptr g_gpu)
{
    if (mpz_poly_cmp(f_cpu, f_gpu) == 0 && mpz_poly_cmp(g_cpu, g_gpu) == 0) {
        #pragma omp atomic
        sopt_test_eq++;
        return;
    }
    double n_cpu = L2_lognorm(f_cpu, L2_skewness(f_cpu));
    double n_gpu = L2_lognorm(f_gpu, L2_skewness(f_gpu));
    if (n_gpu <= n_cpu + 1e-6) {
        #pragma omp atomic
        sopt_test_close++;
        return;
    }
    #pragma omp atomic
    sopt_test_worse++;
    gmp_fprintf(stderr, "# gps1: sopt-test WORSE gpu %.6f cpu %.6f f_cpu %Zd...\n",
                n_gpu, n_cpu, mpz_poly_coeff_const(f_cpu, 0));
}

/* ----------------------------------------------------------------- driver */

static unsigned long n_matches;
static double t_gpu;
/* the prep and flush threads update these while the main thread reads them */
static std::atomic<double> t_roots(0), t_crt(0), t_match(0), t_sopt(0);

/* the (q, rq) work items of one leading coefficient; item 0 is q = 1 */
struct items_t { std::vector<uint64_t> q; std::vector<cxx_mpz> rq; };

/* a GPU match waiting for host processing */
struct hit_t { cxx_mpz ad; uint32_t p1, p2; int64_t i; uint64_t q; cxx_mpz rq; };

/* a raw candidate between the two host passes */
struct cand_t { int ok = 0, pair = -1; cxx_mpz_poly f, g; std::string head; };

/* process pending matches: build the raw candidates on all cores, run one
 * GPU size-optimization batch, redo GPU failures on the CPU; returns the
 * polynomial count */
static unsigned long flush_hits(const params_t *pa,
                                const std::vector<uint32_t> &primes,
                                std::vector<hit_t> &hits)
{
    double tm = omp_get_wtime();
    std::vector<cand_t> cands(hits.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t mi = 0; mi < hits.size(); mi++) {
        const hit_t &ht = hits[mi];
        cand_t &c = cands[mi];
        header_t h;
        header_set_ad(&h, pa->N, pa->d, ht.ad);
        char *buf; size_t len;
        FILE *out = open_memstream(&buf, &len);
        c.ok = build_raw(out, pa, &h, primes, ht.p1, ht.p2, ht.i, ht.q,
                         ht.q == 1 ? NULL : (mpz_srcptr) ht.rq, c.f, c.g);
        fclose(out);
        c.head.assign(buf, len);
        free(buf);
    }

    /* one GPU batch over the surviving candidates */
    std::vector<sopt_pair_t> pairs;
    if (pa->sopt_gpu)
        for (cand_t &c : cands) {
            sopt_pair_t P;
            if (c.ok && pair_from_polys(&P, c.f, c.g)) {
                c.pair = pairs.size();
                pairs.push_back(P);
            }
        }
    double tg = omp_get_wtime();
    gps1_sopt_run(pairs.data(), pairs.size(), pa->sopt_effort);
    t_sopt.fetch_add(omp_get_wtime() - tg);

    unsigned long printed = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:printed)
    for (size_t mi = 0; mi < cands.size(); mi++) {
        cand_t &c = cands[mi];
        char *buf; size_t len;
        FILE *out = open_memstream(&buf, &len);
        fwrite(c.head.data(), 1, c.head.size(), out);
        if (c.ok) {
            cxx_mpz_poly f, g;
            if (c.pair >= 0 && !pairs[c.pair].fail) {
                pair_to_polys(f, g, &pairs[c.pair]);
                if (pa->sopt_test) {
                    cxx_mpz_poly fc, gc;
                    size_optimization(fc, gc, c.f, c.g, pa->sopt_effort, 0);
                    sopt_test_compare(fc, gc, f, g);
                }
            } else {
                size_optimization(f, g, c.f, c.g, pa->sopt_effort, 0);
                #pragma omp atomic
                sopt_test_fallback++;
            }
            printed += finish_match(out, pa, c.f, c.g, f, g);
        }
        fclose(out);
        #pragma omp critical
        fwrite(buf, 1, len, stdout);
        free(buf);
    }
    printf("# gps1: flush %zu hits, %lu polys, %zu gpu-sopt, %lu cpu-redo, "
           "%.2fs, %d threads\n",
           hits.size(), printed, pairs.size(), sopt_test_fallback,
           omp_get_wtime() - tm, omp_get_max_threads());
    fflush(stdout);
    hits.clear();
    t_match.fetch_add(omp_get_wtime() - tm);
    return printed;
}

static std::atomic<unsigned long> n_polys(0);
static std::thread flush_thread;

/* run flush_hits on its own thread, so the raw search of the next ad
 * values continues while this batch is size-optimized and printed */
static void start_flush(const params_t *pa,
                        const std::vector<uint32_t> &primes,
                        std::vector<hit_t> &hits)
{
    if (flush_thread.joinable())
        flush_thread.join();
    flush_thread = std::thread([pa, &primes, batch = std::move(hits)]() mutable {
        n_polys += flush_hits(pa, primes, batch);
    });
    hits.clear();
}

/* the host input of one leading coefficient: its header, the roots
 * mod p^2 of all P primes in the flat GPU layout with their sort-entry
 * offsets, and the (q, rq) items: q = 1 (collision_on_p), then the
 * special-q CRT residues (collision_on_sq) */
struct prep_t {
    header_t h;
    std::vector<uint32_t> root_prime, entry_off;
    std::vector<uint64_t> roots;
    items_t items;
};

static prep_t prepare(const params_t *pa, const std::vector<uint32_t> &primes,
                      int64_t M, gmp_randstate_ptr rstate, cxx_mpz ad)
{
    prep_t P;
    header_set_ad(&P.h, pa->N, pa->d, ad);

    double tr = omp_get_wtime();
    {
        std::vector<std::vector<uint64_t>> tmp(primes.size());
        #pragma omp parallel
        {
            gmp_randstate_t rs;
            gmp_randinit_default(rs);
            uint64_t rp[10];
            #pragma omp for schedule(dynamic, 64)
            for (size_t i = 0; i < primes.size(); i++) {
                unsigned long p = primes[i];
                if (header_skip(&P.h, p))
                    continue;
                unsigned long n = roots_mod_uint64(rp, mpz_fdiv_ui(P.h.Ntilde, p), pa->d, p, rs);
                n = lift_roots(rp, P.h.Ntilde, pa->d, P.h.m0, p, n);
                tmp[i].assign(rp, rp + n);
            }
            gmp_randclear(rs);
        }
        for (size_t i = 0; i < primes.size(); i++)
            for (uint64_t rt : tmp[i]) {
                P.root_prime.push_back(i);
                P.roots.push_back(rt);
            }
    }
    P.entry_off.resize(P.root_prime.size() + 1);
    for (size_t c = 0; c < P.root_prime.size(); c++) {
        uint64_t pp = (uint64_t) primes[P.root_prime[c]] * primes[P.root_prime[c]];
        P.entry_off[c + 1] = P.entry_off[c] + (2 * (uint64_t) M - 1) / pp + 1;
    }
    t_roots.fetch_add(omp_get_wtime() - tr);

    P.items.q.push_back(1);
    P.items.rq.emplace_back(0);
    double tc = omp_get_wtime();
    qroots_t SQ = comp_sq_roots(&P.h, rstate);
    unsigned long k;
    unsigned long lq = find_suitable_lq(pa, &P.h, SQ, &k);
    std::vector<unsigned long> idx_q(lq);
    first_comb(k, idx_q.data());
    unsigned long curr_nq = 0;
    while (curr_nq < pa->nq) {
        uint64_t q = 1;
        for (unsigned long j = 0; j < k; j++)
            q *= SQ.q[idx_q[j]];

        std::vector<unsigned int> ind(k, 0), len(k);
        for (unsigned long j = 0; j < k; j++)
            len[j] = SQ.nr[idx_q[j]];
        cxx_mpz qqz;
        unsigned long ql[8], rl[8];
        for (int re = 1; re && curr_nq < pa->nq; ) {
            for (unsigned long j = 0; j < k; j++) {
                ql[j] = SQ.q[idx_q[j]];
                rl[j] = SQ.roots[idx_q[j]][ind[j]];
            }
            P.items.q.push_back(q);
            P.items.rq.emplace_back();
            crt_sq(qqz, P.items.rq.back(), ql, rl, k);
            re = aux_nextcomb(ind.data(), k, len.data());
            curr_nq++;
        }
        if (next_comb(lq, k, idx_q.data()) == k)
            break;
    }
    t_crt.fetch_add(omp_get_wtime() - tc);
    return P;
}

/* run one GPU collision pass for items [first, first + batch) and queue the
 * matches for flush_hits */
static void run_batch(const header_t *h, gpu_t *G, uint32_t nroots,
                      const items_t &items, uint32_t first,
                      uint32_t batch, int64_t M, uint64_t slots,
                      std::vector<hit_t> &hits)
{
    if (nroots == 0)
        return;
    std::vector<uint64_t> qs, rq_hi(batch), rq_lo(batch);
    std::vector<uint32_t> qidx(batch);
    for (uint32_t b = 0; b < batch; b++) {
        uint64_t q = items.q[first + b];
        if (qs.empty() || qs.back() != q)
            qs.push_back(q);
        qidx[b] = qs.size() - 1;
        mpz_srcptr rq = items.rq[first + b];
        unsigned __int128 v = 0;
        assert(mpz_size(rq) <= 2);
        for (size_t li = mpz_size(rq); li--; )
            v = (v << 64) | mpz_getlimbn(rq, li);
        rq_hi[b] = (uint64_t)(v >> 64);
        rq_lo[b] = (uint64_t) v;
    }
    CUDA_CHECK(cudaMemcpy(G->qs, qs.data(), qs.size() * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G->qidx, qidx.data(), batch * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G->rq_hi, rq_hi.data(), batch * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G->rq_lo, rq_lo.data(), batch * 8, cudaMemcpyHostToDevice));

    double tg = omp_get_wtime();
    uint64_t nmatches = gps1_run(G, nroots, qs.size(), batch, M, slots);
    t_gpu += omp_get_wtime() - tg;
    n_matches += nmatches;
    std::vector<match_t> matches(nmatches);
    if (nmatches)
        CUDA_CHECK(cudaMemcpy(matches.data(), G->matches, nmatches * sizeof(match_t),
                              cudaMemcpyDeviceToHost));

    for (const match_t &mt : matches) {
        hit_t ht;
        mpz_set(ht.ad, h->ad);
        ht.p1 = mt.p1; ht.p2 = mt.p2; ht.i = mt.i;
        ht.q = items.q[first + mt.b];
        mpz_set(ht.rq, items.rq[first + mt.b]);
        hits.push_back(std::move(ht));
    }
}

int main(int argc, char **argv)
{
    params_t pa;
    mpz_init(pa.N);
    mpz_init_set_ui(pa.admin, 0);
    mpz_init_set_ui(pa.admax, 0);
    pa.d = 0; pa.P = 0; pa.incr = 60; pa.nq = 1000; pa.sopt_effort = 0;
    pa.flush_at = 0; pa.prep_ahead = 1; pa.sopt_gpu = 0; pa.sopt_test = 0;
    for (int a = 1; a + 1 < argc; a += 2) {
        std::string s = argv[a];
        if (s == "-N") mpz_set_str(pa.N, argv[a + 1], 10);
        else if (s == "-d") pa.d = atol(argv[a + 1]);
        else if (s == "-P") pa.P = atol(argv[a + 1]);
        else if (s == "-admin") mpz_set_str(pa.admin, argv[a + 1], 10);
        else if (s == "-admax") mpz_set_str(pa.admax, argv[a + 1], 10);
        else if (s == "-incr") pa.incr = atol(argv[a + 1]);
        else if (s == "-nq") pa.nq = atol(argv[a + 1]);
        else if (s == "-sopteffort") pa.sopt_effort = atol(argv[a + 1]);
        else if (s == "-sopt-gpu") pa.sopt_gpu = atoi(argv[a + 1]);
        else if (s == "-flush-at") pa.flush_at = atol(argv[a + 1]);
        else if (s == "-prep-ahead") pa.prep_ahead = atoi(argv[a + 1]);
        else if (s == "-sopt-test") pa.sopt_test = atoi(argv[a + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
    }
    assert(mpz_sgn(pa.N) > 0 && pa.d > 0 && pa.P > 0 && mpz_sgn(pa.admax) > 0);
    /* the sort key packs the batch index above the u + M offset field,
     * which holds [0, 2M) with M = 4P^2; P <= 2^30 keeps M in the int64
     * range and leaves at least one key bit for the batch index */
    assert(pa.P <= (1ul << 30) && "P > 2^30 does not fit the 64-bit sort key");
    /* with GPU sopt a batch of this size keeps the item kernel full,
     * and the flush thread hides it behind the next raw searches */
    if (pa.flush_at == 0)
        pa.flush_at = pa.sopt_gpu ? 8192 : 512;

    std::vector<uint32_t> primes = init_primes(pa.P);
    int64_t M = (int64_t)(2 * pa.P) * (int64_t)(2 * pa.P);
    uint32_t offset_bits = gps1_offset_bits(M);

    /* smallest prime bounds the batch size: every batch item costs
     * entry_off[nroots] <= sum over roots of (2M/p^2 + 1) sort entries */
    uint64_t max_slots = 0;
    for (uint32_t p : primes)
        max_slots += pa.d * ((2 * (uint64_t) M - 1) / ((uint64_t) p * p) + 1);
    uint32_t max_batch = std::min<uint64_t>(GPS1_MAX_ENTRIES / max_slots,
                                            ((uint64_t) 1 << (64 - offset_bits)) - 1);
    assert(max_batch >= 1 && "one batch item exceeds GPS1_MAX_ENTRIES; lower P or d");

    gpu_t G;
    G.nprimes = primes.size();
    CUDA_CHECK(cudaMalloc(&G.primes, primes.size() * 4));
    CUDA_CHECK(cudaMalloc(&G.root_prime, primes.size() * pa.d * 4));
    CUDA_CHECK(cudaMalloc(&G.roots, primes.size() * pa.d * 8));
    CUDA_CHECK(cudaMalloc(&G.entry_off, (primes.size() * pa.d + 1) * 4));
    CUDA_CHECK(cudaMalloc(&G.qs, (uint64_t) max_batch * 8));
    CUDA_CHECK(cudaMalloc(&G.qidx, (uint64_t) max_batch * 4));
    CUDA_CHECK(cudaMalloc(&G.rq_hi, (uint64_t) max_batch * 8));
    CUDA_CHECK(cudaMalloc(&G.rq_lo, (uint64_t) max_batch * 8));
    CUDA_CHECK(cudaMalloc(&G.invqq, (uint64_t) max_batch * primes.size() * 8));
    CUDA_CHECK(cudaMalloc(&G.keys, (uint64_t) GPS1_MAX_ENTRIES * 8));
    CUDA_CHECK(cudaMalloc(&G.keys_alt, (uint64_t) GPS1_MAX_ENTRIES * 8));
    CUDA_CHECK(cudaMalloc(&G.vals, (uint64_t) GPS1_MAX_ENTRIES * 4));
    CUDA_CHECK(cudaMalloc(&G.vals_alt, (uint64_t) GPS1_MAX_ENTRIES * 4));
    G.sort_tmp_bytes = gps1_sort_tmp_bytes(GPS1_MAX_ENTRIES);
    CUDA_CHECK(cudaMalloc(&G.sort_tmp, G.sort_tmp_bytes));
    CUDA_CHECK(cudaMalloc(&G.matches, GPS1_MAX_MATCHES * sizeof(match_t)));
    CUDA_CHECK(cudaMemcpy(G.primes, primes.data(), primes.size() * 4, cudaMemcpyHostToDevice));

    gmp_randstate_t rstate;
    gmp_randinit_default(rstate);

    std::vector<hit_t> hits;
    double t0 = omp_get_wtime();

    cxx_mpz ad = pa.admin;
    /* ad = 0 is degenerate (Ntilde = m0 = 0); start from incr and round
     * up to a multiple of incr, same as cado-nfs polyselect */
    if (mpz_sgn(ad) == 0)
        mpz_set_ui(ad, pa.incr);
    mpz_add_ui(ad, ad, (pa.incr - mpz_fdiv_ui(ad, pa.incr)) % pa.incr);
    /* the host input of an ad is prepared while the GPU searches the
     * previous one; with -prep-ahead 0 it is prepared at the get */
    auto prepare_ad = [&](const cxx_mpz &a) {
        auto policy = pa.prep_ahead ? std::launch::async : std::launch::deferred;
        return std::async(policy, prepare, &pa, std::cref(primes), M,
                          (gmp_randstate_ptr) rstate, a);
    };
    std::future<prep_t> next = prepare_ad(ad);
    for (; mpz_cmp(ad, pa.admax) < 0; mpz_add_ui(ad, ad, pa.incr)) {
        prep_t P = next.get();
        cxx_mpz ad1;
        mpz_add_ui(ad1, ad, pa.incr);
        if (mpz_cmp(ad1, pa.admax) < 0)
            next = prepare_ad(ad1);

        CUDA_CHECK(cudaMemcpy(G.root_prime, P.root_prime.data(), P.root_prime.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.roots, P.roots.data(), P.roots.size() * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.entry_off, P.entry_off.data(), P.entry_off.size() * 4, cudaMemcpyHostToDevice));
        uint64_t slots = P.entry_off.back();
        for (uint32_t first = 0; first < P.items.q.size(); first += max_batch) {
            uint32_t batch = std::min<uint64_t>(max_batch, P.items.q.size() - first);
            run_batch(&P.h, &G, P.root_prime.size(), P.items, first, batch, M, slots, hits);
        }
        if (hits.size() >= pa.flush_at)
            start_flush(&pa, primes, hits);
        gmp_printf("# gps1: ad=%Zd done, %lu matches, %lu polynomials, %.2fs "
                   "(roots %.2f gpu %.2f crt %.2f match %.2f)\n",
                   (mpz_srcptr) ad, n_matches, n_polys.load(), omp_get_wtime() - t0,
                   t_roots.load(), t_gpu, t_crt.load(), t_match.load());
        fflush(stdout);
    }
    start_flush(&pa, primes, hits);
    flush_thread.join();
    printf("# gps1: total %lu matches, %lu polynomials in %.2fs (sopt-gpu %.2f)\n",
           n_matches, n_polys.load(), omp_get_wtime() - t0, t_sopt.load());
    if (pa.sopt_test) {
        printf("# gps1: sopt-test %lu equal, %lu close, %lu worse, %lu cpu-redo\n",
               sopt_test_eq, sopt_test_close, sopt_test_worse,
               sopt_test_fallback);
        if (sopt_test_worse)
            return 1;
    }
    return 0;
}
