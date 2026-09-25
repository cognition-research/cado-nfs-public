/* gps1_sopt_test: run the GPU size-optimization algorithm on the host CPU
 * (gps1_sopt_run_host) and the CADO reference (size_optimization) on the
 * same deterministic random candidates, then compare the results.
 *
 * A result passes when the coefficients are equal, or when the GPU pair
 * lognorm is not worse than the CPU pair lognorm plus 1e-6 (device and
 * host log() round differently, so a near-tie can pick another
 * translation). A candidate the GPU path rejects counts as a fallback,
 * not as a failure: gps1 redoes such a candidate on the CPU.
 *
 * Usage: gps1_sopt_test [count] [seed]
 */
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <gmp.h>

#include "cado.h" // IWYU pragma: keep
#include "mpz_poly.h"
#include "cxx_mpz.hpp"
#include "size_optimization.hpp"
#include "auxiliary.hpp"
#include "polyselect_norms.hpp"

#include "gps1_sopt.h"

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

/* one raw candidate with the coefficient scales of a c155-class number:
 * g = l x - m with l ~ 2^45, m ~ 2^90, ad small, f coefficients up to
 * ~2^90 */
static void random_pair(cxx_mpz_poly &f, cxx_mpz_poly &g, gmp_randstate_t rs,
                        int d)
{
    mpz_urandomb(mpz_poly_coeff(g, 1), rs, 45);
    mpz_add_ui(mpz_poly_coeff(g, 1), mpz_poly_coeff(g, 1), 1);
    mpz_urandomb(mpz_poly_coeff(g, 0), rs, 90);
    mpz_neg(mpz_poly_coeff(g, 0), mpz_poly_coeff(g, 0));
    mpz_poly_cleandeg(g, 1);

    mpz_urandomb(mpz_poly_coeff(f, d), rs, 20);
    mpz_add_ui(mpz_poly_coeff(f, d), mpz_poly_coeff(f, d), 60);
    for (int i = 0; i < d; i++) {
        mpz_urandomb(mpz_poly_coeff(f, i), rs, 60 + 5 * i);
        if (gmp_urandomb_ui(rs, 1))
            mpz_neg(mpz_poly_coeff(f, i), mpz_poly_coeff(f, i));
    }
    mpz_poly_cleandeg(f, d);
}

int main(int argc, char **argv)
{
    int count = argc > 1 ? atoi(argv[1]) : 200;
    unsigned long seed = argc > 2 ? strtoul(argv[2], NULL, 10) : 42;
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, seed);

    int eq = 0, close = 0, worse = 0, fallback = 0;
    for (int t = 0; t < count; t++) {
        int d = 5 + t % 2;
        cxx_mpz_poly f, g;
        random_pair(f, g, rs, d);

        sopt_pair_t P;
        P.deg = d;
        P.fail = 0;
        int fits = 1;
        for (int i = 0; i <= d; i++)
            fits &= mpz_to_soptz(&P.f[i], mpz_poly_coeff_const(f, i));
        for (int i = 0; i <= 1; i++)
            fits &= mpz_to_soptz(&P.g[i], mpz_poly_coeff_const(g, i));
        assert(fits);
        P.skew = pow(fabs(mpz_get_d(mpz_poly_coeff_const(g, 0)) /
                          mpz_get_d(mpz_poly_coeff_const(f, d))),
                     1.0 / (double) d);

        gps1_sopt_run_host(&P, 1, 0);
        if (P.fail) {
            fallback++;
            continue;
        }

        cxx_mpz_poly fg, gg;
        for (int i = 0; i <= d; i++)
            soptz_to_mpz(mpz_poly_coeff(fg, i), &P.f[i]);
        for (int i = 0; i <= 1; i++)
            soptz_to_mpz(mpz_poly_coeff(gg, i), &P.g[i]);
        mpz_poly_cleandeg(fg, d);
        mpz_poly_cleandeg(gg, 1);

        cxx_mpz_poly fc, gc;
        size_optimization(fc, gc, f, g, 0, 0);

        if (mpz_poly_cmp(fc, fg) == 0 && mpz_poly_cmp(gc, gg) == 0) {
            eq++;
            continue;
        }
        double n_cpu = L2_lognorm(fc, L2_skewness(fc));
        double n_gpu = L2_lognorm(fg, L2_skewness(fg));
        if (n_gpu <= n_cpu + 1e-6) {
            close++;
            continue;
        }
        worse++;
        gmp_printf("WORSE t=%d gpu %.6f cpu %.6f\nf_raw %Zd %Zd %Zd %Zd %Zd %Zd %Zd\n",
                   t, n_gpu, n_cpu,
                   mpz_poly_coeff_const(f, 0), mpz_poly_coeff_const(f, 1),
                   mpz_poly_coeff_const(f, 2), mpz_poly_coeff_const(f, 3),
                   mpz_poly_coeff_const(f, 4), mpz_poly_coeff_const(f, 5),
                   d == 6 ? mpz_poly_coeff_const(f, 6) : mpz_poly_coeff_const(f, 0));
    }
    gmp_randclear(rs);

    printf("gps1_sopt_test: %d candidates, %d equal, %d close, %d worse, "
           "%d fallback\n", count, eq, close, worse, fallback);
    return worse != 0;
}
