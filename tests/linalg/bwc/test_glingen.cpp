#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <algorithm>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "gmp_aux.h"
#include "lingen.hpp"
#include "lingen_bmstatus.hpp"
#include "lingen_ft_plugin.hpp"
#include "lingen_gpu.hpp"
#include "lingen_matpoly_select.hpp"
#include "lingen_qcode_binary.hpp"
#include "macros.h"
#include "params.hpp"
#include "select_mpi.h"

/* The operations of dmatpoly (lingen_gpu.hpp) against those of
 * matpoly<true>, on the plugin named by lingen_ft_impl. */

/* columns 37 z mod n for z < zeros are zero, so that the base case
 * sees lucky columns */
static matpoly<true> random_matpoly(matpoly<true>::arith_hard * ab,
        unsigned int m, unsigned int n, size_t L, unsigned int zeros,
        cxx_gmp_randstate & rstate)
{
    matpoly<true> E(ab, m, n, L);
    E.zero_pad(L);
    E.fill_random(0, L, rstate);
    for (unsigned int z = 0; z < zeros; z++)
        for (unsigned int i = 0; i < m; i++)
            std::fill_n(E.part(i, (37 * z) % n), E.capacity() / ULONG_BITS, 0UL);
    return E;
}

static matpoly<true> copy(matpoly<true> const & a)
{
    matpoly<true> c;
    c.truncate(a, a.get_size());
    return c;
}

static void expect_same(const char * what, matpoly<true> const & a, dmatpoly const & b)
{
    matpoly<true> const h = b.download(a.ab);
    bool const ok = a.get_size() == h.get_size() && a.cmp(h) == 0;
    fprintf(stderr, "%s: %s (size %zu vs %zu)\n", what, ok ? "ok" : "MISMATCH",
            a.get_size(), h.get_size());
    ASSERT_ALWAYS(ok);
}

static bmstatus<true> fresh_bm(unsigned int m, unsigned int n, size_t L)
{
    cxx_mpz const p = 2;
    bmstatus<true> bm(m, n, p);
    bm.set_t0(iceildiv(m, n));
    bm.hints = glingen_hints(bm.d, L, 2 * L);
    return bm;
}

static void expect_same(bmstatus<true> const & a, bmstatus<true> const & b)
{
    ASSERT_ALWAYS(a.t == b.t);
    ASSERT_ALWAYS(a.done == b.done);
    ASSERT_ALWAYS(a.delta == b.delta);
    ASSERT_ALWAYS(a.lucky == b.lucky);
}

static void test_shifts(matpoly<true> const & E)
{
    size_t const L = E.get_size();
    size_t const half = L - L / 2;
    for (size_t k : { (size_t) 0, (size_t) 1, half / 3, half }) {
        matpoly<true> h = copy(E);
        dmatpoly d(E);
        matpoly<true> const hlow = h.truncate_and_rshift(half, k);
        dmatpoly const dlow = d.truncate_and_rshift(half, k);
        expect_same("truncate", hlow, dlow);
        expect_same("rshift", h, d);
        ASSERT_ALWAYS(h.coeff_is_zero(0) == d.coeff_is_zero(0));
        ASSERT_ALWAYS(h.coeff_is_zero(h.get_size() - 1) == d.coeff_is_zero(d.get_size() - 1));
    }
}

static void test_products(matpoly<true> const & E, matpoly<true> const & pi)
{
    dmatpoly const dE(E), dpi(pi);
    if (E.get_size() >= pi.get_size())
        expect_same("mp", matpoly<true>::mp(E, pi), dmatpoly::mp(dE, dpi));
    expect_same("mul", matpoly<true>::mul(pi, pi), dmatpoly::mul(dpi, dpi));
}

static matpoly<true> test_basecase(matpoly<true> const & E, unsigned int m, unsigned int n)
{
    bmstatus<true> hbm = fresh_bm(m, n, E.get_size());
    bmstatus<true> dbm = fresh_bm(m, n, E.get_size());
    matpoly<true> hE = copy(E);
    dmatpoly dE(E);
    matpoly<true> hpi = bw_lingen_basecase(hbm, hE);
    dmatpoly const dpi = glingen_basecase(dbm, dE);
    expect_same(hbm, dbm);
    expect_same("basecase", hpi, dpi);
    return hpi;
}

static void test_recursion(matpoly<true> const & E, unsigned int m, unsigned int n,
        unsigned int threshold)
{
    bmstatus<true> hbm = fresh_bm(m, n, E.get_size());
    bmstatus<true> dbm = fresh_bm(m, n, E.get_size());
    hbm.hints = dbm.hints = glingen_hints(hbm.d, E.get_size(), threshold);
    matpoly<true> hE = copy(E);
    matpoly<true> dE = copy(E);
    matpoly<true> const hpi = bw_lingen_single(hbm, hE);
    matpoly<true> const dpi = bw_glingen(dbm, dE);
    bool const ok = hpi.cmp(dpi) == 0;
    fprintf(stderr, "recursion: %s\n", ok ? "ok" : "MISMATCH");
    ASSERT_ALWAYS(ok);
    expect_same(hbm, dbm);
}

// coverity[root_function]
int main(int argc, char const * argv[])
{
    MPI_Init(&argc, (char ***) &argv);

    unsigned int m = 64, n = 64, L = 200, threshold = 60, zeros = 0;
    unsigned long seed = 0;
    cxx_param_list pl;
    pl.process_command_line(argc, argv, false);
    pl.parse("m", m);
    pl.parse("n", n);
    pl.parse("L", L);
    pl.parse("threshold", threshold);
    pl.parse("seed", seed);
    pl.parse("zeros", zeros);
    ASSERT_ALWAYS(zeros <= m + n);
    lingen_ft_plugin_init(pl);
    ASSERT_ALWAYS(bwc_lingen_ft_plugin);

    cxx_gmp_randstate rstate;
    gmp_randseed_ui(rstate, seed);
    cxx_mpz const p = 2;
    matpoly<true>::arith_hard ab(p, 1U);
    matpoly<true>::memory_guard const blanket(SIZE_MAX);

    matpoly<true> const E = random_matpoly(&ab, m, m + n, L, zeros, rstate);
    test_shifts(E);
    matpoly<true> const pi = test_basecase(E, m, n);
    test_products(E, pi);
    /* with zero columns, the CPU recursion stops at its pi length check */
    if (!zeros)
        test_recursion(E, m, n, threshold);

    MPI_Finalize();
}
