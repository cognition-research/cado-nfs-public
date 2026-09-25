#include "cado.h" // IWYU pragma: keep

#include <cstdint>

#include <algorithm>
#include <utility>
#include <vector>

#include "gf2x-fft.h"

#include "lingen_bmstatus.hpp"
#include "lingen_expected_pi_length.hpp"
#include "lingen_gpu.hpp"
#include "lingen_qcode_binary.hpp"
#include "macros.h"

static bwc_lingen_ft_plugin_api const & P()
{
    ASSERT_ALWAYS(bwc_lingen_ft_plugin);
    return *bwc_lingen_ft_plugin;
}

static bwc_lingen_ft_polys host_polys(matpoly<true> const & a)
{
    return { (uint64_t *) a.data_area(), a.ncols(),
        a.data_entry_alloc_size_in_words(), a.get_size() };
}

static bwc_lingen_ft_cantor cantor(gf2x_cantor_fft_info const & fti)
{
    return { fti.k, fti.n, fti.mp_shift };
}

dmatpoly::dmatpoly(unsigned int m, unsigned int n, size_t size)
    : x(P().mp_alloc(m, n, size)), m(m)
{}

dmatpoly::dmatpoly(matpoly<true> const & a)
    : dmatpoly(a.nrows(), a.ncols(), a.get_size())
{
    P().mp_copy(x, host_polys(a), m);
    clear_high();
}

void dmatpoly::clear_high() const
{
    P().mp_rshift(x, x, m, 0);
}

dmatpoly & dmatpoly::operator=(dmatpoly && o) noexcept
{
    std::swap(x, o.x);
    std::swap(m, o.m);
    return *this;
}

void dmatpoly::clear()
{
    if (!x.x) return;
    P().mp_free(x);
    x = { nullptr, 0, 0, 0 };
}

matpoly<true> dmatpoly::download(matpoly<true>::arith_hard * ab) const
{
    matpoly<true> a(ab, m, ncols(), get_size());
    a.set_size(get_size());
    P().mp_copy(host_polys(a), x, m);
    return a;
}

void dmatpoly::set_size(size_t s)
{
    ASSERT_ALWAYS(s <= x.bits);
    x.bits = s;
    clear_high();
}

int dmatpoly::coeff_is_zero(size_t k) const
{
    return P().mp_coeff_is_zero(x, m, k);
}

void dmatpoly::rshift(dmatpoly const & src, size_t k)
{
    ASSERT_ALWAYS(&src == this && k <= x.bits);
    bwc_lingen_ft_polys d = x;
    d.bits -= k;
    P().mp_rshift(d, x, m, k);
    x.bits = d.bits;
}

dmatpoly dmatpoly::truncate_and_rshift(size_t size, size_t k)
{
    ASSERT_ALWAYS(size <= x.bits);
    dmatpoly low(m, ncols(), size);
    P().mp_rshift(low.x, x, m, 0);
    rshift(*this, k);
    return low;
}

static dmatpoly product(dmatpoly const & a, dmatpoly const & b, size_t csize,
        gf2x_cantor_fft_info const & fti)
{
    ASSERT_ALWAYS(a.ncols() == b.nrows());
    dmatpoly c(a.nrows(), b.ncols(), csize);
    P().mp_product(c.polys(), a.polys(), b.polys(), a.nrows(), a.ncols(),
            b.ncols(), cantor(fti));
    return c;
}

dmatpoly dmatpoly::mp(dmatpoly const & E, dmatpoly const & pi)
{
    ASSERT_ALWAYS(E.get_size() >= pi.get_size());
    return product(E, pi, E.get_size() - pi.get_size() + 1,
            gf2x_cantor_fft_info::mp_info(E.get_size(), pi.get_size()));
}

dmatpoly dmatpoly::mul(dmatpoly const & a, dmatpoly const & b)
{
    return product(a, b, a.get_size() + b.get_size() - 1,
            gf2x_cantor_fft_info::mul_info(a.get_size(), b.get_size()));
}

/* bw_lingen_basecase_raw_fast on the device; the reports come back in
 * a log and print afterwards */
dmatpoly glingen_basecase_raw(bmstatus<true> & bm, dmatpoly const & E)
{
    unsigned int const m = bm.d.m, b = m + bm.d.n, L = E.get_size();
    std::vector<unsigned int> log((size_t) L * (1 + b));
    bwc_lingen_bc bc { E.polys(), m, b, bm.delta.data(), bm.lucky.data(),
        expected_pi_length(bm.d), 0, log.data(), 0 };
    dmatpoly pi { P().basecase(&bc), b };
    std::vector<int> lucky(b);
    for(unsigned int i = 0 ; i < bc.nlog ; i++) {
        unsigned int const * rec = &log[(size_t) i * (1 + b)];
        std::copy(rec + 1, rec + 1 + b, lucky.begin());
        luck_report(bm.t + rec[0], lucky, bm.d.n, bc.luck_mini);
    }
    bm.t += bc.t;
    bm.done = bc.t < L;
    return pi;
}

dmatpoly glingen_basecase(bmstatus<true> & bm, dmatpoly & E)
{
    lingen_call_companion const & C = bm.companion(bm.depth, E.get_size());
    tree_stats::sentinel const dummy(bm.stats, "basecase", E.get_size(), C.total_ncalls, true);
    bmstatus<true>::depth_sentinel const ddummy(bm);
    bm.stats.plan_smallstep("basecase", C.ttb);
    bm.stats.begin_smallstep("basecase");
    dmatpoly pi = glingen_basecase_raw(bm, E);
    bm.stats.end_smallstep();
    E.clear();
    return pi;
}

int load_checkpoint_file(bmstatus<true> & bm, cp_which which, dmatpoly & X, unsigned int t0, unsigned int t1)
{
    matpoly<true> h;
    if (!load_checkpoint_file(bm, which, h, t0, t1)) return 0;
    X = dmatpoly(h);
    return 1;
}

int save_checkpoint_file(bmstatus<true> & bm, cp_which which, dmatpoly const & X, unsigned int t0, unsigned int t1)
{
    if ((t1 - t0) < lingen_checkpoint<true>::threshold) return 0;
    return save_checkpoint_file(bm, which, X.download(&bm.d.ab), t0, t1);
}
