/* The lingen transform plugin (lingen_ft_impl=cuda): glingen behind
 * the plugin hook of the CADO tree (linalg/bwc/lingen_ft_plugin.hpp).
 * The hook passes the matrices as void *, and its structs are those
 * of glingen.h member for member. */
#include "lingen_ft_plugin.hpp"

#include "glingen.h"

static glingen_sub sub(bwc_lingen_ft_sub R)
{
    return { R.i0, R.j0, R.nrows, R.ncols };
}

static glingen_polys polys(bwc_lingen_ft_polys a)
{
    return { a.x, a.ld, a.words, a.bits };
}

static bwc_lingen_ft_polys cpolys(glingen_polys a)
{
    return { a.x, a.ld, a.words, a.bits };
}

static glingen_cantor cantor(bwc_lingen_ft_cantor f)
{
    return { f.k, f.n, f.mp_shift };
}

static glingen_tmat * tmat(void * t)
{
    return static_cast<glingen_tmat *>(t);
}

static const glingen_tmat * tmat(const void * t)
{
    return static_cast<const glingen_tmat *>(t);
}

static bwc_lingen_ft_plugin_api const api = {
    [](unsigned int nrows, unsigned int ncols, unsigned int k) -> void * {
        return glingen_alloc(nrows, ncols, k);
    },
    [](void * t) {
        glingen_free(tmat(t));
    },
    [](void * t, bwc_lingen_ft_sub R) {
        glingen_zero(tmat(t), sub(R));
    },
    [](void * t, bwc_lingen_ft_sub R, bwc_lingen_ft_polys a,
            bwc_lingen_ft_cantor fti) {
        glingen_dft(tmat(t), sub(R), polys(a), cantor(fti));
    },
    [](void * c, bwc_lingen_ft_sub Rc, const void * a, bwc_lingen_ft_sub Ra,
            const void * b, bwc_lingen_ft_sub Rb, size_t npts) {
        glingen_addcompose(tmat(c), sub(Rc), tmat(a), sub(Ra), tmat(b), sub(Rb),
                npts);
    },
    [](bwc_lingen_ft_polys a, const void * t, bwc_lingen_ft_sub R,
            bwc_lingen_ft_cantor fti) {
        glingen_ift(polys(a), tmat(t), sub(R), cantor(fti));
    },
    [](unsigned int nrows, unsigned int ncols, size_t bits) {
        return cpolys(glingen_mp_alloc(nrows, ncols, bits));
    },
    [](bwc_lingen_ft_polys a) {
        glingen_mp_free(polys(a));
    },
    [](bwc_lingen_ft_polys dst, bwc_lingen_ft_polys src, unsigned int nrows) {
        glingen_mp_copy(polys(dst), polys(src), nrows);
    },
    [](bwc_lingen_ft_polys dst, bwc_lingen_ft_polys src, unsigned int nrows,
            size_t k) {
        glingen_mp_rshift(polys(dst), polys(src), nrows, k);
    },
    [](bwc_lingen_ft_polys a, unsigned int nrows, size_t k) {
        return glingen_mp_coeff_is_zero(polys(a), nrows, k);
    },
    [](bwc_lingen_ft_polys c, bwc_lingen_ft_polys a, bwc_lingen_ft_polys b,
            unsigned int nrows, unsigned int nadd, unsigned int ncols,
            bwc_lingen_ft_cantor fti) {
        glingen_mp_product(polys(c), polys(a), polys(b), nrows, nadd, ncols,
                cantor(fti));
    },
    [](bwc_lingen_bc * bc) {
        glingen_bc g = { polys(bc->E), bc->m, bc->b, bc->delta, bc->lucky,
            bc->luck_mini, 0, bc->log, 0 };
        bwc_lingen_ft_polys pi = cpolys(glingen_basecase(&g));
        bc->t = g.t;
        bc->nlog = g.nlog;
        return pi;
    },
};

extern "C" bwc_lingen_ft_plugin_api const * bwc_lingen_ft_solib_reach_api()
{
    return &api;
}
