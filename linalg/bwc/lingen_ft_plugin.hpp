#ifndef CADO_LINGEN_FT_PLUGIN_HPP
#define CADO_LINGEN_FT_PLUGIN_HPP

#include <cstddef>
#include <cstdint>

#include "params.hpp"

/* A transform plugin holds the matrices of Cantor transforms of
 * matpoly_ft<gf2x_cantor_fft_info> in its own memory (a GPU), and does
 * the dft, the pointwise matrix product and the ift there; only the
 * polynomial coefficients cross. lingen_ft_impl=<impl> loads
 * libbwclingenft_<impl> through dlopen, with the same naming scheme as
 * the matmul backends. Without the parameter, the gf2x code runs.
 *
 * The results are word for word those of gf2x. The tuning of lingen
 * times the transforms and products through the plugin too, so its
 * choices (fft or plain product, batch sizes) account for it.
 */

/* the block of rows i0.. and columns j0.. of a matrix of transforms */
struct bwc_lingen_ft_sub {
    unsigned int i0, j0, nrows, ncols;
};

/* A block of a matpoly in host memory: entry (i,j) of the block is the
 * 64-bit words at x + (i*ld + j) * words, of which the low bits are
 * the coefficients. */
struct bwc_lingen_ft_polys {
    uint64_t * x;
    size_t ld, words, bits;
};

/* the k, n and mp_shift of gf2x_cantor_fft_info */
struct bwc_lingen_ft_cantor {
    unsigned int k;
    size_t n, mp_shift;
};

/* The base case of lingen_qcode_binary with E (m x b, E.bits
 * coefficients) in plugin memory: delta and lucky (b entries) are
 * bm.delta and bm.lucky, updated as the CPU code does; t receives the
 * steps done; log (L * (1 + b) entries) receives nlog reports, each
 * the step and lucky after it, for the steps generator_found reports. */
struct bwc_lingen_bc {
    bwc_lingen_ft_polys E;
    unsigned int m, b;
    unsigned int * delta;
    int * lucky;
    unsigned int luck_mini;
    unsigned int t;
    unsigned int * log;
    unsigned int nlog;
};

struct bwc_lingen_ft_plugin_api {
    /* nrows x ncols transforms of 2^k points, zero */
    void * (*alloc)(unsigned int nrows, unsigned int ncols, unsigned int k);
    void (*free)(void * t);
    void (*zero)(void * t, bwc_lingen_ft_sub R);
    /* the transforms of the polynomials of a into the block R of t */
    void (*dft)(void * t, bwc_lingen_ft_sub R, bwc_lingen_ft_polys a,
            bwc_lingen_ft_cantor fti);
    /* c[i][j] += sum over k of a[i][k] * b[k][j] in GF(2^128), for each
     * of the first npts points */
    void (*addcompose)(void * c, bwc_lingen_ft_sub Rc,
            const void * a, bwc_lingen_ft_sub Ra,
            const void * b, bwc_lingen_ft_sub Rb, size_t npts);
    /* the inverse transforms of the block R of t into a, a.bits
     * coefficients each */
    void (*ift)(bwc_lingen_ft_polys a, const void * t, bwc_lingen_ft_sub R,
            bwc_lingen_ft_cantor fti);

    /* Polynomial matrices in the plugin's memory, for glingen: the
     * recursion keeps its matrices there, in the matpoly layout, with
     * the bits above .bits zero. dft and ift take them as well. */
    bwc_lingen_ft_polys (*mp_alloc)(unsigned int nrows, unsigned int ncols,
            size_t bits);
    void (*mp_free)(bwc_lingen_ft_polys a);
    /* dst = src, each in host or plugin memory, nrows x ld entries */
    void (*mp_copy)(bwc_lingen_ft_polys dst, bwc_lingen_ft_polys src,
            unsigned int nrows);
    /* dst = src / x^k, dst.bits coefficients kept, the bits above
     * cleared; dst may be src */
    void (*mp_rshift)(bwc_lingen_ft_polys dst, bwc_lingen_ft_polys src,
            unsigned int nrows, size_t k);
    /* 1 when coefficient k of a is zero */
    int (*mp_coeff_is_zero)(bwc_lingen_ft_polys a, unsigned int nrows,
            size_t k);
    /* c = a * b (nrows x nadd times nadd x ncols), the middle product
     * when fti.mp_shift > 0; c.bits coefficients */
    void (*mp_product)(bwc_lingen_ft_polys c, bwc_lingen_ft_polys a,
            bwc_lingen_ft_polys b, unsigned int nrows, unsigned int nadd,
            unsigned int ncols, bwc_lingen_ft_cantor fti);

    /* pi, sized to its degree */
    bwc_lingen_ft_polys (*basecase)(bwc_lingen_bc * bc);
};

void lingen_ft_plugin_decl_usage(cxx_param_list & pl);
void lingen_ft_plugin_init(cxx_param_list & pl);

extern bwc_lingen_ft_plugin_api const * bwc_lingen_ft_plugin;

#endif  /* CADO_LINGEN_FT_PLUGIN_HPP */
