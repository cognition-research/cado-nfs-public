#ifndef GLINGEN_H
#define GLINGEN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GPU Cantor transforms of polynomial matrices over GF(2) for the
 * lingen matrix products of CADO-NFS (lingen_matpoly_ft): the dft, the
 * product of transformed matrices, and the ift, all on the device, the
 * word-for-word result of the gf2x code. A transform is 2^k elements
 * of GF(2^128), two 64-bit words each, modulus x^128 + x^7 + x^2 + x
 * + 1, of which the first n carry data. */

/* a matrix of transforms on the device */
struct glingen_tmat;

/* the block of rows i0.. and columns j0.. of a matrix */
struct glingen_sub {
    unsigned int i0, j0, nrows, ncols;
};

/* A block of a matrix of polynomials in host memory, the lingen
 * matpoly layout: entry (i,j) of the block is the 64-bit words at
 * x + (i*ld + j) * words, of which the low bits are the coefficients.
 * The block has the dimensions of the device block it goes with. */
struct glingen_polys {
    uint64_t * x;
    size_t ld, words, bits;
};

/* The Cantor transform parameters of gf2x_cantor_fft_info: transforms
 * of 2^k points, of which n are significant; mp_shift is the shift of
 * the middle product, 0 for a plain product. */
struct glingen_cantor {
    unsigned int k;
    size_t n, mp_shift;
};

/* nrows x ncols transforms of 2^k points, zero */
struct glingen_tmat * glingen_alloc(unsigned int nrows, unsigned int ncols,
                                    unsigned int k);
void glingen_free(struct glingen_tmat * t);
void glingen_zero(struct glingen_tmat * t, struct glingen_sub R);

/* the transforms of the polynomials of a into the block R of t */
void glingen_dft(struct glingen_tmat * t, struct glingen_sub R,
                 struct glingen_polys a, struct glingen_cantor fti);

/* c += a * b over the first npts points: the product is pointwise in
 * the transform, c[i][j] += sum over k of a[i][k] * b[k][j]. Rc is
 * nrows x ncols, Ra is nrows x nadd, Rb is nadd x ncols. */
void glingen_addcompose(struct glingen_tmat * c, struct glingen_sub Rc,
                        const struct glingen_tmat * a, struct glingen_sub Ra,
                        const struct glingen_tmat * b, struct glingen_sub Rb,
                        size_t npts);

/* the inverse transforms of the block R of t into the polynomials of
 * a, of a.bits coefficients */
void glingen_ift(struct glingen_polys a, const struct glingen_tmat * t,
                 struct glingen_sub R, struct glingen_cantor fti);

/* ---- polynomial matrices of the glingen program ----
 *
 * A glingen_polys in host memory (ld = ncols), in the matpoly layout:
 * the recursion of lingen keeps its matrices there, and the products
 * move the blocks they transform to the devices, as many at a time as
 * the device memory holds. The bits above .bits of an entry are zero. */

/* nrows x ncols zero polynomials of bits coefficients */
struct glingen_polys glingen_mp_alloc(unsigned int nrows, unsigned int ncols,
                                      size_t bits);
void glingen_mp_free(struct glingen_polys a);

/* dst = src, nrows x ld entries */
void glingen_mp_copy(struct glingen_polys dst, struct glingen_polys src,
                     unsigned int nrows);

/* dst = src / x^k, dst.bits coefficients kept, the bits above cleared;
 * dst may be src */
void glingen_mp_rshift(struct glingen_polys dst, struct glingen_polys src,
                       unsigned int nrows, size_t k);

/* 1 when coefficient k of a is the zero matrix */
int glingen_mp_coeff_is_zero(struct glingen_polys a, unsigned int nrows,
                             size_t k);

/* c = a * b, or the middle product when fti.mp_shift > 0, with fti the
 * parameters of that product: the transforms, the pointwise product and
 * the inverse transform all run on the devices. c is nrows x ncols of
 * c.bits coefficients, a is nrows x nadd, b is nadd x ncols. */
void glingen_mp_product(struct glingen_polys c, struct glingen_polys a,
                        struct glingen_polys b, unsigned int nrows,
                        unsigned int nadd, unsigned int ncols,
                        struct glingen_cantor fti);

/* ---- the binary base case of lingen on the device ----
 *
 * The iteration of lingen_qcode_binary (generator_found, the pivot
 * search of bpack_view::ple, the update of E and pi) on the device,
 * with the same pivot order as the CPU code. b and m are multiples of
 * 64, b at most 2048. */
struct glingen_bc {
    struct glingen_polys E;     /* m x b, L = E.bits coefficients */
    unsigned int m, b;
    unsigned int * delta;       /* b entries, in and out: bm.delta */
    int * lucky;                /* b entries, in and out: bm.lucky */
    unsigned int luck_mini;
    unsigned int t;             /* out: the steps done, L when no generator was found */
    unsigned int * log;         /* out: L * (1 + b) entries for the luck reports */
    unsigned int nlog;          /* out: the reports, each the step and lucky after it */
};

/* pi as b x b polynomials, sized to its degree */
struct glingen_polys glingen_basecase(struct glingen_bc * bc);

#ifdef __cplusplus
}
#endif

#endif
