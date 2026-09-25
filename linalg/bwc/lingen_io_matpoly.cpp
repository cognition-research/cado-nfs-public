#include "cado.h" // IWYU pragma: keep

#ifdef LINGEN_BINARY
#include <climits>
#include <cstdlib>
#endif
#include <cstdio>

#include <fstream>
#include <ostream>

#ifndef LINGEN_BINARY
#include <gmp.h>
#endif
#ifdef LINGEN_BINARY
#include <sys/types.h>
#include <unistd.h>     // pread
#endif

#include "bblas_level3a.hpp"
#include "lingen_io_matpoly.hpp"
#include "lingen_matpoly_select.hpp"
#include "macros.h"
#include "params.hpp"
#ifdef HAVE_OPENMP
#include "omp_proxy.h"
#endif



/* {{{ I/O helpers */

/* This is an indication of the number of bytes we read at a time for A
 * (input) and F (output) */
template<bool is_binary>
unsigned int lingen_io_matpoly<is_binary>::block_size = 1 << 20;

template<bool is_binary>
void lingen_io_matpoly<is_binary>::decl_usage(cxx_param_list & pl)
{
    pl.declare_usage("io-block-size",
            "chunk size for reading the input or writing the output");
}

template<bool is_binary>
void lingen_io_matpoly<is_binary>::lookup_parameters(cxx_param_list & pl)
{
    pl.lookup("io-block-size");
}

template<bool is_binary>
void lingen_io_matpoly<is_binary>::interpret_parameters(cxx_param_list & pl)
{
    pl.parse("io-block-size", block_size);
}

/* {{{ matpoly_write
 * writes some of the matpoly data to f, either in ascii or binary
 * format. This can be used to write only part of the data (degrees
 * [k0..k1[). Returns the number of coefficients (i.e., matrices, so at
 * most k1-k0) successfully written, or
 * -1 on error (e.g. when some matrix was only partially written).
 */

#ifndef LINGEN_BINARY
template<>
int lingen_io_matpoly<false>::write(matpoly<false>::arith_hard * ab, std::ostream& os, matpoly<false> const & M, unsigned int k0, unsigned int k1, int ascii, int transpose)
{
    unsigned int const m = transpose ? M.n : M.m;
    unsigned int const n = transpose ? M.m : M.n;
    ASSERT_ALWAYS(k0 == k1 || (k0 < M.get_size() && k1 <= M.get_size()));
    for(unsigned int k = k0 ; k < k1 ; k++) {
        int err = 0;
        int matnb = 0;
        for(unsigned int i = 0 ; !err && i < m ; i++) {
            for(unsigned int j = 0 ; !err && j < n ; j++) {
                matpoly<false>::arith_hard::elt const & x = M.coeff(transpose?j:i,transpose?i:j,k);
                if (ascii) {
                    if (j) err = !(os << " ");
                    if (!err) err = !(ab->cxx_out(os, x));
                } else {
                    err = !ab->write(os, x);
                }
                if (!err) matnb++;
            }
            if (!err && ascii) err = !(os << "\n");
        }
        if (ascii) err = err || !(os << "\n");
        if (err) {
            return (matnb == 0) ? (int) (k - k0) : -1;
        }
    }
    return k1 - k0;
}
#else
/* In the file, one coefficient matrix is ns rows of pc words, each word
 * packing 64 entries of the other dimension (the transpose flag swaps
 * the roles of the rows and columns of M). In memory, the 64
 * coefficients 64*kq..64*kq+63 of an entry are the word kq of that
 * entry, so the word kq of 64 consecutive packed entries is a 64x64 bit
 * matrix whose transpose has the file word of coefficient 64*kq+kr in
 * row kr. The conversion goes 64 coefficients at a time through that
 * transpose.
 */
static_assert(ULONG_BITS == 64, "the block transpose uses mat64");
struct packed_layout {
    unsigned int ns, np, pc;
    bool transpose;
    packed_layout(matpoly<true> const & M, int transpose)
        : ns(transpose ? M.n : M.m)
        , np(transpose ? M.m : M.n)
        , pc(iceildiv(np, 64))
        , transpose(transpose)
    {}
    size_t ulongs_per_mat() const { return ns * pc; }
    unsigned long const * part(matpoly<true> const & M, unsigned int s, unsigned int p) const {
        return transpose ? M.part(p, s) : M.part(s, p);
    }
    unsigned long * part(matpoly<true> & M, unsigned int s, unsigned int p) const {
        return transpose ? M.part(p, s) : M.part(s, p);
    }
};

/* the coefficients k0..k1-1, all with the same word index kq, as
 * consecutive file matrices in buf */
static void pack_block(packed_layout const & P, matpoly<true> const & M, unsigned int k0, unsigned int k1, unsigned long * buf)
{
    unsigned int const kq = k0 / 64;
    for(unsigned int s = 0 ; s < P.ns ; s++) {
        for(unsigned int pq = 0 ; pq < P.pc ; pq++) {
            mat64 A, T;
            for(unsigned int pr = 0 ; pr < 64 ; pr++) {
                unsigned int const p = pq * 64 + pr;
                A[pr] = p < P.np ? P.part(M, s, p)[kq] : 0;
            }
            mat64_transpose(T, A);
            for(unsigned int k = k0 ; k < k1 ; k++)
                buf[(k - k0) * P.ulongs_per_mat() + s * P.pc + pq] = T[k % 64];
        }
    }
}

static void unpack_block(packed_layout const & P, matpoly<true> & M, unsigned int k0, unsigned int k1, unsigned long const * buf)
{
    unsigned int const kq = k0 / 64;
    unsigned long const keep = ~(((k1 - k0 == 64) ? ~0UL : ((1UL << (k1 - k0)) - 1)) << (k0 % 64));
    for(unsigned int s = 0 ; s < P.ns ; s++) {
        for(unsigned int pq = 0 ; pq < P.pc ; pq++) {
            mat64 A, T;
            for(unsigned int kr = 0 ; kr < 64 ; kr++) {
                unsigned int const k = kq * 64 + kr;
                A[kr] = (k0 <= k && k < k1) ? buf[(k - k0) * P.ulongs_per_mat() + s * P.pc + pq] : 0;
            }
            mat64_transpose(T, A);
            for(unsigned int pr = 0 ; pr < 64 && pq * 64 + pr < P.np ; pr++) {
                unsigned long & w = P.part(M, s, pq * 64 + pr)[kq];
                w = (w & keep) | T[pr];
            }
        }
    }
}

template<>
int lingen_io_matpoly<true>::write(matpoly<true>::arith_hard *, std::ostream& os, matpoly<true> const & M, unsigned int k0, unsigned int k1, int ascii, int transpose)
{
    /* do we have an endian-robust wordsize-robust convention for
     * printing bitstrings in hex ? it's not even clear that we should
     * care -- after all, as long as mksol follows a consistent
     * convention too, we should be fine. */
    ASSERT_ALWAYS(!ascii);
    ASSERT_ALWAYS(k0 == k1 || (k0 < M.get_size() && k1 <= M.get_size()));
    packed_layout const P(M, transpose);
    std::vector<unsigned long> buf(P.ulongs_per_mat() * 64);
    for(unsigned int k = k0 ; k < k1 ; ) {
        unsigned int const kk = MIN(k1, (k / 64 + 1) * 64);
        pack_block(P, M, k, kk, buf.data());
        if (!os.write((const char*) buf.data(), (kk - k) * P.ulongs_per_mat() * sizeof(unsigned long)))
            return -1;
        k = kk;
    }
    return k1 - k0;
}
#endif
/* }}} */

/* fw must be an array of ofstreams of exactly the same size as the
 * matrix to be written.
 */

#ifdef LINGEN_BINARY
template<>
int lingen_io_matpoly<true>::write_split(matpoly<true>::arith_hard * ab MAYBE_UNUSED, std::vector<std::ofstream> & fw, matpoly<true> const & M, unsigned int k0, unsigned int k1, int ascii)
{
    constexpr const unsigned int splitwidth = 64;
    ASSERT_ALWAYS(k0 == k1 || (k0 < M.get_size() && k1 <= M.get_size()));
    size_t const ulongs_per_mat = splitwidth * splitwidth / ULONG_BITS;
    std::vector<unsigned long> buf(ulongs_per_mat);
    for(unsigned int k = k0 ; k < k1 ; k++) {
        int err = 0;
        int matnb = 0;
        for(unsigned int i = 0 ; !err && i < M.m ; i += splitwidth) {
            for(unsigned int j = 0 ; !err && j < M.n ; j += splitwidth) {
                std::ostream& os = fw[i/splitwidth*M.n/splitwidth+j/splitwidth];
                /* In fact, we _do_ have code that we might want to use
                 * in the b64 layer.
                 */
                if (ascii)
                    abort();
                buf.assign(ulongs_per_mat, 0);
                size_t const kq = k / ULONG_BITS;
                size_t const kr = k % ULONG_BITS;
                for(unsigned int di = 0 ; di < splitwidth ; di++) {
                    unsigned int const ii = i + di;
                    unsigned int const ulongs_per_row = splitwidth / ULONG_BITS;
                    for(unsigned int dj0 = 0 ; dj0 < ulongs_per_row ; dj0++) {
                        for(unsigned int dj = 0 ; dj < ULONG_BITS ; dj++) {
                            unsigned int const jj = j + dj0 * ULONG_BITS + dj;
                            const unsigned long * mij = M.part(ii, jj);
                            unsigned long const bit = (mij[kq] >> kr) & 1;
                            buf[di * ulongs_per_row + dj0] ^= bit << dj;
                        }
                    }
                }
                err = !(os.write((const char *) buf.data(), ulongs_per_mat * sizeof(unsigned long)));
                if (!err) matnb++;
            }
        }
        if (err) {
            return (matnb == 0) ? (int) (k - k0) : -1;
        }
    }
    return k1 - k0;
}
#endif

#ifndef LINGEN_BINARY
template<>
int lingen_io_matpoly<false>::write_split(matpoly<false>::arith_hard * ab MAYBE_UNUSED, std::vector<std::ofstream> & fw, matpoly<false> const & M, unsigned int k0, unsigned int k1, int ascii)
{
    constexpr const unsigned int splitwidth = 1;
    ASSERT_ALWAYS(k0 == k1 || (k0 < M.get_size() && k1 <= M.get_size()));
    for(unsigned int k = k0 ; k < k1 ; k++) {
        int err = 0;
        int matnb = 0;
        for(unsigned int i = 0 ; !err && i < M.m ; i += splitwidth) {
            for(unsigned int j = 0 ; !err && j < M.n ; j += splitwidth) {
                std::ostream& os = fw[i/splitwidth*M.n/splitwidth+j/splitwidth];
                auto const & x = M.coeff(i, j, k);
                if (ascii) {
                    err = !(ab->cxx_out(os, x));
                    if (!err) err = !(os << "\n");
                } else {
                    err = !ab->write(os, x);
                }
                if (!err) matnb++;
            }
        }
        if (err) {
            return (matnb == 0) ? (int) (k - k0) : -1;
        }
    }
    return k1 - k0;
}
#endif

/* }}} */

/* {{{ matpoly_read
 * reads some of the matpoly data from f, either in ascii or binary
 * format. This can be used to parse only part of the data (degrees
 * [k0..k1[, k1 being an upper bound). Returns the number of coefficients
 * (i.e., matrices, so at most k1-k0) successfully read, or
 * -1 on error (e.g. when some matrix was only partially read).
 *
 * Note that the matrix must *not* be in pre-init state. It must have
 * been already allocated.
 */

#ifndef LINGEN_BINARY
template<>
int lingen_io_matpoly<false>::read(matpoly<false>::arith_hard * ab, FILE * f, matpoly<false> & M, unsigned int k0, unsigned int k1, int ascii, int transpose)
{
    ASSERT_ALWAYS(!M.check_pre_init());
    unsigned int const m = transpose ? M.n : M.m;
    unsigned int const n = transpose ? M.m : M.n;
    ASSERT_ALWAYS(k0 == k1 || (k0 < M.get_size() && k1 <= M.get_size()));
    for(unsigned int k = k0 ; k < k1 ; k++) {
        int err = 0;
        int matnb = 0;
        for(unsigned int i = 0 ; !err && i < m ; i++) {
            for(unsigned int j = 0 ; !err && j < n ; j++) {
                auto & x = M.coeff(transpose?j:i,transpose?i:j,k);
                if (ascii) {
                    err = ab->fscan(f, x) == 0;
                } else {
                    err = ab->fread(f, x) < 1;
                }
                if (!err) matnb++;
            }
        }
        if (err) {
            if (matnb == 0) 
                return (int) (k - k0);
            fprintf(stderr, "I/O error after %d,%d coefficients (matrix size %u*%u)\n",
                    (int) (k-k0), matnb, m, n);
            return -1;
        }
    }
    return k1 - k0;
}
#else
template<>
int lingen_io_matpoly<true>::read(matpoly<true>::arith_hard *, FILE * f, matpoly<true> & M, unsigned int k0, unsigned int k1, int ascii, int transpose)
{
    ASSERT_ALWAYS(!ascii);
    ASSERT_ALWAYS(!M.check_pre_init());
    ASSERT_ALWAYS(k0 == k1 || (k0 < M.get_size() && k1 <= M.get_size()));
    packed_layout const P(M, transpose);
    size_t const one = P.ulongs_per_mat() * sizeof(unsigned long);
    off_t const pos0 = ftell(f);
    unsigned int rc = 0;
    /* one block of (at most) 64 coefficients per iteration, read with
     * pread so that the blocks go in parallel */
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:rc)
#endif
    for(unsigned int kq = k0 / 64 ; kq < iceildiv(k1, 64) ; kq++) {
        unsigned int const a = MAX(k0, kq * 64);
        unsigned int const b = MIN(k1, (kq + 1) * 64);
        std::vector<unsigned long> buf(P.ulongs_per_mat() * (b - a));
        ssize_t const r = pread(fileno(f), buf.data(), (b - a) * one, pos0 + (a - k0) * one);
        unsigned int const good = r < 0 ? 0 : r / one;
        unpack_block(P, M, a, a + good, buf.data());
        rc += good;
    }
    fseek(f, pos0 + rc * one, SEEK_SET);
    return rc;
}

#endif
/* }}} */

#ifdef LINGEN_BINARY
template struct lingen_io_matpoly<true>;
#else
template struct lingen_io_matpoly<false>;
#endif
