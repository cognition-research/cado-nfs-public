#ifndef CADO_MATRIX_OUTPUT_HPP
#define CADO_MATRIX_OUTPUT_HPP

/* Output of the final filtered matrix: the sparse and dense matrix files,
 * their row and column weight files, and the index file. replay and
 * merge write these files with the functions below.
 *
 * The rows come from a row source R with:
 *   R::empty(i)          true if row i has no entry
 *   R::for_each(i, f)    calls f(typerow_t const & cell) for each entry
 *                        of row i, in increasing order of the original
 *                        column index
 * code[c] - 1 is the new index of the column c that the cells refer to.
 * The dense part gets the columns with new index below skip.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <vector>

#include <unistd.h>

#include "fix-endianness.h"
#include "gzip.h"
#include "macros.h"
#include "misc.h"
#include "omp_proxy.h"
#include "sparse.h"
#include "typedefs.h"

#ifdef FOR_DL
#define ENTRY_WORDS 2   /* 32-bit words per matrix entry in binary format */
static inline index_t cell_col(typerow_t const & c) { return c.id; }
#else
#define ENTRY_WORDS 1
static inline index_t cell_col(typerow_t const & c) { return c; }
#endif

/* Writes 32-bit little-endian words to a binary output file. The words go
 * to a buffer first. A plain file gets each full buffer with pwrite at the
 * byte position pos, so several word_writer objects can write disjoint
 * ranges of one file at the same time. A pipe gets each full buffer with
 * fwrite, so only one word_writer at a time can write to a pipe. */
struct word_writer {
    FILE * f;
    bool seekable;
    off_t pos;
    std::vector<uint32_t> buf;
    static constexpr size_t cap = 1 << 20;

    word_writer(FILE * f, bool seekable, off_t pos)
        : f(f), seekable(seekable), pos(pos)
    { buf.reserve(cap); }
    word_writer(word_writer const &) = delete;
    ~word_writer() { flush(); }

    void put(uint32_t x) {
        buf.push_back(bfix32(x));
        if (buf.size() == cap) flush();
    }
    void flush() {
        if (buf.empty()) return;
        const char * p = (const char *) buf.data();
        size_t n = buf.size() * sizeof(uint32_t);
        if (!seekable) {
            ASSERT_ALWAYS(fwrite(p, 1, n, f) == n);
            buf.clear();
            return;
        }
        while (n) {
            ssize_t w = pwrite(fileno(f), p, n, pos);
            ASSERT_ALWAYS(w > 0);
            n -= w; p += w; pos += w;
        }
        buf.clear();
    }
};

/* Writes the rows [i0, i1) in binary format. Each row is a 32-bit weight
 * followed by its entries. The row weight files get one word per non-empty
 * row. The offsets tell where row i0 starts in each file. */
template<typename R>
static void
write_rows_bin(R const & rows, index_t i0, index_t i1,
               const index_t * code, index_t skip,
               const uint32_t * sw, const uint32_t * dw,
               FILE * smatfile, FILE * srwfile, FILE * dmatfile,
               FILE * drwfile, bool seekable,
               off_t soff, off_t srwoff, off_t doff, off_t drwoff)
{
    word_writer smat(smatfile, seekable, soff);
    word_writer srw(srwfile, seekable, srwoff);
    word_writer dmat(dmatfile, seekable, doff);
    word_writer drw(drwfile, seekable, drwoff);
    for (index_t i = i0; i < i1; i++) {
        if (rows.empty(i)) {
            smat.put(0);
            if (skip) dmat.put(0);
            continue;
        }
        smat.put(sw[i]);
        srw.put(sw[i]);
        if (skip) {
            dmat.put(dw[i]);
            drw.put(dw[i]);
        }
        rows.for_each(i, [&](typerow_t const & c) {
            uint32_t x = code[cell_col(c)] - 1;
            if (x < skip) {
                dmat.put(x);
                return;
            }
            smat.put(x - skip);
#ifdef FOR_DL
            /* exponents are always int32_t */
            smat.put((uint32_t) c.e);
#endif
        });
    }
}

/* Writes the rows [0, small_nrows) in text format. */
template<typename R>
static void
write_rows_txt(R const & rows, index_t small_nrows,
               const index_t * code, index_t skip,
               const uint32_t * sw, const uint32_t * dw,
               FILE * smatfile, FILE * srwfile, FILE * dmatfile,
               FILE * drwfile)
{
    for (index_t i = 0; i < small_nrows; i++) {
        if (rows.empty(i)) {
            fprintf(smatfile, "0\n");
            if (skip) fprintf(dmatfile, "0\n");
            continue;
        }
        fprintf(smatfile, "%" PRIu32 "", sw[i]);
        fprintf(srwfile, "%" PRIu32 "\n", sw[i]);
        if (skip) {
            fprintf(dmatfile, "%" PRIu32 "", dw[i]);
            fprintf(drwfile, "%" PRIu32 "\n", dw[i]);
        }
        rows.for_each(i, [&](typerow_t const & c) {
            uint32_t x = code[cell_col(c)] - 1;
            if (x < skip) {
                fprintf(dmatfile, " %" PRIu32 "", x);
                return;
            }
            fprintf(smatfile, " %" PRIu32 "", (uint32_t) (x - skip));
#ifdef FOR_DL
            fprintf(smatfile, ":%d", c.e);
#endif
        });
        fprintf(smatfile, "\n");
        if (skip) fprintf(dmatfile, "\n");
    }
}

/* Writes the matrix files derived from sparsename: the sparse matrix
 * (rows [0, small_nrows), columns [skip, small_ncols) after renumbering),
 * the dense matrix (columns [0, skip)) and their row and column weight
 * files. Returns the weight of the sparse matrix. */
template<typename R>
static unsigned long
flushSparse(const char *sparsename, R const & rows, index_t small_nrows,
            index_t small_ncols, const index_t * code, index_t skip, int bin)
{
#ifdef FOR_DL
  ASSERT_ALWAYS (skip == 0);
#endif
    /* the matrix files store column indices as uint32_t */
    ASSERT_ALWAYS (small_ncols <= UINT32_MAX);
    const struct {
        const char * ext;
        const char * smat;
        const char * srw;
        const char * scw;
        const char * dmat;
        const char * drw;
        const char * dcw;
    } suffixes[2] = {
        {
          .ext = ".txt",
          .smat = "txt",
          .srw = "rw.txt",
          .scw = "cw.txt",
          .dmat = "dense.txt",
          .drw = "dense.rw.txt",
          .dcw = "dense.cw.txt",
        },
        {
          .ext = ".bin",
          .smat = "bin",
          .srw = "rw.bin",
          .scw = "cw.bin",
          .dmat = "dense.bin",
          .drw = "dense.rw.bin",
          .dcw = "dense.cw.bin",
        },
    }, * suf = &(suffixes[bin]);

    unsigned long W = 0;
    unsigned long DW = 0;
    const char * zip = has_suffix(sparsename, ".gz") ? ".gz" : NULL;
    index_t * weights = (index_t *) malloc(small_ncols * sizeof(index_t));
    ASSERT_ALWAYS(weights != NULL);
    memset(weights, 0, small_ncols * sizeof(index_t));

    char wmode[3] = "w";
    int pipeflag = 0;

    char * base = strdup(sparsename);
    if (zip) { base[strlen(base)-3]='\0'; }
    if (has_suffix(base, suf->ext)) { base[strlen(base)-4]='\0'; }

    char * smatname = nullptr;
    FILE * smatfile = nullptr;
    char * srwname  = nullptr;
    FILE * srwfile  = nullptr;
    char * scwname  = nullptr;
    FILE * scwfile  = nullptr;

    {
        smatname = derived_filename(base, suf->smat, zip);
        srwname  = derived_filename(base, suf->srw, zip);
        scwname = derived_filename(base, suf->scw, zip);
        smatfile = fopen_maybe_compressed2(smatname, wmode, &pipeflag, NULL);
        srwfile  = fopen_maybe_compressed(srwname, wmode);
        /* XXX sm-outside-matrix creates a square matrix here */
        if (!bin) fprintf(smatfile, "%" PRIu64 " %" PRIu64 "\n",
                          (uint64_t) small_nrows,
                          (uint64_t) small_ncols - skip);
    }

    char * dmatname = nullptr;
    FILE * dmatfile = nullptr;
    char * drwname  = nullptr;
    FILE * drwfile  = nullptr;
    char * dcwname  = nullptr;
    FILE * dcwfile  = nullptr;

    if (skip) {
        /* arrange so that we don't get file names like .sparse.dense */
        char * dbase = strdup(base);
        char * tmp = strstr(dbase, ".sparse");
        if (tmp) {
            memmove(tmp, tmp + 7, strlen(tmp + 7) + 1);
        }
        dmatname = derived_filename(dbase, suf->dmat, zip);
        drwname  = derived_filename(dbase, suf->drw, zip);
        dcwname = derived_filename(dbase, suf->dcw, zip);
        dmatfile = fopen_maybe_compressed(dmatname, wmode);
        drwfile  = fopen_maybe_compressed(drwname, wmode);
        if (!bin)
            fprintf(dmatfile, "%" PRIu64 " %" PRIu64 "\n",
                    (uint64_t) small_nrows, (uint64_t) skip);
        free(dbase);
    }

    /* pass 1: weight of each row in the sparse and dense parts, and
     * weight of each column. */
    uint32_t * sw = (uint32_t *) malloc(small_nrows * sizeof(uint32_t));
    uint32_t * dw = (uint32_t *) malloc(small_nrows * sizeof(uint32_t));
    ASSERT_ALWAYS(sw != NULL && dw != NULL);
#pragma omp parallel for schedule(static) reduction(+:W,DW)
    for (index_t i = 0; i < small_nrows; i++) {
        sw[i] = dw[i] = 0;
        if (rows.empty(i)) continue;
        rows.for_each(i, [&](typerow_t const & c) {
            ASSERT_ALWAYS(code[cell_col(c)]-1 <= (index_t) UINT32_MAX);
            uint32_t x = code[cell_col(c)] - 1;
#pragma omp atomic
            weights[x]++;
            if (x < skip) dw[i]++; else sw[i]++;
        });
        W += sw[i];
        DW += dw[i];
    }

    /* pass 2: write the rows. In binary format each thread writes one
     * range of rows to plain files. A pipe gets all rows from one thread.
     */
    if (!bin) {
        write_rows_txt(rows, small_nrows, code, skip, sw, dw,
                       smatfile, srwfile, dmatfile, drwfile);
    } else {
        const int nchunks = pipeflag ? 1 : omp_get_max_threads();
        std::vector<index_t> bound(nchunks + 1);
        std::vector<off_t> soff(nchunks + 1), srwoff(nchunks + 1);
        std::vector<off_t> doff(nchunks + 1), drwoff(nchunks + 1);
        for (int c = 0; c <= nchunks; c++)
            bound[c] = (index_t) ((uint64_t) small_nrows * c / nchunks);
        for (int c = 0; c < nchunks; c++) {
            off_t ns = 0, nd = 0, ne = 0;
            for (index_t i = bound[c]; i < bound[c+1]; i++) {
                ns += sw[i];
                nd += dw[i];
                ne += !rows.empty(i);
            }
            index_t nr = bound[c+1] - bound[c];
            soff[c+1] = soff[c] + 4 * (nr + ENTRY_WORDS * ns);
            srwoff[c+1] = srwoff[c] + 4 * ne;
            doff[c+1] = doff[c] + 4 * (nr + nd);
            drwoff[c+1] = drwoff[c] + 4 * ne;
        }
#pragma omp parallel for schedule(static)
        for (int c = 0; c < nchunks; c++)
            write_rows_bin(rows, bound[c], bound[c+1], code, skip,
                           sw, dw, smatfile, srwfile, dmatfile, drwfile,
                           !pipeflag, soff[c], srwoff[c], doff[c], drwoff[c]);
    }
    free(sw);
    free(dw);

    {
        fclose_maybe_compressed (smatfile, smatname);
        if (srwfile) fclose_maybe_compressed (srwfile, srwname);
    }
    if (skip) {
        printf("%lu coeffs (out of %lu total) put into %s (%.1f%%)\n",
                DW, DW+W, dmatname,
                100.0 * (double) DW / (DW+W+(DW==0&&W==0)));
        fflush(stdout);

        fclose_maybe_compressed (dmatfile, dmatname);
        fclose_maybe_compressed (drwfile, drwname);
    }

    if (skip) {
        dcwfile = fopen_maybe_compressed(dcwname, wmode);
        for (unsigned int j = 0; j < skip; j++){
	    ASSERT_ALWAYS(weights[j] <= (index_t) UINT32_MAX);
            uint32_t x = weights[j];
            if (bin) {
                fwrite32_little(&x, 1, dcwfile);
            } else {
                fprintf(dcwfile, "%" PRIu32 "\n", x);
            }
        }
        fclose_maybe_compressed(dcwfile, dcwname);
    }

    {
        scwfile = fopen_maybe_compressed(scwname, wmode);
        for(index_t j = skip; j < small_ncols; j++){
	    ASSERT_ALWAYS(weights[j] <= (index_t) UINT32_MAX);
            uint32_t x = weights[j];
            if (bin) {
                fwrite32_little(&x, 1, scwfile);
            } else {
                fprintf(scwfile, "%" PRIu32 "\n", x);
            }
        }
        fclose_maybe_compressed(scwfile, scwname);
    }

    {
        free(smatname);
        free(srwname);
        free(scwname);
    }
    if (skip) {
        free(dmatname);
        free(drwname);
        free(dcwname);
    }

    free(base);
    free(weights);

    return W;
}

/* On input, colweight[c] is the weight of column c (0 for an absent
 * column, small_ncols columns have a positive weight). On output,
 * colweight[c] - 1 is the new index of column c: heavier columns come
 * first, and columns of equal weight keep their order.
 * FOR_DL: writes the ideals file; orig[c] is the original index of column
 * c in the purged file, or orig == NULL if c is the original index. */
void renumber (index_t small_ncols, index_t *colweight, index_t ncols,
               const index_t * orig, const char *idealsfilename);

/* True if the matrix files derived from sparsename get the binary format */
int matrix_name_is_binary(const char * sparsename);

/* Writes the index file: the number of rows, then per row the number of
 * relations and their indices (FOR_DL: with their coefficients). */
void writeIndex(const char *indexname, index_data_t index_data,
                index_t small_nrows);

#endif  /* CADO_MATRIX_OUTPUT_HPP */
