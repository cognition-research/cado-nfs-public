/* replay --- replaying history of merges to build the sparse matrix

Copyright 2008-2019 Francois Morain, Emmanuel Thome, Paul Zimmermann,
          Cyril Bouvier, Pierrick Gaudry

This file is part of CADO-NFS.

CADO-NFS is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2.1 of the License, or (at your option)
any later version.

CADO-NFS is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License
along with CADO-NFS; see the file COPYING.  If not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdint>

#include <algorithm>
#include <future>
#include <string>
#include <vector>

#include "fmt/base.h"

#include "purgedfile.h"
#include "typedefs.h"
#include "filter_config.h"
#include "filter_io.hpp"
#include "gzip.h"
#include "misc.h"
#include "params.hpp"
#include "sparse.h"
#include "stats.h"
#include "timing.h"
#include "verbose.hpp"
#include "portability.h"
#include "macros.h"
#include "matrix_output.hpp"
#include "merge_heap.h"
#include "merge_replay_matrix.h"
#include "runtime_numeric_cast.hpp"
#include "read_purgedfile_in_parallel.hpp"
#include "staged_rows.hpp"

#include "utils_cxx.hpp"
#include "omp_proxy.h"

#define DEBUG 0

// newrows[i] contains a new row formed of old rows that correspond to
// true original relations (and not multirelations).
//
// After computing newrows, we deduce for all old rows the list of newrows
// containing it.

/* Row source for the matrix output: the rows of replay */
struct replay_rows {
    typerow_t ** rows;
    bool empty(index_t i) const { return rows[i] == NULL; }
    template<typename F>
    void for_each(index_t i, F && f) const {
        for (unsigned int j = 1; j <= rowLength(rows, i); j++)
            f(rowFullCell(rows[i], j));
    }
};

// A line is "i i1 ... ik [#j]".
// If i >= 0 then
//     row[i] is to be added to rows i1...ik and destroyed at the end of
//     the process.
//     Works also if i is alone (hence: destroyed row).
// If i < 0 then
//     row[-i-1] is to be added to rows i1...ik and NOT destroyed.
//
// If given, j is the index of the column used for pivoting (used in DL).
struct his_line {
    index_signed_t ind[MERGE_LEVEL_MAX];
    int ni;
    index_t j;

    his_line() = default;
    /* a comment line gives ni == 0 */
    explicit his_line(char const * str) {
        ni = *str == '#' ? 0 : parse_hisfile_line(ind, str, &j);
    }
    bool destroy() const { return ind[0] >= 0; }
    index_t source() const { return destroy() ? ind[0] : -ind[0]-1; }
};

// If newrows=NULL, we only compute the index.
static void
doAllAdds(typerow_t **newrows, his_line const & line, index_data_t index_data)
{
  index_t i0 = line.source();

  for (int k = 1; k < line.ni; k++)
    addRowsUpdateIndex(newrows, index_data, line.ind[k], i0, line.j);

  if (!line.destroy())
    return;

  //destroy initial row!
  if (newrows != NULL)
    {
      heap_destroy_row(newrows[i0]);
      newrows[i0] = NULL;
    }

  if (index_data != NULL) // ie we want an index
    {
      heap_destroy_row(index_data[i0]);
      index_data[i0] = NULL;
    }
}


#define STRLENMAX 2048

// sparsemat is small_nrows x small_ncols
static void
toFlush (const char *sparsename, typerow_t **sparsemat, index_t *colweight,
         index_t ncols, index_t small_nrows, index_t small_ncols, int skip,
         int bin, const char *idealsfilename, unsigned int nsquare_matrices)

{
    unsigned long W;

    printf("Renumbering columns (including sorting w.r.t. weight)\n");
    fflush(stdout);
    renumber (small_ncols, colweight, ncols, NULL, idealsfilename);

    printf ("Sparse submatrix: nrows=%" PRIu64 " ncols=%" PRIu64 "\n",
            (uint64_t) small_nrows, (uint64_t) small_ncols - skip);
    ASSERT_ALWAYS(small_nrows >= small_ncols);

    if (nsquare_matrices)
    {
        ASSERT_ALWAYS(nsquare_matrices <= (small_nrows-small_ncols+1));
        printf ("Will output %u matri%s of size %" PRIu64 "x%" PRIu64 "\n",
                nsquare_matrices, nsquare_matrices == 1 ? "x" : "ces",
                (uint64_t) small_ncols, (uint64_t) small_ncols);
    }

    double tt = seconds();
    double wt = wct_seconds();
    printf("Writing sparse representation to %s\n", sparsename);
    fflush(stdout);
    W = flushSparse(sparsename, replay_rows { sparsemat },
                    nsquare_matrices ? small_ncols : small_nrows,
                    small_ncols, colweight, skip, bin);
    printf("# Writing matrix took %.1lfs (cpu), %.1lfs (wct)\n",
           seconds()-tt, wct_seconds()-wt);
    printf("# Weight of the sparse submatrix: %lu\n", W);
    fflush(stdout);

    if (nsquare_matrices > 1)
    {
        int zip = has_suffix(sparsename, ".gz");
        const char *ext = bin ? (zip ? ".bin.gz" : ".bin")
                              : (zip ? ".txt.gz" : ".txt");
        char istr[] = { '1', '\0' };

        for (unsigned int i = 1; i < nsquare_matrices; i++)
        {
            double tt = seconds();
            double wt = wct_seconds();
            char * smatname = derived_filename(sparsename, istr, ext);
            printf("Writing sparse representation to %s\n", smatname);
            fflush(stdout);
            W = flushSparse(smatname, replay_rows { sparsemat+i }, small_ncols, small_ncols,
                            colweight, skip, bin);
            istr[0]++;
            printf("# Writing matrix took %.1lfs (cpu), %.1lfs (wct)\n",
                   seconds()-tt, wct_seconds()-wt);
            printf("# Weight of the sparse submatrix: %lu\n", W);
            fflush(stdout);
            free(smatname);
        }
    }
}

/* A merge is a group of consecutive history lines. Its last line destroys
 * the source row of the merge. merge writes the history pass by pass, and
 * the merges of one pass touch disjoint rows.
 *
 * his_reader reads the history in blocks of whole merges. One thread
 * reads the next block from the file while the other threads parse the
 * lines of the current block in parallel. The reader then groups the
 * parsed lines merge by merge into batches. All merges of one batch
 * touch disjoint rows, so replay can apply the merges of one batch in
 * parallel. A batch ends when the next merge touches a row that is busy
 * in the batch, at max_lines lines, or at the end of the block. */
struct his_reader {
    static constexpr size_t max_lines = 1 << 18;
    static constexpr size_t block_bytes = size_t(64) << 20;
    FILE * hisfile;
    index_t nrows;
    index_t Nmax;
    uint64_t addread = 0;
    stats_data_t stats; /* struct for printing progress */
    std::vector<char> block, next_block; /* whole merges of the history */
    std::future<void> prefetch; /* the read of next_block */
    std::vector<his_line> parsed; /* the lines of block, comments dropped */
    size_t pend = 0; /* parsed[pend, next) is the merge read last, not yet in a batch */
    size_t next = 0; /* the first line of parsed not yet read */
    std::vector<uint8_t> busy; /* busy[i] != 0 if row i is in the batch */
    std::vector<index_t> busy_rows;

    his_reader(FILE * hisfile, index_t nrows, index_t Nmax)
        : hisfile(hisfile), nrows(nrows), Nmax(Nmax), busy(nrows, 0)
    {
        /* will print report at 2^10, 2^11, ... 2^23 computed primes and every
         * 2^23 primes after that */
        stats_init (stats, stdout, &addread, 23, "Read", "row additions", "", "lines");
        start_prefetch();
    }
    ~his_reader() { prefetch.wait(); }

    /* Fills buf with the next block_bytes of the history, extended to the
     * end of its last merge. An empty buf is the end of the history. */
    void read_block(std::vector<char> & buf)
    {
        buf.resize(block_bytes);
        buf.resize(fread(buf.data(), 1, block_bytes, hisfile));
        if (buf.empty())
            return;
        char const * nl = (char const *) memrchr(buf.data(), '\n', buf.size());
        size_t line = nl ? nl + 1 - buf.data() : 0;
        for (;;) {
            /* read the first byte of the line at buf[line] if it is not in */
            if (line == buf.size()) {
                int c = getc(hisfile);
                if (c == EOF)
                    return;
                buf.push_back(c);
            }
            /* complete the line */
            while (buf.back() != '\n') {
                int c = getc(hisfile);
                if (c == EOF) {
                    buf.push_back('\n');
                    return;
                }
                buf.push_back(c);
            }
            bool destroy = buf[line] != '-' && buf[line] != '#';
            if (destroy)
                return;
            line = buf.size();
        }
    }

    void start_prefetch()
    {
        prefetch = std::async(std::launch::async,
                              [this]() { read_block(next_block); });
    }

    /* Parses the next block into parsed. Returns false at the end of the
     * history. */
    bool parse_block()
    {
        prefetch.get();
        block.swap(next_block);
        start_prefetch();
        if (block.empty())
            return false;
        std::vector<char const *> lines;
        char const * end = block.data() + block.size();
        for (char const * p = block.data(); p < end;
             p = (char const *) memchr(p, '\n', end - p) + 1)
            lines.push_back(p);
        parsed.resize(lines.size());
#pragma omp parallel for schedule(static)
        for (size_t l = 0; l < lines.size(); l++)
            parsed[l] = his_line(lines[l]);
        auto comment = [](his_line const & l) { return l.ni == 0; };
        parsed.erase(std::remove_if(parsed.begin(), parsed.end(), comment),
                     parsed.end());
        pend = next = 0;
        return true;
    }

    /* Reads one merge of the block into parsed[pend, next). Returns false
     * when the number of rows is below Nmax. */
    bool read_merge()
    {
        while (next < parsed.size() && nrows >= Nmax)
        {
            his_line const & line = parsed[next++];
            addread++;

            if (stats_test_progress(stats))
                stats_print_progress (stats, addread, 0, 0, 0);

            /* a merge that destroys its source row removes one row */
            nrows -= line.destroy();
            if (line.destroy())
                return true;
        }
        return pend != next;
    }

    template<typename F> void for_each_pending_row(F f) const
    {
        for (size_t l = pend; l < next; l++) {
            f(parsed[l].source());
            for (int k = 1; k < parsed[l].ni; k++)
                f((index_t) parsed[l].ind[k]);
        }
    }

    /* Fills lines with the next merges that touch disjoint rows. The
     * pointers are valid until the next call. merge_start[m] is the first
     * line of merge m; the last entry is lines.size(). Returns false when
     * there is no merge left. */
    bool next_batch(std::vector<his_line const *> & lines,
                    std::vector<size_t> & merge_start)
    {
        lines.clear();
        merge_start.clear();
        for (index_t i : busy_rows) busy[i] = 0;
        busy_rows.clear();
        while (lines.size() < max_lines) {
            if (pend == next) {
                if (next == parsed.size()) {
                    /* the next block replaces parsed: end the batch here */
                    if (!lines.empty()) break;
                    if (!parse_block()) break;
                }
                if (!read_merge()) break;
            }
            bool conflict = false;
            for_each_pending_row([&](index_t i) { conflict |= busy[i]; });
            if (conflict) break;
            for_each_pending_row([&](index_t i) {
                    busy[i] = 1;
                    busy_rows.push_back(i);
                    });
            merge_start.push_back(lines.size());
            for (size_t l = pend; l < next; l++)
                lines.push_back(&parsed[l]);
            pend = next;
        }
        merge_start.push_back(lines.size());
        return !lines.empty();
    }
};

/* If newrows = NULL, only compute the index */
static void
build_newrows_from_file(typerow_t **newrows, FILE *hisfile,
                        index_data_t index_data, index_t nrows, index_t Nmax)
{
    printf("Reading row additions\n");
    fflush(stdout);
    his_reader reader(hisfile, nrows, Nmax);
    std::vector<his_line const *> lines;
    std::vector<size_t> merge_start;
    double t_read = 0, t_add = 0, t_gc = 0;
    size_t nbatches = 0, nlines = 0;
    auto lap = [t = wct_seconds()](double & acc) mutable {
        double now = wct_seconds(); acc += now - t; t = now;
    };
    while (reader.next_batch(lines, merge_start)) {
        lap(t_read);
        nbatches++;
        nlines += lines.size();
        size_t nmerges = merge_start.size() - 1;
#pragma omp parallel for schedule(dynamic, 16)
        for (size_t m = 0; m < nmerges; m++)
            for (size_t l = merge_start[m]; l < merge_start[m+1]; l++)
                doAllAdds(newrows, *lines[l], index_data);
        lap(t_add);
        if (heap_waste_ratio() > 0.25)
            heap_garbage_collection(newrows, index_data);
        lap(t_gc);
    }
    stats_print_progress (reader.stats, reader.addread, 0, 0, 1);
    printf("Row additions: read %.1fs, add %.1fs, collect %.1fs (wct)\n",
           t_read, t_add, t_gc);
    printf("Row additions: %zu batches of %.0f lines on average\n",
           nbatches, nbatches ? (double) nlines / nbatches : 0.0);
}

/* TODO: refactor with purged_file_reader in merge.cpp !!!
 */
struct replay_read_data {
    typerow_t **mat = nullptr; // not owned
#ifndef FOR_DL
    staged_rows staged;
#endif
    parameter<std::string,
        "purged",
        "input purged file">
            purgedname;
    parameter_with_default<index_t,
        "col0",
        "print only columns with index >= col0",
        "0">
            col0;
    parameter_with_default<index_t,
        "colmax",
        "print only columns with index < colmax",
        "0">
            colmax;
    size_t nrows = 0;
    index_t ncols = 0;

    static void configure(cxx_param_list & pl) {
        decltype(purgedname)::configure(pl);
        decltype(col0)::configure(pl);
        decltype(colmax)::configure(pl);
    }

    void fetch_nrows_ncols() {// {{{
        uint64_t nr, nc;
        /* Read number of rows and cols on first line of purged file */
        purgedfile_read_firstline (purgedname().c_str(), &nr, &nc);
        if (nr>= UINT32_MAX)
        {
            fprintf (stderr, "Error, cannot handle 2^32 rows or more after purge\n");
            fprintf (stderr, "change ind_row from uint32_t to uint64_t in sparse.h\n");
            exit (EXIT_FAILURE);
        }
#if SIZEOF_INDEX == 4
        if (nc>= UINT32_MAX) {
            fprintf(stderr, "You must recompile with -DSIZEOF_INDEX=8\n");
            exit(EXIT_FAILURE);
        }
#endif
        nrows = nr;
        ncols = runtime_numeric_cast<index_t>(nc);
        fmt::print("Sparse matrix has {} rows and {} cols\n",
                nrows, ncols);
        fflush(stdout);
    }// }}}

    explicit replay_read_data (cxx_param_list & pl)
        : purgedname(pl)
        , col0(pl)
        , colmax(pl)
    {
        fetch_nrows_ncols();
        if (colmax() == 0)
            colmax() = std::numeric_limits<index_t>::max();
    }

#ifdef FOR_DL
    template<typename relation_type>
    void fill_in_rows (relation_type & rel)
    {
        typerow_t buf[UMAX(weight_t)];

        unsigned int nb = 0;
        for (auto const & pe : rel.primes) {
            index_t h = pe.h;
            if (h < col0 || h >= colmax) continue;
            nb++;
            exponent_t e = pe.e;
            buf[nb] = (ideal_merge_t) {.id = h, .e = e};
            ASSERT (h < ncols);
        }
        buf[0].id = nb;

        qsort (&(buf[1]), nb, sizeof(typerow_t), cmp_typerow_t);

        mat[rel.num] = heap_alloc_row (rel.num, nb);
        compressRow (mat[rel.num], buf, nb);
    }

    void build_rows() {}
#else
    template<typename relation_type>
    void fill_in_rows (relation_type & rel)
    {
        index_t buf[UMAX(weight_t)];

        unsigned int nb = 0;
        for (auto const & pe : rel.primes) {
            index_t h = pe.h;
            if (h < col0 || h >= colmax) continue;
            ASSERT_ALWAYS (pe.e == 1);
            ASSERT (h < ncols);
            buf[nb++] = h;
        }
        std::sort(buf, buf + nb);
        staged.add(rel.num, buf, nb);
    }

    void build_rows() {
        ncols = ::build_rows(staged, ncols, mat);
        fmt::print("{} columns appear in the matrix\n", ncols);
        fflush(stdout);
    }
#endif

    size_t read_serial() {
        using R = cado::relation_building_blocks::primes_block<
            prime_type_for_indexed_relations,
            cado::relation_building_blocks::ab_ignore<16>>;
        return filter_rels<R>(purgedname, nullptr, nullptr,
                [&](R & rel) { fill_in_rows(rel); });
    }

#ifdef FOR_DL
    size_t read_rows() { return read_serial(); }
#else
    size_t read_rows() {
        if (!purgedfile_is_seekable(purgedname().c_str()))
            return read_serial();
        return read_purgedfile_in_parallel(purgedname().c_str(), col0,
                colmax, NULL, staged);
    }
#endif

    size_t read(typerow_t ** newrows)
    {
        mat = newrows;
        fmt::print("Reading sparse matrix from {}\n", purgedname());
        fflush(stdout);
        size_t nread = read_rows();
        ASSERT_ALWAYS (nread == nrows);
        build_rows();
        return nread;
    }

    void read_for_msieve(typerow_t ** newrows) {
        mat = newrows;
        /* to generate the .cyc file for msieve, we only need to start from the
           identity matrix with newnrows relation-sets, where relation-set i
           contains only relation i. Thus we only need to read the first line of
           the purged file, to get the number of relations-sets. */

        for (index_t i = 0; i < nrows; i++) {
            mat[i] = heap_alloc_row(i, 1);
            setCell(mat[i], 1, i, 1);
        }
    }
};

/* if for_msieve=1, generate the *.cyc file needed by msieve to construct
   its matrix, which is of the following (binary) format:
      small_nrows
      n1 i1 i2 ... in1
      n2 j1 j2 ... jn2
      ...
      nk ...
   where each value is stored as a 32-bit integer (no linebreak),
   small_nrows is the number of relation-sets of the matrix,
   n1 is the number of relations in the first relation-set,
   i1 is the index of the first relation in the first relation-set
   (should correspond to line i1+2 in *.purged.gz), and so on */


static void
generate_cyc (const char *outname, typerow_t **rows, index_t nrows)
{
    auto outfile = fopen_helper(outname, "w");

    /* msieve's .cyc format is apparently 32-bit only */

    uint32_t t = runtime_numeric_cast<uint32_t>(nrows);

    /* first write the number of relations */
    fwrite (&t, sizeof(uint32_t), 1, outfile.get());

    /* then for each relation-set write a 32-bit integer giving the number k
       of its element, followed by those k elements */
    for (index_t i = 0; i < nrows; i++)
    {
        t = runtime_numeric_cast<uint32_t>(rowLength(rows, i));
        ASSERT_ALWAYS(t != 0);
        fwrite (&t, sizeof(uint32_t), 1, outfile.get());
        for (index_t k = 1; k <= t; k++)
        {
            auto u = runtime_numeric_cast<uint32_t>(rowCell(rows[i], k));
            fwrite (&u, sizeof(uint32_t), 1, outfile.get());
        }
    }
}

static void
fasterVersion (typerow_t **newrows, const char *sparsename,
               const char *indexname, const char *hisname, index_t nrows,
               index_t ncols, int skip, int bin, const char *idealsfilename,
               int for_msieve, index_t Nmax, unsigned int nsquare_matrices)
{
  FILE *hisfile = NULL;
  index_t *colweight = NULL;
  index_t small_nrows = 0, small_ncols;
  index_data_t index_data = NULL;
  auto phase = [t0 = wct_seconds()](char const * what) {
      printf("# %s at %.1fs (wct)\n", what, wct_seconds() - t0);
      fflush(stdout);
  };

  hisfile = fopen_maybe_compressed (hisname, "r");
  ASSERT_ALWAYS(hisfile != NULL);

  if (indexname != NULL)
  {
    /* At the beginning, the index_data consists of relsets that are just single
     * relations.*/
    index_data = (relset_t *) malloc (nrows * sizeof(relset_t));
    ASSERT_ALWAYS (index_data != NULL);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < nrows; ++i)
      index_data[i] = relset_singleton(i);
  }
  else
    index_data = NULL;

  /* read merges in the *.merge.his file and replay them */
  phase("row additions start");
  build_newrows_from_file (newrows, hisfile, index_data, nrows, Nmax);
  phase("row additions done");

  if (sparsename != NULL)
    {
      /* crunch empty rows first to save memory and compute small_nrows */
      index_t j = 0;
      for (index_t i = 0; i < nrows; i++)
        if (newrows[i] != NULL)
          newrows[j++] = newrows[i]; /* we always have j <= i */
      small_nrows = j;
      checked_realloc(newrows, small_nrows);
    }

  /* if index was asked: crunch the empty rows as above, create the index and
   * free index_data before calling toFlush(), in order to decrease the total
   * memory usage */
  if (indexname != NULL)
  {
    index_t j = 0;
    for (index_t i = 0; i < nrows; i++)
    {
      if (index_data[i] != NULL)
        index_data[j++] = index_data[i];
    }
    if (sparsename != NULL)
      ASSERT_ALWAYS (j == small_nrows);
    else
      small_nrows = j;

    phase("index write start");
    writeIndex (indexname, index_data, small_nrows);
    phase("index write done");

    free (index_data);
    phase("index freed");
  }

  if (sparsename == NULL)
    goto close_and_exit;

  /* compute column weights */
  colweight = (index_t*) malloc (ncols * sizeof(index_t));
  ASSERT_ALWAYS (colweight != NULL);
  memset (colweight, 0, ncols * sizeof(index_t));
#pragma omp parallel for schedule(dynamic, 1024)
  for (index_t i = 0; i < small_nrows; i++)
    for(unsigned int k = 1; k <= rowLength(newrows, i); k++)
    {
      index_t j = rowCell(newrows[i], k);
#pragma omp atomic
      colweight[j] ++;
    }
  /* small_ncols is the number of columns with non-empty weight,
     i.e., the number of columns of the final matrix */
  small_ncols = 0;
#pragma omp parallel for reduction(+:small_ncols)
  for (index_t j = 0; j < ncols; j++)
    small_ncols += (colweight[j] != 0);

#if defined FOR_DL && defined STAT_DL
  index_t count[11] = {0,0,0,0,0,0,0,0,0,0,0};
  index_t nonzero = 0;
  for (index_t i = 0; i < small_nrows ; i++)
    {
      for(unsigned int k = 1; k <= rowLength(newrows, i); k++)
        {
          if (abs(newrows[i][k].e) > 10)
            count[0]++;
          else
            count[abs(newrows[i][k].e)]++;
          nonzero++;
        }
    }
  fprintf (stderr, "# of non zero coeff: %lu\n", nonzero);
  for (int i = 1; i <= 10 ; i++)
    fprintf (stderr, "# of %d: %lu(%.2f%%)\n", i, count[i],
                     100 * (double) count[i]/nonzero);
  fprintf (stderr, "# of > 10: %lu(%.2f%%)\n", count[0],
                   100 * (double) count[0]/nonzero);
#endif

  phase("column weights done");
  if (for_msieve)
    {
      /* generate the <dat_file_name>.cyc file in "indexname" */
      if (skip != 0)
        {
          fprintf (stderr, "Error, skip should be 0 with --for_msieve\n");
          exit (1);
        }
      if (bin != 0)
        {
          fprintf (stderr, "Error, --binary incompatible with --for_msieve\n");
          exit (1);
        }
      generate_cyc (sparsename, newrows, small_nrows);
    }
  else if (sparsename != NULL)
    /* renumber columns after sorting them by decreasing weight */
    toFlush (sparsename, newrows, colweight, ncols, small_nrows, small_ncols,
             skip, bin, idealsfilename, nsquare_matrices);

  phase("matrix written");
  /* Free */
  free (colweight);
  heap_clear ();
  free (newrows);
  phase("rows freed");

 close_and_exit:
  fclose_maybe_compressed (hisfile, hisname);
}

static void declare_usage(cxx_param_list & pl)
{
  pl.declare_usage("his", "input history file");
  pl.declare_usage("out", "basename for output matrices");
#ifndef FOR_DL
  pl.declare_usage("skip", "number of heaviest columns that go to the "
                            "dense matrix (default " CADO_STRINGIZE(DEFAULT_MERGE_SKIP) ")");
#endif
  pl.declare_usage("index", "file containing description of rows "
                                     "(relations-sets) of the matrix");
  pl.declare_usage("ideals", "file containing correspondence between "
                                      "ideals and matrix columns");
  pl.declare_usage("path_antebuffer", "path to antebuffer program");
  pl.declare_usage("for_msieve", "output matrix in msieve format");
  pl.declare_usage("Nmax", "stop at Nmax number of rows (default 0)");

  pl.declare_usage("nsquare_matrices", "if non-zero, output that many "
                                "square matrices by skipping rows (default 0)");
  pl.declare_usage("t", "number of threads (default 1)");

  verbose_decl_usage(pl);
}

// We start from M_purged which is nrows x ncols;
// we build M_small which is small_nrows x small_ncols.
// newrows[i] if != NULL, contains a list of the indices of the rows in
// M_purged that were added together to form this new row in M_small.
// TODO: replace this index by the index to rels directly to skip one
// indirection???
int main(int argc, char const * argv[])
{
    const char * argv0 = argv[0];
    uint64_t Nmax = 0;
    typerow_t **newrows;
    int bin = -1, skip = DEFAULT_MERGE_SKIP, for_msieve = 0;
    int nthreads = 1;
    double cpu0 = seconds ();
    double wct0 = wct_seconds ();

    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);

    cxx_param_list pl;
    declare_usage(pl);

    replay_read_data::configure(pl);

    pl.configure_switch_old("for_msieve", &for_msieve);
    cado::filter_io_details::configure(pl);

    pl.process_command_line(argc, argv, false);

    verbose_interpret_parameters(pl);
    cado::filter_io_details::interpret_parameters(pl);
    pl.print_command_line(stdout);
    fflush(stdout);

    const char * hisname = pl.lookup_old("his");
    const char * sparsename = pl.lookup_old("out");
    const char * indexname = pl.lookup_old("index");
    const char * idealsfilename = pl.lookup_old("ideals");
#ifndef FOR_DL
    pl.parse("skip", skip);
#endif
    pl.parse("Nmax", Nmax);
    pl.parse("t", nthreads);
    ASSERT_ALWAYS(nthreads >= 1);
    omp_set_num_threads(nthreads);
    heap_setup();
    const char * path_antebuffer = pl.lookup_old("path_antebuffer");


    unsigned int nsquare_matrices = 0;
    pl.parse("nsquare_matrices", nsquare_matrices);
    if (nsquare_matrices)
    {
        printf ("# Will output %u square matri%s\n", nsquare_matrices,
                nsquare_matrices == 1 ? "x" : "ces" );
        ASSERT_ALWAYS (skip == 0); /* not compatible with skip */
        ASSERT_ALWAYS (nsquare_matrices < 10);
    }

    replay_read_data rr(pl);

    /* Some checks on command line arguments */
    if (pl.warn_unused())
        pl.fail("Error, unused parameters are given\n");

    if (hisname == NULL)
        pl.fail("Error, missing -his command line argument\n");
    if (sparsename == NULL && indexname == NULL)
        pl.fail("Error, at least one of -out and -index is required\n");
#ifdef FOR_DL
    if (idealsfilename == NULL)
        pl.fail("Error, missing -ideals command line argument\n");
    ASSERT_ALWAYS (skip == 0);
#endif
    if (sparsename != NULL)
    {
        bin = matrix_name_is_binary(sparsename);
        printf ("# Output matrices will be written in %s format\n",
                bin ? "binary" : "text");
    }

    set_antebuffer_path (argv0, path_antebuffer);


    /* Allocate memory for rows of the matrix */
    if (sparsename != NULL)
    {
        newrows = (typerow_t **) malloc (rr.nrows * sizeof(typerow_t *));
        ASSERT_ALWAYS(newrows != NULL);

        if (for_msieve)
            rr.read_for_msieve(newrows);
        else
            rr.read(newrows);
#if DEBUG >=1
        for (index_t i = 0; i < nrows; i++)
        {
            fprintf(stderr, "row[%" PRIu64 "] :", i);
            fprintRow(stderr, newrows[i]);
            fprintf(stderr, "\n");
        }
#endif
    } else {
        newrows = NULL;
    }


    fasterVersion (newrows, sparsename, indexname, hisname,
            rr.nrows, rr.ncols, skip,
            bin, idealsfilename, for_msieve, Nmax, nsquare_matrices);

    print_timing_and_memory (stdout, cpu0, wct0);
    return 0;
}
