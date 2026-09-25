#include "cado.h" // IWYU pragma: keep

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "filter_config.h"
#include "gzip.h"
#include "macros.h"
#include "merge_heap.h"
#include "merge_replay_matrix.h"
#include "omp_proxy.h"
#include "read_purgedfile_in_parallel.hpp"
#include "sparse.h"
#include "staged_rows.hpp"
#include "timing.h"
#include "typedefs.h"

#include "utils_cxx.hpp"

uint64_t * rows_per_thread;
off_t * spos_tab;
size_t global_bytes = 0;
size_t global_next_report = 1024;
double tt0;
size_t global_nrows = 0;
size_t global_nthreads;

void global_init(int nthreads, off_t endpos)
{
    global_nthreads = nthreads;
    rows_per_thread = (uint64_t *) malloc(nthreads * sizeof(uint64_t));
    tt0 = wct_seconds();
    spos_tab = (off_t *) malloc((nthreads + 1) * sizeof(off_t));

    for (int i = 0; i < nthreads; i++)
        spos_tab[i] = (endpos * i) / nthreads;
    spos_tab[nthreads] = endpos;
}

void global_clear()
{
    free(rows_per_thread);
    free(spos_tab);
}

void global_print_report() {
    double dt = wct_seconds() - tt0;
    printf
        ("# Read %zu relations in %.1fs -- %.1f MB/s -- %.1f rels/s\n",
         global_nrows, dt, (global_bytes >> 20) / dt,
         global_nrows / dt);
    fflush(stdout);

    global_next_report *= 2;
}

static inline char * hacked_strtoul16(index_t * px, char * p)/*{{{*/
{
    /* functionally equivalent to:
         char * q;
         * px = strtoul(p, &q, 16);
         return q;
     */
    static const unsigned char ugly[256] = {
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        0,   1,   2,   3,   4,   5,   6,   7,  
        8,   9,   255, 255, 255, 255, 255, 255,
        255, 10,  11,  12,  13,  14,  15,  255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 10,  11,  12,  13,  14,  15,  255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255 
    };
    index_t x = 0;
    for(int c ; (c = ugly[(int) (unsigned char) *p++]) != 255 ; x = x*16+c) ;
    p--;
    *px = x;
    return p;
}
/*}}}*/

/* The rows read by one thread: staged with their original columns
   (factorization), or in the heap. */
#ifndef FOR_DL
typedef staged_rows local_rows;
#else
typedef std::vector<typerow_t *> local_rows;
#endif

/* Parses one line of the purged file into its sorted columns in [col0,
   colmax). The columns below col0 go to the mask of the row in buried
   instead (buried != NULL requires col0 <= 64). */
static void parse_row(char * line, index_t col0, index_t colmax,
                      std::vector<index_t> & cols, std::vector<uint64_t> * buried)
{
    cols.clear();
    uint64_t buried_mask = 0;
    char * p = line;
    char * z = p + strlen(line) - 1; /* the EOL delimiter */
    for( ; *p && *p != ':' ; p++);
    for( ; p++ != z ; ) {
        index_t x;
        p = hacked_strtoul16(&x, p);
        if (x < col0) {
            if (buried) buried_mask ^= (uint64_t) 1 << x;
            continue;
        }
        if (x >= colmax) continue;
        cols.push_back(x);
    }
    std::sort(cols.begin(), cols.end());
    if (buried) buried->push_back(buried_mask);
}

#ifndef FOR_DL
/* A column that appears an even number of times cancels. */
static void store_row(local_rows & V, size_t i, std::vector<index_t> & cols)
{
    size_t jt = 0;
    for (size_t it = 0; it < cols.size(); ) {
        size_t kt = it + 1;
        for (; kt < cols.size() && cols[kt] == cols[it]; kt++);
        cols[jt] = cols[it];
        jt += (kt - it) & 1;
        it = kt;
    }
    V.add(i, cols.data(), jt);
}
#else
/* One cell per distinct column, with the multiplicity as exponent. The
   row index is written later, when the rows of all threads are counted. */
static void store_row(local_rows & V, size_t, std::vector<index_t> & cols)
{
    std::vector<typerow_t> buf(1);
    for (size_t it = 0; it < cols.size(); ) {
        size_t kt = it + 1;
        for (; kt < cols.size() && cols[kt] == cols[it]; kt++);
        typerow_t x;
        setCell(&x, 0, cols[it], (exponent_t) (kt - it));
        buf.push_back(x);
        it = kt;
    }
    index_t const z = buf.size() - 1;
    setCell(buf.data(), 0, z, 0);
    typerow_t *newrow = heap_alloc_row(0, z);
    compressRow(newrow, buf.data(), z);
    V.push_back(newrow);
}
#endif

/* Returns the number of rows read. buried != NULL: gets one mask per row
   with the columns below col0 */
static size_t read_local_rows(local_rows & V, std::vector<uint64_t> * buried,
                              FILE * fi, off_t bytes_to_read, index_t col0, index_t colmax)
{
    size_t local_next_report = 256;
    size_t local_nrows_at_last_report = 0;
    size_t local_bytes_at_last_report = 0;
    size_t nrows = 0;

    char line[4096];
    std::vector<index_t> cols;

    long start = ftell(fi);

    for (long pos ; ((pos = ftell(fi)) - start)  < bytes_to_read ; ) {
        if (fgets(line, sizeof(line), fi) == NULL) break;
        if (line[0] == '#') continue;
        /* otherwise 4096 is not enough ! */
        ASSERT_ALWAYS(strlen(line) < (sizeof(line) - 1));

        parse_row(line, col0, colmax, cols, buried);
        store_row(V, nrows++, cols);

        /* At this point we should consider reporting. */
        size_t local_bytes = pos - start;
        if (nrows >= local_next_report) {
#pragma omp critical
            {
                global_nrows += nrows - local_nrows_at_last_report;
                global_bytes += local_bytes - local_bytes_at_last_report;
                local_nrows_at_last_report = nrows;
                local_bytes_at_last_report = local_bytes;
                if (global_nrows >= global_next_report)
                    global_print_report();
                local_next_report += global_next_report / 2 / global_nthreads;
            }
        }
    }
#pragma omp critical
    {
        global_nrows += nrows - local_nrows_at_last_report;
        size_t local_bytes = ftell(fi) - start;
        global_bytes += local_bytes - local_bytes_at_last_report;
    }
    return nrows;
}

int purgedfile_is_seekable(const char * filename)
{
    FILE * f = fopen_maybe_compressed(filename, "r");
    ASSERT_ALWAYS(f != NULL);
    int t = fseek(f, 0, SEEK_END) == 0;
    fclose_maybe_compressed(f, filename);
    return t;
}

#ifndef FOR_DL
uint64_t read_purgedfile_in_parallel(const char * filename, index_t col0,
                                     index_t colmax, uint64_t * buried,
                                     staged_rows & staged)
#else
uint64_t read_purgedfile_in_parallel(filter_matrix_t * mat,
				     const char *filename, uint64_t * buried)
#endif
{
#ifdef FOR_DL
    index_t col0 = mat->skip;
    index_t colmax = std::numeric_limits<index_t>::max();
#endif
    ASSERT_ALWAYS(buried == NULL || col0 <= 64);
    off_t endpos;

    {
	FILE * f = fopen(filename, "r");
        DIE_ERRNO_DIAG(f == NULL, "fopen(%s)", filename);
	int rc = fseek(f, 0, SEEK_END);
        DIE_ERRNO_DIAG(rc < 0, "fseek(%s)", filename);
        endpos = ftell(f);
        fclose(f);
    }

    /* Find accurate starting positions for everyone */
    unsigned int nthreads = omp_get_max_threads();

    /* cap the number of I/O threads */
    if (nthreads > MAX_IO_THREADS)
	nthreads = MAX_IO_THREADS;

    fprintf(stderr, "# %s: Doing I/O with %u threads\n", filename,
		 nthreads);

    global_init(nthreads, endpos);

    /* All threads get their private reading head. */
#pragma omp parallel num_threads(nthreads)
    {
        int i = omp_get_thread_num();
        FILE * fi;
        char buffer[1 << 16];

        fi = fopen(filename, "r");
        DIE_ERRNO_DIAG(fi == NULL, "fopen(%s)", filename);

        setbuffer(fi, buffer, sizeof(buffer));

	int rc = fseek(fi, spos_tab[i], SEEK_SET);
        DIE_ERRNO_DIAG(rc < 0, "fseek(%s)", filename);

        /* Except when we're at the beginning of the stream, read until
         * we get a newline */
        for (; i > 0 && fgetc(fi) != '\n';);
        spos_tab[i] = ftell(fi);

#pragma omp barrier

        off_t bytes_to_read = spos_tab[i + 1] - spos_tab[i];
        local_rows V;
        std::vector<uint64_t> local_buried;
        rows_per_thread[i] = read_local_rows(V, buried ? &local_buried : NULL, fi, bytes_to_read, col0, colmax);

#pragma omp barrier

        /* now we want to collect the number of rows for everyone,
         * renumber all relations, and store them in order in mat->rows.
         */
        uint64_t index = 0;
        for(int j = 0 ; j < i ; j++)
            index += rows_per_thread[j];
        if (buried) std::copy(local_buried.begin(), local_buried.end(), buried + index);
#ifndef FOR_DL
#pragma omp critical
        staged.append(V, index);
#else
        for (typerow_t * r : V) {
            rowCell((r - 1), 0) = index;
            mat->rows[index++] = r;
        }
#endif

        fclose(fi);
    }
    global_print_report();
    global_clear();

    return global_nrows;
}


/* BAD IDEAS FOR PARSING LOOP */

/* In the critical parsing loop, there are several ways to parse the
 * string s and make a relation out of it.  The whole point of reading in
 * parallel is that we're going to have several threads that keep the I/O
 * layer happy, so that what happens CPU-bound shouldn't be *that*
 * important.
 *
 * However, we do care about lock contention. And apparently, there are
 * several bad ideas that one can come up with, which eventually destroy
 * performance.
 */

#if 0
    /* The #1 outrageously bad idea is this one.  Doing _just_ this
     * initialization of an istringstream, and nothing else, divides the
     * 16-thread reading performance by about 10 (from ~ 900MB/s to
     * 90MB/s from beegfs, with really nothing beyond progress printing
     * down the line.
     * */
    std::istringstream ss(s);
    /* This is another terrible idea, with catastrophic impact on
     * performance. On the same metric as above, we drop from 90MB/s to
     * 20MB/s.
     */
    ss.imbue(std::locale(std::locale(), new csv_reader()));
#if 0   /* corresponding companion structure */
        // https://stackoverflow.com/questions/1894886/parsing-a-comma-delimited-stdstring
        struct csv_reader: std::ctype<char> {
            csv_reader(): std::ctype<char>(get_table()) {}
            static std::ctype_base::mask const* get_table() {
                static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());

                rc[','] = std::ctype_base::space;
                rc['\n'] = std::ctype_base::space;
                rc[' '] = std::ctype_base::space;
                rc[':'] = std::ctype_base::space;
                return &rc[0];
            }
        }; 
#endif

    /* This first step of parsing over a,b doesn't degrade performance
     * too much.  */
    ss >> std::hex;
    {
        std::string a,b;
        ss >> a >> b;
    }
    /* this for loop, however, seems to tax the CPU somewhat (so yet
     * another small degradation, but we've already reached a pretty
     * degraded situation at this point).  I also see many futex calls, I
     * suspect that some libstc++ locking is at stake here. Maybe it's a
     * stupid thing with reference counting of a locale that goes with a
     * shared ptr, or something like this.
     */
    for( ; ss.good() ; ) {
        index_t x;
        ss >> x;
        typerow_t xx;
        setCell(&xx, 0, x, 1);
        primes.push_back(xx);
    }
#endif

#if 0
    /* it is also worth noting that hacked_strtoul16 is moderately, but
     * still measurably slower than its strtoul-based equivalent
     */
#endif

#if 0
/* We also see many lseek(fd, 0, SEEK_CUR) in strace that correspond to
 * our tellg() calls. We tried to kill them as follows, but it isn't
 * great, and raises double buffering questions that I don't really want
 * to go into.
 */
class countbuf : public std::streambuf {
    std::streambuf & sbuf;
    std::streamsize size = 0;
    char ch;
public:
    countbuf(std::streambuf & sbuf): sbuf(sbuf) {}
protected:
    virtual int underflow()
    {
        size++;
        ch = sbuf.sbumpc();
        setg(&ch, &ch, &ch+1);
        return ch;
    }
public:
    std::streamsize count() { return this->size; }
};
// this would be used as follows, the ifstream fi being replaced by fi2
// as follows.
        // countbuf cc(*fi.rdbuf());
        // std::istream fi2(&cc);

#endif

