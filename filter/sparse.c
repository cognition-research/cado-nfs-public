/*
 * Program: history
 * Author : F. Morain
 * Purpose: managing history of merges
 *
 * Algorithm:
 *
 */

#include "cado.h" // IWYU pragma: keep

#include <stdint.h>
#ifdef FOR_DL
#include <inttypes.h> // for PRId32
#endif
#include <stdio.h>
#include <stdlib.h>

#include "macros.h"
#ifdef FOR_DL
#include "gcd.h" // gcd_uint64
#endif
#include "merge_heap.h"
#include "merge_replay_matrix.h"
#include "sparse.h"
#include "typedefs.h"

#define DEBUG 0

void fprintRow(FILE * file, typerow_t * row)
{
    index_t i;
#ifdef FOR_DL
    fprintf(file, "[%" PRid "]", row[0].id);
    for (i = 1; i <= row[0].id; i++)
        fprintf(file, " %" PRid "(%" PRId32 ")", row[i].id, row[i].e);
#else
    fprintf(file, "[%" PRid "]", (index_t) rowCell(row, 0));
    for (i = 1; i <= rowCell(row, 0); i++)
        fprintf(file, " %" PRid "", (index_t) rowCell(row, i));
#endif
}

static multirel_t
singleton_rel(index_t i)
{
    multirel_t r;
    r.ind_row = i;
#ifdef FOR_DL
    r.e = 1;
#endif
    return r;
}

/* heap cells that hold n relations after the count cell */
static size_t relset_cells(uint32_t n)
{
    return 1 + (n * sizeof(multirel_t) + sizeof(typerow_t) - 1) / sizeof(typerow_t);
}

static relset_t relset_alloc(index_t i, uint32_t n)
{
    relset_t r = heap_alloc_relset(i, relset_cells(n));
    rowCell(r, 1) = n;
    return r;
}

relset_t relset_singleton(index_t i)
{
    relset_t r = relset_alloc(i, 1);
    relset_rels(r)[0] = singleton_rel(i);
    return r;
}

/* view of a relation set; a NULL set holds the row alone, in *one */
struct relset_view { uint32_t n; multirel_t * rels; };

static struct relset_view view(relset_t r, multirel_t * one)
{
    if (r == NULL)
        return (struct relset_view) { 1, one };
    return (struct relset_view) { relset_n(r), relset_rels(r) };
}

/* The sum is built in this per-thread buffer, then copied to a heap row of
   its exact size. */
static __thread multirel_t * sum_buf = NULL;
static __thread uint32_t sum_buf_n = 0;

void addRelsets(index_data_t index_data, index_t i1, index_t i2,
                MAYBE_UNUSED int32_t e1, MAYBE_UNUSED int32_t e2)
{
    uint32_t k = 0, k1 = 0, k2 = 0;
    multirel_t one1 = singleton_rel(i1);
    multirel_t one2 = singleton_rel(i2);
    struct relset_view r1 = view(index_data[i1], &one1);
    struct relset_view r2 = view(index_data[i2], &one2);

    if (sum_buf_n < r1.n + r2.n) {
        sum_buf_n = r1.n + r2.n;
        CHECKED_REALLOC(sum_buf, sum_buf_n, multirel_t);
    }
    multirel_t * tmp = sum_buf;
    while ((k1 < r1.n) && (k2 < r2.n)) {
        if (r1.rels[k1].ind_row < r2.rels[k2].ind_row) {
#ifdef FOR_DL
            tmp[k].e = e2 * r1.rels[k1].e;
#endif
            tmp[k++].ind_row = r1.rels[k1++].ind_row;
        } else if (r1.rels[k1].ind_row > r2.rels[k2].ind_row) {
#ifdef FOR_DL
            tmp[k].e = e1 * r2.rels[k2].e;
#endif
            tmp[k++].ind_row = r2.rels[k2++].ind_row;
        } else {
#ifdef FOR_DL
            int32_t e = e2 * r1.rels[k1].e + e1 * r2.rels[k2].e;
            if (e != 0) {
                tmp[k].ind_row = r1.rels[k1].ind_row;
                tmp[k++].e = e;
            }
#endif
            k1++;
            k2++;
        }
    }
    // finish with k1 and k2
    for (; k1 < r1.n;) {
#ifdef FOR_DL
        tmp[k].e = e2 * r1.rels[k1].e;
#endif
        tmp[k++].ind_row = r1.rels[k1++].ind_row;
    }
    for (; k2 < r2.n;) {
#ifdef FOR_DL
        tmp[k].e = e1 * r2.rels[k2].e;
#endif
        tmp[k++].ind_row = r2.rels[k2++].ind_row;
    }
    ASSERT(k <= r1.n + r2.n);

    relset_t sum = relset_alloc(i1, k);
    memcpy(relset_rels(sum), tmp, k * sizeof(multirel_t));
    if (index_data[i1] != NULL)
        heap_destroy_row(index_data[i1]);
    index_data[i1] = sum;
}

// i1 += i2
// A row is row[0..max] where row[0] = max and the real components are
// row[1..max].
// If len != -1, then it is the real length of row[i1]+row[i2].
//
// If j is given, it is the index of the column that is used for
// pivoting in the case of DL. Then, the operation is
//   i1 = e2*i1 + e1*i2
// where e1 and e2 are adjusted so that the j-th column is zero in i1.
//
// Also update the data for the index file, if needed (i.e. if the given
// pointer is not NULL).
// If rows = NULL, do not update the matrix.
void addRowsUpdateIndex(typerow_t ** rows, index_data_t index_data, index_t i1,
                        index_t i2, MAYBE_UNUSED index_t j)
{
    uint32_t k1, k2, k, len;
    typerow_t * tmp;

    ASSERT(i1 != i2);

#ifndef FOR_DL
    if (rows == NULL) {
        addRelsets(index_data, i1, i2, 0, 0);
        return;
    }
#endif

    ASSERT(rows[i1] != NULL);
    ASSERT(rows[i2] != NULL);
#if DEBUG >= 1
    fprintf(stderr, "R[%d] =", i1);
    fprintRow(stderr, rows[i1]);
    fprintf(stderr, "\n");
    fprintf(stderr, "R[%d] =", i2);
    fprintRow(stderr, rows[i2]);
    fprintf(stderr, "\n");
#endif

#ifdef FOR_DL /* look for the exponents of j in i1 i2*/
    int e1 = 0, e2 = 0;
    int d;
    unsigned int l;
    for (l = 1; l <= rowLength(rows, i1); l++)
        if (rowCell(rows[i1], l) == j)
            e1 = rows[i1][l].e;
    for (l = 1; l <= rowLength(rows, i2); l++)
        if (rowCell(rows[i2], l) == j)
            e2 = rows[i2][l].e;

    ASSERT(e1 != 0 && e2 != 0);

    d = (int)gcd_int64((int64_t)e1, (int64_t)e2);
    e1 /= -d;
    e2 /= d;

    for (l = 1; l <= rowLength(rows, i2); l++)
        rows[i2][l].e *= e1;
    for (l = 1; l <= rowLength(rows, i1); l++)
        rows[i1][l].e *= e2;

#if DEBUG >= 1
    fprintf(stdout, "Computing %d*rows[%d] + %d*rows[%d] for j=%d\n", e2, i1,
            e1, i2, j);
#endif

#endif

    len = 1 + rowLength(rows, i1) + rowLength(rows, i2);
    tmp = heap_alloc_row(i1, len - 1);

    // loop while everybody is here
    k = k1 = k2 = 1;
    while ((k1 <= rowLength(rows, i1)) && (k2 <= rowLength(rows, i2))) {
        if (rowCell(rows[i1], k1) < rowCell(rows[i2], k2)) {
            tmp[k++] = rowFullCell(rows[i1], k1);
            k1++;
        } else if (rowCell(rows[i1], k1) > rowCell(rows[i2], k2)) {
            tmp[k++] = rowFullCell(rows[i2], k2);
            k2++;
        } else {
#ifdef FOR_DL
            if (rows[i1][k1].e + rows[i2][k2].e != 0) {
                tmp[k].id = rows[i1][k1].id;
                tmp[k++].e = rows[i1][k1].e + rows[i2][k2].e;
            }
#else
#if DEBUG >= 1
            fprintf(stderr, "WARNING: j1=j2=%d in addRows\n", k1);
#endif
#endif
            k1++;
            k2++;
        }
    }
    // finish with k1
    for (; k1 <= rowLength(rows, i1); k1++)
        tmp[k++] = rowFullCell(rows[i1], k1);
    // finish with k2
    for (; k2 <= rowLength(rows, i2); k2++)
        tmp[k++] = rowFullCell(rows[i2], k2);
    ASSERT(k <= len);

    heap_resize_last_row(tmp, k - 1);
    heap_destroy_row(rows[i1]);
    rows[i1] = tmp;
#ifdef FOR_DL
    /* restore old coeff for row i2 */
    for (l = 1; l <= rowLength(rows, i2); l++)
        rows[i2][l].e /= e1;
#endif

    if (index_data != NULL)
#ifdef FOR_DL
        addRelsets(index_data, i1, i2, e1, e2);
#else
        addRelsets(index_data, i1, i2, 0, 0);
#endif

#if DEBUG >= 1
    fprintf(stderr, "row[%d]+row[%d] =", i1, i2);
    fprintRow(stderr, rows[i1]);
    fprintf(stderr, "\n");
#endif
}

// A line is "[-]i i1 ... ik [#j]"
int parse_hisfile_line(index_signed_t * ind, char const * t, index_t * j)
{
    int ni = 0;
    index_signed_t sg = 1;

    if (*t == '-') {
        sg = -1;
        t++;
    }

    ind[0] = 0;
    while (1) {
        if ((*t == '\n') || (*t == '#'))
            break;
        else if (*t == ' ') {
            ni++;
            ind[ni] = 0;
        } else
            ind[ni] = 10 * ind[ni] + (*t - '0');

        t++;
    }

    ind[0] = sg * ind[0];
    *j = 0;

    if (*t == '#') {
        for (t++; (*t != '\n'); t++)
            *j = 10 * (*j) + (*t - '0');
    } else {
        ni++;
        *j = -1;
    }

    return ni;
}

/* experimental code for factorization with 5 <= SIZEOF_INDEX <= 7 */
#if !defined(FOR_DL) && 5 <= SIZEOF_INDEX && SIZEOF_INDEX <= 7

/* We use the following data structure for a matrix row:

 * each entry has SIZEOF_INDEX bytes which are stored consecutively
   (little endian, i.e., smallest byte first)
 * the first entry represents the length (say n) of the row
 * thus we have in total (n+1)*SIZEOF_INDEX bytes */

index_t rowCell(index_t * row, int k)
{
    index_t res = 0;
    uint8_t * ptr = ((uint8_t *)row) + k * SIZEOF_INDEX;

    for (int t = 0; t < SIZEOF_INDEX; t++)
        res += ((index_t)ptr[t]) << (t * 8);
    return res;
}

/* this function is only used for factorization, when 5 <= SIZEOF_INDEX <= 7 */
void setCell(index_t * row, int k, index_t j)
{
    uint8_t * ptr = ((uint8_t *)row) + k * SIZEOF_INDEX;
    index_t j0 = j;

    for (int t = 0; t < SIZEOF_INDEX; t++) {
        ptr[t] = (uint8_t)j;
        j >>= 8;
    }
    /* we should have used all bits of j */
    if (j != 0)
        printf("j0=%lu\n", j0);
    ASSERT_ALWAYS(j == 0);
}

/* set (compressed) row[0..n] from buf[0..n] */
void compressRow(index_t * row, index_t * buf, int n)
{
    for (int k = 0; k <= n; k++)
        setCell(row, k, buf[k]);
}

#endif
