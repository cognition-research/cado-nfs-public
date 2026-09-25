#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>

#include "filter_config.h"
#include "gzip.h"
#include "macros.h"
#include "misc.h"
#include "matrix_output.hpp"
#include "sparse.h"
#include "typedefs.h"

void
renumber (index_t small_ncols, index_t *colweight, index_t ncols,
          MAYBE_UNUSED const index_t * orig,
          MAYBE_UNUSED const char *idealsfilename)
{
    index_t k, nb;
    index_t *tmp;

#ifdef FOR_DL
    FILE *renumberfile = fopen_maybe_compressed (idealsfilename, "w");
    if (renumberfile == NULL)
    {
      fprintf (stderr, "Error while opening file to save permutation of"
                       "ideals\n");
      exit(EXIT_FAILURE);
    }
#endif

    ASSERT_ALWAYS(small_ncols);
    tmp = (index_t *) malloc ((small_ncols << 1) * sizeof(index_t));
    ASSERT_ALWAYS(tmp != NULL);
    memset(tmp, 0, (small_ncols << 1) * sizeof(index_t));
    for (k = 0, nb = 0; k < ncols; k++)
      if (colweight[k] > 0)
	{
	  tmp[nb++] = colweight[k];
	  tmp[nb++] = k;
	}
    ASSERT_ALWAYS(nb == 2 * small_ncols);
#ifdef FOR_DL
    fprintf (renumberfile, "# %" PRIu64 "\n", (uint64_t) small_ncols);
#endif
    printf ("Sorting %" PRIu64 " columns by decreasing weight\n",
            (uint64_t) small_ncols);
    fflush (stdout);
    qsort (tmp, small_ncols, 2*sizeof(index_t), cmp_index2);
    memset (colweight, 0, ncols * sizeof(index_t));
    // useful for BW + skipping heavy part only...
    index_signed_t j;
    ASSERT_ALWAYS(nb >= 1);
    for (j = (index_signed_t) (nb - 1), k = 1; j >= 0; j -= 2)
      {
        colweight[tmp[j]] = k++; // always this +1 trick
#ifdef FOR_DL
        fprintf (renumberfile, "%" PRIu64 " %" PRIx64 "\n",
                 (uint64_t) colweight[tmp[j]]-1,
                 (uint64_t) (orig ? orig[tmp[j]] : tmp[j]));
#endif
      }

#ifdef FOR_DL
    fclose_maybe_compressed (renumberfile, idealsfilename);
#endif
    free(tmp);
}

int
matrix_name_is_binary(const char * sparsename)
{
    return has_suffix(sparsename, ".bin") || has_suffix(sparsename, ".bin.gz");
}

void
writeIndex(const char *indexname, index_data_t index_data, index_t small_nrows)
{
    FILE *indexfile = NULL;
    indexfile = fopen_maybe_compressed(indexname, "w");
    ASSERT_ALWAYS (indexfile != NULL);
    fprintf(indexfile, "%" PRIu64 "\n", (uint64_t) small_nrows);

    for (index_t i = 0; i < small_nrows; ++i) {
        uint32_t n = relset_n(index_data[i]);
        multirel_t const * rels = relset_rels(index_data[i]);
        ASSERT (n > 0);
        fprintf(indexfile, "%" PRIu32, n);
        for (uint32_t j = 0; j < n; ++j) {
#ifdef FOR_DL
            fprintf(indexfile, " %" PRIx64 ":%d",
                    (uint64_t) rels[j].ind_row, rels[j].e);
#else
            fprintf(indexfile, " %" PRIx64 "", (uint64_t) rels[j].ind_row);
#endif
        }
        fprintf(indexfile, "\n");
    }
    fclose_maybe_compressed(indexfile, indexname);
}
