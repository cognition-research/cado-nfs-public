#ifndef CADO_READ_PURGEDFILE_IN_PARALLEL_H
#define CADO_READ_PURGEDFILE_IN_PARALLEL_H

#include <stdint.h>
#include "merge_replay_matrix.h"
#include "staged_rows.hpp"

/* maximal number of threads when reading purged file
 *
 * note that this is mostly a limitation of the filesystem more than
 * anything else.
 */
#define MAX_IO_THREADS 16

/* The parallel reader needs a seekable (uncompressed) file. */
int purgedfile_is_seekable(const char * filename);

/* buried != NULL: buried[i] gets bit c set for each column c < col0 of
 * row i (this requires col0 <= 64). factorization: the rows keep the
 * columns in [col0, colmax) and go to staged with their original column
 * indices, see staged_rows.hpp. DL: col0 is mat->skip, the rows go to
 * mat->rows. */
#ifndef FOR_DL
uint64_t read_purgedfile_in_parallel(const char * filename, index_t col0, index_t colmax, uint64_t * buried, staged_rows & staged);
#else
uint64_t read_purgedfile_in_parallel(filter_matrix_t * mat, const char * filename, uint64_t * buried);
#endif

#endif	/* CADO_READ_PURGEDFILE_IN_PARALLEL_H */
