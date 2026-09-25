#ifndef SPMV_GF2_H
#define SPMV_GF2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GF(2)^(64*words) sparse matrix-vector products on one CUDA device.
 * A vector element is `words` consecutive 64-bit words; bit k of word w
 * belongs to vector 64*w+k of a group of 64*words independent GF(2)
 * vectors. */

typedef struct spmv_gf2 spmv_gf2;

/* rows is a flat stream of nrows stored rows. Each row is a 32-bit
 * length followed by that many 32-bit column indices. nwords is the
 * total length of the stream. The setup may reorder the columns of a
 * row in place; the caller must not rely on their order after the
 * call. max_dim is the largest vector length in elements that the
 * product functions will see. words is the element width and must be
 * 1, 2, or 4. The context binds to one device, selected from the MPI
 * local rank and the count of contexts already made in the process. */
spmv_gf2 * spmv_gf2_new(unsigned int nrows, uint64_t nwords,
                        uint32_t * rows, unsigned int max_dim,
                        unsigned int words);

/* dst[i] = xor of src[col] over the entries col of stored row i.
 * ndst and nsrc count elements. */
void spmv_gf2_gather(spmv_gf2 *, uint64_t * dst, const uint64_t * src,
                     unsigned int ndst, unsigned int nsrc);

/* dst[col] ^= src[i] over the entries col of each stored row i,
 * with dst set to zero first. ndst and nsrc count elements. */
void spmv_gf2_scatter(spmv_gf2 *, uint64_t * dst, const uint64_t * src,
                      unsigned int ndst, unsigned int nsrc);

void spmv_gf2_free(spmv_gf2 *);

#ifdef __cplusplus
}

#include <vector>

/* Builds the flat row stream of a random matrix on the device selected
 * for the calling context: n stored rows with Poisson(mean) lengths
 * and entries uniform in [0,range). Row i draws from Philox streams
 * (seed, 2i) and (seed, 2i+1), so the stream is a pure function of the
 * arguments.
 * Entries of a row can repeat (probability ~ mean^2/2/range per row);
 * repeats cancel under xor. bwc finds this symbol with dlsym when it
 * runs with random_matrix= (see random_matrix_stream_solib in cado's
 * random_matrix.hpp). Only the CUDA backend implements it. */
extern "C" std::vector<uint32_t> random_matrix_stream_solib(
        unsigned long n, unsigned long range, double mean,
        unsigned long seed);
#endif

#endif
