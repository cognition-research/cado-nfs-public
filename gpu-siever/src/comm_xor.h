#ifndef COMM_XOR_H
#define COMM_XOR_H

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dst[i] ^= src[i] for words device words, on the given stream */
void comm_xor_gf2(uint64_t * dst, const uint64_t * src, size_t words,
                  cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
