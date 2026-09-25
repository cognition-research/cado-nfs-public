#include <cstdio>
#include <cstdlib>

#include "comm_xor.h"

static __global__ void xor_kernel(uint64_t * dst, const uint64_t * src,
                                  size_t words)
{
    size_t const stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < words; i += stride)
        dst[i] ^= src[i];
}

extern "C" void comm_xor_gf2(uint64_t * dst, const uint64_t * src,
                             size_t words, cudaStream_t stream)
{
    int const block = 256;
    size_t const nblocks = (words + block - 1) / block;
    int const grid = nblocks == 0 ? 1 : (int) (nblocks < 1024 ? nblocks : 1024);
    xor_kernel<<<grid, block, 0, stream>>>(dst, src, words);
    cudaError_t const e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,
                cudaGetErrorString(e));
        abort();
    }
}
