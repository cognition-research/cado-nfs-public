#include <cstdio>
#include <cstdlib>

#include "comm_multimem.h"

#if __CUDA_ARCH__ >= 900

static __device__ inline uint64_t mm_ldxor(const uint64_t * p)
{
    uint64_t v;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.xor.b64 %0, [%1];"
                 : "=l"(v) : "l"(p) : "memory");
    return v;
}

static __device__ inline void mm_st(uint64_t * p, uint64_t v)
{
    asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;"
                 :: "l"(p), "l"(v) : "memory");
}

#else

static __device__ inline uint64_t mm_ldxor(const uint64_t *) { return 0; }
static __device__ inline void mm_st(uint64_t *, uint64_t) {}

#endif

static __global__ void ldxor_kernel(uint64_t * dst, const uint64_t * mc_src,
                                    size_t words)
{
    size_t const stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < words; i += stride)
        dst[i] = mm_ldxor(mc_src + i);
}

static __global__ void bcast_kernel(uint64_t * mc_dst, const uint64_t * src,
                                    size_t words)
{
    size_t const stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < words; i += stride)
        mm_st(mc_dst + i, src[i]);
}

static __global__ void barrier_kernel(unsigned long long * mc_flag,
        const unsigned long long * uc_flag, unsigned long long target)
{
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;"
                 :: "l"(mc_flag), "l"(1ull) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
    unsigned long long v;
    do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                     : "=l"(v) : "l"(uc_flag) : "memory");
    } while (v < target);
#endif
}

static void check_launch(int line)
{
    cudaError_t const e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s\n", __FILE__, line,
                cudaGetErrorString(e));
        abort();
    }
}

static int grid_of(size_t words)
{
    size_t const nblocks = (words + 255) / 256;
    return nblocks == 0 ? 1 : (int) (nblocks < 1024 ? nblocks : 1024);
}

extern "C" void comm_multimem_ldxor(uint64_t * dst, const uint64_t * mc_src,
        size_t words, cudaStream_t stream)
{
    ldxor_kernel<<<grid_of(words), 256, 0, stream>>>(dst, mc_src, words);
    check_launch(__LINE__);
}

extern "C" void comm_multimem_bcast(uint64_t * mc_dst, const uint64_t * src,
        size_t words, cudaStream_t stream)
{
    bcast_kernel<<<grid_of(words), 256, 0, stream>>>(mc_dst, src, words);
    check_launch(__LINE__);
}

extern "C" void comm_multimem_barrier(unsigned long long * mc_flag,
        const unsigned long long * uc_flag, unsigned long long target,
        cudaStream_t stream)
{
    barrier_kernel<<<1, 1, 0, stream>>>(mc_flag, uc_flag, target);
    check_launch(__LINE__);
}
