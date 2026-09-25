#ifndef COMM_MULTIMEM_H
#define COMM_MULTIMEM_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

/* The multimem (NVLink SHARP) kernels of the nvls communication plugin
 * (src/bwccomm_nvls.cpp). The mc pointers are multicast virtual
 * addresses of a bound multicast group; the other pointers are
 * ordinary device addresses of this device. */

extern "C" {

/* dst[i] = the XOR of word i of mc_src over every replica */
void comm_multimem_ldxor(uint64_t * dst, const uint64_t * mc_src,
        size_t words, cudaStream_t stream);

/* word i of mc_dst on every replica = src[i] */
void comm_multimem_bcast(uint64_t * mc_dst, const uint64_t * src,
        size_t words, cudaStream_t stream);

/* The arrival barrier of the CUDA programming guide (4.16.4.4): add 1
 * to the counter on every replica, then wait on the local replica
 * until the count reaches target. The proxy fences of the kernel
 * order the multimem accesses of this device against its ordinary
 * accesses of the same memory. */
void comm_multimem_barrier(unsigned long long * mc_flag,
        const unsigned long long * uc_flag, unsigned long long target,
        cudaStream_t stream);

}

#endif
