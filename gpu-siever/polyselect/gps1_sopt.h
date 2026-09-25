/* gps1_sopt: interface between the CUDA size optimization (gps1_sopt.cu)
 * and the host driver (gps1.cpp).
 *
 * One candidate is one raw polynomial pair (f, g) with deg(g) = 1. One GPU
 * thread runs the algorithm of cado-nfs size_optimization() on one
 * candidate. Coefficients travel as fixed-width sign-magnitude integers
 * (little-endian 32-bit limbs). The host computes the LLL skewness seed
 * (sopt_get_skewness) with glibc pow() and passes it in skew. When a
 * candidate overflows the fixed width or hits an iteration cap, the GPU
 * sets the fail flag and the host redoes that candidate on the CPU. */
#ifndef GPS1_SOPT_H
#define GPS1_SOPT_H

#include <stdint.h>

#define SOPT_GPU_LIMBS 168
#define SOPT_GPU_MAXDEG 7

struct sopt_z { int32_t n, sign; uint32_t l[SOPT_GPU_LIMBS]; };

struct sopt_pair_t {
    sopt_z f[SOPT_GPU_MAXDEG + 1];
    sopt_z g[2];
    double skew;
    int32_t deg;
    int32_t fail;
};

extern "C" void gps1_sopt_run(sopt_pair_t *pairs, uint32_t n,
                              unsigned int effort);

/* the same optimization on the host CPU, for tests on a machine without
 * a GPU */
extern "C" void gps1_sopt_run_host(sopt_pair_t *pairs, uint32_t n,
                                   unsigned int effort);

#endif
