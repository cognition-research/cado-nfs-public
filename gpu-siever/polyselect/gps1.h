/* gps1: interface between the CUDA kernels (gps1_kernels.cu, c++17) and the
 * host driver (gps1.cpp, c++20, uses cado-nfs headers).
 *
 * Sort engine: for one leading coefficient, the host enumerates every
 * (q, rq) pair (q = 1 first, then the special-q CRT residues) and hands the
 * GPU a batch of them. The GPU computes every transformed root
 * (rp - rq)/q^2 mod p^2, expands its class over [-M, M) into (key, root)
 * entries with key = (batch index << offset_bits) | (u + M), radix-sorts all
 * entries of the batch at once, and scans adjacent equal keys for collisions.
 * offset_bits comes from M, so large P values fit; the batch index gets the
 * remaining high bits. */
#ifndef GPS1_H
#define GPS1_H

#include <stdint.h>

#define GPS1_MAX_ENTRIES (128u << 20) /* (key, val) pairs per sort */
#define GPS1_MAX_MATCHES (1 << 20)

struct match_t { uint32_t p1, p2, b, pad; int64_t i; };

/* width of the u + M key field: the smallest width that holds [0, 2M) */
static inline uint32_t gps1_offset_bits(int64_t M)
{
    return 64 - __builtin_clzll(2 * (uint64_t) M - 1);
}

/* device buffers */
struct gpu_t {
    uint32_t *primes;       /* [nprimes] */
    uint32_t *root_prime;   /* [nroots] prime index of each root */
    uint64_t *roots;        /* [nroots] rp mod p^2 */
    uint32_t *entry_off;    /* [nroots + 1] exclusive scan of slots per root */
    uint64_t *qs;           /* [ndq] distinct q values of the batch */
    uint32_t *qidx;         /* [batch] item -> distinct q index */
    uint64_t *rq_hi, *rq_lo;/* [batch] rq of each item, 128 bits */
    uint64_t *invqq;        /* [ndq * nprimes] 1/q^2 mod p^2 */
    uint64_t *keys, *keys_alt;
    uint32_t *vals, *vals_alt;
    void *sort_tmp; size_t sort_tmp_bytes;
    match_t *matches;
    uint32_t nprimes;
};

/* fill, sort, and scan one batch; returns the number of matches written to
 * G->matches. nentries = entry_off[nroots] * batch must be <= GPS1_MAX_ENTRIES. */
extern "C" uint64_t gps1_run(const gpu_t *G, uint32_t nroots, uint32_t ndq,
                             uint32_t batch, int64_t M, uint64_t slots);

/* temp-storage size the radix sort needs for nentries pairs */
extern "C" size_t gps1_sort_tmp_bytes(uint64_t nentries);

#endif
