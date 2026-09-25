#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <utility>
#include <vector>

#include "arith-generic.hpp"
#include "macros.h"
#include "matmul.hpp"
#include "matmul-common.hpp"
#include "matrix_u32.hpp"
#include "params.hpp"

#include "spmv_gf2.h"

/* A CADO-NFS matmul backend that runs the GF(2) sparse product on a
 * CUDA device, for the b64, b128, and b256 arithmetic layers. Select
 * it with mm_impl=cuda. */

#define MM_MAGIC_FAMILY        0xa063UL
#define MM_MAGIC_VERSION       0x1001UL
#define MM_MAGIC (MM_MAGIC_FAMILY << 16 | MM_MAGIC_VERSION)

/* Same convention as the CPU backends: for the optimized direction, the
 * stored rows are the output coordinates, so the product is a gather. */
#define MM_DIR0_PREFERS_TRANSP_MULT   1

struct matmul_cuda : public matmul_interface {
    std::vector<uint32_t> q;
    spmv_gf2 * gpu = nullptr;
    unsigned int words;

    void build_cache(matrix_u32 &&) override;
    int reload_cache_private() override;
    void save_cache_private() override;
    void mul(void *, const void *, int) override;
    ~matmul_cuda() override { spmv_gf2_free(gpu); }

    matmul_cuda(matmul_public &&, arith_generic *, cxx_param_list &, int);

    matmul_cuda(matmul_cuda const &) = delete;
    matmul_cuda& operator=(matmul_cuda const &) = delete;
    matmul_cuda(matmul_cuda &&) noexcept = default;
    matmul_cuda& operator=(matmul_cuda &&) noexcept = default;

    private:
    void upload();
};

matmul_cuda::matmul_cuda(matmul_public && P, arith_generic * arith, cxx_param_list & pl, int optimized_direction)
    : matmul_interface(std::move(P))
{
    ASSERT_ALWAYS(arith->simd_groupsize() % 64 == 0);
    ASSERT_ALWAYS(arith->elt_stride() == arith->simd_groupsize() / 8);
    words = arith->simd_groupsize() / 64;
    int const suggest = optimized_direction ^ MM_DIR0_PREFERS_TRANSP_MULT;
    store_transposed = suggest;
    pl.parse("mm_store_transposed", store_transposed);
    if (store_transposed != suggest) {
        fprintf(stderr, "Warning, mm_store_transposed"
                " overrides suggested matrix storage ordering\n");
    }
}

void matmul_cuda::build_cache(matrix_u32 && m)
{
    q = std::move(m.p);
    ncoeffs = q.size() - dim[store_transposed];
}

int matmul_cuda::reload_cache_private()
{
    /* the cached data is the raw row stream; it does not depend on the
     * element width, so every layer shares one cache file */
    auto f = matmul_common_reload_cache_fopen(sizeof(uint64_t), *this, MM_MAGIC);
    if (!f) return 0;

    uint64_t datasize;

    MATMUL_COMMON_READ_ONE64(datasize, f.get());
    resize_and_check_meaningful(q, datasize, f.get());
    MATMUL_COMMON_READ_MANY32(q.data(), datasize, f.get());

    return 1;
}

void matmul_cuda::save_cache_private()
{
    auto f = matmul_common_save_cache_fopen(sizeof(uint64_t), *this, MM_MAGIC);
    if (!f) return;

    MATMUL_COMMON_WRITE_ONE64(q.size(), f.get());
    MATMUL_COMMON_WRITE_MANY32(q.data(), q.size(), f.get());
}

/* The device copy is made at the first product, after the framework is
 * done with build_cache and save_cache. The flat data is then released:
 * the framework does not save the cache after products. */
void matmul_cuda::upload()
{
    if (gpu) return;
    gpu = spmv_gf2_new(dim[store_transposed], q.size(), q.data(),
            std::max(dim[0], dim[1]), words);
    q = std::vector<uint32_t>();
}

void matmul_cuda::mul(void * xdst, void const * xsrc, int d)
{
    upload();
    auto * dst = (uint64_t *) xdst;
    auto const * src = (uint64_t const *) xsrc;
    if (d == !store_transposed) {
        spmv_gf2_gather(gpu, dst, src, dim[!d], dim[d]);
    } else {
        spmv_gf2_scatter(gpu, dst, src, dim[!d], dim[d]);
    }
    iteration[d]++;
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
matmul_interface * CADO_CONCATENATE4(new_matmul_, ARITH_LAYER, _, MM_IMPL)(
        matmul_public && P,
        arith_generic * arith,
        cxx_param_list & pl,
        int optimized_direction)
{
    return new matmul_cuda(std::move(P), arith, pl, optimized_direction);
}
