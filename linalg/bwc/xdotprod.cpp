#include "cado.h" // IWYU pragma: keep
#include <cstdint>

#include <vector>

#include "arith-generic.hpp"
#include "matmul_top_vec.hpp"
#include "parallelizing_info.hpp" // for parallelizing_info_s, pi_comm, seria...
#include "xdotprod.hpp"

/* byte offsets, within the local vector buffer, of the words in the
 * index range [vi0, vi1) that the dot product reads */
static std::vector<uint64_t> x_word_offsets(std::vector<uint32_t> const & xv,
        unsigned int j0, unsigned int j1, unsigned int nx,
        mmt_vec const & v, unsigned int vi0, unsigned int vi1)
{
    size_t const stride = v.abase->vec_elt_stride(1);
    ASSERT_ALWAYS(stride % sizeof(uint64_t) == 0);
    std::vector<uint64_t> off;
    for (unsigned int j = j0; j < j1; j++) {
        for (unsigned int t = 0; t < nx; t++) {
            uint32_t const i = xv[j * nx + t];
            if (i < vi0 || i >= vi1)
                continue;
            for (size_t b = 0; b < stride; b += sizeof(uint64_t))
                off.push_back((uint64_t) (i - v.i0) * stride + b);
        }
    }
    return off;
}

void x_dotprod(arith_generic::elt * dst, std::vector<uint32_t> const & xv,
               unsigned int j0, unsigned int j1, unsigned int nx,
               mmt_vec const & v, int sign)
{
    /* We're reading from the shared right vector data -- this area is
     * written to by the other threads in the column. Some of them might
     * be lingering in reduce operations, so we have to wait for them
     */
    if (mmt_vec_is_shared(v)) {
        v.pi.wr[v.d].serialize_threads(__FILE__, __LINE__);
    } else {
        // I'd presume that no locking is needed here. But it's unchecked
        // ASSERT_ALWAYS(0);
    }

    unsigned int const vi0 = v.i0 + mmt_my_own_offset_in_items(v);
    unsigned int const vi1 = vi0 + mmt_my_own_size_in_items(v);

    /* fetch the read items in one batch when the vector lives in a
     * device arena */
    if (bwc_dev_find(v.v)) {
        std::vector<uint64_t> const off =
            x_word_offsets(xv, j0, j1, nx, v, vi0, vi1);
        bwc_dev_fetch_words(v.v, off.data(), off.size());
    }

    for (unsigned int j = j0; j < j1; j++) {
        arith_generic::elt & where = v.abase->vec_item(dst, j - j0);
        for (unsigned int t = 0; t < nx; t++) {
            uint32_t const i = xv[j * nx + t];
            if (i < vi0 || i >= vi1)
                continue;
            arith_generic::elt const & coeff = v.abase->vec_item(v.v, i - v.i0);
            if (sign > 0) {
                v.abase->add_and_reduce(where, coeff);
            } else {
                v.abase->sub_and_reduce(where, coeff);
            }
        }
    }
}

/* The index range that one call of the deferred dot product covers.
 * Sibling threads share one arena and one capture queue, so for a
 * shared vector only thread 0 queues, and it covers the ranges of all
 * sibling threads. The later allreduce of the results makes this equal
 * to the per-thread split of x_dotprod. */
static void x_queue_range(mmt_vec const & v,
        unsigned int & vi0, unsigned int & vi1)
{
    pi_comm & wr = v.pi.wr[v.d];
    unsigned int const nthreads =
        mmt_vec_is_shared(v) ? (wr.trank ? 0 : wr.ncores) : 1;
    vi0 = v.i0 + mmt_my_own_offset_in_items(v);
    vi1 = vi0 + nthreads * mmt_my_own_size_in_items(v);
}

int x_dotprod_queue(std::vector<uint32_t> const & xv,
        unsigned int j0, unsigned int j1, unsigned int nx,
        mmt_vec const & v, size_t slot)
{
    if (!v.abase->is_characteristic_two())
        return 0;
    /* same wait as in x_dotprod: the reduce writers must be done */
    if (mmt_vec_is_shared(v)) {
        v.pi.wr[v.d].serialize_threads(__FILE__, __LINE__);
    }
    unsigned int vi0, vi1;
    x_queue_range(v, vi0, vi1);
    std::vector<uint64_t> const off =
        x_word_offsets(xv, j0, j1, nx, v, vi0, vi1);
    /* also consult the registry when off is empty: sibling threads
     * that queue nothing must get the same verdict as thread 0 */
    return bwc_dev_words_queue(v.v, off.data(), off.size(), slot);
}

void x_dotprod_flush(arith_generic::elt * dst, size_t dst_stride,
        std::vector<uint32_t> const & xv,
        unsigned int j0, unsigned int j1, unsigned int nx,
        mmt_vec const & v, size_t nslots)
{
    unsigned int vi0, vi1;
    x_queue_range(v, vi0, vi1);
    std::vector<uint64_t> const off =
        x_word_offsets(xv, j0, j1, nx, v, vi0, vi1);
    if (off.empty())
        return;
    std::vector<uint64_t> val(off.size() * nslots);
    ASSERT_ALWAYS(bwc_dev_words_flush(v.v, val.data(), off.size(), nslots));

    size_t const stride = v.abase->vec_elt_stride(1);
    const uint64_t * w = val.data();
    for (size_t s = 0; s < nslots; s++) {
        for (unsigned int j = j0; j < j1; j++) {
            arith_generic::elt & where =
                v.abase->vec_item(dst, s * dst_stride + (j - j0));
            for (unsigned int t = 0; t < nx; t++) {
                uint32_t const i = xv[j * nx + t];
                if (i < vi0 || i >= vi1)
                    continue;
                for (size_t b = 0; b < stride; b += sizeof(uint64_t))
                    ((uint64_t *) &where)[b / sizeof(uint64_t)] ^= *w++;
            }
        }
    }
}
