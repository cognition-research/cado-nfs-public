#ifndef CADO_MATMUL_TOP_VEC_HPP
#define CADO_MATMUL_TOP_VEC_HPP

#include <cstdint>

#include <string>

#include "gmp_aux.h"
#include "parallelizing_info.hpp"
#include "arith-generic.hpp"
#include "bwc_device_arena.hpp"
#include "bwc_file.hpp"
#include "macros.h"

struct matmul_top_data;

// the ``all_v'' field collects all the pointers to the per-thread vector
// values. There are exactly pi->wr[0]->ncores such pointers in
// mmt->wr[0]. In some cases, these pointers may be equal (for source
// vectors, never used as destination), and in some cases not.

struct mmt_vec {
    arith_generic * abase = nullptr;
    parallelizing_info & pi;
    int d = -1;
    pi_datatype * pitype = nullptr;

    unsigned int n = 0;          // total size in items
    unsigned int i0 = 0;
    unsigned int i1 = 0;

    arith_generic::owned_vector owned_v;

    // v == owned_v except for shared vector, where this holds only at
    // pi->wr[d]->trank == 0
    arith_generic::elt * v = nullptr;

    /* pointers to other vectors are held in shared memory areas (shared
     * among cores, of course)
     */
    using pointer_to_others = parallelizing_info::shared_array<mmt_vec *>;

    /* pi->wr[d]->ncores siblings ; only in case all cores in the
     * communicator have their own data area v */
    pointer_to_others siblings;

    /* pi->m->ncores siblings, always */
    pointer_to_others mpals;

    /* pi->wr[0]->ncores and */
    /* pi->wr[1]->ncores siblings, always. */
    pointer_to_others wrpals[2];

    private:
    void set_pointer_to_others(bool with_siblings);

    public:

    // mmt_vec() = default;
    mmt_vec(matmul_top_data & mmt, arith_generic * abase, pi_datatype * pitype, int d, int flags, unsigned int n);
    mmt_vec(mmt_vec const &) = delete;
    mmt_vec& operator=(mmt_vec const &) = delete;
    ~mmt_vec() { if (owned_v) bwc_dev_unregister(owned_v.get()); }

    mmt_vec& operator=(mmt_vec &&) = delete;

    /* We need to do some stuff here! */
    mmt_vec(mmt_vec && o) noexcept;

    size_t rsbuf_items = 0;      // auto-expanded on demand.

    // ATTENTION: elements are laid out flat in here! rsbuf[0][i] is not
    // a thing.
    arith_generic::owned_vector rsbuf[2];            // only for RS_CHOICE == RS_CHOICE_MINE
    int consistency = 0;    /* 0 == inconsistent ; 1 == partial ; 2 == full */

    // XXX private ?
    mmt_vec & sibling(unsigned int i) { return siblings ? *(siblings.get()[i]) : *this; }
    mmt_vec const & sibling(unsigned int i) const { return siblings ? *(siblings.get()[i]) : *this; }
};

/* some handy macros to access portions of mmt_vec's */
#define SUBVEC(v,w,offset) (v)->abase->vec_subvec((v)->abase, (v)->w, offset)
#define SUBVEC_const(v,w,offset) (v)->abase->vec_subvec_const((v)->abase, (v)->w, offset)

/* These are flags for the distributed vectors. For the moment we have
 * only one flag */
#define THREAD_SHARED_VECTOR    1

extern size_t mmt_my_own_size_in_items(mmt_vec const & v);
extern size_t mmt_my_own_size_in_bytes(mmt_vec const & v);
extern size_t mmt_my_own_offset_in_items(mmt_vec const & v);
extern size_t mmt_my_own_offset_in_items(mmt_vec const & v, unsigned int);
extern size_t mmt_my_own_offset_in_bytes(mmt_vec const & v);
extern arith_generic::elt * mmt_my_own_subvec(mmt_vec & v);
extern arith_generic::elt * mmt_my_own_subvec(mmt_vec & v, unsigned int);
extern arith_generic::elt const * mmt_my_own_subvec(mmt_vec const & v);

/* do not use this function if you want consistency when the splitting
 * changes ! */
extern void mmt_vec_set_random_inconsistent(mmt_vec & v, cxx_gmp_randstate & rstate);
extern unsigned long mmt_vec_hamming_weight(mmt_vec const & y);
extern void mmt_vec_set_x_indices(mmt_vec & y, std::vector<uint32_t> const & gxvecs, unsigned int j0, unsigned int j1, unsigned int nx);
extern void mmt_vec_set_expanded_copy_of_local_data(mmt_vec & y, const void * v, unsigned int n);

extern void mmt_own_vec_set(mmt_vec & w, mmt_vec const & v);
extern void mmt_own_vec_set2(mmt_vec const & z, mmt_vec & w, mmt_vec const & v);
// extern void mmt_vec_swap(mmt_vec & w, mmt_vec & v);
extern void mmt_full_vec_set(mmt_vec & w, mmt_vec const & v);
extern void mmt_full_vec_set_zero(mmt_vec & v);
extern void mmt_vec_set_basis_vector_at(mmt_vec & v, int k, unsigned int j);
extern void mmt_vec_set_basis_vector(mmt_vec & v, unsigned int j);
extern void mmt_vec_add_basis_vector_at(mmt_vec & v, int k, unsigned int j);
extern void mmt_vec_add_basis_vector(mmt_vec & v, unsigned int j);
#if 0
extern void matmul_top_fill_random_source_generic(matmul_top_data & mmt, size_t stride, mmt_vec & v, int d);
#endif
extern int mmt_vec_load(mmt_vec & v, std::string const & name, unsigned int itemsondisk, unsigned int block_position) ATTRIBUTE_WARN_UNUSED_RESULT;
extern int mmt_vec_save(mmt_vec & v, std::string const & name, unsigned int itemsondisk, unsigned int block_position);
/* the files that mmt_vec_save writes; data is at the leader only */
extern std::vector<bwc_file> mmt_vec_gather(mmt_vec & v, std::string const & name, unsigned int itemsondisk, unsigned int block_position);
/* its two halves, as pi_gather_stage and pi_gather_exchange: the pieces
 * of the job (thread 0; the other threads get empty data), then the
 * exchange over comm that thread 0 of every job runs, in a thread of
 * its own if it likes */
struct mmt_vec_staged {
    std::vector<bwc_file> files;
    size_t outsize;
    int d;
};
extern mmt_vec_staged mmt_vec_stage(mmt_vec & v, std::string const & name, unsigned int itemsondisk, unsigned int block_position);
extern std::vector<bwc_file> mmt_vec_exchange(parallelizing_info & pi, MPI_Comm comm, mmt_vec_staged st);
/* f holds the 64-bit items of a gathered twisted vector; it becomes the
 * unpadded items of the untwisted vector (perm from mmt_untwist_perm) */
extern void mmt_vec_untwist_file(bwc_file & f, std::vector<uint32_t> const & perm, size_t unpadded);
extern void mmt_vec_clear_padding(mmt_vec & v, size_t unpadded, size_t padded);

static inline int mmt_vec_is_shared(mmt_vec const & v) {
    return v.siblings.get() == nullptr;
}

/* device arena coherence, for host code around the vector data; both
 * are no-ops when no arena covers the vector (bwc_device_arena.hpp).
 * The threads of the wiring share one buffer and one arena when the
 * vector is shared, so only thread 0 touches the arena; every call
 * site serializes the threads before another thread reads the data. */
static inline void mmt_vec_dev_fetch(mmt_vec const & v) {
    if (mmt_vec_is_shared(v) && v.pi.wr[v.d].trank)
        return;
    bwc_dev_fetch(v.v, v.abase->vec_elt_stride(v.i1 - v.i0));
}
static inline void mmt_vec_dev_invalidate(mmt_vec const & v) {
    if (mmt_vec_is_shared(v) && v.pi.wr[v.d].trank)
        return;
    bwc_dev_invalidate(v.v);
}
static inline void mmt_vec_dev_adopt(mmt_vec const & v) {
    if (mmt_vec_is_shared(v) && v.pi.wr[v.d].trank)
        return;
    bwc_dev_adopt(v.v);
}
static inline void mmt_vec_dev_mirror(mmt_vec const & v, mmt_vec const & like) {
    if (mmt_vec_is_shared(v) && v.pi.wr[v.d].trank)
        return;
    bwc_dev_mirror(v.v, v.abase->vec_elt_stride(v.i1 - v.i0), like.v);
}

/* copy sz items, on the device mirrors when both sides hold their
 * current data there, else on the host */
static inline void mmt_vec_copy_items(arith_generic * A,
        arith_generic::elt * dst, arith_generic::elt const * src, size_t sz)
{
    size_t const bytes = A->vec_elt_stride(sz);
    if (bwc_dev_copy(dst, src, bytes))
        return;
    bwc_dev_fetch(src, bytes);
    A->vec_set(dst, src, sz);
    bwc_dev_push(dst, bytes);
}

extern void mmt_vec_share_across_threads(mmt_vec & v);

extern void mmt_vec_set_random_through_file(mmt_vec & v, std::string const & name, unsigned int itemsondisk, cxx_gmp_randstate & rstate, unsigned int block_position);

// private ?
extern void mmt_vec_downgrade_consistency(mmt_vec & v);

#endif	/* MATMUL_TOP_VEC_HPP_ */
