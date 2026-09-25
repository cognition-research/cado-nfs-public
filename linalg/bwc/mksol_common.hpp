#ifndef CADO_LINALG_BWC_MKSOL_COMMON_HPP
#define CADO_LINALG_BWC_MKSOL_COMMON_HPP

/* The parts of mksol that do not depend on how the iteration loop is
 * scheduled: the setup, the V vectors of a block, the F coefficients
 * of a window, the iterations, the host combine, and the main function.
 * mksol and gmksol share them. */

#include <cstddef>
#include <climits>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "arith-cross.hpp"
#include "arith-generic.hpp"
#include "async.hpp"
#include "bwc_file.hpp"
#include "gmp_aux.h"
#include "matmul_top.hpp"
#include "matmul_top_vec.hpp"
#include "mmt_vector_pair.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"

struct mksol_state {
    parallelizing_info & pi;
    int fake;
    int tcan_print;
    std::array<unsigned int, 2> solutions;
    unsigned int Av_width;      /* 64 in characteristic two, else 1 */
    unsigned int Av_multiplex;  /* the number of V vectors, n / Av_width */
    unsigned int As_width;      /* solutions[1] - solutions[0] */
    unsigned int Af_multiplex;  /* F files per coefficient, As_width / Av_width */
    std::unique_ptr<arith_generic> Av;
    pi_datatype * Av_pi;
    std::unique_ptr<arith_generic> As;
    std::unique_ptr<arith_cross_generic> AvxAs;
    matmul_top_data mmt;
    mmt_vector_pair ymy;
    std::vector<mmt_vec> vi;
    unsigned int unpadded;
    cxx_gmp_randstate rstate;
    unsigned int end;           /* bw->end, lowered by a short read of F */
    unsigned int block_end;     /* min(end, s + checkpoint_precious) */
    size_t one_fcoeff;          /* bytes of one coefficient of F */
    std::vector<arith_generic::elt *> fcoeffs;  /* interval coefficients per V vector */
    std::vector<arith_generic::elt *> fnext;    /* read ahead, at the leader */
    unsigned int fnext_s = UINT_MAX;            /* the window in fnext */
    unsigned int fnext_n = 0;
    arith_generic::elt * fcoeff_tmp = nullptr;
    struct timing_data timing[1];
    bwc_background writer;
    bwc_background freader;

    mksol_state(parallelizing_info & pi, cxx_param_list & pl);
    ~mksol_state();

    /* the start of block [s, s + checkpoint_precious): the V vectors at
     * iteration s, twisted, and ymy[0] zero */
    void load(unsigned int s);

    /* the same, once the untwisted V vectors at iteration s are in vi */
    void begin_block(unsigned int s);

    /* the coefficients [s0, s1) of F on every rank. A short read lowers
     * end; the return value is the new s1, equal to s0 when the block
     * is complete. */
    unsigned int read_f(unsigned int s0, unsigned int s1);

    /* the leader reads the coefficients [s0, s0 + interval) of F in the
     * background while the current window runs; the next read_f takes
     * them without touching the disk */
    void start_read_f(unsigned int s0);

    /* coefficient c of the current window, for the V vector i */
    const arith_generic::elt * f(unsigned int i, unsigned int c) const;

    /* y += sum_i vi[i] * f(i, c) over the items [item0, item0 + items)
     * on the device mirrors; returns 0 when the width or the mirrors do
     * not allow it, and nothing ran */
    int dev_combine(mmt_vec & y, unsigned int c, size_t item0, size_t items);

    /* the iterations of block [s, block_end): for each coefficient of
     * F from the top down, combine(c) into the iterate then the product
     * into the other pair, as in gkrylov; returns the vector that holds
     * the final iterate of the block. The timing lines carry the stage
     * name. */
    mmt_vec & iterate(unsigned int s, mmt_vector_pair & other,
            void (*combine)(mksol_state &, mmt_vec &, unsigned int),
            const char * stage);

    /* the S file pattern of block [s, s + checkpoint_precious) */
    std::string s_pattern(unsigned int s) const;

    /* waits for the writer, then the final tally */
    void finish(const char * stage);
};

/* y += sum_i vi[i] * f(i, c), each thread on the piece it owns, then
 * the broadcast */
void mksol_combine(mksol_state & M, mmt_vec & y, unsigned int c);

int mksol_main(int argc, char const * argv[],
        void * (*prog)(parallelizing_info &, cxx_param_list &, void *));

#endif	/* CADO_LINALG_BWC_MKSOL_COMMON_HPP */
