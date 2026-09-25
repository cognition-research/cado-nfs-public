#ifndef CADO_LINALG_BWC_KRYLOV_COMMON_HPP
#define CADO_LINALG_BWC_KRYLOV_COMMON_HPP

/* The parts of krylov that do not depend on how the iteration loop is
 * scheduled: the online check data, the setup before the loop, the
 * output of one interval, and the main function. krylov and gkrylov
 * share them. */

#include <cstdint>

#include <array>
#include <memory>
#include <vector>

#include "arith-cross.hpp"
#include "arith-generic.hpp"
#include "async.hpp"
#include "bwc_file.hpp"
#include "matmul_top.hpp"
#include "matmul_top_vec.hpp"
#include "mmt_vector_pair.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"

struct check_data {
    matmul_top_data & mmt;
    parallelizing_info & pi;
    int nchecks;
    arith_generic * A;
    std::unique_ptr<arith_generic> Ac;
    pi_datatype * Ac_pi;
    std::unique_ptr<arith_cross_generic> AxAc;
    mmt_vec check_vector;
    arith_generic::elt * Tdata = nullptr;
    arith_generic::elt * ahead = nullptr;

    int tcan_print = 0;

    check_data(matmul_top_data & mmt, arith_generic * A);
    ~check_data();

    void load();

    /* at the start of an interval; y in the order of check_vector
     * (untwisted, unless keep_twisted has twisted check_vector) */
    void plan_ahead(mmt_vec const & y);

    /* at the end of the interval, y twisted; true when the last dot
     * product cancels ahead */
    bool verify(mmt_vec const & y, std::vector<uint32_t> const & gxvecs,
            int nx);
};

struct krylov_state {
    parallelizing_info & pi;
    int fake;
    int tcan_print;
    std::array<int, 2> ys;
    std::unique_ptr<arith_generic> A;
    matmul_top_data mmt;
    std::shared_ptr<check_data> C;
    mmt_vector_pair ymy;
    unsigned int unpadded;
    std::vector<uint32_t> gxvecs;
    unsigned int nx = 0;
    arith_generic::elt * xymats;
    struct timing_data timing[1];
    /* the items gathered per checkpoint (the unpadded items of an
     * untwisted vector, every item of a twisted one) and, at the
     * leader, the untwist the writer then applies to them */
    unsigned int gathered;
    std::vector<uint32_t> untwist;
    /* the communicator of the writer thread's gather (thread 0 of each
     * job), so that it never shares one with the main thread */
    MPI_Comm gather_comm;
    /* MPI_THREAD_MULTIPLE was provided; otherwise (MPI_THREAD_SERIALIZED)
     * the gather runs on the main thread and only the write is in the
     * writer thread */
    bool threaded_mpi;
    /* last member: its thread reads untwist, unpadded and gather_comm */
    bwc_background writer;

    /* everything before the iteration loop: the matrix, the x vectors,
     * the starting vector, the interval, the check data */
    krylov_state(parallelizing_info & pi, cxx_param_list & pl);
    ~krylov_state();

    /* the products of interval [s, s + interval) on the twisted vector
     * ymy[0], with the dot products by x into xymats */
    void iterate(int s);

    /* the vector stays twisted for the whole run: output untwists the
     * checkpoint in the writer thread, and the check vector is twisted
     * along so that plan_ahead takes the twisted vector */
    void keep_twisted();

    /* the end of interval [s, s + interval): the A block and the
     * checkpoint of the vector y, untwisted unless keep_twisted was
     * called. Every job gathers the checkpoint and the leader writes
     * the files in the background; finish waits for them. */
    void output(int s, mmt_vec & y);

    void finish(cxx_param_list & pl, const char * stage);
};

int krylov_main(int argc, char const * argv[],
        void * (*prog)(parallelizing_info &, cxx_param_list &, void *));

#endif	/* CADO_LINALG_BWC_KRYLOV_COMMON_HPP */
