#include "cado.h" // IWYU pragma: keep

#include <cstdlib>

#include "fmt/base.h"

#include "bw-common.hpp"
#include "krylov_common.hpp"
#include "macros.h"
#include "matmul_top.hpp"
#include "matmul_top_vec.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"

static void * krylov_prog(parallelizing_info & pi, cxx_param_list & pl, void * arg MAYBE_UNUSED)
{
    krylov_state K(pi, pl);
    mmt_vec & y = K.ymy[0];

    pi.interleaving_flip();
    pi.interleaving_flip();

    epoch_timer epoch;
    for(int s = bw->start ; s < bw->end ; s += bw->interval ) {
        if (K.C) K.C->plan_ahead(y);

        /* Create an empty slot in program execution, so that we don't
         * impose strong constraints on twist/untwist_vector being free of
         * MPI calls (well, it's *not* free of MPI calls, to start
         * with...).
         */
        pi.interleaving_flip();
        pi.interleaving_flip();
        mmt_vec_twist(K.mmt, y);

        epoch_timer it;
        K.iterate(s);
        double const t_iter = it.lap();

        if (K.C && !K.C->verify(y, K.gxvecs, K.nx)) {
            fmt::print("Failed check at iteration {}\n", s + bw->interval);
            exit(1);
        }

        mmt_vec_untwist(K.mmt, y);

        K.output(s, y);

        pi.m.serialize(__FILE__, __LINE__);

        // reached s + bw->interval. Count our time on cpu, and compute the sum.
        timing_disp_collective_oneline(pi, K.timing, s + bw->interval, K.tcan_print, "krylov");
        epoch_report(K.tcan_print, "krylov", s + bw->interval, bw->interval,
                t_iter, epoch.lap() - t_iter);
    }

    K.finish(pl, "krylov");

    return nullptr;
}

int main(int argc, char const * argv[])
{
    return krylov_main(argc, argv, krylov_prog);
}
