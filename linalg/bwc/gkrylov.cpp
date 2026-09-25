#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include "fmt/base.h"

#include "async.hpp"
#include "bw-common.hpp"
#include "krylov_common.hpp"
#include "macros.h"
#include "matmul_top.hpp"
#include "matmul_top_vec.hpp"
#include "mmt_vector_pair.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"
#include "xdotprod.hpp"

/* gkrylov is krylov for device-resident vectors.
 *
 * krylov twists the vector at the start of each interval and untwists
 * it at the end, so that the vector it checks and saves is in the
 * untwisted order. Each twist and untwist is a permutation of the whole
 * vector on the host, with two allreduces, and it pulls the vector off
 * the device and pushes it back.
 *
 * gkrylov twists the vector once, before the first interval, and keeps
 * it twisted for the whole run. At the end of each interval it checks
 * the twisted vector (the check vector is twisted along) and gathers
 * it as it is; the untwist is a fixed permutation of the gathered
 * items, computed once, that the writer thread applies to the V files.
 * The A and V files are identical to those of krylov. The permutation
 * needs the characteristic 2, so gkrylov is for GF(2) only.
 *
 * The line "gkrylov epoch" gives the wall time of the work between two
 * intervals, split by step, and the wall time of the interval itself
 * (iterations and epoch work) in milliseconds per iteration, so that a
 * run shows what the checkpoint costs next to the iterations.
 *
 * gkrylov iterates on two vector pairs: iteration i reads y[i & 1] and
 * writes y[!(i & 1)], through the matrix temporary of the pair it
 * reads. So no device operation of an iteration writes memory that
 * another operation of the same iteration reads, and the exchange
 * chain on the device orders its lanes by the data they carry alone.
 * The
 * interval is even, so that every interval starts and ends on the pair
 * that krylov_state loads, checks and saves.
 */

/* krylov_state::iterate over the pairs K.ymy and other: the dot
 * products of iteration i queue on the device as slot (i - q0) / 2 of
 * the pair it reads, from iteration q0 on, and the flush of each pair
 * fills every other block of xymats */
static void iterate(krylov_state & K, int s, mmt_vector_pair & other)
{
    mmt_vector_pair * const ymy[2] = { &K.ymy, &other };
    arith_generic * Ap = K.A.get();
    Ap->vec_set_zero(K.xymats, bw->m * bw->interval);
    K.pi.m.serialize(__FILE__, __LINE__);
    K.pi.interleaving_flip();

    int q0 = -1;
    auto x_flush = [&](int i) {
        if (q0 < 0) return;
        for(int j = q0 ; j < i && j < q0 + 2 ; j++)
            x_dotprod_flush(Ap->vec_subvec(K.xymats, j * bw->m), 2 * bw->m,
                    K.gxvecs, 0, bw->m, K.nx, (*ymy[j & 1])[0],
                    (i - j + 1) / 2);
        q0 = -1;
    };

    for(int i = 0 ; i < bw->interval ; i++) {
        mmt_vec & y = (*ymy[i & 1])[0];
        if (x_dotprod_queue(K.gxvecs, 0, bw->m, K.nx, y,
                    q0 < 0 ? 0 : (i - q0) / 2)) {
            if (q0 < 0) q0 = i;
        } else {
            x_flush(i);
            x_dotprod(Ap->vec_subvec(K.xymats, i * bw->m),
                    K.gxvecs, 0, bw->m, K.nx, y, 1);
        }

        matmul_top_mul_to(K.mmt, ymy[i & 1]->vectors(), (*ymy[!(i & 1)])[0], K.timing);
        timing_check(K.pi, K.timing, s + i + 1, K.tcan_print);
    }
    x_flush(bw->interval);

    K.pi.interleaving_flip();
    K.pi.interleaving_flip();
    K.pi.m.serialize(__FILE__, __LINE__);
}

static void * gkrylov_prog(parallelizing_info & pi, cxx_param_list & pl, void * arg MAYBE_UNUSED)
{
    krylov_state K(pi, pl);
    if (bw->interval & 1) {
        fmt::print(stderr, "gkrylov needs an even interval, not {}\n",
                bw->interval);
        exit(1);
    }
    mmt_vector_pair other(K.mmt, bw->dir);
    mmt_vec & y = K.ymy[0];

    pi.interleaving_flip();
    pi.interleaving_flip();

    K.keep_twisted();
    mmt_vec_twist(K.mmt, y);
    if (K.C) K.C->plan_ahead(y);

    epoch_timer epoch;
    for(int s = bw->start ; s < bw->end ; s += bw->interval ) {
        epoch_timer it;
        iterate(K, s, other);
        double const t_iter = it.lap();

        if (K.C && !K.C->verify(y, K.gxvecs, K.nx)) {
            fmt::print("Failed check at iteration {}\n", s + bw->interval);
            exit(1);
        }

        K.output(s, y);
        if (K.C) K.C->plan_ahead(y);

        pi.m.serialize(__FILE__, __LINE__);
        epoch_report(K.tcan_print, "gkrylov", s + bw->interval, bw->interval,
                t_iter, epoch.lap() - t_iter);
        timing_disp_collective_oneline(pi, K.timing, s + bw->interval,
                K.tcan_print, "gkrylov");
    }

    K.finish(pl, "gkrylov");
    return nullptr;
}

int main(int argc, char const * argv[])
{
    return krylov_main(argc, argv, gkrylov_prog);
}
