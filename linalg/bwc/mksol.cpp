#include "cado.h" // IWYU pragma: keep

#include "bw-common.hpp"
#include "macros.h"
#include "matmul_top.hpp"
#include "matmul_top_vec.hpp"
#include "mksol_common.hpp"
#include "mmt_vector_pair.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"

static void * mksol_prog(parallelizing_info & pi, cxx_param_list & pl, void * arg MAYBE_UNUSED)
{
    mksol_state M(pi, pl);
    mmt_vector_pair other(M.mmt, bw->dir);

    pi.interleaving_flip();
    pi.interleaving_flip();

    epoch_timer epoch;
    for(unsigned int s = bw->start ; s < M.end ; s += bw->checkpoint_precious) {
        M.load(s);
        epoch_timer it;
        mmt_vec & y = M.iterate(s, other, mksol_combine, "mksol");
        double const t_iter = it.lap();
        mmt_vec_untwist(M.mmt, y);
        if (!M.fake)
            mmt_vec_save(y, M.s_pattern(s), M.unpadded, M.solutions[0]);
        epoch_report(M.tcan_print, "mksol", M.block_end, M.block_end - s,
                t_iter, epoch.lap() - t_iter);
    }

    M.finish("mksol");

    return nullptr;
}

int main(int argc, char const * argv[])
{
    return mksol_main(argc, argv, mksol_prog);
}
