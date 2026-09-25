#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <gmp.h>
#include "fmt/base.h"
#include "fmt/format.h"

#include "arith-cross.hpp"
#include "arith-generic.hpp"
#include "async.hpp"
#include "bw-common.hpp"
#include "bwc_file.hpp"
#include "bwc_filenames.hpp"
#include "gmp_aux.h"
#include "krylov_common.hpp"
#include "macros.h"
#include "matmul_top.hpp"
#include "matmul_top_vec.hpp"
#include "misc.h"
#include "mmt_vector_pair.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"
#include "rolling.hpp"
#include "select_mpi.h"
#include "timing.h"
#include "utils_cxx.hpp"
#include "xdotprod.hpp"
#include "xvectors.hpp"

check_data::check_data(matmul_top_data & mmt, arith_generic * A)
    : mmt(mmt)
    , pi(mmt.pi)
    , nchecks(mpz_cmp_ui(bw->p, 2) > 0 ? NCHECKS_CHECK_VECTOR_GFp : NCHECKS_CHECK_VECTOR_GF2)
    , A(A)
    , Ac(arith_generic::instance(bw->p, nchecks))
    , Ac_pi(pi.alloc_arith_datatype(Ac.get()))
    , AxAc(arith_cross_generic::instance(A, Ac.get()))
    , check_vector(mmt, Ac.get(), Ac_pi,
              bw->dir, THREAD_SHARED_VECTOR, mmt.n[bw->dir])
    , tcan_print(bw->can_print && pi.m.trank == 0)
{
}

void check_data::load()
{
    /* We do the dot product by working on the local vector chunks.
     * Therefore, we must really understand the check vector as
     * playing a role in the very same direction of the y vector!
     */
    std::string const Cv_filename_pattern = bwc_Cv_file::pattern(bw->interval);
    const int ok = mmt_vec_load(check_vector, Cv_filename_pattern, mmt.n0[bw->dir], 0);
    if (!ok) {
        if (tcan_print)
            fmt::print(stderr, "check file {} not found", Cv_filename_pattern);
        pi.m.abort(EXIT_FAILURE);
    }
    std::string const Ct_filename = fmt::format("Ct0-{}.0-{}", nchecks, bw->m);
    Tdata = Ac->alloc(bw->m, ALIGNMENT_ON_ALL_BWC_VECTORS);
    if (pi.m.trank == 0 && pi.m.jrank == 0) {
        auto Tfile = fopen_helper(Ct_filename, "rb");
        size_t const rc = fread(Tdata, Ac->vec_elt_stride(bw->m), 1, Tfile.get());
        ASSERT_ALWAYS(rc == 1);
    }
    if (tcan_print) fmt::print("loaded {}\n", Ct_filename);
    pi.m.bcast(Tdata, bw->m, Ac_pi, 0, 0);

    ahead = A->alloc(nchecks, ALIGNMENT_ON_ALL_BWC_VECTORS);
}

check_data::~check_data()
{
    A->free(ahead);
    Ac->free(Tdata);
    pi.free_arith_datatype(Ac_pi);
}

void check_data::plan_ahead(mmt_vec const & y)
{
    // Plan ahead. The check vector is here to predict the final A matrix.
    // Note that our share of the dot product is determined by the
    // intersections of the i0..i1 intervals on both sides.

    /* Note that the check vector is always stored untwisted in
     * memory */

    /* create a matrix of size nchecks * nbys with the dot
     * product with Cv -- in the case where nbys != nchecks,
     * dealing with that data will require some care.
     *
     */
    A->vec_set_zero(ahead, nchecks);
    arith_generic::elt const * my = mmt_my_own_subvec(y);
    arith_generic::elt const * mc = mmt_my_own_subvec(check_vector);
    size_t const n = mmt_my_own_size_in_items(y);
    size_t const bytes = A->vec_elt_stride(n);
    if (Ac->vec_elt_stride(1) == sizeof(uint64_t) &&
            bwc_dev_dotprod64(my, bytes, A->vec_elt_stride(1),
                (const uint64_t *) mc, (uint64_t *) ahead))
        return;
    bwc_dev_fetch(my, bytes);
    /* The syntax of ->dotprod is a bit weird. We compute
     * transpose(data-operand-0)*data-operand1, but data-operand0
     * (check_vector here) actually refers to field-operand1 (Ac
     * here).
     */
    AxAc->add_dotprod(ahead, my, mc, n);
}

bool check_data::verify(mmt_vec const & y,
        std::vector<uint32_t> const & gxvecs, int nx)
{
    /* Last dot product. This must cancel ! */
    arith_generic::elt * tmp1 = A->alloc(nchecks, ALIGNMENT_ON_ALL_BWC_VECTORS);
    for(int c = 0 ; c < bw->m ; c += nchecks) {
        /* First zero out the matrix of size nchecks * nbys.  */
        A->vec_set_zero(tmp1, nchecks);
        x_dotprod(tmp1, gxvecs, c, c + nchecks, nx, y, -1);
        /* And now compute the product transpose(part of
         * T)*ahead_tmp, and subtract that from our check value
         */
        AxAc->add_dotprod(
                ahead,
                tmp1,
                Ac->vec_subvec(Tdata, c),
                nchecks);
    }
    A->free(tmp1);

    pi.m.allreduce(nullptr, ahead, nchecks, mmt.pitype, BWC_PI_SUM);
    return A->vec_is_zero(ahead, nchecks);
}

/* an interleaved run splits the ys range between its two halves */
static std::array<int, 2> krylov_ys(parallelizing_info & pi)
{
    std::array<int, 2> ys { bw->ys[0], bw->ys[1] };
    if (!pi.interleaved)
        return ys;
    ASSERT_ALWAYS((bw->ys[1]-bw->ys[0]) % 2 == 0);
    ys[0] = bw->ys[0] + pi.interleaved->idx * (bw->ys[1]-bw->ys[0])/2;
    ys[1] = ys[0] + (bw->ys[1]-bw->ys[0])/2;
    return ys;
}

static int krylov_fake(cxx_param_list & pl)
{
    int const fake = pl.has("random_matrix") || pl.has("static_random_matrix");
    if (fake) bw->skip_online_checks = 1;
    return fake;
}

krylov_state::krylov_state(parallelizing_info & pi, cxx_param_list & pl)
    : pi(pi)
    , fake(krylov_fake(pl))
    , tcan_print(bw->can_print && pi.m.trank == 0)
    , ys(krylov_ys(pi))
    , A(arith_generic::instance(bw->p, ys[1]-ys[0]))
    , mmt(A.get(), pi, pl, bw->dir)
    , C(bw->skip_online_checks ? nullptr : std::make_shared<check_data>(mmt, A.get()))
    , ymy(mmt, bw->dir)
    , unpadded(MAX(mmt.n0[0], mmt.n0[1]))
    , gathered(unpadded)
    , gather_comm(MPI_COMM_NULL)
    , threaded_mpi(bw_common_threaded_mpi())
{
    pi.m.serialize(__FILE__, __LINE__);
    if (pi.m.trank == 0) MPI_Comm_dup(pi.m.pals, &gather_comm);
    if (tcan_print && !threaded_mpi)
        fmt::print("MPI serves one thread at a time: the checkpoint gather runs on the main thread\n");

    if (!fake) {
        gxvecs = load_x(bw->m, nx, pi);
    } else {
        gxvecs = set_x_fake(bw->m, nx, pi);
    }
    indices_twist(mmt, gxvecs, bw->dir);

    /* let's be generous with interleaving protection. I don't want to be
     * bothered, really */
    pi.interleaving_flip();
    pi.interleaving_flip();
    matmul_top_comm_bench(mmt, bw->dir);
    pi.interleaving_flip();
    pi.interleaving_flip();

    /* I have absolutely no idea why, but the two --apparently useless--
     * serializing calls around the next block seem to have a beneficial
     * impact on the SEGv's we see every now and then with --mca
     * mpi_leave_pinned 1
     */
    pi.m.serialize(__FILE__, __LINE__);
    if (!fake) {
        int const ok = mmt_vec_load(ymy[0], bwc_V_file::pattern(bw->start), unpadded, ys[0]);
        ASSERT_ALWAYS(ok);
    } else {
        cxx_gmp_randstate rstate;
        unsigned long const g = pi.m.jrank * pi.m.ncores + pi.m.trank;
        gmp_randseed_ui(rstate, bw->seed + g);
        mmt_vec_set_random_inconsistent(ymy[0], rstate);
        mmt_vec_truncate(mmt, ymy[0]);
    }
    pi.m.serialize(__FILE__, __LINE__);

    pi.m.serialize_threads(__FILE__, __LINE__);
    if (pi.m.trank == 0) {
        /* the bw object is global ! */
        bw_set_length_and_interval_krylov(bw, mmt.n0);
    }
    pi.m.serialize_threads(__FILE__, __LINE__);
    if (tcan_print) {
        fmt::print("Target iteration is {}\n", bw->end);
    }
    ASSERT_ALWAYS(bw->end % bw->interval == 0);

    if (C) C->load();

    /* We'll store all xy matrices locally before doing reductions. Given
     * the small footprint of these matrices, it's rather innocuous.
     */
    if (tcan_print) {
        fmt::print("Each thread allocates {} kb for the A matrices\n",
                A->vec_elt_stride(bw->m*bw->interval) >> 10);
    }
    xymats = A->alloc(bw->m*bw->interval, ALIGNMENT_ON_ALL_BWC_VECTORS);

    timing_init(timing, 4 * mmt.matrices.size(), bw->start, bw->end);

    for(size_t i = 0 ; i < mmt.matrices.size(); i++) {
        timing_set_timer_name(timing, 4*i, "CPU%zu", i);
        timing_set_timer_items(timing, 4*i, mmt.matrices[i].mm->ncoeffs);
        timing_set_timer_name(timing, 4*i+1, "cpu-wait%zu", i);
        timing_set_timer_name(timing, 4*i+2, "COMM%zu", i);
        timing_set_timer_name(timing, 4*i+3, "comm-wait%zu", i);
    }
}

krylov_state::~krylov_state()
{
    writer.wait();
    if (pi.m.trank == 0) MPI_Comm_free(&gather_comm);
    A->free(xymats);
    timing_clear(timing);
}

void krylov_state::iterate(int s)
{
    arith_generic * Ap = A.get();
    mmt_vec & y = ymy[0];

    Ap->vec_set_zero(xymats, bw->m*bw->interval);
    pi.m.serialize(__FILE__, __LINE__);
    pi.interleaving_flip();

    /* While the vector stays on the device, the dot products are
     * queued there (starting at iteration q0) and read back in one
     * batch. */
    int q0 = -1;
    auto x_flush = [&](int i) {
        if (q0 < 0) return;
        x_dotprod_flush(Ap->vec_subvec(xymats, q0 * bw->m), bw->m,
                gxvecs, 0, bw->m, nx, y, i - q0);
        q0 = -1;
    };

    for(int i = 0 ; i < bw->interval ; i++) {
        if (x_dotprod_queue(gxvecs, 0, bw->m, nx, y, q0 < 0 ? 0 : i - q0)) {
            if (q0 < 0) q0 = i;
        } else {
            x_flush(i);
            x_dotprod(Ap->vec_subvec(xymats, i * bw->m),
                    gxvecs, 0, bw->m, nx, y, 1);
        }

        matmul_top_mul(mmt, ymy.vectors(), timing);
        timing_check(pi, timing, s+i+1, tcan_print);
    }
    x_flush(bw->interval);

    pi.interleaving_flip();
    pi.interleaving_flip();
    pi.m.serialize(__FILE__, __LINE__);
}

void krylov_state::keep_twisted()
{
    untwist = mmt_untwist_perm(mmt, ymy[0]);
    gathered = ymy[0].n;
    if (C) mmt_vec_twist(mmt, C->check_vector);
}

void krylov_state::output(int s, mmt_vec & y)
{
    /* Now (and only now) collect the xy matrices */
    pi.m.allreduce(nullptr, xymats,
            bw->m * bw->interval,
            mmt.pitype, BWC_PI_SUM);

    if (fake)
        return;

    unsigned int const v = s + bw->interval;
    mmt_vec_staged st = mmt_vec_stage(y, bwc_V_file::pattern(v),
            gathered, ys[0]);

    if (pi.m.trank)
        return;

    bwc_file a { fmt::format("A{}-{}.{}-{}", ys[0], ys[1], s, v), {} };
    if (pi.m.jrank == 0) {
        auto const * p = (const char *) xymats;
        a.data.assign(p, p + A->vec_elt_stride(bw->m*bw->interval));
    }
    /* the exchange of the staged pieces runs on every job, in the writer
     * thread while the main thread iterates when MPI allows it; the
     * leader then untwists and writes in the writer thread */
    auto exchange = [this, st = std::move(st)] () mutable {
        return mmt_vec_exchange(pi, gather_comm, std::move(st));
    };
    auto write = [this, a = std::move(a), v] (std::vector<bwc_file> files) mutable {
        if (pi.m.jrank)
            return;
        if (!untwist.empty())
            for(auto & f : files)
                mmt_vec_untwist_file(f, untwist, unpadded);
        files.insert(files.begin(), std::move(a));
        bwc_write_files(files);
        keep_rolling_checkpoints(fmt::format("V{}-{}", ys[0], ys[1]), v);
    };
    if (threaded_mpi) {
        writer.run([exchange = std::move(exchange), write = std::move(write)] () mutable {
            write(exchange());
        });
        return;
    }
    writer.run([files = exchange(), write = std::move(write)] () mutable {
        write(std::move(files));
    });
}

void krylov_state::finish(cxx_param_list & pl, const char * stage)
{
    writer.wait();
    timing_final_tally(pi, timing, tcan_print, stage);

    if (tcan_print)
        fmt::print("Done {}.\n", stage);
    pi.m.serialize(__FILE__, __LINE__);

    int want_full_report = 0;
    pl.parse("full_report", want_full_report);
    matmul_top_report(mmt, 1.0, want_full_report);
}

// coverity[root_function]
int krylov_main(int argc, char const * argv[],
        void * (*prog)(parallelizing_info &, cxx_param_list &, void *))
{
    cxx_param_list pl;

    bw_common_init(bw, &argc, &argv, MPI_THREAD_MULTIPLE);

    parallelizing_info::init_attribute_things();

    bw_common_decl_usage(pl);
    parallelizing_info::declare_usage(pl);
    matmul_top_decl_usage(pl);
    /* declare local parameters and switches: none here (so far). */

    bw_common_parse_cmdline(bw, pl, &argc, &argv);

    bw_common_interpret_parameters(bw, pl);
    parallelizing_info::lookup_parameters(pl);
    matmul_top_lookup_parameters(pl);
    /* interpret our parameters */
    if (bw->ys[0] < 0) { fmt::print(stderr, "no ys value set\n"); exit(1); }

    ASSERT_ALWAYS(pl.has("ys"));
    ASSERT_ALWAYS(!pl.has("solutions"));

    if (pl.warn_unused()) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (!rank) pl.print_usage(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    pi_go(*prog, pl, 0);

    parallelizing_info::clear_attribute_things();

    bw_common_clear(bw);

    return 0;
}
