#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <ctime>

#include <gmp.h>
#include "fmt/base.h"
#include "fmt/format.h"

#include "bw-common.hpp"
#include "bwc_device_arena.hpp"
#include "bwc_filenames.hpp"
#include "macros.h"
#include "matmul_top_comm.hpp"
#include "misc.h"
#include "mksol_common.hpp"
#include "select_mpi.h"
#include "utils_cxx.hpp"

/* the solution range of this program, half of bw->solutions when two
 * interleaved programs share it */
static std::array<unsigned int, 2> my_solutions(parallelizing_info const & pi)
{
    unsigned int const s0 = bw->solutions[0];
    unsigned int const s1 = bw->solutions[1];
    if (!pi.interleaved)
        return { s0, s1 };
    ASSERT_ALWAYS((s1 - s0) % 2 == 0);
    unsigned int const h = (s1 - s0) / 2;
    return { s0 + pi.interleaved->idx * h, s0 + (pi.interleaved->idx + 1) * h };
}

/* one spmv per iteration constrains the number of solutions at a time */
static unsigned int solution_width(std::array<unsigned int, 2> const & solutions, int char2)
{
    unsigned int const w = solutions[1] - solutions[0];
    int const ok = char2 ? (w == 64 || w == 128 || w == 256) : w == 1;
    if (ok)
        return w;
    fmt::print(stderr,
            "We cannot support computing {} solutions at a time "
            "with one single Spmv operation, given the currently "
            "implemented code\n", w);
    exit(EXIT_FAILURE);
}

mksol_state::mksol_state(parallelizing_info & pi, cxx_param_list & pl)
    : pi(pi)
    , fake(pl.has("random_matrix") || pl.has("static_random_matrix"))
    , tcan_print(bw->can_print && pi.m.trank == 0)
    , solutions(my_solutions(pi))
    , Av_width(mpz_cmp_ui(bw->p, 2) == 0 ? 64 : 1)
    , Av_multiplex(bw->n / Av_width)
    , As_width(solution_width(solutions, Av_width == 64))
    , Af_multiplex(As_width / Av_width)
    , Av(arith_generic::instance(bw->p, Av_width))
    , Av_pi(pi.alloc_arith_datatype(Av.get()))
    , As(arith_generic::instance(bw->p, As_width))
    , AvxAs(arith_cross_generic::instance(Av.get(), As.get()))
    , mmt(As.get(), pi, pl, bw->dir)
    , ymy(mmt, bw->dir)
    , unpadded(MAX(mmt.n0[0], mmt.n0[1]))
    , one_fcoeff(As->vec_elt_stride(Av->simd_groupsize()))
{
    if (fake) bw->skip_online_checks = 1;

    /* the vectors we read from disk, n/64 of them in the binary case */
    matmul_top_matrix const & M = mmt.matrices[bw->dir ? (mmt.matrices.size() - 1) : 0];
    for(unsigned int i = 0 ; i < Av_multiplex ; i++) {
        vi.emplace_back(mmt, Av.get(), Av_pi, bw->dir, THREAD_SHARED_VECTOR, M.n[bw->dir]);
        mmt_full_vec_set_zero(vi.back());
    }

    pi.m.serialize(__FILE__, __LINE__);

    /* i/o stats, with generous interleaving protection */
    pi.interleaving_flip();
    pi.interleaving_flip();
    matmul_top_comm_bench(mmt, bw->dir);
    pi.interleaving_flip();
    pi.interleaving_flip();

    if (fake) {
        if (pi.m.trank == 0 && !bw->seed) {
            bw->seed = int(time(nullptr));
            MPI_Bcast(&bw->seed, 1, MPI_INT, 0, pi.m.pals);
        }
        pi.m.serialize_threads(__FILE__, __LINE__);
        gmp_randseed_ui(rstate, bw->seed);
        if (tcan_print)
            fmt::print("// Random generator seeded with {}\n", bw->seed);
    }

    unsigned int expected_last_iteration;
    pi.m.serialize_threads(__FILE__, __LINE__);
    if (pi.m.trank == 0) {
        /* the bw object is global ! */
        expected_last_iteration = bw_set_length_and_interval_mksol(bw, mmt.n0);
    }
    pi.m.thread_bcast(&expected_last_iteration, 1, BWC_PI_UNSIGNED, 0);
    pi.m.serialize_threads(__FILE__, __LINE__);
    if (bw->end == INT_MAX) {
        if (tcan_print)
            fmt::print ("Target iteration is unspecified ;"
                    " going to end of F file\n");
    } else {
        if (tcan_print)
            fmt::print ("Target iteration is {}\n", bw->end);
        expected_last_iteration = bw->end;
    }
    ASSERT_ALWAYS(bw->end == INT_MAX || bw->end % bw->interval == 0);
    end = block_end = bw->end;

    pi.interleaving_flip();
    if (bw->checkpoint_precious) {
        if (tcan_print) {
            fmt::print("As per interval={} checkpoint_precious={}, we'll load vectors every {} iterations, and print timings every {} iterations\n",
                    bw->interval,
                    bw->checkpoint_precious,
                    bw->checkpoint_precious,
                    bw->interval);
        }
    } else {
        bw->checkpoint_precious = bw->interval;
    }
    pi.interleaving_flip();

    /* F plays the role of a right-hand-side in mksol. Vector iterates
     * have to be multiplied on the right by a matrix corresponding to
     * some coefficient of F. Because F has been written to disk in the
     * "proper" order (namely: reversed order, leading coeff first) by
     * lingen, iterate i is to be multiplied by the i-th coefficient of F.
     *
     * The F-coefficients are square matrices having bw->n rows and
     * columns. Columns correspond to candidate solutions; we are
     * interested in the columns solutions[0]..solutions[1]. The bw->n
     * rows correspond to the Av_multiplex vectors: at each iteration,
     * we perform that many multiplications by coefficients of F. Each
     * of the Af_multiplex files of a coefficient holds Av_width
     * columns. We load the F coefficients for a full interval. */
    if (tcan_print) {
        char buf[20];
        fmt::print("Each thread allocates {}*{}*{}*{}*{}={} for the F matrices\n",
                Af_multiplex > 1 ? 2 : 1,
                Av_multiplex,
                1,
                one_fcoeff,
                bw->interval,
                size_disp(
                    (Af_multiplex > 1 ? 2 : 1) *
                    Av_multiplex *
                    one_fcoeff *
                    bw->interval, buf));
    }
    for(unsigned int i = 0 ; i < Av_multiplex ; i++) {
        fcoeffs.push_back(As->alloc(one_fcoeff * bw->interval, ALIGNMENT_ON_ALL_BWC_VECTORS));
        As->vec_set_zero(fcoeffs[i], one_fcoeff * bw->interval);
    }
    if (pi.m.trank == 0 && pi.m.jrank == 0) {
        for(unsigned int i = 0 ; i < Av_multiplex ; i++) {
            fnext.push_back(As->alloc(one_fcoeff * bw->interval, ALIGNMENT_ON_ALL_BWC_VECTORS));
            As->vec_set_zero(fnext[i], one_fcoeff * bw->interval);
        }
    }
    ASSERT_ALWAYS(Av_width * Af_multiplex == (unsigned int) As->simd_groupsize());
    if (Af_multiplex > 1) {
        fcoeff_tmp = As->alloc(one_fcoeff / Af_multiplex * bw->interval, ALIGNMENT_ON_ALL_BWC_VECTORS);
        As->vec_set_zero(fcoeff_tmp, one_fcoeff / Af_multiplex * bw->interval);
    }

    timing_init(timing, 4 * mmt.matrices.size(), bw->start, expected_last_iteration);
    for(size_t i = 0 ; i < mmt.matrices.size(); i++) {
        timing_set_timer_name(timing, 4*i, "CPU%zu", i);
        timing_set_timer_items(timing, 4*i, mmt.matrices[i].mm->ncoeffs);
        timing_set_timer_name(timing, 4*i+1, "cpu-wait%zu", i);
        timing_set_timer_name(timing, 4*i+2, "COMM%zu", i);
        timing_set_timer_name(timing, 4*i+3, "comm-wait%zu", i);
    }
}

mksol_state::~mksol_state()
{
    freader.wait();
    for(arith_generic::elt * ff : fcoeffs)
        As->free(ff);
    for(arith_generic::elt * ff : fnext)
        As->free(ff);
    if (fcoeff_tmp)
        As->free(fcoeff_tmp);
    pi.free_arith_datatype(Av_pi);
    timing_clear(timing);
}

void mksol_state::load(unsigned int s)
{
    pi.m.serialize(__FILE__, __LINE__);
    for(unsigned int i = 0 ; i < Av_multiplex ; i++) {
        auto pat = bwc_V_file::pattern(s);
        if (fake) {
            mmt_vec_set_random_through_file(vi[i], pat, unpadded, rstate, i * Av_width);
        } else {
            int const ok = mmt_vec_load(vi[i], pat, unpadded, i * Av_width);
            ASSERT_ALWAYS(ok);
        }
    }
    begin_block(s);
}

void mksol_state::begin_block(unsigned int s)
{
    for(mmt_vec & v : vi)
        mmt_vec_twist(mmt, v);
    pi.m.serialize(__FILE__, __LINE__);

    mmt_full_vec_set_zero(ymy[0]);

    block_end = MIN(end, s + bw->checkpoint_precious);
    if (tcan_print)
        fmt::print("about to do {} iterations starting from vectors at iteration {} to handle the coefficients of degree [{}..{}] in F\n", block_end-s-1, s, s, block_end-1);
}

/* the coefficients [s0, s0 + interval) of one F file, column block k
 * of the V vector i, into ff; returns the number of coefficients read */
static int read_f_file(mksol_state const & M, unsigned int i, unsigned int k,
        unsigned int s0, arith_generic::elt * ff)
{
    unsigned int const sol0 = M.solutions[0] + k * M.Av_width;
    std::string const name = fmt::format("F.sols{}-{}.{}-{}",
            sol0, sol0 + M.Av_width, i * M.Av_width, (i + 1) * M.Av_width);
    size_t const cbytes = M.one_fcoeff / M.Af_multiplex;
    auto f = fopen_helper(name, "rb");
    if (fseek(f.get(), cbytes * s0, SEEK_SET) < 0)
        return 0;
    /* the coefficients of one file are contiguous in ff only when it
     * is the only file of the coefficient */
    arith_generic::elt * buffer = M.Af_multiplex == 1 ? ff : M.fcoeff_tmp;
    int const rc = (int) fread(buffer, cbytes, bw->interval, f.get());
    ASSERT_ALWAYS(rc <= bw->interval);
    if (M.Af_multiplex == 1)
        return rc;
    arith_generic const * Av = M.Av.get();
    const arith_generic::elt * src = buffer;
    arith_generic::elt * dst = Av->vec_subvec(ff, k);
    for(unsigned int row = 0 ; row < M.Av_width * rc ; row++) {
        memcpy(dst, src, Av->elt_stride());
        src = Av->vec_subvec(src, 1);
        dst = Av->vec_subvec(dst, M.Af_multiplex);
    }
    return rc;
}

/* the leader reads the coefficients [s0, s0 + interval) of every F
 * file into fc, zero past the end of the files; returns the
 * number of coefficients read */
static unsigned int read_f_leader(mksol_state & M, unsigned int s0,
        std::vector<arith_generic::elt *> const & fc)
{
    int n = -1;
    for(unsigned int i = 0 ; i < M.Av_multiplex ; i++) {
        arith_generic::elt * ff = fc[i];
        M.As->vec_set_zero(ff, M.one_fcoeff * bw->interval);
        for(unsigned int k = 0 ; k < M.Af_multiplex ; k++) {
            int const rc = read_f_file(M, i, k, s0, ff);
            if (n < 0)
                n = rc;
            if (rc == n)
                continue;
            fmt::print(stderr, "Inconsistency in number of "
                    "coefficients for F files\n");
            exit(EXIT_FAILURE);
        }
    }
    return n;
}

void mksol_state::start_read_f(unsigned int s0)
{
    if (pi.m.trank || pi.m.jrank || s0 >= end)
        return;
    fnext_s = s0;
    freader.run([this, s0] { fnext_n = read_f_leader(*this, s0, fnext); });
}

unsigned int mksol_state::read_f(unsigned int s0, unsigned int s1)
{
    if (s0 >= end)
        return s0;
    pi.m.serialize(__FILE__, __LINE__);
    unsigned int n = 0;
    if (pi.m.trank == 0 && pi.m.jrank == 0) {
        freader.wait();
        if (fnext_s == s0) {
            std::swap(fcoeffs, fnext);
            fnext_s = UINT_MAX;
            n = fnext_n;
        } else {
            n = read_f_leader(*this, s0, fcoeffs);
        }
    }
    pi.m.bcast(&n, 1, BWC_PI_UNSIGNED, 0, 0);
    if (n < (unsigned int) bw->interval) {
        /* a short read is only fine for the last coefficients */
        if (s1 != block_end) {
            fmt::print(stderr, "Problem while reading coefficients of f for degrees [{}..{}[ ; we should not have a short read given that bw->end={}\n",
                    s0, s1, end);
            exit(EXIT_FAILURE);
        }
        end = block_end = s0 + n;
        if (tcan_print)
            fmt::print("We read {} coefficients from F in total\n", end);
    }
    s1 = MIN(end, s1);
    for(unsigned int i = 0 ; i < Av_multiplex ; i++)
        pi.m.bcast(fcoeffs[i], Av->simd_groupsize() * (s1 - s0), mmt.pitype, 0, 0);
    pi.m.serialize(__FILE__, __LINE__);
    return s1;
}

const arith_generic::elt * mksol_state::f(unsigned int i, unsigned int c) const
{
    return As->vec_subvec(fcoeffs[i], c * Av->simd_groupsize());
}

int mksol_state::dev_combine(mmt_vec & y, unsigned int c, size_t item0, size_t items)
{
    if (As_width != 64)
        return 0;
    std::vector<const void *> v(Av_multiplex), vbase(Av_multiplex);
    std::vector<size_t> vbytes(Av_multiplex);
    std::vector<const uint64_t *> fc(Av_multiplex);
    for(unsigned int i = 0 ; i < Av_multiplex ; i++) {
        mmt_vec const & vi_ = vi[i];
        v[i] = Av->vec_subvec(vi_.v, item0);
        vbase[i] = vi_.v;
        vbytes[i] = Av->vec_elt_stride(vi_.i1 - vi_.i0);
        fc[i] = (const uint64_t *) f(i, c);
    }
    return bwc_dev_addmul64(As->vec_subvec(y.v, item0),
            As->vec_elt_stride(items), Av_multiplex, v.data(),
            vbase.data(), vbytes.data(), fc.data());
}

void mksol_combine(mksol_state & M, mmt_vec & y, unsigned int c)
{
    M.pi.m.serialize_threads(__FILE__, __LINE__);

    size_t const eblock = mmt_my_own_size_in_items(y);
    arith_generic::elt * const yown = mmt_my_own_subvec(y);
    size_t const ybytes = M.As->vec_elt_stride(eblock);
    if (!M.dev_combine(y, c, mmt_my_own_offset_in_items(y), eblock)) {
        bwc_dev_fetch(yown, ybytes);
        for(unsigned int i = 0 ; i < M.Av_multiplex ; i++) {
            arith_generic::elt const * vown = mmt_my_own_subvec(M.vi[i]);
            bwc_dev_fetch(vown, M.Av->vec_elt_stride(eblock));
            M.AvxAs->addmul_tiny(yown, vown, M.f(i, c), eblock);
        }
        bwc_dev_push(yown, ybytes);
    }
    /* addmul_tiny degrades consistency ! */
    y.consistency = 1;
    /* the broadcast makes MPI calls, so it needs an interleaving block
     * of its own */
    M.pi.interleaving_flip();
    mmt_vec_broadcast(y);
    M.pi.interleaving_flip();
    /* we write on the next input vector: a race with the forthcoming
     * spmv, so we serialize */
    M.pi.m.serialize_threads(__FILE__, __LINE__);
}

mmt_vec & mksol_state::iterate(unsigned int s, mmt_vector_pair & other,
        void (*combine)(mksol_state &, mmt_vec &, unsigned int),
        const char * stage)
{
    mmt_vector_pair * const pairs[2] = { &ymy, &other };
    unsigned int j = 0;
    int const n_windows = bw->checkpoint_precious / bw->interval;
    for(int w = n_windows - 1 ; w >= 0 ; w--) {
        unsigned int const s0 = s + w * bw->interval;
        unsigned int const s1 = read_f(s0, s0 + bw->interval);
        start_read_f(w ? s0 - bw->interval : s + 2 * bw->checkpoint_precious - bw->interval);
        if (s1 == s0)
            continue;

        pi.interleaving_flip();
        for(unsigned int k = 0 ; k < s1 - s0 ; k++) {
            combine(*this, (*pairs[j & 1])[0], s1 - s0 - 1 - k);
            /* no product after the last coefficient of the block */
            if (w == 0 && k == s1 - s0 - 1)
                break;
            matmul_top_mul_to(mmt, pairs[j & 1]->vectors(), (*pairs[!(j & 1)])[0], timing);
            j++;
            timing_check(pi, timing, s + block_end - s1 + k + 1, tcan_print);
        }
        pi.interleaving_flip();
        pi.interleaving_flip();
        pi.m.serialize(__FILE__, __LINE__);

        timing_disp_collective_oneline(pi, timing, s + block_end - s0, tcan_print, stage);
    }
    return (*pairs[j & 1])[0];
}

std::string mksol_state::s_pattern(unsigned int s) const
{
    bwc_iteration_range const r { s, s + bw->checkpoint_precious };
    return bwc_S_file::pattern(r);
}

void mksol_state::finish(const char * stage)
{
    freader.wait();
    writer.wait();
    timing->end_mark = bw->start + bw->interval * iceildiv(end - bw->start, bw->interval);
    timing_final_tally(pi, timing, tcan_print, stage);
    if (tcan_print)
        fmt::print("Done {}.\n", stage);
    pi.m.serialize(__FILE__, __LINE__);
}

// coverity[root_function]
int mksol_main(int argc, char const * argv[],
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
    /* interpret our parameters: none here (so far). */

    ASSERT_ALWAYS(!pl.has("ys"));
    ASSERT_ALWAYS(pl.has("solutions"));

    if (pl.warn_unused()) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (!rank) pl.print_usage(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    pi_go(*prog, pl, nullptr);

    parallelizing_info::clear_attribute_things();

    bw_common_clear(bw);

    return 0;
}
