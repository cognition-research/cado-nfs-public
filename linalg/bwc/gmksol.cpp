#include "cado.h" // IWYU pragma: keep

/* gmksol is mksol for a matrix product that runs on a device. The
 * combine of the V vectors into the iterate runs on the device over the
 * whole vector, so the iterate stays consistent and the per-iteration
 * broadcast of mksol disappears. As in gkrylov, the iterations
 * alternate between two vector pairs, so no device operation of an
 * iteration writes memory that another operation of the same iteration
 * reads, and the iterate is never untwisted: the untwist is a fixed
 * permutation of the gathered items, computed once, that the writer
 * thread applies to the S files while the next block runs. The files
 * are identical to those of mksol. The V files of a block are read
 * during the previous block, each rank its own piece, and the F
 * coefficients of a window are read during the previous window. */

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "fmt/base.h"
#include "fmt/format.h"

#include "bw-common.hpp"
#include "bwc_device_arena.hpp"
#include "bwc_file.hpp"
#include "bwc_filenames.hpp"
#include "macros.h"
#include "matmul_top.hpp"
#include "matmul_top_comm.hpp"
#include "matmul_top_vec.hpp"
#include "mksol_common.hpp"
#include "mmt_vector_pair.hpp"
#include "parallelizing_info.hpp"
#include "params.hpp"
#include "select_mpi.h"
#include "timing.h"

/* The pieces of the V files of one block that this thread owns, read
 * straight from the working directory in the background while the
 * previous block runs (mmt_vec_load has the leader read the whole
 * file and send the pieces, between two blocks). The working
 * directory is therefore shared by the ranks. A piece that cannot be
 * read aborts the program when its block is placed: the read of the
 * block after the last one (the end is only known once the F files
 * run short) may fail harmlessly. */
struct v_prefetch {
    mksol_state & M;
    std::vector<std::vector<char>> pieces;  /* one per V vector */
    unsigned int s = UINT_MAX;              /* the block of the pieces */
    double seconds = 0;
    std::string error;
    bwc_background reader;

    explicit v_prefetch(mksol_state & M)
        : M(M)
        , pieces(M.Av_multiplex, std::vector<char>(mmt_my_own_size_in_bytes(M.vi[0])))
    {
        ASSERT_ALWAYS(M.Av->is_characteristic_two());
    }

    std::string name(unsigned int i) const
    {
        return fmt::format(fmt::runtime(bwc_V_file::pattern(s)),
                i * M.Av_width, (i + 1) * M.Av_width);
    }

    void start(unsigned int s0)
    {
        s = s0;
        reader.run([this] {
            seconds = -wct_seconds();
            error.clear();
            for(unsigned int i = 0 ; i < M.Av_multiplex && error.empty() ; i++)
                read(i);
            seconds += wct_seconds();
        });
    }

    void read(unsigned int i)
    {
        mmt_vec const & v = M.vi[i];
        size_t const item = v.abase->vec_elt_stride(1);
        size_t const off = (v.i0 + mmt_my_own_offset_in_items(v)) * item;
        size_t const ondisk = M.unpadded * item;
        std::vector<char> & p = pieces[i];
        size_t const want = off < ondisk ? MIN(p.size(), ondisk - off) : 0;
        memset(p.data() + want, 0, p.size() - want);
        int const fd = open(name(i).c_str(), O_RDONLY);
        size_t done = 0;
        while (fd >= 0 && done < want) {
            ssize_t const r = pread(fd, p.data() + done, want - done, off + done);
            if (r == 0)
                errno = EIO;
            if (r <= 0)
                break;
            done += r;
        }
        if (fd < 0 || done < want)
            error = fmt::format("failed to load {}: {}", name(i), strerror(errno));
        if (fd >= 0)
            close(fd);
    }

    /* the pieces of block s0 into the own ranges of vi, then the
     * vectors complete on every rank, as after mmt_vec_load, and on
     * the device, so that the twist of the block runs there */
    void place(unsigned int s0)
    {
        ASSERT_ALWAYS(s0 == s);
        reader.wait();
        if (!error.empty()) {
            fmt::print(stderr, "ERROR: {}\n", error);
            abort();
        }
        for(unsigned int i = 0 ; i < M.Av_multiplex ; i++) {
            mmt_vec & v = M.vi[i];
            memcpy(mmt_my_own_subvec(v), pieces[i].data(), pieces[i].size());
            mmt_vec_dev_invalidate(v);
            v.consistency = 1;
            mmt_vec_broadcast(v);
            mmt_vec_dev_adopt(v);
            v.pi.m.serialize_threads(__FILE__, __LINE__);
        }
        if (M.tcan_print)
            fmt::print("Loaded the {} V files of iteration {}, read ahead in {:.2f}\n",
                    M.Av_multiplex, s, seconds);
    }
};

/* y += sum_i vi[i] * f(i, c) on the device, over the whole range
 * of the vector: every job of the wiring computes the same result, so
 * no broadcast follows. Thread 0 of the wiring issues the work of the
 * shared vector before the other threads queue their products. An
 * iterate without a device mirror (see warm_up) combines on the host. */
static void dev_combine(mksol_state & M, mmt_vec & y, unsigned int c)
{
    bwc_device_arena const * a = bwc_dev_find(y.v);
    if (M.As_width != 64 || !a || a->state != BWC_ARENA_DEV) {
        mksol_combine(M, y, c);
        return;
    }
    pi_comm & wr = y.pi.wr[y.d];
    if (wr.trank == 0)
        ASSERT_ALWAYS(M.dev_combine(y, c, 0, y.i1 - y.i0));
    wr.serialize_threads(__FILE__, __LINE__);
}

/* Two products of the zero vector: the matmul backend creates the
 * device mirror of an iterate on its first product as a source, and the
 * combines that precede the first real products then run on the device
 * too. The V vectors get their mirrors next to the iterate, so that
 * every block places them on the device (see v_prefetch::place). */
static void warm_up(mksol_state & M, mmt_vector_pair & other)
{
    mmt_full_vec_set_zero(M.ymy[0]);
    M.pi.interleaving_flip();
    matmul_top_mul_to(M.mmt, M.ymy.vectors(), other[0], nullptr);
    matmul_top_mul_to(M.mmt, other.vectors(), M.ymy[0], nullptr);
    M.pi.interleaving_flip();
    M.pi.interleaving_flip();
    M.pi.m.serialize(__FILE__, __LINE__);
    for(mmt_vec & v : M.vi)
        mmt_vec_dev_mirror(v, M.ymy[0]);
}

/* The S file of a block, gathered and written while the next block
 * runs. The iterate stays twisted for the whole run, so the untwist is
 * a fixed permutation of the gathered items that the writer thread
 * applies to the file bytes, and the stage gathers the padded items.
 * The exchange of the staged pieces runs on every job, in the writer
 * thread while the main thread iterates when MPI allows it; the leader
 * then untwists and writes in the writer thread. */
struct s_writer {
    mksol_state & M;
    std::vector<uint32_t> untwist;      /* at the leader */
    MPI_Comm gather_comm = MPI_COMM_NULL;
    bool threaded_mpi;

    explicit s_writer(mksol_state & M)
        : M(M)
        , untwist(mmt_untwist_perm(M.mmt, M.ymy[0]))
        , threaded_mpi(bw_common_threaded_mpi())
    {
        M.pi.m.serialize(__FILE__, __LINE__);
        if (M.pi.m.trank == 0)
            MPI_Comm_dup(M.pi.m.pals, &gather_comm);
        if (M.tcan_print && !threaded_mpi)
            fmt::print("MPI serves one thread at a time: the S gather runs on the main thread\n");
    }

    ~s_writer()
    {
        M.writer.wait();
        if (M.pi.m.trank == 0)
            MPI_Comm_free(&gather_comm);
    }

    void output(unsigned int s, mmt_vec & y)
    {
        mmt_vec_staged st = mmt_vec_stage(y, M.s_pattern(s), y.n, M.solutions[0]);
        if (M.pi.m.trank)
            return;
        auto exchange = [this, st = std::move(st)] () mutable {
            return mmt_vec_exchange(M.pi, gather_comm, std::move(st));
        };
        auto write = [this] (std::vector<bwc_file> files) {
            if (M.pi.m.jrank)
                return;
            for(auto & f : files)
                mmt_vec_untwist_file(f, untwist, M.unpadded);
            bwc_write_files(files);
        };
        if (threaded_mpi) {
            M.writer.run([exchange = std::move(exchange), write = std::move(write)] () mutable {
                write(exchange());
            });
            return;
        }
        M.writer.run([files = exchange(), write = std::move(write)] () mutable {
            write(std::move(files));
        });
    }
};

static void * gmksol_prog(parallelizing_info & pi, cxx_param_list & pl, void * arg MAYBE_UNUSED)
{
    mksol_state M(pi, pl);
    mmt_vector_pair other(M.mmt, bw->dir);
    ASSERT_ALWAYS(mmt_vec_is_shared(M.ymy[0]));
    ASSERT_ALWAYS(mmt_vec_is_shared(other[0]));

    pi.interleaving_flip();
    pi.interleaving_flip();
    warm_up(M, other);

    v_prefetch V(M);
    s_writer S(M);
    if (!M.fake)
        V.start(bw->start);
    epoch_timer epoch;
    for(unsigned int s = bw->start ; s < M.end ; s += bw->checkpoint_precious) {
        epoch_timer it;
        double t_place = 0, t_twist = 0;
        if (M.fake) {
            M.load(s);
        } else {
            V.place(s);
            t_place = it.lap();
            M.begin_block(s);
            t_twist = it.lap();
            if (s + bw->checkpoint_precious < M.end)
                V.start(s + bw->checkpoint_precious);
        }
        it.lap();
        mmt_vec & y = M.iterate(s, other, dev_combine, "gmksol");
        double const t_iter = it.lap();
        if (!M.fake)
            S.output(s, y);
        double const t_stage = it.lap();
        epoch_report(M.tcan_print, "gmksol", M.block_end, M.block_end - s,
                t_iter, epoch.lap() - t_iter);
        if (M.tcan_print)
            fmt::print("gmksol block {}: V place {:.3f} s, twist {:.3f} s, S stage {:.3f} s\n",
                    s, t_place, t_twist, t_stage);
    }

    M.finish("gmksol");

    return nullptr;
}

int main(int argc, char const * argv[])
{
    return mksol_main(argc, argv, gmksol_prog);
}
