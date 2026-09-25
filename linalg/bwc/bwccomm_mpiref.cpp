#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <mutex>
#include <vector>

#include "macros.h"
#include "bwc_device_arena.hpp"
#include "matmul_top_comm_plugin.hpp"
#include "select_mpi.h"

/* Reference implementation of the communication plugin interface. It
 * runs the same operations as the stock MPI path, on host memory. It
 * exists to test the plugin hook, not to run faster.
 * comm_impl=mpiref selects it.
 *
 * A buffer can be one range of a larger arena (one thread chunk of a
 * vector), so the plugin works like host code: it fetches the range it
 * reads and pushes the range it writes, and never marks the whole
 * arena stale.
 */

static void xor_into(uint64_t * dst, const uint64_t * src, size_t words)
{
    for (size_t i = 0; i < words; i++)
        dst[i] ^= src[i];
}

/* One notice per process and per operation on first use, so that test
 * logs show that the plugin path is active. */
static void notice_once(std::once_flag & once, const char * op)
{
    std::call_once(once, [op]() {
        fprintf(stderr, "comm plugin mpiref: first %s\n", op);
    });
}

static void reduce_scatter_xor(void * const * bufs, int nbufs,
        size_t block_bytes, MPI_Comm comm)
{
    static std::once_flag once;
    notice_once(once, "reduce_scatter_xor");
    int njobs;
    int rank;
    MPI_Comm_size(comm, &njobs);
    MPI_Comm_rank(comm, &rank);
    /* one job: the operation is the identity, and the device mirrors
     * stay current */
    if (njobs == 1)
        return;
    ASSERT_ALWAYS(block_bytes % sizeof(uint64_t) == 0);
    size_t const words = block_bytes / sizeof(uint64_t);
    int const srank = (rank + 1) % njobs;
    int const drank = (rank + njobs - 1) % njobs;
    std::vector<uint64_t> acc(words);
    std::vector<uint64_t> tmp(words);
    for (int w = 0; w < nbufs; w++) {
        uint64_t * const v = (uint64_t *) bufs[w];
        bwc_dev_fetch(v, (size_t) njobs * block_bytes);
        std::fill(acc.begin(), acc.end(), 0);
        for (int i = 0; i < njobs; i++) {
            int const l = (rank + 1 + i) % njobs;
            xor_into(acc.data(), v + l * words, words);
            if (i == njobs - 1)
                break;
            MPI_Sendrecv(
                    acc.data(), block_bytes, MPI_BYTE, drank, (i << 16) + rank,
                    tmp.data(), block_bytes, MPI_BYTE, srank, (i << 16) + srank,
                    comm, MPI_STATUS_IGNORE);
            std::swap(acc, tmp);
        }
        memcpy(v, acc.data(), block_bytes);
        bwc_dev_push(v, block_bytes);
    }
}

static void allgather(void * const * bufs, int nbufs,
        size_t block_bytes, MPI_Comm comm)
{
    static std::once_flag once;
    notice_once(once, "allgather");
    int njobs;
    int rank;
    MPI_Comm_size(comm, &njobs);
    MPI_Comm_rank(comm, &rank);
    if (njobs == 1)
        return;
    for (int w = 0; w < nbufs; w++) {
        bwc_dev_fetch((const char *) bufs[w] + (size_t) rank * block_bytes,
                block_bytes);
        int const err = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                bufs[w], block_bytes, MPI_BYTE, comm);
        ASSERT_ALWAYS(!err);
        bwc_dev_push(bufs[w], (size_t) njobs * block_bytes);
    }
}

static bwc_comm_plugin_api const api = { reduce_scatter_xor, allgather, 1 };

extern "C" bwc_comm_plugin_api const * bwc_comm_solib_reach_api()
{
    return &api;
}
