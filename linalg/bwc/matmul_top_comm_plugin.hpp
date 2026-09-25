#ifndef CADO_MATMUL_TOP_COMM_PLUGIN_HPP
#define CADO_MATMUL_TOP_COMM_PLUGIN_HPP

#include <cstddef>

#include "params.hpp"
#include "select_mpi.h"

/* A communication plugin replaces the two MPI legs of
 * matmul_top_mul_comm. comm_impl=<impl> loads libbwccomm_<impl> through
 * dlopen, with the same naming scheme as the matmul backends. Without
 * the parameter, the stock MPI path runs.
 *
 * Both operations act on nbufs host buffers at once. Each buffer holds
 * one block of block_bytes per rank of comm.
 *
 * reduce_scatter_xor: on exit, the XOR across all ranks of the blocks
 * with this rank's index sits at the beginning of the buffer.
 *
 * allgather: on entry the block at this rank's index is set; on exit
 * every block is set.
 *
 * handles_device_arenas says that the plugin looks the buffers up in
 * the device arena registry (bwc_device_arena.hpp), works on the
 * mirrors of registered buffers, and keeps the arena states correct.
 * The matmul backend keeps the vectors on the device only when the
 * loaded plugin sets this flag. Such a plugin may leave its device
 * work queued behind the ready event of each mirror, under the arena
 * header's rules. A host buffer without a current mirror is always
 * settled on exit. Plugins built against an older interface do not
 * match this struct; rebuild them together with this tree.
 */
struct bwc_comm_plugin_api {
    void (*reduce_scatter_xor)(void * const * bufs, int nbufs,
            size_t block_bytes, MPI_Comm comm);
    void (*allgather)(void * const * bufs, int nbufs,
            size_t block_bytes, MPI_Comm comm);
    int handles_device_arenas;
};

void matmul_top_comm_plugin_decl_usage(cxx_param_list & pl);
void matmul_top_comm_plugin_lookup_parameters(cxx_param_list & pl);
void matmul_top_comm_plugin_init(cxx_param_list & pl);

extern bwc_comm_plugin_api const * bwc_comm_plugin;

#endif  /* CADO_MATMUL_TOP_COMM_PLUGIN_HPP */
