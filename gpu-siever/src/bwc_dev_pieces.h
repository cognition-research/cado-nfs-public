#ifndef BWC_DEV_PIECES_H
#define BWC_DEV_PIECES_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

/* The per-device layout behind an arena mirror. The matmul backend
 * (spmv_gf2.cu) computes the product output in one segment per device
 * and reads the source vector from one replica per device. The mirror
 * on the first device stays the current copy for the host and for the
 * arena contract (bwc_device_arena.hpp). This record lets the
 * communication plugin (bwccomm_nccl.cpp) read the segments and write
 * the replicas directly, so the vector data does not funnel through
 * the mirror device.
 *
 * The backend stores this record at the start of the owner object of
 * each arena it creates, and the plugin trusts the magic word. Both
 * sides access the record in the program order of the bwc thread that
 * drives the vector, so it carries no lock. */

/* The arrays below hold the blocks of a product (one per job of the
 * communicator that reduces the vector) and its output segments (the
 * rows of each device cut at the block boundaries, so at most one block
 * per device more than the block count). 64 blocks per communicator
 * cover the grids in use, the widest being 16 jobs. */
#define BWC_DEV_PIECES_MAGIC 0x50494543455f4445ull
#define BWC_DEV_PIECES_MAX 64

struct bwc_dev_pieces;

struct bwc_dev_piece {
    int device;
    char * p;                   /* device address of arena offset off */
    size_t off;                 /* offset inside the arena */
    size_t bytes;
    cudaEvent_t ready;          /* the range is complete on p */
    bwc_dev_pieces * owner;     /* the record of the arena whose memory holds p */
};

struct bwc_dev_pieces {
    uint64_t magic;

    /* The output segments of the last product into this arena: each
     * holds its range of the mirror data behind its ready event, and
     * together they cover [out[0].off, out_end). The backend zeroes
     * nout at every other write of the mirror, and the plugin zeroes
     * it when it writes the mirror, or replaces them with the slices
     * of its reduce-scatter result. A reader records one event per
     * stream it read on in the rd of each segment's owner (a forwarded
     * segment lives in the memory of another arena); the backend waits
     * for them before it writes that memory again. out_settled marks segments that a
     * writer of the mirror may take instead of the ready marker: the
     * segment events cover the write of the segments, and out_before
     * (when not null) covers every access of the mirror before that
     * write. The reduce-scatter settles its result slices, and a copy
     * hands settled segments on to its destination. */
    int nout;
    bool out_settled;
    cudaEvent_t out_before;
    size_t out_end;
    bwc_dev_piece out[BWC_DEV_PIECES_MAX];
    int nrd;
    cudaEvent_t rd[BWC_DEV_PIECES_MAX];

    /* wgen counts the writes of the mirror data. The backend bumps it
     * at each of its own writes; the plugin bumps it when it writes
     * the mirror without the replicas. */
    uint64_t wgen;

    /* The source replicas of the next product from this arena, all
     * with the same off and bytes, each behind rep[k].ready, the last
     * read of that replica. A writer of the mirror that also fills the
     * replicas stores the wgen it leaves in rep_wgen, so the replicas
     * stay current while rep_wgen == wgen; the writes to the replicas
     * are complete when the mirror ready marker is. rep_owner belongs
     * to the backend: the instance that published the replicas. */
    int nrep;
    bwc_dev_piece rep[BWC_DEV_PIECES_MAX];
    uint64_t rep_wgen;
    void * rep_owner;

    /* The order the plugin consumes the next product into this arena:
     * block b is the arena range [b * blk_bytes, (b + 1) * blk_bytes),
     * and blk lists the nblk blocks in consumption order. The backend
     * computes the rows in that order and publishes one output segment
     * per device and block, so the plugin starts on a block while the
     * product computes the later ones. nblk 0 (the initial state) asks
     * for one segment per device. */
    size_t blk_bytes;
    int nblk;
    int blk[BWC_DEV_PIECES_MAX];
};

#endif
