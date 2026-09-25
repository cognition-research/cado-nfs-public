#ifndef CADO_BWC_DEVICE_ARENA_HPP
#define CADO_BWC_DEVICE_ARENA_HPP

#include <cstddef>
#include <cstdint>

/* A device arena is a device-side mirror of one host vector arena. A
 * matmul backend that computes on a device creates the mirror and
 * registers it here. Host code finds the arena with any pointer into
 * the host range. All calls accept pointers outside every registered
 * arena and do nothing for them.
 *
 * state says which copy holds the current data. The backend and the
 * communication plugin set BWC_ARENA_DEV after they write the mirror.
 * A change to BWC_ARENA_DEV asserts that the whole mirror range is
 * current, not only the range of one call. Host code calls
 * bwc_dev_fetch on a range before it reads it, and bwc_dev_invalidate
 * after it writes the host copy.
 *
 * Registration happens only in programs that call bwc_dev_allow. Those
 * programs fetch and invalidate around every host access to the
 * vectors. Other programs keep the pure host behavior.
 *
 * The registry owns the arena objects. bwc_dev_register copies its
 * argument and returns the stored object; bwc_dev_unregister releases
 * the mirror through ops->release and deletes the object. Callers do
 * not keep arena pointers across bwc phases.
 *
 * The registry serves one process. The registry lock only protects the
 * arena list. The backend serializes registration, growth, and state
 * changes for one arena with its own lock, because the threads that
 * share one vector reach the backend concurrently. Host code relies on
 * the bwc thread serialization at every use point.
 */

#define BWC_ARENA_HOST 0
#define BWC_ARENA_DEV  1

struct bwc_device_arena;

/* An operation may leave device work queued instead of complete. The
 * ready field of the arena then holds an opaque device event that
 * fires when every queued device access to the mirror is complete. An
 * operation waits on the event of each mirror it touches before it
 * queues its own accesses, and stores a fresh event when it leaves
 * accesses queued. to_host, words_to_host and words_flush complete on
 * the host before they return, so host code always reads settled
 * data. words_queue can leave its gather queued. A backend or plugin
 * that leaves ready null keeps every operation synchronous.
 *
 * The module that stores an event handle keeps the event object alive
 * for the arena lifetime, and a re-record of the object only moves the
 * fire point later in time. A stale handle therefore still covers the
 * accesses it covered when it was stored. */

/* A producer that computes the mirror in ranges can publish one event
 * per range in bands, so a consumer starts on a range before the whole
 * mirror settles. The bands are sorted, contiguous, and start at
 * offset zero; ready still covers every queued access. An operation
 * that writes the mirror and stores a new ready value clears the
 * bands. A read that stores a new ready value keeps the bands: each
 * band event still covers its range. A consumer that needs a range
 * past the last band waits on ready. The producer owns the band array
 * and its events, and keeps them alive for the arena lifetime. */
struct bwc_device_band {
    size_t off;
    size_t bytes;
    void * ready;       /* fires when the range is complete */
};

struct bwc_device_arena_ops {
    /* copy the mirror range [off, off + bytes) to the host copy */
    void (*to_host)(bwc_device_arena *, size_t off, size_t bytes);
    /* copy the host range [off, off + bytes) to the mirror */
    void (*to_dev)(bwc_device_arena *, size_t off, size_t bytes);
    /* copy the 8-byte mirror words at the n byte offsets to the host
     * copy */
    void (*words_to_host)(bwc_device_arena *, const uint64_t * byte_off,
            size_t n);
    /* copy bytes from the src mirror at soff to the dst mirror at
     * doff */
    void (*dev_copy)(bwc_device_arena * dst, size_t doff,
            bwc_device_arena * src, size_t soff, size_t bytes);
    /* free the mirror and the backend context */
    void (*release)(bwc_device_arena *);
    /* XOR the src mirror range at soff into the dst mirror range at
     * doff */
    void (*dev_xor)(bwc_device_arena * dst, size_t doff,
            bwc_device_arena * src, size_t soff, size_t bytes);
    /* Deferred word gather, for the per-iteration dot products.  queue
     * reads the 8-byte mirror words at the n byte offsets and keeps
     * them on the device as slot number slot.  queue can return with
     * the gather still queued behind ready.  A later write to the
     * mirror does not change a queued slot.  flush completes the
     * queued gathers and copies the slots [0, nslots), n words each,
     * to out.  n is the same for every slot of one queue-flush cycle,
     * and n is never zero: a zero-length capture succeeds in the
     * wrapper without a backend call. */
    void (*words_queue)(bwc_device_arena *, const uint64_t * byte_off,
            size_t n, size_t slot);
    void (*words_flush)(bwc_device_arena *, uint64_t * out, size_t n,
            size_t nslots);
    /* GF(2) combine of n sources in one pass: each 64-bit word w of
     * the dst mirror range [doff, doff + bytes) receives the xor of
     * f[j][k] over the sources j and the set bits k of word w of
     * source j. v[j] points into a host vector; the backend mirrors
     * the host range [v_base[j], v_base[j] + v_bytes[j]), so an
     * invalidation through any pointer in that range reaches the
     * mirror. No source range overlaps the dst range. */
    void (*dev_addmul64)(bwc_device_arena * dst, size_t doff,
            size_t bytes, unsigned int n, const void * const * v,
            const void * const * v_base, const size_t * v_bytes,
            const uint64_t * const * f);
    /* GF(2) transposed combine: out[64w + b] ^= c[i] for each item i
     * of the src mirror range [soff, soff + bytes), stride bytes each,
     * and each set bit b of word w of the item. c and out are host
     * arrays, of one word per item and of 8 * stride bytes; out is
     * complete on return. */
    void (*dev_dotprod64)(bwc_device_arena * src, size_t soff,
            size_t bytes, size_t stride, const uint64_t * c,
            uint64_t * out);
    /* Item scatter between two current mirrors on one device: dst item
     * di[k] receives src item si[k], stride bytes each, for k in
     * [0, n). di and si are current mirrors on the same device whose
     * first n uint32_t are the indices. */
    void (*dev_scatter)(bwc_device_arena * dst, bwc_device_arena * di,
            bwc_device_arena * src, bwc_device_arena * si, size_t n,
            size_t stride);
    /* give the host range [host, host + bytes) a mirror in the host
     * state on the device of the like arena */
    void (*mirror)(bwc_device_arena * like, void * host, size_t bytes);
    /* zero the mirror range [off, off + bytes) */
    void (*set_zero)(bwc_device_arena *, size_t off, size_t bytes);
};

/* Backends and plugins built against an older layout of this struct do
 * not match it; rebuild them together with this tree. */
struct bwc_device_arena {
    void * host;
    size_t bytes;
    void * dev;         /* device pointer of the mirror */
    int device;         /* device ordinal of the mirror */
    int state;
    void * owner;       /* backend context for the ops */
    bwc_device_arena_ops const * ops;
    void * ready = nullptr;  /* opaque device event for queued accesses;
                              * null when the mirror is settled */
    bwc_device_band const * bands = nullptr;  /* per-range events, or
                                               * null */
    int nbands = 0;
};

extern "C" {

/* a program calls this once to accept device arenas; see above */
void bwc_dev_allow();
int bwc_dev_allowed();

/* the backend registers each arena once; the vector destructor
 * unregisters it, so a later allocation at the same address cannot
 * alias a stale entry */
bwc_device_arena * bwc_dev_register(bwc_device_arena const &);
void bwc_dev_unregister(const void * host);

bwc_device_arena * bwc_dev_find(const void * p);
void bwc_dev_fetch(const void * p, size_t bytes);
void bwc_dev_fetch_words(const void * p, const uint64_t * byte_off, size_t n);
/* after a host write of only [p, p + bytes): keep the mirror current */
void bwc_dev_push(const void * p, size_t bytes);
/* after a host write that leaves the whole host copy current */
void bwc_dev_invalidate(const void * p);
/* returns 1 when the copy ran on the device mirrors; a copy over the
 * whole dst mirror makes it current whatever its state was */
int bwc_dev_copy(void * dst, const void * src, size_t bytes);
/* returns 1 when the xor ran on the device mirrors */
int bwc_dev_xor(void * dst, const void * src, size_t bytes);
/* deferred word gather (see the ops); both return 1 when the mirror
 * holds the current data and the backend supports the operation */
int bwc_dev_words_queue(const void * p, const uint64_t * byte_off,
        size_t n, size_t slot);
int bwc_dev_words_flush(const void * p, uint64_t * out, size_t n,
        size_t nslots);
/* returns 1 when the combine ran on the device mirrors (see the ops) */
int bwc_dev_addmul64(void * w, size_t bytes, unsigned int n,
        const void * const * v, const void * const * v_base,
        const size_t * v_bytes, const uint64_t * const * f);
/* returns 1 when the transposed combine ran on the device mirror of v
 * (see the ops) */
int bwc_dev_dotprod64(const void * v, size_t bytes, size_t stride,
        const uint64_t * c, uint64_t * out);
/* returns 1 when the scatter ran on the device mirrors (see the ops);
 * dst and src are the host ranges of two mirrors, and di and si must
 * then be the host ranges of two current mirrors on the same device */
int bwc_dev_scatter(void * dst, const uint32_t * di, const void * src,
        const uint32_t * si, size_t n, size_t stride);
/* after a host write that leaves the whole host copy current: make the
 * mirror current too, so later device operations run on it */
void bwc_dev_adopt(const void * p);
/* mirror a host range on the device that mirrors like; no-op when like
 * has no mirror or host has one already */
void bwc_dev_mirror(void * host, size_t bytes, const void * like);
/* returns 1 when the zeroing ran on the device mirror; zeroing a whole
 * mirror makes it current whatever its state was */
int bwc_dev_set_zero(void * p, size_t bytes);

}

#endif  /* CADO_BWC_DEVICE_ARENA_HPP */
