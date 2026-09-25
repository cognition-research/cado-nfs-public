/* CPU reference implementation of the spmv_gf2 API, for validating the
 * matmul_cuda glue on machines without a CUDA device. The device
 * arenas of the CUDA version become plain host buffers here, so the
 * arena state machine runs the same way. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "bwc_device_arena.hpp"
#include "spmv_gf2.h"

struct spmv_gf2 {
    unsigned int nrows;
    unsigned int words;
    std::vector<uint64_t> rowptr;
    std::vector<uint32_t> cols;
};

spmv_gf2 * spmv_gf2_new(unsigned int nrows, uint64_t nwords,
                        uint32_t * rows, unsigned int max_dim,
                        unsigned int words)
{
    (void) max_dim;
    spmv_gf2 * m = new spmv_gf2;
    m->nrows = nrows;
    m->words = words;
    m->rowptr.resize(nrows + 1);
    m->cols.reserve(nwords - nrows);
    uint64_t k = 0;
    for (unsigned int i = 0; i < nrows; i++) {
        m->rowptr[i] = m->cols.size();
        uint32_t const len = rows[k++];
        m->cols.insert(m->cols.end(), rows + k, rows + k + len);
        k += len;
    }
    m->rowptr[nrows] = m->cols.size();
    if (k != nwords) {
        fprintf(stderr, "spmv_gf2_new: row stream length mismatch\n");
        abort();
    }
    return m;
}

struct arena_ctx {
    std::vector<uint64_t> q;    /* queued word captures */
};

static void arena_to_host(bwc_device_arena * a, size_t off, size_t bytes)
{
    memcpy((char *) a->host + off, (char *) a->dev + off, bytes);
}

static void arena_to_dev(bwc_device_arena * a, size_t off, size_t bytes)
{
    memcpy((char *) a->dev + off, (char *) a->host + off, bytes);
}

static void arena_words_to_host(bwc_device_arena * a,
        const uint64_t * off, size_t n)
{
    for (size_t i = 0; i < n; i++)
        *(uint64_t *) ((char *) a->host + off[i]) =
            *(const uint64_t *) ((const char *) a->dev + off[i]);
}

static void arena_words_queue(bwc_device_arena * a,
        const uint64_t * off, size_t n, size_t slot)
{
    if (!n)
        return;
    auto * c = (arena_ctx *) a->owner;
    if (c->q.size() < (slot + 1) * n)
        c->q.resize((slot + 1) * n);
    for (size_t i = 0; i < n; i++)
        c->q[slot * n + i] =
            *(const uint64_t *) ((const char *) a->dev + off[i]);
}

static void arena_words_flush(bwc_device_arena * a, uint64_t * out,
        size_t n, size_t nslots)
{
    auto * c = (arena_ctx *) a->owner;
    memcpy(out, c->q.data(), nslots * n * sizeof(uint64_t));
}

static void arena_dev_copy(bwc_device_arena * dst, size_t doff,
        bwc_device_arena * src, size_t soff, size_t bytes)
{
    memcpy((char *) dst->dev + doff, (const char *) src->dev + soff, bytes);
}

static void arena_dev_xor(bwc_device_arena * dst, size_t doff,
        bwc_device_arena * src, size_t soff, size_t bytes)
{
    auto * d = (uint64_t *) ((char *) dst->dev + doff);
    auto * s = (const uint64_t *) ((const char *) src->dev + soff);
    for (size_t i = 0; i < bytes / sizeof(uint64_t); i++)
        d[i] ^= s[i];
}

static bwc_device_arena * mirror(const void * host, size_t bytes);
static void arena_make_dev(bwc_device_arena * a, size_t filled);

/* This creates a mirror for each source vector that has none, and
 * moves it to the device state.  After that, a host write to a source
 * vector must go through bwc_dev_push or bwc_dev_invalidate, or the
 * write is lost. */
static void arena_dev_addmul64(bwc_device_arena * dst, size_t doff,
        size_t bytes, unsigned int n, const void * const * v,
        const void * const * v_base, const size_t * v_bytes,
        const uint64_t * const * f)
{
    auto * d = (uint64_t *) ((char *) dst->dev + doff);
    for (unsigned int i = 0; i < n; i++) {
        bwc_device_arena * const src = mirror(v_base[i], v_bytes[i]);
        arena_make_dev(src, 0);
        size_t const soff = (const char *) v[i] - (const char *) src->host;
        auto * s = (const uint64_t *) ((const char *) src->dev + soff);
        for (size_t j = 0; j < bytes / sizeof(uint64_t); j++) {
            uint64_t const a = s[j];
            for (unsigned int k = 0; k < 64; k++)
                d[j] ^= f[i][k] & (uint64_t) -(int64_t) ((a >> k) & 1);
        }
    }
}

static void arena_dev_dotprod64(bwc_device_arena * src, size_t soff,
        size_t bytes, size_t stride, const uint64_t * c, uint64_t * out)
{
    auto * s = (const uint64_t *) ((const char *) src->dev + soff);
    size_t const words = stride / sizeof(uint64_t);
    for (size_t i = 0; i < bytes / stride; i++)
        for (size_t w = 0; w < words; w++)
            for (unsigned int b = 0; b < 64; b++)
                out[64 * w + b] ^= c[i] & (uint64_t) -(int64_t) ((s[i * words + w] >> b) & 1);
}

static void arena_dev_scatter(bwc_device_arena * dst, bwc_device_arena * di,
        bwc_device_arena * src, bwc_device_arena * si, size_t n,
        size_t stride)
{
    auto const * d = (const uint32_t *) di->dev;
    auto const * s = (const uint32_t *) si->dev;
    for (size_t k = 0; k < n; k++)
        memcpy((char *) dst->dev + d[k] * stride,
                (const char *) src->dev + s[k] * stride, stride);
}

static void arena_mirror(bwc_device_arena *, void * host, size_t bytes)
{
    mirror(host, bytes);
}

static void arena_set_zero(bwc_device_arena * a, size_t off, size_t bytes)
{
    memset((char *) a->dev + off, 0, bytes);
}

static void arena_release(bwc_device_arena * a)
{
    free(a->dev);
    delete (arena_ctx *) a->owner;
}

static bwc_device_arena_ops const arena_ops = {
    arena_to_host, arena_to_dev, arena_words_to_host, arena_dev_copy,
    arena_release, arena_dev_xor, arena_words_queue, arena_words_flush,
    arena_dev_addmul64, arena_dev_dotprod64, arena_dev_scatter,
    arena_mirror, arena_set_zero,
};

/* The bwc threads that share one source vector reach the product
 * functions concurrently, each through its own instance.  One lock
 * serializes mirror creation, growth, and every state change. */
static std::mutex arena_mutex;

/* The same vector base can arrive as the source of one product and the
 * destination of the transposed product with a different size, so an
 * existing mirror grows to the largest observed range.  The host holds
 * the current data for the bytes past the old range. */
static bwc_device_arena * mirror(const void * host, size_t bytes)
{
    if (!bwc_dev_allowed()) return nullptr;
    std::lock_guard<std::mutex> const lock(arena_mutex);
    bwc_device_arena * const a = bwc_dev_find(host);
    if (a) {
        if (a->bytes < bytes) {
            a->dev = realloc(a->dev, bytes);
            if (a->state == BWC_ARENA_DEV)
                arena_to_dev(a, a->bytes, bytes - a->bytes);
            a->bytes = bytes;
        }
        return a;
    }
    return bwc_dev_register({ const_cast<void *>(host), bytes,
            malloc(bytes), 0, BWC_ARENA_HOST, new arena_ctx, &arena_ops,
            nullptr, nullptr, 0 });
}

/* Mark the mirror current.  The caller filled the mirror range
 * [0, filled); the rest of the arena comes from the host copy. */
static void arena_make_dev(bwc_device_arena * a, size_t filled)
{
    std::lock_guard<std::mutex> const lock(arena_mutex);
    if (a->state == BWC_ARENA_HOST && filled < a->bytes)
        arena_to_dev(a, filled, a->bytes - filled);
    a->state = BWC_ARENA_DEV;
}

void spmv_gf2_gather(spmv_gf2 * m, uint64_t * dst, const uint64_t * src,
                     unsigned int ndst, unsigned int nsrc)
{
    unsigned int const W = m->words;
    size_t const esz = (size_t) W * sizeof(uint64_t);
    bwc_device_arena * const asrc = mirror(src, nsrc * esz);
    bwc_device_arena * const adst = mirror(dst, ndst * esz);
    const uint64_t * s = src;
    if (asrc) {
        arena_make_dev(asrc, 0);
        s = (const uint64_t *) asrc->dev;
    }
    uint64_t * const d = adst ? (uint64_t *) adst->dev : dst;
    for (unsigned int i = 0; i < m->nrows; i++) {
        for (unsigned int w = 0; w < W; w++) {
            uint64_t acc = 0;
            for (uint64_t k = m->rowptr[i]; k < m->rowptr[i+1]; k++)
                acc ^= s[(size_t) m->cols[k] * W + w];
            d[(size_t) i * W + w] = acc;
        }
    }
    if (ndst > m->nrows)
        memset(d + (size_t) m->nrows * W, 0, (ndst - m->nrows) * esz);
    if (adst)
        arena_make_dev(adst, ndst * esz);
}

void spmv_gf2_scatter(spmv_gf2 * m, uint64_t * dst, const uint64_t * src,
                      unsigned int ndst, unsigned int nsrc)
{
    unsigned int const W = m->words;
    size_t const esz = (size_t) W * sizeof(uint64_t);
    bwc_device_arena * const asrc = mirror(src, nsrc * esz);
    bwc_device_arena * const adst = mirror(dst, ndst * esz);
    const uint64_t * s = src;
    if (asrc) {
        arena_make_dev(asrc, 0);
        s = (const uint64_t *) asrc->dev;
    }
    uint64_t * const d = adst ? (uint64_t *) adst->dev : dst;
    memset(d, 0, ndst * esz);
    for (unsigned int i = 0; i < m->nrows; i++)
        for (uint64_t k = m->rowptr[i]; k < m->rowptr[i+1]; k++)
            for (unsigned int w = 0; w < W; w++)
                d[(size_t) m->cols[k] * W + w] ^= s[(size_t) i * W + w];
    if (adst)
        arena_make_dev(adst, ndst * esz);
}

void spmv_gf2_free(spmv_gf2 * m)
{
    delete m;
}
