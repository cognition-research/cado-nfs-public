#include "cado.h" // IWYU pragma: keep

#include <cstdint>

#include <mutex>
#include <vector>

#include "bwc_device_arena.hpp"
#include "macros.h"

/* Registration happens a few times per run. Lookups happen a few times
 * per iteration. One mutex over a flat list is enough. */

static std::mutex arenas_mutex;
static std::vector<bwc_device_arena *> arenas;
static int allowed = 0;

void bwc_dev_allow()
{
    allowed = 1;
}

int bwc_dev_allowed()
{
    return allowed;
}

bwc_device_arena * bwc_dev_register(bwc_device_arena const & a)
{
    std::lock_guard<std::mutex> const dummy(arenas_mutex);
    const char * const h = (const char *) a.host;
    for (bwc_device_arena const * b : arenas) {
        const char * const bh = (const char *) b->host;
        ASSERT_ALWAYS(h + a.bytes <= bh || bh + b->bytes <= h);
    }
    auto * p = new bwc_device_arena(a);
    arenas.push_back(p);
    return p;
}

/* The release call runs after the registry lock drops, because a
 * backend may take its own lock inside release, and the same backend
 * lock is held around registration calls. */
void bwc_dev_unregister(const void * host)
{
    bwc_device_arena * a = nullptr;
    {
        std::lock_guard<std::mutex> const dummy(arenas_mutex);
        for (auto it = arenas.begin() ; it != arenas.end() ; ++it) {
            if ((*it)->host != host)
                continue;
            a = *it;
            arenas.erase(it);
            break;
        }
    }
    if (!a)
        return;
    a->ops->release(a);
    delete a;
}

bwc_device_arena * bwc_dev_find(const void * p)
{
    std::lock_guard<std::mutex> const dummy(arenas_mutex);
    for (bwc_device_arena * a : arenas) {
        const char * h = (const char *) a->host;
        if ((const char *) p >= h && (const char *) p < h + a->bytes)
            return a;
    }
    return nullptr;
}

void bwc_dev_fetch(const void * p, size_t bytes)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a || a->state != BWC_ARENA_DEV)
        return;
    size_t const off = (const char *) p - (const char *) a->host;
    ASSERT_ALWAYS(off + bytes <= a->bytes);
    a->ops->to_host(a, off, bytes);
}

void bwc_dev_fetch_words(const void * p, const uint64_t * byte_off, size_t n)
{
    if (!n)
        return;
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a || a->state != BWC_ARENA_DEV)
        return;
    uint64_t const base = (const char *) p - (const char *) a->host;
    std::vector<uint64_t> off(byte_off, byte_off + n);
    for (uint64_t & o : off) {
        o += base;
        ASSERT_ALWAYS(o + sizeof(uint64_t) <= a->bytes);
    }
    a->ops->words_to_host(a, off.data(), n);
}

void bwc_dev_push(const void * p, size_t bytes)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a || a->state != BWC_ARENA_DEV)
        return;
    size_t const off = (const char *) p - (const char *) a->host;
    ASSERT_ALWAYS(off + bytes <= a->bytes);
    a->ops->to_dev(a, off, bytes);
}

void bwc_dev_invalidate(const void * p)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (a)
        a->state = BWC_ARENA_HOST;
}

int bwc_dev_copy(void * dst, const void * src, size_t bytes)
{
    bwc_device_arena * d = bwc_dev_find(dst);
    if (!d)
        return 0;
    size_t const doff = (char *) dst - (char *) d->host;
    int const whole = doff == 0 && bytes == d->bytes;
    if (d->state != BWC_ARENA_DEV && !whole)
        return 0;
    bwc_device_arena * s = bwc_dev_find(src);
    if (!s || s->state != BWC_ARENA_DEV)
        return 0;
    size_t const soff = (const char *) src - (const char *) s->host;
    ASSERT_ALWAYS(doff + bytes <= d->bytes);
    ASSERT_ALWAYS(soff + bytes <= s->bytes);
    if (d == s && doff == soff)
        return 1;
    d->ops->dev_copy(d, doff, s, soff, bytes);
    d->state = BWC_ARENA_DEV;
    return 1;
}

int bwc_dev_xor(void * dst, const void * src, size_t bytes)
{
    if (!bytes)
        return 1;
    bwc_device_arena * d = bwc_dev_find(dst);
    if (!d || d->state != BWC_ARENA_DEV || !d->ops->dev_xor)
        return 0;
    bwc_device_arena * s = bwc_dev_find(src);
    if (!s || s->state != BWC_ARENA_DEV)
        return 0;
    size_t const doff = (char *) dst - (char *) d->host;
    size_t const soff = (const char *) src - (const char *) s->host;
    ASSERT_ALWAYS(doff + bytes <= d->bytes);
    ASSERT_ALWAYS(soff + bytes <= s->bytes);
    d->ops->dev_xor(d, doff, s, soff, bytes);
    return 1;
}

int bwc_dev_words_queue(const void * p, const uint64_t * byte_off,
        size_t n, size_t slot)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a || a->state != BWC_ARENA_DEV || !a->ops->words_queue)
        return 0;
    if (!n)
        return 1;
    uint64_t const base = (const char *) p - (const char *) a->host;
    std::vector<uint64_t> off(byte_off, byte_off + n);
    for (uint64_t & o : off) {
        o += base;
        ASSERT_ALWAYS(o + sizeof(uint64_t) <= a->bytes);
    }
    a->ops->words_queue(a, off.data(), n, slot);
    return 1;
}

int bwc_dev_addmul64(void * w, size_t bytes, unsigned int n,
        const void * const * v, const void * const * v_base,
        const size_t * v_bytes, const uint64_t * const * f)
{
    if (!bytes || !n)
        return 1;
    bwc_device_arena * d = bwc_dev_find(w);
    if (!d || d->state != BWC_ARENA_DEV || !d->ops->dev_addmul64)
        return 0;
    size_t const doff = (char *) w - (char *) d->host;
    ASSERT_ALWAYS(doff + bytes <= d->bytes);
    for (unsigned int j = 0; j < n; j++) {
        ASSERT_ALWAYS((const char *) v[j] >= (const char *) v_base[j]);
        ASSERT_ALWAYS((const char *) v[j] + bytes
                <= (const char *) v_base[j] + v_bytes[j]);
        /* the sources must not overlap the destination */
        ASSERT_ALWAYS((const char *) v[j] + bytes <= (const char *) w
                || (const char *) w + bytes <= (const char *) v[j]);
    }
    d->ops->dev_addmul64(d, doff, bytes, n, v, v_base, v_bytes, f);
    return 1;
}

int bwc_dev_dotprod64(const void * v, size_t bytes, size_t stride,
        const uint64_t * c, uint64_t * out)
{
    bwc_device_arena * s = bwc_dev_find(v);
    if (!s || s->state != BWC_ARENA_DEV || !s->ops->dev_dotprod64)
        return 0;
    size_t const soff = (const char *) v - (const char *) s->host;
    ASSERT_ALWAYS(soff + bytes <= s->bytes);
    ASSERT_ALWAYS(stride % sizeof(uint64_t) == 0);
    ASSERT_ALWAYS(bytes % stride == 0);
    s->ops->dev_dotprod64(s, soff, bytes, stride, c, out);
    return 1;
}

/* the current mirror of an index array, on the given device */
static bwc_device_arena * index_arena(const uint32_t * idx, size_t n,
        int device)
{
    bwc_device_arena * a = bwc_dev_find(idx);
    ASSERT_ALWAYS(a && a->host == idx && a->state == BWC_ARENA_DEV);
    ASSERT_ALWAYS(a->device == device);
    ASSERT_ALWAYS(n * sizeof(uint32_t) <= a->bytes);
    return a;
}

int bwc_dev_scatter(void * dst, const uint32_t * di, const void * src,
        const uint32_t * si, size_t n, size_t stride)
{
    if (!n)
        return 1;
    bwc_device_arena * d = bwc_dev_find(dst);
    if (!d || d->state != BWC_ARENA_DEV || !d->ops->dev_scatter)
        return 0;
    bwc_device_arena * s = bwc_dev_find(src);
    if (!s || s->state != BWC_ARENA_DEV)
        return 0;
    ASSERT_ALWAYS(dst == d->host);
    ASSERT_ALWAYS(src == s->host);
    ASSERT_ALWAYS(d->device == s->device);
    d->ops->dev_scatter(d, index_arena(di, n, d->device), s,
            index_arena(si, n, d->device), n, stride);
    return 1;
}

void bwc_dev_adopt(const void * p)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a)
        return;
    a->ops->to_dev(a, 0, a->bytes);
    a->state = BWC_ARENA_DEV;
}

void bwc_dev_mirror(void * host, size_t bytes, const void * like)
{
    bwc_device_arena * l = bwc_dev_find(like);
    if (!l || bwc_dev_find(host))
        return;
    l->ops->mirror(l, host, bytes);
}

int bwc_dev_set_zero(void * p, size_t bytes)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a || !a->ops->set_zero)
        return 0;
    size_t const off = (char *) p - (char *) a->host;
    int const whole = off == 0 && bytes == a->bytes;
    if (a->state != BWC_ARENA_DEV && !whole)
        return 0;
    ASSERT_ALWAYS(off + bytes <= a->bytes);
    a->ops->set_zero(a, off, bytes);
    a->state = BWC_ARENA_DEV;
    return 1;
}

int bwc_dev_words_flush(const void * p, uint64_t * out, size_t n,
        size_t nslots)
{
    bwc_device_arena * a = bwc_dev_find(p);
    if (!a || !a->ops->words_flush)
        return 0;
    a->ops->words_flush(a, out, n, nslots);
    return 1;
}
