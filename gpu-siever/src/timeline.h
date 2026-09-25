/* Per-device event timeline of a few iterations, for the product and
 * exchange schedule of one rank. SPMV_GF2_TIMELINE=1 turns it on for
 * MPI rank 0. tl_mark records an event on a stream during the window
 * of calls [TL_K0, TL_K0 + TL_N); tl_flush prints every mark as
 * milliseconds after the base of its device. The bases of the devices
 * fire together: the first mark records a gate event on its stream,
 * and the base of every device is an event on an idle stream of the
 * device that waits for the gate. The gate fires when the queued work
 * of that stream drains, long after the host has enqueued every base
 * (the host runs far ahead of the devices), so the devices align to
 * the cross-device event latency. A host stamp at the gate aligns the
 * two libraries. TH lines are host clock stamps. SPMV_GF2_TIMELINE=all
 * marks on every rank; the rank is the last field of the TB and TE
 * lines, so that the ranks of one ring can be aligned on the end of
 * a leg, which they reach together. */
#ifndef TIMELINE_H
#define TIMELINE_H
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>
#include <string>
#include <vector>
#include <cuda_runtime.h>

/* the window starts away from the checkpoint at 1024, where the host
 * drains the devices; SPMV_GF2_TIMELINE_K0 moves it */
#define TL_K0 tl_k0()
#define TL_N 6

struct tl_rec { cudaEvent_t ev; int dev; const char * what; unsigned long k; };
struct tl_state {
    std::vector<tl_rec> recs;
    double base_host = 0;
    cudaEvent_t gate = nullptr;
    cudaEvent_t base[16] = { nullptr };
    const char * tag;
};

static inline int tl_rank()
{
    static int const r = getenv("OMPI_COMM_WORLD_RANK") ? atoi(getenv("OMPI_COMM_WORLD_RANK")) : 0;
    return r;
}

static inline bool tl_on()
{
    static const char * const v = getenv("SPMV_GF2_TIMELINE");
    if (!v) return false;
    static bool const all = std::string(v) == "all";
    return all || tl_rank() == 0;
}

/* a library can take its first call from an environment variable of its
 * own (TL_K0_ENV), since its call count need not match the product's */
#ifndef TL_K0_ENV
#define TL_K0_ENV "SPMV_GF2_TIMELINE_K0"
#endif

static inline unsigned long tl_k0()
{
    static const char * const s = getenv(TL_K0_ENV)
        ? getenv(TL_K0_ENV) : getenv("SPMV_GF2_TIMELINE_K0");
    static unsigned long const k0 = s ? strtoul(s, nullptr, 10) : 1100;
    return k0;
}

static inline double tl_now()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* "<what><i>", with the storage of the marks' lifetime */
static inline const char * tl_name(const char * what, int i)
{
    static std::set<std::string> names;
    if (!tl_on()) return what;
    return names.insert(what + std::to_string(i)).first->c_str();
}

static void CUDART_CB tl_stamp(void * p) { *(double *) p = tl_now(); }

/* the base of the device: an event behind the gate on an idle stream
 * of the device (the current device is dev) */
static inline void tl_base(tl_state & s, cudaStream_t st, int dev)
{
    bool const first = !s.gate;
    if (first) {
        cudaEventCreate(&s.gate);
        cudaEventRecord(s.gate, st);
    }
    cudaStream_t idle;
    cudaStreamCreateWithFlags(&idle, cudaStreamNonBlocking);
    cudaStreamWaitEvent(idle, s.gate, 0);
    if (first)
        cudaLaunchHostFunc(idle, tl_stamp, &s.base_host);
    cudaEventCreate(&s.base[dev]);
    cudaEventRecord(s.base[dev], idle);
}

static inline void tl_mark(tl_state & s, cudaStream_t st, int dev,
        const char * what, unsigned long k)
{
    if (!tl_on() || k < TL_K0 || k >= TL_K0 + TL_N) return;
    int cur;
    cudaGetDevice(&cur);
    cudaSetDevice(dev);
    if (!s.base[dev])
        tl_base(s, st, dev);
    tl_rec r{ nullptr, dev, what, k };
    cudaEventCreate(&r.ev);
    cudaEventRecord(r.ev, st);
    s.recs.push_back(r);
    cudaSetDevice(cur);
}

static inline void tl_flush(tl_state & s, unsigned long k)
{
    if (!tl_on() || k != TL_K0 + TL_N) return;
    for (tl_rec & r : s.recs) cudaEventSynchronize(r.ev);
    for (int d = 0; d < 16; d++) {
        if (!s.base[d]) continue;
        cudaEventSynchronize(s.base[d]);
        fprintf(stderr, "TB %s dev%d host %.6f rank %d\n", s.tag, d,
                s.base_host, tl_rank());
        for (tl_rec & r : s.recs) {
            if (r.dev != d) continue;
            float ms;
            cudaEventElapsedTime(&ms, s.base[d], r.ev);
            fprintf(stderr, "TE %s dev%d %.3f %s %lu rank %d\n", s.tag, d,
                    ms, r.what, r.k, tl_rank());
        }
    }
    s.recs.clear();
}
#endif
