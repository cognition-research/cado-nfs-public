/* glas: GPU lattice siever, a drop-in replacement for the las siever
 * step of CADO-NFS.
 *
 * The program follows the las pipeline:
 *   1. build the q-lattice basis for each special-q (SkewGauss)
 *   2. choose the sieve region (adjust-strategy 0)
 *   3. sieve the region (CPU reference core or CUDA core)
 *   4. process survivors exactly: divide the norms by the sieved
 *      primes, then factor the leftover with the facul library
 *   5. print relations in the CADO-NFS format
 *
 * Steps 1, 2, 4, 5 reuse CADO-NFS library code (libutils + facul) and
 * replicate the las selection logic, so the output is compatible with
 * the CPU siever.
 */

#include "cado.h" // IWYU pragma: keep

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <gmp.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "cado_poly.hpp"
#include "cxx_mpz.hpp"
#include "gmp_aux.h"
#include "mpz_poly.h"
#include "polynomial.hpp"
#include "rootfinder.h"
#include "ecm/facul.hpp"
#include "ecm/facul_strategies.hpp"

#include "cofac_strategy.hpp"

#include "sieve_core.hpp"

namespace {

/* ------------------------------------------------------------------ */
/* command line                                                       */

struct side_params {
    unsigned long lim = 0;
    unsigned int lpb = 0;
    unsigned int mfb = 0;
    double lambda = 0.0;
    int ncurves = -1;
    std::string fb_path;
};

struct params {
    std::string poly_path;
    std::string out_path;
    uint64_t q0 = 0, q1 = 0;
    int logA = 0;
    int sqside = 1;
    int nthreads = 1;
    bool gpu = false;
    bool stats_stderr = false;
    bool allow_largesq = false;
    side_params side[2];
};

[[noreturn]] void usage_error(std::string const & msg)
{
    fprintf(stderr, "Error: %s\n", msg.c_str());
    /* _exit runs no destructors, so it cannot race with the GPU
     * warmup thread. */
    fflush(nullptr);
    _exit(EXIT_FAILURE);
}

params parse_args(int argc, char const ** argv)
{
    params P;
    std::map<std::string, std::string> kv;
    std::vector<std::string> const flags = {"v", "sync", "never-discard"};
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (key.rfind('-', 0) != 0)
            usage_error("unexpected argument " + key);
        key = key.substr(1);
        if (key == "gpu") {
            P.gpu = true;
            continue;
        }
        if (key == "stats-stderr") {
            P.stats_stderr = true;
            continue;
        }
        if (key == "allow-largesq") {
            P.allow_largesq = true;
            continue;
        }
        if (std::find(flags.begin(), flags.end(), key) != flags.end())
            continue;
        if (i + 1 >= argc)
            usage_error("missing value for -" + key);
        kv[key] = argv[++i];
    }

    auto take = [&kv](char const * key) -> std::optional<std::string> {
        auto it = kv.find(key);
        if (it == kv.end())
            return std::nullopt;
        std::string v = it->second;
        kv.erase(it);
        return v;
    };
    auto need = [&take](char const * key) -> std::string {
        auto v = take(key);
        if (!v)
            usage_error(std::string("missing required -") + key);
        return *v;
    };

    P.poly_path = need("poly");
    P.q0 = strtoull(need("q0").c_str(), nullptr, 10);
    P.q1 = strtoull(need("q1").c_str(), nullptr, 10);
    if (auto v = take("I"))
        P.logA = 2 * atoi(v->c_str()) - 1;
    if (auto v = take("A"))
        P.logA = atoi(v->c_str());
    if (P.logA == 0)
        usage_error("one of -I or -A is required");
    if (auto v = take("sqside"))
        P.sqside = atoi(v->c_str());
    if (auto v = take("t"))
        P.nthreads = std::max(1, atoi(v->c_str()));
    if (auto v = take("out"))
        P.out_path = *v;
    if (auto v = take("adjust-strategy")) {
        if (atoi(v->c_str()) != 0)
            usage_error("only -adjust-strategy 0 is supported");
    }
    /* -fill-bands feeds the GLAS_FILL_BANDS default of the GPU core
     * (sieve_core_gpu.cu); an explicit environment value wins. */
    if (auto v = take("fill-bands"))
        setenv("GLAS_FILL_BANDS", v->c_str(), 0);
    /* -host-mr feeds GLAS_FLT_HOST_MR the same way. */
    if (auto v = take("host-mr"))
        setenv("GLAS_FLT_HOST_MR", v->c_str(), 0);
    for (int s = 0; s < 2; s++) {
        std::string const n = std::to_string(s);
        P.side[s].lim = strtoul(need(("lim" + n).c_str()).c_str(), nullptr, 10);
        P.side[s].lpb = atoi(need(("lpb" + n).c_str()).c_str());
        P.side[s].mfb = atoi(need(("mfb" + n).c_str()).c_str());
        if (auto v = take(("lambda" + n).c_str()))
            P.side[s].lambda = atof(v->c_str());
        if (auto v = take(("ncurves" + n).c_str()))
            P.side[s].ncurves = atoi(v->c_str());
        if (auto v = take(("fb" + n).c_str()))
            P.side[s].fb_path = *v;
    }
    /* The filter chain (renumber, dup2, purge) indexes primes below
     * 2^lpb only, so a special-q above it cannot be consumed; refuse it
     * up front, as las does. */
    if (!P.allow_largesq && P.q1 > (UINT64_C(1) << P.side[P.sqside].lpb))
        usage_error("special-q range [q0,q1) exceeds 2^lpb" +
                    std::to_string(P.sqside) + "; pass -allow-largesq to sieve it anyway");
    for (auto const & [key, value] : kv)
        fprintf(stderr, "# Warning: ignored option -%s %s\n", key.c_str(),
                value.c_str());
    return P;
}

/* ------------------------------------------------------------------ */
/* prime enumeration (simple sieves; the factor bases are small)      */

std::vector<uint32_t> primes_up_to(uint32_t n)
{
    std::vector<uint8_t> composite(n + 1, 0);
    std::vector<uint32_t> out;
    for (uint32_t p = 2; p <= n; p++) {
        if (composite[p])
            continue;
        out.push_back(p);
        for (uint64_t m = (uint64_t)p * p; m <= n; m += p)
            composite[m] = 1;
    }
    return out;
}

std::vector<uint64_t> primes_in(uint64_t lo, uint64_t hi)
{
    if (hi <= lo)
        return {};
    auto const base = primes_up_to((uint32_t)std::sqrt((double)hi) + 2);
    std::vector<uint8_t> composite(hi - lo, 0);
    for (uint64_t p : base) {
        uint64_t m = std::max(p * p, (lo + p - 1) / p * p);
        for (; m < hi; m += p)
            composite[m - lo] = 1;
    }
    std::vector<uint64_t> out;
    for (uint64_t v = lo; v < hi; v++)
        if (v >= 2 && !composite[v - lo])
            out.push_back(v);
    return out;
}

/* a^-1 mod m for coprime a and m below 2^63 */
uint64_t inv_mod(uint64_t a, uint64_t m)
{
    int64_t t = 0, tn = 1;
    uint64_t r = m, rn = a % m;
    while (rn) {
        uint64_t const q = r / rn;
        int64_t const t2 = t - (int64_t)(q * tn);
        t = tn;
        tn = t2;
        uint64_t const r2 = r - q * rn;
        r = rn;
        rn = r2;
    }
    assert(r == 1);
    return t < 0 ? (uint64_t)(t + (int64_t)m) : (uint64_t)t;
}

/* ------------------------------------------------------------------ */
/* factor bases                                                       */

uint8_t weight_units(uint32_t base_prime, int mult)
{
    long const w = std::lround(LOG_SCALE * mult * std::log2((double)base_prime));
    return (uint8_t)std::min(w, 255L);
}

/* Base prime of a prime power (makefb only emits prime powers). */
uint32_t base_prime_of(uint32_t q)
{
    static std::map<uint32_t, uint32_t> const base = [] {
        std::map<uint32_t, uint32_t> m;
        for (uint64_t b = 2; b * b <= UINT32_MAX; b++)
            for (uint64_t v = b * b; v <= UINT32_MAX; v *= b)
                m.emplace((uint32_t)v, (uint32_t)b);
        return m;
    }();
    auto const it = base.find(q);
    return it == base.end() ? q : it->second;
}

void parse_fb_line(char * line, std::vector<fb_entry> & out)
{
    if (line[0] == '#' || line[0] == 0)
        return;
    char * colon1 = strchr(line, ':');
    if (!colon1)
        return;
    *colon1 = 0;
    uint32_t const q = strtoul(line, nullptr, 10);
    char * rest = colon1 + 1;
    int n1 = 1, n2 = 0;
    char * colon2 = strchr(rest, ':');
    if (colon2) {
        *colon2 = 0;
        sscanf(rest, "%d,%d", &n1, &n2);
        rest = colon2 + 1;
    }
    uint32_t const p = base_prime_of(q);
    fb_entry e;
    e.q = q;
    e.weight = weight_units(p, n1 - n2);
    e.resieve = (p == q) && n1 == 1 && n2 == 0 && q > TINY_PRIME_BOUND;
    while (*rest) {
        e.r = strtoul(rest, &rest, 10);
        out.push_back(e);
        if (*rest != ',')
            break;
        rest++;
    }
}

void parse_fb_chunk(char * s, char * const end, std::vector<fb_entry> & out)
{
    while (s < end) {
        char * const nl = (char *)memchr(s, '\n', (size_t)(end - s));
        *nl = 0;
        parse_fb_line(s, out);
        s = nl + 1;
    }
}

/* Concatenate parsed parts; each thread fills part[i] from one range. */
template <typename Fill>
std::vector<fb_entry> parallel_fb(unsigned n_chunks, Fill && fill)
{
    std::vector<std::vector<fb_entry>> part(n_chunks);
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < n_chunks; i++)
        threads.emplace_back([&, i] { fill(i, part[i]); });
    for (auto & th : threads)
        th.join();
    size_t total = 0;
    for (auto const & p : part)
        total += p.size();
    std::vector<fb_entry> out;
    out.reserve(total);
    for (auto const & p : part)
        out.insert(out.end(), p.begin(), p.end());
    return out;
}

bool is_gz(std::string const & path)
{
    return path.size() > 3 && path.substr(path.size() - 3) == ".gz";
}

/* popen without a shell: a stream from the stdout (mode "r") or to the
 * stdin (mode "w") of argv; the child's other standard stream is fd.
 * close() waits for the child and returns its wait status. */
struct child_pipe {
    FILE * f;
    pid_t pid;

    int close()
    {
        fclose(f);
        int status;
        waitpid(pid, &status, 0);
        return status;
    }
};

child_pipe pipe_open(std::vector<std::string> const & argv, char const * mode,
                     int fd)
{
    bool const reading = mode[0] == 'r';
    int p[2];
    if (pipe2(p, O_CLOEXEC) != 0)
        return {nullptr, 0};
    std::vector<char *> av;
    for (auto const & a : argv)
        av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    pid_t const pid = fork();
    if (pid == 0) {
        dup2(p[reading], reading);
        dup2(fd, !reading);
        execvp(av[0], av.data());
        _exit(127);
    }
    close(p[reading]);
    return {pid < 0 ? nullptr : fdopen(p[!reading], mode), pid};
}

std::vector<fb_entry> read_fb_file(std::string const & path)
{
    bool const gz = is_gz(path);
    child_pipe zcat = {nullptr, 0};
    FILE * f;
    if (gz) {
        zcat = pipe_open({"zcat", path}, "r", 0);
        f = zcat.f;
    } else {
        f = fopen(path.c_str(), "r");
    }
    if (!f)
        usage_error("cannot open factor base " + path);

    std::string text;
    char buf[1 << 20];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;)
        text.append(buf, n);
    if (gz ? zcat.close() != 0 : fclose(f) != 0)
        usage_error("cannot read factor base " + path);
    if (text.empty() || text.back() != '\n')
        text.push_back('\n');

    unsigned const n_chunks = std::max(1u, std::thread::hardware_concurrency());
    std::vector<size_t> cut = {0};
    for (unsigned i = 1; i < n_chunks; i++)
        cut.push_back(text.find('\n', text.size() * i / n_chunks) + 1);
    cut.push_back(text.size());
    return parallel_fb(n_chunks, [&](unsigned i, std::vector<fb_entry> & out) {
        parse_fb_chunk(text.data() + cut[i], text.data() + cut[i + 1], out);
    });
}

/* Factor base for a rational side (degree-1 polynomial g1*x + g0):
 * one root per prime, projective root 0 when p divides g1. */
std::vector<fb_entry> make_rational_fb(cxx_mpz_poly const & g, unsigned long lim)
{
    constexpr uint32_t POWER_BOUND = 4096;
    unsigned const n_chunks = std::max(1u, std::thread::hardware_concurrency());
    uint64_t const bound = lim + 1;
    return parallel_fb(n_chunks, [&](unsigned i, std::vector<fb_entry> & out) {
        uint64_t const lo = 2 + (bound - 2) * i / n_chunks;
        uint64_t const hi = 2 + (bound - 2) * (i + 1) / n_chunks;
        for (uint64_t p64 : primes_in(lo, hi)) {
            uint32_t const p = (uint32_t)p64;
            uint8_t const w = weight_units(p, 1);
            bool const resieve = p > TINY_PRIME_BOUND;
            if (mpz_fdiv_ui(mpz_poly_coeff_const(g, 1), p) == 0) {
                out.push_back({p, p + 0, w, resieve}); /* projective root 0 */
                continue;
            }
            uint64_t const q_max = std::max((uint64_t)p, (uint64_t)POWER_BOUND);
            for (uint64_t q = p; q <= q_max; q *= p) {
                /* root of g mod q = p^k: r = -g0 / g1 */
                uint64_t const g0 = mpz_fdiv_ui(mpz_poly_coeff_const(g, 0), q);
                uint64_t const g1 = mpz_fdiv_ui(mpz_poly_coeff_const(g, 1), q);
                uint64_t const r = g0 == 0 ? 0 :
                    (uint64_t)((unsigned __int128)(q - g0) *
                               inv_mod(g1, q) % q);
                out.push_back({(uint32_t)q, (uint32_t)r, w,
                               q == p && resieve});
                if (q > POWER_BOUND / p)
                    break;
            }
        }
    });
}

/* The factor base cache: the two parsed fb_entry vectors, written once
 * next to the side-1 factor base file and read by every later process
 * with the same polynomial, bounds, and file. Reading the file of
 * entries takes about a second; parsing the gzipped text and
 * generating the rational side takes about twelve. The key holds
 * everything the vectors depend on; the hash of the binary stands for
 * the parser, the entry layout, and the constants, so a new build
 * writes a new cache. GLAS_FB_CACHE=0 turns the cache off (the A/B
 * control). */
struct fb_cache_key {
    uint64_t magic = 0x676c617366626331; /* "glasfbc1" */
    uint64_t exe_hash = 0;  /* FNV-1a of /proc/self/exe */
    uint64_t poly_hash = 0; /* FNV-1a of the polynomial file */
    uint64_t lim[2] = {0, 0};
    uint64_t fb_size[2] = {0, 0};  /* size of the fb file, 0 if generated */
    uint64_t fb_mtime[2] = {0, 0}; /* its modification time, 0 if generated */

    bool operator==(fb_cache_key const & o) const
    {
        return memcmp(this, &o, sizeof *this) == 0;
    }
};

struct fb_cache_header {
    fb_cache_key key;
    uint64_t count[2];
};

uint64_t fnv1a_file(std::string const & path)
{
    FILE * f = fopen(path.c_str(), "rb");
    if (!f)
        usage_error("cannot open " + path);
    uint64_t h = 0xcbf29ce484222325;
    char buf[1 << 16];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;)
        for (size_t i = 0; i < n; i++)
            h = (h ^ (uint8_t)buf[i]) * 0x100000001b3;
    fclose(f);
    return h;
}

struct stat file_stat(std::string const & path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        usage_error("cannot stat factor base " + path);
    return st;
}

fb_cache_key fb_key(params const & P)
{
    fb_cache_key k;
    k.exe_hash = fnv1a_file("/proc/self/exe");
    k.poly_hash = fnv1a_file(P.poly_path);
    for (int side = 0; side < 2; side++) {
        k.lim[side] = P.side[side].lim;
        if (P.side[side].fb_path.empty())
            continue;
        struct stat const st = file_stat(P.side[side].fb_path);
        k.fb_size[side] = (uint64_t)st.st_size;
        k.fb_mtime[side] = (uint64_t)st.st_mtime;
    }
    return k;
}

/* Read the cache if it holds the key; false when the file is absent,
 * made for other inputs, or short (the process then parses and
 * rewrites it). */
bool read_fb_cache(std::string const & path, fb_cache_key const & key,
                   std::vector<fb_entry> (&fb)[2])
{
    FILE * f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    fb_cache_header h;
    bool ok = fread(&h, sizeof h, 1, f) == 1 && h.key == key;
    for (int side = 0; ok && side < 2; side++) {
        fb[side].resize(h.count[side]);
        ok = fread(fb[side].data(), sizeof(fb_entry), fb[side].size(), f) ==
             fb[side].size();
    }
    fclose(f);
    if (!ok)
        for (auto & v : fb)
            v.clear();
    return ok;
}

/* One writer per cache: the lock is a file made with O_EXCL, and a
 * process that cannot make it leaves the write to the holder. A lock
 * older than ten minutes (a write takes seconds) is a dead writer's
 * and is removed. */
bool take_lock(std::string const & lock)
{
    struct stat st;
    if (stat(lock.c_str(), &st) == 0 && time(nullptr) - st.st_mtime > 600)
        unlink(lock.c_str());
    int const fd = open(lock.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0)
        return false;
    close(fd);
    return true;
}

bool write_file(std::string const & path, fb_cache_header const & h,
                std::vector<fb_entry> const (&fb)[2])
{
    FILE * f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    bool ok = fwrite(&h, sizeof h, 1, f) == 1;
    for (int side = 0; side < 2; side++)
        ok = ok && fwrite(fb[side].data(), sizeof(fb_entry), fb[side].size(),
                          f) == fb[side].size();
    return fclose(f) == 0 && ok;
}

/* Write to a private name, then rename: a reader sees the whole file
 * or none. A write failure is reported, not fatal: the process has its
 * factor bases. Returns the log line. */
std::string write_fb_cache(std::string const & path, fb_cache_key const & key,
                           std::vector<fb_entry> const (&fb)[2])
{
    std::string const lock = path + ".lock";
    if (!take_lock(lock))
        return "# factor base cache: another process writes " + path + "\n";
    fb_cache_header const h = {key, {fb[0].size(), fb[1].size()}};
    char host[256] = "";
    gethostname(host, sizeof host - 1);
    std::string const tmp =
        path + ".tmp." + host + "." + std::to_string(getpid());
    bool const ok = write_file(tmp, h, fb) && rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok)
        unlink(tmp.c_str());
    unlink(lock.c_str());
    return "# factor base cache: " + std::string(ok ? "wrote " : "cannot write ") +
           path + "\n";
}

/* Load both factor bases: from the cache when one is valid, else from
 * the files and the polynomial, and then write the cache. Returns the
 * log line for the relation file header. */
std::string load_factor_bases(params const & P, cxx_cado_poly const & cpoly,
                              std::vector<fb_entry> (&fb)[2])
{
    char const * env = getenv("GLAS_FB_CACHE");
    bool const use_cache = !(env && strcmp(env, "0") == 0) &&
                           !P.side[1].fb_path.empty();
    std::string const cache_path = P.side[1].fb_path + ".cache";
    fb_cache_key key;
    if (use_cache) {
        key = fb_key(P);
        if (read_fb_cache(cache_path, key, fb))
            return "# factor base cache: read " + cache_path + "\n";
    }
    for (int side = 0; side < 2; side++) {
        if (!P.side[side].fb_path.empty())
            fb[side] = read_fb_file(P.side[side].fb_path);
        else if (cpoly[side].degree() == 1)
            fb[side] = make_rational_fb(cpoly[side], P.side[side].lim);
        else
            usage_error("no factor base file for side " + std::to_string(side));
    }
    if (!use_cache)
        return "";
    return write_fb_cache(cache_path, key, fb);
}

/* ------------------------------------------------------------------ */
/* q-lattice basis and sieve region (las adjust-strategy 0)           */

constexpr int LOG_BUCKET_REGION = 16;

struct basis64 {
    int64_t a0, b0, a1, b1;
};

/* SkewGauss reduction (las-qlattice.cpp).  Returns false when the
 * reduced basis does not fit in int64. */
bool skew_gauss(basis64 & B, uint64_t q, uint64_t rho, double skewness)
{
    double mantissa;
    int exponent;
    mantissa = frexp(skewness * skewness, &exponent);
    int64_t const mantissa_z = (int64_t)ldexp(mantissa, 53);
    exponent -= 53;

    cxx_mpz a[2], b[2], N[2], S, quot, tmp;
    mpz_set_uint64(a[0], q);
    mpz_set_uint64(a[1], rho);
    mpz_set_ui(b[0], 0);
    mpz_set_ui(b[1], 1);

    auto quadform = [&](cxx_mpz & dst, int i0, int i1) {
        mpz_mul(dst, a[i0], a[i1]);
        mpz_mul(tmp, b[i0], b[i1]);
        if (exponent < 0)
            mpz_mul_2exp(dst, dst, -exponent);
        if (exponent > 0)
            mpz_mul_2exp(tmp, tmp, exponent);
        mpz_addmul_int64(dst, tmp, mantissa_z);
    };
    quadform(N[0], 0, 0);
    quadform(N[1], 1, 1);
    quadform(S, 0, 1);

    for (;;) {
        mpz_rdiv_q(quot, S, N[1]);
        if (mpz_cmp_ui(quot, 0) == 0)
            break;
        mpz_submul(a[0], quot, a[1]);
        mpz_submul(b[0], quot, b[1]);
        mpz_set(tmp, S);
        mpz_submul(S, quot, N[1]);
        mpz_add(tmp, tmp, S);
        mpz_submul(N[0], quot, tmp);

        mpz_rdiv_q(quot, S, N[0]);
        if (mpz_cmp_ui(quot, 0) == 0)
            break;
        mpz_submul(a[1], quot, a[0]);
        mpz_submul(b[1], quot, b[0]);
        mpz_set(tmp, S);
        mpz_submul(S, quot, N[0]);
        mpz_add(tmp, tmp, S);
        mpz_submul(N[1], quot, tmp);
    }
    if (mpz_cmp(N[0], N[1]) > 0) {
        mpz_swap(a[0], a[1]);
        mpz_swap(b[0], b[1]);
    }
    if (!mpz_fits_sint64_p(a[0]) || !mpz_fits_sint64_p(b[0]) ||
        !mpz_fits_sint64_p(a[1]) || !mpz_fits_sint64_p(b[1]))
        return false;
    B.a0 = mpz_get_int64(a[0]);
    B.b0 = mpz_get_int64(b[0]);
    B.a1 = mpz_get_int64(a[1]);
    B.b1 = mpz_get_int64(b[1]);
    return true;
}

bool fits_31bits(basis64 const & B)
{
    constexpr int64_t t31 = INT64_C(1) << 31;
    return B.a0 >= -t31 && B.a0 < t31 && B.a1 >= -t31 && B.a1 < t31 &&
           B.b0 >= -t31 && B.b0 < t31 && B.b1 >= -t31 && B.b1 < t31;
}

/* max |f(x)| for x in [-s, s] (las get_maxnorm_aux) */
double maxnorm_aux(polynomial<double> const & f, double s)
{
    if (f.degree() <= 0)
        return std::fabs(f[0]);
    double gmax = std::max(std::fabs(f(-s)), std::fabs(f(s)));
    for (auto x : f.derivative().template roots<double>(s))
        gmax = std::max(gmax, std::fabs(f(x)));
    return gmax;
}

/* max |F(x,y)| for -X <= x <= X, 0 <= y <= Y (las get_maxnorm_rectangular) */
double maxnorm_rectangular(polynomial<double> const & f, double X, double Y)
{
    double const d = f.degree();
    double const b = maxnorm_aux(f, X / Y) * std::pow(Y, d);
    double const a = maxnorm_aux(f.reciprocal(), Y / X) * std::pow(X, d);
    return std::max(a, b);
}

/* max |F(X cos t, Y sin t)| for 0 <= t <= pi (las get_maxnorm_circular) */
double maxnorm_circular(polynomial<double> const & f, double X, double Y)
{
    int const d = f.degree();
    polynomial<double> poly;
    poly[0] = 0.0;
    poly[1] = 0.0;
    for (int i = 1; i <= d; i++) {
        double const t = std::pow(X / Y, (double)i - 1.0);
        poly[i + 1] = X * (double)i * f[i] * t;
        poly[i - 1] += X * (double)i * f[i] * t;
    }
    for (int i = 0; i <= d; i++) {
        double const t = f[i] * std::pow(X / Y, (double)i);
        poly[i + 1] -= (double)d * Y * t;
    }
    double max_norm = std::fabs(f[d] * std::pow(X, (double)d));
    for (double const u : poly.template roots<double>()) {
        double const x = X * u / std::sqrt(1.0 + u * u);
        double const y = Y / std::sqrt(1.0 + u * u);
        max_norm = std::max(max_norm, std::fabs(f(x, y)));
    }
    return max_norm;
}

/* One survivor that passed the leftover-size bounds; the leftovers
 * still need splitting into large primes. */
struct candidate {
    int64_t a, b;
    std::vector<uint64_t> factors[2];
    cxx_mpz leftover[2];
};

/* Pass-1 output for one special-q: the report header plus the
 * candidates whose cofactorization is still pending. */
struct cofac_job {
    size_t idx;
    std::string text;
    std::vector<candidate> cands;
};

/* GLAS_STREAM state of one special-q: workers store each chunk's
 * candidates in parts; pending counts the sieve plus the queued
 * chunks, and the task is complete when it reaches zero. */
struct task_state {
    size_t idx;
    sq_task t;
    std::atomic<size_t> nsurv{0};
    std::atomic<int> pending{1};
    std::mutex m;
    std::vector<std::vector<candidate>> parts;
};

/* One work-queue item: a chunk to process, or, with chunk == SIZE_MAX,
 * a complete task whose candidates go to cofactorization.  make
 * assembles the chunk's sieve_result; the worker runs it, so the
 * siever thread hands a chunk off without unpacking it. */
struct chunk_work {
    std::shared_ptr<task_state> ts;
    size_t chunk;
    chunk_result make;
};

struct siever {
    params P;
    cxx_cado_poly cpoly;
    std::vector<fb_entry> fb[2];
    std::unique_ptr<facul_strategies> strategies;
    tiny_table tiny;
    filter_params flt;

    std::optional<sq_task> make_task(uint64_t q, uint64_t rho) const;
    std::vector<candidate> pass1_cands(sq_task const & t,
                                       sieve_result const & res) const;
    cofac_job pass1(size_t idx, sq_task const & t,
                    sieve_result const & res) const;
    void cofactor_cpu(std::vector<candidate> const & cands,
                      std::vector<cofac_result> & fr) const;
    void cofactor_finish(std::vector<candidate> const & cands,
                         cofac_result * fr) const;
    void pass3(cofac_job & job, cofac_result const * fr,
               unsigned long & nreports) const;
};

/* Build the task for one special-q: q-lattice basis, sieve region and
 * survivor bounds.  Returns nothing when las would discard the q. */
std::optional<sq_task> siever::make_task(uint64_t q, uint64_t rho) const
{
    basis64 B;
    if (!skew_gauss(B, q, rho, cpoly.skew) || !fits_31bits(B))
        return std::nullopt;

    double const skew = cpoly.skew;
    double const rt_skew = std::sqrt(skew);
    auto skewed_norm = [skew](int64_t a, int64_t b) {
        return (double)a * (double)a / skew + (double)b * (double)b * skew;
    };
    if (skewed_norm(B.a0, B.b0) > skewed_norm(B.a1, B.b1)) {
        std::swap(B.a0, B.a1);
        std::swap(B.b0, B.b1);
    }

    int const logA = P.logA;
    int const logI = (logA + 1) / 2;
    uint32_t const I = 1u << logI;
    double const maxab1 = std::max(std::fabs((double)B.a1) / rt_skew,
                                   std::fabs((double)B.b1) * rt_skew);
    double const bnd = std::sqrt(2.0 * (double)q / std::sqrt(3.0));
    uint64_t J = (uint64_t)(bnd / maxab1 * (double)(1UL << ((logA - 1) / 2)));
    J = std::min(J, (uint64_t)I >> 1);

    uint32_t const j_round = 1u << std::max(LOG_BUCKET_REGION - logI, 0);
    if (J / j_round == 0)
        return std::nullopt;  /* too-skewed basis: discard */

    sq_task t;
    t.q = q;
    t.rho = rho;
    t.sq_side = P.sqside;
    t.a0 = B.a0;
    t.b0 = B.b0;
    t.a1 = B.a1;
    t.b1 = B.b1;
    t.logI = logI;

    /* norm polynomials in (i,j) coordinates */
    polynomial<double> fijd[2];
    for (int side = 0; side < 2; side++) {
        cxx_mpz_poly fij = cpoly[side].homography({B.a0, B.b0, B.a1, B.b1});
        if (fij.degree() < cpoly[side].degree())
            return std::nullopt;  /* homography tripped over a root */
        if (side == P.sqside)
            fij = fij.divexact(cxx_mpz(q));
        fijd[side] = polynomial<double>(fij);
        t.degree[side] = fijd[side].degree();
        for (int k = 0; k <= t.degree[side]; k++)
            t.fijd[side][k] = fijd[side][k];
    }

    /* largest J such that the rectangle norm stays below the ideal
     * circular norm (las sieve_info_update_norm_data_Jmax, fudge 2.0) */
    double const fudge_factor = 2.0;
    double const A = (double)I * std::sqrt((double)q * skew);
    double const Bc = (double)(1u << (logA - logI)) * std::sqrt((double)q / skew);
    uint32_t Jmax = 1u << (logA - logI);
    for (int side = 0; side < 2; side++) {
        polynomial<double> dpoly(cpoly[side]);
        double maxnorm = maxnorm_circular(dpoly, fudge_factor * A / 2.0,
                                          fudge_factor * Bc);
        if (side == P.sqside)
            maxnorm /= (double)q;
        auto const norm = [&](uint32_t j) {
            return maxnorm_rectangular(fijd[side], (double)I / 2, (double)j);
        };
        if (norm(Jmax) > maxnorm) {
            /* invariant: norm(hi) >= maxnorm, and norm(lo) < maxnorm or
             * lo == 0; hi ends as the smallest such integer */
            uint32_t lo = 0, hi = Jmax;
            while (hi - lo > 1) {
                uint32_t const mid = lo + (hi - lo) / 2;
                if (norm(mid) < maxnorm)
                    lo = mid;
                else
                    hi = mid;
            }
            Jmax = hi;
        }
    }
    J = Jmax;
    J = J / j_round * j_round;
    if (J < (1u << std::max(1, LOG_BUCKET_REGION - logI)))
        return std::nullopt;
    t.J = (uint32_t)J;

    /* survivor bounds (las lognorm_base::ctor_common), capped at
     * mfb + a margin: the host rejects any leftover cofactor above
     * mfb bits, so cells past that cannot yield a relation.  The
     * margin covers the sieve contributions that are not subtracted
     * (prime powers above the power bound) plus the norm-estimate
     * rounding. */
    double const mfb_margin = 8.0;
    for (int side = 0; side < 2; side++) {
        double const maxlog2 =
            std::log2(maxnorm_rectangular(fijd[side], (double)(I / 2), (double)J));
        double lambda = P.side[side].lambda;
        if (lambda == 0.0)
            lambda = 0.3 + (double)P.side[side].mfb / (double)P.side[side].lpb;
        double const r = std::min({maxlog2, lambda * (double)P.side[side].lpb,
                                   (double)P.side[side].mfb + mfb_margin});
        t.bound[side] = (int32_t)std::lround(r * LOG_SCALE);
    }
    return t;
}

/* ------------------------------------------------------------------ */
/* survivor processing                                                */

/* fixed-width unsigned arithmetic for exact norms (device wide twin) */
struct wide {
    uint64_t v[NORM_LIMBS];
};

using u128 = unsigned __int128;

int wide_cmp(wide const & a, wide const & b)
{
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        if (a.v[i] != b.v[i])
            return a.v[i] < b.v[i] ? -1 : 1;
    return 0;
}

void wide_add(wide & a, wide const & b)
{
    uint64_t carry = 0;
    for (int i = 0; i < NORM_LIMBS; i++) {
        u128 const s = (u128)a.v[i] + b.v[i] + carry;
        a.v[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
}

void wide_sub(wide & a, wide const & b)   /* needs a >= b */
{
    uint64_t borrow = 0;
    for (int i = 0; i < NORM_LIMBS; i++) {
        u128 const s = (u128)a.v[i] - b.v[i] - borrow;
        a.v[i] = (uint64_t)s;
        borrow = (uint64_t)(s >> 64) & 1;
    }
}

void wide_mul_u64(wide & a, uint64_t m)
{
    uint64_t carry = 0;
    for (int i = 0; i < NORM_LIMBS; i++) {
        u128 const p = (u128)a.v[i] * m + carry;
        a.v[i] = (uint64_t)p;
        carry = (uint64_t)(p >> 64);
    }
}

uint64_t wide_divrem_u64(wide & a, uint64_t p)
{
    uint64_t rem = 0;
    for (int i = NORM_LIMBS - 1; i >= 0; i--) {
        u128 const cur = (u128)rem << 64 | a.v[i];
        a.v[i] = (uint64_t)(cur / p);
        rem = (uint64_t)(cur % p);
    }
    return rem;
}

void wide_divide_out(wide & n, uint64_t p, std::vector<uint64_t> & factors)
{
    for (;;) {
        wide q = n;
        if (wide_divrem_u64(q, p))
            return;
        n = q;
        factors.push_back(p);
    }
}

int wide_bits(wide const & a)
{
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        if (a.v[i])
            return 64 * (i + 1) - __builtin_clzll(a.v[i]);
    return 0;
}

double wide_to_double(wide const & a)
{
    double v = 0.0;
    for (int i = NORM_LIMBS - 1; i >= 0; i--)
        v = v * 18446744073709551616.0 + (double)a.v[i];
    return v;
}

void wide_to_mpz(cxx_mpz & out, wide const & a)
{
    mpz_import(out, NORM_LIMBS, -1, 8, 0, 0, a.v);
}

/* |F_side(a, b)| with b >= 0: signed sum of c_k a^k b^(d-k) */
void wide_norm(filter_params const & flt, int side, int d, int64_t a,
               uint64_t b, wide & out)
{
    uint64_t const aa = a < 0 ? (uint64_t)-a : (uint64_t)a;
    wide pos = {}, neg = {};
    for (int k = 0; k <= d; k++) {
        wide term;
        for (int i = 0; i < NORM_LIMBS; i++)
            term.v[i] = flt.coeff[side][k].limb[i];
        for (int m = 0; m < k; m++)
            wide_mul_u64(term, aa);
        for (int m = k; m < d; m++)
            wide_mul_u64(term, b);
        bool const tneg = flt.coeff[side][k].neg ^ (a < 0 && (k & 1));
        wide_add(tneg ? neg : pos, term);
    }
    if (wide_cmp(pos, neg) >= 0) {
        wide_sub(pos, neg);
        out = pos;
    } else {
        wide_sub(neg, pos);
        out = neg;
    }
}

/* las check_leftover_norm: can this leftover norm yield a relation? */
bool leftover_ok(wide const & n, side_params const & sp)
{
    int const s = wide_bits(n);
    if (s == 0 || s > (int)sp.mfb)
        return false;
    if (s <= (int)sp.lpb)
        return true;
    double const nd = wide_to_double(n);
    double const B = (double)sp.lim;
    double kB = B * B;
    for (unsigned int klpb = sp.lpb; klpb < (unsigned int)s;
         klpb += sp.lpb, kB *= B)
        if (nd < kB)
            return false;
    cxx_mpz z;
    wide_to_mpz(z, n);
    if (mpz_probab_prime_p(z, 1))
        return false;  /* prime larger than the large prime bound */
    return true;
}

/* process-phase profile; GLAS_GPU_STATS prints it at exit */
std::atomic<uint64_t> proc_ns[4];
std::atomic<uint64_t> proc_n[3];
constexpr int PH_NORM = 0, PH_DIVIDE = 1, PH_TINY = 2, PH_FACUL = 3;
constexpr int PN_SURVIVOR = 0, PN_COPRIME = 1, PN_FACUL = 2;

double wallclock()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

struct stopwatch {
    double last = wallclock();
    void lap(std::atomic<uint64_t> & ns)
    {
        double const now = wallclock();
        ns += (uint64_t)(1e9 * (now - last));
        last = now;
    }
};

/* GLAS_COFAC_DUMP=path appends every candidate leftover pair to a
 * binary file for the standalone cofactorization harness
 * (cofac_bench).  Header: magic then per side lim, lpb, mfb, ncurves;
 * records: the two leftovers as two 64-bit limbs each. */
constexpr uint64_t COFAC_DUMP_MAGIC = UINT64_C(0x676c61732d636f66);

void dump_cofac_pairs(std::vector<candidate> const & cands,
                      params const & P)
{
    static std::mutex mtx;
    static FILE * fp = nullptr;
    static bool init = false;
    std::lock_guard<std::mutex> lock(mtx);
    if (!init) {
        init = true;
        if (char const * path = getenv("GLAS_COFAC_DUMP")) {
            fp = fopen(path, "wb");
            if (fp) {
                uint64_t hdr[9] = {COFAC_DUMP_MAGIC};
                for (int side = 0; side < 2; side++) {
                    hdr[1 + 4 * side] = P.side[side].lim;
                    hdr[2 + 4 * side] = P.side[side].lpb;
                    hdr[3 + 4 * side] = P.side[side].mfb;
                    hdr[4 + 4 * side] = (uint64_t)(int64_t)P.side[side].ncurves;
                }
                fwrite(hdr, sizeof hdr, 1, fp);
            }
        }
    }
    if (!fp)
        return;
    for (candidate const & c : cands) {
        uint64_t rec[4];
        for (int side = 0; side < 2; side++) {
            rec[2 * side] = mpz_getlimbn(c.leftover[side], 0);
            rec[2 * side + 1] = mpz_getlimbn(c.leftover[side], 1);
        }
        fwrite(rec, sizeof rec, 1, fp);
    }
    fflush(fp);
}

std::string task_header(sq_task const & t, size_t nsurv)
{
    char buf[256];
    snprintf(buf, sizeof buf,
             "# Sieving side-%d q=%" PRIu64 "; rho=%" PRIu64
             "; a0=%" PRId64 "; b0=%" PRId64 "; a1=%" PRId64 "; b1=%" PRId64
             "; J=%u; %zu survivors\n",
             t.sq_side, t.q, t.rho, t.a0, t.b0, t.a1, t.b1, t.J, nsurv);
    return buf;
}

/* One GPU-assembled survivor: rebuild (a, b) from the position and
 * unpack the factors and leftovers the filter kernel recorded.  False
 * when a leftover that owes its primality test is prime. */
bool from_gpu(sq_task const & t, gpu_cand const & g, candidate & c)
{
    uint32_t const I = 1u << t.logI;
    int64_t const i = (int64_t)(g.x & (I - 1)) - I / 2;
    int64_t const j = g.x >> t.logI;
    c.a = i * t.a0 + j * t.a1;
    c.b = i * t.b0 + j * t.b1;
    if (c.b < 0) {
        c.a = -c.a;
        c.b = -c.b;
    }
    uint32_t const * f = g.fac;
    for (int side = 0; side < 2; side++) {
        if (side == t.sq_side)
            c.factors[side].insert(c.factors[side].end(), g.nq, t.q);
        c.factors[side].insert(c.factors[side].end(), f, f + g.nf[side]);
        f += g.nf[side];
        mpz_import(c.leftover[side], 2, -1, 8, 0, 0, g.leftover[side]);
        if (g.pend >> side & 1 && mpz_probab_prime_p(c.leftover[side], 1))
            return false;
    }
    return true;
}

/* pass 1: exact norms, sieved-prime division, candidate collection.
 * res.cands come assembled from the GPU; res.survivors get the norm
 * work here.  Both lists are sorted by position and the output
 * merges them back into position order. */
std::vector<candidate> siever::pass1_cands(sq_task const & t,
                                           sieve_result const & res) const
{
    uint32_t const I = 1u << t.logI;
    std::vector<candidate> cands;
    std::vector<resieve_hit> const & hits = res.hits;
    wide norm[2];
    size_t hi = 0;
    size_t gi = 0;
    auto take_gpu = [&](uint64_t xbound) {
        for (; gi < res.cands.size() && res.cands[gi].x < xbound; gi++) {
            candidate c;
            if (from_gpu(t, res.cands[gi], c))
                cands.push_back(std::move(c));
        }
    };
    proc_n[PN_SURVIVOR] += res.nsurv();
    proc_n[PN_COPRIME] += res.cands.size();
    for (survivor const & sv : res.survivors) {
        uint64_t const x = sv.x;
        take_gpu(x);
        while (hi < hits.size() && hits[hi].x < x)
            hi++;
        size_t const h0 = hi;
        size_t h1 = hi;
        while (h1 < hits.size() && hits[h1].x == x)
            h1++;

        int64_t const i = (int64_t)(x & (I - 1)) - I / 2;
        int64_t const j = x >> t.logI;
        if (std::gcd(std::abs(i), j) != 1)
            continue;
        int64_t a = i * t.a0 + j * t.a1;
        int64_t b = i * t.b0 + j * t.b1;
        if (b < 0) {
            a = -a;
            b = -b;
        }
        if (b == 0)
            continue;

        proc_n[PN_COPRIME]++;
        std::vector<uint64_t> factors[2];
        stopwatch sw;
        bool ok = true;
        for (int side = 0; side < 2 && ok; side++) {
            wide_norm(flt, side, cpoly[side]->deg, a, (uint64_t)b,
                      norm[side]);
            sw.lap(proc_ns[PH_NORM]);
            if (side == t.sq_side) {
                wide q = norm[side];
                if (wide_divrem_u64(q, t.q)) {
                    ok = false;
                    break;
                }
                norm[side] = q;
                factors[side].push_back(t.q);
                wide_divide_out(norm[side], t.q, factors[side]);
            }
            for (size_t h = h0; h < h1; h++)
                if (hits[h].side == side)
                    wide_divide_out(norm[side], hits[h].q, factors[side]);
            sw.lap(proc_ns[PH_DIVIDE]);
            for (uint64_t m = sv.tiny[side]; m; m &= m - 1)
                wide_divide_out(norm[side], tiny.primes[__builtin_ctzll(m)],
                                factors[side]);
            ok = leftover_ok(norm[side], P.side[side]);
            sw.lap(proc_ns[PH_TINY]);
        }
        if (!ok)
            continue;

        candidate c;
        c.a = a;
        c.b = b;
        for (int side = 0; side < 2; side++) {
            c.factors[side] = std::move(factors[side]);
            wide_to_mpz(c.leftover[side], norm[side]);
        }
        cands.push_back(std::move(c));
    }
    take_gpu(UINT64_MAX);
    proc_n[PN_FACUL] += cands.size();
    dump_cofac_pairs(cands, P);
    return cands;
}

cofac_job siever::pass1(size_t idx, sq_task const & t,
                        sieve_result const & res) const
{
    cofac_job job;
    job.idx = idx;
    job.text = task_header(t, res.nsurv());
    job.cands = pass1_cands(t, res);
    return job;
}

/* pass 2 on the host: per-candidate facul (MR + ECM chains) */
void siever::cofactor_cpu(std::vector<candidate> const & cands,
                          std::vector<cofac_result> & fr) const
{
    fr.resize(cands.size());
    stopwatch sw;
    for (size_t k = 0; k < cands.size(); k++) {
        std::vector<cxx_mpz> leftover(2);
        leftover[0] = cands[k].leftover[0];
        leftover[1] = cands[k].leftover[1];
        auto const fac = facul_all(leftover, *strategies);
        cofac_result & r = fr[k];
        r = {};
        r.smooth = fac[0].status == FACUL_SMOOTH &&
                   fac[1].status == FACUL_SMOOTH;
        for (int side = 0; r.smooth && side < 2; side++)
            for (cxx_mpz const & pz : fac[side].primes) {
                assert(r.nf[side] < COFAC_MAX_FACTORS);
                r.f[side][r.nf[side]++] = mpz_get_uint64(pz);
            }
    }
    sw.lap(proc_ns[PH_FACUL]);
}

/* Finish the sides the GPU left pending: facul (ECM chains) on the
 * hard composites. */
void siever::cofactor_finish(std::vector<candidate> const & cands,
                             cofac_result * fr) const
{
    for (size_t k = 0; k < cands.size(); k++) {
        cofac_result & r = fr[k];
        if (!r.smooth || !r.pending)
            continue;
        std::vector<cxx_mpz> leftover(2);
        for (int side = 0; side < 2; side++)
            if (r.pending & (1 << side))
                leftover[side] = cands[k].leftover[side];
            else
                mpz_set_ui(leftover[side], 1);
        auto const fac = facul_all(leftover, *strategies);
        r.smooth = fac[0].status == FACUL_SMOOTH &&
                   fac[1].status == FACUL_SMOOTH;
        for (int side = 0; r.smooth && side < 2; side++)
            for (cxx_mpz const & pz : fac[side].primes) {
                assert(r.nf[side] < COFAC_MAX_FACTORS);
                r.f[side][r.nf[side]++] = mpz_get_uint64(pz);
            }
    }
}

/* pass 3: check the split against the leftover, then print */
void siever::pass3(cofac_job & job, cofac_result const * fr,
                   unsigned long & nreports) const
{
    std::vector<candidate> & cands = job.cands;
    std::string & out = job.text;
    char buf[256];
    cxx_mpz prod, pz;
    for (size_t k = 0; k < cands.size(); k++) {
        if (!fr[k].smooth)
            continue;
        candidate & c = cands[k];
        bool ok = true;
        for (int side = 0; ok && side < 2; side++) {
            mpz_set_ui(prod, 1);
            for (int m = 0; m < fr[k].nf[side]; m++) {
                uint64_t const p = fr[k].f[side][m];
                mpz_set_uint64(pz, p);
                mpz_mul(prod, prod, pz);
                ok = ok && mpz_probab_prime_p(pz, 1);
                c.factors[side].push_back(p);
            }
            ok = ok && mpz_cmp(prod, c.leftover[side]) == 0;
        }
        if (!ok)
            continue;

        snprintf(buf, sizeof buf, "%" PRId64 ",%" PRId64, c.a, c.b);
        out += buf;
        for (int side = 0; side < 2; side++) {
            std::sort(c.factors[side].begin(), c.factors[side].end());
            char sep = ':';
            for (uint64_t const p : c.factors[side]) {
                snprintf(buf, sizeof buf, "%c%" PRIx64, sep, p);
                out += buf;
                sep = ',';
            }
            if (c.factors[side].empty())
                out += sep == ':' ? ":" : "";
        }
        out += '\n';
        nreports++;
    }
}

}  // namespace

int main(int argc, char const ** argv)
{
    double const t_start = wallclock();
    siever S;
    S.P = parse_args(argc, argv);
    params const & P = S.P;

    if (!S.cpoly.read(P.poly_path))
        usage_error("cannot read polynomial file " + P.poly_path);
    if (S.cpoly.nsides() != 2)
        usage_error("only two-side polynomials are supported");

    FILE * out = stdout;
    child_pipe gzip = {nullptr, 0};
    if (!P.out_path.empty()) {
        int const fd = open(P.out_path.c_str(),
                            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
        if (fd < 0)
            usage_error("cannot open output file " + P.out_path);
        if (is_gz(P.out_path)) {
            gzip = pipe_open({"gzip", "-c"}, "w", fd);
            close(fd);
            out = gzip.f;
        } else {
            out = fdopen(fd, "w");
        }
        if (!out)
            usage_error("cannot open output file " + P.out_path);
    }
    fprintf(out, "# glas with q0=%" PRIu64 " q1=%" PRIu64 "\n", P.q0, P.q1);

    std::thread warmup;
    if (P.gpu)
        warmup = std::thread(gpu_warmup);

    /* factor bases */
    std::string const fb_log = load_factor_bases(P, S.cpoly, S.fb);
    for (int side = 0; side < 2; side++)
        fprintf(out, "# side-%d factor base: %zu entries\n", side,
                S.fb[side].size());
    fputs(fb_log.c_str(), out);
    S.tiny.primes = primes_up_to(TINY_PRIME_BOUND);
    for (int side = 0; side < 2; side++) {
        S.tiny.cmod[side].assign(
            S.tiny.primes.size() * (GLAS_MAX_DEGREE + 1), 0);
        for (size_t k = 0; k < S.tiny.primes.size(); k++)
            for (int d = 0; d <= S.cpoly[side]->deg; d++)
                S.tiny.cmod[side][k * (GLAS_MAX_DEGREE + 1) + d] =
                    mpz_fdiv_ui(mpz_poly_coeff_const(S.cpoly[side], d),
                                S.tiny.primes[k]);
    }
    double const t_fb = wallclock();

    /* exact-norm filter constants: |coefficients| as fixed-width limbs.
     * |a|, |b| < 2^(32 + logI) with a 31-bit-bounded basis, so each of
     * the d multiplications in wide_norm adds at most 32 + logI bits,
     * and the sum of the d + 1 <= 9 terms adds at most 4 more. */
    int const logI = (P.logA + 1) / 2;
    S.flt = {};
    for (int side = 0; side < 2; side++) {
        int const d = S.cpoly[side]->deg;
        for (int k = 0; k <= d; k++) {
            mpz_srcptr const c = mpz_poly_coeff_const(S.cpoly[side], k);
            if (mpz_sizeinbase(c, 2) + (32 + (size_t)logI) * d + 4 >
                64 * NORM_LIMBS)
                usage_error("polynomial too large for exact-norm filter");
            mpz_export(S.flt.coeff[side][k].limb, nullptr, -1, 8, 0, 0, c);
            S.flt.coeff[side][k].neg = mpz_sgn(c) < 0;
        }
        S.flt.lim[side] = P.side[side].lim;
        S.flt.lpb[side] = P.side[side].lpb;
        S.flt.mfb[side] = P.side[side].mfb;
        assert(P.side[side].mfb <= 120);
    }

    if (warmup.joinable())
        warmup.join();
    if (P.gpu && !gpu_init(S.fb, logI, 1u << (P.logA - logI), S.tiny, S.flt))
        usage_error("-gpu requested but no CUDA device is available");
    double const t_gpu = wallclock();

    /* cofactorization strategies */
    {
        std::vector<unsigned long> lim = {P.side[0].lim, P.side[1].lim};
        std::vector<unsigned int> lpb = {P.side[0].lpb, P.side[1].lpb};
        std::vector<unsigned int> mfb = {P.side[0].mfb, P.side[1].mfb};
        std::vector<int> ncurves = {P.side[0].ncurves, P.side[1].ncurves};
        for (int side = 0; side < 2; side++)
            if (ncurves[side] < 0)
                ncurves[side] = nb_curves(lpb[side], mfb[side]);
        char const * order = getenv("GLAS_COFAC_ORDER");
        S.strategies = std::make_unique<facul_strategies>(
            make_strategies(order ? order : "cheap-first", lim, lpb, mfb, ncurves));
    }

    /* special-q list */
    std::vector<std::pair<uint64_t, uint64_t>> sq_list;
    {
        cxx_gmp_randstate rstate;
        for (uint64_t const q : primes_in(P.q0, P.q1)) {
            uint64_t roots[GLAS_MAX_DEGREE + 1];
            unsigned int const n =
                mpz_poly_roots_uint64(roots, S.cpoly[P.sqside], q, rstate);
            std::sort(roots, roots + n);
            for (unsigned int k = 0; k < n; k++)
                sq_list.emplace_back(q, roots[k]);
        }
    }

    fprintf(out, "# Startup: %0.2f s fb, %0.2f s gpu, %0.2f s rest\n",
            t_fb - t_start, t_gpu - t_fb, wallclock() - t_gpu);

    /* process the special-q tasks with a thread pool; print in order */
    std::atomic<size_t> next{0};
    std::atomic<unsigned long> total_reports{0};
    std::atomic<uint64_t> sieve_us{0}, process_us{0};
    std::mutex out_mutex;
    std::map<size_t, std::string> pending;
    size_t next_print = 0;
    /* phase split: first sieve start and last task output */
    std::atomic<double> t_first_sieve{0.0};
    double t_last_done = 0.0;

    auto emit = [&](size_t idx, std::string text) {
        std::lock_guard<std::mutex> const lock(out_mutex);
        t_last_done = wallclock();
        pending[idx] = std::move(text);
        while (!pending.empty() && pending.begin()->first == next_print) {
            fputs(pending.begin()->second.c_str(), out);
            pending.erase(pending.begin());
            next_print++;
        }
    };
    auto discard = [&](size_t idx, uint64_t q, uint64_t rho) {
        emit(idx, "# Discarding side-" + std::to_string(P.sqside) +
                      " q=" + std::to_string(q) +
                      "; rho=" + std::to_string(rho) + "\n");
    };
    auto sieve_task = [&](sq_task const & t, chunk_sink const & sink) {
        double const t0 = wallclock();
        double expected = 0.0;
        t_first_sieve.compare_exchange_strong(expected, t0);
        if (P.gpu)
            sieve_one_gpu(t, S.fb, S.tiny, sink);
        else {
            sieve_result r = sieve_one_cpu(t, S.fb, S.tiny);
            sink([r = std::move(r)]() mutable { return std::move(r); });
        }
        sieve_us += (uint64_t)(1e6 * (wallclock() - t0));
    };

    /* The GPU cofactor kernels need tens of thousands of candidates
     * per launch to fill the device, but one special-q yields only a
     * few hundred.  The workers park their pass-1 jobs in one shared
     * pool; the worker whose push crosses the flush bound takes the
     * whole pool and cofactors it in one call, so a launch carries
     * every worker's jobs instead of one worker's. */
    size_t flush_bound = 1u << 16;
    if (auto const * v = getenv("GLAS_COFAC_BATCH"))
        flush_bound = strtoul(v, nullptr, 10);
    /* GLAS_COFAC_HYBRID and GLAS_COFAC_GPU are experimental diagnostics
     * of the device path: no real configuration sets them, and
     * benchmarks of a change need only the adaptive default. Both
     * measure slower than adaptive on large inputs, and
     * GLAS_COFAC_GPU loses relations. GLAS_COFAC_ADAPT=0
     * pins cofactorization to the host: the fixed-routing reference for
     * relation-set comparisons. */
    bool const hybrid = getenv("GLAS_COFAC_HYBRID") != nullptr;
    bool const gpu_cofac =
        P.gpu && (hybrid || getenv("GLAS_COFAC_GPU") != nullptr);
    bool const adaptive =
        P.gpu && !gpu_cofac && env_flag("GLAS_COFAC_ADAPT", true);
    bool const finish_host = hybrid || adaptive;
    bool const stream = P.gpu && env_flag("GLAS_STREAM", true);
    std::atomic<uint64_t> host_cands{0}, device_cands{0};

    /* A device batch comes back as one finish item per job (host
     * finish of the pending sides, then relation output).  The stream
     * path queues the items so every worker takes a share instead of
     * the flushing worker walking the whole batch alone. */
    struct finish_item {
        cofac_job job;
        std::vector<cofac_result> fr;
    };
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<finish_item> finish_queue;

    auto finish_device = [&](finish_item it) {
        stopwatch sw;
        if (finish_host)
            S.cofactor_finish(it.job.cands, it.fr.data());
        unsigned long n = 0;
        S.pass3(it.job, it.fr.data(), n);
        total_reports += n;
        emit(it.job.idx, std::move(it.job.text));
        sw.lap(proc_ns[PH_FACUL]);
    };

    auto flush = [&](std::vector<cofac_job> jobs) {
        std::vector<cofac_pair> batch;
        for (cofac_job const & j : jobs)
            for (candidate const & c : j.cands) {
                cofac_pair p;
                for (int side = 0; side < 2; side++) {
                    assert(mpz_size(c.leftover[side]) <= 2);
                    p.n[side][0] = mpz_getlimbn(c.leftover[side], 0);
                    p.n[side][1] = mpz_getlimbn(c.leftover[side], 1);
                }
                batch.push_back(p);
            }
        std::vector<cofac_result> fr;
        int const lpb[2] = {(int)P.side[0].lpb, (int)P.side[1].lpb};
        stopwatch sw;
        device_cands += batch.size();
        gpu_cofactor(batch, fr, lpb, !finish_host);
        sw.lap(proc_ns[PH_FACUL]);
        size_t off = 0;
        for (cofac_job & j : jobs) {
            size_t const n = j.cands.size();
            finish_item it{std::move(j),
                           {fr.begin() + off, fr.begin() + off + n}};
            off += n;
            if (!stream) {
                finish_device(std::move(it));
                continue;
            }
            {
                std::lock_guard<std::mutex> const lk(qm);
                finish_queue.push_back(std::move(it));
            }
            qcv.notify_one();
        }
    };

    auto finish_cpu = [&](cofac_job job) {
        host_cands += job.cands.size();
        std::vector<cofac_result> fr;
        S.cofactor_cpu(job.cands, fr);
        unsigned long n = 0;
        S.pass3(job, fr.data(), n);
        total_reports += n;
        emit(job.idx, std::move(job.text));
    };

    std::mutex pool_mutex;
    std::vector<cofac_job> pool_jobs;
    size_t pool_cands = 0;

    auto pool_push = [&](cofac_job job) {
        std::vector<cofac_job> jobs;
        {
            std::lock_guard<std::mutex> const lk(pool_mutex);
            pool_cands += job.cands.size();
            pool_jobs.push_back(std::move(job));
            if (pool_cands < flush_bound)
                return;
            jobs.swap(pool_jobs);
            pool_cands = 0;
        }
        flush(std::move(jobs));
    };
    /* Empties the pool when the route turns to the host and when a
     * worker exits, so no job waits past the route change or the
     * end of the run. */
    auto pool_drain = [&]() {
        std::vector<cofac_job> jobs;
        {
            std::lock_guard<std::mutex> const lk(pool_mutex);
            if (pool_jobs.empty())
                return;
            jobs.swap(pool_jobs);
            pool_cands = 0;
        }
        flush(std::move(jobs));
    };
    auto pool_pending = [&]() {
        std::lock_guard<std::mutex> const lk(pool_mutex);
        return !pool_jobs.empty();
    };

    auto worker = [&]() {
        for (;;) {
            size_t const idx = next.fetch_add(1);
            if (idx >= sq_list.size()) {
                pool_drain();
                return;
            }
            auto const [q, rho] = sq_list[idx];
            auto const t = S.make_task(q, rho);
            if (!t) {
                discard(idx, q, rho);
                continue;
            }
            sieve_result res;
            sieve_task(*t, [&](chunk_result && make) {
                sieve_result c = make();
                std::move(c.survivors.begin(), c.survivors.end(),
                          std::back_inserter(res.survivors));
                std::move(c.hits.begin(), c.hits.end(),
                          std::back_inserter(res.hits));
                std::move(c.cands.begin(), c.cands.end(),
                          std::back_inserter(res.cands));
            });
            double const t1 = wallclock();
            cofac_job job = S.pass1(idx, *t, res);
            if (gpu_cofac)
                pool_push(std::move(job));
            else
                finish_cpu(std::move(job));
            process_us += (uint64_t)(1e6 * (wallclock() - t1));
        }
    };

    /* GLAS_STREAM (on by default, 0 disables) decouples sieving from
     * survivor processing.  A few threads keep the GPU slots fed and
     * queue each chunk's survivors as the chunk completes; the worker
     * pool processes queued chunks, so a task's cofactorization
     * overlaps its own later chunks and the tail after the last chunk
     * shrinks to one chunk's work. */
    std::deque<chunk_work> queue;
    int busy = 0;
    bool sieving = true;

    /* In the stream path the -t thread count is the CPU budget of
     * the worker pool.  The siever threads that feed the GPU sleep
     * while the device works and only run between chunks, so they
     * come on top of it. */
    int const nsievers = gpu_slots() + 1;
    int const nworkers = P.nthreads;

    /* Adaptive placement: cofactorization runs on the host until the
     * chunk queue shows that the workers cannot keep up.  Above the
     * high mark completed tasks go to the device batch; below the low
     * mark they go to the host again.  Once sieving ends the GPU sits
     * idle, so the remaining backlog drains to the device
     * (GLAS_COFAC_DRAIN=0 disables).  GLAS_COFAC_QHIGH sets
     * the high mark in units of workers. */
    std::atomic<bool> device_route{false};
    size_t qhigh = 4 * (size_t)nworkers;
    if (auto const * v = getenv("GLAS_COFAC_QHIGH"))
        qhigh = strtoul(v, nullptr, 10) * (size_t)nworkers;
    size_t const qlow = std::min(qhigh / 2, (size_t)nworkers);
    bool const drain = env_flag("GLAS_COFAC_DRAIN", true);

    auto push = [&](chunk_work w) {
        {
            std::lock_guard<std::mutex> const lk(qm);
            queue.push_back(std::move(w));
        }
        qcv.notify_one();
    };

    /* Tasks prepared ahead of GPU demand.  A siever thread takes a
     * ready task the moment a slot frees, and tops the queue back up
     * from the chunk sink while the GPU sieves, so several tasks that
     * finish together all find a prepared successor.  The depth
     * follows the slot count. */
    struct prep_task {
        size_t idx;
        sq_task t;
    };
    std::mutex pm;
    std::deque<prep_task> prepared;
    size_t const prep_depth = (size_t)gpu_slots() + 1;

    auto prepare_one = [&]() -> bool {
        trace_scope const tr(TR_PREP);
        size_t const idx = next.fetch_add(1);
        if (idx >= sq_list.size())
            return false;
        auto const [q, rho] = sq_list[idx];
        auto const t = S.make_task(q, rho);
        if (!t) {
            discard(idx, q, rho);
            return true;
        }
        std::lock_guard<std::mutex> const lk(pm);
        prepared.push_back({idx, *t});
        return true;
    };
    auto take_prepared = [&]() -> std::optional<prep_task> {
        for (;;) {
            {
                std::lock_guard<std::mutex> const lk(pm);
                if (!prepared.empty()) {
                    prep_task p = std::move(prepared.front());
                    prepared.pop_front();
                    return p;
                }
            }
            if (!prepare_one())
                return std::nullopt;
        }
    };
    /* One task per call bounds the pause in the caller's chunk loop. */
    auto refill_prepared = [&]() {
        {
            std::lock_guard<std::mutex> const lk(pm);
            if (prepared.size() >= prep_depth)
                return;
        }
        prepare_one();
    };

    auto stream_siever = [&]() {
        for (;;) {
            auto const p = take_prepared();
            if (!p)
                return;
            auto ts = std::make_shared<task_state>();
            ts->idx = p->idx;
            ts->t = p->t;
            size_t nchunks = 0;
            sieve_task(p->t, [&](chunk_result && make) {
                trace_scope const tr(TR_SINK);
                {
                    std::lock_guard<std::mutex> const lk(ts->m);
                    ts->parts.emplace_back();
                }
                ts->pending++;
                push({ts, nchunks++, std::move(make)});
                refill_prepared();
            });
            if (--ts->pending == 0)
                push({ts, SIZE_MAX, {}});
        }
    };

    /* The workers run candidate assembly and host cofactorization at
     * a higher nice value, so a siever thread that wakes to feed the
     * GPU preempts them at once (GLAS_WORKER_NICE, 0 disables). */
    int worker_nice = 5;
    if (auto const * v = getenv("GLAS_WORKER_NICE"))
        worker_nice = atoi(v);

    auto stream_worker = [&]() {
        trace_worker = true;
        if (worker_nice)
            setpriority(PRIO_PROCESS, 0, worker_nice);
        for (;;) {
            /* Finish items first, then chunks.  With both queues empty
             * after sieving, the last pool goes to the device (w.ts
             * stays null) and its finish items come back through the
             * finish queue; when the pool is empty too, the run is
             * over. */
            std::optional<finish_item> fin;
            chunk_work w;
            {
                std::unique_lock<std::mutex> lk(qm);
                qcv.wait(lk, [&] {
                    return !finish_queue.empty() || !queue.empty() ||
                           (!sieving && busy == 0);
                });
                if (!finish_queue.empty()) {
                    fin = std::move(finish_queue.front());
                    finish_queue.pop_front();
                } else if (queue.empty()) {
                    if (!pool_pending())
                        break;
                } else {
                    w = std::move(queue.front());
                    queue.pop_front();
                    if (queue.size() >= qhigh || (drain && !sieving))
                        device_route = true;
                    else if (queue.size() <= qlow)
                        device_route = false;
                }
                busy++;
            }
            double const t0 = wallclock();
            if (fin) {
                finish_device(std::move(*fin));
            } else if (!w.ts) {
                pool_drain();
            } else if (w.chunk == SIZE_MAX) {
                task_state & ts = *w.ts;
                cofac_job job;
                job.idx = ts.idx;
                job.text = task_header(ts.t, ts.nsurv);
                for (std::vector<candidate> & part : ts.parts)
                    std::move(part.begin(), part.end(),
                              std::back_inserter(job.cands));
                if (gpu_cofac || (adaptive && device_route)) {
                    pool_push(std::move(job));
                } else {
                    pool_drain();
                    finish_cpu(std::move(job));
                }
            } else {
                task_state & ts = *w.ts;
                sieve_result const res = w.make();
                ts.nsurv += res.nsurv();
                std::vector<candidate> cands = S.pass1_cands(ts.t, res);
                {
                    std::lock_guard<std::mutex> const lk(ts.m);
                    ts.parts[w.chunk] = std::move(cands);
                }
                if (--ts.pending == 0)
                    push({w.ts, SIZE_MAX, {}});
            }
            process_us += (uint64_t)(1e6 * (wallclock() - t0));
            {
                std::lock_guard<std::mutex> const lk(qm);
                busy--;
            }
            qcv.notify_all();
        }
    };

    /* GLAS_PIN=1 pins each siever thread to its own CPU from the
     * process affinity mask.  The workers run on every CPU of the
     * mask at the higher nice value, so a siever that wakes to feed
     * the GPU preempts the worker on its CPU.  Each process needs a
     * disjoint CPU set from its launcher (for example taskset);
     * without one, concurrent processes pin to the same CPUs. */
    std::vector<int> siever_cpus;
    if (env_flag("GLAS_PIN", false)) {
        cpu_set_t mask;
        if (sched_getaffinity(0, sizeof mask, &mask) == 0) {
            std::vector<int> cpus;
            for (int c = 0; c < CPU_SETSIZE; c++)
                if (CPU_ISSET(c, &mask))
                    cpus.push_back(c);
            if ((int)cpus.size() > nsievers)
                siever_cpus.assign(cpus.begin(), cpus.begin() + nsievers);
        }
    }

    std::vector<std::thread> threads;
    if (stream) {
        if (getenv("GLAS_GPU_STATS"))
            fprintf(stderr, "# Threads: %d sievers, %d workers%s\n",
                    nsievers, nworkers,
                    siever_cpus.empty() ? "" : ", pinned");
        std::vector<std::thread> sievers;
        for (int k = 0; k < nsievers; k++)
            sievers.emplace_back([&, k] {
                if (!siever_cpus.empty()) {
                    cpu_set_t one;
                    CPU_ZERO(&one);
                    CPU_SET(siever_cpus[k], &one);
                    pthread_setaffinity_np(pthread_self(), sizeof one,
                                           &one);
                }
                stream_siever();
            });
        for (int k = 0; k < nworkers; k++)
            threads.emplace_back(stream_worker);
        for (auto & th : sievers)
            th.join();
        {
            std::lock_guard<std::mutex> const lk(qm);
            sieving = false;
        }
        qcv.notify_all();
    } else {
        for (int k = 0; k < P.nthreads; k++)
            threads.emplace_back(worker);
    }
    for (auto & th : threads)
        th.join();

    double const elapsed = wallclock() - t_start;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double const cpu = (double)ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec +
                       (double)ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec;
    unsigned long const reports = total_reports.load();
    char total_line[160];
    snprintf(total_line, sizeof total_line,
             "# Total %lu reports [%0.6fs/r, %0.2fr/sq] in %0.2f elapsed s "
             "[%0.1f%% CPU]\n",
             reports, reports ? elapsed / (double)reports : 0.0,
             sq_list.empty() ? 0.0 : (double)reports / (double)sq_list.size(),
             elapsed, elapsed > 0 ? 100.0 * cpu / elapsed : 0.0);
    fputs(total_line, out);
    if (P.stats_stderr)
        fputs(total_line, stderr);
    fprintf(out, "# Thread seconds: %0.2f sieve, %0.2f process\n",
            1e-6 * (double)sieve_us.load(), 1e-6 * (double)process_us.load());
    /* startup = process start to the first special-q sieve start; steady =
     * that to the last special-q output; tail = the rest.  A run that
     * sieved nothing is all startup. */
    double const t_end = t_start + elapsed;
    double const t_first = t_first_sieve.load() ? t_first_sieve.load() : t_end;
    double const t_last = std::max(t_last_done, t_first);
    fprintf(out, "# Phases: %0.2f s startup, %0.2f s steady, %0.2f s tail, %zu special-q\n",
            t_first - t_start, t_last - t_first, t_end - t_last, sq_list.size());
    fprintf(out, "# Cofactor candidates: %" PRIu64 " host, %" PRIu64 " device\n",
            host_cands.load(), device_cands.load());
    if (getenv("GLAS_GPU_STATS"))
        fprintf(stderr,
                "# Process seconds: %0.2f norm, %0.2f divide, %0.2f tiny,"
                " %0.2f facul (%" PRIu64 " survivors, %" PRIu64 " coprime,"
                " %" PRIu64 " facul calls)\n",
                1e-9 * (double)proc_ns[PH_NORM].load(),
                1e-9 * (double)proc_ns[PH_DIVIDE].load(),
                1e-9 * (double)proc_ns[PH_TINY].load(),
                1e-9 * (double)proc_ns[PH_FACUL].load(),
                proc_n[PN_SURVIVOR].load(), proc_n[PN_COPRIME].load(),
                proc_n[PN_FACUL].load());
    if (gzip.f)
        return gzip.close() != 0;
    if (out != stdout)
        fclose(out);
    return 0;
}
