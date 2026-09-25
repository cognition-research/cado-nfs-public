/* check_matrix: an independent check of the matrix files written by
   merge or replay. With -matrix, checks that the matrix (dense and
   sparse parts together, and their row and column weight files) is the
   mod-2 expansion, through the index file, of the purged relations, up
   to the column renumbering (which keeps the left kernel). With -ker,
   checks that every dependency of the kernel file written by
   characters expands through the index to a set of purged relations
   whose ideals all cancel mod 2, so a wrong dependency is caught
   before the square root.

   Each row and each column gets a 128-bit hash, and the check compares
   XOR aggregates of these hashes, so it reads each file once and a
   mismatch escapes detection with probability about 2^-128. It shares
   no code with the merge and replay row construction. */

#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "fmt/base.h"

#include "fix-endianness.h"
#include "gzip.h"
#include "macros.h"
#include "misc.h"
#include "params.hpp"
#include "purgedfile.h"

struct sig128 {
    uint64_t lo = 0, hi = 0;
    void operator^=(sig128 const & o) { lo ^= o.lo; hi ^= o.hi; }
    bool operator==(sig128 const & o) const { return lo == o.lo && hi == o.hi; }
    bool operator<(sig128 const & o) const { return lo != o.lo ? lo < o.lo : hi < o.hi; }
    bool zero() const { return !(lo | hi); }
};

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

uint64_t seed;

/* the 128-bit hash of row or column x; the domain separates the two */
static sig128 hash128(uint64_t domain, uint64_t x)
{
    uint64_t const a = mix64(seed ^ mix64(domain + x));
    uint64_t const b = mix64(~seed ^ mix64(domain ^ mix64(x)));
    return { a, b };
}

static constexpr uint64_t ROWS = 1;
static constexpr uint64_t COLUMNS = 2;

static FILE * open_or_die(std::string const & name, const char * mode)
{
    FILE * f = fopen_maybe_compressed(name.c_str(), mode);
    if (!f) {
        fmt::print(stderr, "cannot open {}\n", name);
        exit(EXIT_FAILURE);
    }
    return f;
}

/* "<n> <j> <j> ..." with n in decimal and the row indices j in hex */
static void parse_index_row(char const * line, std::vector<uint64_t> & row)
{
    char * q;
    row.resize(strtoull(line, &q, 10));
    ASSERT_ALWAYS(!row.empty());
    for (auto & j : row) {
        char const * p = q;
        j = strtoull(p, &q, 16);
        ASSERT_ALWAYS(q != p);
    }
}

/* calls f(i, row) for each index row i */
template<typename F>
static uint64_t stream_index(std::string const & indexname, F f)
{
    FILE * fp = open_or_die(indexname, "r");
    char * line = nullptr;
    size_t alloc = 0;
    ASSERT_ALWAYS(getline(&line, &alloc, fp) >= 0);
    uint64_t const small_nrows = strtoull(line, nullptr, 10);
    std::vector<uint64_t> row;
    for (uint64_t i = 0; i < small_nrows; i++) {
        ASSERT_ALWAYS(getline(&line, &alloc, fp) >= 0);
        parse_index_row(line, row);
        f(i, row);
    }
    ASSERT_ALWAYS(getline(&line, &alloc, fp) < 0);
    free(line);
    fclose_maybe_compressed(fp, indexname.c_str());
    return small_nrows;
}

/* calls f(j, h) for each ideal occurrence h of purged relation j */
template<typename F>
static uint64_t stream_purged(std::string const & purgedname, F f)
{
    FILE * fp = open_or_die(purgedname, "r");
    char * line = nullptr;
    size_t alloc = 0;
    uint64_t j = 0;
    while (getline(&line, &alloc, fp) >= 0) {
        if (line[0] == '#')
            continue;
        char * p = strchr(line, ':');
        ASSERT_ALWAYS(p);
        for (p++; *p && *p != '\n'; p += *p == ',') {
            uint64_t const h = strtoull(p, &p, 16);
            f(j, h);
        }
        j++;
    }
    free(line);
    fclose_maybe_compressed(fp, purgedname.c_str());
    return j;
}

static std::vector<uint32_t> read_u32_file(std::string const & name)
{
    FILE * f = fopen(name.c_str(), "rb");
    ASSERT_ALWAYS(f);
    struct stat sbuf[1];
    ASSERT_ALWAYS(fstat(fileno(f), sbuf) == 0);
    ASSERT_ALWAYS(sbuf->st_size % 4 == 0);
    std::vector<uint32_t> v(sbuf->st_size / 4);
    ASSERT_ALWAYS(fread32_little(v.data(), v.size(), f) == v.size());
    fclose(f);
    return v;
}

/* one of the two matrix parts (dense or sparse), read row by row; each
 * row is a 32-bit weight followed by its column entries, which the
 * part offsets by j0 into the renumbered column space */
struct matrix_part {
    FILE * f;
    std::string name;
    std::vector<uint32_t> rw, cw;
    uint64_t j0;
    size_t nrw = 0;
    std::vector<uint32_t> row;

    matrix_part(std::string const & base, uint64_t j0)
        : f(fopen((base + ".bin").c_str(), "rb"))
        , name(base + ".bin")
        , rw(read_u32_file(base + ".rw.bin"))
        , cw(read_u32_file(base + ".cw.bin"))
        , j0(j0)
    {
        ASSERT_ALWAYS(f);
    }

    void next()
    {
        uint32_t w;
        ASSERT_ALWAYS(fread32_little(&w, 1, f) == 1);
        row.resize(w);
        ASSERT_ALWAYS(fread32_little(row.data(), w, f) == w);
    }

    void finish() const
    {
        char c;
        if (nrw != rw.size() || fread(&c, 1, 1, f) == 1) {
            fmt::print(stderr, "{} has data beyond the expected rows\n", name);
            exit(EXIT_FAILURE);
        }
    }
};

/* The matrix equals the expansion iff the two multisets of nonzero
 * column signatures agree, where the signature of a column is the XOR
 * of the hashes of the rows it appears in: this proves the matrix
 * columns and the expanded columns are the same up to a bijection,
 * which keeps the left kernel. */
static int check_matrix(std::string const & purgedname,
                        std::string const & indexname,
                        std::string const & matrixname)
{
    uint64_t nrows_purged, colbound;
    purgedfile_read_firstline(purgedname.c_str(), &nrows_purged, &colbound);

    /* S[j] = XOR of the hashes of the index rows that use purged row j */
    std::vector<sig128> S(nrows_purged);
    uint64_t const small_nrows = stream_index(indexname,
        [&](uint64_t i, std::vector<uint64_t> const & row) {
            for (uint64_t const j : row) {
                ASSERT_ALWAYS(j < nrows_purged);
                S[j] ^= hash128(ROWS, i);
            }
        });

    /* T[h] = signature of original column h in the expanded matrix;
     * an even number of occurrences of h cancels, as in the row XOR */
    std::vector<sig128> T(colbound);
    uint64_t const nread = stream_purged(purgedname,
        [&](uint64_t j, uint64_t h) {
            ASSERT_ALWAYS(h < colbound);
            T[h] ^= S[j];
        });
    ASSERT_ALWAYS(nread == nrows_purged);

    ASSERT_ALWAYS(matrixname.ends_with(".bin"));
    std::string const base = matrixname.substr(0, matrixname.size() - 4);
    std::string dbase = base;
    if (size_t const pos = dbase.rfind(".sparse"); pos != std::string::npos)
        dbase.erase(pos, 7);
    dbase += ".dense";

    std::vector<matrix_part> parts;
    if (struct stat sbuf[1]; stat((dbase + ".bin").c_str(), sbuf) == 0)
        parts.emplace_back(dbase, 0);
    uint64_t const skip = parts.empty() ? 0 : parts.front().cw.size();
    parts.emplace_back(base, skip);

    /* The row weight files hold one entry per row the producer saw as
     * non-empty (matrix_output.hpp). The fused merge output writes an
     * entry for every row; replay skips the rows with no entry in any
     * part. The file length tells the two apart. */
    for (auto const & part : parts)
        ASSERT_ALWAYS(part.rw.size() == parts.front().rw.size());
    bool const rw_all_rows = parts.front().rw.size() == small_nrows;

    /* U[c] = signature of renumbered column c in the matrix files */
    uint64_t const ncols = skip + parts.back().cw.size();
    std::vector<sig128> U(ncols);
    std::vector<uint32_t> weights(ncols);
    for (uint64_t i = 0; i < small_nrows; i++) {
        bool nonempty = false;
        for (auto & part : parts) {
            part.next();
            nonempty = nonempty || !part.row.empty();
        }
        for (auto & part : parts) {
            if (rw_all_rows || nonempty) {
                ASSERT_ALWAYS(part.nrw < part.rw.size());
                ASSERT_ALWAYS(part.rw[part.nrw++] == part.row.size());
            }
            for (uint32_t const c : part.row) {
                ASSERT_ALWAYS(c < part.cw.size());
                U[part.j0 + c] ^= hash128(ROWS, i);
                weights[part.j0 + c]++;
            }
        }
    }
    for (auto const & part : parts) {
        part.finish();
        for (size_t k = 0; k < part.cw.size(); k++)
            ASSERT_ALWAYS(part.cw[k] == weights[part.j0 + k]);
    }

    std::vector<sig128> t;
    for (auto const & s : T)
        if (!s.zero())
            t.push_back(s);
    std::sort(t.begin(), t.end());
    std::sort(U.begin(), U.end());
    if (t != U) {
        fmt::print(stderr,
                   "FAILED: {} is not the expansion of {} through {}"
                   " ({} matrix columns, {} expanded columns, seed {})\n",
                   matrixname, purgedname, indexname, ncols, t.size(), seed);
        return 1;
    }
    fmt::print("{} == {} * {} over GF(2): ok"
               " ({} rows, {} columns, {} in the dense part, seed {})\n",
               matrixname, indexname, purgedname,
               small_nrows, ncols, skip, seed);
    return 0;
}

/* one row of little-endian 64-bit words per index row */
static std::vector<uint64_t> read_kernel_file(std::string const & name)
{
    FILE * f = fopen(name.c_str(), "rb");
    ASSERT_ALWAYS(f);
    struct stat sbuf[1];
    ASSERT_ALWAYS(fstat(fileno(f), sbuf) == 0);
    ASSERT_ALWAYS(sbuf->st_size % 8 == 0);
    std::vector<uint32_t> halves(sbuf->st_size / 4);
    ASSERT_ALWAYS(fread32_little(halves.data(), halves.size(), f) ==
                  halves.size());
    fclose(f);
    std::vector<uint64_t> words(halves.size() / 2);
    for (size_t k = 0; k < words.size(); k++)
        words[k] = halves[2 * k] | (uint64_t) halves[2 * k + 1] << 32;
    return words;
}

static int check_kernel(std::string const & purgedname,
                        std::string const & indexname,
                        std::string const & kername)
{
    uint64_t nrows_purged, colbound;
    purgedfile_read_firstline(purgedname.c_str(), &nrows_purged, &colbound);

    /* R[j] = XOR of the column hashes of purged relation j */
    std::vector<sig128> R(nrows_purged);
    uint64_t const nread = stream_purged(purgedname,
        [&](uint64_t j, uint64_t h) { R[j] ^= hash128(COLUMNS, h); });
    ASSERT_ALWAYS(nread == nrows_purged);

    /* rowsig[i] = signature of the ideal parity vector of index row i */
    std::vector<sig128> rowsig;
    stream_index(indexname,
        [&](uint64_t i, std::vector<uint64_t> const & row) {
            ASSERT_ALWAYS(i == rowsig.size());
            sig128 s;
            for (uint64_t const j : row) {
                ASSERT_ALWAYS(j < nrows_purged);
                s ^= R[j];
            }
            rowsig.push_back(s);
        });

    std::vector<uint64_t> const kernel = read_kernel_file(kername);
    ASSERT_ALWAYS(!kernel.empty());
    ASSERT_ALWAYS(kernel.size() % rowsig.size() == 0);
    size_t const nwords = kernel.size() / rowsig.size();

    /* acc[d] = signature of the ideal parity vector of dependency d */
    std::vector<sig128> acc(64 * nwords);
    for (size_t i = 0; i < rowsig.size(); i++)
        for (size_t w = 0; w < nwords; w++)
            for (uint64_t b = kernel[i * nwords + w]; b; b &= b - 1)
                acc[64 * w + cado_ctz64(b)] ^= rowsig[i];

    int nfailed = 0;
    for (size_t d = 0; d < acc.size(); d++) {
        if (acc[d].zero())
            continue;
        fmt::print(stderr,
                   "FAILED: dependency {} of {} does not cancel the ideals"
                   " of {} (seed {})\n",
                   d, kername, purgedname, seed);
        nfailed++;
    }
    if (nfailed)
        return 1;
    fmt::print("all {} dependencies of {} cancel the ideals of {}: ok"
               " (seed {})\n",
               acc.size(), kername, purgedname, seed);
    return 0;
}

int main(int argc, char const * argv[])
{
    cxx_param_list pl;
    pl.declare_usage("purged", "purged relations file, as produced by purge");
    pl.declare_usage("index", "index file, as produced by merge or replay");
    pl.declare_usage("matrix", "binary sparse matrix file, as produced by"
                     " merge or replay: check matrix = index * purged");
    pl.declare_usage("ker", "kernel file, as produced by characters:"
                     " check that each dependency cancels all ideals");
    pl.declare_usage("seed", "seed of the row and column hashes"
                     " (default: random)");
    pl.process_command_line(argc, argv, true);

    std::string purgedname, indexname, matrixname, kername;
    pl.parse_mandatory("purged", purgedname);
    pl.parse_mandatory("index", indexname);
    pl.parse("matrix", matrixname);
    pl.parse("ker", kername);
    seed = std::random_device()();
    pl.parse("seed", seed);
    if (pl.warn_unused())
        pl.fail("unused parameters are given");
    if (matrixname.empty() && kername.empty())
        pl.fail("at least one of -matrix and -ker is needed");

    int err = 0;
    if (!matrixname.empty())
        err |= check_matrix(purgedname, indexname, matrixname);
    if (!kername.empty())
        err |= check_kernel(purgedname, indexname, kername);
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
