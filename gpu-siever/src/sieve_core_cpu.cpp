/* CPU reference sieve core.  Mirrors the las algorithms:
 *  - root transform to the q-lattice (las-fbroot-qlattice.hpp)
 *  - row stepping for small and projective entries (las-smallsieve.cpp)
 *  - Franke-Kleinjung enumeration for large primes (las-plattice.hpp)
 */

#include "sieve_core.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace {

uint64_t invmod_u64(uint64_t a, uint64_t m)
{
    /* extended gcd; returns 0 when gcd(a, m) != 1 */
    int64_t t = 0, newt = 1;
    uint64_t r = m, newr = a % m;
    while (newr != 0) {
        uint64_t const qq = r / newr;
        int64_t const t2 = t - (int64_t)qq * newt;
        t = newt; newt = t2;
        uint64_t const r2 = r - qq * newr;
        r = newr; newr = r2;
    }
    if (r != 1)
        return 0;
    return t < 0 ? (uint64_t)(t + (int64_t)m) : (uint64_t)t;
}

uint32_t invmod_po2(uint32_t v)
{
    /* inverse of odd v mod 2^32, by Newton iteration */
    uint32_t x = v;
    for (int k = 0; k < 4; k++)
        x = x * (2 - v * x);
    return x;
}

uint64_t mod_i128(__int128 v, uint64_t m)
{
    __int128 const r = v % (__int128)m;
    return r < 0 ? (uint64_t)(r + (__int128)m) : (uint64_t)r;
}

uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m)
{
    return (uint64_t)(((unsigned __int128)a * b) % m);
}

/* Root transform for powers of 2 (las fb_root_in_qlattice_po2). */
uint32_t transform_root_po2(uint32_t q, uint32_t r_enc, sq_task const & t)
{
    bool const proj = r_enc >= q;
    uint32_t const R = proj ? r_enc - q : r_enc;
    uint32_t u, v;
    if (!proj) {
        u = (uint32_t)((int64_t)R * (t.b1 % (int64_t)q) - t.a1);
        v = (uint32_t)(t.a0 - (int64_t)R * (t.b0 % (int64_t)q));
    } else {
        u = (uint32_t)(t.b1 - (int64_t)R * (t.a1 % (int64_t)q));
        v = (uint32_t)((int64_t)R * (t.a0 % (int64_t)q) - t.b0);
    }
    if (v & 1)
        return (u * invmod_po2(v)) & (q - 1);
    if (!(u & 1))
        return UINT32_MAX;  /* degenerate; skip the entry */
    return q + ((invmod_po2(u) * v) & (q - 1));
}

/* Root transform to the q-lattice.  Returns the root in (i,j)
 * coordinates, with q added when the transformed root is projective, or
 * UINT32_MAX when the entry must be skipped. */
uint32_t transform_root(uint32_t q, uint32_t r_enc, sq_task const & t)
{
    if ((q & 1) == 0)
        return transform_root_po2(q, r_enc, t);
    bool const proj = r_enc >= q;
    int64_t const R = proj ? r_enc - q : r_enc;
    __int128 num, den;
    if (!proj) {
        num = (__int128)R * t.b1 - t.a1;
        den = (__int128)t.a0 - (__int128)R * t.b0;
    } else {
        num = (__int128)t.b1 - (__int128)R * t.a1;
        den = (__int128)R * t.a0 - t.b0;
    }
    uint64_t const u = mod_i128(num, q);
    uint64_t const v = mod_i128(den, q);
    uint64_t const vinv = invmod_u64(v, q);
    if (vinv)
        return (uint32_t)mulmod_u64(u, vinv, q);
    uint64_t const uinv = invmod_u64(u, q);
    if (!uinv)
        return UINT32_MAX;  /* degenerate; skip the entry */
    return q + (uint32_t)mulmod_u64(v, uinv, q);
}

/* Franke-Kleinjung p-lattice basis, reduced exactly as las does
 * (las-plattice.hpp, las-reduce-plattice-simplistic.hpp). */
struct plattice {
    uint32_t mi0 = 0, j0 = 0, i1 = 0, j1 = 0;  /* vectors (-mi0,j0), (i1,j1) */

    plattice(uint32_t q, uint32_t r, uint32_t I)
    {
        mi0 = q; j0 = 0; i1 = r; j1 = 1;
        if (i1 == 0) {
            reduce_with_vertical_vector(I);
            return;
        }
        for (;;) {
            if (i1 < I) {
                if (i1 == 0) {
                    j0 = j1 - j0;
                    reduce_with_vertical_vector(I);
                    return;
                }
                uint32_t const a = (mi0 + i1 - I) / i1;
                mi0 -= a * i1;
                j0 += a * j1;
                return;
            }
            { uint32_t const k = mi0 / i1; mi0 -= k * i1; j0 += k * j1; }
            if (mi0 < I) {
                if (mi0 == 0) {
                    mi0 = i1;
                    i1 = j0; j0 = j1; j1 = i1;
                    i1 = 0;
                    reduce_with_vertical_vector(I);
                    return;
                }
                uint32_t const a = (mi0 + i1 - I) / mi0;
                i1 -= a * mi0;
                j1 += a * j0;
                return;
            }
            { uint32_t const k = i1 / mi0; i1 -= k * mi0; j1 += k * j0; }
        }
    }

    void reduce_with_vertical_vector(uint32_t I)
    {
        uint32_t const a = (I - 1) % mi0;
        uint32_t mi2 = (I - 1) - a - i1;
        uint32_t const j2 = j1;
        if (mi0 + mi2 < I)
            mi2 += mi0;
        i1 = mi0;
        j1 = j0;
        mi0 = mi2;
        j0 = j2;
    }
};

struct region {
    int logI;
    uint32_t I, J;
    uint64_t area;
    region(sq_task const & t)
        : logI(t.logI), I(1u << t.logI), J(t.J),
          area((uint64_t)t.J << t.logI) {}
};

/* Walk the FK lattice for an affine root of a large entry (q >= I).
 * Calls f(x) for every hit that can be coprime. */
template <typename F>
void fk_walk(region const & g, uint32_t q, uint32_t root, F && f)
{
    plattice const L(q, root, g.I);
    if (L.i1 >> g.logI)
        return;  /* vertical line i=0: no coprime hit except (0,1) */
    uint64_t const inc_warp = ((uint64_t)L.j0 << g.logI) - L.mi0;
    uint64_t const inc_step = ((uint64_t)L.j1 << g.logI) + L.i1;
    uint32_t const bound_warp = g.I - L.i1;
    uint32_t const bound_step = L.mi0;
    uint64_t const even_mask = ((uint64_t)1 << g.logI) | 1;
    uint32_t const maskI = g.I - 1;
    uint64_t x = g.I / 2;
    while (x < g.area) {
        if (x & even_mask)
            f(x);
        uint32_t const i = x & maskI;
        uint64_t inc = 0;
        if (i >= bound_warp)
            inc += inc_warp;
        if (i < bound_step)
            inc += inc_step;
        x += inc;
        if (inc == 0)
            return;  /* cannot happen after a valid reduction */
    }
}

/* Row stepping for an affine root of a small entry (q < I).
 * Calls f(x) for every hit with j >= 1. */
template <typename F>
void row_walk_affine(region const & g, uint32_t q, uint32_t root, F && f)
{
    uint64_t m = ((uint64_t)root + g.I / 2) % q;  /* (root*j + I/2) mod q, j=1 */
    for (uint32_t j = 1; j < g.J; j++) {
        uint64_t const row = (uint64_t)j << g.logI;
        for (uint64_t off = m; off < g.I; off += q)
            f(row + off);
        m += root;
        if (m >= q)
            m -= q;
    }
}

/* Row stepping for a projective root U in the q-lattice: hits satisfy
 * i*U == j (mod q).  With g = gcd(U, q), qq = q/g: rows j = n*g, and
 * i == n * (U/g)^-1 (mod qq).  Calls f(x) for every hit. */
template <typename F>
void row_walk_proj(region const & g, uint32_t q, uint32_t U, F && f)
{
    uint32_t const gg = U ? (uint32_t)std::gcd(U, q) : q;
    uint32_t const qq = q / gg;
    uint64_t const t0 = U ? invmod_u64(U / gg, qq) : 0;
    if (U && t0 == 0 && qq > 1)
        return;  /* cannot happen: gcd(U/gg, qq) == 1 */
    for (uint64_t j = gg, n = 1; j < g.J; j += gg, n++) {
        uint64_t const row = j << g.logI;
        uint64_t const start = qq > 1 ? (t0 * n + g.I / 2) % qq : 0;
        for (uint64_t off = start; off < g.I; off += qq)
            f(row + off);
    }
}

/* Dispatch one factor-base entry to the proper walk. */
template <typename F>
void entry_walk(region const & g, uint32_t q, uint32_t root_ij, F && f)
{
    if (root_ij >= q)
        row_walk_proj(g, q, root_ij - q, f);
    else if (q < g.I)
        row_walk_affine(g, q, root_ij, f);
    else
        fk_walk(g, q, root_ij, f);
}

int16_t log2_units(double v)
{
    double const bits = std::log2(std::fabs(v) + 1.0);
    return (int16_t)std::lround(bits * LOG_SCALE);
}

void norm_init(std::vector<int16_t> & rem, sq_task const & t, int side,
               region const & g)
{
    int const d = t.degree[side];
    double const * c = t.fijd[side];
    rem.assign(g.area, INT16_MAX);
    std::vector<double> cj(d + 1);
    for (uint32_t j = 1; j < g.J; j++) {
        double jp = 1.0;
        for (int k = d; k >= 0; k--) {
            cj[k] = c[k] * jp;
            jp *= (double)j;
        }
        int16_t * row = rem.data() + ((uint64_t)j << g.logI);
        for (uint32_t ii = 0; ii < g.I; ii++) {
            double const x = (double)((int32_t)ii - (int32_t)(g.I / 2));
            double v = cj[d];
            for (int k = d - 1; k >= 0; k--)
                v = v * x + cj[k];
            row[ii] = log2_units(v);
        }
    }
}

bool skip_entry(sq_task const & t, int side, fb_entry const & e)
{
    return side == t.sq_side && e.q == t.q && e.r == t.rho;
}

/* which tiny primes divide the side norm F(a, b): evaluate F mod p */
uint64_t tiny_mask(tiny_table const & tt, int side, int degree, int64_t a,
                   int64_t b)
{
    uint64_t m = 0;
    for (size_t k = 0; k < tt.primes.size(); k++) {
        uint32_t const p = tt.primes[k];
        uint32_t const * c = tt.cmod[side].data() + k * (GLAS_MAX_DEGREE + 1);
        uint64_t const am = (uint64_t)(a % p + p) % p;
        uint64_t const bm = (uint64_t)(b % p + p) % p;
        uint64_t r = c[degree], bp = 1;
        for (int d = degree - 1; d >= 0; d--) {
            bp = bp * bm % p;
            r = (r * am + c[d] * bp) % p;
        }
        m |= (uint64_t)(r == 0) << k;
    }
    return m;
}

}  // namespace

/* LSD radix sort by x (x < 2^48: three 16-bit passes) */
void sort_hits(std::vector<resieve_hit> & hits)
{
    std::vector<resieve_hit> tmp(hits.size());
    resieve_hit * a = hits.data();
    resieve_hit * b = tmp.data();
    for (int shift = 0; shift < 48; shift += 16) {
        std::vector<uint32_t> pos(1 << 16, 0);
        for (size_t k = 0; k < hits.size(); k++)
            pos[(a[k].x >> shift) & 0xffff]++;
        uint32_t sum = 0;
        for (uint32_t & c : pos) {
            uint32_t const n = c;
            c = sum;
            sum += n;
        }
        for (size_t k = 0; k < hits.size(); k++)
            b[pos[(a[k].x >> shift) & 0xffff]++] = a[k];
        std::swap(a, b);
    }
    if (a != hits.data())
        hits.swap(tmp);
}

sieve_result sieve_one_cpu(sq_task const & task,
                           std::vector<fb_entry> const * fb,
                           tiny_table const & tt)
{
    region const g(task);
    sieve_result res;

    std::vector<int16_t> rem[2];
    for (int side = 0; side < 2; side++) {
        norm_init(rem[side], task, side, g);
        int16_t * S = rem[side].data();
        for (fb_entry const & e : fb[side]) {
            if (skip_entry(task, side, e))
                continue;
            uint32_t const root = transform_root(e.q, e.r, task);
            if (root == UINT32_MAX)
                continue;
            uint8_t const w = e.weight;
            entry_walk(g, e.q, root, [S, w](uint32_t x) { S[x] -= w; });
        }
    }

    /* survivor scan: both sides below their bound, i and j not both
     * even (they cannot be coprime) */
    int32_t const b0 = task.bound[0] + SURVIVOR_SLACK;
    int32_t const b1 = task.bound[1] + SURVIVOR_SLACK;
    uint64_t const even_mask = ((uint64_t)1 << g.logI) | 1;
    std::vector<uint8_t> is_survivor(g.area, 0);
    for (uint64_t x = g.I; x < g.area; x++) {
        if (rem[0][x] <= b0 && rem[1][x] <= b1 && (x & even_mask)) {
            int64_t const i = (int64_t)(x & (g.I - 1)) - g.I / 2;
            int64_t const j = (int64_t)(x >> g.logI);
            int64_t const a = i * task.a0 + j * task.a1;
            int64_t const b = i * task.b0 + j * task.b1;
            res.survivors.push_back(
                {x, {tiny_mask(tt, 0, task.degree[0], a, b),
                     tiny_mask(tt, 1, task.degree[1], a, b)}});
            is_survivor[x] = 1;
        }
    }
    if (res.survivors.empty())
        return res;

    /* resieve: walk the prime entries again and report hits on
     * survivors */
    for (uint8_t side = 0; side < 2; side++) {
        for (fb_entry const & e : fb[side]) {
            if (!e.resieve || skip_entry(task, side, e))
                continue;
            uint32_t const root = transform_root(e.q, e.r, task);
            if (root == UINT32_MAX)
                continue;
            uint32_t const q = e.q;
            entry_walk(g, q, root,
                       [&res, &is_survivor, q, side](uint64_t x) {
                           if (is_survivor[x])
                               res.hits.push_back({x, q, side});
                       });
        }
    }
    sort_hits(res.hits);
    return res;
}
