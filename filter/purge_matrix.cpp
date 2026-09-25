#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "macros.h"
#include "memusage.h"
#include "misc.h"
#include "purge_matrix.hpp"
#include "subdivision.hpp"
#include "timing.h"
#include "typedefs.h"
#include "utils_cxx.hpp"

static double const wct_program_start = wct_seconds();

double purge_matrix::wct_elapsed()
{
    return wct_seconds() - wct_program_start;
}

purge_matrix::column_weight_counter::~column_weight_counter()
{
    for (auto & slot: chunks)
        delete[] slot.load();
}

void purge_matrix::column_weight_counter::increment(size_t h)
{
    ASSERT_ALWAYS(h < max_chunks * chunk_size);
    auto & slot = chunks[h >> chunk_bits];
    weight_t * chunk = slot.load(std::memory_order_acquire);
    if (!chunk) {
        /* The first thread to reach this range of columns installs
         * the chunk; the others take it. */
        auto * fresh = new weight_t[chunk_size]();
        if (slot.compare_exchange_strong(chunk, fresh, std::memory_order_acq_rel))
            chunk = fresh;
        else
            delete[] fresh;
    }
    std::atomic_ref const w(chunk[h % chunk_size]);
    weight_t old = w.load(std::memory_order_relaxed);
    while (old < OVERWEIGHT &&
           !w.compare_exchange_weak(old, old + 1, std::memory_order_relaxed))
        ;
}

void purge_matrix::set_column_range(size_t col0, size_t col1)
{
    col_min_index = col0;
    col_max_index = col1;
    index_bytes = 1;
    for (; index_bytes < sizeof(index_t) && col1 >> (8 * index_bytes);
         index_bytes++)
        ;
    end_of_row = ~index_t(0) >> (8 * (sizeof(index_t) - index_bytes));
    ASSERT_ALWAYS(col1 <= end_of_row);
}

void purge_matrix::append(row_builder & b)
{
    rows.insert(rows.end(), b.rows.begin(), b.rows.end());
    remaining_rows += b.rows.size();
    take_blocks(b);
    b.rows = {};
}

void purge_matrix::take_column_weights(column_weight_counter const & weights)
{
    size_t n = weights.chunks.size();
    for (; n && !weights.chunks[n - 1].load(); n--)
        ;
    column_weights.assign(n * weights.chunk_size, 0);
    for (size_t j = 0; j < n; j++) {
        auto const * chunk = weights.chunks[j].load();
        if (chunk)
            std::copy_n(chunk, weights.chunk_size,
                        column_weights.begin() + j * weights.chunk_size);
    }
    while (!column_weights.empty() && column_weights.back() == 0)
        column_weights.pop_back();
    remaining_columns = column_weights.size()
        - std::count(column_weights.begin(), column_weights.end(), 0);
}

void purge_matrix::print_memory(std::string const & label) const
{
    fmt::print("# MEMORY: {}: rss {} (peak vsize {});"
               " row pointers {}, row data {}, column weights {},"
               " row xors {}\n",
               label, size_disp(Memusage2() << 10),
               size_disp(PeakMemusage() << 10),
               size_disp(rows.capacity() * sizeof(uint8_t *)),
               size_disp(get_allocated_bytes()),
               size_disp(column_weights.capacity() * sizeof(weight_t)),
               size_disp(row_xors.capacity() * sizeof(index_t)));
    fflush(stdout);
}

void purge_matrix::print_stats_generic(FILE * out, std::vector<size_t> const & w,
                        char const * name, char const * unit, int verbose)
{
    size_t av = 0, std = 0, nzero = 0;
    size_t min = std::numeric_limits<size_t>::max();
    size_t max = 0;
    for (auto c: w) {
        if (c) {
            nzero++;
            min = std::min(min, c);
            max = std::max(max, c);
            av += c;
            std += c * c;
        }
    }
    double const av_f = double_ratio(av, nzero + !nzero);
    double const std_f = sqrt(double_ratio(std, nzero + !nzero) - av_f * av_f);

    fmt::print(out, "# STATS on {}: #{} = {}\n", name, name, nzero);
    fmt::print(out, "# STATS on {}: min {} = {}\n", name, unit, min);
    fmt::print(out, "# STATS on {}: max {} = {}\n", name, unit, max);
    fmt::print(out, "# STATS on {}: av {} = {:.2f}\n", name, unit, av_f);
    fmt::print(out, "# STATS on {}: std {} = {:.2f}\n", name, unit, std_f);

    if (verbose > 1) {
        std::vector<size_t> dist(max - min + 1, 0);
        for (auto c: w)
            if (c)
                dist[c - min]++;

        for (size_t i = min; i <= max; i++)
            if (dist[i - min] > 0)
                fmt::print(out, "# STATS on {}: #{} of {} {} : {}\n", name,
                           name, unit, i, dist[i - min]);
    }
    fflush(out);
}

void purge_matrix::print_stats_column_weights(FILE * out, int verbose) const
{
    std::vector<size_t> w(column_weights.size(), 0);

    for (size_t i = 0; i < rows.size(); i++) {
        if (is_active(i))
            for (auto h: columns(i))
                w[h]++;
    }

    print_stats_generic(out, w, "cols", "weight", verbose);
}

void purge_matrix::print_stats_row_weights(FILE * out, int verbose) const
{
    std::vector<size_t> w(rows.size(), 0);

    for (size_t i = 0; i < rows.size(); i++) {
        if (is_active(i))
            for ([[maybe_unused]] auto h: columns(i))
                w[i]++;
    }
    print_stats_generic(out, w, "rows", "weight", verbose);
}

template <typename F>
static void apply_fragment_multithread(size_t nrows, size_t nthreads, F const & f)
{
    std::vector<std::thread> T;
    T.reserve(nthreads);
    auto const S = subdivision(nrows, nthreads);
    for (size_t i = 0; i < nthreads; i++) {
        auto [i0, i1] = S.nth_block(i);
        T.emplace_back(f, i0, i1);
    }
    for (auto & t: T)
        t.join();
}

void purge_matrix::compute_row_xors(size_t nthreads)
{
    ASSERT_ALWAYS(rows.size() < std::numeric_limits<index_t>::max());
    row_xors.assign(column_weights.size(), 0);
    auto fragment = [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; i++) {
            if (!is_active(i))
                continue;
            for (auto h: columns(i)) {
                std::atomic_ref(row_xors[h])
                    .fetch_xor(index_t(i + 1), std::memory_order_relaxed);
            }
        }
    };
    apply_fragment_multithread(rows.size(), nthreads, fragment);
}

bool purge_matrix::delete_row(size_t i, std::vector<index_t> & weight_one)
{
    auto const * p = std::atomic_ref(rows[i]).exchange(nullptr);
    if (!p)
        return false;
    size_t killed = 0;
    for (auto h: columns(p)) {
        auto w = std::atomic_ref(column_weights[h]);
        ASSERT(w.load());
        /* Decrease only if not equal to the maximum value */
        /* If weight becomes 0, we just remove a column */
        if (w.load() == OVERWEIGHT)
            continue;
        /* The xor comes first, so that the thread which sees the weight
         * fall to one sees the xor of the other deleted rows as well. */
        std::atomic_ref(row_xors[h])
            .fetch_xor(index_t(i + 1), std::memory_order_relaxed);
        weight_t const nw = --w;
        if (nw == 0)
            killed++;
        if (nw == 1)
            weight_one.push_back(h);
    }
    if (killed)
        std::atomic_ref(remaining_columns) -= killed;
    std::atomic_ref(remaining_rows)--;
    return true;
}

auto purge_matrix::columns_of_weight(weight_t w, size_t nthreads) const
    -> column_bitset
{
    column_bitset res((column_weights.size() + 63) / 64, 0);
    auto fragment = [&](size_t j0, size_t j1) {
        for (size_t h = 64 * j0; h < std::min(64 * j1, column_weights.size()); h++)
            if (column_weights[h] == w)
                res[h / 64] |= uint64_t(1) << (h % 64);
    };
    apply_fragment_multithread(res.size(), nthreads, fragment);
    return res;
}

void purge_matrix::add_singleton_columns(std::vector<index_t> const & weight_one)
{
    std::lock_guard const lock(singleton_columns_mutex);
    singleton_columns.insert(singleton_columns.end(),
            weight_one.begin(), weight_one.end());
}

void purge_matrix::delete_rows_with_weight_one_column(size_t nthreads)
{
    auto const weight_one_columns = columns_of_weight(1, nthreads);
    auto fragment = [&](size_t i0, size_t i1) {
        std::vector<index_t> weight_one;
        for (size_t i = i0; i < i1; i++) {
            if (!is_active(i))
                continue;
            for (auto h: columns(i)) {
                if (test_bit(weight_one_columns, h)) {
                    ASSERT_ALWAYS(delete_row(i, weight_one));
                    break;
                }
            }
        }
        add_singleton_columns(weight_one);
    };
    apply_fragment_multithread(rows.size(), nthreads, fragment);
}

/* Delete the last row of each column in singleton_columns, and the
 * rows that these deletions make singletons in turn. A row that has two
 * columns of weight one is the last row of both: the thread that
 * deletes it first wins.
 */
void purge_matrix::delete_singletons(size_t nthreads)
{
    auto const columns = std::move(singleton_columns);
    singleton_columns.clear();
    auto fragment = [&](size_t k0, size_t k1) {
        std::vector<index_t> todo(columns.begin() + k0, columns.begin() + k1);
        for (size_t k = 0; k < todo.size(); k++) {
            index_t const x = std::atomic_ref(row_xors[todo[k]])
                .load(std::memory_order_relaxed);
            /* zero: the last row is deleted already */
            if (x)
                delete_row(x - 1, todo);
        }
    };
    apply_fragment_multithread(columns.size(), nthreads, fragment);
}

/* Perform a complete singleton removal step.
 * Return the excess at the end *
 */
int64_t purge_matrix::singleton_removal(size_t nthreads, int verbose)
{
    if (verbose >= 0) {
        fmt::print("Sing. rem.: begin with:"
                   " nrows={} ncols={} excess={} at {:2.2f} (wct {:2.2f})\n",
                   remaining_rows, remaining_columns, excess(), seconds(),
                   wct_elapsed());
        fflush(stdout);
    }

    if (row_xors.empty()) {
        compute_row_xors(nthreads);
        if (verbose >= 0)
            fmt::print("# MEMORY: row xors: {}\n",
                       size_disp(row_xors.size() * sizeof(index_t)));
        delete_rows_with_weight_one_column(nthreads);
    }
    delete_singletons(nthreads);

    if (verbose >= 0)
        fmt::print("Sing. rem.: No more singletons:"
                   " nrows={} ncols={} excess={} finished at {:2.2f} (wct {:2.2f})\n",
                   remaining_rows, remaining_columns, excess(), seconds(),
                   wct_elapsed());

    return excess();
}
