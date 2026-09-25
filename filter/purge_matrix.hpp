#ifndef CADO_PURGE_MATRIX_HPP
#define CADO_PURGE_MATRIX_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <atomic>
#include <compare>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "memalloc.hpp"
#include "typedefs.h"

/* XXX implementation for this structure is split in purge_matrix.cpp
 * (most stuff, including singleton removal) and clique_removal.cpp
 */
template <typename T>
concept container_with_h = requires {
    T().begin();
    T().end();
    T()[0].h;
};

template <typename F>
concept weight_function = requires { float(F()(3)); };

/* A row is the list of its column indices, each packed on index_bytes
 * bytes, little endian, followed by end_of_row. The rows live in
 * blocks with 8 spare bytes at the end, so that a row is read with
 * 8-byte loads.
 */
using purge_row_allocator =
    simple_minded_chunk_allocator<uint8_t, size_t(1) << 20, 8>;

struct purge_matrix : purge_row_allocator {
    static constexpr weight_t OVERWEIGHT = std::numeric_limits<weight_t>::max();
    /* only the columns in [col_min_index, col_max_index) are stored */
    size_t col_min_index = 0;
    size_t col_max_index = std::numeric_limits<index_t>::max();
    unsigned int index_bytes = sizeof(index_t);
    /* all ones on index_bytes bytes: the mask of a packed index, and
     * the sentinel after the last index of a row */
    index_t end_of_row = std::numeric_limits<index_t>::max();
    std::vector<uint8_t *> rows;
    std::vector<weight_t> column_weights;
    /* xor of i+1 over the active rows i of each column. It gives the
     * last row of a column of weight one, and the other row of a column
     * of weight two. The first call of singleton_removal creates it,
     * and delete_row updates it. */
    std::vector<index_t> row_xors;
    /* columns whose weight fell to one since the last singleton
     * removal: their last rows are the singletons to remove. */
    std::vector<index_t> singleton_columns;
    std::mutex singleton_columns_mutex;
    size_t remaining_rows = 0;
    size_t remaining_columns = 0;

    std::make_signed_t<size_t> excess() const
    {
        return remaining_rows - remaining_columns;
    }

    bool is_active(size_t i) const { return rows[i] != nullptr; }

    /* the other row of the column h of weight two, given one row i */
    size_t other_row(size_t h, size_t i) const
    {
        index_t const x = std::atomic_ref(row_xors[h])
            .load(std::memory_order_relaxed);
        return (x ^ index_t(i + 1)) - 1;
    }

    /* Set the column range; index_bytes is the smallest width that
     * holds col_max_index - 1 and the sentinel. */
    void set_column_range(size_t col0, size_t col1);

    index_t unpack(uint8_t const * p) const
    {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v & end_of_row;
    }

    /* The column indices of one row, in the order of the relation: a
     * range that is its own iterator, up to end_of_row. */
    struct row_columns {
        purge_matrix const & M;
        uint8_t const * p;
        struct sentinel {};
        row_columns begin() const { return *this; }
        sentinel end() const { return {}; }
        index_t operator*() const { return M.unpack(p); }
        row_columns & operator++() { p += M.index_bytes; return *this; }
        bool operator==(sentinel) const { return **this == M.end_of_row; }
    };
    row_columns columns(uint8_t const * p) const { return {*this, p}; }
    row_columns columns(size_t i) const { return columns(rows[i]); }

    /* The column weights while several threads count them at the same
     * time: chunks of 2^24 weights, which a thread creates when it
     * first meets a column of their range, and saturating increments.
     * take_column_weights() copies them into column_weights.
     */
    struct column_weight_counter {
        static constexpr unsigned int chunk_bits = 24;
        static constexpr size_t chunk_size = size_t(1) << chunk_bits;
        /* 2^40 columns at most: their weights alone would fill 1 TB. */
        static constexpr size_t max_chunks = size_t(1) << 16;
        std::vector<std::atomic<weight_t *>> chunks =
            std::vector<std::atomic<weight_t *>>(max_chunks);
        ~column_weight_counter();
        void increment(size_t h);
    };

    /* The rows of one group of relations. One thread fills one
     * row_builder. append() moves its rows into the matrix.
     */
    struct row_builder : purge_row_allocator {
        std::vector<uint8_t *> rows;

        /* Add a row that contains the primes in the given list that
         * are within the column range of M, and count every prime in
         * the column weights. The row gets the next row number.
         */
        template <container_with_h C>
        void new_row(column_weight_counter & weights, purge_matrix const & M,
                     C const & primes)
        {
            size_t active_weight = 0;
            for (auto & c: primes)
                active_weight += M.is_stored(c.h);

            auto * p = alloc(M.index_bytes * (active_weight + 1));
            auto * q = p;
            for (auto & c: primes) {
                auto h = c.h;
                weights.increment(h);
                if (M.is_stored(h))
                    q = M.pack(q, h);
            }
            M.pack(q, M.end_of_row);
            rows.push_back(p);
        }
    };

    bool is_stored(size_t h) const
    {
        return h >= col_min_index && h < col_max_index;
    }

    /* Writes h at q, and returns the position after it. */
    uint8_t * pack(uint8_t * q, index_t h) const
    {
        std::memcpy(q, &h, index_bytes);
        return q + index_bytes;
    }

    /* Put the rows of b after the rows of the matrix. b is empty
     * afterwards.
     */
    void append(row_builder & b);

    /* Set column_weights from the counter, and count the non-empty
     * columns.
     */
    void take_column_weights(column_weight_counter const & weights);

    /* one bit per column */
    using column_bitset = std::vector<uint64_t>;
    static bool test_bit(column_bitset const & b, size_t h)
    {
        return b[h / 64] >> (h % 64) & 1;
    }

    /* The resident size of the process and the bytes of the matrix
     * structures, on one line that starts with "# MEMORY: <label>". */
    void print_memory(std::string const & label) const;

    void print_stats_column_weights(FILE *, int verbose) const;
    void print_stats_row_weights(FILE *, int verbose) const;
    void print_stats_on_cliques (FILE *, column_bitset const & weight_two,
            int verbose) const;

    int64_t singleton_removal(size_t nthreads, int verbose);

    /* everything about clique removal is in clique_removal.cpp */
    void clique_removal(int64_t target_excess, size_t nthreads, int verbose);

    struct connected_component {
        size_t i = 0;     /* smallest row */
        float w = 0;      /* weight */

        /* compare weights, first. */
        std::partial_ordering operator<=>(connected_component const & o) const {
            if (auto r = w <=> o.w; r != 0) return r;
            /* same weight, use first row to be deterministic */
            return i <=> o.i;
        }
        auto operator==(connected_component const & o) const {
            return i == o.i;
        }
    };

    /* in clique_removal.cpp */
    static void print_clique_removal_weight_function();

    private:

    /* compute_connected_component() has a hidden quadratic behavior.
     * It's fine if all connected components remain tiny, but it can
     * become a pain quite soon if they don't. The set version has better
     * complexity. As a stopgap measure, we keep both for the moment, and
     * we'll see if there's a point in having only the set version in
     * longer term
     */
    static constexpr size_t ccc_quadratic_abort = 10;

    /* The connected components follow the columns of weight two. The
     * bitset weight_two marks the columns that had weight two when the
     * components were computed: a column that falls to weight two
     * during the deletions is not followed.
     */
    template <weight_function F>
    std::pair<size_t, connected_component>
    compute_connected_component(size_t i0, column_bitset const & weight_two,
                                    std::vector<size_t> & buf,
                                    F const & f) const;

    template <weight_function F>
    std::pair<size_t, connected_component>
    compute_connected_component_with_set(size_t i0,
            column_bitset const & weight_two,
            std::vector<size_t>  & buf,
            F const & f) const;

    void delete_connected_component(size_t i0,
            column_bitset const & weight_two,
            std::vector<size_t> & buf, std::vector<index_t> & weight_one);

    size_t clique_removal_core (column_bitset const & weight_two,
            int64_t target_excess, size_t max_nb_comp,
            size_t nthreads, int verbose);

    column_bitset columns_of_weight(weight_t w, size_t nthreads) const;
    void compute_row_xors(size_t nthreads);
    void delete_rows_with_weight_one_column(size_t nthreads);
    void delete_singletons(size_t nthreads);
    void add_singleton_columns(std::vector<index_t> const & weight_one);

    /* this function needs to touch the matrix data atomically. For this
     * reason, we prefer to limit its exposition. It is used concurrently
     * within singleton_removal and clique_removal.
     *
     * The columns whose weight falls to one are appended to weight_one.
     * Returns false, and does nothing, if the row is deleted already.
     */
    bool delete_row(size_t i, std::vector<index_t> & weight_one);

    static void print_stats_generic(FILE * out, std::vector<size_t> const & w,
                        char const * name, char const * unit, int verbose);

    /* wall time since program start, comparable to seconds() */
    static double wct_elapsed();
};

#endif /* CADO_PURGE_MATRIX_HPP */
