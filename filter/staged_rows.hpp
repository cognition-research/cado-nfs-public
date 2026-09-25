#ifndef CADO_STAGED_ROWS_HPP
#define CADO_STAGED_ROWS_HPP

/* merge and replay store their rows with 32-bit columns (factorization), so they
   read the purged file, whose columns go up to 2^64, in two steps: the
   rows are staged with their original columns, then the columns that
   appear are numbered consecutively and the rows go to the heap. */

#include <bit>
#include <cstdint>
#include <vector>

#include "macros.h"
#include "merge_heap.h"
#include "merge_replay_matrix.h"
#include "typedefs.h"

/* The rows of the purged file, with their original column indices, as
   records [row number, length, columns...] in chunks of cells. */
struct staged_rows {
    static constexpr size_t chunk_cells = size_t(1) << 26;
    std::vector<std::vector<index_t>> chunks;

    void add(index_t i, index_t const * cols, index_t n) {
        if (chunks.empty() || chunks.back().size() + n + 2 > chunk_cells)
            chunks.emplace_back().reserve(chunk_cells);
        auto & chunk = chunks.back();
        chunk.push_back(i);
        chunk.push_back(n);
        chunk.insert(chunk.end(), cols, cols + n);
    }

    /* Takes the rows of other, numbered from row_offset on. */
    void append(staged_rows & other, index_t row_offset) {
        for (auto & chunk : other.chunks)
            for (size_t p = 0; p < chunk.size(); p += 2 + chunk[p+1])
                chunk[p] += row_offset;
        for (auto & chunk : other.chunks)
            chunks.push_back(std::move(chunk));
        other.chunks.clear();
    }

    template<typename F>
    void for_each_row(size_t k, F && f) const {
        auto const & chunk = chunks[k];
        for (size_t p = 0; p < chunk.size(); p += 2 + chunk[p+1])
            f(chunk[p], chunk[p+1], &chunk[p+2]);
    }

    void release(size_t k) { std::vector<index_t>().swap(chunks[k]); }
};

/* Numbers the columns that appear in the matrix consecutively, in the
   order of their original indices. */
struct column_map {
    std::vector<uint64_t> bits;
    std::vector<uint32_t> rank;
    index_t n = 0;

    explicit column_map(index_t ncols)
        : bits((ncols + 63) / 64), rank(bits.size()) {}

    void mark(index_t h) {
#pragma omp atomic
        bits[h / 64] |= uint64_t(1) << (h % 64);
    }

    void finish() {
        for (size_t w = 0; w < bits.size(); w++) {
            rank[w] = n;
            n += std::popcount(bits[w]);
        }
        ASSERT_ALWAYS(n < UINT32_MAX);
    }

    uint32_t operator()(index_t h) const {
        uint64_t below = bits[h / 64] & ((uint64_t(1) << (h % 64)) - 1);
        return rank[h / 64] + std::popcount(below);
    }
};

#ifndef FOR_DL
/* Renumbers the columns that appear in the staged rows, stores the rows
   in the heap with the new numbers, and returns the number of columns. */
static inline index_t build_rows(staged_rows & staged, index_t ncols,
                                 typerow_t ** rows)
{
    size_t nchunks = staged.chunks.size();
    column_map cols(ncols);
#pragma omp parallel for
    for (size_t k = 0; k < nchunks; k++)
        staged.for_each_row(k, [&](index_t, index_t n, index_t const * h) {
            for (index_t c = 0; c < n; c++)
                cols.mark(h[c]);
        });
    cols.finish();
#pragma omp parallel for
    for (size_t k = 0; k < nchunks; k++) {
        staged.for_each_row(k, [&](index_t i, index_t n, index_t const * h) {
            rows[i] = heap_alloc_row(i, n);
            for (index_t c = 0; c < n; c++)
                rows[i][c + 1] = cols(h[c]);
        });
        staged.release(k);
    }
    return cols.n;
}
#endif

#endif	/* CADO_STAGED_ROWS_HPP */
