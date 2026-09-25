/* purge --- perform singleton removal and clique removal
 *
 * Copyright 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2026
 * Cyril Bouvier, Alain Filbois, Francois Morain, Paul Zimmermann, Emmanuel Thomé
 *
 * This file is part of CADO-NFS.
 *
 * CADO-NFS is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * CADO-NFS is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with CADO-NFS; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/* References:
 * On the Number Field Sieve Integer Factorisation Algorithm,
 * Stefania Cavallar, PhD Thesis, University of Leiden, 2002.
 *
 * The filtering step of discrete logarithm and integer factorization
 * algorithms.
 * Cyril Bouvier, preprint, 2013, https://hal.inria.fr/hal-00734654.
 */

/*
 * Important remark:
 *   A relation corresponds to a row of the matrix and a ideal corresponds to
 *   a column of the matrix.
 *
 * This program works in two passes over the relation files:
 * - the first pass loads in memory only indices of columns >= col_min_index
 *   and keeps a count of the weight of each column in column_weights.
 *   Then, a first step of singleton removal is performed followed by 'nsteps'
 *   steps of singleton removal and clique removal, in order to obtained the
 *   final excess 'keep'.
 * - the second pass goes through the relations again, and dumps the remaining
 *   ones in the format needed by 'merge'.

 * This program uses the purge_matrix structure, where most of the stuff
 * happens.
 *
 * M.rows[i] is the list of indices of columns (above col_min_index) of
 * row i, terminated by a -1 sentinel.
 *
 * M.column_weights[h] is the weight of the column h in current rows
 * (saturates at the max value of weight_t)
 */

/*
 * Exit value:
 * - 0 if enough relations
 * - 1 if an error occurred (then we should abort the factorization)
 * - 2 if not enough relations
 */

#include "cado.h" // IWYU pragma: keep
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "fmt/base.h"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "filelist.hpp"
#include "filter_config.h"
#include "filter_io.hpp"
#include "fstream_maybe_compressed.hpp"
#include "gzip.h"
#include "macros.h"
#include "misc.h"
#include "params.hpp"
#include "portability.h"
#include "purge_matrix.hpp"
#include "timing.h"
#include "typedefs.h"
#include "utils_cxx.hpp"
#include "verbose.hpp"

// #define TRACE_J 0x5b841 /* trace column J */

/* Output file that pass 2 writes as one part per input file, so that
 * several threads can write at the same time. The parts have the same
 * compression as the final file. concatenate() appends the parts to the
 * final file in the order of the input files: a sequence of gzip members
 * or of zstd frames is a valid file, and plain parts concatenate as
 * bytes. */
struct parted_output { /* {{{ */
    std::string name;
    size_t nparts;

    /* Creates the final file with only the header line. */
    parted_output(std::string const & name, size_t nparts,
                  std::string const & header)
        : name(name)
        , nparts(nparts)
    {
        ofstream_maybe_compressed out(name);
        if (!out.good())
            throw cado::error("cannot open {} for writing", name);
        out << header;
    }

    std::string part_name(size_t i) const
    {
        auto const suffix = get_suffix(name);
        return fmt::format("{}.part{}{}",
                           name.substr(0, name.size() - suffix.size()), i,
                           suffix);
    }

    void concatenate() const
    {
        std::vector<char> buf(size_t(1) << 24);
        std::ofstream out(name, std::ios::binary | std::ios::app);
        for (size_t i = 0; i < nparts; i++) {
            std::ifstream in(part_name(i), std::ios::binary);
            while (in.read(buf.data(), buf.size()), in.gcount() > 0)
                out.write(buf.data(), in.gcount());
            unlink(part_name(i).c_str());
        }
        if (!out.good())
            throw cado::error("write error on {}", name);
    }
};
/* }}} */

struct purge_output_specification { /* {{{ */
    parameter<std::string, "out", "outfile for remaining relations">
        purgedname;
    parameter<std::string, "outdel", "outfile for deleted relations (for DL)">
        deletedname;

    static void configure(cxx_param_list & pl)
    {
        decltype(purgedname)::configure(pl);
        decltype(deletedname)::configure(pl);
    }

    explicit purge_output_specification(cxx_param_list & pl)
        : purgedname(pl)
        , deletedname(pl)
    {
    }
};
/* }}} */
struct purge_process : purge_output_specification {
        /* {{{ */
    parameter_with_default<
        size_t, "col-min-index",
        "only take into account columns with indices >= col-min-index", "0">
        col_min_index;
    parameter_with_default<
        size_t, "col-max-index",
        "only take into account columns with indices < col-max-index", "0">
        col_max_index;
    parameter_with_default<int64_t, "keep", "wanted excess at the end of purge",
                           CADO_STRINGIZE(DEFAULT_FILTER_EXCESS)>
        keep;
    parameter_with_default<
        int, "nsteps",
        R"(maximal number of steps of clique removal (-1 means choose between 0 and DEFAULT_PURGE_NSTEPS so that each step removes at least about 1% of the number of columns))",
        "-1">
        nsteps;
    parameter_with_default<
        double, "required_excess",
        R"(% of excess required at the end of the 1st singleton removal step)",
        CADO_STRINGIZE(DEFAULT_PURGE_REQUIRED_EXCESS)>
        required_excess;
    parameter_with_default<int, "t", "number of threads",
                           CADO_STRINGIZE(DEFAULT_PURGE_NTHREADS)>
        nthreads;

    parameter_switch<"v", "verbose mode"> verbose;

    purge_matrix M;

    /* row number of the first relation of each input file, set by pass 1 */
    std::vector<size_t> row_base;

    /* }}} */
    static void configure(cxx_param_list & pl) /* {{{ */
    {
        purge_output_specification::configure(pl);
        decltype(col_min_index)::configure(pl);
        decltype(col_max_index)::configure(pl);
        decltype(keep)::configure(pl);
        decltype(nsteps)::configure(pl);
        decltype(required_excess)::configure(pl);
        decltype(nthreads)::configure(pl);
        decltype(verbose)::configure(pl);
    }
    /* }}} */
    explicit purge_process(cxx_param_list & pl) /* {{{ */
        : purge_output_specification(pl)
        , col_min_index(pl)
        , col_max_index(pl)
        , keep(pl)
        , nsteps(pl)
        , required_excess(pl)
        , nthreads(pl)
        , verbose(pl)
    {
        if (col_max_index == 0)
            col_max_index() = std::numeric_limits<index_t>::max();
        if (nthreads == 0)
            pl.fail("cannot have nthreads == 0");
        if (col_min_index >= col_max_index)
            pl.fail("cannot have col-min-index >= col-max-index\n");
        /* If col_max_index > 2^32, then we need index_t to be 64-bit */
        if (col_max_index > std::numeric_limits<index_t>::max())
            pl.fail("Error, -col-max-index is too large for a 32-bit "
                    "program\nSee #define SIZEOF_INDEX in typedefs.h\n");
        M.set_column_range(col_min_index, col_max_index);

        print_information();
    }
    /* }}} */
    /* {{{ print_information: some info and hash-table related stats */
    void print_information() const
    {
        /* this one is in clique_removal.cpp */
        purge_matrix::print_clique_removal_weight_function();

        fmt::print("# INFO: number of rows: automatic\n");
        fmt::print("# INFO: maximum possible index of a column: {}\n",
                   col_max_index());
        fmt::print("# INFO: bytes per stored column index: {}\n",
                   M.index_bytes);
        fmt::print("# INFO: number of threads: {}\n", nthreads());
        fmt::print("# INFO: number of clique removal steps: ");
        if (nsteps < 0)
            fmt::print("will be chosen by the program\n");
        else
            fmt::print("{}\n", nsteps());
        ASSERT_ALWAYS(keep >= 0);
        fmt::print("# INFO: target excess: {}\n", keep());
        fflush(stdout);
    }
    /*}}}*/
    void print_optional_weight_statistics() /* {{{ */
    {
        /* prints some stats on columns and rows weight if verbose > 0. */
        if (verbose) {
            M.print_stats_column_weights(stdout, verbose);
            M.print_stats_row_weights(stdout, verbose);
        }
    }
    /* }}} */
    /* Calls f(i) for each of the nfiles input files, with at most
     * nthreads calls at the same time. An exception from f stops the
     * loop and is thrown again in the calling thread. */
    template <typename F>
    void for_each_file(size_t nfiles, F const & f) const /* {{{ */
    {
        std::atomic<size_t> next = 0;
        std::mutex m;
        std::exception_ptr error;
        auto worker = [&]() {
            try {
                for (size_t i; (i = next++) < nfiles;)
                    f(i);
            } catch (...) {
                std::scoped_lock const dummy(m);
                error = std::current_exception();
                next = nfiles;
            }
        };
        {
            std::vector<std::jthread> threads;
            for (size_t k = std::min<size_t>(nthreads(), nfiles); k--;)
                threads.emplace_back(worker);
        }
        if (error)
            std::rethrow_exception(error);
    }
    /* }}} */

    /* Reads the rows and the column weights of the files into M. The
     * column weight counter and the row builders are freed on return. */
    void read_rows(std::vector<std::string> const & files) /* {{{ */
    {
        using relation_type = cado::relation_building_blocks::primes_block<
            prime_type_for_indexed_relations,
            cado::relation_building_blocks::ab_ignore<16>>;

        std::vector<purge_matrix::row_builder> builders(files.size());
        purge_matrix::column_weight_counter weights;

        for_each_file(files.size(), [&](size_t i) {
            filter_rels<relation_type>(files[i], nullptr, nullptr,
                [&](relation_type & rel) {
                    builders[i].new_row(weights, M, rel.primes);
            });
        });

        size_t builder_bytes = 0;
        for (auto const & b: builders)
            builder_bytes += b.get_allocated_bytes()
                + b.rows.capacity() * sizeof(uint8_t *);
        M.print_memory(fmt::format("pass 1 read, builders {}",
                                   size_disp(builder_bytes)));

        for (auto & b: builders) {
            row_base.push_back(M.rows.size());
            M.append(b);
        }
        M.take_column_weights(weights);
    }
    /* }}} */

    void pass1(filelist const & input) /* {{{ */
    {
        fmt::print("\n"
                   "Pass 1, reading and storing columns with index h >= {}\n",
                   col_min_index());

        read_rows(input.create_file_list());
        M.print_memory("pass 1 matrix");

#ifdef TRACE_J
        printf("TRACE: weight of ideal 0x%x is %u\n", TRACE_J,
               M.column_weights[TRACE_J]);
#endif

        print_optional_weight_statistics();

        /* MAIN FUNCTIONS: do singletons and cliques removal. */
        singleton_and_clique_removal();
        M.print_memory("end of pass 1");

#ifdef TRACE_J
        printf("TRACE: weight of ideal 0x%x is %u\n", TRACE_J,
               M.column_weights[TRACE_J]);
#endif

        print_optional_weight_statistics();

        if (M.remaining_rows < M.remaining_columns + keep) {
            /* XXX Warning: This output line gets ***PARSED*** (eek!) by the
             * Python script to decide whether we have enough excess */
            fmt::print("number of rows < number of columns + keep\n");
            print_final_values(0);
            exit(2);
        }
        if (M.remaining_rows == 0 || M.remaining_columns == 0) {
            fmt::print("number of rows or number of columns is 0\n");
            print_final_values(0);
            exit(2);
        }
    }
    /* }}} */

    void pass2(filelist const & input)/* {{{ */
    {
        fmt::print("\n"
                   "Pass 2, reading and writing output file{}...\n",
                   deletedname().empty() ? "" : "s");

        auto const files = input.create_file_list();
        std::optional<parted_output> out;
        std::optional<parted_output> outdel;

        /* The header line for the file of remaining relations has a
         * bound B such that all non empty columns have indices j < B
         */
        if (purgedname.is_provided()) {
            size_t bound = M.column_weights.size();
            for (; bound && M.column_weights[bound - 1] == 0; bound--)
                ;
            out.emplace(purgedname, files.size(),
                        fmt::format("# {} {} {}\n", M.remaining_rows, bound,
                                    M.remaining_columns));
        }

        if (deletedname.is_provided())
            outdel.emplace(deletedname, files.size(),
                           fmt::format("# {}\n",
                                       M.rows.size() - M.remaining_rows));

        /* second pass over relations in files */
        using relation_type = cado::relation_building_blocks::line_block<
            cado::relation_building_blocks::primecount_block<
                cado::relation_building_blocks::ab_ignore<16>>>;
        std::vector<double> W(files.size(), 0);

        for_each_file(files.size(), [&](size_t i) {
            ofstream_maybe_compressed out_i;
            ofstream_maybe_compressed outdel_i;
            if (out)
                if (out_i.open(out->part_name(i)); !out_i.good())
                    throw cado::error("cannot open {} for writing",
                                      out->part_name(i));
            if (outdel)
                if (outdel_i.open(outdel->part_name(i)); !outdel_i.good())
                    throw cado::error("cannot open {} for writing",
                                      outdel->part_name(i));

            size_t const base = row_base[i];
            filter_rels<relation_type>(files[i], nullptr, nullptr,
                [&](relation_type & rel) {
                    if (M.is_active(base + rel.num)) {
                        W[i] += static_cast<double>(rel.weight);
                        if (out_i.is_open())
                            fmt::print(out_i, "{}\n", rel.line);
                    } else if (outdel_i.is_open()) {
                        fmt::print(outdel_i, "{}\n", rel.line);
                    }
                });
        });

        if (out)
            out->concatenate();
        if (outdel)
            outdel->concatenate();
        M.print_memory("end of pass 2");

        /* write final values to stdout */
        /* This output, incl. "Final values:", is required by the script */
        print_final_values(std::accumulate(W.begin(), W.end(), 0.0));
    }
/* }}} */

    /* {{{ write final values to stdout */
    /* This output, incl. "Final values:", is required by the script */
    void print_final_values(double weight) const
    {
        fmt::print("Final values:\n"
                   "nrows={} ncols={} excess={}\n"
                   "weight={:1.0f} weight*nrows={:1.2e}\n",
                   M.remaining_rows, M.remaining_columns, M.excess(), weight,
                   weight * static_cast<double>(M.remaining_rows));
        fflush(stdout);
    }
    /* }}} */

    /* Note: if nsteps is negative, then the value is chosen by the
     * function. */
    void singleton_and_clique_removal() /* {{{ */
    {
        /* First step of singletons removal */
        fmt::print("\nStep 0: only singleton removal\n");
        M.singleton_removal(nthreads, verbose);
        M.print_memory("step 0");

#ifdef TRACE_J
        printf("# TRACE: weight of ideal 0x%x is %u\n", TRACE_J,
               M.column_weights[TRACE_J]);
#endif

        if (M.excess() <= 0) /* covers case nrows = ncols = 0 */
        {
            /* XXX Warning: This output line gets ***PARSED*** (eek!) by the
             * Python script to decide whether we have enough excess */
            fmt::print("number of rows < number of columns + keep\n");
            print_final_values(0);
            exit(2);
        }

        if ((double)M.excess() <
            required_excess * (double)M.remaining_columns) {
            /* XXX Warning: This output line gets ***PARSED*** (eek!) by the
             * Python script to decide whether we have enough excess */
            fmt::print("(excess / ncols) = {:.2f} < {:.2f}."
                       " See -required_excess argument.\n",
                       double_ratio(M.excess(), M.remaining_columns),
                       static_cast<double>(required_excess()));
            print_final_values(0);
            exit(2);
        }

        /* delta between the current excess and the desired final excess. */
        auto delta_excess = M.excess() - keep;

        /* If nsteps was not given in the command line, adjust nsteps in
           [1..DEFAULT_PURGE_NSTEPS] so that each step removes at least about 1%
           wrt the number of columns */
        if (nsteps < 0) {
            /* If we are less than or equal to the wanted final excess,
             * there is no need for clique removal, so set nsteps to 0*/
            if (delta_excess <= 0)
                nsteps() = 0;
            else if ((size_t)delta_excess / DEFAULT_PURGE_NSTEPS <
                     M.remaining_columns / 100)
                nsteps() = 1 + (100 * delta_excess) / M.remaining_columns;
            else
                nsteps() = DEFAULT_PURGE_NSTEPS;
        }

        int64_t chunk = 0;
        if (nsteps > 0) {
            chunk = delta_excess / nsteps;
            fmt::print("# INFO: number of clique removal steps: {}\n",
                       nsteps());
            fmt::print("# INFO: At each step, excess will be decreased by {}\n",
                       chunk);
            fflush(stdout);
        } else
            fmt::print("# INFO: No step of clique removal will be done\n");

        auto one_step = [&](std::string const & s, int64_t target_excess) {
            size_t const oldnrows = M.remaining_rows;
            int64_t const oldexcess = M.excess();
            fmt::print("\nStep {}: target excess is {}\n", s, target_excess);
            fflush(stdout);

            print_optional_weight_statistics();

            double const wct0 = wct_seconds();
            M.clique_removal(target_excess, nthreads, verbose);
            double const wct1 = wct_seconds();
            M.singleton_removal(nthreads, verbose);
            double const wct2 = wct_seconds();

#ifdef TRACE_J
            printf("TRACE: weight of ideal 0x%x is %u\n", TRACE_J,
                   M.column_weights[TRACE_J]);
#endif

            fmt::print("This step removed {} rows and decreased excess by {}\n",
                       oldnrows - M.remaining_rows, oldexcess - M.excess());
            fmt::print("This step took {:.2f}s wall: clique removal {:.2f}s,"
                       " singleton removal {:.2f}s\n",
                       wct2 - wct0, wct1 - wct0, wct2 - wct1);
            if (oldexcess > M.excess())
                fmt::print("Each excess row deleted {:2.2f} rows\n",
                           double_ratio(oldnrows - M.remaining_rows,
                                        oldexcess - M.excess()));
        };
        /* nsteps steps of clique removal + singletons removal */
        for (int count = 0; count < nsteps && M.excess() > 0; count++) {
            one_step(fmt::format("{} of {}", count + 1, nsteps()),
                     std::max(M.excess() - chunk, keep()));
        }

        /* May need an extra step of clique removal + singletons removal
         * if excess is still larger than keep. It may happen due to the
         * fact that each clique does not make the excess go down by one
         * but can (rarely) leave the excess unchanged. */
        if (M.excess() > keep && nsteps > 0)
            one_step("extra", keep);
    }
    /* }}} */
};

/*************************** main ********************************************/

int main(int argc, char const * argv[])
{
    cxx_param_list pl;

    double const cpu0 = seconds();
    double const wct0 = wct_seconds();

    verbose_decl_usage(pl);
    filelist::configure(pl);
    cado::filter_io_details::configure(pl);
    purge_process::configure(pl);

    pl.process_command_line(argc, argv, true);

    /* print command-line arguments */
    verbose_interpret_parameters(pl);
    cado::filter_io_details::interpret_parameters(pl);
    pl.print_command_line(stdout);
    fflush(stdout);

    filelist const input(pl, argc, argv);

    purge_process pp(pl);

    if (pl.warn_unused())
        pl.fail("Error, unused parameters are given\n");

    pp.pass1(input);
    pp.pass2(input);

    /* print usage of time and memory */
    print_timing_and_memory(stdout, cpu0, wct0);

    return 0;
}
