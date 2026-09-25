#ifndef COFAC_STRATEGY_HPP_
#define COFAC_STRATEGY_HPP_

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ecm/facul_strategies.hpp"

/* Host cofactorization strategy from CADO's default method chain, in
 * one of three orders.  "default" is CADO's own: the whole chain of the
 * larger side, then the whole chain of the smaller side.  "cheap-first"
 * runs P-1, P+1 and the small ECM curve on both sides before the large
 * curves.  "interleave" alternates one method per side, larger side
 * first.  All three find the same relations; only the work spent on
 * pairs that fail differs. */
inline facul_strategies make_strategies(std::string const & order,
                                        std::vector<unsigned long> const & lim,
                                        std::vector<unsigned int> const & lpb,
                                        std::vector<unsigned int> const & mfb,
                                        std::vector<int> const & ncurves)
{
    if (order == "default")
        return facul_strategies(lim, lpb, mfb, ncurves, true, 0);
    if (order != "cheap-first" && order != "interleave") {
        fprintf(stderr, "unknown cofactor order '%s' (want default, "
                "cheap-first or interleave)\n", order.c_str());
        exit(1);
    }
    int const nmax = std::max(ncurves[0], ncurves[1]);
    auto const chain = facul_strategy_oneside::default_strategy(nmax);
    int const ncheap = chain.size() - nmax;
    assert(ncheap > 0);
    size_t const cut = order == "cheap-first" ? ncheap : 1;
    facul_strategies::strategy_file sf;
    for (unsigned s0 = 0; s0 <= mfb[0]; s0++)
        for (unsigned s1 = 0; s1 <= mfb[1]; s1++) {
            int const first = s0 < s1;
            std::vector<facul_method::parameters_with_side> v;
            for (size_t k = 0; k < chain.size(); k += cut)
                for (int side : {first, 1 - first})
                    for (size_t i = k; i < std::min(k + cut, chain.size());
                         i++)
                        if ((int)i < ncheap + ncurves[side])
                            v.emplace_back(side, chain[i]);
            sf[{s0, s1}] = std::move(v);
        }
    return facul_strategies(lim, lpb, mfb, true, sf, 0);
}

#endif /* COFAC_STRATEGY_HPP_ */
