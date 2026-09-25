#ifndef CADO_LINGEN_QCODE_BINARY_HPP
#define CADO_LINGEN_QCODE_BINARY_HPP

#include <cstddef>

#include <vector>

#include <gmp.h>

#include "gmp_aux.h"
#include "lingen_matpoly_select.hpp"

template<bool is_binary> struct bmstatus;

extern matpoly<true> bw_lingen_basecase(bmstatus<true> & bm, matpoly<true> & E);

/* the report of a step with canceled columns (lucky[j] > 0); true when
 * they make a complete generator: all n columns, each luck_mini times */
bool luck_report(unsigned int t, std::vector<int> const & lucky, unsigned int n, unsigned int luck_mini);

extern void test_basecase(matpoly<true>::arith_hard * ab, unsigned int m, unsigned int n, size_t L, cxx_gmp_randstate & rstate);

extern void test_basecase_bblas(matpoly<true>::arith_hard * ab, unsigned int m, unsigned int n, size_t L, cxx_gmp_randstate & rstate);

#endif	/* LINGEN_QCODE_BINARY_HPP_ */
