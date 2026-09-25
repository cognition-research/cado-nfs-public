#ifndef CADO_LINALG_BWC_LINGEN_HPP
#define CADO_LINALG_BWC_LINGEN_HPP

#include "lingen_bigmatpoly.hpp"
#include "lingen_bmstatus.hpp"
#include "lingen_matpoly_select.hpp"

template<typename matpoly_type>
matpoly_type bw_lingen_single(bmstatus<matpoly_type::is_binary> & bm, matpoly_type & E);

template<bool is_binary>
bigmatpoly<is_binary> bw_biglingen_collective(bmstatus<is_binary> & bm, bigmatpoly<is_binary> & E);

#ifdef LINGEN_GPU
/* glingen: the recursion, the products and the base case on the GPU of
 * the transform plugin; E is moved there and pi comes back. The hints
 * replace the tuning: Cantor products, recursion down to threshold. */
lingen_hints glingen_hints(bw_dimensions<true> const & d, size_t L, unsigned int threshold);
matpoly<true> bw_glingen(bmstatus<true> & bm, matpoly<true> & E);
#endif

#endif	/* LINALG_BWC_LINGEN_HPP_ */
