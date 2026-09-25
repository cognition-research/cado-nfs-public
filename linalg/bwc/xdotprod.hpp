#ifndef CADO_XDOTPROD_HPP
#define CADO_XDOTPROD_HPP

#include <cstdint>              // for uint32_t

#include <vector>

#include "matmul_top.hpp"

/* This interface is relevant to both krylov and mksol, since it's used
 * both for checking and computing A files (obviously only krylov is
 * concerned by the latter aspect). */

void x_dotprod(arith_generic::elt * dst, std::vector<uint32_t> const & xv, unsigned int j0, unsigned int j1, unsigned int nx, mmt_vec const & v, int sign);

/* Deferred variant for device-resident vectors, only for
 * characteristic two. x_dotprod_queue captures the words that the dot
 * product of v reads, on the device, as slot number slot, and returns
 * 1. It returns 0 when the capture cannot run on the device; the
 * caller then uses x_dotprod (with sign 1) instead. x_dotprod_flush
 * adds the nslots captured dot products to dst, in slot order, one
 * block of (j1-j0) elements per slot, the blocks dst_stride elements
 * apart. */
int x_dotprod_queue(std::vector<uint32_t> const & xv, unsigned int j0, unsigned int j1, unsigned int nx, mmt_vec const & v, size_t slot);
void x_dotprod_flush(arith_generic::elt * dst, size_t dst_stride, std::vector<uint32_t> const & xv, unsigned int j0, unsigned int j1, unsigned int nx, mmt_vec const & v, size_t nslots);

#endif	/* XDOTPROD_HPP_ */
