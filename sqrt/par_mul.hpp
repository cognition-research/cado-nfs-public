#ifndef CADO_SQRT_PAR_MUL_HPP
#define CADO_SQRT_PAR_MUL_HPP

#include <gmp.h>
#include "cxx_mpz.hpp"

/* Multithreaded arithmetic on huge integers, on top of GMP's mpn layer.
 *
 * Linking this file also replaces mpz_mul, mpz_addmul, mpz_submul,
 * mpz_tdiv_qr, mpz_tdiv_q and mpz_mod for the whole program: products of
 * two large operands are split Karatsuba-style into three half-size
 * products that run on separate threads, recursively to a depth fixed by
 * the threads the calling OpenMP context leaves free (a product called
 * from a team of k threads gets 1/k of the threads), and divisions with
 * a large divisor and a large quotient go through a Newton reciprocal
 * made of such products. Smaller operations go to GMP unchanged.
 */

/* {rp, na+nb} <- {ap, na} * {bp, nb}, with na >= nb >= 1 */
void par_mul(mp_ptr rp, mp_srcptr ap, mp_size_t na, mp_srcptr bp, mp_size_t nb);

/* root <- floor(sqrt({pp, np})), returns {pp, np} - root^2. pp[np-1] != 0 */
cxx_mpz par_sqrtrem(cxx_mpz & root, mp_srcptr pp, mp_size_t np);

/* q, r <- a / d, a mod d for a >= d > 0 */
void par_tdiv_qr(cxx_mpz & q, cxx_mpz & r, mpz_srcptr a, mpz_srcptr d);

#endif	/* CADO_SQRT_PAR_MUL_HPP */
