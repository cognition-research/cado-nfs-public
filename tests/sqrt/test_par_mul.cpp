#include "cado.h" // IWYU pragma: keep

#include <cstdlib>
#include <memory>

#include <dlfcn.h>
#include <gmp.h>

#include "cxx_mpz.hpp"
#include "macros.h"
#include "sqrt/par_mul.hpp"
#include "tests_common.h"

/* Run with PAR_MUL_DEPTH=12 OMP_NUM_THREADS=64 (see CMakeLists.txt) so
 * that the splits go several levels down at these sizes; the checks are
 * against GMP's own mpn functions, which the interposed mpz_mul does not
 * touch. */

static void random_limbs(mp_ptr p, mp_size_t n, gmp_randstate_t rstate)
{
    mpz_t z;
    mpz_init(z);
    mpz_urandomb(z, rstate, n * GMP_NUMB_BITS);
    mpn_zero(p, n);
    mpn_copyi(p, mpz_limbs_read(z), mpz_size(z));
    mpz_clear(z);
}

static void test_mul(mp_size_t na, mp_size_t nb, gmp_randstate_t rstate)
{
    std::unique_ptr<mp_limb_t[]> a(new mp_limb_t[na]), b(new mp_limb_t[nb]);
    std::unique_ptr<mp_limb_t[]> r(new mp_limb_t[na + nb]), s(new mp_limb_t[na + nb]);
    random_limbs(a.get(), na, rstate);
    random_limbs(b.get(), nb, rstate);
    par_mul(r.get(), a.get(), na, b.get(), nb);
    mpn_mul(s.get(), a.get(), na, b.get(), nb);
    ASSERT_ALWAYS(mpn_cmp(r.get(), s.get(), na + nb) == 0);
    /* and through the interposed mpz_mul, with a sign */
    cxx_mpz x, y, z;
    mpz_t v;
    mpz_neg(x, mpz_roinit_n(v, a.get(), na));
    mpz_set(y, mpz_roinit_n(v, b.get(), nb));
    mpz_mul(z, x, y);
    mpz_neg(z, z);
    ASSERT_ALWAYS(mpz_cmp(z, mpz_roinit_n(v, s.get(), na + nb)) == 0);
    /* w + x*y and w - x*y, with w of the size of the product */
    cxx_mpz w, w_ref;
    mpz_urandomb(w, rstate, (na + nb) * GMP_NUMB_BITS);
    mpz_sub(w_ref, w, z);
    mpz_addmul(w, x, y);
    ASSERT_ALWAYS(mpz_cmp(w, w_ref) == 0);
    mpz_add(w_ref, w_ref, z);
    mpz_submul(w, x, y);
    ASSERT_ALWAYS(mpz_cmp(w, w_ref) == 0);
}

static void test_sqrt(mp_size_t np, gmp_randstate_t rstate, bool square)
{
    std::unique_ptr<mp_limb_t[]> p(new mp_limb_t[np]);
    random_limbs(p.get(), np, rstate);
    if (square) {
        /* p <- (p mod B^(np/2))^2, up to np limbs */
        std::unique_ptr<mp_limb_t[]> r(new mp_limb_t[np]);
        mpn_zero(r.get(), np);
        mpn_sqr(r.get(), p.get(), np / 2);
        mpn_copyi(p.get(), r.get(), np);
    }
    for (; p[np - 1] == 0; np--)
        ;
    cxx_mpz root, rem_ref, root_ref;
    mpz_t v;
    const cxx_mpz rem = par_sqrtrem(root, p.get(), np);
    mpz_sqrtrem(root_ref, rem_ref, mpz_roinit_n(v, p.get(), np));
    ASSERT_ALWAYS(mpz_cmp(root, root_ref) == 0);
    ASSERT_ALWAYS(mpz_cmp(rem, rem_ref) == 0);
    ASSERT_ALWAYS(!square || mpz_sgn(rem) == 0);
}

/* a of na bits over d of nd bits; the reference division is GMP's, since
 * the interposed mpz_tdiv_qr leaves operands of these sizes to it */
static void test_div(mp_bitcnt_t na, mp_bitcnt_t nd, gmp_randstate_t rstate)
{
    cxx_mpz a, d, q, r, q_ref, r_ref;
    mpz_urandomb(a, rstate, na);
    mpz_setbit(a, na - 1);
    mpz_urandomb(d, rstate, nd);
    mpz_setbit(d, nd - 1);
    par_tdiv_qr(q, r, a, d);
    mpz_tdiv_qr(q_ref, r_ref, a, d);
    ASSERT_ALWAYS(mpz_cmp(q, q_ref) == 0);
    ASSERT_ALWAYS(mpz_cmp(r, r_ref) == 0);
    /* an exact quotient, and a dividend just below a multiple */
    mpz_mul(a, q, d);
    par_tdiv_qr(q_ref, r_ref, a, d);
    ASSERT_ALWAYS(mpz_cmp(q, q_ref) == 0 && mpz_sgn(r_ref) == 0);
    mpz_sub_ui(a, a, 1);
    par_tdiv_qr(q_ref, r_ref, a, d);
    mpz_sub_ui(q, q, 1);
    mpz_sub_ui(d, d, 1);
    ASSERT_ALWAYS(mpz_cmp(q, q_ref) == 0 && mpz_cmp(r_ref, d) == 0);
}

/* the interposed mpz_tdiv_qr, mpz_tdiv_q and mpz_mod on operands large
 * enough to leave GMP, with signs: checked by q d + r = a */
static void test_div_interposed(mp_size_t na, mp_size_t nd, gmp_randstate_t rstate)
{
    cxx_mpz a, d, q, r, t;
    mpz_urandomb(a, rstate, na * GMP_NUMB_BITS);
    mpz_urandomb(d, rstate, nd * GMP_NUMB_BITS);
    mpz_neg(a, a);
    mpz_tdiv_qr(q, r, a, d);
    ASSERT_ALWAYS(mpz_sgn(r) <= 0 && mpz_cmpabs(r, d) < 0);
    mpz_mul(t, q, d);
    mpz_add(t, t, r);
    ASSERT_ALWAYS(mpz_cmp(t, a) == 0);
    mpz_tdiv_q(t, a, d);
    ASSERT_ALWAYS(mpz_cmp(t, q) == 0);
    mpz_neg(d, d);
    mpz_mod(t, a, d);
    mpz_sub(t, t, r);
    ASSERT_ALWAYS(mpz_sgn(t) == 0 || mpz_cmpabs(t, d) == 0);
}

int main(int argc, char const * argv[])
{
    unsigned long iter = 200;
    tests_common_cmdline(&argc, &argv, PARSE_SEED | PARSE_ITER);
    tests_common_get_iter(&iter);
    gmp_randstate_t rstate;
    gmp_randinit_default(rstate);
    gmp_randseed_ui(rstate, 42);
    /* the definitions of this executable, not GMP's, are the ones called */
    for (const char * name : { "__gmpz_mul", "__gmpz_addmul", "__gmpz_submul",
                               "__gmpz_tdiv_qr", "__gmpz_tdiv_q", "__gmpz_mod" })
        ASSERT_ALWAYS(dlsym(RTLD_DEFAULT, name) != dlsym(RTLD_NEXT, name));
    ASSERT_ALWAYS(dlsym(RTLD_DEFAULT, "__gmpz_mul") == (void *) &mpz_mul);
    for (unsigned long i = 0; i < iter; i++) {
        const mp_size_t na = 1 + gmp_urandomm_ui(rstate, 3000);
        const mp_size_t nb = 1 + gmp_urandomm_ui(rstate, na);
        test_mul(na, nb, rstate);
        test_sqrt(1 + gmp_urandomm_ui(rstate, 6000), rstate, i & 1);
        const mp_bitcnt_t nd = 1 + gmp_urandomm_ui(rstate, 300000);
        test_div(nd + gmp_urandomm_ui(rstate, 300000), nd, rstate);
    }
    /* quotients much longer and much shorter than the divisor */
    test_div(3000000, 1000, rstate);
    test_div(300000, 299000, rstate);
    test_div_interposed(3 << 20, 1 << 20, rstate);
    test_div_interposed((2 << 20) + 5, (1 << 20) + 3, rstate);
    /* large enough for the interposed mpz_mul to split */
    test_mul((1 << 20) + 12345, (1 << 20) + 777, rstate);
    /* a few sizes past the base precision of the square root */
    for (mp_size_t np : { 2047, 2048, 2049, 4096, 8191, 20000 }) {
        test_sqrt(np, rstate, true);
        test_sqrt(np, rstate, false);
    }
    gmp_randclear(rstate);
    tests_common_clear();
    return EXIT_SUCCESS;
}
