#include "cado.h" // IWYU pragma: keep

#include <cstdint>

#include <memory>

#include <gmp.h>

#ifdef HAVE_SSE2
#include <x86intrin.h>
#endif

#include "arith-cross.hpp"
#include "bblas_level3a.hpp"
#include "bblas_level3a1.hpp"
#include "bblas_level3c.hpp"
#include "bblas_mat64.hpp"
#include "macros.h"
#include "misc.h"

/* code for add_dotprod {{{ */
/* Parameter L may be fixed at the template level, and used as a way to
 * special-case the code. The default version does not do that, though.
 */
template<unsigned int, unsigned int>
struct add_dotprod {
    void operator()(
            uint64_t * w,           // 64L at a time
            const uint64_t * u,     // 64K at a time
            const uint64_t * v,     // 64L at a time
            unsigned int n,
            unsigned int K,
            unsigned int L) const
    {
        /* u has n rows of 64K bits
         * v has n rows of 64L bits.
         * Compute (u|v) == tr(u)*v into the area pointed to by v: 64K
         * rows of 64L bits.
         */
        /* This is meant for L > 2 -- whether K == 1 or not does not
         * matter much, as usual. We're not using sse-2 here because of
         * lazyness -- and little expected returns.
         */
        /* Apparently we don't have this code in bblas at this point.
         */
        for(unsigned int i = 0 ; i < n ; i++) {
            uint64_t * w0 = (uint64_t *) w;
            for(unsigned int l = 0 ; l < L ; l++) {
                uint64_t const b = *v++;
                uint64_t *sw = w0;
                const uint64_t * u0 = u;
                for(unsigned int k = 0 ; k < K ; k++) {
                    uint64_t a = *u0++;
                    for (unsigned int j = 0; j < 64; j ++) {
                        *sw ^= b & -(a & 1);
                        a >>= 1;
                        sw += L;
                    }
                }
                w0++;
            }
            u += K;
        }
    }
};

/* u has n rows of 64K bits
 * v has n rows of 64 bits.
 * Compute (u|v) == tr(u)*v into the area pointed to by v: 64K rows of 64 bits.
 */
template<>
struct add_dotprod<0,1> {
    void operator()(
            uint64_t * w,
            const uint64_t * u,
            const uint64_t * v, unsigned int n,
            unsigned int K,
            unsigned int /* L=1 is hard-coded */) const
    {
        addmul_TN64K_N64(w, u, v, n, K);
    }
};

#if    defined(HAVE_SSE2) && GMP_LIMB_BITS == 64
/* u has n rows of 64K bits
 * v has n rows of 128 bits.
 * Compute (u|v) == tr(u)*v into the area pointed to by v: 64K rows of 128 bits.
 */
template<>
struct add_dotprod<0,2> {
    void operator()(
            uint64_t * w,           // 128 at a time
            const uint64_t * u,     // 64K at a time
            const uint64_t * v,     // 128 at a time
            unsigned int n,
            unsigned int K,
            unsigned int /* L=2 is hard-coded */) const
    {
        /* okay, we converted the m128i* to u64* for interchange, and now
         * we're casting it back...  */
        /* Apparently we don't have this code in bblas at this point.
         */
        __m128i const * vv = reinterpret_cast<const __m128i*>(v);
        for(unsigned int i = 0 ; i < n ; i++) {
            __m128i * w0 = (__m128i*) w;
            __m128i const mb[4][2] = {
                {_mm_setzero_si128(), _mm_setzero_si128()},
                {*vv, _mm_setzero_si128()},
                {_mm_setzero_si128(), *vv},
                {*vv, *vv},
            };
            vv++;
            __m128i *sw = w0;
            for(unsigned int k = 0 ; k < K ; k++) {
                uint64_t a = *u++;
                for (unsigned int j = 0; j < 64; j += 2) {
                    _mm_storeu_si128(sw, _mm_xor_si128(_mm_loadu_si128(sw), mb[a & 3][0]));
                    sw ++;
                    _mm_storeu_si128(sw, _mm_xor_si128(_mm_loadu_si128(sw), mb[a & 3][1]));
                    sw ++;
                    a >>= 2;
                }
            }
        }
    }
};
#endif

/* Here is code for another specialization. (for <K,2>)
 *
static inline void add_dotprod_64K_128(uint64_t * b, const uint64_t * A, const uint64_t * x, unsigned int ncol, unsigned int K)
{
    uint64_t idx, i, rA;
    uint64_t rx[2];

    abort();    // untested.

    for(idx = 0; idx < ncol; idx++) {
        rx[0] = *x++;
        rx[1] = *x++;
        uint64_t* pb = b;
        for(unsigned int j = 0 ; j < K ; j++) {
            rA = *A++;
            for(i = 0; i < 64; i++) {
                *pb++ ^= rx[0] & -(rA & 1);
                *pb++ ^= rx[1] & -(rA & 1);
                rA >>= 1;
            }
        }
    }
}
*/

/* }}} */

/* {{{ code for addmul_tiny */
template<unsigned int, unsigned int>
struct addmul_tiny {
    /* multiply the n times 64K-bit vector u by the 64K by
     * 64L matrix v -- n must be even. Result is put in w.
     */
    void operator()(
            uint64_t * w,           // 64L at a time
            const uint64_t * u,     // 64K at a time
            const uint64_t * v,     // 64L at a time
            unsigned int n,
            unsigned int K,
            unsigned int L) const
    {
#if     defined(HAVE_SSE2) && GMP_LIMB_BITS == 64
        ASSERT_ALWAYS((n & 1) == 0);
        /* We're doing addmul, so don't clear the output. */
        /* We'll read and write rows two at a time. So we maintain two
         * pointers for each, interlaced. */
        uint64_t * u0 = (uint64_t *) u;
        uint64_t * u1 = (uint64_t *) (u + K);
        uint64_t * w0 = (uint64_t *) w;
        uint64_t * w1 = (uint64_t *) (w + L);
        for (unsigned int j = 0; j < n; j += 2 ) {
            const uint64_t * v0 = v;
            for(unsigned int l = 0 ; l < L ; l++) {
                __m128i r = _mm_setzero_si128();
                const uint64_t * vv = v0;
                for(unsigned int k = 0 ; k < K ; k++) {
                    __m128i a = _mm_setr_epi64(_mm_cvtsi64_m64(u0[k]), _mm_cvtsi64_m64(u1[k]));
                    __m128i const one = _mm_set1_epi64(_mm_cvtsi64_m64(1));
                    for (unsigned int i = 0; i < 64; i++) {
                        __m128i const zw = _mm_set1_epi64(_mm_cvtsi64_m64(*vv));
                        r = _mm_xor_si128(r, _mm_and_si128(zw, _mm_sub_epi64(_mm_setzero_si128(), _mm_and_si128(a, one))));
                        a = _mm_srli_epi64(a, 1);
                        vv += L;
                    }
                }
                /* Because L > 1, the destination words are not contiguous */
                w0[l] ^= _mm_cvtsi128_si64(r);
                w1[l] ^= _mm_cvtsi128_si64(_mm_srli_si128(r, 8));
                v0++; /* next column in v */
            }
            u0 += 2 * K; u1 += 2 * K;
            w0 += 2 * L; w1 += 2 * L;
        }
#else   /* HAVE_SSE2 */
        /* This is just a direct translation of the sse version. Has been
         * tested once. */
        uint64_t * u0 = (uint64_t *) u;
        uint64_t * w0 = (uint64_t *) w;
        for (unsigned int j = 0; j < n; j ++ ) {
            const uint64_t * v0 = v;
            for(unsigned int l = 0 ; l < L ; l++) {
                uint64_t rx = 0;
                const uint64_t * vv = v0;
                for(unsigned int k = 0 ; k < K ; k++) {
                    uint64_t a = u0[k];
                    for (unsigned int i = 0; i < 64; i++) {
                        rx ^= (*vv & -(a & (uint64_t) 1));
                        a >>= 1;
                        vv += L;
                    }
                }
                /* Because L > 1, the destination words are not contiguous */
                w0[l] ^= rx;
                v0++; /* next column in v */
            }
            u0 += K;
            w0 += L;
        }
#endif
    }
};

/* The L == 1 case computes w[j] ^= u[j] * F for each of the n rows
 * u[j], where F is the 64 by 64 bit matrix at v (row i of F is v[i]).
 * mksol calls this once per iteration on the local vector block, so
 * this case carries most of the mksol CPU time.  The table path
 * (method of four Russians) works on every processor; the GFNI path
 * replaces it at run time on processors with GFNI and AVX-512. */

static void addmul_tiny_64_64_m4r(uint64_t * w, const uint64_t * u,
        const uint64_t * v, unsigned int n)
{
    /* T[b][x] holds the product of byte value x by rows 8b..8b+7 */
    uint64_t T[8][256];
    for (unsigned int b = 0; b < 8; b++) {
        T[b][0] = 0;
        for (unsigned int x = 1; x < 256; x++) {
            unsigned int const lsb = x & -x;
            T[b][x] = T[b][x ^ lsb] ^ v[8 * b + cado_ctz(lsb)];
        }
    }
    for (unsigned int j = 0; j < n; j++) {
        uint64_t const a = u[j];
        w[j] ^= T[0][a & 0xff]
              ^ T[1][(a >> 8) & 0xff]
              ^ T[2][(a >> 16) & 0xff]
              ^ T[3][(a >> 24) & 0xff]
              ^ T[4][(a >> 32) & 0xff]
              ^ T[5][(a >> 40) & 0xff]
              ^ T[6][(a >> 48) & 0xff]
              ^ T[7][(a >> 56) & 0xff];
    }
}

#if defined(__x86_64__) && defined(__GNUC__) && defined(HAVE_SSE2)
#define ADDMUL_TINY_HAVE_GFNI 1

static bool addmul_tiny_use_gfni()
{
    static bool const yes = __builtin_cpu_supports("gfni")
        && __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vbmi");
    return yes;
}

/* Pack the 8x8 bit block B (y = x * B, bit k = 2^k) into the operand
 * qword of gf2p8affineqb.  The instruction computes, for output bit j,
 * the parity of (operand byte (7 - j)) AND the input byte; so operand
 * byte (7 - j) holds column j of B. */
static uint64_t addmul_tiny_pack_affine(const uint8_t B[8])
{
    uint64_t A = 0;
    for (unsigned int j = 0; j < 8; j++) {
        uint8_t col = 0;
        for (unsigned int k = 0; k < 8; k++)
            col |= ((B[k] >> j) & 1) << k;
        A |= (uint64_t) col << (8 * (7 - j));
    }
    return A;
}

/* Eight rows per step: one byte transpose, eight affine products (one
 * per output byte column), and three vpermt2q rounds that fold the
 * eight qword lanes of the eight affine outputs into one vector. */
__attribute__((target("gfni,avx512f,avx512bw,avx512vbmi")))
static void addmul_tiny_64_64_gfni(uint64_t * w, const uint64_t * u,
        const uint64_t * v, unsigned int n)
{
    __m512i M[8];
    for (unsigned int c = 0; c < 8; c++) {
        uint64_t m[8];
        for (unsigned int l = 0; l < 8; l++) {
            uint8_t B[8];
            for (unsigned int k = 0; k < 8; k++)
                B[k] = (uint8_t) (v[8 * l + k] >> (8 * c));
            m[l] = addmul_tiny_pack_affine(B);
        }
        M[c] = _mm512_loadu_si512(m);
    }
    /* qword l byte r of the transpose takes byte l of row r */
    uint8_t tp[64];
    for (unsigned int l = 0; l < 8; l++)
        for (unsigned int r = 0; r < 8; r++)
            tp[8 * l + r] = (uint8_t) (8 * r + l);
    __m512i const TP = _mm512_loadu_si512(tp);
    __m512i const EV = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    __m512i const OD = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

    unsigned int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m512i const T = _mm512_permutexvar_epi8(TP,
                _mm512_loadu_si512(u + j));
        __m512i Y[8], A[4], B[2];
        for (unsigned int c = 0; c < 8; c++)
            Y[c] = _mm512_gf2p8affine_epi64_epi8(T, M[c], 0);
        for (unsigned int c = 0; c < 4; c++)
            A[c] = _mm512_xor_si512(
                _mm512_permutex2var_epi64(Y[2*c], EV, Y[2*c+1]),
                _mm512_permutex2var_epi64(Y[2*c], OD, Y[2*c+1]));
        for (unsigned int c = 0; c < 2; c++)
            B[c] = _mm512_xor_si512(
                _mm512_permutex2var_epi64(A[2*c], EV, A[2*c+1]),
                _mm512_permutex2var_epi64(A[2*c], OD, A[2*c+1]));
        __m512i const G = _mm512_xor_si512(
                _mm512_permutex2var_epi64(B[0], EV, B[1]),
                _mm512_permutex2var_epi64(B[0], OD, B[1]));
        _mm512_storeu_si512(w + j, _mm512_xor_si512(
                    _mm512_loadu_si512(w + j),
                    _mm512_permutexvar_epi8(TP, G)));
    }
    if (j < n)
        addmul_tiny_64_64_m4r(w + j, u + j, v, n - j);
}
#endif

template<>
struct addmul_tiny<0,1> {
    void operator()(
            uint64_t * w,
            const uint64_t * u,
            const uint64_t * v,
            unsigned int n,
            unsigned int K,
            unsigned int L) const
    {
        if (K != 1) {
            ::addmul_tiny<0,0>()(w, u, v, n, K, L);
            return;
        }
#ifdef ADDMUL_TINY_HAVE_GFNI
        if (addmul_tiny_use_gfni()) {
            addmul_tiny_64_64_gfni(w, u, v, n);
            return;
        }
#endif
        addmul_tiny_64_64_m4r(w, u, v, n);
    }
};
/* }}} */

/* Transpose a matrix of 64K rows of 64L bits */
struct transpose {
    void operator()(uint64_t * w, const uint64_t * u, unsigned int K, unsigned int L) const
    {
#if 1
        /* We have 1*(64*K)*L*1 64-bit words */
        /* We want 1*K*L*64*1 64-bit words */
        std::unique_ptr<uint64_t[]> temp(new uint64_t[K*L*64]);
        if (u != w)
            generic_transpose_words<uint64_t>(w, u, 1, 64, K*L, 1);
        else
            generic_transpose_words_inplace<uint64_t>(w, 1, 64, K*L, 1, temp.get());
        for (unsigned int k = 0; k < K * L; k++) {
            mat64& d(reinterpret_cast<mat64&>(w[k]));
            mat64_transpose(d, d);
        }
        /* We have 1*K*L*64*1 64-bit words */
        /* We want 1*L*(64*K)*1 64-bit words */
        generic_transpose_words_inplace<uint64_t>(w, 1, K, L*64, 1, temp.get());
#else
        uint64_t md = 1UL;
        unsigned int od = 0;
        uint64_t ms = 1UL;
        unsigned int os = 0;
        uint64_t * dst = (uint64_t *) w;
        const uint64_t * src = (const uint64_t *) u;
        for(unsigned int l = 0 ; l < 64*L ; l++) {
            for(unsigned int k = 0 ; k < 64*K ; k++) {
                dst[od] |= md & -((src[os+k*L] & ms) != 0);
                md <<= 1;
                od += md == 0;
                md += md == 0;
            }
            ms <<= 1;
            os += ms == 0;
            ms += ms == 0;
        }
#endif
    }
};


template<unsigned int K, unsigned int L> struct arith_cross_gf2;

template<unsigned int L>
struct arith_cross_gf2<0, L>
: public arith_cross_generic
{
    unsigned int g0;
    unsigned int g1;
    arith_cross_gf2(arith_generic const * A0, arith_generic const * A1)
        : g0(A0->simd_groupsize())
        , g1(A1->simd_groupsize())
    {
        ASSERT_ALWAYS(L == 0 || g1 == L * 64);
    }
    void add_dotprod(arith_generic::elt * w, arith_generic::elt const * u, arith_generic::elt const * v, unsigned int n) const override
    {
        auto * xw = reinterpret_cast<uint64_t *>(w);
        auto const * xv = reinterpret_cast<const uint64_t *>(v);
        auto const * xu = reinterpret_cast<const uint64_t *>(u);
        ::add_dotprod<0,L>()(xw,xu,xv,n,g0/64,g1/64);
    }

    void addmul_tiny(arith_generic::elt * w, arith_generic::elt const * u, arith_generic::elt const * v, unsigned int n) const override {
        auto * xw = reinterpret_cast<uint64_t *>(w);
        auto const * xv = reinterpret_cast<const uint64_t *>(v);
        auto const * xu = reinterpret_cast<const uint64_t *>(u);
        ::addmul_tiny<0,L>()(xw,xu,xv,n,g0/64,g1/64);
    }

    void transpose(arith_generic::elt * w, arith_generic::elt const * u) const override {
        auto * xw = reinterpret_cast<uint64_t *>(w);
        auto const * xu = reinterpret_cast<const uint64_t *>(u);
        ::transpose()(xw,xu,g0/64,g1/64);
    }

    ~arith_cross_gf2() override = default;
    arith_cross_gf2(arith_cross_gf2 const &) = delete;
    arith_cross_gf2(arith_cross_gf2 &&) = delete;
    arith_cross_gf2& operator=(arith_cross_gf2 const &) = delete;
    arith_cross_gf2& operator=(arith_cross_gf2 &&) = delete;
};

struct arith_cross_gfp : public arith_cross_generic
{
    arith_generic * A;
    arith_cross_gfp(arith_generic * A0, arith_generic const * A1)
        : A(A0)
    {
        ASSERT_ALWAYS(A1->simd_groupsize() == A->simd_groupsize());
        ASSERT_ALWAYS(A->simd_groupsize() == 1);
        ASSERT_ALWAYS(mpz_cmp(A->characteristic(), A1->characteristic()) == 0);
        ASSERT_ALWAYS(A->impl_name() == A1->impl_name());
    }
    void add_dotprod(arith_generic::elt * w, arith_generic::elt const * u, arith_generic::elt const * v, unsigned int n) const override
    {
        A->vec_add_dotprod(*w, u, v, n);
    }

    void addmul_tiny(arith_generic::elt * w, arith_generic::elt const * u, arith_generic::elt const * v, unsigned int n) const override
    {
        A->vec_addmul_and_reduce(w, u, *v, n);
    }

    void transpose(arith_generic::elt * w, arith_generic::elt const * u) const override
    {
        A->set(*w, *u);
    }

    ~arith_cross_gfp() override = default;
    arith_cross_gfp(arith_cross_gfp const &) = delete;
    arith_cross_gfp(arith_cross_gfp &&) = delete;
    arith_cross_gfp& operator=(arith_cross_gfp const &) = delete;
    arith_cross_gfp& operator=(arith_cross_gfp &&) = delete;
};

arith_cross_generic * arith_cross_generic::instance(arith_generic * A0, arith_generic * A1)
{
    if (A0->is_characteristic_two() && A1->is_characteristic_two()) {
        if (A1->simd_groupsize() == 64)
            return new arith_cross_gf2<0,1>(A0, A1);
        else if (A1->simd_groupsize() == 128)
            return new arith_cross_gf2<0,2>(A0, A1);
        else
            return new arith_cross_gf2<0,0>(A0, A1);
    } else if (mpz_cmp(A0->characteristic(), A1->characteristic()) == 0) {
        if (A0->simd_groupsize() != 1) return nullptr;
        if (A1->simd_groupsize() != 1) return nullptr;
        return new arith_cross_gfp(A0,A1);
    } else {
        return nullptr;
    }
}
