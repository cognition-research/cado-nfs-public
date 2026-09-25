#include "cado.h" // IWYU pragma: keep

#ifndef HAVE_AVX2
#error "This file assumes AVX2 support!"
#endif /* HAVE_AVX2 */

#include <cstdint>
#include <cstdlib>

#include <vector>

#include <immintrin.h>

#include "las-unsieve.hpp"
#include "macros.h"
#include "arithxx/u64arith.h"

static const __m256i sign_conversion = _mm256_set1_epi8(-128);
static const __m256i ff = _mm256_set1_epi8(0xff);

void
search_survivors_in_line_avx2_siqs(
        unsigned char * SS,
        unsigned char bound,
        unsigned int length,
        std::vector<uint16_t> &survivors)
{
    __m256i const B = _mm256_xor_si256(_mm256_set1_epi8(bound+1), sign_conversion);
    const unsigned int x_step = sizeof(__m256i);

    for (unsigned int x_start = 0; x_start < length; x_start += x_step)
    {
        /* Do bounds check using AVX2 pattern, set non-survivors in SS[0] array
           to 255 */
        __m256i * ptrS = (__m256i *)(SS + x_start);
        __m256i const s = *ptrS;
        __m256i m = _mm256_cmpgt_epi8(B, _mm256_xor_si256(s, sign_conversion));
        /* movemask returns an int, we want to zero-extend the result into an
         * uint64_t. To avoid sign-extension we need to first cast it into an
         * unsigned int and then into an uint64_t */
        uint64_t bitmask = (uint64_t) (unsigned int) _mm256_movemask_epi8(m);
        m = _mm256_xor_si256(m, ff);
        *ptrS = _mm256_or_si256(s, m);

        for (unsigned int x = x_start; UNLIKELY(bitmask != 0); ++x) {
            unsigned int const tz = u64arith_ctz(bitmask);
            x += tz;
            bitmask >>= tz + 1u;

            survivors.push_back(x);
        }
    }
}
