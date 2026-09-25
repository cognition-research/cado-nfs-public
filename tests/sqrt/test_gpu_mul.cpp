/* gpu_mul against mpn_mul: sizes around the transform lengths, random
 * limbs and all-ones limbs (the worst case for the carries), then timings
 * of the two at growing sizes. Built with the shared library:
 *   g++ -O2 test_gpu_mul.cpp -L$B -lgpumul -lgmp */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <gmp.h>

extern "C" int gpu_mul_devices();
extern "C" void gpu_mul(uint64_t * rp, const uint64_t * ap, size_t na, const uint64_t * bp, size_t nb);

static double seconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<mp_limb_t> limbs(size_t n, bool ones)
{
    std::vector<mp_limb_t> v(n);
    if (ones)
        for (auto & x : v)
            x = ~(mp_limb_t) 0;
    else
        mpn_random2(v.data(), n);
    return v;
}

static int check(size_t na, size_t nb, bool ones)
{
    const std::vector<mp_limb_t> a = limbs(na, ones), b = limbs(nb, ones);
    std::vector<mp_limb_t> r(na + nb), s(na + nb);
    if (na >= nb)
        mpn_mul(r.data(), a.data(), na, b.data(), nb);
    else
        mpn_mul(r.data(), b.data(), nb, a.data(), na);
    gpu_mul(s.data(), a.data(), na, b.data(), nb);
    if (r == s)
        return 0;
    size_t i = 0;
    while (r[i] == s[i])
        i++;
    printf("FAIL %zu x %zu %s: first difference at limb %zu\n", na, nb, ones ? "ones" : "random", i);
    return 1;
}

int main(int argc, char ** argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("%d devices\n", gpu_mul_devices());
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    int fails = 0;
    const size_t sizes[] = { 1, 2, 3, 4, 5, 7, 8, 9, 100, 1000, 1023, 1024, 1025, 65536, 65537, 1 << 20, (1 << 20) + 1 };
    for (size_t na : sizes)
        for (size_t nb : sizes)
            for (int ones = 0; ones < 2; ones++)
                fails += check(na, nb, ones);
    for (int i = 0; i < 20; i++)
        fails += check(1 + gmp_urandomm_ui(rs, 1 << 22), 1 + gmp_urandomm_ui(rs, 1 << 22), false);
    printf("%d failures\n", fails);
    const int max_log = argc > 1 ? atoi(argv[1]) : 26;
    for (int log = 20; log <= max_log; log += 2) {
        const size_t n = (size_t) 1 << (log - 1);
        const std::vector<mp_limb_t> a = limbs(n, false), b = limbs(n, false);
        std::vector<mp_limb_t> r(2 * n), s(2 * n);
        double t0 = seconds();
        gpu_mul(s.data(), a.data(), n, b.data(), n);
        double t1 = seconds();
        gpu_mul(s.data(), a.data(), n, b.data(), n);
        double t2 = seconds();
        mpn_mul(r.data(), a.data(), n, b.data(), n);
        double t3 = seconds();
        const bool same = r == s;
        printf("2^%d x 2^%d limbs: gpu %.3f s (first %.3f s), gmp %.3f s, %s\n", log - 1, log - 1,
               t2 - t1, t1 - t0, t3 - t2, same ? "equal" : "DIFFERENT");
        fails += !same;
    }
    return fails != 0;
}
