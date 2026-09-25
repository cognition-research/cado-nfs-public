#include "eprime.hpp"

#include <cstdio>
#include <cstdlib>

#include "cado_poly.hpp"

namespace {

void usage(char const *program)
{
    std::fprintf(stderr,
                 "usage: %s POLY Bf Bg AREA STEPS ALPHA_BOUND\n",
                 program);
    std::exit(EXIT_FAILURE);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 7)
        usage(argv[0]);

    cxx_cado_poly poly;
    if (!poly.read(argv[1]))
        return EXIT_FAILURE;

    double const bound_f = std::strtod(argv[2], nullptr);
    double const bound_g = std::strtod(argv[3], nullptr);
    double const area = std::strtod(argv[4], nullptr);
    unsigned int const steps = std::strtoul(argv[5], nullptr, 10);
    unsigned long const alpha_bound = std::strtoul(argv[6], nullptr, 10);

    eprime_result const result = eprime_score(
        poly, bound_f, bound_g, area, steps, alpha_bound);
    std::printf(
        "Eprime %.17g\nsmooth_part_mean %.17g\nsmooth_part_variance %.17g\n"
        "algebraic_alpha %.17g\nrational_alpha %.17g\n",
        result.score, result.smooth_part_mean, result.smooth_part_variance,
        result.algebraic_alpha, result.rational_alpha);
    return EXIT_SUCCESS;
}
