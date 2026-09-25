#include "eprime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include <gmp.h>

#include "getprime.h"
#include "mpz_poly.h"
#include "polynomial.hpp"
#include "rho.h"
#include "rootfinder.h"

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr int rational_side = 0;
constexpr int algebraic_side = 1;

struct moments {
    double first;
    double second;
};

cxx_mpz_poly divide_content(cxx_mpz_poly const &f, unsigned long p,
                            unsigned int &valuation)
{
    cxx_mpz content;
    mpz_poly_content(content, f);
    assert(mpz_sgn(content) != 0);

    cxx_mpz_poly g = f;
    valuation = 0;
    while (mpz_divisible_ui_p(content, p)) {
        mpz_divexact_ui(content, content, p);
        for (int i = 0; i <= g.degree(); ++i)
            mpz_divexact_ui(mpz_poly_coeff(g, i), mpz_poly_coeff_const(g, i), p);
        ++valuation;
    }
    return g;
}

cxx_mpz_poly translate_scale(cxx_mpz_poly const &f, unsigned long root,
                             unsigned long p)
{
    cxx_mpz_poly h = f;
    for (int i = h.degree() - 1; i >= 0; --i)
        for (int j = i; j < h.degree(); ++j)
            mpz_addmul_ui(mpz_poly_coeff(h, j),
                          mpz_poly_coeff_const(h, j + 1), root);

    cxx_mpz power = 1;
    for (int i = 0; i <= h.degree(); ++i) {
        mpz_mul(mpz_poly_coeff(h, i), mpz_poly_coeff_const(h, i), power);
        mpz_mul_ui(power, power, p);
    }
    return h;
}

cxx_mpz_poly reverse_scale(cxx_mpz_poly const &f, unsigned long p)
{
    cxx_mpz_poly h;
    cxx_mpz power = 1;
    for (int i = 0; i <= f.degree(); ++i) {
        cxx_mpz coefficient;
        mpz_mul(coefficient, f.coeff(f.degree() - i), power);
        mpz_poly_setcoeff(h, i, coefficient);
        mpz_mul_ui(power, power, p);
    }
    return h;
}

moments affine_moments(cxx_mpz_poly const &f, unsigned long p,
                       unsigned int previous_valuation,
                       gmp_randstate_ptr random_state)
{
    unsigned int content_valuation;
    cxx_mpz_poly g = divide_content(f, p, content_valuation);
    unsigned int const valuation = previous_valuation + content_valuation;

    moments result{
        static_cast<double>(content_valuation),
        static_cast<double>(valuation) * static_cast<double>(valuation),
    };

    std::vector<unsigned long> roots(std::max(g.degree(), 1));
    unsigned int const count =
        mpz_poly_roots_ulong(roots.data(), g, p, random_state);
    cxx_mpz derivative;
    double const denominator = static_cast<double>(p - 1);

    for (unsigned int i = 0; i < count; ++i) {
        unsigned long const root = roots[i];
        mpz_poly_eval_diff_ui(derivative, g, root);
        if (!mpz_divisible_ui_p(derivative, p)) {
            result.first += 1.0 / denominator;
            result.second +=
                2.0 * valuation / denominator +
                static_cast<double>(p + 1) / (denominator * denominator);
            continue;
        }

        moments const child = affine_moments(
            translate_scale(g, root, p), p, valuation, random_state);
        result.first += child.first / p;
        result.second +=
            (child.second - static_cast<double>(valuation) * valuation) / p;
    }
    return result;
}

moments valuation_moments(cxx_mpz_poly const &f, unsigned long p,
                          gmp_randstate_ptr random_state)
{
    moments const affine = affine_moments(f, p, 0, random_state);
    moments const projective =
        affine_moments(reverse_scale(f, p), p, 0, random_state);
    double const scale = 1.0 / static_cast<double>(p + 1);
    double const first = (p * affine.first + projective.first) * scale;
    double const second = (p * affine.second + projective.second) * scale;
    return {first, second - first * first};
}

struct smooth_part {
    double mean;
    double variance;
    double baseline;
};

smooth_part smooth_part_moments(cxx_mpz_poly const &f, unsigned long bound)
{
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    gmp_randseed_ui(random_state, 1);

    prime_info primes;
    prime_info_init(primes);

    smooth_part result{};
    for (unsigned long p = 2; p <= bound; p = getprime_mt(primes)) {
        moments const value = valuation_moments(f, p, random_state);
        double const log_p = std::log(static_cast<double>(p));
        result.mean += value.first * log_p;
        result.variance += value.second * log_p * log_p;
        result.baseline += log_p / static_cast<double>(p - 1);
    }

    prime_info_clear(primes);
    gmp_randclear(random_state);
    return result;
}

/* modified Bessel function of the first kind; the standard routine rejects
   negative orders, so orders in (-1, 0) use the reflection formula
   I_{-nu}(x) = I_nu(x) + (2/pi) sin(nu pi) K_nu(x) */
double bessel_i(double order, double x)
{
    if (order >= 0.0)
        return std::cyl_bessel_i(order, x);
    assert(order > -1.0);
    assert(x > 0.0);
    double const nu = -order;
    return std::cyl_bessel_i(nu, x) +
           2.0 / pi * std::sin(nu * pi) * std::cyl_bessel_k(nu, x);
}

double noncentral_chi_squared_density(double x, double degrees, double lambda)
{
    double const bessel =
        bessel_i(degrees / 2.0 - 1.0, std::sqrt(lambda * x));
    return 0.5 * std::exp(-(x + lambda) / 2.0) *
           std::pow(x / lambda, degrees / 4.0 - 0.5) * bessel;
}

} // namespace

eprime_result eprime_score(cxx_cado_poly const &poly, double bound_f,
                           double bound_g, double area, unsigned int steps,
                           unsigned long alpha_bound)
{
    assert(poly.nsides() == 2);
    assert(poly[algebraic_side].degree() > 1);
    assert(poly[rational_side].degree() == 1);
    assert(poly.skew > 0.0);
    assert(bound_f > 1.0);
    assert(bound_g > 1.0);
    assert(area > 0.0);
    assert(steps > 0);
    assert(alpha_bound >= 2);

    smooth_part const smooth =
        smooth_part_moments(poly[algebraic_side], alpha_bound);
    smooth_part const rational =
        smooth_part_moments(poly[rational_side], alpha_bound);
    double const rational_alpha = rational.baseline - rational.mean;
    double const lambda = smooth.variance / 2.0 - smooth.mean;
    double const degrees = smooth.mean - lambda;
    assert(lambda > 0.0);
    assert(degrees > 0.0);
    double const height = steps / (6.0 * smooth.baseline);

    polynomial<cxx_mpz> const f(poly[algebraic_side]);
    polynomial<cxx_mpz> const g(poly[rational_side]);
    double const x_scale = std::sqrt(area * poly.skew);
    double const y_scale = std::sqrt(area / poly.skew);
    double const inverse_log_f = 1.0 / std::log(bound_f);
    double const inverse_log_g = 1.0 / std::log(bound_g);

    std::vector<double> smooth_values(steps);
    std::vector<double> densities(steps);
    for (unsigned int i = 0; i < steps; ++i) {
        smooth_values[i] = (static_cast<double>(i) + 0.5) / height;
        densities[i] =
            noncentral_chi_squared_density(smooth_values[i], degrees, lambda);
    }

    double score = 0.0;
    for (unsigned int i = 0; i < steps; ++i) {
        double const theta = pi / steps * (static_cast<double>(i) + 0.5);
        double const x = x_scale * std::cos(theta);
        double const y = y_scale * std::sin(theta);
        double const log_f = std::log(std::abs(f(x, y)));
        double const rational_probability =
            dickman_rho((std::log(std::abs(g(x, y))) + rational_alpha) *
                        inverse_log_g);

        for (unsigned int j = 0; j < steps; ++j) {
            double const correction = smooth.baseline - smooth_values[j];
            score += dickman_rho((log_f + correction) * inverse_log_f) *
                     rational_probability * densities[j];
        }
    }

    return {
        score / (steps * height),
        smooth.mean,
        smooth.variance,
        smooth.baseline - smooth.mean,
        rational_alpha,
    };
}
