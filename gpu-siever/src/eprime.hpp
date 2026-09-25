#pragma once

#include "cado_poly.hpp"

struct eprime_result {
    double score;
    double smooth_part_mean;
    double smooth_part_variance;
    double algebraic_alpha;
    double rational_alpha;
};

eprime_result eprime_score(cxx_cado_poly const &poly, double bound_f,
                           double bound_g, double area, unsigned int steps,
                           unsigned long alpha_bound);
