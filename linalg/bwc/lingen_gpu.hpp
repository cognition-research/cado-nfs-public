#ifndef CADO_LINGEN_GPU_HPP
#define CADO_LINGEN_GPU_HPP

#include <cstddef>

#include <utility>

#include "lingen_checkpoints.hpp"
#include "lingen_ft_plugin.hpp"
#include "lingen_matpoly_select.hpp"

template<bool is_binary> struct bmstatus;

/* A polynomial matrix in the memory of the transform plugin (the GPU),
 * with the operations of matpoly<true> that the lingen recursion uses,
 * so that bw_lingen_recursive runs on it unchanged. The entries above
 * get_size() are zero. */
class dmatpoly {
    bwc_lingen_ft_polys x { nullptr, 0, 0, 0 };
    unsigned int m = 0;
    dmatpoly(bwc_lingen_ft_polys x, unsigned int m) : x(x), m(m) {}
    void clear_high() const;
public:
    static constexpr bool is_binary = true;
    dmatpoly() = default;
    dmatpoly(unsigned int m, unsigned int n, size_t size);
    explicit dmatpoly(matpoly<true> const & a);
    dmatpoly(dmatpoly && o) noexcept { *this = std::move(o); }
    dmatpoly & operator=(dmatpoly && o) noexcept;
    dmatpoly(dmatpoly const &) = delete;
    dmatpoly & operator=(dmatpoly const &) = delete;
    ~dmatpoly() { clear(); }

    matpoly<true> download(matpoly<true>::arith_hard * ab) const;
    bwc_lingen_ft_polys const & polys() const { return x; }
    unsigned int nrows() const { return m; }
    unsigned int ncols() const { return x.ld; }
    size_t get_size() const { return x.bits; }
    /* s <= get_size(); the coefficients from s are cleared */
    void set_size(size_t s);
    void clear_high_word() const {}
    int coeff_is_zero(size_t k) const;
    dmatpoly similar_shell() const { return { { nullptr, ncols(), 0, 0 }, m }; }
    void clear();
    /* this mod x^size is returned, this becomes this / x^k */
    dmatpoly truncate_and_rshift(size_t size, size_t k);
    void rshift(dmatpoly const & src, size_t k);
    static dmatpoly mp(dmatpoly const & E, dmatpoly const & pi);
    static dmatpoly mul(dmatpoly const & a, dmatpoly const & b);

    friend dmatpoly glingen_basecase_raw(bmstatus<true> & bm, dmatpoly const & E);
};

/* bw_lingen_basecase with E and pi in plugin memory; E is cleared */
dmatpoly glingen_basecase(bmstatus<true> & bm, dmatpoly & E);

int load_checkpoint_file(bmstatus<true> & bm, cp_which which, dmatpoly & X, unsigned int t0, unsigned int t1);
int save_checkpoint_file(bmstatus<true> & bm, cp_which which, dmatpoly const & X, unsigned int t0, unsigned int t1);

#endif  /* CADO_LINGEN_GPU_HPP */
