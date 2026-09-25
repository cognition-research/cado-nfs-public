#ifndef CADO_LINGEN_FFT_SELECT_HPP
#define CADO_LINGEN_FFT_SELECT_HPP

#ifdef LINGEN_BINARY
#include "gf2x-fft.h"   // IWYU pragma: export
#include "gf2x-fake-fft.h"      // IWYU pragma: export
#include "gf2x-cantor-fft.h"    // IWYU pragma: export
#include "gf2x-ternary-fft.h"   // IWYU pragma: export
#else
#include "flint-fft/transform_interface.h"      // IWYU pragma: export
#endif

template<typename T> struct is_binary_fft;
#ifdef LINGEN_BINARY
template<> struct is_binary_fft<gf2x_fake_fft_info> {
    static constexpr const bool value = true;
};
template<> struct is_binary_fft<gf2x_cantor_fft_info> {
    static constexpr const bool value = true;
};
template<> struct is_binary_fft<gf2x_ternary_fft_info> {
    static constexpr const bool value = true;
};
#else
template<> struct is_binary_fft<fft_transform_info> {
    static constexpr const bool value = false;
};
#endif
template<typename T>
inline constexpr bool is_binary_fft_v = is_binary_fft<T>::value;

/* the transforms a lingen_ft_impl plugin (lingen_ft_plugin.hpp) can hold */
template<typename T> struct has_lingen_ft_plugin {
    static constexpr const bool value = false;
};
#ifdef LINGEN_BINARY
template<> struct has_lingen_ft_plugin<gf2x_cantor_fft_info> {
    static constexpr const bool value = true;
};
#endif
template<typename T>
inline constexpr bool has_lingen_ft_plugin_v = has_lingen_ft_plugin<T>::value;


#endif	/* LINGEN_FFT_SELECT_HPP_ */
