#include "cado.h" // IWYU pragma: keep

#include <cstdlib>
#include <string>

#include "bwc_config.h" // BUILD_DYNAMICALLY_LINKABLE_BWC // IWYU pragma: keep

#ifdef BUILD_DYNAMICALLY_LINKABLE_BWC
#include <dlfcn.h>
#include "solib-naming.h" // IWYU pragma: keep
#endif

#include "fmt/base.h"

#include "macros.h"
#include "lingen_ft_plugin.hpp"
#include "params.hpp"

bwc_lingen_ft_plugin_api const * bwc_lingen_ft_plugin = nullptr;

void lingen_ft_plugin_decl_usage(cxx_param_list & pl)
{
    pl.declare_usage("lingen_ft_impl",
            "load a plugin for the products of Cantor-transformed matrices");
}

void lingen_ft_plugin_init(cxx_param_list & pl)
{
    const char * impl = pl.lookup_old("lingen_ft_impl");
    if (!impl) return;
#ifdef BUILD_DYNAMICALLY_LINKABLE_BWC
    std::string const solib = fmt::format(
            SOLIB_PREFIX "bwclingenft_{}" SOLIB_SUFFIX, impl);
    /* the solib stays loaded until the process exits */
    void * handle = dlopen(solib.c_str(), RTLD_NOW);
    if (handle == nullptr) {
        fmt::print(stderr, "loading {}: {}\n", solib, dlerror());
        abort();
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    auto api_getter = (bwc_lingen_ft_plugin_api const * (*)())
        dlsym(handle, "bwc_lingen_ft_solib_reach_api");
    if (api_getter == nullptr) {
        fmt::print(stderr, "loading {}: {}\n", solib, dlerror());
        abort();
    }
    bwc_lingen_ft_plugin = (*api_getter)();
    ASSERT_ALWAYS(bwc_lingen_ft_plugin != nullptr);
#else
    fmt::print(stderr, "lingen_ft_impl={} needs a build with ENABLE_SHARED=1\n", impl);
    abort();
#endif
}
