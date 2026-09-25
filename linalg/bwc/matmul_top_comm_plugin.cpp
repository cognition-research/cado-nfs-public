#include "cado.h" // IWYU pragma: keep

#include <mutex>
#include <string>

#include "bwc_config.h" // BUILD_DYNAMICALLY_LINKABLE_BWC // IWYU pragma: keep

#ifdef BUILD_DYNAMICALLY_LINKABLE_BWC
#include <dlfcn.h>
#include "solib-naming.h" // IWYU pragma: keep
#endif

#include "fmt/base.h"

#include "macros.h"
#include "bwc_device_arena.hpp"
#include "matmul_top_comm_plugin.hpp"
#include "params.hpp"

bwc_comm_plugin_api const * bwc_comm_plugin = nullptr;

/* With one MPI job, every communicator the two legs use has one rank,
 * so both collectives are identities. This builtin plugin replaces
 * them with no-ops. It also accepts device arenas, so a device
 * backend keeps the vectors resident on the device instead of a full
 * round trip through the host on every iteration. */
static void one_job_reduce_scatter_xor(void * const *, int, size_t, MPI_Comm)
{
}

static void one_job_allgather(void * const *, int, size_t, MPI_Comm)
{
}

static bwc_comm_plugin_api const one_job_api = {
    one_job_reduce_scatter_xor, one_job_allgather, 1
};

void matmul_top_comm_plugin_decl_usage(cxx_param_list & pl)
{
    pl.declare_usage("comm_impl",
            "load a communication plugin for the vector reduce and broadcast steps");
}

void matmul_top_comm_plugin_lookup_parameters(cxx_param_list & pl)
{
    pl.lookup("comm_impl");
}

void matmul_top_comm_plugin_init(cxx_param_list & pl)
{
    static std::mutex pp;
    std::lock_guard<std::mutex> const dummy(pp);
    if (bwc_comm_plugin) return;
    const char * impl = pl.lookup_old("comm_impl");
    if (!impl) {
        int njobs;
        MPI_Comm_size(MPI_COMM_WORLD, &njobs);
        if (njobs > 1)
            return;
        fmt::print(stderr, "comm plugin builtin: one MPI job,"
                " identity collectives\n");
        bwc_comm_plugin = &one_job_api;
        bwc_dev_allow();
        return;
    }
#ifdef BUILD_DYNAMICALLY_LINKABLE_BWC
    std::string const solib = fmt::format(
            SOLIB_PREFIX "bwccomm_{}" SOLIB_SUFFIX, impl);
    /* the solib stays loaded until the process exits */
    void * handle = dlopen(solib.c_str(), RTLD_NOW);
    if (handle == nullptr) {
        fmt::print(stderr, "loading {}: {}\n", solib, dlerror());
        abort();
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    auto api_getter = (bwc_comm_plugin_api const * (*)())
        dlsym(handle, "bwc_comm_solib_reach_api");
    if (api_getter == nullptr) {
        fmt::print(stderr, "loading {}: {}\n", solib, dlerror());
        abort();
    }
    bwc_comm_plugin = (*api_getter)();
    ASSERT_ALWAYS(bwc_comm_plugin != nullptr);
    if (bwc_comm_plugin->handles_device_arenas)
        bwc_dev_allow();
#else
    fmt::print(stderr, "comm_impl={} needs a build with ENABLE_SHARED=1\n", impl);
    abort();
#endif
}
