# Build with shared libraries so that bwc can dlopen out-of-tree
# matmul backends (mm_impl=<name> loads libmatmul_<layer>_<name>.so).
ENABLE_SHARED=1

# The libhwloc x86 backend crashes with SIGFPE on some hypervisors.
# hwloc_aux_topology_load (utils/hwloc-aux.cpp) disables it by
# default. Set HWLOC_COMPONENTS to override.
