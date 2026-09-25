#!/usr/bin/env bash
# Build cado-nfs and the GPU programs of this tree.
#
#   CUDA_ARCH=<sm> ./build.sh
#
# CADO_SRC is the cado-nfs tree, by default the parent directory (see
# README.md). CUDA_ARCH is the compute capability without the
# dot (90 for Hopper, 100 for Blackwell). Three cado-nfs
# build trees come out of it, under BUILD_ROOT (default ./build):
#
#   cado-sieve   the plain build glas and gps1 link against
#   cado         the pipeline build that cado-nfs.py runs: shared
#                libraries (bwc loads the CUDA matmul backend with
#                dlopen) and 8-byte renumber entries, for large prime
#                bounds past 32 bits and renumber tables past 2^32
#                entries
#   cado-la      the MPI build of the linear algebra programs, with the
#                CUDA matmul backends, the NCCL exchange plugin and the
#                GPU lingen plugin installed
#
# and, in this directory: glas-gpu, gps1, eprime.
set -Eeuo pipefail
cd "$(dirname "$0")"

CADO_SRC=$(cd "${CADO_SRC:-..}" && pwd)
CUDA_ARCH=${CUDA_ARCH:-100}
BUILD_ROOT=${BUILD_ROOT:-$PWD/build}
JOBS=${JOBS:-$(nproc)}
CXXFLAGS=${CXXFLAGS:-"-O3 -std=c++20 -march=native -g -Wall -Wextra"}

CADO_SIEVE_BUILD=$BUILD_ROOT/cado-sieve
CADO_BUILD=$BUILD_ROOT/cado
CADO_LA_BUILD=$BUILD_ROOT/cado-la

force_build_tree=$CADO_SIEVE_BUILD \
  make -C "$CADO_SRC" -j "$JOBS"

FLAGS_SIZE="-DSIZEOF_P_R_VALUES=8 -DSIZEOF_INDEX=8" ENABLE_SHARED=1 \
  force_build_tree=$CADO_BUILD \
  make -C "$CADO_SRC" -j "$JOBS"

NO_PYTHON_CHECK=1 ENABLE_SHARED=1 MPI=1 \
  force_build_tree=$CADO_LA_BUILD \
  make -C "$CADO_SRC" -j "$JOBS" bwc_full_gf2

# The plugins link against the tree that loads them, so each set builds
# from a clean state.
make clean
make matmul-cuda libbwccomm_nccl.so libbwccomm_nvls.so libbwclingenft_cuda.so \
  CADO_SRC="$CADO_SRC" CADO_BUILD="$CADO_LA_BUILD" \
  CUDA_ARCH="$CUDA_ARCH" CXXFLAGS="$CXXFLAGS"
make install-matmul install-comm install-lingenft CADO_BUILD="$CADO_LA_BUILD"

make clean
make matmul-cuda \
  CADO_SRC="$CADO_SRC" CADO_BUILD="$CADO_BUILD" \
  CUDA_ARCH="$CUDA_ARCH" CXXFLAGS="$CXXFLAGS"
make install-matmul CADO_BUILD="$CADO_BUILD"

make clean
make glas-gpu gps1 eprime \
  CADO_SRC="$CADO_SRC" CADO_BUILD="$CADO_SIEVE_BUILD" \
  CUDA_ARCH="$CUDA_ARCH" CXXFLAGS="$CXXFLAGS"

cat <<EOF
built:
  glas-gpu gps1 eprime           in $PWD
  cado-nfs.py                    $CADO_BUILD/cado-nfs.py
  bwc.pl (MPI, GPU plugins)      $CADO_LA_BUILD/linalg/bwc/bwc.pl
EOF
