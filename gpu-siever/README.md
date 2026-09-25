(The following is primarily LLM-generated and lightly edited.)

# gpu-siever

GPU implementations of: lattice sieving, polynomial selection,
and block Wiedemann. The programs build against
the CADO-NFS tree in the parent directory, which is modified for compatibility.

This is the `rsa260` version.

## Contents

- `src/glas.cpp`, `src/sieve_core_*` — `glas`, a GPU lattice siever
  that replaces the CADO-NFS `las` siever. It accepts the `las`
  command line, reads the same polynomial and factor-base files, and
  writes relations in the same format. The sieve stage runs on a CUDA
  device; norm computation and cofactorization run on the host, with
  a batched ECM on the device taking over when the host falls behind.
- `src/matmul_cuda.cpp`, `src/spmv_gf2.cu` — `libmatmul_b{64,128,256}_cuda.so`,
  the CUDA matrix-vector product backend of the `bwc` solver
  (`mm_impl=cuda`).
- `src/bwccomm_nccl.cpp`, `src/comm_xor.cu` — `libbwccomm_nccl.so`,
  the NCCL vector exchange plugin of `bwc` (`comm_impl=nccl`).
- `src/bwccomm_nvls.cpp`, `src/comm_multimem.cu` — `libbwccomm_nvls.so`,
  the NVLink SHARP exchange plugin (`comm_impl=nvls`; one NVLink
  domain over all ranks, explicit selection, no fallback).
- `src/glingen.cu`, `src/lingenft_cuda.cpp` — `libbwclingenft_cuda.so`,
  the GPU lingen matrix products (`lingen_ft_impl=cuda`, used by the
  `glingen` program of `cado-nfs`).
- `polyselect/gps1*` — `gps1`, a CUDA port of CADO-NFS polynomial
  selection stage 1 with the size optimization on the GPU.
- `src/eprime*` — `eprime`, an E-prime polynomial scorer.
- `params/params.cNNN` — CADO-NFS parameter files for 150 to 344
  digit inputs, tuned for this siever.
- `scripts/` — the launchers that plug the GPU programs into
  `cado-nfs.py`.
- `build.sh`, `Makefile` — the build.

## Requirements

- Linux, GCC 10 or later, CMake 3.18 or later, GMP, hwloc, zlib,
  Python 3.10 with `flask` and `requests` (the CADO-NFS requirements).
- CUDA 12.4 or later with `nvcc`; NCCL; an MPI implementation with
  `mpicxx` (Open MPI is what the linear algebra ran with).
- One or more NVIDIA GPUs of compute capability 9.0 or later.

## Build

    git clone https://github.com/cognition-research/cado-nfs-public cado-nfs
    cd cado-nfs && git checkout latest
    cd gpu-siever
    CUDA_ARCH=100 ./build.sh

`CUDA_ARCH` is the compute capability without the dot (90 for Hopper,
100 for Blackwell). `build.sh` makes three CADO-NFS build
trees under `build/` (a plain one the siever links against, the shared
pipeline build that `cado-nfs.py` runs, and the MPI build of the
linear algebra programs), installs the CUDA `bwc` plugins into the
last two, and builds `glas-gpu`, `gps1` and `eprime` here. `CADO_SRC`
overrides the CADO-NFS tree (default: the parent directory). Set
`NCCL_HOME` if NCCL is not under `/usr`, and `MPICXX` if the MPI
compiler wrapper is not `mpicxx`.

The individual targets are in the `Makefile`; `make glas` builds a
CPU-only siever and `make matmul-cpu-stub` a CPU implementation of
the matmul backend, both for validation on a machine without a GPU.

## Running a factorization

The pipeline is the stock `cado-nfs.py` of the pipeline build. Three
launchers in `scripts/` put the GPU programs in place of the CPU ones,
through the `execpath`/`execbin` parameters of cadofactor:

    BWC_LA_BUILD=$PWD/build/cado-la \
    build/cado/cado-nfs.py <N> --parameters params/params.c<size> \
      tasks.workdir=<wdir> \
      tasks.polyselect.polyselect.execpath=$PWD/scripts \
      tasks.polyselect.polyselect.execbin=polyselect-gps1 \
      tasks.sieve.las.execpath=$PWD/scripts \
      tasks.sieve.las.execbin=las-glas \
      tasks.linalg.bwc.execpath=$PWD/scripts \
      tasks.linalg.bwc.execbin=bwc-gpu \
      tasks.linalg.m=128 tasks.linalg.n=64 \
      slaves.nrclients=0 --server-threads <cores>

Everything else is stock CADO-NFS: the server hands out polynomial
selection and sieving workunits, the clients run the launchers, and
the filtering and square root stages run on the server host.

- `scripts/polyselect-gps1` runs `gps1` for a stage-1 workunit,
  assigning one GPU per workunit by its `ad` range.
- `scripts/las-glas` runs `glas-gpu` for a sieving workunit,
  assigning one GPU per process by its special-q range and giving the
  process the CPUs of that GPU's share of the host. One client per
  GPU keeps every device busy; the server prints its URL and
  certificate fingerprint at start:

        for gpu in $(seq 0 $((ngpu - 1))); do
          GLAS_GPU=$gpu build/cado/cado-nfs-client.py \
            --server=<url> --certsha1=<fingerprint> \
            --basepath=<client dir>/$gpu --clientid=$(hostname).gpu$gpu &
        done

  A client downloads the launcher from the server, so it needs
  `glas-gpu` (and `gps1`) on its `PATH`, or `GLAS_BIN` (`GPS1_BIN`)
  set to the binary. The first process on a host writes the parsed factor
  base to `<fb file>.cache` and every later process reads it.
- `scripts/bwc-gpu` runs `bwc.pl` of the MPI build with the GPU
  configuration: `mm_impl=cuda` on the `SPMV_GF2_NGPU` devices of
  each rank (default: all GPUs of the host), the device-resident
  `gkrylov`, `gmksol` and `glingen` programs, and with more than one
  rank the NCCL exchange (`comm_impl=nccl`). `BWC_MPI=<r>x<c>` and
  `BWC_HOSTS=<host,...>` place one rank per node; `BWC_MPI_EXTRA`
  passes the `mpirun` transport arguments of the site.

The parameter files carry the sieve and filter parameters; the
polynomial selection block of each is the fallback for a plain CADO
run and is normally overridden on the command line
(`tasks.polyselect.P`, `admax`, `adrange`, `threads`). A polynomial
found elsewhere is imported with `tasks.polyselect.import=<poly>`.

## Running the programs directly

    ./glas-gpu -poly c120.poly -fb1 c120.roots1.gz \
      -q0 600000 -q1 601000 -I 12 \
      -lim0 2500000 -lim1 3400000 -lpb0 27 -lpb1 28 \
      -mfb0 52 -mfb1 54 -lambda0 1.92 -lambda1 1.93 \
      -ncurves0 11 -ncurves1 14 -sqside 1 -t 16 \
      -adjust-strategy 0 -gpu -out out.txt

    ./gps1 -N <n> -d 6 -P 10000000 -admin 0 -admax 1000000 -incr 60 -nq 1296 -sopt-gpu 1

    ./eprime POLY Bf Bg AREA STEPS ALPHA_BOUND

    perl build/cado-la/linalg/bwc/bwc.pl :complete matrix=<matrix> wdir=<wdir> \
      m=128 n=64 nullspace=left mpi=1x1 thr=1x1 mm_impl=cuda \
      krylov_prog=gkrylov mksol_prog=gmksol lingen_prog=glingen lingen_thr=8x8

`glas` supports `-adjust-strategy 0` only (a fixed sieve rectangle),
one sieve region size and one bucket level; a task whose buffers
overflow on the device falls back to the CPU path with a note.
`GLAS_GPU_STATS=1` prints per-kernel statistics at exit.
