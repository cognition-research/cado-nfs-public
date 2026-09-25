# Build glas against an existing CADO-NFS build tree (libutils + facul).
# CADO_SRC is the cado-nfs tree (the parent directory) and CADO_BUILD
# the cado-nfs build tree to link against (see build.sh).
# `make glas` is the CPU-only build; `make glas-gpu` needs nvcc.

CADO_SRC ?= $(abspath ..)
CADO_BUILD ?= $(CURDIR)/build/cado-sieve

CXX ?= g++
NVCC ?= nvcc
CUDA_ARCH ?= 90
CXXFLAGS ?= -O3 -std=c++20 -march=native -g -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) --extended-lambda
INCLUDES = -I$(CADO_SRC) -I$(CADO_SRC)/utils -I$(CADO_SRC)/utils/embedded \
           -I$(CADO_SRC)/sieve -I$(CADO_BUILD)

FACUL_OBJS = $(wildcard $(CADO_BUILD)/sieve/ecm/CMakeFiles/facul.dir/*.o)
ifneq ($(wildcard $(CADO_BUILD)/utils/libutils.a),)
LIBUTILS = $(CADO_BUILD)/utils/libutils.a
else
LIBUTILS = -L$(CADO_BUILD)/utils -lutils -Wl,-rpath,$(CADO_BUILD)/utils
endif
LIBS = $(LIBUTILS) -lgmp -lpthread -fopenmp

SRC = src/glas.cpp src/sieve_core_cpu.cpp
EPRIME_SRC = src/eprime.cpp

glas: $(SRC) src/sieve_core_gpu_stub.cpp src/sieve_core.hpp src/cofac_strategy.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC) src/sieve_core_gpu_stub.cpp \
	    $(FACUL_OBJS) $(LIBS)

eprime: src/eprime_main.cpp $(EPRIME_SRC) src/eprime.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ src/eprime_main.cpp $(EPRIME_SRC) \
	    $(LIBS)

sieve_core_gpu.o: src/sieve_core_gpu.cu src/sieve_core.hpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ src/sieve_core_gpu.cu

glas-gpu: $(SRC) sieve_core_gpu.o src/sieve_core.hpp src/cofac_strategy.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC) sieve_core_gpu.o \
	    $(FACUL_OBJS) $(LIBS) -lcudart -L/usr/local/cuda/lib64

# CUDA backend for the bwc linear algebra step (mm_impl=cuda). It needs
# a CADO build with ENABLE_SHARED=1, because bwc loads matmul backends
# with dlopen. One library per GF(2) arithmetic layer: b64, b128, b256.
# `make matmul-cuda` builds all three; `make install-matmul` puts them
# where bwc finds them. `make matmul-cpu-stub` builds the same
# libraries with a CPU implementation of the device API, for validation
# without a GPU.
BWC_WIDTHS = 64 128 256
BWC_DEFS = -DARITH_MOD2 -DMM_IMPL=cuda
BWC_INCLUDES = $(INCLUDES) -I$(CADO_SRC)/linalg/bwc -I$(CADO_BUILD)/linalg/bwc
BWC_LINK = src/matmul_cuda.cpp $(CADO_SRC)/linalg/bwc/matmul_facade.cpp \
	    -L$(CADO_BUILD)/linalg/bwc -lmatmul_common \
	    -L$(CADO_BUILD)/utils -lutils \
	    -Wl,-rpath,$(CADO_BUILD)/linalg/bwc \
	    -Wl,-rpath,$(CADO_BUILD)/utils

spmv_gf2.o: src/spmv_gf2.cu src/spmv_gf2.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -I$(CADO_SRC)/linalg/bwc \
	    -c -o $@ src/spmv_gf2.cu

libmatmul_b%_cuda.so: src/matmul_cuda.cpp spmv_gf2.o src/spmv_gf2.h
	$(CXX) $(CXXFLAGS) -fPIC -shared $(BWC_DEFS) \
	    -DARITH_LAYER_b$* -DARITH_LAYER=b$* -DARITH_SIMD_GROUPSIZE=$* \
	    $(BWC_INCLUDES) -o $@ spmv_gf2.o $(BWC_LINK) \
	    -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64

matmul-cuda: $(BWC_WIDTHS:%=libmatmul_b%_cuda.so)

matmul-cpu-stub: src/matmul_cuda.cpp src/spmv_gf2_cpu.cpp src/spmv_gf2.h
	for g in $(BWC_WIDTHS); do \
	    $(CXX) $(CXXFLAGS) -fPIC -shared $(BWC_DEFS) \
	        -DARITH_LAYER_b$$g -DARITH_LAYER=b$$g \
	        -DARITH_SIMD_GROUPSIZE=$$g $(BWC_INCLUDES) \
	        -o libmatmul_b$${g}_cuda.so src/spmv_gf2_cpu.cpp $(BWC_LINK) \
	        || exit 1; \
	done

install-matmul:
	cp libmatmul_b*_cuda.so $(CADO_BUILD)/linalg/bwc/

# CUDA port of CADO-NFS polyselect stage 1.
gps1_kernels.o: polyselect/gps1_kernels.cu polyselect/gps1.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ polyselect/gps1_kernels.cu

# -fmad=false and -ffp-contract=off: the size optimization must not
# contract mul+add to fma on the device or the host, so that its doubles
# match the CPU reference sequence of operations.
gps1_sopt.o: polyselect/gps1_sopt.cu polyselect/gps1_sopt.h
	$(NVCC) $(NVCCFLAGS) -fmad=false -Xcompiler -ffp-contract=off \
	    -c -o $@ polyselect/gps1_sopt.cu

gps1: polyselect/gps1.cpp gps1_kernels.o gps1_sopt.o polyselect/gps1.h \
	    polyselect/gps1_sopt.h
	$(CXX) $(CXXFLAGS) -fopenmp $(INCLUDES) -I$(CADO_SRC)/polyselect \
	    -I/usr/local/cuda/include -o $@ polyselect/gps1.cpp gps1_kernels.o \
	    gps1_sopt.o \
	    -L$(CADO_BUILD)/polyselect -lpolyselect_common \
	    -L$(CADO_BUILD)/utils -lutils -lgmp -lpthread \
	    -lcudart -L/usr/local/cuda/lib64 \
	    -Wl,-rpath,$(CADO_BUILD)/polyselect -Wl,-rpath,$(CADO_BUILD)/utils

# CPU-only check of the GPU size optimization: the same code runs on the
# host and the CADO reference checks it.
gps1-sopt-test: polyselect/gps1_sopt_test.cpp gps1_sopt.o polyselect/gps1_sopt.h
	$(CXX) $(CXXFLAGS) -fopenmp $(INCLUDES) -I$(CADO_SRC)/polyselect \
	    -o $@ polyselect/gps1_sopt_test.cpp gps1_sopt.o \
	    -L$(CADO_BUILD)/polyselect -lpolyselect_common \
	    -L$(CADO_BUILD)/utils -lutils -lgmp -lpthread \
	    -lcudart -L/usr/local/cuda/lib64 \
	    -Wl,-rpath,$(CADO_BUILD)/polyselect -Wl,-rpath,$(CADO_BUILD)/utils
	./gps1-sopt-test

# NCCL communication plugin for the bwc vector exchange steps
# (comm_impl=nccl). It needs the comm plugin hook in the CADO tree
# (linalg/bwc/matmul_top_comm_plugin.hpp) and an MPI compiler wrapper.
MPICXX ?= mpicxx
NCCL_HOME ?= /usr

comm_xor.o: src/comm_xor.cu src/comm_xor.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ src/comm_xor.cu

libbwccomm_nccl.so: src/bwccomm_nccl.cpp comm_xor.o src/comm_xor.h
	$(MPICXX) $(CXXFLAGS) -DWITH_MPI -fPIC -shared $(BWC_INCLUDES) \
	    -I$(NCCL_HOME)/include -I/usr/local/cuda/include -o $@ \
	    src/bwccomm_nccl.cpp comm_xor.o \
	    -L$(NCCL_HOME)/lib -lnccl -L/usr/local/cuda/lib64 -lcudart \
	    -Wl,-rpath,/usr/local/cuda/lib64

# NVLink SHARP communication plugin (comm_impl=nvls): the multimem
# instructions reduce and broadcast in the switch, one kernel per
# exchange leg instead of a ring. Explicit selection only; it needs
# multicast and fabric handle support on every device and does not
# fall back to the ring (src/bwccomm_nvls.cpp).
comm_multimem.o: src/comm_multimem.cu src/comm_multimem.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ src/comm_multimem.cu

libbwccomm_nvls.so: src/bwccomm_nvls.cpp comm_multimem.o src/comm_multimem.h
	$(MPICXX) $(CXXFLAGS) -DWITH_MPI -fPIC -shared $(BWC_INCLUDES) \
	    -I/usr/local/cuda/include -o $@ \
	    src/bwccomm_nvls.cpp comm_multimem.o \
	    -L/usr/local/cuda/lib64 -lcudart -lcuda \
	    -Wl,-rpath,/usr/local/cuda/lib64

install-comm: libbwccomm_nccl.so libbwccomm_nvls.so
	cp libbwccomm_nccl.so libbwccomm_nvls.so $(CADO_BUILD)/linalg/bwc/

# glingen: the lingen matrix products of Cantor transforms on the GPU
# (dft, product, ift).
glingen.o: src/glingen.cu src/glingen.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -Xcompiler -fopenmp -c -o $@ src/glingen.cu

# The lingen plugin (lingen_ft_impl=cuda): the hook in the CADO tree
# (linalg/bwc/lingen_ft_plugin.hpp) loads it with dlopen, like the
# matmul backends. `make install-lingenft` puts it where lingen finds
# it.
libbwclingenft_cuda.so: src/lingenft_cuda.cpp glingen.o src/glingen.h
	$(CXX) $(CXXFLAGS) -fPIC -shared -fopenmp $(BWC_INCLUDES) -o $@ \
	    src/lingenft_cuda.cpp glingen.o \
	    -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64

install-lingenft: libbwclingenft_cuda.so
	cp libbwclingenft_cuda.so $(CADO_BUILD)/linalg/bwc/

clean:
	rm -f glas glas-gpu gps1 gps1_kernels.o gps1_sopt.o \
	    gps1-sopt-test \
	    sieve_core_gpu.o \
	    spmv_gf2.o libmatmul_b*_cuda.so comm_xor.o libbwccomm_nccl.so \
	    comm_multimem.o libbwccomm_nvls.so \
	    glingen.o libbwclingenft_cuda.so eprime

.PHONY: clean install-matmul matmul-cuda matmul-cpu-stub install-comm install-lingenft
