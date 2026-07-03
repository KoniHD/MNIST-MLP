#
# Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
# License: CC-BY-SA-4.0
#

# --- CPU sequential path (no CUDA, no MPI, no MLX) ---------------------------
CXX       = g++
CXX_FLAGS = -std=c++23 -O3 -Wall -Werror -march=native

# --- Apple MLX path (macOS only; requires: brew install mlx) -----------------
MLXCC = clang++
MLX_PREFIX = $(shell brew --prefix mlx)
MLX_FLAGS = 

# --- GPU / MPI path (requires nvc++ from NVIDIA HPC SDK and an MPI install) --
NVCC_FLAGS     = -x cu -std=c++20 -O3 -rdc=true -arch=sm_61
NVCC_LINK    = -std=c++20 -O3 -rdc=true
NVCC_LDFLAGS = -lcublas
MPICXX = mpicxx
OMPI_CXX_FLAGS = -DOMPI_SKIP_MPICXX -DUSE_MPI

export OMPI_CXX = nvcc

.PHONY: all clean

all: local student_submission tutor_submission reference_submission

apple: sequential_submission mlx_submission

# CPU-only
sequential_submission: sequential_submission.cpp Utility.hpp
	$(CXX) $(CXX_FLAGS) -o $@ $<

# Apple MLX backend (macOS only)
mlx_submission: mlx_submission.cpp Utility.hpp
	$(MLXCC) $(CXX_FLAGS) -DMLX_BACKEND -o $@ $< \
		-I$(MLX_PREFIX)/include \
		-L$(MLX_PREFIX)/lib \
		-lmlx

# GPU/CUDA backend
Utility.o: Utility.cpp Utility.hpp
	$(MPICXX) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" $(OMPI_CXX_FLAGS) -c -o $@ $<

student_submission: student_submission.cpp Utility.o
	$(MPICXX) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" $(OMPI_CXX_FLAGS) -c -o student_submission.o $<
	$(MPICXX) $(NVCC_LINK) $(OMPI_CXX_FLAGS) -o $@ student_submission.o Utility.o $(NVCC_LDFLAGS)

reference_submission: reference_submission.cpp Utility.o
	$(MPICXX) $(NVCC_FLAGS) $(OMPI_CXX_FLAGS) -c -o reference_submission.o $<
	$(MPICXX) $(NVCC_LINK) $(OMPI_CXX_FLAGS) -o $@ reference_submission.o Utility.o $(NVCC_LDFLAGS)

tutor_submission: tutor_submission.cpp Utility.o
	$(MPICXX) $(NVCC_FLAGS) $(OMPI_CXX_FLAGS) -c -o tutor_submission.o $<
	$(MPICXX) $(NVCC_LINK) $(OMPI_CXX_FLAGS) -o $@ tutor_submission.o Utility.o $(NVCC_LDFLAGS)

clean:
	rm -f sequential_submission mlx_submission student_submission reference_submission tutor_submission Utility.o

remove_reference:
	rm -f *reference* *tutor*
