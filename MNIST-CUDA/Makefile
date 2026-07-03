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
NVCC          = nvc++
NVCC_FLAGS    = -cuda -cudalib=cublas
MPICXX        = mpicxx
OMPI_CXX      = nvc++
OMPI_CXX_FLAGS = -DOMPI_SKIP_MPICXX

.PHONY: all apple clean

all: local student_submission tutor_submission

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
	$(MPICXX) $(CXX_FLAGS) $(NVCC_FLAGS) $(OMPI_CXX_FLAGS) -o $@ $<

student_submission: student_submission.cpp Utility.o
	$(MPICXX) $(CXX_FLAGS) $(NVCC_FLAGS) $(OMPI_CXX_FLAGS) -o $@ $^

tutor_submission: tutor_submission.cpp Utility.o
	$(MPICXX) $(CXX_FLAGS) $(NVCC_FLAGS) $(OMPI_CXX_FLAGS) -o $@ $^

clean:
	rm -f sequential_submission mlx_submission student_submission tutor_submission scratch scratch_sequential Utility.o compute_graph.dot

remove_reference:
	rm -f *reference* *tutor*
