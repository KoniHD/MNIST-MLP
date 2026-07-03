#
# Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
# License: CC-BY-SA-4.0
#

# --- CPU sequential path (no CUDA, no MPI, no MLX) ---------------------------
CXX       = g++
CXX_FLAGS = -std=c++23 -O3 -Wall -Werror -march=native

# --- Apple MLX path (macOS only; requires: brew install mlx) -----------------
MLXCC       = clang++
MLX_PREFIX  = $(shell brew --prefix mlx)

# --- GPU / MPI path (nvcc drives, mpicxx as host compiler) -------------------
NVCC       = nvcc
CUDA_FLAGS := -std=c++20 -O3 -arch=sm_75 -ccbin mpicxx -x cu -rdc=true
CUDA_LINK  := -std=c++20 -O3 -arch=sm_75 -ccbin mpicxx -rdc=true
CUDA_LIBS  := -lcublas -lcudart
DEFINES    := -DOMPI_SKIP_MPICXX -DUSE_MPI

.PHONY: all apple clean remove_reference

all: student_submission tutor_submission reference_submission

apple: sequential_submission mlx_submission

# CPU-only (pure g++, c++23)
sequential_submission: sequential_submission.cpp Utility.hpp
	$(CXX) $(CXX_FLAGS) -o $@ $

# Apple MLX backend (macOS only)
mlx_submission: mlx_submission.cpp Utility.hpp
	$(MLXCC) $(CXX_FLAGS) -DMLX_BACKEND -o $@ $< \
		-I$(MLX_PREFIX)/include -L$(MLX_PREFIX)/lib -lmlx

# Shared CUDA object
Utility.o: Utility.cpp Utility.hpp
	$(NVCC) $(CUDA_FLAGS) $(DEFINES) -c $< -o $@

student_submission: student_submission.cpp Utility.o
	$(NVCC) $(CUDA_FLAGS) $(DEFINES) -c $< -o student_submission.o
	$(NVCC) $(CUDA_LINK) -o $@ student_submission.o Utility.o $(CUDA_LIBS)

reference_submission: reference_submission.cpp Utility.o
	$(NVCC) $(CUDA_FLAGS) $(DEFINES) -c $< -o reference_submission.o
	$(NVCC) $(CUDA_LINK) -o $@ reference_submission.o Utility.o $(CUDA_LIBS)

tutor_submission: tutor_submission.cpp Utility.o
	$(NVCC) $(CUDA_FLAGS) $(DEFINES) -c $< -o tutor_submission.o
	$(NVCC) $(CUDA_LINK) -o $@ tutor_submission.o Utility.o $(CUDA_LIBS)

clean:
	rm -f sequential_submission mlx_submission student_submission \
		reference_submission tutor_submission Utility.o \
		student_submission.o reference_submission.o tutor_submission.o \
		compute_graph.dot

remove_reference:
	rm -f *reference* *tutor*