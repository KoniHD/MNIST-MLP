# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Two-track MNIST classifier assignment:
- `MNIST-PyTorch/` — reference PyTorch 2-layer MLP (784→256→10), trained on `ylecun/mnist` via Hugging Face Datasets.
- `MNIST-CUDA/` — raw C++ port of the same model, implemented across five backends: standalone CPU (`sequential_submission.cpp`), Apple MLX (`mlx_submission.cpp`), instructor CUDA kernels (`Utility.cpp`), CUDA+MPI with full-weight broadcast (`student_submission.cpp`), and CUDA+MPI with seed-only broadcast (`fast_submission.cpp`). All backends share the `Module` interface declared in `Utility.hpp`.

`MNIST-CUDA/README.md` is the single source of truth for the architecture and design rationale (kernel split, DDP `Iallreduce`/`Waitany` overlap scheme, seed-broadcast optimisation, safetensors format, student-vs-instructor member split).

## Commands

### MNIST-PyTorch (Python)

All Python work goes through `uv`. Never suggest `pip install`.

```bash
cd MNIST-PyTorch
uv sync                                                                    # install deps (Python >= 3.14)
uv run mnist-pytorch train --epochs 10 -bs 128 -lr 0.1 -p models/model_weights --plot
uv run mnist-pytorch evaluate -bs 128 -p models/model_weights.safetensors --plot
uv add <pkg>                                                               # add a runtime dep
uv add --dev <pkg>                                                         # add a dev dep
```

`train` writes `<path>.safetensors`; `evaluate` expects the full `.safetensors` path.

### MNIST-CUDA (C++)

```bash
cd MNIST-CUDA
make local                           # builds sequential_submission + mlx_submission (macOS, no GPU)
make all                             # builds everything (requires nvc++ + MPI + CUDA GPU)
make clean

# Run — seed is read from stdin (course harness protocol: READY / seed / accuracy / DONE)
echo 42 | ./sequential_submission
echo 42 | ./mlx_submission
echo 42 | mpirun -np 4 ./student_submission
echo 42 | mpirun -np 4 ./fast_submission

# Inference only — load saved weights, skip training
./sequential_submission model.safetensors
mpirun -np 4 ./student_submission model.safetensors
```

## Conventions

- Test accuracy target: **≥90%** for both the PyTorch and CUDA implementations. Surface anything below this.
- Do not edit `MNIST-PyTorch/notebooks/*.ipynb` — it is reference material.
- Device is auto-selected in `MNIST-PyTorch/src/mnist_pytorch/train.py` (CUDA → MPS → CPU). When debugging device issues, check that path first.
- First PyTorch train run pulls MNIST from HF Hub — requires internet.
- Format C/C++ files with **clang-format**; the config lives at the ParProg-Tut repo root and is auto-discovered by editors and the formatter. Do NOT run clang-tidy on this project — it is intentionally excluded.
