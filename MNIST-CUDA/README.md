# In-class exercise: MNIST Classifier (CUDA + MPI)

This program trains and evaluates the a 2-layer MLP (784 → 256 → 10 neurons, ReLU activation, softmax + cross-entropy output, SGD with lr = 0.1, batch size 128, 5
epochs) on the MNIST-Dataset[^1].

# Your task

Fill the code at the marked `// TODO(Students)` comments.

## How to build

```bash
# macOS developer machine (no GPU required)
make apple          # builds sequential_submission + mlx_submission

# Full build (requires NVIDIA HPC SDK + MPI + CUDA GPU)
make all            # builds local + Utility.o + student_submission + fast_submission

# Individual targets
make sequential_submission
make mlx_submission          # needs: brew install mlx
make Utility.o
make student_submission
make fast_submission

make clean
```

## How to run

### Default course specific

For instance the `sequential_submission` can be run as follows:

```bash
# Sequential (CPU) — any C++23 g++; no GPU or MPI
echo 42 | ./sequential_submission
```

### Perform model inference

Using an additional input you can run a trained model only with inference if you pass the trained weights as `mode.safetensors` file.

If you are curious you can test the effect of model training by running the program in inference mode with the provided `model.safetensors` file which contains **untrained** weights.

*Note:* `*.safetensors` is the prefered model weight format for HuggingFace if you are interested.

```bash
echo 42 | ./sequential_submission model.safetensors
```

If you want to continue improving the model using PyTorch you can simply load it like this using the above model definition.

```python
from safetensors.torch import load_model

model = Classifier()
load_model(model, model.safetensors)
```

## Implementation details

If you are interested the implementation attempts to implement the following `Python` + `PyTorch` model:

```python
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out
```

The model is re-implemented across different backend translation units so
that you can study how data moves between processes and between host and device:

| File | What it is |
|--------------|------------|
| `sequential_submission.cpp` | Standalone CPU implementation built with `g++`. |
| `mlx_submission.cpp` | Standalone Apple MLX backend (unified memory, no host/device copies, high-level API). Builds on macOS with `clang++` + libmlx. |
| `student_submission.cpp` | CUDA + MPI data-parallel training, **whole-model broadcast** baseline. Links `Utility.o`. |

---

## The `Module` class

`Utility.hpp` declares one `Module` class whose public interface is shaped after
`torch.nn.Module`, so the C++ and Python implementations read in parallel. Each backend supplies its own member bodies in
its own translation unit, so exactly one definition of each member is linked per
binary.

```cpp
Module model(seed);              // seeded ctor: deterministic LeCun-style uniform init
model.to("cuda:0");              // mirror params/grads/activations onto a backend
model(X, N);                     // sugar for model.forward(X, N)  (caches activations)
model.backward(target_one_hot);  // softmax+CE -> fc2_backward -> fc1_backward
model.step(0.1f);                // SGD: theta <- theta - lr * grad
model.pull_metrics();            // finish the metric all-reduce (CUDA); no-op elsewhere
float l = model.loss();          // mean CE of the last batch  (softmax+CE byproduct)
int   c = model.num_correct();   // correct predictions in the last batch (byproduct)
```

### Construction & the seed-broadcast optimisation

Two constructors with distinct intended callers:

- **`Module()`** — sizes the buffers but leaves the parameters uninitialised.
  Used by every rank **except** rank 0 in the whole-model-broadcast baseline,
  which then receives the weights over MPI.
- **`explicit Module(uint32_t seed)`** — fan-in-scaled uniform init
  (`±1/√fan_in`; fc1 fan_in = `INPUT_DIM`, fc2 fan_in = `HIDDEN_DIM`) driven by a
  private `std::mt19937` seeded from the argument. Defined `inline` in the header
  (host-only), so all backends share one definition.

Because the seeded ctor is **deterministic**, `fast_submission` broadcasts only
the 4-byte seed and has *every* rank call `Module(seed)` to reconstruct
bit-identical weights — trading the ~800 KB weight broadcast for a 4-byte one.


## Kernel / implementation design (CUDA backend)

### The five kernels

Two are templated (resolved at compile time via `if constexpr`, zero runtime
cost):

1. **`kernel_linear_forward<bool FuseReLU>`** — adds the bias after the GEMM,
   caches the pre-activation, and writes the activation. Layer 1: `<true>` (fused
   ReLU). Layer 2: `<false>`.
2. **`kernel_softmax_ce`** — one block per sample, one thread per class. Two
   shared-memory **tree reductions**: reduction 1 finds the shift `a = max_k z`
   (plus the argmax, tie-broken to the lowest index to match the host); reduction
   2 finds `denom = Σ exp(z−a)`. Writes the fused gradient
   `g = (1/N)(softmax − target)` (reusing the exponential, never forming `log p`),
   and accumulates two **byproducts** into a device `metric[2]` via `atomicAdd`:
   per-sample loss `log(denom) − (z[label]−a)` (= −log p_label) and the
   correct-prediction flag.
3. **`kernel_linear_backward<bool FuseReLU>`** — one thread per
   output column; sweeps the batch. `FuseReLU` (layer 1) masks the upstream
   gradient by `1[h_pre>0]` at the *head* (in place, so the weight-grad GEMM sees
   the masked `D`); always reduces the bias gradient `db = Σ_b dY`.
4. **`kernel_argmax`** — *inference/eval only*. One block per sample, one thread
   per class; tree reduction as softmax reduction 1 (max + lowest-index
   argmax) to produce `pred[b] = argmax_k z[b][k]`.
5. **`kernel_sgd`** — the elementwise `param -= lr*grad` update. It is
   instructor-provided (the update rule is fixed) but **launched** by the
   student-implemented `step()` from within `student_submission.cpp`.

### Mathematical foundations

If you are interested in the mathematical operations you can have a look in `Scratchpad_MNIST.pdf`

## Parallelism design (CUDA + MPI backends)

Training is **data-parallel SGD** across (typically) 4 MPI ranks. Each rank loads
the full dataset **from disk** (shared filesystem assumed — no dataset broadcast),
processes its *own* micro-batch (`BATCH_SIZE / 4 = 32` samples) on its own GPU, and
the per-step gradient all-reduce keeps the four weight copies identical.

## Prediction montage

To visualize running the model writes `predictions.ppm` — a 2 × 8 grid of 16
test-set digits, each tile bordered **green** if correct, **red** if wrong.

## Data

The bundled raw MNIST IDX files live in `data/`; see [`data/README.md`](data/README.md). The data
files are tracked via **Git LFS** (see `.gitattributes`). No download is required
to build or run.

---

> **Authorship.** Created by Konstantin Zeck ([konstantin.zeck@gmail.com](mailto:konstantin.zeck@gmail.com?subject=%5BMNIST-CUDA%5D%20Question&body=Hi%20Konstantin%0A))

[^1]: Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.
