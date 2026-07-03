//
// Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
// License: CC-BY-SA-4.0
//

#include "Utility.hpp"

#if defined(__CUDACC__) || defined(__NVCOMPILER)

#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>

// Singleton cuBLAS handle
static cublasHandle_t cublas_handle()
{
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        checkCublasErrors(cublasCreate(&h));
        return h;
    }();
    return handle;
}

// Student files define relu()
extern __device__ auto relu(float x) -> float;

// One block per sample, one warp of threads (padded power of two >= OUTPUT_DIM).
// Inactive lanes (threadIdx.x >= OUTPUT_DIM) contribute identity values so the
// shared-memory tree reductions in kernel_softmax_ce are well-formed.
static constexpr int SOFTMAX_BLOCK = 32;

/// @brief Bias-add tail after the cuBLAS GEMM, with optional fused ReLU.
/// @tparam FuseReLU true for layer 1 (stores pre-activation in `pre`, ReLU in `post`);
///                  false for layer 2 (pre == post == biased logits).
template<bool FuseReLU>
__global__ void kernel_linear_forward(float *pre, float *post, const float *bias, int B, int out_dim)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int o  = blockIdx.y * blockDim.y + threadIdx.y;
    if (bi >= B || o >= out_dim)
        return;

    int idx  = bi * out_dim + o;
    float v  = pre[idx] + bias[o];
    pre[idx] = v; // pre-activation gets cached for the backward mask
    if constexpr (FuseReLU)
        post[idx] = relu(v);
    else
        post[idx] = v;
}

/// @brief Fused softmax + cross-entropy: one block per sample, one thread per class.
///
/// For sample s, class k:
///   reduction 1: a = max_k z[s][k]  (numerical-stability shift; argmax kept for accuracy)
///   reduction 2: denom = Σ_k exp(z[s][k] - a)
///   gradient:    g[s][k] = (1/N)(exp(z[s][k]-a)/denom - target[s][k])
///
/// Thread 0 folds per-sample loss and correct-count into metric[] via atomicAdd.
__global__ void kernel_softmax_ce(const float *z, // [N][OUTPUT_DIM]
                                  const float *target, // [N][OUTPUT_DIM]  one-hot floats
                                  float *g, // [N][OUTPUT_DIM]  gradient output
                                  float *metric, // [2] {loss_sum, correct_count}
                                  int N)
{
    const int s = blockIdx.x; // sample index
    const int k = threadIdx.x; // class index (0 .. SOFTMAX_BLOCK-1)
    if (s >= N)
        return;

    const float *zrow       = z + s * OUTPUT_DIM;
    const float *target_row = target + s * OUTPUT_DIM;
    float *grow             = g + s * OUTPUT_DIM;

    __shared__ float value_scratch[SOFTMAX_BLOCK]; // value scratch (max, then sum)
    __shared__ int index_scratch[SOFTMAX_BLOCK]; // index scratch (argmax)
    __shared__ float target_logit; // logit of the true class
    __shared__ int target_label; // index of the true class

    const auto active{k < static_cast<int>(OUTPUT_DIM)};
    const auto zk{active ? zrow[k] : -INFINITY};

    // Exactly one lane is hot in the one-hot target row: it records the label.
    if (active && target_row[k] != 0.f) {
        target_logit = zk;
        target_label = k;
    }

    // === reduction 1: z_max = max_k z ===
    value_scratch[k] = zk;
    index_scratch[k] = k;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (k < stride) {
            const auto other{value_scratch[k + stride]};
            if (other > value_scratch[k]
                || (other == value_scratch[k] && index_scratch[k + stride] < index_scratch[k])) {
                value_scratch[k] = other;
                index_scratch[k] = index_scratch[k + stride];
            }
        }
        __syncthreads();
    }
    const auto z_max{value_scratch[0]};
    const auto prediction{index_scratch[0]};
    __syncthreads(); // before reusing value_scratch for the sum

    // ---- reduction 2: denom = Σ_k exp(z_k - a) ----
    const auto e{active ? std::exp(zk - z_max) : 0.f};
    value_scratch[k] = e;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (k < stride)
            value_scratch[k] += value_scratch[k + stride];
        __syncthreads();
    }
    const auto denom{value_scratch[0]};

    // ---- fused gradient g = (1/N)(softmax - target); reuse e, never log p ----
    if (active)
        grow[k] = ((e / denom) - target_row[k]) / static_cast<float>(N);

    // ---- byproducts: thread 0 folds this sample into the metric accumulator ----
    if (k == 0) {
        const auto loss_s{std::log(denom) - (target_logit - z_max)}; // L = -(Z_k* - z_max -(log(denom)))
        const auto correct{(prediction == target_label) ? 1.f : 0.f};
        atomicAdd(&metric[0], loss_s);
        atomicAdd(&metric[1], correct);
    }
}

/// @brief ReLU-grad mask (optional) and bias-gradient column-sum.
/// @tparam FuseReLU  true for layer 1: masks dY in-place with 1[h_pre > 0] before summing.
template<bool FuseReLU>
__global__ void kernel_linear_backward(float *dY, // [B][out_dim] in: upstream grad; out (FuseReLU): masked D
                                       const float *h_pre, // [B][out_dim] pre-activation (read only if FuseReLU)
                                       float *db, // [out_dim] bias gradient
                                       int B, int out_dim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim)
        return;

    float acc = 0.f;
    for (std::size_t bi = 0; bi < B; ++bi) {
        auto idx = bi * out_dim + o;
        auto d   = dY[idx];
        if constexpr (FuseReLU) {
            d       = d * static_cast<float>(h_pre[idx] > 0.f); // ReLU-gradient mask
            dY[idx] = d; // write back so the weight-grad GEMM reads D
        }
        acc += d;
    }
    db[o] = acc;
}

/// @brief Argmax over OUTPUT_DIM per sample — eval/inference path only.
__global__ void kernel_argmax(const float *z, std::size_t *pred, int N)
{
    const int b = blockIdx.x; // sample index
    const int k = threadIdx.x; // class index (0 .. SOFTMAX_BLOCK-1)
    if (b >= N)
        return;

    const float *row = z + b * OUTPUT_DIM;

    __shared__ float value_scratch[SOFTMAX_BLOCK]; // value scratch (running max)
    __shared__ int index_scratch[SOFTMAX_BLOCK]; // index scratch (argmax)

    const auto active{k < static_cast<int>(OUTPUT_DIM)};
    value_scratch[k] = active ? row[k] : -INFINITY;
    index_scratch[k] = k;
    __syncthreads();

    // Tree reduction: max value, argmax tie-broken to the lowest index.
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (k < stride) {
            const float other = value_scratch[k + stride];
            if (other > value_scratch[k]
                || (other == value_scratch[k] && index_scratch[k + stride] < index_scratch[k])) {
                value_scratch[k] = other;
                index_scratch[k] = index_scratch[k + stride];
            }
        }
        __syncthreads();
    }

    if (k == 0)
        pred[b] = static_cast<std::size_t>(index_scratch[0]);
}

/// @brief Elementwise SGD update: param -= lr * grad.
///
/// Launched by the student-implemented step() once each parameter's
/// all-reduce completes.  The update rule is fixed here; students implement
/// the MPI_Waitany scheduling around it.
__global__ void kernel_sgd(float *param, const float *grad, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    param[i] -= lr * grad[i];
}

Module::~Module()
{
    // clang-format off
    auto f = [](float *&p) -> void { if (p) { cudaFree(p); p = nullptr; } };
    f(_dW1); f(_db1); f(_dW2); f(_db2);
    f(_dgW1); f(_dgb1); f(_dgW2); f(_dgb2);
    f(_dx); f(_dh); f(_dhhat); f(_dz); f(_dg); f(_ddH); f(_dtarget); f(_dmetric);
    // clang-format on
}

Module::Module(Module &&other) noexcept :
    W1{std::move(other.W1)},
    b1{std::move(other.b1)},
    W2{std::move(other.W2)},
    b2{std::move(other.b2)},
    gW1{std::move(other.gW1)},
    gb1{std::move(other.gb1)},
    gW2{std::move(other.gW2)},
    gb2{std::move(other.gb2)},
    _device{other._device},
    _training{other._training},
    _N{other._N},
    _rng{other._rng},
    _x{std::move(other._x)},
    _h{std::move(other._h)},
    _hhat{std::move(other._hhat)},
    _z{std::move(other._z)},
    _g{std::move(other._g)},
    _dH{std::move(other._dH)},
    _pred{std::move(other._pred)},
    _loss{other._loss},
    _correct{other._correct},
    _dW1{other._dW1},
    _db1{other._db1},
    _dW2{other._dW2},
    _db2{other._db2},
    _dgW1{other._dgW1},
    _dgb1{other._dgb1},
    _dgW2{other._dgW2},
    _dgb2{other._dgb2},
    _dx{other._dx},
    _dh{other._dh},
    _dhhat{other._dhhat},
    _dz{other._dz},
    _dg{other._dg},
    _ddH{other._ddH},
    _dtarget{other._dtarget},
    _dmetric{other._dmetric}
{
    // Null source so its destructor doesn't double-free
    other._dW1 = other._db1 = other._dW2 = other._db2 = nullptr;
    other._dgW1 = other._dgb1 = other._dgW2 = other._dgb2 = nullptr;
    other._dx = other._dh = other._dhhat = other._dz = other._dg = other._ddH = nullptr;
    other._dtarget = other._dmetric = nullptr;
}

void Module::to(const std::string &device)
{
    if (device == "cpu") {
        _device = Device::CPU;
        return;
    }

    // Parse "cuda" or "cuda:N"
    int gpu_idx{0};
    if (device == "cuda") {
        gpu_idx = 0;
    } else if (device.substr(0, 5) == "cuda:") {
        gpu_idx = std::stoi(device.substr(5));
    } else {
        throw std::runtime_error("Module::to: unknown device \"" + device + "\"");
    }
    checkCudaErrors(cudaSetDevice(gpu_idx));
    _device = Device::CUDA;

    // Convenience: sizes
    auto W1_sz = INPUT_DIM * HIDDEN_DIM;
    auto b1_sz = HIDDEN_DIM;
    auto W2_sz = HIDDEN_DIM * OUTPUT_DIM;
    auto b2_sz = OUTPUT_DIM;
    // Activation buffers: allocate for the max batch (BATCH_SIZE) per rank
    auto act_x = BATCH_SIZE * INPUT_DIM;
    auto act_h = BATCH_SIZE * HIDDEN_DIM;
    auto act_z = BATCH_SIZE * OUTPUT_DIM;

    // --- Parameter buffers ---
    checkCudaErrors(cudaMalloc(&_dW1, W1_sz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_db1, b1_sz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dW2, W2_sz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_db2, b2_sz * sizeof(float)));

    // --- Gradient buffers ---
    checkCudaErrors(cudaMalloc(&_dgW1, W1_sz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dgb1, b1_sz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dgW2, W2_sz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dgb2, b2_sz * sizeof(float)));

    // --- Activation / scratch buffers ---
    checkCudaErrors(cudaMalloc(&_dx, act_x * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dh, act_h * sizeof(float))); // pre-relu hidden (with bias)
    checkCudaErrors(cudaMalloc(&_dhhat, act_h * sizeof(float))); // post-relu hidden
    checkCudaErrors(cudaMalloc(&_dz, act_z * sizeof(float))); // logits
    checkCudaErrors(cudaMalloc(&_dg, act_z * sizeof(float))); // softmax gradient
    checkCudaErrors(cudaMalloc(&_ddH, act_h * sizeof(float))); // grad wrt pre-relu hidden
    checkCudaErrors(cudaMalloc(&_dtarget, act_z * sizeof(float))); // one-hot targets
    checkCudaErrors(cudaMalloc(&_dmetric, 2 * sizeof(float))); // {loss_sum, correct_count}

    // --- Copy parameters H2D ---
    checkCudaErrors(cudaMemcpy(_dW1, W1.data(), W1_sz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_db1, b1.data(), b1_sz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dW2, W2.data(), W2_sz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_db2, b2.data(), b2_sz * sizeof(float), cudaMemcpyHostToDevice));
}

void Module::_fc1_forward()
{
    const auto alpha{1.f}, beta{0.f};
    auto N{static_cast<int>(_N)};
    checkCublasErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                  HIDDEN_DIM, // n (rows of C^T = cols of C)
                                  N, // m (cols of C^T = rows of C = batch size)
                                  INPUT_DIM, // k (inner dimension)
                                  &alpha, _dW1, HIDDEN_DIM, // B^T col-major (W1 row-major = W1^T col-major)
                                  _dx, INPUT_DIM, // A^T col-major (X row-major = X^T col-major)
                                  &beta, _dh, HIDDEN_DIM)); // C^T col-major => _dh row-major H

    dim3 block{16, 16};
    dim3 grid{(static_cast<unsigned>(N) + block.x - 1) / block.x,
              (static_cast<unsigned>(HIDDEN_DIM) + block.y - 1) / block.y}; // ceil(N/block.x) ceil(HIDDEN_DIM/block.y)
    kernel_linear_forward<true><<<grid, block>>>(_dh, _dhhat, _db1, N, HIDDEN_DIM);
    checkCudaErrors(cudaGetLastError());
}

void Module::_fc2_forward()
{
    const auto alpha{1.f}, beta{0.f};
    auto N{static_cast<int>(_N)};
    checkCublasErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                  OUTPUT_DIM, // n
                                  N, // m
                                  HIDDEN_DIM, // k
                                  &alpha, _dW2, OUTPUT_DIM, // B^T col-major
                                  _dhhat, HIDDEN_DIM, // A^T col-major
                                  &beta, _dz, OUTPUT_DIM)); // C^T col-major => _dz row-major Z

    dim3 block{16, 16};
    dim3 grid{(static_cast<unsigned>(N) + block.x - 1) / block.x,
              (static_cast<unsigned>(OUTPUT_DIM) + block.y - 1) / block.y}; // ceil(N/block.x) ceil(HIDDEN_DIM/block.y)
    kernel_linear_forward<false><<<grid, block>>>(_dz, _dz, _db2, N, OUTPUT_DIM);
    checkCudaErrors(cudaGetLastError());

    // In eval mode: compute argmax predictions (host _pred).
    if (not _training) {
        _pred.resize(static_cast<std::size_t>(N));
        std::size_t *d_pred{nullptr};
        checkCudaErrors(cudaMalloc(&d_pred, static_cast<std::size_t>(N) * sizeof(std::size_t)));
        kernel_argmax<<<static_cast<unsigned>(N), SOFTMAX_BLOCK>>>(_dz, d_pred, N);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(_pred.data(), d_pred, static_cast<std::size_t>(N) * sizeof(std::size_t),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_pred));
    }
}

void Module::_softmax_ce(const std::vector<float> &true_labels)
{
    auto N{static_cast<int>(_N)};
    checkCudaErrors(cudaMemcpy(_dtarget, true_labels.data(), static_cast<std::size_t>(N * OUTPUT_DIM) * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(_dmetric, 0, 2 * sizeof(float)));

    // One block per sample; SOFTMAX_BLOCK threads per block (one warp, padded
    // power of two >= OUTPUT_DIM for the in-block tree reductions).
    kernel_softmax_ce<<<static_cast<unsigned>(N), SOFTMAX_BLOCK>>>(_dz, _dtarget, _dg, _dmetric, N);
    checkCudaErrors(cudaGetLastError());
}

void Module::_fc2_backward()
{
    auto N{static_cast<int>(_N)};
    const auto alpha{1.f}, beta{0.f};

    dim3 block{256};
    dim3 grid{(static_cast<unsigned>(OUTPUT_DIM) + block.x - 1) / block.x}; // ceil(OUTPUT_DIM/block.x)

    // Bias gradient gb2 = Σ_b G[b]  (no mask: <FuseReLU=false>)
    kernel_linear_backward<false><<<grid, block>>>(_dg, nullptr, _dgb2, N, OUTPUT_DIM);
    checkCudaErrors(cudaGetLastError());

    // GEMM 1: _dgW2 = Hhat^T · G  [HIDDEN_DIM x OUTPUT_DIM]
    checkCublasErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, OUTPUT_DIM, HIDDEN_DIM, N, &alpha, _dg,
                                  OUTPUT_DIM, // G col-major (OP_N: use G as-is)
                                  _dhhat, HIDDEN_DIM, // Hhat col-major (OP_T: treat as Hhat^T)
                                  &beta, _dgW2, OUTPUT_DIM)); // dgW2 col-major [HIDDEN_DIM x OUTPUT_DIM]

    // GEMM 2: _ddH = G · W2^T  [N x HIDDEN_DIM]
    checkCublasErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_DIM, N, OUTPUT_DIM, &alpha, _dW2,
                                  OUTPUT_DIM, // W2 col-major (OP_N)
                                  _dg, OUTPUT_DIM, // G col-major (OP_T => G^T)
                                  &beta, _ddH,
                                  HIDDEN_DIM)); // ddH col-major [HIDDEN_DIM x N] => row-major [N x HIDDEN_DIM]
}

void Module::_fc1_backward()
{
    auto N{static_cast<int>(_N)};
    const auto alpha{1.f}, beta{0.f};

    dim3 block{256};
    dim3 grid{(static_cast<unsigned>(HIDDEN_DIM) + block.x - 1) / block.x}; // ceil(HIDDEN_DIM/block.x)

    // Mask _ddH by relu'(_dh) and accumulate gb1  (<FuseReLU=true>)
    kernel_linear_backward<true><<<grid, block>>>(_ddH, _dh, _dgb1, N, HIDDEN_DIM);
    checkCudaErrors(cudaGetLastError());

    // GEMM: _dgW1 = X^T · D  [INPUT_DIM x HIDDEN_DIM]
    checkCublasErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_DIM, INPUT_DIM, N, &alpha, _ddH,
                                  HIDDEN_DIM, // D col-major (OP_N)
                                  _dx, INPUT_DIM, // X col-major (OP_T => X^T)
                                  &beta, _dgW1, HIDDEN_DIM)); // dgW1 col-major [INPUT_DIM x HIDDEN_DIM]
}

void Module::pull()
{
    // D2H copy of parameters into host vectors.
    checkCudaErrors(cudaMemcpy(W1.data(), _dW1, static_cast<std::size_t>(INPUT_DIM * HIDDEN_DIM) * sizeof(float),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(b1.data(), _db1, HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(W2.data(), _dW2, static_cast<std::size_t>(HIDDEN_DIM * OUTPUT_DIM) * sizeof(float),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(b2.data(), _db2, OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
}

#endif // defined(__CUDACC__) || defined(__NVCOMPILER)
