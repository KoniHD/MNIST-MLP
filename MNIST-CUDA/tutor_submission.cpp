//
// Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
// License: CC-BY-SA-4.0
//

// === Train a 2-layer MLP (784→256→10) on MNIST ===
// This solution is faster due to less MPI_Bcasts

#include "Utility.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

__device__ auto relu(float x) -> float { return x > 0.0f ? x : 0.0f; }

void Module::backward(const std::vector<float> &true_labels)
{
    // DDP overlap scheme: Start non-blocking communication: training metric + four gradient all-reduces
    // CUDA-aware MPI: device pointers are handed straight to MPI.

    // softmax_ce fills _dg and the loss/accuracy byproducts in _dmetric.
    _softmax_ce(true_labels);
    cudaDeviceSynchronize();
    MPI_Iallreduce(MPI_IN_PLACE, _dmetric, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_metric_req);

    _fc2_backward();
    cudaDeviceSynchronize();
    MPI_Iallreduce(MPI_IN_PLACE, _dgW2, HIDDEN_DIM * OUTPUT_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[0]);
    MPI_Iallreduce(MPI_IN_PLACE, _dgb2, OUTPUT_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[1]);

    _fc1_backward();
    cudaDeviceSynchronize();
    MPI_Iallreduce(MPI_IN_PLACE, _dgW1, INPUT_DIM * HIDDEN_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[2]);
    MPI_Iallreduce(MPI_IN_PLACE, _dgb1, HIDDEN_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[3]);
}

void Module::step(float lr)
{
    // _grad_reqs follow the order {dW2, db2, dW1, db1}.
    float *params[4] = {_dW2, _db2, _dW1, _db1};
    float *grads[4]  = {_dgW2, _dgb2, _dgW1, _dgb1};
    int ns[4]        = {HIDDEN_DIM * OUTPUT_DIM, OUTPUT_DIM, INPUT_DIM * HIDDEN_DIM, HIDDEN_DIM};

    for (std::size_t done = 0; done < 4; ++done) {
        int idx{0};
        MPI_Waitany(4, _grad_reqs, &idx, MPI_STATUS_IGNORE);
        auto n{ns[idx]};
        // This kernel updates the gradients
        kernel_sgd<<<(n + 255) / 256, 256>>>(params[idx], grads[idx], lr, n);
    }
    cudaDeviceSynchronize(); // finish the SGD updates before the next forward reads the params
}

void Module::pullMetrics()
{
    MPI_Wait(&_metric_req, MPI_STATUS_IGNORE);
    float host[2]{0.f, 0.f};
    // Copy the training metrics back to host
    checkCudaErrors(cudaMemcpy(host, _dmetric, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    _loss    = host[0] / static_cast<float>(BATCH_SIZE);
    _correct = static_cast<int>(host[1]);
}

int main(int argc, char **argv)
{
    // === MPI init with FUNNELED thread support ===
    int provided;
    int rank, size;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided != MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::cerr << "MPI does not support the required threading level." << std::endl;

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    // === Inference-only path ===
    // Load weights from a safetensors file
    if (argc >= 2) {
        int rc;
        if (rank == 0)
            rc = Utility::predict_from_safetensors(argv[1], "cuda:0");
        MPI_Finalize();
        return rc;
    }

    // === Training path ===
    const std::size_t micro_batch{BATCH_SIZE / size}; // Split BATCH_SIZE among MPI_Processes into micro-batch
    const float lr{LEARNING_RATE
                   / size}; // fuse the size normalization into learning rate (necessary afer reduce + sum)

    // Seed: rank 0 reads from stdin, broadcasts to all ranks.
    uint32_t seed{0u};
    if (rank == 0)
        seed = Utility::read_seed();
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Dataset: every rank loads from disk (shared filesystem, no MPI needed).
    auto train_images{Utility::load_idx_images(TRAIN_IMAGES, N_TRAIN)};
    auto test_images{Utility::load_idx_images(TEST_IMAGES, N_TEST)};
    auto train_labels{Utility::load_idx_labels(TRAIN_LABELS, N_TRAIN)};
    auto test_labels{Utility::load_idx_labels(TEST_LABELS, N_TEST)};

    Module model{seed};
    model.to("cuda:" + std::to_string(rank)); // passes cuda:0, cuda:1, etc.

    auto loader{Utility::TrainLoader(train_images, train_labels, micro_batch, seed, rank)};
    const auto steps_per_epoch{Utility::TrainLoader::stepsPerEpoch()};
    std::vector<float> x_images, true_labels;

    // === Actual training Loop ===
    for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        model.train();
        loader.shuffle();

        double epoch_loss{0.0};
        long epoch_correct{0l};

        for (std::size_t step = 0; step < steps_per_epoch; ++step) {
            loader.batch(step, x_images, true_labels);

            model.forward(x_images, micro_batch);
            model.backward(true_labels);
            model.step(lr);
            model.pullMetrics();

            epoch_loss += model.loss();
            epoch_correct += model.numCorrect();
        }

        // Output loss/accuracy per epoch
        if (rank == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << EPOCHS << "] Completed | Avg Loss " << std::fixed
                      << std::setprecision(4) << (epoch_loss / steps_per_epoch) << " | Train Acc "
                      << std::setprecision(2)
                      << (100.0 * static_cast<double>(epoch_correct) / (steps_per_epoch * BATCH_SIZE)) << "%\n";
        }
    }

    // === Evaluation on test data ===
    if (rank == 0) {
        std::vector<std::size_t> test_preds;
        auto accuracy{Utility::evaluate(model, test_images, test_labels, N_TEST, &test_preds)};
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%\n";

        // Visualize predictions
        Utility::write_prediction_montage("predictions.ppm", test_images, test_preds, test_labels, 16);

        model.pull();
        Utility::save_safetensors("model.safetensors", model);

        Utility::output_result(accuracy);
    }

    MPI_Finalize();
    return 0;
}
