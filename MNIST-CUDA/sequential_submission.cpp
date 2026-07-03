//
// Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
// License: CC-BY-SA-4.0
//

// === Train a 2-layer MLP (784→256→10) on MNIST ===

#include "Utility.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <print>
#include <stdexcept>
#include <vector>

Module::Module(Module &&other) noexcept = default;

void Module::to(const std::string &device)
{
    if (device == "cpu") {
        _device = Device::CPU;
        return;
    }
    if (device == "cuda" || device.starts_with("cuda:")) {
        throw std::runtime_error("CUDA unavailable in the sequential CPU build.");
    }
    if (device == "mlx") {
        throw std::runtime_error("MLX unavailable in the sequential CPU build.");
    }
    throw std::runtime_error("Unknown device string: \"" + device + "\".");
}

void Module::pull() {}

void Module::pullMetrics() {}

void Module::_fc1_forward()
{
    // fc1_forward: _h = _x · W1 + b1  [N, HIDDEN_DIM]; _hhat = ReLU(_h)
    // _x  : [N, INPUT_DIM]
    // W1  : [INPUT_DIM, HIDDEN_DIM]  (row-major)

    _h.resize(_N * HIDDEN_DIM);
    _hhat.resize(_N * HIDDEN_DIM);

    for (std::size_t s = 0; s < _N; ++s) {
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
            auto sum{b1[j]};
            for (std::size_t i = 0; i < INPUT_DIM; ++i)
                sum += _x[(s * INPUT_DIM) + i] * W1[(i * HIDDEN_DIM) + j];
            _h[(s * HIDDEN_DIM) + j]    = sum;
            _hhat[(s * HIDDEN_DIM) + j] = std::max(sum, 0.0f); // ReLU
        }
    }
}

void Module::_fc2_forward()
{
    // fc2_forward: _z = _hhat · W2 + b2  [N, OUTPUT_DIM]
    // _hhat : [N, HIDDEN_DIM]
    // W2    : [HIDDEN_DIM, OUTPUT_DIM]  (row-major)

    _z.resize(_N * OUTPUT_DIM);

    for (std::size_t s = 0; s < _N; ++s) {
        for (std::size_t k = 0; k < OUTPUT_DIM; ++k) {
            auto sum{b2[k]};
            for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
                sum += _hhat[(s * HIDDEN_DIM) + j] * W2[(j * OUTPUT_DIM) + k];
            _z[(s * OUTPUT_DIM) + k] = sum;
        }
    }

    if (not _training) {
        _pred.resize(_N);
        for (std::size_t s = 0; s < _N; ++s) {
            std::size_t best = 0;
            float best_v     = _z[s * OUTPUT_DIM];
            for (std::size_t k = 1; k < OUTPUT_DIM; ++k) {
                float v = _z[(s * OUTPUT_DIM) + k];
                if (v > best_v) {
                    best_v = v;
                    best   = k;
                }
            }
            _pred[s] = static_cast<int>(best);
        }
    }
}

void Module::_softmax_ce(const std::vector<float> &true_labels)
{
    _g.resize(_N * OUTPUT_DIM);

    const float inv_n{1.0f / static_cast<float>(_N)};
    double loss_sum{0.0};
    int correct{0};

    for (std::size_t s = 0; s < _N; ++s) {
        const float *z_row{_z.data() + (s * OUTPUT_DIM)};
        const float *t_row{true_labels.data() + (s * OUTPUT_DIM)};
        float *g_row{_g.data() + (s * OUTPUT_DIM)};

        // Numerically stable softmax denominator
        float z_max{z_row[0]};
        for (std::size_t k = 1; k < OUTPUT_DIM; ++k)
            z_max = std::max(z_max, z_row[k]);

        float sum{0.0f};
        for (std::size_t k = 0; k < OUTPUT_DIM; ++k) {
            g_row[k] = std::exp(z_row[k] - z_max);
            sum += g_row[k];
        }

        // Byproducts: argmax(z) for accuracy, argmax(one-hot) for the label.
        std::size_t pred{0}, label{0};
        for (std::size_t k = 1; k < OUTPUT_DIM; ++k) {
            if (z_row[k] > z_row[pred])
                pred = k;
            if (t_row[k] > t_row[label])
                label = k;
        }
        if (pred == label)
            ++correct;

        // Cross-entropy of this sample: -log(softmax[label]) = log(sum) - (z[label]-mx)
        loss_sum += std::log(static_cast<double>(sum)) - static_cast<double>(z_row[label] - z_max);

        for (std::size_t k = 0; k < OUTPUT_DIM; ++k)
            g_row[k] = inv_n * ((g_row[k] / sum) - t_row[k]);
    }

    _loss    = static_cast<float>(loss_sum * inv_n); // mean CE over the batch
    _correct = correct;
}

void Module::_fc2_backward()
{
    // gW2 = _hhat^T · _g   [HIDDEN_DIM, OUTPUT_DIM]
    // gb2 = sum_s _g        [OUTPUT_DIM]
    // _dH = _g · W2^T       [N, HIDDEN_DIM]

    gW2.assign(HIDDEN_DIM * OUTPUT_DIM, 0.0f);
    gb2.assign(OUTPUT_DIM, 0.0f);
    _dH.resize(_N * HIDDEN_DIM);

    for (std::size_t s = 0; s < _N; ++s) {
        const float *hhat_row{_hhat.data() + (s * HIDDEN_DIM)};
        const float *g_row{_g.data() + (s * OUTPUT_DIM)};
        float *dH_row{_dH.data() + (s * HIDDEN_DIM)};

        // gW2 += hhat^T * g
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
            for (std::size_t k = 0; k < OUTPUT_DIM; ++k)
                gW2[(j * OUTPUT_DIM) + k] += hhat_row[j] * g_row[k];

        // gb2 += g
        for (std::size_t k = 0; k < OUTPUT_DIM; ++k)
            gb2[k] += g_row[k];

        // _dH = g · W2^T
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
            float val{0.0f};
            for (std::size_t k = 0; k < OUTPUT_DIM; ++k)
                val += g_row[k] * W2[(j * OUTPUT_DIM) + k];
            dH_row[j] = val;
        }
    }
}

void Module::_fc1_backward()
{
    // D = _dH ⊙ 1[_h > 0]   (ReLU mask)
    // gW1 = _x^T · D          [INPUT_DIM, HIDDEN_DIM]
    // gb1 = sum_s D            [HIDDEN_DIM]

    gW1.assign(INPUT_DIM * HIDDEN_DIM, 0.0f);
    gb1.assign(HIDDEN_DIM, 0.0f);

    for (std::size_t s = 0; s < _N; ++s) {
        const float *x_row{_x.data() + (s * INPUT_DIM)};
        const float *h_row{_h.data() + (s * HIDDEN_DIM)};
        const float *dH_row{_dH.data() + (s * HIDDEN_DIM)};

        for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
            float d{(h_row[j] > 0.0f) ? dH_row[j] : 0.0f}; // ReLU mask

            gb1[j] += d;

            for (std::size_t i = 0; i < INPUT_DIM; ++i)
                gW1[(i * HIDDEN_DIM) + j] += x_row[i] * d;
        }
    }
}

void Module::step(float lr)
{
    for (std::size_t i = 0; i < W1.size(); ++i)
        W1[i] -= lr * gW1[i];
    for (std::size_t j = 0; j < b1.size(); ++j)
        b1[j] -= lr * gb1[j];
    for (std::size_t i = 0; i < W2.size(); ++i)
        W2[i] -= lr * gW2[i];
    for (std::size_t k = 0; k < b2.size(); ++k)
        b2[k] -= lr * gb2[k];
}

auto main(int argc, char *argv[]) -> int
{
    // === Inference-only path ===
    // Load weights from a safetensors file
    if (argc == 2)
        return Utility::predict_from_safetensors(argv[1], "cpu");

    // === Training path ===
    auto seed{Utility::read_seed()};

    std::println("Loading MNIST data...");
    auto train_images{Utility::load_idx_images(TRAIN_IMAGES, N_TRAIN)};
    auto test_images{Utility::load_idx_images(TEST_IMAGES, N_TEST)};
    auto train_labels{Utility::load_idx_labels(TRAIN_LABELS, N_TRAIN)};
    auto test_labels{Utility::load_idx_labels(TEST_LABELS, N_TEST)};

    Module model{seed};
    model.to("cpu");

    auto loader{Utility::TrainLoader(train_images, train_labels, BATCH_SIZE, seed)};
    const auto n_batches{Utility::TrainLoader::stepsPerEpoch()};
    std::vector<float> x_images, true_labels;

    for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        model.train();
        loader.shuffle();

        double epoch_loss{0.0};
        long epoch_correct{0l};

        for (std::size_t b = 0; b < n_batches; ++b) {
            loader.batch(b, x_images, true_labels);

            model.forward(x_images, BATCH_SIZE);
            model.backward(true_labels);
            model.step(LEARNING_RATE);
            model.pullMetrics();

            epoch_loss += model.loss();
            epoch_correct += model.numCorrect();
        }

        // Output loss/accuracy per epoch
        std::println("Epoch [{}/{}] Completed | Avg Loss {:.4f} | Train Acc {:.2f}%", epoch + 1, EPOCHS,
                     epoch_loss / static_cast<double>(n_batches),
                     100.0 * static_cast<double>(epoch_correct) / static_cast<double>(n_batches * BATCH_SIZE));
    }

    // === Evaluation on test data ===
    std::vector<std::size_t> test_preds;
    auto accuracy{Utility::evaluate(model, test_images, test_labels, N_TEST, &test_preds)};
    std::println("Test Accuracy: {:.2f}%", accuracy * 100.0f);

    // Visualize predictions
    Utility::write_prediction_montage("predictions.ppm", test_images, test_preds, test_labels, 16);

    model.pull();
    Utility::save_safetensors("model.safetensors", model);

    Utility::output_result(accuracy);
    return 0;
}
