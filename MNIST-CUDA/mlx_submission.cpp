//
// Created by Konstantin Zeck (konstantin.zeck@gmail.com) on 30/07/2026
// License: CC-BY-SA-4.0
//

// === Train a 2-layer MLP (784→256→10) on MNIST using SGD on Apple MLX backend ===

#include "Utility.hpp"

#include <mlx/mlx.h>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mx = mlx::core;

Module::Module(Module &&other) noexcept :
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
    _aW1(std::move(other._aW1)),
    _ab1(std::move(other._ab1)),
    _aW2(std::move(other._aW2)),
    _ab2(std::move(other._ab2)),
    _agW1(std::move(other._agW1)),
    _agb1(std::move(other._agb1)),
    _agW2(std::move(other._agW2)),
    _agb2(std::move(other._agb2)),
    _aH(std::move(other._aH)),
    _aHhat(std::move(other._aHhat)),
    _aZ(std::move(other._aZ)),
    _aG(std::move(other._aG)),
    _adH(std::move(other._adH)),
    W1{std::move(other.W1)},
    b1{std::move(other.b1)},
    W2{std::move(other.W2)},
    b2{std::move(other.b2)},
    gW1{std::move(other.gW1)},
    gb1{std::move(other.gb1)},
    gW2{std::move(other.gW2)},
    gb2{std::move(other.gb2)}
{}

void Module::to(const std::string &device)
{
    if (device != "mlx")
        throw std::runtime_error(R"(The MLX build only supports to("mlx"); got ")" + device + "\".");

    // Every training step reuses these arrays.
    _device = Device::MLX;
    _aW1    = mx::array{W1.data(), mx::Shape{INPUT_DIM, HIDDEN_DIM}, mx::float32};
    _ab1    = mx::array{b1.data(), mx::Shape{HIDDEN_DIM}, mx::float32};
    _aW2    = mx::array{W2.data(), mx::Shape{HIDDEN_DIM, OUTPUT_DIM}, mx::float32};
    _ab2    = mx::array{b2.data(), mx::Shape{OUTPUT_DIM}, mx::float32};
    mx::eval(_aW1, _ab1, _aW2, _ab2);
}

void Module::pull()
{
    // Called once before save_safetensors
    mx::eval(_aW1, _ab1, _aW2, _ab2);
    auto to_host = [](const mx::array &arr, std::vector<float> &dst) -> void {
        const auto *ptr = arr.data<float>();
        dst.assign(ptr, ptr + arr.size());
    };
    to_host(_aW1, W1);
    to_host(_ab1, b1);
    to_host(_aW2, W2);
    to_host(_ab2, b2);
}

void Module::pullMetrics() {}

void Module::_fc1_forward()
{
    auto X = mx::array{_x.data(), mx::Shape{static_cast<int>(_N), INPUT_DIM}, mx::float32};

    _aH = mx::matmul(X, _aW1) + _ab1;

    // ReLU
    _aHhat = mx::maximum(_aH, mx::array(0.0f));
    mx::eval(_aH, _aHhat);
}

void Module::_fc2_forward()
{
    _aZ = mx::matmul(_aHhat, _aW2) + _ab2;
    mx::eval(_aZ);

    if (not _training) {
        auto predictions = mx::argmax(_aZ, 1);
        mx::eval(predictions);
        const auto *ptr = predictions.data<uint32_t>();
        _pred.assign(ptr, ptr + _N);
    }
}

void Module::_softmax_ce(const std::vector<float> &true_labels)
{
    auto one_hot = mx::array{true_labels.data(), mx::Shape{static_cast<int>(_N), OUTPUT_DIM}, mx::float32};

    // G = (1/N) * (softmax(Z) - one_hot)
    auto predictions = mx::softmax(_aZ, 1); // numerically stable in MLX
    _aG              = (predictions - one_hot) * mx::array(1.0f / static_cast<float>(_N));

    // Byproducts: mean CE; correct = #(argmax match).
    auto cross_entropy = -mx::sum(one_hot * mx::log(predictions), 1, false);
    auto loss          = mx::mean(cross_entropy, false);
    auto correct       = mx::sum(mx::astype(mx::equal(mx::argmax(_aZ, 1), mx::argmax(one_hot, 1)), mx::int32), false);

    mx::eval(_aG, loss, correct);
    _loss    = loss.item<float>();
    _correct = correct.item<int>();
}

void Module::_fc2_backward()
{
    _agW2 = mx::matmul(mx::transpose(_aHhat, {1, 0}), _aG);
    _agb2 = mx::sum(_aG, 0, false);
    _adH  = mx::matmul(_aG, mx::transpose(_aW2, {1, 0}));
    mx::eval(_agW2, _agb2, _adH);
}

void Module::_fc1_backward()
{
    auto X = mx::array{_x.data(), mx::Shape{static_cast<int>(_N), INPUT_DIM}, mx::float32};

    auto mask = mx::astype(mx::greater(_aH, mx::array(0.0f)), mx::float32);
    auto D    = _adH * mask;

    _agW1 = mx::matmul(mx::transpose(X, {1, 0}), D);
    _agb1 = mx::sum(D, 0, false);
    mx::eval(_agW1, _agb1);
}

void Module::step(float lr)
{
    // SGD in place
    mx::array a_lr = mx::array(lr);
    _aW1           = _aW1 - a_lr * _agW1;
    _ab1           = _ab1 - a_lr * _agb1;
    _aW2           = _aW2 - a_lr * _agW2;
    _ab2           = _ab2 - a_lr * _agb2;
    mx::eval(_aW1, _ab1, _aW2, _ab2);
}

auto main(int argc, char *argv[]) -> int
{
    // === Inference-only path ===
    // Load weights from a safetensors file
    if (argc == 2)
        return Utility::predict_from_safetensors(argv[1], "mlx");

    // === Training path ===
    auto seed = Utility::read_seed();

    std::println("Loading MNIST data...");
    auto train_images = Utility::load_idx_images(TRAIN_IMAGES, N_TRAIN);
    auto train_labels = Utility::load_idx_labels(TRAIN_LABELS, N_TRAIN);
    auto test_images  = Utility::load_idx_images(TEST_IMAGES, N_TEST);
    auto test_labels  = Utility::load_idx_labels(TEST_LABELS, N_TEST);

    Module model{seed};
    model.to("mlx");

    auto loader          = Utility::TrainLoader(train_images, train_labels, BATCH_SIZE, seed);
    const auto n_batches = Utility::TrainLoader::stepsPerEpoch();
    std::vector<float> x_images, true_labels;

    for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        model.train();
        loader.shuffle();

        double epoch_loss  = 0.0;
        long epoch_correct = 0;

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
    float accuracy = Utility::evaluate(model, test_images, test_labels, N_TEST, &test_preds);
    std::println("Test Accuracy: {:.2f}%", accuracy * 100.0f);

    // Visualize predictions
    Utility::write_prediction_montage("predictions.ppm", test_images, test_preds, test_labels, 16);

    model.pull();
    Utility::save_safetensors("model.safetensors", model);

    Utility::output_result(accuracy);
    return 0;
}
