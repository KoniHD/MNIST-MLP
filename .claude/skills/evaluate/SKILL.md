---
name: evaluate
description: Run the MNIST-PyTorch evaluation against saved weights, report test accuracy, and flag if it falls below the 90% target. Use when the user asks to evaluate, check accuracy, or verify the trained model.
---

Run the evaluator from `MNIST-PyTorch/`:

```bash
cd MNIST-PyTorch
uv run mnist-pytorch evaluate -bs 128 -p models/model_weights.safetensors
```

If the user passes `$ARGUMENTS`, treat it as an alternate weights path (with or without the `.safetensors` suffix — normalize it).

After the run:
- Report the final test accuracy.
- If accuracy is below **90%**, surface it explicitly as a regression and suggest checking: epoch count, learning rate, whether the weights file matches the current model architecture in `src/mnist_pytorch/models/classifier.py`.
- If the weights file is missing, tell the user to train first with `/train` or `uv run mnist-pytorch train ...`.
