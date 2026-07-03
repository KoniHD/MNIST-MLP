---
name: train
description: Train the MNIST-PyTorch MLP with project defaults. Use when the user asks to train the model, fit, or produce new weights. Accepts an optional epoch count as $ARGUMENTS.
---

Run training from `MNIST-PyTorch/` with project defaults (`-bs 128 -lr 0.1`):

```bash
cd MNIST-PyTorch
uv run mnist-pytorch train --epochs ${EPOCHS:-10} -bs 128 -lr 0.1 -p models/model_weights
```

If `$ARGUMENTS` is provided and is a positive integer, use it as `--epochs`. Otherwise default to 10.

After training:
- Confirm `models/model_weights.safetensors` was written.
- Suggest running `/evaluate` to check test accuracy against the ≥90% target.
- Device selection is automatic (CUDA → MPS → CPU); mention the chosen device from the run output if relevant.
