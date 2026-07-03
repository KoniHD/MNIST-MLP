# MNIST Data

## Provenance

Original dataset: LeCun et al. —
<http://yann.lecun.com/exdb/mnist//> [^1]

**License: CC BY-SA 3.0**.


## Files

| Filename                    | Role          | Items  | Size (bytes) |
|-----------------------------|---------------|--------|--------------|
| `train-images-idx3-ubyte`   | Train images  | 60 000 | 47 040 016   |
| `train-labels-idx1-ubyte`   | Train labels  | 60 000 |     60 008   |
| `t10k-images-idx3-ubyte`    | Test images   | 10 000 |  7 840 016   |
| `t10k-labels-idx1-ubyte`    | Test labels   | 10 000 |     10 008   |

---

## IDX File Format

All integers are **big-endian 32-bit** unless noted otherwise.

### Image files (magic 0x00000803 / 2051)

```
offset  type    value       description
0       int32   0x00000803  magic number
4       int32   N           number of images
8       int32   28          number of rows
12      int32   28          number of columns
16      uint8[] …           N × 784 pixels, row-major, values in [0, 255]
```

### Label files (magic 0x00000801 / 2049)

```
offset  type    value       description
0       int32   0x00000801  magic number
4       int32   N           number of labels
8       uint8[] …           N label bytes, values in [0, 9]
```

[^1] Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.
