import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def to_tensor(batch):
    """Transforms raw image batches into normalized, flattened tensors."""
    batch["x"] = [
        torch.tensor(np.array(img), dtype=torch.float32).view(-1) / 255.0
        for img in batch["image"]
    ]
    batch["y"] = batch["label"]
    return batch


def collate(batch):
    """Stacks individual dictionary items into batched tensors."""
    return torch.stack([b["x"] for b in batch]), torch.tensor([b["y"] for b in batch])


def get_train_loader(batch_size: int = 128):
    """Loads the MNIST dataset and returns train dataloaders."""
    ds = load_dataset("ylecun/mnist")
    ds = ds.with_transform(to_tensor)

    train_loader = DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    return train_loader


def get_test_loader(batch_size: int = 256):
    """Loads the MNIST dataset and returns test dataloaders."""
    ds = load_dataset("ylecun/mnist")
    ds = ds.with_transform(to_tensor)

    test_loader = DataLoader(ds["test"], batch_size=batch_size, collate_fn=collate)

    return test_loader
