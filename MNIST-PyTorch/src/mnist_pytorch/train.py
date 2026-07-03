import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

from .data.dataset import get_train_loader
from .models.classifier import Classifier


def main(args):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Using Batch size: {args.batch_size}")
    train_loader = get_train_loader(batch_size=args.batch_size)

    model = Classifier().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1)
            opt.zero_grad()
            prediction = model(x)
            loss = F.cross_entropy(prediction, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / n_batches
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{args.epochs}] Completed | Avg Loss {avg_loss:.4f}")

    if not args.weights_path == "":
        save_dir = Path("models")
        save_path = (save_dir / args.weights_path).with_suffix(".safetensors").resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), save_path)
        print(f"Saved model weights to {save_path}.")

    if args.plot:
        df = pd.DataFrame(history)

        fig = px.line(df, x="epoch", y="loss", title="Training loss", markers=True)
        fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom MNIST Classifier.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="epochs",
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        metavar="size",
        help="Batch size for the training.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.1,
        metavar="lr",
        help="Learning rate for the training.",
    )
    parser.add_argument(
        "-p",
        "--weights_path",
        type=str,
        default="models/model_weights",
        metavar="path",
        help="Path to the safed model weights.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a plot of the loss curve.",
    )
    args = parser.parse_args()
    main(args)
