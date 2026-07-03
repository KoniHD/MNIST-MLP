import argparse
from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from safetensors.torch import load_model
from tqdm import tqdm

from .data.dataset import get_test_loader
from .models.classifier import Classifier


def main(args):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using dvice: {device}")

    test_loader = get_test_loader(batch_size=args.batch_size)
    pbar = tqdm(test_loader, desc="Testing on test dataset")

    model = Classifier().to(device)
    weights_path = Path(args.weights_path).with_suffix(".safetensors").resolve()

    if not weights_path.exists():
        print(f"Error: Could not find model weights at {weights_path}.")

    load_model(model, weights_path)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")

    if args.plot:
        print("Generating prediction visualization grid...")

        # Grab a single batch from the test_loader
        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)
        x_flat = x.view(x.shape[0], -1)

        with torch.no_grad():
            preds = model(x_flat).argmax(dim=1)

        # 1. Pre-generate the HTML-formatted titles for each subplot
        titles = []
        for i in range(16):
            color = "green" if preds[i] == y[i] else "red"
            titles.append(
                f"<span style='color:{color}; font-size:12px'>pred {preds[i].item()}<br>true {y[i].item()}</span>"
            )

        # 2. Initialize the 2x8 subplot grid
        fig = make_subplots(
            rows=2,
            cols=8,
            subplot_titles=titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.15,
        )

        # 3. Populate the grid with the image arrays
        for i in range(16):
            row = (i // 8) + 1
            col = (i % 8) + 1

            # Plotly expects numpy arrays, not PyTorch tensors
            img = x[i].cpu().view(28, 28).numpy()

            fig.add_trace(
                go.Heatmap(
                    z=img,
                    colorscale="gray",
                    showscale=False,  # Hide the colorbar for each individual image
                    hoverinfo="none",  # Disable the hover tooltip to keep it clean
                ),
                row=row,
                col=col,
            )

            # Plotly heatmaps draw from bottom to top by default.
            # We must reverse the Y-axis so the MNIST digit isn't upside down.
            fig.update_yaxes(autorange="reversed", visible=False, row=row, col=col)
            fig.update_xaxes(visible=False, row=row, col=col)

        # 4. Clean up the layout and display
        fig.update_layout(
            height=400,
            width=1000,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="white",
        )

        fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the custom MNIST Classifier."
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        metavar="size",
        help="Batch size for the evaluation,",
    )
    parser.add_argument(
        "-p",
        "--weights_path",
        type=str,
        default="models/model_weights.safetensors",
        metavar="path",
        help="Path to the safed model weights.",
    )
    parser.add_argument("--plot", action="store_true", help="Plot test examples.")
    args = parser.parse_args()
    main(args)
