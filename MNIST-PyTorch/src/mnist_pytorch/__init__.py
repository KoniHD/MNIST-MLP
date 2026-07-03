import argparse

from . import evaluate, train
from .data.dataset import get_test_loader, get_train_loader
from .models.classifier import Classifier

__all__ = ["Classifier", "get_train_loader", "get_test_loader"]


def main():
    """Master entry point for the mnist-pytorch CLI."""
    parser = argparse.ArgumentParser(
        description="The ultimate MNIST PyTorch toolkit.", prog="mnist-pytorch"
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available sub-commands"
    )

    # ---------------------------------------------------------
    # Sub-command: TRAIN
    # ---------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train", help="Train the MNIST model from scratch"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="epochs",
        help="Number of epochs to train.",
    )
    train_parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        metavar="size",
        help="Batch size for training.",
    )
    train_parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.1,
        metavar="lr",
        help="Learning rate for training.",
    )
    train_parser.add_argument(
        "-p",
        "--weights_path",
        type=str,
        default="models/model_weights",
        metavar="path",
        help="Path to the safed model weights.",
    )
    train_parser.add_argument(
        "--plot", action="store_true", help="Generate a plot of the loss curve."
    )

    # ---------------------------------------------------------
    # Sub-command: EVALUATE
    # ---------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained MNIST model"
    )
    eval_parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        metavar="size",
        help="Batch size for the evaluation,",
    )
    eval_parser.add_argument(
        "-p",
        "--weights_path",
        type=str,
        default="models/model_weights.safetensors",
        metavar="path",
        help="Path to the safed model weights.",
    )
    eval_parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot test examples.",
    )

    # Parse all arguments from the terminal
    args = parser.parse_args()

    # Route the execution to the appropriate module's main() function
    if args.command == "train":
        train.main(args)
    elif args.command == "evaluate":
        evaluate.main(args)
