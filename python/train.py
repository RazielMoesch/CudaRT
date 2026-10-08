"""Train the matching model, or export a supported existing state dictionary."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from model import ARCHITECTURE, PREPROCESSING, export_model, load_model, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--checkpoint", type=Path, default=Path("python/artifacts/mnist.pt"))
    train.add_argument("--crt", type=Path, default=Path("python/artifacts/mnist.crt"))
    train.add_argument("--data-dir", type=Path, default=Path("python/data"))
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--crt", type=Path, required=True)
    export.add_argument("--prefixes", nargs=2)
    args = parser.parse_args()
    if args.command == "export":
        model, _ = load_model(args.checkpoint, args.prefixes)
        export_model(model, args.crt)
        print(f"Exported {args.crt}")
        return
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    model = make_model().to(args.device)
    dataset = MNIST(str(args.data_dir), train=True, download=True)
    # No mean/std normalization: this matches the C++ division exactly.
    inputs = dataset.data.float().div(255.0)
    loader = DataLoader(TensorDataset(inputs, dataset.targets), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        print(f"epoch {epoch + 1}/{args.epochs}: train loss={total_loss / len(dataset):.6f}")
    model.cpu().eval()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": {
        "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
        "seed": args.seed, "epochs": args.epochs, "training_samples": len(dataset),
        "torch_version": str(torch.__version__),
    }}, args.checkpoint)
    export_model(model, args.crt)
    print(f"Saved {args.checkpoint} and {args.crt}; run compare.py to measure test accuracy")


if __name__ == "__main__":
    main()
