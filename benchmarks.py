#!/usr/bin/env python3
"""Build, train if needed, and run the existing MNIST comparison workflow."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(command):
    print(f"\n> {shlex.join(str(item) for item in command)}", flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=project_path, default=ROOT / "python/artifacts/mnist.pt")
    parser.add_argument("--retrain", action="store_true", help="Retrain even if the checkpoint exists")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs when training is needed")
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA device for both implementations")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing build/cudart_example")
    parser.add_argument("--quick", action="store_true", help="Use 256 images, batches 1/32, 10 warmups and 20 iterations")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--data-dir", type=project_path, default=ROOT / "python/data")
    parser.add_argument("--output-dir", type=project_path, default=ROOT / "python/artifacts/comparison")
    parser.add_argument("--eager-only", action="store_true", help="Skip compiled PyTorch baselines")
    parser.add_argument("--prefixes", nargs=2, help="Layer names in an existing checkpoint")
    args = parser.parse_args()
    samples = args.samples if args.samples is not None else (256 if args.quick else 10000)
    batches = args.batch_sizes if args.batch_sizes is not None else ([1, 32] if args.quick else [1, 32, 128])
    warmup = args.warmup if args.warmup is not None else (10 if args.quick else 50)
    iterations = args.iterations if args.iterations is not None else (20 if args.quick else 200)
    if (args.epochs < 1 or args.device < 0 or not 1 <= samples <= 10000
            or min(batches) < 1 or max(batches) > samples or warmup < 1 or iterations < 1):
        parser.error("Invalid epoch, device, sample, batch, warmup or iteration count")
    train = args.retrain or not args.checkpoint.is_file()
    if train and args.prefixes:
        parser.error("--prefixes applies only to an existing checkpoint without --retrain")

    if not args.skip_build:
        run(["cmake", "-S", ROOT, "-B", ROOT / "build", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_CUDA_ARCHITECTURES=native"])
        run(["cmake", "--build", ROOT / "build", "--config", "Release", "--parallel"])
    if train:
        run([sys.executable, ROOT / "python/train.py", "train",
             "--checkpoint", args.checkpoint, "--crt", args.checkpoint.with_suffix(".crt"),
             "--epochs", args.epochs, "--device", f"cuda:{args.device}", "--data-dir", args.data_dir])
    else:
        print(f"\nUsing existing checkpoint: {args.checkpoint}", flush=True)
    command = [sys.executable, ROOT / "python/compare.py", "--checkpoint", args.checkpoint,
               "--executable", ROOT / "build/cudart_example", "--download",
               "--data-dir", args.data_dir, "--output-dir", args.output_dir,
               "--device", args.device, "--samples", samples,
               "--warmup", warmup, "--iterations", iterations, "--batch-sizes", *batches]
    if args.eager_only:
        command.append("--eager-only")
    if args.prefixes:
        command.extend(["--prefixes", *args.prefixes])
    run(command)
    print(f"\nBenchmark report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"\nCommand failed (exit {error.returncode}); stopping.", file=sys.stderr)
        sys.exit(error.returncode)
    except FileNotFoundError as error:
        print(f"\nRequired file or command missing: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
