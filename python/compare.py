"""Validate and benchmark the same MNIST FP32 model in PyTorch and CudaRT."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST

from model import ARCHITECTURE, PREPROCESSING, export_model, load_model

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values, batch):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid timing samples")
    median = float(np.median(values))
    return {"median_ms": median, "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(values.mean()), "std_ms": float(values.std()),
            "images_per_second": batch * 1000.0 / median,
            "samples_ms": values.tolist()}


def correctness(actual, expected, labels, atol, rtol):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Output shape mismatch or non-finite logits")
    difference = np.abs(actual - expected)
    predictions = actual.argmax(axis=1)
    reference_predictions = expected.argmax(axis=1)
    close = np.isclose(actual, expected, atol=atol, rtol=rtol)
    matching = predictions == reference_predictions
    return {"passed": bool(close.all() and matching.all()),
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "max_relative_error": float((difference / np.maximum(np.abs(expected), atol)).max()),
            "logits_outside_tolerance": int((~close).sum()),
            "prediction_agreement": float(matching.mean()),
            "prediction_mismatches": int((~matching).sum()),
            "test_accuracy": float((predictions == labels).mean()),
            "atol": atol, "rtol": rtol}


@torch.inference_mode()
def evaluate(model, images, batch, device):
    outputs = []
    for start in range(0, len(images), batch):
        result = model(images[start:start + batch].to(device))
        outputs.append(result.cpu().numpy().copy())
    return np.concatenate(outputs)


@torch.inference_mode()
def benchmark(model, host_batch, device, warmup, iterations):
    device_batch = host_batch.to(device)
    for _ in range(warmup):
        output = model(device_batch)
        del output
    # Warm the request path too, including transfers and fresh allocations.
    for _ in range(warmup):
        x = host_batch.to(device)
        y = model(x)
        cpu = y.cpu()
        del x, y, cpu
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize event resources before sampling.
    start.record()
    end.record()
    end.synchronize()
    forward, request = [], []
    for _ in range(iterations):
        start.record()
        output = model(device_batch)
        end.record()
        end.synchronize()
        forward.append(start.elapsed_time(end))
        del output
        # The blocking GPU->CPU copy guarantees completed inference.
        tick = time.perf_counter_ns()
        x = host_batch.to(device)
        y = model(x)
        cpu = y.cpu()
        del x, y, cpu
        request.append((time.perf_counter_ns() - tick) / 1e6)
    return {"forward": summarize(forward, len(host_batch)),
            "request": summarize(request, len(host_batch))}


def markdown(report):
    lines = ["# MNIST CudaRT vs PyTorch", "",
             f"Architecture: {ARCHITECTURE}. Precision: FP32, IEEE matmul; input: {PREPROCESSING}.",
             f"GPU: {report['environment']['gpu']}. Test images: {report['samples']}.",
             "", "All durations exclude process startup, model loading, compilation, and image preprocessing.",
             "Forward uses CUDA events with resident inputs; request uses wall time for allocation, H2D, forward, D2H and cleanup.",
             "Native allocator behavior is included. Forward is a runtime interval, not isolated kernel time.",
             "", "| Batch | Implementation | Accuracy | Prediction agreement | Max abs logit error | Forward median / p95 ms | Request median / p95 ms | Request images/s |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        for name, result in run["implementations"].items():
            if "error" in result:
                lines.append(f"| {run['batch_size']} | {name} | unavailable | — | — | — | — | — |")
                continue
            check, forward, request = result["correctness"], result["timing"]["forward"], result["timing"]["request"]
            lines.append(f"| {run['batch_size']} | {name} | {check['test_accuracy']:.2%} | "
                         f"{check['prediction_agreement']:.2%} | {check['max_absolute_error']:.3g} | "
                         f"{forward['median_ms']:.6f} / {forward['p95_ms']:.6f} | "
                         f"{request['median_ms']:.6f} / {request['p95_ms']:.6f} | {request['images_per_second']:,.0f} |")
    lines.extend(["", "## Measured comparison", ""])
    for run in report["runs"]:
        if "best_pytorch_request" not in run:
            lines.append(f"Batch {run['batch_size']}: validation failed; no performance claim generated.")
            continue
        best = run["best_pytorch_request"]
        ratio = run["custom_request_speedup_vs_best_pytorch"]
        lines.append(f"Batch {run['batch_size']}: fastest passing PyTorch request baseline was `{best}`; "
                     f"CudaRT request speedup (PyTorch ms / CudaRT ms) = **{ratio:.3f}×**. "
                     "Values below 1 mean CudaRT was slower.")
    lines.extend(["", "## Resume wording from this run", ""])
    eligible = [run for run in report["runs"] if "best_pytorch_request" in run]
    if report["passed"] and report["completed"] and eligible:
        run = eligible[0]
        custom = run["implementations"]["cudart"]
        lines.append(f"Implemented a CUDA inference runtime for a 784→128→10 MNIST MLP; validated "
                     f"{custom['correctness']['prediction_agreement']:.2%} prediction agreement with FP32 PyTorch "
                     f"on {report['samples']:,} held-out images. Measured "
                     f"{custom['timing']['request']['median_ms']:.4f} ms median batch-{run['batch_size']} request latency "
                     f"on {report['environment']['gpu']}; compared against eager and available compiled PyTorch baselines.")
    else:
        lines.append("The run is incomplete or validation failed. Complete a passing run before citing these results.")
    lines.extend(["", "These measurements cover this architecture, device, precision and workload only; "
                  "the fastest tested configuration is not a claim of globally optimal PyTorch performance.",
                  "See report.json for raw timing samples, errors, checkpoint hashes and software versions."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefixes", nargs=2)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/cudart_example")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "python/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "python/artifacts/comparison")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 128])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eager-only", action="store_true", help="Skip compiled variants; narrows baseline coverage")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if (not 1 <= args.samples <= 10000 or min(args.batch_sizes) < 1
            or max(args.batch_sizes) > args.samples or args.warmup < 1 or args.iterations < 1
            or not np.isfinite([args.atol, args.rtol]).all() or args.atol <= 0 or args.rtol < 0):
        parser.error("Invalid sample, batch, warmup, iteration count or tolerances")
    if not torch.cuda.is_available():
        parser.error("A working CUDA GPU is required for comparison with the CUDA executable")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("CUDA device index is out of range")
    if not args.executable.is_file():
        parser.error("C++ executable missing; run cmake -S . -B build && cmake --build build")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    # New and legacy precision controls must not be mixed.
    try:
        torch.backends.cuda.matmul.fp32_precision
    except (AttributeError, AssertionError):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    model, checkpoint_metadata = load_model(args.checkpoint, args.prefixes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "model.crt"
    export_model(model, weights_path)
    model.to(device).eval()
    dataset = MNIST(str(args.data_dir), train=False, download=args.download)
    raw = dataset.data[:args.samples].numpy().copy()
    labels = dataset.targets[:args.samples].numpy().copy()
    images = torch.from_numpy(raw.astype(np.float32) / np.float32(255.0))
    image_path = args.output_dir / "images.bin"
    image_path.write_bytes(raw.tobytes(order="C"))
    np.save(args.output_dir / "labels.npy", labels)
    properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
        "precision": "FP32 / IEEE matmul", "samples": args.samples,
        "dataset": "first N images of the official MNIST test split (no shuffling)",
        "configuration": {"batch_sizes": args.batch_sizes, "warmup": args.warmup,
                          "iterations": args.iterations, "atol": args.atol, "rtol": args.rtol,
                          "eager_only": args.eager_only, "device_index": args.device},
        "environment": {"gpu": properties.name, "compute_capability": f"{properties.major}.{properties.minor}",
                        "gpu_memory_bytes": properties.total_memory, "torch": str(torch.__version__),
                        "torchvision": str(torchvision.__version__), "numpy": np.__version__,
                        "torch_cuda": torch.version.cuda, "python": platform.python_version(),
                        "platform": platform.platform(), "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES")},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint),
                       "metadata": checkpoint_metadata},
        "artifacts": {"weights_sha256": sha256(weights_path), "images_sha256": sha256(image_path),
                      "executable_sha256": sha256(args.executable)},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for folder in ("src", "include", "examples", "python")
                          for path in ((ROOT / folder).glob("*.py") if folder == "python" else (ROOT / folder).rglob("*"))
                          if path.suffix in (".cpp", ".cu", ".hpp", ".py")},
        "runs": [], "passed": True, "completed": False,
    }
    report["source_sha256"]["CMakeLists.txt"] = sha256(ROOT / "CMakeLists.txt")
    cache = args.executable.parent / "CMakeCache.txt"
    if cache.is_file():
        report["cmake_configuration"] = [line for line in cache.read_text().splitlines()
                                         if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CUDA_ARCHITECTURES:",
                                                             "CMAKE_CUDA_FLAGS", "CMAKE_CXX_FLAGS",
                                                             "CMAKE_CUDA_COMPILER:", "CMAKE_CXX_COMPILER:"))]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(markdown(report))
    for batch in args.batch_sizes:
        print(f"Evaluating and benchmarking batch {batch}...", flush=True)
        run = {"batch_size": batch, "implementations": {}}
        reference = evaluate(model, images, batch, device)
        np.save(args.output_dir / f"pytorch_eager_logits_b{batch}.npy", reference)
        eager = {"correctness": correctness(reference, reference, labels, args.atol, args.rtol),
                 "timing": benchmark(model, images[:batch], device, args.warmup, args.iterations)}
        run["implementations"]["pytorch_eager"] = eager
        result_path = args.output_dir / f"cudart_b{batch}.json"
        command = [str(args.executable.resolve()), "--benchmark", str(weights_path.resolve()),
                   str(image_path.resolve()), str(result_path.resolve()), str(batch),
                   str(args.warmup), str(args.iterations), str(args.device)]
        subprocess.run(command, check=True)
        cpp = json.loads(result_path.read_text())
        if cpp["build_type"] not in ("Release", "RelWithDebInfo"):
            raise ValueError("Benchmark requires a Release or RelWithDebInfo C++ build")
        if (cpp["image_count"] != args.samples or cpp["batch_size"] != batch
                or cpp["device_index"] != args.device
                or cpp["compute_capability"] != report["environment"]["compute_capability"]
                or len(cpp["forward_ms"]) != args.iterations or len(cpp["request_ms"]) != args.iterations):
            raise ValueError("C++ result does not match the requested workload or device")
        custom = np.asarray(cpp["logits"], dtype=np.float32).reshape(args.samples, 10)
        run["implementations"]["cudart"] = {
            "correctness": correctness(custom, reference, labels, args.atol, args.rtol),
            "timing": {"forward": summarize(cpp["forward_ms"], batch),
                       "request": summarize(cpp["request_ms"], batch)},
            "command": command, "cuda_runtime_version": cpp["cuda_runtime_version"],
            "cuda_driver_version": cpp["cuda_driver_version"],
            "build_type": cpp["build_type"],
        }
        if not args.eager_only:
            for mode in ("reduce-overhead", "max-autotune"):
                name = f"pytorch_compile_{mode}"
                try:
                    compiled = torch.compile(model, mode=mode, fullgraph=True)
                    # Evaluate first: compilation and any partial-batch recompilation are untimed.
                    actual = evaluate(compiled, images, batch, device)
                    np.save(args.output_dir / f"{name}_logits_b{batch}.npy", actual)
                    check = correctness(actual, reference, labels, args.atol, args.rtol)
                    run["implementations"][name] = {
                        "correctness": check,
                        "timing": benchmark(compiled, images[:batch], device, args.warmup, args.iterations),
                    }
                    del compiled
                except Exception as error:
                    # Explicitly record an unavailable baseline; never invent a fallback timing.
                    run["implementations"][name] = {"error": f"{type(error).__name__}: {error}"}
                    print(f"{name} unavailable: {error}", flush=True)
        candidates = {name: result for name, result in run["implementations"].items()
                      if name.startswith("pytorch_") and "error" not in result and result["correctness"]["passed"]}
        valid = all(result["correctness"]["passed"] for result in run["implementations"].values() if "error" not in result)
        report["passed"] = report["passed"] and valid
        if valid and candidates:
            for metric in ("request", "forward"):
                best = min(candidates, key=lambda name: candidates[name]["timing"][metric]["median_ms"])
                run[f"best_pytorch_{metric}"] = best
                run[f"custom_{metric}_speedup_vs_best_pytorch"] = (
                    candidates[best]["timing"][metric]["median_ms"]
                    / run["implementations"]["cudart"]["timing"][metric]["median_ms"])
        report["runs"].append(run)
        # Persist completed batches so a later process failure does not erase results.
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (args.output_dir / "report.md").write_text(markdown(report))
    report["completed"] = True
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(markdown(report))
    print(f"Report: {args.output_dir / 'report.md'}")
    if not report["passed"]:
        raise SystemExit("Numerical validation failed; see report.json (no resume claim generated)")


if __name__ == "__main__":
    main()
