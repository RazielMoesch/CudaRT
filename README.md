# CudaRT — CUDA Inference Engine

A C++17 inference runtime with custom CUDA kernels, GPU tensor storage, and a PyTorch weight export pipeline. The project evaluates a handwritten MNIST multilayer perceptron against eager and compiled PyTorch using identical weights, inputs, and FP32 precision.

## Implementation

- Move-only tensors with GPU memory ownership, host/device transfers, and reshaping.
- Sequential model execution with custom Linear and ReLU kernels and a Flatten layer.
- A validated binary weight loader and Python exporter using the CRT1 format.
- Numerical validation and benchmarking against PyTorch, including latency distributions, throughput, and reproducibility metadata.

The example uses `Flatten → Linear(784,128) → ReLU → Linear(128,10)` with pixels scaled to `[0,1]`.

## Quick start

Requires an NVIDIA GPU with a working driver, CUDA Toolkit, CMake ≥3.24, a CUDA-compatible C++ compiler, and Python ≥3.10.

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
python -m pip install -r python/requirements.txt
python benchmarks.py
```

The launcher builds C++ in Release mode, downloads MNIST, trains matching weights if no checkpoint exists, and runs the comparison. Later runs reuse the checkpoint.

```bash
python benchmarks.py --quick                         # Short comparison
python benchmarks.py --checkpoint path/to/model.pt   # Existing matching weights
python benchmarks.py --help                          # All options
```

## Benchmarking

The default run evaluates all 10,000 MNIST test images at batch sizes 1, 32, and 128. It compares logits within recorded FP32 tolerances and requires exact prediction agreement. Timing repeats the first full batch after warmup and excludes training, compilation, model loading, and process startup.

Reports separate CUDA-event forward latency from request latency, which includes allocation, host/device transfers, inference, and cleanup. Native allocator behavior is included in both implementations. The fastest passing PyTorch configuration is selected from eager and available compiled variants; this is a comparison for this workload, not a claim of globally optimal performance.

Results are written to `python/artifacts/comparison/report.md` and `report.json`, with raw samples, accuracy, numerical errors, hardware/software details, and artifact hashes. Performance figures should be taken from a completed, passing run on the reported hardware.

## Project layout

| Path | Purpose |
|---|---|
| `include/cudart/` | Tensor, model, layer, and loader interfaces |
| `src/` | Runtime implementation and CUDA kernels |
| `examples/mnist.cpp` | C++ inference and benchmark runner |
| `python/` | Training, checkpoint loading, export, comparison, and tests |
| `benchmarks.py` | Single-command workflow launcher |

## Tests

```bash
python -m unittest discover -s python -p 'test_*.py' -v
```

Unit tests cover the export format, checkpoint compatibility, and comparison checks. Cross-runtime validation requires the GPU benchmark workflow.

See [the Python workflow guide](python/README.md) for checkpoint requirements, the binary format, and detailed measurement methodology.
