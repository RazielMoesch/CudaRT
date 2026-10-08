# MNIST: PyTorch vs the custom CUDA runtime

This workflow trains or loads the exact model in `examples/mnist.cpp`, exports weights for `src/loader.cpp`, checks every supplied logit and predicted class, and benchmarks the C++ executable against eager PyTorch plus `torch.compile` in `reduce-overhead` and `max-autotune` modes.

The architecture is `Flatten → Linear(784,128) → ReLU → Linear(128,10)`, with biases and FP32 weights. Inputs are raw 28×28 uint8 images divided by 255, with no mean/std normalization or softmax. Existing checkpoints must have this architecture and preprocessing; weight shapes alone cannot prove that their original model had the same activations.

## Setup

Run from the repository root on a machine with a working NVIDIA CUDA driver:

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
python -m pip install -r python/requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

Choose a PyTorch wheel compatible with your driver using the [official installation selector](https://pytorch.org/get-started/locally/) if the default pip wheel does not work. Training and exporting can run on CPU; the direct C++ comparison requires CUDA. The C++ runner remains `build/cudart_example`.

## Train matching weights

For a single command after setup, run `python3 benchmarks.py` from the repository root.
It builds in Release mode for the local GPU, trains if the checkpoint is missing,
and calls the existing comparison script. Later runs reuse the checkpoint.
Use `python3 benchmarks.py --quick` for a short comparison (training still uses
10 epochs if needed), `--retrain` to train again, or `--checkpoint path/to/existing.pt`
to use matching weights. `--skip-build` reuses the existing executable.
Run `python3 benchmarks.py --help` for benchmark options. The launcher uses the
same Python interpreter you invoke it with and installs no dependencies.

```bash
python python/train.py train --epochs 10 --device cuda:0
```

This uses the official 60,000-image MNIST training split, seed 42, Adam at 0.001, and batch size 128. It downloads MNIST to `python/data`, writes `python/artifacts/mnist.pt`, and exports `python/artifacts/mnist.crt`. It does not train on test images or pick weights using test accuracy. Use `--device cpu` for CPU training. Epoch count, seed, paths and batch size are configurable; a fixed seed is not a promise of bitwise reproducibility across platforms.

## Load an existing matching model

The exporter supports a plain state dictionary or a dictionary containing `state_dict` or `model_state_dict`. Supported layer keys are `1`/`3` (the matching Sequential), `fc1`/`fc2`, or `linear0`/`linear1`. An optional `module.` prefix is removed. Specify other names with `--prefixes FIRST SECOND`.

```bash
python python/train.py export \
  --checkpoint path/to/existing.pt \
  --crt python/artifacts/existing.crt
```

Loading uses `torch.load(..., weights_only=True)`. Whole pickled model objects, extra parameters, mismatched shapes, non-FP32 weights and non-finite weights are rejected. The training command saves architecture and normalization metadata; compatible external checkpoints without that metadata require you to verify those properties yourself.

## Compare and benchmark

```bash
python python/compare.py \
  --checkpoint python/artifacts/mnist.pt \
  --batch-sizes 1 32 128 \
  --samples 10000 --warmup 50 --iterations 200 \
  --output-dir python/artifacts/comparison
```

Use `--download` if the MNIST test data has not been downloaded. `--device 0` selects the same logical CUDA device in both processes. For a quick smoke run, use `--samples 256 --batch-sizes 1 32 --warmup 10 --iterations 20`. `--eager-only` skips compilation and explicitly narrows the baseline coverage. Existing checkpoints accept the same `--prefixes` option here.

The comparison uses the first N images of the official held-out test split without shuffling. Every implementation sees identical normalized images and exported weights. Correctness is checked on all N images in the requested batches, including a final partial batch. Timing repeats the first full batch. Training, process startup, disk I/O, preprocessing, checkpoint loading, compilation and warmup are excluded from reported latency.

Two measurements are reported separately:

- **Forward:** CUDA event interval around the model with input transfer excluded. This includes the native runtime's allocations, intermediate cleanup, scheduling gaps and input ownership behavior. It is not an isolated sum of kernel execution times. PyTorch uses its caching allocator; CudaRT uses `cudaMalloc`/`cudaFree` and consumes the input tensor. These are intentional properties of the implementations being compared.
- **Request:** wall time starting with a preprocessed CPU batch, including fresh GPU input allocation, H2D transfer, forward inference, blocking D2H output transfer and cleanup. Native allocator behavior is included; images are not pinned in either implementation. This is an in-process inference request, not application startup or raw-image decoding.

The compiled candidates run under `torch.inference_mode`, and compilation and graph warmup happen before timing. Both sides use IEEE FP32 matmul, without TF32 or mixed precision. Each candidate's logits are validated against eager FP32 PyTorch. Defaults are `atol=1e-4`, `rtol=1e-4`; predicted classes must agree exactly. Floating-point accumulation orders differ, so bitwise equality is not expected. Tolerances and maximum errors are recorded, and numerical failures exit unsuccessfully and suppress the resume wording.

The report chooses the fastest **passing, tested** PyTorch candidate separately for forward and request latency. Compiled candidates that are unavailable are recorded with their error; the report does not silently replace them or call an eager-only result the global best. Full GPU runtime errors still fail the command. Baseline availability, device, batch size, precision and tested modes must accompany any performance claim.

## Artifacts and resume use

`python/artifacts/comparison/` contains:

- `model.crt`: the exported weights loaded by the C++ runner.
- `images.bin` and `labels.npy`: the exact evaluation inputs and labels.
- `cudart_b*.json`: all custom logits and raw timing samples.
- `pytorch*_logits_b*.npy`: reference and compiled logits.
- `report.json`: medians, p95, mean, standard deviation, throughput, agreement, accuracy, raw samples, hashes, software versions, CMake flags and run completion state.
- `report.md`: comparison table, measured ratios and factual resume wording generated only after a completed passing run.

Speedup is `PyTorch latency / CudaRT latency`: below 1 means the custom version is slower. Request throughput is `batch_size / median_request_seconds`, not a measurement of an asynchronously saturated serving pipeline. A slow custom implementation can still support a strong resume claim about runtime design and numerical validation. The report does not invent an improvement.

Run without unrelated GPU workloads, preserve the raw reports, and repeat measurements to assess run-to-run variability before quoting a speedup. Timing distributions within a single run are not confidence intervals. This suite compares the matching FP32 MLP; it does not establish the best MNIST classifier or compare TensorRT, other architectures, quantization or mixed precision.

To exercise the original single-image C++ interface, extract one image and use the exported weights:

```bash
python -c 'from pathlib import Path; p=Path("python/artifacts/comparison"); (p/"image.bin").write_bytes((p/"images.bin").read_bytes()[:784])'
./build/cudart_example python/artifacts/comparison/model.crt python/artifacts/comparison/image.bin
```

## CRT1 wire format

All integers and float data are little-endian, with no padding:

```text
"CRT1"                              4 bytes
parameter_count                      uint32 (4)
for each parameter:
    UTF-8 name length                 uint32
    UTF-8 name                        name_length bytes
    dimension count                   uint32
    dimensions                        ndim × uint64
    row-major tensor values           product(dimensions) × float32
```

Names and shapes are `linear0.weight [128,784]`, `linear0.bias [128]`, `linear1.weight [10,128]`, `linear1.bias [10]`. No weight transpose is performed: both implementations index weights `[out_features,in_features]`. The resulting file is 407,220 bytes. The C++ loader reads float payloads directly on the little-endian CUDA host.

## Verification

```bash
python -m unittest discover -s python -p 'test_*.py' -v
```

The wire-format tests use only the standard library. Checkpoint/inference and comparison tests also run when dependencies are installed; they do not require a GPU. Full cross-runtime validation requires the CUDA comparison command above.

The measurement choices follow [PyTorch CUDA timing guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html), [FP32 precision controls](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html), and [numerical accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
