"""Run with python3 -m unittest discover -s python -p 'test_*.py'."""
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest

from crt import PARAMETER_SHAPES, write_crt


class FormatTests(unittest.TestCase):
    def parameters(self):
        return {name: (shape, [0.25] * math.prod(shape)) for name, shape in PARAMETER_SHAPES.items()}

    def test_exact_loader_wire_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.crt"
            write_crt(path, self.parameters())
            payload = path.read_bytes()
        self.assertEqual(payload[:8], b"CRT1\x04\x00\x00\x00")
        offset = 8
        for expected_name, expected_shape in PARAMETER_SHAPES.items():
            length, = struct.unpack_from("<I", payload, offset)
            offset += 4
            name = payload[offset:offset + length].decode("utf-8")
            offset += length
            ndim, = struct.unpack_from("<I", payload, offset)
            offset += 4
            shape = struct.unpack_from(f"<{ndim}Q", payload, offset)
            offset += 8 * ndim
            values = struct.unpack_from(f"<{math.prod(shape)}f", payload, offset)
            offset += 4 * len(values)
            self.assertEqual(name, expected_name)
            self.assertEqual(shape, expected_shape)
            self.assertTrue(all(value == 0.25 for value in values))
        self.assertEqual(offset, len(payload))
        self.assertEqual(len(payload), 407220)

    def test_bad_parameters_are_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.crt"
            cases = []
            missing = self.parameters()
            del missing["linear1.bias"]
            cases.append(missing)
            wrong_shape = self.parameters()
            wrong_shape["linear0.weight"] = ((784, 128), [0.0] * (784 * 128))
            cases.append(wrong_shape)
            wrong_count = self.parameters()
            wrong_count["linear1.bias"] = ((10,), [0.0])
            cases.append(wrong_count)
            not_finite = self.parameters()
            not_finite["linear1.bias"] = ((10,), [float("nan")] * 10)
            cases.append(not_finite)
            for case in cases:
                with self.subTest(case=len(case)), self.assertRaises(ValueError):
                    write_crt(path, case)
                self.assertFalse(path.exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class CheckpointTests(unittest.TestCase):
    def test_existing_named_checkpoint_roundtrip_and_inference(self):
        import torch
        from model import export_model, load_model, make_model
        torch.manual_seed(7)
        original = make_model().eval()
        state = {key.replace("1.", "fc1.").replace("3.", "fc2."): value
                 for key, value in original.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "existing.pt"
            torch.save({"model_state_dict": state}, checkpoint)
            loaded, _ = load_model(checkpoint)
            images = torch.rand(17, 28, 28)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(images), original(images), rtol=0, atol=0)
            crt = Path(directory) / "model.crt"
            export_model(loaded, crt)
            self.assertEqual(crt.read_bytes()[:8], b"CRT1\x04\x00\x00\x00")

    def test_rejects_incompatible_checkpoint(self):
        import torch
        from model import load_model, make_model
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            torch.save({"state_dict": make_model().state_dict(),
                        "metadata": {"preprocessing": "mean/std"}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_model(checkpoint)
            state = make_model().state_dict()
            state["1.weight"] = torch.zeros(64, 784)
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_model(checkpoint)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "numpy")),
                     "PyTorch, torchvision and numpy are needed")
class ComparisonTests(unittest.TestCase):
    def test_prediction_mismatch_fails_even_when_logits_are_close(self):
        import numpy as np
        from compare import correctness
        expected = np.array([[0.50001, 0.5]], dtype=np.float32)
        actual = np.array([[0.5, 0.50001]], dtype=np.float32)
        result = correctness(actual, expected, np.array([0]), atol=1e-4, rtol=1e-4)
        self.assertEqual(result["logits_outside_tolerance"], 0)
        self.assertEqual(result["prediction_mismatches"], 1)
        self.assertFalse(result["passed"])

    def test_nonfinite_output_and_invalid_timings_rejected(self):
        import numpy as np
        from compare import correctness, summarize
        with self.assertRaises(ValueError):
            correctness(np.array([[np.nan]]), np.array([[0.0]]), np.array([0]), 1e-4, 1e-4)
        for timings in ([], [0], [-1], [np.inf]):
            with self.assertRaises(ValueError):
                summarize(timings, 1)
        result = summarize([1, 2, 3], 32)
        self.assertEqual(result["median_ms"], 2.0)
        self.assertEqual(result["images_per_second"], 16000.0)


if __name__ == "__main__":
    unittest.main()
