"""
ONNX export tests for CGA1D Cl(2,1) algebra.

Tests that CGA1D operations can be exported to ONNX format
with the required constraints:
- No loop operations
- Only basic arithmetic operators (Add, Mul, Neg, Sub)
"""

import pytest
import torch
import tempfile
import os

from fast_clifford.cga import CGA
from fast_clifford.cga.layers import CliffordTransformLayer, CGAEncoder, CGADecoder, CGAPipeline


# =============================================================================
# ONNX Export Tests
# =============================================================================

class TestONNXExport:
    """Test ONNX export capability."""

    @pytest.fixture
    def algebra(self):
        """Get CGA1D algebra instance."""
        return CGA(1)

    def test_cga1d_transform_layer_export(self, algebra):
        """CliffordTransformLayer can be exported to ONNX."""
        layer = CliffordTransformLayer(algebra)

        ev = torch.randn(1, 4)
        point = torch.randn(1, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga1d_care.onnx")

            torch.onnx.export(
                layer,
                (ev, point),
                onnx_path,
                input_names=["ev", "point"],
                output_names=["output"],
                dynamic_axes={
                    "ev": {0: "batch"},
                    "point": {0: "batch"},
                    "output": {0: "batch"},
                },
                opset_version=17,
            )

            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0

    def test_upgc1d_encoder_export(self, algebra):
        """CGAEncoder can be exported to ONNX."""
        encoder = CGAEncoder(algebra)

        x = torch.randn(1, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "upgc1d_encoder.onnx")

            torch.onnx.export(
                encoder,
                x,
                onnx_path,
                input_names=["x"],
                output_names=["point"],
                dynamic_axes={
                    "x": {0: "batch"},
                    "point": {0: "batch"},
                },
                opset_version=17,
            )

            assert os.path.exists(onnx_path)

    def test_upgc1d_decoder_export(self, algebra):
        """CGADecoder can be exported to ONNX."""
        decoder = CGADecoder(algebra)

        point = torch.randn(1, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "upgc1d_decoder.onnx")

            torch.onnx.export(
                decoder,
                point,
                onnx_path,
                input_names=["point"],
                output_names=["x"],
                dynamic_axes={
                    "point": {0: "batch"},
                    "x": {0: "batch"},
                },
                opset_version=17,
            )

            assert os.path.exists(onnx_path)

    def test_full_pipeline_export(self, algebra):
        """CGAPipeline can be exported to ONNX."""
        pipeline = CGAPipeline(algebra)

        ev = torch.randn(1, 4)
        x = torch.randn(1, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga1d_pipeline.onnx")

            torch.onnx.export(
                pipeline,
                (ev, x),
                onnx_path,
                input_names=["ev", "x"],
                output_names=["y"],
                dynamic_axes={
                    "ev": {0: "batch"},
                    "x": {0: "batch"},
                    "y": {0: "batch"},
                },
                opset_version=17,
            )

            assert os.path.exists(onnx_path)


# =============================================================================
# No Loops Tests
# =============================================================================

class TestONNXNoLoops:
    """Test that ONNX exports contain no loop operations."""

    @pytest.fixture
    def algebra(self):
        """Get CGA1D algebra instance."""
        return CGA(1)

    def test_cga1d_transform_layer_no_loops(self, algebra):
        """CliffordTransformLayer ONNX has no Loop nodes."""
        pytest.importorskip("onnx")
        import onnx

        layer = CliffordTransformLayer(algebra)

        ev = torch.randn(1, 4)
        point = torch.randn(1, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")

            torch.onnx.export(
                layer,
                (ev, point),
                onnx_path,
                opset_version=17,
            )

            model = onnx.load(onnx_path)
            op_types = {node.op_type for node in model.graph.node}

            assert "Loop" not in op_types
            assert "Scan" not in op_types
            assert "SequenceConstruct" not in op_types


# =============================================================================
# Numerical Equivalence Tests
# =============================================================================

class TestONNXNumericalEquivalence:
    """Test ONNX output matches PyTorch output."""

    @pytest.fixture
    def algebra(self):
        """Get CGA1D algebra instance."""
        return CGA(1)

    def test_onnx_pytorch_equivalence(self, algebra):
        """ONNX model output matches PyTorch."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        layer = CliffordTransformLayer(algebra)

        ev = torch.randn(4, 4)
        point = torch.randn(4, 3)

        # PyTorch output
        torch_output = layer(ev, point).numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")

            torch.onnx.export(
                layer,
                (ev, point),
                onnx_path,
                input_names=["versor", "point"],
                output_names=["output"],
                opset_version=17,
            )

            # ONNX Runtime output
            session = ort.InferenceSession(onnx_path)
            ort_output = session.run(
                None,
                {
                    "versor": ev.numpy(),
                    "point": point.numpy(),
                }
            )[0]

            import numpy as np
            np.testing.assert_allclose(torch_output, ort_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
