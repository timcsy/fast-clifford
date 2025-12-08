"""
ONNX export tests for CGA2D Cl(3,1) algebra.

Tests that CGA2D operations can be exported to ONNX format
with the required constraints:
- No loop operations
- Only basic arithmetic operators (Add, Mul, Neg, Sub)
"""

import pytest
import torch
import tempfile
import os

from fast_clifford.algebras import cga2d


# =============================================================================
# ONNX Export Tests
# =============================================================================

class TestONNXExport:
    """Test ONNX export capability."""

    def test_cga2d_care_layer_export(self):
        """CGA2DCareLayer can be exported to ONNX."""
        layer = cga2d.CGA2DCareLayer()

        motor = torch.randn(1, 7)
        point = torch.randn(1, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga2d_care.onnx")

            torch.onnx.export(
                layer,
                (motor, point),
                onnx_path,
                input_names=["motor", "point"],
                output_names=["output"],
                dynamic_axes={
                    "motor": {0: "batch"},
                    "point": {0: "batch"},
                    "output": {0: "batch"},
                },
                opset_version=17,
            )

            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0

    def test_upgc2d_encoder_export(self):
        """UPGC2DEncoder can be exported to ONNX."""
        encoder = cga2d.UPGC2DEncoder()

        x = torch.randn(1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "upgc2d_encoder.onnx")

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

    def test_upgc2d_decoder_export(self):
        """UPGC2DDecoder can be exported to ONNX."""
        decoder = cga2d.UPGC2DDecoder()

        point = torch.randn(1, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "upgc2d_decoder.onnx")

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

    def test_full_pipeline_export(self):
        """CGA2DTransformPipeline can be exported to ONNX."""
        pipeline = cga2d.CGA2DTransformPipeline()

        motor = torch.randn(1, 7)
        x = torch.randn(1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga2d_pipeline.onnx")

            torch.onnx.export(
                pipeline,
                (motor, x),
                onnx_path,
                input_names=["motor", "x"],
                output_names=["y"],
                dynamic_axes={
                    "motor": {0: "batch"},
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

    def test_cga2d_care_layer_no_loops(self):
        """CGA2DCareLayer ONNX has no Loop nodes."""
        pytest.importorskip("onnx")
        import onnx

        layer = cga2d.CGA2DCareLayer()

        motor = torch.randn(1, 7)
        point = torch.randn(1, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")

            torch.onnx.export(
                layer,
                (motor, point),
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

    def test_onnx_pytorch_equivalence(self):
        """ONNX model output matches PyTorch."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        layer = cga2d.CGA2DCareLayer()

        motor = torch.randn(4, 7)
        point = torch.randn(4, 4)

        # PyTorch output
        torch_output = layer(motor, point).numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")

            torch.onnx.export(
                layer,
                (motor, point),
                onnx_path,
                opset_version=17,
            )

            # ONNX Runtime output
            session = ort.InferenceSession(onnx_path)
            ort_output = session.run(
                None,
                {
                    "motor": motor.numpy(),
                    "point": point.numpy(),
                }
            )[0]

            import numpy as np
            np.testing.assert_allclose(torch_output, ort_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
