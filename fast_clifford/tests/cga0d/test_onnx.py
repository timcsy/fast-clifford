"""
CGA0D ONNX Export Tests

Tests for ONNX compatibility:
- Export without Loop nodes
- Numerical consistency after export
"""

import pytest
import torch
import tempfile
import os

from fast_clifford.algebras import cga0d


class TestONNXExport:
    """Test ONNX export functionality."""

    @pytest.fixture
    def onnx_available(self):
        """Check if ONNX is available."""
        try:
            import onnx
            import onnxruntime
            return True
        except ImportError:
            pytest.skip("onnx or onnxruntime not available")

    def test_care_layer_onnx_export(self, onnx_available):
        """Test that CGA0DCareLayer can be exported to ONNX without Loop nodes."""
        import onnx

        layer = cga0d.CGA0DCareLayer()

        # Create sample inputs
        motor = torch.randn(1, 2)
        point = torch.randn(1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_care.onnx")

            # Export to ONNX
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

            # Load and verify no Loop nodes
            model = onnx.load(onnx_path)
            op_types = {n.op_type for n in model.graph.node}

            assert "Loop" not in op_types, f"Found Loop node in ONNX graph: {op_types}"

    def test_care_layer_onnx_numerical_consistency(self, onnx_available):
        """Test that ONNX export produces numerically consistent results."""
        import onnx
        import onnxruntime as ort

        layer = cga0d.CGA0DCareLayer()

        # Create sample inputs
        motor = torch.randn(8, 2)
        point = torch.randn(8, 2)

        # PyTorch computation
        with torch.no_grad():
            pytorch_output = layer(motor, point)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_care.onnx")

            # Export to ONNX
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

            # ONNX Runtime inference
            session = ort.InferenceSession(onnx_path)
            ort_output = session.run(
                None,
                {
                    "motor": motor.numpy(),
                    "point": point.numpy(),
                }
            )[0]

            # Compare results
            assert pytorch_output.numpy().shape == ort_output.shape
            assert torch.allclose(
                pytorch_output,
                torch.tensor(ort_output),
                atol=1e-5
            )

    def test_encoder_onnx_export(self, onnx_available):
        """Test that UPGC0DEncoder can be exported to ONNX."""
        import onnx

        encoder = cga0d.UPGC0DEncoder()

        # Create sample input
        x = torch.randn(1, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_encoder.onnx")

            # Export to ONNX
            torch.onnx.export(
                encoder,
                (x,),
                onnx_path,
                input_names=["x"],
                output_names=["point"],
                dynamic_axes={
                    "x": {0: "batch"},
                    "point": {0: "batch"},
                },
                opset_version=17,
            )

            # Load and verify no Loop nodes
            model = onnx.load(onnx_path)
            op_types = {n.op_type for n in model.graph.node}

            assert "Loop" not in op_types

    def test_transform_pipeline_onnx_export(self, onnx_available):
        """Test that CGA0DTransformPipeline can be exported to ONNX."""
        import onnx

        pipeline = cga0d.CGA0DTransformPipeline()

        # Create sample inputs
        motor = torch.randn(1, 2)
        x = torch.randn(1, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_pipeline.onnx")

            # Export to ONNX
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

            # Load and verify no Loop nodes
            model = onnx.load(onnx_path)
            op_types = {n.op_type for n in model.graph.node}

            assert "Loop" not in op_types
