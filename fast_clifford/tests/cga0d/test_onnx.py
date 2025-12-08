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

from fast_clifford.cga import CGA
from fast_clifford.cga.layers import CliffordTransformLayer, CGAEncoder, CGAPipeline


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

    @pytest.fixture
    def algebra(self):
        """Get CGA0D algebra instance."""
        return CGA(0)

    def test_care_layer_onnx_export(self, onnx_available, algebra):
        """Test that CliffordTransformLayer can be exported to ONNX without Loop nodes."""
        import onnx

        layer = CliffordTransformLayer(algebra)

        # Create sample inputs
        ev = torch.randn(1, 2)
        point = torch.randn(1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_care.onnx")

            # Export to ONNX
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

            # Load and verify no Loop nodes
            model = onnx.load(onnx_path)
            op_types = {n.op_type for n in model.graph.node}

            assert "Loop" not in op_types, f"Found Loop node in ONNX graph: {op_types}"

    def test_care_layer_onnx_numerical_consistency(self, onnx_available, algebra):
        """Test that ONNX export produces numerically consistent results."""
        import onnx
        import onnxruntime as ort

        layer = CliffordTransformLayer(algebra)

        # Create sample inputs
        ev = torch.randn(8, 2)
        point = torch.randn(8, 2)

        # PyTorch computation
        with torch.no_grad():
            pytorch_output = layer(ev, point)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_care.onnx")

            # Export to ONNX
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

            # ONNX Runtime inference
            session = ort.InferenceSession(onnx_path)
            ort_output = session.run(
                None,
                {
                    "ev": ev.numpy(),
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

    def test_encoder_onnx_export(self, onnx_available, algebra):
        """Test that CGAEncoder can be exported to ONNX."""
        import onnx

        encoder = CGAEncoder(algebra)

        # Create sample input (CGA0D has 0 euclidean dimensions)
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

    def test_transform_pipeline_onnx_export(self, onnx_available, algebra):
        """Test that CGAPipeline can be exported to ONNX."""
        import onnx

        pipeline = CGAPipeline(algebra)

        # Create sample inputs
        ev = torch.randn(1, 2)
        x = torch.randn(1, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "cga0d_pipeline.onnx")

            # Export to ONNX
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

            # Load and verify no Loop nodes
            model = onnx.load(onnx_path)
            op_types = {n.op_type for n in model.graph.node}

            assert "Loop" not in op_types
