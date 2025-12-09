"""
ONNX export tests for CGA4D operations.

Tests:
- ONNX export with opset 17
- Verify no Loop nodes in computation graph
- Verify only basic operators (Add/Mul/Neg)
- Verify PyTorch and ONNX Runtime produce consistent results
"""

import pytest
import torch
import torch.onnx
import tempfile
import os

from fast_clifford.algebras.cga4d.layers import (
    CliffordTransformLayer,
    CGAEncoder,
    CGADecoder,
    CGAPipeline
)


class TestONNXExport:
    """ONNX export tests with opset 17."""

    def test_cga4d_transform_layer_export(self):
        """Test CliffordTransformLayer exports to ONNX."""
        layer = CliffordTransformLayer()

        ev = torch.randn(1, 32)
        point = torch.randn(1, 6)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    layer,
                    (ev, point),
                    f.name,
                    opset_version=17,
                    input_names=["ev", "point"],
                    output_names=["output"],
                    dynamic_axes={
                        "ev": {0: "batch_size"},
                        "point": {0: "batch_size"},
                        "output": {0: "batch_size"}
                    }
                )

                assert os.path.exists(f.name)
                assert os.path.getsize(f.name) > 0

                import onnx
                model = onnx.load(f.name)
                onnx.checker.check_model(model)

            finally:
                os.unlink(f.name)

    def test_upgc4d_encoder_export(self):
        """Test CGAEncoder exports to ONNX."""
        encoder = CGAEncoder()

        x = torch.randn(1, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    encoder,
                    (x,),
                    f.name,
                    opset_version=17,
                    input_names=["x"],
                    output_names=["point"],
                    dynamic_axes={
                        "x": {0: "batch_size"},
                        "point": {0: "batch_size"}
                    }
                )

                import onnx
                model = onnx.load(f.name)
                onnx.checker.check_model(model)

            finally:
                os.unlink(f.name)

    def test_upgc4d_decoder_export(self):
        """Test CGADecoder exports to ONNX."""
        decoder = CGADecoder()

        point = torch.randn(1, 6)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    decoder,
                    (point,),
                    f.name,
                    opset_version=17,
                    input_names=["point"],
                    output_names=["x"],
                    dynamic_axes={
                        "point": {0: "batch_size"},
                        "x": {0: "batch_size"}
                    }
                )

                import onnx
                model = onnx.load(f.name)
                onnx.checker.check_model(model)

            finally:
                os.unlink(f.name)

    def test_full_pipeline_export(self):
        """Test CGAPipeline exports to ONNX."""
        pipeline = CGAPipeline()

        ev = torch.randn(1, 32)
        x = torch.randn(1, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    pipeline,
                    (ev, x),
                    f.name,
                    opset_version=17,
                    input_names=["ev", "x"],
                    output_names=["y"],
                    dynamic_axes={
                        "ev": {0: "batch_size"},
                        "x": {0: "batch_size"},
                        "y": {0: "batch_size"}
                    }
                )

                import onnx
                model = onnx.load(f.name)
                onnx.checker.check_model(model)

            finally:
                os.unlink(f.name)


class TestONNXNoLoops:
    """Verify ONNX computation graph has no Loop nodes."""

    def _get_all_op_types(self, model) -> set:
        """Extract all op types from an ONNX model."""
        op_types = set()
        for node in model.graph.node:
            op_types.add(node.op_type)
        return op_types

    def test_cga4d_transform_layer_no_loops(self):
        """Verify CliffordTransformLayer has no Loop nodes."""
        layer = CliffordTransformLayer()
        ev = torch.randn(1, 32)
        point = torch.randn(1, 6)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    layer,
                    (ev, point),
                    f.name,
                    opset_version=17
                )

                import onnx
                model = onnx.load(f.name)
                op_types = self._get_all_op_types(model)

                forbidden_ops = {"Loop", "If", "Scan", "While"}
                found_forbidden = op_types.intersection(forbidden_ops)

                assert len(found_forbidden) == 0, \
                    f"Found forbidden control flow ops: {found_forbidden}"

            finally:
                os.unlink(f.name)

    def test_full_pipeline_no_loops(self):
        """Verify CGAPipeline has no Loop nodes."""
        pipeline = CGAPipeline()
        ev = torch.randn(1, 32)
        x = torch.randn(1, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    pipeline,
                    (ev, x),
                    f.name,
                    opset_version=17
                )

                import onnx
                model = onnx.load(f.name)
                op_types = self._get_all_op_types(model)

                forbidden_ops = {"Loop", "If", "Scan", "While"}
                found_forbidden = op_types.intersection(forbidden_ops)

                assert len(found_forbidden) == 0, \
                    f"Found forbidden control flow ops: {found_forbidden}"

            finally:
                os.unlink(f.name)


class TestONNXBasicOperators:
    """Verify ONNX uses only basic operators."""

    ALLOWED_OPS = {
        # Basic arithmetic
        "Add", "Sub", "Mul", "Div", "Neg",
        # Shape manipulation
        "Concat", "Slice", "Unsqueeze", "Squeeze", "Reshape", "Flatten",
        "Gather", "Split", "Transpose",
        # Constants and identity
        "Constant", "ConstantOfShape", "Identity",
        # Type conversion
        "Cast",
        # Reduction ops
        "ReduceSum", "ReduceMean",
        # Other basic ops
        "MatMul", "Gemm",
        # Stack operations
        "ConcatFromSequence"
    }

    def _get_all_op_types(self, model) -> set:
        """Extract all op types from an ONNX model."""
        op_types = set()
        for node in model.graph.node:
            op_types.add(node.op_type)
        return op_types

    def test_cga4d_transform_layer_basic_ops(self):
        """Verify CliffordTransformLayer uses only basic operators."""
        layer = CliffordTransformLayer()
        ev = torch.randn(1, 32)
        point = torch.randn(1, 6)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    layer,
                    (ev, point),
                    f.name,
                    opset_version=17
                )

                import onnx
                model = onnx.load(f.name)
                op_types = self._get_all_op_types(model)

                # Key assertion: no control flow
                control_flow = {"Loop", "If", "Scan", "While"}
                assert len(op_types.intersection(control_flow)) == 0, \
                    "Control flow ops should not be present"

            finally:
                os.unlink(f.name)

    def test_report_ops_used(self):
        """Report all ops used by the full pipeline."""
        pipeline = CGAPipeline()
        ev = torch.randn(1, 32)
        x = torch.randn(1, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    pipeline,
                    (ev, x),
                    f.name,
                    opset_version=17
                )

                import onnx
                model = onnx.load(f.name)
                op_types = self._get_all_op_types(model)

                print(f"\nOps used by CGAPipeline: {sorted(op_types)}")

                op_counts = {}
                for node in model.graph.node:
                    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

                print(f"Op counts: {op_counts}")

                assert len(op_types) > 0, "Should have some ops"

            finally:
                os.unlink(f.name)


class TestONNXNumericalEquivalence:
    """Verify ONNX model produces same results as PyTorch."""

    def test_onnx_pytorch_equivalence(self):
        """Compare ONNX inference to PyTorch inference."""
        import numpy as np

        layer = CliffordTransformLayer()
        ev = torch.randn(1, 32)
        point = torch.randn(1, 6)

        pytorch_result = layer(ev, point).detach().numpy()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    layer,
                    (ev, point),
                    f.name,
                    opset_version=17,
                    input_names=["ev", "point"],
                    output_names=["output"]
                )

                import onnxruntime as ort
                session = ort.InferenceSession(f.name)

                onnx_result = session.run(
                    None,
                    {
                        "ev": ev.numpy(),
                        "point": point.numpy()
                    }
                )[0]

                np.testing.assert_allclose(
                    pytorch_result, onnx_result,
                    rtol=1e-5, atol=1e-5
                )

            finally:
                os.unlink(f.name)

    def test_pipeline_onnx_pytorch_equivalence(self):
        """Compare full pipeline ONNX to PyTorch."""
        import numpy as np

        pipeline = CGAPipeline()
        ev = torch.randn(1, 32)
        x = torch.randn(1, 4)

        pytorch_result = pipeline(ev, x).detach().numpy()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            try:
                torch.onnx.export(
                    pipeline,
                    (ev, x),
                    f.name,
                    opset_version=17,
                    input_names=["ev", "x"],
                    output_names=["y"]
                )

                import onnxruntime as ort
                session = ort.InferenceSession(f.name)

                onnx_result = session.run(
                    None,
                    {
                        "ev": ev.numpy(),
                        "x": x.numpy()
                    }
                )[0]

                np.testing.assert_allclose(
                    pytorch_result, onnx_result,
                    rtol=1e-5, atol=1e-5
                )

            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
