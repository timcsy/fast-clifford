"""
Runtime CGA Tests

Tests for RuntimeCGAAlgebra (CGA6D+):
- Basic functionality
- Numerical correctness (vs clifford library)
- Gradient computation
- ONNX export
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from fast_clifford import CGA
from fast_clifford.cga.runtime import RuntimeCGAAlgebra


class TestRuntimeCGABasic:
    """Basic tests for RuntimeCGAAlgebra."""

    def test_cga6_blade_count(self):
        """Test CGA(6) has 256 blades."""
        cga = CGA(6)
        assert cga.blade_count == 256

    def test_cga6_point_count(self):
        """Test CGA(6) has 8 point components."""
        cga = CGA(6)
        assert cga.point_count == 8

    def test_cga6_signature(self):
        """Test CGA(6) has correct signature."""
        cga = CGA(6)
        # Cl(7, 1): 7 positive, 1 negative
        expected = tuple([1] * 7 + [-1])
        assert cga.signature == expected

    def test_cga6_clifford_notation(self):
        """Test CGA(6) clifford notation."""
        cga = CGA(6)
        assert cga.clifford_notation == "Cl(7,1,0)"

    def test_cga6_returns_runtime_algebra(self):
        """Test CGA(6) returns RuntimeCGAAlgebra."""
        cga = CGA(6)
        assert isinstance(cga, RuntimeCGAAlgebra)


class TestRuntimeCGAOperations:
    """Test RuntimeCGAAlgebra operations."""

    @pytest.fixture
    def cga6(self):
        """Create CGA(6) algebra."""
        return CGA(6)

    def test_upgc_encode_shape(self, cga6):
        """Test that upgc_encode produces correct shape."""
        batch_size = 4
        x = torch.randn(batch_size, 6)
        point = cga6.upgc_encode(x)
        assert point.shape == (batch_size, 8)

    def test_upgc_decode_shape(self, cga6):
        """Test that upgc_decode produces correct shape."""
        batch_size = 4
        point = torch.randn(batch_size, 8)
        x = cga6.upgc_decode(point)
        assert x.shape == (batch_size, 6)

    def test_upgc_roundtrip(self, cga6):
        """Test encode/decode roundtrip preserves Euclidean coordinates."""
        batch_size = 4
        x = torch.randn(batch_size, 6)
        point = cga6.upgc_encode(x)
        x_decoded = cga6.upgc_decode(point)
        assert torch.allclose(x, x_decoded, atol=1e-5)

    def test_geometric_product_shape(self, cga6):
        """Test that geometric_product_full produces correct shape."""
        batch_size = 4
        a = torch.randn(batch_size, 256)
        b = torch.randn(batch_size, 256)
        result = cga6.geometric_product_full(a, b)
        assert result.shape == (batch_size, 256)

    def test_geometric_product_identity(self, cga6):
        """Test that multiplying by identity preserves value."""
        batch_size = 4
        a = torch.randn(batch_size, 256)
        identity = torch.zeros(batch_size, 256)
        identity[..., 0] = 1.0  # scalar = 1

        result = cga6.geometric_product_full(identity, a)
        assert torch.allclose(result, a, atol=1e-5)

    def test_sandwich_product_shape(self, cga6):
        """Test that sandwich_product_sparse produces correct shape."""
        batch_size = 4
        motor = torch.randn(batch_size, cga6.motor_count)
        point = torch.randn(batch_size, cga6.point_count)
        result = cga6.sandwich_product_sparse(motor, point)
        assert result.shape == (batch_size, 8)

    def test_sandwich_product_identity_motor(self, cga6):
        """Test that identity motor preserves point."""
        batch_size = 4

        # Create identity motor (scalar = 1, rest = 0)
        motor = torch.zeros(batch_size, cga6.motor_count)
        motor[..., 0] = 1.0

        point = torch.randn(batch_size, cga6.point_count)
        result = cga6.sandwich_product_sparse(motor, point)

        assert torch.allclose(result, point, atol=1e-5)

    def test_reverse_full_shape(self, cga6):
        """Test that reverse_full produces correct shape."""
        batch_size = 4
        mv = torch.randn(batch_size, 256)
        result = cga6.reverse_full(mv)
        assert result.shape == (batch_size, 256)

    def test_reverse_motor_shape(self, cga6):
        """Test that reverse_motor produces correct shape."""
        batch_size = 4
        motor = torch.randn(batch_size, cga6.motor_count)
        result = cga6.reverse_motor(motor)
        assert result.shape == motor.shape


class TestRuntimeCGALayers:
    """Test RuntimeCGAAlgebra layer factories."""

    @pytest.fixture
    def cga6(self):
        return CGA(6)

    def test_get_care_layer(self, cga6):
        """Test get_care_layer returns working module."""
        layer = cga6.get_care_layer()
        assert isinstance(layer, torch.nn.Module)

        motor = torch.randn(4, cga6.motor_count)
        point = torch.randn(4, cga6.point_count)
        result = layer(motor, point)
        assert result.shape == (4, cga6.point_count)

    def test_get_encoder(self, cga6):
        """Test get_encoder returns working module."""
        encoder = cga6.get_encoder()
        assert isinstance(encoder, torch.nn.Module)

        x = torch.randn(4, 6)
        point = encoder(x)
        assert point.shape == (4, 8)

    def test_get_decoder(self, cga6):
        """Test get_decoder returns working module."""
        decoder = cga6.get_decoder()
        assert isinstance(decoder, torch.nn.Module)

        point = torch.randn(4, 8)
        x = decoder(point)
        assert x.shape == (4, 6)

    def test_get_transform_pipeline(self, cga6):
        """Test get_transform_pipeline returns working module."""
        pipeline = cga6.get_transform_pipeline()
        assert isinstance(pipeline, torch.nn.Module)

        motor = torch.randn(4, cga6.motor_count)
        x = torch.randn(4, 6)
        y = pipeline(motor, x)
        assert y.shape == (4, 6)


class TestRuntimeCGAGradients:
    """Test gradient computation for RuntimeCGAAlgebra."""

    @pytest.fixture
    def cga6(self):
        return CGA(6)

    def test_geometric_product_gradient(self, cga6):
        """Test that geometric product supports gradients."""
        a = torch.randn(4, 256, requires_grad=True)
        b = torch.randn(4, 256, requires_grad=True)

        result = cga6.geometric_product_full(a, b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

    def test_sandwich_product_gradient(self, cga6):
        """Test that sandwich product supports gradients."""
        motor = torch.randn(4, cga6.motor_count, requires_grad=True)
        point = torch.randn(4, cga6.point_count, requires_grad=True)

        result = cga6.sandwich_product_sparse(motor, point)
        loss = result.sum()
        loss.backward()

        assert motor.grad is not None
        assert point.grad is not None

    def test_transform_pipeline_gradient(self, cga6):
        """Test that transform pipeline supports gradients."""
        pipeline = cga6.get_transform_pipeline()

        motor = torch.randn(4, cga6.motor_count, requires_grad=True)
        x = torch.randn(4, 6, requires_grad=True)

        y = pipeline(motor, x)
        loss = y.sum()
        loss.backward()

        assert motor.grad is not None
        assert x.grad is not None


class TestRuntimeCGAONNX:
    """Test ONNX export for RuntimeCGAAlgebra."""

    @pytest.fixture
    def onnx_available(self):
        try:
            import onnx
            import onnxruntime
            return True
        except ImportError:
            pytest.skip("onnx or onnxruntime not available")

    @pytest.fixture
    def cga6(self):
        return CGA(6)

    def test_care_layer_onnx_no_loop(self, onnx_available, cga6):
        """Test that CareLayer ONNX export has no Loop nodes."""
        import onnx

        layer = cga6.get_care_layer()

        motor = torch.randn(1, cga6.motor_count)
        point = torch.randn(1, cga6.point_count)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "runtime_cga6_care.onnx")

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

            model = onnx.load(onnx_path)
            op_types = {n.op_type for n in model.graph.node}

            assert "Loop" not in op_types, f"Found Loop in ONNX: {op_types}"

    def test_care_layer_onnx_numerical(self, onnx_available, cga6):
        """Test ONNX numerical consistency."""
        import onnx
        import onnxruntime as ort

        layer = cga6.get_care_layer()

        motor = torch.randn(4, cga6.motor_count)
        point = torch.randn(4, cga6.point_count)

        with torch.no_grad():
            pytorch_output = layer(motor, point)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "runtime_cga6_care.onnx")

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

            session = ort.InferenceSession(onnx_path)
            ort_output = session.run(
                None,
                {
                    "motor": motor.numpy(),
                    "point": point.numpy(),
                }
            )[0]

            assert torch.allclose(
                pytorch_output,
                torch.tensor(ort_output),
                atol=1e-4
            )


class TestRuntimeCGACliffordVerification:
    """Verify RuntimeCGAAlgebra against clifford library."""

    @pytest.fixture
    def clifford_cga6(self):
        try:
            from clifford import Cl, conformalize
            G_6, _ = Cl(6)
            layout, blades, stuff = conformalize(G_6)
            return layout, blades, stuff
        except ImportError:
            pytest.skip("clifford library not available")

    @pytest.fixture
    def cga6(self):
        return CGA(6)

    def test_geometric_product_vs_clifford(self, clifford_cga6, cga6):
        """Compare geometric product with clifford library (small sample)."""
        layout, blades, stuff = clifford_cga6

        np.random.seed(42)
        a_values = np.random.randn(256)
        b_values = np.random.randn(256)

        # Clifford computation
        a_cliff = layout.MultiVector(value=a_values)
        b_cliff = layout.MultiVector(value=b_values)
        result_cliff = a_cliff * b_cliff

        # PyTorch computation
        a_torch = torch.tensor(a_values, dtype=torch.float32).unsqueeze(0)
        b_torch = torch.tensor(b_values, dtype=torch.float32).unsqueeze(0)
        result_torch = cga6.geometric_product_full(a_torch, b_torch)

        assert torch.allclose(
            result_torch.squeeze(0),
            torch.tensor(result_cliff.value, dtype=torch.float32),
            atol=1e-4
        )

    def test_upgc_encode_vs_clifford(self, clifford_cga6, cga6):
        """Compare UPGC encoding with clifford library."""
        layout, blades, stuff = clifford_cga6
        up_func = stuff['up']

        np.random.seed(42)
        x_values = np.random.randn(6)

        # Clifford computation
        x_mv = sum(x_values[i] * blades[f'e{i+1}'] for i in range(6))
        point_cliff = up_func(x_mv)

        # Extract Grade 1 components
        grade_1_indices = [i for i in range(256) if len(layout.bladeTupList[i]) == 1]
        point_cliff_sparse = np.array([point_cliff.value[i] for i in grade_1_indices])

        # PyTorch computation
        x_torch = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0)
        point_torch = cga6.upgc_encode(x_torch)

        assert torch.allclose(
            point_torch.squeeze(0),
            torch.tensor(point_cliff_sparse, dtype=torch.float32),
            atol=1e-4
        )
