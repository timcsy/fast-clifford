"""
Tests for Exponential Map (User Story 3)

T077-T085: Test exp_bivector operations for generating rotors.

Tests verify:
- T078: Zero element: exp_bivector(0) == identity
- T079: 90° rotation correctness
- T080: Small angle numerical stability (θ < 1e-10)
- T081: Inverse operation: compose(exp(B), exp(-B)) ≈ identity
- T082: Comparison with clifford library
- T083: Batch dimension support
- T084: ONNX export (no Loop/If nodes)
- T085: Autograd gradient propagation

Note: Current implementation only supports pure rotation bivectors (B² < 0).
"""

import pytest
import torch
import numpy as np


# Test dimensions to cover
TEST_DIMS = [0, 1, 2, 3]


def get_functional_module(dim: int):
    """Dynamically import functional module for a given dimension."""
    if dim == 0:
        from fast_clifford.algebras.cga0d import functional
    elif dim == 1:
        from fast_clifford.algebras.cga1d import functional
    elif dim == 2:
        from fast_clifford.algebras.cga2d import functional
    elif dim == 3:
        from fast_clifford.algebras.cga3d import functional
    elif dim == 4:
        from fast_clifford.algebras.cga4d import functional
    elif dim == 5:
        from fast_clifford.algebras.cga5d import functional
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    return functional


def get_identity_even_versor(dim: int) -> torch.Tensor:
    """Create identity EvenVersor for given CGA dimension."""
    func = get_functional_module(dim)
    even_versor_count = len(func.EVEN_VERSOR_MASK)
    identity = torch.zeros(even_versor_count)
    identity[0] = 1.0  # scalar = 1
    return identity


class TestExpBivectorZero:
    """T078: Zero element tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_zero_bivector_gives_identity(self, dim: int):
        """exp_bivector(0) == identity = (1, 0, 0, ...)"""
        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        B = torch.zeros(bivector_count)
        result = func.exp_bivector(B)
        identity = get_identity_even_versor(dim)

        torch.testing.assert_close(
            result, identity, rtol=1e-5, atol=1e-5,
            msg=f"exp(0) should be identity for dim={dim}"
        )


class TestExpBivector90Rotation:
    """T079: 90° rotation tests."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_90_degree_rotation(self, dim: int):
        """exp(π/4 * e1e2) should produce 90° rotation.

        A rotation by angle θ uses bivector B with magnitude θ/2.
        So 90° = π/2 rotation uses B with magnitude π/4.
        """
        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        # Create a pure e1e2 bivector with angle π/4
        B = torch.zeros(bivector_count)
        B[0] = np.pi / 4  # e1e2 is first grade-2 blade

        # exp(B) = cos(θ) + sin(θ)/θ * B where θ = |B|
        result = func.exp_bivector(B)

        # For pure rotation bivector:
        # R = cos(θ) + sin(θ) * e1e2 (normalized)
        # θ = |B| = π/4
        expected_scalar = np.cos(np.pi / 4)
        expected_bivector = np.sin(np.pi / 4)

        # Check scalar component
        assert abs(result[0].item() - expected_scalar) < 1e-5, \
            f"Scalar component should be cos(π/4)"

        # Check e1e2 component (index 1 in even versor)
        assert abs(result[1].item() - expected_bivector) < 1e-5, \
            f"e1e2 component should be sin(π/4)"


class TestExpBivectorSmallAngle:
    """T080: Small angle numerical stability tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_very_small_angle_no_nan(self, dim: int):
        """θ < 1e-10 should not produce NaN/Inf."""
        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        # Very small bivector
        B = torch.zeros(bivector_count)
        if bivector_count > 0:
            B[0] = 1e-12

        result = func.exp_bivector(B)

        assert not torch.isnan(result).any(), "exp_bivector should not produce NaN"
        assert not torch.isinf(result).any(), "exp_bivector should not produce Inf"

        # Should be close to identity
        identity = get_identity_even_versor(dim)
        torch.testing.assert_close(
            result, identity, rtol=1e-3, atol=1e-3,
            msg="exp(small B) should be close to identity"
        )


class TestExpBivectorInverse:
    """T081: Inverse operation tests."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_exp_negative_is_inverse(self, dim: int):
        """compose(exp(B), exp(-B)) ≈ identity for pure rotation."""
        func = get_functional_module(dim)
        identity = get_identity_even_versor(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        # Use a pure rotation bivector (e1e2 only)
        B = torch.zeros(bivector_count)
        B[0] = 0.3  # e1e2 component

        # exp(B) and exp(-B)
        R = func.exp_bivector(B)
        R_inv = func.exp_bivector(-B)

        # R * R_inv should be identity
        result = func.compose_even_versor(R, R_inv)

        torch.testing.assert_close(
            result, identity, rtol=1e-3, atol=1e-3,
            msg="exp(B) * exp(-B) should be identity"
        )


class TestExpBivectorCliffordComparison:
    """T082: Comparison with clifford library."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_vs_clifford(self, dim: int):
        """Compare exp_bivector with clifford library for pure rotation."""
        try:
            from clifford import Cl
            from clifford.tools import exp as cliff_exp
        except ImportError:
            pytest.skip("clifford library or tools not installed")

        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        # Create clifford algebra
        layout, blades = Cl(dim + 1, 1)

        # Pure rotation bivector (e1e2 only)
        angle = 0.5
        B = torch.zeros(bivector_count)
        B[0] = angle  # e1e2

        # Our implementation
        result_ours = func.exp_bivector(B)

        # Clifford implementation
        e1, e2 = blades['e1'], blades['e2']
        B_cliff = angle * (e1 ^ e2)

        # clifford's exp function
        result_cliff = cliff_exp(B_cliff)

        # Compare scalar and e1e2 components
        assert abs(result_ours[0].item() - result_cliff.value[0]) < 1e-4, \
            "Scalar component should match clifford"

        # e1e2 index in full algebra (for CGA(n), e1e2 is typically index 5 or similar)
        # Map from our EVEN_VERSOR_MASK to full blade index
        e1e2_full_idx = func.EVEN_VERSOR_MASK[1]
        assert abs(result_ours[1].item() - result_cliff.value[e1e2_full_idx]) < 1e-4, \
            "e1e2 component should match clifford"


class TestExpBivectorBatch:
    """T083: Batch dimension tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_dimension(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)
        even_versor_count = len(func.EVEN_VERSOR_MASK)
        batch_size = 16

        B = torch.randn(batch_size, bivector_count) * 0.1  # Small angles

        result = func.exp_bivector(B)

        assert result.shape == (batch_size, even_versor_count)


class TestExpBivectorONNX:
    """T084: ONNX export tests."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_onnx_export_no_control_flow(self, dim: int):
        """Verify ONNX export has no Loop/If nodes."""
        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        class ExpBivectorModel(torch.nn.Module):
            def forward(self, B):
                return func.exp_bivector(B)

        model = ExpBivectorModel()
        B = torch.randn(1, bivector_count) * 0.1

        try:
            import io
            import onnx

            buffer = io.BytesIO()
            torch.onnx.export(
                model, (B,), buffer,
                input_names=['B'],
                output_names=['result'],
                opset_version=14
            )

            buffer.seek(0)
            onnx_model = onnx.load(buffer)

            # Check for Loop/If nodes
            forbidden_ops = {'Loop', 'If', 'SequenceConstruct'}
            found_ops = set()
            for node in onnx_model.graph.node:
                if node.op_type in forbidden_ops:
                    found_ops.add(node.op_type)

            assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

        except ImportError:
            pytest.skip("ONNX not installed")


class TestExpBivectorAutograd:
    """T085: Autograd gradient propagation tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_gradient_propagation(self, dim: int):
        """Test gradient flows through exp_bivector."""
        func = get_functional_module(dim)
        bivector_count = len(func.GRADE_2_INDICES)

        # Create leaf tensor with requires_grad=True
        B = torch.randn(bivector_count) * 0.1
        B = B.clone().detach().requires_grad_(True)

        result = func.exp_bivector(B)
        loss = result.sum()
        loss.backward()

        assert B.grad is not None, "B should have gradient"
        assert not torch.isnan(B.grad).any(), "B gradient should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
