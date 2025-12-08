"""
Tests for EvenVersor Composition (User Story 1)

T028-T036: Test compose_even_versor and compose_similitude operations.

Tests verify:
- T029: Identity composition
- T030: Associativity
- T031: Inverse composition
- T032: Comparison with clifford library
- T033: Batch dimension support
- T034: ONNX export (no Loop/If nodes)
- T035: Autograd gradient propagation
- T036: Unified API routing
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


def get_random_even_versor(dim: int, batch_size: int = None) -> torch.Tensor:
    """Create random normalized EvenVersor for given CGA dimension."""
    func = get_functional_module(dim)
    even_versor_count = len(func.EVEN_VERSOR_MASK)

    if batch_size is not None:
        ev = torch.randn(batch_size, even_versor_count)
    else:
        ev = torch.randn(even_versor_count)

    # Normalize (simple normalization for testing)
    norm = torch.sqrt((ev ** 2).sum(dim=-1, keepdim=True))
    return ev / norm


class TestComposeEvenVersorIdentity:
    """T029: Identity element tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_left_identity(self, dim: int):
        """compose(identity, V) == V"""
        func = get_functional_module(dim)
        identity = get_identity_even_versor(dim)
        v = get_random_even_versor(dim)

        result = func.compose_even_versor(identity, v)
        torch.testing.assert_close(result, v, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_right_identity(self, dim: int):
        """compose(V, identity) == V"""
        func = get_functional_module(dim)
        identity = get_identity_even_versor(dim)
        v = get_random_even_versor(dim)

        result = func.compose_even_versor(v, identity)
        torch.testing.assert_close(result, v, rtol=1e-5, atol=1e-5)


class TestComposeEvenVersorAssociativity:
    """T030: Associativity tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_associativity(self, dim: int):
        """compose(compose(A,B), C) == compose(A, compose(B,C))"""
        func = get_functional_module(dim)

        # Use fixed seed for reproducibility
        torch.manual_seed(123 + dim)

        a = get_random_even_versor(dim)
        b = get_random_even_versor(dim)
        c = get_random_even_versor(dim)

        # (A * B) * C
        ab = func.compose_even_versor(a, b)
        ab_c = func.compose_even_versor(ab, c)

        # A * (B * C)
        bc = func.compose_even_versor(b, c)
        a_bc = func.compose_even_versor(a, bc)

        # Relaxed tolerance for float32 accumulation errors
        torch.testing.assert_close(ab_c, a_bc, rtol=1e-3, atol=1e-3)


class TestComposeEvenVersorInverse:
    """T031: Inverse element tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_inverse_composition(self, dim: int):
        """compose(V, reverse(V)) ≈ scalar * identity for unit rotors.

        Note: This property only holds for unit versors (rotors).
        For a general EvenVersor, V * ~V does not necessarily equal a scalar.

        We test with a pure Euclidean rotation bivector (only eiej where i,j are
        Euclidean basis indices), which guarantees B² < 0 and produces a unit rotor.
        """
        func = get_functional_module(dim)
        identity = get_identity_even_versor(dim)

        # Use fixed seed for reproducibility
        torch.manual_seed(456 + dim)

        # For CGA(n), the Euclidean rotation bivector is only e1e2 (index 0 in Grade-2)
        # For dim 0: no Euclidean rotation possible (0D space)
        # For dim 1: only e1, no rotation plane
        # For dim 2+: e1e2 is a valid rotation bivector
        bivector_count = len(func.GRADE_2_INDICES)
        B = torch.zeros(bivector_count)

        if dim >= 2:
            # Use a pure rotation in the e1e2 plane
            # e1e2 is typically the first Grade-2 blade in CGA
            angle = 0.3  # 0.3 radians
            B[0] = angle  # e1e2 component only

        v = func.exp_bivector(B)

        # Compute reverse
        v_rev = func.reverse_even_versor(v)

        # Compose V * ~V
        result = func.compose_even_versor(v, v_rev)

        # For unit rotor from rotation bivector: R * ~R = 1
        # Result should be identity (scalar = 1, all others = 0)
        torch.testing.assert_close(
            result, identity, rtol=1e-3, atol=1e-3,
            msg=f"R * ~R should be identity for dim={dim}"
        )


class TestComposeEvenVersorCliffordComparison:
    """T032: Comparison with clifford library."""

    @pytest.mark.parametrize("dim", [1, 2, 3])  # Skip dim=0 for clifford
    def test_vs_clifford(self, dim: int):
        """Compare compose_even_versor with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)

        # Create clifford algebra
        layout, blades = Cl(dim + 1, 1)

        # Get EvenVersor indices mapping
        even_versor_mask = func.EVEN_VERSOR_MASK

        # Create random EvenVersors with fixed seed for reproducibility
        torch.manual_seed(42 + dim)
        v1_vals = torch.randn(len(even_versor_mask)) * 0.5
        v2_vals = torch.randn(len(even_versor_mask)) * 0.5

        # Our implementation
        result_ours = func.compose_even_versor(v1_vals, v2_vals)

        # Clifford implementation - build multivectors
        v1_mv = layout.MultiVector()
        v2_mv = layout.MultiVector()

        for sparse_idx, full_idx in enumerate(even_versor_mask):
            v1_mv.value[full_idx] = v1_vals[sparse_idx].item()
            v2_mv.value[full_idx] = v2_vals[sparse_idx].item()

        # Compute product in clifford
        result_cliff = (v1_mv * v2_mv)

        # Extract EvenVersor components from clifford result
        # Cast to float32 to match our implementation dtype
        result_cliff_ev = torch.tensor([
            result_cliff.value[full_idx] for full_idx in even_versor_mask
        ], dtype=torch.float32)

        # Use relaxed tolerance (1e-5) to account for float32 precision
        torch.testing.assert_close(
            result_ours, result_cliff_ev, rtol=1e-5, atol=1e-5,
            msg=f"CGA{dim}D compose_even_versor differs from clifford"
        )


class TestComposeEvenVersorBatch:
    """T033: Batch dimension tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_dimension(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        batch_size = 16

        v1 = get_random_even_versor(dim, batch_size)
        v2 = get_random_even_versor(dim, batch_size)

        result = func.compose_even_versor(v1, v2)

        assert result.shape == (batch_size, len(func.EVEN_VERSOR_MASK))

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_broadcast_batch(self, dim: int):
        """Test broadcasting with different batch dimensions."""
        func = get_functional_module(dim)

        v1 = get_random_even_versor(dim, batch_size=8)
        v2 = get_random_even_versor(dim)  # No batch

        result = func.compose_even_versor(v1, v2)
        assert result.shape == (8, len(func.EVEN_VERSOR_MASK))


class TestComposeEvenVersorONNX:
    """T034: ONNX export tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])  # Test a few dimensions
    def test_onnx_export_no_control_flow(self, dim: int):
        """Verify ONNX export has no Loop/If nodes."""
        func = get_functional_module(dim)
        even_versor_count = len(func.EVEN_VERSOR_MASK)

        class ComposeModel(torch.nn.Module):
            def forward(self, v1, v2):
                return func.compose_even_versor(v1, v2)

        model = ComposeModel()
        v1 = torch.randn(1, even_versor_count)
        v2 = torch.randn(1, even_versor_count)

        try:
            import io
            import onnx

            buffer = io.BytesIO()
            torch.onnx.export(
                model, (v1, v2), buffer,
                input_names=['v1', 'v2'],
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


class TestComposeEvenVersorAutograd:
    """T035: Autograd gradient propagation tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_gradient_propagation(self, dim: int):
        """Test gradient flows through compose_even_versor."""
        func = get_functional_module(dim)

        v1 = get_random_even_versor(dim).requires_grad_(True)
        v2 = get_random_even_versor(dim).requires_grad_(True)

        result = func.compose_even_versor(v1, v2)
        loss = result.sum()
        loss.backward()

        assert v1.grad is not None, "v1 should have gradient"
        assert v2.grad is not None, "v2 should have gradient"
        assert not torch.isnan(v1.grad).any(), "v1 gradient should not be NaN"
        assert not torch.isnan(v2.grad).any(), "v2 gradient should not be NaN"


class TestComposeSimilitude:
    """T047-T052: Similitude composition tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_similitude_equals_even_versor(self, dim: int):
        """compose_similitude should give same result as compose_even_versor."""
        func = get_functional_module(dim)

        v1 = get_random_even_versor(dim)
        v2 = get_random_even_versor(dim)

        result_ev = func.compose_even_versor(v1, v2)
        result_sim = func.compose_similitude(v1, v2)

        torch.testing.assert_close(result_ev, result_sim, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
