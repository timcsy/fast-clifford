"""
Tests for Geometric Inner Product (User Story 2)

T059-T067: Test inner_product_full operations across CGA dimensions.

Tests verify:
- T060: Null basis inner product: <eo, einf> = -1
- T061: Symmetry: <a, b> = <b, a>
- T062: Orthogonality: orthogonal blades have zero inner product
- T063: Comparison with clifford library
- T065: Batch dimension support
- T066: ONNX export (no Loop/If nodes)
- T067: Autograd gradient propagation
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


def get_blade_count(dim: int) -> int:
    """Get total blade count for CGA(n) = Cl(n+1,1)."""
    return 2 ** (dim + 2)


class TestInnerProductNullBasis:
    """T060: Null basis inner product tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_eo_einf_inner_product(self, dim: int):
        """<e_o, e_inf> = -1

        In CGA, the null basis vectors satisfy:
        - e_o = 0.5 * (e- - e+)
        - e_inf = e- + e+
        - <e_o, e_inf> = -1

        Note: inner_product_full computes the scalar part of the geometric product,
        which for two vectors is their metric inner product.
        """
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # Create e_o and e_inf in the full multivector representation
        # Blade ordering: [scalar, e1, e2, ..., en, e+, e-, ...]
        # For CGA(n), e+ is at index n+1, e- is at index n+2
        e_plus_idx = dim + 1
        e_minus_idx = dim + 2

        eo = torch.zeros(blade_count)
        eo[e_plus_idx] = -0.5  # e+ coefficient in e_o
        eo[e_minus_idx] = 0.5  # e- coefficient in e_o

        einf = torch.zeros(blade_count)
        einf[e_plus_idx] = 1.0  # e+ coefficient in e_inf
        einf[e_minus_idx] = 1.0  # e- coefficient in e_inf

        result = func.inner_product_full(eo, einf)

        # Result should be a scalar = -1
        expected = torch.tensor([-1.0])
        torch.testing.assert_close(
            result, expected, rtol=1e-5, atol=1e-5,
            msg=f"<e_o, e_inf> should be -1 for dim={dim}"
        )


class TestInnerProductSymmetry:
    """T061: Symmetry tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_symmetry(self, dim: int):
        """<a, b> == <b, a>"""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        torch.manual_seed(42 + dim)
        a = torch.randn(blade_count)
        b = torch.randn(blade_count)

        ab = func.inner_product_full(a, b)
        ba = func.inner_product_full(b, a)

        torch.testing.assert_close(ab, ba, rtol=1e-5, atol=1e-5)


class TestInnerProductOrthogonality:
    """T062: Orthogonality tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_orthogonal_euclidean_basis(self, dim: int):
        """<e_i, e_j> = 0 for i != j (Euclidean basis)."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # e1 is at index 1, e2 is at index 2
        e1 = torch.zeros(blade_count)
        e1[1] = 1.0

        e2 = torch.zeros(blade_count)
        e2[2] = 1.0

        result = func.inner_product_full(e1, e2)

        expected = torch.tensor([0.0])
        torch.testing.assert_close(
            result, expected, rtol=1e-5, atol=1e-5,
            msg="<e1, e2> should be 0"
        )

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_euclidean_basis_self_inner_product(self, dim: int):
        """<e_i, e_i> = 1 for Euclidean basis (positive signature)."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        if dim >= 1:
            e1 = torch.zeros(blade_count)
            e1[1] = 1.0

            result = func.inner_product_full(e1, e1)

            expected = torch.tensor([1.0])
            torch.testing.assert_close(
                result, expected, rtol=1e-5, atol=1e-5,
                msg="<e1, e1> should be 1"
            )


class TestInnerProductCliffordComparison:
    """T063: Comparison with clifford library."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_vs_clifford(self, dim: int):
        """Compare inner_product_full with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # Create clifford algebra
        layout, blades = Cl(dim + 1, 1)

        # Random multivectors
        torch.manual_seed(42 + dim)
        a_vals = torch.randn(blade_count) * 0.5
        b_vals = torch.randn(blade_count) * 0.5

        # Our implementation
        result_ours = func.inner_product_full(a_vals, b_vals)

        # Clifford implementation
        a_mv = layout.MultiVector(value=a_vals.numpy().astype(np.float64))
        b_mv = layout.MultiVector(value=b_vals.numpy().astype(np.float64))

        # Geometric product and extract scalar (grade 0)
        result_cliff = (a_mv * b_mv).value[0]

        torch.testing.assert_close(
            result_ours.item(), float(result_cliff), rtol=1e-5, atol=1e-5,
            msg=f"CGA{dim}D inner_product differs from clifford"
        )


class TestInnerProductZero:
    """T064: Zero vector tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_zero_inner_product(self, dim: int):
        """<0, 0> = 0"""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        zero = torch.zeros(blade_count)

        result = func.inner_product_full(zero, zero)

        expected = torch.tensor([0.0])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestInnerProductBatch:
    """T065: Batch dimension tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_dimension(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        batch_size = 16

        a = torch.randn(batch_size, blade_count)
        b = torch.randn(batch_size, blade_count)

        result = func.inner_product_full(a, b)

        assert result.shape == (batch_size, 1)

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_broadcast(self, dim: int):
        """Test broadcasting with different batch dimensions."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        a = torch.randn(8, blade_count)
        b = torch.randn(blade_count)  # No batch

        result = func.inner_product_full(a, b)
        assert result.shape == (8, 1)


class TestInnerProductONNX:
    """T066: ONNX export tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_onnx_export_no_control_flow(self, dim: int):
        """Verify ONNX export has no Loop/If nodes."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        class InnerProductModel(torch.nn.Module):
            def forward(self, a, b):
                return func.inner_product_full(a, b)

        model = InnerProductModel()
        a = torch.randn(1, blade_count)
        b = torch.randn(1, blade_count)

        try:
            import io
            import onnx

            buffer = io.BytesIO()
            torch.onnx.export(
                model, (a, b), buffer,
                input_names=['a', 'b'],
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


class TestInnerProductAutograd:
    """T067: Autograd gradient propagation tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_gradient_propagation(self, dim: int):
        """Test gradient flows through inner_product_full."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        a = torch.randn(blade_count, requires_grad=True)
        b = torch.randn(blade_count, requires_grad=True)

        result = func.inner_product_full(a, b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None, "a should have gradient"
        assert b.grad is not None, "b should have gradient"
        assert not torch.isnan(a.grad).any(), "a gradient should not be NaN"
        assert not torch.isnan(b.grad).any(), "b gradient should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
