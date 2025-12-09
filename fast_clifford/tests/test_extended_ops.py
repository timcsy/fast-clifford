"""
Tests for Extended Operations (User Stories 5-9)

Covers outer_product, left/right_contraction, grade_select, dual, normalize.

Tests verify:
- Outer product: e1 ∧ e2 = e12
- Contractions: Grade lowering behavior
- Grade select: Component extraction
- Dual: Pseudoscalar relationship
- Normalize: Unit norm output
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


# =============================================================================
# Outer Product Tests (US5)
# =============================================================================

class TestOuterProduct:
    """User Story 5: Outer Product (Wedge Product) tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_orthogonal_vectors_wedge(self, dim: int):
        """e1 ∧ e2 should produce e12 bivector."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        e1 = torch.zeros(blade_count)
        e1[1] = 1.0

        e2 = torch.zeros(blade_count)
        e2[2] = 1.0

        result = func.outer_product_full(e1, e2)

        # e1∧e2 = e12
        # Find e12 index (first bivector in grade-2)
        e12_idx = func.GRADE_2_INDICES[0]

        assert abs(result[e12_idx].item() - 1.0) < 1e-5, \
            f"e1∧e2 should have e12 component = 1"

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_self_wedge_is_zero(self, dim: int):
        """v ∧ v = 0 for any vector."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        v = torch.randn(blade_count)
        # Make it a pure vector (grade 1 only)
        v_vec = torch.zeros(blade_count)
        for idx in func.GRADE_1_INDICES:
            v_vec[idx] = v[idx]

        result = func.outer_product_full(v_vec, v_vec)

        # All components should be zero
        torch.testing.assert_close(
            result, torch.zeros(blade_count), rtol=1e-5, atol=1e-5,
            msg="v∧v should be 0"
        )

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_outer_product(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        batch_size = 8

        a = torch.randn(batch_size, blade_count)
        b = torch.randn(batch_size, blade_count)

        result = func.outer_product_full(a, b)

        assert result.shape == (batch_size, blade_count)


# =============================================================================
# Contraction Tests (US6)
# =============================================================================

class TestContractions:
    """User Story 6: Left/Right Contraction tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_left_contraction_grade_lowering(self, dim: int):
        """Left contraction of vector with bivector should give vector."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # e1 (vector)
        e1 = torch.zeros(blade_count)
        e1[1] = 1.0

        # e12 (bivector)
        e12 = torch.zeros(blade_count)
        e12_idx = func.GRADE_2_INDICES[0]
        e12[e12_idx] = 1.0

        result = func.left_contraction_full(e1, e12)

        # e1 ⌋ e12 = e1 ⌋ (e1∧e2) = (e1·e1)*e2 - (e1·e2)*e1 = e2
        # Result should be a vector (grade 1)
        has_grade_1 = any(abs(result[idx].item()) > 1e-6 for idx in func.GRADE_1_INDICES)
        assert has_grade_1, "Left contraction should produce grade-1 component"

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_right_contraction_grade_lowering(self, dim: int):
        """Right contraction of bivector with vector should give vector."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # e12 (bivector)
        e12 = torch.zeros(blade_count)
        e12_idx = func.GRADE_2_INDICES[0]
        e12[e12_idx] = 1.0

        # e2 (vector)
        e2 = torch.zeros(blade_count)
        e2[2] = 1.0

        result = func.right_contraction_full(e12, e2)

        # e12 ⌊ e2 should give a vector
        has_grade_1 = any(abs(result[idx].item()) > 1e-6 for idx in func.GRADE_1_INDICES)
        assert has_grade_1, "Right contraction should produce grade-1 component"

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_contractions(self, dim: int):
        """Test batch dimension support for contractions."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        batch_size = 8

        a = torch.randn(batch_size, blade_count)
        b = torch.randn(batch_size, blade_count)

        left_result = func.left_contraction_full(a, b)
        right_result = func.right_contraction_full(a, b)

        assert left_result.shape == (batch_size, blade_count)
        assert right_result.shape == (batch_size, blade_count)


# =============================================================================
# Grade Selection Tests (US7)
# =============================================================================

class TestGradeSelect:
    """User Story 7: Grade Selection tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_grade_0_extraction(self, dim: int):
        """grade_select(mv, 0) returns scalar component."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # Create multivector with all components
        mv = torch.randn(blade_count)
        scalar_val = mv[0].item()

        result = func.grade_select(mv, 0)

        # Only scalar should be non-zero
        assert abs(result[0].item() - scalar_val) < 1e-6
        # All other components should be zero
        assert torch.allclose(result[1:], torch.zeros(blade_count - 1), atol=1e-6)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_grade_1_extraction(self, dim: int):
        """grade_select(mv, 1) returns vector component."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        mv = torch.randn(blade_count)

        result = func.grade_select(mv, 1)

        # Only grade-1 components should be preserved
        for i, val in enumerate(result):
            if i in func.GRADE_1_INDICES:
                assert abs(val.item() - mv[i].item()) < 1e-6
            else:
                assert abs(val.item()) < 1e-6

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_grade_select(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        batch_size = 8

        mv = torch.randn(batch_size, blade_count)

        result = func.grade_select(mv, 0)

        assert result.shape == (batch_size, blade_count)


# =============================================================================
# Dual Tests (US8)
# =============================================================================

class TestDual:
    """User Story 8: Dual operation tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_scalar_dual_is_pseudoscalar(self, dim: int):
        """dual(1) = pseudoscalar."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        scalar = torch.zeros(blade_count)
        scalar[0] = 1.0

        result = func.dual(scalar)

        # The pseudoscalar is the last blade
        pseudoscalar_idx = blade_count - 1
        assert abs(result[pseudoscalar_idx].item()) > 0.1, \
            "dual(1) should have pseudoscalar component"

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_dual(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        batch_size = 8

        mv = torch.randn(batch_size, blade_count)

        result = func.dual(mv)

        assert result.shape == (batch_size, blade_count)


# =============================================================================
# Normalize Tests (US9)
# =============================================================================

class TestNormalize:
    """User Story 9: Normalize operation tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_unit_vector_output(self, dim: int):
        """normalize(v) should have unit norm."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # Create a non-unit vector
        v = torch.zeros(blade_count)
        v[1] = 3.0  # e1 component

        result = func.normalize(v)

        # Check norm is 1 (using inner product)
        inner = func.inner_product_full(result, result)
        assert abs(inner[0].item() - 1.0) < 1e-5, \
            f"Normalized vector should have unit norm, got {inner[0].item()}"

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_zero_vector_stability(self, dim: int):
        """normalize(0) should not produce NaN."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        zero = torch.zeros(blade_count)

        result = func.normalize(zero)

        assert not torch.isnan(result).any(), "normalize(0) should not produce NaN"
        assert not torch.isinf(result).any(), "normalize(0) should not produce Inf"

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_batch_normalize(self, dim: int):
        """Test batch dimension support."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        batch_size = 8

        mv = torch.randn(batch_size, blade_count)

        result = func.normalize(mv)

        assert result.shape == (batch_size, blade_count)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_normalize_autograd(self, dim: int):
        """Test gradient propagation through normalize."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        v = torch.randn(blade_count, requires_grad=True)

        result = func.normalize(v)
        loss = result.sum()
        loss.backward()

        assert v.grad is not None, "v should have gradient"
        assert not torch.isnan(v.grad).any(), "v gradient should not be NaN"


# =============================================================================
# ONNX Export Tests
# =============================================================================

class TestExtendedOpsONNX:
    """ONNX export tests for extended operations."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_outer_product_onnx(self, dim: int):
        """Verify outer_product ONNX export."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return func.outer_product_full(a, b)

        model = Model()
        a = torch.randn(1, blade_count)
        b = torch.randn(1, blade_count)

        try:
            import io
            import onnx

            buffer = io.BytesIO()
            torch.onnx.export(model, (a, b), buffer, opset_version=14)

            buffer.seek(0)
            onnx_model = onnx.load(buffer)

            forbidden_ops = {'Loop', 'If'}
            found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
            assert not found_ops, f"Found forbidden ops: {found_ops}"

        except ImportError:
            pytest.skip("ONNX not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
