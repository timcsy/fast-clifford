"""
6D+ Runtime Extended Operations Tests (User Story 4)

T095-T100, T107: Tests for runtime extended operations on high-dimensional CGAs.

Tests verify:
- T095: Test framework setup
- T096: CGA(6) compose clifford comparison
- T097: CGA(6) inner_product clifford comparison
- T098: CGA(6) exp_bivector clifford comparison
- T099: CGA(7) basic functionality
- T100: Batch dimension support for 6D+
- T107: Execute runtime test verification
"""

import pytest
import torch
import numpy as np


def get_cga(dim: int):
    """Get CGA algebra for given dimension."""
    from fast_clifford.cga import CGA
    return CGA(dim)


def is_runtime_algebra(algebra):
    """Check if algebra uses runtime implementation."""
    from fast_clifford.cga.runtime import RuntimeCGAAlgebra
    return isinstance(algebra, RuntimeCGAAlgebra)


# =============================================================================
# T095: Test Framework Setup
# =============================================================================

class TestRuntimeExtendedFramework:
    """T095: Basic runtime extended operations test framework."""

    def test_cga6_uses_runtime(self):
        """CGA(6) should use RuntimeCGAAlgebra."""
        algebra = get_cga(6)
        assert is_runtime_algebra(algebra), "CGA(6) should use RuntimeCGAAlgebra"

    def test_cga7_uses_runtime(self):
        """CGA(7) should use RuntimeCGAAlgebra."""
        algebra = get_cga(7)
        assert is_runtime_algebra(algebra), "CGA(7) should use RuntimeCGAAlgebra"

    def test_runtime_has_extended_ops(self):
        """Runtime algebra should have all extended operations."""
        algebra = get_cga(6)

        assert hasattr(algebra, 'compose_even_versor')
        assert hasattr(algebra, 'inner_product')
        assert hasattr(algebra, 'exp_bivector')
        assert hasattr(algebra, 'outer_product')
        assert hasattr(algebra, 'left_contraction')
        assert hasattr(algebra, 'right_contraction')
        assert hasattr(algebra, 'grade_select')
        assert hasattr(algebra, 'dual')
        assert hasattr(algebra, 'normalize')


# =============================================================================
# T096: CGA(6) compose clifford comparison
# =============================================================================

class TestCGA6Compose:
    """T096: CGA(6) compose_even_versor clifford comparison."""

    def test_compose_identity(self):
        """compose(identity, V) == V for CGA(6)."""
        algebra = get_cga(6)

        identity = torch.zeros(algebra.even_versor_count)
        identity[0] = 1.0

        v = torch.randn(algebra.even_versor_count)
        v = v / torch.norm(v)

        result = algebra.compose_even_versor(identity, v)
        torch.testing.assert_close(result, v, rtol=1e-4, atol=1e-4)

    def test_compose_batch(self):
        """Batch compose for CGA(6)."""
        algebra = get_cga(6)
        batch_size = 8

        v1 = torch.randn(batch_size, algebra.even_versor_count)
        v2 = torch.randn(batch_size, algebra.even_versor_count)

        result = algebra.compose_even_versor(v1, v2)
        assert result.shape == (batch_size, algebra.even_versor_count)

    def test_compose_vs_clifford(self):
        """Compare CGA(6) compose with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        from fast_clifford.codegen.cga_factory import get_even_versor_indices

        algebra = get_cga(6)

        # Create clifford algebra Cl(7,1) for CGA(6)
        layout, _ = Cl(7, 1)

        # Get even versor indices
        even_indices = get_even_versor_indices(6)

        # Random even versors
        torch.manual_seed(42)
        v1_vals = torch.randn(algebra.even_versor_count) * 0.3
        v2_vals = torch.randn(algebra.even_versor_count) * 0.3

        # Our implementation
        result_ours = algebra.compose_even_versor(v1_vals, v2_vals)

        # Clifford implementation
        v1_mv = layout.MultiVector()
        v2_mv = layout.MultiVector()

        for i, full_idx in enumerate(even_indices):
            v1_mv.value[full_idx] = v1_vals[i].item()
            v2_mv.value[full_idx] = v2_vals[i].item()

        result_cliff = v1_mv * v2_mv

        # Extract even versor components
        result_cliff_ev = torch.tensor([
            result_cliff.value[full_idx] for full_idx in even_indices
        ], dtype=torch.float32)

        torch.testing.assert_close(
            result_ours, result_cliff_ev, rtol=1e-3, atol=1e-3,
            msg="CGA(6) compose differs from clifford"
        )


# =============================================================================
# T097: CGA(6) inner_product clifford comparison
# =============================================================================

class TestCGA6InnerProduct:
    """T097: CGA(6) inner_product clifford comparison."""

    def test_inner_product_basic(self):
        """Basic inner_product for CGA(6)."""
        algebra = get_cga(6)

        a = torch.randn(algebra.blade_count)
        b = torch.randn(algebra.blade_count)

        result = algebra.inner_product(a, b)
        assert result.shape == (1,)

    def test_inner_product_symmetric(self):
        """inner_product(a, b) == inner_product(b, a)."""
        algebra = get_cga(6)

        a = torch.randn(algebra.blade_count)
        b = torch.randn(algebra.blade_count)

        result_ab = algebra.inner_product(a, b)
        result_ba = algebra.inner_product(b, a)

        torch.testing.assert_close(result_ab, result_ba, rtol=1e-5, atol=1e-5)

    def test_inner_product_vs_clifford(self):
        """Compare CGA(6) inner_product with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        algebra = get_cga(6)
        layout, _ = Cl(7, 1)

        torch.manual_seed(42)
        a_vals = torch.randn(algebra.blade_count) * 0.3
        b_vals = torch.randn(algebra.blade_count) * 0.3

        # Our implementation
        result_ours = algebra.inner_product(a_vals, b_vals)

        # Clifford implementation
        a_mv = layout.MultiVector(a_vals.numpy().astype(np.float64))
        b_mv = layout.MultiVector(b_vals.numpy().astype(np.float64))

        # Geometric inner product is grade-0 of geometric product
        result_cliff = (a_mv * b_mv).value[0]

        torch.testing.assert_close(
            result_ours[0], torch.tensor(result_cliff, dtype=torch.float32),
            rtol=1e-3, atol=1e-3,
            msg="CGA(6) inner_product differs from clifford"
        )


# =============================================================================
# T098: CGA(6) exp_bivector clifford comparison
# =============================================================================

class TestCGA6ExpBivector:
    """T098: CGA(6) exp_bivector clifford comparison."""

    def test_exp_bivector_zero(self):
        """exp_bivector(0) == identity."""
        algebra = get_cga(6)

        zero_biv = torch.zeros(algebra.bivector_count)
        result = algebra.exp_bivector(zero_biv)

        # Identity: scalar = 1, all others = 0
        assert abs(result[0].item() - 1.0) < 1e-5, "exp(0)[0] should be 1"
        for i in range(1, len(result)):
            assert abs(result[i].item()) < 1e-5, f"exp(0)[{i}] should be 0"

    def test_exp_bivector_small_angle(self):
        """exp_bivector for small angles should be stable."""
        algebra = get_cga(6)

        small_biv = torch.zeros(algebra.bivector_count)
        small_biv[0] = 1e-8

        result = algebra.exp_bivector(small_biv)

        assert not torch.isnan(result).any(), "exp_bivector should not produce NaN"
        assert not torch.isinf(result).any(), "exp_bivector should not produce Inf"

    def test_exp_bivector_vs_clifford(self):
        """Compare CGA(6) exp_bivector with clifford library."""
        try:
            from clifford import Cl
            import numpy as np
        except ImportError:
            pytest.skip("clifford library not installed")

        from fast_clifford.codegen.cga_factory import compute_grade_indices, get_even_versor_indices

        algebra = get_cga(6)
        layout, blades = Cl(7, 1)

        # Create a small Euclidean rotation bivector
        # Use e1^e2 (first Euclidean bivector)
        grade_indices = compute_grade_indices(6)
        bivector_indices = list(grade_indices[2])
        even_indices = get_even_versor_indices(6)

        B = torch.zeros(algebra.bivector_count)
        angle = 0.3
        B[0] = angle  # First bivector component

        # Our implementation
        result_ours = algebra.exp_bivector(B)

        # Clifford implementation
        B_mv = layout.MultiVector()
        for i, full_idx in enumerate(bivector_indices):
            B_mv.value[full_idx] = B[i].item()

        # Use clifford's exp
        result_cliff_mv = np.exp(B_mv)

        # Extract even versor components
        result_cliff = torch.tensor([
            result_cliff_mv.value[full_idx] for full_idx in even_indices
        ], dtype=torch.float32)

        torch.testing.assert_close(
            result_ours, result_cliff, rtol=1e-3, atol=1e-3,
            msg="CGA(6) exp_bivector differs from clifford"
        )


# =============================================================================
# T099: CGA(7) Basic Functionality
# =============================================================================

class TestCGA7Basic:
    """T099: CGA(7) basic functionality tests."""

    def test_cga7_dimensions(self):
        """CGA(7) should have correct dimensions."""
        algebra = get_cga(7)

        # CGA(7) = Cl(8,1) -> 2^9 = 512 blades
        assert algebra.blade_count == 512
        assert algebra.euclidean_dim == 7

    def test_cga7_compose(self):
        """Basic compose for CGA(7)."""
        algebra = get_cga(7)

        v1 = torch.randn(algebra.even_versor_count)
        v2 = torch.randn(algebra.even_versor_count)

        result = algebra.compose_even_versor(v1, v2)
        assert result.shape == (algebra.even_versor_count,)

    def test_cga7_inner_product(self):
        """Basic inner_product for CGA(7)."""
        algebra = get_cga(7)

        a = torch.randn(algebra.blade_count)
        b = torch.randn(algebra.blade_count)

        result = algebra.inner_product(a, b)
        assert result.shape == (1,)

    def test_cga7_exp_bivector(self):
        """Basic exp_bivector for CGA(7)."""
        algebra = get_cga(7)

        B = torch.zeros(algebra.bivector_count)
        result = algebra.exp_bivector(B)

        assert result.shape == (algebra.even_versor_count,)
        assert abs(result[0].item() - 1.0) < 1e-5


# =============================================================================
# T100: Batch Dimension Tests (6D+)
# =============================================================================

class TestRuntimeBatch:
    """T100: Batch dimension tests for 6D+ runtime operations."""

    @pytest.mark.parametrize("dim", [6, 7])
    def test_compose_batch(self, dim: int):
        """Test batch compose for 6D+."""
        algebra = get_cga(dim)
        batch_size = 4

        v1 = torch.randn(batch_size, algebra.even_versor_count)
        v2 = torch.randn(batch_size, algebra.even_versor_count)

        result = algebra.compose_even_versor(v1, v2)
        assert result.shape == (batch_size, algebra.even_versor_count)

    @pytest.mark.parametrize("dim", [6, 7])
    def test_inner_product_batch(self, dim: int):
        """Test batch inner_product for 6D+."""
        algebra = get_cga(dim)
        batch_size = 4

        a = torch.randn(batch_size, algebra.blade_count)
        b = torch.randn(batch_size, algebra.blade_count)

        result = algebra.inner_product(a, b)
        assert result.shape == (batch_size, 1)

    @pytest.mark.parametrize("dim", [6, 7])
    def test_exp_bivector_batch(self, dim: int):
        """Test batch exp_bivector for 6D+."""
        algebra = get_cga(dim)
        batch_size = 4

        B = torch.randn(batch_size, algebra.bivector_count) * 0.1

        result = algebra.exp_bivector(B)
        assert result.shape == (batch_size, algebra.even_versor_count)


# =============================================================================
# T107: Execute Runtime Test Verification
# =============================================================================

class TestRuntimeVerification:
    """T107: Comprehensive runtime test verification."""

    def test_full_runtime_workflow(self):
        """Test complete runtime workflow for CGA(6)."""
        algebra = get_cga(6)

        # 1. Create EvenVersor from exp_bivector
        B = torch.zeros(algebra.bivector_count)
        B[0] = 0.3
        rotor = algebra.exp_bivector(B)

        # 2. Compose with another
        identity = torch.zeros(algebra.even_versor_count)
        identity[0] = 1.0
        composed = algebra.compose_even_versor(rotor, identity)

        # 3. Verify shape
        assert composed.shape == rotor.shape

        # 4. Use other extended ops
        mv = torch.randn(algebra.blade_count)
        _ = algebra.inner_product(mv, mv)
        _ = algebra.outer_product(mv, mv)
        _ = algebra.grade_select(mv, 0)
        _ = algebra.dual(mv)
        _ = algebra.normalize(mv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
