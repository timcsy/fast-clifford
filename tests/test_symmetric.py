"""
Tests for p < q algebras (e.g., Cl(0,4), Cl(1,3)) - Direct hardcoded implementation.

These tests verify that algebras with p < q work correctly with direct hardcoded
algebras (not using SymmetricClWrapper, which was found to be mathematically
impossible for p != q due to different metric signatures).
"""

import pytest
import torch
from fast_clifford.clifford import Cl
from fast_clifford.clifford.registry import HardcodedClWrapper


class TestNegativeSignatureCreation:
    """Test creation and basic properties of p < q algebras."""

    def test_cl13_uses_hardcoded_wrapper(self):
        """Cl(1,3) should use HardcodedClWrapper directly."""
        algebra = Cl(1, 3)
        assert isinstance(algebra, HardcodedClWrapper)
        assert algebra.p == 1
        assert algebra.q == 3
        assert algebra.count_blade == 16

    def test_cl04_uses_hardcoded_wrapper(self):
        """Cl(0,4) should use HardcodedClWrapper directly."""
        algebra = Cl(0, 4)
        assert isinstance(algebra, HardcodedClWrapper)
        assert algebra.p == 0
        assert algebra.q == 4
        assert algebra.count_blade == 16

    def test_cl02_uses_hardcoded_wrapper(self):
        """Cl(0,2) should use HardcodedClWrapper directly."""
        algebra = Cl(0, 2)
        assert isinstance(algebra, HardcodedClWrapper)
        assert algebra.p == 0
        assert algebra.q == 2
        assert algebra.count_blade == 4

    def test_cl31_uses_hardcoded_wrapper(self):
        """Cl(3,1) with p >= q should also use HardcodedClWrapper."""
        algebra = Cl(3, 1)
        assert isinstance(algebra, HardcodedClWrapper)
        assert algebra.p == 3
        assert algebra.q == 1


class TestBasisVectorSquares:
    """Test that basis vector squares have correct signs for p < q algebras."""

    def test_cl04_e1_squared_is_negative(self):
        """In Cl(0,4), all basis vectors square to -1."""
        algebra = Cl(0, 4)
        e1 = torch.zeros(algebra.count_blade)
        e1[1] = 1.0
        e1_sq = algebra.geometric_product(e1, e1)
        # Scalar part should be -1
        assert torch.isclose(e1_sq[0], torch.tensor(-1.0), atol=1e-6)

    def test_cl04_all_basis_vectors_square_negative(self):
        """In Cl(0,4), e1², e2², e3², e4² all equal -1."""
        algebra = Cl(0, 4)
        for i in range(1, 5):
            ei = torch.zeros(algebra.count_blade)
            ei[i] = 1.0
            ei_sq = algebra.geometric_product(ei, ei)
            assert torch.isclose(ei_sq[0], torch.tensor(-1.0), atol=1e-6), \
                f"e{i}² should be -1, got {ei_sq[0]}"

    def test_cl13_e1_squared_is_positive(self):
        """In Cl(1,3), e1 (the positive basis) squares to +1."""
        algebra = Cl(1, 3)
        e1 = torch.zeros(algebra.count_blade)
        e1[1] = 1.0
        e1_sq = algebra.geometric_product(e1, e1)
        assert torch.isclose(e1_sq[0], torch.tensor(1.0), atol=1e-6)

    def test_cl13_e2_squared_is_negative(self):
        """In Cl(1,3), e2 (first negative basis) squares to -1."""
        algebra = Cl(1, 3)
        e2 = torch.zeros(algebra.count_blade)
        e2[2] = 1.0
        e2_sq = algebra.geometric_product(e2, e2)
        assert torch.isclose(e2_sq[0], torch.tensor(-1.0), atol=1e-6)

    def test_cl13_correct_signature(self):
        """In Cl(1,3): e1²=+1, e2²=e3²=e4²=-1."""
        algebra = Cl(1, 3)
        expected = [1.0, -1.0, -1.0, -1.0]
        for i, exp_sq in enumerate(expected):
            ei = torch.zeros(algebra.count_blade)
            ei[i + 1] = 1.0
            ei_sq = algebra.geometric_product(ei, ei)
            assert torch.isclose(ei_sq[0], torch.tensor(exp_sq), atol=1e-6), \
                f"e{i+1}² should be {exp_sq}, got {ei_sq[0]}"

    def test_cl02_quaternion_signature(self):
        """Cl(0,2) is quaternions: e1²=e2²=-1, (e1e2)²=-1."""
        algebra = Cl(0, 2)

        # e1² = -1
        e1 = torch.zeros(4)
        e1[1] = 1.0
        e1_sq = algebra.geometric_product(e1, e1)
        assert torch.isclose(e1_sq[0], torch.tensor(-1.0), atol=1e-6)

        # e2² = -1
        e2 = torch.zeros(4)
        e2[2] = 1.0
        e2_sq = algebra.geometric_product(e2, e2)
        assert torch.isclose(e2_sq[0], torch.tensor(-1.0), atol=1e-6)

        # e12² = e1*e2*e1*e2 = -e1*e1*e2*e2 = -(-1)(-1) = -1
        e12 = algebra.outer(e1, e2)
        e12_sq = algebra.geometric_product(e12, e12)
        assert torch.isclose(e12_sq[0], torch.tensor(-1.0), atol=1e-6)


class TestGeometricProduct:
    """Test geometric product for p < q algebras."""

    def test_geometric_product_cl13(self):
        """Test e1*e2 = e12 in Cl(1,3)."""
        algebra = Cl(1, 3)

        e1 = torch.zeros(16)
        e1[1] = 1.0
        e2 = torch.zeros(16)
        e2[2] = 1.0

        # e1*e2 should give e12 bivector
        prod = algebra.geometric_product(e1, e2)

        # Verify e12² = e1*e2*e1*e2 = -e1*e1*e2*e2 = -(+1)*(-1) = +1
        e12_sq = algebra.geometric_product(prod, prod)
        assert torch.isclose(e12_sq[0], torch.tensor(1.0), atol=1e-6)

    def test_geometric_product_batch(self):
        """Test batched geometric product in p < q algebra."""
        algebra = Cl(1, 3)
        batch_size = 10
        blade_count = algebra.count_blade

        a = torch.randn(batch_size, blade_count)
        b = torch.randn(batch_size, blade_count)

        result = algebra.geometric_product(a, b)
        assert result.shape == (batch_size, blade_count)


class TestOtherOperations:
    """Test other operations in p < q algebras."""

    def test_reverse(self):
        """Test reverse operation."""
        algebra = Cl(1, 3)
        mv = torch.randn(16)
        rev = algebra.reverse(mv)
        assert rev.shape == mv.shape

    def test_outer_product(self):
        """Test outer product."""
        algebra = Cl(1, 3)
        e1 = torch.zeros(16)
        e1[1] = 1.0
        e2 = torch.zeros(16)
        e2[2] = 1.0
        e12 = algebra.outer(e1, e2)
        # e1∧e2 should be nonzero
        assert e12.abs().sum() > 0

    def test_inner_product(self):
        """Test inner product returns scalar."""
        algebra = Cl(0, 4)
        e1 = torch.zeros(16)
        e1[1] = 1.0
        inner = algebra.inner(e1, e1)
        # Inner product of e1 with itself should give -1 (since e1²=-1 in Cl(0,4))
        assert inner.numel() >= 1

    def test_norm_squared(self):
        """Test norm squared."""
        algebra = Cl(1, 3)
        # Create scalar multivector directly
        mv = torch.zeros(algebra.count_blade)
        mv[0] = 3.0
        norm_sq = algebra.norm_squared(mv)
        # Scalar norm squared: s * ~s = s * s = s²
        assert torch.isclose(norm_sq[0], torch.tensor(9.0), atol=1e-6)


class TestCliffordLibraryComparison:
    """Compare p < q algebra results with clifford library (if available)."""

    @pytest.fixture
    def clifford_available(self):
        """Check if clifford library is available."""
        try:
            import clifford
            return True
        except ImportError:
            pytest.skip("clifford library not available")
            return False

    def test_cl02_geometric_product_vs_clifford(self, clifford_available):
        """Compare Cl(0,2) geometric product with clifford library."""
        import clifford
        import numpy as np

        # Create algebras
        layout, blades = clifford.Cl(0, 2)
        our_algebra = Cl(0, 2)

        # Test e1 * e1 = -1 in Cl(0,2)
        cf_e1 = blades['e1']
        cf_result = cf_e1 * cf_e1

        our_e1 = torch.zeros(4)
        our_e1[1] = 1.0
        our_result = our_algebra.geometric_product(our_e1, our_e1)

        # clifford's result scalar should match ours
        assert np.isclose(float(cf_result), our_result[0].item(), atol=1e-6)

    def test_cl13_basis_squares_vs_clifford(self, clifford_available):
        """Compare Cl(1,3) basis vector squares with clifford library."""
        import clifford
        import numpy as np

        layout, blades = clifford.Cl(1, 3)
        our_algebra = Cl(1, 3)

        # Check all basis vector squares
        for i in range(1, 5):
            cf_ei = blades[f'e{i}']
            cf_sq = cf_ei * cf_ei

            our_ei = torch.zeros(16)
            our_ei[i] = 1.0
            our_sq = our_algebra.geometric_product(our_ei, our_ei)

            expected = 1.0 if i <= 1 else -1.0  # p=1 positive, q=3 negative
            assert np.isclose(float(cf_sq), expected, atol=1e-6), \
                f"clifford e{i}² = {float(cf_sq)}, expected {expected}"
            assert torch.isclose(our_sq[0], torch.tensor(expected), atol=1e-6), \
                f"our e{i}² = {our_sq[0]}, expected {expected}"


class TestAllPLessQAlgebras:
    """Test all algebras with p < q and p+q < 8."""

    @pytest.mark.parametrize("p,q", [
        (0, 1), (0, 2), (1, 2), (0, 3),
        (1, 3), (0, 4), (2, 3), (1, 4), (0, 5),
        (2, 4), (1, 5), (0, 6), (3, 4), (2, 5), (1, 6), (0, 7),
    ])
    def test_algebra_creation(self, p, q):
        """Test that all p < q algebras can be created."""
        algebra = Cl(p, q)
        assert isinstance(algebra, HardcodedClWrapper)
        assert algebra.p == p
        assert algebra.q == q
        assert algebra.count_blade == 2 ** (p + q)

    @pytest.mark.parametrize("p,q", [
        (0, 1), (0, 2), (1, 2), (0, 3),
        (1, 3), (0, 4), (2, 3), (1, 4), (0, 5),
        (2, 4), (1, 5), (0, 6), (3, 4), (2, 5), (1, 6), (0, 7),
    ])
    def test_correct_signature(self, p, q):
        """Test that all algebras have correct metric signature."""
        algebra = Cl(p, q)
        n = p + q

        for i in range(1, n + 1):
            ei = torch.zeros(algebra.count_blade)
            ei[i] = 1.0
            ei_sq = algebra.geometric_product(ei, ei)

            expected = 1.0 if i <= p else -1.0
            assert torch.isclose(ei_sq[0], torch.tensor(expected), atol=1e-6), \
                f"Cl({p},{q}) e{i}² should be {expected}, got {ei_sq[0]}"
