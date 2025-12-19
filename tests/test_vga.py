"""
Test VGA (Vanilla Geometric Algebra) - Cl(n, 0)

Tests VGA specialization against clifford library for correctness verification.
"""

import pytest
import torch
import numpy as np

from fast_clifford import VGA

# Try to import clifford library for comparison
try:
    import clifford as cf
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False


class TestVGABasic:
    """Basic VGA functionality tests."""

    def test_vga1_creation(self):
        """Test VGA(1) creation."""
        vga = VGA(1)
        assert vga.p == 1
        assert vga.q == 0
        assert vga.r == 0
        assert vga.count_blade == 2
        assert vga.count_rotor == 1
        assert vga.dim_euclidean == 1
        assert vga.algebra_type == "vga"

    def test_vga2_creation(self):
        """Test VGA(2) creation."""
        vga = VGA(2)
        assert vga.count_blade == 4
        assert vga.count_rotor == 2
        assert vga.dim_euclidean == 2

    def test_vga3_creation(self):
        """Test VGA(3) creation."""
        vga = VGA(3)
        assert vga.count_blade == 8
        assert vga.count_rotor == 4
        assert vga.dim_euclidean == 3

    def test_vga4_creation(self):
        """Test VGA(4) creation."""
        vga = VGA(4)
        assert vga.count_blade == 16
        assert vga.count_rotor == 8
        assert vga.dim_euclidean == 4

    def test_vga5_creation(self):
        """Test VGA(5) creation."""
        vga = VGA(5)
        assert vga.count_blade == 32
        assert vga.count_rotor == 16
        assert vga.dim_euclidean == 5


class TestVGAEncodeDecode:
    """Test VGA encode/decode operations."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_encode_decode_roundtrip(self, n):
        """Test that encode -> decode recovers original vector."""
        vga = VGA(n)
        x = torch.randn(n)
        point = vga.encode(x)
        x_back = vga.decode(point)
        assert torch.allclose(x, x_back, atol=1e-5)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_encode_batch(self, n):
        """Test batch encoding."""
        vga = VGA(n)
        batch_size = 10
        x = torch.randn(batch_size, n)
        points = vga.encode(x)
        assert points.shape == (batch_size, vga.count_blade)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_decode_batch(self, n):
        """Test batch decoding."""
        vga = VGA(n)
        batch_size = 10
        x = torch.randn(batch_size, n)
        points = vga.encode(x)
        x_back = vga.decode(points)
        assert x_back.shape == (batch_size, n)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_encode_zeros(self):
        """Test encoding zero vector."""
        vga = VGA(3)
        x = torch.zeros(3)
        point = vga.encode(x)
        # Zero vector should give zero multivector
        assert torch.allclose(point, torch.zeros(8), atol=1e-5)


class TestVGAGeometricProduct:
    """Test VGA geometric product."""

    def test_basis_vectors_square_positive(self):
        """Test that basis vectors square to +1 in VGA."""
        vga = VGA(3)
        e1 = vga.encode(torch.tensor([1., 0., 0.]))
        e2 = vga.encode(torch.tensor([0., 1., 0.]))
        e3 = vga.encode(torch.tensor([0., 0., 1.]))

        # e1^2 = +1
        e1e1 = vga.geometric_product(e1, e1)
        assert torch.allclose(e1e1[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(e1e1[1:], torch.zeros(7), atol=1e-5)

        # e2^2 = +1
        e2e2 = vga.geometric_product(e2, e2)
        assert torch.allclose(e2e2[0], torch.tensor(1.0), atol=1e-5)

        # e3^2 = +1
        e3e3 = vga.geometric_product(e3, e3)
        assert torch.allclose(e3e3[0], torch.tensor(1.0), atol=1e-5)

    def test_basis_vectors_anticommute(self):
        """Test that distinct basis vectors anticommute."""
        vga = VGA(3)
        e1 = vga.encode(torch.tensor([1., 0., 0.]))
        e2 = vga.encode(torch.tensor([0., 1., 0.]))

        # e1*e2 should equal -e2*e1
        e1e2 = vga.geometric_product(e1, e2)
        e2e1 = vga.geometric_product(e2, e1)
        assert torch.allclose(e1e2, -e2e1, atol=1e-5)

    def test_bivector_product(self):
        """Test geometric product produces bivector."""
        vga = VGA(3)
        e1 = vga.encode(torch.tensor([1., 0., 0.]))
        e2 = vga.encode(torch.tensor([0., 1., 0.]))

        e12 = vga.geometric_product(e1, e2)
        # Result should be pure bivector (grade 2)
        # In VGA(3): indices 0=scalar, 1-3=vectors, 4-6=bivectors, 7=trivector
        assert torch.allclose(e12[:4], torch.zeros(4), atol=1e-5)
        assert e12[4].abs() > 0.5  # e12 component should be nonzero
        assert torch.allclose(e12[5:7], torch.zeros(2), atol=1e-5)


class TestVGAReverseInvolute:
    """Test VGA reverse and involute operations."""

    def test_reverse_vector(self):
        """Test reverse of a vector (should be unchanged)."""
        vga = VGA(3)
        v = vga.encode(torch.tensor([1., 2., 3.]))
        v_rev = vga.reverse(v)
        # Reverse of grade-1 element is unchanged
        assert torch.allclose(v, v_rev, atol=1e-5)

    def test_reverse_bivector(self):
        """Test reverse of a bivector (should be negated)."""
        vga = VGA(3)
        e1 = vga.encode(torch.tensor([1., 0., 0.]))
        e2 = vga.encode(torch.tensor([0., 1., 0.]))
        e12 = vga.geometric_product(e1, e2)

        e12_rev = vga.reverse(e12)
        # Reverse of grade-2 element is negated
        assert torch.allclose(e12, -e12_rev, atol=1e-5)

    def test_double_reverse(self):
        """Test that reverse(reverse(x)) = x."""
        vga = VGA(3)
        mv = torch.randn(8)
        mv_rev_rev = vga.reverse(vga.reverse(mv))
        assert torch.allclose(mv, mv_rev_rev, atol=1e-5)

    def test_involute_vector(self):
        """Test involute of a vector (should be negated)."""
        vga = VGA(3)
        v = vga.encode(torch.tensor([1., 2., 3.]))
        v_inv = vga.involute(v)
        # Involute of grade-1 element is negated
        assert torch.allclose(v, -v_inv, atol=1e-5)


class TestVGAOuterInner:
    """Test VGA outer and inner products."""

    def test_outer_product_vectors(self):
        """Test outer product of two vectors."""
        vga = VGA(3)
        e1 = vga.encode(torch.tensor([1., 0., 0.]))
        e2 = vga.encode(torch.tensor([0., 1., 0.]))

        e1_outer_e2 = vga.outer(e1, e2)
        e1_gp_e2 = vga.geometric_product(e1, e2)

        # For orthogonal vectors, outer product equals geometric product
        assert torch.allclose(e1_outer_e2, e1_gp_e2, atol=1e-5)

    def test_inner_product_same_vector(self):
        """Test inner product of vector with itself."""
        vga = VGA(3)
        v = vga.encode(torch.tensor([1., 2., 3.]))
        v_inner_v = vga.inner(v, v)

        # Inner product should give scalar = |v|^2
        expected_scalar = 1.0 + 4.0 + 9.0  # 14
        assert torch.allclose(v_inner_v[0], torch.tensor(expected_scalar), atol=1e-5)


@pytest.mark.skipif(not HAS_CLIFFORD, reason="clifford library not installed")
class TestVGACliffordComparison:
    """Compare VGA operations with clifford library."""

    def test_geometric_product_vga3(self):
        """Compare geometric product with clifford library for VGA(3)."""
        # Create clifford algebra
        layout, blades = cf.Cl(3)
        e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']

        # Create fast-clifford algebra
        vga = VGA(3)
        fc_e1 = vga.encode(torch.tensor([1., 0., 0.]))
        fc_e2 = vga.encode(torch.tensor([0., 1., 0.]))
        fc_e3 = vga.encode(torch.tensor([0., 0., 1.]))

        # Test e1 * e2
        cf_result = e1 * e2
        fc_result = vga.geometric_product(fc_e1, fc_e2)

        # Both should have non-zero bivector component
        # clifford result is a multivector with e12 blade
        assert np.any(cf_result.value != 0)  # Has some non-zero component
        assert fc_result[4].item() != 0  # Our e12 is at index 4

    def test_reverse_vga3(self):
        """Compare reverse with clifford library for VGA(3)."""
        layout, blades = cf.Cl(3)
        e1, e2 = blades['e1'], blades['e2']

        # Create bivector
        cf_e12 = e1 * e2
        cf_rev = ~cf_e12

        # fast-clifford
        vga = VGA(3)
        fc_e1 = vga.encode(torch.tensor([1., 0., 0.]))
        fc_e2 = vga.encode(torch.tensor([0., 1., 0.]))
        fc_e12 = vga.geometric_product(fc_e1, fc_e2)
        fc_rev = vga.reverse(fc_e12)

        # Reverse of bivector should negate it in both libraries
        # clifford: ~e12 = -e12
        assert np.allclose(cf_rev.value, -cf_e12.value)
        assert torch.allclose(fc_rev, -fc_e12, atol=1e-5)


class TestVGABatchOperations:
    """Test VGA batch operations."""

    def test_geometric_product_batch(self):
        """Test batched geometric product."""
        vga = VGA(3)
        batch_size = 10

        a = torch.randn(batch_size, 8)
        b = torch.randn(batch_size, 8)

        result = vga.geometric_product(a, b)
        assert result.shape == (batch_size, 8)

    def test_reverse_batch(self):
        """Test batched reverse."""
        vga = VGA(3)
        batch_size = 10

        mv = torch.randn(batch_size, 8)
        result = vga.reverse(mv)
        assert result.shape == (batch_size, 8)
