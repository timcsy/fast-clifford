"""
Numerical tests for CGA2D Cl(3,1) algebra.

Tests geometric product, reverse, and sandwich product correctness
against the clifford library reference implementation.
"""

import pytest
import torch
import numpy as np
from clifford import Cl, conformalize

from fast_clifford.algebras import cga2d


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reference_algebra():
    """Create reference CGA2D algebra using clifford."""
    G2, _ = Cl(2)
    layout, blades, stuff = conformalize(G2)
    return layout, blades, stuff


@pytest.fixture
def random_multivector():
    """Generate random 16-component multivector."""
    return torch.randn(16, dtype=torch.float32)


@pytest.fixture
def random_batch_multivector():
    """Generate batch of random multivectors."""
    return torch.randn(8, 16, dtype=torch.float32)


# =============================================================================
# Geometric Product Tests
# =============================================================================

class TestGeometricProduct:
    """Test full 16x16 geometric product."""

    def test_scalar_multiplication(self):
        """Scalar × Scalar = Scalar."""
        a = torch.zeros(16)
        a[0] = 2.0
        b = torch.zeros(16)
        b[0] = 3.0

        result = cga2d.geometric_product_full(a, b)

        assert result[0].item() == pytest.approx(6.0)
        assert torch.allclose(result[1:], torch.zeros(15))

    def test_vector_square(self, reference_algebra):
        """e1² = +1 (Euclidean)."""
        layout, blades, _ = reference_algebra

        a = torch.zeros(16)
        a[1] = 1.0  # e1

        result = cga2d.geometric_product_full(a, a)

        # e1² = +1
        assert result[0].item() == pytest.approx(1.0)

    def test_ep_em_anticommute(self, reference_algebra):
        """e+ × e- = e+e- (bivector)."""
        layout, blades, _ = reference_algebra

        ep = torch.zeros(16)
        ep[3] = 1.0  # e+ at index 3

        em = torch.zeros(16)
        em[4] = 1.0  # e- at index 4

        result = cga2d.geometric_product_full(ep, em)

        # Should have e+e- component non-zero
        assert result[10].item() != 0  # e+e- at index 10

    def test_em_square_negative(self, reference_algebra):
        """e-² = -1 (negative signature)."""
        layout, blades, _ = reference_algebra

        em = torch.zeros(16)
        em[4] = 1.0  # e-

        result = cga2d.geometric_product_full(em, em)

        # e-² = -1
        assert result[0].item() == pytest.approx(-1.0)

    def test_associativity(self):
        """(a × b) × c = a × (b × c)."""
        a = torch.randn(16)
        b = torch.randn(16)
        c = torch.randn(16)

        left = cga2d.geometric_product_full(
            cga2d.geometric_product_full(a, b), c
        )
        right = cga2d.geometric_product_full(
            a, cga2d.geometric_product_full(b, c)
        )

        assert torch.allclose(left, right, atol=1e-5)

    def test_batch_support(self):
        """Batch operations work correctly."""
        batch_size = 8
        a = torch.randn(batch_size, 16)
        b = torch.randn(batch_size, 16)

        result = cga2d.geometric_product_full(a, b)

        assert result.shape == (batch_size, 16)

        # Verify each batch element
        for i in range(batch_size):
            single = cga2d.geometric_product_full(a[i], b[i])
            assert torch.allclose(result[i], single)

    def test_matches_clifford_reference(self, reference_algebra):
        """Result matches clifford library."""
        layout, _, _ = reference_algebra

        # Random multivectors
        a_np = np.random.randn(16).astype(np.float32)
        b_np = np.random.randn(16).astype(np.float32)

        # Clifford reference
        a_mv = layout.MultiVector(value=a_np.astype(np.float64))
        b_mv = layout.MultiVector(value=b_np.astype(np.float64))
        ref_result = (a_mv * b_mv).value

        # Our implementation
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        our_result = cga2d.geometric_product_full(a_torch, b_torch).numpy()

        np.testing.assert_allclose(our_result, ref_result.astype(np.float32), atol=1e-5)


# =============================================================================
# Reverse Tests
# =============================================================================

class TestReverse:
    """Test reverse operation."""

    def test_scalar_unchanged(self):
        """Scalar is unchanged by reverse."""
        mv = torch.zeros(16)
        mv[0] = 5.0

        result = cga2d.reverse_full(mv)

        assert result[0].item() == pytest.approx(5.0)

    def test_vector_unchanged(self):
        """Vector (grade 1) is unchanged by reverse."""
        mv = torch.zeros(16)
        mv[1] = 1.0  # e1
        mv[2] = 2.0  # e2
        mv[3] = 3.0  # e+
        mv[4] = 4.0  # e-

        result = cga2d.reverse_full(mv)

        assert result[1].item() == pytest.approx(1.0)
        assert result[2].item() == pytest.approx(2.0)
        assert result[3].item() == pytest.approx(3.0)
        assert result[4].item() == pytest.approx(4.0)

    def test_bivector_negated(self):
        """Bivector (grade 2) is negated by reverse."""
        mv = torch.zeros(16)
        mv[5] = 1.0   # e12
        mv[10] = 2.0  # e+-

        result = cga2d.reverse_full(mv)

        assert result[5].item() == pytest.approx(-1.0)
        assert result[10].item() == pytest.approx(-2.0)

    def test_trivector_negated(self):
        """Trivector (grade 3) is negated by reverse."""
        mv = torch.zeros(16)
        mv[11] = 1.0  # e12+

        result = cga2d.reverse_full(mv)

        assert result[11].item() == pytest.approx(-1.0)

    def test_quadvector_unchanged(self):
        """Quadvector (grade 4) is unchanged by reverse."""
        mv = torch.zeros(16)
        mv[15] = 1.0  # e12+-

        result = cga2d.reverse_full(mv)

        assert result[15].item() == pytest.approx(1.0)

    def test_double_reverse_identity(self):
        """Reverse twice returns original."""
        mv = torch.randn(16)
        result = cga2d.reverse_full(cga2d.reverse_full(mv))
        assert torch.allclose(mv, result)


# =============================================================================
# UPGC Encode/Decode Tests
# =============================================================================

class TestUPGCEncoding:
    """Test UPGC point encoding and decoding."""

    def test_encode_origin(self):
        """Encoding origin gives n_o."""
        x = torch.zeros(2)
        point = cga2d.upgc_encode(x)

        # n_o = 0.5*(e- - e+) -> e+ = -0.5, e- = 0.5
        assert point[0].item() == pytest.approx(0.0)  # e1
        assert point[1].item() == pytest.approx(0.0)  # e2
        assert point[2].item() == pytest.approx(-0.5)  # e+
        assert point[3].item() == pytest.approx(0.5)   # e-

    def test_encode_unit_x(self):
        """Encoding (1,0) gives correct point."""
        x = torch.tensor([1.0, 0.0])
        point = cga2d.upgc_encode(x)

        # X = n_o + x + 0.5|x|² n_inf
        # |x|² = 1, so:
        # e1 = 1, e2 = 0
        # e+ = -0.5 + 0.5*1 = 0
        # e- = 0.5 + 0.5*1 = 1
        assert point[0].item() == pytest.approx(1.0)   # e1
        assert point[1].item() == pytest.approx(0.0)   # e2
        assert point[2].item() == pytest.approx(0.0)   # e+
        assert point[3].item() == pytest.approx(1.0)   # e-

    def test_decode_inverts_encode(self):
        """Decode inverts encode for various points."""
        for _ in range(10):
            x = torch.randn(2)
            point = cga2d.upgc_encode(x)
            decoded = cga2d.upgc_decode(point)
            assert torch.allclose(x, decoded)

    def test_batch_encode_decode(self):
        """Batch encoding/decoding works."""
        batch_size = 8
        x = torch.randn(batch_size, 2)

        points = cga2d.upgc_encode(x)
        decoded = cga2d.upgc_decode(points)

        assert points.shape == (batch_size, 4)
        assert decoded.shape == (batch_size, 2)
        assert torch.allclose(x, decoded)


# =============================================================================
# Sandwich Product Tests
# =============================================================================

class TestSandwichProduct:
    """Test sparse sandwich product M × X × M̃."""

    def test_identity_ev(self):
        """Identity EvenVersor leaves point unchanged."""
        ev = torch.zeros(8)
        ev[0] = 1.0  # scalar = 1

        x = torch.randn(2)
        point = cga2d.upgc_encode(x)

        result = cga2d.sandwich_product_sparse(ev, point)

        assert torch.allclose(point, result)

    def test_matches_full_product(self, reference_algebra):
        """Sparse sandwich matches full geometric product."""
        layout, blades, stuff = reference_algebra

        # Create an EvenVersor using clifford
        e1, e2 = blades['e1'], blades['e2']
        up = stuff['up']
        down = stuff['down']

        # Rotation by 30 degrees in e1e2 plane
        angle = np.pi / 6
        R = np.cos(angle / 2) + np.sin(angle / 2) * (e1 ^ e2)

        # Get expected result from clifford first
        x_mv = 1.0 * e1
        X_mv = up(x_mv)
        X_new_mv = R * X_mv * ~R
        x_out_mv = down(X_new_mv)
        expected = torch.tensor(
            [float(x_out_mv.value[1]), float(x_out_mv.value[2])],
            dtype=torch.float32
        )

        # Convert EvenVersor to sparse representation
        ev = torch.zeros(8, dtype=torch.float32)
        ev[0] = float(R.value[0])   # scalar
        ev[1] = float(R.value[5])   # e12 at index 5 in full, index 1 in sparse

        # Create a point using our implementation
        x = torch.tensor([1.0, 0.0], dtype=torch.float32)
        point = cga2d.upgc_encode(x)

        # Transform
        result = cga2d.sandwich_product_sparse(ev, point)
        decoded = cga2d.upgc_decode(result)

        assert torch.allclose(decoded, expected, atol=1e-5)

    def test_batch_sandwich(self):
        """Batch sandwich product works correctly."""
        batch_size = 8

        ev = torch.zeros(batch_size, 8)
        ev[:, 0] = 1.0  # Identity EvenVersors

        points = torch.randn(batch_size, 4)
        # Normalize to valid UPGC points
        x_batch = torch.randn(batch_size, 2)
        points = cga2d.upgc_encode(x_batch)

        result = cga2d.sandwich_product_sparse(ev, points)

        assert result.shape == (batch_size, 4)
        assert torch.allclose(points, result)


# =============================================================================
# Null Basis Tests
# =============================================================================

class TestNullBasis:
    """Test null basis properties."""

    def test_eo_squared_zero(self, reference_algebra):
        """n_o² = 0."""
        props = cga2d.verify_null_basis()
        assert props['eo_squared_zero']

    def test_einf_squared_zero(self, reference_algebra):
        """n_∞² = 0."""
        props = cga2d.verify_null_basis()
        assert props['einf_squared_zero']

    def test_eo_einf_inner_product(self, reference_algebra):
        """n_o · n_∞ = -1."""
        props = cga2d.verify_null_basis()
        assert props['eo_einf_minus_one']


# =============================================================================
# EvenVersor Reverse Tests
# =============================================================================

class TestEvenVersorReverse:
    """Test EvenVersor-specific reverse operation."""

    def test_identity_ev_reverse(self):
        """Identity EvenVersor reverse is identity."""
        ev = torch.zeros(8)
        ev[0] = 1.0

        result = cga2d.reverse_even_versor(ev)

        assert result[0].item() == pytest.approx(1.0)
        assert torch.allclose(result[1:], torch.zeros(7))

    def test_grade2_negated(self):
        """Grade 2 components are negated."""
        ev = torch.zeros(8)
        ev[1] = 1.0  # e12 (Grade 2)

        result = cga2d.reverse_even_versor(ev)

        assert result[1].item() == pytest.approx(-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
