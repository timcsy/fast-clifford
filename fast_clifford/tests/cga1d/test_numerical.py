"""
Numerical tests for CGA1D Cl(2,1) algebra.

Tests geometric product, reverse, and sandwich product correctness
against the clifford library reference implementation.
"""

import pytest
import torch
import numpy as np
from clifford import Cl, conformalize

from fast_clifford.algebras import cga1d


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reference_algebra():
    """Create reference CGA1D algebra using clifford."""
    G1, _ = Cl(1)
    layout, blades, stuff = conformalize(G1)
    return layout, blades, stuff


@pytest.fixture
def random_multivector():
    """Generate random 8-component multivector."""
    return torch.randn(8, dtype=torch.float32)


@pytest.fixture
def random_batch_multivector():
    """Generate batch of random multivectors."""
    return torch.randn(8, 8, dtype=torch.float32)


# =============================================================================
# Geometric Product Tests
# =============================================================================

class TestGeometricProduct:
    """Test full 8x8 geometric product."""

    def test_scalar_multiplication(self):
        """Scalar × Scalar = Scalar."""
        a = torch.zeros(8)
        a[0] = 2.0
        b = torch.zeros(8)
        b[0] = 3.0

        result = cga1d.geometric_product_full(a, b)

        assert result[0].item() == pytest.approx(6.0)
        assert torch.allclose(result[1:], torch.zeros(7))

    def test_e1_square(self, reference_algebra):
        """e1² = +1 (Euclidean)."""
        layout, blades, _ = reference_algebra

        a = torch.zeros(8)
        a[1] = 1.0  # e1

        result = cga1d.geometric_product_full(a, a)

        # e1² = +1
        assert result[0].item() == pytest.approx(1.0)

    def test_ep_square(self, reference_algebra):
        """e+² = +1."""
        a = torch.zeros(8)
        a[2] = 1.0  # e+ at index 2

        result = cga1d.geometric_product_full(a, a)

        # e+² = +1
        assert result[0].item() == pytest.approx(1.0)

    def test_em_square_negative(self, reference_algebra):
        """e-² = -1 (negative signature)."""
        a = torch.zeros(8)
        a[3] = 1.0  # e- at index 3

        result = cga1d.geometric_product_full(a, a)

        # e-² = -1
        assert result[0].item() == pytest.approx(-1.0)

    def test_associativity(self):
        """(a × b) × c = a × (b × c)."""
        a = torch.randn(8)
        b = torch.randn(8)
        c = torch.randn(8)

        left = cga1d.geometric_product_full(
            cga1d.geometric_product_full(a, b), c
        )
        right = cga1d.geometric_product_full(
            a, cga1d.geometric_product_full(b, c)
        )

        assert torch.allclose(left, right, atol=1e-5)

    def test_batch_support(self):
        """Batch operations work correctly."""
        batch_size = 8
        a = torch.randn(batch_size, 8)
        b = torch.randn(batch_size, 8)

        result = cga1d.geometric_product_full(a, b)

        assert result.shape == (batch_size, 8)

        # Verify each batch element
        for i in range(batch_size):
            single = cga1d.geometric_product_full(a[i], b[i])
            assert torch.allclose(result[i], single)

    def test_matches_clifford_reference(self, reference_algebra):
        """Result matches clifford library."""
        layout, _, _ = reference_algebra

        # Random multivectors
        a_np = np.random.randn(8).astype(np.float32)
        b_np = np.random.randn(8).astype(np.float32)

        # Clifford reference
        a_mv = layout.MultiVector(value=a_np.astype(np.float64))
        b_mv = layout.MultiVector(value=b_np.astype(np.float64))
        ref_result = (a_mv * b_mv).value

        # Our implementation
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        our_result = cga1d.geometric_product_full(a_torch, b_torch).numpy()

        np.testing.assert_allclose(our_result, ref_result.astype(np.float32), atol=1e-5)


# =============================================================================
# Reverse Tests
# =============================================================================

class TestReverse:
    """Test reverse operation."""

    def test_scalar_unchanged(self):
        """Scalar is unchanged by reverse."""
        mv = torch.zeros(8)
        mv[0] = 5.0

        result = cga1d.reverse_full(mv)

        assert result[0].item() == pytest.approx(5.0)

    def test_vector_unchanged(self):
        """Vector (grade 1) is unchanged by reverse."""
        mv = torch.zeros(8)
        mv[1] = 1.0  # e1
        mv[2] = 2.0  # e+
        mv[3] = 3.0  # e-

        result = cga1d.reverse_full(mv)

        assert result[1].item() == pytest.approx(1.0)
        assert result[2].item() == pytest.approx(2.0)
        assert result[3].item() == pytest.approx(3.0)

    def test_bivector_negated(self):
        """Bivector (grade 2) is negated by reverse."""
        mv = torch.zeros(8)
        mv[4] = 1.0  # e1+
        mv[5] = 2.0  # e1-
        mv[6] = 3.0  # e+-

        result = cga1d.reverse_full(mv)

        assert result[4].item() == pytest.approx(-1.0)
        assert result[5].item() == pytest.approx(-2.0)
        assert result[6].item() == pytest.approx(-3.0)

    def test_trivector_negated(self):
        """Trivector (grade 3) is negated by reverse."""
        mv = torch.zeros(8)
        mv[7] = 1.0  # e1+-

        result = cga1d.reverse_full(mv)

        assert result[7].item() == pytest.approx(-1.0)

    def test_double_reverse_identity(self):
        """Reverse twice returns original."""
        mv = torch.randn(8)
        result = cga1d.reverse_full(cga1d.reverse_full(mv))
        assert torch.allclose(mv, result)


# =============================================================================
# UPGC Encode/Decode Tests
# =============================================================================

class TestUPGCEncoding:
    """Test UPGC point encoding and decoding."""

    def test_encode_origin(self):
        """Encoding origin gives n_o."""
        x = torch.zeros(1)
        point = cga1d.upgc_encode(x)

        # n_o = 0.5*(e- - e+) -> e+ = -0.5, e- = 0.5
        assert point[0].item() == pytest.approx(0.0)   # e1
        assert point[1].item() == pytest.approx(-0.5)  # e+
        assert point[2].item() == pytest.approx(0.5)   # e-

    def test_encode_unit_x(self):
        """Encoding x=1 gives correct point."""
        x = torch.tensor([1.0])
        point = cga1d.upgc_encode(x)

        # X = n_o + x + 0.5|x|² n_inf
        # |x|² = 1, so:
        # e1 = 1
        # e+ = -0.5 + 0.5*1 = 0
        # e- = 0.5 + 0.5*1 = 1
        assert point[0].item() == pytest.approx(1.0)   # e1
        assert point[1].item() == pytest.approx(0.0)   # e+
        assert point[2].item() == pytest.approx(1.0)   # e-

    def test_encode_x_equals_2(self):
        """Encoding x=2 gives correct point."""
        x = torch.tensor([2.0])
        point = cga1d.upgc_encode(x)

        # |x|² = 4, so:
        # e1 = 2
        # e+ = -0.5 + 0.5*4 = 1.5
        # e- = 0.5 + 0.5*4 = 2.5
        assert point[0].item() == pytest.approx(2.0)   # e1
        assert point[1].item() == pytest.approx(1.5)   # e+
        assert point[2].item() == pytest.approx(2.5)   # e-

    def test_decode_inverts_encode(self):
        """Decode inverts encode for various points."""
        for _ in range(10):
            x = torch.randn(1)
            point = cga1d.upgc_encode(x)
            decoded = cga1d.upgc_decode(point)
            assert torch.allclose(x, decoded)

    def test_batch_encode_decode(self):
        """Batch encoding/decoding works."""
        batch_size = 8
        x = torch.randn(batch_size, 1)

        points = cga1d.upgc_encode(x)
        decoded = cga1d.upgc_decode(points)

        assert points.shape == (batch_size, 3)
        assert decoded.shape == (batch_size, 1)
        assert torch.allclose(x, decoded)


# =============================================================================
# Sandwich Product Tests
# =============================================================================

class TestSandwichProduct:
    """Test sparse sandwich product M × X × M̃."""

    def test_identity_motor(self):
        """Identity motor leaves point unchanged."""
        motor = torch.zeros(4)
        motor[0] = 1.0  # scalar = 1

        x = torch.randn(1)
        point = cga1d.upgc_encode(x)

        result = cga1d.sandwich_product_sparse(motor, point)

        assert torch.allclose(point, result, atol=1e-6)

    def test_translation(self, reference_algebra):
        """Translation by t works correctly."""
        layout, blades, stuff = reference_algebra

        # Translation motor: T = 1 + 0.5 * t * einf * e1 (positive sign for forward translation)
        e1 = blades['e1']
        einf = stuff['einf']
        up = stuff['up']
        down = stuff['down']

        t = 3.0  # translate by 3

        # Create translation motor using clifford
        T = 1 + 0.5 * t * einf * e1

        # Get expected result from clifford first
        x_mv = 2.0 * e1
        X_mv = up(x_mv)
        X_new_mv = T * X_mv * ~T
        x_out_mv = down(X_new_mv)
        expected = torch.tensor([float(x_out_mv.value[1])], dtype=torch.float32)

        # Convert to sparse representation
        # Motor layout: [scalar, e1+, e1-, e+-]
        motor = torch.zeros(4, dtype=torch.float32)
        motor[0] = float(T.value[0])   # scalar
        motor[1] = float(T.value[4])   # e1e+ (index 4 in CGA1D)
        motor[2] = float(T.value[5])   # e1e- (index 5 in CGA1D)
        motor[3] = float(T.value[6])   # e+e- (index 6 in CGA1D)

        # Create a point at x=2
        x = torch.tensor([2.0], dtype=torch.float32)
        point = cga1d.upgc_encode(x)

        # Transform
        result = cga1d.sandwich_product_sparse(motor, point)
        decoded = cga1d.upgc_decode(result)

        assert torch.allclose(decoded, expected, atol=1e-5)

    def test_batch_sandwich(self):
        """Batch sandwich product works correctly."""
        batch_size = 8

        motor = torch.zeros(batch_size, 4)
        motor[:, 0] = 1.0  # Identity motors

        x_batch = torch.randn(batch_size, 1)
        points = cga1d.upgc_encode(x_batch)

        result = cga1d.sandwich_product_sparse(motor, points)

        assert result.shape == (batch_size, 3)
        assert torch.allclose(points, result, atol=1e-6)


# =============================================================================
# Null Basis Tests
# =============================================================================

class TestNullBasis:
    """Test null basis properties."""

    def test_eo_squared_zero(self, reference_algebra):
        """n_o² = 0."""
        props = cga1d.verify_null_basis()
        assert props['eo_squared_zero']

    def test_einf_squared_zero(self, reference_algebra):
        """n_∞² = 0."""
        props = cga1d.verify_null_basis()
        assert props['einf_squared_zero']

    def test_eo_einf_inner_product(self, reference_algebra):
        """n_o · n_∞ = -1."""
        props = cga1d.verify_null_basis()
        assert props['eo_einf_minus_one']


# =============================================================================
# Motor Reverse Tests
# =============================================================================

class TestMotorReverse:
    """Test motor-specific reverse operation."""

    def test_identity_motor_reverse(self):
        """Identity motor reverse is identity."""
        motor = torch.zeros(4)
        motor[0] = 1.0

        result = cga1d.reverse_motor(motor)

        assert result[0].item() == pytest.approx(1.0)
        assert torch.allclose(result[1:], torch.zeros(3))

    def test_grade2_negated(self):
        """Grade 2 components are negated."""
        motor = torch.zeros(4)
        motor[1] = 1.0  # e1+ (Grade 2)
        motor[2] = 2.0  # e1- (Grade 2)
        motor[3] = 3.0  # e+- (Grade 2)

        result = cga1d.reverse_motor(motor)

        assert result[1].item() == pytest.approx(-1.0)
        assert result[2].item() == pytest.approx(-2.0)
        assert result[3].item() == pytest.approx(-3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
