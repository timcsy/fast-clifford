"""
CGA0D Numerical Tests

Tests for CGA0D Cl(1,1) operations:
- Geometric product correctness
- Sandwich product correctness
- UPGC encoding/decoding
"""

import pytest
import torch
import numpy as np
from fast_clifford.algebras import cga0d


class TestGeometricProduct:
    """Test geometric product operations."""

    def test_geometric_product_identity(self):
        """Test that multiplying by identity (scalar 1) preserves value."""
        batch_size = 8
        a = torch.randn(batch_size, 4)
        identity = torch.zeros(batch_size, 4)
        identity[..., 0] = 1.0  # scalar = 1

        result = cga0d.geometric_product_full(identity, a)
        assert torch.allclose(result, a, atol=1e-6)

        result = cga0d.geometric_product_full(a, identity)
        assert torch.allclose(result, a, atol=1e-6)

    def test_geometric_product_e_plus_squared(self):
        """Test that e+^2 = 1."""
        batch_size = 4
        e_plus = torch.zeros(batch_size, 4)
        e_plus[..., 1] = 1.0

        result = cga0d.geometric_product_full(e_plus, e_plus)
        expected = torch.zeros(batch_size, 4)
        expected[..., 0] = 1.0  # scalar = 1

        assert torch.allclose(result, expected, atol=1e-6)

    def test_geometric_product_e_minus_squared(self):
        """Test that e-^2 = -1."""
        batch_size = 4
        e_minus = torch.zeros(batch_size, 4)
        e_minus[..., 2] = 1.0

        result = cga0d.geometric_product_full(e_minus, e_minus)
        expected = torch.zeros(batch_size, 4)
        expected[..., 0] = -1.0  # scalar = -1

        assert torch.allclose(result, expected, atol=1e-6)

    def test_geometric_product_bivector_squared(self):
        """Test that (e+-)^2 = 1 (in Cl(1,1) with signature +1,-1)."""
        batch_size = 4
        e_pm = torch.zeros(batch_size, 4)
        e_pm[..., 3] = 1.0

        result = cga0d.geometric_product_full(e_pm, e_pm)
        expected = torch.zeros(batch_size, 4)
        # e+-^2 = e+*e-*e+*e- = -e+*e+*e-*e- = -(1)*(-1) = 1
        expected[..., 0] = 1.0  # scalar = 1

        assert torch.allclose(result, expected, atol=1e-6)

    def test_geometric_product_anticommutator(self):
        """Test that e+ * e- + e- * e+ = 0 (orthogonal vectors anticommute)."""
        batch_size = 4
        e_plus = torch.zeros(batch_size, 4)
        e_plus[..., 1] = 1.0
        e_minus = torch.zeros(batch_size, 4)
        e_minus[..., 2] = 1.0

        ab = cga0d.geometric_product_full(e_plus, e_minus)
        ba = cga0d.geometric_product_full(e_minus, e_plus)

        # Inner product: 0.5 * (ab + ba) should be zero
        inner = 0.5 * (ab + ba)
        # The scalar part should be zero
        assert torch.allclose(inner[..., 0], torch.zeros(batch_size), atol=1e-6)


class TestReverse:
    """Test reverse operation."""

    def test_reverse_scalar(self):
        """Test that reverse of scalar is scalar."""
        batch_size = 4
        mv = torch.zeros(batch_size, 4)
        mv[..., 0] = 2.0

        result = cga0d.reverse_full(mv)
        assert torch.allclose(result, mv, atol=1e-6)

    def test_reverse_vector(self):
        """Test that reverse of grade 1 vectors is unchanged."""
        batch_size = 4
        mv = torch.zeros(batch_size, 4)
        mv[..., 1] = 1.0
        mv[..., 2] = 2.0

        result = cga0d.reverse_full(mv)
        assert torch.allclose(result, mv, atol=1e-6)

    def test_reverse_bivector(self):
        """Test that reverse of grade 2 bivector is negated."""
        batch_size = 4
        mv = torch.zeros(batch_size, 4)
        mv[..., 3] = 1.0

        result = cga0d.reverse_full(mv)
        expected = torch.zeros(batch_size, 4)
        expected[..., 3] = -1.0

        assert torch.allclose(result, expected, atol=1e-6)


class TestUPGC:
    """Test UPGC encoding and decoding."""

    def test_upgc_encode_shape(self):
        """Test that encoding produces correct shape."""
        batch_size = 8
        x = torch.randn(batch_size, 0)  # 0D input
        point = cga0d.upgc_encode(x)
        assert point.shape == (batch_size, 2)

    def test_upgc_encode_is_origin(self):
        """Test that encoded point is always the origin n_o."""
        batch_size = 8
        x = torch.randn(batch_size, 0)
        point = cga0d.upgc_encode(x)

        # Origin n_o = 0.5 * (e- - e+) = -0.5*e+ + 0.5*e-
        expected_e_plus = torch.full((batch_size,), -0.5)
        expected_e_minus = torch.full((batch_size,), 0.5)

        assert torch.allclose(point[..., 0], expected_e_plus, atol=1e-6)
        assert torch.allclose(point[..., 1], expected_e_minus, atol=1e-6)

    def test_upgc_decode_shape(self):
        """Test that decoding produces correct shape."""
        batch_size = 8
        point = torch.randn(batch_size, 2)
        x = cga0d.upgc_decode(point)
        assert x.shape == (batch_size, 0)


class TestSandwichProduct:
    """Test sandwich product operations."""

    def test_sandwich_identity_motor(self):
        """Test that identity motor (1, 0) preserves point."""
        batch_size = 8
        motor = torch.zeros(batch_size, 2)
        motor[..., 0] = 1.0  # scalar = 1, e+- = 0

        point = torch.randn(batch_size, 2)
        result = cga0d.sandwich_product_sparse(motor, point)

        assert torch.allclose(result, point, atol=1e-6)

    def test_sandwich_product_shape(self):
        """Test that sandwich product produces correct shape."""
        batch_size = 8
        motor = torch.randn(batch_size, 2)
        point = torch.randn(batch_size, 2)

        result = cga0d.sandwich_product_sparse(motor, point)
        assert result.shape == (batch_size, 2)

    def test_sandwich_product_normalized_motor(self):
        """Test sandwich product with normalized motor."""
        batch_size = 8
        # Create a normalized motor
        angle = torch.rand(batch_size) * 2 * np.pi
        motor = torch.stack([torch.cos(angle / 2), torch.sin(angle / 2)], dim=-1)

        point = cga0d.upgc_encode(torch.randn(batch_size, 0))
        result = cga0d.sandwich_product_sparse(motor, point)

        assert result.shape == (batch_size, 2)
        # The result should still be finite
        assert torch.isfinite(result).all()


class TestCliffordVerification:
    """Verify against clifford library (if available)."""

    @pytest.fixture
    def clifford_cga0d(self):
        """Create clifford CGA0D algebra."""
        try:
            from clifford import Cl, conformalize
            G_0, _ = Cl(0)
            layout, blades, stuff = conformalize(G_0)
            return layout, blades, stuff
        except ImportError:
            pytest.skip("clifford library not available")

    def test_null_basis_properties(self, clifford_cga0d):
        """Verify null basis properties."""
        layout, blades, stuff = clifford_cga0d
        eo = stuff['eo']
        einf = stuff['einf']

        # eo^2 = 0
        assert np.allclose((eo * eo).value, 0, atol=1e-10)

        # einf^2 = 0
        assert np.allclose((einf * einf).value, 0, atol=1e-10)

        # eo * einf = -1
        inner = eo * einf
        assert np.allclose(inner.value[0], -1.0, atol=1e-10)

    def test_geometric_product_vs_clifford(self, clifford_cga0d):
        """Compare geometric product with clifford library."""
        layout, blades, stuff = clifford_cga0d

        # Test with random multivectors
        np.random.seed(42)
        a_values = np.random.randn(4)
        b_values = np.random.randn(4)

        # Clifford computation
        a_cliff = layout.MultiVector(value=a_values)
        b_cliff = layout.MultiVector(value=b_values)
        result_cliff = a_cliff * b_cliff

        # PyTorch computation
        a_torch = torch.tensor(a_values, dtype=torch.float32).unsqueeze(0)
        b_torch = torch.tensor(b_values, dtype=torch.float32).unsqueeze(0)
        result_torch = cga0d.geometric_product_full(a_torch, b_torch)

        # Compare
        assert torch.allclose(
            result_torch.squeeze(0),
            torch.tensor(result_cliff.value, dtype=torch.float32),
            atol=1e-5
        )


class TestGradient:
    """Test gradient computation."""

    def test_geometric_product_gradient(self):
        """Test that geometric product supports gradient computation."""
        a = torch.randn(4, 4, requires_grad=True)
        b = torch.randn(4, 4, requires_grad=True)

        result = cga0d.geometric_product_full(a, b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

    def test_sandwich_product_gradient(self):
        """Test that sandwich product supports gradient computation."""
        motor = torch.randn(4, 2, requires_grad=True)
        point = torch.randn(4, 2, requires_grad=True)

        result = cga0d.sandwich_product_sparse(motor, point)
        loss = result.sum()
        loss.backward()

        assert motor.grad is not None
        assert point.grad is not None
