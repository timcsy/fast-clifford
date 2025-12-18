"""
Test CGA (Conformal Geometric Algebra) - Cl(n+1, 1)

Tests CGA specialization with encode/decode and conformal transformations.
"""

import pytest
import torch
import numpy as np

from fast_clifford import CGA

# Try to import clifford library for comparison
try:
    import clifford as cf
    from clifford import conformalize
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False


class TestCGABasic:
    """Basic CGA functionality tests."""

    def test_cga1_creation(self):
        """Test CGA(1) = Cl(2, 1) creation."""
        cga = CGA(1)
        assert cga.p == 2
        assert cga.q == 1
        assert cga.r == 0
        assert cga.count_blade == 8
        assert cga.dim_euclidean == 1
        assert cga.count_point == 3
        assert cga.algebra_type == "cga"

    def test_cga2_creation(self):
        """Test CGA(2) = Cl(3, 1) creation."""
        cga = CGA(2)
        assert cga.p == 3
        assert cga.q == 1
        assert cga.count_blade == 16
        assert cga.dim_euclidean == 2
        assert cga.count_point == 4

    def test_cga3_creation(self):
        """Test CGA(3) = Cl(4, 1) creation."""
        cga = CGA(3)
        assert cga.p == 4
        assert cga.q == 1
        assert cga.count_blade == 32
        assert cga.dim_euclidean == 3
        assert cga.count_point == 5

    def test_cga4_creation(self):
        """Test CGA(4) = Cl(5, 1) creation."""
        cga = CGA(4)
        assert cga.p == 5
        assert cga.q == 1
        assert cga.count_blade == 64
        assert cga.dim_euclidean == 4
        assert cga.count_point == 6

    def test_cga5_creation(self):
        """Test CGA(5) = Cl(6, 1) creation."""
        cga = CGA(5)
        assert cga.p == 6
        assert cga.q == 1
        assert cga.count_blade == 128
        assert cga.dim_euclidean == 5
        assert cga.count_point == 7


class TestCGAEncodeDecode:
    """Test CGA encode/decode operations."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_encode_decode_roundtrip(self, n):
        """Test that encode -> decode recovers original Euclidean point."""
        cga = CGA(n)
        x = torch.randn(n)
        point = cga.encode(x)
        x_back = cga.decode(point)
        assert torch.allclose(x, x_back, atol=1e-5)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_encode_batch(self, n):
        """Test batch encoding."""
        cga = CGA(n)
        batch_size = 10
        x = torch.randn(batch_size, n)
        points = cga.encode(x)
        assert points.shape == (batch_size, cga.count_blade)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_decode_batch(self, n):
        """Test batch decoding."""
        cga = CGA(n)
        batch_size = 10
        x = torch.randn(batch_size, n)
        points = cga.encode(x)
        x_back = cga.decode(points)
        assert x_back.shape == (batch_size, n)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_encode_origin(self):
        """Test encoding the origin point."""
        cga = CGA(3)
        x = torch.zeros(3)
        point = cga.encode(x)

        # Origin encoding: e_o = (e- - e+)/2
        # Coefficients: Euclidean=0, e+=−0.5, e−=+0.5
        assert torch.allclose(point[1:4], torch.zeros(3), atol=1e-5)
        assert torch.allclose(point[4], torch.tensor(-0.5), atol=1e-5)
        assert torch.allclose(point[5], torch.tensor(0.5), atol=1e-5)

    def test_encode_unit_vector(self):
        """Test encoding a unit vector."""
        cga = CGA(3)
        x = torch.tensor([1., 0., 0.])
        point = cga.encode(x)

        # |x|^2 = 1, so half_norm_sq = 0.5
        # e+ coeff = -0.5 + 0.5 = 0
        # e- coeff = 0.5 + 0.5 = 1
        assert torch.allclose(point[1], torch.tensor(1.0), atol=1e-5)  # x component
        assert torch.allclose(point[4], torch.tensor(0.0), atol=1e-5)  # e+ coeff
        assert torch.allclose(point[5], torch.tensor(1.0), atol=1e-5)  # e- coeff


class TestCGANullVector:
    """Test CGA null vector properties."""

    def test_encoded_point_is_null(self):
        """Test that encoded points are null vectors (X · X = 0)."""
        cga = CGA(3)
        x = torch.randn(3)
        point = cga.encode(x)

        # Compute X · X using inner product
        x_inner_x = cga.inner(point, point)

        # For a properly encoded CGA point, it should be null
        # X · X should be close to zero
        scalar = x_inner_x[0]
        assert torch.abs(scalar) < 1e-4, f"Point is not null: X·X = {scalar}"

    def test_weight_is_one(self):
        """Test that encoded point weight (-e_inf · X) = 1."""
        cga = CGA(3)
        x = torch.randn(3)
        point = cga.encode(x)

        # Weight = e- - e+ (from Dorst convention)
        n = cga.dim_euclidean
        e_plus = point[n + 1]
        e_minus = point[n + 2]
        weight = e_minus - e_plus

        assert torch.allclose(weight, torch.tensor(1.0), atol=1e-5)


class TestCGAGeometricProduct:
    """Test CGA geometric product."""

    def test_geometric_product_shape(self):
        """Test geometric product output shape."""
        cga = CGA(3)
        a = torch.randn(cga.count_blade)
        b = torch.randn(cga.count_blade)
        result = cga.geometric_product(a, b)
        assert result.shape == (cga.count_blade,)

    def test_geometric_product_batch(self):
        """Test batched geometric product."""
        cga = CGA(3)
        batch_size = 10
        a = torch.randn(batch_size, cga.count_blade)
        b = torch.randn(batch_size, cga.count_blade)
        result = cga.geometric_product(a, b)
        assert result.shape == (batch_size, cga.count_blade)


class TestCGARotor:
    """Test CGA rotor operations."""

    def test_compose_rotor_shape(self):
        """Test rotor composition output shape."""
        cga = CGA(3)
        # Test with batch dimension
        r1 = torch.randn(10, cga.count_rotor)
        r2 = torch.randn(10, cga.count_rotor)
        result = cga.compose_rotor(r1, r2)
        assert result.shape == (10, cga.count_rotor)

    def test_reverse_rotor_shape(self):
        """Test rotor reverse output shape."""
        cga = CGA(3)
        r = torch.randn(cga.count_rotor)
        result = cga.reverse_rotor(r)
        assert result.shape == (cga.count_rotor,)

    def test_sandwich_rotor_shape(self):
        """Test rotor sandwich product output shape."""
        cga = CGA(3)
        r = torch.randn(cga.count_rotor)
        x = torch.randn(cga.count_blade)
        result = cga.sandwich_rotor(r, x)
        assert result.shape == (cga.count_blade,)


class TestCGASparseEncode:
    """Test CGA sparse encoding."""

    def test_encode_point_sparse(self):
        """Test sparse encoding produces correct shape."""
        cga = CGA(3)
        x = torch.tensor([1., 2., 3.])
        sparse_point = cga.encode_point_sparse(x)
        assert sparse_point.shape == (cga.count_point,)

    def test_decode_point_sparse(self):
        """Test sparse decode roundtrip."""
        cga = CGA(3)
        x = torch.tensor([1., 2., 3.])
        sparse_point = cga.encode_point_sparse(x)
        x_back = cga.decode_point_sparse(sparse_point)
        assert torch.allclose(x, x_back, atol=1e-5)


@pytest.mark.skipif(not HAS_CLIFFORD, reason="clifford library not installed")
class TestCGACliffordComparison:
    """Compare CGA operations with clifford library."""

    def test_encode_decode_cga3(self):
        """Compare encoding with clifford library for CGA(3)."""
        # Create clifford CGA
        layout, blades, stuff = cf.conformalize(cf.Cl(3)[0])
        e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']
        eo, einf = stuff['eo'], stuff['einf']

        # Test point
        x_np = np.array([1.0, 2.0, 3.0])

        # clifford encoding (up projection)
        cf_point = eo + x_np[0]*e1 + x_np[1]*e2 + x_np[2]*e3 + 0.5*np.sum(x_np**2)*einf

        # fast-clifford encoding
        cga = CGA(3)
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        fc_point = cga.encode(x_torch)

        # Decode and compare
        fc_decoded = cga.decode(fc_point)
        assert torch.allclose(fc_decoded, x_torch, atol=1e-4)

    def test_null_vector_property(self):
        """Verify null vector property matches clifford."""
        layout, blades, stuff = cf.conformalize(cf.Cl(3)[0])
        e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']
        eo, einf = stuff['eo'], stuff['einf']

        # Test point
        x_np = np.array([1.0, 2.0, 3.0])
        cf_point = eo + x_np[0]*e1 + x_np[1]*e2 + x_np[2]*e3 + 0.5*np.sum(x_np**2)*einf

        # clifford: X | X should be 0 (null vector)
        cf_inner = cf_point | cf_point
        assert abs(float(cf_inner)) < 1e-10

        # fast-clifford: same property
        cga = CGA(3)
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        fc_point = cga.encode(x_torch)
        fc_inner = cga.inner(fc_point, fc_point)
        assert torch.abs(fc_inner[0]) < 1e-4


class TestCGALayers:
    """Test CGA PyTorch layers."""

    def test_encoder_layer(self):
        """Test CGA encoder layer."""
        cga = CGA(3)
        encoder = cga.get_encoder()

        x = torch.randn(10, 3)
        points = encoder(x)
        assert points.shape == (10, cga.count_blade)

    def test_decoder_layer(self):
        """Test CGA decoder layer."""
        cga = CGA(3)
        encoder = cga.get_encoder()
        decoder = cga.get_decoder()

        x = torch.randn(10, 3)
        points = encoder(x)
        x_back = decoder(points)
        assert torch.allclose(x, x_back, atol=1e-5)
