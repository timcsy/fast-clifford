"""
CGA Unified Interface Tests

Tests for:
- CGA(n) factory function
- Cl(p, q, r) factory function
- CGAAlgebraBase interface compliance
"""

import pytest
import torch
import warnings

from fast_clifford import CGA, Cl
from fast_clifford.cga import CGAAlgebraBase, HardcodedCGAWrapper


class TestCGAFactory:
    """Test CGA(n) factory function."""

    @pytest.mark.parametrize("n,expected_blades", [
        (0, 4),
        (1, 8),
        (2, 16),
        (3, 32),
        (4, 64),
        (5, 128),
    ])
    def test_cga_hardcoded_blade_counts(self, n, expected_blades):
        """Test that CGA(n) for n=0-5 returns correct blade count."""
        cga = CGA(n)
        assert cga.blade_count == expected_blades

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_cga_hardcoded_returns_wrapper(self, n):
        """Test that CGA(n) for n=0-5 returns HardcodedCGAWrapper."""
        cga = CGA(n)
        assert isinstance(cga, HardcodedCGAWrapper)

    def test_cga_runtime_returns_runtime_algebra(self):
        """Test that CGA(6+) returns RuntimeCGAAlgebra."""
        from fast_clifford.cga.runtime import RuntimeCGAAlgebra
        cga = CGA(6)
        assert isinstance(cga, RuntimeCGAAlgebra)

    def test_cga_negative_dimension_raises(self):
        """Test that CGA(-1) raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            CGA(-1)

    def test_cga_high_dimension_warning(self):
        """Test that CGA(15+) emits memory warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CGA(15)
            assert len(w) == 1
            assert "memory" in str(w[0].message).lower()

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_cga_implements_interface(self, n):
        """Test that CGA(n) implements CGAAlgebraBase interface."""
        cga = CGA(n)
        assert isinstance(cga, CGAAlgebraBase)

    @pytest.mark.parametrize("n,expected_notation", [
        (0, "Cl(1,1,0)"),
        (1, "Cl(2,1,0)"),
        (2, "Cl(3,1,0)"),
        (3, "Cl(4,1,0)"),
    ])
    def test_cga_clifford_notation(self, n, expected_notation):
        """Test that clifford_notation is correct."""
        cga = CGA(n)
        assert cga.clifford_notation == expected_notation


class TestClFactory:
    """Test Cl(p, q, r) factory function."""

    @pytest.mark.parametrize("p,q,r,expected_n", [
        (1, 1, 0, 0),
        (2, 1, 0, 1),
        (3, 1, 0, 2),
        (4, 1, 0, 3),
        (5, 1, 0, 4),
        (6, 1, 0, 5),
    ])
    def test_cl_cga_signature_routes_to_cga(self, p, q, r, expected_n):
        """Test that Cl(p, 1, 0) routes to CGA(p-1)."""
        algebra = Cl(p, q, r)
        assert algebra.euclidean_dim == expected_n

    def test_cl_default_r_is_zero(self):
        """Test that r defaults to 0."""
        algebra = Cl(4, 1)  # No r specified
        assert algebra.euclidean_dim == 3  # CGA3D

    def test_cl_non_cga_signature_warning(self):
        """Test that non-CGA signatures emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Cl(3, 0, 0)  # Pure 3D GA, not CGA
            assert len(w) == 1
            assert "not a CGA" in str(w[0].message)

    def test_cl_negative_raises(self):
        """Test that negative signature raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Cl(-1, 1, 0)


class TestCGAAlgebraBaseInterface:
    """Test CGAAlgebraBase interface compliance."""

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_properties_exist(self, n):
        """Test that all required properties exist."""
        cga = CGA(n)

        # Properties
        assert hasattr(cga, 'euclidean_dim')
        assert hasattr(cga, 'blade_count')
        assert hasattr(cga, 'point_count')
        assert hasattr(cga, 'even_versor_count')
        assert hasattr(cga, 'signature')
        assert hasattr(cga, 'clifford_notation')

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_core_methods_exist(self, n):
        """Test that all core methods exist."""
        cga = CGA(n)

        # Core operations
        assert callable(cga.upgc_encode)
        assert callable(cga.upgc_decode)
        assert callable(cga.geometric_product_full)
        assert callable(cga.sandwich_product_sparse)
        assert callable(cga.reverse_full)
        assert callable(cga.reverse_even_versor)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_layer_factory_methods_exist(self, n):
        """Test that layer factory methods exist."""
        cga = CGA(n)

        # Layer factories
        assert callable(cga.get_care_layer)
        assert callable(cga.get_encoder)
        assert callable(cga.get_decoder)
        assert callable(cga.get_transform_pipeline)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_upgc_encode_shape(self, n):
        """Test that upgc_encode produces correct shape."""
        cga = CGA(n)
        batch_size = 8
        x = torch.randn(batch_size, n)
        point = cga.upgc_encode(x)
        assert point.shape == (batch_size, n + 2)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_upgc_decode_shape(self, n):
        """Test that upgc_decode produces correct shape."""
        cga = CGA(n)
        batch_size = 8
        point = torch.randn(batch_size, n + 2)
        x = cga.upgc_decode(point)
        assert x.shape == (batch_size, n)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sandwich_product_shape(self, n):
        """Test that sandwich_product_sparse produces correct shape."""
        cga = CGA(n)
        batch_size = 8
        ev = torch.randn(batch_size, cga.even_versor_count)
        point = torch.randn(batch_size, cga.point_count)
        result = cga.sandwich_product_sparse(ev, point)
        assert result.shape == (batch_size, cga.point_count)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_get_care_layer_returns_module(self, n):
        """Test that get_care_layer returns nn.Module."""
        cga = CGA(n)
        layer = cga.get_care_layer()
        assert isinstance(layer, torch.nn.Module)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_get_encoder_returns_module(self, n):
        """Test that get_encoder returns nn.Module."""
        cga = CGA(n)
        encoder = cga.get_encoder()
        assert isinstance(encoder, torch.nn.Module)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_get_decoder_returns_module(self, n):
        """Test that get_decoder returns nn.Module."""
        cga = CGA(n)
        decoder = cga.get_decoder()
        assert isinstance(decoder, torch.nn.Module)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_get_transform_pipeline_returns_module(self, n):
        """Test that get_transform_pipeline returns nn.Module."""
        cga = CGA(n)
        pipeline = cga.get_transform_pipeline()
        assert isinstance(pipeline, torch.nn.Module)


class TestCGAConsistency:
    """Test consistency across different access methods."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_cga_vs_direct_module_upgc_encode(self, n):
        """Test that CGA(n).upgc_encode matches direct module access."""
        cga = CGA(n)

        # Direct module access
        if n == 1:
            from fast_clifford.algebras import cga1d as direct
        elif n == 2:
            from fast_clifford.algebras import cga2d as direct
        elif n == 3:
            from fast_clifford.algebras import cga3d as direct

        x = torch.randn(8, n)
        unified_result = cga.upgc_encode(x)
        direct_result = direct.upgc_encode(x)

        assert torch.allclose(unified_result, direct_result, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_cga_vs_direct_module_sandwich(self, n):
        """Test that CGA(n).sandwich_product_sparse matches direct module access."""
        cga = CGA(n)

        # Direct module access
        if n == 1:
            from fast_clifford.algebras import cga1d as direct
        elif n == 2:
            from fast_clifford.algebras import cga2d as direct
        elif n == 3:
            from fast_clifford.algebras import cga3d as direct

        ev = torch.randn(8, cga.even_versor_count)
        point = torch.randn(8, cga.point_count)

        unified_result = cga.sandwich_product_sparse(ev, point)
        direct_result = direct.sandwich_product_sparse(ev, point)

        assert torch.allclose(unified_result, direct_result, atol=1e-6)
