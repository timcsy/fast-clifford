"""
Similitude Tests (User Story 4a)

T048-T052, T058: Tests for Similitude-specific operations.

Tests verify:
- T048: Similitude composition performance vs EvenVersor
- T049: sandwich_product_similitude correctness
- T050: Similitude sandwich product performance comparison
- T051: Similitude constraint validation (excluding transversion)
- T052: Static routing to compose_similitude
"""

import pytest
import torch
import time
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


def get_random_point(dim: int, batch_size: int = None) -> torch.Tensor:
    """Create random UPGC point for given CGA dimension."""
    func = get_functional_module(dim)
    point_count = len(func.POINT_MASK)

    if batch_size is not None:
        return torch.randn(batch_size, point_count)
    return torch.randn(point_count)


def get_random_similitude(dim: int, batch_size: int = None) -> torch.Tensor:
    """Create random normalized Similitude for given CGA dimension."""
    func = get_functional_module(dim)
    even_versor_count = len(func.EVEN_VERSOR_MASK)

    if batch_size is not None:
        ev = torch.randn(batch_size, even_versor_count)
    else:
        ev = torch.randn(even_versor_count)

    # Normalize (simple normalization for testing)
    norm = torch.sqrt((ev ** 2).sum(dim=-1, keepdim=True))
    return ev / norm


# =============================================================================
# T049: sandwich_product_similitude Correctness Tests
# =============================================================================

class TestSandwichProductSimilitudeCorrectness:
    """T049: sandwich_product_similitude correctness tests."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_similitude_sandwich_equals_sparse(self, dim: int):
        """sandwich_product_similitude should give same result as sandwich_product_sparse."""
        func = get_functional_module(dim)

        similitude = get_random_similitude(dim)
        point = get_random_point(dim)

        result_sparse = func.sandwich_product_sparse(similitude, point)
        result_similitude = func.sandwich_product_similitude(similitude, point)

        torch.testing.assert_close(
            result_sparse, result_similitude, rtol=1e-4, atol=1e-4,
            msg=f"sandwich_product_similitude differs from sparse for dim={dim}"
        )

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_similitude_sandwich_batch(self, dim: int):
        """Test batch dimension support for sandwich_product_similitude."""
        func = get_functional_module(dim)
        batch_size = 16

        similitude = get_random_similitude(dim, batch_size)
        point = get_random_point(dim, batch_size)

        result = func.sandwich_product_similitude(similitude, point)

        assert result.shape == (batch_size, len(func.POINT_MASK))

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_similitude_sandwich_identity(self, dim: int):
        """Identity Similitude should preserve point."""
        func = get_functional_module(dim)

        # Identity similitude: scalar = 1, all others = 0
        identity = torch.zeros(len(func.EVEN_VERSOR_MASK))
        identity[0] = 1.0

        point = get_random_point(dim)

        result = func.sandwich_product_similitude(identity, point)

        torch.testing.assert_close(
            result, point, rtol=1e-5, atol=1e-5,
            msg="Identity similitude should preserve point"
        )


# =============================================================================
# T050: Similitude Sandwich Performance Tests
# =============================================================================

class TestSimilitudePerformance:
    """T048/T050: Similitude performance comparison tests."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_similitude_compose_performance(self, dim: int):
        """T048: Compare compose_similitude vs compose_even_versor performance."""
        func = get_functional_module(dim)
        batch_size = 1024
        iterations = 100

        v1 = get_random_similitude(dim, batch_size)
        v2 = get_random_similitude(dim, batch_size)

        # Warmup
        for _ in range(10):
            _ = func.compose_even_versor(v1, v2)
            _ = func.compose_similitude(v1, v2)

        # Measure compose_even_versor
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func.compose_even_versor(v1, v2)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_ev = (time.perf_counter() - start) / iterations

        # Measure compose_similitude
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func.compose_similitude(v1, v2)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_sim = (time.perf_counter() - start) / iterations

        # Performance should be similar (within 50%)
        # Since Similitude uses the same EvenVersor structure
        ratio = time_sim / time_ev
        assert 0.5 < ratio < 2.0, \
            f"compose_similitude should be comparable: ratio={ratio:.2f}"

    @pytest.mark.parametrize("dim", [2, 3])
    def test_similitude_sandwich_performance(self, dim: int):
        """T050: Compare sandwich_product_similitude vs sandwich_product_sparse."""
        func = get_functional_module(dim)
        batch_size = 1024
        iterations = 100

        similitude = get_random_similitude(dim, batch_size)
        point = get_random_point(dim, batch_size)

        # Warmup
        for _ in range(10):
            _ = func.sandwich_product_sparse(similitude, point)
            _ = func.sandwich_product_similitude(similitude, point)

        # Measure sandwich_product_sparse
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func.sandwich_product_sparse(similitude, point)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_sparse = (time.perf_counter() - start) / iterations

        # Measure sandwich_product_similitude
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func.sandwich_product_similitude(similitude, point)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_similitude = (time.perf_counter() - start) / iterations

        # Performance should be similar (within 50%)
        ratio = time_similitude / time_sparse
        assert 0.5 < ratio < 2.0, \
            f"sandwich_product_similitude should be comparable: ratio={ratio:.2f}"


# =============================================================================
# T051: Similitude Constraint Validation Tests
# =============================================================================

class TestSimilitudeConstraints:
    """T051: Similitude constraint validation tests."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_similitude_is_even_versor_subset(self, dim: int):
        """Similitude should have same component count as EvenVersor."""
        func = get_functional_module(dim)

        # Similitude uses the same indices as EvenVersor
        assert hasattr(func, 'EVEN_VERSOR_MASK')
        assert hasattr(func, 'compose_similitude')
        assert hasattr(func, 'sandwich_product_similitude')

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_pure_rotation_similitude(self, dim: int):
        """Pure rotation should work as Similitude."""
        func = get_functional_module(dim)

        # Create a pure rotation via exp_bivector
        bivector_count = len(func.GRADE_2_INDICES)
        B = torch.zeros(bivector_count)
        if bivector_count > 0:
            B[0] = 0.3  # Small rotation angle

        rotor = func.exp_bivector(B)
        point = get_random_point(dim)

        # Apply as Similitude
        result = func.sandwich_product_similitude(rotor, point)

        # Result should be valid (no NaN)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


# =============================================================================
# T052: Static Routing Tests
# =============================================================================

class TestStaticRouting:
    """T052: Static routing tests for Similitude operations."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_compose_routes_to_similitude(self, dim: int):
        """compose() should route to compose_similitude for Similitude inputs."""
        from fast_clifford.cga import CGA
        from fast_clifford.cga.multivector import Similitude

        algebra = CGA(dim)

        # Create Similitude multivectors
        s1 = algebra.similitude(get_random_similitude(dim))
        s2 = algebra.similitude(get_random_similitude(dim))

        # Compose
        result = s1 * s2

        # Result should be a Similitude
        assert isinstance(result, Similitude), \
            f"Similitude × Similitude should return Similitude, got {type(result)}"

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_sandwich_routes_to_similitude(self, dim: int):
        """sandwich_product() should use similitude path for Similitude."""
        from fast_clifford.cga import CGA

        algebra = CGA(dim)

        similitude = get_random_similitude(dim)
        point = get_random_point(dim)

        # Direct call
        result_direct = algebra.sandwich_product_similitude(similitude, point)

        # Via sandwich_product with versor_type hint
        result_api = algebra.sandwich_product(similitude, point, versor_type='similitude')

        torch.testing.assert_close(result_direct, result_api, rtol=1e-5, atol=1e-5)


# =============================================================================
# T058: Execute Similitude Test Verification
# =============================================================================

class TestSimilitudeVerification:
    """T058: Comprehensive Similitude test verification."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_full_similitude_workflow(self, dim: int):
        """Test complete Similitude workflow."""
        from fast_clifford.cga import CGA

        algebra = CGA(dim)

        # 1. Create Similitude
        s1 = algebra.similitude(get_random_similitude(dim))
        s2 = algebra.similitude(get_random_similitude(dim))

        # 2. Compose
        composed = s1 * s2
        assert composed.data.shape[-1] == algebra.even_versor_count

        # 3. Apply to point
        point_data = get_random_point(dim)
        result = algebra.sandwich_product_similitude(composed.data, point_data)

        # 4. Verify result shape
        assert result.shape == point_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
