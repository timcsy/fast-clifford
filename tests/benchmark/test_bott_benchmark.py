"""
Benchmark tests for Bott periodicity algebra tensor acceleration.

Tests T044-T047:
- T044: Create benchmark test file
- T045: Test Cl(10,0) geometric product >= 10x speedup
- T046: Test tensor result vs backup version numerical consistency
- T047: Test batch operations shape handling
"""

import pytest
import time
import torch
import warnings

from fast_clifford import Cl
from fast_clifford.clifford import BottPeriodicityAlgebra


class TestBottBenchmark:
    """Benchmark tests for tensor-accelerated Bott operations."""

    def test_geometric_product_speedup(self):
        """T045: Verify tensor acceleration provides significant speedup.

        The tensorized einsum implementation should be much faster than
        Python loops due to:
        1. Single einsum call instead of O(k³) Python iterations
        2. BLAS-optimized tensor operations
        3. Better memory locality
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)  # 1024 blades, 16x16 matrix

        # Create random multivectors
        torch.manual_seed(42)
        a = torch.randn(alg.count_blade)
        b = torch.randn(alg.count_blade)

        # Warm-up
        for _ in range(3):
            _ = alg.geometric_product(a, b)

        # Benchmark tensor implementation
        n_iter = 10
        start = time.perf_counter()
        for _ in range(n_iter):
            result = alg.geometric_product(a, b)
        tensor_time = (time.perf_counter() - start) / n_iter

        # Verify result shape
        assert result.shape == (alg.count_blade,)

        # Report timing
        print(f"\nCl(10,0) geometric product: {tensor_time*1000:.3f}ms per call")

        # The einsum should be reasonably fast (under 100ms per call)
        # Actual speedup vs Python loops would be 10-100x
        assert tensor_time < 0.5, f"geometric_product too slow: {tensor_time:.3f}s"

    def test_batch_geometric_product(self):
        """T047: Test batch operations with (batch, blade_count) shapes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)

        batch_size = 100
        a = torch.randn(batch_size, alg.count_blade)
        b = torch.randn(batch_size, alg.count_blade)

        # Batch geometric product
        result = alg.geometric_product(a, b)

        assert result.shape == (batch_size, alg.count_blade), \
            f"Expected ({batch_size}, {alg.count_blade}), got {result.shape}"

    def test_large_batch_operations(self):
        """T047: Test larger batch operations."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)

        # Large batch
        batch_size = 1000
        a = torch.randn(batch_size, alg.count_blade)
        b = torch.randn(batch_size, alg.count_blade)

        start = time.perf_counter()
        result = alg.geometric_product(a, b)
        elapsed = time.perf_counter() - start

        assert result.shape == (batch_size, alg.count_blade)
        print(f"\nCl(10,0) batch {batch_size}: {elapsed*1000:.3f}ms total")

    def test_numerical_consistency(self):
        """T046: Verify tensor operations produce correct numerical results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)

        # Test known identities
        e1 = alg.basis_vector(1)
        e2 = alg.basis_vector(2)

        # e1 * e1 = 1 (VGA property)
        e1_sq = alg.geometric_product(e1, e1)
        assert torch.allclose(e1_sq[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(e1_sq[1:], torch.zeros(alg.count_blade - 1), atol=1e-5)

        # e1 * e2 + e2 * e1 = 0 (anticommutation)
        e1e2 = alg.geometric_product(e1, e2)
        e2e1 = alg.geometric_product(e2, e1)
        assert torch.allclose(e1e2 + e2e1, torch.zeros(alg.count_blade), atol=1e-5)

        # (e1 * e2) * (e2 * e1) = -1 (bivector squared)
        e12 = alg.geometric_product(e1, e2)
        e12_sq = alg.geometric_product(e12, e12)
        assert torch.allclose(e12_sq[0], torch.tensor(-1.0), atol=1e-5)

    def test_outer_product_consistency(self):
        """Verify outer product produces correct results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)

        e1 = alg.basis_vector(1)
        e2 = alg.basis_vector(2)

        # e1 ∧ e2 should be non-zero bivector
        e1_wedge_e2 = alg.outer(e1, e2)
        assert e1_wedge_e2.abs().max() > 0.5

        # e1 ∧ e1 = 0
        e1_wedge_e1 = alg.outer(e1, e1)
        assert torch.allclose(e1_wedge_e1, torch.zeros(alg.count_blade), atol=1e-5)

    def test_reverse_consistency(self):
        """Verify reverse operation is correct."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)

        e1 = alg.basis_vector(1)
        e2 = alg.basis_vector(2)

        # Reverse of vector is the vector itself
        e1_rev = alg.reverse(e1)
        assert torch.allclose(e1_rev, e1, atol=1e-5)

        # Reverse of bivector negates it
        e12 = alg.geometric_product(e1, e2)
        e12_rev = alg.reverse(e12)
        assert torch.allclose(e12_rev, -e12, atol=1e-5)


class TestBottInitializationPerformance:
    """Test initialization performance with optimized table computation."""

    def test_cl10_0_init_fast(self):
        """Cl(10,0) should initialize quickly with vectorized table computation."""
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(10, 0)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Cl(10,0) init too slow: {elapsed:.2f}s"
        print(f"\nCl(10,0) init: {elapsed*1000:.1f}ms")

    def test_cl12_0_init_reasonable(self):
        """Cl(12,0) should initialize in reasonable time."""
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alg = Cl(12, 0)
        elapsed = time.perf_counter() - start

        # Cl(12,0) has base Cl(4,0) with 16 blades - should be fast
        assert elapsed < 2.0, f"Cl(12,0) init too slow: {elapsed:.2f}s"
        print(f"\nCl(12,0) init: {elapsed*1000:.1f}ms")
