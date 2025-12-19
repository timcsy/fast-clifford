"""
Performance benchmark comparing fast-clifford with clifford library.

Run with: pytest tests/benchmark/test_benchmark.py -v -s
"""

import pytest
import time
import torch
import numpy as np

from fast_clifford import Cl, VGA, CGA

# Try to import clifford library for comparison
try:
    import clifford as cf
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False


def benchmark(func, warmup=5, iterations=100):
    """Run benchmark and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


class TestGeometricProductBenchmark:
    """Benchmark geometric product performance."""

    @pytest.mark.skipif(not HAS_CLIFFORD, reason="clifford library not installed")
    def test_vga3_geometric_product(self):
        """Compare VGA(3) geometric product: fast-clifford vs clifford."""
        print("\n" + "="*60)
        print("VGA(3) Geometric Product Benchmark")
        print("="*60)

        batch_size = 1000
        iterations = 100

        # fast-clifford setup
        vga = VGA(3)
        fc_a = torch.randn(batch_size, vga.count_blade)
        fc_b = torch.randn(batch_size, vga.count_blade)

        def fc_gp():
            return vga.geometric_product(fc_a, fc_b)

        fc_mean, fc_std = benchmark(fc_gp, iterations=iterations)
        print(f"fast-clifford: {fc_mean:.3f} +/- {fc_std:.3f} ms ({batch_size} ops)")

        # clifford setup
        layout, blades = cf.Cl(3)
        cf_a = [layout.MultiVector(np.random.randn(8)) for _ in range(batch_size)]
        cf_b = [layout.MultiVector(np.random.randn(8)) for _ in range(batch_size)]

        def cf_gp():
            return [a * b for a, b in zip(cf_a, cf_b)]

        cf_mean, cf_std = benchmark(cf_gp, iterations=iterations)
        print(f"clifford:      {cf_mean:.3f} +/- {cf_std:.3f} ms ({batch_size} ops)")

        speedup = cf_mean / fc_mean
        print(f"Speedup:       {speedup:.1f}x")
        print("="*60)

        # Assert we're faster (should be significantly faster due to batching)
        assert speedup > 1.0, f"Expected speedup > 1x, got {speedup:.2f}x"

    @pytest.mark.skipif(not HAS_CLIFFORD, reason="clifford library not installed")
    def test_cga3_geometric_product(self):
        """Compare CGA(3) = Cl(4,1) geometric product: fast-clifford vs clifford."""
        print("\n" + "="*60)
        print("CGA(3) = Cl(4,1) Geometric Product Benchmark")
        print("="*60)

        batch_size = 1000
        iterations = 100

        # fast-clifford setup
        cga = CGA(3)  # Cl(4, 1)
        fc_a = torch.randn(batch_size, cga.count_blade)
        fc_b = torch.randn(batch_size, cga.count_blade)

        def fc_gp():
            return cga.geometric_product(fc_a, fc_b)

        fc_mean, fc_std = benchmark(fc_gp, iterations=iterations)
        print(f"fast-clifford: {fc_mean:.3f} +/- {fc_std:.3f} ms ({batch_size} ops)")

        # clifford setup
        layout, blades = cf.Cl(4, 1)
        cf_a = [layout.MultiVector(np.random.randn(32)) for _ in range(batch_size)]
        cf_b = [layout.MultiVector(np.random.randn(32)) for _ in range(batch_size)]

        def cf_gp():
            return [a * b for a, b in zip(cf_a, cf_b)]

        cf_mean, cf_std = benchmark(cf_gp, iterations=iterations)
        print(f"clifford:      {cf_mean:.3f} +/- {cf_std:.3f} ms ({batch_size} ops)")

        speedup = cf_mean / fc_mean
        print(f"Speedup:       {speedup:.1f}x")
        print("="*60)

        assert speedup > 1.0, f"Expected speedup > 1x, got {speedup:.2f}x"


class TestRotorBenchmark:
    """Benchmark rotor operations."""

    def test_compose_rotor_vga3(self):
        """Benchmark rotor composition in VGA(3)."""
        print("\n" + "="*60)
        print("VGA(3) Rotor Composition Benchmark")
        print("="*60)

        batch_size = 1000
        iterations = 100

        vga = VGA(3)
        r1 = torch.randn(batch_size, vga.count_rotor)
        r2 = torch.randn(batch_size, vga.count_rotor)

        def compose():
            return vga.compose_rotor(r1, r2)

        mean, std = benchmark(compose, iterations=iterations)
        ops_per_sec = batch_size * iterations / (mean * iterations / 1000)
        print(f"Time:          {mean:.3f} +/- {std:.3f} ms ({batch_size} ops)")
        print(f"Throughput:    {ops_per_sec/1e6:.2f}M ops/sec")
        print("="*60)

    def test_sandwich_rotor_cga3(self):
        """Benchmark rotor sandwich product in CGA(3)."""
        print("\n" + "="*60)
        print("CGA(3) Sandwich Rotor Benchmark")
        print("="*60)

        batch_size = 1000
        iterations = 100

        cga = CGA(3)
        rotor = torch.randn(batch_size, cga.count_rotor)
        point = torch.randn(batch_size, cga.count_blade)

        def sandwich():
            return cga.sandwich_rotor(rotor, point)

        mean, std = benchmark(sandwich, iterations=iterations)
        ops_per_sec = batch_size * iterations / (mean * iterations / 1000)
        print(f"Time:          {mean:.3f} +/- {std:.3f} ms ({batch_size} ops)")
        print(f"Throughput:    {ops_per_sec/1e6:.2f}M ops/sec")
        print("="*60)


class TestEncodeDecode:
    """Benchmark encode/decode operations."""

    def test_cga_encode_decode(self):
        """Benchmark CGA encode/decode."""
        print("\n" + "="*60)
        print("CGA(3) Encode/Decode Benchmark")
        print("="*60)

        batch_size = 10000
        iterations = 100

        cga = CGA(3)
        points = torch.randn(batch_size, 3)

        def encode():
            return cga.encode(points)

        def decode():
            encoded = cga.encode(points)
            return cga.decode(encoded)

        enc_mean, enc_std = benchmark(encode, iterations=iterations)
        dec_mean, dec_std = benchmark(decode, iterations=iterations)

        print(f"Encode:        {enc_mean:.3f} +/- {enc_std:.3f} ms ({batch_size} points)")
        print(f"Round-trip:    {dec_mean:.3f} +/- {dec_std:.3f} ms ({batch_size} points)")
        print(f"Encode rate:   {batch_size / (enc_mean / 1000) / 1e6:.2f}M points/sec")
        print("="*60)


class TestHighDimensional:
    """Benchmark high-dimensional algebras."""

    @pytest.mark.parametrize("p,q", [(5, 0), (6, 0), (4, 1), (5, 1)])
    def test_geometric_product_scaling(self, p, q):
        """Benchmark how geometric product scales with dimension."""
        print(f"\nCl({p}, {q}) Geometric Product:")

        algebra = Cl(p, q)
        batch_size = 100
        iterations = 50

        a = torch.randn(batch_size, algebra.count_blade)
        b = torch.randn(batch_size, algebra.count_blade)

        def gp():
            return algebra.geometric_product(a, b)

        mean, std = benchmark(gp, warmup=3, iterations=iterations)
        ops_per_sec = batch_size / (mean / 1000)
        print(f"  Blades: {algebra.count_blade}, Time: {mean:.3f} +/- {std:.3f} ms, Rate: {ops_per_sec:.0f} ops/sec")


if __name__ == "__main__":
    # Run quick benchmark
    print("Fast-Clifford Performance Benchmark")
    print("="*60)

    test = TestGeometricProductBenchmark()
    if HAS_CLIFFORD:
        test.test_vga3_geometric_product()
        test.test_cga3_geometric_product()
    else:
        print("clifford library not installed, skipping comparison tests")

    TestRotorBenchmark().test_compose_rotor_vga3()
    TestRotorBenchmark().test_sandwich_rotor_cga3()
    TestEncodeDecode().test_cga_encode_decode()
