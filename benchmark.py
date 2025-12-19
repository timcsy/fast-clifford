#!/usr/bin/env python3
"""
fast-clifford Comprehensive Benchmark

Run with: uv run python benchmark.py
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass

# Try to import clifford for comparison
try:
    import clifford as cf
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False
    print("Note: clifford library not installed, skipping comparison benchmarks")
    print("Install with: uv pip install clifford\n")

from fast_clifford import Cl, VGA, CGA, PGA


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    ops_per_sec: float
    batch_size: int = 1
    extra_info: str = ""


def timeit(func: Callable, warmup: int = 5, iterations: int = 50) -> float:
    """Time a function, return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    # Sync if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        func()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    return (end - start) / iterations * 1000  # Convert to ms


def format_time(ms: float) -> str:
    """Format time nicely."""
    if ms < 0.001:
        return f"{ms * 1000:.3f}μs"
    elif ms < 1:
        return f"{ms:.3f}ms"
    else:
        return f"{ms:.2f}ms"


def format_ops(ops: float) -> str:
    """Format operations per second."""
    if ops >= 1e9:
        return f"{ops/1e9:.2f}B"
    elif ops >= 1e6:
        return f"{ops/1e6:.2f}M"
    elif ops >= 1e3:
        return f"{ops/1e3:.2f}K"
    else:
        return f"{ops:.1f}"


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_algebra_creation() -> List[BenchmarkResult]:
    """Benchmark algebra creation time."""
    results = []

    print("\n" + "=" * 70)
    print("ALGEBRA CREATION BENCHMARK")
    print("=" * 70)

    algebras = [
        ("VGA(3)", lambda: VGA(3)),
        ("CGA(3)", lambda: CGA(3)),
        ("PGA(3)", lambda: PGA(3)),
        ("Cl(2,2)", lambda: Cl(2, 2)),
        ("Cl(4,1)", lambda: Cl(4, 1)),
        ("Cl(7,0)", lambda: Cl(7, 0)),
        ("Cl(10,0) [Bott]", lambda: Cl(10, 0)),
        ("Cl(12,0) [Bott]", lambda: Cl(12, 0)),
    ]

    print(f"\n{'Algebra':<20} {'Time':>12} {'Blades':>10}")
    print("-" * 45)

    for name, create_fn in algebras:
        time_ms = timeit(create_fn, warmup=3, iterations=20)
        alg = create_fn()
        results.append(BenchmarkResult(
            name=f"create_{name}",
            time_ms=time_ms,
            ops_per_sec=1000 / time_ms,
            extra_info=f"{alg.count_blade} blades"
        ))
        print(f"{name:<20} {format_time(time_ms):>12} {alg.count_blade:>10}")

    return results


def benchmark_basic_operations() -> List[BenchmarkResult]:
    """Benchmark basic algebraic operations."""
    results = []

    print("\n" + "=" * 70)
    print("BASIC OPERATIONS BENCHMARK (single element)")
    print("=" * 70)

    vga = VGA(3)
    cga = CGA(3)

    # VGA operations
    a_vga = torch.randn(vga.count_blade)
    b_vga = torch.randn(vga.count_blade)
    r_vga = torch.randn(vga.count_rotor)

    # CGA operations
    a_cga = torch.randn(cga.count_blade)
    b_cga = torch.randn(cga.count_blade)
    r_cga = torch.randn(cga.count_rotor)

    operations = [
        ("VGA(3) geometric_product", lambda: vga.geometric_product(a_vga, b_vga)),
        ("VGA(3) outer", lambda: vga.outer(a_vga, b_vga)),
        ("VGA(3) inner", lambda: vga.inner(a_vga, b_vga)),
        ("VGA(3) reverse", lambda: vga.reverse(a_vga)),
        ("VGA(3) dual", lambda: vga.dual(a_vga)),
        ("VGA(3) compose_rotor", lambda: vga.compose_rotor(r_vga, r_vga)),
        ("VGA(3) sandwich_rotor", lambda: vga.sandwich_rotor(r_vga, a_vga)),
        ("CGA(3) geometric_product", lambda: cga.geometric_product(a_cga, b_cga)),
        ("CGA(3) outer", lambda: cga.outer(a_cga, b_cga)),
        ("CGA(3) inner", lambda: cga.inner(a_cga, b_cga)),
        ("CGA(3) reverse", lambda: cga.reverse(a_cga)),
        ("CGA(3) sandwich_rotor", lambda: cga.sandwich_rotor(r_cga, a_cga)),
    ]

    print(f"\n{'Operation':<30} {'Time':>12} {'Ops/sec':>12}")
    print("-" * 56)

    for name, op in operations:
        time_ms = timeit(op, warmup=20, iterations=200)
        ops = 1000 / time_ms
        results.append(BenchmarkResult(name=name, time_ms=time_ms, ops_per_sec=ops))
        print(f"{name:<30} {format_time(time_ms):>12} {format_ops(ops):>12}")

    return results


def benchmark_batch_operations() -> List[BenchmarkResult]:
    """Benchmark batch operations."""
    results = []

    print("\n" + "=" * 70)
    print("BATCH OPERATIONS BENCHMARK")
    print("=" * 70)

    batch_sizes = [1, 100, 1000]

    vga = VGA(3)
    cga = CGA(3)

    print(f"\n{'Operation':<25} ", end="")
    for bs in batch_sizes:
        print(f"{'n=' + str(bs):>12}", end="")
    print()
    print("-" * (25 + 12 * len(batch_sizes)))

    # VGA geometric product
    print(f"{'VGA(3) geometric':<25} ", end="")
    for bs in batch_sizes:
        a = torch.randn(bs, vga.count_blade)
        b = torch.randn(bs, vga.count_blade)
        time_ms = timeit(lambda: vga.geometric_product(a, b), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"VGA_geometric_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # CGA geometric product
    print(f"{'CGA(3) geometric':<25} ", end="")
    for bs in batch_sizes:
        a = torch.randn(bs, cga.count_blade)
        b = torch.randn(bs, cga.count_blade)
        time_ms = timeit(lambda: cga.geometric_product(a, b), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"CGA_geometric_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # VGA sandwich
    print(f"{'VGA(3) sandwich':<25} ", end="")
    for bs in batch_sizes:
        r = torch.randn(bs, vga.count_rotor)
        x = torch.randn(bs, vga.count_blade)
        time_ms = timeit(lambda: vga.sandwich_rotor(r, x), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"VGA_sandwich_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # CGA sandwich
    print(f"{'CGA(3) sandwich':<25} ", end="")
    for bs in batch_sizes:
        r = torch.randn(bs, cga.count_rotor)
        x = torch.randn(bs, cga.count_blade)
        time_ms = timeit(lambda: cga.sandwich_rotor(r, x), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"CGA_sandwich_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    return results


def benchmark_bott_periodicity() -> List[BenchmarkResult]:
    """Benchmark Bott periodicity algebras."""
    results = []

    print("\n" + "=" * 70)
    print("BOTT PERIODICITY BENCHMARK (High-dimensional algebras)")
    print("=" * 70)

    algebras = [
        ("Cl(8,0)", 8, 0),
        ("Cl(10,0)", 10, 0),
        ("Cl(12,0)", 12, 0),
    ]

    print(f"\n{'Algebra':<12} {'Blades':>8} {'Base':>10} {'Periods':>8} {'Init':>10} {'GeomProd':>12}")
    print("-" * 65)

    for name, p, q in algebras:
        # Measure init time
        init_time = timeit(lambda: Cl(p, q), warmup=2, iterations=5)

        alg = Cl(p, q)
        base_info = f"Cl({alg.base_p},{alg.base_q})" if hasattr(alg, 'base_p') else "hardcoded"
        periods = alg.periods if hasattr(alg, 'periods') else 0

        # Measure geometric product
        a = torch.randn(alg.count_blade)
        b = torch.randn(alg.count_blade)
        gp_time = timeit(lambda: alg.geometric_product(a, b), warmup=5, iterations=30)

        results.append(BenchmarkResult(
            name=f"Bott_{name}_init",
            time_ms=init_time,
            ops_per_sec=1000 / init_time,
            extra_info=f"{alg.count_blade} blades"
        ))
        results.append(BenchmarkResult(
            name=f"Bott_{name}_geom",
            time_ms=gp_time,
            ops_per_sec=1000 / gp_time
        ))

        print(f"{name:<12} {alg.count_blade:>8} {base_info:>10} {periods:>8} {format_time(init_time):>10} {format_time(gp_time):>12}")

    # Batch benchmark for Cl(10,0)
    print("\nCl(10,0) batch geometric product:")
    cl10 = Cl(10, 0)
    batch_sizes = [1, 100, 1000]

    print(f"  ", end="")
    for bs in batch_sizes:
        print(f"{'n=' + str(bs):>12}", end="")
    print()
    print(f"  ", end="")

    for bs in batch_sizes:
        a = torch.randn(bs, cl10.count_blade)
        b = torch.randn(bs, cl10.count_blade)
        time_ms = timeit(lambda: cl10.geometric_product(a, b), warmup=3, iterations=20)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"Bott_Cl10_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    return results


def benchmark_encode_decode() -> List[BenchmarkResult]:
    """Benchmark encode/decode operations."""
    results = []

    print("\n" + "=" * 70)
    print("ENCODE/DECODE BENCHMARK")
    print("=" * 70)

    batch_sizes = [1, 100, 1000]

    vga3 = VGA(3)
    cga3 = CGA(3)

    print(f"\n{'Operation':<25} ", end="")
    for bs in batch_sizes:
        print(f"{'n=' + str(bs):>12}", end="")
    print()
    print("-" * (25 + 12 * len(batch_sizes)))

    # VGA encode
    print(f"{'VGA(3) encode':<25} ", end="")
    for bs in batch_sizes:
        x = torch.randn(bs, 3)
        time_ms = timeit(lambda: vga3.encode(x), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"VGA_encode_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # CGA encode
    print(f"{'CGA(3) encode':<25} ", end="")
    for bs in batch_sizes:
        x = torch.randn(bs, 3)
        time_ms = timeit(lambda: cga3.encode(x), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"CGA_encode_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # CGA decode
    print(f"{'CGA(3) decode':<25} ", end="")
    for bs in batch_sizes:
        x = torch.randn(bs, 3)
        encoded = cga3.encode(x)
        time_ms = timeit(lambda: cga3.decode(encoded), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"CGA_decode_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    return results


def benchmark_exp_log() -> List[BenchmarkResult]:
    """Benchmark exponential and logarithm operations."""
    results = []

    print("\n" + "=" * 70)
    print("EXPONENTIAL/LOGARITHM BENCHMARK")
    print("=" * 70)

    vga3 = VGA(3)
    cga3 = CGA(3)

    batch_sizes = [1, 100, 1000]

    print(f"\n{'Operation':<25} ", end="")
    for bs in batch_sizes:
        print(f"{'n=' + str(bs):>12}", end="")
    print()
    print("-" * (25 + 12 * len(batch_sizes)))

    # VGA exp_bivector
    print(f"{'VGA(3) exp_bivector':<25} ", end="")
    for bs in batch_sizes:
        B = torch.randn(bs, vga3.count_bivector) * 0.5
        time_ms = timeit(lambda: vga3.exp_bivector(B), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"VGA_exp_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # VGA log_rotor
    print(f"{'VGA(3) log_rotor':<25} ", end="")
    for bs in batch_sizes:
        B = torch.randn(bs, vga3.count_bivector) * 0.5
        R = vga3.exp_bivector(B)
        time_ms = timeit(lambda: vga3.log_rotor(R), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"VGA_log_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    # VGA slerp
    print(f"{'VGA(3) slerp_rotor':<25} ", end="")
    for bs in batch_sizes:
        B1 = torch.randn(bs, vga3.count_bivector) * 0.5
        B2 = torch.randn(bs, vga3.count_bivector) * 0.5
        R1 = vga3.exp_bivector(B1)
        R2 = vga3.exp_bivector(B2)
        t = 0.5
        time_ms = timeit(lambda: vga3.slerp_rotor(R1, R2, t), warmup=5, iterations=30)
        throughput = bs * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"VGA_slerp_batch_{bs}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=bs
        ))
        print(f"{format_ops(throughput) + '/s':>12}", end="")
    print()

    return results


def benchmark_vs_clifford() -> List[BenchmarkResult]:
    """Benchmark against clifford library."""
    results = []

    if not HAS_CLIFFORD:
        print("\n" + "=" * 70)
        print("COMPARISON WITH CLIFFORD LIBRARY (skipped - not installed)")
        print("=" * 70)
        return results

    print("\n" + "=" * 70)
    print("COMPARISON WITH CLIFFORD LIBRARY")
    print("=" * 70)

    batch_size = 500

    # Setup VGA(3)
    vga = VGA(3)
    layout_vga, blades_vga = cf.Cl(3)

    # Setup CGA(3)
    cga = CGA(3)
    layout_cga, blades_cga = cf.Cl(4, 1)

    print(f"\nBatch size: {batch_size}")
    print(f"\n{'Operation':<30} {'fast-clifford':>15} {'clifford':>15} {'Speedup':>10}")
    print("-" * 75)

    # VGA geometric product
    a_fc = torch.randn(batch_size, vga.count_blade)
    b_fc = torch.randn(batch_size, vga.count_blade)

    a_cf = [cf.MultiVector(layout_vga, np.random.randn(vga.count_blade)) for _ in range(batch_size)]
    b_cf = [cf.MultiVector(layout_vga, np.random.randn(vga.count_blade)) for _ in range(batch_size)]

    fc_time = timeit(lambda: vga.geometric_product(a_fc, b_fc), warmup=5, iterations=30)
    cf_time = timeit(lambda: [a * b for a, b in zip(a_cf, b_cf)], warmup=2, iterations=10)
    speedup = cf_time / fc_time

    results.append(BenchmarkResult(
        name="VGA3_geom_vs_clifford",
        time_ms=fc_time,
        ops_per_sec=batch_size * 1000 / fc_time,
        extra_info=f"speedup={speedup:.1f}x"
    ))
    print(f"{'VGA(3) geometric_product':<30} {format_time(fc_time):>15} {format_time(cf_time):>15} {speedup:>9.1f}x")

    # CGA geometric product
    a_fc = torch.randn(batch_size, cga.count_blade)
    b_fc = torch.randn(batch_size, cga.count_blade)

    a_cf = [cf.MultiVector(layout_cga, np.random.randn(cga.count_blade)) for _ in range(batch_size)]
    b_cf = [cf.MultiVector(layout_cga, np.random.randn(cga.count_blade)) for _ in range(batch_size)]

    fc_time = timeit(lambda: cga.geometric_product(a_fc, b_fc), warmup=5, iterations=30)
    cf_time = timeit(lambda: [a * b for a, b in zip(a_cf, b_cf)], warmup=2, iterations=10)
    speedup = cf_time / fc_time

    results.append(BenchmarkResult(
        name="CGA3_geom_vs_clifford",
        time_ms=fc_time,
        ops_per_sec=batch_size * 1000 / fc_time,
        extra_info=f"speedup={speedup:.1f}x"
    ))
    print(f"{'CGA(3) geometric_product':<30} {format_time(fc_time):>15} {format_time(cf_time):>15} {speedup:>9.1f}x")

    # VGA outer product
    a_fc = torch.randn(batch_size, vga.count_blade)
    b_fc = torch.randn(batch_size, vga.count_blade)

    fc_time = timeit(lambda: vga.outer(a_fc, b_fc), warmup=5, iterations=30)
    cf_time = timeit(lambda: [a ^ b for a, b in zip(a_cf[:batch_size], b_cf[:batch_size])], warmup=2, iterations=10)
    speedup = cf_time / fc_time

    results.append(BenchmarkResult(
        name="VGA3_outer_vs_clifford",
        time_ms=fc_time,
        ops_per_sec=batch_size * 1000 / fc_time,
        extra_info=f"speedup={speedup:.1f}x"
    ))
    print(f"{'VGA(3) outer':<30} {format_time(fc_time):>15} {format_time(cf_time):>15} {speedup:>9.1f}x")

    # VGA reverse
    a_fc = torch.randn(batch_size, vga.count_blade)
    a_cf_vga = [cf.MultiVector(layout_vga, np.random.randn(vga.count_blade)) for _ in range(batch_size)]

    fc_time = timeit(lambda: vga.reverse(a_fc), warmup=5, iterations=30)
    cf_time = timeit(lambda: [~a for a in a_cf_vga], warmup=2, iterations=10)
    speedup = cf_time / fc_time

    results.append(BenchmarkResult(
        name="VGA3_reverse_vs_clifford",
        time_ms=fc_time,
        ops_per_sec=batch_size * 1000 / fc_time,
        extra_info=f"speedup={speedup:.1f}x"
    ))
    print(f"{'VGA(3) reverse':<30} {format_time(fc_time):>15} {format_time(cf_time):>15} {speedup:>9.1f}x")

    return results


def benchmark_multivector_operators() -> List[BenchmarkResult]:
    """Benchmark Multivector class with operator overloading."""
    results = []

    print("\n" + "=" * 70)
    print("MULTIVECTOR OPERATOR OVERLOADING BENCHMARK")
    print("=" * 70)

    vga = VGA(3)

    # Create multivectors
    a = vga.multivector(torch.randn(vga.count_blade))
    b = vga.multivector(torch.randn(vga.count_blade))

    operations = [
        ("a * b (geometric)", lambda: a * b),
        ("a ^ b (outer)", lambda: a ^ b),
        ("a | b (inner)", lambda: a | b),
        ("a << b (left contract)", lambda: a << b),
        ("a >> b (right contract)", lambda: a >> b),
        ("~a (reverse)", lambda: ~a),
    ]

    print(f"\n{'Operation':<25} {'Time':>12} {'Ops/sec':>12}")
    print("-" * 51)

    for name, op in operations:
        time_ms = timeit(op, warmup=20, iterations=200)
        ops = 1000 / time_ms
        results.append(BenchmarkResult(name=f"Multivector_{name}", time_ms=time_ms, ops_per_sec=ops))
        print(f"{name:<25} {format_time(time_ms):>12} {format_ops(ops):>12}")

    return results


def benchmark_device_comparison() -> List[BenchmarkResult]:
    """Benchmark CPU vs GPU (if available)."""
    results = []

    devices = [("CPU", "cpu")]

    if torch.cuda.is_available():
        devices.append(("CUDA", "cuda"))
    if torch.backends.mps.is_available():
        devices.append(("MPS", "mps"))

    if len(devices) == 1:
        print("\n" + "=" * 70)
        print("DEVICE COMPARISON (only CPU available)")
        print("=" * 70)
        return results

    print("\n" + "=" * 70)
    print("DEVICE COMPARISON")
    print("=" * 70)

    batch_size = 5000
    cga = CGA(3)

    print(f"\nBatch size: {batch_size}")
    print(f"\n{'Operation':<25} ", end="")
    for name, _ in devices:
        print(f"{name:>15}", end="")
    print()
    print("-" * (25 + 15 * len(devices)))

    # geometric_product
    print(f"{'CGA(3) geometric':<25} ", end="")
    for dev_name, device in devices:
        a = torch.randn(batch_size, cga.count_blade, device=device)
        b = torch.randn(batch_size, cga.count_blade, device=device)

        def op():
            result = cga.geometric_product(a, b)
            if device != "cpu":
                torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
            return result

        time_ms = timeit(op, warmup=5, iterations=20)
        throughput = batch_size * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"CGA_geom_{dev_name}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=batch_size
        ))
        print(f"{format_ops(throughput) + '/s':>15}", end="")
    print()

    # sandwich_rotor
    print(f"{'CGA(3) sandwich':<25} ", end="")
    for dev_name, device in devices:
        r = torch.randn(batch_size, cga.count_rotor, device=device)
        x = torch.randn(batch_size, cga.count_blade, device=device)

        def op():
            result = cga.sandwich_rotor(r, x)
            if device != "cpu":
                torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
            return result

        time_ms = timeit(op, warmup=5, iterations=20)
        throughput = batch_size * 1000 / time_ms
        results.append(BenchmarkResult(
            name=f"CGA_sandwich_{dev_name}",
            time_ms=time_ms,
            ops_per_sec=throughput,
            batch_size=batch_size
        ))
        print(f"{format_ops(throughput) + '/s':>15}", end="")
    print()

    return results


def print_summary(all_results: List[BenchmarkResult]):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"""
Key Performance Metrics:
------------------------
- VGA(3) geometric product: {next((r for r in all_results if r.name == 'VGA(3) geometric_product'), None)}
- CGA(3) geometric product: {next((r for r in all_results if r.name == 'CGA(3) geometric_product'), None)}
- Bott Cl(10,0) initialization: {next((r for r in all_results if r.name == 'Bott_Cl(10,0)_init'), None)}

Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
PyTorch: {torch.__version__}
""")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("fast-clifford Comprehensive Benchmark")
    print("=" * 70)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available")
    print(f"clifford library: {'Available' if HAS_CLIFFORD else 'Not installed'}")

    all_results = []

    # Run all benchmarks
    all_results.extend(benchmark_algebra_creation())
    all_results.extend(benchmark_basic_operations())
    all_results.extend(benchmark_batch_operations())
    all_results.extend(benchmark_bott_periodicity())
    all_results.extend(benchmark_encode_decode())
    all_results.extend(benchmark_exp_log())
    all_results.extend(benchmark_multivector_operators())
    all_results.extend(benchmark_vs_clifford())
    all_results.extend(benchmark_device_comparison())

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nTotal benchmarks run: {len(all_results)}")

    return all_results


if __name__ == "__main__":
    main()
