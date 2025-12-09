"""
Numerical correctness tests for CGA4D operations.

Tests verify:
- Geometric product correctness (vs clifford library)
- Null basis properties
- Associativity of geometric product
- Reverse operation correctness
- Sparse sandwich product correctness
- 4D rotation and translation transformations
"""

import pytest
import torch
import numpy as np
from clifford import Cl, conformalize


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cga_reference():
    """Create reference CGA4D algebra using clifford library."""
    G4, _ = Cl(4)
    layout, blades, stuff = conformalize(G4)
    return layout, blades, stuff


@pytest.fixture(scope="module")
def functional_module():
    """Import the generated functional module."""
    from fast_clifford.algebras.cga4d import functional
    return functional


# =============================================================================
# Geometric Product Correctness
# =============================================================================

class TestGeometricProductCorrectness:
    """Test geometric product against clifford library reference."""

    def test_scalar_times_scalar(self, cga_reference, functional_module):
        """Test scalar × scalar = scalar."""
        layout, _, _ = cga_reference

        a_val = np.zeros(64)
        a_val[0] = 3.0
        b_val = np.zeros(64)
        b_val[0] = 4.0

        a_mv = layout.MultiVector(value=a_val.copy())
        b_mv = layout.MultiVector(value=b_val.copy())
        ref_result = (a_mv * b_mv).value

        a_tensor = torch.tensor(a_val, dtype=torch.float32).unsqueeze(0)
        b_tensor = torch.tensor(b_val, dtype=torch.float32).unsqueeze(0)
        our_result = functional_module.geometric_product_full(a_tensor, b_tensor)

        assert torch.allclose(
            our_result,
            torch.tensor(ref_result, dtype=torch.float32).unsqueeze(0),
            atol=1e-6
        )

    def test_vector_times_vector(self, cga_reference, functional_module):
        """Test vector × vector produces scalar + bivector."""
        layout, _, _ = cga_reference

        a_val = np.zeros(64)
        a_val[1] = 1.0  # e1
        a_val[2] = 2.0  # e2
        b_val = np.zeros(64)
        b_val[2] = 1.0  # e2
        b_val[3] = 3.0  # e3

        a_mv = layout.MultiVector(value=a_val.copy())
        b_mv = layout.MultiVector(value=b_val.copy())
        ref_result = (a_mv * b_mv).value

        a_tensor = torch.tensor(a_val, dtype=torch.float32).unsqueeze(0)
        b_tensor = torch.tensor(b_val, dtype=torch.float32).unsqueeze(0)
        our_result = functional_module.geometric_product_full(a_tensor, b_tensor)

        assert torch.allclose(
            our_result,
            torch.tensor(ref_result, dtype=torch.float32).unsqueeze(0),
            atol=1e-6
        ), f"Expected {ref_result}, got {our_result}"

    def test_random_multivectors(self, cga_reference, functional_module):
        """Test with random multivector inputs."""
        layout, _, _ = cga_reference

        np.random.seed(42)
        a_val = np.random.randn(64)
        b_val = np.random.randn(64)

        a_mv = layout.MultiVector(value=a_val.copy())
        b_mv = layout.MultiVector(value=b_val.copy())
        ref_result = (a_mv * b_mv).value

        a_tensor = torch.tensor(a_val, dtype=torch.float32).unsqueeze(0)
        b_tensor = torch.tensor(b_val, dtype=torch.float32).unsqueeze(0)
        our_result = functional_module.geometric_product_full(a_tensor, b_tensor)

        assert torch.allclose(
            our_result,
            torch.tensor(ref_result, dtype=torch.float32).unsqueeze(0),
            atol=1e-5
        )

    def test_batched_computation(self, cga_reference, functional_module):
        """Test batch dimension handling."""
        layout, _, _ = cga_reference

        batch_size = 10
        np.random.seed(123)

        a_vals = np.random.randn(batch_size, 64)
        b_vals = np.random.randn(batch_size, 64)

        ref_results = []
        for i in range(batch_size):
            a_mv = layout.MultiVector(value=a_vals[i].copy())
            b_mv = layout.MultiVector(value=b_vals[i].copy())
            ref_results.append((a_mv * b_mv).value)
        ref_results = np.array(ref_results)

        a_tensor = torch.tensor(a_vals, dtype=torch.float32)
        b_tensor = torch.tensor(b_vals, dtype=torch.float32)
        our_result = functional_module.geometric_product_full(a_tensor, b_tensor)

        assert torch.allclose(
            our_result,
            torch.tensor(ref_results, dtype=torch.float32),
            atol=1e-5
        )


# =============================================================================
# Null Basis Properties
# =============================================================================

class TestNullBasisProperties:
    """Test n_o · n_inf = -1 and related properties."""

    def test_eo_einf_product(self, cga_reference, functional_module):
        """Test that n_o · n_inf = -1."""
        layout, _, stuff = cga_reference

        eo = stuff['eo'].value
        einf = stuff['einf'].value

        eo_tensor = torch.tensor(eo, dtype=torch.float32).unsqueeze(0)
        einf_tensor = torch.tensor(einf, dtype=torch.float32).unsqueeze(0)
        result = functional_module.geometric_product_full(eo_tensor, einf_tensor)

        scalar_component = result[0, 0].item()
        assert np.isclose(scalar_component, -1.0, atol=1e-6), \
            f"Expected n_o · n_inf = -1, got {scalar_component}"

    def test_eo_squared_zero(self, cga_reference, functional_module):
        """Test that n_o² = 0."""
        _, _, stuff = cga_reference

        eo = stuff['eo'].value
        eo_tensor = torch.tensor(eo, dtype=torch.float32).unsqueeze(0)
        result = functional_module.geometric_product_full(eo_tensor, eo_tensor)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6), \
            f"Expected n_o² = 0, got {result}"

    def test_einf_squared_zero(self, cga_reference, functional_module):
        """Test that n_inf² = 0."""
        _, _, stuff = cga_reference

        einf = stuff['einf'].value
        einf_tensor = torch.tensor(einf, dtype=torch.float32).unsqueeze(0)
        result = functional_module.geometric_product_full(einf_tensor, einf_tensor)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6), \
            f"Expected n_inf² = 0, got {result}"


# =============================================================================
# Associativity Test
# =============================================================================

class TestAssociativity:
    """Test that (a · b) · c = a · (b · c)."""

    def test_associativity_random(self, cga_reference, functional_module):
        """Test associativity with random multivectors."""
        np.random.seed(42)

        for _ in range(5):  # 5 random tests
            a_val = np.random.randn(64)
            b_val = np.random.randn(64)
            c_val = np.random.randn(64)

            a = torch.tensor(a_val, dtype=torch.float32).unsqueeze(0)
            b = torch.tensor(b_val, dtype=torch.float32).unsqueeze(0)
            c = torch.tensor(c_val, dtype=torch.float32).unsqueeze(0)

            gp = functional_module.geometric_product_full

            ab = gp(a, b)
            ab_c = gp(ab, c)

            bc = gp(b, c)
            a_bc = gp(a, bc)

            assert torch.allclose(ab_c, a_bc, atol=1e-4), \
                f"Associativity failed: (a·b)·c != a·(b·c)"

    def test_associativity_basis_vectors(self, cga_reference, functional_module):
        """Test associativity with basis vectors."""
        e1 = torch.zeros(1, 64)
        e1[0, 1] = 1.0
        e2 = torch.zeros(1, 64)
        e2[0, 2] = 1.0
        e3 = torch.zeros(1, 64)
        e3[0, 3] = 1.0

        gp = functional_module.geometric_product_full

        e1e2 = gp(e1, e2)
        e1e2_e3 = gp(e1e2, e3)

        e2e3 = gp(e2, e3)
        e1_e2e3 = gp(e1, e2e3)

        assert torch.allclose(e1e2_e3, e1_e2e3, atol=1e-6)


# =============================================================================
# Module Import Test
# =============================================================================

class TestModuleImport:
    """Test that algebra.py can be imported."""

    def test_algebra_importable(self):
        """Verify algebra module is importable."""
        from fast_clifford.algebras.cga4d import algebra
        assert algebra.BLADE_COUNT == 64
        assert algebra.EUCLIDEAN_DIM == 4

    def test_product_table_complete(self):
        """Verify product table has all entries."""
        from fast_clifford.algebras.cga4d import algebra
        table = algebra.get_product_table()
        # Should have 4096 entries (64 × 64)
        assert len(table) == 4096


# =============================================================================
# Reverse Operation Tests
# =============================================================================

class TestReverseOperation:
    """Test the reverse operation."""

    def test_reverse_grades(self, cga_reference, functional_module):
        """Test reverse signs by grade."""
        layout, _, _ = cga_reference

        # Test with a random blade from each grade
        test_indices = [0, 1, 7, 22, 42, 57, 63]  # One from each grade

        for idx in test_indices:
            mv_val = np.zeros(64)
            mv_val[idx] = 1.0

            mv = layout.MultiVector(value=mv_val.copy())
            ref_rev = (~mv).value

            mv_tensor = torch.tensor(mv_val, dtype=torch.float32).unsqueeze(0)
            our_rev = functional_module.reverse_full(mv_tensor)

            assert torch.allclose(
                our_rev,
                torch.tensor(ref_rev, dtype=torch.float32).unsqueeze(0),
                atol=1e-6
            ), f"Reverse failed for index {idx}"

    def test_reverse_random(self, cga_reference, functional_module):
        """Test reverse with random multivector."""
        layout, _, _ = cga_reference

        np.random.seed(42)
        mv_val = np.random.randn(64)

        mv = layout.MultiVector(value=mv_val.copy())
        ref_rev = (~mv).value

        mv_tensor = torch.tensor(mv_val, dtype=torch.float32).unsqueeze(0)
        our_rev = functional_module.reverse_full(mv_tensor)

        assert torch.allclose(
            our_rev,
            torch.tensor(ref_rev, dtype=torch.float32).unsqueeze(0),
            atol=1e-6
        )


# =============================================================================
# Sparse Sandwich Product Correctness
# =============================================================================

class TestSparseSandwichProduct:
    """Test sparse sandwich product against full computation."""

    def test_identity_ev(self, functional_module):
        """Test with identity EvenVersor (scalar = 1, all else zero)."""
        ev = torch.zeros(1, 32)
        ev[0, 0] = 1.0  # scalar

        point = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, 1.5]])

        result = functional_module.sandwich_product_sparse(ev, point)

        assert torch.allclose(result, point, atol=1e-6), \
            f"Identity EvenVersor should not change point. Got {result}"

    def test_against_full_computation(self, cga_reference, functional_module):
        """Test sparse sandwich product against full computation."""
        layout, _, stuff = cga_reference

        # Create a simple EvenVersor (pure rotation in e12 plane)
        theta = np.pi / 4  # 45 degrees
        ev_full = np.zeros(64)
        ev_full[0] = np.cos(theta / 2)   # scalar
        ev_full[7] = np.sin(theta / 2)   # e12

        # Create a sparse EvenVersor (31 components)
        ev_sparse = torch.zeros(1, 32)
        ev_sparse[0, 0] = ev_full[0]  # scalar
        ev_sparse[0, 1] = ev_full[7]  # e12

        # Create test point using up()
        x_4d = np.array([1.0, 0.0, 0.0, 0.0])
        up_func = stuff['up']
        e1, e2, e3, e4 = (layout.blades['e1'], layout.blades['e2'],
                          layout.blades['e3'], layout.blades['e4'])
        x_mv = x_4d[0] * e1 + x_4d[1] * e2 + x_4d[2] * e3 + x_4d[3] * e4
        X_full = up_func(x_mv).value

        # Get sparse point (Grade 1 only)
        point_sparse = torch.tensor([[
            X_full[1], X_full[2], X_full[3], X_full[4], X_full[5], X_full[6]
        ]], dtype=torch.float32)

        # Compute with sparse function
        result_sparse = functional_module.sandwich_product_sparse(ev_sparse, point_sparse)

        # Compute with full function for reference
        ev_full_t = torch.tensor(ev_full, dtype=torch.float32).unsqueeze(0)
        X_full_t = torch.tensor(X_full, dtype=torch.float32).unsqueeze(0)

        gp = functional_module.geometric_product_full
        rev = functional_module.reverse_full

        MX = gp(ev_full_t, X_full_t)
        M_rev = rev(ev_full_t)
        result_full = gp(MX, M_rev)

        # Extract Grade 1 from full result
        result_full_grade1 = result_full[0, 1:7]

        assert torch.allclose(result_sparse[0], result_full_grade1, atol=1e-5), \
            f"Sparse result {result_sparse[0]} != Full result {result_full_grade1}"

    def test_batched_sandwich_product(self, functional_module):
        """Test batched computation."""
        batch_size = 5

        evs = torch.randn(batch_size, 32)
        evs[:, 0] = 1.0  # Ensure non-zero scalar

        points = torch.randn(batch_size, 6)

        result = functional_module.sandwich_product_sparse(evs, points)

        assert result.shape == (batch_size, 6)


# =============================================================================
# Multiplication Count Test
# =============================================================================

class TestMultiplicationCount:
    """Verify sparse sandwich product has significant reduction vs full."""

    def test_multiplication_count(self):
        """Count multiplications in generated sandwich product code."""
        from fast_clifford.codegen.sparse_analysis import (
            get_sandwich_product_terms_generic,
            count_multiplication_ops,
        )

        terms = get_sandwich_product_terms_generic(4)
        total_muls = count_multiplication_ops(terms)
        print(f"CGA4D Total multiplications: {total_muls}")

        # Full sandwich product would require ~2 * 64 * 64 = 8192 multiplications
        full_muls = 2 * 64 * 64
        reduction_ratio = total_muls / full_muls
        print(f"Reduction ratio: {reduction_ratio:.2%} of full ({full_muls})")

        assert total_muls < full_muls, \
            f"Sparse implementation should be faster than full"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases for sandwich product."""

    def test_zero_vector_point(self, functional_module):
        """Test with zero 4D vector (origin)."""
        ev = torch.zeros(1, 32)
        ev[0, 0] = 1.0

        zero_4d = torch.zeros(1, 4)
        point = functional_module.cga_encode(zero_4d)

        result = functional_module.sandwich_product_sparse(ev, point)

        assert torch.allclose(result, point, atol=1e-6), \
            f"Zero vector point should be unchanged by identity. Got {result}"

    def test_identity_ev_preserves_point(self, functional_module):
        """Test identity EvenVersor (scalar=1) preserves any point."""
        ev = torch.zeros(1, 32)
        ev[0, 0] = 1.0  # Identity: M = 1

        for _ in range(5):
            point = torch.randn(1, 6)
            result = functional_module.sandwich_product_sparse(ev, point)

            assert torch.allclose(result, point, atol=1e-6), \
                "Identity EvenVersor must preserve point"

    def test_cga_encode_decode_roundtrip(self, functional_module):
        """Test UPGC encode/decode roundtrip."""
        x_4d = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        encoded = functional_module.cga_encode(x_4d)
        decoded = functional_module.cga_decode(encoded)

        assert torch.allclose(decoded, x_4d, atol=1e-6), \
            f"Roundtrip failed: {x_4d} -> {encoded} -> {decoded}"

    def test_cga_encode_origin(self, functional_module):
        """Test UPGC encoding of origin."""
        origin = torch.zeros(1, 4)
        point = functional_module.cga_encode(origin)

        # At origin: e1=e2=e3=e4=0, e+ = -0.5, e- = 0.5
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, -0.5, 0.5]])

        assert torch.allclose(point, expected, atol=1e-6), \
            f"Origin encoding failed. Expected {expected}, got {point}"


# =============================================================================
# CliffordTransformLayer Tests
# =============================================================================

class TestCliffordTransformLayer:
    """Test CliffordTransformLayer module."""

    def test_layer_basic_functionality(self):
        """Test basic forward pass."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()

        ev = torch.zeros(1, 32)
        ev[0, 0] = 1.0

        point = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, 1.5]])

        result = layer(ev, point)

        assert result.shape == point.shape
        assert torch.allclose(result, point, atol=1e-6)

    def test_layer_batched(self):
        """Test batched computation."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()

        batch_size = 10
        ev = torch.randn(batch_size, 32)
        ev[:, 0] = 1.0  # Ensure valid EvenVersor

        point = torch.randn(batch_size, 6)

        result = layer(ev, point)

        assert result.shape == (batch_size, 6)

    def test_precision_handling_float16(self):
        """Test that float16 inputs are handled correctly."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()

        ev = torch.zeros(1, 32, dtype=torch.float16)
        ev[0, 0] = 1.0

        point = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, 1.5]], dtype=torch.float16)

        result = layer(ev, point)

        assert result.dtype == torch.float16
        assert torch.allclose(result, point, atol=1e-3)

    def test_precision_handling_float32(self):
        """Test that float32 inputs produce float32 outputs."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()

        ev = torch.randn(1, 32, dtype=torch.float32)
        ev[0, 0] = 1.0

        point = torch.randn(1, 6, dtype=torch.float32)

        result = layer(ev, point)

        assert result.dtype == torch.float32


class TestCGAPipeline:
    """Test complete CGA4D transformation pipeline."""

    def test_pipeline_roundtrip(self):
        """Test that identity EvenVersor preserves 4D point."""
        from fast_clifford.algebras.cga4d.layers import CGAPipeline

        pipeline = CGAPipeline()

        ev = torch.zeros(1, 32)
        ev[0, 0] = 1.0

        x_4d = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        y_4d = pipeline(ev, x_4d)

        assert y_4d.shape == x_4d.shape
        assert torch.allclose(y_4d, x_4d, atol=1e-6)

    def test_pipeline_rotation_e12(self):
        """Test rotation in e1-e2 plane."""
        from fast_clifford.algebras.cga4d.layers import CGAPipeline
        import numpy as np

        pipeline = CGAPipeline()

        # Rotation in e12 plane by 90 degrees
        theta = np.pi / 2
        ev = torch.zeros(1, 32)
        ev[0, 0] = np.cos(theta / 2)  # scalar
        ev[0, 1] = np.sin(theta / 2)  # e12

        # Point on e1 axis
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        y = pipeline(ev, x)

        # After 90 deg rotation: (1,0,0,0) -> (0,-1,0,0) in GA convention
        expected = torch.tensor([[0.0, -1.0, 0.0, 0.0]])

        assert torch.allclose(y, expected, atol=1e-5), \
            f"Expected {expected}, got {y}"


# =============================================================================
# Cross-platform Tests
# =============================================================================

class TestCrossPlatform:
    """Cross-platform tests (MPS/CUDA/CPU)."""

    def test_cpu_computation(self):
        """Test computation on CPU."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()
        ev = torch.randn(4, 32, device='cpu')
        point = torch.randn(4, 6, device='cpu')

        result = layer(ev, point)
        assert result.device.type == 'cpu'
        assert result.shape == (4, 6)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_computation(self):
        """Test computation on CUDA."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer().cuda()
        ev = torch.randn(4, 32, device='cuda')
        point = torch.randn(4, 6, device='cuda')

        result = layer(ev, point)
        assert result.device.type == 'cuda'
        assert result.shape == (4, 6)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_computation(self):
        """Test computation on MPS (Apple Silicon)."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer().to('mps')
        ev = torch.randn(4, 32, device='mps')
        point = torch.randn(4, 6, device='mps')

        result = layer(ev, point)
        assert result.device.type == 'mps'
        assert result.shape == (4, 6)


# =============================================================================
# Precision Tests
# =============================================================================

class TestPrecision:
    """Precision tests (float32 vs float16)."""

    def test_float32_precision(self):
        """Test computation in float32."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()
        ev = torch.randn(4, 32, dtype=torch.float32)
        point = torch.randn(4, 6, dtype=torch.float32)

        result = layer(ev, point)
        assert result.dtype == torch.float32

    def test_float16_precision(self):
        """Test computation in float16 (with internal float32)."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()
        ev = torch.randn(4, 32, dtype=torch.float16)
        point = torch.randn(4, 6, dtype=torch.float16)

        result = layer(ev, point)
        assert result.dtype == torch.float16

    def test_float16_vs_float32_consistency(self):
        """Verify float16 produces similar results to float32."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()

        ev_f32 = torch.randn(4, 32, dtype=torch.float32)
        point_f32 = torch.randn(4, 6, dtype=torch.float32)

        ev_f16 = ev_f32.to(torch.float16)
        point_f16 = point_f32.to(torch.float16)

        result_f32 = layer(ev_f32, point_f32)
        result_f16 = layer(ev_f16, point_f16)

        result_f16_as_f32 = result_f16.to(torch.float32)

        # fp16 has ~3 decimal digits of precision, use relative tolerance
        # CGA4D has more terms than CGA3D, so allow more tolerance
        assert torch.allclose(result_f32, result_f16_as_f32, rtol=5e-2, atol=1e-1), \
            f"Float16 and float32 results differ too much"


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients:
    """Test gradient computation for CGA4D operations."""

    def test_transform_layer_gradients(self):
        """Test that CliffordTransformLayer supports gradient computation."""
        from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer

        layer = CliffordTransformLayer()

        ev = torch.randn(4, 32, requires_grad=True)
        point = torch.randn(4, 6, requires_grad=True)

        result = layer(ev, point)
        loss = result.sum()
        loss.backward()

        assert ev.grad is not None
        assert point.grad is not None
        assert ev.grad.shape == ev.shape
        assert point.grad.shape == point.shape

    def test_gradcheck_transform_layer(self):
        """Test gradient correctness with gradcheck."""
        from fast_clifford.algebras.cga4d import functional as F

        def func(ev, point):
            return F.sandwich_product_sparse(ev, point)

        ev = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        point = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(func, (ev, point), eps=1e-6, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
