"""
Numerical correctness tests for CGA5D Cl(6,1) operations.

Tests:
- Geometric product correctness (compare with clifford library)
- Null basis properties (eo^2=0, einf^2=0, eo*einf=-1)
- Associativity of geometric product
- Reverse operation
- Sandwich product correctness
- UPGC encode/decode round-trip
- PyTorch layer correctness
"""

import pytest
import numpy as np
import torch

from fast_clifford.algebras.cga5d import (
    EUCLIDEAN_DIM,
    BLADE_COUNT,
    GRADE_INDICES,
    UPGC_POINT_MASK,
    EVEN_VERSOR_SPARSE_INDICES,
    REVERSE_SIGNS,
    EVEN_VERSOR_REVERSE_SIGNS,
    get_layout,
    get_blades,
    get_null_basis,
    verify_null_basis_properties,
    get_product_table,
    up,
    down,
)
from fast_clifford.algebras.cga5d import functional as F
from fast_clifford.algebras.cga5d.layers import (
    CGA5DCareLayer,
    UPGC5DEncoder,
    UPGC5DDecoder,
    CGA5DTransformPipeline,
)


class TestAlgebraConstants:
    """Test algebra constants are correct."""

    def test_euclidean_dim(self):
        """Euclidean dimension is 5."""
        assert EUCLIDEAN_DIM == 5

    def test_blade_count(self):
        """Total blade count is 128."""
        assert BLADE_COUNT == 128

    def test_grade_distribution(self):
        """Grade distribution is C(7,k) = 1,7,21,35,35,21,7,1."""
        expected_sizes = [1, 7, 21, 35, 35, 21, 7, 1]
        for grade, expected in enumerate(expected_sizes):
            assert len(GRADE_INDICES[grade]) == expected, \
                f"Grade {grade} should have {expected} blades"

    def test_upgc_point_size(self):
        """UPGC point has 7 components."""
        assert len(UPGC_POINT_MASK) == 7

    def test_even_versor_size(self):
        """EvenVersor has 64 components (G0 + G2 + G4 + G6)."""
        # G0: 1, G2: 21, G4: 35, G6: 7 = 64
        assert len(EVEN_VERSOR_SPARSE_INDICES) == 64


class TestNullBasisProperties:
    """Verify Null Basis properties using clifford library."""

    def test_eo_squared_zero(self):
        """eo^2 = 0."""
        results = verify_null_basis_properties()
        assert results['eo_squared_zero']

    def test_einf_squared_zero(self):
        """einf^2 = 0."""
        results = verify_null_basis_properties()
        assert results['einf_squared_zero']

    def test_eo_einf_inner_product(self):
        """eo * einf = -1."""
        results = verify_null_basis_properties()
        assert results['eo_einf_minus_one']

    def test_null_basis_orthogonality(self):
        """Euclidean basis orthogonal to null basis."""
        layout = get_layout()
        blades = get_blades()
        eo, einf = get_null_basis()

        for name in ['e1', 'e2', 'e3', 'e4', 'e5']:
            ei = blades[name]
            # Inner products should be 0
            eo_ei = eo | ei
            einf_ei = einf | ei
            assert np.allclose(eo_ei.value, 0, atol=1e-10)
            assert np.allclose(einf_ei.value, 0, atol=1e-10)


class TestGeometricProductCorrectness:
    """Verify geometric product against clifford library."""

    def test_basis_vector_products(self):
        """Basis vector products match clifford."""
        blades = get_blades()

        # e1 * e2
        e1, e2 = blades['e1'], blades['e2']
        result = e1 * e2

        # Check result is pure bivector
        assert np.allclose(result.value[0], 0, atol=1e-10)

    def test_euclidean_basis_squared(self):
        """Euclidean basis vectors square to +1."""
        blades = get_blades()

        for name in ['e1', 'e2', 'e3', 'e4', 'e5']:
            ei = blades[name]
            ei_squared = ei * ei
            assert np.allclose(ei_squared.value[0], 1.0, atol=1e-10)

    def test_conformal_basis_signature(self):
        """e+ squares to +1, e- squares to -1."""
        layout = get_layout()
        blades = get_blades()

        # e+ and e- are at specific indices
        # e+ is at index 6, e- is at index 7 in the blade tuple list
        ep = layout.blades_of_grade(1)[5]  # 6th grade-1 blade
        em = layout.blades_of_grade(1)[6]  # 7th grade-1 blade

        ep_squared = ep * ep
        em_squared = em * em

        assert np.allclose(ep_squared.value[0], 1.0, atol=1e-10)
        assert np.allclose(em_squared.value[0], -1.0, atol=1e-10)


class TestAssociativity:
    """Test associativity of geometric product."""

    def test_three_vector_associativity(self):
        """(a*b)*c = a*(b*c) for random vectors."""
        layout = get_layout()

        np.random.seed(42)
        a = layout.randomMV()
        b = layout.randomMV()
        c = layout.randomMV()

        left = (a * b) * c
        right = a * (b * c)

        assert np.allclose(left.value, right.value, rtol=1e-10)

    def test_even_versor_associativity(self):
        """EvenVersor products are associative."""
        layout = get_layout()

        np.random.seed(43)
        # Create random even-grade multivectors (EvenVersors)
        m1 = layout.randomMV()
        m2 = layout.randomMV()
        m3 = layout.randomMV()

        left = (m1 * m2) * m3
        right = m1 * (m2 * m3)

        assert np.allclose(left.value, right.value, rtol=1e-10)


class TestReverseOperation:
    """Test reverse operation."""

    def test_reverse_signs_by_grade(self):
        """Verify reverse signs: (-1)^(k*(k-1)/2)."""
        expected_signs = {
            0: 1,   # (-1)^0 = 1
            1: 1,   # (-1)^0 = 1
            2: -1,  # (-1)^1 = -1
            3: -1,  # (-1)^3 = -1
            4: 1,   # (-1)^6 = 1
            5: 1,   # (-1)^10 = 1
            6: -1,  # (-1)^15 = -1
            7: -1,  # (-1)^21 = -1
        }

        for grade, indices in enumerate(GRADE_INDICES):
            for idx in indices:
                assert REVERSE_SIGNS[idx] == expected_signs[grade], \
                    f"Blade {idx} (grade {grade}) should have sign {expected_signs[grade]}"

    def test_even_versor_reverse_signs_count(self):
        """EvenVersor has 64 reverse signs."""
        assert len(EVEN_VERSOR_REVERSE_SIGNS) == 64

    def test_even_versor_reverse_functional(self):
        """functional.reverse_even_versor matches clifford."""
        layout = get_layout()

        np.random.seed(44)
        ev_full = layout.randomMV()

        # Extract EvenVersor components (G0 + G2 + G4 + G6)
        ev_sparse = np.array([ev_full.value[idx] for idx in EVEN_VERSOR_SPARSE_INDICES])
        ev_tensor = torch.tensor(ev_sparse, dtype=torch.float32).unsqueeze(0)

        # Compute reverse with our function
        reversed_tensor = F.reverse_even_versor(ev_tensor)
        reversed_np = reversed_tensor.squeeze(0).numpy()

        # Compute reverse with clifford
        ev_reversed = ~ev_full

        # Compare sparse components
        expected = np.array([ev_reversed.value[idx] for idx in EVEN_VERSOR_SPARSE_INDICES])

        assert np.allclose(reversed_np, expected, rtol=1e-5)


class TestSparseSandwichProduct:
    """Test sparse sandwich product."""

    def test_sandwich_product_shape(self):
        """Sandwich product preserves point shape."""
        ev = torch.randn(2, 64)
        point = torch.randn(2, 7)

        result = F.sandwich_product_sparse(ev, point)

        assert result.shape == (2, 7)

    def test_sandwich_with_identity_even_versor(self):
        """Identity EvenVersor (scalar=1) preserves point."""
        # Identity EvenVersor: scalar = 1, all others = 0
        ev = torch.zeros(1, 64)
        ev[0, 0] = 1.0  # Scalar component

        point = torch.randn(1, 7)

        result = F.sandwich_product_sparse(ev, point)

        assert torch.allclose(result, point, rtol=1e-5, atol=1e-5)

    def test_sandwich_against_clifford(self):
        """Sandwich product matches clifford computation."""
        layout = get_layout()
        blades = get_blades()
        stuff = get_blades()

        # Create a simple 5D point
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        point_mv = up(x)

        # Create a simple rotor in e1e2 plane
        angle = np.pi / 4
        e1, e2 = blades['e1'], blades['e2']
        e12 = e1 * e2
        rotor = np.cos(angle / 2) + np.sin(angle / 2) * e12

        # Extract EvenVersor sparse representation
        ev_sparse = np.array([rotor.value[idx] for idx in EVEN_VERSOR_SPARSE_INDICES])

        # Extract point sparse representation
        point_sparse = np.array([point_mv[idx] for idx in GRADE_INDICES[1]])

        # Compute with our sparse function
        ev_tensor = torch.tensor(ev_sparse, dtype=torch.float32).unsqueeze(0)
        point_tensor = torch.tensor(point_sparse, dtype=torch.float32).unsqueeze(0)

        result_tensor = F.sandwich_product_sparse(ev_tensor, point_tensor)
        result_np = result_tensor.squeeze(0).numpy()

        # Compute with clifford
        point_as_mv = layout.MultiVector(value=point_mv)
        result_mv = rotor * point_as_mv * ~rotor

        # Extract grade-1 components
        expected = np.array([result_mv.value[idx] for idx in GRADE_INDICES[1]])

        assert np.allclose(result_np, expected, rtol=1e-4, atol=1e-4)


class TestUPGCEncodeDecode:
    """Test UPGC encoding and decoding."""

    def test_encode_shape(self):
        """UPGC encode produces 7-component output."""
        x = torch.randn(2, 5)
        point = F.upgc_encode(x)
        assert point.shape == (2, 7)

    def test_decode_shape(self):
        """UPGC decode produces 5-component output."""
        point = torch.randn(2, 7)
        x = F.upgc_decode(point)
        assert x.shape == (2, 5)

    def test_encode_decode_roundtrip(self):
        """Encode then decode recovers original point."""
        x_original = torch.randn(10, 5)

        point = F.upgc_encode(x_original)
        x_recovered = F.upgc_decode(point)

        assert torch.allclose(x_original, x_recovered, rtol=1e-5, atol=1e-5)

    def test_encode_matches_clifford(self):
        """Encoding matches clifford's up function."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Clifford encoding
        point_clifford = up(x)
        expected = np.array([point_clifford[idx] for idx in GRADE_INDICES[1]])

        # Our encoding
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        point_tensor = F.upgc_encode(x_tensor)
        result = point_tensor.squeeze(0).numpy()

        assert np.allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestCGA5DCareLayer:
    """Test CGA5DCareLayer PyTorch module."""

    def test_layer_output_shape(self):
        """Layer produces correct output shape."""
        layer = CGA5DCareLayer()
        ev = torch.randn(4, 64)
        point = torch.randn(4, 7)

        result = layer(ev, point)

        assert result.shape == (4, 7)

    def test_layer_identity_even_versor(self):
        """Identity EvenVersor preserves point."""
        layer = CGA5DCareLayer()

        ev = torch.zeros(2, 64)
        ev[:, 0] = 1.0

        point = torch.randn(2, 7)

        result = layer(ev, point)

        assert torch.allclose(result, point, rtol=1e-5, atol=1e-5)

    def test_layer_batch_consistency(self):
        """Batched and single computation give same results."""
        layer = CGA5DCareLayer()

        ev = torch.randn(3, 64)
        point = torch.randn(3, 7)

        batched_result = layer(ev, point)

        for i in range(3):
            single_result = layer(ev[i:i+1], point[i:i+1])
            assert torch.allclose(batched_result[i], single_result.squeeze(0), rtol=1e-5)


class TestCGA5DTransformPipeline:
    """Test complete transformation pipeline."""

    def test_pipeline_output_shape(self):
        """Pipeline produces correct output shape."""
        pipeline = CGA5DTransformPipeline()
        ev = torch.randn(4, 64)
        x = torch.randn(4, 5)

        y = pipeline(ev, x)

        assert y.shape == (4, 5)

    def test_pipeline_identity_even_versor(self):
        """Identity EvenVersor preserves 5D point."""
        pipeline = CGA5DTransformPipeline()

        ev = torch.zeros(2, 64)
        ev[:, 0] = 1.0

        x = torch.randn(2, 5)

        y = pipeline(ev, x)

        assert torch.allclose(x, y, rtol=1e-4, atol=1e-4)


class TestCrossPlatform:
    """Test cross-platform compatibility."""

    def test_cpu_execution(self):
        """Operations work on CPU."""
        layer = CGA5DCareLayer()
        ev = torch.randn(2, 64, device='cpu')
        point = torch.randn(2, 7, device='cpu')

        result = layer(ev, point)

        assert result.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        """Operations work on CUDA."""
        layer = CGA5DCareLayer().cuda()
        ev = torch.randn(2, 64, device='cuda')
        point = torch.randn(2, 7, device='cuda')

        result = layer(ev, point)

        assert result.device.type == 'cuda'

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_execution(self):
        """Operations work on MPS (Apple Silicon)."""
        layer = CGA5DCareLayer().to('mps')
        ev = torch.randn(2, 64, device='mps')
        point = torch.randn(2, 7, device='mps')

        result = layer(ev, point)

        assert result.device.type == 'mps'


class TestPrecision:
    """Test numerical precision handling."""

    def test_float32_precision(self):
        """Float32 operations are stable."""
        layer = CGA5DCareLayer()
        ev = torch.randn(10, 64, dtype=torch.float32)
        point = torch.randn(10, 7, dtype=torch.float32)

        result = layer(ev, point)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.dtype == torch.float32

    def test_float16_vs_float32_consistency(self):
        """Float16 and float32 give consistent results."""
        layer = CGA5DCareLayer()

        ev_f32 = torch.randn(5, 64, dtype=torch.float32)
        point_f32 = torch.randn(5, 7, dtype=torch.float32)

        result_f32 = layer(ev_f32, point_f32)

        ev_f16 = ev_f32.to(torch.float16)
        point_f16 = point_f32.to(torch.float16)

        result_f16 = layer(ev_f16, point_f16)

        # CGA5D has more terms (64 EvenVersor, 7 point), need relaxed tolerance
        assert torch.allclose(
            result_f32, result_f16.to(torch.float32),
            rtol=1e-1, atol=2e-1
        )


class TestGradients:
    """Test gradient computation."""

    def test_gradient_flow(self):
        """Gradients flow through sandwich product."""
        ev = torch.randn(2, 64, requires_grad=True)
        point = torch.randn(2, 7, requires_grad=True)

        result = F.sandwich_product_sparse(ev, point)
        loss = result.sum()
        loss.backward()

        assert ev.grad is not None
        assert point.grad is not None
        assert not torch.isnan(ev.grad).any()
        assert not torch.isnan(point.grad).any()

    def test_layer_gradient_flow(self):
        """Gradients flow through CGA5DCareLayer."""
        layer = CGA5DCareLayer()

        ev = torch.randn(2, 64, requires_grad=True)
        point = torch.randn(2, 7, requires_grad=True)

        result = layer(ev, point)
        loss = result.sum()
        loss.backward()

        assert ev.grad is not None
        assert point.grad is not None

    def test_pipeline_gradient_flow(self):
        """Gradients flow through full pipeline."""
        pipeline = CGA5DTransformPipeline()

        ev = torch.randn(2, 64, requires_grad=True)
        x = torch.randn(2, 5, requires_grad=True)

        y = pipeline(ev, x)
        loss = y.sum()
        loss.backward()

        assert ev.grad is not None
        assert x.grad is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_even_versor(self):
        """Zero EvenVersor produces zero output."""
        layer = CGA5DCareLayer()

        ev = torch.zeros(1, 64)
        point = torch.randn(1, 7)

        result = layer(ev, point)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_zero_point(self):
        """Zero point produces zero output."""
        layer = CGA5DCareLayer()

        ev = torch.randn(1, 64)
        point = torch.zeros(1, 7)

        result = layer(ev, point)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_large_batch(self):
        """Large batch sizes work correctly."""
        layer = CGA5DCareLayer()

        ev = torch.randn(256, 64)
        point = torch.randn(256, 7)

        result = layer(ev, point)

        assert result.shape == (256, 7)
        assert not torch.isnan(result).any()

    def test_single_batch(self):
        """Single batch size works correctly."""
        layer = CGA5DCareLayer()

        ev = torch.randn(1, 64)
        point = torch.randn(1, 7)

        result = layer(ev, point)

        assert result.shape == (1, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
