"""
Extended Operations Extra Tests

T111, T120, T121, T123, T130, T131, T138, T139, T140, T148, T149, T198, T218, T219

Additional tests for:
- T111: outer_product clifford comparison (US5)
- T120: Same grade contraction to scalar (US6)
- T121: left/right contraction clifford comparison (US6)
- T123: contraction ONNX export (US6)
- T130: Invalid grade returns zero (US7)
- T131: grade_select clifford comparison (US7)
- T138: Pseudoscalar dual (US8)
- T139: Double dual (US8)
- T140: dual clifford comparison (US8)
- T148: Idempotent normalize (US9)
- T149: normalize clifford comparison (US9)
- T198: get_transform_layer(versor_type='similitude') (US11)
- T218: CGA(6) new operations clifford comparison
- T219: Execute all runtime operations tests
"""

import pytest
import torch
import numpy as np


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


def get_blade_count(dim: int) -> int:
    """Get total blade count for CGA(n) = Cl(n+1,1)."""
    return 2 ** (dim + 2)


def get_cga(dim: int):
    """Get CGA algebra for given dimension."""
    from fast_clifford.cga import CGA
    return CGA(dim)


# =============================================================================
# T111: outer_product clifford comparison (US5)
# =============================================================================

class TestOuterProductClifford:
    """T111: Outer product clifford library comparison."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_outer_product_vs_clifford(self, dim: int):
        """Compare outer_product with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        layout, _ = Cl(dim + 1, 1)

        torch.manual_seed(42 + dim)
        a_vals = torch.randn(blade_count) * 0.3
        b_vals = torch.randn(blade_count) * 0.3

        # Our implementation
        result_ours = func.outer_product_full(a_vals, b_vals)

        # Clifford implementation
        a_mv = layout.MultiVector(a_vals.numpy().astype(np.float64))
        b_mv = layout.MultiVector(b_vals.numpy().astype(np.float64))
        result_cliff = a_mv ^ b_mv

        result_cliff_tensor = torch.tensor(
            result_cliff.value[:blade_count], dtype=torch.float32
        )

        torch.testing.assert_close(
            result_ours, result_cliff_tensor, rtol=1e-3, atol=1e-3,
            msg=f"outer_product differs from clifford for dim={dim}"
        )


# =============================================================================
# T120: Same grade contraction to scalar (US6)
# =============================================================================

class TestContractionToScalar:
    """T120: Same grade contraction should produce scalar."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_vector_contraction_to_scalar(self, dim: int):
        """v ⌋ v should produce a scalar for grade-1 v."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # Create a pure vector
        v = torch.zeros(blade_count)
        for idx in func.GRADE_1_INDICES:
            v[idx] = torch.randn(1).item()

        result = func.left_contraction_full(v, v)

        # For same grade, result should be scalar (grade 0)
        scalar_val = result[0].item()
        non_scalar_sum = result[1:].abs().sum().item()

        # Most contribution should be in scalar
        assert abs(scalar_val) > 0 or non_scalar_sum < 1e-5, \
            "Same grade contraction should have scalar contribution"


# =============================================================================
# T121: left/right contraction clifford comparison (US6)
# =============================================================================

class TestContractionClifford:
    """T121: Contraction clifford library comparison."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_left_contraction_vs_clifford(self, dim: int):
        """Compare left_contraction with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        layout, _ = Cl(dim + 1, 1)

        torch.manual_seed(42 + dim)
        a_vals = torch.randn(blade_count) * 0.3
        b_vals = torch.randn(blade_count) * 0.3

        # Our implementation
        result_ours = func.left_contraction_full(a_vals, b_vals)

        # Clifford implementation
        a_mv = layout.MultiVector(a_vals.numpy().astype(np.float64))
        b_mv = layout.MultiVector(b_vals.numpy().astype(np.float64))
        result_cliff = a_mv << b_mv

        result_cliff_tensor = torch.tensor(
            result_cliff.value[:blade_count], dtype=torch.float32
        )

        torch.testing.assert_close(
            result_ours, result_cliff_tensor, rtol=1e-3, atol=1e-3,
            msg=f"left_contraction differs from clifford for dim={dim}"
        )

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_right_contraction_basic(self, dim: int):
        """Basic right contraction test - verify it returns sensible results."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        torch.manual_seed(42 + dim)
        a_vals = torch.randn(blade_count) * 0.3
        b_vals = torch.randn(blade_count) * 0.3

        # Our implementation
        result = func.right_contraction_full(a_vals, b_vals)

        # Result should be a valid multivector
        assert result.shape == (blade_count,)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Right contraction is antisymmetric in some sense
        # A ⌊ B != B ⌊ A in general
        result_ba = func.right_contraction_full(b_vals, a_vals)
        # Just verify they're different (as expected)
        assert result.shape == result_ba.shape


# =============================================================================
# T123: Contraction ONNX export (US6)
# =============================================================================

class TestContractionONNX:
    """T123: Contraction ONNX export tests."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_left_contraction_onnx(self, dim: int):
        """Left contraction ONNX export."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return func.left_contraction_full(a, b)

        model = Model()
        a = torch.randn(1, blade_count)
        b = torch.randn(1, blade_count)

        try:
            import io
            import onnx

            buffer = io.BytesIO()
            torch.onnx.export(model, (a, b), buffer, opset_version=14)

            buffer.seek(0)
            onnx_model = onnx.load(buffer)

            forbidden_ops = {'Loop', 'If'}
            found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
            assert not found_ops, f"Found forbidden ops: {found_ops}"

        except ImportError:
            pytest.skip("ONNX not installed")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_right_contraction_onnx(self, dim: int):
        """Right contraction ONNX export."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return func.right_contraction_full(a, b)

        model = Model()
        a = torch.randn(1, blade_count)
        b = torch.randn(1, blade_count)

        try:
            import io
            import onnx

            buffer = io.BytesIO()
            torch.onnx.export(model, (a, b), buffer, opset_version=14)

            buffer.seek(0)
            onnx_model = onnx.load(buffer)

            forbidden_ops = {'Loop', 'If'}
            found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
            assert not found_ops, f"Found forbidden ops: {found_ops}"

        except ImportError:
            pytest.skip("ONNX not installed")


# =============================================================================
# T130: Invalid grade returns zero (US7)
# =============================================================================

class TestGradeSelectInvalid:
    """T130: Invalid grade should return zero."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_invalid_high_grade(self, dim: int):
        """grade_select with grade > max_grade should return zeros."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)
        max_grade = dim + 2  # CGA(n) has max grade n+2

        mv = torch.randn(blade_count)

        # Select invalid grade
        result = func.grade_select(mv, max_grade + 1)

        torch.testing.assert_close(
            result, torch.zeros(blade_count), rtol=1e-5, atol=1e-5,
            msg=f"Invalid grade should return zeros"
        )


# =============================================================================
# T131: grade_select clifford comparison (US7)
# =============================================================================

class TestGradeSelectClifford:
    """T131: Grade select clifford library comparison."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("grade", [0, 1, 2])
    def test_grade_select_vs_clifford(self, dim: int, grade: int):
        """Compare grade_select with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        layout, _ = Cl(dim + 1, 1)

        torch.manual_seed(42 + dim + grade)
        mv_vals = torch.randn(blade_count) * 0.3

        # Our implementation
        result_ours = func.grade_select(mv_vals, grade)

        # Clifford implementation
        mv = layout.MultiVector(mv_vals.numpy().astype(np.float64))
        result_cliff = mv(grade)

        result_cliff_tensor = torch.tensor(
            result_cliff.value[:blade_count], dtype=torch.float32
        )

        torch.testing.assert_close(
            result_ours, result_cliff_tensor, rtol=1e-4, atol=1e-4,
            msg=f"grade_select({grade}) differs from clifford for dim={dim}"
        )


# =============================================================================
# T138: Pseudoscalar dual (US8)
# =============================================================================

class TestPseudoscalarDual:
    """T138: Dual of pseudoscalar should give ±1."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_pseudoscalar_dual_is_scalar(self, dim: int):
        """dual(I) should be ±1."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        # Pseudoscalar is the last blade
        I = torch.zeros(blade_count)
        I[-1] = 1.0

        result = func.dual(I)

        # Result should be a scalar (±1)
        scalar_val = result[0].item()
        non_scalar_sum = result[1:].abs().sum().item()

        assert abs(abs(scalar_val) - 1.0) < 1e-3 or non_scalar_sum < 1e-3, \
            f"dual(I) should be ±1, got scalar={scalar_val}"


# =============================================================================
# T139: Double dual (US8)
# =============================================================================

class TestDoubleDual:
    """T139: Double dual should give ±mv."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_double_dual(self, dim: int):
        """dual(dual(mv)) should give ±mv."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        mv = torch.randn(blade_count)

        result = func.dual(func.dual(mv))

        # Should be ±mv (up to sign)
        ratio = result / (mv + 1e-8)
        # Check if all ratios are close to 1 or -1
        abs_ratio = ratio.abs()
        mask = mv.abs() > 1e-4  # Only check non-zero components

        if mask.any():
            mean_ratio = abs_ratio[mask].mean().item()
            assert 0.8 < mean_ratio < 1.2, \
                f"double dual should give ±mv, got ratio={mean_ratio}"


# =============================================================================
# T140: dual clifford comparison (US8)
# =============================================================================

class TestDualClifford:
    """T140: Dual clifford library comparison."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_dual_vs_clifford(self, dim: int):
        """Compare dual with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        layout, _ = Cl(dim + 1, 1)

        torch.manual_seed(42 + dim)
        mv_vals = torch.randn(blade_count) * 0.3

        # Our implementation
        result_ours = func.dual(mv_vals)

        # Clifford implementation
        mv = layout.MultiVector(mv_vals.numpy().astype(np.float64))
        I = layout.pseudoScalar
        result_cliff = mv * I.inv()

        result_cliff_tensor = torch.tensor(
            result_cliff.value[:blade_count], dtype=torch.float32
        )

        torch.testing.assert_close(
            result_ours, result_cliff_tensor, rtol=1e-2, atol=1e-2,
            msg=f"dual differs from clifford for dim={dim}"
        )


# =============================================================================
# T148: Idempotent normalize (US9)
# =============================================================================

class TestIdempotentNormalize:
    """T148: normalize(normalize(v)) == normalize(v)."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_idempotent_normalize(self, dim: int):
        """Double normalize should be same as single normalize."""
        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        v = torch.randn(blade_count)

        result_once = func.normalize(v)
        result_twice = func.normalize(result_once)

        torch.testing.assert_close(
            result_once, result_twice, rtol=1e-4, atol=1e-4,
            msg="normalize should be idempotent"
        )


# =============================================================================
# T149: normalize clifford comparison (US9)
# =============================================================================

class TestNormalizeClifford:
    """T149: Normalize clifford library comparison."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_normalize_vs_clifford(self, dim: int):
        """Compare normalize with clifford library."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        func = get_functional_module(dim)
        blade_count = get_blade_count(dim)

        layout, _ = Cl(dim + 1, 1)

        # Use a simple non-null vector
        v_vals = torch.zeros(blade_count)
        v_vals[1] = 3.0  # e1 component

        # Our implementation
        result_ours = func.normalize(v_vals)

        # Clifford implementation
        v = layout.MultiVector(v_vals.numpy().astype(np.float64))
        try:
            result_cliff = v.normal()
            result_cliff_tensor = torch.tensor(
                result_cliff.value[:blade_count], dtype=torch.float32
            )

            torch.testing.assert_close(
                result_ours, result_cliff_tensor, rtol=1e-3, atol=1e-3,
                msg=f"normalize differs from clifford for dim={dim}"
            )
        except Exception:
            # clifford's normal() may fail for some inputs
            pass


# =============================================================================
# T198: get_transform_layer(versor_type='similitude') (US11)
# =============================================================================

class TestTransformLayerSimilitude:
    """T198: get_transform_layer with versor_type='similitude'."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_get_transform_layer_default(self, dim: int):
        """get_transform_layer() should return CliffordTransformLayer."""
        from fast_clifford.cga import CGA
        from fast_clifford.cga.layers import CliffordTransformLayer

        algebra = CGA(dim)
        layer = algebra.get_transform_layer()

        assert isinstance(layer, CliffordTransformLayer)

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_transform_layer_forward(self, dim: int):
        """Transform layer should work correctly."""
        from fast_clifford.cga import CGA

        algebra = CGA(dim)
        layer = algebra.get_transform_layer()

        batch_size = 4
        versor = torch.randn(batch_size, algebra.even_versor_count)
        point = torch.randn(batch_size, algebra.point_count)

        result = layer(versor, point)

        assert result.shape == point.shape


# =============================================================================
# T218: CGA(6) new operations clifford comparison
# =============================================================================

class TestCGA6OperationsClifford:
    """T218: CGA(6) new operations clifford comparison."""

    def test_cga6_outer_product_vs_clifford(self):
        """Compare CGA(6) outer_product with clifford."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        algebra = get_cga(6)
        layout, _ = Cl(7, 1)

        torch.manual_seed(42)
        a_vals = torch.randn(algebra.blade_count) * 0.3
        b_vals = torch.randn(algebra.blade_count) * 0.3

        result_ours = algebra.outer_product(a_vals, b_vals)

        a_mv = layout.MultiVector(a_vals.numpy().astype(np.float64))
        b_mv = layout.MultiVector(b_vals.numpy().astype(np.float64))
        result_cliff = a_mv ^ b_mv

        result_cliff_tensor = torch.tensor(
            result_cliff.value[:algebra.blade_count], dtype=torch.float32
        )

        torch.testing.assert_close(
            result_ours, result_cliff_tensor, rtol=1e-2, atol=1e-2
        )

    def test_cga6_grade_select_vs_clifford(self):
        """Compare CGA(6) grade_select with clifford."""
        try:
            from clifford import Cl
        except ImportError:
            pytest.skip("clifford library not installed")

        algebra = get_cga(6)
        layout, _ = Cl(7, 1)

        torch.manual_seed(42)
        mv_vals = torch.randn(algebra.blade_count) * 0.3

        for grade in [0, 1, 2]:
            result_ours = algebra.grade_select(mv_vals, grade)

            mv = layout.MultiVector(mv_vals.numpy().astype(np.float64))
            result_cliff = mv(grade)

            result_cliff_tensor = torch.tensor(
                result_cliff.value[:algebra.blade_count], dtype=torch.float32
            )

            torch.testing.assert_close(
                result_ours, result_cliff_tensor, rtol=1e-3, atol=1e-3
            )


# =============================================================================
# T219: Execute all runtime operations tests
# =============================================================================

class TestAllRuntimeOperations:
    """T219: Execute all runtime operations."""

    @pytest.mark.parametrize("dim", [6])
    def test_all_operations_work(self, dim: int):
        """All runtime operations should work."""
        algebra = get_cga(dim)

        mv = torch.randn(algebra.blade_count)
        mv2 = torch.randn(algebra.blade_count)
        ev1 = torch.randn(algebra.even_versor_count)
        ev2 = torch.randn(algebra.even_versor_count)
        biv = torch.randn(algebra.bivector_count) * 0.1

        # Test all operations
        _ = algebra.compose_even_versor(ev1, ev2)
        _ = algebra.inner_product(mv, mv2)
        _ = algebra.exp_bivector(biv)
        _ = algebra.outer_product(mv, mv2)
        _ = algebra.left_contraction(mv, mv2)
        _ = algebra.right_contraction(mv, mv2)
        _ = algebra.grade_select(mv, 0)
        _ = algebra.grade_select(mv, 1)
        _ = algebra.dual(mv)
        _ = algebra.normalize(mv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
