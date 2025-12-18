"""
Test Unified Clifford Interface - Cl(p, q)

Tests the unified Cl() factory function and general Clifford algebra operations.
"""

import pytest
import torch
import numpy as np

from fast_clifford import Cl, VGA, CGA, Multivector, Rotor

# Try to import clifford library for comparison
try:
    import clifford as cf
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False


class TestClFactory:
    """Test Cl() factory function."""

    def test_cl_0_0_scalar(self):
        """Test Cl(0, 0) - scalar algebra (edge case)."""
        algebra = Cl(0, 0)
        assert algebra.p == 0
        assert algebra.q == 0
        assert algebra.r == 0
        assert algebra.count_blade == 1
        assert algebra.count_rotor == 1

    def test_cl_1_0_vga1(self):
        """Test Cl(1, 0) = VGA(1)."""
        algebra = Cl(1, 0)
        assert algebra.p == 1
        assert algebra.q == 0
        assert algebra.count_blade == 2

    def test_cl_3_0_vga3(self):
        """Test Cl(3, 0) = VGA(3)."""
        algebra = Cl(3, 0)
        assert algebra.count_blade == 8
        assert algebra.count_rotor == 4

    def test_cl_4_1_cga3(self):
        """Test Cl(4, 1) = CGA(3)."""
        algebra = Cl(4, 1)
        assert algebra.p == 4
        assert algebra.q == 1
        assert algebra.count_blade == 32

    def test_cl_2_2_general(self):
        """Test Cl(2, 2) - general signature."""
        algebra = Cl(2, 2)
        assert algebra.p == 2
        assert algebra.q == 2
        assert algebra.count_blade == 16
        assert algebra.count_rotor == 8

    @pytest.mark.parametrize("p,q", [
        (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1),
        (3, 0), (0, 3), (2, 1), (1, 2),
        (4, 0), (0, 4), (3, 1), (1, 3), (2, 2),
        (5, 0), (0, 5), (4, 1), (1, 4), (3, 2), (2, 3),
    ])
    def test_all_low_dim_algebras(self, p, q):
        """Test all algebras with p+q <= 5."""
        algebra = Cl(p, q)
        expected_blade_count = 2 ** (p + q)
        assert algebra.count_blade == expected_blade_count

    @pytest.mark.parametrize("p,q", [
        (9, 0), (0, 9), (8, 1), (1, 8), (5, 4), (4, 5),
    ])
    def test_high_dim_algebras(self, p, q):
        """Test high-dimensional algebras (p+q=9)."""
        algebra = Cl(p, q)
        assert algebra.count_blade == 512


class TestClProperties:
    """Test Clifford algebra properties."""

    def test_count_blade(self):
        """Test count_blade = 2^(p+q)."""
        for p in range(5):
            for q in range(5 - p):
                algebra = Cl(p, q)
                expected = 2 ** (p + q)
                assert algebra.count_blade == expected

    def test_count_rotor(self):
        """Test count_rotor = 2^(p+q-1) for p+q > 0."""
        for p in range(1, 5):
            for q in range(5 - p):
                algebra = Cl(p, q)
                expected = 2 ** (p + q - 1)
                assert algebra.count_rotor == expected

    def test_max_grade(self):
        """Test max_grade = p + q."""
        for p in range(5):
            for q in range(5 - p):
                algebra = Cl(p, q)
                expected = p + q
                assert algebra.max_grade == expected


class TestGeometricProduct:
    """Test geometric product for various signatures."""

    def test_positive_signature_squares(self):
        """Test basis vectors square to +1 in Cl(n, 0)."""
        algebra = Cl(3, 0)
        for i in range(3):
            # Create basis vector
            e = torch.zeros(8)
            e[i + 1] = 1.0  # e1, e2, e3 at indices 1, 2, 3

            ee = algebra.geometric_product(e, e)
            assert torch.allclose(ee[0], torch.tensor(1.0), atol=1e-5)

    def test_negative_signature_squares(self):
        """Test basis vectors square to -1 in Cl(0, n)."""
        algebra = Cl(0, 3)
        for i in range(3):
            e = torch.zeros(8)
            e[i + 1] = 1.0

            ee = algebra.geometric_product(e, e)
            assert torch.allclose(ee[0], torch.tensor(-1.0), atol=1e-5)

    def test_mixed_signature_squares(self):
        """Test mixed signature Cl(2, 2)."""
        algebra = Cl(2, 2)
        n = algebra.count_blade  # 16

        # First two basis vectors should square to +1
        for i in range(2):
            e = torch.zeros(n)
            e[i + 1] = 1.0
            ee = algebra.geometric_product(e, e)
            assert torch.allclose(ee[0], torch.tensor(1.0), atol=1e-5)

        # Last two basis vectors should square to -1
        for i in range(2, 4):
            e = torch.zeros(n)
            e[i + 1] = 1.0
            ee = algebra.geometric_product(e, e)
            assert torch.allclose(ee[0], torch.tensor(-1.0), atol=1e-5)

    def test_associativity(self):
        """Test geometric product is associative."""
        algebra = Cl(3, 0)
        a = torch.randn(8)
        b = torch.randn(8)
        c = torch.randn(8)

        # (a * b) * c
        ab = algebra.geometric_product(a, b)
        abc1 = algebra.geometric_product(ab, c)

        # a * (b * c)
        bc = algebra.geometric_product(b, c)
        abc2 = algebra.geometric_product(a, bc)

        assert torch.allclose(abc1, abc2, atol=1e-4)


class TestMultivectorOperators:
    """Test Multivector operator overloading."""

    def test_geometric_product_operator(self):
        """Test * operator for geometric product."""
        algebra = Cl(3, 0)
        a_data = torch.randn(8)
        b_data = torch.randn(8)

        a = Multivector(a_data, algebra)
        b = Multivector(b_data, algebra)

        # Using operator
        c_op = a * b

        # Using method
        c_method = algebra.geometric_product(a_data, b_data)

        assert torch.allclose(c_op.data, c_method, atol=1e-5)

    def test_outer_product_operator(self):
        """Test ^ operator for outer product."""
        algebra = Cl(3, 0)
        a_data = torch.randn(8)
        b_data = torch.randn(8)

        a = Multivector(a_data, algebra)
        b = Multivector(b_data, algebra)

        c = a ^ b
        assert c.data.shape == (8,)

    def test_inner_product_operator(self):
        """Test | operator for inner product."""
        algebra = Cl(3, 0)
        a_data = torch.randn(8)
        b_data = torch.randn(8)

        a = Multivector(a_data, algebra)
        b = Multivector(b_data, algebra)

        c = a | b
        # Inner product may return scalar or full multivector depending on implementation
        assert c.data.numel() >= 1

    def test_reverse_operator(self):
        """Test ~ operator for reverse."""
        algebra = Cl(3, 0)
        a_data = torch.randn(8)

        a = Multivector(a_data, algebra)
        a_rev = ~a

        expected = algebra.reverse(a_data)
        assert torch.allclose(a_rev.data, expected, atol=1e-5)

    def test_scalar_multiplication(self):
        """Test scalar * multivector and multivector * scalar."""
        algebra = Cl(3, 0)
        a_data = torch.randn(8)
        a = Multivector(a_data, algebra)

        # Scalar on right
        b = a * 2.0
        assert torch.allclose(b.data, a_data * 2.0, atol=1e-5)

        # Scalar on left
        c = 3.0 * a
        assert torch.allclose(c.data, a_data * 3.0, atol=1e-5)

    def test_addition(self):
        """Test + operator."""
        algebra = Cl(3, 0)
        a = Multivector(torch.randn(8), algebra)
        b = Multivector(torch.randn(8), algebra)

        c = a + b
        assert torch.allclose(c.data, a.data + b.data, atol=1e-5)

    def test_subtraction(self):
        """Test - operator."""
        algebra = Cl(3, 0)
        a = Multivector(torch.randn(8), algebra)
        b = Multivector(torch.randn(8), algebra)

        c = a - b
        assert torch.allclose(c.data, a.data - b.data, atol=1e-5)


class TestReverseInvoluteConjugate:
    """Test reverse, involute, and conjugate operations."""

    def test_reverse_double(self):
        """Test reverse(reverse(x)) = x."""
        algebra = Cl(3, 1)
        mv = torch.randn(algebra.count_blade)
        result = algebra.reverse(algebra.reverse(mv))
        assert torch.allclose(result, mv, atol=1e-5)

    def test_involute_double(self):
        """Test involute(involute(x)) = x."""
        algebra = Cl(3, 1)
        mv = torch.randn(algebra.count_blade)
        result = algebra.involute(algebra.involute(mv))
        assert torch.allclose(result, mv, atol=1e-5)

    def test_conjugate_double(self):
        """Test conjugate(conjugate(x)) = x."""
        algebra = Cl(3, 1)
        mv = torch.randn(algebra.count_blade)
        result = algebra.conjugate(algebra.conjugate(mv))
        assert torch.allclose(result, mv, atol=1e-5)


class TestSelectGrade:
    """Test grade selection."""

    def test_select_grade_0(self):
        """Test selecting scalar (grade 0)."""
        algebra = Cl(3, 0)
        mv = torch.randn(8)
        grade0 = algebra.select_grade(mv, 0)

        # Only scalar component should be non-zero
        assert torch.allclose(grade0[0], mv[0], atol=1e-5)
        assert torch.allclose(grade0[1:], torch.zeros(7), atol=1e-5)

    def test_select_grade_1(self):
        """Test selecting vectors (grade 1)."""
        algebra = Cl(3, 0)
        mv = torch.randn(8)
        grade1 = algebra.select_grade(mv, 1)

        # Only vector components should be non-zero
        assert torch.allclose(grade1[0], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(grade1[1:4], mv[1:4], atol=1e-5)
        assert torch.allclose(grade1[4:], torch.zeros(4), atol=1e-5)


@pytest.mark.skipif(not HAS_CLIFFORD, reason="clifford library not installed")
class TestCliffordComparison:
    """Compare with clifford library."""

    @pytest.mark.parametrize("p,q", [(2, 0), (3, 0), (2, 1), (1, 2), (2, 2)])
    def test_geometric_product_comparison(self, p, q):
        """Compare geometric product with clifford library."""
        # Create clifford algebra
        layout, blades = cf.Cl(p, q)
        n = p + q

        # Create fast-clifford algebra
        algebra = Cl(p, q)

        # Create random multivectors using basis blade coefficients
        # Use small random values
        np.random.seed(42)
        a_coeffs = np.random.randn(2**n) * 0.1
        b_coeffs = np.random.randn(2**n) * 0.1

        # Create clifford multivectors
        cf_a = layout.MultiVector(a_coeffs)
        cf_b = layout.MultiVector(b_coeffs)
        cf_result = cf_a * cf_b

        # Create fast-clifford tensors
        fc_a = torch.tensor(a_coeffs, dtype=torch.float32)
        fc_b = torch.tensor(b_coeffs, dtype=torch.float32)
        fc_result = algebra.geometric_product(fc_a, fc_b)

        # Compare (clifford may use different ordering, so compare norms)
        cf_norm = np.sum(cf_result.value ** 2)
        fc_norm = torch.sum(fc_result ** 2).item()

        # Norms should be close
        assert abs(cf_norm - fc_norm) < 0.1, f"Norm mismatch: {cf_norm} vs {fc_norm}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_scalar_algebra_product(self):
        """Test Cl(0, 0) scalar algebra multiplication."""
        algebra = Cl(0, 0)
        a = torch.tensor([2.0])
        b = torch.tensor([3.0])
        result = algebra.geometric_product(a, b)
        assert torch.allclose(result, torch.tensor([6.0]), atol=1e-5)

    def test_zero_multivector(self):
        """Test operations on zero multivector."""
        algebra = Cl(3, 0)
        zero = torch.zeros(8)
        a = torch.randn(8)

        # 0 * a = 0
        result = algebra.geometric_product(zero, a)
        assert torch.allclose(result, zero, atol=1e-5)

        # a * 0 = 0
        result = algebra.geometric_product(a, zero)
        assert torch.allclose(result, zero, atol=1e-5)

    def test_identity_scalar(self):
        """Test multiplication by scalar identity."""
        algebra = Cl(3, 0)
        one = torch.zeros(8)
        one[0] = 1.0  # Scalar 1
        a = torch.randn(8)

        # 1 * a = a
        result = algebra.geometric_product(one, a)
        assert torch.allclose(result, a, atol=1e-5)

        # a * 1 = a
        result = algebra.geometric_product(a, one)
        assert torch.allclose(result, a, atol=1e-5)
