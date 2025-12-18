"""
Test rotor exponential and logarithm operations.

Tests exp_bivector, log_rotor, and slerp_rotor functions.
"""

import pytest
import torch
import math

from fast_clifford import VGA, CGA, Cl


class TestExpBivector:
    """Test bivector exponential function."""

    def test_exp_zero_bivector_vga3(self):
        """exp(0) should be identity rotor [1, 0, 0, ...]."""
        vga = VGA(3)
        B = torch.zeros(vga.count_bivector)
        R = vga.exp_bivector(B)

        assert R.shape == (vga.count_rotor,)
        assert torch.allclose(R[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(R[1:], torch.zeros(vga.count_rotor - 1), atol=1e-5)

    def test_exp_small_angle_vga3(self):
        """exp of small bivector should be close to 1 + B."""
        vga = VGA(3)
        angle = 0.01
        B = torch.zeros(vga.count_bivector)
        B[0] = angle  # e12 component

        R = vga.exp_bivector(B)

        # For small angles: exp(B) ≈ 1 + B
        assert torch.allclose(R[0], torch.tensor(1.0), atol=1e-3)
        assert torch.allclose(R[1], torch.tensor(angle), atol=1e-3)

    def test_exp_pi_half_rotation_vga3(self):
        """exp(π/4 * e12) should give 45° rotation."""
        vga = VGA(3)
        angle = math.pi / 4
        B = torch.zeros(vga.count_bivector)
        B[0] = angle  # e12 component

        R = vga.exp_bivector(B)

        # exp(θ * e12) = cos(θ) + sin(θ) * e12
        expected_scalar = math.cos(angle)
        expected_e12 = math.sin(angle)

        assert torch.allclose(R[0], torch.tensor(expected_scalar), atol=1e-5)
        assert torch.allclose(R[1], torch.tensor(expected_e12), atol=1e-5)

    def test_exp_rotor_normalized_vga3(self):
        """exp(B) should produce a unit rotor."""
        vga = VGA(3)
        B = torch.randn(vga.count_bivector) * 0.5

        R = vga.exp_bivector(B)
        norm_sq = vga.norm_squared_rotor(R)

        assert torch.allclose(norm_sq, torch.tensor([[1.0]]), atol=1e-4)

    def test_exp_cga3(self):
        """Test exp_bivector in CGA(3)."""
        cga = CGA(3)
        B = torch.zeros(cga.count_bivector)
        B[0] = 0.5

        R = cga.exp_bivector(B)

        assert R.shape == (cga.count_rotor,)
        # Should be normalized
        norm_sq = cga.norm_squared_rotor(R)
        assert torch.allclose(norm_sq, torch.tensor([[1.0]]), atol=1e-4)

    def test_exp_batch(self):
        """Test batched exp_bivector."""
        vga = VGA(3)
        batch_size = 10
        B = torch.randn(batch_size, vga.count_bivector) * 0.3

        R = vga.exp_bivector(B)

        assert R.shape == (batch_size, vga.count_rotor)


class TestLogRotor:
    """Test rotor logarithm function."""

    def test_log_identity_rotor_vga3(self):
        """log(identity) should be zero bivector."""
        vga = VGA(3)
        R = torch.zeros(vga.count_rotor)
        R[0] = 1.0  # Identity rotor

        B = vga.log_rotor(R)

        assert B.shape == (vga.count_bivector,)
        assert torch.allclose(B, torch.zeros(vga.count_bivector), atol=1e-5)

    def test_log_small_rotation_vga3(self):
        """log of small rotation should recover the bivector."""
        vga = VGA(3)
        angle = 0.1
        B_original = torch.zeros(vga.count_bivector)
        B_original[0] = angle

        R = vga.exp_bivector(B_original)
        B_recovered = vga.log_rotor(R)

        assert torch.allclose(B_recovered, B_original, atol=1e-5)

    def test_log_larger_rotation_vga3(self):
        """log should work for larger rotations too."""
        vga = VGA(3)
        angle = math.pi / 3  # 60 degrees
        B_original = torch.zeros(vga.count_bivector)
        B_original[0] = angle

        R = vga.exp_bivector(B_original)
        B_recovered = vga.log_rotor(R)

        assert torch.allclose(B_recovered, B_original, atol=1e-4)

    def test_log_batch(self):
        """Test batched log_rotor."""
        vga = VGA(3)
        batch_size = 10
        B = torch.randn(batch_size, vga.count_bivector) * 0.3
        R = vga.exp_bivector(B)

        B_recovered = vga.log_rotor(R)

        assert B_recovered.shape == (batch_size, vga.count_bivector)
        assert torch.allclose(B_recovered, B, atol=1e-4)


class TestExpLogRoundtrip:
    """Test exp/log roundtrip consistency."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_roundtrip_vga(self, n):
        """exp(log(R)) = R and log(exp(B)) = B for VGA."""
        vga = VGA(n)
        B = torch.randn(vga.count_bivector) * 0.5

        R = vga.exp_bivector(B)
        B_recovered = vga.log_rotor(R)

        assert torch.allclose(B_recovered, B, atol=1e-4)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_roundtrip_cga_simple_bivector(self, n):
        """exp(log(R)) = R for simple bivectors in CGA.

        Note: CGA has mixed-signature bivectors (some square to +, some to -).
        The simple formula works best for single-plane rotations.
        """
        cga = CGA(n)
        # Use only the first bivector component (e1e2 plane, purely Euclidean)
        B = torch.zeros(cga.count_bivector)
        B[0] = 0.3  # Only in e1e2 plane

        R = cga.exp_bivector(B)
        B_recovered = cga.log_rotor(R)

        # For simple bivectors, should work well
        assert torch.allclose(B_recovered, B, atol=1e-4)


class TestSlerpRotor:
    """Test rotor spherical linear interpolation."""

    def test_slerp_t0_returns_r1(self):
        """slerp(r1, r2, 0) = r1."""
        vga = VGA(3)
        B1 = torch.zeros(vga.count_bivector)
        B1[0] = 0.2
        B2 = torch.zeros(vga.count_bivector)
        B2[0] = 0.8

        R1 = vga.exp_bivector(B1)
        R2 = vga.exp_bivector(B2)
        t = torch.tensor(0.0)

        R = vga.slerp_rotor(R1, R2, t)

        assert torch.allclose(R, R1, atol=1e-4)

    def test_slerp_t1_returns_r2(self):
        """slerp(r1, r2, 1) = r2."""
        vga = VGA(3)
        B1 = torch.zeros(vga.count_bivector)
        B1[0] = 0.2
        B2 = torch.zeros(vga.count_bivector)
        B2[0] = 0.8

        R1 = vga.exp_bivector(B1)
        R2 = vga.exp_bivector(B2)
        t = torch.tensor(1.0)

        R = vga.slerp_rotor(R1, R2, t)

        assert torch.allclose(R, R2, atol=1e-4)

    def test_slerp_t05_midpoint(self):
        """slerp(r1, r2, 0.5) should be midpoint rotation."""
        vga = VGA(3)
        B1 = torch.zeros(vga.count_bivector)
        B2 = torch.zeros(vga.count_bivector)
        B2[0] = 1.0  # 1 radian in e12 plane

        R1 = vga.exp_bivector(B1)
        R2 = vga.exp_bivector(B2)
        t = torch.tensor(0.5)

        R_mid = vga.slerp_rotor(R1, R2, t)

        # Expected: exp(0.5 * B2)
        B_expected = torch.zeros(vga.count_bivector)
        B_expected[0] = 0.5
        R_expected = vga.exp_bivector(B_expected)

        assert torch.allclose(R_mid, R_expected, atol=1e-4)

    def test_slerp_normalized_output(self):
        """slerp should produce normalized rotors."""
        vga = VGA(3)
        B1 = torch.randn(vga.count_bivector) * 0.3
        B2 = torch.randn(vga.count_bivector) * 0.5

        R1 = vga.exp_bivector(B1)
        R2 = vga.exp_bivector(B2)
        t = torch.tensor(0.5)

        R = vga.slerp_rotor(R1, R2, t)
        norm_sq = vga.norm_squared_rotor(R)

        assert torch.allclose(norm_sq, torch.tensor([[1.0]]), atol=1e-3)

    def test_slerp_cga3(self):
        """Test slerp in CGA(3)."""
        cga = CGA(3)
        B1 = torch.zeros(cga.count_bivector)
        B2 = torch.zeros(cga.count_bivector)
        B2[0] = 0.6

        R1 = cga.exp_bivector(B1)
        R2 = cga.exp_bivector(B2)
        t = torch.tensor(0.5)

        R_mid = cga.slerp_rotor(R1, R2, t)

        # Expected midpoint
        B_expected = torch.zeros(cga.count_bivector)
        B_expected[0] = 0.3
        R_expected = cga.exp_bivector(B_expected)

        assert torch.allclose(R_mid, R_expected, atol=1e-4)


class TestMultipleDimensions:
    """Test across multiple algebra dimensions."""

    @pytest.mark.parametrize("p,q", [(2, 0), (3, 0), (4, 0), (5, 0)])
    def test_exp_log_roundtrip_vga(self, p, q):
        """Test exp/log roundtrip for VGA signatures (pure Euclidean)."""
        alg = Cl(p, q)
        B = torch.randn(alg.count_bivector) * 0.3

        R = alg.exp_bivector(B)
        B_recovered = alg.log_rotor(R)

        assert torch.allclose(B_recovered, B, atol=1e-4)

    @pytest.mark.parametrize("p,q", [(2, 1), (3, 1), (4, 1)])
    def test_exp_log_roundtrip_cga_simple(self, p, q):
        """Test exp/log roundtrip for CGA with simple bivectors."""
        alg = Cl(p, q)
        # Only use first bivector component (Euclidean plane)
        B = torch.zeros(alg.count_bivector)
        B[0] = 0.3

        R = alg.exp_bivector(B)
        B_recovered = alg.log_rotor(R)

        assert torch.allclose(B_recovered, B, atol=1e-4)

    @pytest.mark.parametrize("p,q", [(2, 0), (3, 0)])
    def test_slerp_midpoint_vga(self, p, q):
        """Test slerp midpoint for VGA signatures."""
        alg = Cl(p, q)
        B1 = torch.zeros(alg.count_bivector)
        B2 = torch.randn(alg.count_bivector) * 0.4

        R1 = alg.exp_bivector(B1)
        R2 = alg.exp_bivector(B2)

        R_mid = alg.slerp_rotor(R1, R2, torch.tensor(0.5))
        R_expected = alg.exp_bivector(B2 * 0.5)

        assert torch.allclose(R_mid, R_expected, atol=1e-3)

    @pytest.mark.parametrize("p,q", [(2, 1), (3, 1)])
    def test_slerp_midpoint_cga_simple(self, p, q):
        """Test slerp midpoint for CGA with simple bivectors."""
        alg = Cl(p, q)
        B1 = torch.zeros(alg.count_bivector)
        B2 = torch.zeros(alg.count_bivector)
        B2[0] = 0.4  # Only first component

        R1 = alg.exp_bivector(B1)
        R2 = alg.exp_bivector(B2)

        R_mid = alg.slerp_rotor(R1, R2, torch.tensor(0.5))
        R_expected = alg.exp_bivector(B2 * 0.5)

        assert torch.allclose(R_mid, R_expected, atol=1e-3)
