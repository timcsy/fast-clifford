"""
Test PGA (Projective Geometric Algebra) - Cl(n, 0, 1)

Tests PGA specialization via CGA embedding.
"""

import pytest
import torch

from fast_clifford import PGA


class TestPGABasic:
    """Basic PGA functionality tests."""

    def test_pga2_creation(self):
        """Test PGA(2) creation."""
        pga = PGA(2)
        assert pga.p == 2
        assert pga.q == 0
        assert pga.r == 1
        assert pga.count_blade == 8  # 2^(2+1)
        assert pga.count_rotor == 4  # 2^2
        assert pga.dim_euclidean == 2
        assert pga.algebra_type == "pga"

    def test_pga3_creation(self):
        """Test PGA(3) creation."""
        pga = PGA(3)
        assert pga.p == 3
        assert pga.q == 0
        assert pga.r == 1
        assert pga.count_blade == 16  # 2^(3+1)
        assert pga.count_rotor == 8   # 2^3
        assert pga.dim_euclidean == 3
        assert pga.algebra_type == "pga"

    def test_pga4_creation(self):
        """Test PGA(4) creation."""
        pga = PGA(4)
        assert pga.count_blade == 32  # 2^(4+1)
        assert pga.count_rotor == 16  # 2^4


class TestPGAPrimitives:
    """Test PGA geometric primitives."""

    def test_point_creation(self):
        """Test PGA point creation."""
        pga = PGA(3)
        x = torch.tensor([1., 2., 3.])
        point = pga.point(x)

        assert point.shape == (16,)
        # Point: x*e1 + y*e2 + z*e3 + e0
        assert torch.allclose(point[1], torch.tensor(1.0))  # e1
        assert torch.allclose(point[2], torch.tensor(2.0))  # e2
        assert torch.allclose(point[3], torch.tensor(3.0))  # e3
        assert torch.allclose(point[4], torch.tensor(1.0))  # e0

    def test_direction_creation(self):
        """Test PGA direction (ideal point) creation."""
        pga = PGA(3)
        d = torch.tensor([0., 0., 1.])
        direction = pga.direction(d)

        assert direction.shape == (16,)
        # Direction: dx*e1 + dy*e2 + dz*e3 (no e0)
        assert torch.allclose(direction[3], torch.tensor(1.0))  # e3
        assert torch.allclose(direction[4], torch.tensor(0.0))  # e0 = 0

    def test_plane_creation(self):
        """Test PGA plane creation."""
        pga = PGA(3)
        # Plane z = 5, i.e., z - 5 = 0, coeffs [0, 0, 1, -5]
        plane = pga.plane(torch.tensor([0., 0., 1., -5.]))

        assert plane.shape == (16,)
        assert torch.allclose(plane[3], torch.tensor(1.0))   # normal z
        assert torch.allclose(plane[4], torch.tensor(-5.0))  # distance

    def test_point_batch(self):
        """Test batch point creation."""
        pga = PGA(3)
        batch_size = 10
        x = torch.randn(batch_size, 3)
        points = pga.point(x)

        assert points.shape == (batch_size, 16)


class TestPGAOperations:
    """Test PGA operations."""

    def test_line_from_points(self):
        """Test creating a line from two points."""
        pga = PGA(3)
        p1 = torch.tensor([0., 0., 0.])
        p2 = torch.tensor([1., 0., 0.])

        line = pga.line_from_points(p1, p2)
        assert line.shape == (16,)
        # Line should be a bivector (grade 2)

    def test_reverse(self):
        """Test reverse operation."""
        pga = PGA(3)
        point = pga.point(torch.tensor([1., 2., 3.]))
        point_rev = pga.reverse(point)

        # Reverse of a vector (grade 1) is unchanged
        assert point_rev.shape == (16,)

    def test_outer_product(self):
        """Test outer product."""
        pga = PGA(3)
        a = pga.point(torch.tensor([1., 0., 0.]))
        b = pga.point(torch.tensor([0., 1., 0.]))

        ab = pga.outer(a, b)
        assert ab.shape == (16,)

    def test_inner_product(self):
        """Test inner product."""
        pga = PGA(3)
        a = pga.point(torch.tensor([1., 0., 0.]))
        b = pga.point(torch.tensor([0., 1., 0.]))

        ab = pga.inner(a, b)
        assert ab.shape == (16,)


class TestPGA2D:
    """Test 2D PGA specific operations."""

    def test_pga2_point(self):
        """Test 2D PGA point."""
        pga = PGA(2)
        point = pga.point(torch.tensor([3., 4.]))

        assert point.shape == (8,)
        assert torch.allclose(point[1], torch.tensor(3.0))  # e1
        assert torch.allclose(point[2], torch.tensor(4.0))  # e2
        assert torch.allclose(point[3], torch.tensor(1.0))  # e0

    def test_pga2_line(self):
        """Test 2D PGA line from two points."""
        pga = PGA(2)
        p1 = torch.tensor([0., 0.])
        p2 = torch.tensor([1., 1.])

        line = pga.line_from_points(p1, p2)
        assert line.shape == (8,)
