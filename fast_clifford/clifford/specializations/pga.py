"""
PGA (Projective Geometric Algebra) Specialization via CGA Embedding

PGA(n) = Cl(n, 0, 1) - Projective geometric algebra with one degenerate dimension

This specialization implements PGA operations by embedding into CGA:
- PGA(n) = Cl(n, 0, 1) embeds into CGA(n) = Cl(n+1, 1)
- The degenerate basis e0 (e0² = 0) maps to e_inf in CGA

This approach leverages the hardcoded CGA operations for performance.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch import Tensor

from .cga import CGAWrapper

if TYPE_CHECKING:
    from ..base import CliffordAlgebraBase


class PGAEmbedding:
    """
    PGA(n) = Cl(n, 0, 1) implementation via CGA embedding.

    PGA has a degenerate basis vector e0 with e0² = 0.
    We embed into CGA where e_inf² = 0, mapping e0 -> e_inf.

    Attributes:
        cga: The underlying CGA(n) algebra
        dim_euclidean: Euclidean dimension n
        pga_blade_count: Number of blades in PGA(n)

    Example:
        >>> pga = PGA(3)  # Creates PGA(3) via CGA(3)
        >>> point = pga.point(torch.tensor([1., 2., 3.]))
        >>> plane = pga.plane(torch.tensor([0., 0., 1., -5.]))
    """

    def __init__(self, n: int):
        """
        Initialize PGA(n) via CGA(n) embedding.

        Args:
            n: Euclidean dimension
        """
        from .. import CGA

        self._n = n
        self._cga = CGA(n)

        # PGA has n+1 basis vectors: e1, ..., en, e0
        # Total blades: 2^(n+1)
        self._pga_blade_count = 2 ** (n + 1)
        self._pga_rotor_count = 2 ** n

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def p(self) -> int:
        """Positive signature dimension."""
        return self._n

    @property
    def q(self) -> int:
        """Negative signature dimension (always 0)."""
        return 0

    @property
    def r(self) -> int:
        """Degenerate dimension (always 1)."""
        return 1

    @property
    def dim_euclidean(self) -> int:
        """Euclidean dimension n."""
        return self._n

    @property
    def count_blade(self) -> int:
        """Total blade count in PGA."""
        return self._pga_blade_count

    @property
    def count_rotor(self) -> int:
        """Rotor component count in PGA."""
        return self._pga_rotor_count

    @property
    def max_grade(self) -> int:
        """Maximum grade."""
        return self._n + 1

    @property
    def algebra_type(self) -> str:
        """Algebra type (always 'pga')."""
        return "pga"

    # =========================================================================
    # PGA Primitives
    # =========================================================================

    def point(self, x: Tensor) -> Tensor:
        """
        Create a PGA point from Euclidean coordinates.

        In PGA, a point at (x, y, z) is represented as:
            P = x*e1 + y*e2 + z*e3 + e0

        Args:
            x: Euclidean point, shape (..., n)

        Returns:
            PGA point (grade-1), shape (..., count_blade)
        """
        batch_shape = x.shape[:-1]
        n = self._n

        result = torch.zeros(*batch_shape, self.count_blade, dtype=x.dtype, device=x.device)

        # Euclidean components (e1, ..., en at indices 1..n)
        result[..., 1:n+1] = x

        # e0 component (at index n+1 in our ordering)
        result[..., n+1] = 1.0

        return result

    def direction(self, d: Tensor) -> Tensor:
        """
        Create a PGA ideal point (direction/point at infinity).

        In PGA, a direction is a point with e0 = 0:
            D = dx*e1 + dy*e2 + dz*e3

        Args:
            d: Direction vector, shape (..., n)

        Returns:
            PGA ideal point (grade-1), shape (..., count_blade)
        """
        batch_shape = d.shape[:-1]
        n = self._n

        result = torch.zeros(*batch_shape, self.count_blade, dtype=d.dtype, device=d.device)
        result[..., 1:n+1] = d

        return result

    def plane(self, coeffs: Tensor) -> Tensor:
        """
        Create a PGA plane from coefficients.

        In 3D PGA, a plane ax + by + cz + d = 0 is:
            P = a*e1 + b*e2 + c*e3 + d*e0

        Args:
            coeffs: Plane coefficients (a, b, c, d), shape (..., n+1)

        Returns:
            PGA plane (grade-1), shape (..., count_blade)
        """
        batch_shape = coeffs.shape[:-1]
        n = self._n

        result = torch.zeros(*batch_shape, self.count_blade, dtype=coeffs.dtype, device=coeffs.device)

        # Normal components (e1, ..., en)
        result[..., 1:n+1] = coeffs[..., :n]

        # Distance component (e0)
        result[..., n+1] = coeffs[..., n]

        return result

    def line_from_points(self, p1: Tensor, p2: Tensor) -> Tensor:
        """
        Create a PGA line from two points.

        Line = P1 ^ P2 (outer product of two points)

        Args:
            p1: First point, shape (..., n)
            p2: Second point, shape (..., n)

        Returns:
            PGA line (grade-2), shape (..., count_blade)
        """
        point1 = self.point(p1)
        point2 = self.point(p2)
        return self.outer(point1, point2)

    # =========================================================================
    # Operations via CGA
    # =========================================================================

    def _embed_to_cga(self, pga_mv: Tensor) -> Tensor:
        """
        Embed PGA multivector into CGA.

        Maps: e0 (PGA) -> e_inf (CGA)

        Args:
            pga_mv: PGA multivector, shape (..., pga_blade_count)

        Returns:
            CGA multivector, shape (..., cga_blade_count)
        """
        batch_shape = pga_mv.shape[:-1]
        n = self._n

        # Create CGA multivector (larger)
        cga_mv = torch.zeros(*batch_shape, self._cga.count_blade,
                            dtype=pga_mv.dtype, device=pga_mv.device)

        # Copy scalar (grade 0)
        cga_mv[..., 0] = pga_mv[..., 0]

        # Copy Euclidean vectors (e1, ..., en)
        cga_mv[..., 1:n+1] = pga_mv[..., 1:n+1]

        # Map e0 -> e_inf = e+ + e-
        # In CGA, e+ is at index n+1, e- is at index n+2
        e0_coeff = pga_mv[..., n+1]
        cga_mv[..., n+1] = cga_mv[..., n+1] + e0_coeff  # e+
        cga_mv[..., n+2] = cga_mv[..., n+2] + e0_coeff  # e-

        # Higher grades need more complex mapping (simplified for now)
        # TODO: Full grade mapping for complete embedding

        return cga_mv

    def _project_from_cga(self, cga_mv: Tensor) -> Tensor:
        """
        Project CGA multivector back to PGA.

        Maps: e_inf (CGA) -> e0 (PGA)

        Args:
            cga_mv: CGA multivector, shape (..., cga_blade_count) or (..., 1) for scalar

        Returns:
            PGA multivector, shape (..., pga_blade_count)
        """
        n = self._n

        # Handle case where CGA returns just scalar (e.g., from inner product)
        if cga_mv.shape[-1] == 1:
            # Just scalar result - expand to full PGA multivector
            batch_shape = cga_mv.shape[:-1]
            pga_mv = torch.zeros(*batch_shape, self.count_blade,
                                dtype=cga_mv.dtype, device=cga_mv.device)
            pga_mv[..., 0] = cga_mv[..., 0]
            return pga_mv

        batch_shape = cga_mv.shape[:-1]

        pga_mv = torch.zeros(*batch_shape, self.count_blade,
                            dtype=cga_mv.dtype, device=cga_mv.device)

        # Copy scalar
        pga_mv[..., 0] = cga_mv[..., 0]

        # Copy Euclidean vectors (if enough components)
        if cga_mv.shape[-1] > n:
            pga_mv[..., 1:n+1] = cga_mv[..., 1:n+1]

        # Map e_inf = e+ + e- -> e0
        if cga_mv.shape[-1] > n + 2:
            pga_mv[..., n+1] = cga_mv[..., n+1] + cga_mv[..., n+2]

        return pga_mv

    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Geometric product via CGA embedding.

        Args:
            a: Left operand, shape (..., count_blade)
            b: Right operand, shape (..., count_blade)

        Returns:
            Product, shape (..., count_blade)
        """
        cga_a = self._embed_to_cga(a)
        cga_b = self._embed_to_cga(b)
        cga_result = self._cga.geometric_product(cga_a, cga_b)
        return self._project_from_cga(cga_result)

    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        """Outer product via CGA embedding."""
        cga_a = self._embed_to_cga(a)
        cga_b = self._embed_to_cga(b)
        cga_result = self._cga.outer(cga_a, cga_b)
        return self._project_from_cga(cga_result)

    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        """Inner product via CGA embedding."""
        cga_a = self._embed_to_cga(a)
        cga_b = self._embed_to_cga(b)
        cga_result = self._cga.inner(cga_a, cga_b)
        return self._project_from_cga(cga_result)

    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        """Regressive product (meet) via CGA embedding."""
        cga_a = self._embed_to_cga(a)
        cga_b = self._embed_to_cga(b)
        cga_result = self._cga.regressive(cga_a, cga_b)
        return self._project_from_cga(cga_result)

    def sandwich(self, v: Tensor, x: Tensor) -> Tensor:
        """Sandwich product via CGA embedding."""
        cga_v = self._embed_to_cga(v)
        cga_x = self._embed_to_cga(x)
        cga_result = self._cga.sandwich(cga_v, cga_x)
        return self._project_from_cga(cga_result)

    def reverse(self, mv: Tensor) -> Tensor:
        """Reverse via CGA embedding."""
        cga_mv = self._embed_to_cga(mv)
        cga_result = self._cga.reverse(cga_mv)
        return self._project_from_cga(cga_result)

    def dual(self, mv: Tensor) -> Tensor:
        """Dual via CGA embedding."""
        cga_mv = self._embed_to_cga(mv)
        cga_result = self._cga.dual(cga_mv)
        return self._project_from_cga(cga_result)

    def normalize(self, mv: Tensor) -> Tensor:
        """Normalize via CGA embedding."""
        cga_mv = self._embed_to_cga(mv)
        cga_result = self._cga.normalize(cga_mv)
        return self._project_from_cga(cga_result)
