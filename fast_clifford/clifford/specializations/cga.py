"""
CGA (Conformal Geometric Algebra) Specialization

CGA(n) = Cl(n+1, 1) - Conformal geometric algebra with signature (+1,...,+1,-1)

This specialization provides:
- encode: Embed Euclidean point as CGA null vector (UPGC representation)
- decode: Extract Euclidean coordinates from CGA point
- dim_euclidean: Euclidean dimension property
- count_point: Number of CGA point components

Null basis convention (Dorst):
- e_o = (e_- - e_+) / 2  (origin)
- e_inf = e_- + e_+      (point at infinity)

Where:
- e_+ is the positive extra dimension (e_+² = +1)
- e_- is the negative extra dimension (e_-² = -1)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..base import CliffordAlgebraBase


class CGAWrapper:
    """
    CGA(n) = Cl(n+1, 1) specialization wrapper.

    Wraps a base Clifford algebra with CGA-specific operations for
    conformal geometric transformations.

    Attributes:
        base_algebra: The underlying Cl(n+1, 1) algebra
        dim_euclidean: Euclidean dimension (n)
        count_point: Number of CGA point components (n+2)

    Example:
        >>> cga = CGA(3)  # Creates CGA(3) = Cl(4, 1)
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> point = cga.encode(x)  # Embed as CGA null vector
        >>> x_back = cga.decode(point)  # Extract Euclidean coords
    """

    def __init__(self, base_algebra: "CliffordAlgebraBase", euclidean_dim: int):
        """
        Initialize CGA wrapper.

        Args:
            base_algebra: A Clifford algebra Cl(n+1, 1)
            euclidean_dim: The Euclidean dimension n

        Raises:
            ValueError: If algebra signature doesn't match CGA(n)
        """
        expected_p = euclidean_dim + 1
        expected_q = 1

        if base_algebra.p != expected_p or base_algebra.q != expected_q:
            raise ValueError(
                f"CGAWrapper for CGA({euclidean_dim}) requires Cl({expected_p}, {expected_q}), "
                f"got Cl({base_algebra.p}, {base_algebra.q}, {base_algebra.r})"
            )

        self._base = base_algebra
        self._euclidean_dim = euclidean_dim

        # CGA point uses grade-1 components: n Euclidean + e+ + e-
        self._count_point = euclidean_dim + 2

        # Indices in the multivector for grade-1 components
        # For CGA(n) = Cl(n+1, 1), grade-1 has n+2 components
        # Order: e1, e2, ..., en, e(n+1), e(n+2)
        # Where e(n+1) = e+ and e(n+2) = e-
        self._point_start_idx = 1  # Grade-1 starts at index 1

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def p(self) -> int:
        """Positive dimension."""
        return self._base.p

    @property
    def q(self) -> int:
        """Negative dimension."""
        return self._base.q

    @property
    def r(self) -> int:
        """Degenerate dimension (always 0 for CGA)."""
        return 0

    @property
    def dim_euclidean(self) -> int:
        """Euclidean dimension n."""
        return self._euclidean_dim

    @property
    def count_point(self) -> int:
        """Number of CGA point components (n+2)."""
        return self._count_point

    @property
    def count_blade(self) -> int:
        """Total blade count."""
        return self._base.count_blade

    @property
    def count_rotor(self) -> int:
        """Rotor component count."""
        return self._base.count_rotor

    @property
    def count_bivector(self) -> int:
        """Bivector component count."""
        return self._base.count_bivector

    @property
    def max_grade(self) -> int:
        """Maximum grade."""
        return self._base.max_grade

    @property
    def algebra_type(self) -> str:
        """Algebra type (always 'cga')."""
        return "cga"

    # =========================================================================
    # CGA-Specific Operations
    # =========================================================================

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode Euclidean point as CGA null vector (UPGC representation).

        The conformal embedding is:
            X = x + (1/2)|x|² e_inf + e_o

        Using Dorst convention:
            e_o = (e_- - e_+) / 2
            e_inf = e_- + e_+

        This gives:
            e+ coefficient = -1/2 + (1/2)|x|²
            e- coefficient = 1/2 + (1/2)|x|²

        Args:
            x: Euclidean point, shape (..., dim_euclidean)

        Returns:
            CGA point (grade-1 multivector), shape (..., count_blade)
        """
        batch_shape = x.shape[:-1]
        n = self.dim_euclidean

        # Compute |x|² / 2
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        half_norm_sq = 0.5 * x_norm_sq

        # Create full multivector with zeros
        result = torch.zeros(*batch_shape, self.count_blade, dtype=x.dtype, device=x.device)

        # Place Euclidean components (e1, e2, ..., en)
        result[..., 1:n+1] = x

        # e+ coefficient: -0.5 + half_norm_sq (from e_o and e_inf)
        result[..., n+1] = (-0.5 + half_norm_sq).squeeze(-1)

        # e- coefficient: 0.5 + half_norm_sq (from e_o and e_inf)
        result[..., n+2] = (0.5 + half_norm_sq).squeeze(-1)

        return result

    def decode(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to Euclidean coordinates.

        For a properly normalized CGA point X = x + (1/2)|x|² e_inf + e_o,
        the Euclidean coordinates are simply the e1, e2, ..., en components.

        For a general CGA point, we normalize by dividing by (-e_inf · X).

        Weight calculation with Dorst convention:
        - e_inf = e_+ + e_-
        - e_inf · X = (e+ + e-) · (... + α e+ + β e-)
                    = α e+² + β e-² = α(+1) + β(-1) = α - β
        - weight = -e_inf · X = β - α = (e- coeff) - (e+ coeff)

        Args:
            point: CGA point (full multivector), shape (..., count_blade)

        Returns:
            Euclidean point, shape (..., dim_euclidean)
        """
        n = self.dim_euclidean

        # Extract Euclidean components directly
        euclidean = point[..., 1:n+1]

        # Weight = -e_inf · X = e- coeff - e+ coeff
        # For UPGC (unit point) this equals 1
        e_plus = point[..., n+1:n+2]
        e_minus = point[..., n+2:n+3]
        weight = e_minus - e_plus

        # Avoid division by zero
        weight = torch.where(
            torch.abs(weight) < 1e-10,
            torch.ones_like(weight),
            weight
        )

        return euclidean / weight

    def encode_point_sparse(self, x: Tensor) -> Tensor:
        """
        Encode Euclidean point to sparse CGA point representation.

        Returns only the grade-1 components (n+2 values).

        Args:
            x: Euclidean point, shape (..., dim_euclidean)

        Returns:
            Sparse CGA point, shape (..., count_point)
        """
        batch_shape = x.shape[:-1]
        n = self.dim_euclidean

        # Compute |x|² / 2
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        half_norm_sq = 0.5 * x_norm_sq

        # Create sparse point
        result = torch.zeros(*batch_shape, self.count_point, dtype=x.dtype, device=x.device)

        # Euclidean components
        result[..., :n] = x

        # e+ and e- coefficients
        result[..., n] = (-0.5 + half_norm_sq).squeeze(-1)
        result[..., n+1] = (0.5 + half_norm_sq).squeeze(-1)

        return result

    def decode_point_sparse(self, point: Tensor) -> Tensor:
        """
        Decode sparse CGA point to Euclidean coordinates.

        Args:
            point: Sparse CGA point, shape (..., count_point)

        Returns:
            Euclidean point, shape (..., dim_euclidean)
        """
        n = self.dim_euclidean

        # Extract Euclidean components
        euclidean = point[..., :n]

        # Weight = e- coeff - e+ coeff
        e_plus = point[..., n:n+1]
        e_minus = point[..., n+1:n+2]
        weight = e_minus - e_plus

        weight = torch.where(
            torch.abs(weight) < 1e-10,
            torch.ones_like(weight),
            weight
        )

        return euclidean / weight

    # =========================================================================
    # Delegated Operations
    # =========================================================================

    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        """Geometric product."""
        return self._base.geometric_product(a, b)

    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        """Inner product."""
        return self._base.inner(a, b)

    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        """Outer product."""
        return self._base.outer(a, b)

    def contract_left(self, a: Tensor, b: Tensor) -> Tensor:
        """Left contraction."""
        return self._base.contract_left(a, b)

    def contract_right(self, a: Tensor, b: Tensor) -> Tensor:
        """Right contraction."""
        return self._base.contract_right(a, b)

    def scalar(self, a: Tensor, b: Tensor) -> Tensor:
        """Scalar product."""
        return self._base.scalar(a, b)

    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        """Regressive product (meet)."""
        return self._base.regressive(a, b)

    def sandwich(self, v: Tensor, x: Tensor) -> Tensor:
        """Sandwich product."""
        return self._base.sandwich(v, x)

    def reverse(self, mv: Tensor) -> Tensor:
        """Reverse."""
        return self._base.reverse(mv)

    def involute(self, mv: Tensor) -> Tensor:
        """Grade involution."""
        return self._base.involute(mv)

    def conjugate(self, mv: Tensor) -> Tensor:
        """Clifford conjugate."""
        return self._base.conjugate(mv)

    def select_grade(self, mv: Tensor, grade: int) -> Tensor:
        """Select grade."""
        return self._base.select_grade(mv, grade)

    def dual(self, mv: Tensor) -> Tensor:
        """Dual."""
        return self._base.dual(mv)

    def normalize(self, mv: Tensor) -> Tensor:
        """Normalize."""
        return self._base.normalize(mv)

    def inverse(self, mv: Tensor) -> Tensor:
        """Inverse."""
        return self._base.inverse(mv)

    def norm_squared(self, mv: Tensor) -> Tensor:
        """Norm squared."""
        return self._base.norm_squared(mv)

    def exp(self, mv: Tensor) -> Tensor:
        """Exponential."""
        return self._base.exp(mv)

    # Rotor operations
    def compose_rotor(self, r1: Tensor, r2: Tensor) -> Tensor:
        """Rotor composition."""
        return self._base.compose_rotor(r1, r2)

    def reverse_rotor(self, r: Tensor) -> Tensor:
        """Rotor reverse."""
        return self._base.reverse_rotor(r)

    def sandwich_rotor(self, r: Tensor, x: Tensor) -> Tensor:
        """Rotor sandwich product."""
        return self._base.sandwich_rotor(r, x)

    def norm_squared_rotor(self, r: Tensor) -> Tensor:
        """Rotor norm squared."""
        return self._base.norm_squared_rotor(r)

    def inverse_rotor(self, r: Tensor) -> Tensor:
        """Rotor inverse."""
        return self._base.inverse_rotor(r)

    def normalize_rotor(self, r: Tensor) -> Tensor:
        """Rotor normalize."""
        return self._base.normalize_rotor(r)

    def exp_bivector(self, B: Tensor) -> Tensor:
        """Bivector exponential."""
        return self._base.exp_bivector(B)

    def log_rotor(self, r: Tensor) -> Tensor:
        """Rotor logarithm."""
        return self._base.log_rotor(r)

    def slerp_rotor(self, r1: Tensor, r2: Tensor, t: float) -> Tensor:
        """Rotor slerp."""
        return self._base.slerp_rotor(r1, r2, t)

    # Factory methods
    def multivector(self, data: Tensor):
        """Create Multivector wrapper."""
        return self._base.multivector(data)

    def rotor(self, data: Tensor):
        """Create Rotor wrapper."""
        return self._base.rotor(data)

    def get_transform_layer(self):
        """Get transform layer."""
        return self._base.get_transform_layer()

    def get_encoder(self):
        """Get encoder layer for CGA."""
        return CGAEncoderLayer(self)

    def get_decoder(self):
        """Get decoder layer for CGA."""
        return CGADecoderLayer(self)


# =============================================================================
# CGA-specific PyTorch Layers
# =============================================================================

class CGAEncoderLayer(torch.nn.Module):
    """
    CGA encoder layer.

    Embeds Euclidean points as CGA null vectors.
    """

    def __init__(self, cga: CGAWrapper):
        super().__init__()
        self.cga = cga

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode Euclidean point to CGA null vector.

        Args:
            x: Euclidean point, shape (..., dim_euclidean)

        Returns:
            CGA point (full multivector), shape (..., count_blade)
        """
        # Force float32 for numerical stability (FR-054)
        return self.cga.encode(x.float())


class CGADecoderLayer(torch.nn.Module):
    """
    CGA decoder layer.

    Extracts Euclidean coordinates from CGA points.
    """

    def __init__(self, cga: CGAWrapper):
        super().__init__()
        self.cga = cga

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to Euclidean coordinates.

        Args:
            point: CGA point (full multivector), shape (..., count_blade)

        Returns:
            Euclidean point, shape (..., dim_euclidean)
        """
        return self.cga.decode(point.float())
