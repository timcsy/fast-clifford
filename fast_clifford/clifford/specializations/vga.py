"""
VGA (Vanilla Geometric Algebra) Specialization

VGA(n) = Cl(n, 0) - Pure Euclidean vector algebra with signature (+1, +1, ..., +1)

This specialization provides:
- encode: Embed vector as grade-1 multivector
- decode: Extract grade-1 components
- dim_euclidean: Euclidean dimension property
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..base import CliffordAlgebraBase


class VGAWrapper:
    """
    VGA(n) = Cl(n, 0) specialization wrapper.

    Wraps a base Clifford algebra with VGA-specific operations.

    Attributes:
        base_algebra: The underlying Cl(n, 0) algebra
        dim_euclidean: Euclidean dimension (n)

    Example:
        >>> vga = VGA(3)  # Creates VGA(3) = Cl(3, 0)
        >>> v = torch.tensor([1.0, 2.0, 3.0])
        >>> encoded = vga.encode(v)  # Embed as grade-1
        >>> decoded = vga.decode(encoded)  # Extract back
    """

    def __init__(self, base_algebra: "CliffordAlgebraBase"):
        """
        Initialize VGA wrapper.

        Args:
            base_algebra: A Clifford algebra Cl(n, 0)

        Raises:
            ValueError: If algebra is not VGA type (q != 0 or r != 0)
        """
        if base_algebra.q != 0 or base_algebra.r != 0:
            raise ValueError(
                f"VGAWrapper requires Cl(n, 0, 0), got Cl({base_algebra.p}, {base_algebra.q}, {base_algebra.r})"
            )
        self._base = base_algebra

    # =========================================================================
    # Properties - Delegate to base algebra
    # =========================================================================

    @property
    def p(self) -> int:
        """Positive dimension."""
        return self._base.p

    @property
    def q(self) -> int:
        """Negative dimension (always 0 for VGA)."""
        return 0

    @property
    def r(self) -> int:
        """Degenerate dimension (always 0 for VGA)."""
        return 0

    @property
    def dim_euclidean(self) -> int:
        """Euclidean dimension n."""
        return self._base.p

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
        """Algebra type (always 'vga')."""
        return "vga"

    # =========================================================================
    # VGA-Specific Operations
    # =========================================================================

    def encode(self, x: Tensor) -> Tensor:
        """
        Embed Euclidean vector as grade-1 multivector.

        For VGA, this simply places the vector components in the grade-1 slots.

        Args:
            x: Euclidean vector, shape (..., dim_euclidean)

        Returns:
            Multivector with only grade-1 components, shape (..., count_blade)
        """
        batch_shape = x.shape[:-1]
        n = self.dim_euclidean

        # Create full multivector with zeros
        result = torch.zeros(*batch_shape, self.count_blade, dtype=x.dtype, device=x.device)

        # Place vector components in grade-1 slots (indices 1 to n)
        result[..., 1:n+1] = x

        return result

    def decode(self, mv: Tensor) -> Tensor:
        """
        Extract grade-1 components as Euclidean vector.

        Args:
            mv: Multivector, shape (..., count_blade)

        Returns:
            Euclidean vector, shape (..., dim_euclidean)
        """
        n = self.dim_euclidean
        return mv[..., 1:n+1]

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
        """Get encoder layer for VGA."""
        from ..layers import VGAEncoderLayer
        return VGAEncoderLayer(self)

    def get_decoder(self):
        """Get decoder layer for VGA."""
        from ..layers import VGADecoderLayer
        return VGADecoderLayer(self)


# =============================================================================
# VGA-specific PyTorch Layers
# =============================================================================

class VGAEncoderLayer(torch.nn.Module):
    """
    VGA encoder layer.

    Embeds Euclidean vectors as grade-1 multivectors.
    """

    def __init__(self, vga: VGAWrapper):
        super().__init__()
        self.vga = vga

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode vector to multivector.

        Args:
            x: Euclidean vector, shape (..., dim_euclidean)

        Returns:
            Multivector, shape (..., count_blade)
        """
        return self.vga.encode(x.float())


class VGADecoderLayer(torch.nn.Module):
    """
    VGA decoder layer.

    Extracts grade-1 components as Euclidean vectors.
    """

    def __init__(self, vga: VGAWrapper):
        super().__init__()
        self.vga = vga

    def forward(self, mv: Tensor) -> Tensor:
        """
        Decode multivector to vector.

        Args:
            mv: Multivector, shape (..., count_blade)

        Returns:
            Euclidean vector, shape (..., dim_euclidean)
        """
        return self.vga.decode(mv.float())
