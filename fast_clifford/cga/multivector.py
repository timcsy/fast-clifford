"""
Multivector - High-level wrapper for CGA multivectors with operator overloading.

Provides intuitive Python operators for geometric algebra operations:
- a * b: Geometric product
- a ^ b: Outer product (wedge)
- a | b: Inner product
- a << b: Left contraction
- a >> b: Right contraction
- m @ x: Sandwich product
- ~a: Reverse
- a ** -1: Inverse

Classes:
- Multivector: General multivector (blade_count components)
- EvenVersor: Even-grade versor (even_versor_count components)
- Similitude: CGA-specific subset (same storage as EvenVersor)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, Literal
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .base import CGAAlgebraBase


class Multivector:
    """
    General multivector wrapper with operator overloading.

    Wraps a tensor of shape (..., blade_count) and provides
    intuitive operators for geometric algebra operations.

    Attributes:
        data: Underlying tensor of shape (..., blade_count)
        algebra: Reference to the CGA algebra instance
    """

    __slots__ = ('data', 'algebra')

    def __init__(self, data: Tensor, algebra: 'CGAAlgebraBase'):
        """
        Create a Multivector.

        Args:
            data: Tensor of shape (..., blade_count)
            algebra: CGA algebra instance
        """
        self.data = data
        self.algebra = algebra

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other: Union[Multivector, Tensor, float]) -> Multivector:
        """Addition: a + b"""
        if isinstance(other, Multivector):
            return Multivector(self.data + other.data, self.algebra)
        elif isinstance(other, (int, float)):
            # Add scalar to grade-0 component
            result = self.data.clone()
            result[..., 0] = result[..., 0] + other
            return Multivector(result, self.algebra)
        elif isinstance(other, Tensor):
            return Multivector(self.data + other, self.algebra)
        return NotImplemented

    def __radd__(self, other: Union[Tensor, float]) -> Multivector:
        """Right addition: scalar + a"""
        return self.__add__(other)

    def __sub__(self, other: Union[Multivector, Tensor, float]) -> Multivector:
        """Subtraction: a - b"""
        if isinstance(other, Multivector):
            return Multivector(self.data - other.data, self.algebra)
        elif isinstance(other, (int, float)):
            result = self.data.clone()
            result[..., 0] = result[..., 0] - other
            return Multivector(result, self.algebra)
        elif isinstance(other, Tensor):
            return Multivector(self.data - other, self.algebra)
        return NotImplemented

    def __rsub__(self, other: Union[Tensor, float]) -> Multivector:
        """Right subtraction: scalar - a"""
        return Multivector(-self.data, self.algebra).__add__(other)

    def __neg__(self) -> Multivector:
        """Negation: -a"""
        return Multivector(-self.data, self.algebra)

    # =========================================================================
    # Geometric Product Operators
    # =========================================================================

    def __mul__(self, other: Union[Multivector, Tensor, float]) -> Multivector:
        """
        Geometric product or scalar multiplication: a * b

        If other is a Multivector, computes geometric product.
        If other is a scalar, computes scalar multiplication.
        """
        if isinstance(other, Multivector):
            result = self.algebra.geometric_product_full(self.data, other.data)
            return Multivector(result, self.algebra)
        elif isinstance(other, (int, float)):
            return Multivector(self.data * other, self.algebra)
        elif isinstance(other, Tensor):
            if other.shape[-1] == self.algebra.blade_count:
                result = self.algebra.geometric_product_full(self.data, other)
                return Multivector(result, self.algebra)
            else:
                return Multivector(self.data * other, self.algebra)
        return NotImplemented

    def __rmul__(self, other: Union[Tensor, float]) -> Multivector:
        """Right multiplication: scalar * a or tensor * a"""
        if isinstance(other, (int, float)):
            return Multivector(other * self.data, self.algebra)
        elif isinstance(other, Tensor):
            if other.shape[-1] == self.algebra.blade_count:
                result = self.algebra.geometric_product_full(other, self.data)
                return Multivector(result, self.algebra)
            else:
                return Multivector(other * self.data, self.algebra)
        return NotImplemented

    def __truediv__(self, other: Union[Multivector, float]) -> Multivector:
        """
        Division: a / b

        If other is a scalar, computes scalar division.
        If other is a Multivector, computes a * b^-1.
        """
        if isinstance(other, (int, float)):
            return Multivector(self.data / other, self.algebra)
        elif isinstance(other, Multivector):
            return self * other.inverse()
        return NotImplemented

    # =========================================================================
    # Wedge Product (Outer Product)
    # =========================================================================

    def __xor__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """Outer product (wedge): a ^ b"""
        if isinstance(other, Multivector):
            result = self.algebra.outer_product(self.data, other.data)
            return Multivector(result, self.algebra)
        elif isinstance(other, Tensor):
            result = self.algebra.outer_product(self.data, other)
            return Multivector(result, self.algebra)
        return NotImplemented

    def __rxor__(self, other: Tensor) -> Multivector:
        """Right outer product: tensor ^ a"""
        if isinstance(other, Tensor):
            result = self.algebra.outer_product(other, self.data)
            return Multivector(result, self.algebra)
        return NotImplemented

    # =========================================================================
    # Inner Product
    # =========================================================================

    def __or__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """Inner product: a | b"""
        if isinstance(other, Multivector):
            result = self.algebra.inner_product(self.data, other.data)
            return Multivector(result, self.algebra)
        elif isinstance(other, Tensor):
            result = self.algebra.inner_product(self.data, other)
            return Multivector(result, self.algebra)
        return NotImplemented

    def __ror__(self, other: Tensor) -> Multivector:
        """Right inner product: tensor | a"""
        if isinstance(other, Tensor):
            result = self.algebra.inner_product(other, self.data)
            return Multivector(result, self.algebra)
        return NotImplemented

    # =========================================================================
    # Contractions
    # =========================================================================

    def __lshift__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """Left contraction: a << b"""
        if isinstance(other, Multivector):
            result = self.algebra.left_contraction(self.data, other.data)
            return Multivector(result, self.algebra)
        elif isinstance(other, Tensor):
            result = self.algebra.left_contraction(self.data, other)
            return Multivector(result, self.algebra)
        return NotImplemented

    def __rshift__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """Right contraction: a >> b"""
        if isinstance(other, Multivector):
            result = self.algebra.right_contraction(self.data, other.data)
            return Multivector(result, self.algebra)
        elif isinstance(other, Tensor):
            result = self.algebra.right_contraction(self.data, other)
            return Multivector(result, self.algebra)
        return NotImplemented

    # =========================================================================
    # Sandwich Product
    # =========================================================================

    def __matmul__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """
        Sandwich product: m @ x = m * x * ~m

        For EvenVersor/Similitude, uses optimized sparse implementation.
        """
        if isinstance(other, Multivector):
            other_data = other.data
        elif isinstance(other, Tensor):
            other_data = other
        else:
            return NotImplemented

        # Use full sandwich: m * x * ~m
        mx = self.algebra.geometric_product_full(self.data, other_data)
        rev_m = self.algebra.reverse_full(self.data)
        result = self.algebra.geometric_product_full(mx, rev_m)
        return Multivector(result, self.algebra)

    # =========================================================================
    # Unary Operators
    # =========================================================================

    def __invert__(self) -> Multivector:
        """Reverse: ~a"""
        result = self.algebra.reverse_full(self.data)
        return Multivector(result, self.algebra)

    def __pow__(self, n: int) -> Multivector:
        """
        Power: a ** n

        For n >= 0: repeated geometric product
        For n == -1: inverse
        For n < -1: inverse then power
        """
        if n == 0:
            # Return scalar 1
            result = torch.zeros_like(self.data)
            result[..., 0] = 1.0
            return Multivector(result, self.algebra)
        elif n == 1:
            return Multivector(self.data.clone(), self.algebra)
        elif n == -1:
            return self.inverse()
        elif n > 1:
            result = self
            for _ in range(n - 1):
                result = result * self
            return result
        else:  # n < -1
            inv = self.inverse()
            result = inv
            for _ in range(-n - 1):
                result = result * inv
            return result

    # =========================================================================
    # Methods
    # =========================================================================

    def reverse(self) -> Multivector:
        """Compute the reverse of this multivector."""
        return ~self

    def dual(self) -> Multivector:
        """Compute the dual of this multivector."""
        result = self.algebra.dual(self.data)
        return Multivector(result, self.algebra)

    def normalize(self) -> Multivector:
        """Normalize this multivector to unit norm."""
        result = self.algebra.normalize(self.data)
        return Multivector(result, self.algebra)

    def grade(self, k: int) -> Multivector:
        """Extract grade-k component."""
        result = self.algebra.grade_select(self.data, k)
        return Multivector(result, self.algebra)

    def inverse(self) -> Multivector:
        """
        Compute the inverse of this multivector.

        For a versor V, V^-1 = ~V / (V * ~V)_0
        """
        rev = self.algebra.reverse_full(self.data)
        norm_sq = self.algebra.inner_product(self.data, rev)
        # norm_sq is shape (..., 1), need to broadcast
        result = rev / (norm_sq + 1e-12)
        return Multivector(result, self.algebra)

    def exp(self) -> Multivector:
        """
        Exponential map for bivector.

        Only valid for bivector (grade-2) multivectors.
        Returns exp(B) = cos(theta) + sin(theta)/theta * B
        """
        # Extract bivector part
        B = self.algebra.grade_select(self.data, 2)
        # Get bivector indices for sparse representation
        from fast_clifford.codegen.cga_factory import compute_grade_indices
        grade_indices = compute_grade_indices(self.algebra.euclidean_dim)
        biv_indices = list(grade_indices[2])
        B_sparse = B[..., biv_indices]

        # Compute exp
        ev = self.algebra.exp_bivector(B_sparse)

        # Embed back to full multivector
        result = torch.zeros_like(self.data)
        from fast_clifford.codegen.cga_factory import get_even_versor_indices
        ev_indices = list(get_even_versor_indices(self.algebra.euclidean_dim))
        for i, idx in enumerate(ev_indices):
            result[..., idx] = ev[..., i]

        return Multivector(result, self.algebra)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the underlying tensor."""
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying tensor."""
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self.data.dtype

    def to(self, device: Union[str, torch.device]) -> Multivector:
        """Move to a device."""
        return Multivector(self.data.to(device), self.algebra)

    def clone(self) -> Multivector:
        """Clone this multivector."""
        return Multivector(self.data.clone(), self.algebra)

    def detach(self) -> Multivector:
        """Detach from computation graph."""
        return Multivector(self.data.detach(), self.algebra)

    def __repr__(self) -> str:
        return f"Multivector(shape={self.shape}, algebra={self.algebra})"


class EvenVersor(Multivector):
    """
    Even-grade versor (even_versor_count components).

    Represents rotations, translations, and their compositions.
    Uses sparse representation for efficiency.

    Data shape: (..., even_versor_count)
    """

    def __init__(self, data: Tensor, algebra: 'CGAAlgebraBase'):
        """
        Create an EvenVersor.

        Args:
            data: Tensor of shape (..., even_versor_count)
            algebra: CGA algebra instance
        """
        # Store sparse data directly
        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'algebra', algebra)

    def __mul__(self, other: Union[EvenVersor, Multivector, Tensor, float]) -> Union[EvenVersor, Multivector]:
        """
        Composition or scalar multiplication.

        EvenVersor * EvenVersor -> EvenVersor (composition)
        EvenVersor * Multivector -> Multivector (geometric product)
        EvenVersor * scalar -> EvenVersor
        """
        if isinstance(other, EvenVersor):
            result = self.algebra.compose_even_versor(self.data, other.data)
            return EvenVersor(result, self.algebra)
        elif isinstance(other, (int, float)):
            return EvenVersor(self.data * other, self.algebra)
        elif isinstance(other, Multivector):
            # Embed to full and compute
            full_self = self.to_multivector()
            return full_self * other
        elif isinstance(other, Tensor):
            if other.shape[-1] == self.algebra.even_versor_count:
                result = self.algebra.compose_even_versor(self.data, other)
                return EvenVersor(result, self.algebra)
            else:
                return EvenVersor(self.data * other, self.algebra)
        return NotImplemented

    def __matmul__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """
        Sandwich product: ev @ x = ev * x * ~ev

        Uses sparse sandwich product for UPGC points.
        """
        if isinstance(other, Multivector):
            other_data = other.data
        elif isinstance(other, Tensor):
            other_data = other
        else:
            return NotImplemented

        # Check if other is a point (grade-1)
        if other_data.shape[-1] == self.algebra.point_count:
            # Use sparse sandwich
            result = self.algebra.sandwich_product_sparse(self.data, other_data)
            # Create point Multivector (grade-1)
            full_result = torch.zeros(
                *other_data.shape[:-1], self.algebra.blade_count,
                device=other_data.device, dtype=other_data.dtype
            )
            from fast_clifford.codegen.cga_factory import compute_grade_indices
            grade_indices = compute_grade_indices(self.algebra.euclidean_dim)
            point_indices = list(grade_indices[1])
            for i, idx in enumerate(point_indices):
                full_result[..., idx] = result[..., i]
            return Multivector(full_result, self.algebra)
        else:
            # Full sandwich
            full_self = self.to_multivector()
            return full_self @ Multivector(other_data, self.algebra)

    def __invert__(self) -> EvenVersor:
        """Reverse: ~ev"""
        result = self.algebra.reverse_even_versor(self.data)
        return EvenVersor(result, self.algebra)

    def inverse(self) -> EvenVersor:
        """Compute the inverse of this EvenVersor."""
        rev = self.algebra.reverse_even_versor(self.data)
        # For normalized versor: inverse = reverse
        # For non-normalized: need to divide by norm squared
        full_data = self.to_multivector().data
        rev_full = (~self).to_multivector().data
        norm_sq = self.algebra.inner_product(full_data, rev_full)
        result = rev / (norm_sq + 1e-12)
        return EvenVersor(result, self.algebra)

    def to_multivector(self) -> Multivector:
        """Convert to full Multivector representation."""
        from fast_clifford.codegen.cga_factory import get_even_versor_indices
        ev_indices = list(get_even_versor_indices(self.algebra.euclidean_dim))

        full = torch.zeros(
            *self.data.shape[:-1], self.algebra.blade_count,
            device=self.data.device, dtype=self.data.dtype
        )
        for i, idx in enumerate(ev_indices):
            full[..., idx] = self.data[..., i]

        return Multivector(full, self.algebra)

    def __repr__(self) -> str:
        return f"EvenVersor(shape={self.shape}, algebra={self.algebra})"


class Similitude(EvenVersor):
    """
    CGA-specific Similitude (rotation + translation + scaling).

    Same storage as EvenVersor, but uses optimized operations
    that exploit the Similitude constraint (no transversion).

    Data shape: (..., even_versor_count)
    """

    def __mul__(self, other: Union[Similitude, EvenVersor, Multivector, Tensor, float]) -> Union[Similitude, EvenVersor, Multivector]:
        """
        Composition with optimized Similitude operations.

        Similitude * Similitude -> Similitude (optimized)
        Similitude * EvenVersor -> EvenVersor
        Similitude * Multivector -> Multivector
        """
        if isinstance(other, Similitude):
            result = self.algebra.compose_similitude(self.data, other.data)
            return Similitude(result, self.algebra)
        elif isinstance(other, EvenVersor):
            result = self.algebra.compose_even_versor(self.data, other.data)
            return EvenVersor(result, self.algebra)
        elif isinstance(other, (int, float)):
            return Similitude(self.data * other, self.algebra)
        elif isinstance(other, Multivector):
            full_self = self.to_multivector()
            return full_self * other
        elif isinstance(other, Tensor):
            if other.shape[-1] == self.algebra.even_versor_count:
                result = self.algebra.compose_similitude(self.data, other)
                return Similitude(result, self.algebra)
            else:
                return Similitude(self.data * other, self.algebra)
        return NotImplemented

    def __matmul__(self, other: Union[Multivector, Tensor]) -> Multivector:
        """
        Sandwich product with optimized Similitude operations.
        """
        if isinstance(other, Multivector):
            other_data = other.data
        elif isinstance(other, Tensor):
            other_data = other
        else:
            return NotImplemented

        # Check if other is a point
        if other_data.shape[-1] == self.algebra.point_count:
            # Use Similitude-optimized sandwich
            result = self.algebra.sandwich_product_similitude(self.data, other_data)
            # Create point Multivector
            full_result = torch.zeros(
                *other_data.shape[:-1], self.algebra.blade_count,
                device=other_data.device, dtype=other_data.dtype
            )
            from fast_clifford.codegen.cga_factory import compute_grade_indices
            grade_indices = compute_grade_indices(self.algebra.euclidean_dim)
            point_indices = list(grade_indices[1])
            for i, idx in enumerate(point_indices):
                full_result[..., idx] = result[..., i]
            return Multivector(full_result, self.algebra)
        else:
            # Fall back to full sandwich
            return super().__matmul__(other)

    def __invert__(self) -> Similitude:
        """Reverse: ~s"""
        result = self.algebra.reverse_even_versor(self.data)
        return Similitude(result, self.algebra)

    def structure_normalize(self, eps: float = 1e-8) -> Similitude:
        """Apply structure normalization to maintain Similitude constraints."""
        result = self.algebra.structure_normalize(self.data, eps)
        return Similitude(result, self.algebra)

    def __repr__(self) -> str:
        return f"Similitude(shape={self.shape}, algebra={self.algebra})"
