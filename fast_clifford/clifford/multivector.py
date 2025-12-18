"""
Multivector and Rotor wrapper classes with operator overloading.

This module provides high-level wrapper classes that enable intuitive
operator syntax for Clifford algebra operations.

Operator Mapping:
    * : geometric_product (a * b)
    ^ : outer product (a ^ b)
    | : inner product (a | b)
    << : left contraction (a << b)
    >> : right contraction (a >> b)
    @ : sandwich product (m @ x)
    & : regressive/meet (a & b)
    ~ : reverse (~a)
    ** -1 : inverse (a ** -1)
    () : grade selection (mv(k))
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .base import CliffordAlgebraBase


class Multivector:
    """
    Multivector wrapper with operator overloading.

    Provides an intuitive interface for Clifford algebra operations using
    Python's operator syntax.

    Attributes:
        data: The underlying tensor of shape (..., count_blade)
        algebra: The Clifford algebra this multivector belongs to

    Example:
        >>> vga = VGA(3)
        >>> a = vga.multivector(tensor_a)
        >>> b = vga.multivector(tensor_b)
        >>> c = a * b          # geometric product
        >>> d = a ^ b          # outer product
        >>> e = ~a             # reverse
        >>> f = a(2)           # grade-2 selection
    """

    __slots__ = ("data", "algebra")

    def __init__(self, data: Tensor, algebra: "CliffordAlgebraBase") -> None:
        """
        Initialize a Multivector.

        Args:
            data: Tensor of shape (..., count_blade)
            algebra: The Clifford algebra instance
        """
        self.data = data
        self.algebra = algebra

    # =========================================================================
    # Geometric Product and Scalar Operations
    # =========================================================================

    def _same_algebra(self, other: "Multivector") -> bool:
        """Check if two multivectors are from the same algebra (by signature)."""
        return (
            self.algebra.p == other.algebra.p and
            self.algebra.q == other.algebra.q and
            self.algebra.r == other.algebra.r
        )

    def __mul__(self, other: Union["Multivector", float, int]) -> "Multivector":
        """
        Geometric product (a * b) or scalar multiplication.

        Maps to: geometric_product(a, b) for Multivector
                 scalar multiplication for float/int
        """
        if isinstance(other, Multivector):
            if not self._same_algebra(other):
                raise ValueError("Cannot multiply multivectors from different algebras")
            result = self.algebra.geometric_product(self.data, other.data)
            return Multivector(result, self.algebra)
        elif isinstance(other, (float, int)):
            return Multivector(self.data * other, self.algebra)
        return NotImplemented

    def __rmul__(self, other: Union[float, int]) -> "Multivector":
        """Scalar multiplication (s * a)."""
        if isinstance(other, (float, int)):
            return Multivector(self.data * other, self.algebra)
        return NotImplemented

    def __truediv__(self, scalar: Union[float, int]) -> "Multivector":
        """Scalar division (a / s)."""
        if isinstance(scalar, (float, int)):
            return Multivector(self.data / scalar, self.algebra)
        return NotImplemented

    # =========================================================================
    # Outer and Inner Products
    # =========================================================================

    def __xor__(self, other: "Multivector") -> "Multivector":
        """
        Outer product (a ^ b).

        Maps to: outer(a, b)
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compute outer product of multivectors from different algebras")
        result = self.algebra.outer(self.data, other.data)
        return Multivector(result, self.algebra)

    def __or__(self, other: "Multivector") -> "Multivector":
        """
        Inner product (a | b).

        Maps to: inner(a, b)
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compute inner product of multivectors from different algebras")
        result = self.algebra.inner(self.data, other.data)
        return Multivector(result, self.algebra)

    # =========================================================================
    # Contractions
    # =========================================================================

    def __lshift__(self, other: "Multivector") -> "Multivector":
        """
        Left contraction (a << b).

        Maps to: contract_left(a, b)
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compute left contraction of multivectors from different algebras")
        result = self.algebra.contract_left(self.data, other.data)
        return Multivector(result, self.algebra)

    def __rshift__(self, other: "Multivector") -> "Multivector":
        """
        Right contraction (a >> b).

        Maps to: contract_right(a, b)
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compute right contraction of multivectors from different algebras")
        result = self.algebra.contract_right(self.data, other.data)
        return Multivector(result, self.algebra)

    # =========================================================================
    # Sandwich Product and Meet
    # =========================================================================

    def __matmul__(self, other: "Multivector") -> "Multivector":
        """
        Sandwich product (m @ x = m * x * ~m).

        Maps to: sandwich(m, x)

        Note: self is the versor, other is the operand.
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compute sandwich product of multivectors from different algebras")
        result = self.algebra.sandwich(self.data, other.data)
        return Multivector(result, self.algebra)

    def __and__(self, other: "Multivector") -> "Multivector":
        """
        Regressive product / meet (a & b).

        Maps to: regressive(a, b)
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compute meet of multivectors from different algebras")
        result = self.algebra.regressive(self.data, other.data)
        return Multivector(result, self.algebra)

    # =========================================================================
    # Unary Operations
    # =========================================================================

    def __invert__(self) -> "Multivector":
        """
        Reverse (~a).

        Maps to: reverse(a)
        """
        result = self.algebra.reverse(self.data)
        return Multivector(result, self.algebra)

    def __pow__(self, exp: int) -> "Multivector":
        """
        Power / inverse (a ** n).

        For exp == -1: returns inverse(a)
        For exp == 0: returns scalar 1
        For exp > 0: returns a * a * ... (exp times)
        """
        if not isinstance(exp, int):
            return NotImplemented

        if exp == -1:
            result = self.algebra.inverse(self.data)
            return Multivector(result, self.algebra)
        elif exp == 0:
            # Return identity (scalar 1)
            result = torch.zeros_like(self.data)
            result[..., 0] = 1.0
            return Multivector(result, self.algebra)
        elif exp > 0:
            result = self.data.clone()
            for _ in range(exp - 1):
                result = self.algebra.geometric_product(result, self.data)
            return Multivector(result, self.algebra)
        else:
            # Negative exponent: compute inverse then positive power
            inv = self.algebra.inverse(self.data)
            result = inv.clone()
            for _ in range(-exp - 1):
                result = self.algebra.geometric_product(result, inv)
            return Multivector(result, self.algebra)

    def __neg__(self) -> "Multivector":
        """Negation (-a)."""
        return Multivector(-self.data, self.algebra)

    def __pos__(self) -> "Multivector":
        """Positive (+a) - returns self."""
        return self

    # =========================================================================
    # Addition and Subtraction
    # =========================================================================

    def __add__(self, other: "Multivector") -> "Multivector":
        """Addition (a + b)."""
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot add multivectors from different algebras")
        return Multivector(self.data + other.data, self.algebra)

    def __radd__(self, other: Union[float, int]) -> "Multivector":
        """Right addition for scalar (s + a)."""
        if isinstance(other, (float, int)):
            result = self.data.clone()
            result[..., 0] = result[..., 0] + other
            return Multivector(result, self.algebra)
        return NotImplemented

    def __sub__(self, other: "Multivector") -> "Multivector":
        """Subtraction (a - b)."""
        if not isinstance(other, Multivector):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot subtract multivectors from different algebras")
        return Multivector(self.data - other.data, self.algebra)

    def __rsub__(self, other: Union[float, int]) -> "Multivector":
        """Right subtraction for scalar (s - a)."""
        if isinstance(other, (float, int)):
            result = -self.data.clone()
            result[..., 0] = result[..., 0] + other
            return Multivector(result, self.algebra)
        return NotImplemented

    # =========================================================================
    # Grade Selection
    # =========================================================================

    def __call__(self, grade: int) -> "Multivector":
        """
        Grade selection (mv(k)).

        Maps to: select_grade(mv, k)

        Args:
            grade: The grade to extract (0 to max_grade)

        Returns:
            Multivector containing only the selected grade components
        """
        result = self.algebra.select_grade(self.data, grade)
        return Multivector(result, self.algebra)

    def grade(self, k: int) -> "Multivector":
        """
        Grade selection (alternative method syntax).

        Equivalent to mv(k).
        """
        return self(k)

    # =========================================================================
    # Methods
    # =========================================================================

    def dual(self) -> "Multivector":
        """Poincare dual."""
        result = self.algebra.dual(self.data)
        return Multivector(result, self.algebra)

    def normalize(self) -> "Multivector":
        """Normalize to unit norm."""
        result = self.algebra.normalize(self.data)
        return Multivector(result, self.algebra)

    def inverse(self) -> "Multivector":
        """Multiplicative inverse."""
        result = self.algebra.inverse(self.data)
        return Multivector(result, self.algebra)

    def exp(self) -> "Multivector":
        """Exponential map."""
        result = self.algebra.exp(self.data)
        return Multivector(result, self.algebra)

    def conjugate(self) -> "Multivector":
        """Clifford conjugate."""
        result = self.algebra.conjugate(self.data)
        return Multivector(result, self.algebra)

    def involute(self) -> "Multivector":
        """Grade involution."""
        result = self.algebra.involute(self.data)
        return Multivector(result, self.algebra)

    def norm_squared(self) -> Tensor:
        """Norm squared |m|^2."""
        return self.algebra.norm_squared(self.data)

    def norm(self) -> Tensor:
        """Norm |m|."""
        return torch.sqrt(torch.abs(self.norm_squared()) + 1e-12)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_rotor(self) -> "Rotor":
        """
        Convert to Rotor (extracts even grades).

        Note: This assumes the multivector represents a rotor.
        Use with caution as it does not validate the rotor property.
        """
        # Extract rotor components from full multivector
        # The algebra must provide a way to do this
        # For now, we delegate to the algebra's rotor factory
        # which should handle the extraction
        return self.algebra.rotor(self.data)

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensor."""
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensor."""
        return self.data.dtype

    def __repr__(self) -> str:
        return f"Multivector(shape={self.shape}, algebra=Cl({self.algebra.p},{self.algebra.q},{self.algebra.r}))"


class Rotor:
    """
    Rotor wrapper with accelerated operations.

    Rotors are even-grade versors used for rotations and transformations.
    This class automatically uses accelerated rotor-specific operations.

    Attributes:
        data: The underlying tensor of shape (..., count_rotor)
        algebra: The Clifford algebra this rotor belongs to

    Example:
        >>> vga = VGA(3)
        >>> r1 = vga.rotor(tensor_r1)
        >>> r2 = vga.rotor(tensor_r2)
        >>> r3 = r1 * r2       # rotor composition
        >>> p_transformed = r1 @ point  # sandwich product
        >>> r_inv = ~r1        # rotor inverse (via reverse)
    """

    __slots__ = ("data", "algebra")

    def __init__(self, data: Tensor, algebra: "CliffordAlgebraBase") -> None:
        """
        Initialize a Rotor.

        Args:
            data: Tensor of shape (..., count_rotor)
            algebra: The Clifford algebra instance
        """
        self.data = data
        self.algebra = algebra

    # =========================================================================
    # Rotor Composition
    # =========================================================================

    def __mul__(self, other: "Rotor") -> "Rotor":
        """
        Rotor composition (r1 * r2).

        Maps to: compose_rotor(r1, r2)

        Uses accelerated rotor multiplication.
        """
        if not isinstance(other, Rotor):
            return NotImplemented
        if not self._same_algebra(other):
            raise ValueError("Cannot compose rotors from different algebras")
        result = self.algebra.compose_rotor(self.data, other.data)
        return Rotor(result, self.algebra)

    def __rmul__(self, other: Union[float, int]) -> "Rotor":
        """Scalar multiplication (s * r)."""
        if isinstance(other, (float, int)):
            return Rotor(self.data * other, self.algebra)
        return NotImplemented

    def __truediv__(self, scalar: Union[float, int]) -> "Rotor":
        """Scalar division (r / s)."""
        if isinstance(scalar, (float, int)):
            return Rotor(self.data / scalar, self.algebra)
        return NotImplemented

    # =========================================================================
    # Sandwich Product
    # =========================================================================

    def __matmul__(self, point: Tensor) -> Tensor:
        """
        Rotor sandwich product (r @ x = r * x * ~r).

        Maps to: sandwich_rotor(r, x)

        Args:
            point: Tensor to transform (shape depends on algebra type)

        Returns:
            Transformed tensor (same shape as input)
        """
        result = self.algebra.sandwich_rotor(self.data, point)
        return result

    # =========================================================================
    # Reverse (Inverse for Unit Rotors)
    # =========================================================================

    def __invert__(self) -> "Rotor":
        """
        Rotor reverse (~r).

        Maps to: reverse_rotor(r)

        For unit rotors, reverse equals inverse.
        """
        result = self.algebra.reverse_rotor(self.data)
        return Rotor(result, self.algebra)

    def __pow__(self, exp: int) -> "Rotor":
        """
        Power / inverse (r ** n).

        For exp == -1: returns inverse_rotor(r)
        For exp == 0: returns identity rotor
        For exp > 0: returns r * r * ... (exp times)
        """
        if not isinstance(exp, int):
            return NotImplemented

        if exp == -1:
            result = self.algebra.inverse_rotor(self.data)
            return Rotor(result, self.algebra)
        elif exp == 0:
            # Return identity rotor (scalar 1, rest 0)
            result = torch.zeros_like(self.data)
            result[..., 0] = 1.0
            return Rotor(result, self.algebra)
        elif exp > 0:
            result = self.data.clone()
            for _ in range(exp - 1):
                result = self.algebra.compose_rotor(result, self.data)
            return Rotor(result, self.algebra)
        else:
            # Negative exponent: compute inverse then positive power
            inv = self.algebra.inverse_rotor(self.data)
            result = inv.clone()
            for _ in range(-exp - 1):
                result = self.algebra.compose_rotor(result, inv)
            return Rotor(result, self.algebra)

    def __neg__(self) -> "Rotor":
        """Negation (-r)."""
        return Rotor(-self.data, self.algebra)

    # =========================================================================
    # Methods
    # =========================================================================

    def normalize(self) -> "Rotor":
        """
        Normalize to unit rotor.

        Maps to: normalize_rotor(r)
        """
        result = self.algebra.normalize_rotor(self.data)
        return Rotor(result, self.algebra)

    def inverse(self) -> "Rotor":
        """
        Rotor inverse.

        Maps to: inverse_rotor(r)
        """
        result = self.algebra.inverse_rotor(self.data)
        return Rotor(result, self.algebra)

    def log(self) -> Tensor:
        """
        Rotor logarithm -> Bivector.

        Maps to: log_rotor(r)

        Returns:
            Bivector tensor of shape (..., count_bivector)
        """
        return self.algebra.log_rotor(self.data)

    def norm_squared(self) -> Tensor:
        """
        Rotor norm squared.

        Maps to: norm_squared_rotor(r)
        """
        return self.algebra.norm_squared_rotor(self.data)

    def norm(self) -> Tensor:
        """Rotor norm."""
        return torch.sqrt(torch.abs(self.norm_squared()) + 1e-12)

    def slerp(self, other: "Rotor", t: float) -> "Rotor":
        """
        Spherical linear interpolation.

        Maps to: slerp_rotor(self, other, t)

        Args:
            other: Target rotor
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated rotor
        """
        if not isinstance(other, Rotor):
            raise TypeError("slerp requires another Rotor")
        if not self._same_algebra(other):
            raise ValueError("Cannot slerp rotors from different algebras")
        result = self.algebra.slerp_rotor(self.data, other.data, t)
        return Rotor(result, self.algebra)

    # =========================================================================
    # Conversion
    # =========================================================================

    def to_multivector(self) -> Multivector:
        """
        Convert to full Multivector.

        Expands rotor components to full multivector representation.
        """
        # Create full multivector with rotor components in even grades
        # This requires knowing the rotor mask from the algebra
        # For now, create a multivector and let the algebra handle it
        return self.algebra.multivector(self.data)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensor."""
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensor."""
        return self.data.dtype

    def __repr__(self) -> str:
        return f"Rotor(shape={self.shape}, algebra=Cl({self.algebra.p},{self.algebra.q},{self.algebra.r}))"
