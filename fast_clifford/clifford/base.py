"""
CliffordAlgebraBase - Abstract base class for all Clifford algebras Cl(p,q,r)

This module defines the unified interface that all Clifford algebra implementations
must follow, including HardcodedClWrapper, BottPeriodicityAlgebra, and RuntimeCliffordAlgebra.
"""

from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .multivector import Multivector, Rotor


class CliffordAlgebraBase(ABC):
    """
    Abstract base class for any Clifford algebra Cl(p, q, r).

    This class defines the unified API that all implementations must provide.
    Implementations include:
    - HardcodedClWrapper: For pre-generated algebras (p+q <= 9)
    - BottPeriodicityAlgebra: For high-dimensional algebras using Bott periodicity
    - RuntimeCliffordAlgebra: Fallback for degenerate algebras (r > 0)
    """

    # =========================================================================
    # Signature Properties
    # =========================================================================

    @property
    @abstractmethod
    def p(self) -> int:
        """Positive dimension (number of e_i with e_i² = +1)"""
        ...

    @property
    @abstractmethod
    def q(self) -> int:
        """Negative dimension (number of e_i with e_i² = -1)"""
        ...

    @property
    @abstractmethod
    def r(self) -> int:
        """Degenerate dimension (number of e_i with e_i² = 0)"""
        ...

    # =========================================================================
    # Count Properties
    # =========================================================================

    @property
    def count_blade(self) -> int:
        """Total blade count = 2^(p+q+r)"""
        return 2 ** (self.p + self.q + self.r)

    @property
    @abstractmethod
    def count_rotor(self) -> int:
        """Rotor component count (sum of even grade dimensions)"""
        ...

    @property
    @abstractmethod
    def count_bivector(self) -> int:
        """Bivector component count = C(p+q+r, 2)"""
        ...

    @property
    def max_grade(self) -> int:
        """Maximum grade = p+q+r"""
        return self.p + self.q + self.r

    # =========================================================================
    # Type Detection
    # =========================================================================

    @property
    def algebra_type(self) -> Literal["vga", "cga", "pga", "general"]:
        """
        Detect algebra type based on signature.

        Returns:
            'vga' if q == 0 and r == 0
            'cga' if q == 1 and r == 0
            'pga' if r > 0
            'general' otherwise
        """
        if self.r > 0:
            return "pga"
        elif self.q == 0:
            return "vga"
        elif self.q == 1:
            return "cga"
        else:
            return "general"

    # =========================================================================
    # Core Operations (Full Multivector)
    # =========================================================================

    @abstractmethod
    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Geometric product ab.

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Inner product (scalar product) <ab>₀.

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., 1)
        """
        ...

    @abstractmethod
    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Outer product (wedge product) a ∧ b.

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def contract_left(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Left contraction a ⌋ b.

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def contract_right(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Right contraction a ⌊ b.

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    def scalar(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Scalar product <ab>₀ (alias for inner).

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., 1)
        """
        return self.inner(a, b)

    @abstractmethod
    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Regressive product (meet) a ∨ b = (a* ∧ b*)*.

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    def sandwich(self, v: Tensor, x: Tensor) -> Tensor:
        """
        Sandwich product vxṽ.

        Args:
            v: shape (..., count_blade) - versor
            x: shape (..., count_blade) - operand

        Returns:
            shape (..., count_blade)
        """
        vx = self.geometric_product(v, x)
        v_rev = self.reverse(v)
        return self.geometric_product(vx, v_rev)

    # =========================================================================
    # Unary Operations
    # =========================================================================

    @abstractmethod
    def reverse(self, mv: Tensor) -> Tensor:
        """
        Reversion m̃ (reverses order of basis vectors in each blade).

        For grade k: reverse sign is (-1)^(k(k-1)/2)

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def involute(self, mv: Tensor) -> Tensor:
        """
        Grade involution m̂ (negates odd grades).

        For grade k: involution sign is (-1)^k

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def conjugate(self, mv: Tensor) -> Tensor:
        """
        Clifford conjugate m† = reverse(involute(m)).

        For grade k: conjugate sign is (-1)^(k(k+1)/2)

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def select_grade(self, mv: Tensor, grade: int) -> Tensor:
        """
        Extract specific grade components.

        Args:
            mv: shape (..., count_blade)
            grade: 0 to max_grade

        Returns:
            shape (..., count_blade) with only grade-k components non-zero
        """
        ...

    @abstractmethod
    def dual(self, mv: Tensor) -> Tensor:
        """
        Poincaré dual: m* = m ⌋ I (left contraction with pseudoscalar).

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    def normalize(self, mv: Tensor) -> Tensor:
        """
        Normalize to unit norm.

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        norm_sq = self.norm_squared(mv)
        norm = torch.sqrt(torch.abs(norm_sq) + 1e-12)
        return mv / norm

    def inverse(self, mv: Tensor) -> Tensor:
        """
        Multiplicative inverse m⁻¹ = m̃ / (m * m̃).

        Only valid for versors (products of invertible vectors).

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        rev = self.reverse(mv)
        norm_sq = self.norm_squared(mv)
        return rev / (norm_sq + 1e-12)

    @abstractmethod
    def norm_squared(self, mv: Tensor) -> Tensor:
        """
        Norm squared |m|² = <m * m̃>₀.

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., 1)
        """
        ...

    def exp(self, mv: Tensor) -> Tensor:
        """
        General exponential map exp(m).

        For bivectors, this produces rotors. For general multivectors,
        uses Taylor series approximation.

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        # Default implementation using Taylor series
        # Subclasses should override with optimized versions for bivectors
        result = torch.zeros_like(mv)
        result[..., 0] = 1.0  # Start with identity (scalar = 1)

        term = mv.clone()
        for k in range(1, 12):  # 12 terms for reasonable precision
            result = result + term / float(_factorial(k))
            term = self.geometric_product(term, mv)

        return result

    # =========================================================================
    # Rotor Accelerated Operations
    # =========================================================================

    @abstractmethod
    def compose_rotor(self, r1: Tensor, r2: Tensor) -> Tensor:
        """
        Rotor composition r1 r2 (accelerated geometric product for rotors).

        Args:
            r1: shape (..., count_rotor)
            r2: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def reverse_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor reversion r̃ (accelerated).

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def sandwich_rotor(self, r: Tensor, x: Tensor) -> Tensor:
        """
        Rotor sandwich product rxr̃ (accelerated).

        Args:
            r: shape (..., count_rotor)
            x: shape (..., count_point) or (..., count_blade)

        Returns:
            same shape as x
        """
        ...

    @abstractmethod
    def norm_squared_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor norm squared (accelerated).

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., 1)
        """
        ...

    def inverse_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor inverse r⁻¹ = r̃ / |r|².

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        rev = self.reverse_rotor(r)
        norm_sq = self.norm_squared_rotor(r)
        return rev / (norm_sq + 1e-12)

    def normalize_rotor(self, r: Tensor) -> Tensor:
        """
        Normalize rotor to unit norm.

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        norm_sq = self.norm_squared_rotor(r)
        norm = torch.sqrt(torch.abs(norm_sq) + 1e-12)
        return r / norm

    # =========================================================================
    # Rotor-Specific Operations
    # =========================================================================

    @abstractmethod
    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Bivector exponential map exp(B) → Rotor.

        For a simple bivector B with B² = -|B|²:
            exp(B) = cos(|B|) + sin(|B|)/|B| * B

        Args:
            B: shape (..., count_bivector)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def log_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor logarithm log(r) → Bivector.

        Inverse of exp_bivector.

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_bivector)
        """
        ...

    def slerp_rotor(self, r1: Tensor, r2: Tensor, t: float) -> Tensor:
        """
        Spherical linear interpolation between rotors.

        slerp(r1, r2, t) = r1 * exp(t * log(r1⁻¹ * r2))

        Args:
            r1: shape (..., count_rotor)
            r2: shape (..., count_rotor)
            t: interpolation parameter [0, 1]

        Returns:
            shape (..., count_rotor)
        """
        # r1_inv * r2
        r1_inv = self.inverse_rotor(r1)
        r_delta = self.compose_rotor(r1_inv, r2)

        # log(r_delta) -> scale by t -> exp
        log_delta = self.log_rotor(r_delta)
        log_scaled = log_delta * t
        r_interp = self.exp_bivector(log_scaled)

        # r1 * r_interp
        return self.compose_rotor(r1, r_interp)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    def multivector(self, data: Tensor) -> "Multivector":
        """
        Create a Multivector wrapper.

        Args:
            data: shape (..., count_blade)

        Returns:
            Multivector instance
        """
        from .multivector import Multivector

        return Multivector(data, self)

    def rotor(self, data: Tensor) -> "Rotor":
        """
        Create a Rotor wrapper.

        Args:
            data: shape (..., count_rotor)

        Returns:
            Rotor instance
        """
        from .multivector import Rotor

        return Rotor(data, self)

    # =========================================================================
    # Layer Factory (will be implemented in layers.py)
    # =========================================================================

    def get_transform_layer(self) -> "torch.nn.Module":
        """Get sandwich product layer."""
        from .layers import CliffordTransformLayer

        return CliffordTransformLayer(self)

    def get_encoder(self) -> "torch.nn.Module":
        """Get encoder layer (for CGA/VGA specializations)."""
        raise NotImplementedError(
            "Encoder only available for CGA/VGA specializations"
        )

    def get_decoder(self) -> "torch.nn.Module":
        """Get decoder layer (for CGA/VGA specializations)."""
        raise NotImplementedError(
            "Decoder only available for CGA/VGA specializations"
        )


def _factorial(n: int) -> int:
    """Compute factorial for Taylor series."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
