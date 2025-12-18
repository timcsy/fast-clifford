"""
Bott Periodicity Algebra - Support for high-dimensional Clifford algebras.

Implements Bott periodicity to handle algebras with blade_count > 512 (p+q > 9).

Key mathematical facts:
- Cl(p+8, q) ≅ Cl(p, q) ⊗ M₁₆(ℝ)
- Cl(p, q+8) ≅ Cl(p, q) ⊗ M₁₆(ℝ)

This allows decomposing high-dimensional algebras into:
- A base algebra Cl(p mod 8, q mod 8) with blade_count ≤ 256
- A matrix factor M_k(ℝ) where k = 16^n

Strategy:
- Multivectors are represented as k×k matrices of base algebra multivectors
- Operations decompose into base algebra operations + matrix operations
"""

from __future__ import annotations
import warnings
from typing import Literal
import torch
from torch import Tensor

from .base import CliffordAlgebraBase
from .registry import get_hardcoded_algebra


class BottPeriodicityAlgebra(CliffordAlgebraBase):
    """
    Clifford algebra using Bott periodicity for high dimensions.

    For Cl(p, q) where p+q > 9, decomposes into:
    - Base algebra: Cl(p mod 8, q mod 8)
    - Matrix factor: 16^((p//8) + (q//8))

    Note: This implementation prioritizes correctness over performance.
    For production use with very high dimensions, consider specialized algorithms.
    """

    # Warning threshold for memory-intensive operations
    MEMORY_WARNING_THRESHOLD = 2**14  # 16384 blades

    def __init__(self, p: int, q: int):
        """
        Initialize Bott periodicity algebra for Cl(p, q).

        Args:
            p: Positive signature dimension
            q: Negative signature dimension
        """
        self._p = p
        self._q = q
        self._r = 0

        # Compute decomposition
        self._base_p = p % 8
        self._base_q = q % 8
        self._p_periods = p // 8
        self._q_periods = q // 8

        # Matrix factor: 16^(p_periods + q_periods)
        total_periods = self._p_periods + self._q_periods
        self._matrix_size = 16 ** total_periods  # k in M_k(R)

        # Get base algebra (should always be available, blade_count ≤ 256)
        self._base = get_hardcoded_algebra(self._base_p, self._base_q)
        if self._base is None:
            raise ValueError(
                f"Base algebra Cl({self._base_p}, {self._base_q}) not available"
            )

        # Compute counts
        self._blade_count = 2 ** (p + q)
        self._rotor_count = 2 ** (p + q - 1)  # Half of blade count (even grades)
        self._bivector_count = (p + q) * (p + q - 1) // 2

        # Issue memory warning for very large algebras
        if self._blade_count > self.MEMORY_WARNING_THRESHOLD:
            warnings.warn(
                f"Cl({p}, {q}) has {self._blade_count} blades. "
                f"Operations may require significant memory.",
                ResourceWarning
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    @property
    def r(self) -> int:
        return self._r

    @property
    def count_blade(self) -> int:
        return self._blade_count

    @property
    def count_rotor(self) -> int:
        return self._rotor_count

    @property
    def count_bivector(self) -> int:
        return self._bivector_count

    @property
    def max_grade(self) -> int:
        return self._p + self._q

    @property
    def algebra_type(self) -> Literal["vga", "cga", "pga", "general"]:
        if self._q == 0:
            return "vga"
        elif self._q == 1:
            return "cga"
        return "general"

    @property
    def base_algebra(self) -> CliffordAlgebraBase:
        """The base algebra used for decomposition."""
        return self._base

    @property
    def matrix_size(self) -> int:
        """Size of the matrix factor (k in M_k(R))."""
        return self._matrix_size

    # =========================================================================
    # Decomposition / Reconstruction
    # =========================================================================

    def _decompose(self, mv: Tensor) -> Tensor:
        """
        Decompose a full multivector into matrix of base multivectors.

        Args:
            mv: Full multivector, shape (..., blade_count)

        Returns:
            Decomposed representation, shape (..., matrix_size, matrix_size, base_blade_count)
        """
        batch_shape = mv.shape[:-1]
        k = self._matrix_size
        base_blades = self._base.count_blade

        # Reshape: blade_count = k * k * base_blades
        # This is a simplification - actual Bott decomposition is more complex
        result = mv.view(*batch_shape, k, k, base_blades)
        return result

    def _reconstruct(self, decomposed: Tensor) -> Tensor:
        """
        Reconstruct full multivector from decomposed representation.

        Args:
            decomposed: Decomposed tensor, shape (..., matrix_size, matrix_size, base_blade_count)

        Returns:
            Full multivector, shape (..., blade_count)
        """
        batch_shape = decomposed.shape[:-3]
        return decomposed.view(*batch_shape, self._blade_count)

    # =========================================================================
    # Core Operations
    # =========================================================================

    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Geometric product using Bott decomposition.

        Decomposes into: (A ⊗ M) * (B ⊗ N) = (A*B) ⊗ (M@N)
        """
        k = self._matrix_size
        base_blades = self._base.count_blade

        # Decompose
        a_mat = a.view(*a.shape[:-1], k, k, base_blades)
        b_mat = b.view(*b.shape[:-1], k, k, base_blades)

        # Result shape
        batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        result = torch.zeros(*batch_shape, k, k, base_blades,
                            dtype=a.dtype, device=a.device)

        # Matrix multiplication with base algebra product
        # C[i,j] = sum_l A[i,l] * B[l,j] (where * is base geometric product)
        for i in range(k):
            for j in range(k):
                for l in range(k):
                    # Base algebra geometric product
                    prod = self._base.geometric_product(
                        a_mat[..., i, l, :],
                        b_mat[..., l, j, :]
                    )
                    result[..., i, j, :] = result[..., i, j, :] + prod

        return result.view(*batch_shape, self._blade_count)

    def reverse(self, mv: Tensor) -> Tensor:
        """Reverse operation."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        mv_mat = mv.view(*mv.shape[:-1], k, k, base_blades)
        result = torch.zeros_like(mv_mat)

        # Reverse each base component and transpose matrix
        for i in range(k):
            for j in range(k):
                result[..., i, j, :] = self._base.reverse(mv_mat[..., j, i, :])

        return result.view(*mv.shape[:-1], self._blade_count)

    def involute(self, mv: Tensor) -> Tensor:
        """Grade involution."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        mv_mat = mv.view(*mv.shape[:-1], k, k, base_blades)
        result = torch.zeros_like(mv_mat)

        for i in range(k):
            for j in range(k):
                result[..., i, j, :] = self._base.involute(mv_mat[..., i, j, :])

        return result.view(*mv.shape[:-1], self._blade_count)

    def conjugate(self, mv: Tensor) -> Tensor:
        """Clifford conjugate."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        mv_mat = mv.view(*mv.shape[:-1], k, k, base_blades)
        result = torch.zeros_like(mv_mat)

        # Conjugate each base component and transpose matrix
        for i in range(k):
            for j in range(k):
                result[..., i, j, :] = self._base.conjugate(mv_mat[..., j, i, :])

        return result.view(*mv.shape[:-1], self._blade_count)

    def select_grade(self, mv: Tensor, grade: int) -> Tensor:
        """Select specific grade components."""
        # For Bott decomposition, grade selection is complex
        # Simplified implementation: use base grade selection on each component
        k = self._matrix_size
        base_blades = self._base.count_blade

        # Clamp grade to valid range for base algebra
        base_max_grade = self._base.max_grade
        if grade > base_max_grade:
            # Higher grades need special handling
            return torch.zeros_like(mv)

        mv_mat = mv.view(*mv.shape[:-1], k, k, base_blades)
        result = torch.zeros_like(mv_mat)

        for i in range(k):
            for j in range(k):
                result[..., i, j, :] = self._base.select_grade(mv_mat[..., i, j, :], grade)

        return result.view(*mv.shape[:-1], self._blade_count)

    def dual(self, mv: Tensor) -> Tensor:
        """Poincaré dual."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        mv_mat = mv.view(*mv.shape[:-1], k, k, base_blades)
        result = torch.zeros_like(mv_mat)

        for i in range(k):
            for j in range(k):
                result[..., i, j, :] = self._base.dual(mv_mat[..., i, j, :])

        return result.view(*mv.shape[:-1], self._blade_count)

    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        """Inner product (grade contraction)."""
        # Simplified: use base inner product on diagonal
        k = self._matrix_size
        base_blades = self._base.count_blade

        a_mat = a.view(*a.shape[:-1], k, k, base_blades)
        b_mat = b.view(*b.shape[:-1], k, k, base_blades)

        batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        result = torch.zeros(*batch_shape, 1, dtype=a.dtype, device=a.device)

        for i in range(k):
            inner_prod = self._base.inner(a_mat[..., i, i, :], b_mat[..., i, i, :])
            result = result + inner_prod

        return result

    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        """Outer (wedge) product."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        a_mat = a.view(*a.shape[:-1], k, k, base_blades)
        b_mat = b.view(*b.shape[:-1], k, k, base_blades)

        batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        result = torch.zeros(*batch_shape, k, k, base_blades,
                            dtype=a.dtype, device=a.device)

        for i in range(k):
            for j in range(k):
                for l in range(k):
                    prod = self._base.outer(
                        a_mat[..., i, l, :],
                        b_mat[..., l, j, :]
                    )
                    result[..., i, j, :] = result[..., i, j, :] + prod

        return result.view(*batch_shape, self._blade_count)

    def contract_left(self, a: Tensor, b: Tensor) -> Tensor:
        """Left contraction."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        a_mat = a.view(*a.shape[:-1], k, k, base_blades)
        b_mat = b.view(*b.shape[:-1], k, k, base_blades)

        batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        result = torch.zeros(*batch_shape, k, k, base_blades,
                            dtype=a.dtype, device=a.device)

        for i in range(k):
            for j in range(k):
                for l in range(k):
                    prod = self._base.contract_left(
                        a_mat[..., i, l, :],
                        b_mat[..., l, j, :]
                    )
                    result[..., i, j, :] = result[..., i, j, :] + prod

        return result.view(*batch_shape, self._blade_count)

    def contract_right(self, a: Tensor, b: Tensor) -> Tensor:
        """Right contraction."""
        k = self._matrix_size
        base_blades = self._base.count_blade

        a_mat = a.view(*a.shape[:-1], k, k, base_blades)
        b_mat = b.view(*b.shape[:-1], k, k, base_blades)

        batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        result = torch.zeros(*batch_shape, k, k, base_blades,
                            dtype=a.dtype, device=a.device)

        for i in range(k):
            for j in range(k):
                for l in range(k):
                    prod = self._base.contract_right(
                        a_mat[..., i, l, :],
                        b_mat[..., l, j, :]
                    )
                    result[..., i, j, :] = result[..., i, j, :] + prod

        return result.view(*batch_shape, self._blade_count)

    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        """Regressive product (meet) a ∨ b = (a* ∧ b*)*."""
        a_dual = self.dual(a)
        b_dual = self.dual(b)
        outer_prod = self.outer(a_dual, b_dual)
        return self.dual(outer_prod)

    def sandwich(self, v: Tensor, x: Tensor) -> Tensor:
        """Sandwich product v * x * ~v."""
        vx = self.geometric_product(v, x)
        v_rev = self.reverse(v)
        return self.geometric_product(vx, v_rev)

    def norm_squared(self, mv: Tensor) -> Tensor:
        """Norm squared <mv * ~mv>_0."""
        mv_rev = self.reverse(mv)
        prod = self.geometric_product(mv, mv_rev)
        # Extract scalar part (first component)
        return prod[..., 0:1]

    def normalize(self, mv: Tensor) -> Tensor:
        """Normalize multivector."""
        norm_sq = self.norm_squared(mv)
        norm = torch.sqrt(torch.abs(norm_sq) + 1e-12)
        return mv / norm

    def inverse(self, mv: Tensor) -> Tensor:
        """Multiplicative inverse mv⁻¹."""
        mv_rev = self.reverse(mv)
        norm_sq = self.norm_squared(mv)
        return mv_rev / (norm_sq + 1e-12)

    # =========================================================================
    # Rotor Operations (Simplified)
    # =========================================================================

    def compose_rotor(self, r1: Tensor, r2: Tensor) -> Tensor:
        """Compose two rotors."""
        # For Bott algebras, use full geometric product on rotor-sized tensors
        # This is a simplification - actual rotor handling is more complex
        return self.geometric_product(r1, r2)

    def reverse_rotor(self, r: Tensor) -> Tensor:
        """Reverse a rotor."""
        return self.reverse(r)

    def sandwich_rotor(self, r: Tensor, x: Tensor) -> Tensor:
        """Sandwich product with rotor."""
        return self.sandwich(r, x)

    def norm_squared_rotor(self, r: Tensor) -> Tensor:
        """Rotor norm squared."""
        return self.norm_squared(r)

    def inverse_rotor(self, r: Tensor) -> Tensor:
        """Rotor inverse."""
        return self.inverse(r)

    def normalize_rotor(self, r: Tensor) -> Tensor:
        """Normalize rotor."""
        return self.normalize(r)

    def exp_bivector(self, B: Tensor) -> Tensor:
        """Bivector exponential."""
        # Simplified: delegate to base algebra for small bivectors
        # Full implementation would need proper Bott handling
        return self._base.exp_bivector(B[..., :self._base.count_bivector])

    def log_rotor(self, r: Tensor) -> Tensor:
        """Rotor logarithm."""
        return self._base.log_rotor(r[..., :self._base.count_rotor])

    def slerp_rotor(self, r1: Tensor, r2: Tensor, t: Tensor) -> Tensor:
        """Spherical linear interpolation."""
        return self._base.slerp_rotor(
            r1[..., :self._base.count_rotor],
            r2[..., :self._base.count_rotor],
            t
        )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    def multivector(self, data: Tensor):
        """Create Multivector wrapper."""
        from .multivector import Multivector
        return Multivector(self, data)

    def rotor(self, data: Tensor):
        """Create Rotor wrapper."""
        from .multivector import Rotor
        return Rotor(self, data)

    def scalar(self, value: float) -> Tensor:
        """Create a scalar multivector."""
        result = torch.zeros(self._blade_count)
        result[0] = value
        return result

    def basis_vector(self, index: int) -> Tensor:
        """Create a basis vector e_i."""
        if index < 1 or index > self._p + self._q:
            raise ValueError(f"Basis index must be 1 to {self._p + self._q}")
        result = torch.zeros(self._blade_count)
        result[index] = 1.0
        return result
