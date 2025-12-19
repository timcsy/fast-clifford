"""
Clifford Algebra Registry - HardcodedClWrapper

Maps generated hardcoded algebras to the unified CliffordAlgebraBase interface.

For p+q <= 9 (blade_count <= 512), uses pre-generated hardcoded implementations.
For higher dimensions, falls back to Bott periodicity or runtime computation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Literal
import importlib
import torch
from torch import Tensor

from .base import CliffordAlgebraBase


class HardcodedClWrapper(CliffordAlgebraBase):
    """
    Wrapper for pre-generated hardcoded Clifford algebras.

    Provides the unified CliffordAlgebraBase interface backed by
    auto-generated loop-free PyTorch functions.

    Attributes:
        p: Positive signature dimension
        q: Negative signature dimension
        _module: The loaded generated algebra module
    """

    def __init__(self, p: int, q: int):
        """
        Initialize wrapper for Cl(p, q).

        Args:
            p: Positive signature dimension
            q: Negative signature dimension

        Raises:
            ValueError: If the requested algebra is not available
        """
        self._p = p
        self._q = q
        self._r = 0

        # Try to load the generated module
        module_name = f"fast_clifford.algebras.generated.cl_{p}_{q}"
        try:
            self._module = importlib.import_module(module_name)
        except ImportError:
            raise ValueError(
                f"No pre-generated algebra found for Cl({p}, {q}). "
                f"Expected module: {module_name}"
            )

        # Cache frequently used values from module
        self._blade_count = self._module.BLADE_COUNT
        self._rotor_count = self._module.ROTOR_COUNT
        self._bivector_count = self._module.BIVECTOR_COUNT
        self._rotor_mask = self._module.ROTOR_MASK
        self._bivector_mask = self._module.BIVECTOR_MASK

    # =========================================================================
    # Signature Properties
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

    # =========================================================================
    # Count Properties
    # =========================================================================

    @property
    def count_rotor(self) -> int:
        return self._rotor_count

    @property
    def count_bivector(self) -> int:
        return self._bivector_count

    # =========================================================================
    # Core Operations
    # =========================================================================

    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        return self._module.geometric_product(a, b)

    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        return self._module.inner(a, b)

    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        return self._module.outer(a, b)

    def contract_left(self, a: Tensor, b: Tensor) -> Tensor:
        return self._module.contract_left(a, b)

    def contract_right(self, a: Tensor, b: Tensor) -> Tensor:
        return self._module.contract_right(a, b)

    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        # Regressive product: a ∨ b = (a* ∧ b*)*
        a_dual = self.dual(a)
        b_dual = self.dual(b)
        result_dual = self.outer(a_dual, b_dual)
        return self.dual(result_dual)

    # =========================================================================
    # Unary Operations
    # =========================================================================

    def reverse(self, mv: Tensor) -> Tensor:
        return self._module.reverse(mv)

    def involute(self, mv: Tensor) -> Tensor:
        return self._module.involute(mv)

    def conjugate(self, mv: Tensor) -> Tensor:
        return self._module.conjugate(mv)

    def select_grade(self, mv: Tensor, grade: int) -> Tensor:
        # Dynamically call select_grade_N function
        func_name = f"select_grade_{grade}"
        if hasattr(self._module, func_name):
            return getattr(self._module, func_name)(mv)
        else:
            # Fallback: return zeros if grade > max_grade
            return torch.zeros_like(mv)

    def dual(self, mv: Tensor) -> Tensor:
        return self._module.dual(mv)

    def norm_squared(self, mv: Tensor) -> Tensor:
        return self._module.norm_squared(mv)

    # =========================================================================
    # Rotor Operations
    # =========================================================================

    def compose_rotor(self, r1: Tensor, r2: Tensor) -> Tensor:
        return self._module.compose_rotor(r1, r2)

    def reverse_rotor(self, r: Tensor) -> Tensor:
        return self._module.reverse_rotor(r)

    def sandwich_rotor(self, r: Tensor, x: Tensor) -> Tensor:
        return self._module.sandwich_rotor(r, x)

    def norm_squared_rotor(self, r: Tensor) -> Tensor:
        return self._module.norm_squared_rotor(r)

    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Bivector exponential (hardcoded).

        Converts a bivector (generator) to a rotor (rotation).
        Formula: exp(B) = cos(θ) + sin(θ)/θ * B where θ = sqrt(-B²)

        Args:
            B: Input bivector, shape (..., count_bivector)

        Returns:
            Rotor, shape (..., count_rotor)
        """
        return self._module.exp_bivector(B)

    def log_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor logarithm (hardcoded).

        Converts a rotor (rotation) to a bivector (generator).
        Formula: log(R) = atan2(|B|, s) / |B| * B where R = s + B

        Args:
            r: Input rotor, shape (..., count_rotor)

        Returns:
            Bivector, shape (..., count_bivector)
        """
        return self._module.log_rotor(r)

    def slerp_rotor(self, r1: Tensor, r2: Tensor, t: Tensor) -> Tensor:
        """
        Spherical linear interpolation between rotors (hardcoded).

        Formula: slerp(r1, r2, t) = r1 * exp(t * log(r1~ * r2))

        Args:
            r1: Start rotor, shape (..., count_rotor)
            r2: End rotor, shape (..., count_rotor)
            t: Interpolation parameter [0, 1], shape (...) or scalar

        Returns:
            Interpolated rotor, shape (..., count_rotor)
        """
        if not isinstance(t, Tensor):
            t = torch.tensor(t, dtype=r1.dtype, device=r1.device)
        return self._module.slerp_rotor(r1, r2, t)


# =============================================================================
# Module Cache and Factory
# =============================================================================

_algebra_cache: Dict[tuple, HardcodedClWrapper] = {}


def get_hardcoded_algebra(p: int, q: int) -> Optional[HardcodedClWrapper]:
    """
    Get or create a HardcodedClWrapper for Cl(p, q).

    Args:
        p: Positive signature dimension
        q: Negative signature dimension

    Returns:
        HardcodedClWrapper instance, or None if not available
    """
    key = (p, q)
    if key in _algebra_cache:
        return _algebra_cache[key]

    try:
        wrapper = HardcodedClWrapper(p, q)
        _algebra_cache[key] = wrapper
        return wrapper
    except ValueError:
        return None


def list_available_hardcoded() -> list:
    """
    List all available pre-generated algebras.

    Returns:
        List of (p, q) tuples
    """
    from ..algebras.generated import list_available_algebras
    return list_available_algebras()


def is_hardcoded_available(p: int, q: int) -> bool:
    """
    Check if a hardcoded algebra is available.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension

    Returns:
        True if pre-generated algebra exists
    """
    module_name = f"fast_clifford.algebras.generated.cl_{p}_{q}"
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# =============================================================================
# SymmetricClWrapper - Support for p < q algebras via Cl(q, p) mapping
# =============================================================================

class SymmetricClWrapper(CliffordAlgebraBase):
    """
    Wrapper that provides Cl(p, q) interface using Cl(q, p) as the underlying algebra.

    For p < q, instead of storing separate pre-generated code for both Cl(p,q) and Cl(q,p),
    we use the algebraic isomorphism between them with a basis permutation.

    Key insight: The algebras Cl(p,q) and Cl(q,p) are isomorphic. We establish an
    isomorphism by permuting basis vectors so that signature-matching vectors align:
    - Cl(p,q) positive vectors e_1...e_p → Cl(q,p) positive vectors e_{q+1}...e_{q+p}
    - Cl(p,q) negative vectors e_{p+1}...e_{p+q} → Cl(q,p) negative vectors e_1...e_q

    Example:
        >>> # Cl(1,3): e₁²=+1, e₂²=e₃²=e₄²=-1
        >>> # Uses Cl(3,1): e₁²=e₂²=e₃²=-1, e₄²=+1
        >>> # Mapping: Cl(1,3).e1 ↔ Cl(3,1).e4, Cl(1,3).e2 ↔ Cl(3,1).e1, etc.
        >>> wrapper = SymmetricClWrapper(base_algebra=Cl(3,1), p=1, q=3)

    Note: The isomorphism preserves algebraic structure while matching signatures.
    """

    def __init__(self, base_algebra: CliffordAlgebraBase, p: int, q: int):
        """
        Initialize SymmetricClWrapper.

        Args:
            base_algebra: The underlying Cl(q, p) algebra
            p: Positive signature dimension (p < q)
            q: Negative signature dimension
        """
        if p >= q:
            raise ValueError(f"SymmetricClWrapper requires p < q, got p={p}, q={q}")

        self._base = base_algebra
        self._p = p
        self._q = q
        self._r = 0
        self._n = p + q  # Total dimension

        # Compute the basis vector permutation
        # perm[i] maps Cl(p,q) basis vector i to Cl(q,p) basis vector perm[i]
        # such that they have the same signature (both +1 or both -1)
        self._basis_perm = self._compute_basis_perm()

        # Compute blade index and sign mappings
        self._swap_indices, self._swap_signs = self._compute_blade_mapping()
        self._inverse_indices, self._inverse_signs = self._compute_inverse_mapping()

    def _compute_basis_perm(self) -> list:
        """
        Compute the basis vector permutation from Cl(p,q) to Cl(q,p).

        The key is to map vectors with matching signatures:
        - Cl(p,q): vectors 0..p-1 have +1 square, vectors p..p+q-1 have -1 square
        - Cl(q,p): vectors 0..q-1 have -1 square, vectors q..q+p-1 have +1 square

        Mapping:
        - Cl(p,q) positive [0..p-1] → Cl(q,p) positive [q..q+p-1]
        - Cl(p,q) negative [p..p+q-1] → Cl(q,p) negative [0..q-1]
        """
        p, q = self._p, self._q
        n = p + q
        perm = [0] * n

        # Map positive-signature vectors
        for i in range(p):
            perm[i] = q + i  # Cl(p,q).e_{i+1} (+1) → Cl(q,p).e_{q+i+1} (+1)

        # Map negative-signature vectors
        for j in range(q):
            perm[p + j] = j  # Cl(p,q).e_{p+j+1} (-1) → Cl(q,p).e_{j+1} (-1)

        return perm

    def _compute_blade_mapping(self) -> tuple:
        """
        Compute blade index mapping and sign corrections from Cl(p,q) to Cl(q,p).

        The sign correction accounts for reordering of basis vectors in multi-vector blades.
        """
        n = self._n
        blade_count = 2 ** n
        perm = self._basis_perm

        swap_indices = torch.zeros(blade_count, dtype=torch.long)
        swap_signs = torch.ones(blade_count, dtype=torch.float32)

        for blade_idx in range(blade_count):
            # Get basis vectors in this blade (by bit position)
            original_bits = [bit for bit in range(n) if blade_idx & (1 << bit)]

            # Compute new blade index by permuting bits
            new_blade_idx = 0
            for bit in original_bits:
                new_blade_idx |= (1 << perm[bit])
            swap_indices[blade_idx] = new_blade_idx

            # Compute sign from permutation inversions (antisymmetry of wedge product)
            if len(original_bits) > 1:
                new_positions = [perm[bit] for bit in original_bits]
                inversions = sum(1 for i in range(len(new_positions))
                               for j in range(i + 1, len(new_positions))
                               if new_positions[i] > new_positions[j])
                swap_signs[blade_idx] = (-1.0) ** inversions

        return swap_indices, swap_signs

    def _compute_inverse_mapping(self) -> tuple:
        """Compute inverse blade mapping from Cl(q,p) back to Cl(p,q)."""
        blade_count = 2 ** self._n

        inverse_indices = torch.zeros(blade_count, dtype=torch.long)
        inverse_signs = torch.ones(blade_count, dtype=torch.float32)

        for i in range(blade_count):
            j = self._swap_indices[i].item()
            inverse_indices[j] = i
            inverse_signs[j] = self._swap_signs[i]

        return inverse_indices, inverse_signs

    def _to_base(self, mv: Tensor) -> Tensor:
        """Transform multivector from Cl(p,q) to Cl(q,p) representation.

        For each component at index i in Cl(p,q), place it at index swap_indices[i] in Cl(q,p).
        """
        # Apply signs first, then gather with inverse indices
        # Since we want: result[swap_indices[i]] = mv[i] * signs[i]
        # This is equivalent to: result[j] = mv[inverse_indices[j]] * signs[inverse_indices[j]]
        # Which is: result = mv[inverse_indices] * signs[inverse_indices]
        # But inverse_indices maps j -> i, so we need the inverse of swap_indices
        #
        # Simpler approach: use the inverse mapping for gather
        # result[j] = (mv * signs)[inverse_indices[j]]
        signed_mv = mv * self._swap_signs
        return signed_mv[..., self._inverse_indices]

    def _from_base(self, mv: Tensor) -> Tensor:
        """Transform multivector from Cl(q,p) to Cl(p,q) representation.

        This is the inverse of _to_base.
        """
        # Reverse the transformation: gather using swap_indices, then apply inverse signs
        # We want: result[i] = mv[swap_indices[i]] * inverse_sign
        # Since inverse operation needs to undo the sign, we need to track what sign was applied
        return (mv[..., self._swap_indices]) * self._swap_signs

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
        return self._base.count_blade

    @property
    def count_rotor(self) -> int:
        return self._base.count_rotor

    @property
    def count_bivector(self) -> int:
        return self._base.count_bivector

    @property
    def max_grade(self) -> int:
        return self._n

    @property
    def algebra_type(self):
        if self._q == 0:
            return "vga"
        return "general"

    # =========================================================================
    # Core Operations (to_base → base op → from_base)
    # =========================================================================

    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        a_base = self._to_base(a)
        b_base = self._to_base(b)
        result = self._base.geometric_product(a_base, b_base)
        return self._from_base(result)

    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        a_base = self._to_base(a)
        b_base = self._to_base(b)
        result = self._base.outer(a_base, b_base)
        return self._from_base(result)

    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        a_base = self._to_base(a)
        b_base = self._to_base(b)
        return self._base.inner(a_base, b_base)

    def contract_left(self, a: Tensor, b: Tensor) -> Tensor:
        a_base = self._to_base(a)
        b_base = self._to_base(b)
        result = self._base.contract_left(a_base, b_base)
        return self._from_base(result)

    def contract_right(self, a: Tensor, b: Tensor) -> Tensor:
        a_base = self._to_base(a)
        b_base = self._to_base(b)
        result = self._base.contract_right(a_base, b_base)
        return self._from_base(result)

    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        a_dual = self.dual(a)
        b_dual = self.dual(b)
        result_dual = self.outer(a_dual, b_dual)
        return self.dual(result_dual)

    # =========================================================================
    # Unary Operations
    # =========================================================================

    def reverse(self, mv: Tensor) -> Tensor:
        mv_base = self._to_base(mv)
        result = self._base.reverse(mv_base)
        return self._from_base(result)

    def involute(self, mv: Tensor) -> Tensor:
        mv_base = self._to_base(mv)
        result = self._base.involute(mv_base)
        return self._from_base(result)

    def conjugate(self, mv: Tensor) -> Tensor:
        mv_base = self._to_base(mv)
        result = self._base.conjugate(mv_base)
        return self._from_base(result)

    def select_grade(self, mv: Tensor, grade: int) -> Tensor:
        mv_base = self._to_base(mv)
        result = self._base.select_grade(mv_base, grade)
        return self._from_base(result)

    def dual(self, mv: Tensor) -> Tensor:
        mv_base = self._to_base(mv)
        result = self._base.dual(mv_base)
        return self._from_base(result)

    def norm_squared(self, mv: Tensor) -> Tensor:
        mv_base = self._to_base(mv)
        return self._base.norm_squared(mv_base)

    # =========================================================================
    # Rotor Operations
    # =========================================================================

    def compose_rotor(self, r1: Tensor, r2: Tensor) -> Tensor:
        return self.geometric_product(r1, r2)

    def reverse_rotor(self, r: Tensor) -> Tensor:
        return self.reverse(r)

    def sandwich_rotor(self, r: Tensor, x: Tensor) -> Tensor:
        rx = self.geometric_product(r, x)
        r_rev = self.reverse(r)
        return self.geometric_product(rx, r_rev)

    def norm_squared_rotor(self, r: Tensor) -> Tensor:
        return self.norm_squared(r)

    def exp_bivector(self, B: Tensor) -> Tensor:
        B_base = self._to_base(B)
        result = self._base.exp_bivector(B_base)
        return self._from_base(result)

    def log_rotor(self, r: Tensor) -> Tensor:
        r_base = self._to_base(r)
        result = self._base.log_rotor(r_base)
        return self._from_base(result)

    def slerp_rotor(self, r1: Tensor, r2: Tensor, t: Tensor) -> Tensor:
        r1_base = self._to_base(r1)
        r2_base = self._to_base(r2)
        if not isinstance(t, Tensor):
            t = torch.tensor(t, dtype=r1.dtype, device=r1.device)
        result = self._base.slerp_rotor(r1_base, r2_base, t)
        return self._from_base(result)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    def scalar(self, value: float) -> Tensor:
        """Create a scalar multivector."""
        result = torch.zeros(self.count_blade)
        result[0] = value
        return result

    def basis_vector(self, index: int) -> Tensor:
        """
        Create a basis vector e_i.

        For Cl(p,q) with p < q:
        - e₁...eₚ have square +1 (positive signature)
        - eₚ₊₁...eₚ₊q have square -1 (negative signature)

        Uses lexicographic (dictionary) ordering convention:
        - e1 at index 1, e2 at index 2, ..., en at index n

        The basis vector is created in Cl(p,q) space and must be usable
        with the symmetric operations that internally swap to Cl(q,p).
        """
        if index < 1 or index > self._n:
            raise ValueError(f"Basis index must be 1 to {self._n}")

        # Create basis vector using lexicographic indexing (not bit pattern)
        # In lexicographic order, e_i is at index i (not 2^(i-1))
        result = torch.zeros(self.count_blade)
        result[index] = 1.0
        return result
