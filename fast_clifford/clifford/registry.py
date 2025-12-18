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
