"""
CGA(n) Unified Interface - Conformal Geometric Algebra Factory

This module provides:
- CGA(n): Factory function to create CGA algebra by Euclidean dimension
- Cl(p, q, r): Factory function to create Clifford algebra by signature

Supported algebras:
- CGA0D-CGA5D: Hardcoded fast algorithms
- CGA6D+: Runtime computed algorithms

Example:
    >>> from fast_clifford import CGA, Cl
    >>> cga3d = CGA(3)  # CGA3D Cl(4,1,0)
    >>> cga3d = Cl(4, 1)  # Same as CGA(3)
"""

from .base import CGAAlgebraBase
from .registry import HardcodedCGAWrapper
from .multivector import Multivector, EvenVersor, Similitude
from .layers import CliffordTransformLayer, CGAEncoder, CGADecoder, CGAPipeline

__all__ = [
    "CGA",
    "Cl",
    "CGAAlgebraBase",
    "HardcodedCGAWrapper",
    "Multivector",
    "EvenVersor",
    "Similitude",
    "CliffordTransformLayer",
    "CGAEncoder",
    "CGADecoder",
    "CGAPipeline",
]


def CGA(n: int) -> CGAAlgebraBase:
    """
    Create a CGA algebra by Euclidean dimension.

    Args:
        n: Euclidean dimension (0-5 uses hardcoded, 6+ uses runtime)

    Returns:
        CGAAlgebraBase instance

    Raises:
        ValueError: If n < 0

    Examples:
        >>> cga3d = CGA(3)
        >>> cga3d.blade_count
        32
    """
    import warnings

    if n < 0:
        raise ValueError(f"Euclidean dimension must be non-negative, got {n}")

    if n >= 15:
        warnings.warn(
            f"CGA({n}) creates an algebra with {2**(n+2)} blades. "
            "This may cause memory issues.",
            UserWarning,
        )

    if n <= 5:
        # Use hardcoded fast algorithms
        return HardcodedCGAWrapper(n)
    else:
        # Use runtime algorithm
        from .runtime import RuntimeCGAAlgebra
        return RuntimeCGAAlgebra(n)


def Cl(p: int, q: int, r: int = 0) -> CGAAlgebraBase:
    """
    Create a Clifford algebra by signature Cl(p, q, r).

    Args:
        p: Number of positive-signature dimensions
        q: Number of negative-signature dimensions
        r: Number of degenerate dimensions (default 0)

    Returns:
        CGAAlgebraBase instance (or RuntimeCliffordAlgebra for non-CGA)

    Warnings:
        Emits warning if not a CGA signature (q != 1 or r != 0)

    Examples:
        >>> cga3d = Cl(4, 1, 0)  # CGA3D
        >>> cga3d = Cl(4, 1)  # Same, r=0 is default
    """
    import warnings

    if p < 0 or q < 0 or r < 0:
        raise ValueError(f"Signature components must be non-negative, got ({p}, {q}, {r})")

    # Check if this is a CGA signature
    is_cga = (q == 1 and r == 0)

    if is_cga:
        # CGA signature Cl(n+1, 1, 0) -> CGA(n)
        n = p - 1
        return CGA(n)
    else:
        # Non-CGA signature
        warnings.warn(
            f"Cl({p},{q},{r}) is not a CGA signature. "
            "Some CGA-specific operations may not be available.",
            UserWarning,
        )
        from .runtime import RuntimeCliffordAlgebra
        return RuntimeCliffordAlgebra(p, q, r)
