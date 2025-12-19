"""
Unified Clifford Algebra Interface

This module provides a unified interface for working with Clifford algebras Cl(p,q,r).

Factory Functions:
- Cl(p, q, r=0): Create arbitrary Clifford algebra
- VGA(n): Create VGA(n) = Cl(n, 0)
- CGA(n): Create CGA(n) = Cl(n+1, 1)
- PGA(n): Create PGA(n) = Cl(n, 0, 1)

Example:
    >>> from fast_clifford.clifford import Cl, VGA
    >>> vga3d = VGA(3)  # Cl(3, 0)
    >>> algebra = Cl(2, 2)  # General Cl(2, 2)
"""

from typing import Union

from .base import CliffordAlgebraBase
from .registry import HardcodedClWrapper, get_hardcoded_algebra, is_hardcoded_available
from .bott import BottPeriodicityAlgebra
from .multivector import Multivector, Rotor
from .specializations.vga import VGAWrapper
from .specializations.cga import CGAWrapper
from .specializations.pga import PGAEmbedding


# Threshold for using hardcoded vs Bott periodicity
# After 007-bott-optimization: All 36 algebras with p+q < 8 are hardcoded (128 blades max)
HARDCODED_THRESHOLD = 128  # p+q < 8


def Cl(p: int, q: int = 0, r: int = 0) -> CliffordAlgebraBase:
    """
    Create a Clifford algebra Cl(p, q, r).

    Args:
        p: Positive signature dimension (e_i² = +1)
        q: Negative signature dimension (e_i² = -1)
        r: Degenerate dimension (e_i² = 0), must be 0 for now

    Returns:
        CliffordAlgebraBase instance

    Routing:
        - blade_count <= 128 (p+q < 8): Uses pre-generated hardcoded algebra
        - blade_count > 128 (p+q >= 8): Uses Bott periodicity decomposition

    Raises:
        ValueError: If r != 0 (not yet supported)
        ValueError: If no implementation available for (p, q)

    Example:
        >>> algebra = Cl(3, 0)   # VGA(3) - hardcoded
        >>> algebra = Cl(4, 1)   # CGA(3) - hardcoded
        >>> algebra = Cl(1, 3)   # hardcoded
        >>> algebra = Cl(10, 0)  # VGA(10) - Bott periodicity
    """
    if r != 0:
        raise ValueError(
            f"Degenerate signatures (r={r}) not yet supported. "
            "Use r=0 for hardcoded algebras."
        )

    blade_count = 2 ** (p + q)

    # For blade_count <= 128 (p+q < 8), use hardcoded algebra
    if blade_count <= HARDCODED_THRESHOLD:
        algebra = get_hardcoded_algebra(p, q)
        if algebra is not None:
            return algebra
        else:
            raise ValueError(
                f"No pre-generated algebra for Cl({p}, {q}). "
                "Generate it first using ClCodeGenerator."
            )
    else:
        # For high dimensions (p+q >= 8), use Bott periodicity
        return BottPeriodicityAlgebra(p, q)


def VGA(n: int) -> VGAWrapper:
    """
    Create VGA(n) = Cl(n, 0) - Vanilla Geometric Algebra.

    VGA is pure Euclidean vector algebra with signature (+1, +1, ..., +1).

    Args:
        n: Euclidean dimension

    Returns:
        VGAWrapper instance

    Example:
        >>> vga = VGA(3)  # 3D VGA
        >>> v = vga.encode(torch.tensor([1., 2., 3.]))
    """
    base_algebra = Cl(n, 0, 0)
    return VGAWrapper(base_algebra)


def CGA(n: int) -> CGAWrapper:
    """
    Create CGA(n) = Cl(n+1, 1) - Conformal Geometric Algebra.

    CGA embeds n-dimensional Euclidean space into a (n+2)-dimensional
    conformal space, enabling unified treatment of rotations, translations,
    and dilations as versors.

    Args:
        n: Euclidean dimension

    Returns:
        CGAWrapper instance

    Example:
        >>> cga = CGA(3)  # 3D CGA = Cl(4, 1)
        >>> point = cga.encode(torch.tensor([1., 2., 3.]))
        >>> x = cga.decode(point)
    """
    # CGA(n) = Cl(n+1, 1)
    base_algebra = Cl(n + 1, 1, 0)
    return CGAWrapper(base_algebra, euclidean_dim=n)


def PGA(n: int) -> PGAEmbedding:
    """
    Create PGA(n) = Cl(n, 0, 1) - Projective Geometric Algebra.

    PGA represents n-dimensional projective geometry using a degenerate
    Clifford algebra. This implementation embeds PGA into CGA for
    leveraging hardcoded operations.

    Args:
        n: Euclidean dimension

    Returns:
        PGAEmbedding instance

    Example:
        >>> pga = PGA(3)  # 3D PGA = Cl(3, 0, 1)
        >>> point = pga.point(torch.tensor([1., 2., 3.]))
        >>> plane = pga.plane(torch.tensor([0., 0., 1., -5.]))
    """
    return PGAEmbedding(n)


__all__ = [
    # Factory functions
    "Cl",
    "VGA",
    "CGA",
    "PGA",
    # Base classes
    "CliffordAlgebraBase",
    "HardcodedClWrapper",
    "BottPeriodicityAlgebra",
    # Wrapper classes
    "Multivector",
    "Rotor",
    "VGAWrapper",
    "CGAWrapper",
    "PGAEmbedding",
]
