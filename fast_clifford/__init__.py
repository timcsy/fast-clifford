"""
fast-clifford: High-performance Clifford algebra code generator for PyTorch

This package provides:
- Code generation for Clifford algebra operations
- Optimized implementations for VGA, CGA, and general Cl(p,q) algebras
- ONNX-exportable PyTorch modules for TensorRT deployment

Factory Functions:
- Cl(p, q, r=0): Create arbitrary Clifford algebra Cl(p, q, r)
- VGA(n): Create VGA(n) = Cl(n, 0) - Vanilla Geometric Algebra
- CGA(n): Create CGA(n) = Cl(n+1, 1) - Conformal Geometric Algebra (coming soon)
- PGA(n): Create PGA(n) = Cl(n, 0, 1) - Projective Geometric Algebra (coming soon)

Example:
    >>> from fast_clifford import Cl, VGA
    >>> vga3d = VGA(3)  # Cl(3, 0)
    >>> algebra = Cl(2, 2)  # General Cl(2, 2)

Note:
    This is version 0.2.0 with the unified Clifford algebra interface.
    Breaking change: Old cga0d-cga5d modules have been removed.
"""

__version__ = "0.2.0"

# Import from unified interface
from .clifford import (
    Cl,
    VGA,
    CGA,
    PGA,
    CliffordAlgebraBase,
    HardcodedClWrapper,
    Multivector,
    Rotor,
    VGAWrapper,
    CGAWrapper,
    PGAEmbedding,
)

__all__ = [
    "__version__",
    # Factory functions
    "Cl",
    "VGA",
    "CGA",
    "PGA",
    # Base classes
    "CliffordAlgebraBase",
    "HardcodedClWrapper",
    # Wrapper classes
    "Multivector",
    "Rotor",
    "VGAWrapper",
    "CGAWrapper",
    "PGAEmbedding",
]
