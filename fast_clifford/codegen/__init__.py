"""
Code generation framework for Clifford algebras

This module provides tools for generating optimized PyTorch code
for any Clifford algebra Cl(p,q,r).

Usage:
    from fast_clifford.codegen.generator import generate_algebra_module
    generate_algebra_module(3, 0)  # Generates VGA(3) = Cl(3,0)
    generate_algebra_module(4, 1)  # Generates CGA(3) = Cl(4,1)
"""

from .generator import ClCodeGenerator, generate_algebra_module
from .clifford_factory import ClFactory

__all__ = [
    "ClCodeGenerator",
    "generate_algebra_module",
    "ClFactory",
]
