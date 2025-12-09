"""
fast-clifford: High-performance CGA code generator for PyTorch

This package provides:
- Code generation for Clifford algebra operations
- Optimized sparse implementations for CGA (Conformal Geometric Algebra)
- ONNX-exportable PyTorch modules for TensorRT deployment

Supported CGA algebras:
- cga0d: Cl(1,1) - 0D Conformal Geometric Algebra (4 blades)
- cga1d: Cl(2,1) - 1D Conformal Geometric Algebra (8 blades)
- cga2d: Cl(3,1) - 2D Conformal Geometric Algebra (16 blades)
- cga3d: Cl(4,1) - 3D Conformal Geometric Algebra (32 blades)
- cga4d: Cl(5,1) - 4D Conformal Geometric Algebra (64 blades)
- cga5d: Cl(6,1) - 5D Conformal Geometric Algebra (128 blades)
- CGA(6+): Runtime computed for higher dimensions

Unified interface:
- CGA(n): Create CGA algebra by Euclidean dimension
- Cl(p, q, r): Create Clifford algebra by signature

Extended Operations (Feature 005):
- compose(): EvenVersor or Similitude composition
- sandwich_product(): Transform points/multivectors
- inner_product(): Scalar product <a*b>_0
- outer_product(): Wedge product a ^ b
- left_contraction(): a left-contract b
- right_contraction(): a right-contract b
- grade_select(): Extract grade-k component
- dual(): Poincare duality
- normalize(): Unit normalization
- exp_bivector(): Bivector exponential map
"""

__version__ = "0.1.0"

# Expose all CGA algebras
from .algebras import cga0d, cga1d, cga2d, cga3d, cga4d, cga5d

# Expose unified interface
from .cga import CGA, Cl, CGAAlgebraBase, Multivector, EvenVersor, Similitude

__all__ = [
    # Version
    "__version__",
    # Unified interface
    "CGA",
    "Cl",
    "CGAAlgebraBase",
    # Multivector classes
    "Multivector",
    "EvenVersor",
    "Similitude",
    # Direct module access
    "cga0d",
    "cga1d",
    "cga2d",
    "cga3d",
    "cga4d",
    "cga5d",
]
