"""
fast-clifford: High-performance CGA code generator for PyTorch

This package provides:
- Code generation for Clifford algebra operations
- Optimized sparse implementations for CGA (Conformal Geometric Algebra)
- ONNX-exportable PyTorch modules for TensorRT deployment

Supported CGA algebras:
- cga1d: Cl(2,1) - 1D Conformal Geometric Algebra (8 blades)
- cga2d: Cl(3,1) - 2D Conformal Geometric Algebra (16 blades)
- cga3d: Cl(4,1) - 3D Conformal Geometric Algebra (32 blades)
- cga4d: Cl(5,1) - 4D Conformal Geometric Algebra (64 blades)
- cga5d: Cl(6,1) - 5D Conformal Geometric Algebra (128 blades)
"""

__version__ = "0.1.0"

# Expose all CGA algebras
from .algebras import cga1d, cga2d, cga3d, cga4d, cga5d

__all__ = ["cga1d", "cga2d", "cga3d", "cga4d", "cga5d", "__version__"]
