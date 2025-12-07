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
"""

__version__ = "0.1.0"

# Expose all CGA algebras
from .algebras import cga1d, cga2d, cga3d

__all__ = ["cga1d", "cga2d", "cga3d", "__version__"]
