"""
fast-clifford: High-performance CGA code generator for PyTorch

This package provides:
- Code generation for Clifford algebra operations
- Optimized sparse implementations for CGA (Conformal Geometric Algebra)
- ONNX-exportable PyTorch modules for TensorRT deployment
"""

__version__ = "0.1.0"

# Expose cga3d as the primary API
from .algebras import cga3d

__all__ = ["cga3d", "__version__"]
