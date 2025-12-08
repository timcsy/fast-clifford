"""
CGA0D - 0D Conformal Geometric Algebra Cl(1,1)

This module provides high-performance CGA0D operations for PyTorch,
optimized for ONNX export and TensorRT deployment.

Algebra properties:
- Base space: 0D Euclidean (no Euclidean basis)
- Conformal basis: e+, e-
- Signature: (+, -)
- Total blades: 4
- UPGC Point: 2 components
- Motor: 2 components
"""

from .algebra import (
    EUCLIDEAN_DIM,
    BLADE_COUNT,
    SIGNATURE,
    GRADE_INDICES,
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    UPGC_POINT_MASK,
    MOTOR_MASK,
    REVERSE_SIGNS,
    MOTOR_REVERSE_SIGNS,
    BLADE_NAMES,
    get_layout,
    get_blades,
    get_stuff,
    get_product_table,
    get_null_basis,
    verify_null_basis,
    get_blade_grade,
    get_blade_info,
)

from .functional import (
    geometric_product_full,
    reverse_full,
    upgc_encode,
    upgc_decode,
    reverse_motor,
    sandwich_product_sparse,
)

from .layers import (
    CGA0DCareLayer,
    UPGC0DEncoder,
    UPGC0DDecoder,
    CGA0DTransformPipeline,
)

__all__ = [
    # Constants
    "EUCLIDEAN_DIM",
    "BLADE_COUNT",
    "SIGNATURE",
    "GRADE_INDICES",
    "GRADE_0_INDICES",
    "GRADE_1_INDICES",
    "GRADE_2_INDICES",
    "UPGC_POINT_MASK",
    "MOTOR_MASK",
    "REVERSE_SIGNS",
    "MOTOR_REVERSE_SIGNS",
    "BLADE_NAMES",
    # Algebra functions
    "get_layout",
    "get_blades",
    "get_stuff",
    "get_product_table",
    "get_null_basis",
    "verify_null_basis",
    "get_blade_grade",
    "get_blade_info",
    # PyTorch operations
    "geometric_product_full",
    "reverse_full",
    "upgc_encode",
    "upgc_decode",
    "reverse_motor",
    "sandwich_product_sparse",
    # PyTorch layers
    "CGA0DCareLayer",
    "UPGC0DEncoder",
    "UPGC0DDecoder",
    "CGA0DTransformPipeline",
]
