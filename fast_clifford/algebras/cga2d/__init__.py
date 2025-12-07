"""
CGA2D - 2D Conformal Geometric Algebra Cl(3,1)

This module provides high-performance CGA2D operations for PyTorch,
optimized for ONNX export and TensorRT deployment.

Algebra properties:
- Base space: 2D Euclidean (e1, e2)
- Conformal basis: e+, e-
- Signature: (+, +, +, -)
- Total blades: 16
- UPGC Point: 4 components
- Motor: 8 components
"""

from .algebra import (
    EUCLIDEAN_DIM,
    BLADE_COUNT,
    SIGNATURE,
    GRADE_INDICES,
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
    GRADE_4_INDICES,
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
    up,
    down,
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
    CGA2DCareLayer,
    UPGC2DEncoder,
    UPGC2DDecoder,
    CGA2DTransformPipeline,
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
    "GRADE_3_INDICES",
    "GRADE_4_INDICES",
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
    "up",
    "down",
    # PyTorch operations
    "geometric_product_full",
    "reverse_full",
    "upgc_encode",
    "upgc_decode",
    "reverse_motor",
    "sandwich_product_sparse",
    # PyTorch layers
    "CGA2DCareLayer",
    "UPGC2DEncoder",
    "UPGC2DDecoder",
    "CGA2DTransformPipeline",
]
