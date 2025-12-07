"""
3D Conformal Geometric Algebra Cl(4,1)

This module provides:
- CGA algebra definitions and blade indexing
- Generated hard-coded geometric product functions
- CGACareLayer for PyTorch integration
"""

from .functional import (
    # Constants
    BLADE_COUNT,
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
    GRADE_4_INDICES,
    GRADE_5_INDICES,
    UPGC_POINT_MASK,
    MOTOR_MASK,
    REVERSE_SIGNS,
    MOTOR_REVERSE_SIGNS,
    # Core functions
    geometric_product_full,
    reverse_full,
    # Sparse functions
    sandwich_product_sparse,
    reverse_motor,
    upgc_encode,
    upgc_decode,
)

from .layers import (
    CGACareLayer,
    UPGCEncoder,
    UPGCDecoder,
    CGATransformPipeline,
)

__all__ = [
    # Constants
    "BLADE_COUNT",
    "GRADE_0_INDICES",
    "GRADE_1_INDICES",
    "GRADE_2_INDICES",
    "GRADE_3_INDICES",
    "GRADE_4_INDICES",
    "GRADE_5_INDICES",
    "UPGC_POINT_MASK",
    "MOTOR_MASK",
    "REVERSE_SIGNS",
    "MOTOR_REVERSE_SIGNS",
    # Core functions
    "geometric_product_full",
    "reverse_full",
    # Sparse functions
    "sandwich_product_sparse",
    "reverse_motor",
    "upgc_encode",
    "upgc_decode",
    # Layers
    "CGACareLayer",
    "UPGCEncoder",
    "UPGCDecoder",
    "CGATransformPipeline",
]
