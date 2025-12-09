"""
3D Conformal Geometric Algebra Cl(4,1)

This module provides:
- CGA algebra definitions and blade indexing
- Generated hard-coded geometric product functions
- Extended operations (compose, inner product, exp, etc.)
"""

from .functional import (
    # Constants
    BLADE_COUNT,
    EUCLIDEAN_DIM,
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
    GRADE_4_INDICES,
    GRADE_5_INDICES,
    GRADE_0_MASK,
    GRADE_1_MASK,
    GRADE_2_MASK,
    GRADE_3_MASK,
    GRADE_4_MASK,
    GRADE_5_MASK,
    POINT_MASK,
    EVEN_VERSOR_MASK,
    REVERSE_SIGNS,
    EVEN_VERSOR_REVERSE_SIGNS,
    ROTOR_INDICES,
    TRANSLATION_PAIRS,
    DILATION_INDEX,
    # Core functions
    geometric_product_full,
    reverse_full,
    # Sparse functions
    sandwich_product_sparse,
    reverse_even_versor,
    cga_encode,
    cga_decode,
    # Extended operations
    compose_even_versor,
    compose_similitude,
    sandwich_product_similitude,
    inner_product_full,
    outer_product_full,
    left_contraction_full,
    right_contraction_full,
    grade_select,
    dual,
    normalize,
    norm_squared,
    exp_bivector,
    bivector_squared_scalar,
    structure_normalize,
    soft_structure_normalize,
)

from . import functional

__all__ = [
    # Module
    "functional",
    # Constants
    "BLADE_COUNT",
    "EUCLIDEAN_DIM",
    "GRADE_0_INDICES",
    "GRADE_1_INDICES",
    "GRADE_2_INDICES",
    "GRADE_3_INDICES",
    "GRADE_4_INDICES",
    "GRADE_5_INDICES",
    "GRADE_0_MASK",
    "GRADE_1_MASK",
    "GRADE_2_MASK",
    "GRADE_3_MASK",
    "GRADE_4_MASK",
    "GRADE_5_MASK",
    "POINT_MASK",
    "EVEN_VERSOR_MASK",
    "REVERSE_SIGNS",
    "EVEN_VERSOR_REVERSE_SIGNS",
    "ROTOR_INDICES",
    "TRANSLATION_PAIRS",
    "DILATION_INDEX",
    # Core functions
    "geometric_product_full",
    "reverse_full",
    # Sparse functions
    "sandwich_product_sparse",
    "reverse_even_versor",
    "cga_encode",
    "cga_decode",
    # Extended operations
    "compose_even_versor",
    "compose_similitude",
    "sandwich_product_similitude",
    "inner_product_full",
    "outer_product_full",
    "left_contraction_full",
    "right_contraction_full",
    "grade_select",
    "dual",
    "normalize",
    "norm_squared",
    "exp_bivector",
    "bivector_squared_scalar",
    "structure_normalize",
    "soft_structure_normalize",
]
