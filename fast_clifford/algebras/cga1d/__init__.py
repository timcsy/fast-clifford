"""
CGA1D - 1D Conformal Geometric Algebra Cl(2,1)

This module provides high-performance CGA1D operations for PyTorch,
optimized for ONNX export and TensorRT deployment.

Algebra properties:
- Base space: 1D Euclidean (e1)
- Conformal basis: e+, e-
- Signature: (+, +, -)
- Total blades: 8
- CGA Point: 3 components
- EvenVersor: 4 components
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
    POINT_MASK,
    EVEN_VERSOR_MASK,
    REVERSE_SIGNS,
    EVEN_VERSOR_REVERSE_SIGNS,
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
    cga_encode,
    cga_decode,
    reverse_even_versor,
    sandwich_product_sparse,
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
    "EUCLIDEAN_DIM",
    "BLADE_COUNT",
    "SIGNATURE",
    "GRADE_INDICES",
    "GRADE_0_INDICES",
    "GRADE_1_INDICES",
    "GRADE_2_INDICES",
    "GRADE_3_INDICES",
    "POINT_MASK",
    "EVEN_VERSOR_MASK",
    "REVERSE_SIGNS",
    "EVEN_VERSOR_REVERSE_SIGNS",
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
    # Core operations
    "geometric_product_full",
    "reverse_full",
    "cga_encode",
    "cga_decode",
    "reverse_even_versor",
    "sandwich_product_sparse",
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
