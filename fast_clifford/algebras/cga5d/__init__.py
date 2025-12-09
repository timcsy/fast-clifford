"""
CGA5D (Conformal Geometric Algebra) Cl(6,1) Module.

This module provides:
- CGA5D algebra definition and product tables
- Sparse functional operations for PyTorch
- nn.Module layers for Transformer integration

CGA5D specifications:
- Signature: (+,+,+,+,+,+,-) = 6 positive, 1 negative
- 128 blades total (2^7)
- CGA Point: 7 components (Grade 1)
- EvenVersor: 64 components (Grade 0 + 2 + 4 + 6)
"""

from .functional import (
    # Core operations
    geometric_product_full,
    reverse_full,
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

from .algebra import (
    EUCLIDEAN_DIM,
    BLADE_COUNT,
    GRADE_INDICES,
    POINT_MASK,
    EVEN_VERSOR_MASK,
    EVEN_VERSOR_SPARSE_INDICES,
    REVERSE_SIGNS,
    EVEN_VERSOR_REVERSE_SIGNS,
    get_layout,
    get_blades,
    get_stuff,
    get_null_basis,
    verify_null_basis_properties,
    get_product_table,
    get_blade_grade,
    get_blade_names,
    up,
    down,
)

from . import functional

__all__ = [
    # Module
    "functional",
    # Algebra constants
    "EUCLIDEAN_DIM",
    "BLADE_COUNT",
    "GRADE_INDICES",
    "POINT_MASK",
    "EVEN_VERSOR_MASK",
    "EVEN_VERSOR_SPARSE_INDICES",
    "REVERSE_SIGNS",
    "EVEN_VERSOR_REVERSE_SIGNS",
    # Algebra functions
    "get_layout",
    "get_blades",
    "get_stuff",
    "get_null_basis",
    "verify_null_basis_properties",
    "get_product_table",
    "get_blade_grade",
    "get_blade_names",
    "up",
    "down",
    # Core operations
    "geometric_product_full",
    "reverse_full",
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
