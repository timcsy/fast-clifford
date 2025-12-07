"""
CGA5D (Conformal Geometric Algebra) Cl(6,1) Module.

This module provides:
- CGA5D algebra definition and product tables
- Sparse functional operations for PyTorch
- nn.Module layers for Transformer integration

CGA5D specifications:
- Signature: (+,+,+,+,+,+,-) = 6 positive, 1 negative
- 128 blades total (2^7)
- UPGC Point: 7 components (Grade 1)
- Motor: 64 components (Grade 0 + 2 + 4 + 6)
"""

from .layers import (
    CGA5DCareLayer,
    UPGC5DEncoder,
    UPGC5DDecoder,
    CGA5DTransformPipeline,
)

from .functional import (
    geometric_product_full,
    reverse_full,
    sandwich_product_sparse,
    reverse_motor,
    upgc_encode,
    upgc_decode,
)

from .algebra import (
    EUCLIDEAN_DIM,
    BLADE_COUNT,
    GRADE_INDICES,
    UPGC_POINT_MASK,
    MOTOR_MASK,
    MOTOR_SPARSE_INDICES,
    REVERSE_SIGNS,
    MOTOR_REVERSE_SIGNS,
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

__all__ = [
    # Layers
    "CGA5DCareLayer",
    "UPGC5DEncoder",
    "UPGC5DDecoder",
    "CGA5DTransformPipeline",
    # Functional
    "geometric_product_full",
    "reverse_full",
    "sandwich_product_sparse",
    "reverse_motor",
    "upgc_encode",
    "upgc_decode",
    # Algebra
    "EUCLIDEAN_DIM",
    "BLADE_COUNT",
    "GRADE_INDICES",
    "UPGC_POINT_MASK",
    "MOTOR_MASK",
    "MOTOR_SPARSE_INDICES",
    "REVERSE_SIGNS",
    "MOTOR_REVERSE_SIGNS",
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
]
