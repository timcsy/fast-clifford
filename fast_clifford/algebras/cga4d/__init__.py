"""
CGA4D - Conformal Geometric Algebra for 4D Euclidean Space

CGA4D uses Cl(5,1) algebra with signature (+,+,+,+,+,-):
- 64 blades total
- EvenVersor: 31 components (Grade 0 + 2 + 4)
- CGA Point: 6 components (Grade 1)

Main classes:
- CliffordTransformLayer: PyTorch layer for EvenVersor × point × EvenVersor̃ transformation
- CGAEncoder: Encode 4D coordinates to CGA representation
- CGADecoder: Decode CGA representation to 4D coordinates
- CGAPipeline: Complete encode → transform → decode pipeline

Example:
    >>> import torch
    >>> from fast_clifford.algebras.cga4d.layers import CliffordTransformLayer, CGAEncoder, CGADecoder
    >>>
    >>> # Encode 4D points
    >>> encoder = CGAEncoder()
    >>> points_4d = torch.randn(batch_size, 4)
    >>> cga_points = encoder(points_4d)  # shape: (batch_size, 6)
    >>>
    >>> # Apply EvenVersor transformation
    >>> layer = CliffordTransformLayer()
    >>> evs = torch.randn(batch_size, 31)
    >>> transformed = layer(evs, cga_points)  # shape: (batch_size, 6)
    >>>
    >>> # Decode back to 4D
    >>> decoder = CGADecoder()
    >>> result_4d = decoder(transformed)  # shape: (batch_size, 4)
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
    GRADE_6_INDICES,
    POINT_MASK,
    EVEN_VERSOR_MASK,
    REVERSE_SIGNS,
    EVEN_VERSOR_REVERSE_SIGNS,
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
    "GRADE_6_INDICES",
    "POINT_MASK",
    "EVEN_VERSOR_MASK",
    "REVERSE_SIGNS",
    "EVEN_VERSOR_REVERSE_SIGNS",
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
