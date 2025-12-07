"""
CGA4D - Conformal Geometric Algebra for 4D Euclidean Space

CGA4D uses Cl(5,1) algebra with signature (+,+,+,+,+,-):
- 64 blades total
- Motor: 31 components (Grade 0 + 2 + 4)
- UPGC Point: 6 components (Grade 1)

Main classes:
- CGA4DCareLayer: PyTorch layer for motor × point × motor̃ transformation
- UPGC4DEncoder: Encode 4D coordinates to UPGC representation
- UPGC4DDecoder: Decode UPGC representation to 4D coordinates
- CGA4DTransformPipeline: Complete encode → transform → decode pipeline

Example:
    >>> import torch
    >>> from fast_clifford.algebras.cga4d import CGA4DCareLayer, UPGC4DEncoder, UPGC4DDecoder
    >>>
    >>> # Encode 4D points
    >>> encoder = UPGC4DEncoder()
    >>> points_4d = torch.randn(batch_size, 4)
    >>> upgc_points = encoder(points_4d)  # shape: (batch_size, 6)
    >>>
    >>> # Apply motor transformation
    >>> layer = CGA4DCareLayer()
    >>> motors = torch.randn(batch_size, 31)
    >>> transformed = layer(motors, upgc_points)  # shape: (batch_size, 6)
    >>>
    >>> # Decode back to 4D
    >>> decoder = UPGC4DDecoder()
    >>> result_4d = decoder(transformed)  # shape: (batch_size, 4)
"""

from .layers import (
    CGA4DCareLayer,
    UPGC4DEncoder,
    UPGC4DDecoder,
    CGA4DTransformPipeline,
)

from .functional import (
    geometric_product_full,
    reverse_full,
    sandwich_product_sparse,
    reverse_motor,
    upgc_encode,
    upgc_decode,
    BLADE_COUNT,
    EUCLIDEAN_DIM,
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
    GRADE_4_INDICES,
    GRADE_5_INDICES,
    GRADE_6_INDICES,
    UPGC_POINT_MASK,
    MOTOR_MASK,
    REVERSE_SIGNS,
    MOTOR_REVERSE_SIGNS,
)

__all__ = [
    # Layers
    "CGA4DCareLayer",
    "UPGC4DEncoder",
    "UPGC4DDecoder",
    "CGA4DTransformPipeline",
    # Functional
    "geometric_product_full",
    "reverse_full",
    "sandwich_product_sparse",
    "reverse_motor",
    "upgc_encode",
    "upgc_decode",
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
    "UPGC_POINT_MASK",
    "MOTOR_MASK",
    "REVERSE_SIGNS",
    "MOTOR_REVERSE_SIGNS",
]
