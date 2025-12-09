"""
Unified CGA Layer Classes

Provides dimension-agnostic PyTorch layers for CGA operations:
- CliffordTransformLayer: Sandwich product layer (M x X x ~M)
- CGAEncoder: Euclidean to CGA encoding
- CGADecoder: CGA to Euclidean decoding
- CGAPipeline: Complete encode-transform-decode pipeline
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .base import CGAAlgebraBase


class CliffordTransformLayer(nn.Module):
    """
    Unified Clifford transform layer (sandwich product).

    Computes M x X x ~M where M is an EvenVersor/Similitude
    and X is a CGA point.

    Args:
        algebra: CGA algebra instance
        versor_type: 'even_versor', 'similitude', or 'auto'
        use_fp32: Whether to use fp32 for computation
    """

    def __init__(
        self,
        algebra: 'CGAAlgebraBase',
        versor_type: Literal['even_versor', 'similitude', 'auto'] = 'auto',
        use_fp32: bool = True
    ):
        super().__init__()
        self.algebra = algebra
        self.versor_type = versor_type
        self.use_fp32 = use_fp32

    def forward(self, versor: Tensor, point: Tensor) -> Tensor:
        """
        Apply sandwich product transformation.

        Args:
            versor: EvenVersor/Similitude, shape (..., even_versor_count)
            point: CGA point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        original_dtype = point.dtype

        if self.use_fp32:
            versor = versor.to(torch.float32)
            point = point.to(torch.float32)

        if self.versor_type == 'similitude':
            result = self.algebra.sandwich_product_similitude(versor, point)
        else:
            result = self.algebra.sandwich_product_sparse(versor, point)

        return result.to(original_dtype)


class CGAEncoder(nn.Module):
    """
    Euclidean to CGA point encoder.

    Encodes n-dimensional Euclidean coordinates to (n+2)-dimensional
    CGA point representation.

    Args:
        algebra: CGA algebra instance
        use_fp32: Whether to use fp32 for computation
    """

    def __init__(self, algebra: 'CGAAlgebraBase', use_fp32: bool = True):
        super().__init__()
        self.algebra = algebra
        self.use_fp32 = use_fp32

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode Euclidean coordinates to CGA point.

        Args:
            x: Euclidean coordinates, shape (..., n)

        Returns:
            CGA point, shape (..., n+2)
        """
        original_dtype = x.dtype

        if self.use_fp32:
            x = x.to(torch.float32)

        result = self.algebra.cga_encode(x)

        return result.to(original_dtype)


class CGADecoder(nn.Module):
    """
    CGA point to Euclidean decoder.

    Decodes (n+2)-dimensional CGA point to n-dimensional
    Euclidean coordinates.

    Args:
        algebra: CGA algebra instance
    """

    def __init__(self, algebra: 'CGAAlgebraBase'):
        super().__init__()
        self.algebra = algebra

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to Euclidean coordinates.

        Args:
            point: CGA point, shape (..., n+2)

        Returns:
            Euclidean coordinates, shape (..., n)
        """
        return self.algebra.cga_decode(point)


class CGAPipeline(nn.Module):
    """
    Complete CGA transformation pipeline.

    Combines encoding, transformation, and decoding into a single module:
    1. Encode Euclidean -> CGA
    2. Transform via sandwich product
    3. Decode CGA -> Euclidean

    Args:
        algebra: CGA algebra instance
        versor_type: 'even_versor', 'similitude', or 'auto'
        use_fp32: Whether to use fp32 for computation
    """

    def __init__(
        self,
        algebra: 'CGAAlgebraBase',
        versor_type: Literal['even_versor', 'similitude', 'auto'] = 'auto',
        use_fp32: bool = True
    ):
        super().__init__()
        self.encoder = CGAEncoder(algebra, use_fp32)
        self.transform = CliffordTransformLayer(algebra, versor_type, use_fp32)
        self.decoder = CGADecoder(algebra)

    def forward(self, versor: Tensor, x: Tensor) -> Tensor:
        """
        Apply complete transformation pipeline.

        Args:
            versor: EvenVersor/Similitude, shape (..., even_versor_count)
            x: Euclidean coordinates, shape (..., n)

        Returns:
            Transformed Euclidean coordinates, shape (..., n)
        """
        point = self.encoder(x)
        transformed = self.transform(versor, point)
        return self.decoder(transformed)


