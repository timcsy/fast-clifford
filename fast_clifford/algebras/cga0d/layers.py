"""
PyTorch nn.Module wrappers for CGA0D operations.

Provides CliffordTransformLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CliffordTransformLayer(nn.Module):
    """
    CGA0D sandwich product layer for point transformation.

    Computes M x X x M~ where:
    - M is an EvenVersor (2 components: Grade 0, 2)
    - X is a CGA point (2 components: Grade 1)
    - Output is a transformed CGA point (2 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CliffordTransformLayer()
        >>> ev = torch.randn(batch_size, 2)
        >>> point = torch.randn(batch_size, 2)
        >>> output = layer(ev, point)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the CliffordTransformLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 2)
                   Layout: [scalar, e+-]
            point: CGA point tensor, shape (..., 2)
                   Layout: [e+, e-]

        Returns:
            Transformed point, shape (..., 2)
        """
        # Save original dtype for output conversion
        original_dtype = point.dtype

        # Convert to float32 for stable CGA computation
        ev_f32 = ev.to(torch.float32)
        point_f32 = point.to(torch.float32)

        # Compute sandwich product
        result = F.sandwich_product_sparse(ev_f32, point_f32)

        # Convert back to original dtype
        return result.to(original_dtype)


class CGAEncoder(nn.Module):
    """
    Encoder for converting 0D points to CGA representation.

    For 0D CGA, the encoded point is always the origin:
    X = n_o = 0.5 * (e- - e+)

    Example:
        >>> encoder = CGAEncoder()
        >>> x_0d = torch.randn(batch_size, 0)
        >>> point = encoder(x_0d)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 0D vector to CGA point.

        Args:
            x: 0D vector, shape (..., 0)

        Returns:
            CGA point, shape (..., 2)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.cga_encode(x_f32)
        return result.to(original_dtype)


class CGADecoder(nn.Module):
    """
    Decoder for converting CGA representation back to 0D points.

    For 0D CGA, returns an empty tensor.

    Example:
        >>> decoder = CGADecoder()
        >>> point = torch.randn(batch_size, 2)
        >>> x_0d = decoder(point)  # shape: (batch_size, 0)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to 0D vector.

        Args:
            point: CGA point, shape (..., 2)

        Returns:
            0D vector, shape (..., 0)
        """
        return F.cga_decode(point)


class CGAPipeline(nn.Module):
    """
    Complete CGA0D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 0D point to CGA representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 0D point

    Example:
        >>> pipeline = CGAPipeline()
        >>> ev = torch.randn(batch_size, 2)
        >>> x_0d = torch.randn(batch_size, 0)
        >>> y_0d = pipeline(ev, x_0d)  # shape: (batch_size, 0)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = CGAEncoder()
        self.transform_layer = CliffordTransformLayer()
        self.decoder = CGADecoder()

    def forward(self, ev: Tensor, x: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to 0D point.

        Args:
            ev: EvenVersor tensor, shape (..., 2)
            x: 0D point, shape (..., 0)

        Returns:
            Transformed 0D point, shape (..., 0)
        """
        point = self.encoder(x)
        transformed = self.transform_layer(ev, point)
        return self.decoder(transformed)
