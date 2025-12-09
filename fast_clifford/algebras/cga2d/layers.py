"""
PyTorch nn.Module wrappers for CGA2D operations.

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
    CGA2D sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is an EvenVersor (8 components: Grade 0, 2, 4)
    - X is a CGA point (4 components: Grade 1)
    - Output is a transformed CGA point (4 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CliffordTransformLayer()
        >>> ev = torch.randn(batch_size, 8)
        >>> point = torch.randn(batch_size, 4)
        >>> output = layer(ev, point)  # shape: (batch_size, 4)
    """

    def __init__(self):
        """Initialize the CliffordTransformLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 8)
                Layout: [scalar, e12, e1+, e1-, e2+, e2-, e+-, e12+-]
            point: CGA point tensor, shape (..., 4)
                   Layout: [e1, e2, e+, e-]

        Returns:
            Transformed point, shape (..., 4)
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
    Encoder for converting 2D points to CGA representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = CGAEncoder()
        >>> x_2d = torch.randn(batch_size, 2)
        >>> point = encoder(x_2d)  # shape: (batch_size, 4)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 2D vector to CGA point.

        Args:
            x: 2D vector, shape (..., 2)

        Returns:
            CGA point, shape (..., 4)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.cga_encode(x_f32)
        return result.to(original_dtype)


class CGADecoder(nn.Module):
    """
    Decoder for converting CGA representation back to 2D points.

    Example:
        >>> decoder = CGADecoder()
        >>> point = torch.randn(batch_size, 4)
        >>> x_2d = decoder(point)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to 2D vector.

        Args:
            point: CGA point, shape (..., 4)

        Returns:
            2D vector, shape (..., 2)
        """
        return F.cga_decode(point)


class CGAPipeline(nn.Module):
    """
    Complete CGA2D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 2D point to CGA representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 2D point

    Example:
        >>> pipeline = CGAPipeline()
        >>> ev = torch.randn(batch_size, 8)
        >>> x_2d = torch.randn(batch_size, 2)
        >>> y_2d = pipeline(ev, x_2d)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = CGAEncoder()
        self.transform_layer = CliffordTransformLayer()
        self.decoder = CGADecoder()

    def forward(self, ev: Tensor, x: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to 2D point.

        Args:
            ev: EvenVersor tensor, shape (..., 8)
            x: 2D point, shape (..., 2)

        Returns:
            Transformed 2D point, shape (..., 2)
        """
        point = self.encoder(x)
        transformed = self.transform_layer(ev, point)
        return self.decoder(transformed)
