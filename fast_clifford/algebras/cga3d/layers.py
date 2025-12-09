"""
PyTorch nn.Module wrappers for CGA operations.

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
    CGA sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is an EvenVersor (16 components: Grade 0, 2, 4)
    - X is a CGA point (5 components: Grade 1)
    - Output is a transformed CGA point (5 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CliffordTransformLayer()
        >>> ev = torch.randn(batch_size, 16)
        >>> point = torch.randn(batch_size, 5)
        >>> output = layer(ev, point)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the CliffordTransformLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 16)
                   Layout: [scalar, e12, e13, e1+, e1-, e23, e2+, e2-,
                           e3+, e3-, e+-, e123+, e123-, e12+-, e13+-, e23+-]
            point: CGA point tensor, shape (..., 5)
                   Layout: [e1, e2, e3, e+, e-]

        Returns:
            Transformed point, shape (..., 5)
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
    Encoder for converting 3D points to CGA representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = CGAEncoder()
        >>> x_3d = torch.randn(batch_size, 3)
        >>> point = encoder(x_3d)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 3D vector to CGA point.

        Args:
            x: 3D vector, shape (..., 3)

        Returns:
            CGA point, shape (..., 5)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.cga_encode(x_f32)
        return result.to(original_dtype)


class CGADecoder(nn.Module):
    """
    Decoder for converting CGA representation back to 3D points.

    Example:
        >>> decoder = CGADecoder()
        >>> point = torch.randn(batch_size, 5)
        >>> x_3d = decoder(point)  # shape: (batch_size, 3)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to 3D vector.

        Args:
            point: CGA point, shape (..., 5)

        Returns:
            3D vector, shape (..., 3)
        """
        return F.cga_decode(point)


class CGAPipeline(nn.Module):
    """
    Complete CGA transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 3D point to CGA representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 3D point

    Example:
        >>> pipeline = CGAPipeline()
        >>> ev = torch.randn(batch_size, 16)
        >>> x_3d = torch.randn(batch_size, 3)
        >>> y_3d = pipeline(ev, x_3d)  # shape: (batch_size, 3)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = CGAEncoder()
        self.transform_layer = CliffordTransformLayer()
        self.decoder = CGADecoder()

    def forward(self, ev: Tensor, x: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to 3D point.

        Args:
            ev: EvenVersor tensor, shape (..., 16)
            x: 3D point, shape (..., 3)

        Returns:
            Transformed 3D point, shape (..., 3)
        """
        point = self.encoder(x)
        transformed = self.transform_layer(ev, point)
        return self.decoder(transformed)
