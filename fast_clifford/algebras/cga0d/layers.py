"""
PyTorch nn.Module wrappers for CGA0D operations.

Provides CGA0DCareLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CGA0DCareLayer(nn.Module):
    """
    CGA0D sandwich product layer for point transformation.

    Computes M x X x M~ where:
    - M is an EvenVersor (2 components: Grade 0, 2)
    - X is a UPGC point (2 components: Grade 1)
    - Output is a transformed UPGC point (2 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CGA0DCareLayer()
        >>> ev = torch.randn(batch_size, 2)
        >>> point = torch.randn(batch_size, 2)
        >>> output = layer(ev, point)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the CGA0DCareLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 2)
                   Layout: [scalar, e+-]
            point: UPGC point tensor, shape (..., 2)
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


class UPGC0DEncoder(nn.Module):
    """
    Encoder for converting 0D points to UPGC representation.

    For 0D CGA, the encoded point is always the origin:
    X = n_o = 0.5 * (e- - e+)

    Example:
        >>> encoder = UPGC0DEncoder()
        >>> x_0d = torch.randn(batch_size, 0)
        >>> point = encoder(x_0d)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 0D vector to UPGC point.

        Args:
            x: 0D vector, shape (..., 0)

        Returns:
            UPGC point, shape (..., 2)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.upgc_encode(x_f32)
        return result.to(original_dtype)


class UPGC0DDecoder(nn.Module):
    """
    Decoder for converting UPGC representation back to 0D points.

    For 0D CGA, returns an empty tensor.

    Example:
        >>> decoder = UPGC0DDecoder()
        >>> point = torch.randn(batch_size, 2)
        >>> x_0d = decoder(point)  # shape: (batch_size, 0)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to 0D vector.

        Args:
            point: UPGC point, shape (..., 2)

        Returns:
            0D vector, shape (..., 0)
        """
        return F.upgc_decode(point)


class CGA0DTransformPipeline(nn.Module):
    """
    Complete CGA0D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 0D point to UPGC representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 0D point

    Example:
        >>> pipeline = CGA0DTransformPipeline()
        >>> ev = torch.randn(batch_size, 2)
        >>> x_0d = torch.randn(batch_size, 0)
        >>> y_0d = pipeline(ev, x_0d)  # shape: (batch_size, 0)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = UPGC0DEncoder()
        self.care_layer = CGA0DCareLayer()
        self.decoder = UPGC0DDecoder()

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
        transformed = self.care_layer(ev, point)
        return self.decoder(transformed)
