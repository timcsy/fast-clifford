"""
PyTorch nn.Module wrappers for CGA5D operations.

Provides CliffordTransformLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models

CGA5D Cl(6,1) specifications:
- 128 blades total
- EvenVersor: 64 components (Grade 0 + 2 + 4 + 6)
- UPGC Point: 7 components (Grade 1)
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CliffordTransformLayer(nn.Module):
    """
    CGA5D sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is an EvenVersor (64 components: Grade 0, 2, 4, 6)
    - X is a CGA point (7 components: Grade 1)
    - Output is a transformed CGA point (7 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CliffordTransformLayer()
        >>> ev = torch.randn(batch_size, 64)
        >>> point = torch.randn(batch_size, 7)
        >>> output = layer(ev, point)  # shape: (batch_size, 7)
    """

    def __init__(self):
        """Initialize the CliffordTransformLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 64)
                Layout: [scalar (1), Grade 2 (21), Grade 4 (35), Grade 6 (7)]
            point: CGA point tensor, shape (..., 7)
                   Layout: [e1, e2, e3, e4, e5, e+, e-]

        Returns:
            Transformed point, shape (..., 7)
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
    Encoder for converting 5D points to CGA representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = CGAEncoder()
        >>> x_5d = torch.randn(batch_size, 5)
        >>> point = encoder(x_5d)  # shape: (batch_size, 7)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 5D vector to CGA point.

        Args:
            x: 5D vector, shape (..., 5)

        Returns:
            CGA point, shape (..., 7)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.cga_encode(x_f32)
        return result.to(original_dtype)


class CGADecoder(nn.Module):
    """
    Decoder for converting CGA representation back to 5D points.

    Example:
        >>> decoder = CGADecoder()
        >>> point = torch.randn(batch_size, 7)
        >>> x_5d = decoder(point)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode CGA point to 5D vector.

        Args:
            point: CGA point, shape (..., 7)

        Returns:
            5D vector, shape (..., 5)
        """
        return F.cga_decode(point)


class CGAPipeline(nn.Module):
    """
    Complete CGA5D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 5D point to CGA representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 5D point

    Example:
        >>> pipeline = CGAPipeline()
        >>> ev = torch.randn(batch_size, 64)
        >>> x_5d = torch.randn(batch_size, 5)
        >>> y_5d = pipeline(ev, x_5d)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = CGAEncoder()
        self.transform_layer = CliffordTransformLayer()
        self.decoder = CGADecoder()

    def forward(self, ev: Tensor, x: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to 5D point.

        Args:
            ev: EvenVersor tensor, shape (..., 64)
            x: 5D point, shape (..., 5)

        Returns:
            Transformed 5D point, shape (..., 5)
        """
        point = self.encoder(x)
        transformed = self.transform_layer(ev, point)
        return self.decoder(transformed)
