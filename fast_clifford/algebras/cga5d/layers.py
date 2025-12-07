"""
PyTorch nn.Module wrappers for CGA5D operations.

Provides CGA5DCareLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models

CGA5D Cl(6,1) specifications:
- 128 blades total
- Motor: 64 components (Grade 0 + 2 + 4 + 6)
- UPGC Point: 7 components (Grade 1)
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CGA5DCareLayer(nn.Module):
    """
    CGA5D sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is a motor (64 components: Grade 0, 2, 4, 6)
    - X is a UPGC point (7 components: Grade 1)
    - Output is a transformed UPGC point (7 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CGA5DCareLayer()
        >>> motor = torch.randn(batch_size, 64)
        >>> point = torch.randn(batch_size, 7)
        >>> output = layer(motor, point)  # shape: (batch_size, 7)
    """

    def __init__(self):
        """Initialize the CGA5DCareLayer."""
        super().__init__()

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        Apply motor transformation to point via sandwich product.

        Args:
            motor: Motor tensor, shape (..., 64)
                   Layout: [scalar (1), Grade 2 (21), Grade 4 (35), Grade 6 (7)]
            point: UPGC point tensor, shape (..., 7)
                   Layout: [e1, e2, e3, e4, e5, e+, e-]

        Returns:
            Transformed point, shape (..., 7)
        """
        # Save original dtype for output conversion
        original_dtype = point.dtype

        # Convert to float32 for stable CGA computation
        motor_f32 = motor.to(torch.float32)
        point_f32 = point.to(torch.float32)

        # Compute sandwich product
        result = F.sandwich_product_sparse(motor_f32, point_f32)

        # Convert back to original dtype
        return result.to(original_dtype)


class UPGC5DEncoder(nn.Module):
    """
    Encoder for converting 5D points to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = UPGC5DEncoder()
        >>> x_5d = torch.randn(batch_size, 5)
        >>> point = encoder(x_5d)  # shape: (batch_size, 7)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 5D vector to UPGC point.

        Args:
            x: 5D vector, shape (..., 5)

        Returns:
            UPGC point, shape (..., 7)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.upgc_encode(x_f32)
        return result.to(original_dtype)


class UPGC5DDecoder(nn.Module):
    """
    Decoder for converting UPGC representation back to 5D points.

    Example:
        >>> decoder = UPGC5DDecoder()
        >>> point = torch.randn(batch_size, 7)
        >>> x_5d = decoder(point)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to 5D vector.

        Args:
            point: UPGC point, shape (..., 7)

        Returns:
            5D vector, shape (..., 5)
        """
        return F.upgc_decode(point)


class CGA5DTransformPipeline(nn.Module):
    """
    Complete CGA5D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 5D point to UPGC representation
    2. Apply motor transformation via sandwich product
    3. Decode back to 5D point

    Example:
        >>> pipeline = CGA5DTransformPipeline()
        >>> motor = torch.randn(batch_size, 64)
        >>> x_5d = torch.randn(batch_size, 5)
        >>> y_5d = pipeline(motor, x_5d)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = UPGC5DEncoder()
        self.care_layer = CGA5DCareLayer()
        self.decoder = UPGC5DDecoder()

    def forward(self, motor: Tensor, x: Tensor) -> Tensor:
        """
        Apply motor transformation to 5D point.

        Args:
            motor: Motor tensor, shape (..., 64)
            x: 5D point, shape (..., 5)

        Returns:
            Transformed 5D point, shape (..., 5)
        """
        point = self.encoder(x)
        transformed = self.care_layer(motor, point)
        return self.decoder(transformed)
