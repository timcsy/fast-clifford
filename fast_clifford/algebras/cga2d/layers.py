"""
PyTorch nn.Module wrappers for CGA2D operations.

Provides CGA2DCareLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CGA2DCareLayer(nn.Module):
    """
    CGA2D sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is a motor (7 components: Grade 0, 2)
    - X is a UPGC point (4 components: Grade 1)
    - Output is a transformed UPGC point (4 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CGA2DCareLayer()
        >>> motor = torch.randn(batch_size, 7)
        >>> point = torch.randn(batch_size, 4)
        >>> output = layer(motor, point)  # shape: (batch_size, 4)
    """

    def __init__(self):
        """Initialize the CGA2DCareLayer."""
        super().__init__()

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        Apply motor transformation to point via sandwich product.

        Args:
            motor: Motor tensor, shape (..., 7)
                   Layout: [scalar, e12, e1+, e1-, e2+, e2-, e+-]
            point: UPGC point tensor, shape (..., 4)
                   Layout: [e1, e2, e+, e-]

        Returns:
            Transformed point, shape (..., 4)
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


class UPGC2DEncoder(nn.Module):
    """
    Encoder for converting 2D points to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = UPGC2DEncoder()
        >>> x_2d = torch.randn(batch_size, 2)
        >>> point = encoder(x_2d)  # shape: (batch_size, 4)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 2D vector to UPGC point.

        Args:
            x: 2D vector, shape (..., 2)

        Returns:
            UPGC point, shape (..., 4)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.upgc_encode(x_f32)
        return result.to(original_dtype)


class UPGC2DDecoder(nn.Module):
    """
    Decoder for converting UPGC representation back to 2D points.

    Example:
        >>> decoder = UPGC2DDecoder()
        >>> point = torch.randn(batch_size, 4)
        >>> x_2d = decoder(point)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to 2D vector.

        Args:
            point: UPGC point, shape (..., 4)

        Returns:
            2D vector, shape (..., 2)
        """
        return F.upgc_decode(point)


class CGA2DTransformPipeline(nn.Module):
    """
    Complete CGA2D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 2D point to UPGC representation
    2. Apply motor transformation via sandwich product
    3. Decode back to 2D point

    Example:
        >>> pipeline = CGA2DTransformPipeline()
        >>> motor = torch.randn(batch_size, 7)
        >>> x_2d = torch.randn(batch_size, 2)
        >>> y_2d = pipeline(motor, x_2d)  # shape: (batch_size, 2)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = UPGC2DEncoder()
        self.care_layer = CGA2DCareLayer()
        self.decoder = UPGC2DDecoder()

    def forward(self, motor: Tensor, x: Tensor) -> Tensor:
        """
        Apply motor transformation to 2D point.

        Args:
            motor: Motor tensor, shape (..., 7)
            x: 2D point, shape (..., 2)

        Returns:
            Transformed 2D point, shape (..., 2)
        """
        point = self.encoder(x)
        transformed = self.care_layer(motor, point)
        return self.decoder(transformed)
