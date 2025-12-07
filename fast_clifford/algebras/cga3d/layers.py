"""
PyTorch nn.Module wrappers for CGA operations.

Provides CGACareLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CGACareLayer(nn.Module):
    """
    CGA sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is a motor (16 components: Grade 0, 2, 4)
    - X is a UPGC point (5 components: Grade 1)
    - Output is a transformed UPGC point (5 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CGACareLayer()
        >>> motor = torch.randn(batch_size, 16)
        >>> point = torch.randn(batch_size, 5)
        >>> output = layer(motor, point)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the CGACareLayer."""
        super().__init__()

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        Apply motor transformation to point via sandwich product.

        Args:
            motor: Motor tensor, shape (..., 16)
                   Layout: [scalar, e12, e13, e1+, e1-, e23, e2+, e2-,
                           e3+, e3-, e+-, e123+, e123-, e12+-, e13+-, e23+-]
            point: UPGC point tensor, shape (..., 5)
                   Layout: [e1, e2, e3, e+, e-]

        Returns:
            Transformed point, shape (..., 5)
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


class UPGCEncoder(nn.Module):
    """
    Encoder for converting 3D points to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = UPGCEncoder()
        >>> x_3d = torch.randn(batch_size, 3)
        >>> point = encoder(x_3d)  # shape: (batch_size, 5)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 3D vector to UPGC point.

        Args:
            x: 3D vector, shape (..., 3)

        Returns:
            UPGC point, shape (..., 5)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.upgc_encode(x_f32)
        return result.to(original_dtype)


class UPGCDecoder(nn.Module):
    """
    Decoder for converting UPGC representation back to 3D points.

    Example:
        >>> decoder = UPGCDecoder()
        >>> point = torch.randn(batch_size, 5)
        >>> x_3d = decoder(point)  # shape: (batch_size, 3)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to 3D vector.

        Args:
            point: UPGC point, shape (..., 5)

        Returns:
            3D vector, shape (..., 3)
        """
        return F.upgc_decode(point)


class CGATransformPipeline(nn.Module):
    """
    Complete CGA transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 3D point to UPGC representation
    2. Apply motor transformation via sandwich product
    3. Decode back to 3D point

    Example:
        >>> pipeline = CGATransformPipeline()
        >>> motor = torch.randn(batch_size, 16)
        >>> x_3d = torch.randn(batch_size, 3)
        >>> y_3d = pipeline(motor, x_3d)  # shape: (batch_size, 3)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = UPGCEncoder()
        self.care_layer = CGACareLayer()
        self.decoder = UPGCDecoder()

    def forward(self, motor: Tensor, x: Tensor) -> Tensor:
        """
        Apply motor transformation to 3D point.

        Args:
            motor: Motor tensor, shape (..., 16)
            x: 3D point, shape (..., 3)

        Returns:
            Transformed 3D point, shape (..., 3)
        """
        point = self.encoder(x)
        transformed = self.care_layer(motor, point)
        return self.decoder(transformed)
