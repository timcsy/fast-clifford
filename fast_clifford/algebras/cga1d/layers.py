"""
PyTorch nn.Module wrappers for CGA1D operations.

Provides CGA1DCareLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CGA1DCareLayer(nn.Module):
    """
    CGA1D sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is an EvenVersor (4 components: Grade 0, 2)
    - X is a UPGC point (3 components: Grade 1)
    - Output is a transformed UPGC point (3 components)

    Note: CGA1D does not have Grade 4 (max grade is 3).

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CGA1DCareLayer()
        >>> ev = torch.randn(batch_size, 4)
        >>> point = torch.randn(batch_size, 3)
        >>> output = layer(ev, point)  # shape: (batch_size, 3)
    """

    def __init__(self):
        """Initialize the CGA1DCareLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 4)
                   Layout: [scalar, e1+, e1-, e+-]
            point: UPGC point tensor, shape (..., 3)
                   Layout: [e1, e+, e-]

        Returns:
            Transformed point, shape (..., 3)
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


class UPGC1DEncoder(nn.Module):
    """
    Encoder for converting 1D points to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = UPGC1DEncoder()
        >>> x_1d = torch.randn(batch_size, 1)
        >>> point = encoder(x_1d)  # shape: (batch_size, 3)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 1D vector to UPGC point.

        Args:
            x: 1D vector, shape (..., 1)

        Returns:
            UPGC point, shape (..., 3)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.upgc_encode(x_f32)
        return result.to(original_dtype)


class UPGC1DDecoder(nn.Module):
    """
    Decoder for converting UPGC representation back to 1D points.

    Example:
        >>> decoder = UPGC1DDecoder()
        >>> point = torch.randn(batch_size, 3)
        >>> x_1d = decoder(point)  # shape: (batch_size, 1)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to 1D vector.

        Args:
            point: UPGC point, shape (..., 3)

        Returns:
            1D vector, shape (..., 1)
        """
        return F.upgc_decode(point)


class CGA1DTransformPipeline(nn.Module):
    """
    Complete CGA1D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 1D point to UPGC representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 1D point

    Example:
        >>> pipeline = CGA1DTransformPipeline()
        >>> ev = torch.randn(batch_size, 4)
        >>> x_1d = torch.randn(batch_size, 1)
        >>> y_1d = pipeline(ev, x_1d)  # shape: (batch_size, 1)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = UPGC1DEncoder()
        self.care_layer = CGA1DCareLayer()
        self.decoder = UPGC1DDecoder()

    def forward(self, ev: Tensor, x: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to 1D point.

        Args:
            ev: EvenVersor tensor, shape (..., 4)
            x: 1D point, shape (..., 1)

        Returns:
            Transformed 1D point, shape (..., 1)
        """
        point = self.encoder(x)
        transformed = self.care_layer(ev, point)
        return self.decoder(transformed)
