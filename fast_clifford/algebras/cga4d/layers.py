"""
PyTorch nn.Module wrappers for CGA4D operations.

Provides CGA4DCareLayer that wraps the sandwich product with:
- Automatic precision handling (fp16 -> fp32 -> fp16)
- PyTorch autograd compatibility
- Clean API for use in Transformer models

CGA4D Cl(5,1) specifications:
- 64 blades total
- EvenVersor: 31 components (Grade 0 + 2 + 4)
- UPGC Point: 6 components (Grade 1)
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F


class CGA4DCareLayer(nn.Module):
    """
    CGA4D sandwich product layer for point transformation.

    Computes M × X × M̃ where:
    - M is an EvenVersor (31 components: Grade 0, 2, 4)
    - X is a UPGC point (6 components: Grade 1)
    - Output is a transformed UPGC point (6 components)

    This layer handles:
    - Precision conversion (fp16 input -> fp32 computation -> fp16 output)
    - ONNX-compatible operations (no loops)

    Example:
        >>> layer = CGA4DCareLayer()
        >>> ev = torch.randn(batch_size, 31)
        >>> point = torch.randn(batch_size, 6)
        >>> output = layer(ev, point)  # shape: (batch_size, 6)
    """

    def __init__(self):
        """Initialize the CGA4DCareLayer."""
        super().__init__()

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to point via sandwich product.

        Args:
            ev: EvenVersor tensor, shape (..., 31)
                   Layout: [scalar (1), Grade 2 (15), Grade 4 (15)]
            point: UPGC point tensor, shape (..., 6)
                   Layout: [e1, e2, e3, e4, e+, e-]

        Returns:
            Transformed point, shape (..., 6)
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


class UPGC4DEncoder(nn.Module):
    """
    Encoder for converting 4D points to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Example:
        >>> encoder = UPGC4DEncoder()
        >>> x_4d = torch.randn(batch_size, 4)
        >>> point = encoder(x_4d)  # shape: (batch_size, 6)
    """

    def __init__(self):
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode 4D vector to UPGC point.

        Args:
            x: 4D vector, shape (..., 4)

        Returns:
            UPGC point, shape (..., 6)
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.upgc_encode(x_f32)
        return result.to(original_dtype)


class UPGC4DDecoder(nn.Module):
    """
    Decoder for converting UPGC representation back to 4D points.

    Example:
        >>> decoder = UPGC4DDecoder()
        >>> point = torch.randn(batch_size, 6)
        >>> x_4d = decoder(point)  # shape: (batch_size, 4)
    """

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to 4D vector.

        Args:
            point: UPGC point, shape (..., 6)

        Returns:
            4D vector, shape (..., 4)
        """
        return F.upgc_decode(point)


class CGA4DTransformPipeline(nn.Module):
    """
    Complete CGA4D transformation pipeline.

    Combines encoding, transformation, and decoding:
    1. Encode 4D point to UPGC representation
    2. Apply EvenVersor transformation via sandwich product
    3. Decode back to 4D point

    Example:
        >>> pipeline = CGA4DTransformPipeline()
        >>> ev = torch.randn(batch_size, 31)
        >>> x_4d = torch.randn(batch_size, 4)
        >>> y_4d = pipeline(ev, x_4d)  # shape: (batch_size, 4)
    """

    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.encoder = UPGC4DEncoder()
        self.care_layer = CGA4DCareLayer()
        self.decoder = UPGC4DDecoder()

    def forward(self, ev: Tensor, x: Tensor) -> Tensor:
        """
        Apply EvenVersor transformation to 4D point.

        Args:
            ev: EvenVersor tensor, shape (..., 31)
            x: 4D point, shape (..., 4)

        Returns:
            Transformed 4D point, shape (..., 4)
        """
        point = self.encoder(x)
        transformed = self.care_layer(ev, point)
        return self.decoder(transformed)
