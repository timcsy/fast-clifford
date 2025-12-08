"""
HardcodedCGAWrapper - Wrapper for existing cga0d-cga5d modules

Adapts existing hardcoded CGA modules to the unified CGAAlgebraBase interface.
"""

from typing import Tuple
from torch import Tensor, nn

from .base import CGAAlgebraBase


class HardcodedCGAWrapper(CGAAlgebraBase):
    """
    Wrapper that adapts existing hardcoded CGA modules to CGAAlgebraBase.

    Supported dimensions: 0, 1, 2, 3, 4, 5
    """

    def __init__(self, euclidean_dim: int):
        """
        Initialize wrapper for a specific dimension.

        Args:
            euclidean_dim: Euclidean dimension (0-5)

        Raises:
            ValueError: If dimension not in [0, 5]
        """
        if euclidean_dim < 0 or euclidean_dim > 5:
            raise ValueError(f"Hardcoded CGA only supports dimensions 0-5, got {euclidean_dim}")

        self._euclidean_dim = euclidean_dim
        self._module = self._load_module(euclidean_dim)

    def _load_module(self, dim: int):
        """Load the appropriate CGA module."""
        if dim == 0:
            from fast_clifford.algebras import cga0d
            return cga0d
        elif dim == 1:
            from fast_clifford.algebras import cga1d
            return cga1d
        elif dim == 2:
            from fast_clifford.algebras import cga2d
            return cga2d
        elif dim == 3:
            from fast_clifford.algebras import cga3d
            return cga3d
        elif dim == 4:
            from fast_clifford.algebras import cga4d
            return cga4d
        elif dim == 5:
            from fast_clifford.algebras import cga5d
            return cga5d
        else:
            raise ValueError(f"Unknown dimension: {dim}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def euclidean_dim(self) -> int:
        return self._euclidean_dim

    @property
    def blade_count(self) -> int:
        return self._module.BLADE_COUNT

    @property
    def point_count(self) -> int:
        return len(self._module.UPGC_POINT_MASK)

    @property
    def motor_count(self) -> int:
        return len(self._module.MOTOR_MASK)

    @property
    def signature(self) -> Tuple[int, ...]:
        # CGA(n) has signature Cl(n+1, 1, 0): n+1 positive, 1 negative
        # Dynamically compute to avoid dependency on module having SIGNATURE
        return tuple([1] * (self._euclidean_dim + 1) + [-1])

    # =========================================================================
    # Core Operations
    # =========================================================================

    def upgc_encode(self, x: Tensor) -> Tensor:
        return self._module.upgc_encode(x)

    def upgc_decode(self, point: Tensor) -> Tensor:
        return self._module.upgc_decode(point)

    def geometric_product_full(self, a: Tensor, b: Tensor) -> Tensor:
        return self._module.geometric_product_full(a, b)

    def sandwich_product_sparse(self, motor: Tensor, point: Tensor) -> Tensor:
        return self._module.sandwich_product_sparse(motor, point)

    def reverse_full(self, mv: Tensor) -> Tensor:
        return self._module.reverse_full(mv)

    def reverse_motor(self, motor: Tensor) -> Tensor:
        return self._module.reverse_motor(motor)

    # =========================================================================
    # Layer Factory Methods
    # =========================================================================

    def get_care_layer(self) -> nn.Module:
        """Get CareLayer for this dimension."""
        dim = self._euclidean_dim
        if dim == 0:
            return self._module.CGA0DCareLayer()
        elif dim == 1:
            return self._module.CGA1DCareLayer()
        elif dim == 2:
            return self._module.CGA2DCareLayer()
        elif dim == 3:
            # cga3d uses old naming convention (CGACareLayer instead of CGA3DCareLayer)
            return self._module.CGACareLayer()
        elif dim == 4:
            return self._module.CGA4DCareLayer()
        elif dim == 5:
            return self._module.CGA5DCareLayer()

    def get_encoder(self) -> nn.Module:
        """Get UPGC encoder for this dimension."""
        dim = self._euclidean_dim
        if dim == 0:
            return self._module.UPGC0DEncoder()
        elif dim == 1:
            return self._module.UPGC1DEncoder()
        elif dim == 2:
            return self._module.UPGC2DEncoder()
        elif dim == 3:
            # cga3d uses old naming convention (UPGCEncoder instead of UPGC3DEncoder)
            return self._module.UPGCEncoder()
        elif dim == 4:
            return self._module.UPGC4DEncoder()
        elif dim == 5:
            return self._module.UPGC5DEncoder()

    def get_decoder(self) -> nn.Module:
        """Get UPGC decoder for this dimension."""
        dim = self._euclidean_dim
        if dim == 0:
            return self._module.UPGC0DDecoder()
        elif dim == 1:
            return self._module.UPGC1DDecoder()
        elif dim == 2:
            return self._module.UPGC2DDecoder()
        elif dim == 3:
            # cga3d uses old naming convention (UPGCDecoder instead of UPGC3DDecoder)
            return self._module.UPGCDecoder()
        elif dim == 4:
            return self._module.UPGC4DDecoder()
        elif dim == 5:
            return self._module.UPGC5DDecoder()

    def get_transform_pipeline(self) -> nn.Module:
        """Get complete transform pipeline for this dimension."""
        dim = self._euclidean_dim
        if dim == 0:
            return self._module.CGA0DTransformPipeline()
        elif dim == 1:
            return self._module.CGA1DTransformPipeline()
        elif dim == 2:
            return self._module.CGA2DTransformPipeline()
        elif dim == 3:
            # cga3d uses old naming convention (CGATransformPipeline instead of CGA3DTransformPipeline)
            return self._module.CGATransformPipeline()
        elif dim == 4:
            return self._module.CGA4DTransformPipeline()
        elif dim == 5:
            return self._module.CGA5DTransformPipeline()
