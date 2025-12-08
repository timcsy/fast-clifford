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
    def even_versor_count(self) -> int:
        return len(self._module.EVEN_VERSOR_MASK)

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

    def sandwich_product_sparse(self, ev: Tensor, point: Tensor) -> Tensor:
        return self._module.sandwich_product_sparse(ev, point)

    def reverse_full(self, mv: Tensor) -> Tensor:
        return self._module.reverse_full(mv)

    def reverse_even_versor(self, ev: Tensor) -> Tensor:
        return self._module.reverse_even_versor(ev)

    # =========================================================================
    # Layer Factory Methods (using unified layers)
    # =========================================================================

    def get_care_layer(self) -> nn.Module:
        """Get CareLayer (CliffordTransformLayer) for this dimension."""
        from .layers import CliffordTransformLayer
        return CliffordTransformLayer(self)

    def get_transform_layer(self) -> nn.Module:
        """Get CliffordTransformLayer for this dimension (alias for get_care_layer)."""
        return self.get_care_layer()

    def get_encoder(self) -> nn.Module:
        """Get CGAEncoder for this dimension."""
        from .layers import CGAEncoder
        return CGAEncoder(self)

    def get_decoder(self) -> nn.Module:
        """Get CGADecoder for this dimension."""
        from .layers import CGADecoder
        return CGADecoder(self)

    def get_transform_pipeline(self) -> nn.Module:
        """Get complete transform pipeline for this dimension."""
        from .layers import CGAPipeline
        return CGAPipeline(self)

    # =========================================================================
    # Extended Properties
    # =========================================================================

    @property
    def bivector_count(self) -> int:
        """Number of Bivector components (Grade 2)."""
        return len(self._module.GRADE_2_INDICES)

    @property
    def max_grade(self) -> int:
        """Maximum grade in the algebra (= n+2 for CGA(n))."""
        return self._euclidean_dim + 2

    # =========================================================================
    # Extended Operations
    # =========================================================================

    def compose_even_versor(self, v1: Tensor, v2: Tensor) -> Tensor:
        """Compose two EvenVersors via geometric product."""
        return self._module.functional.compose_even_versor(v1, v2)

    def compose_similitude(self, s1: Tensor, s2: Tensor) -> Tensor:
        """Compose two Similitudes via optimized geometric product."""
        return self._module.functional.compose_similitude(s1, s2)

    def sandwich_product_even_versor(self, versor: Tensor, point: Tensor) -> Tensor:
        """Compute sandwich product V x X x ~V for EvenVersor."""
        # Use the existing sparse sandwich product
        return self._module.sandwich_product_sparse(versor, point)

    def sandwich_product_similitude(self, similitude: Tensor, point: Tensor) -> Tensor:
        """Compute sandwich product S x X x ~S for Similitude."""
        return self._module.functional.sandwich_product_similitude(similitude, point)

    def inner_product(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute geometric inner product (metric inner product)."""
        return self._module.functional.inner_product_full(a, b)

    def outer_product(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute outer product (wedge product)."""
        return self._module.functional.outer_product_full(a, b)

    def left_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute left contraction."""
        return self._module.functional.left_contraction_full(a, b)

    def right_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute right contraction."""
        return self._module.functional.right_contraction_full(a, b)

    def exp_bivector(self, B: Tensor) -> Tensor:
        """Compute exponential map from Bivector to EvenVersor."""
        return self._module.functional.exp_bivector(B)

    def grade_select(self, mv: Tensor, grade: int) -> Tensor:
        """Extract components of a specific grade."""
        return self._module.functional.grade_select(mv, grade)

    def dual(self, mv: Tensor) -> Tensor:
        """Compute the dual of a multivector."""
        return self._module.functional.dual(mv)

    def normalize(self, mv: Tensor) -> Tensor:
        """Normalize a multivector to unit norm."""
        return self._module.functional.normalize(mv)

    def structure_normalize(self, similitude: Tensor, eps: float = 1e-8) -> Tensor:
        """Structure normalize a Similitude to maintain geometric constraints."""
        # Check if the module has structure_normalize
        if hasattr(self._module.functional, 'structure_normalize'):
            return self._module.functional.structure_normalize(similitude, eps)
        else:
            # Fallback: just return the input (no structure normalization available)
            return similitude
