"""
CGAAlgebraBase - Abstract Base Class for CGA Algebras

Defines the unified interface that all CGA algebras must implement.

Extended Operations (Feature 005):
- EvenVersor composition and sandwich product
- Similitude (CGA-specific accelerated subset of EvenVersor)
- Inner product, outer product, contractions
- Grade selection, dual, normalize
- Exponential map (bivector -> EvenVersor)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Literal, Union
import torch
from torch import Tensor, nn


class CGAAlgebraBase(ABC):
    """
    Abstract base class for CGA algebras.

    All CGA algebras (hardcoded and runtime) implement this interface,
    providing a unified API regardless of the underlying implementation.

    Naming Convention:
    - EvenVersor: General even-grade versor (rotation + translation + scaling + transversion)
    - Similitude: CGA-specific subset (rotation + translation + scaling, no transversion)
    """

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    @abstractmethod
    def euclidean_dim(self) -> int:
        """Euclidean dimension n (CGA = Cl(n+1, 1, 0))"""
        ...

    @property
    @abstractmethod
    def blade_count(self) -> int:
        """Total number of blades = 2^(n+2)"""
        ...

    @property
    @abstractmethod
    def point_count(self) -> int:
        """Number of UPGC point components = n+2 (Grade 1)"""
        ...

    @property
    @abstractmethod
    def even_versor_count(self) -> int:
        """Number of EvenVersor components (Grade 0, 2, 4... even grades)"""
        ...


    @property
    def similitude_count(self) -> int:
        """Number of Similitude components (same as EvenVersor for storage)"""
        return self.even_versor_count

    @property
    @abstractmethod
    def bivector_count(self) -> int:
        """Number of Bivector components (Grade 2)"""
        ...

    @property
    @abstractmethod
    def max_grade(self) -> int:
        """Maximum grade in the algebra (= n+2 for CGA(n))"""
        ...

    @property
    @abstractmethod
    def signature(self) -> Tuple[int, ...]:
        """Clifford signature (+1, +1, ..., -1)"""
        ...

    @property
    def clifford_notation(self) -> str:
        """Clifford notation, e.g., 'Cl(4,1,0)'"""
        p = self.euclidean_dim + 1
        return f"Cl({p},1,0)"

    # =========================================================================
    # Core Operations
    # =========================================================================

    @abstractmethod
    def upgc_encode(self, x: Tensor) -> Tensor:
        """
        Encode Euclidean coordinates to UPGC point.

        Args:
            x: Euclidean coordinates, shape (..., n)

        Returns:
            UPGC point, shape (..., n+2)
        """
        ...

    @abstractmethod
    def upgc_decode(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to Euclidean coordinates.

        Args:
            point: UPGC point, shape (..., n+2)

        Returns:
            Euclidean coordinates, shape (..., n)
        """
        ...

    @abstractmethod
    def geometric_product_full(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute full geometric product.

        Args:
            a: Left operand, shape (..., blade_count)
            b: Right operand, shape (..., blade_count)

        Returns:
            Result multivector, shape (..., blade_count)
        """
        ...

    @abstractmethod
    def sandwich_product_sparse(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Compute sparse sandwich product M x X x M~.

        Args:
            ev: EvenVersor, shape (..., even_versor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        ...

    @abstractmethod
    def reverse_full(self, mv: Tensor) -> Tensor:
        """
        Compute reverse of a multivector.

        Args:
            mv: Multivector, shape (..., blade_count)

        Returns:
            Reversed multivector, shape (..., blade_count)
        """
        ...

    @abstractmethod
    def reverse_even_versor(self, ev: Tensor) -> Tensor:
        """
        Compute reverse of an EvenVersor.

        Args:
            ev: EvenVersor, shape (..., even_versor_count)

        Returns:
            Reversed EvenVersor, shape (..., even_versor_count)
        """
        ...

    # =========================================================================
    # Layer Factory Methods
    # =========================================================================

    @abstractmethod
    def get_care_layer(self) -> nn.Module:
        """Get CareLayer (sandwich product layer)"""
        ...

    def get_transform_layer(self) -> nn.Module:
        """Get CliffordTransformLayer (alias for get_care_layer)"""
        return self.get_care_layer()

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        """Get UPGC encoder layer"""
        ...

    @abstractmethod
    def get_decoder(self) -> nn.Module:
        """Get UPGC decoder layer"""
        ...

    @abstractmethod
    def get_transform_pipeline(self) -> nn.Module:
        """Get complete transform pipeline (encode + sandwich + decode)"""
        ...

    # =========================================================================
    # T025: Extended Operations - Abstract Methods
    # =========================================================================

    # --- EvenVersor Composition ---

    @abstractmethod
    def compose_even_versor(self, v1: Tensor, v2: Tensor) -> Tensor:
        """
        Compose two EvenVersors via geometric product.

        Args:
            v1: First EvenVersor, shape (..., even_versor_count)
            v2: Second EvenVersor, shape (..., even_versor_count)

        Returns:
            Composed EvenVersor, shape (..., even_versor_count)
        """
        ...

    @abstractmethod
    def compose_similitude(self, s1: Tensor, s2: Tensor) -> Tensor:
        """
        Compose two Similitudes via optimized geometric product.

        Faster than compose_even_versor by utilizing Similitude constraints.

        Args:
            s1: First Similitude, shape (..., even_versor_count)
            s2: Second Similitude, shape (..., even_versor_count)

        Returns:
            Composed Similitude, shape (..., even_versor_count)
        """
        ...

    @abstractmethod
    def sandwich_product_even_versor(self, versor: Tensor, point: Tensor) -> Tensor:
        """
        Compute sandwich product V x X x ~V for EvenVersor.

        Args:
            versor: EvenVersor, shape (..., even_versor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        ...

    @abstractmethod
    def sandwich_product_similitude(self, similitude: Tensor, point: Tensor) -> Tensor:
        """
        Compute sandwich product S x X x ~S for Similitude.

        Faster than sandwich_product_even_versor.

        Args:
            similitude: Similitude, shape (..., even_versor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        ...

    # --- Products ---

    @abstractmethod
    def inner_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute geometric inner product (metric inner product).

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Scalar inner product, shape (..., 1)
        """
        ...

    @abstractmethod
    def outer_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute outer product (wedge product).

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Wedge product, shape (..., blade_count)
        """
        ...

    @abstractmethod
    def left_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute left contraction.

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Left contraction result, shape (..., blade_count)
        """
        ...

    @abstractmethod
    def right_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute right contraction.

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Right contraction result, shape (..., blade_count)
        """
        ...

    # --- Unary Operations ---

    @abstractmethod
    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Compute exponential map from Bivector to EvenVersor.

        Args:
            B: Bivector, shape (..., bivector_count)

        Returns:
            EvenVersor, shape (..., even_versor_count)
        """
        ...

    @abstractmethod
    def grade_select(self, mv: Tensor, grade: int) -> Tensor:
        """
        Extract components of a specific grade.

        Args:
            mv: Multivector, shape (..., blade_count)
            grade: Grade to extract (0 to max_grade)

        Returns:
            Grade components, shape (..., grade_component_count)
        """
        ...

    @abstractmethod
    def dual(self, mv: Tensor) -> Tensor:
        """
        Compute the dual of a multivector.

        Args:
            mv: Multivector, shape (..., blade_count)

        Returns:
            Dual multivector, shape (..., blade_count)
        """
        ...

    @abstractmethod
    def normalize(self, mv: Tensor) -> Tensor:
        """
        Normalize a multivector to unit norm.

        Args:
            mv: Multivector, shape (..., blade_count)

        Returns:
            Normalized multivector, shape (..., blade_count)
        """
        ...

    # --- Structure Normalize (Similitude) ---

    @abstractmethod
    def structure_normalize(self, similitude: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Structure normalize a Similitude to maintain geometric constraints.

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)
            eps: Numerical stability constant

        Returns:
            Structure-normalized Similitude, shape (..., even_versor_count)
        """
        ...

    def soft_structure_normalize(self, similitude: Tensor, strength: float = 0.1) -> Tensor:
        """
        Soft structure normalize (gradient-friendly version).

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)
            strength: Interpolation strength (0 = no change, 1 = full normalize)

        Returns:
            Softly normalized Similitude, shape (..., even_versor_count)
        """
        normalized = self.structure_normalize(similitude)
        return similitude + strength * (normalized - similitude)

    def structure_normalize_ste(self, similitude: Tensor) -> Tensor:
        """
        Structure normalize with Straight-Through Estimator.

        Forward: uses structure_normalize(similitude)
        Backward: gradients pass through similitude unchanged

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)

        Returns:
            Structure-normalized Similitude (with STE gradient)
        """
        clean = self.structure_normalize(similitude)
        return similitude + (clean - similitude).detach()

    # =========================================================================
    # T026: Unified API (Static Routing)
    # =========================================================================

    def compose(
        self,
        v1: Tensor,
        v2: Tensor,
        versor_type: Literal['even_versor', 'similitude', 'auto'] = 'auto'
    ) -> Tensor:
        """
        Unified composition API with static routing.

        Automatically routes to the fastest implementation based on versor_type.

        Args:
            v1: First versor, shape (..., even_versor_count)
            v2: Second versor, shape (..., even_versor_count)
            versor_type: 'even_versor', 'similitude', or 'auto'

        Returns:
            Composed versor, shape (..., even_versor_count)
        """
        if versor_type == 'similitude':
            return self.compose_similitude(v1, v2)
        else:
            return self.compose_even_versor(v1, v2)

    def sandwich_product(
        self,
        versor: Tensor,
        point: Tensor,
        versor_type: Literal['even_versor', 'similitude', 'auto'] = 'auto'
    ) -> Tensor:
        """
        Unified sandwich product API with static routing.

        Args:
            versor: Versor, shape (..., even_versor_count)
            point: UPGC point, shape (..., point_count)
            versor_type: 'even_versor', 'similitude', or 'auto'

        Returns:
            Transformed point, shape (..., point_count)
        """
        if versor_type == 'similitude':
            return self.sandwich_product_similitude(versor, point)
        else:
            return self.sandwich_product_even_versor(versor, point)

    def reverse(
        self,
        v: Tensor,
        versor_type: Literal['full', 'even_versor'] = 'even_versor'
    ) -> Tensor:
        """
        Unified reverse API.

        Args:
            v: Input tensor
            versor_type: 'full' for full multivector, 'even_versor' for sparse

        Returns:
            Reversed tensor
        """
        if versor_type == 'full':
            return self.reverse_full(v)
        else:
            return self.reverse_even_versor(v)

    # =========================================================================
    # Multivector Factory Methods (T190)
    # =========================================================================

    def multivector(self, data: Tensor) -> 'Multivector':
        """
        Create a Multivector wrapper for a tensor.

        Args:
            data: Tensor of shape (..., blade_count)

        Returns:
            Multivector instance
        """
        from .multivector import Multivector
        return Multivector(data, self)

    def even_versor(self, data: Tensor) -> 'EvenVersor':
        """
        Create an EvenVersor wrapper for a tensor.

        Args:
            data: Tensor of shape (..., even_versor_count)

        Returns:
            EvenVersor instance
        """
        from .multivector import EvenVersor
        return EvenVersor(data, self)

    def similitude(self, data: Tensor) -> 'Similitude':
        """
        Create a Similitude wrapper for a tensor.

        Args:
            data: Tensor of shape (..., even_versor_count)

        Returns:
            Similitude instance
        """
        from .multivector import Similitude
        return Similitude(data, self)

    def point(self, x: Tensor) -> 'Multivector':
        """
        Create a UPGC point Multivector from Euclidean coordinates.

        Args:
            x: Euclidean coordinates, shape (..., n)

        Returns:
            Multivector representing the UPGC point
        """
        from .multivector import Multivector
        from fast_clifford.codegen.cga_factory import compute_grade_indices

        point_data = self.upgc_encode(x)
        grade_indices = compute_grade_indices(self.euclidean_dim)
        point_indices = list(grade_indices[1])

        full = torch.zeros(
            *x.shape[:-1], self.blade_count,
            device=x.device, dtype=x.dtype
        )
        for i, idx in enumerate(point_indices):
            full[..., idx] = point_data[..., i]

        return Multivector(full, self)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        return f"CGA{self.euclidean_dim}D({self.clifford_notation})"
