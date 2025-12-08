"""
CGAAlgebraBase - Abstract Base Class for CGA Algebras

Defines the unified interface that all CGA algebras must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import Tensor, nn


class CGAAlgebraBase(ABC):
    """
    Abstract base class for CGA algebras.

    All CGA algebras (hardcoded and runtime) implement this interface,
    providing a unified API regardless of the underlying implementation.
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
    def motor_count(self) -> int:
        """Number of Motor components (Grade 0, 2, 4... even grades)"""
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
    def sandwich_product_sparse(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        Compute sparse sandwich product M x X x M~.

        Args:
            motor: Motor, shape (..., motor_count)
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
    def reverse_motor(self, motor: Tensor) -> Tensor:
        """
        Compute reverse of a motor.

        Args:
            motor: Motor, shape (..., motor_count)

        Returns:
            Reversed motor, shape (..., motor_count)
        """
        ...

    # =========================================================================
    # Layer Factory Methods
    # =========================================================================

    @abstractmethod
    def get_care_layer(self) -> nn.Module:
        """Get CareLayer (sandwich product layer)"""
        ...

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
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        return f"CGA{self.euclidean_dim}D({self.clifford_notation})"
