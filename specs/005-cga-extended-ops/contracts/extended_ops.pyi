"""
Type stubs for CGA Extended Operations

Defines the API contracts for motor_compose, inner_product, and exp_bivector.
"""

from torch import Tensor
from typing import Protocol, runtime_checkable


@runtime_checkable
class CGAAlgebraExtendedOps(Protocol):
    """Extended operations protocol for CGA algebras."""

    @property
    def bivector_count(self) -> int:
        """Number of Bivector components (Grade 2)"""
        ...

    def motor_compose(self, m1: Tensor, m2: Tensor) -> Tensor:
        """
        Compose two motors via geometric product.

        Computes M_result = M1 * M2

        Args:
            m1: First motor, shape (..., motor_count)
            m2: Second motor, shape (..., motor_count)

        Returns:
            Composed motor, shape (..., motor_count)

        Note:
            - Motor × Motor = Motor (even grades preserved)
            - Non-commutative: motor_compose(m1, m2) != motor_compose(m2, m1)
            - Associative: compose(compose(a,b),c) == compose(a,compose(b,c))
        """
        ...

    def inner_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute geometric inner product (metric inner product).

        Returns the Grade 0 component of the geometric product,
        with proper CGA metric signature handling.

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Scalar inner product, shape (..., 1)

        Note:
            - CGA metric is (+,+,...,+,-), not Euclidean
            - For null basis: inner_product(eo, einf) == -1
            - Symmetric: inner_product(a, b) == inner_product(b, a)
        """
        ...

    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Compute exponential map from Bivector to Motor.

        Implements exp(B) = cos(θ) + sin(θ)/θ * B
        where θ² = -B²

        Args:
            B: Bivector, shape (..., bivector_count)

        Returns:
            Motor, shape (..., motor_count)

        Note:
            - exp(0) returns identity motor (1, 0, 0, ...)
            - Numerically stable for small angles (uses sinc)
            - exp(B) * exp(-B) ≈ identity
        """
        ...


# Hardcoded functional signatures (per dimension)

def motor_compose_sparse_0d(m1: Tensor, m2: Tensor) -> Tensor:
    """CGA0D motor composition. m1, m2: (..., 2) -> (..., 2)"""
    ...

def motor_compose_sparse_1d(m1: Tensor, m2: Tensor) -> Tensor:
    """CGA1D motor composition. m1, m2: (..., 4) -> (..., 4)"""
    ...

def motor_compose_sparse_2d(m1: Tensor, m2: Tensor) -> Tensor:
    """CGA2D motor composition. m1, m2: (..., 7) -> (..., 7)"""
    ...

def motor_compose_sparse_3d(m1: Tensor, m2: Tensor) -> Tensor:
    """CGA3D motor composition. m1, m2: (..., 16) -> (..., 16)"""
    ...

def motor_compose_sparse_4d(m1: Tensor, m2: Tensor) -> Tensor:
    """CGA4D motor composition. m1, m2: (..., 31) -> (..., 31)"""
    ...

def motor_compose_sparse_5d(m1: Tensor, m2: Tensor) -> Tensor:
    """CGA5D motor composition. m1, m2: (..., 64) -> (..., 64)"""
    ...


def inner_product_full_0d(a: Tensor, b: Tensor) -> Tensor:
    """CGA0D inner product. a, b: (..., 4) -> (..., 1)"""
    ...

def inner_product_full_1d(a: Tensor, b: Tensor) -> Tensor:
    """CGA1D inner product. a, b: (..., 8) -> (..., 1)"""
    ...

def inner_product_full_2d(a: Tensor, b: Tensor) -> Tensor:
    """CGA2D inner product. a, b: (..., 16) -> (..., 1)"""
    ...

def inner_product_full_3d(a: Tensor, b: Tensor) -> Tensor:
    """CGA3D inner product. a, b: (..., 32) -> (..., 1)"""
    ...

def inner_product_full_4d(a: Tensor, b: Tensor) -> Tensor:
    """CGA4D inner product. a, b: (..., 64) -> (..., 1)"""
    ...

def inner_product_full_5d(a: Tensor, b: Tensor) -> Tensor:
    """CGA5D inner product. a, b: (..., 128) -> (..., 1)"""
    ...


def exp_bivector_0d(B: Tensor) -> Tensor:
    """CGA0D exp map. B: (..., 1) -> Motor: (..., 2)"""
    ...

def exp_bivector_1d(B: Tensor) -> Tensor:
    """CGA1D exp map. B: (..., 3) -> Motor: (..., 4)"""
    ...

def exp_bivector_2d(B: Tensor) -> Tensor:
    """CGA2D exp map. B: (..., 6) -> Motor: (..., 7)"""
    ...

def exp_bivector_3d(B: Tensor) -> Tensor:
    """CGA3D exp map. B: (..., 10) -> Motor: (..., 16)"""
    ...

def exp_bivector_4d(B: Tensor) -> Tensor:
    """CGA4D exp map. B: (..., 15) -> Motor: (..., 31)"""
    ...

def exp_bivector_5d(B: Tensor) -> Tensor:
    """CGA5D exp map. B: (..., 21) -> Motor: (..., 64)"""
    ...
