"""
PyTorch Neural Network Layers for Clifford Algebras

This module provides torch.nn.Module wrappers for Clifford algebra operations,
enabling integration with PyTorch neural networks and ONNX export.

All layers are:
- ONNX-compatible (loop-free operations)
- torch.jit.script compatible
- Input dtype forced to float32 for numerical stability (FR-054)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .base import CliffordAlgebraBase


class CliffordTransformLayer(nn.Module):
    """
    Sandwich product transformation layer.

    Applies versor transformation: output = v @ x @ ~v

    This layer is useful for applying learned transformations
    (rotations, translations, etc.) in neural networks.

    Attributes:
        algebra: The Clifford algebra instance
        use_rotor: If True, uses accelerated rotor operations
    """

    def __init__(
        self,
        algebra: "CliffordAlgebraBase",
        use_rotor: bool = True,
    ):
        """
        Initialize transform layer.

        Args:
            algebra: The Clifford algebra to use
            use_rotor: If True, expects rotor input and uses accelerated ops
        """
        super().__init__()
        self.algebra = algebra
        self.use_rotor = use_rotor

    def forward(self, versor: Tensor, x: Tensor) -> Tensor:
        """
        Apply sandwich transformation.

        Args:
            versor: Transformation versor
                - If use_rotor=True: shape (..., count_rotor)
                - If use_rotor=False: shape (..., count_blade)
            x: Input to transform, shape (..., count_blade)

        Returns:
            Transformed output, shape (..., count_blade)
        """
        # Force float32 for numerical stability (FR-054)
        versor = versor.float()
        x = x.float()

        if self.use_rotor:
            return self.algebra.sandwich_rotor(versor, x)
        else:
            return self.algebra.sandwich(versor, x)


class GeometricProductLayer(nn.Module):
    """
    Geometric product layer.

    Computes ab for multivectors a and b.
    """

    def __init__(self, algebra: "CliffordAlgebraBase"):
        """
        Initialize geometric product layer.

        Args:
            algebra: The Clifford algebra to use
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute geometric product.

        Args:
            a: Left operand, shape (..., count_blade)
            b: Right operand, shape (..., count_blade)

        Returns:
            Product result, shape (..., count_blade)
        """
        a = a.float()
        b = b.float()
        return self.algebra.geometric_product(a, b)


class RotorCompositionLayer(nn.Module):
    """
    Rotor composition layer.

    Computes r1 * r2 using accelerated rotor multiplication.
    """

    def __init__(self, algebra: "CliffordAlgebraBase"):
        """
        Initialize rotor composition layer.

        Args:
            algebra: The Clifford algebra to use
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, r1: Tensor, r2: Tensor) -> Tensor:
        """
        Compose two rotors.

        Args:
            r1: Left rotor, shape (..., count_rotor)
            r2: Right rotor, shape (..., count_rotor)

        Returns:
            Composed rotor, shape (..., count_rotor)
        """
        r1 = r1.float()
        r2 = r2.float()
        return self.algebra.compose_rotor(r1, r2)


class ExpBivectorLayer(nn.Module):
    """
    Bivector exponential layer.

    Maps bivector to rotor: exp(B) -> R
    """

    def __init__(self, algebra: "CliffordAlgebraBase"):
        """
        Initialize exp_bivector layer.

        Args:
            algebra: The Clifford algebra to use
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, B: Tensor) -> Tensor:
        """
        Compute rotor from bivector.

        Args:
            B: Bivector, shape (..., count_bivector)

        Returns:
            Rotor, shape (..., count_rotor)
        """
        B = B.float()
        return self.algebra.exp_bivector(B)


class NormLayer(nn.Module):
    """
    Multivector norm layer.

    Computes |mv| = sqrt(|<mv * ~mv>_0|)
    """

    def __init__(self, algebra: "CliffordAlgebraBase", squared: bool = False):
        """
        Initialize norm layer.

        Args:
            algebra: The Clifford algebra to use
            squared: If True, returns norm squared instead of norm
        """
        super().__init__()
        self.algebra = algebra
        self.squared = squared

    def forward(self, mv: Tensor) -> Tensor:
        """
        Compute norm.

        Args:
            mv: Multivector, shape (..., count_blade)

        Returns:
            Norm (or norm squared), shape (..., 1)
        """
        mv = mv.float()
        norm_sq = self.algebra.norm_squared(mv)
        if self.squared:
            return norm_sq
        return torch.sqrt(torch.abs(norm_sq) + 1e-12)


class GradeProjectionLayer(nn.Module):
    """
    Grade projection layer.

    Extracts specific grade components from multivector.
    """

    def __init__(self, algebra: "CliffordAlgebraBase", grade: int):
        """
        Initialize grade projection layer.

        Args:
            algebra: The Clifford algebra to use
            grade: Grade to extract (0 to max_grade)
        """
        super().__init__()
        self.algebra = algebra
        self.grade = grade

    def forward(self, mv: Tensor) -> Tensor:
        """
        Extract grade components.

        Args:
            mv: Multivector, shape (..., count_blade)

        Returns:
            Grade-projected multivector, shape (..., count_blade)
        """
        mv = mv.float()
        return self.algebra.select_grade(mv, self.grade)


class DualLayer(nn.Module):
    """
    Poincare dual layer.

    Computes mv* = mv << I (left contraction with pseudoscalar)
    """

    def __init__(self, algebra: "CliffordAlgebraBase"):
        """
        Initialize dual layer.

        Args:
            algebra: The Clifford algebra to use
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, mv: Tensor) -> Tensor:
        """
        Compute dual.

        Args:
            mv: Multivector, shape (..., count_blade)

        Returns:
            Dual multivector, shape (..., count_blade)
        """
        mv = mv.float()
        return self.algebra.dual(mv)


class ReverseLayer(nn.Module):
    """
    Reversion layer.

    Computes ~mv (reverses basis vector order in each blade)
    """

    def __init__(self, algebra: "CliffordAlgebraBase"):
        """
        Initialize reverse layer.

        Args:
            algebra: The Clifford algebra to use
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, mv: Tensor) -> Tensor:
        """
        Compute reverse.

        Args:
            mv: Multivector, shape (..., count_blade)

        Returns:
            Reversed multivector, shape (..., count_blade)
        """
        mv = mv.float()
        return self.algebra.reverse(mv)


class LearnableRotorLayer(nn.Module):
    """
    Layer with learnable rotor parameters.

    Useful for learning transformations in geometric deep learning.
    """

    def __init__(
        self,
        algebra: "CliffordAlgebraBase",
        num_rotors: int = 1,
        init_identity: bool = True,
    ):
        """
        Initialize learnable rotor layer.

        Args:
            algebra: The Clifford algebra to use
            num_rotors: Number of independent rotors to learn
            init_identity: If True, initialize to identity rotor
        """
        super().__init__()
        self.algebra = algebra
        self.num_rotors = num_rotors

        # Initialize rotor parameters
        rotor_data = torch.zeros(num_rotors, algebra.count_rotor)
        if init_identity:
            rotor_data[:, 0] = 1.0  # Scalar = 1 is identity rotor

        self.rotor = nn.Parameter(rotor_data)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply learned rotor transformation.

        Args:
            x: Input, shape (..., count_blade)

        Returns:
            Transformed output, shape (..., count_blade)
        """
        x = x.float()

        # Normalize rotor to unit length
        rotor_normalized = self.algebra.normalize_rotor(self.rotor)

        # Apply sandwich product
        # Handle broadcasting for multiple rotors
        if self.num_rotors == 1:
            return self.algebra.sandwich_rotor(rotor_normalized[0], x)
        else:
            # Apply each rotor sequentially
            result = x
            for i in range(self.num_rotors):
                result = self.algebra.sandwich_rotor(rotor_normalized[i], result)
            return result

    def get_rotor(self, normalized: bool = True) -> Tensor:
        """
        Get the learned rotor(s).

        Args:
            normalized: If True, return unit-normalized rotor

        Returns:
            Rotor tensor, shape (num_rotors, count_rotor)
        """
        if normalized:
            return self.algebra.normalize_rotor(self.rotor)
        return self.rotor


class LearnableBivectorLayer(nn.Module):
    """
    Layer with learnable bivector parameters.

    The bivector is exponentiated to produce a rotor for transformation.
    This is often more stable for optimization than learning rotors directly.
    """

    def __init__(
        self,
        algebra: "CliffordAlgebraBase",
        num_bivectors: int = 1,
        init_zero: bool = True,
    ):
        """
        Initialize learnable bivector layer.

        Args:
            algebra: The Clifford algebra to use
            num_bivectors: Number of independent bivectors to learn
            init_zero: If True, initialize to zero (identity transformation)
        """
        super().__init__()
        self.algebra = algebra
        self.num_bivectors = num_bivectors

        # Initialize bivector parameters
        bivector_data = torch.zeros(num_bivectors, algebra.count_bivector)
        self.bivector = nn.Parameter(bivector_data)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply transformation via exp(bivector).

        Args:
            x: Input, shape (..., count_blade)

        Returns:
            Transformed output, shape (..., count_blade)
        """
        x = x.float()

        # Convert bivector to rotor via exp
        rotor = self.algebra.exp_bivector(self.bivector)

        # Apply sandwich product
        if self.num_bivectors == 1:
            return self.algebra.sandwich_rotor(rotor[0], x)
        else:
            result = x
            for i in range(self.num_bivectors):
                result = self.algebra.sandwich_rotor(rotor[i], result)
            return result

    def get_rotor(self) -> Tensor:
        """
        Get the rotor corresponding to current bivector.

        Returns:
            Rotor tensor, shape (num_bivectors, count_rotor)
        """
        return self.algebra.exp_bivector(self.bivector)

    def get_bivector(self) -> Tensor:
        """
        Get the learned bivector(s).

        Returns:
            Bivector tensor, shape (num_bivectors, count_bivector)
        """
        return self.bivector
