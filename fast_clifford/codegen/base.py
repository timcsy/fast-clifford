"""
Base classes and interfaces for Clifford algebra code generation.

This module provides abstract base classes for:
- Algebra definition interface
- Code generator interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class AlgebraDefinition(ABC):
    """
    Abstract base class for Clifford algebra definitions.

    Subclasses must implement methods to provide:
    - Algebra signature and blade count
    - Geometric product multiplication table
    - Grade structure
    - Reverse signs
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algebra (e.g., 'cga3d', 'pga3d')."""
        pass

    @property
    @abstractmethod
    def signature(self) -> Tuple[int, ...]:
        """Metric signature (e.g., (1, 1, 1, 1, -1) for CGA)."""
        pass

    @property
    @abstractmethod
    def blade_count(self) -> int:
        """Total number of blades (2^n for n basis vectors)."""
        pass

    @abstractmethod
    def get_grade_indices(self, grade: int) -> Tuple[int, ...]:
        """Get blade indices for a given grade."""
        pass

    @abstractmethod
    def get_product_table(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Get geometric product lookup table.

        Returns:
            Dict mapping (left_idx, right_idx) -> (result_idx, sign)
        """
        pass

    @abstractmethod
    def get_reverse_signs(self) -> Tuple[int, ...]:
        """Get reverse sign for each blade (+1 or -1)."""
        pass


class CodeGenerator(ABC):
    """
    Abstract base class for code generators.

    Subclasses implement specific code generation strategies.
    """

    def __init__(self, algebra: AlgebraDefinition):
        """
        Initialize generator with an algebra definition.

        Args:
            algebra: The algebra definition to generate code for
        """
        self.algebra = algebra

    @abstractmethod
    def generate_constants(self) -> str:
        """Generate constant definitions (blade indices, masks, etc.)."""
        pass

    @abstractmethod
    def generate_geometric_product(self) -> str:
        """Generate the full geometric product function."""
        pass

    @abstractmethod
    def generate_reverse(self) -> str:
        """Generate the reverse operation function."""
        pass

    @abstractmethod
    def generate_module(self) -> str:
        """Generate the complete module code."""
        pass


class SparsityPattern:
    """
    Represents the sparsity pattern of a multivector type.

    Used for optimized code generation when input/output
    sparsity is known at code generation time.
    """

    def __init__(self, name: str, nonzero_indices: Tuple[int, ...], blade_count: int = 32):
        """
        Create a sparsity pattern.

        Args:
            name: Human-readable name (e.g., 'point', 'even_versor')
            nonzero_indices: Tuple of blade indices that may be non-zero
            blade_count: Total number of blades in the algebra
        """
        self.name = name
        self.nonzero_indices = nonzero_indices
        self.blade_count = blade_count
        self.sparse_count = len(nonzero_indices)

        # Create mappings between full and sparse indices
        self._full_to_sparse = {full: sparse for sparse, full in enumerate(nonzero_indices)}
        self._sparse_to_full = {sparse: full for sparse, full in enumerate(nonzero_indices)}

    def full_to_sparse(self, full_index: int) -> int:
        """Convert full index to sparse index."""
        return self._full_to_sparse[full_index]

    def sparse_to_full(self, sparse_index: int) -> int:
        """Convert sparse index to full index."""
        return self._sparse_to_full[sparse_index]

    def contains(self, full_index: int) -> bool:
        """Check if a full index is in the non-zero set."""
        return full_index in self._full_to_sparse

    def __repr__(self) -> str:
        return f"SparsityPattern('{self.name}', {self.nonzero_indices})"
