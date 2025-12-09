"""
Code generation framework for Clifford algebras

This module provides tools for generating optimized PyTorch code
from symbolic Clifford algebra definitions.

Supports multiple CGA dimensions:
- CGA1D Cl(2,1): 8 blades, 3 Point, 4 EvenVersor
- CGA2D Cl(3,1): 16 blades, 4 Point, 8 EvenVersor
- CGA3D Cl(4,1): 32 blades, 5 Point, 16 EvenVersor
"""

from .base import AlgebraDefinition, CodeGenerator, SparsityPattern
from .cga_factory import (
    create_cga_algebra,
    compute_blade_count,
    compute_grade_indices,
    compute_reverse_signs,
    get_product_table,
    get_point_indices,
    get_even_versor_indices,
    get_blade_names,
    verify_null_basis_properties,
)
from .sparse_analysis import (
    get_point_pattern,
    get_even_versor_pattern,
    get_sandwich_product_terms_generic,
    count_sandwich_product_ops,
)
from .generate import (
    CGANDAlgebra,
    CGANDCodeGenerator,
    generate_cgand_functional,
    CGA3DAlgebra,
    CGA3DCodeGenerator,
    generate_cga3d_functional,
)

__all__ = [
    # Base classes
    "AlgebraDefinition",
    "CodeGenerator",
    "SparsityPattern",
    # Factory functions
    "create_cga_algebra",
    "compute_blade_count",
    "compute_grade_indices",
    "compute_reverse_signs",
    "get_product_table",
    "get_point_indices",
    "get_even_versor_indices",
    "get_blade_names",
    "verify_null_basis_properties",
    # Sparsity analysis
    "get_point_pattern",
    "get_even_versor_pattern",
    "get_sandwich_product_terms_generic",
    "count_sandwich_product_ops",
    # Code generators
    "CGANDAlgebra",
    "CGANDCodeGenerator",
    "generate_cgand_functional",
    "CGA3DAlgebra",
    "CGA3DCodeGenerator",
    "generate_cga3d_functional",
]
