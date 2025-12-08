"""
CGA1D (Conformal Geometric Algebra) Cl(2,1) Definition

This module provides:
- CGA1D algebra initialization using clifford library
- Geometric product multiplication table extraction
- Null basis (n_o, n_inf) definitions
- Blade indexing and grade mappings
- Reverse sign table

For 1D Conformal Geometric Algebra:
- Base Euclidean space: 1D (e1)
- Conformal basis: e+, e-
- Signature: (+, +, -)
- Total blades: 8
"""

from typing import Dict, List, Tuple
import numpy as np
from fast_clifford.codegen.cga_factory import (
    create_cga_algebra,
    compute_blade_count,
    compute_grade_indices,
    compute_reverse_signs,
    get_product_table as factory_get_product_table,
    get_upgc_point_indices,
    get_even_versor_indices,
    get_blade_names,
    verify_null_basis_properties,
)


# =============================================================================
# CGA1D Constants
# =============================================================================

EUCLIDEAN_DIM = 1
BLADE_COUNT = 8
SIGNATURE = (1, 1, -1)  # e1+, e++, e--

# Grade distribution: C(3,k) for k=0..3 gives 1,3,3,1
GRADE_SIZES = {0: 1, 1: 3, 2: 3, 3: 1}

# Blade indices by grade (from cga_factory)
_grade_indices = compute_grade_indices(EUCLIDEAN_DIM)
GRADE_0_INDICES = _grade_indices[0]  # (0,)
GRADE_1_INDICES = _grade_indices[1]  # (1, 2, 3) - e1, e+, e-
GRADE_2_INDICES = _grade_indices[2]  # (4, 5, 6) - bivectors
GRADE_3_INDICES = _grade_indices[3]  # (7,) - trivector

GRADE_INDICES = (
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
)


# =============================================================================
# Sparsity Masks
# =============================================================================

# UPGC Point: Grade 1 only (3 components)
# Layout: [e1, e+, e-]
UPGC_POINT_MASK = GRADE_1_INDICES

# EvenVersor: Grade 0, 2 (1 + 3 = 4 components)
# Note: CGA1D has no Grade 4 (max grade is 3)
# Layout: [scalar, e1+, e1-, e+-]
EVEN_VERSOR_MASK = get_even_versor_indices(EUCLIDEAN_DIM)

# EvenVersor sparse indices (full index -> sparse index mapping)
EVEN_VERSOR_SPARSE_INDICES = EVEN_VERSOR_MASK


# =============================================================================
# Reverse Signs
# =============================================================================

REVERSE_SIGNS_BY_GRADE = {
    0: 1,   # (-1)^0 = 1
    1: 1,   # (-1)^0 = 1
    2: -1,  # (-1)^1 = -1
    3: -1,  # (-1)^3 = -1
}

# Precomputed reverse signs for all 8 blades
REVERSE_SIGNS = compute_reverse_signs(EUCLIDEAN_DIM)

# EvenVersor reverse signs (4 components)
# Grade 0 (+1), Grade 2 (-1 x 3)
EVEN_VERSOR_REVERSE_SIGNS = tuple(REVERSE_SIGNS[idx] for idx in EVEN_VERSOR_SPARSE_INDICES)


# =============================================================================
# Blade Names
# =============================================================================

BLADE_NAMES = get_blade_names(EUCLIDEAN_DIM)


# =============================================================================
# CGA1D Algebra Initialization
# =============================================================================

# Global CGA1D algebra instance (lazy initialization)
_layout = None
_blades = None
_stuff = None


def _ensure_algebra():
    """Ensure the algebra is initialized."""
    global _layout, _blades, _stuff
    if _layout is None:
        _layout, _blades, _stuff = create_cga_algebra(EUCLIDEAN_DIM)


def get_layout():
    """Get the CGA1D layout object."""
    _ensure_algebra()
    return _layout


def get_blades():
    """Get the CGA1D blades dictionary."""
    _ensure_algebra()
    return _blades


def get_stuff():
    """Get the CGA1D stuff (eo, einf, up, down, etc.)."""
    _ensure_algebra()
    return _stuff


# =============================================================================
# Geometric Product Table
# =============================================================================

# Cached product table
_product_table = None


def get_product_table() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Get geometric product lookup table.

    Returns:
        Dictionary mapping (left_idx, right_idx) -> (result_idx, sign)
    """
    global _product_table
    if _product_table is None:
        _product_table = factory_get_product_table(EUCLIDEAN_DIM)
    return _product_table


# =============================================================================
# Null Basis
# =============================================================================

def get_null_basis():
    """
    Get the Null Basis vectors for CGA1D.

    Returns:
        eo: Origin point (n_o = (e- - e+) / 2)
        einf: Point at infinity (n_inf = e- + e+)

    Convention: eo * einf = -1
    """
    _ensure_algebra()
    eo = _stuff['eo']
    einf = _stuff['einf']
    return eo, einf


def verify_null_basis() -> Dict[str, bool]:
    """
    Verify the mathematical properties of the Null Basis.

    Returns:
        Dictionary of property names to verification results
    """
    return verify_null_basis_properties(EUCLIDEAN_DIM)


# =============================================================================
# Utility Functions
# =============================================================================

def get_blade_grade(index: int) -> int:
    """
    Get the grade of a blade by its index.

    Args:
        index: Blade index (0-7)

    Returns:
        Grade (0-3)
    """
    for grade, indices in enumerate(GRADE_INDICES):
        if index in indices:
            return grade
    raise ValueError(f"Invalid blade index: {index}")


def get_blade_info() -> List[Dict]:
    """
    Get detailed information about all 8 blades.

    Returns:
        List of dicts with keys: index, grade, name
    """
    _ensure_algebra()
    blade_tuples = _layout.bladeTupList
    info = []

    for idx in range(BLADE_COUNT):
        info.append({
            'index': idx,
            'grade': get_blade_grade(idx),
            'name': BLADE_NAMES[idx],
        })

    return info


def get_even_versor_sparse_index_map() -> Dict[int, int]:
    """
    Get mapping from full 8-index to sparse 4-index for EvenVersors.

    Returns:
        Dict mapping full_index -> sparse_index
    """
    return {full: sparse for sparse, full in enumerate(EVEN_VERSOR_SPARSE_INDICES)}


def get_even_versor_full_index_map() -> Dict[int, int]:
    """
    Get mapping from sparse 4-index to full 8-index for EvenVersors.

    Returns:
        Dict mapping sparse_index -> full_index
    """
    return {sparse: full for sparse, full in enumerate(EVEN_VERSOR_SPARSE_INDICES)}


def up(x_1d: np.ndarray) -> np.ndarray:
    """
    Project a 1D point to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Args:
        x_1d: 1D coordinates as (1,) array or scalar

    Returns:
        UPGC point as multivector value array (8,)
    """
    _ensure_algebra()
    up_func = _stuff['up']
    e1 = _blades['e1']

    # Handle scalar or array input
    x_val = float(x_1d[0]) if hasattr(x_1d, '__len__') else float(x_1d)
    x_mv = x_val * e1
    result = up_func(x_mv)
    return result.value


def down(X: np.ndarray) -> np.ndarray:
    """
    Project a UPGC point back to 1D coordinates.

    Args:
        X: UPGC point as multivector value array (8,) or (3,) for sparse

    Returns:
        1D coordinates as (1,) array
    """
    _ensure_algebra()
    down_func = _stuff['down']

    if len(X) == 8:
        mv = _layout.MultiVector(value=X)
    else:
        # Sparse representation - fill in grade 1 components
        mv = _layout.MultiVector()
        mv.value[1:4] = X
    result = down_func(mv)
    return np.array([result.value[1]])


if __name__ == "__main__":
    print("=== CGA1D Cl(2,1) Algebra ===")
    print(f"Blade count: {BLADE_COUNT}")
    print(f"Signature: {SIGNATURE}")
    print()

    print("Grade indices:")
    for grade, indices in enumerate(GRADE_INDICES):
        print(f"  Grade {grade}: {indices}")
    print()

    print(f"UPGC Point mask: {UPGC_POINT_MASK} ({len(UPGC_POINT_MASK)} components)")
    print(f"EvenVersor mask: {EVEN_VERSOR_MASK} ({len(EVEN_VERSOR_MASK)} components)")
    print()

    print("Blade info:")
    for info in get_blade_info():
        print(f"  {info['index']:2d}: Grade {info['grade']}, {info['name']}")
    print()

    print("Null basis verification:")
    props = verify_null_basis()
    for prop, value in props.items():
        print(f"  {prop}: {value}")
