"""
CGA0D (Conformal Geometric Algebra) Cl(1,1) Definition

This module provides:
- CGA0D algebra initialization
- Geometric product multiplication table
- Null basis (n_o, n_inf) definitions
- Blade indexing and grade mappings
- Reverse sign table

For 0D Conformal Geometric Algebra:
- Base Euclidean space: 0D (no Euclidean basis)
- Conformal basis: e+, e-
- Signature: (+, -)
- Total blades: 4
"""

from typing import Dict, List, Tuple
import numpy as np


# =============================================================================
# CGA0D Constants
# =============================================================================

EUCLIDEAN_DIM = 0
BLADE_COUNT = 4
SIGNATURE = (1, -1)  # e++, e--

# Grade distribution: C(2,k) for k=0..2 gives 1,2,1
GRADE_SIZES = {0: 1, 1: 2, 2: 1}

# Blade indices by grade
# Index 0: scalar (1)
# Index 1: e+
# Index 2: e-
# Index 3: e+-
GRADE_0_INDICES = (0,)
GRADE_1_INDICES = (1, 2)
GRADE_2_INDICES = (3,)

GRADE_INDICES = (
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
)


# =============================================================================
# Sparsity Masks
# =============================================================================

# UPGC Point: Grade 1 only (2 components)
# Layout: [e+, e-]
POINT_MASK = GRADE_1_INDICES

# EvenVersor: Grade 0, 2 (1 + 1 = 2 components)
# Layout: [scalar, e+-]
EVEN_VERSOR_MASK = (0, 3)

# EvenVersor sparse indices (full index -> sparse index mapping)
EVEN_VERSOR_SPARSE_INDICES = EVEN_VERSOR_MASK


# =============================================================================
# Reverse Signs
# =============================================================================

REVERSE_SIGNS_BY_GRADE = {
    0: 1,   # (-1)^0 = 1
    1: 1,   # (-1)^0 = 1
    2: -1,  # (-1)^1 = -1
}

# Precomputed reverse signs for all 4 blades
# Grade 0: +1, Grade 1: +1, +1, Grade 2: -1
REVERSE_SIGNS = (1, 1, 1, -1)

# EvenVersor reverse signs (2 components)
# Grade 0 (+1), Grade 2 (-1)
EVEN_VERSOR_REVERSE_SIGNS = (1, -1)


# =============================================================================
# Blade Names
# =============================================================================

BLADE_NAMES = ['1', 'e+', 'e-', 'e+-']


# =============================================================================
# CGA0D Algebra Initialization (using clifford library)
# =============================================================================

# Global CGA0D algebra instance (lazy initialization)
_layout = None
_blades = None
_stuff = None


def _ensure_algebra():
    """Ensure the algebra is initialized."""
    global _layout, _blades, _stuff
    if _layout is None:
        from clifford import Cl, conformalize
        # For CGA0D, we use Cl(0) conformalized
        # However, clifford doesn't support Cl(0) directly
        # We manually construct the algebra
        G_0, _ = Cl(0)  # 0D Euclidean algebra (just scalar)
        _layout, _blades, _stuff = conformalize(G_0)


def get_layout():
    """Get the CGA0D layout object."""
    _ensure_algebra()
    return _layout


def get_blades():
    """Get the CGA0D blades dictionary."""
    _ensure_algebra()
    return _blades


def get_stuff():
    """Get the CGA0D stuff (eo, einf, up, down, etc.)."""
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

    Cayley table for Cl(1,1) with signature (+1, -1):
    (Indices: 0=1, 1=e+, 2=e-, 3=e+-)

    From clifford library GMT:
         |  1    e+   e-   e+-
    -----|-----------------------
      1  |  1    e+   e-   e+-
      e+ |  e+   1    e+-  e-
      e- |  e-  -e+- -1    e+
     e+- | e+-  -e-  -e+   1

    Key: e+^2 = 1, e-^2 = -1, e+-^2 = 1
    """
    global _product_table
    if _product_table is None:
        _product_table = {
            (0, 0): (0, 1),   # 1 * 1 = 1
            (0, 1): (1, 1),   # 1 * e+ = e+
            (0, 2): (2, 1),   # 1 * e- = e-
            (0, 3): (3, 1),   # 1 * e+- = e+-
            (1, 0): (1, 1),   # e+ * 1 = e+
            (1, 1): (0, 1),   # e+ * e+ = 1
            (1, 2): (3, 1),   # e+ * e- = e+-
            (1, 3): (2, 1),   # e+ * e+- = e-
            (2, 0): (2, 1),   # e- * 1 = e-
            (2, 1): (3, -1),  # e- * e+ = -e+-
            (2, 2): (0, -1),  # e- * e- = -1
            (2, 3): (1, 1),   # e- * e+- = e+
            (3, 0): (3, 1),   # e+- * 1 = e+-
            (3, 1): (2, -1),  # e+- * e+ = -e-
            (3, 2): (1, -1),  # e+- * e- = -e+
            (3, 3): (0, 1),   # e+- * e+- = 1
        }
    return _product_table


# =============================================================================
# Null Basis
# =============================================================================

def get_null_basis():
    """
    Get the Null Basis vectors for CGA0D.

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
    _ensure_algebra()

    eo = _stuff['eo']
    einf = _stuff['einf']

    # eo^2 = 0
    eo_squared = eo * eo
    eo_sq_zero = np.allclose(eo_squared.value, 0, atol=1e-10)

    # einf^2 = 0
    einf_squared = einf * einf
    einf_sq_zero = np.allclose(einf_squared.value, 0, atol=1e-10)

    # eo * einf = -1
    eo_einf = eo * einf
    inner_product_value = float(eo_einf.value[0])
    inner_is_minus_one = np.allclose(inner_product_value, -1.0, atol=1e-10)

    return {
        'eo_squared_zero': eo_sq_zero,
        'einf_squared_zero': einf_sq_zero,
        'eo_einf_minus_one': inner_is_minus_one,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_blade_grade(index: int) -> int:
    """
    Get the grade of a blade by its index.

    Args:
        index: Blade index (0-3)

    Returns:
        Grade (0-2)
    """
    for grade, indices in enumerate(GRADE_INDICES):
        if index in indices:
            return grade
    raise ValueError(f"Invalid blade index: {index}")


def get_blade_info() -> List[Dict]:
    """
    Get detailed information about all 4 blades.

    Returns:
        List of dicts with keys: index, grade, name
    """
    info = []
    for idx in range(BLADE_COUNT):
        info.append({
            'index': idx,
            'grade': get_blade_grade(idx),
            'name': BLADE_NAMES[idx],
        })
    return info


if __name__ == "__main__":
    print("=== CGA0D Cl(1,1) Algebra ===")
    print(f"Blade count: {BLADE_COUNT}")
    print(f"Signature: {SIGNATURE}")
    print()

    print("Grade indices:")
    for grade, indices in enumerate(GRADE_INDICES):
        print(f"  Grade {grade}: {indices}")
    print()

    print(f"UPGC Point mask: {POINT_MASK} ({len(POINT_MASK)} components)")
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
