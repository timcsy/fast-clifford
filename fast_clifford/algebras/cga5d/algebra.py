"""
CGA5D (Conformal Geometric Algebra) Cl(6,1) Definition

This module provides:
- CGA5D algebra initialization using clifford library
- Geometric product multiplication table extraction
- Null basis (n_o, n_inf) definitions
- Blade indexing and grade mappings
- Reverse sign table

CGA5D extends 5D Euclidean space with conformal model:
- Signature: (+,+,+,+,+,+,-) = 6 positive, 1 negative
- 128 blades total (2^7)
- UPGC Point: 7 components (Grade 1)
- EvenVersor: 64 components (Grade 0 + 2 + 4 + 6)
"""

from typing import Dict, List, Tuple
import numpy as np
from clifford import Cl, conformalize


# =============================================================================
# CGA Algebra Initialization
# =============================================================================

def create_cga_algebra():
    """
    Create CGA5D Cl(6,1) algebra using clifford's conformalize function.

    Returns:
        layout: The CGA layout object
        blades: Dictionary of blade objects
        stuff: Additional CGA objects (eo, einf, up, down, etc.)
    """
    # Create base 5D Euclidean algebra
    G5, blades_g5 = Cl(5)

    # Conformalize to get CGA
    layout, blades, stuff = conformalize(G5)

    return layout, blades, stuff


# Global CGA algebra instance
_layout, _blades, _stuff = create_cga_algebra()


def get_layout():
    """Get the CGA layout object."""
    return _layout


def get_blades():
    """Get the CGA blades dictionary."""
    return _blades


def get_stuff():
    """Get the CGA stuff (eo, einf, up, down, etc.)."""
    return _stuff


# =============================================================================
# Geometric Product Multiplication Table
# =============================================================================

def extract_gmt_dense() -> np.ndarray:
    """
    Extract the geometric multiplication table as a dense array.

    The GMT is stored as gmt[i, k, j] where:
    - i: left operand blade index
    - j: right operand blade index
    - k: result blade index
    - gmt[i, k, j] is the coefficient

    Returns:
        Dense array of shape (128, 128, 128) containing multiplication coefficients
    """
    gmt = _layout.gmt
    return np.asarray(gmt.todense())


def get_product_rules() -> List[Tuple[int, int, int, int]]:
    """
    Extract all non-zero geometric product rules.

    Returns:
        List of tuples (left_idx, right_idx, result_idx, sign)
        where sign is +1 or -1
    """
    gmt_dense = extract_gmt_dense()
    rules = []

    for i in range(BLADE_COUNT):
        for j in range(BLADE_COUNT):
            # gmt is indexed as [left, result, right]
            result_vec = gmt_dense[i, :, j]
            nonzero_indices = np.where(result_vec != 0)[0]

            for k in nonzero_indices:
                coeff = result_vec[k]
                # Coefficient should be +1, -1, or 0
                sign = int(np.sign(coeff))
                if sign != 0:
                    rules.append((i, j, k, sign))

    return rules


def get_product_table() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Get geometric product lookup table.

    Returns:
        Dictionary mapping (left_idx, right_idx) -> (result_idx, sign)
    """
    rules = get_product_rules()
    table = {}
    for left, right, result, sign in rules:
        table[(left, right)] = (result, sign)
    return table


# =============================================================================
# Null Basis Definition
# =============================================================================

def get_null_basis():
    """
    Get the Null Basis vectors for CGA.

    Returns:
        eo: Origin point (n_o = (e- - e+) / 2)
        einf: Point at infinity (n_inf = e- + e+)

    Convention: eo * einf = -1
    """
    eo = _stuff['eo']
    einf = _stuff['einf']
    return eo, einf


def verify_null_basis_properties() -> Dict[str, bool]:
    """
    Verify the mathematical properties of the Null Basis.

    Returns:
        Dictionary of property names to verification results
    """
    eo, einf = get_null_basis()

    # eo^2 = 0
    eo_squared = eo * eo
    eo_sq_zero = np.allclose(eo_squared.value, 0, atol=1e-10)

    # einf^2 = 0
    einf_squared = einf * einf
    einf_sq_zero = np.allclose(einf_squared.value, 0, atol=1e-10)

    # eo * einf = -1
    eo_einf = eo * einf
    # The inner product should give -1 in the scalar part
    inner_product_value = float(eo_einf.value[0])
    inner_is_minus_one = np.allclose(inner_product_value, -1.0, atol=1e-10)

    return {
        'eo_squared_zero': eo_sq_zero,
        'einf_squared_zero': einf_sq_zero,
        'eo_einf_minus_one': inner_is_minus_one,
    }


# =============================================================================
# Blade Indexing and Grade Mapping
# =============================================================================

# Euclidean dimension
EUCLIDEAN_DIM = 5

# Number of blades in Cl(6,1)
BLADE_COUNT = 128

# Grade distribution: C(7,k) for k=0..7 gives 1,7,21,35,35,21,7,1
GRADE_SIZES = {0: 1, 1: 7, 2: 21, 3: 35, 4: 35, 5: 21, 6: 7, 7: 1}

# Blade indices by grade (computed from clifford layout)
GRADE_0_INDICES = (0,)
GRADE_1_INDICES = (1, 2, 3, 4, 5, 6, 7)  # e1, e2, e3, e4, e5, e+, e-
GRADE_2_INDICES = tuple(range(8, 29))  # 21 components
GRADE_3_INDICES = tuple(range(29, 64))  # 35 components
GRADE_4_INDICES = tuple(range(64, 99))  # 35 components
GRADE_5_INDICES = tuple(range(99, 120))  # 21 components
GRADE_6_INDICES = tuple(range(120, 127))  # 7 components
GRADE_7_INDICES = (127,)

# All grade indices as a tuple of tuples
GRADE_INDICES = (
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
    GRADE_4_INDICES,
    GRADE_5_INDICES,
    GRADE_6_INDICES,
    GRADE_7_INDICES,
)


def get_blade_grade(index: int) -> int:
    """
    Get the grade of a blade by its index.

    Args:
        index: Blade index (0-127)

    Returns:
        Grade (0-7)
    """
    for grade, indices in enumerate(GRADE_INDICES):
        if index in indices:
            return grade
    raise ValueError(f"Invalid blade index: {index}")


def get_blade_names() -> List[str]:
    """
    Get human-readable names for all 128 blades.

    Returns:
        List of blade names in index order
    """
    return list(_layout.bladeTupList)


def get_blade_info() -> List[Dict]:
    """
    Get detailed information about all 128 blades.

    Returns:
        List of dicts with keys: index, grade, name, basis_vectors
    """
    blade_tuples = _layout.bladeTupList
    info = []

    for idx, blade_tuple in enumerate(blade_tuples):
        info.append({
            'index': idx,
            'grade': get_blade_grade(idx),
            'name': str(blade_tuple),
            'basis_vectors': blade_tuple,
        })

    return info


# Sparsity masks for common multivector types
POINT_MASK = GRADE_1_INDICES  # Grade 1 only (7 components)

# EvenVersor mask: Grade 0, 2, 4, 6 (1 + 21 + 35 + 7 = 64 components)
EVEN_VERSOR_MASK = GRADE_0_INDICES + GRADE_2_INDICES + GRADE_4_INDICES + GRADE_6_INDICES

# EvenVersor sparse indices (ordered for the 64-component representation)
EVEN_VERSOR_SPARSE_INDICES = GRADE_0_INDICES + GRADE_2_INDICES + GRADE_4_INDICES + GRADE_6_INDICES


def get_even_versor_sparse_index_map() -> Dict[int, int]:
    """
    Get mapping from full 128-index to sparse 64-index for EvenVersors.

    Returns:
        Dict mapping full_index -> sparse_index
    """
    return {full: sparse for sparse, full in enumerate(EVEN_VERSOR_SPARSE_INDICES)}


def get_even_versor_full_index_map() -> Dict[int, int]:
    """
    Get mapping from sparse 64-index to full 128-index for EvenVersors.

    Returns:
        Dict mapping sparse_index -> full_index
    """
    return {sparse: full for sparse, full in enumerate(EVEN_VERSOR_SPARSE_INDICES)}


# =============================================================================
# Reverse Sign Table
# =============================================================================

def compute_reverse_sign(grade: int) -> int:
    """
    Compute the sign for the reverse operation on a blade of given grade.

    Formula: (-1)^(k*(k-1)/2) where k is the grade

    Args:
        grade: The grade of the blade (0-7)

    Returns:
        +1 or -1
    """
    exponent = grade * (grade - 1) // 2
    return (-1) ** exponent


# Precomputed reverse signs by grade
REVERSE_SIGNS_BY_GRADE = {
    0: 1,   # (-1)^0 = 1
    1: 1,   # (-1)^0 = 1
    2: -1,  # (-1)^1 = -1
    3: -1,  # (-1)^3 = -1
    4: 1,   # (-1)^6 = 1
    5: 1,   # (-1)^10 = 1
    6: -1,  # (-1)^15 = -1
    7: -1,  # (-1)^21 = -1
}


def get_reverse_signs() -> Tuple[int, ...]:
    """
    Get the reverse sign for all 128 blades.

    Returns:
        Tuple of 128 signs (+1 or -1)
    """
    signs = []
    for idx in range(BLADE_COUNT):
        grade = get_blade_grade(idx)
        signs.append(REVERSE_SIGNS_BY_GRADE[grade])
    return tuple(signs)


# Precomputed reverse signs for all 128 blades
REVERSE_SIGNS = get_reverse_signs()


def get_even_versor_reverse_signs() -> Tuple[int, ...]:
    """
    Get the reverse signs for the 64 EvenVersor components.

    EvenVersor components are in order:
    - Grade 0: 1 component (sign = +1)
    - Grade 2: 21 components (sign = -1)
    - Grade 4: 35 components (sign = +1)
    - Grade 6: 7 components (sign = -1)

    Returns:
        Tuple of 64 signs
    """
    signs = []
    for full_idx in EVEN_VERSOR_SPARSE_INDICES:
        grade = get_blade_grade(full_idx)
        signs.append(REVERSE_SIGNS_BY_GRADE[grade])
    return tuple(signs)


# Precomputed EvenVersor reverse signs
EVEN_VERSOR_REVERSE_SIGNS = get_even_versor_reverse_signs()


# =============================================================================
# Utility Functions
# =============================================================================

def up(x_5d: np.ndarray) -> np.ndarray:
    """
    Project a 5D point to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Args:
        x_5d: 5D coordinates as (5,) array

    Returns:
        UPGC point as multivector value array (128,)
    """
    up_func = _stuff['up']
    e1, e2, e3, e4, e5 = (_blades['e1'], _blades['e2'], _blades['e3'],
                          _blades['e4'], _blades['e5'])
    x_mv = x_5d[0] * e1 + x_5d[1] * e2 + x_5d[2] * e3 + x_5d[3] * e4 + x_5d[4] * e5
    result = up_func(x_mv)
    return result.value


def down(X: np.ndarray) -> np.ndarray:
    """
    Project a UPGC point back to 5D coordinates.

    Args:
        X: UPGC point as multivector value array (128,) or (7,) for sparse

    Returns:
        5D coordinates as (5,) array
    """
    down_func = _stuff['down']
    # Create multivector from value array
    mv = _layout.MultiVector(value=X if len(X) == 128 else np.zeros(128))
    if len(X) == 7:
        # Sparse representation - fill in grade 1 components
        mv.value[1:8] = X
    result = down_func(mv)
    return np.array([result.value[1], result.value[2], result.value[3],
                     result.value[4], result.value[5]])
