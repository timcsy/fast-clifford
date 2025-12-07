"""
CGA (Conformal Geometric Algebra) Cl(4,1) Definition

This module provides:
- CGA algebra initialization using clifford library
- Geometric product multiplication table extraction
- Null basis (n_o, n_inf) definitions
- Blade indexing and grade mappings
- Reverse sign table
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from clifford import Cl, conformalize


# =============================================================================
# CGA Algebra Initialization (T008)
# =============================================================================

def create_cga_algebra():
    """
    Create CGA Cl(4,1) algebra using clifford's conformalize function.

    Returns:
        layout: The CGA layout object
        blades: Dictionary of blade objects
        stuff: Additional CGA objects (eo, einf, up, down, etc.)
    """
    # Create base 3D Euclidean algebra
    G3, blades_g3 = Cl(3)

    # Conformalize to get CGA
    layout, blades, stuff = conformalize(G3)

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
# Geometric Product Multiplication Table (T009)
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
        Dense array of shape (32, 32, 32) containing multiplication coefficients
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

    for i in range(32):
        for j in range(32):
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
# Null Basis Definition (T010)
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
# Blade Indexing and Grade Mapping (T011)
# =============================================================================

# Number of blades in Cl(4,1)
BLADE_COUNT = 32

# Grade distribution: C(5,k) for k=0..5 gives 1,5,10,10,5,1
GRADE_SIZES = {0: 1, 1: 5, 2: 10, 3: 10, 4: 5, 5: 1}

# Blade indices by grade
GRADE_0_INDICES = (0,)
GRADE_1_INDICES = (1, 2, 3, 4, 5)
GRADE_2_INDICES = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
GRADE_3_INDICES = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
GRADE_4_INDICES = (26, 27, 28, 29, 30)
GRADE_5_INDICES = (31,)

# All grade indices as a tuple of tuples
GRADE_INDICES = (
    GRADE_0_INDICES,
    GRADE_1_INDICES,
    GRADE_2_INDICES,
    GRADE_3_INDICES,
    GRADE_4_INDICES,
    GRADE_5_INDICES,
)


def get_blade_grade(index: int) -> int:
    """
    Get the grade of a blade by its index.

    Args:
        index: Blade index (0-31)

    Returns:
        Grade (0-5)
    """
    for grade, indices in enumerate(GRADE_INDICES):
        if index in indices:
            return grade
    raise ValueError(f"Invalid blade index: {index}")


def get_blade_names() -> List[str]:
    """
    Get human-readable names for all 32 blades.

    Returns:
        List of blade names in index order
    """
    return list(_layout.bladeTupList)


def get_blade_info() -> List[Dict]:
    """
    Get detailed information about all 32 blades.

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
UPGC_POINT_MASK = GRADE_1_INDICES  # Grade 1 only (5 components)

# Motor mask: Grade 0, 2, 4 (1 + 10 + 5 = 16 components)
MOTOR_MASK = GRADE_0_INDICES + GRADE_2_INDICES + GRADE_4_INDICES

# Motor sparse indices (ordered for the 16-component representation)
MOTOR_SPARSE_INDICES = (0,) + tuple(range(6, 16)) + tuple(range(26, 31))


def get_motor_sparse_index_map() -> Dict[int, int]:
    """
    Get mapping from full 32-index to sparse 16-index for motors.

    Returns:
        Dict mapping full_index -> sparse_index
    """
    return {full: sparse for sparse, full in enumerate(MOTOR_SPARSE_INDICES)}


def get_motor_full_index_map() -> Dict[int, int]:
    """
    Get mapping from sparse 16-index to full 32-index for motors.

    Returns:
        Dict mapping sparse_index -> full_index
    """
    return {sparse: full for sparse, full in enumerate(MOTOR_SPARSE_INDICES)}


# =============================================================================
# Reverse Sign Table (T012)
# =============================================================================

def compute_reverse_sign(grade: int) -> int:
    """
    Compute the sign for the reverse operation on a blade of given grade.

    Formula: (-1)^(k*(k-1)/2) where k is the grade

    Args:
        grade: The grade of the blade (0-5)

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
}


def get_reverse_signs() -> Tuple[int, ...]:
    """
    Get the reverse sign for all 32 blades.

    Returns:
        Tuple of 32 signs (+1 or -1)
    """
    signs = []
    for idx in range(BLADE_COUNT):
        grade = get_blade_grade(idx)
        signs.append(REVERSE_SIGNS_BY_GRADE[grade])
    return tuple(signs)


# Precomputed reverse signs for all 32 blades
REVERSE_SIGNS = get_reverse_signs()


def get_motor_reverse_signs() -> Tuple[int, ...]:
    """
    Get the reverse signs for the 16 motor components.

    Motor components are in order:
    - Grade 0: 1 component (sign = +1)
    - Grade 2: 10 components (sign = -1)
    - Grade 4: 5 components (sign = +1)

    Returns:
        Tuple of 16 signs
    """
    signs = []
    for full_idx in MOTOR_SPARSE_INDICES:
        grade = get_blade_grade(full_idx)
        signs.append(REVERSE_SIGNS_BY_GRADE[grade])
    return tuple(signs)


# Precomputed motor reverse signs
MOTOR_REVERSE_SIGNS = get_motor_reverse_signs()


# =============================================================================
# Utility Functions
# =============================================================================

def up(x_3d: np.ndarray) -> np.ndarray:
    """
    Project a 3D point to UPGC representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Args:
        x_3d: 3D coordinates as (3,) array

    Returns:
        UPGC point as multivector value array (32,)
    """
    up_func = _stuff['up']
    e1, e2, e3 = _blades['e1'], _blades['e2'], _blades['e3']
    x_mv = x_3d[0] * e1 + x_3d[1] * e2 + x_3d[2] * e3
    result = up_func(x_mv)
    return result.value


def down(X: np.ndarray) -> np.ndarray:
    """
    Project a UPGC point back to 3D coordinates.

    Args:
        X: UPGC point as multivector value array (32,) or (5,) for sparse

    Returns:
        3D coordinates as (3,) array
    """
    down_func = _stuff['down']
    # Create multivector from value array
    mv = _layout.MultiVector(value=X if len(X) == 32 else np.zeros(32))
    if len(X) == 5:
        # Sparse representation - fill in grade 1 components
        mv.value[1:6] = X
    result = down_func(mv)
    return np.array([result.value[1], result.value[2], result.value[3]])
