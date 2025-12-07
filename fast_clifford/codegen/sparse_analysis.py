"""
Sparsity analysis for CGA operations.

Defines sparsity patterns for common multivector types:
- UPGC Point (Grade 1 only)
- Motor (Grade 0, 2, 4)

Supports multiple CGA dimensions:
- CGA1D Cl(2,1): 8 blades, 3 Point, 4 Motor
- CGA2D Cl(3,1): 16 blades, 4 Point, 8 Motor
- CGA3D Cl(4,1): 32 blades, 5 Point, 16 Motor

And analyzes sandwich product output sparsity.
"""

from typing import Dict, List, Set, Tuple
from .base import SparsityPattern
from .cga_factory import (
    compute_blade_count,
    compute_grade_indices,
    get_upgc_point_indices,
    get_motor_indices,
    get_product_table,
    compute_reverse_signs,
)


# =============================================================================
# T022: UPGC Point Sparsity Pattern
# =============================================================================

# UPGC Point: X = n_o + x + 0.5|x|^2 * n_inf
# Only Grade 1 components are non-zero
UPGC_POINT_FULL_INDICES = (1, 2, 3, 4, 5)  # e1, e2, e3, e+, e-

UPGC_POINT_PATTERN = SparsityPattern(
    name="upgc_point",
    nonzero_indices=UPGC_POINT_FULL_INDICES,
    blade_count=32
)


# =============================================================================
# T023: Motor Sparsity Pattern
# =============================================================================

# Motor: M = R * T where R is rotor, T is translator
# Grade 0: 1 component (scalar)
# Grade 2: 10 components (bivectors)
# Grade 4: 5 components (quadvectors)
MOTOR_FULL_INDICES = (
    0,                              # Grade 0: scalar
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # Grade 2: e12, e13, e1+, e1-, e23, e2+, e2-, e3+, e3-, e+-
    26, 27, 28, 29, 30              # Grade 4: e123+, e123-, e12+-, e13+-, e23+-
)

MOTOR_PATTERN = SparsityPattern(
    name="motor",
    nonzero_indices=MOTOR_FULL_INDICES,
    blade_count=32
)


# =============================================================================
# T024: Sandwich Product Sparsity Analysis
# =============================================================================

def analyze_sandwich_output_sparsity(
    product_table: Dict[Tuple[int, int], Tuple[int, int]],
    motor_indices: Tuple[int, ...],
    point_indices: Tuple[int, ...]
) -> Set[int]:
    """
    Analyze which blade indices can be non-zero in the sandwich product output.

    Sandwich product: M × X × M̃

    For CGA:
    - M has Grade 0, 2, 4 (motor_indices)
    - X has Grade 1 (point_indices)
    - M̃ has same pattern as M (with sign changes in Grade 2)

    Mathematical result: Output should only have Grade 1 (point transforms to point)

    Args:
        product_table: Geometric product lookup table
        motor_indices: Non-zero indices in motor
        point_indices: Non-zero indices in point

    Returns:
        Set of blade indices that can be non-zero in output
    """
    # First compute M × X possible outputs
    mx_possible = set()
    for m_idx in motor_indices:
        for x_idx in point_indices:
            if (m_idx, x_idx) in product_table:
                result_idx, _ = product_table[(m_idx, x_idx)]
                mx_possible.add(result_idx)

    # Then compute (M × X) × M̃ possible outputs
    # M̃ has same indices as M
    output_possible = set()
    for mx_idx in mx_possible:
        for m_rev_idx in motor_indices:
            if (mx_idx, m_rev_idx) in product_table:
                result_idx, _ = product_table[(mx_idx, m_rev_idx)]
                output_possible.add(result_idx)

    return output_possible


def verify_grade1_output(output_indices: Set[int]) -> bool:
    """
    Verify that output indices are all Grade 1.

    Args:
        output_indices: Set of possible output blade indices

    Returns:
        True if all indices are Grade 1
    """
    grade_1_set = set(UPGC_POINT_FULL_INDICES)
    return output_indices.issubset(grade_1_set)


def get_sandwich_product_terms(
    product_table: Dict[Tuple[int, int], Tuple[int, int]],
    motor_indices: Tuple[int, ...],
    point_indices: Tuple[int, ...],
    reverse_signs: Tuple[int, ...]
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Get all terms in the expanded sandwich product.

    For each output blade index k, returns list of terms:
    (motor_sparse_i, point_sparse_j, motor_sparse_l, sign)

    The generated code will compute:
        result[k] = sum of (sign * motor[i] * point[j] * motor_reversed[l])

    where motor_reversed[l] is computed by the generated code itself
    (negating Grade 2 components).

    IMPORTANT: The sign here does NOT include the reverse sign, because
    the generated code applies the reverse operation separately.

    Args:
        product_table: Geometric product lookup table
        motor_indices: Non-zero motor blade indices (16 for motor)
        point_indices: Non-zero point blade indices (5 for UPGC)
        reverse_signs: Reverse signs for all 32 blades (used for reference only)

    Returns:
        Dict mapping output_index -> list of term tuples
    """
    # Build result dictionary for each output grade-1 component
    result_terms = {k: [] for k in point_indices}

    # For each combination of motor × point × motor_reversed
    for m_i in motor_indices:
        for p_j in point_indices:
            # First product: motor[i] × point[j]
            if (m_i, p_j) not in product_table:
                continue
            intermediate, sign_1 = product_table[(m_i, p_j)]

            # Second product: intermediate × motor[l]
            # Note: we use motor[l] here, not motor_reversed[l]
            # The reverse will be applied in the generated code
            for m_l in motor_indices:
                if (intermediate, m_l) not in product_table:
                    continue
                output_idx, sign_2 = product_table[(intermediate, m_l)]

                # Only keep Grade 1 outputs
                if output_idx not in point_indices:
                    continue

                # Combined sign from the two geometric products only
                # Do NOT include reverse sign here - it's applied in generated code
                total_sign = sign_1 * sign_2

                # Store term: (motor_sparse_i, point_sparse_j, motor_sparse_l, total_sign)
                # Convert to sparse indices
                m_sparse_i = MOTOR_PATTERN.full_to_sparse(m_i)
                p_sparse_j = UPGC_POINT_PATTERN.full_to_sparse(p_j)
                m_sparse_l = MOTOR_PATTERN.full_to_sparse(m_l)

                result_terms[output_idx].append((
                    m_sparse_i, p_sparse_j, m_sparse_l, total_sign
                ))

    return result_terms


def count_multiplication_ops(terms: Dict[int, List[Tuple]]) -> int:
    """
    Count total multiplication operations in the sandwich product.

    Each term requires 2 multiplications (m[i] * p[j] * m[l]).

    Args:
        terms: Output from get_sandwich_product_terms

    Returns:
        Total number of multiplications
    """
    total_terms = sum(len(t) for t in terms.values())
    # Each term is motor * point * motor_rev = 2 multiplications
    return total_terms * 2


# =============================================================================
# Utility Functions
# =============================================================================

def get_sparse_to_full_mapping(pattern: SparsityPattern) -> Dict[int, int]:
    """Get mapping from sparse index to full index."""
    return {s: f for s, f in enumerate(pattern.nonzero_indices)}


def get_full_to_sparse_mapping(pattern: SparsityPattern) -> Dict[int, int]:
    """Get mapping from full index to sparse index."""
    return {f: s for s, f in enumerate(pattern.nonzero_indices)}


# Motor sparse index order (matches MOTOR_FULL_INDICES)
MOTOR_SPARSE_ORDER = """
0: scalar (index 0)
1: e12 (index 6)
2: e13 (index 7)
3: e1+ (index 8)
4: e1- (index 9)
5: e23 (index 10)
6: e2+ (index 11)
7: e2- (index 12)
8: e3+ (index 13)
9: e3- (index 14)
10: e+- (index 15)
11: e123+ (index 26)
12: e123- (index 27)
13: e12+- (index 28)
14: e13+- (index 29)
15: e23+- (index 30)
"""

# Point sparse index order (matches UPGC_POINT_FULL_INDICES)
POINT_SPARSE_ORDER = """
0: e1 (index 1)
1: e2 (index 2)
2: e3 (index 3)
3: e+ (index 4)
4: e- (index 5)
"""


# =============================================================================
# 通用化稀疏分析工廠函數
# =============================================================================

def get_upgc_point_pattern(euclidean_dim: int) -> SparsityPattern:
    """
    取得指定維度 CGA 的 UPGC Point 稀疏性模式。

    Args:
        euclidean_dim: 歐幾里得空間維度 (1, 2, 或 3)

    Returns:
        SparsityPattern 物件
    """
    blade_count = compute_blade_count(euclidean_dim)
    point_indices = get_upgc_point_indices(euclidean_dim)

    return SparsityPattern(
        name=f"upgc_point_{euclidean_dim}d",
        nonzero_indices=point_indices,
        blade_count=blade_count
    )


def get_motor_pattern(euclidean_dim: int) -> SparsityPattern:
    """
    取得指定維度 CGA 的 Motor 稀疏性模式。

    Args:
        euclidean_dim: 歐幾里得空間維度 (1, 2, 或 3)

    Returns:
        SparsityPattern 物件
    """
    blade_count = compute_blade_count(euclidean_dim)
    motor_indices = get_motor_indices(euclidean_dim)

    return SparsityPattern(
        name=f"motor_{euclidean_dim}d",
        nonzero_indices=motor_indices,
        blade_count=blade_count
    )


def analyze_sandwich_output_sparsity_generic(
    euclidean_dim: int
) -> Set[int]:
    """
    分析指定維度 CGA 的 sandwich product 輸出稀疏性。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        輸出可能非零的 blade 索引集合
    """
    product_table = get_product_table(euclidean_dim)
    motor_indices = get_motor_indices(euclidean_dim)
    point_indices = get_upgc_point_indices(euclidean_dim)

    return analyze_sandwich_output_sparsity(
        product_table, motor_indices, point_indices
    )


def get_sandwich_product_terms_generic(
    euclidean_dim: int
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    取得指定維度 CGA 的 sandwich product 所有項。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_index -> [(motor_sparse_i, point_sparse_j, motor_sparse_l, sign), ...]
    """
    product_table = get_product_table(euclidean_dim)
    motor_indices = get_motor_indices(euclidean_dim)
    point_indices = get_upgc_point_indices(euclidean_dim)
    reverse_signs = compute_reverse_signs(euclidean_dim)

    motor_pattern = get_motor_pattern(euclidean_dim)
    point_pattern = get_upgc_point_pattern(euclidean_dim)

    # 建立結果字典
    result_terms = {k: [] for k in point_indices}

    # 對每個 motor × point × motor_reversed 組合
    for m_i in motor_indices:
        for p_j in point_indices:
            # 第一個乘積: motor[i] × point[j]
            if (m_i, p_j) not in product_table:
                continue
            intermediate, sign_1 = product_table[(m_i, p_j)]

            # 第二個乘積: intermediate × motor[l]
            for m_l in motor_indices:
                if (intermediate, m_l) not in product_table:
                    continue
                output_idx, sign_2 = product_table[(intermediate, m_l)]

                # 只保留 Grade 1 輸出
                if output_idx not in point_indices:
                    continue

                # 合併符號（不包含 reverse 符號，由生成的代碼處理）
                total_sign = sign_1 * sign_2

                # 轉換為稀疏索引
                m_sparse_i = motor_pattern.full_to_sparse(m_i)
                p_sparse_j = point_pattern.full_to_sparse(p_j)
                m_sparse_l = motor_pattern.full_to_sparse(m_l)

                result_terms[output_idx].append((
                    m_sparse_i, p_sparse_j, m_sparse_l, total_sign
                ))

    return result_terms


def count_sandwich_product_ops(euclidean_dim: int) -> int:
    """
    計算指定維度 CGA sandwich product 的乘法操作數。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        總乘法操作數
    """
    terms = get_sandwich_product_terms_generic(euclidean_dim)
    return count_multiplication_ops(terms)


# =============================================================================
# CGA1D 預計算模式
# =============================================================================

# CGA1D Cl(2,1): 8 blades
# Grade 分佈: 1, 3, 3, 1
CGA1D_UPGC_POINT_FULL_INDICES = (1, 2, 3)  # e1, e+, e-
CGA1D_MOTOR_FULL_INDICES = (0, 4, 5, 6)  # scalar, e1+, e1-, e+-

CGA1D_UPGC_POINT_PATTERN = SparsityPattern(
    name="upgc_point_1d",
    nonzero_indices=CGA1D_UPGC_POINT_FULL_INDICES,
    blade_count=8
)

CGA1D_MOTOR_PATTERN = SparsityPattern(
    name="motor_1d",
    nonzero_indices=CGA1D_MOTOR_FULL_INDICES,
    blade_count=8
)


# =============================================================================
# CGA2D 預計算模式
# =============================================================================

# CGA2D Cl(3,1): 16 blades
# Grade 分佈: 1, 4, 6, 4, 1
CGA2D_UPGC_POINT_FULL_INDICES = (1, 2, 3, 4)  # e1, e2, e+, e-
CGA2D_MOTOR_FULL_INDICES = (0, 5, 6, 7, 8, 9, 10, 15)  # Grade 0, 2, 4

CGA2D_UPGC_POINT_PATTERN = SparsityPattern(
    name="upgc_point_2d",
    nonzero_indices=CGA2D_UPGC_POINT_FULL_INDICES,
    blade_count=16
)

CGA2D_MOTOR_PATTERN = SparsityPattern(
    name="motor_2d",
    nonzero_indices=CGA2D_MOTOR_FULL_INDICES,
    blade_count=16
)


if __name__ == "__main__":
    print("=== Sparsity Analysis ===")

    for dim in [1, 2, 3]:
        print(f"\n--- CGA{dim}D ---")

        point_pattern = get_upgc_point_pattern(dim)
        motor_pattern = get_motor_pattern(dim)

        print(f"UPGC Point: {point_pattern.sparse_count} components")
        print(f"  Indices: {point_pattern.nonzero_indices}")

        print(f"Motor: {motor_pattern.sparse_count} components")
        print(f"  Indices: {motor_pattern.nonzero_indices}")

        # 驗證輸出稀疏性
        output_indices = analyze_sandwich_output_sparsity_generic(dim)
        print(f"Sandwich output indices: {output_indices}")

        # 乘法操作數
        ops = count_sandwich_product_ops(dim)
        full_ops = motor_pattern.sparse_count * point_pattern.sparse_count * motor_pattern.sparse_count * 2
        print(f"Multiplication ops: {ops} (vs {full_ops} naive)")
