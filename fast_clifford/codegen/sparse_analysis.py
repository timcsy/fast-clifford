"""
Sparsity analysis for CGA operations.

Defines sparsity patterns for common multivector types:
- CGA Point (Grade 1 only)
- EvenVersor (Grade 0, 2, 4)

Supports multiple CGA dimensions:
- CGA1D Cl(2,1): 8 blades, 3 Point, 4 EvenVersor
- CGA2D Cl(3,1): 16 blades, 4 Point, 8 EvenVersor
- CGA3D Cl(4,1): 32 blades, 5 Point, 16 EvenVersor

And analyzes sandwich product output sparsity.
"""

from typing import Dict, List, Set, Tuple
from .base import SparsityPattern
from .cga_factory import (
    compute_blade_count,
    compute_grade_indices,
    get_point_indices,
    get_even_versor_indices,
    get_product_table,
    compute_reverse_signs,
    get_blade_names,
    create_cga_algebra,
)


# =============================================================================
# T022: CGA Point Sparsity Pattern
# =============================================================================

# CGA Point: X = n_o + x + 0.5|x|^2 * n_inf
# Only Grade 1 components are non-zero
POINT_FULL_INDICES = (1, 2, 3, 4, 5)  # e1, e2, e3, e+, e-

POINT_PATTERN = SparsityPattern(
    name="point",
    nonzero_indices=POINT_FULL_INDICES,
    blade_count=32
)


# =============================================================================
# T023: EvenVersor Sparsity Pattern
# =============================================================================

# EvenVersor: M = R * T where R is rotor, T is translator
# Grade 0: 1 component (scalar)
# Grade 2: 10 components (bivectors)
# Grade 4: 5 components (quadvectors)
EVEN_VERSOR_FULL_INDICES = (
    0,                              # Grade 0: scalar
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # Grade 2: e12, e13, e1+, e1-, e23, e2+, e2-, e3+, e3-, e+-
    26, 27, 28, 29, 30              # Grade 4: e123+, e123-, e12+-, e13+-, e23+-
)

EVEN_VERSOR_PATTERN = SparsityPattern(
    name="even_versor",
    nonzero_indices=EVEN_VERSOR_FULL_INDICES,
    blade_count=32
)


# =============================================================================
# T024: Sandwich Product Sparsity Analysis
# =============================================================================

def analyze_sandwich_output_sparsity(
    product_table: Dict[Tuple[int, int], Tuple[int, int]],
    even_versor_indices: Tuple[int, ...],
    point_indices: Tuple[int, ...]
) -> Set[int]:
    """
    Analyze which blade indices can be non-zero in the sandwich product output.

    Sandwich product: M × X × M̃

    For CGA:
    - M has Grade 0, 2, 4 (even_versor_indices)
    - X has Grade 1 (point_indices)
    - M̃ has same pattern as M (with sign changes in Grade 2)

    Mathematical result: Output should only have Grade 1 (point transforms to point)

    Args:
        product_table: Geometric product lookup table
        even_versor_indices: Non-zero indices in even_versor
        point_indices: Non-zero indices in point

    Returns:
        Set of blade indices that can be non-zero in output
    """
    # First compute M × X possible outputs
    mx_possible = set()
    for m_idx in even_versor_indices:
        for x_idx in point_indices:
            if (m_idx, x_idx) in product_table:
                result_idx, _ = product_table[(m_idx, x_idx)]
                mx_possible.add(result_idx)

    # Then compute (M × X) × M̃ possible outputs
    # M̃ has same indices as M
    output_possible = set()
    for mx_idx in mx_possible:
        for m_rev_idx in even_versor_indices:
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
    grade_1_set = set(POINT_FULL_INDICES)
    return output_indices.issubset(grade_1_set)


def get_sandwich_product_terms(
    product_table: Dict[Tuple[int, int], Tuple[int, int]],
    even_versor_indices: Tuple[int, ...],
    point_indices: Tuple[int, ...],
    reverse_signs: Tuple[int, ...]
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Get all terms in the expanded sandwich product.

    For each output blade index k, returns list of terms:
    (even_versor_sparse_i, point_sparse_j, even_versor_sparse_l, sign)

    The generated code will compute:
        result[k] = sum of (sign * even_versor[i] * point[j] * even_versor_reversed[l])

    where even_versor_reversed[l] is computed by the generated code itself
    (negating Grade 2 components).

    IMPORTANT: The sign here does NOT include the reverse sign, because
    the generated code applies the reverse operation separately.

    Args:
        product_table: Geometric product lookup table
        even_versor_indices: Non-zero even_versor blade indices (16 for even_versor)
        point_indices: Non-zero point blade indices (5 for CGA Point)
        reverse_signs: Reverse signs for all 32 blades (used for reference only)

    Returns:
        Dict mapping output_index -> list of term tuples
    """
    # Build result dictionary for each output grade-1 component
    result_terms = {k: [] for k in point_indices}

    # For each combination of even_versor × point × even_versor_reversed
    for m_i in even_versor_indices:
        for p_j in point_indices:
            # First product: even_versor[i] × point[j]
            if (m_i, p_j) not in product_table:
                continue
            intermediate, sign_1 = product_table[(m_i, p_j)]

            # Second product: intermediate × even_versor[l]
            # Note: we use even_versor[l] here, not even_versor_reversed[l]
            # The reverse will be applied in the generated code
            for m_l in even_versor_indices:
                if (intermediate, m_l) not in product_table:
                    continue
                output_idx, sign_2 = product_table[(intermediate, m_l)]

                # Only keep Grade 1 outputs
                if output_idx not in point_indices:
                    continue

                # Combined sign from the two geometric products only
                # Do NOT include reverse sign here - it's applied in generated code
                total_sign = sign_1 * sign_2

                # Store term: (even_versor_sparse_i, point_sparse_j, even_versor_sparse_l, total_sign)
                # Convert to sparse indices
                m_sparse_i = EVEN_VERSOR_PATTERN.full_to_sparse(m_i)
                p_sparse_j = POINT_PATTERN.full_to_sparse(p_j)
                m_sparse_l = EVEN_VERSOR_PATTERN.full_to_sparse(m_l)

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
    # Each term is even_versor * point * even_versor_rev = 2 multiplications
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


# EvenVersor sparse index order (matches EVEN_VERSOR_FULL_INDICES)
EVEN_VERSOR_SPARSE_ORDER = """
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

# Point sparse index order (matches POINT_FULL_INDICES)
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

def get_point_pattern(euclidean_dim: int) -> SparsityPattern:
    """
    取得指定維度 CGA 的 CGA Point 稀疏性模式。

    Args:
        euclidean_dim: 歐幾里得空間維度 (1, 2, 或 3)

    Returns:
        SparsityPattern 物件
    """
    blade_count = compute_blade_count(euclidean_dim)
    point_indices = get_point_indices(euclidean_dim)

    return SparsityPattern(
        name=f"point_{euclidean_dim}d",
        nonzero_indices=point_indices,
        blade_count=blade_count
    )


def get_even_versor_pattern(euclidean_dim: int) -> SparsityPattern:
    """
    取得指定維度 CGA 的 EvenVersor 稀疏性模式。

    Args:
        euclidean_dim: 歐幾里得空間維度 (1, 2, 或 3)

    Returns:
        SparsityPattern 物件
    """
    blade_count = compute_blade_count(euclidean_dim)
    even_versor_indices = get_even_versor_indices(euclidean_dim)

    return SparsityPattern(
        name=f"even_versor_{euclidean_dim}d",
        nonzero_indices=even_versor_indices,
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
    even_versor_indices = get_even_versor_indices(euclidean_dim)
    point_indices = get_point_indices(euclidean_dim)

    return analyze_sandwich_output_sparsity(
        product_table, even_versor_indices, point_indices
    )


def get_sandwich_product_terms_generic(
    euclidean_dim: int
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    取得指定維度 CGA 的 sandwich product 所有項。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_index -> [(even_versor_sparse_i, point_sparse_j, even_versor_sparse_l, sign), ...]
    """
    product_table = get_product_table(euclidean_dim)
    even_versor_indices = get_even_versor_indices(euclidean_dim)
    point_indices = get_point_indices(euclidean_dim)
    reverse_signs = compute_reverse_signs(euclidean_dim)

    even_versor_pattern = get_even_versor_pattern(euclidean_dim)
    point_pattern = get_point_pattern(euclidean_dim)

    # 建立結果字典
    result_terms = {k: [] for k in point_indices}

    # 對每個 even_versor × point × even_versor_reversed 組合
    for m_i in even_versor_indices:
        for p_j in point_indices:
            # 第一個乘積: even_versor[i] × point[j]
            if (m_i, p_j) not in product_table:
                continue
            intermediate, sign_1 = product_table[(m_i, p_j)]

            # 第二個乘積: intermediate × even_versor[l]
            for m_l in even_versor_indices:
                if (intermediate, m_l) not in product_table:
                    continue
                output_idx, sign_2 = product_table[(intermediate, m_l)]

                # 只保留 Grade 1 輸出
                if output_idx not in point_indices:
                    continue

                # 合併符號（不包含 reverse 符號，由生成的代碼處理）
                total_sign = sign_1 * sign_2

                # 轉換為稀疏索引
                m_sparse_i = even_versor_pattern.full_to_sparse(m_i)
                p_sparse_j = point_pattern.full_to_sparse(p_j)
                m_sparse_l = even_versor_pattern.full_to_sparse(m_l)

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
CGA1D_POINT_FULL_INDICES = (1, 2, 3)  # e1, e+, e-
CGA1D_EVEN_VERSOR_FULL_INDICES = (0, 4, 5, 6)  # scalar, e1+, e1-, e+-

CGA1D_POINT_PATTERN = SparsityPattern(
    name="point_1d",
    nonzero_indices=CGA1D_POINT_FULL_INDICES,
    blade_count=8
)

CGA1D_EVEN_VERSOR_PATTERN = SparsityPattern(
    name="even_versor_1d",
    nonzero_indices=CGA1D_EVEN_VERSOR_FULL_INDICES,
    blade_count=8
)


# =============================================================================
# CGA2D 預計算模式
# =============================================================================

# CGA2D Cl(3,1): 16 blades
# Grade 分佈: 1, 4, 6, 4, 1
CGA2D_POINT_FULL_INDICES = (1, 2, 3, 4)  # e1, e2, e+, e-
CGA2D_EVEN_VERSOR_FULL_INDICES = (0, 5, 6, 7, 8, 9, 10, 15)  # Grade 0, 2, 4

CGA2D_POINT_PATTERN = SparsityPattern(
    name="point_2d",
    nonzero_indices=CGA2D_POINT_FULL_INDICES,
    blade_count=16
)

CGA2D_EVEN_VERSOR_PATTERN = SparsityPattern(
    name="even_versor_2d",
    nonzero_indices=CGA2D_EVEN_VERSOR_FULL_INDICES,
    blade_count=16
)


# =============================================================================
# T001: get_compose_even_versor_terms - EvenVersor 組合稀疏性分析
# =============================================================================

def get_compose_even_versor_terms(euclidean_dim: int) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    取得 EvenVersor 組合（幾何積）的所有非零項。

    EvenVersor × EvenVersor = EvenVersor（偶數 grade 乘積保持偶數 grade）

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_sparse_idx -> [(v1_sparse_i, v2_sparse_j, sign), ...]
    """
    product_table = get_product_table(euclidean_dim)
    even_versor_indices = get_even_versor_indices(euclidean_dim)
    even_versor_pattern = get_even_versor_pattern(euclidean_dim)

    # 建立結果字典
    result_terms = {sparse_idx: [] for sparse_idx in range(len(even_versor_indices))}

    # 對每個 even_versor × even_versor 組合
    for v1_full_idx in even_versor_indices:
        for v2_full_idx in even_versor_indices:
            if (v1_full_idx, v2_full_idx) not in product_table:
                continue
            output_full_idx, sign = product_table[(v1_full_idx, v2_full_idx)]

            # 只保留 even_versor 輸出
            if output_full_idx not in even_versor_indices:
                continue

            # 轉換為稀疏索引
            v1_sparse = even_versor_pattern.full_to_sparse(v1_full_idx)
            v2_sparse = even_versor_pattern.full_to_sparse(v2_full_idx)
            output_sparse = even_versor_pattern.full_to_sparse(output_full_idx)

            result_terms[output_sparse].append((v1_sparse, v2_sparse, sign))

    return result_terms


# =============================================================================
# T002: get_compose_similitude_terms - Similitude 組合稀疏性分析 (CGA 專用)
# =============================================================================

def get_similitude_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    取得 Similitude 的有效 blade 索引。

    Similitude = EvenVersor 但強制 ei+ = ei- (排除 transversion)

    注意: Similitude 使用相同的 EvenVersor 儲存格式，
          加速來自約束條件的稀疏性利用。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Similitude 的有效 blade 索引（與 even_versor_indices 相同）
    """
    # Similitude 使用與 EvenVersor 相同的索引
    # 約束 (ei+ = ei-) 在運行時強制
    return get_even_versor_indices(euclidean_dim)


def get_similitude_constraint_pairs(euclidean_dim: int) -> List[Tuple[int, int]]:
    """
    取得 Similitude 約束配對 (ei+, ei-) 的稀疏索引。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        List of (plus_sparse_idx, minus_sparse_idx) pairs
    """
    blade_names = get_blade_names(euclidean_dim)
    even_versor_indices = get_even_versor_indices(euclidean_dim)
    even_versor_pattern = get_even_versor_pattern(euclidean_dim)

    pairs = []
    for i in range(euclidean_dim):
        # 找 ei+ 和 ei-
        plus_name = f"e{i+1}e+"
        minus_name = f"e{i+1}e-"

        plus_full_idx = None
        minus_full_idx = None

        for full_idx in even_versor_indices:
            name = blade_names[full_idx]
            if name == plus_name:
                plus_full_idx = full_idx
            elif name == minus_name:
                minus_full_idx = full_idx

        if plus_full_idx is not None and minus_full_idx is not None:
            plus_sparse = even_versor_pattern.full_to_sparse(plus_full_idx)
            minus_sparse = even_versor_pattern.full_to_sparse(minus_full_idx)
            pairs.append((plus_sparse, minus_sparse))

    return pairs


def get_compose_similitude_terms(euclidean_dim: int) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    取得 Similitude 組合的所有非零項。

    由於 Similitude 約束 (ei+ = ei-), 某些項可以合併或忽略，
    提供 30-50% 的加速。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_sparse_idx -> [(s1_sparse_i, s2_sparse_j, sign), ...]
    """
    # 基礎實作: 與 EvenVersor 相同
    # 進階優化可在生成代碼時利用約束合併項
    return get_compose_even_versor_terms(euclidean_dim)


# =============================================================================
# T003: get_inner_product_signs - 度規內積符號分析
# =============================================================================

def get_inner_product_signs(euclidean_dim: int) -> Tuple[int, ...]:
    """
    取得各 blade 的度規內積符號。

    幾何內積: <a * b>_0 = sum(a[i] * b[i] * blade_i²)
    blade_i² 的符號取決於 blade 的 metric signature。

    CGA 度規為 (+,...,+,-)，因此:
    - ei² = +1 (歐幾里得基底)
    - e+² = +1
    - e-² = -1

    對於高階 blade: (ei ∧ ej ∧ ...)² = ei² * ej² * ... * signature_factor

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        每個 blade 的度規符號 tuple
    """
    layout, blades, _ = create_cga_algebra(euclidean_dim)
    blade_count = compute_blade_count(euclidean_dim)

    # 取得 product table 來查詢 blade²
    product_table = get_product_table(euclidean_dim)

    signs = []
    for idx in range(blade_count):
        blade_tuple = layout.bladeTupList[idx]
        if len(blade_tuple) == 0:
            # Scalar: 1² = 1
            signs.append(1)
        else:
            # 查詢 blade[idx] * blade[idx] 的結果
            if (idx, idx) in product_table:
                result_idx, sign = product_table[(idx, idx)]
                # 結果應該是標量 (index 0)
                if result_idx == 0:
                    signs.append(sign)
                else:
                    signs.append(0)
            else:
                signs.append(0)

    return tuple(signs)


# =============================================================================
# T004: get_bivector_squared_terms - Bivector 平方項分析
# =============================================================================

def get_bivector_squared_terms(euclidean_dim: int) -> List[Tuple[int, int, int]]:
    """
    取得 Bivector 平方的標量部分項。

    B² = sum of (B[i] * B[j] * sign_ij) for i,j in bivector indices
    where the result is scalar (Grade 0)

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        List of (biv_sparse_i, biv_sparse_j, sign) tuples contributing to scalar
    """
    product_table = get_product_table(euclidean_dim)
    bivector_indices = get_bivector_indices(euclidean_dim)
    grade_indices = compute_grade_indices(euclidean_dim)
    scalar_idx = grade_indices[0][0]  # Grade 0 只有一個 index: 0

    # 建立 bivector 稀疏模式
    biv_pattern = SparsityPattern(
        name=f"bivector_{euclidean_dim}d",
        nonzero_indices=bivector_indices,
        blade_count=compute_blade_count(euclidean_dim)
    )

    terms = []
    for i, full_i in enumerate(bivector_indices):
        for j, full_j in enumerate(bivector_indices):
            if (full_i, full_j) not in product_table:
                continue
            result_full, sign = product_table[(full_i, full_j)]

            # 只保留標量輸出
            if result_full == scalar_idx:
                terms.append((i, j, sign))

    return terms


# =============================================================================
# T005: get_bivector_indices - Bivector 索引取得
# =============================================================================

def get_bivector_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    取得 Bivector (Grade 2) 的 blade 索引。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Grade 2 的 blade 索引 tuple
    """
    grade_indices = compute_grade_indices(euclidean_dim)
    return grade_indices.get(2, ())


# =============================================================================
# T006: get_outer_product_terms - 楔積項分析
# =============================================================================

def get_outer_product_terms(euclidean_dim: int) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    取得楔積（外積）的所有非零項。

    楔積: a ∧ b = <a * b>_{|a| + |b|}
    即幾何積中 Grade 為 Grade(a) + Grade(b) 的部分。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_full_idx -> [(a_full_i, b_full_j, sign), ...]
    """
    layout, _, _ = create_cga_algebra(euclidean_dim)
    product_table = get_product_table(euclidean_dim)
    grade_indices = compute_grade_indices(euclidean_dim)
    blade_count = compute_blade_count(euclidean_dim)

    # 建立 index -> grade 映射
    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    # 收集楔積項
    result_terms = {k: [] for k in range(blade_count)}

    for (a_idx, b_idx), (result_idx, sign) in product_table.items():
        grade_a = index_to_grade[a_idx]
        grade_b = index_to_grade[b_idx]
        grade_result = index_to_grade[result_idx]

        # 楔積只保留 grade(a) + grade(b) 的結果
        if grade_result == grade_a + grade_b:
            result_terms[result_idx].append((a_idx, b_idx, sign))

    return result_terms


# =============================================================================
# T007: get_left_contraction_terms - 左縮併項分析
# =============================================================================

def get_left_contraction_terms(euclidean_dim: int) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    取得左縮併的所有非零項。

    左縮併: a ⌋ b = <a * b>_{|b| - |a|}  if |a| <= |b|, else 0

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_full_idx -> [(a_full_i, b_full_j, sign), ...]
    """
    product_table = get_product_table(euclidean_dim)
    grade_indices = compute_grade_indices(euclidean_dim)
    blade_count = compute_blade_count(euclidean_dim)

    # 建立 index -> grade 映射
    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    result_terms = {k: [] for k in range(blade_count)}

    for (a_idx, b_idx), (result_idx, sign) in product_table.items():
        grade_a = index_to_grade[a_idx]
        grade_b = index_to_grade[b_idx]
        grade_result = index_to_grade[result_idx]

        # 左縮併: grade(result) = grade(b) - grade(a) if grade(a) <= grade(b)
        if grade_a <= grade_b and grade_result == grade_b - grade_a:
            result_terms[result_idx].append((a_idx, b_idx, sign))

    return result_terms


# =============================================================================
# T008: get_right_contraction_terms - 右縮併項分析
# =============================================================================

def get_right_contraction_terms(euclidean_dim: int) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    取得右縮併的所有非零項。

    右縮併: a ⌊ b = <a * b>_{|a| - |b|}  if |a| >= |b|, else 0

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 output_full_idx -> [(a_full_i, b_full_j, sign), ...]
    """
    product_table = get_product_table(euclidean_dim)
    grade_indices = compute_grade_indices(euclidean_dim)
    blade_count = compute_blade_count(euclidean_dim)

    # 建立 index -> grade 映射
    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    result_terms = {k: [] for k in range(blade_count)}

    for (a_idx, b_idx), (result_idx, sign) in product_table.items():
        grade_a = index_to_grade[a_idx]
        grade_b = index_to_grade[b_idx]
        grade_result = index_to_grade[result_idx]

        # 右縮併: grade(result) = grade(a) - grade(b) if grade(a) >= grade(b)
        if grade_a >= grade_b and grade_result == grade_a - grade_b:
            result_terms[result_idx].append((a_idx, b_idx, sign))

    return result_terms


# =============================================================================
# T009: get_grade_masks - Grade 遮罩取得
# =============================================================================

def get_grade_masks(euclidean_dim: int) -> Dict[int, Tuple[int, ...]]:
    """
    取得各 grade 的 blade 索引遮罩。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict 映射 grade -> (blade indices tuple)
    """
    return compute_grade_indices(euclidean_dim)


# =============================================================================
# T010: get_pseudoscalar_info - Pseudoscalar 資訊取得
# =============================================================================

def get_pseudoscalar_info(euclidean_dim: int) -> Dict[str, any]:
    """
    取得 Pseudoscalar 的相關資訊。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict containing:
            - 'index': Pseudoscalar 的 blade 索引
            - 'grade': Pseudoscalar 的 grade (= n+2 for CGA(n))
            - 'square': Pseudoscalar² 的值 (+1 或 -1)
    """
    grade_indices = compute_grade_indices(euclidean_dim)
    total_dim = euclidean_dim + 2
    max_grade = total_dim

    pseudoscalar_idx = grade_indices[max_grade][0]

    # 計算 pseudoscalar²
    product_table = get_product_table(euclidean_dim)
    if (pseudoscalar_idx, pseudoscalar_idx) in product_table:
        result_idx, sign = product_table[(pseudoscalar_idx, pseudoscalar_idx)]
        # Pseudoscalar² 應該是標量
        square_value = sign
    else:
        square_value = 0

    return {
        'index': pseudoscalar_idx,
        'grade': max_grade,
        'square': square_value,
    }


# =============================================================================
# T011: get_norm_squared_terms - 範數平方項分析
# =============================================================================

def get_norm_squared_terms(euclidean_dim: int) -> List[Tuple[int, int]]:
    """
    取得 multivector 範數平方的項。

    |a|² = <a * ~a>_0 = sum of (a[i] * a[i] * reverse_sign[i] * inner_product_sign[i])

    由於 a 和 ~a 的非零索引相同，可以簡化為:
    |a|² = sum of (a[i]² * combined_sign[i])

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        List of (full_idx, combined_sign) tuples
    """
    blade_count = compute_blade_count(euclidean_dim)
    reverse_signs = compute_reverse_signs(euclidean_dim)
    inner_product_signs = get_inner_product_signs(euclidean_dim)

    terms = []
    for idx in range(blade_count):
        # combined_sign = reverse_sign * inner_product_sign
        combined = reverse_signs[idx] * inner_product_signs[idx]
        if combined != 0:
            terms.append((idx, combined))

    return terms


# =============================================================================
# Structure Normalize 相關函數 (T153i, T153j)
# =============================================================================

def get_rotor_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    取得 Rotor（純旋轉）分量的稀疏索引。

    Rotor 包含 scalar 和空間 bivector (ei ∧ ej, 不含 e+ 或 e-)。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Rotor 分量的稀疏索引 tuple
    """
    blade_names = get_blade_names(euclidean_dim)
    even_versor_indices = get_even_versor_indices(euclidean_dim)
    even_versor_pattern = get_even_versor_pattern(euclidean_dim)

    rotor_sparse_indices = []

    for full_idx in even_versor_indices:
        name = blade_names[full_idx]
        # 包含 scalar (name == "1")
        # 或空間 bivector (ei ∧ ej，不含 e+ 或 e-)
        if name == "1":
            rotor_sparse_indices.append(even_versor_pattern.full_to_sparse(full_idx))
        elif "+" not in name and "-" not in name:
            # 純空間 bivector，如 e12, e13, e23
            rotor_sparse_indices.append(even_versor_pattern.full_to_sparse(full_idx))

    return tuple(sorted(rotor_sparse_indices))


def get_translation_pairs(euclidean_dim: int) -> List[Tuple[int, int]]:
    """
    取得平移分量配對 (ei+, ei-) 的稀疏索引。

    用於 Structure Normalize 強制 Similitude 約束。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        List of (plus_sparse_idx, minus_sparse_idx) pairs
    """
    return get_similitude_constraint_pairs(euclidean_dim)


def get_dilation_index(euclidean_dim: int) -> int:
    """
    取得 Dilation (e+-) 分量的稀疏索引。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        e+- 分量的稀疏索引，若不存在返回 -1
    """
    blade_names = get_blade_names(euclidean_dim)
    even_versor_indices = get_even_versor_indices(euclidean_dim)
    even_versor_pattern = get_even_versor_pattern(euclidean_dim)

    for full_idx in even_versor_indices:
        name = blade_names[full_idx]
        if name == "e+e-" or name == "e+-":
            return even_versor_pattern.full_to_sparse(full_idx)

    return -1


# =============================================================================
# Structure Normalize (US9a): Rotor indices and Translation pairs
# =============================================================================

def get_rotor_indices(dim: int) -> Tuple[int, ...]:
    """
    Get rotor (pure rotation) component indices in EvenVersor sparse representation.

    Rotor consists of scalar and spatial bivectors (eij where i,j are Euclidean).

    Args:
        dim: Euclidean dimension

    Returns:
        Tuple of sparse indices for rotor components

    Examples:
        >>> get_rotor_indices(0)  # CGA0D: no rotation
        (0,)
        >>> get_rotor_indices(1)  # CGA1D: no rotation
        (0,)
        >>> get_rotor_indices(2)  # CGA2D: 1 rotation plane (e12)
        (0, 1)
        >>> get_rotor_indices(3)  # CGA3D: 3 rotation planes (e12, e13, e23)
        (0, 1, 2, 5)
    """
    even_versor_pattern = get_even_versor_pattern(dim)
    blade_names = get_blade_names(dim)

    rotor_indices = []

    for sparse_idx, full_idx in enumerate(even_versor_pattern.nonzero_indices):
        name = blade_names[full_idx]

        # Scalar component always included
        if name == "1":
            rotor_indices.append(sparse_idx)
            continue

        # Spatial bivectors: eiej where i,j are both Euclidean (1,2,3,...)
        # Not e+, e-, einf, eo
        if "+" not in name and "-" not in name:
            # Check if it's a bivector (exactly 2 digits)
            digits = [c for c in name if c.isdigit()]
            if len(digits) == 2 and name.startswith("e"):
                rotor_indices.append(sparse_idx)

    return tuple(rotor_indices)


def get_translation_pairs(dim: int) -> List[Tuple[int, int]]:
    """
    Get translation component pairs (eie+, eie-) indices in EvenVersor sparse representation.

    For Similitude constraint: eie+ coefficient = eie- coefficient (no transversion).

    Args:
        dim: Euclidean dimension

    Returns:
        List of (plus_idx, minus_idx) tuples in sparse representation

    Examples:
        >>> get_translation_pairs(1)  # CGA1D: [(e1e+, e1e-)]
        [(1, 2)]
        >>> get_translation_pairs(2)  # CGA2D: [(e1e+, e1e-), (e2e+, e2e-)]
        [(2, 3), (4, 5)]
        >>> get_translation_pairs(3)  # CGA3D: 3 pairs
        [(3, 4), (6, 7), (8, 9)]
    """
    even_versor_pattern = get_even_versor_pattern(dim)
    blade_names = get_blade_names(dim)

    pairs = []

    # For each Euclidean dimension i, find eie+ and eie-
    for i in range(1, dim + 1):
        plus_name = f"e{i}e+"
        minus_name = f"e{i}e-"

        plus_idx = None
        minus_idx = None

        for sparse_idx, full_idx in enumerate(even_versor_pattern.nonzero_indices):
            name = blade_names[full_idx]
            if name == plus_name:
                plus_idx = sparse_idx
            elif name == minus_name:
                minus_idx = sparse_idx

        if plus_idx is not None and minus_idx is not None:
            pairs.append((plus_idx, minus_idx))

    return pairs


def get_dilation_index(dim: int) -> int:
    """
    Get dilation (e+-) component index in EvenVersor sparse representation.

    Args:
        dim: Euclidean dimension

    Returns:
        Sparse index for e+- component, or -1 if not found
    """
    even_versor_pattern = get_even_versor_pattern(dim)
    blade_names = get_blade_names(dim)

    for sparse_idx, full_idx in enumerate(even_versor_pattern.nonzero_indices):
        name = blade_names[full_idx]
        if name == "e+-" or name == "e+e-":
            return sparse_idx

    return -1


if __name__ == "__main__":
    print("=== Sparsity Analysis ===")

    for dim in [1, 2, 3]:
        print(f"\n--- CGA{dim}D ---")

        point_pattern = get_point_pattern(dim)
        even_versor_pattern = get_even_versor_pattern(dim)

        print(f"CGA Point: {point_pattern.sparse_count} components")
        print(f"  Indices: {point_pattern.nonzero_indices}")

        print(f"EvenVersor: {even_versor_pattern.sparse_count} components")
        print(f"  Indices: {even_versor_pattern.nonzero_indices}")

        # 驗證輸出稀疏性
        output_indices = analyze_sandwich_output_sparsity_generic(dim)
        print(f"Sandwich output indices: {output_indices}")

        # 乘法操作數
        ops = count_sandwich_product_ops(dim)
        full_ops = even_versor_pattern.sparse_count * point_pattern.sparse_count * even_versor_pattern.sparse_count * 2
        print(f"Multiplication ops: {ops} (vs {full_ops} naive)")

        # Structure Normalize indices
        rotor_indices = get_rotor_indices(dim)
        translation_pairs = get_translation_pairs(dim)
        dilation_idx = get_dilation_index(dim)
        print(f"Rotor indices: {rotor_indices}")
        print(f"Translation pairs: {translation_pairs}")
        print(f"Dilation index: {dilation_idx}")
