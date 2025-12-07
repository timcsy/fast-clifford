"""
CGA Factory - 通用化 CGA 代數建立工廠

支援任意歐幾里得維度的 CGA 代數:
- CGA1D: Cl(2,1) - 1D 歐幾里得空間
- CGA2D: Cl(3,1) - 2D 歐幾里得空間
- CGA3D: Cl(4,1) - 3D 歐幾里得空間
- CGA4D: Cl(5,1) - 4D 歐幾里得空間
- CGA5D: Cl(6,1) - 5D 歐幾里得空間

所有 CGA(n) 使用 Cl(n+1,1) 代數，簽名為 (+,...,+,-):
- n 個正簽名的歐幾里得基底 (e1, e2, ..., en)
- 1 個正簽名的額外維度 (e+)
- 1 個負簽名的額外維度 (e-)
"""

from typing import Dict, List, Tuple
import numpy as np
from clifford import Cl, conformalize


def create_cga_algebra(euclidean_dim: int):
    """
    建立 CGA Cl(n+1,1) 代數。

    Args:
        euclidean_dim: 歐幾里得空間維度 (1, 2, 3, 4, 或 5)

    Returns:
        layout: CGA 代數布局物件
        blades: Blade 字典
        stuff: CGA 特殊物件 (eo, einf, up, down)

    Raises:
        ValueError: 若 euclidean_dim 不在 [1, 5] 範圍內
    """
    if euclidean_dim < 1 or euclidean_dim > 5:
        raise ValueError(f"euclidean_dim 必須在 [1, 5] 範圍內，收到: {euclidean_dim}")

    # 建立基底歐幾里得代數 Cl(n)
    G_n, _ = Cl(euclidean_dim)

    # Conformalize 得到 CGA Cl(n+1,1)
    layout, blades, stuff = conformalize(G_n)

    return layout, blades, stuff


def compute_blade_count(euclidean_dim: int) -> int:
    """
    計算 CGA 代數的 blade 總數。

    CGA(n) = Cl(n+1,1) 有 2^(n+2) 個 blade。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Blade 總數
    """
    total_dim = euclidean_dim + 2  # n+1 正簽名 + 1 負簽名
    return 2 ** total_dim


def compute_grade_indices(euclidean_dim: int) -> Dict[int, Tuple[int, ...]]:
    """
    計算各 grade 的 blade 索引。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict[grade, tuple of blade indices]
    """
    layout, _, _ = create_cga_algebra(euclidean_dim)
    total_dim = euclidean_dim + 2
    blade_count = 2 ** total_dim

    # 使用 layout 的 grade 資訊
    grade_indices = {g: [] for g in range(total_dim + 1)}

    for idx in range(blade_count):
        # 透過 blade tuple 計算 grade
        blade_tuple = layout.bladeTupList[idx]
        grade = len(blade_tuple)
        grade_indices[grade].append(idx)

    return {g: tuple(indices) for g, indices in grade_indices.items()}


def compute_reverse_sign(grade: int) -> int:
    """
    計算給定 grade 的 reverse 符號。

    公式: (-1)^(k*(k-1)/2)，其中 k 是 grade

    Args:
        grade: Blade 的 grade

    Returns:
        +1 或 -1
    """
    exponent = grade * (grade - 1) // 2
    return (-1) ** exponent


def compute_reverse_signs(euclidean_dim: int) -> Tuple[int, ...]:
    """
    計算所有 blade 的 reverse 符號。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        每個 blade 的 reverse 符號 tuple
    """
    grade_indices = compute_grade_indices(euclidean_dim)
    blade_count = compute_blade_count(euclidean_dim)

    # 建立 index -> grade 映射
    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    # 計算每個 blade 的 reverse 符號
    signs = []
    for idx in range(blade_count):
        grade = index_to_grade[idx]
        signs.append(compute_reverse_sign(grade))

    return tuple(signs)


def get_product_table(euclidean_dim: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    取得幾何積乘法表。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Dict[(left_idx, right_idx), (result_idx, sign)]
    """
    layout, _, _ = create_cga_algebra(euclidean_dim)
    blade_count = compute_blade_count(euclidean_dim)

    # 從 layout 提取 GMT (Geometric Multiplication Table)
    gmt_dense = np.asarray(layout.gmt.todense())

    table = {}
    for i in range(blade_count):
        for j in range(blade_count):
            # GMT 索引為 [left, result, right]
            result_vec = gmt_dense[i, :, j]
            nonzero_indices = np.where(result_vec != 0)[0]

            for k in nonzero_indices:
                coeff = result_vec[k]
                sign = int(np.sign(coeff))
                if sign != 0:
                    table[(i, j)] = (k, sign)

    return table


def get_upgc_point_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    取得 UPGC Point 的非零 blade 索引。

    UPGC Point 只有 Grade 1 分量。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Grade 1 的 blade 索引
    """
    grade_indices = compute_grade_indices(euclidean_dim)
    return grade_indices[1]


def get_motor_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    取得 Motor 的非零 blade 索引。

    Motor 包含偶數 grade 分量，但必須排除 pseudoscalar（最高 grade）。
    當最高 grade 是偶數時，該 grade 是 pseudoscalar，必須排除。

    - CGA1D (n=3): Grade 0, 2 (4 分量) - G3 是奇數，不影響
    - CGA2D (n=4): Grade 0, 2 (7 分量) - G4 是 pseudoscalar，排除
    - CGA3D (n=5): Grade 0, 2, 4 (16 分量) - G5 是奇數，不影響
    - CGA4D (n=6): Grade 0, 2, 4 (31 分量) - G6 是 pseudoscalar，排除
    - CGA5D (n=7): Grade 0, 2, 4, 6 (64 分量) - G7 是奇數，不影響
    - CGA6D (n=8): Grade 0, 2, 4, 6 (127 分量) - G8 是 pseudoscalar，排除
    - CGA7D (n=9): Grade 0, 2, 4, 6, 8 (256 分量) - G9 是奇數，不影響
    - ...

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Motor 的 blade 索引
    """
    grade_indices = compute_grade_indices(euclidean_dim)
    total_dim = euclidean_dim + 2
    max_grade = total_dim  # CGA 代數的最高 grade = pseudoscalar

    indices = list(grade_indices[0])  # Grade 0
    indices.extend(grade_indices[2])  # Grade 2

    # 對於每個偶數 grade，只有當它不是 pseudoscalar 時才加入
    for grade in range(4, max_grade + 1, 2):
        if grade in grade_indices and grade != max_grade:
            indices.extend(grade_indices[grade])

    return tuple(sorted(indices))


def get_blade_names(euclidean_dim: int) -> List[str]:
    """
    取得所有 blade 的人類可讀名稱。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        Blade 名稱列表（使用 e1, e2, ..., e+, e- 命名）
    """
    layout, _, _ = create_cga_algebra(euclidean_dim)
    total_dim = euclidean_dim + 2

    # clifford conformalize 使用 1-based 索引
    # 順序: e1, e2, ..., en, e+, e-
    # 在 bladeTupList 中，基底 1 對應 e1, 基底 n+1 對應 e+, 基底 n+2 對應 e-
    basis_names = {}
    for i in range(euclidean_dim):
        basis_names[i + 1] = f"e{i+1}"
    basis_names[euclidean_dim + 1] = "e+"
    basis_names[euclidean_dim + 2] = "e-"

    names = []
    for blade_tuple in layout.bladeTupList:
        if len(blade_tuple) == 0:
            names.append("1")  # Scalar
        else:
            blade_name = "".join(basis_names[i] for i in blade_tuple)
            names.append(blade_name)

    return names


def verify_null_basis_properties(euclidean_dim: int) -> Dict[str, bool]:
    """
    驗證 Null Basis 的數學性質。

    Args:
        euclidean_dim: 歐幾里得空間維度

    Returns:
        性質驗證結果字典
    """
    _, _, stuff = create_cga_algebra(euclidean_dim)

    eo = stuff['eo']
    einf = stuff['einf']

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
# 預定義 CGA 代數參數
# =============================================================================

# CGA1D Cl(2,1) 參數
CGA1D_EUCLIDEAN_DIM = 1
CGA1D_BLADE_COUNT = 8
CGA1D_SIGNATURE = (1, 1, -1)  # e1+, e++, e--

# CGA2D Cl(3,1) 參數
CGA2D_EUCLIDEAN_DIM = 2
CGA2D_BLADE_COUNT = 16
CGA2D_SIGNATURE = (1, 1, 1, -1)  # e1+, e2+, e++, e--

# CGA3D Cl(4,1) 參數
CGA3D_EUCLIDEAN_DIM = 3
CGA3D_BLADE_COUNT = 32
CGA3D_SIGNATURE = (1, 1, 1, 1, -1)  # e1+, e2+, e3+, e++, e--

# CGA4D Cl(5,1) 參數
CGA4D_EUCLIDEAN_DIM = 4
CGA4D_BLADE_COUNT = 64
CGA4D_SIGNATURE = (1, 1, 1, 1, 1, -1)  # e1+, e2+, e3+, e4+, e++, e--

# CGA5D Cl(6,1) 參數
CGA5D_EUCLIDEAN_DIM = 5
CGA5D_BLADE_COUNT = 128
CGA5D_SIGNATURE = (1, 1, 1, 1, 1, 1, -1)  # e1+, e2+, e3+, e4+, e5+, e++, e--


if __name__ == "__main__":
    # 驗證各維度 CGA 代數
    for dim in [1, 2, 3, 4, 5]:
        print(f"\n=== CGA{dim}D Cl({dim+1},1) ===")
        print(f"Blade count: {compute_blade_count(dim)}")

        grade_indices = compute_grade_indices(dim)
        print(f"Grade indices: {grade_indices}")

        upgc_indices = get_upgc_point_indices(dim)
        print(f"UPGC Point indices: {upgc_indices} ({len(upgc_indices)} components)")

        motor_indices = get_motor_indices(dim)
        print(f"Motor indices: {motor_indices} ({len(motor_indices)} components)")

        props = verify_null_basis_properties(dim)
        print(f"Null basis properties: {props}")

        blade_names = get_blade_names(dim)
        print(f"Blade names: {blade_names}")
