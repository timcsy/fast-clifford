"""
CGA4D Cl(5,1) Functional Interface Type Stubs

自動生成的型別定義檔案，描述 CGA4D 模組的公開 API。
"""

from torch import Tensor

# =============================================================================
# 代數常數
# =============================================================================

EUCLIDEAN_DIM: int = 4
BLADE_COUNT: int = 64
UPGC_POINT_DIM: int = 6
MOTOR_DIM: int = 31

SIGNATURE: tuple[int, ...]  # (1, 1, 1, 1, 1, -1)

GRADE_0_INDICES: tuple[int, ...]  # (0,) - 1 分量
GRADE_1_INDICES: tuple[int, ...]  # (1, 2, 3, 4, 5, 6) - 6 分量
GRADE_2_INDICES: tuple[int, ...]  # 15 分量
GRADE_3_INDICES: tuple[int, ...]  # 20 分量
GRADE_4_INDICES: tuple[int, ...]  # 15 分量
GRADE_5_INDICES: tuple[int, ...]  # 6 分量
GRADE_6_INDICES: tuple[int, ...]  # (63,) - 1 分量

MOTOR_SPARSE_INDICES: tuple[int, ...]  # 31 分量
MOTOR_REVERSE_SIGNS: tuple[int, ...]   # 31 個符號

# =============================================================================
# 幾何積函式
# =============================================================================

def geometric_product_full(a: Tensor, b: Tensor) -> Tensor:
    """
    完整幾何積：a * b

    Args:
        a: 左運算元，shape (..., 64)
        b: 右運算元，shape (..., 64)

    Returns:
        幾何積結果，shape (..., 64)
    """
    ...

# =============================================================================
# 稀疏運算函式
# =============================================================================

def sandwich_product_sparse(motor: Tensor, point: Tensor) -> Tensor:
    """
    稀疏三明治積：Motor × Point × Motor̃

    Args:
        motor: 馬達，shape (..., 31)
               格式：[Grade 0 (1), Grade 2 (15), Grade 4 (15)]
        point: UPGC 點，shape (..., 6)
               格式：[e1, e2, e3, e4, e+, e-]

    Returns:
        變換後的點，shape (..., 6)
    """
    ...

def motor_reverse(motor: Tensor) -> Tensor:
    """
    計算馬達的反轉：M̃

    Args:
        motor: 馬達，shape (..., 31)

    Returns:
        反轉後的馬達，shape (..., 31)
    """
    ...

# =============================================================================
# UPGC 編解碼
# =============================================================================

def upgc_encode(x: Tensor) -> Tensor:
    """
    將 4D 歐幾里得座標編碼為 UPGC 表示。

    公式：X = n_o + x + 0.5|x|² n_inf

    Args:
        x: 4D 座標，shape (..., 4)

    Returns:
        UPGC 點，shape (..., 6)
    """
    ...

def upgc_decode(point: Tensor) -> Tensor:
    """
    將 UPGC 表示解碼為 4D 歐幾里得座標。

    Args:
        point: UPGC 點，shape (..., 6)

    Returns:
        4D 座標，shape (..., 4)
    """
    ...

# =============================================================================
# 輔助函式
# =============================================================================

def reverse_full(mv: Tensor) -> Tensor:
    """
    計算完整多向量的反轉。

    Args:
        mv: 多向量，shape (..., 64)

    Returns:
        反轉後的多向量，shape (..., 64)
    """
    ...

def grade_select(mv: Tensor, grade: int) -> Tensor:
    """
    選取指定 grade 的分量。

    Args:
        mv: 多向量，shape (..., 64)
        grade: 目標 grade (0-6)

    Returns:
        選取的分量，shape 根據 grade 而定
    """
    ...
