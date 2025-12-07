"""
CGA5D Cl(6,1) Functional Interface Type Stubs

自動生成的型別定義檔案，描述 CGA5D 模組的公開 API。
"""

from torch import Tensor

# =============================================================================
# 代數常數
# =============================================================================

EUCLIDEAN_DIM: int = 5
BLADE_COUNT: int = 128
UPGC_POINT_DIM: int = 7
MOTOR_DIM: int = 64

SIGNATURE: tuple[int, ...]  # (1, 1, 1, 1, 1, 1, -1)

GRADE_0_INDICES: tuple[int, ...]  # (0,) - 1 分量
GRADE_1_INDICES: tuple[int, ...]  # (1, 2, 3, 4, 5, 6, 7) - 7 分量
GRADE_2_INDICES: tuple[int, ...]  # 21 分量
GRADE_3_INDICES: tuple[int, ...]  # 35 分量
GRADE_4_INDICES: tuple[int, ...]  # 35 分量
GRADE_5_INDICES: tuple[int, ...]  # 21 分量
GRADE_6_INDICES: tuple[int, ...]  # 7 分量
GRADE_7_INDICES: tuple[int, ...]  # (127,) - 1 分量

MOTOR_SPARSE_INDICES: tuple[int, ...]  # 64 分量
MOTOR_REVERSE_SIGNS: tuple[int, ...]   # 64 個符號

# =============================================================================
# 幾何積函式
# =============================================================================

def geometric_product_full(a: Tensor, b: Tensor) -> Tensor:
    """
    完整幾何積：a * b

    Args:
        a: 左運算元，shape (..., 128)
        b: 右運算元，shape (..., 128)

    Returns:
        幾何積結果，shape (..., 128)
    """
    ...

# =============================================================================
# 稀疏運算函式
# =============================================================================

def sandwich_product_sparse(motor: Tensor, point: Tensor) -> Tensor:
    """
    稀疏三明治積：Motor × Point × Motor̃

    Args:
        motor: 馬達，shape (..., 64)
               格式：[Grade 0 (1), Grade 2 (21), Grade 4 (35), Grade 6 (7)]
        point: UPGC 點，shape (..., 7)
               格式：[e1, e2, e3, e4, e5, e+, e-]

    Returns:
        變換後的點，shape (..., 7)
    """
    ...

def motor_reverse(motor: Tensor) -> Tensor:
    """
    計算馬達的反轉：M̃

    Args:
        motor: 馬達，shape (..., 64)

    Returns:
        反轉後的馬達，shape (..., 64)
    """
    ...

# =============================================================================
# UPGC 編解碼
# =============================================================================

def upgc_encode(x: Tensor) -> Tensor:
    """
    將 5D 歐幾里得座標編碼為 UPGC 表示。

    公式：X = n_o + x + 0.5|x|² n_inf

    Args:
        x: 5D 座標，shape (..., 5)

    Returns:
        UPGC 點，shape (..., 7)
    """
    ...

def upgc_decode(point: Tensor) -> Tensor:
    """
    將 UPGC 表示解碼為 5D 歐幾里得座標。

    Args:
        point: UPGC 點，shape (..., 7)

    Returns:
        5D 座標，shape (..., 5)
    """
    ...

# =============================================================================
# 輔助函式
# =============================================================================

def reverse_full(mv: Tensor) -> Tensor:
    """
    計算完整多向量的反轉。

    Args:
        mv: 多向量，shape (..., 128)

    Returns:
        反轉後的多向量，shape (..., 128)
    """
    ...

def grade_select(mv: Tensor, grade: int) -> Tensor:
    """
    選取指定 grade 的分量。

    Args:
        mv: 多向量，shape (..., 128)
        grade: 目標 grade (0-7)

    Returns:
        選取的分量，shape 根據 grade 而定
    """
    ...
