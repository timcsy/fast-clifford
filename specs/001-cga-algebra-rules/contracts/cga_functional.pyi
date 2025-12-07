"""
CGA Functional Module Type Stubs

自動生成的幾何代數函式型別定義
"""

from typing import Tuple
import torch
from torch import Tensor


# ============================================================
# 常數定義
# ============================================================

# Blade 索引範圍
BLADE_COUNT: int  # = 32
GRADE_0_INDICES: Tuple[int, ...]  # = (0,)
GRADE_1_INDICES: Tuple[int, ...]  # = (1, 2, 3, 4, 5)
GRADE_2_INDICES: Tuple[int, ...]  # = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
GRADE_3_INDICES: Tuple[int, ...]  # = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
GRADE_4_INDICES: Tuple[int, ...]  # = (26, 27, 28, 29, 30)
GRADE_5_INDICES: Tuple[int, ...]  # = (31,)

# 稀疏性遮罩
UPGC_POINT_MASK: Tuple[int, ...]  # = (1, 2, 3, 4, 5) - Grade 1 only
MOTOR_MASK: Tuple[int, ...]       # = (0, 6, 7, ..., 30) - Grade 0, 2, 4

# Reverse 符號表 (32 elements)
REVERSE_SIGNS: Tensor  # shape: (32,), values: +1 or -1


# ============================================================
# 核心函式
# ============================================================

def geometric_product_full(
    a: Tensor,
    b: Tensor
) -> Tensor:
    """
    完整的幾何積計算（32 × 32 → 32）

    Args:
        a: 左運算元 multivector, shape (..., 32)
        b: 右運算元 multivector, shape (..., 32)

    Returns:
        結果 multivector, shape (..., 32)

    Note:
        此函式為完整展開版本，無迴圈，ONNX 相容
    """
    ...


def geometric_product_motor_point(
    motor: Tensor,
    point: Tensor
) -> Tensor:
    """
    Motor × UPGC Point 的稀疏幾何積

    利用稀疏性假設優化計算量

    Args:
        motor: Motor (偶數 grade), shape (..., 16)
        point: UPGC Point (Grade 1), shape (..., 5)

    Returns:
        中間結果 multivector, shape (..., 32) 或稀疏形式
    """
    ...


def reverse_motor(motor: Tensor) -> Tensor:
    """
    計算 Motor 的 Reverse

    對 Grade 2 分量取反符號，Grade 0, 4 保持不變

    Args:
        motor: Motor, shape (..., 16)

    Returns:
        Reversed Motor, shape (..., 16)
    """
    ...


def sandwich_product_sparse(
    motor: Tensor,
    point: Tensor
) -> Tensor:
    """
    稀疏三明治積：M × X × M̃

    這是 fast-clifford 的核心函式，利用以下稀疏性假設：
    - 輸入 X: 只有 Grade 1 有非零值（5 個分量）
    - Motor M: 只有 Grade 0, 2, 4 有非零值（16 個分量）
    - 輸出: 只有 Grade 1 有非零值（5 個分量）

    Args:
        motor: Motor, shape (..., 16)
               索引順序: [scalar, e12, e13, e1+, e1-, e23, e2+, e2-,
                         e3+, e3-, e+-, e123+, e123-, e12+-, e13+-, e23+-]
        point: UPGC Point, shape (..., 5)
               索引順序: [e1, e2, e3, e+, e-]

    Returns:
        變換後的 UPGC Point, shape (..., 5)
        索引順序: [e1, e2, e3, e+, e-]

    Note:
        - 所有計算在 float32 下進行（調用方需處理精度轉換）
        - 無迴圈，完全展開的算術運算
        - ONNX 相容，匯出後只有 Add/Mul/Neg 節點
    """
    ...


# ============================================================
# 輔助函式
# ============================================================

def upgc_encode(x: Tensor) -> Tensor:
    """
    將 3D 向量編碼為 UPGC Point

    X = n_o + x + 0.5|x|^2 * n_∞

    Args:
        x: 3D 向量, shape (..., 3)

    Returns:
        UPGC Point, shape (..., 5)
    """
    ...


def upgc_decode(point: Tensor) -> Tensor:
    """
    從 UPGC Point 解碼 3D 向量

    Args:
        point: UPGC Point, shape (..., 5)

    Returns:
        3D 向量, shape (..., 3)
    """
    ...


def motor_from_rotor_translator(
    rotor: Tensor,
    translator: Tensor
) -> Tensor:
    """
    從旋轉和平移組合 Motor

    M = T × R

    Args:
        rotor: 旋轉 rotor (Grade 0, 2), shape (..., 8)
        translator: 平移 translator, shape (..., 8)

    Returns:
        Motor, shape (..., 16)
    """
    ...
