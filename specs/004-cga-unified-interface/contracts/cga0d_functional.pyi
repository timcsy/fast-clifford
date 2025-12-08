"""
CGA0D Cl(1,1) 功能操作型別定義

硬編碼的 0 維共形幾何代數操作。
"""

from torch import Tensor
from typing import Dict

# =============================================================================
# 常數
# =============================================================================

BLADE_COUNT: int  # = 4

# 各級 blade 索引
GRADE_0_INDICES: tuple  # = (0,)
GRADE_1_INDICES: tuple  # = (1, 2)
GRADE_2_INDICES: tuple  # = (3,)

# 稀疏遮罩
UPGC_POINT_MASK: tuple  # = (1, 2)  # 2 分量
MOTOR_MASK: tuple       # = (0, 3)  # 2 分量

# 反轉符號
REVERSE_SIGNS: tuple    # = (1, 1, 1, -1)
MOTOR_REVERSE_SIGNS: tuple  # = (1, -1)

# =============================================================================
# 幾何積
# =============================================================================

def geometric_product_full(a: Tensor, b: Tensor) -> Tensor:
    """
    完整 4×4 幾何積。

    Args:
        a: 左運算元，shape (..., 4)
        b: 右運算元，shape (..., 4)

    Returns:
        結果多向量，shape (..., 4)

    Note:
        完全展開，無迴圈，ONNX 相容。
    """
    ...

# =============================================================================
# 反轉操作
# =============================================================================

def reverse_full(mv: Tensor) -> Tensor:
    """
    完整多向量反轉。

    Grade 0, 1: 不變
    Grade 2: 取負

    Args:
        mv: 多向量，shape (..., 4)

    Returns:
        反轉後的多向量，shape (..., 4)
    """
    ...

def reverse_motor(motor: Tensor) -> Tensor:
    """
    Motor 反轉（稀疏版本）。

    Args:
        motor: Motor，shape (..., 2)

    Returns:
        反轉後的 Motor，shape (..., 2)
    """
    ...

# =============================================================================
# UPGC 編碼/解碼
# =============================================================================

def upgc_encode(x: Tensor) -> Tensor:
    """
    歐幾里得座標編碼為 UPGC 點。

    對於 0D，沒有歐幾里得分量，返回原點 n_o。

    Args:
        x: 歐幾里得座標，shape (..., 0) 或 shape (...)

    Returns:
        UPGC 點，shape (..., 2)
        [e+, e-] = [-0.5, 0.5] (原點)
    """
    ...

def upgc_decode(point: Tensor) -> Tensor:
    """
    UPGC 點解碼為歐幾里得座標。

    對於 0D，返回空張量。

    Args:
        point: UPGC 點，shape (..., 2)

    Returns:
        歐幾里得座標，shape (..., 0)
    """
    ...

# =============================================================================
# 三明治積
# =============================================================================

def sandwich_product_sparse(motor: Tensor, point: Tensor) -> Tensor:
    """
    稀疏三明治積 M × X × M̃。

    Args:
        motor: Motor [scalar, e+-]，shape (..., 2)
        point: UPGC 點 [e+, e-]，shape (..., 2)

    Returns:
        變換後的點，shape (..., 2)

    Note:
        利用稀疏性優化，僅計算非零項。
    """
    ...

# =============================================================================
# 驗證函式
# =============================================================================

def verify_null_basis() -> Dict[str, bool]:
    """
    驗證空基底屬性。

    Returns:
        包含以下鍵的字典：
        - 'eo_squared_zero': n_o² = 0
        - 'einf_squared_zero': n_∞² = 0
        - 'eo_einf_minus_one': n_o · n_∞ = -1
    """
    ...
