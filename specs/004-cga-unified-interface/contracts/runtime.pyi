"""
RuntimeCGAAlgebra 型別定義

運行時計算的 CGA 代數，支援任意維度。
"""

from torch import Tensor, nn
from typing import Tuple, Optional


class RuntimeCGAAlgebra(nn.Module):
    """
    運行時計算的 CGA 代數。

    使用張量化批次操作在運行時計算幾何積，
    無需預先生成硬編碼展開程式碼。

    適用於 CGA6D 及更高維度。

    Attributes:
        euclidean_dim: 歐幾里得維度
        blade_count: 總 blade 數
        point_count: UPGC 點分量數
        motor_count: Motor 分量數
    """

    euclidean_dim: int
    blade_count: int
    point_count: int
    motor_count: int
    signature: Tuple[int, ...]

    # 內部 buffers（延遲初始化）
    left_idx: Tensor
    right_idx: Tensor
    result_idx: Tensor
    signs: Tensor
    point_mask: Tensor
    motor_mask: Tensor
    reverse_signs: Tensor

    def __init__(self, euclidean_dim: int) -> None:
        """
        建立運行時 CGA 代數。

        Args:
            euclidean_dim: 歐幾里得維度 n

        Note:
            實際的代數參數在首次呼叫時延遲初始化，
            以避免未使用代數的計算開銷。
        """
        ...

    def _ensure_initialized(self) -> None:
        """
        確保代數參數已初始化。

        如果尚未初始化，計算 Cayley 表並註冊 buffers。
        """
        ...

    # 核心操作
    def upgc_encode(self, x: Tensor) -> Tensor:
        """
        歐幾里得座標編碼為 UPGC 點。

        Args:
            x: 歐幾里得座標，shape (..., n)

        Returns:
            UPGC 點，shape (..., n+2)
        """
        ...

    def upgc_decode(self, point: Tensor) -> Tensor:
        """
        UPGC 點解碼為歐幾里得座標。

        Args:
            point: UPGC 點，shape (..., n+2)

        Returns:
            歐幾里得座標，shape (..., n)
        """
        ...

    def geometric_product_full(self, a: Tensor, b: Tensor) -> Tensor:
        """
        完整幾何積（張量化批次操作）。

        使用預計算的索引和符號張量進行計算，
        無 Python 迴圈，ONNX 相容。

        Args:
            a: 左運算元，shape (..., blade_count)
            b: 右運算元，shape (..., blade_count)

        Returns:
            結果多向量，shape (..., blade_count)
        """
        ...

    def sandwich_product_sparse(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        稀疏三明治積 M × X × M̃。

        Args:
            motor: Motor，shape (..., motor_count)
            point: UPGC 點，shape (..., point_count)

        Returns:
            變換後的點，shape (..., point_count)
        """
        ...

    def reverse_full(self, mv: Tensor) -> Tensor:
        """
        多向量反轉。

        Args:
            mv: 多向量，shape (..., blade_count)

        Returns:
            反轉後的多向量，shape (..., blade_count)
        """
        ...

    def reverse_motor(self, motor: Tensor) -> Tensor:
        """
        Motor 反轉。

        Args:
            motor: Motor，shape (..., motor_count)

        Returns:
            反轉後的 Motor，shape (..., motor_count)
        """
        ...

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        前向傳播（三明治積）。

        可直接作為 nn.Module 使用。

        Args:
            motor: Motor，shape (..., motor_count)
            point: UPGC 點，shape (..., point_count)

        Returns:
            變換後的點，shape (..., point_count)
        """
        ...


class RuntimeCGACareLayer(nn.Module):
    """運行時 CGA Care Layer"""

    def __init__(self, algebra: RuntimeCGAAlgebra) -> None:
        ...

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        ...


class RuntimeUPGCEncoder(nn.Module):
    """運行時 UPGC 編碼器"""

    def __init__(self, algebra: RuntimeCGAAlgebra) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...


class RuntimeUPGCDecoder(nn.Module):
    """運行時 UPGC 解碼器"""

    def __init__(self, algebra: RuntimeCGAAlgebra) -> None:
        ...

    def forward(self, point: Tensor) -> Tensor:
        ...


class RuntimeCGATransformPipeline(nn.Module):
    """運行時 CGA 完整轉換流水線"""

    def __init__(self, algebra: RuntimeCGAAlgebra) -> None:
        ...

    def forward(self, motor: Tensor, x: Tensor) -> Tensor:
        """
        完整轉換：編碼 → 三明治積 → 解碼

        Args:
            motor: Motor，shape (..., motor_count)
            x: 歐幾里得座標，shape (..., n)

        Returns:
            變換後的歐幾里得座標，shape (..., n)
        """
        ...
