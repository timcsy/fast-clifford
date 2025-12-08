"""
CGAAlgebraBase 型別定義

所有 CGA 代數的統一介面抽象基底類別。
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import Tensor, nn


class CGAAlgebraBase(ABC):
    """CGA 代數抽象基底類別"""

    # 屬性
    @property
    @abstractmethod
    def euclidean_dim(self) -> int:
        """歐幾里得維度 n（CGA = Cl(n+1, 1, 0)）"""
        ...

    @property
    @abstractmethod
    def blade_count(self) -> int:
        """總 blade 數 = 2^(n+2)"""
        ...

    @property
    @abstractmethod
    def point_count(self) -> int:
        """UPGC 點分量數 = n+2（Grade 1）"""
        ...

    @property
    @abstractmethod
    def motor_count(self) -> int:
        """Motor 分量數（Grade 0, 2, 4... 偶數級）"""
        ...

    @property
    @abstractmethod
    def signature(self) -> Tuple[int, ...]:
        """Clifford 簽名（+1, +1, ..., -1）"""
        ...

    @property
    def clifford_notation(self) -> str:
        """Clifford 表示法，如 'Cl(4,1,0)'"""
        p = self.euclidean_dim + 1
        return f"Cl({p},1,0)"

    # 核心操作
    @abstractmethod
    def upgc_encode(self, x: Tensor) -> Tensor:
        """
        歐幾里得座標編碼為 UPGC 點。

        Args:
            x: 歐幾里得座標，shape (..., n)

        Returns:
            UPGC 點，shape (..., n+2)
        """
        ...

    @abstractmethod
    def upgc_decode(self, point: Tensor) -> Tensor:
        """
        UPGC 點解碼為歐幾里得座標。

        Args:
            point: UPGC 點，shape (..., n+2)

        Returns:
            歐幾里得座標，shape (..., n)
        """
        ...

    @abstractmethod
    def geometric_product_full(self, a: Tensor, b: Tensor) -> Tensor:
        """
        完整幾何積。

        Args:
            a: 左運算元，shape (..., blade_count)
            b: 右運算元，shape (..., blade_count)

        Returns:
            結果多向量，shape (..., blade_count)
        """
        ...

    @abstractmethod
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

    @abstractmethod
    def reverse_full(self, mv: Tensor) -> Tensor:
        """
        多向量反轉。

        Args:
            mv: 多向量，shape (..., blade_count)

        Returns:
            反轉後的多向量，shape (..., blade_count)
        """
        ...

    @abstractmethod
    def reverse_motor(self, motor: Tensor) -> Tensor:
        """
        Motor 反轉。

        Args:
            motor: Motor，shape (..., motor_count)

        Returns:
            反轉後的 Motor，shape (..., motor_count)
        """
        ...

    # 層工廠方法
    @abstractmethod
    def get_care_layer(self) -> nn.Module:
        """取得 CareLayer（三明治積層）"""
        ...

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        """取得 UPGC 編碼器層"""
        ...

    @abstractmethod
    def get_decoder(self) -> nn.Module:
        """取得 UPGC 解碼器層"""
        ...

    @abstractmethod
    def get_transform_pipeline(self) -> nn.Module:
        """取得完整轉換流水線（編碼 + 三明治積 + 解碼）"""
        ...


class HardcodedCGAWrapper(CGAAlgebraBase):
    """包裝現有 cga0d-cga5d 模組的適配器"""

    def __init__(self, module) -> None:
        """
        Args:
            module: cga0d, cga1d, ..., cga5d 模組
        """
        ...


class RuntimeCGAAlgebra(CGAAlgebraBase, nn.Module):
    """運行時計算的 CGA 代數（用於 CGA6D+）"""

    def __init__(self, euclidean_dim: int) -> None:
        """
        Args:
            euclidean_dim: 歐幾里得維度 n（n >= 6）
        """
        ...


# 工廠函式
def CGA(n: int) -> CGAAlgebraBase:
    """
    建立 CGA 代數。

    Args:
        n: 歐幾里得維度（0-5 使用硬編碼，6+ 使用運行時）

    Returns:
        CGAAlgebraBase 實例

    Raises:
        ValueError: 如果 n < 0

    Examples:
        >>> cga3d = CGA(3)
        >>> cga3d.blade_count
        32
    """
    ...


def Cl(p: int, q: int, r: int = 0) -> CGAAlgebraBase:
    """
    使用 Clifford 簽名建立代數。

    Args:
        p: 正簽名維度數
        q: 負簽名維度數
        r: 退化維度數（預設 0）

    Returns:
        CliffordAlgebraBase 實例

    Warnings:
        如果不是 CGA 簽名（q != 1 或 r != 0），會發出警告

    Examples:
        >>> cga3d = Cl(4, 1, 0)  # 等同於 CGA(3)
        >>> ga3d = Cl(3, 0, 0)   # 純 GA，會發出警告
    """
    ...
