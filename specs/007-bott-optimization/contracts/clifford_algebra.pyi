"""
API 類型定義：Clifford 代數介面

此檔案定義更新後的 API 類型簽章。
"""

from typing import Literal, Optional
from torch import Tensor

class CliffordAlgebraBase:
    """Clifford 代數基底類別"""

    @property
    def p(self) -> int: ...
    @property
    def q(self) -> int: ...
    @property
    def r(self) -> int: ...
    @property
    def count_blade(self) -> int: ...
    @property
    def count_rotor(self) -> int: ...
    @property
    def count_bivector(self) -> int: ...
    @property
    def max_grade(self) -> int: ...
    @property
    def algebra_type(self) -> Literal["vga", "cga", "pga", "general"]: ...

    # 核心運算
    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor: ...
    def outer(self, a: Tensor, b: Tensor) -> Tensor: ...
    def inner(self, a: Tensor, b: Tensor) -> Tensor: ...
    def reverse(self, mv: Tensor) -> Tensor: ...
    def dual(self, mv: Tensor) -> Tensor: ...


class SymmetricClWrapper(CliffordAlgebraBase):
    """
    對稱代數包裝器：將 Cl(p,q) 映射到 Cl(q,p)

    當 p < q 時使用，透過索引重排提供 Cl(p,q) 介面。
    """

    def __init__(self, base_algebra: CliffordAlgebraBase, p: int, q: int) -> None:
        """
        Args:
            base_algebra: 基底代數 Cl(q,p)
            p: 正簽章維度（p < q）
            q: 負簽章維度
        """
        ...

    @property
    def base_algebra(self) -> CliffordAlgebraBase:
        """取得基底代數 Cl(q,p)"""
        ...

    def _swap_indices(self, mv: Tensor) -> Tensor:
        """將 Cl(p,q) 多重向量轉換為 Cl(q,p) 表示"""
        ...


class BottPeriodicityAlgebra(CliffordAlgebraBase):
    """
    Bott 週期性代數：使用張量化運算的高維 Clifford 代數

    對於 p+q >= 8，分解為基底代數 + 矩陣因子。
    """

    def __init__(self, p: int, q: int) -> None:
        """
        Args:
            p: 正簽章維度
            q: 負簽章維度
        """
        ...

    @property
    def base_algebra(self) -> CliffordAlgebraBase:
        """取得基底代數 Cl(p mod 8, q mod 8)"""
        ...

    @property
    def matrix_size(self) -> int:
        """取得矩陣因子大小（16^週期數）"""
        ...

    @property
    def periods(self) -> int:
        """取得 Bott 週期數"""
        ...


# 工廠函數

def Cl(p: int, q: int = 0, r: int = 0) -> CliffordAlgebraBase:
    """
    建立 Clifford 代數 Cl(p, q, r)

    路由邏輯：
    - p+q < 8 且 p >= q: HardcodedClWrapper
    - p+q < 8 且 p < q: SymmetricClWrapper
    - p+q >= 8: BottPeriodicityAlgebra

    Args:
        p: 正簽章維度
        q: 負簽章維度
        r: 退化維度（目前必須為 0）

    Returns:
        CliffordAlgebraBase 實例
    """
    ...


def VGA(n: int) -> CliffordAlgebraBase:
    """建立 VGA(n) = Cl(n, 0)"""
    ...


def CGA(n: int) -> CliffordAlgebraBase:
    """建立 CGA(n) = Cl(n+1, 1)"""
    ...


def PGA(n: int) -> CliffordAlgebraBase:
    """建立 PGA(n) = Cl(n, 0, 1)"""
    ...
