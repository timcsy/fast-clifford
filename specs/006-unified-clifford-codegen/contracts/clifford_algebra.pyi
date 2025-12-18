"""
Type stubs for Unified Clifford Algebra System

Feature: 006-unified-clifford-codegen
Date: 2025-12-18
"""

from abc import ABC, abstractmethod
from typing import Tuple, Literal, Optional, overload
import torch
from torch import Tensor, nn


# =============================================================================
# Core Abstract Base Class
# =============================================================================

class CliffordAlgebraBase(ABC):
    """任意 Clifford 代數 Cl(p,q,r) 的統一介面"""

    # === Signature Properties ===
    @property
    def p(self) -> int:
        """正維度（e_i² = +1 的數量）"""
        ...

    @property
    def q(self) -> int:
        """負維度（e_i² = -1 的數量）"""
        ...

    @property
    def r(self) -> int:
        """退化維度（e_i² = 0 的數量）"""
        ...

    # === Count Properties ===
    @property
    def count_blade(self) -> int:
        """總 blade 數 = 2^(p+q+r)"""
        ...

    @property
    def count_rotor(self) -> int:
        """Rotor 分量數（偶數 grade 總和）"""
        ...

    @property
    def count_bivector(self) -> int:
        """Bivector 分量數 = C(p+q+r, 2)"""
        ...

    @property
    def max_grade(self) -> int:
        """最大 grade = p+q+r"""
        ...

    # === Type Detection ===
    @property
    def algebra_type(self) -> Literal['vga', 'cga', 'pga', 'general']:
        """代數類型"""
        ...

    # === Core Operations (Full Multivector) ===
    @abstractmethod
    def geometric_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        幾何積 ab

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def inner(self, a: Tensor, b: Tensor) -> Tensor:
        """
        內積（純量積）<ab>₀

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., 1)
        """
        ...

    @abstractmethod
    def outer(self, a: Tensor, b: Tensor) -> Tensor:
        """
        外積 a ∧ b

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def contract_left(self, a: Tensor, b: Tensor) -> Tensor:
        """
        左縮並 a ⌋ b

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def contract_right(self, a: Tensor, b: Tensor) -> Tensor:
        """
        右縮並 a ⌊ b

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def scalar(self, a: Tensor, b: Tensor) -> Tensor:
        """
        純量積 <ab>₀（同 inner）

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., 1)
        """
        ...

    @abstractmethod
    def regressive(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Meet 運算 a ∨ b

        Args:
            a: shape (..., count_blade)
            b: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def sandwich(self, v: Tensor, x: Tensor) -> Tensor:
        """
        三明治積 vxṽ

        Args:
            v: shape (..., count_blade) - versor
            x: shape (..., count_blade) - operand

        Returns:
            shape (..., count_blade)
        """
        ...

    # === Unary Operations ===
    @abstractmethod
    def reverse(self, mv: Tensor) -> Tensor:
        """
        反轉 m̃

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def involute(self, mv: Tensor) -> Tensor:
        """
        Grade 反演 m̂

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def conjugate(self, mv: Tensor) -> Tensor:
        """
        Clifford 共軛 m†

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def select_grade(self, mv: Tensor, grade: int) -> Tensor:
        """
        提取特定 grade

        Args:
            mv: shape (..., count_blade)
            grade: 0 to max_grade

        Returns:
            shape (..., count_grade_k)
        """
        ...

    @abstractmethod
    def dual(self, mv: Tensor) -> Tensor:
        """
        Poincaré 對偶

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def normalize(self, mv: Tensor) -> Tensor:
        """
        正規化到單位範數

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def inverse(self, mv: Tensor) -> Tensor:
        """
        乘法逆元 m⁻¹

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    @abstractmethod
    def norm_squared(self, mv: Tensor) -> Tensor:
        """
        範數平方 |m|²

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., 1)
        """
        ...

    @abstractmethod
    def exp(self, mv: Tensor) -> Tensor:
        """
        通用指數映射 exp(m)

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., count_blade)
        """
        ...

    # === Rotor Accelerated Operations ===
    @abstractmethod
    def compose_rotor(self, r1: Tensor, r2: Tensor) -> Tensor:
        """
        Rotor 組合 r1 r2

        Args:
            r1: shape (..., count_rotor)
            r2: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def reverse_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor 反轉 r̃

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def sandwich_rotor(self, r: Tensor, x: Tensor) -> Tensor:
        """
        Rotor 三明治積（加速版）rxr̃

        Args:
            r: shape (..., count_rotor)
            x: shape (..., count_point) or (..., count_blade)

        Returns:
            same shape as x
        """
        ...

    @abstractmethod
    def norm_squared_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor 範數平方

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., 1)
        """
        ...

    @abstractmethod
    def inverse_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor 逆元

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def normalize_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor 正規化

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_rotor)
        """
        ...

    # === Rotor-Specific Operations ===
    @abstractmethod
    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Bivector 指數映射 exp(B) → Rotor

        Args:
            B: shape (..., count_bivector)

        Returns:
            shape (..., count_rotor)
        """
        ...

    @abstractmethod
    def log_rotor(self, r: Tensor) -> Tensor:
        """
        Rotor 對數映射 log(r) → Bivector

        Args:
            r: shape (..., count_rotor)

        Returns:
            shape (..., count_bivector)
        """
        ...

    @abstractmethod
    def slerp_rotor(self, r1: Tensor, r2: Tensor, t: float) -> Tensor:
        """
        Rotor 球面線性插值

        Args:
            r1: shape (..., count_rotor)
            r2: shape (..., count_rotor)
            t: interpolation parameter [0, 1]

        Returns:
            shape (..., count_rotor)
        """
        ...

    # === Factory Methods ===
    def multivector(self, data: Tensor) -> 'Multivector':
        """建立 Multivector 包裝"""
        ...

    def rotor(self, data: Tensor) -> 'Rotor':
        """建立 Rotor 包裝"""
        ...

    # === Layer Factory ===
    def get_transform_layer(self) -> nn.Module:
        """取得 sandwich product layer"""
        ...

    def get_encoder(self) -> nn.Module:
        """取得編碼器 layer"""
        ...

    def get_decoder(self) -> nn.Module:
        """取得解碼器 layer"""
        ...


# =============================================================================
# CGA Specialization
# =============================================================================

class CGAWrapper(CliffordAlgebraBase):
    """CGA(n) = Cl(n+1, 1) 特化"""

    @property
    def dim_euclidean(self) -> int:
        """歐幾里得維度 n"""
        ...

    @property
    def count_point(self) -> int:
        """CGA 點分量數 = n + 2"""
        ...

    def encode(self, x: Tensor) -> Tensor:
        """
        歐幾里得座標 → CGA 點

        Args:
            x: shape (..., dim_euclidean)

        Returns:
            shape (..., count_point)
        """
        ...

    def decode(self, point: Tensor) -> Tensor:
        """
        CGA 點 → 歐幾里得座標

        Args:
            point: shape (..., count_point)

        Returns:
            shape (..., dim_euclidean)
        """
        ...


# =============================================================================
# VGA Specialization
# =============================================================================

class VGAWrapper(CliffordAlgebraBase):
    """VGA(n) = Cl(n, 0) 特化"""

    @property
    def dim_euclidean(self) -> int:
        """歐幾里得維度 n"""
        ...

    def encode(self, x: Tensor) -> Tensor:
        """
        向量嵌入為 Grade-1

        Args:
            x: shape (..., dim_euclidean)

        Returns:
            shape (..., dim_euclidean)  # Grade-1 部分
        """
        ...

    def decode(self, mv: Tensor) -> Tensor:
        """
        提取 Grade-1 部分

        Args:
            mv: shape (..., count_blade)

        Returns:
            shape (..., dim_euclidean)
        """
        ...


# =============================================================================
# PGA Specialization
# =============================================================================

class PGAEmbedding(CliffordAlgebraBase):
    """PGA(n) = Cl(n, 0, 1) 透過 CGA 嵌入"""

    @property
    def dim_euclidean(self) -> int:
        """歐幾里得維度 n"""
        ...

    def embed(self, pga_mv: Tensor) -> Tensor:
        """
        PGA multivector → CGA multivector

        Args:
            pga_mv: shape (..., pga_count_blade)

        Returns:
            shape (..., cga_count_blade)
        """
        ...

    def project(self, cga_mv: Tensor) -> Tensor:
        """
        CGA multivector → PGA multivector

        Args:
            cga_mv: shape (..., cga_count_blade)

        Returns:
            shape (..., pga_count_blade)
        """
        ...


# =============================================================================
# Wrapper Classes
# =============================================================================

class Multivector:
    """Multivector 運算子包裝"""

    data: Tensor
    algebra: CliffordAlgebraBase

    def __init__(self, data: Tensor, algebra: CliffordAlgebraBase) -> None: ...

    # 幾何積與純量運算
    def __mul__(self, other: 'Multivector') -> 'Multivector': ...      # a * b 幾何積
    def __rmul__(self, other: float) -> 'Multivector': ...             # s * a 純量乘
    def __truediv__(self, scalar: float) -> 'Multivector': ...         # a / s 純量除

    # 外積與內積
    def __xor__(self, other: 'Multivector') -> 'Multivector': ...      # a ^ b 外積
    def __or__(self, other: 'Multivector') -> 'Multivector': ...       # a | b 內積

    # 縮並
    def __lshift__(self, other: 'Multivector') -> 'Multivector': ...   # a << b 左縮並
    def __rshift__(self, other: 'Multivector') -> 'Multivector': ...   # a >> b 右縮並

    # 三明治積與 Meet
    def __matmul__(self, other: 'Multivector') -> 'Multivector': ...   # m @ x 三明治積
    def __and__(self, other: 'Multivector') -> 'Multivector': ...      # a & b meet (regressive)

    # 單元運算
    def __invert__(self) -> 'Multivector': ...                         # ~a 反轉
    def __pow__(self, exp: int) -> 'Multivector': ...                  # a ** n 冪次/逆元
    def __neg__(self) -> 'Multivector': ...                            # -a 取負

    # 加減法
    def __add__(self, other: 'Multivector') -> 'Multivector': ...      # a + b
    def __sub__(self, other: 'Multivector') -> 'Multivector': ...      # a - b

    # Grade 選取（使用 __call__）
    def __call__(self, grade: int) -> 'Multivector': ...               # mv(k) 選取 grade-k

    # 方法
    def grade(self, k: int) -> 'Multivector': ...                      # 同 __call__
    def dual(self) -> 'Multivector': ...
    def normalize(self) -> 'Multivector': ...
    def inverse(self) -> 'Multivector': ...
    def exp(self) -> 'Multivector': ...


class Rotor:
    """Rotor 運算子包裝（自動使用加速運算）"""

    data: Tensor
    algebra: CliffordAlgebraBase

    def __init__(self, data: Tensor, algebra: CliffordAlgebraBase) -> None: ...
    def __mul__(self, other: 'Rotor') -> 'Rotor': ...
    def __matmul__(self, point: Tensor) -> Tensor: ...
    def __invert__(self) -> 'Rotor': ...
    def normalize(self) -> 'Rotor': ...


# =============================================================================
# Factory Functions
# =============================================================================

def Cl(p: int, q: int = 0, r: int = 0) -> CliffordAlgebraBase:
    """
    建立 Clifford 代數 Cl(p, q, r)

    Args:
        p: 正維度
        q: 負維度
        r: 退化維度

    Returns:
        CliffordAlgebraBase 實例
    """
    ...


def VGA(n: int) -> VGAWrapper:
    """
    建立 VGA(n) = Cl(n, 0)

    Args:
        n: 歐幾里得維度

    Returns:
        VGAWrapper 實例
    """
    ...


def CGA(n: int) -> CGAWrapper:
    """
    建立 CGA(n) = Cl(n+1, 1)

    Args:
        n: 歐幾里得維度

    Returns:
        CGAWrapper 實例
    """
    ...


def PGA(n: int) -> PGAEmbedding:
    """
    建立 PGA(n) = Cl(n, 0, 1)

    Args:
        n: 歐幾里得維度

    Returns:
        PGAEmbedding 實例
    """
    ...
