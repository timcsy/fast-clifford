# Data Model: Unified Cl(p,q,0) Codegen System

**Feature**: 006-unified-clifford-codegen
**Date**: 2025-12-18

## Core Entities

### 1. CliffordAlgebraBase (Abstract)

所有 Clifford 代數實作的抽象基類。

```python
class CliffordAlgebraBase(ABC):
    """任意 Clifford 代數 Cl(p,q,r) 的統一介面"""

    # === 簽名屬性 ===
    p: int                    # 正維度（e_i² = +1 的數量）
    q: int                    # 負維度（e_i² = -1 的數量）
    r: int                    # 退化維度（e_i² = 0 的數量）

    # === 計數屬性 ===
    count_blade: int          # 總 blade 數 = 2^(p+q+r)
    count_rotor: int          # Rotor 分量數（偶數 grade 總和）
    count_bivector: int       # Bivector 分量數 = C(p+q+r, 2)
    max_grade: int            # 最大 grade = p+q+r

    # === 類型識別 ===
    algebra_type: str         # 'vga' | 'cga' | 'pga' | 'general'
```

**Validation Rules**:
- `p >= 0`, `q >= 0`, `r >= 0`
- `count_blade == 2 ** (p + q + r)`
- `algebra_type` 自動推導：
  - `q == 0 and r == 0` → 'vga'
  - `q == 1 and r == 0` → 'cga'
  - `r > 0` → 'pga'
  - else → 'general'

---

### 2. HardcodedClWrapper

包裝預生成的硬編碼代數（p+q ≤ 9）。

```python
class HardcodedClWrapper(CliffordAlgebraBase):
    """包裝自動生成的硬編碼代數模組"""

    _module: ModuleType       # 對應的 algebras/generated/cl_{p}_{q} 模組

    # 狀態：無狀態，純函數包裝
```

**Relationships**:
- 1:1 對應 `algebras/generated/cl_{p}_{q}/` 模組
- 繼承 `CliffordAlgebraBase` 所有屬性和方法

---

### 3. BottPeriodicityAlgebra

利用 Bott 週期性實作的高維度代數（p+q > 9）。

```python
class BottPeriodicityAlgebra(CliffordAlgebraBase):
    """利用 Bott 週期性的高維度代數"""

    _base_algebra: HardcodedClWrapper  # 基礎代數 Cl(p mod 8, q mod 8)
    _period_count: int                  # Bott 週期數 k = (p+q) // 8
    _tensor_shape: Tuple[int, ...]      # 張量形狀 (base_blades, 16, 16, ...)
```

**State Transitions**: 無（純計算，無狀態）

**Validation Rules**:
- `_period_count >= 1`
- `_base_algebra.count_blade <= 512`

---

### 4. Multivector

統一的 multivector 包裝類別。

```python
class Multivector:
    """Multivector 包裝，支援運算子重載"""

    data: Tensor                # 底層張量 shape (..., count_blade)
    algebra: CliffordAlgebraBase  # 所屬代數

    # 支援的運算子
    __mul__: Multivector        # 幾何積 a * b
    __xor__: Multivector        # 外積 a ^ b
    __or__: Multivector         # 內積 a | b
    __lshift__: Multivector     # 左縮並 a << b
    __rshift__: Multivector     # 右縮並 a >> b
    __invert__: Multivector     # 反轉 ~a
    __pow__: Multivector        # a ** -1 = 逆元
```

**Validation Rules**:
- `data.shape[-1] == algebra.count_blade`
- `data.dtype in [torch.float32, torch.float64]`

---

### 5. Rotor

Rotor（偶數 grade versor）包裝類別。

```python
class Rotor:
    """Rotor 包裝，自動使用加速運算"""

    data: Tensor                # 底層張量 shape (..., count_rotor)
    algebra: CliffordAlgebraBase  # 所屬代數

    # 支援的運算子
    __mul__: Rotor              # 組合 compose_rotor
    __matmul__: Multivector     # sandwich r @ x
    __invert__: Rotor           # 反轉 reverse_rotor
```

**Validation Rules**:
- `data.shape[-1] == algebra.count_rotor`

---

### 6. CGAWrapper (CGA Specialization)

CGA 特化包裝，提供 encode/decode。

```python
class CGAWrapper(CliffordAlgebraBase):
    """CGA(n) = Cl(n+1, 1) 特化包裝"""

    _base: CliffordAlgebraBase  # 底層 Cl(n+1, 1) 實作
    dim_euclidean: int          # 歐幾里得維度 n
    count_point: int            # CGA 點分量數 = n + 2

    # Null basis 定義（Dorst 慣例）
    # e_o = (e_- - e_+) / 2
    # e_inf = e_- + e_+
```

**Relationships**:
- 包裝 `HardcodedClWrapper` 或 `BottPeriodicityAlgebra`

---

### 7. VGAWrapper (VGA Specialization)

VGA 特化包裝。

```python
class VGAWrapper(CliffordAlgebraBase):
    """VGA(n) = Cl(n, 0) 特化包裝"""

    _base: CliffordAlgebraBase  # 底層 Cl(n, 0) 實作
    dim_euclidean: int          # 歐幾里得維度 n
```

---

### 8. PGAEmbedding (PGA Specialization)

PGA 透過 CGA 嵌入的實作。

```python
class PGAEmbedding(CliffordAlgebraBase):
    """PGA(n) = Cl(n, 0, 1) 嵌入 CGA(n) 實作"""

    _cga: CGAWrapper            # 底層 CGA(n) 實作
    dim_euclidean: int          # 歐幾里得維度 n

    # 映射表
    _pga_to_cga: Dict[int, int]  # PGA blade → CGA blade
    _cga_to_pga: Dict[int, int]  # CGA blade → PGA blade (partial)
```

---

## Index Constants

每個生成的代數模組包含以下常數：

```python
# algebras/generated/cl_4_1/constants.py

# Blade 索引（按 grade 排序）
SCALAR_INDEX = 0
VECTOR_INDICES = (1, 2, 3, 4, 5)           # Grade 1
BIVECTOR_INDICES = (6, 7, 8, 9, ...)       # Grade 2
...
PSEUDOSCALAR_INDEX = 31

# 稀疏表示
ROTOR_MASK = [0, 6, 7, 8, ...]             # 偶數 grade 索引
ROTOR_INDICES = {0: 0, 6: 1, 7: 2, ...}    # 完整 → 稀疏映射

# 反轉符號
REVERSE_SIGNS = [1, 1, 1, 1, 1, -1, -1, ...]  # 按 blade 索引

# Grade 映射
GRADE_SLICES = {
    0: slice(0, 1),
    1: slice(1, 6),
    2: slice(6, 16),
    ...
}
```

---

## Tensor Shapes

### Shape Conventions

| 概念 | 形狀 | 說明 |
|------|------|------|
| Batch | `(...)` | 任意批次維度 |
| Full Multivector | `(..., count_blade)` | 完整 blade 係數 |
| Rotor | `(..., count_rotor)` | 偶數 grade 係數 |
| Bivector | `(..., count_bivector)` | Grade-2 係數 |
| Point (CGA) | `(..., count_point)` | Grade-1 係數 |
| Scalar | `(..., 1)` | Grade-0 係數 |

### Examples for CGA3D (Cl(4,1))

```python
count_blade = 32
count_rotor = 16
count_bivector = 10
count_point = 5

# Input shapes
full_mv: Tensor[..., 32]
rotor: Tensor[..., 16]
bivector: Tensor[..., 10]
point: Tensor[..., 5]
euclidean: Tensor[..., 3]
```

---

## Blade Ordering

採用二進制排序（與現有 CGA 實作一致）：

```
Cl(4,1) Blade Order:
Index  Binary    Basis       Grade
0      00000     1           0
1      00001     e1          1
2      00010     e2          1
3      00011     e12         2
4      00100     e3          1
5      00101     e13         2
6      00110     e23         2
7      00111     e123        3
8      01000     e4          1
...
31     11111     e12345      5
```

---

## Entity Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    CliffordAlgebraBase                       │
│                       (Abstract)                             │
└────────────┬──────────────┬──────────────┬─────────────────┘
             │              │              │
             ▼              ▼              ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
│HardcodedClWrapper│ │BottPeriodicity │ │RuntimeClifford    │
│  (p+q ≤ 9)     │ │   Algebra      │ │   Algebra         │
│                │ │  (p+q > 9)     │ │  (r > 0, fallback)│
└───────┬────────┘ └───────┬────────┘ └────────────────────┘
        │                  │
        └──────────────────┴───────────────┐
                                           │
        ┌──────────────────────────────────┼──────────────────┐
        │                                  │                  │
        ▼                                  ▼                  ▼
┌────────────────┐              ┌────────────────┐  ┌────────────────┐
│   VGAWrapper   │              │   CGAWrapper   │  │  PGAEmbedding  │
│   Cl(n,0)      │              │   Cl(n+1,1)    │  │   Cl(n,0,1)    │
└────────────────┘              └────────────────┘  └────────────────┘
                                        │
                                        ▼
                                ┌────────────────┐
                                │  PGAEmbedding  │
                                │  (uses CGA)    │
                                └────────────────┘
```

---

## Factory Function Logic

```python
def Cl(p: int, q: int = 0, r: int = 0) -> CliffordAlgebraBase:
    """統一工廠函數"""

    if r > 0:
        # 退化維度：使用 Runtime 或 PGA 嵌入
        if q == 0:  # PGA pattern
            return PGAEmbedding(p)
        else:
            return RuntimeCliffordAlgebra(p, q, r)

    blade_count = 2 ** (p + q)
    if blade_count <= 512:  # p+q <= 9
        return HardcodedClWrapper(p, q)
    else:
        return BottPeriodicityAlgebra(p, q)


def VGA(n: int) -> VGAWrapper:
    """VGA(n) = Cl(n, 0)"""
    return VGAWrapper(Cl(n, 0))


def CGA(n: int) -> CGAWrapper:
    """CGA(n) = Cl(n+1, 1)"""
    return CGAWrapper(Cl(n + 1, 1), euclidean_dim=n)


def PGA(n: int) -> PGAEmbedding:
    """PGA(n) = Cl(n, 0, 1)"""
    return PGAEmbedding(n)
```
