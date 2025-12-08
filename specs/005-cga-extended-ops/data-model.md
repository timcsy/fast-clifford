# Data Model: CGA Extended Operations

## 核心實體

### EvenVersor (偶數 Versor)

偶數 Grade 多向量，用於表示幾何變換（旋轉 + 平移 + 縮放）。

| 維度 | EvenVersor 分量數 | Grade 組成 | 稀疏索引 |
|------|-------------------|-----------|---------|
| CGA0D | 2 | G0(1) + G2(1) | (0, 3) |
| CGA1D | 4 | G0(1) + G2(3) | (0, 3, 4, 5) |
| CGA2D | 8 | G0(1) + G2(6) + G4(1) | (0, 6-11, 15) |
| CGA3D | 16 | G0(1) + G2(10) + G4(5) | 見下表 |
| CGA4D | 32 | G0(1) + G2(15) + G4(15) + G6(1) | ... |
| CGA5D | 64 | G0(1) + G2(21) + G4(35) + G6(7) | ... |

**CGA3D EvenVersor 索引表**:
```
Index | Blade | Grade
------|-------|------
0     | 1     | 0
1     | e12   | 2
2     | e13   | 2
3     | e1+   | 2
4     | e1-   | 2
5     | e23   | 2
6     | e2+   | 2
7     | e2-   | 2
8     | e3+   | 2
9     | e3-   | 2
10    | e+-   | 2
11    | e123+ | 4
12    | e123- | 4
13    | e12+- | 4
14    | e13+- | 4
15    | e23+- | 4
```

### Similitude (相似變換)

EvenVersor 的子類別，僅包含平移、旋轉、縮放（排除 transversion）。

**重要**：Similitude 使用相同的 EvenVersor 儲存格式，加速來自計算路徑優化。

| 維度 | EvenVersor 分量 | Transversion 約束 | Similitude 有效自由度 |
|------|----------------|-------------------|---------------------|
| CGA0D | 2 | 0 | 2 |
| CGA1D | 4 | 1 | 3 |
| CGA2D | 8 | 2 | 6 |
| CGA3D | 16 | 3 | 13 |
| CGA4D | 32 | 4 | 28 |
| CGA5D | 64 | 5 | 59 |

**Similitude 約束**（orthonormal basis）：
```
對每個 i: coefficient(ei+) = coefficient(ei-)
```
這等價於排除 `ei ∧ eo` 分量。

### Bivector

Grade 2 多向量，用於表示旋轉軸/平面，是 `exp_bivector` 的輸入。

| 維度 | Bivector 分量數 | 說明 |
|------|----------------|------|
| CGA0D | 1 | e+- |
| CGA1D | 3 | e1+, e1-, e+- |
| CGA2D | 6 | e12, e1+, e1-, e2+, e2-, e+- |
| CGA3D | 10 | e12, e13, e1+, e1-, e23, e2+, e2-, e3+, e3-, e+- |
| CGA4D | 15 | ... |
| CGA5D | 21 | ... |

### Point (UPGC 表示)

Grade 1 多向量，用於表示空間中的點。

| 維度 | Point 分量數 | 組成 |
|------|-------------|------|
| CGA0D | 2 | e+, e- |
| CGA1D | 3 | e1, e+, e- |
| CGA2D | 4 | e1, e2, e+, e- |
| CGA3D | 5 | e1, e2, e3, e+, e- |
| CGA4D | 6 | e1, e2, e3, e4, e+, e- |
| CGA5D | 7 | e1, e2, e3, e4, e5, e+, e- |

### Multivector (完整表示)

完整 Clifford 代數元素。

| 維度 | Blade 總數 | Grade 分布 |
|------|-----------|-----------|
| CGA0D | 4 | [1, 2, 1] |
| CGA1D | 8 | [1, 3, 3, 1] |
| CGA2D | 16 | [1, 4, 6, 4, 1] |
| CGA3D | 32 | [1, 5, 10, 10, 5, 1] |
| CGA4D | 64 | [1, 6, 15, 20, 15, 6, 1] |
| CGA5D | 128 | [1, 7, 21, 35, 35, 21, 7, 1] |

### TRS 參數

Translation-Rotation-Scaling 參數，用於 `from_trs` / `to_trs` 轉換。

| 維度 | Translation | Rotation | Scaling | TRS 總計 |
|------|-------------|----------|---------|----------|
| CGA0D | 0 | 0 | 1 | 1 |
| CGA1D | 1 | 0 | 1 | 2 |
| CGA2D | 2 | 1 (角度) | 1 | 4 |
| CGA3D | 3 | 4 (四元數) 或 3 (Euler) | 1 | 7-8 |
| CGA4D | 4 | 6 (bivector) | 1 | 11 |
| CGA5D | 5 | 10 (bivector) | 1 | 16 |

**旋轉格式**：

| 格式 | 維度 | 形狀 | 說明 |
|------|------|------|------|
| `angle` | 2D | (..., 1) | 單一角度 |
| `quaternion` | 3D | (..., 4) | [w, x, y, z] |
| `euler` | 3D | (..., 3) | [roll, pitch, yaw] |
| `bivector` | 任意 | (..., C(n,2)) | 原生 bivector |

---

## 新增常數

### INNER_PRODUCT_SIGNS

各維度的度規符號向量，用於 `inner_product` 計算。

**CGA3D 範例** (32 個符號):
```python
INNER_PRODUCT_SIGNS = (
    1,   # 0:  1 (scalar)
    1,   # 1:  e1
    1,   # 2:  e2
    1,   # 3:  e3
    1,   # 4:  e+
    -1,  # 5:  e-
    -1,  # 6:  e12
    -1,  # 7:  e13
    -1,  # 8:  e1+
    1,   # 9:  e1-
    -1,  # 10: e23
    -1,  # 11: e2+
    1,   # 12: e2-
    -1,  # 13: e3+
    1,   # 14: e3-
    1,   # 15: e+-
    -1,  # 16: e123
    -1,  # 17: e12+
    1,   # 18: e12-
    -1,  # 19: e13+
    1,   # 20: e13-
    -1,  # 21: e23+
    1,   # 22: e23-
    1,   # 23: e1+-
    1,   # 24: e2+-
    1,   # 25: e3+-
    1,   # 26: e123+
    -1,  # 27: e123-
    -1,  # 28: e12+-
    -1,  # 29: e13+-
    -1,  # 30: e23+-
    -1,  # 31: e123+-
)
```

### BIVECTOR_MASK

各維度的 Bivector 索引遮罩。

```python
# CGA0D
BIVECTOR_MASK_0D = (3,)  # e+-

# CGA1D
BIVECTOR_MASK_1D = (3, 4, 5)  # e1+, e1-, e+-

# CGA2D
BIVECTOR_MASK_2D = (6, 7, 8, 9, 10, 15)  # e12, e1+, e1-, e2+, e2-, e+-

# CGA3D
BIVECTOR_MASK_3D = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
```

### GRADE_MASKS

各維度的 Grade 索引遮罩。

**CGA3D 範例**:
```python
GRADE_MASKS_3D = {
    0: [0],                                      # scalar
    1: [1, 2, 3, 4, 5],                          # vectors
    2: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],     # bivectors
    3: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25], # trivectors
    4: [26, 27, 28, 29, 30],                     # quadvectors
    5: [31],                                      # pseudoscalar
}
```

### PSEUDOSCALAR_SQUARE

各維度的 Pseudoscalar 平方值。

```python
PSEUDOSCALAR_SQUARE = {
    0: +1,   # CGA0D: I² = +1
    1: -1,   # CGA1D: I² = -1
    2: -1,   # CGA2D: I² = -1
    3: -1,   # CGA3D: I² = -1
    4: +1,   # CGA4D: I² = +1
    5: -1,   # CGA5D: I² = -1
}
```

### ROTOR_INDICES (用於 Structure Normalize)

各維度的 Rotor（純旋轉）分量索引，包含 scalar 和空間 bivector。

```python
# CGA1D: 無旋轉自由度
ROTOR_INDICES_1D = (0,)  # scalar only

# CGA2D: 1 個旋轉平面 (e12)
ROTOR_INDICES_2D = (0, 1)  # scalar, e12

# CGA3D: 3 個旋轉平面 (e12, e13, e23)
ROTOR_INDICES_3D = (0, 1, 2, 5)  # scalar, e12, e13, e23

# CGA4D: 6 個旋轉平面
ROTOR_INDICES_4D = (0, 1, 2, 3, 6, 7, 10)  # scalar, e12, e13, e14, e23, e24, e34

# CGA5D: 10 個旋轉平面
ROTOR_INDICES_5D = (0, 1, 2, 3, 4, 7, 8, 9, 12, 13, 16)
```

### TRANSLATION_PAIRS (用於 Structure Normalize)

各維度的平移分量配對索引 `(ei+, ei-)`，用於強制 Similitude 約束。

```python
# CGA1D
TRANSLATION_PAIRS_1D = [(1, 2)]  # (e1+, e1-)

# CGA2D
TRANSLATION_PAIRS_2D = [(2, 3), (4, 5)]  # (e1+, e1-), (e2+, e2-)

# CGA3D
TRANSLATION_PAIRS_3D = [(3, 4), (6, 7), (8, 9)]  # (e1+, e1-), (e2+, e2-), (e3+, e3-)

# CGA4D
TRANSLATION_PAIRS_4D = [(4, 5), (8, 9), (11, 12), (14, 15)]

# CGA5D
TRANSLATION_PAIRS_5D = [(5, 6), (10, 11), (14, 15), (17, 18), (20, 21)]
```

### DILATION_INDEX (用於 Structure Normalize)

各維度的 Dilation (e+-) 分量索引。

```python
DILATION_INDEX = {
    1: 3,   # CGA1D: e+- at index 3
    2: 6,   # CGA2D: e+- at index 6
    3: 10,  # CGA3D: e+- at index 10
    4: 19,  # CGA4D: e+- at index 19
    5: 34,  # CGA5D: e+- at index 34
}
```

---

## 張量形狀規範

### compose_even_versor

```
輸入:
  v1: Tensor[..., even_versor_count]
  v2: Tensor[..., even_versor_count]

輸出:
  result: Tensor[..., even_versor_count]
```

### compose_similitude

```
輸入:
  s1: Tensor[..., even_versor_count]  # 使用 EvenVersor 格式
  s2: Tensor[..., even_versor_count]

輸出:
  result: Tensor[..., even_versor_count]
```

### inner_product

```
輸入:
  a: Tensor[..., blade_count]
  b: Tensor[..., blade_count]

輸出:
  result: Tensor[..., 1]
```

### exp_bivector

```
輸入:
  B: Tensor[..., bivector_count]

輸出:
  even_versor: Tensor[..., even_versor_count]
```

### outer_product

```
輸入:
  a: Tensor[..., blade_count]
  b: Tensor[..., blade_count]

輸出:
  result: Tensor[..., blade_count]
```

### left_contraction / right_contraction

```
輸入:
  a: Tensor[..., blade_count]
  b: Tensor[..., blade_count]

輸出:
  result: Tensor[..., blade_count]
```

### grade_select

```
輸入:
  mv: Tensor[..., blade_count]
  grade: int

輸出:
  result: Tensor[..., grade_count]  # 該 grade 的分量數
```

### dual

```
輸入:
  mv: Tensor[..., blade_count]

輸出:
  result: Tensor[..., blade_count]
```

### normalize

```
輸入:
  mv: Tensor[..., blade_count]

輸出:
  result: Tensor[..., blade_count]
```

---

## Multivector 類別

### 屬性

```python
class Multivector:
    data: Tensor           # (..., blade_count)
    algebra: CGAAlgebraBase
    kind: str | None       # 類型標記
```

### Kind 值

| Kind | 含義 | 優化路由 |
|------|------|---------|
| None | 一般多向量 | geometric_product_full |
| 'versor' | 可逆多向量 | - |
| 'even_versor' | 偶數 Versor | compose_even_versor |
| 'similitude' | CGA Similitude | compose_similitude |
| 'bivector' | Grade-2 | exp_bivector |
| 'point' | CGA 點 | - |

### 運算子映射

| 運算子 | 操作 | 返回類型 |
|--------|------|---------|
| `*` | 幾何積 / compose | Multivector |
| `^` | 楔積 | Multivector |
| `\|` | 內積 | Tensor (標量) |
| `<<` | 左縮併 | Multivector |
| `>>` | 右縮併 | Multivector |
| `@` | 三明治積 | Multivector |
| `~` | 反向 | Multivector |
| `**` | 冪次 | Multivector |
| `+` | 加法 | Multivector |
| `-` | 減法 | Multivector |

---

## 狀態轉換

### EvenVersor Composition

```
EvenVersor × EvenVersor → EvenVersor
Similitude × Similitude → Similitude
Similitude × EvenVersor → EvenVersor (退化)

初始狀態: 兩個獨立變換
結束狀態: 組合後的單一變換
不變量: |V_result| ≈ |V1| × |V2| (對正規化 Versor)
```

### Exponential Map

```
Bivector → EvenVersor

初始狀態: 旋轉軸/平面表示
結束狀態: 旋轉 Versor
不變量: exp(0) = identity = (1, 0, 0, ...)
不變量: exp(B) × exp(-B) ≈ identity
```

---

## 驗證規則

### compose_even_versor 驗證

1. **單位元**: `compose(identity, V) == V`
2. **結合律**: `compose(compose(A, B), C) == compose(A, compose(B, C))`
3. **逆元**: `compose(V, reverse(V)) ≈ identity`

### inner_product 驗證

1. **對稱性**: `inner_product(a, b) == inner_product(b, a)`
2. **Null basis**: `inner_product(eo, einf) == -1`
3. **正交性**: 正交 blade 內積為 0

### exp_bivector 驗證

1. **零元**: `exp_bivector(0) == (1, 0, 0, ...)`
2. **小角度穩定性**: θ < 1e-10 時無 NaN/Inf
3. **逆運算**: `compose(exp(B), exp(-B)) ≈ identity`

### outer_product 驗證

1. **反對稱**: `a ^ b == -(b ^ a)`
2. **冪零**: `a ^ a == 0`
3. **Grade 加法**: `Grade(a ^ b) == Grade(a) + Grade(b)`

### contraction 驗證

1. **Grade 規則**: `Grade(a ⌋ b) == Grade(b) - Grade(a)`
2. **零條件**: `Grade(a) > Grade(b) → a ⌋ b == 0`

### dual 驗證

1. **Grade 翻轉**: `Grade(dual(a)) == max_grade - Grade(a)`
2. **雙對偶**: `dual(dual(a)) == ±a`

### normalize 驗證

1. **單位範數**: `|normalize(a)| == 1`
2. **零向量**: `normalize(0) == 0` (無 NaN)

---

## 統一 Layer 類別

### 命名對照表

| 移除的舊名稱 | 統一名稱 | 說明 |
|--------------|----------|------|
| `CGA{n}DCareLayer` | `CliffordTransformLayer` | EvenVersor sandwich product 變換層 |
| `RuntimeCGACareLayer` | `CliffordTransformLayer` | 運行時變換層 (n≥6) |
| `UPGC{n}DEncoder` | `CGAEncoder` | 歐氏座標 → CGA 點 |
| `UPGC{n}DDecoder` | `CGADecoder` | CGA 點 → 歐氏座標 |
| `CGA{n}DTransformPipeline` | `CGAPipeline` | 完整變換管線 |
| `get_care_layer()` | `get_transform_layer()` | 取得變換層的工廠方法 |

**注意**: 舊名稱將完全移除，不提供向後相容別名。

### CliffordTransformLayer

```python
class CliffordTransformLayer(nn.Module):
    """
    統一的 Clifford 代數變換層，執行 EvenVersor sandwich product。

    Attributes:
        dim: 歐氏維度 (0-5 硬編碼, 6+ 運行時)
        even_versor_count: EvenVersor 分量數
        point_count: 點分量數
    """
    def __init__(self, dim: int): ...
    def forward(self, versor: Tensor, point: Tensor) -> Tensor: ...
```

### CGAEncoder / CGADecoder

```python
class CGAEncoder(nn.Module):
    """
    統一的 UPGC 編碼器。

    Args:
        dim: 歐氏維度

    Input: (..., dim) 歐氏座標
    Output: (..., point_count) CGA 點表示
    """
    def forward(self, x: Tensor) -> Tensor: ...

class CGADecoder(nn.Module):
    """
    統一的 UPGC 解碼器。

    Input: (..., point_count) CGA 點表示
    Output: (..., dim) 歐氏座標
    """
    def forward(self, p: Tensor) -> Tensor: ...
```

### CGAPipeline

```python
class CGAPipeline(nn.Module):
    """
    統一的變換管線：Encoder → Transform → Decoder。

    Args:
        dim: 歐氏維度

    Input:
        versor: (..., even_versor_count)
        x: (..., dim) 歐氏座標
    Output: (..., dim) 變換後的歐氏座標
    """
    def forward(self, versor: Tensor, x: Tensor) -> Tensor: ...
```
