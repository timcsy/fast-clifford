# Data Model: CGA Extended Operations

## 核心實體

### Motor (馬達)

偶數 Grade 多向量，用於表示剛體變換（旋轉 + 平移）。

| 維度 | Motor 分量數 | Grade 組成 | 稀疏索引 |
|------|-------------|-----------|---------|
| CGA0D | 2 | G0(1) + G2(1) | (0, 3) |
| CGA1D | 4 | G0(1) + G2(3) | (0, 3, 4, 5) |
| CGA2D | 7 | G0(1) + G2(5) + G4(1) | (0, 6, 7, 8, 9, 10, 15) |
| CGA3D | 16 | G0(1) + G2(10) + G4(5) | 見下表 |
| CGA4D | 31 | G0(1) + G2(15) + G4(15) | ... |
| CGA5D | 64 | G0(1) + G2(21) + G4(35) + G6(7) | ... |

**CGA3D Motor 索引表**:
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

完整 Clifford 代數元素，用於 `inner_product` 的輸入。

| 維度 | Blade 總數 |
|------|-----------|
| CGA0D | 4 |
| CGA1D | 8 |
| CGA2D | 16 |
| CGA3D | 32 |
| CGA4D | 64 |
| CGA5D | 128 |

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

---

## 張量形狀規範

### motor_compose

```
輸入:
  m1: Tensor[..., motor_count]
  m2: Tensor[..., motor_count]

輸出:
  result: Tensor[..., motor_count]
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
  motor: Tensor[..., motor_count]
```

---

## 狀態轉換

### Motor Composition

```
Motor × Motor → Motor

初始狀態: 兩個獨立變換馬達
結束狀態: 組合後的單一馬達
不變量: |M_result| ≈ |M1| × |M2| (對正規化馬達)
```

### Exponential Map

```
Bivector → Motor

初始狀態: 旋轉軸/平面表示
結束狀態: 旋轉馬達
不變量: exp(0) = identity motor = (1, 0, 0, ...)
不變量: exp(B) × exp(-B) ≈ identity
```

---

## 驗證規則

### motor_compose 驗證

1. **單位元**: `motor_compose(identity, M) == M`
2. **結合律**: `motor_compose(motor_compose(A, B), C) == motor_compose(A, motor_compose(B, C))`
3. **逆元**: `motor_compose(M, reverse(M)) ≈ identity`

### inner_product 驗證

1. **對稱性**: `inner_product(a, b) == inner_product(b, a)`
2. **Null basis**: `inner_product(eo, einf) == -1`
3. **正交性**: 正交 blade 內積為 0

### exp_bivector 驗證

1. **零元**: `exp_bivector(0) == (1, 0, 0, ...)`
2. **小角度穩定性**: θ < 1e-10 時無 NaN/Inf
3. **逆運算**: `motor_compose(exp(B), exp(-B)) ≈ identity`

---

## 統一 Layer 類別

### 命名對照表

| 移除的舊名稱 | 統一名稱 | 說明 |
|--------------|----------|------|
| `CGA{n}DCareLayer` | `CGATransformLayer` | Motor sandwich product 變換層 |
| `RuntimeCGACareLayer` | `CGATransformLayer` | 運行時變換層 (n≥6) |
| `UPGC{n}DEncoder` | `CGAEncoder` | 歐氏座標 → CGA 點 |
| `UPGC{n}DDecoder` | `CGADecoder` | CGA 點 → 歐氏座標 |
| `CGA{n}DTransformPipeline` | `CGAPipeline` | 完整變換管線 |
| `get_care_layer()` | `get_transform_layer()` | 取得變換層的工廠方法 |

**注意**: 舊名稱將完全移除，不提供向後相容別名。

### CGATransformLayer

```python
class CGATransformLayer(nn.Module):
    """
    統一的 CGA 變換層，執行 Motor sandwich product。

    Attributes:
        dim: 歐氏維度 (0-5 硬編碼, 6+ 運行時)
        motor_count: 馬達分量數
        point_count: 點分量數
    """
    def __init__(self, dim: int): ...
    def forward(self, motor: Tensor, point: Tensor) -> Tensor: ...
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
        motor: (..., motor_count)
        x: (..., dim) 歐氏座標
    Output: (..., dim) 變換後的歐氏座標
    """
    def forward(self, motor: Tensor, x: Tensor) -> Tensor: ...
```

