# 資料模型：CGA2D 與 CGA1D

**日期**: 2025-12-07
**功能**: 002-cga-2d-1d

## 實體定義

本文件定義 CGA2D (Cl(3,1)) 和 CGA1D (Cl(2,1)) 的資料模型。

---

## 1. CGA2D Cl(3,1) 資料模型

### 1.1 Blade（基底刃）

**描述**: Clifford 代數 Cl(3,1) 中的基底多重向量元素

| 欄位 | 型別 | 說明 | 範例 |
|------|------|------|------|
| index | int | 在 16 維空間中的索引 (0-15) | `0`, `5`, `15` |
| grade | int | 階數 (0-4) | `0`=scalar, `1`=vector |
| basis_vectors | tuple[int] | 組成的基底向量索引 | `()`, `(0,)`, `(0,1,2,3)` |
| name | str | 人類可讀名稱 | `"1"`, `"e1"`, `"e12+-"` |

**Grade 分佈**:
```
Grade 0: 1 個 (scalar)     → index 0
Grade 1: 4 個 (vector)     → index 1-4
Grade 2: 6 個 (bivector)   → index 5-10
Grade 3: 4 個 (trivector)  → index 11-14
Grade 4: 1 個 (quadvector) → index 15
```

**完整 Blade 索引表**（按 clifford 庫標準順序）:

| Index | Grade | Name | Basis Vectors |
|-------|-------|------|---------------|
| 0 | 0 | 1 | () |
| 1 | 1 | e1 | (0,) |
| 2 | 1 | e2 | (1,) |
| 3 | 1 | e+ | (2,) |
| 4 | 1 | e- | (3,) |
| 5 | 2 | e12 | (0,1) |
| 6 | 2 | e1+ | (0,2) |
| 7 | 2 | e1- | (0,3) |
| 8 | 2 | e2+ | (1,2) |
| 9 | 2 | e2- | (1,3) |
| 10 | 2 | e+- | (2,3) |
| 11 | 3 | e12+ | (0,1,2) |
| 12 | 3 | e12- | (0,1,3) |
| 13 | 3 | e1+- | (0,2,3) |
| 14 | 3 | e2+- | (1,2,3) |
| 15 | 4 | e12+- | (0,1,2,3) |

### 1.2 BasisVector（基底向量）

**描述**: CGA2D 的 4 個基底向量

| 索引 | 名稱 | 度規 | 說明 |
|------|------|------|------|
| 0 | e1 | +1 | 歐幾里得 x 軸 |
| 1 | e2 | +1 | 歐幾里得 y 軸 |
| 2 | e+ | +1 | 正簽名額外維度 |
| 3 | e- | -1 | 負簽名額外維度 |

### 1.3 NullBasis（Null 基底）

**描述**: CGA2D 特有的 null 向量定義

| 名稱 | 定義 | 性質 | 用途 |
|------|------|------|------|
| $n_o$ | $\frac{1}{2}(e_- - e_+)$ | $n_o^2 = 0$ | 原點表示 |
| $n_\infty$ | $e_- + e_+$ | $n_\infty^2 = 0$ | 無窮遠點表示 |

**約定**: $n_o \cdot n_\infty = -1$

### 1.4 SparsityMask（稀疏性遮罩）

**描述**: 特定類型 multivector 的非零 blade 索引

| 類型 | 非零索引 | 數量 | 說明 |
|------|----------|------|------|
| UPGC_POINT_2D | [1, 2, 3, 4] | 4 | Grade 1 only |
| MOTOR_2D | [0, 5, 6, 7, 8, 9, 10, 15] | 8 | Grade 0, 2, 4 |
| RESULT_2D | [1, 2, 3, 4] | 4 | Grade 1 only |

### 1.5 ReverseSign（Reverse 符號）

**描述**: Reverse 操作對每個 blade 的符號影響

| Grade | 符號公式 | 結果 |
|-------|---------|------|
| 0 | $(-1)^{0 \times -1 / 2} = 1$ | +1 |
| 1 | $(-1)^{1 \times 0 / 2} = 1$ | +1 |
| 2 | $(-1)^{2 \times 1 / 2} = -1$ | -1 |
| 3 | $(-1)^{3 \times 2 / 2} = -1$ | -1 |
| 4 | $(-1)^{4 \times 3 / 2} = 1$ | +1 |

---

## 2. CGA1D Cl(2,1) 資料模型

### 2.1 Blade（基底刃）

**描述**: Clifford 代數 Cl(2,1) 中的基底多重向量元素

| 欄位 | 型別 | 說明 | 範例 |
|------|------|------|------|
| index | int | 在 8 維空間中的索引 (0-7) | `0`, `4`, `7` |
| grade | int | 階數 (0-3) | `0`=scalar, `1`=vector |
| basis_vectors | tuple[int] | 組成的基底向量索引 | `()`, `(0,)`, `(0,1,2)` |
| name | str | 人類可讀名稱 | `"1"`, `"e1"`, `"e1+-"` |

**Grade 分佈**:
```
Grade 0: 1 個 (scalar)     → index 0
Grade 1: 3 個 (vector)     → index 1-3
Grade 2: 3 個 (bivector)   → index 4-6
Grade 3: 1 個 (trivector)  → index 7
```

**完整 Blade 索引表**（按 clifford 庫標準順序）:

| Index | Grade | Name | Basis Vectors |
|-------|-------|------|---------------|
| 0 | 0 | 1 | () |
| 1 | 1 | e1 | (0,) |
| 2 | 1 | e+ | (1,) |
| 3 | 1 | e- | (2,) |
| 4 | 2 | e1+ | (0,1) |
| 5 | 2 | e1- | (0,2) |
| 6 | 2 | e+- | (1,2) |
| 7 | 3 | e1+- | (0,1,2) |

### 2.2 BasisVector（基底向量）

**描述**: CGA1D 的 3 個基底向量

| 索引 | 名稱 | 度規 | 說明 |
|------|------|------|------|
| 0 | e1 | +1 | 歐幾里得 x 軸 |
| 1 | e+ | +1 | 正簽名額外維度 |
| 2 | e- | -1 | 負簽名額外維度 |

### 2.3 NullBasis（Null 基底）

**描述**: CGA1D 特有的 null 向量定義

| 名稱 | 定義 | 性質 | 用途 |
|------|------|------|------|
| $n_o$ | $\frac{1}{2}(e_- - e_+)$ | $n_o^2 = 0$ | 原點表示 |
| $n_\infty$ | $e_- + e_+$ | $n_\infty^2 = 0$ | 無窮遠點表示 |

**約定**: $n_o \cdot n_\infty = -1$

### 2.4 SparsityMask（稀疏性遮罩）

**描述**: 特定類型 multivector 的非零 blade 索引

| 類型 | 非零索引 | 數量 | 說明 |
|------|----------|------|------|
| UPGC_POINT_1D | [1, 2, 3] | 3 | Grade 1 only |
| MOTOR_1D | [0, 4, 5, 6] | 4 | Grade 0, 2 (無 Grade 4) |
| RESULT_1D | [1, 2, 3] | 3 | Grade 1 only |

**注意**: CGA1D 沒有 Grade 4（空間維度為 3，最高 grade 為 3），因此 Motor 只包含 Grade 0 和 Grade 2。

### 2.5 ReverseSign（Reverse 符號）

**描述**: Reverse 操作對每個 blade 的符號影響

| Grade | 符號公式 | 結果 |
|-------|---------|------|
| 0 | $(-1)^{0 \times -1 / 2} = 1$ | +1 |
| 1 | $(-1)^{1 \times 0 / 2} = 1$ | +1 |
| 2 | $(-1)^{2 \times 1 / 2} = -1$ | -1 |
| 3 | $(-1)^{3 \times 2 / 2} = -1$ | -1 |

---

## 3. 張量表示

### 3.1 CGA2D 張量

#### Multivector 張量
```
形狀: (..., 16)
dtype: float32 (強制)
最後一維對應 16 個 blade 係數
```

#### UPGC 2D Point 稀疏張量
```
形狀: (..., 4)
dtype: float32
對應: [e1_coeff, e2_coeff, e+_coeff, e-_coeff]
```

#### 2D Motor 稀疏張量
```
形狀: (..., 8)
dtype: float32
對應: [scalar, e12, e1+, e1-, e2+, e2-, e+-, e12+-]
```

### 3.2 CGA1D 張量

#### Multivector 張量
```
形狀: (..., 8)
dtype: float32 (強制)
最後一維對應 8 個 blade 係數
```

#### UPGC 1D Point 稀疏張量
```
形狀: (..., 3)
dtype: float32
對應: [e1_coeff, e+_coeff, e-_coeff]
```

#### 1D Motor 稀疏張量
```
形狀: (..., 4)
dtype: float32
對應: [scalar, e1+, e1-, e+-]
```

---

## 4. 關係圖

### CGA2D 關係
```
BasisVector (4)
    │
    ├──組成──▶ Blade (16)
    │              │
    │              ├──左運算元──┐
    │              │            ▼
    │              └──右運算元──▶ GeometricProductRule (256)
    │
    └──定義──▶ NullBasis (2)
                   │
                   └──用於──▶ UPGC 2D Point 編碼

SparsityMask
    │
    ├── UPGC_POINT_2D: 輸入稀疏性 (4)
    ├── MOTOR_2D: 變換算子稀疏性 (8)
    └── RESULT_2D: 輸出稀疏性 (4)
```

### CGA1D 關係
```
BasisVector (3)
    │
    ├──組成──▶ Blade (8)
    │              │
    │              ├──左運算元──┐
    │              │            ▼
    │              └──右運算元──▶ GeometricProductRule (64)
    │
    └──定義──▶ NullBasis (2)
                   │
                   └──用於──▶ UPGC 1D Point 編碼

SparsityMask
    │
    ├── UPGC_POINT_1D: 輸入稀疏性 (3)
    ├── MOTOR_1D: 變換算子稀疏性 (4)
    └── RESULT_1D: 輸出稀疏性 (3)
```

---

## 5. 狀態轉換

本功能不涉及狀態轉換（純計算規則定義）。

---

## 6. 驗證規則

### CGA2D 驗證
- index 必須在 [0, 15] 範圍內
- grade 必須在 [0, 4] 範圍內
- basis_vectors 元素必須在 [0, 3] 範圍內且無重複
- UPGC_POINT_2D 必須只包含 Grade 1 索引 [1, 2, 3, 4]
- MOTOR_2D 必須只包含 Grade 0, 2, 4 索引

### CGA1D 驗證
- index 必須在 [0, 7] 範圍內
- grade 必須在 [0, 3] 範圍內
- basis_vectors 元素必須在 [0, 2] 範圍內且無重複
- UPGC_POINT_1D 必須只包含 Grade 1 索引 [1, 2, 3]
- MOTOR_1D 必須只包含 Grade 0, 2 索引（無 Grade 4）

---

## 7. 與 CGA3D 的對比

| 屬性 | CGA3D Cl(4,1) | CGA2D Cl(3,1) | CGA1D Cl(2,1) |
|------|---------------|---------------|---------------|
| 簽名 | (+,+,+,+,-) | (+,+,+,-) | (+,+,-) |
| 總 Blade 數 | 32 | 16 | 8 |
| Grade 0 | 1 | 1 | 1 |
| Grade 1 | 5 | 4 | 3 |
| Grade 2 | 10 | 6 | 3 |
| Grade 3 | 10 | 4 | 1 |
| Grade 4 | 5 | 1 | - |
| Grade 5 | 1 | - | - |
| UPGC Point | 5 分量 | 4 分量 | 3 分量 |
| Motor | 16 分量 | 8 分量 | 4 分量 |
| 乘法規則數 | 1024 | 256 | 64 |
