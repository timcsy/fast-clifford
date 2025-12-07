# 研究報告：CGA4D 與 CGA5D 代數規則

**日期**: 2025-12-07
**分支**: `003-cga4d-cga5d`

## 1. CGA4D Cl(5,1) 代數規則

### 決策：採用標準 Clifford 代數定義

**理由**：clifford 函式庫是業界標準的幾何代數實作，已在 CGA1D/2D/3D 中驗證可靠。

**代數規格**:

| 屬性 | 值 |
|------|-----|
| Clifford 代數 | Cl(5,1) |
| 歐幾里得維度 | 4 |
| 總維度 | 6 (4 + 2 conformal) |
| Blade 數量 | 64 (2^6) |
| 簽名 | (+,+,+,+,+,-) |
| 基底 | e1, e2, e3, e4, e+, e- |

### Grade 分佈（組合數 C(6,k)）

| Grade | Blade 數量 | 說明 |
|-------|-----------|------|
| 0 | 1 | 標量 |
| 1 | 6 | 向量（e1, e2, e3, e4, e+, e-） |
| 2 | 15 | 二向量 |
| 3 | 20 | 三向量 |
| 4 | 15 | 四向量 |
| 5 | 6 | 五向量 |
| 6 | 1 | 偽標量 |

### UPGC Point 定義

- **分量數**：6（Grade 1）
- **索引**：(1, 2, 3, 4, 5, 6) 對應 (e1, e2, e3, e4, e+, e-)
- **編碼公式**：X = n_o + x + 0.5|x|² n_inf
  - 其中 n_o = (e- - e+)/2, n_inf = e+ + e-

### Motor 定義

- **分量數**：31（Grade 0 + Grade 2 + Grade 4）
- **Grade 0**：1 分量（標量）
- **Grade 2**：15 分量（所有二向量）
- **Grade 4**：15 分量（所有四向量）

---

## 2. CGA5D Cl(6,1) 代數規則

### 決策：採用標準 Clifford 代數定義

**理由**：與 CGA4D 相同，延續使用 clifford 函式庫。

**代數規格**:

| 屬性 | 值 |
|------|-----|
| Clifford 代數 | Cl(6,1) |
| 歐幾里得維度 | 5 |
| 總維度 | 7 (5 + 2 conformal) |
| Blade 數量 | 128 (2^7) |
| 簽名 | (+,+,+,+,+,+,-) |
| 基底 | e1, e2, e3, e4, e5, e+, e- |

### Grade 分佈（組合數 C(7,k)）

| Grade | Blade 數量 | 說明 |
|-------|-----------|------|
| 0 | 1 | 標量 |
| 1 | 7 | 向量（e1, e2, e3, e4, e5, e+, e-） |
| 2 | 21 | 二向量 |
| 3 | 35 | 三向量 |
| 4 | 35 | 四向量 |
| 5 | 21 | 五向量 |
| 6 | 7 | 六向量 |
| 7 | 1 | 偽標量 |

### UPGC Point 定義

- **分量數**：7（Grade 1）
- **索引**：(1, 2, 3, 4, 5, 6, 7) 對應 (e1, e2, e3, e4, e5, e+, e-)

### Motor 定義

- **分量數**：64（Grade 0 + Grade 2 + Grade 4 + Grade 6）
- **Grade 0**：1 分量
- **Grade 2**：21 分量
- **Grade 4**：35 分量
- **Grade 6**：7 分量
- **總計**：1 + 21 + 35 + 7 = **64 分量**

---

## 3. 稀疏性驗證

### 決策：Motor × Point × Motor_rev 輸出僅有 Grade 1

**理由**：三明治積的數學性質保證 Grade 保持。

**驗證方法**：
```python
from clifford import Cl, conformalize

# CGA4D
G4, _ = Cl(4)
layout, blades, stuff = conformalize(G4)
eo, einf = stuff['eo'], stuff['einf']

# 建立測試 motor (rotor + translator)
motor = 1 + 0.1 * blades['e12'] + 0.05 * blades['e1'] * einf

# 建立測試 point
point = stuff['up'](blades['e1'] + 2*blades['e2'] + 3*blades['e3'] + 4*blades['e4'])

# 三明治積
result = motor * point * ~motor

# 驗證：結果僅有 Grade 1 分量
assert all(result(g) == 0 for g in [0, 2, 3, 4, 5, 6])
```

**結論**：稀疏性假設成立，與 CGA1D/2D/3D 一致。

---

## 4. 稀疏乘法次數計算

### CGA4D 稀疏三明治積

- Motor × Point：31 × 6 = 186 次乘法（每個 motor 分量與每個 point 分量）
- 中間結果 × Motor_rev：結果分量 × 31
- **估計總乘法**：31 × 6 × 31 ≈ **5,766 次**
- **完整計算**：64 × 64 × 64 = 262,144 次
- **減少比例**：97.8%

### CGA5D 稀疏三明治積

- Motor × Point：64 × 7 = 448 次乘法
- 中間結果 × Motor_rev：結果分量 × 64
- **估計總乘法**：64 × 7 × 64 ≈ **28,672 次**
- **完整計算**：128 × 128 × 128 = 2,097,152 次
- **減少比例**：98.6%

---

## 5. 生成器擴展驗證

### 決策：修改 cga_factory.py 支援維度 4 和 5

**理由**：現有 `create_cga_algebra()` 函數限制 euclidean_dim 在 [1, 3] 範圍，需擴展。

**變更內容**：
```python
# 原始
if euclidean_dim < 1 or euclidean_dim > 3:
    raise ValueError(...)

# 修改為
if euclidean_dim < 1 or euclidean_dim > 5:
    raise ValueError(...)
```

**新增常數**：
```python
# CGA4D Cl(5,1) 參數
CGA4D_EUCLIDEAN_DIM = 4
CGA4D_BLADE_COUNT = 64
CGA4D_SIGNATURE = (1, 1, 1, 1, 1, -1)

# CGA5D Cl(6,1) 參數
CGA5D_EUCLIDEAN_DIM = 5
CGA5D_BLADE_COUNT = 128
CGA5D_SIGNATURE = (1, 1, 1, 1, 1, 1, -1)
```

---

## 6. 效能預估

### CGA4D 預估效能

基於 CGA3D 的效能數據推算：

| 指標 | CGA3D | CGA4D 預估 |
|------|-------|-----------|
| Blade 數量 | 32 | 64 (2x) |
| Motor 分量 | 16 | 31 (~2x) |
| 三明治積乘法 | 800 | ~5,800 (7x) |
| CPU 吞吐量 | 2.7M 點/秒 | ~500K 點/秒 |

### CGA5D 預估效能

| 指標 | CGA3D | CGA5D 預估 |
|------|-------|-----------|
| Blade 數量 | 32 | 128 (4x) |
| Motor 分量 | 16 | 64 (4x) |
| 三明治積乘法 | 800 | ~28,700 (36x) |
| CPU 吞吐量 | 2.7M 點/秒 | ~100K 點/秒 |

**結論**：CGA5D 的效能會明顯低於 CGA4D，但對於高維度應用仍可接受。

---

## 7. Reverse 符號規則

### 公式

```
reverse_sign(grade) = (-1)^(grade * (grade - 1) / 2)
```

### CGA4D Reverse 符號

| Grade | 指數 | 符號 |
|-------|------|------|
| 0 | 0 | +1 |
| 1 | 0 | +1 |
| 2 | 1 | -1 |
| 3 | 3 | -1 |
| 4 | 6 | +1 |
| 5 | 10 | +1 |
| 6 | 15 | -1 |

### CGA5D Reverse 符號

| Grade | 指數 | 符號 |
|-------|------|------|
| 0 | 0 | +1 |
| 1 | 0 | +1 |
| 2 | 1 | -1 |
| 3 | 3 | -1 |
| 4 | 6 | +1 |
| 5 | 10 | +1 |
| 6 | 15 | -1 |
| 7 | 21 | -1 |

---

## 8. 結論與考量的替代方案

### 採用的方案

1. **擴展現有 cga_factory.py**：最小修改，重用驗證過的邏輯
2. **遵循現有模組結構**：algebra.py + functional.py + layers.py
3. **使用 clifford 函式庫生成規則**：避免手動計算錯誤

### 考量但不採用的替代方案

| 替代方案 | 不採用原因 |
|----------|-----------|
| 手動編寫 functional.py | CGA5D 有 128 blades，手動編寫容易出錯 |
| 使用 Cayley 表查詢 | 違反憲法原則 IV（硬編碼代數展開） |
| 建立新的生成器 | 現有生成器已驗證可靠，無需重寫 |
