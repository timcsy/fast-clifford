# Research Notes: CGA Extended Operations

## 1. Motor Composition

### 數學定義

Motor Composition 是兩個馬達的幾何積：
```
M_result = M1 * M2
```

**馬達結構**：
- 馬達是偶數 Grade 多向量：Grade 0 + Grade 2 + Grade 4 + ...
- CGA3D: 16 分量 (1 + 10 + 5)
- CGA2D: 7 分量 (1 + 5 + 1)
- CGA1D: 4 分量 (1 + 3)
- CGA0D: 2 分量 (1 + 1)

### 稀疏性分析

Motor × Motor 的結果仍為偶數 Grade：
- 偶數 × 偶數 = 偶數

因此 `motor_compose` 可以使用稀疏表示：
- 輸入：兩個 `motor_count` 分量張量
- 輸出：一個 `motor_count` 分量張量

### 實作決策

**Decision**: 使用 codegen 生成 `motor_compose_sparse` 函式，僅計算馬達分量。

**Rationale**:
- Motor × Motor 只會產生偶數 Grade 分量
- 不需要計算完整 blade_count 分量
- 減少約 50% 計算量

**Alternatives considered**:
1. 使用完整幾何積後提取馬達分量 - 效率較低
2. 手動編碼 - 容易出錯，難以維護

### 稀疏項計算

對於 CGA3D (16 motor 分量)：
- 輸入 1: 16 分量
- 輸入 2: 16 分量
- 有效乘法數: ~150-200 次（依據 Cayley 表非零項）
- 理論最大: 16 × 16 = 256 次

---

## 2. Geometric Inner Product (Metric Inner Product)

### 數學定義

幾何內積是幾何積的 Grade 0 分量：
```
<A · B>_0 = Σ a[i] * b[i] * sign[i]
```

其中 `sign[i]` 來自 CGA 度規符號。

### CGA 度規符號

CGA(n) 的度規符號為 Cl(n+1, 1)：
- 前 n+1 個基底：+1 (e1, e2, ..., en, e+)
- 最後 1 個基底：-1 (e-)

**重要**：高階 blade 的符號由基底平方的乘積決定。

### Blade 平方符號表

對於 CGA3D Cl(4,1)：
| Blade | 符號 | 說明 |
|-------|------|------|
| 1 (scalar) | +1 | 1² = 1 |
| e1 | +1 | e1² = +1 |
| e2 | +1 | e2² = +1 |
| e3 | +1 | e3² = +1 |
| e+ | +1 | e+² = +1 |
| e- | -1 | e-² = -1 |
| e12 | -1 | (e1*e2)² = -e1²*e2² = -1 |
| e13 | -1 | -1 |
| e1+ | -1 | -1 |
| e1- | +1 | e1²*e-² = (+1)*(-1) = -1, (e1-)² = -e1²*e-² = +1 |
| ... | ... | ... |

### 實作決策

**Decision**: 硬編碼符號融合優化，直接計算 `sum(a[i] * b[i] * sign[i])`

**Rationale**:
- 避免兩步計算（先乘符號再累加）
- 編譯器可更好地優化
- ONNX 圖更簡潔

**Alternatives considered**:
1. 先做完整幾何積再取 Grade 0 - 計算量大 10 倍以上
2. 分步計算（vec * sign，再 sum）- 額外記憶體和操作

### 符號預計算

每個維度的符號向量需預先計算並儲存為常數：
```python
# CGA3D 範例（需從 clifford 驗證）
INNER_PRODUCT_SIGNS = (
    1,   # scalar
    1,   # e1
    1,   # e2
    1,   # e3
    1,   # e+
    -1,  # e-
    -1,  # e12
    ...  # 32 個符號
)
```

---

## 3. Exponential Map

### 數學定義

Bivector 的指數映射：
```
exp(B) = cos(θ) + sin(θ)/θ * B
```

其中 θ 滿足：
```
θ² = -B²  (對於旋轉 bivector)
```

### 數值穩定性問題

當 θ → 0 時，`sin(θ)/θ` 需要特殊處理：
- 直接計算會得到 0/0 = NaN
- 需要使用 Taylor 展開或 sinc 函數

### 實作決策

**Decision**: 使用 `torch.sinc(θ/π)` 處理數值穩定性

**Rationale**:
- PyTorch 的 `sinc(x) = sin(πx)/(πx)` 對 x=0 返回 1
- 轉換：`sin(θ)/θ = sinc(θ/π)`
- 無需自定義 Taylor 展開

**Alternatives considered**:
1. 自定義 Taylor 展開 - 需要 `torch.where`，可能產生 ONNX If 節點
2. 使用 `torch.where(θ < eps, 1.0, sin(θ)/θ)` - 可能有 ONNX 相容性問題

### Bivector 平方計算

需要新增 `bivector_squared_scalar` 輔助函式：
- 輸入：Bivector (Grade 2 分量)
- 輸出：標量 (B² 的 Grade 0 分量)

這也是稀疏計算：Bivector × Bivector 的標量部分。

### ONNX 相容性驗證

關鍵檢查點：
1. `torch.sinc` 是否可匯出 ONNX - ✅ 可以
2. `torch.sqrt` 對負數 - 需 clamp 處理
3. `torch.cos` - ✅ 可以

---

## 4. codegen 擴展策略

### 新增函式

需要在 `sparse_analysis.py` 新增：
1. `get_motor_compose_terms(dim)` - Motor × Motor 稀疏項
2. `get_inner_product_signs(dim)` - 度規符號向量
3. `get_bivector_squared_terms(dim)` - Bivector² 標量項

需要在 `generate.py` 新增：
1. `_generate_motor_compose_sparse()` - 生成 motor_compose
2. `_generate_inner_product_full()` - 生成 inner_product
3. `_generate_exp_bivector()` - 生成 exp_bivector
4. `_generate_bivector_squared_scalar()` - 輔助函式

### 各維度 Bivector 分量數

| 維度 | Bivector 分量 | Motor 分量 |
|------|--------------|-----------|
| CGA0D | 1 (e+-) | 2 |
| CGA1D | 3 | 4 |
| CGA2D | 6 | 7 |
| CGA3D | 10 | 16 |
| CGA4D | 15 | 31 |
| CGA5D | 21 | 64 |

---

## 5. 運行時實作策略

### RuntimeCGAAlgebra 擴展

對於 n≥6，使用完整幾何積實作：

```python
def motor_compose(self, m1, m2):
    m1_full = self._embed_motor(m1)  # 展開至完整 blade
    m2_full = self._embed_motor(m2)
    result_full = self.geometric_product_full(m1_full, m2_full)
    return self._extract_motor(result_full)  # 提取馬達分量

def inner_product(self, a, b):
    return (a * b * self._inner_product_signs).sum(dim=-1, keepdim=True)

def exp_bivector(self, B):
    B_full = self._embed_bivector(B)
    B_sq = self.geometric_product_full(B_full, B_full)
    theta_sq = torch.clamp(-B_sq[..., 0], min=1e-12)
    theta = torch.sqrt(theta_sq)
    cos_theta = torch.cos(theta)
    sinc_theta = torch.sinc(theta / torch.pi)
    result_full = cos_theta.unsqueeze(-1) + sinc_theta.unsqueeze(-1) * B_full
    return self._extract_motor(result_full)
```

---

## 6. 測試對照策略

### clifford 庫對照

```python
from clifford import conformalize, Cl

# Motor Composition
G, blades, stuff = conformalize(Cl(3)[0])
M1_cliff = ...  # clifford 馬達
M2_cliff = ...
result_cliff = M1_cliff * M2_cliff  # 幾何積

# Inner Product
inner_cliff = float((A_cliff * B_cliff).value[0])  # Grade 0

# Exponential Map
from clifford import exp
rotor_cliff = exp(B_cliff)
```

### 數值容差

- float32: atol=1e-6
- float64: atol=1e-10

---

## 7. 結論

所有 NEEDS CLARIFICATION 已解決：

| 項目 | 解決方案 |
|------|---------|
| Motor Composition 稀疏性 | 使用 motor × motor 稀疏計算 |
| Inner Product 度規符號 | 預計算符號向量，符號融合優化 |
| Exponential Map 數值穩定 | 使用 torch.sinc 處理 θ→0 |
| codegen 擴展 | 新增 3 個分析函式 + 4 個生成函式 |
| 運行時實作 | 使用完整幾何積 + embed/extract |
