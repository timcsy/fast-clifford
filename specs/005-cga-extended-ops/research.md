# Research Notes: Clifford Algebra Extended Operations

**Date**: 2025-12-08
**Branch**: `005-cga-extended-ops`

## 摘要

本文件記錄 005-cga-extended-ops 功能的研究與設計決策，包括命名重構（Motor → EvenVersor）、新增 Similitude 加速、統一 API 設計等。

---

## 1. EvenVersor vs Motor 命名

### 決策
採用 **EvenVersor** 取代 Motor，作為通用 Clifford Algebra 術語。

### 理由
- **Motor** 是 CGA 特定術語，在一般 Clifford Algebra 文獻中較少使用
- **EvenVersor** (偶數 Versor) 是數學上更精確的名稱
- 保持與 Geometric Algebra 學術社群的術語一致性
- 建立清晰的類型層級：`Versor > EvenVersor > Similitude`

### 考慮的替代方案
| 替代方案 | 評估 | 拒絕原因 |
|---------|------|---------|
| Rotor | 僅表示旋轉 | 不包含平移，語義不完整 |
| Spinor | 物理學術語 | 可能與量子力學 Spinor 混淆 |
| Motor | 原方案 | 過於 CGA 特定 |

---

## 2. Similitude 設計

### 決策
新增 **Similitude** 作為 EvenVersor 的 CGA 專用子類別，排除 transversion。

### 數學定義
```
Similitude = 平移 × 旋轉 × 縮放
          = Translation × Rotation × Dilation
          ⊂ EvenVersor (排除 transversion 相關分量)
```

### 稀疏性分析

| 維度 | EvenVersor 分量 | Similitude 分量 | 減少比例 |
|------|----------------|-----------------|----------|
| CGA0D | 2 | 2 | 0% |
| CGA1D | 4 | 3 | 25% |
| CGA2D | 8 | 5 | 37.5% |
| CGA3D | 16 | 10 | 37.5% |
| CGA4D | 32 | 19 | 40.6% |
| CGA5D | 64 | 36 | 43.8% |

### 效能預期
- **compose_similitude** vs **compose_even_versor**: 30-50% 更快
- **sandwich_product_similitude** vs **sandwich_product_even_versor**: 30-50% 更快

### 考慮的替代方案
| 替代方案 | 評估 | 拒絕原因 |
|---------|------|---------|
| 不區分 | 統一使用 EvenVersor | 犧牲效能 |
| Rigid Motion | 僅旋轉+平移 | 不包含縮放，應用範圍較窄 |

---

## 3. EvenVersor Composition

### 數學定義

EvenVersor Composition 是兩個偶數 Versor 的幾何積：
```
V_result = V1 * V2
```

**EvenVersor 結構**：偶數 Grade 多向量：Grade 0 + Grade 2 + Grade 4 + ...

| 維度 | EvenVersor 分量 | Grade 組成 |
|------|----------------|-----------|
| CGA0D | 2 | G0(1) + G2(1) |
| CGA1D | 4 | G0(1) + G2(3) |
| CGA2D | 8 | G0(1) + G2(6) + G4(1) |
| CGA3D | 16 | G0(1) + G2(10) + G4(5) |
| CGA4D | 32 | G0(1) + G2(15) + G4(15) + G6(1) |
| CGA5D | 64 | G0(1) + G2(21) + G4(35) + G6(7) |

### 稀疏性分析
EvenVersor × EvenVersor 的結果仍為偶數 Grade（偶數 × 偶數 = 偶數）。

### 實作決策
**Decision**: 使用 codegen 生成 `compose_even_versor` 和 `compose_similitude` 函式。

**Rationale**:
- EvenVersor × EvenVersor 只產生偶數 Grade 分量
- Similitude × Similitude 更稀疏
- 減少約 50% 計算量（相比完整幾何積）

---

## 4. Geometric Inner Product (Metric Inner Product)

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

### Blade 平方符號表 (CGA3D Cl(4,1))

| Blade | 符號 | 說明 |
|-------|------|------|
| 1 (scalar) | +1 | 1² = 1 |
| e1, e2, e3 | +1 | ei² = +1 |
| e+ | +1 | e+² = +1 |
| e- | -1 | e-² = -1 |
| e12, e13, e23 | -1 | (ei*ej)² = -1 |
| e1-, e2-, e3- | +1 | ei² * e-² = (+1)*(-1) → (ei-)² = +1 |
| e+- | +1 | e+² * e-² = -1 → (e+-)² = +1 |

### 實作決策
**Decision**: 硬編碼符號融合優化，直接計算 `sum(a[i] * b[i] * sign[i])`

**Rationale**:
- 避免兩步計算（先乘符號再累加）
- 編譯器可更好地優化
- ONNX 圖更簡潔

---

## 5. Exponential Map

### 數學定義

Bivector 的指數映射：
```
exp(B) = cos(θ) + sin(θ)/θ × B
```

其中 θ 滿足：`θ² = -B²` (對於旋轉 bivector)

### 數值穩定性問題

當 θ → 0 時，`sin(θ)/θ` 需要特殊處理：
- 直接計算會得到 0/0 = NaN
- 使用 `torch.sinc()` 自動處理

### 實作決策
**Decision**: 使用 `torch.sinc(θ/π)` 處理數值穩定性

**Rationale**:
- PyTorch 的 `sinc(x) = sin(πx)/(πx)` 對 x=0 返回 1
- 轉換：`sin(θ)/θ = sinc(θ/π)`
- 無需自定義 Taylor 展開，無 ONNX If 節點

### Bivector 分量數

| 維度 | Bivector 分量 | EvenVersor 分量 |
|------|--------------|----------------|
| CGA0D | 1 (e+-) | 2 |
| CGA1D | 3 | 4 |
| CGA2D | 6 | 8 |
| CGA3D | 10 | 16 |
| CGA4D | 15 | 32 |
| CGA5D | 21 | 64 |

---

## 6. 統一 API 與靜態路由

### 決策
提供 `compose()`, `sandwich_product()`, `reverse()` 統一 API，基於 `kind` 屬性在 Python 圖構建時靜態路由。

### 路由規則
```python
def compose(v1, v2):
    if v1.kind == 'similitude' and v2.kind == 'similitude':
        return compose_similitude(v1.data, v2.data), kind='similitude'
    elif v1.kind in ('even_versor', 'similitude') and v2.kind in ('even_versor', 'similitude'):
        return compose_even_versor(v1.data, v2.data), kind='even_versor'
    else:
        return geometric_product_full(v1.data, v2.data), kind=None
```

### 類型退化規則
```
Similitude × Similitude → Similitude
Similitude × EvenVersor → EvenVersor (退化)
EvenVersor × EvenVersor → EvenVersor
其他 → Multivector (使用 full 版本)
```

### ONNX 相容性
靜態路由在圖構建時決定，不產生 If 節點。

---

## 7. 運算子重載設計

### 決策
採用以下運算子映射：

| 運算子 | 操作 | 理由 |
|--------|------|------|
| `*` | 幾何積 / compose | 傳統 GA 庫慣例 |
| `^` | 楔積 | 數學符號 ∧ 的近似 |
| `\|` | 內積 | 數學符號 · 的替代 |
| `<<` | 左縮併 | 方向性暗示（左到右） |
| `>>` | 右縮併 | 方向性暗示（右到左） |
| `@` | 三明治積 | PyTorch 矩陣乘法慣例延伸 |
| `~` | 反向 | 波浪號暗示「反轉」|
| `**` | 冪次/逆元 | Python 標準冪次 |
| `.exp()` | 指數映射 | 方法呼叫（非運算子） |
| `.inverse()` | 逆元 | 方法呼叫（非運算子） |

---

## 8. Layer 命名統一

### 決策
使用 `CliffordTransformLayer` 取代維度特定的 `CGA{n}DCareLayer`。

### 理由
- **通用性**: 適用於任意 Clifford Algebra，不僅限 CGA
- **簡潔性**: 單一類別名稱，無需記憶維度後綴
- **移除 CARE 依賴**: CARE 是特定論文名稱，不應成為通用 API 名稱

### 命名對照
| 舊名稱 | 新名稱 |
|--------|--------|
| CGA{n}DCareLayer | CliffordTransformLayer |
| UPGC{n}DEncoder | CGAEncoder |
| UPGC{n}DDecoder | CGADecoder |
| CGA{n}DTransformPipeline | CGAPipeline |
| get_care_layer() | get_transform_layer() |

---

## 9. Multivector 包裝類別設計

### 屬性
```python
class Multivector:
    data: Tensor      # 底層張量
    algebra: CliffordAlgebraBase  # 代數實例
    kind: str | None  # 類型標記: 'even_versor', 'similitude', 'point', 'bivector', None
```

### 類型層級
```python
class Versor(Multivector):
    order: str  # 'full', 'even', 'odd'

class EvenVersor(Versor):  # = Versor(order='even')
    pass

class Similitude(EvenVersor):  # CGA 專用
    pass
```

---

## 10. ONNX 匯出策略

### 決策
- 運算子重載：適合原型開發，內部優先使用 full 版本函式
- functional API：適合生產部署，直接控制計算路徑

### 文檔建議
```python
# 原型開發（方便但可能較慢）
result = versor1 * versor2

# 生產部署（直接、高效、ONNX 相容）
result = compose_even_versor(versor1_tensor, versor2_tensor)
```

---

## 11. codegen 擴展策略

### 新增分析函式 (sparse_analysis.py)
1. `get_compose_even_versor_terms(dim)` - EvenVersor × EvenVersor 稀疏項
2. `get_compose_similitude_terms(dim)` - Similitude × Similitude 稀疏項
3. `get_inner_product_signs(dim)` - 度規符號向量
4. `get_bivector_squared_terms(dim)` - Bivector² 標量項
5. `get_outer_product_terms(dim)` - 楔積稀疏項
6. `get_left_contraction_terms(dim)` - 左縮併稀疏項
7. `get_right_contraction_terms(dim)` - 右縮併稀疏項

### 新增生成函式 (generate.py)
1. `_generate_compose_even_versor()` - 生成 compose_even_versor
2. `_generate_compose_similitude()` - 生成 compose_similitude
3. `_generate_sandwich_product_similitude()` - 生成 sandwich_product_similitude
4. `_generate_inner_product_full()` - 生成 inner_product
5. `_generate_exp_bivector()` - 生成 exp_bivector
6. `_generate_outer_product_full()` - 生成 outer_product
7. `_generate_left_contraction_full()` - 生成 left_contraction
8. `_generate_right_contraction_full()` - 生成 right_contraction
9. `_generate_grade_select()` - 生成 grade_select
10. `_generate_dual()` - 生成 dual
11. `_generate_normalize()` - 生成 normalize

---

## 12. 測試對照策略

### clifford 庫對照
```python
from clifford import conformalize, Cl

# EvenVersor Composition
G, blades, stuff = conformalize(Cl(3)[0])
V1_cliff = ...  # clifford 馬達
V2_cliff = ...
result_cliff = V1_cliff * V2_cliff  # 幾何積

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

## 13. Outer Product (楔積)

### 數學定義
楔積 (Outer Product / Wedge Product) 產生兩個多向量的外積：
```
a ∧ b = Σ <a>_r ∧ <b>_s  (Grade r+s 項)
```

### Grade 規則
```
Grade(a ∧ b) = Grade(a) + Grade(b)
```

但需要滿足：當 a 和 b 共享基底時，結果為 0。

### 基本性質
```python
e1 ^ e2 = e12        # Grade 1+1 = 2
e1 ^ e1 = 0          # 共享基底 → 0
e2 ^ e1 = -e12       # 反對稱
e1 ^ e12 = 0         # e1 ⊂ e12 → 0
```

### 稀疏性分析
- 只有不共享基底的項才產生非零結果
- Grade r × Grade s → Grade r+s（若可能）
- 超過最大 Grade 則為 0

### 實作策略
**Decision**: 使用 codegen 生成 `outer_product_full()`

**Rationale**:
- 預計算所有非零項對
- 避免運行時判斷基底重疊
- 稀疏性可達 50-70%

---

## 14. Left/Right Contraction (縮併)

### 數學定義
**左縮併 (Left Contraction)**:
```
a ⌋ b = Σ <a ⌋ b>_{Grade(b) - Grade(a)}
```
結果 Grade = Grade(b) - Grade(a)，若 Grade(a) > Grade(b) 則為 0。

**右縮併 (Right Contraction)**:
```
a ⌊ b = reverse(reverse(b) ⌋ reverse(a))
```
結果 Grade = Grade(a) - Grade(b)，若 Grade(a) < Grade(b) 則為 0。

### 基本性質
```python
# Left Contraction
e1 ⌋ e12 = e2         # Grade 2-1=1
e1 ⌋ e1 = 1           # Grade 1-1=0 (scalar)
e1 ⌋ e2 = 0           # 正交基底
e12 ⌋ e1 = 0          # Grade 1 < Grade 2

# Right Contraction
e12 ⌊ e1 = -e2        # Grade 2-1=1
e1 ⌊ e12 = 0          # Grade 1 < Grade 2
```

### Grade 規則總結
| 操作 | 輸入 Grade | 結果 Grade | 條件 |
|------|-----------|-----------|------|
| a ⌋ b | r, s | s - r | r ≤ s |
| a ⌊ b | r, s | r - s | r ≥ s |

### 實作策略
**Decision**: 使用 codegen 生成 `left_contraction_full()` 和 `right_contraction_full()`

**Rationale**:
- Grade 規則可在編譯時確定
- 只生成滿足 Grade 條件的項
- 避免運行時 Grade 檢查

---

## 15. Grade Selection

### 數學定義
從多向量中提取特定 Grade 的分量：
```
<a>_k = 僅保留 Grade k 的 blade 分量
```

### CGA Grade 遮罩

| 維度 | Grade 分布 (C(n+2, k)) |
|------|------------------------|
| CGA0D Cl(1,1) | [1, 2, 1] |
| CGA1D Cl(2,1) | [1, 3, 3, 1] |
| CGA2D Cl(3,1) | [1, 4, 6, 4, 1] |
| CGA3D Cl(4,1) | [1, 5, 10, 10, 5, 1] |
| CGA4D Cl(5,1) | [1, 6, 15, 20, 15, 6, 1] |
| CGA5D Cl(6,1) | [1, 7, 21, 35, 35, 21, 7, 1] |

### CGA3D Grade 索引對照
```
Grade 0: [0]           - 1 個 (scalar)
Grade 1: [1-5]         - 5 個 (e1, e2, e3, e+, e-)
Grade 2: [6-15]        - 10 個 (bivectors)
Grade 3: [16-25]       - 10 個 (trivectors)
Grade 4: [26-30]       - 5 個 (quadvectors)
Grade 5: [31]          - 1 個 (pseudoscalar)
```

### 實作策略
**Decision**: 預計算 Grade 遮罩常數，使用索引選擇

```python
# 硬編碼遮罩
GRADE_MASKS = {
    0: [0],
    1: [1, 2, 3, 4, 5],
    2: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    # ...
}

def grade_select(mv, grade):
    mask = GRADE_MASKS[grade]
    return mv[..., mask]
```

**Rationale**:
- 遮罩在編譯時已知
- 純索引操作，無計算
- ONNX 友好

---

## 16. Dual (對偶)

### 數學定義
多向量的對偶是與 Pseudoscalar 的乘積：
```
dual(a) = a * I^{-1}
```

其中 I 是 Pseudoscalar (最高 Grade blade)。

### Pseudoscalar 性質

| 維度 | Pseudoscalar | I² |
|------|--------------|-----|
| CGA0D | e+- | +1 |
| CGA1D | e1+- | -1 |
| CGA2D | e12+- | -1 |
| CGA3D | e123+- | -1 |
| CGA4D | e1234+- | +1 |
| CGA5D | e12345+- | -1 |

### 符號規則
```
I² = (-1)^{n(n-1)/2 + 1}  (對於 Cl(n+1,1))
I^{-1} = I / I² = I * sign(I²)
```

### 對偶效果
```
Grade(dual(a)) = max_grade - Grade(a)
```

CGA3D 例子：
```python
dual(1) = I^{-1}           # scalar → pseudoscalar
dual(e1) = e1 * I^{-1} = -e2345  # Grade 1 → Grade 4
dual(e12) = e12 * I^{-1} = e345   # Grade 2 → Grade 3
```

### 實作策略
**Decision**: 預計算 I^{-1} 符號，使用幾何積

```python
def dual(mv):
    # I_inv_sign = +1 或 -1 取決於 I²
    return geometric_product_full(mv, I_INV)
```

**Rationale**:
- I^{-1} 是常數多向量
- 可重用幾何積實作
- 符號預計算避免運行時判斷

---

## 17. Normalize (正規化)

### 數學定義
正規化將多向量縮放至單位範數：
```
normalize(a) = a / |a|
|a|² = <a * ~a>_0
```

### 範數計算
使用幾何積的 Grade 0 分量：
```python
norm_sq = (a * reverse(a))[..., 0]  # scalar 分量
norm = sqrt(norm_sq)
```

### 數值穩定性
處理零向量或近零範數：
```python
def normalize(mv, eps=1e-12):
    norm_sq = inner_product(mv, mv)  # |mv|²
    norm = torch.sqrt(torch.clamp(norm_sq, min=eps))
    return mv / norm
```

### 零向量處理
**Decision**: 使用 `torch.where` 保持原值

```python
def normalize(mv, eps=1e-12):
    norm_sq = inner_product(mv, mv)
    norm = torch.sqrt(norm_sq)
    safe_norm = torch.where(norm > eps, norm, torch.ones_like(norm))
    result = mv / safe_norm.unsqueeze(-1)
    # 零向量保持不變
    return torch.where(norm.unsqueeze(-1) > eps, result, mv)
```

**Rationale**:
- 避免 NaN 傳播
- ONNX 相容（無 If 節點）
- 符合數學直覺（零向量無法正規化）

---

## 18. Similitude 深入分析

### 正確理解

**重要澄清**：Similitude 加速並非減少儲存分量，而是利用代數稀疏性。

### Orthonormal vs Null Basis

在 orthonormal basis {e1,...,en, e+, e-} 中：
- `ei ∧ e+` 和 `ei ∧ e-` 混合了平移和 transversion
- 無法直接區分 Similitude 和 EvenVersor

在 null basis {e1,...,en, eo, einf} 中：
- `ei ∧ einf` = 平移分量
- `ei ∧ eo` = Transversion 分量

### 轉換關係
```
eo = 0.5*(e- - e+)
einf = e+ + e-

ei ∧ einf = ei+ + ei-    (平移)
ei ∧ eo = 0.5*(ei- - ei+)  (transversion)
```

### Similitude 約束
在 orthonormal basis 下，Similitude 滿足約束：
```
對每個 i: coefficient(ei+) = coefficient(ei-)
```
這等價於排除 `ei ∧ eo` 分量。

### 加速策略

儲存：仍使用完整 EvenVersor 格式（16 分量 for CGA3D）

計算優化：
1. `compose_similitude(S1, S2)` 知道結果的 `ei ∧ eo` 分量為零
2. 可跳過產生 transversion 分量的乘法項
3. 減少約 30-40% 計算量

### Similitude 有效分量數（含約束）
| 維度 | EvenVersor | Transversion 約束 | Similitude 有效 |
|------|------------|-------------------|-----------------|
| CGA0D | 2 | 0 | 2 |
| CGA1D | 4 | 1 | 3 |
| CGA2D | 8 | 2 | 6 |
| CGA3D | 16 | 3 | 13 |
| CGA4D | 32 | 4 | 28 |
| CGA5D | 64 | 5 | 59 |

---

## 19. TRS 轉換函式

### 設計決策
提供 TRS (Translation-Rotation-Scaling) 與 Similitude 之間的轉換函式，方便使用者輸入，內部仍使用 EvenVersor 格式計算。

### TRS 參數數量

| 維度 | T (平移) | R (旋轉) | S (縮放) | TRS 總計 | EvenVersor |
|------|---------|---------|---------|----------|------------|
| CGA0D | 0 | 0 | 1 | 1 | 2 |
| CGA1D | 1 | 0 | 1 | 2 | 4 |
| CGA2D | 2 | 1 (角度) | 1 | 4 | 8 |
| CGA3D | 3 | 3 (四元數-1) | 1 | 7 | 16 |
| CGA4D | 4 | 6 | 1 | 11 | 32 |
| CGA5D | 5 | 10 | 1 | 16 | 64 |

**註**：旋轉自由度 = C(n,2) = n(n-1)/2

### 數學公式

**Similitude = Translation × Rotation × Dilation**

各分量的 CGA 表示：

```
Translation: T = 1 + (1/2) * t * einf
           = 1 + (1/2) * (t1*e1 + t2*e2 + ...) * (e+ + e-)

Rotation:   R = cos(θ/2) + sin(θ/2) * B
           其中 B 是單位 bivector（旋轉平面）

Dilation:   D = cosh(λ/2) + sinh(λ/2) * e+-
           其中 s = e^λ 是縮放因子
```

**組合順序**：`S = T * R * D`（先縮放，再旋轉，最後平移）

### API 設計

```python
class CGAAlgebraBase:
    def from_trs(
        self,
        translation: Tensor,  # (..., dim) 平移向量
        rotation: Tensor,     # (..., rot_params) 旋轉參數
        scale: Tensor,        # (..., 1) 縮放因子
        rotation_format: str = 'quaternion'  # 'quaternion', 'euler', 'bivector'
    ) -> Tensor:
        """
        從 TRS 參數建立 Similitude。

        Args:
            translation: 平移向量，shape (..., dim)
            rotation: 旋轉參數
                - quaternion (3D): (..., 4) [w, x, y, z]
                - euler (3D): (..., 3) [roll, pitch, yaw]
                - bivector: (..., bivector_count)
                - angle (2D): (..., 1)
            scale: 均勻縮放因子，shape (..., 1)
            rotation_format: 旋轉參數格式

        Returns:
            Similitude tensor, shape (..., even_versor_count)
        """
        ...

    def to_trs(
        self,
        similitude: Tensor,
        rotation_format: str = 'quaternion'
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        從 Similitude 提取 TRS 參數。

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)
            rotation_format: 輸出旋轉格式

        Returns:
            (translation, rotation, scale) 元組
        """
        ...
```

### 實作細節

**from_trs 實作**：
```python
def from_trs(self, t, r, s, rotation_format='quaternion'):
    # 1. 建立 Dilation: D = cosh(λ/2) + sinh(λ/2) * e+-
    lambda_ = torch.log(s)
    D = self._make_dilation(lambda_)

    # 2. 建立 Rotation: R = exp(B)
    if rotation_format == 'quaternion':
        B = self._quaternion_to_bivector(r)
    elif rotation_format == 'euler':
        B = self._euler_to_bivector(r)
    elif rotation_format == 'bivector':
        B = r
    elif rotation_format == 'angle':  # 2D only
        B = self._angle_to_bivector(r)
    R = self.exp_bivector(B)

    # 3. 建立 Translation: T = 1 + (1/2) * t * einf
    T = self._make_translation(t)

    # 4. 組合: S = T * R * D
    RD = self.compose_similitude(R, D)
    TRD = self.compose_similitude(T, RD)

    return TRD
```

**to_trs 實作**：
```python
def to_trs(self, similitude, rotation_format='quaternion'):
    # 1. 提取縮放 (從 e+- 分量)
    scale = self._extract_scale(similitude)

    # 2. 移除縮放
    D_inv = self._make_dilation(-torch.log(scale))
    TR = self.compose_similitude(similitude, D_inv)

    # 3. 提取平移 (從 ei+ + ei- 分量)
    translation = self._extract_translation(TR)

    # 4. 移除平移
    T_inv = self._make_translation(-translation)
    R = self.compose_similitude(T_inv, TR)

    # 5. 提取旋轉
    if rotation_format == 'quaternion':
        rotation = self._rotor_to_quaternion(R)
    elif rotation_format == 'euler':
        rotation = self._rotor_to_euler(R)
    elif rotation_format == 'bivector':
        rotation = self._extract_bivector(R)
    elif rotation_format == 'angle':
        rotation = self._rotor_to_angle(R)

    return translation, rotation, scale
```

### 維度特定處理

**CGA2D (2D 旋轉)**：
```python
# 旋轉只有一個角度
angle = rotation[..., 0]  # (..., 1)
B = angle / 2 * e12  # bivector
R = cos(angle/2) + sin(angle/2) * e12
```

**CGA3D (3D 旋轉)**：
```python
# 四元數 [w, x, y, z] -> Rotor
# R = w + x*e23 + y*e31 + z*e12
# 注意：CGA 的 bivector 順序可能不同
```

### 數值穩定性

1. **縮放接近 0**：`scale = max(scale, eps)` 避免 log(0)
2. **縮放接近 1**：使用 `log1p(scale - 1)` 提高精度
3. **旋轉提取**：正規化 Rotor 後再轉換

### ONNX 相容性

- `from_trs`：包含 log、sinh、cosh，但都是標準 ONNX 運算子
- `to_trs`：包含 acos、atan2 等，需確認 ONNX opset 支援

---

## 20. Multivector 類別詳細設計

### 類別層級
```
        Multivector
             │
          Versor
         /   │   \
  OddVersor  │  Bivector
        EvenVersor
             │
        Similitude (CGA專用)
```

### 完整屬性
```python
class Multivector:
    data: Tensor           # (..., blade_count)
    algebra: CGAAlgebraBase
    kind: str | None       # 'even_versor', 'similitude', 'bivector', 'point', None
```

### Kind 值定義
| Kind | 含義 | 用途 |
|------|------|------|
| None | 一般多向量 | 無特殊優化 |
| 'versor' | 可逆多向量 | - |
| 'even_versor' | 偶數 Versor | compose_even_versor |
| 'similitude' | CGA Similitude | compose_similitude |
| 'bivector' | Grade-2 | exp_bivector |
| 'point' | CGA 點 | - |

### 運算子實作要點

**幾何積 `*`**：
```python
def __mul__(self, other):
    if isinstance(other, (int, float, Tensor)):
        return Multivector(self.data * other, self.algebra, self.kind)

    # 靜態路由
    if self.kind == 'similitude' and other.kind == 'similitude':
        return Multivector(
            self.algebra.compose_similitude(self.data, other.data),
            self.algebra, 'similitude'
        )
    elif self._both_even(other):
        return Multivector(
            self.algebra.compose_even_versor(self.data, other.data),
            self.algebra, 'even_versor'
        )
    else:
        return Multivector(
            self.algebra.geometric_product_full(self.data, other.data),
            self.algebra, None
        )
```

**三明治積 `@`**：
```python
def __matmul__(self, other):
    # versor @ point = sandwich(versor, point)
    return Multivector(
        self.algebra.sandwich_product(self.data, other.data),
        self.algebra, other.kind  # 保持被變換對象類型
    )
```

### 工廠方法
```python
class CGAAlgebraBase:
    def multivector(self, data, kind=None) -> Multivector
    def even_versor(self, data) -> EvenVersor
    def similitude(self, data) -> Similitude
    def bivector(self, data) -> Bivector
    def point(self, xyz) -> Multivector  # kind='point'
```

---

## 21. Structure Normalize (結構正規化)

### 動機

在深度學習訓練過程中，Similitude 參數可能逐漸偏離有效的幾何結構。`structure_normalize` 用於：

1. **保持 Rotor 單位性**：確保旋轉部分維持單位模長
2. **強制 Similitude 約束**：確保 `ei+ = ei-`（排除 transversion）
3. **穩定 Dilation**：避免縮放因子過大或過小

### Similitude 結構分解

以 CGA3D 為例，Similitude = T × R × D：

```
S = T * R * D
  = (1 + t·einf/2) * (cos(θ/2) + sin(θ/2)*B) * (cosh(λ/2) + sinh(λ/2)*e+-)
```

展開後的 EvenVersor 分量：

| 分量群組 | 索引 (CGA3D) | 說明 |
|----------|-------------|------|
| Scalar | 0 | cos(θ/2) * cosh(λ/2) |
| Rotor Bivector | 1,2,5 (e12, e13, e23) | sin(θ/2) * B * cosh(λ/2) |
| Dilation | 10 (e+-) | cos(θ/2) * sinh(λ/2) + ... |
| Translation | 3,4,6,7,8,9 (ei+, ei-) | t·einf/2 * R * D |
| Grade 4 | 11-15 | T * R * D 的 Grade 4 部分 |

### Structure Normalize 演算法

```python
def structure_normalize(self, similitude: Tensor, eps: float = 1e-8) -> Tensor:
    """
    對 Similitude 進行結構正規化。

    步驟：
    1. 正規化 Rotor 部分（保持旋轉為單位四元數）
    2. 強制 Similitude 約束（ei+ = ei-）
    3. 可選：限制 Dilation 範圍

    Args:
        similitude: (..., even_versor_count) Similitude 張量
        eps: 數值穩定性常數

    Returns:
        結構正規化後的 Similitude
    """
    result = similitude.clone()

    # === Step 1: 正規化 Rotor 部分 ===
    # Rotor = scalar + bivector (純旋轉平面的 bivector)
    # CGA3D: scalar=0, spatial_bivectors=1,2,5 (e12, e13, e23)
    rotor_indices = self.ROTOR_INDICES  # [0, 1, 2, 5] for CGA3D
    rotor_part = similitude[..., rotor_indices]

    # 計算 Rotor 模長
    rotor_norm = torch.norm(rotor_part, dim=-1, keepdim=True) + eps

    # 正規化 Rotor
    normalized_rotor = rotor_part / rotor_norm
    result[..., rotor_indices] = normalized_rotor

    # === Step 2: 強制 Similitude 約束 ===
    # 約束: coefficient(ei+) = coefficient(ei-)
    # 這等價於排除 transversion (ei ∧ eo)
    # CGA3D: (3,4), (6,7), (8,9) 是 (e1+,e1-), (e2+,e2-), (e3+,e3-)
    for plus_idx, minus_idx in self.TRANSLATION_PAIRS:  # [(3,4), (6,7), (8,9)]
        # 取平均值強制相等
        avg = (result[..., plus_idx] + result[..., minus_idx]) / 2
        result[..., plus_idx] = avg
        result[..., minus_idx] = avg

    # === Step 3 (可選): 限制 Dilation 範圍 ===
    # e+- 分量控制縮放: D = cosh(λ/2) + sinh(λ/2) * e+-
    # 可用 clamp 限制 λ 範圍，避免過大/過小縮放
    # (此步驟可選，取決於應用需求)

    return result
```

### 各維度的 Rotor 和 Translation 索引

**CGA2D** (8 分量 EvenVersor):
```
Rotor: [0, 1]           # scalar, e12
Translation pairs: [(2,3), (4,5)]  # (e1+,e1-), (e2+,e2-)
Dilation: [6]           # e+-
Grade 4: [7]            # e12+-
```

**CGA3D** (16 分量 EvenVersor):
```
Rotor: [0, 1, 2, 5]     # scalar, e12, e13, e23
Translation pairs: [(3,4), (6,7), (8,9)]  # (e1+,e1-), (e2+,e2-), (e3+,e3-)
Dilation: [10]          # e+-
Grade 4: [11-15]        # e123+, e123-, e12+-, e13+-, e23+-
```

### 數學正確性

**Rotor 正規化**：
- Rotor R 滿足 R * ~R = 1（單位 versor）
- |R|² = scalar² + bivector·bivector = 1
- 正規化保持旋轉角度和軸正確

**Similitude 約束強制**：
- 平移: `T = 1 + t·einf/2`，其中 `einf = e+ + e-`
- 展開: `t1·e1·einf/2 = t1/2 * (e1+ + e1-)`
- 約束 `ei+ = ei-` 等價於只保留平移，排除 transversion

### ONNX 相容性

此演算法無迴圈、無條件分支：
- 索引選取：使用預定義常數索引
- 範數計算：`torch.norm`
- 平均操作：簡單算術

完全 ONNX 相容。

### 使用場景

```python
# 在 forward pass 中使用
class CliffordTransformLayer(nn.Module):
    def forward(self, similitude, point):
        # 可選：每次 forward 前正規化
        clean_similitude = self.algebra.structure_normalize(similitude)
        return self.algebra.sandwich_product_similitude(clean_similitude, point)

# 或作為正則化 loss
def structure_regularization_loss(similitude, algebra):
    clean = algebra.structure_normalize(similitude)
    return F.mse_loss(similitude, clean)
```

### 進階變體

**Soft Structure Normalize**（可微分版本）：
```python
def soft_structure_normalize(self, similitude, strength=0.1):
    """軟性正規化，用於訓練時的梯度友好版本"""
    target = self.structure_normalize(similitude)
    return similitude + strength * (target - similitude)
```

**Gradient-Friendly Structure Normalize**：
```python
def structure_normalize_ste(self, similitude):
    """使用 Straight-Through Estimator 的版本"""
    clean = self.structure_normalize(similitude)
    # STE: forward 用 clean，backward 用 similitude 的梯度
    return similitude + (clean - similitude).detach()
```

---

## 22. 效能測試方法論

### 測試標準

為確保效能量測的可重複性和準確性，定義以下標準測試參數：

| 參數 | 值 | 說明 |
|------|-----|------|
| Batch Size | 1024 | 標準測試批次大小 |
| Warmup Iterations | 100 | 預熱迴圈數（排除 JIT 編譯和 GPU 啟動時間） |
| Measurement Iterations | 1000 | 實際測量迴圈數 |
| Device | CPU (single thread) | 主要測試環境，避免 GPU 變異性 |
| dtype | torch.float32 | 標準精度 |
| Synchronization | torch.cuda.synchronize() / mps_sync | GPU 測試時同步 |

### 測試環境

**主要環境** (Apple M3)：
```python
import torch
torch.set_num_threads(1)  # 單執行緒，確保可重複性
device = 'cpu'  # 或 'mps'
```

**驗證環境** (NVIDIA GPU)：
```python
device = 'cuda'
torch.backends.cudnn.benchmark = False  # 確保可重複性
```

### 效能量測程式碼範本

```python
import torch
import time

def benchmark_operation(op_fn, args, batch_size=1024, warmup=100, iterations=1000):
    """
    標準效能量測函式。

    Returns:
        dict: {
            'mean_ms': float,      # 平均每次呼叫時間 (毫秒)
            'std_ms': float,       # 標準差 (毫秒)
            'throughput': float,   # 每秒處理批次數
        }
    """
    # Warmup
    for _ in range(warmup):
        _ = op_fn(*args)

    # Synchronize (如果在 GPU)
    if args[0].is_cuda:
        torch.cuda.synchronize()

    # Measurement
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = op_fn(*args)
        if args[0].is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = torch.tensor(times) * 1000  # 轉換為毫秒
    return {
        'mean_ms': times.mean().item(),
        'std_ms': times.std().item(),
        'throughput': batch_size / (times.mean().item() / 1000),
    }
```

### 加速效果驗證 (SC-001a)

Similitude vs EvenVersor 效能比較標準：

```python
def verify_similitude_speedup(cga, batch_size=1024):
    """
    驗證 Similitude 操作比 EvenVersor 快 30-50%。

    Returns:
        dict: {
            'compose_speedup': float,    # compose 加速倍率
            'sandwich_speedup': float,   # sandwich 加速倍率
            'pass': bool,                # 是否通過 30% 門檻
        }
    """
    s1 = torch.randn(batch_size, cga.similitude_count)
    s2 = torch.randn(batch_size, cga.similitude_count)
    v1 = torch.randn(batch_size, cga.even_versor_count)
    v2 = torch.randn(batch_size, cga.even_versor_count)

    # Compose benchmark
    ev_compose = benchmark_operation(cga.compose_even_versor, (v1, v2))
    sim_compose = benchmark_operation(cga.compose_similitude, (s1, s2))

    compose_speedup = ev_compose['mean_ms'] / sim_compose['mean_ms']

    return {
        'compose_speedup': compose_speedup,
        'pass': compose_speedup >= 1.3,  # 至少 30% 加速
    }
```

### 效能報告格式

效能測試結果應包含以下資訊：

```markdown
## 效能報告

**測試環境**: Apple M3 Pro, macOS 14.0, PyTorch 2.1.0

| 操作 | 維度 | Batch | Mean (ms) | Std (ms) | Throughput |
|------|------|-------|-----------|----------|------------|
| compose_even_versor | CGA3D | 1024 | 0.123 | 0.005 | 8.3M ops/s |
| compose_similitude | CGA3D | 1024 | 0.089 | 0.004 | 11.5M ops/s |
| **加速比** | | | **1.38x** | | |
```

### 邊界情況效能測試

除標準批次測試外，應測試以下邊界情況：

1. **小批次** (batch_size=1)：驗證低延遲
2. **大批次** (batch_size=16384)：驗證吞吐量
3. **高維度** (CGA5D, CGA6D)：確認可接受的效能下降

---

## 23. 結論

所有研究項目已完成：

| 項目 | 解決方案 |
|------|---------|
| Motor → EvenVersor | 採用 EvenVersor 作為通用術語 |
| Similitude 設計 | EvenVersor 子類別，利用代數稀疏性加速（非儲存壓縮）|
| Outer Product | Grade r+s 規則，共享基底為零 |
| Left/Right Contraction | Grade 差規則，codegen 生成 |
| Grade Selection | 預計算遮罩，純索引操作 |
| Dual | I^{-1} 預計算，重用幾何積 |
| Normalize | torch.where 處理零向量 |
| Multivector 類別 | kind 屬性靜態路由，ONNX 友好 |
| 運算子重載 | `*` `^` `\|` `<<` `>>` `@` `~` `**` |
| 統一 API | compose(), sandwich_product(), reverse() |
| Layer 命名 | CliffordTransformLayer |
| 數值穩定性 | torch.sinc, torch.where |
| ONNX 相容 | 靜態路由無 If 節點 |

---

## 參考資料

1. Dorst, L., Fontijne, D., & Mann, S. (2007). *Geometric Algebra for Computer Science*
2. Perwass, C. (2009). *Geometric Algebra with Applications in Engineering*
3. fast-clifford 現有 codegen 系統文檔
4. PyTorch ONNX 匯出指南
