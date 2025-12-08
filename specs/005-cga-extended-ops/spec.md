# Feature Specification: Clifford Algebra Extended Operations

**Feature Branch**: `005-cga-extended-ops`
**Created**: 2025-12-08
**Status**: Draft
**Input**: User description: "重構命名架構，使用 Versor/EvenVersor 取代 Motor，新增 CGA 專用 Similitude 加速，統一 Layer 命名為 CliffordTransformLayer"

## 架構概覽

```
Clifford Algebra (通用，任意度規)
├── Versor                         # 基礎類別
│   ├── order='full'               # 完整 Versor
│   ├── order='even'               # 偶數 Versor (EvenVersor 語法糖)
│   └── order='odd'                # 奇數 Versor
├── EvenVersor                     # = Versor(order='even')，低維度加速
├── CliffordTransformLayer         # 統一 PyTorch Layer
└── 底層函式:
    ├── compose_even_versor        # 偶數 Versor 組合
    ├── sandwich_product_even_versor # 偶數 Versor 三明治積
    └── reverse_even_versor        # 偶數 Versor 反向

CGA (專用，保形幾何代數)
└── Similitude(EvenVersor)         # 子類別：平移 + 旋轉 + 縮放
    └── 排除 transversion（更稀疏，更激進加速）
    └── 底層函式:
        ├── compose_similitude     # Similitude 組合（更快）
        ├── sandwich_product_similitude # Similitude 三明治積（更快）
        └── reverse_similitude     # Similitude 反向

統一 API (靜態分派):
├── compose(v1, v2)                # 自動路由到最佳 compose_* 實作
├── sandwich_product(v, x)         # 自動路由到最佳 sandwich_product_* 實作
└── reverse(v)                     # 自動路由到最佳 reverse_* 實作
```

## User Scenarios & Testing *(mandatory)*

### User Story 1 - EvenVersor Composition for Transform Chaining (Priority: P1)

開發者需要組合多個幾何變換（如先旋轉再平移），透過 EvenVersor Composition 將兩個偶數 Versor 合併為單一 Versor，用於機器人運動學、3D 動畫等應用。此功能適用於任意 Clifford Algebra。

**Why this priority**: EvenVersor Composition 是最基礎的變換組合操作，幾乎所有複雜幾何變換都需要將多個操作串接。沒有此功能，使用者必須多次執行 sandwich product，效能較差。

**Independent Test**: 可透過建立兩個已知變換（如 90° 旋轉 + 平移），組合後驗證等同於單一複合變換的效果。

**Acceptance Scenarios**:

1. **Given** 兩個 CGA3D EvenVersor V1（旋轉）和 V2（平移），**When** 呼叫 `compose(V1, V2)`，**Then** 自動路由到 `compose_even_versor` 並返回正確組合的 EvenVersor
2. **Given** 單位 EvenVersor identity，**When** 呼叫 `compose(identity, V)`，**Then** 返回 V 本身
3. **Given** EvenVersor V 及其逆元 V_rev，**When** 呼叫 `compose(V, V_rev)`，**Then** 返回近似單位 EvenVersor

---

### User Story 2 - Geometric Inner Product for Attention & Loss (Priority: P1)

開發者需要計算 CGA 多向量的幾何內積，用於深度學習中的 Attention Score 計算和損失函數。必須正確處理 CGA 的非歐幾里得度規 (+,+,+,+,-)。

**Why this priority**: 這是 CARE Transformer 等幾何深度學習模型的核心計算，直接影響模型訓練和推理的正確性。

**Independent Test**: 可透過計算已知 Null Basis 向量 (eo, einf) 的內積驗證，應得到 -1。

**Acceptance Scenarios**:

1. **Given** CGA3D 的 eo 和 einf 向量，**When** 呼叫 `inner_product(eo, einf)`，**Then** 返回 -1
2. **Given** 任意 CGA 多向量 a 和 b，**When** 呼叫 `inner_product(a, b)`，**Then** 返回正確的標量內積（幾何積的 Grade 0 分量）
3. **Given** 正交的 basis blades，**When** 計算其內積，**Then** 返回 0

---

### User Story 3 - Exponential Map for EvenVersor Generation (Priority: P2)

開發者需要從旋轉軸和角度生成 EvenVersor，透過 Bivector 的指數映射 exp(B) 產生 Rotor/EvenVersor。此功能用於插值、平滑動畫、從李代數生成變換等場景。

**Why this priority**: 雖然使用者可直接建構 EvenVersor，但 Exponential Map 提供更直觀的數學介面，對於旋轉插值（slerp）和李代數運算至關重要。

**Independent Test**: 可透過將已知旋轉角度的 Bivector 傳入 exp_bivector，驗證產生的 EvenVersor 是否正確旋轉點。

**Acceptance Scenarios**:

1. **Given** 零 Bivector B=0，**When** 呼叫 `exp_bivector(B)`，**Then** 返回單位 EvenVersor (1, 0, 0, ...)
2. **Given** 代表 90° 旋轉的 Bivector，**When** 呼叫 `exp_bivector(B)`，**Then** 產生正確的旋轉 EvenVersor
3. **Given** 極小 Bivector (θ < 1e-6)，**When** 呼叫 `exp_bivector(B)`，**Then** 數值穩定地返回近似單位 EvenVersor（無 NaN 或 Inf）

---

### User Story 4 - High-Dimensional Runtime Support (Priority: P2)

開發者需要在 6D 及以上維度使用相同的 API，系統自動切換至運行時一般化算法，確保功能完整性。

**Why this priority**: 保持 API 一致性，讓高維度研究者無需學習不同介面。效能不是主要考量（高維度本來就較慢）。

**Independent Test**: 可透過 CGA(6) 呼叫新操作，驗證功能正確且無錯誤。

**Acceptance Scenarios**:

1. **Given** CGA(6) 代數實例，**When** 呼叫 `compose`，**Then** 返回正確結果（與 clifford 庫對照）
2. **Given** CGA(7) 代數實例，**When** 呼叫 `inner_product`，**Then** 返回正確標量
3. **Given** CGA(6) 代數實例，**When** 呼叫 `exp_bivector`，**Then** 返回正確 EvenVersor

---

### User Story 4a - CGA Similitude Accelerated Operations (Priority: P1)

開發者在 CGA 應用中需要更高效能的變換操作。Similitude（平移 + 旋轉 + 縮放，排除 transversion）是 EvenVersor 的子集，具有更稀疏的結構，可實現更激進的加速。

**Why this priority**: 大多數 CGA 深度學習應用只需要平移、旋轉、縮放，不需要 transversion。Similitude 的稀疏結構可減少 30-50% 計算量。

**Independent Test**: 可透過 Similitude 變換驗證結果與 EvenVersor 一致，但效能更高。

**Acceptance Scenarios**:

1. **Given** 兩個 CGA3D Similitude S1 和 S2，**When** 呼叫 `compose(S1, S2)`，**Then** 自動路由到 `compose_similitude` 且效能優於 `compose_even_versor`
2. **Given** Similitude S 和 Point P，**When** 呼叫 `sandwich_product(S, P)`，**Then** 自動路由到 `sandwich_product_similitude` 且效能優於 `sandwich_product_even_versor`
3. **Given** 純旋轉 + 純平移 + 純縮放，**When** 組合為 Similitude，**Then** 結果等同於使用 EvenVersor

---

### User Story 5 - Outer Product (Wedge Product) (Priority: P3)

開發者需要計算兩個多向量的外積（楔積），用於建立高階幾何物件（如平面、球面等）和進行投影幾何運算。

**Why this priority**: 外積是建構幾何物件的基礎運算，但在深度學習應用中較少直接使用。優先級低於核心操作但仍是完整代數實作的必要部分。

**Independent Test**: 可透過計算兩個正交向量的外積驗證結果為對應的 Bivector。

**Acceptance Scenarios**:

1. **Given** 兩個正交 Grade 1 向量 e1 和 e2，**When** 呼叫 `outer_product(e1, e2)`，**Then** 返回 e12 Bivector
2. **Given** 同一向量 v，**When** 呼叫 `outer_product(v, v)`，**Then** 返回 0
3. **Given** 任意多向量 a 和 b，**When** 呼叫 `outer_product(a, b)`，**Then** 結果等於幾何積中 Grade |a|+|b| 的分量

---

### User Story 6 - Left/Right Contraction (Priority: P3)

開發者需要計算左縮併和右縮併，用於投影運算、距離計算和幾何分析。

**Why this priority**: 縮併運算用於高階幾何分析，在一般深度學習應用中使用頻率較低。

**Independent Test**: 可透過計算向量與 Bivector 的左縮併驗證結果的 Grade 降低。

**Acceptance Scenarios**:

1. **Given** Grade 1 向量 v 和 Grade 2 Bivector B，**When** 呼叫 `left_contraction(v, B)`，**Then** 返回 Grade 1 結果
2. **Given** Grade 2 Bivector B 和 Grade 1 向量 v，**When** 呼叫 `right_contraction(B, v)`，**Then** 返回 Grade 1 結果
3. **Given** 相同 Grade 的元素 a 和 b，**When** 計算 `left_contraction(a, b)`，**Then** 結果為標量

---

### User Story 7 - Grade Selection (Priority: P3)

開發者需要從完整多向量中提取特定 Grade 的分量，用於分析和處理多向量的特定部分。

**Why this priority**: Grade 提取是基礎工具函式，用於調試和進階分析，但深度學習模型通常使用稀疏表示不需此功能。

**Independent Test**: 可透過從已知多向量提取 Grade 0 分量驗證正確性。

**Acceptance Scenarios**:

1. **Given** 完整多向量 mv，**When** 呼叫 `grade_select(mv, 0)`，**Then** 返回標量分量
2. **Given** 完整多向量 mv，**When** 呼叫 `grade_select(mv, 1)`，**Then** 返回 Grade 1 分量（向量）
3. **Given** 完整多向量 mv 和無效 Grade k，**When** 呼叫 `grade_select(mv, k)`，**Then** 返回零向量

---

### User Story 8 - Dual (Priority: P3)

開發者需要計算多向量的對偶，用於幾何物件的互補表示（如點↔球面、線↔平面等）。

**Why this priority**: 對偶運算在 CGA 理論中重要，但深度學習應用通常不需要此轉換。

**Independent Test**: 可透過計算 Pseudoscalar 的對偶驗證返回標量 1。

**Acceptance Scenarios**:

1. **Given** 標量 1，**When** 呼叫 `dual(1)`，**Then** 返回 Pseudoscalar
2. **Given** Pseudoscalar I，**When** 呼叫 `dual(I)`，**Then** 返回 ±1（依度規符號）
3. **Given** 多向量 mv，**When** 呼叫 `dual(dual(mv))`，**Then** 返回 ±mv

---

### User Story 9 - Normalize (Priority: P3)

開發者需要正規化多向量為單位範數，用於確保數值穩定性和一致的幾何意義。

**Why this priority**: 正規化是常見操作但較為簡單，且可由使用者自行實作。

**Independent Test**: 可透過正規化任意非零向量驗證範數為 1。

**Acceptance Scenarios**:

1. **Given** 非零向量 v，**When** 呼叫 `normalize(v)`，**Then** 返回單位向量（內積為 1）
2. **Given** 零向量，**When** 呼叫 `normalize(0)`，**Then** 返回零向量（不會 NaN）
3. **Given** 正規化後向量 v_norm，**When** 呼叫 `normalize(v_norm)`，**Then** 返回相同向量

---

### User Story 10 - Operator Overloading (Priority: P2)

開發者需要使用直觀的 Python 運算子來操作多向量，使代碼更接近數學公式，提升可讀性和開發效率。

**Why this priority**: 運算子重載是 Python 風格的核心特色，讓幾何代數運算更直觀。相較於函式呼叫（如 `geometric_product(a, b)`），運算子（如 `a * b`）更接近數學表達式，降低認知負擔。

**Independent Test**: 可透過 `a * b` 驗證幾何積、`a ^ b` 驗證楔積、`a | b` 驗證內積。

**Acceptance Scenarios**:

1. **Given** 兩個多向量 a 和 b，**When** 使用 `a * b`，**Then** 返回幾何積結果
2. **Given** 兩個多向量 a 和 b，**When** 使用 `a ^ b`，**Then** 返回楔積（外積）結果
3. **Given** 兩個多向量 a 和 b，**When** 使用 `a | b`，**Then** 返回內積結果
4. **Given** 兩個多向量 a 和 b，**When** 使用 `a + b` 和 `a - b`，**Then** 返回加減結果
5. **Given** 多向量 a 和標量 s，**When** 使用 `a * s` 或 `s * a`，**Then** 返回標量乘積
6. **Given** 多向量 a，**When** 使用 `~a`，**Then** 返回反向（reverse）結果
7. **Given** 多向量 a，**When** 使用 `-a`，**Then** 返回取負結果
8. **Given** 多向量 a 和 b，**When** 使用 `a << b`，**Then** 返回左縮併（left contraction）結果
9. **Given** 多向量 a 和 b，**When** 使用 `a >> b`，**Then** 返回右縮併（right contraction）結果
10. **Given** EvenVersor v 和 Point/Multivector x，**When** 使用 `v @ x`，**Then** 返回三明治積 `v * x * ~v`
11. **Given** 可逆多向量 a 和 b，**When** 使用 `a / b`，**Then** 返回 `a * b^(-1)` 結果
12. **Given** 可逆多向量 a，**When** 使用 `a.inverse()`，**Then** 返回逆元 `a^(-1)`
13. **Given** 多向量 a 和整數 n，**When** 使用 `a ** n`，**Then** 返回 a 的 n 次幾何積冪次
14. **Given** 可逆多向量 a，**When** 使用 `a ** -1`，**Then** 返回逆元（等同 `a.inverse()`）
15. **Given** Bivector B，**When** 使用 `B.exp()`，**Then** 返回指數映射 EvenVersor `exp(B)`

---

### User Story 11 - Unified Layer Naming (Priority: P2)

開發者需要一致的 Layer 命名，不論維度和代數類型都使用相同的類別名稱。移除 CARE 論文特定的命名（如 `CGA3DCareLayer`），改為通用的 `CliffordTransformLayer`。

**Why this priority**: 當前命名過於強調 CARE 論文和 CGA，但這是通用的 Clifford Algebra 運算。統一命名提升 API 一致性和可讀性。

**Independent Test**: 可透過 `from fast_clifford import CliffordTransformLayer` 驗證統一名稱可用。

**Acceptance Scenarios**:

1. **Given** 任意 Clifford Algebra，**When** 使用 `algebra.get_transform_layer()`，**Then** 返回對應的 `CliffordTransformLayer` 實例
2. **Given** CGA 代數實例，**When** 使用 `cga.get_transform_layer(versor_type='similitude')`，**Then** 返回使用 Similitude 加速的 Layer
3. **Given** 任意維度，**When** 從 `fast_clifford` 匯入 `CliffordTransformLayer`，**Then** 可直接使用
4. **Given** CGA 代數實例，**When** 使用 `CGAEncoder` 編碼歐氏座標，**Then** 返回正確形狀的 CGA 點表示
5. **Given** CGA 代數實例，**When** 使用 `CGADecoder` 解碼 CGA 點，**Then** 返回正確形狀的歐氏座標
6. **Given** CGA 代數實例，**When** 使用 `CGAPipeline` 執行完整變換，**Then** 輸入輸出維度一致且變換正確

---

### Edge Cases

- **零向量輸入**: inner_product(0, 0) 應返回 0，exp_bivector(0) 應返回單位 EvenVersor
- **極小角度**: exp_bivector 對 θ < 1e-10 應數值穩定（使用 sinc 或 Taylor 展開）
- **非正規化 EvenVersor**: even_versor_compose 對未正規化的 EvenVersor 仍應正確計算
- **混合精度**: 支援 float32 和 float64 輸入
- **批次維度**: 所有操作支援任意 batch 形狀 (..., component_count)
- **零向量正規化**: normalize(0) 應返回零向量而非 NaN
- **無效 Grade**: grade_select 對超出範圍的 Grade 應返回零向量
- **自楔積**: outer_product(v, v) 對任意 v 應返回 0
- **不可逆多向量**: inverse() 對 null vector 或零向量 MUST 返回全 NaN 張量
- **單位元逆元**: 標量 1 的逆元應為 1
- **Similitude 邊界**: similitude_compose 對包含 transversion 成分的輸入 MUST 拋出 `ValueError("Input contains transversion components")`

## Requirements *(mandatory)*

### Functional Requirements

#### 統一 API (靜態分派)

- **FR-001**: 系統 MUST 提供 `compose(v1, v2)` 統一函式，根據輸入類型靜態路由到最佳實作
- **FR-001a**: 系統 MUST 提供 `sandwich_product(v, x)` 統一函式，根據輸入類型靜態路由到最佳實作
- **FR-001b**: 系統 MUST 提供 `reverse(v)` 統一函式，根據輸入類型靜態路由到最佳實作
- **FR-001c**: 靜態路由 MUST 在 Python 圖構建時決定（非運行時），確保 ONNX 匯出無 If 節點

#### EvenVersor 底層實作 (通用 Clifford Algebra)

- **FR-002**: 系統 MUST 提供 `compose_even_versor(v1, v2)` 底層函式，計算兩個偶數 Versor 的幾何積
- **FR-003**: 輸入輸出格式 MUST 為稀疏 EvenVersor 表示 (even_versor_count 分量)
- **FR-004**: 對於 n≤5，系統 MUST 使用硬編碼展開實作（無迴圈）
- **FR-004a**: 對於 n≥6，系統 MUST 使用運行時一般化算法
- **FR-004b**: 系統 MUST 提供 `sandwich_product_even_versor(v, x)` 底層函式，計算 `v * x * ~v`
- **FR-004c**: 系統 MUST 提供 `reverse_even_versor(v)` 底層函式，計算 EvenVersor 反向
- **FR-004d**: 系統 MUST 提供 `even_versor_count` 屬性，返回 EvenVersor 分量數

#### Similitude 底層實作 (CGA 專用加速)

- **FR-004e**: 系統 MUST 提供 `compose_similitude(s1, s2)` 底層函式，計算兩個 Similitude 的幾何積（更激進加速）
- **FR-004f**: 系統 MUST 提供 `sandwich_product_similitude(s, x)` 底層函式，計算 Similitude 三明治積（更激進加速）
- **FR-004g**: 系統 MUST 提供 `reverse_similitude(s)` 底層函式，計算 Similitude 反向
- **FR-004h**: 系統 MUST 提供 `similitude_count` 屬性，返回 Similitude 分量數（比 even_versor_count 更少）
- **FR-004i**: Similitude 底層函式 SHOULD 比對應的 EvenVersor 函式效能提升 30-50%

#### 靜態路由規則

- **FR-004j**: `compose(Similitude, Similitude)` MUST 路由到 `compose_similitude`
- **FR-004k**: `compose(EvenVersor, EvenVersor)` MUST 路由到 `compose_even_versor`
- **FR-004l**: `compose(Similitude, EvenVersor)` MUST 路由到 `compose_even_versor`（類型降級）
- **FR-004m**: `compose(Multivector, Multivector)` MUST 路由到 `geometric_product_full`
- **FR-004n**: `sandwich_product(Similitude, *)` MUST 路由到 `sandwich_product_similitude`
- **FR-004o**: `sandwich_product(EvenVersor, *)` MUST 路由到 `sandwich_product_even_versor`

#### Geometric Inner Product

- **FR-005**: 系統 MUST 提供 `inner_product(a, b)` 函式，計算兩個多向量的標量內積
- **FR-006**: 內積計算 MUST 正確處理度規符號
- **FR-007**: 實作 MUST 使用符號融合優化（`sum(a[i] * b[i] * sign[i])`）而非分步計算
- **FR-008**: 輸出 MUST 為形狀 (..., 1) 的標量張量

#### Exponential Map

- **FR-009**: 系統 MUST 提供 `exp_bivector(B)` 函式，從 Bivector 生成 EvenVersor
- **FR-010**: 系統 MUST 處理數值穩定性，對 θ→0 使用 sinc 或 Taylor 展開
- **FR-011**: 輸入 MUST 為稀疏 Bivector 表示（Grade 2 分量）
- **FR-012**: 輸出 MUST 為稀疏 EvenVersor 表示

#### Outer Product (Wedge Product)

- **FR-013**: 系統 MUST 提供 `outer_product(a, b)` 函式，計算兩個多向量的外積
- **FR-014**: 外積計算 MUST 返回幾何積中 Grade |a|+|b| 的分量
- **FR-015**: 對於 n≤5，系統 MUST 使用硬編碼展開實作

#### Left/Right Contraction

- **FR-016**: 系統 MUST 提供 `left_contraction(a, b)` 函式，計算左縮併
- **FR-017**: 系統 MUST 提供 `right_contraction(a, b)` 函式，計算右縮併
- **FR-018**: 縮併運算 MUST 返回幾何積中 Grade ||b|-|a|| 的分量

#### Grade Selection

- **FR-019**: 系統 MUST 提供 `grade_select(mv, k)` 函式，提取特定 Grade 分量
- **FR-020**: 對於無效 Grade（k > max_grade 或 k < 0），MUST 返回零向量
- **FR-021**: 輸出 MUST 為完整多向量格式（blade_count 分量）

#### Dual

- **FR-022**: 系統 MUST 提供 `dual(mv)` 函式，計算多向量的對偶
- **FR-023**: 對偶計算 MUST 使用 Pseudoscalar：`dual(a) = a * I^(-1)`

#### Normalize

- **FR-024**: 系統 MUST 提供 `normalize(mv)` 函式，正規化多向量為單位範數
- **FR-025**: 對於零向量輸入，MUST 返回零向量（不會產生 NaN）
- **FR-026**: 正規化 MUST 使用幾何內積計算範數

#### Structure Normalize (Similitude 專用)

- **FR-026a**: 系統 MUST 提供 `structure_normalize(similitude)` 函式，對 Similitude 進行結構正規化
- **FR-026b**: 結構正規化 MUST 包含 Rotor 正規化（保持旋轉為單位四元數）
- **FR-026c**: 結構正規化 MUST 強制 Similitude 約束 `ei+ = ei-`（排除 transversion）
- **FR-026d**: 結構正規化 SHOULD 為 ONNX 相容（無迴圈、無條件分支）
- **FR-026e**: 系統 SHOULD 提供 `soft_structure_normalize(similitude, strength)` 軟性正規化變體
- **FR-026f**: 系統 SHOULD 提供 `structure_normalize_ste(similitude)` STE 變體（梯度穿透）

#### Operator Overloading

- **FR-027**: 系統 MUST 提供 `Multivector` 包裝類別，封裝張量、代數實例和可選的類型標記 (`kind`)
- **FR-028**: `Multivector` MUST 實作 `__mul__` 運算子，對應幾何積 `a * b`
- **FR-029**: `Multivector` MUST 實作 `__xor__` 運算子，對應楔積 `a ^ b`
- **FR-030**: `Multivector` MUST 實作 `__or__` 運算子，對應內積 `a | b`
- **FR-031**: `Multivector` MUST 實作 `__lshift__` 運算子，對應左縮併 `a << b`
- **FR-032**: `Multivector` MUST 實作 `__rshift__` 運算子，對應右縮併 `a >> b`
- **FR-033**: `Multivector` MUST 實作 `__matmul__` 運算子，對應三明治積 `m @ x` = `m * x * ~m`
- **FR-034**: `Multivector` MUST 實作 `__add__` 和 `__sub__` 運算子，對應加減法
- **FR-035**: `Multivector` MUST 實作 `__neg__` 運算子，對應取負 `-a`
- **FR-036**: `Multivector` MUST 實作 `__invert__` 運算子，對應反向 `~a`
- **FR-037**: `Multivector` MUST 實作 `__rmul__` 運算子，支援標量左乘 `s * a`
- **FR-038**: `Multivector` MUST 實作 `__truediv__` 運算子，支援標量除法 `a / s` 和多向量除法 `a / b`
- **FR-039**: `Multivector` MUST 實作 `__pow__` 運算子，支援整數冪次 `a ** n` 和逆元 `a ** -1`
- **FR-040**: `Multivector` MUST 實作 `inverse()` 方法，計算多向量逆元 `a^(-1) = ~a / (a * ~a)`
- **FR-041**: `Multivector` MUST 實作 `exp()` 方法，對 Bivector 計算指數映射
- **FR-042**: 多向量除法 `a / b` MUST 等價於 `a * b.inverse()`
- **FR-043**: 對於不可逆多向量（`a * ~a == 0`），`inverse()` MUST 返回全 NaN 張量（ONNX 相容，避免控制流）
- **FR-044**: 所有運算子 MUST 支援 PyTorch autograd（可微分）
- **FR-045**: 所有運算子 MUST 支援任意 batch 維度

#### 類型標記與工廠方法

- **FR-046**: `Multivector` MUST 支援 `kind` 屬性，可選值為 `None`、`'even_versor'`、`'similitude'`、`'point'`、`'bivector'` 等
- **FR-047**: CliffordAlgebraBase MUST 提供 `even_versor(tensor)`、`point(tensor)`、`bivector(tensor)` 工廠方法
- **FR-047a**: CGAAlgebraBase MUST 額外提供 `similitude(tensor)` 工廠方法
- **FR-048**: 運算子 `*` 對 `kind='even_versor'` SHOULD 內部呼叫 `compose()` 統一 API
- **FR-048a**: 運算子 `*` 對 `kind='similitude'` SHOULD 內部呼叫 `compose()` 統一 API
- **FR-049**: 運算子 `@` SHOULD 內部呼叫 `sandwich_product()` 統一 API
- **FR-050**: 統一 API 的靜態路由 MUST 在 Python 圖構建時決定（非運行時），確保 ONNX 匯出無 If 節點
- **FR-051**: 未標記類型 (`kind=None`) 的多向量 MUST 使用 full 版本函式（保證正確性）

#### ONNX 相容性策略

- **FR-052**: Multivector 運算子 SHOULD 優先使用 full 版本函式，確保 ONNX 相容
- **FR-053**: 生產環境和 ONNX 匯出 SHOULD 直接使用 functional API（如 `even_versor_compose`、`similitude_compose`）而非 Multivector 類別
- **FR-054**: 文檔 MUST 清楚說明：運算子適合原型開發，functional API 適合生產部署

#### 統一介面

- **FR-055**: 所有新函式 MUST 加入 CliffordAlgebraBase 抽象類別（通用操作）或 CGAAlgebraBase（CGA 專用）
- **FR-056**: HardcodedCGAWrapper MUST 對 n=0-5 委派至硬編碼實作
- **FR-057**: RuntimeCGAAlgebra MUST 對 n≥6 提供一般化實作

#### ONNX 相容性（硬編碼實作）

- **FR-058**: 所有硬編碼實作 MUST 可匯出為無 Loop/If 節點的 ONNX 模型
- **FR-059**: 運行時實作 SHOULD 盡可能支援 ONNX 匯出

#### PyTorch 整合

- **FR-060**: 所有操作 MUST 支援 PyTorch autograd（可微分）
- **FR-061**: 所有操作 MUST 支援任意 batch 維度

#### Layer 統一命名

- **FR-062**: 系統 MUST 提供統一的 `CliffordTransformLayer` 類別，取代各維度的 `CGA{n}DCareLayer`
- **FR-063**: 系統 MUST 提供統一的 `CGAEncoder` 和 `CGADecoder` 類別，取代 `UPGC{n}DEncoder/Decoder`
- **FR-064**: 系統 MUST 提供統一的 `CGAPipeline` 類別，取代 `CGA{n}DTransformPipeline`
- **FR-065**: CliffordAlgebraBase MUST 提供 `get_transform_layer()` 方法，取代 `get_care_layer()`
- **FR-065a**: `get_transform_layer()` MUST 支援 `versor_type` 參數，可選 `'even_versor'`（預設）或 `'similitude'`（CGA 專用）
- **FR-066**: 統一命名 MUST 適用於所有維度（包含運行時 n≥6）
- **FR-067**: 舊的維度特定 Layer 類別 MUST 移除（不向後相容）

### Key Entities

- **Versor**: 多向量的子集，可透過基向量的幾何積表示，用於各種變換
  - **EvenVersor**: 偶數 Grade 多向量 (Grade 0 + Grade 2 + Grade 4 + ...)，用於旋轉等保向變換（= `Versor(order='even')`）
  - **OddVersor**: 奇數 Grade 多向量，用於反射等變換（= `Versor(order='odd')`）
- **Similitude** (CGA 專用): EvenVersor 的子類別，僅包含平移 + 旋轉 + 縮放，排除 transversion，具有更稀疏的結構
- **Bivector**: Grade 2 多向量，用於表示旋轉軸/平面
- **Multivector**: 包裝類別，封裝張量與代數實例，提供運算子重載
- **Metric Signature**: Clifford Algebra 度規，定義內積的符號規則（如 CGA 為 (+,+,...,+,-)）
- **CliffordTransformLayer**: 統一的 PyTorch Layer，執行 Versor sandwich product 變換
- **CGAEncoder**: 統一的 UPGC 編碼器，將歐氏座標轉換為 CGA 點表示
- **CGADecoder**: 統一的 UPGC 解碼器，將 CGA 點表示轉換回歐氏座標
- **CGAPipeline**: 統一的變換管線，組合 Encoder → Transform → Decoder

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有新操作對 n=0-5 的硬編碼實作，效能至少達到完整幾何積的 50%（因為只計算部分分量）
- **SC-001a**: Similitude 操作效能比對應 EvenVersor 操作提升 30-50%
- **SC-002**: 所有操作對 clifford 庫的數值誤差小於 1e-6（float32）或 1e-10（float64）
- **SC-003**: exp_bivector 對極小角度（θ < 1e-10）數值穩定，無 NaN 或 Inf
- **SC-004**: normalize 對零向量不產生 NaN，返回零向量
- **SC-005**: 所有硬編碼實作可匯出為 ONNX 模型，且無 Loop 或 If 節點
- **SC-006**: 測試覆蓋率達到 90% 以上，包含邊界情況和數值穩定性測試
- **SC-007**: API 使用方式與現有 sandwich_product_even_versor 一致，學習成本低
- **SC-008**: 統一 Layer 命名後，所有 Clifford Algebra 使用相同類別名稱（CliffordTransformLayer 等）
- **SC-009**: 舊的維度特定 Layer 類別完全移除
- **SC-010**: outer_product(v, v) 對任意 v 返回 0
- **SC-011**: 運算子重載 `a * b` 與 `geometric_product(a, b)` 數值等價
- **SC-012**: 運算子重載 `a ^ b` 與 `outer_product(a, b)` 數值等價
- **SC-013**: 運算子重載 `a | b` 與 `inner_product(a, b)` 數值等價
- **SC-014**: 運算子使用符合幾何代數慣例（`*` 幾何積、`^` 楔積、`|` 內積、`<<` `>>` 縮併、`@` 三明治積、`/` 除法、`**` 冪次）
- **SC-015**: `a * a.inverse()` 對可逆多向量返回近似標量 1
- **SC-016**: `a / b` 等價於 `a * b.inverse()`
- **SC-017**: `v @ x` 對 EvenVersor v 和 Point x，等價於 `sandwich_product_even_versor(v, x)`
- **SC-017a**: `s @ x` 對 Similitude s 和 Point x，等價於 `sandwich_product_similitude(s, x)`
- **SC-018**: `a << b` 與 `left_contraction(a, b)` 數值等價
- **SC-019**: `a >> b` 與 `right_contraction(a, b)` 數值等價
- **SC-020**: `a ** n` 對整數 n 返回 n 次幾何積冪次
- **SC-021**: `B.exp()` 對 Bivector B 與 `exp_bivector(B)` 數值等價
- **SC-022**: 帶類型標記的 EvenVersor 相乘自動路由到 `even_versor_compose`（效能優化）
- **SC-022a**: 帶類型標記的 Similitude 相乘自動路由到 `similitude_compose`（更高效能）
- **SC-023**: 未標記類型的 Multivector 使用 full 版本保證正確性

## Assumptions

- 使用者已安裝 PyTorch 2.0+ 和 clifford 庫（用於測試對照）
- 硬編碼實作由 codegen 系統自動生成
- 運行時實作使用 scatter_add/gather 張量操作
- 度規符號預先計算並儲存為常數

## Background: 現有運算與架構

### 命名架構變更

本功能重構命名系統，從 CGA 特定命名改為通用 Clifford Algebra 命名：

| 舊名稱 | 新名稱 | 說明 |
|--------|--------|------|
| Motor | EvenVersor | 偶數 Grade Versor（通用） |
| motor_compose | even_versor_compose | EvenVersor 組合（通用） |
| motor_count | even_versor_count | EvenVersor 分量數（通用） |
| reverse_motor | reverse_even_versor | EvenVersor 反向（通用） |
| sandwich_product_sparse | sandwich_product_even_versor | EvenVersor 三明治積（通用） |
| - | Similitude | 平移+旋轉+縮放（CGA 專用子類別） |
| - | similitude_compose | Similitude 組合（CGA 專用加速） |
| - | similitude_count | Similitude 分量數（CGA 專用） |
| - | reverse_similitude | Similitude 反向（CGA 專用） |
| - | sandwich_product_similitude | Similitude 三明治積（CGA 專用加速） |
| CGATransformLayer | CliffordTransformLayer | 統一 PyTorch Layer |

### 已實作的運算

本功能建立在現有運算基礎上。以下運算已在所有維度實作：

| 運算 | 函式名稱 | 說明 |
|------|----------|------|
| 幾何積 | `geometric_product_full(a, b)` | 完整多向量幾何積 |
| 反向 | `reverse_full(mv)` | 多向量反向操作 |
| EvenVersor 反向 | `reverse_even_versor(v)` | EvenVersor 專用反向（原 reverse_motor） |
| 三明治積 | `sandwich_product_even_versor(v, x)` | V × X × V~ 變換（原 sandwich_product_sparse） |
| UPGC 編碼 | `upgc_encode(x)` | 歐氏座標 → CGA 點表示 |
| UPGC 解碼 | `upgc_decode(point)` | CGA 點表示 → 歐氏座標 |

### 本功能新增的運算

| 運算 | 狀態 | 說明 |
|------|------|------|
| EvenVersor 組合 | 🔨 本功能 | `even_versor_compose(v1, v2)` - 通用 |
| Similitude 組合 | 🔨 本功能 | `similitude_compose(s1, s2)` - CGA 加速 |
| Similitude 三明治積 | 🔨 本功能 | `sandwich_product_similitude(s, x)` - CGA 加速 |
| 幾何內積 | 🔨 本功能 | `inner_product(a, b)` - 度規內積 (Grade 0) |
| 指數映射 | 🔨 本功能 | `exp_bivector(B)` - Bivector → EvenVersor |
| 楔積 | 🔨 本功能 | `outer_product(a, b)` - a ∧ b |
| 左縮併 | 🔨 本功能 | `left_contraction(a, b)` - a ⌋ b |
| 右縮併 | 🔨 本功能 | `right_contraction(a, b)` - a ⌊ b |
| Grade 提取 | 🔨 本功能 | `grade_select(mv, k)` - ⟨a⟩_k |
| 對偶 | 🔨 本功能 | `dual(mv)` - a* |
| 正規化 | 🔨 本功能 | `normalize(mv)` - a / |a| |

### Similitude vs EvenVersor

```
EvenVersor (通用 Clifford Algebra):
├── 包含所有偶數 Grade 分量
├── CGA3D: 16 分量 (Grade 0 + Grade 2 + Grade 4)
└── 可表示: 旋轉、平移、縮放、transversion、及其組合

Similitude (CGA 專用):
├── EvenVersor 的子集
├── 排除 transversion 相關分量
├── CGA3D: 11 分量（比 EvenVersor 的 16 少 31%）
├── 可表示: 旋轉、平移、縮放
└── 更稀疏 → 更快計算（30-50% 提升）
```

### 運算關係

```
楔積:     a ∧ b = ⟨ab⟩_{|a|+|b|}     (Grade 提升)
左縮併:   a ⌋ b = ⟨ab⟩_{|b|-|a|}     (Grade 降低)
幾何內積: a · b = ⟨ab⟩_0             (本功能實作)
幾何積:   ab = a ∧ b + a ⌋ b + ...   (已實作)
```
