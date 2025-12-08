# Feature Specification: CGA Extended Operations

**Feature Branch**: `005-cga-extended-ops`
**Created**: 2025-12-08
**Status**: Draft
**Input**: User description: "新增 Motor Composition、Geometric Inner Product、Exponential Map 三個 CGA 操作，5D 內硬編碼加速，6D 以上運行時一般化算法"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Motor Composition for Transform Chaining (Priority: P1)

開發者需要組合多個幾何變換（如先旋轉再平移），透過 Motor Composition 將兩個馬達合併為單一馬達，用於機器人運動學、3D 動畫等應用。

**Why this priority**: Motor Composition 是最基礎的變換組合操作，幾乎所有複雜幾何變換都需要將多個操作串接。沒有此功能，使用者必須多次執行 sandwich product，效能較差。

**Independent Test**: 可透過建立兩個已知變換馬達（如 90° 旋轉 + 平移），組合後驗證等同於單一複合變換的效果。

**Acceptance Scenarios**:

1. **Given** 兩個 CGA3D 馬達 M1（旋轉）和 M2（平移），**When** 呼叫 `motor_compose(M1, M2)`，**Then** 返回正確組合的馬達 M_result
2. **Given** 單位馬達 identity，**When** 呼叫 `motor_compose(identity, M)`，**Then** 返回 M 本身
3. **Given** 馬達 M 及其逆元 M_rev，**When** 呼叫 `motor_compose(M, M_rev)`，**Then** 返回近似單位馬達

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

### User Story 3 - Exponential Map for Motor Generation (Priority: P2)

開發者需要從旋轉軸和角度生成旋轉馬達，透過 Bivector 的指數映射 exp(B) 產生 Rotor/Motor。此功能用於插值、平滑動畫、從李代數生成變換等場景。

**Why this priority**: 雖然使用者可直接建構馬達，但 Exponential Map 提供更直觀的數學介面，對於旋轉插值（slerp）和李代數運算至關重要。

**Independent Test**: 可透過將已知旋轉角度的 Bivector 傳入 exp_bivector，驗證產生的馬達是否正確旋轉點。

**Acceptance Scenarios**:

1. **Given** 零 Bivector B=0，**When** 呼叫 `exp_bivector(B)`，**Then** 返回單位馬達 (1, 0, 0, ...)
2. **Given** 代表 90° 旋轉的 Bivector，**When** 呼叫 `exp_bivector(B)`，**Then** 產生正確的旋轉馬達
3. **Given** 極小 Bivector (θ < 1e-6)，**When** 呼叫 `exp_bivector(B)`，**Then** 數值穩定地返回近似單位馬達（無 NaN 或 Inf）

---

### User Story 4 - High-Dimensional Runtime Support (Priority: P2)

開發者需要在 6D 及以上維度使用相同的 API，系統自動切換至運行時一般化算法，確保功能完整性。

**Why this priority**: 保持 API 一致性，讓高維度研究者無需學習不同介面。效能不是主要考量（高維度本來就較慢）。

**Independent Test**: 可透過 CGA(6) 呼叫三個新操作，驗證功能正確且無錯誤。

**Acceptance Scenarios**:

1. **Given** CGA(6) 代數實例，**When** 呼叫 `motor_compose`，**Then** 返回正確結果（與 clifford 庫對照）
2. **Given** CGA(7) 代數實例，**When** 呼叫 `inner_product`，**Then** 返回正確標量
3. **Given** CGA(6) 代數實例，**When** 呼叫 `exp_bivector`，**Then** 返回正確馬達

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
10. **Given** Motor m 和 Point/Multivector x，**When** 使用 `m @ x`，**Then** 返回三明治積 `m * x * ~m`
11. **Given** 可逆多向量 a 和 b，**When** 使用 `a / b`，**Then** 返回 `a * b^(-1)` 結果
12. **Given** 可逆多向量 a，**When** 使用 `a.inverse()`，**Then** 返回逆元 `a^(-1)`
13. **Given** 多向量 a 和整數 n，**When** 使用 `a ** n`，**Then** 返回 a 的 n 次幾何積冪次
14. **Given** 可逆多向量 a，**When** 使用 `a ** -1`，**Then** 返回逆元（等同 `a.inverse()`）
15. **Given** Bivector B，**When** 使用 `B.exp()`，**Then** 返回指數映射馬達 `exp(B)`

---

### User Story 11 - Unified Layer Naming (Priority: P2)

開發者需要一致的 Layer 命名，不論維度都使用相同的類別名稱。移除 CARE 論文特定的命名（如 `CGA3DCareLayer`），改為通用的統一名稱。

**Why this priority**: 當前命名過於強調 CARE 論文，但這是通用的幾何代數運算。統一命名提升 API 一致性和可讀性。

**Independent Test**: 可透過 `from fast_clifford import CGATransformLayer` 驗證統一名稱可用。

**Acceptance Scenarios**:

1. **Given** 任意維度 n=0-5，**When** 使用 `cga.get_transform_layer()`，**Then** 返回對應維度的 `CGATransformLayer` 實例
2. **Given** 運行時代數 n≥6，**When** 使用 `cga.get_transform_layer()`，**Then** 返回統一的 `CGATransformLayer` 實例
3. **Given** 任意維度，**When** 從 `fast_clifford` 匯入 `CGATransformLayer`，**Then** 可直接使用
4. **Given** 任意維度，**When** 使用 `CGAEncoder` 編碼歐氏座標，**Then** 返回正確形狀的 CGA 點表示
5. **Given** 任意維度，**When** 使用 `CGADecoder` 解碼 CGA 點，**Then** 返回正確形狀的歐氏座標
6. **Given** 任意維度，**When** 使用 `CGAPipeline` 執行完整變換，**Then** 輸入輸出維度一致且變換正確

---

### Edge Cases

- **零向量輸入**: inner_product(0, 0) 應返回 0，exp_bivector(0) 應返回單位馬達
- **極小角度**: exp_bivector 對 θ < 1e-10 應數值穩定（使用 sinc 或 Taylor 展開）
- **非正規化馬達**: motor_compose 對未正規化的馬達仍應正確計算
- **混合精度**: 支援 float32 和 float64 輸入
- **批次維度**: 所有操作支援任意 batch 形狀 (..., component_count)
- **零向量正規化**: normalize(0) 應返回零向量而非 NaN
- **無效 Grade**: grade_select 對超出範圍的 Grade 應返回零向量
- **自楔積**: outer_product(v, v) 對任意 v 應返回 0
- **不可逆多向量**: inverse() 對 null vector 或零向量應拋出錯誤或返回 NaN
- **單位元逆元**: 標量 1 的逆元應為 1

## Requirements *(mandatory)*

### Functional Requirements

#### Motor Composition

- **FR-001**: 系統 MUST 提供 `motor_compose(m1, m2)` 函式，計算兩個馬達的幾何積
- **FR-002**: 輸入輸出格式 MUST 為稀疏馬達表示 (motor_count 分量)
- **FR-003**: 對於 n≤5，系統 MUST 使用硬編碼展開實作（無迴圈）
- **FR-004**: 對於 n≥6，系統 MUST 使用運行時一般化算法

#### Geometric Inner Product

- **FR-005**: 系統 MUST 提供 `inner_product(a, b)` 函式，計算兩個多向量的標量內積
- **FR-006**: 內積計算 MUST 正確處理 CGA 度規符號 (+,+,...,+,-)
- **FR-007**: 實作 MUST 使用符號融合優化（`sum(a[i] * b[i] * sign[i])`）而非分步計算
- **FR-008**: 輸出 MUST 為形狀 (..., 1) 的標量張量

#### Exponential Map

- **FR-009**: 系統 MUST 提供 `exp_bivector(B)` 函式，從 Bivector 生成馬達
- **FR-010**: 系統 MUST 處理數值穩定性，對 θ→0 使用 sinc 或 Taylor 展開
- **FR-011**: 輸入 MUST 為稀疏 Bivector 表示（Grade 2 分量）
- **FR-012**: 輸出 MUST 為稀疏馬達表示

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
- **FR-043**: 對於不可逆多向量（`a * ~a == 0`），`inverse()` SHOULD 拋出 `ValueError` 或返回 NaN
- **FR-044**: 所有運算子 MUST 支援 PyTorch autograd（可微分）
- **FR-045**: 所有運算子 MUST 支援任意 batch 維度

#### 類型標記與靜態路由

- **FR-046**: `Multivector` MUST 支援 `kind` 屬性，可選值為 `None`、`'motor'`、`'point'`、`'bivector'` 等
- **FR-047**: CGAAlgebraBase MUST 提供 `motor(tensor)`、`point(tensor)`、`bivector(tensor)` 工廠方法，建立帶類型標記的 Multivector
- **FR-048**: 當 `kind='motor'` 的兩個多向量相乘時，SHOULD 靜態路由到 `motor_compose_sparse` 以優化效能
- **FR-049**: 當 `kind='motor'` 對 `kind='point'` 執行三明治積 (`@`) 時，SHOULD 靜態路由到 `sandwich_product_sparse`
- **FR-050**: 靜態路由 MUST 在 Python 圖構建時決定（非運行時），確保 ONNX 匯出無 If 節點
- **FR-051**: 未標記類型 (`kind=None`) 的多向量 MUST 使用 full 版本函式（保證正確性）

#### ONNX 相容性策略

- **FR-052**: Multivector 運算子 SHOULD 優先使用 full 版本函式，確保 ONNX 相容
- **FR-053**: 生產環境和 ONNX 匯出 SHOULD 直接使用 functional API（如 `motor_compose_sparse`）而非 Multivector 類別
- **FR-054**: 文檔 MUST 清楚說明：運算子適合原型開發，functional API 適合生產部署

#### 統一介面

- **FR-055**: 所有新函式 MUST 加入 CGAAlgebraBase 抽象類別
- **FR-056**: HardcodedCGAWrapper MUST 對 n=0-5 委派至硬編碼實作
- **FR-057**: RuntimeCGAAlgebra MUST 對 n≥6 提供一般化實作

#### ONNX 相容性（硬編碼實作）

- **FR-058**: 所有硬編碼實作 MUST 可匯出為無 Loop/If 節點的 ONNX 模型
- **FR-059**: 運行時實作 SHOULD 盡可能支援 ONNX 匯出

#### PyTorch 整合

- **FR-060**: 所有操作 MUST 支援 PyTorch autograd（可微分）
- **FR-061**: 所有操作 MUST 支援任意 batch 維度

#### Layer 統一命名

- **FR-062**: 系統 MUST 提供統一的 `CGATransformLayer` 類別，取代各維度的 `CGA{n}DCareLayer`
- **FR-063**: 系統 MUST 提供統一的 `CGAEncoder` 和 `CGADecoder` 類別，取代 `UPGC{n}DEncoder/Decoder`
- **FR-064**: 系統 MUST 提供統一的 `CGAPipeline` 類別，取代 `CGA{n}DTransformPipeline`
- **FR-065**: CGAAlgebraBase MUST 提供 `get_transform_layer()` 方法，取代 `get_care_layer()`
- **FR-066**: 統一命名 MUST 適用於所有維度（包含運行時 n≥6）
- **FR-067**: 舊的維度特定 Layer 類別 MUST 移除（不向後相容）

### Key Entities

- **Motor**: 偶數 Grade 多向量 (Grade 0 + Grade 2 + Grade 4 + ...)，用於表示剛體變換
- **Bivector**: Grade 2 多向量，用於表示旋轉軸/平面
- **Multivector**: 包裝類別，封裝張量與代數實例，提供運算子重載
- **Metric Signature**: CGA 度規 (+,+,...,+,-)，定義內積的符號規則
- **CGATransformLayer**: 統一的 PyTorch Layer，執行 Motor sandwich product 變換
- **CGAEncoder**: 統一的 UPGC 編碼器，將歐氏座標轉換為 CGA 點表示
- **CGADecoder**: 統一的 UPGC 解碼器，將 CGA 點表示轉換回歐氏座標
- **CGAPipeline**: 統一的變換管線，組合 Encoder → Transform → Decoder

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有新操作對 n=0-5 的硬編碼實作，效能至少達到完整幾何積的 50%（因為只計算部分分量）
- **SC-002**: 所有操作對 clifford 庫的數值誤差小於 1e-6（float32）或 1e-10（float64）
- **SC-003**: exp_bivector 對極小角度（θ < 1e-10）數值穩定，無 NaN 或 Inf
- **SC-004**: normalize 對零向量不產生 NaN，返回零向量
- **SC-005**: 所有硬編碼實作可匯出為 ONNX 模型，且無 Loop 或 If 節點
- **SC-006**: 測試覆蓋率達到 90% 以上，包含邊界情況和數值穩定性測試
- **SC-007**: API 使用方式與現有 sandwich_product_sparse 一致，學習成本低
- **SC-008**: 統一 Layer 命名後，所有維度使用相同類別名稱（CGATransformLayer 等）
- **SC-009**: 舊的維度特定 Layer 類別完全移除
- **SC-010**: outer_product(v, v) 對任意 v 返回 0
- **SC-011**: 運算子重載 `a * b` 與 `geometric_product(a, b)` 數值等價
- **SC-012**: 運算子重載 `a ^ b` 與 `outer_product(a, b)` 數值等價
- **SC-013**: 運算子重載 `a | b` 與 `inner_product(a, b)` 數值等價
- **SC-014**: 運算子使用符合幾何代數慣例（`*` 幾何積、`^` 楔積、`|` 內積、`<<` `>>` 縮併、`@` 三明治積、`/` 除法、`**` 冪次）
- **SC-015**: `a * a.inverse()` 對可逆多向量返回近似標量 1
- **SC-016**: `a / b` 等價於 `a * b.inverse()`
- **SC-017**: `m @ x` 對 Motor m 和 Point x，等價於 `sandwich_product(m, x)`
- **SC-018**: `a << b` 與 `left_contraction(a, b)` 數值等價
- **SC-019**: `a >> b` 與 `right_contraction(a, b)` 數值等價
- **SC-020**: `a ** n` 對整數 n 返回 n 次幾何積冪次
- **SC-021**: `B.exp()` 對 Bivector B 與 `exp_bivector(B)` 數值等價
- **SC-022**: 帶類型標記的 Motor 相乘自動路由到 `motor_compose_sparse`（效能優化）
- **SC-023**: 未標記類型的 Multivector 使用 full 版本保證正確性

## Assumptions

- 使用者已安裝 PyTorch 2.0+ 和 clifford 庫（用於測試對照）
- 硬編碼實作由 codegen 系統自動生成
- 運行時實作使用 scatter_add/gather 張量操作
- 度規符號預先計算並儲存為常數

## Background: 現有 CGA 運算

### 已實作的運算

本功能建立在現有 CGA 運算基礎上。以下運算已在所有維度 (CGA0D-CGA5D + 運行時 6D+) 實作：

| 運算 | 函式名稱 | 說明 |
|------|----------|------|
| 幾何積 | `geometric_product_full(a, b)` | 完整多向量幾何積，輸入輸出皆為 blade_count 分量 |
| 反向 | `reverse_full(mv)` | 多向量反向操作，Grade k 乘以 (-1)^(k(k-1)/2) |
| 馬達反向 | `reverse_motor(motor)` | 馬達專用反向，稀疏表示 |
| 三明治積 | `sandwich_product_sparse(motor, point)` | M × X × M~ 變換，用於點的剛體變換 |
| UPGC 編碼 | `upgc_encode(x)` | 歐氏座標 → CGA 點表示 |
| UPGC 解碼 | `upgc_decode(point)` | CGA 點表示 → 歐氏座標 |

### 尚未實作的運算

以下基礎 Clifford 代數運算尚未獨立實作（本功能將新增前三項）：

| 運算 | 狀態 | 說明 |
|------|------|------|
| 馬達組合 | 🔨 本功能 | `motor_compose(m1, m2)` - Motor × Motor |
| 幾何內積 | 🔨 本功能 | `inner_product(a, b)` - 度規內積 (Grade 0) |
| 指數映射 | 🔨 本功能 | `exp_bivector(B)` - Bivector → Motor |
| 楔積 | 🔨 本功能 | `outer_product(a, b)` - a ∧ b - Outer Product |
| 左縮併 | 🔨 本功能 | `left_contraction(a, b)` - a ⌋ b - Left Contraction |
| 右縮併 | 🔨 本功能 | `right_contraction(a, b)` - a ⌊ b - Right Contraction |
| Grade 提取 | 🔨 本功能 | `grade_select(mv, k)` - ⟨a⟩_k - 提取特定 Grade 分量 |
| 對偶 | 🔨 本功能 | `dual(mv)` - a* - Dual |
| 正規化 | 🔨 本功能 | `normalize(mv)` - a / |a| - Normalize |

**注意**：加法/減法直接使用 PyTorch 張量運算 (`+`/`-`) 即可，無需額外實作。

### 運算關係

```
楔積:     a ∧ b = ⟨ab⟩_{|a|+|b|}     (Grade 提升)
左縮併:   a ⌋ b = ⟨ab⟩_{|b|-|a|}     (Grade 降低)
幾何內積: a · b = ⟨ab⟩_0             (本功能實作)
幾何積:   ab = a ∧ b + a ⌋ b + ...   (已實作)
```
