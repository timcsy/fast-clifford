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

### User Story 5 - Unified Layer Naming (Priority: P2)

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

#### 統一介面

- **FR-013**: 三個新函式 MUST 加入 CGAAlgebraBase 抽象類別
- **FR-014**: HardcodedCGAWrapper MUST 對 n=0-5 委派至硬編碼實作
- **FR-015**: RuntimeCGAAlgebra MUST 對 n≥6 提供一般化實作

#### ONNX 相容性

- **FR-016**: 所有硬編碼實作 MUST 可匯出為無 Loop/If 節點的 ONNX 模型
- **FR-017**: 運行時實作 SHOULD 盡可能支援 ONNX 匯出

#### PyTorch 整合

- **FR-018**: 所有操作 MUST 支援 PyTorch autograd（可微分）
- **FR-019**: 所有操作 MUST 支援任意 batch 維度

#### Layer 統一命名

- **FR-020**: 系統 MUST 提供統一的 `CGATransformLayer` 類別，取代各維度的 `CGA{n}DCareLayer`
- **FR-021**: 系統 MUST 提供統一的 `CGAEncoder` 和 `CGADecoder` 類別，取代 `UPGC{n}DEncoder/Decoder`
- **FR-022**: 系統 MUST 提供統一的 `CGAPipeline` 類別，取代 `CGA{n}DTransformPipeline`
- **FR-023**: CGAAlgebraBase MUST 提供 `get_transform_layer()` 方法，取代 `get_care_layer()`
- **FR-024**: 統一命名 MUST 適用於所有維度（包含運行時 n≥6）
- **FR-025**: 舊的維度特定 Layer 類別 MUST 移除（不向後相容）

### Key Entities

- **Motor**: 偶數 Grade 多向量 (Grade 0 + Grade 2 + Grade 4 + ...)，用於表示剛體變換
- **Bivector**: Grade 2 多向量，用於表示旋轉軸/平面
- **Multivector**: 完整 Clifford 代數元素，包含所有 Grade 分量
- **Metric Signature**: CGA 度規 (+,+,...,+,-)，定義內積的符號規則
- **CGATransformLayer**: 統一的 PyTorch Layer，執行 Motor sandwich product 變換
- **CGAEncoder**: 統一的 UPGC 編碼器，將歐氏座標轉換為 CGA 點表示
- **CGADecoder**: 統一的 UPGC 解碼器，將 CGA 點表示轉換回歐氏座標
- **CGAPipeline**: 統一的變換管線，組合 Encoder → Transform → Decoder

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 三個新操作對 n=0-5 的硬編碼實作，效能至少達到完整幾何積的 50%（因為只計算部分分量）
- **SC-002**: 所有操作對 clifford 庫的數值誤差小於 1e-6（float32）或 1e-10（float64）
- **SC-003**: exp_bivector 對極小角度（θ < 1e-10）數值穩定，無 NaN 或 Inf
- **SC-004**: 所有硬編碼實作可匯出為 ONNX 模型，且無 Loop 或 If 節點
- **SC-005**: 測試覆蓋率達到 90% 以上，包含邊界情況和數值穩定性測試
- **SC-006**: API 使用方式與現有 sandwich_product_sparse 一致，學習成本低
- **SC-007**: 統一 Layer 命名後，所有維度使用相同類別名稱（CGATransformLayer 等）
- **SC-008**: 舊的維度特定 Layer 類別完全移除

## Assumptions

- 使用者已安裝 PyTorch 2.0+ 和 clifford 庫（用於測試對照）
- 硬編碼實作由 codegen 系統自動生成
- 運行時實作使用 scatter_add/gather 張量操作
- 度規符號預先計算並儲存為常數
