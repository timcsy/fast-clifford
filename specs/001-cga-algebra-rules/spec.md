# 功能規格書：CGA 幾何代數規則定義

**功能分支**: `001-cga-algebra-rules`
**建立日期**: 2025-12-05
**狀態**: Draft
**輸入**: 定義 CGA ($Cl(4,1)$) 的幾何代數規則與稀疏性假設，以便程式碼生成器使用

## 使用者情境與測試 *(必填)*

### User Story 1 - 生成器讀取代數規則 (Priority: P1)

作為程式碼生成器腳本，我需要讀取完整的 CGA 代數規則定義，以便生成硬編碼的幾何積函式。

**優先級理由**: 這是整個 CGA-CARE 專案的基礎。沒有正確的代數規則定義，無法生成任何幾何積程式碼。

**獨立測試**: 可透過驗證生成器輸出的乘法結果與已知 CGA 計算結果相符來測試。

**驗收情境**:

1. **Given** 完整的基底向量定義與度規簽名，**When** 生成器查詢任意兩個基底 blade 的乘積，**Then** 返回正確的結果 blade 與符號
2. **Given** Null Basis 定義 ($n_o$, $n_\infty$)，**When** 計算 $n_o \cdot n_\infty$，**Then** 返回 $-1$（根據 CGA 約定）
3. **Given** 32 個基底 blade 的索引映射，**When** 查詢任意 blade 索引，**Then** 返回對應的基底向量組合

---

### User Story 2 - 稀疏性假設應用 (Priority: P1)

作為程式碼生成器，我需要知道輸入 $X$ (UPGC 點) 和變換算子 $M$ (Motor) 的稀疏性假設，以便只生成非零項的計算程式碼。

**優先級理由**: 稀疏性假設是效能優化的核心。完整的 32×32 乘法有 1024 項，但利用稀疏性可大幅減少到數十項。

**獨立測試**: 可透過計算 $M \times X \times \widetilde{M}$ 的結果並驗證只有預期的 grade 有非零值來測試。

**驗收情境**:

1. **Given** UPGC 點 $X = n_o + x + 0.5|x|^2 n_\infty$ 的稀疏性定義，**When** 查詢 $X$ 的非零 blade 索引，**Then** 返回 Grade 1 中的 5 個分量索引
2. **Given** Motor $M$ 的稀疏性定義，**When** 查詢 $M$ 的非零 blade 索引，**Then** 返回 Grade 0, 2, 4 的分量索引（共 16 個）
3. **Given** 三明治積 $X_{new} = M \times X \times \widetilde{M}$ 的公式，**When** 分析輸出稀疏性，**Then** 確認 $X_{new}$ 只有 Grade 1 有非零值（5 個分量）

---

### User Story 3 - Reverse 操作定義 (Priority: P2)

作為程式碼生成器，我需要 Reverse ($\widetilde{M}$) 操作的符號規則定義，以便正確計算三明治積的第二個乘數。

**優先級理由**: Reverse 是三明治積的必要操作，但相對簡單（只是符號翻轉）。

**獨立測試**: 可透過驗證 $\widetilde{M}$ 的每個 blade 係數符號是否正確來測試。

**驗收情境**:

1. **Given** Grade $k$ 的 blade，**When** 計算 Reverse，**Then** 係數乘以 $(-1)^{k(k-1)/2}$
2. **Given** Motor $M$（偶數 grade），**When** 計算 $\widetilde{M}$，**Then** Grade 0 和 4 符號不變，Grade 2 符號反轉

---

### Edge Cases

- 當 3D 向量 $x = 0$ 時，UPGC 點 $X$ 退化為原點 $n_o$
- 當 Motor $M$ 為單位元（純 scalar = 1）時，三明治積結果應等於輸入
- 當 Motor $M$ 為純旋轉時（無平移），$X$ 的 $n_o$ 和 $n_\infty$ 分量不變
- 數值精度：在 float32 下，$|x|^2$ 的計算範圍限制

## 需求 *(必填)*

### 功能需求

#### 代數空間定義

- **FR-001**: 系統必須定義 $Cl(4,1)$ Clifford 代數，具有 5 個基底向量
- **FR-002**: 系統必須使用度規簽名 $(+,+,+,+,-)$，對應 $e_1^2 = e_2^2 = e_3^2 = e_+^2 = 1$，$e_-^2 = -1$
- **FR-003**: 系統必須定義 Null Basis：$n_o = \frac{1}{2}(e_- - e_+)$（原點），$n_\infty = e_- + e_+$（無窮遠點）
- **FR-004**: 系統必須提供 32 個基底 blade 的標準索引順序（按 grade 排列）

#### Blade 索引映射

- **FR-005**: 系統必須定義 Grade 0（scalar）的索引：索引 0
- **FR-006**: 系統必須定義 Grade 1（vector）的索引：$e_1, e_2, e_3, e_+, e_-$ 對應索引 1-5
- **FR-007**: 系統必須定義 Grade 2（bivector）的索引：10 個分量對應索引 6-15
- **FR-008**: 系統必須定義 Grade 3（trivector）的索引：10 個分量對應索引 16-25
- **FR-009**: 系統必須定義 Grade 4（quadvector）的索引：5 個分量對應索引 26-30
- **FR-010**: 系統必須定義 Grade 5（pseudoscalar）的索引：索引 31

#### 幾何積規則

- **FR-011**: 系統必須為所有 32×32 = 1024 個 blade 配對定義幾何積結果
- **FR-012**: 每個幾何積結果必須包含：結果 blade 索引、符號（+1 或 -1）
- **FR-013**: 幾何積必須滿足結合律：$(a \cdot b) \cdot c = a \cdot (b \cdot c)$

#### 稀疏性假設

- **FR-014**: 系統必須定義 UPGC 點 $X$ 的稀疏模式：只有 Grade 1 的 5 個分量非零
- **FR-015**: 系統必須定義 Motor $M$ 的稀疏模式：只有 Grade 0, 2, 4 的 16 個分量非零
- **FR-016**: 系統必須定義三明治積輸出 $X_{new}$ 的稀疏模式：只有 Grade 1 的 5 個分量非零

#### Reverse 操作

- **FR-017**: 系統必須定義 Reverse 操作的符號規則：Grade $k$ 乘以 $(-1)^{k(k-1)/2}$
- **FR-018**: 系統必須提供 32 個 blade 的 Reverse 符號查詢表

### 關鍵實體

- **Blade**: 基底多重向量元素，由基底向量的外積組成，具有索引、grade、符號屬性
- **Multivector**: 32 個 blade 係數的集合，表示為張量的最後一個維度
- **GeometricProductRule**: 定義兩個 blade 相乘的結果，包含輸入索引對、輸出索引、符號
- **SparsityMask**: 定義特定類型 multivector 的非零 blade 索引集合

## 成功標準 *(必填)*

### 可量測成果

- **SC-001**: 生成器輸出的幾何積函式計算結果與參考實現（如 Python clifford 庫）相符，誤差 < 1e-6
- **SC-002**: 利用稀疏性假設，三明治積 $M \times X \times \widetilde{M}$ 的計算量從 2048 次乘法減少到 200 次以下
- **SC-003**: 生成的程式碼通過 ONNX 匯出驗證，且計算圖不包含 Loop 節點
- **SC-004**: 代數規則定義文件可被生成器腳本直接解析，無需人工介入

## 假設與限制

### 假設

- 輸入 $X$ 始終為有效的 UPGC 點（由 3D 向量映射而來）
- Motor $M$ 始終為正規化的（$M \widetilde{M} = 1$）
- 所有計算在 float32 精度下進行
- 基底向量順序遵循 $e_1, e_2, e_3, e_+, e_-$ 的標準約定

### 限制

- 本規格僅定義代數規則，不涉及具體程式碼實現
- Null Basis 約定採用 $n_o \cdot n_\infty = -1$（部分文獻使用 $+1$，需注意一致性）

## 術語表

| 術語 | 全稱 | 定義 |
|------|------|------|
| **CGA** | Conformal Geometric Algebra | 共形幾何代數，將 3D 歐幾里得空間嵌入到 5D 空間以統一處理旋轉、平移、縮放等變換 |
| **UPGC** | Up-Projected Geometric Conformal | 上投影保形幾何點，將 3D 點 $x$ 嵌入 CGA 的標準表示：$X = n_o + x + \frac{1}{2}\|x\|^2 n_\infty$ |
| **Motor** | — | 偶子代數元素，表示旋轉和平移的組合變換，只有 Grade 0, 2, 4 有非零值 |
| **Blade** | — | 基底多重向量，由基底向量的外積組成 |
| **Null Basis** | — | CGA 特有的 null 向量對 ($n_o$, $n_\infty$)，滿足 $n_o^2 = n_\infty^2 = 0$ |
| **Sandwich Product** | — | 三明治積，$X_{new} = M \times X \times \widetilde{M}$，用於對點施加變換 |
| **Reverse** | — | 逆運算 $\widetilde{M}$，對 Grade $k$ 乘以 $(-1)^{k(k-1)/2}$ |
