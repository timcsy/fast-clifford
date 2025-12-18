# Feature Specification: Unified Cl(p,q,0) Codegen System

**Feature Branch**: `006-unified-clifford-codegen`
**Created**: 2025-12-18
**Status**: Draft
**Input**: 統一 Cl(p,q,r) 在 r=0 時的 codegen，CGA、VGA 作為特例，善用 Bott 週期性

## Clarifications

### Session 2025-12-18

- Q: CGA null basis 定義使用哪種慣例？ → A: Dorst 慣例 `e_o = (e_- - e_+)/2`, `e_inf = e_- + e_+`（與 clifford 庫相容）
- Q: 高維度代數是否有記憶體使用上限？ → A: 無硬性限制，但 blade_count > 2^14 (16384) 時輸出警告
- Q: 自動生成的代數模組應該在什麼時候生成？ → A: 預生成並提交到 git repo（安裝後即可使用）
- Q: 應該預生成哪些代數？ → A: 所有 p+q ≤ 9 的組合（約 55 個代數）

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 使用 VGA 進行基本向量運算 (Priority: P1)

開發者希望使用 VGA(3) = Cl(3,0) 進行 3D 向量和旋轉運算，期望獲得與 CGA 相同的效能和 API 風格。

**Why this priority**: VGA 是最基礎的幾何代數，許多應用只需要純向量空間運算，不需要共形擴展。

**Independent Test**: 可以獨立測試 VGA(3) 的幾何積、外積、sandwich product，對照 clifford 庫驗證正確性。

**Acceptance Scenarios**:

1. **Given** `vga = VGA(3)`, **When** 計算 `vga.geometric_product(a, b)`, **Then** 結果與 clifford 庫一致
2. **Given** 兩個 rotor `r1, r2`, **When** 計算 `vga.compose_rotor(r1, r2)`, **Then** 等價於 `geometric_product` 但只返回偶數 grade
3. **Given** 一個 bivector `B`, **When** 計算 `vga.exp_bivector(B)`, **Then** 返回對應的 rotor

---

### User Story 2 - 使用統一工廠創建任意 Clifford 代數 (Priority: P1)

開發者希望用統一的 `Cl(p, q)` 工廠函數創建任意簽名的 Clifford 代數，無需關心底層實作。

**Why this priority**: 統一介面是整個系統的核心，讓使用者無需關心硬編碼 vs Bott 週期性的差異。

**Independent Test**: 可以創建 Cl(3,0), Cl(4,1), Cl(2,2) 等不同代數，驗證它們都有相同的 API。

**Acceptance Scenarios**:

1. **Given** `algebra = Cl(4, 1)`, **When** 檢查 `algebra.count_blade`, **Then** 返回 32（2^5）
2. **Given** `algebra = Cl(9, 0)`, **When** 調用任何運算, **Then** 自動使用 Bott 週期性（blade_count > 512）
3. **Given** `algebra = Cl(3, 0)`, **When** 檢查 `algebra.algebra_type`, **Then** 返回 'vga'

---

### User Story 3 - CGA 特化運算（編解碼）(Priority: P1)

開發者使用 CGA(3) 進行 3D 共形幾何運算，需要歐幾里得座標與 CGA 點之間的轉換。

**Why this priority**: CGA 的核心價值在於共形座標系統，encode/decode 是必備功能。

**Independent Test**: 可以獨立測試 encode → transform → decode 的完整流程。

**Acceptance Scenarios**:

1. **Given** `cga = CGA(3)` 和歐幾里得點 `x = [1, 2, 3]`, **When** 計算 `cga.encode(x)`, **Then** 返回正確的 UPGC 點（n+2 維）
2. **Given** 一個 CGA 點 `p`, **When** 計算 `cga.decode(p)`, **Then** 返回原始歐幾里得座標
3. **Given** `CGA(3)`, **When** 檢查 `cga.dim_euclidean`, **Then** 返回 3

---

### User Story 4 - Rotor 加速運算 (Priority: P2)

開發者頻繁使用 rotor 進行變換組合和 sandwich product，期望獲得優於通用 multivector 運算的效能。

**Why this priority**: Rotor 是最常用的變換表示，加速運算直接影響應用效能。

**Independent Test**: 可以比較 `compose_rotor` vs `geometric_product` 的結果和效能。

**Acceptance Scenarios**:

1. **Given** 兩個 rotor `r1, r2`, **When** 計算 `compose_rotor(r1, r2)`, **Then** 結果等價於 `geometric_product` 的偶數 grade 部分
2. **Given** rotor `r` 和點 `x`, **When** 計算 `sandwich_rotor(r, x)`, **Then** 等價於 `sandwich(r, x)` 但更快
3. **Given** 大量 rotor 組合運算, **When** 使用 `compose_rotor`, **Then** 效能優於 `geometric_product`

---

### User Story 5 - 高維度代數（Bott 週期性）(Priority: P2)

開發者需要使用高維度 Clifford 代數（如 Cl(10,0)），期望系統自動利用 Bott 週期性提供高效實作。

**Why this priority**: 支援任意維度是系統完整性的要求，但大多數應用使用低維度。

**Independent Test**: 可以創建 Cl(10,0) 並執行基本運算，驗證結果正確性。

**Acceptance Scenarios**:

1. **Given** `algebra = Cl(10, 0)`, **When** 檢查實作類型, **Then** 使用 BottPeriodicityAlgebra
2. **Given** Cl(10, 0) 的兩個 multivector, **When** 計算 `geometric_product`, **Then** 結果正確（對照數值驗證）
3. **Given** 高維度代數, **When** 輸出形狀檢查, **Then** 所有輸出形狀符合預期

---

### User Story 6 - PGA 嵌入 CGA (Priority: P3)

開發者使用 PGA(3) = Cl(3,0,1) 進行投影幾何運算，系統透過嵌入 CGA 提供高效實作。

**Why this priority**: PGA 有退化維度，實作較複雜，但透過 CGA 嵌入可利用現有優化。

**Independent Test**: 可以獨立測試 PGA 的 sandwich product，驗證投影幾何變換正確性。

**Acceptance Scenarios**:

1. **Given** `pga = PGA(3)`, **When** 執行 sandwich product, **Then** 結果與純 PGA 實作等價
2. **Given** PGA multivector, **When** 內部轉換, **Then** 正確嵌入到 CGA 並投影回 PGA
3. **Given** `PGA(3)`, **When** 檢查 `algebra_type`, **Then** 返回 'pga'

---

### Edge Cases

- 當 p+q = 0 時（Cl(0,0)）：系統應返回只有純量的 1 維代數
- 當輸入 tensor 形狀不匹配時：應拋出明確的形狀錯誤訊息
- 當 rotor 不是單位長度時：`normalize_rotor` 應正確正規化
- 當 bivector 為零時：`exp_bivector` 應返回單位 rotor
- 當 Bott 週期數 k > 2 時（blade_count > 256^2）：應能正確處理多層張量積
- 當 blade_count > 2^14 (16384) 時：系統應輸出記憶體警告但允許繼續執行

## Requirements *(mandatory)*

### Functional Requirements

**核心 API**:
- **FR-001**: System MUST 提供 `Cl(p, q, r=0)` 工廠函數創建任意 Clifford 代數
- **FR-002**: System MUST 提供 `VGA(n)` 工廠函數作為 `Cl(n, 0)` 的便捷包裝
- **FR-003**: System MUST 提供 `CGA(n)` 工廠函數作為 `Cl(n+1, 1)` 的便捷包裝（含 encode/decode）
- **FR-004**: System MUST 提供 `PGA(n)` 工廠函數作為 `Cl(n, 0, 1)` 的便捷包裝

**屬性**:
- **FR-005**: System MUST 提供 `count_blade` 屬性（總 blade 數）
- **FR-006**: System MUST 提供 `count_rotor` 屬性（rotor 分量數）
- **FR-007**: System MUST 提供 `count_bivector` 屬性（bivector 分量數）
- **FR-008**: System MUST 提供 `p`, `q`, `r` 簽名屬性

**通用運算**:
- **FR-009**: System MUST 實作 `geometric_product(a, b)` 幾何積
- **FR-010**: System MUST 實作 `inner(a, b)` 內積
- **FR-011**: System MUST 實作 `outer(a, b)` 外積
- **FR-012**: System MUST 實作 `contract_left(a, b)` 左縮並
- **FR-013**: System MUST 實作 `contract_right(a, b)` 右縮並
- **FR-014**: System MUST 實作 `scalar(a, b)` 純量積
- **FR-015**: System MUST 實作 `regressive(a, b)` meet 運算
- **FR-016**: System MUST 實作 `sandwich(v, x)` 三明治積
- **FR-017**: System MUST 實作 `reverse(mv)` 反轉
- **FR-018**: System MUST 實作 `involute(mv)` grade 反演
- **FR-019**: System MUST 實作 `conjugate(mv)` Clifford 共軛
- **FR-020**: System MUST 實作 `select_grade(mv, grade)` 提取特定 grade
- **FR-021**: System MUST 實作 `dual(mv)` Poincaré 對偶
- **FR-022**: System MUST 實作 `normalize(mv)` 正規化
- **FR-023**: System MUST 實作 `inverse(mv)` 乘法逆元
- **FR-024**: System MUST 實作 `norm_squared(mv)` 範數平方
- **FR-025**: System MUST 實作 `exp(mv)` 通用指數映射

**Rotor 加速運算**:
- **FR-026**: System MUST 實作 `compose_rotor(r1, r2)` rotor 組合
- **FR-027**: System MUST 實作 `reverse_rotor(r)` rotor 反轉
- **FR-028**: System MUST 實作 `sandwich_rotor(r, x)` rotor 三明治積
- **FR-029**: System MUST 實作 `norm_squared_rotor(r)` rotor 範數平方
- **FR-030**: System MUST 實作 `inverse_rotor(r)` rotor 逆元
- **FR-031**: System MUST 實作 `normalize_rotor(r)` rotor 正規化
- **FR-032**: System MUST 實作 `exp_bivector(B)` bivector 指數映射
- **FR-033**: System MUST 實作 `log_rotor(r)` rotor 對數映射
- **FR-034**: System MUST 實作 `slerp_rotor(r1, r2, t)` 球面線性插值

**VGA 特化**:
- **FR-035**: VGA algebra MUST 提供 `dim_euclidean` 屬性
- **FR-036**: VGA algebra MUST 實作 `encode(x)` 向量嵌入為 Grade-1
- **FR-037**: VGA algebra MUST 實作 `decode(mv)` 提取 Grade-1 部分

**CGA 特化**:
- **FR-038**: CGA algebra MUST 實作 `encode(x)` 歐幾里得 → CGA 點
- **FR-039**: CGA algebra MUST 實作 `decode(p)` CGA 點 → 歐幾里得
- **FR-040**: CGA algebra MUST 提供 `dim_euclidean` 屬性
- **FR-041**: CGA algebra MUST 提供 `count_point` 屬性
- **FR-042**: CGA algebra MUST 使用 Dorst null basis 慣例：`e_o = (e_- - e_+)/2`, `e_inf = e_- + e_+`（與 clifford 庫相容）

**PGA 特化**:
- **FR-043**: PGA algebra MUST 實作 `embed(pga_mv)` PGA → CGA 嵌入
- **FR-044**: PGA algebra MUST 實作 `project(cga_mv)` CGA → PGA 投影
- **FR-045**: PGA algebra MUST 提供 `dim_euclidean` 屬性

**Bott 週期性**:
- **FR-046**: System MUST 在 blade_count > 512 時自動使用 Bott 週期性（等價於 p+q > 9，因 2^9 = 512）
- **FR-047**: Bott 實作 MUST 利用 Cl(p+8, q) ≅ Cl(p, q) ⊗ M₁₆(ℝ) 同構
- **FR-048**: System MUST 在 blade_count > 2^14 (16384) 時輸出記憶體警告但允許繼續執行

**ONNX 相容性**:
- **FR-049**: 所有運算 MUST 為 loop-free 以支援 ONNX 導出
- **FR-050**: Rotor 加速運算 MUST 使用靜態路由（編譯時決定）

**代碼生成**:
- **FR-051**: System MUST 提供統一的 `ClCodeGenerator` 生成任意 Cl(p,q,0)
- **FR-052**: 生成的代碼 MUST 支援 `@torch.jit.script` 優化
- **FR-053**: 所有 Cl(p,q) 其中 p+q ≤ 9 的代數（約 55 個）MUST 預生成並提交到 git repo

**數值精度**:
- **FR-054**: 所有 CGA 運算層 MUST 在入口強制轉換輸入為 float32 以確保數值穩定性

**運算子重載** (Multivector 類別):
- **FR-055**: `a * b` MUST 映射到 `geometric_product`
- **FR-056**: `a ^ b` MUST 映射到 `outer`（外積）
- **FR-057**: `a | b` MUST 映射到 `inner`（內積）
- **FR-058**: `a << b` MUST 映射到 `contract_left`（左縮並）
- **FR-059**: `a >> b` MUST 映射到 `contract_right`（右縮並）
- **FR-060**: `m @ x` MUST 映射到 `sandwich`（三明治積）
- **FR-061**: `a & b` MUST 映射到 `regressive`（meet 運算）
- **FR-062**: `~a` MUST 映射到 `reverse`（反轉）
- **FR-063**: `a ** -1` MUST 映射到 `inverse`（逆元）
- **FR-064**: `mv(k)` MUST 映射到 `select_grade(mv, k)`（grade 選取）

### Key Entities

- **CliffordAlgebraBase**: 所有 Clifford 代數的抽象基礎類別，定義統一 API
- **HardcodedClWrapper**: 包裝自動生成的硬編碼代數（blade_count ≤ 512）
- **BottPeriodicityAlgebra**: 利用 Bott 週期性的高維度代數實作
- **CGAWrapper**: CGA 特化包裝，提供 encode/decode
- **VGAWrapper**: VGA 特化包裝
- **PGAEmbedding**: PGA 透過 CGA 嵌入的實作
- **Multivector**: 統一的 multivector 包裝類別
- **Rotor**: rotor（偶數 grade versor）包裝類別

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有 Cl(p,q) 其中 p+q ≤ 5 的運算結果與 clifford 庫一致（數值誤差 < 1e-6）
- **SC-002**: Rotor 加速運算效能優於通用 multivector 運算至少 20%
- **SC-003**: 低維度代數（p+q ≤ 4）的 sandwich_rotor 相對 clifford 庫加速 > 10x
- **SC-004**: 高維度代數（p+q > 5）能正確執行（語法無錯誤，輸出形狀正確）
- **SC-005**: 所有運算可成功導出 ONNX 模型
- **SC-006**: 完整測試套件執行時間 < 5 分鐘

### Non-Goals

- 不支援 r ≠ 0 的硬編碼實作（PGA 使用 CGA 嵌入）
- 不提供向後相容（刪除舊 cga*d 實作）
- 不保留 EvenVersor、Similitude 命名（統一使用 Rotor）
- 不優化 Bott 週期性到與硬編碼相同效能水準

## Technical Architecture

### 目錄結構

```
fast_clifford/
├── clifford/                   # 統一 Clifford 介面
│   ├── __init__.py             # Cl(), VGA(), CGA(), PGA() 工廠
│   ├── base.py                 # CliffordAlgebraBase
│   ├── registry.py             # HardcodedClWrapper
│   ├── runtime.py              # RuntimeCliffordAlgebra（fallback）
│   ├── bott.py                 # BottPeriodicityAlgebra
│   ├── multivector.py          # Multivector, Rotor 類別
│   ├── layers.py               # PyTorch layers
│   └── specializations/
│       ├── vga.py              # VGA 特化
│       ├── cga.py              # CGA 特化
│       └── pga.py              # PGA 嵌入
├── algebras/
│   └── generated/              # 自動生成的代數
│       ├── cl_3_0/             # VGA3D
│       ├── cl_4_1/             # CGA3D
│       └── ...
├── codegen/
│   ├── clifford_factory.py     # 通用 Cl(p,q,r) 工廠
│   ├── generator.py            # ClCodeGenerator
│   └── bott_generator.py       # Bott 週期性生成器
└── tests/                      # 測試套件
```

### 命名規範

| 類別 | 舊名稱 | 新名稱 |
|------|--------|--------|
| 類別 | `EvenVersor` | `Rotor` |
| 屬性 | `blade_count` | `count_blade` |
| 屬性 | `even_versor_count` | `count_rotor` |
| 函數 | `geometric_product_full` | `geometric_product` |
| 函數 | `reverse_full` | `reverse` |
| 函數 | `grade_select` | `select_grade` |
| 函數 | `inner_product` | `inner` |
| 函數 | `outer_product` | `outer` |
| 函數 | `left_contraction` | `contract_left` |
| 函數 | `right_contraction` | `contract_right` |
| 函數 | `compose_even_versor` | `compose_rotor` |
| 函數 | `reverse_even_versor` | `reverse_rotor` |
| 函數 | `sandwich_product_sparse` | `sandwich_rotor` |

### 代數路由策略

```python
def Cl(p: int, q: int = 0, r: int = 0):
    if r != 0:
        return RuntimeCliffordAlgebra(p, q, r)

    blade_count = 2 ** (p + q + r)
    if blade_count <= 512:
        return HardcodedClWrapper(p, q)
    else:
        return BottPeriodicityAlgebra(p, q)
```

### Bott 週期性數學

```
Cl(p+8, q) ≅ Cl(p, q) ⊗ M₁₆(ℝ)

對於 blade 索引 I 的分解：
base_idx = I // 256        # 基礎代數 blade 索引
matrix_idx = I % 256       # 16×16 矩陣元素索引
row = matrix_idx // 16
col = matrix_idx % 16
```
