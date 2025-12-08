# Tasks: CGA Extended Operations

**Input**: Design documents from `/specs/005-cga-extended-ops/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 包含測試任務（SC-005 要求 90% 覆蓋率）

**Organization**: 任務按 User Story 分組，支援獨立實作和測試

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可平行執行（不同檔案，無依賴）
- **[Story]**: 所屬 User Story (US1, US2, US3, US4)
- 包含確切檔案路徑

## Path Conventions

```text
fast_clifford/
├── cga/base.py, registry.py, runtime.py
├── codegen/generate.py, sparse_analysis.py
├── algebras/cga{0-5}d/functional.py
└── tests/
```

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 擴展 codegen 系統以支援新操作

### 核心操作 codegen (P1-P2)
- [ ] T001 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_motor_compose_terms(dim)` 函式
- [ ] T002 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_inner_product_signs(dim)` 函式
- [ ] T003 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_bivector_squared_terms(dim)` 函式
- [ ] T004 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_bivector_indices(dim)` 函式

### 代數操作 codegen (P3)
- [ ] T004a [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_outer_product_terms(dim)` 函式
- [ ] T004b [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_left_contraction_terms(dim)` 函式
- [ ] T004c [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_right_contraction_terms(dim)` 函式
- [ ] T004d [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_grade_masks(dim)` 函式
- [ ] T004e [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_pseudoscalar_index(dim)` 函式
- [ ] T004f [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_norm_squared_terms(dim)` 函式

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 擴展 codegen 生成器和統一介面基礎類別

**⚠️ CRITICAL**: 所有 User Story 依賴此 Phase 完成

### 核心操作 codegen 生成器 (P1-P2)
- [ ] T005 在 fast_clifford/codegen/generate.py 新增 `_generate_motor_compose_sparse()` 方法
- [ ] T006 在 fast_clifford/codegen/generate.py 新增 `_generate_inner_product_full()` 方法
- [ ] T007 在 fast_clifford/codegen/generate.py 新增 `_generate_bivector_squared_scalar()` 輔助方法
- [ ] T008 在 fast_clifford/codegen/generate.py 新增 `_generate_exp_bivector()` 方法

### 代數操作 codegen 生成器 (P3)
- [ ] T008a 在 fast_clifford/codegen/generate.py 新增 `_generate_outer_product_full()` 方法
- [ ] T008b 在 fast_clifford/codegen/generate.py 新增 `_generate_left_contraction_full()` 方法
- [ ] T008c 在 fast_clifford/codegen/generate.py 新增 `_generate_right_contraction_full()` 方法
- [ ] T008d 在 fast_clifford/codegen/generate.py 新增 `_generate_grade_select()` 方法
- [ ] T008e 在 fast_clifford/codegen/generate.py 新增 `_generate_dual()` 方法
- [ ] T008f 在 fast_clifford/codegen/generate.py 新增 `_generate_normalize()` 方法

### 整合與介面
- [ ] T009 更新 fast_clifford/codegen/generate.py 的 `generate_module()` 和 `generate_sparse_section()` 整合所有新操作
- [ ] T010 在 fast_clifford/cga/base.py 新增所有新操作的抽象方法
- [ ] T011 在 fast_clifford/cga/base.py 新增 `bivector_count`, `max_grade` 屬性

**Checkpoint**: codegen 和 base.py 準備完成，可開始 User Story 實作

---

## Phase 3: User Story 1 - Motor Composition (Priority: P1) 🎯 MVP

**Goal**: 開發者可組合兩個馬達為單一馬達

**Independent Test**: 驗證 `motor_compose(rotation, translation)` 產生正確複合變換

### Tests for User Story 1

- [ ] T012 [P] [US1] 建立 fast_clifford/tests/test_motor_compose.py 測試框架
- [ ] T013 [P] [US1] 新增單位元測試：`motor_compose(identity, M) == M`
- [ ] T014 [P] [US1] 新增結合律測試：`compose(compose(A,B),C) == compose(A,compose(B,C))`
- [ ] T015 [P] [US1] 新增逆元測試：`motor_compose(M, reverse(M)) ≈ identity`
- [ ] T016 [P] [US1] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T017 [P] [US1] 新增批次維度測試
- [ ] T018 [P] [US1] 新增 ONNX 匯出測試 (無 Loop/If 節點)
- [ ] T018a [P] [US1] 新增 autograd 梯度傳播測試 (FR-018)

### Implementation for User Story 1

- [ ] T019 [P] [US1] 更新 fast_clifford/algebras/cga0d/functional.py 加入 `motor_compose_sparse`
- [ ] T020 [P] [US1] 更新 fast_clifford/algebras/cga1d/functional.py 加入 `motor_compose_sparse`
- [ ] T021 [P] [US1] 更新 fast_clifford/algebras/cga2d/functional.py 加入 `motor_compose_sparse`
- [ ] T022 [P] [US1] 更新 fast_clifford/algebras/cga3d/functional.py 加入 `motor_compose_sparse`
- [ ] T023 [P] [US1] 更新 fast_clifford/algebras/cga4d/functional.py 加入 `motor_compose_sparse`
- [ ] T024 [P] [US1] 更新 fast_clifford/algebras/cga5d/functional.py 加入 `motor_compose_sparse`
- [ ] T025 [US1] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.motor_compose
- [ ] T026 [US1] 更新 fast_clifford/algebras/cga{0-5}d/__init__.py 匯出 motor_compose_sparse
- [ ] T027 [US1] 執行 motor_compose 測試驗證 (T012-T018)

**Checkpoint**: Motor Composition 功能完成且可獨立測試

---

## Phase 4: User Story 2 - Geometric Inner Product (Priority: P1)

**Goal**: 開發者可計算 CGA 多向量的度規內積

**Independent Test**: 驗證 `inner_product(eo, einf) == -1`

### Tests for User Story 2

- [ ] T028 [P] [US2] 建立 fast_clifford/tests/test_inner_product.py 測試框架
- [ ] T029 [P] [US2] 新增 Null Basis 測試：`inner_product(eo, einf) == -1`
- [ ] T030 [P] [US2] 新增對稱性測試：`inner_product(a, b) == inner_product(b, a)`
- [ ] T031 [P] [US2] 新增正交性測試：正交 blade 內積為 0
- [ ] T032 [P] [US2] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T033 [P] [US2] 新增零向量測試：`inner_product(0, 0) == 0`
- [ ] T034 [P] [US2] 新增批次維度測試
- [ ] T035 [P] [US2] 新增 ONNX 匯出測試
- [ ] T035a [P] [US2] 新增 autograd 梯度傳播測試 (FR-018)

### Implementation for User Story 2

- [ ] T036 [P] [US2] 更新 fast_clifford/algebras/cga0d/functional.py 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T037 [P] [US2] 更新 fast_clifford/algebras/cga1d/functional.py 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T038 [P] [US2] 更新 fast_clifford/algebras/cga2d/functional.py 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T039 [P] [US2] 更新 fast_clifford/algebras/cga3d/functional.py 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T040 [P] [US2] 更新 fast_clifford/algebras/cga4d/functional.py 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T041 [P] [US2] 更新 fast_clifford/algebras/cga5d/functional.py 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T042 [US2] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.inner_product
- [ ] T043 [US2] 更新 fast_clifford/algebras/cga{0-5}d/__init__.py 匯出 inner_product_full
- [ ] T044 [US2] 執行 inner_product 測試驗證 (T028-T035)

**Checkpoint**: User Stories 1 和 2 都可獨立運作

---

## Phase 5: User Story 3 - Exponential Map (Priority: P2)

**Goal**: 開發者可從 Bivector 生成旋轉馬達

**Independent Test**: 驗證 `exp_bivector(0) == identity` 且 90° 旋轉正確

### Tests for User Story 3

- [ ] T045 [P] [US3] 建立 fast_clifford/tests/test_exp_bivector.py 測試框架
- [ ] T046 [P] [US3] 新增零元測試：`exp_bivector(0) == (1, 0, 0, ...)`
- [ ] T047 [P] [US3] 新增 90° 旋轉測試：驗證旋轉結果正確
- [ ] T048 [P] [US3] 新增極小角度穩定性測試：θ < 1e-10 無 NaN/Inf
- [ ] T049 [P] [US3] 新增逆運算測試：`compose(exp(B), exp(-B)) ≈ identity`
- [ ] T050 [P] [US3] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T051 [P] [US3] 新增批次維度測試
- [ ] T052 [P] [US3] 新增 ONNX 匯出測試
- [ ] T052a [P] [US3] 新增 autograd 梯度傳播測試 (FR-018)

### Implementation for User Story 3

- [ ] T053 [P] [US3] 更新 fast_clifford/algebras/cga0d/functional.py 加入 `exp_bivector`、`bivector_squared_scalar`、`BIVECTOR_MASK`
- [ ] T054 [P] [US3] 更新 fast_clifford/algebras/cga1d/functional.py 加入 `exp_bivector`、`bivector_squared_scalar`、`BIVECTOR_MASK`
- [ ] T055 [P] [US3] 更新 fast_clifford/algebras/cga2d/functional.py 加入 `exp_bivector`、`bivector_squared_scalar`、`BIVECTOR_MASK`
- [ ] T056 [P] [US3] 更新 fast_clifford/algebras/cga3d/functional.py 加入 `exp_bivector`、`bivector_squared_scalar`、`BIVECTOR_MASK`
- [ ] T057 [P] [US3] 更新 fast_clifford/algebras/cga4d/functional.py 加入 `exp_bivector`、`bivector_squared_scalar`、`BIVECTOR_MASK`
- [ ] T058 [P] [US3] 更新 fast_clifford/algebras/cga5d/functional.py 加入 `exp_bivector`、`bivector_squared_scalar`、`BIVECTOR_MASK`
- [ ] T059 [US3] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.exp_bivector
- [ ] T060 [US3] 更新 fast_clifford/algebras/cga{0-5}d/__init__.py 匯出 exp_bivector
- [ ] T061 [US3] 執行 exp_bivector 測試驗證 (T045-T052)

**Checkpoint**: User Stories 1, 2, 3 都可獨立運作

---

## Phase 6: User Story 4 - High-Dimensional Runtime (Priority: P2)

**Goal**: 6D+ 維度使用相同 API，自動切換運行時算法

**Independent Test**: 驗證 CGA(6) 呼叫三個新操作返回正確結果

### Tests for User Story 4

- [ ] T062 [P] [US4] 建立 fast_clifford/tests/test_runtime_extended.py 測試框架
- [ ] T063 [P] [US4] 新增 CGA(6) motor_compose clifford 對照測試
- [ ] T064 [P] [US4] 新增 CGA(6) inner_product clifford 對照測試
- [ ] T065 [P] [US4] 新增 CGA(6) exp_bivector clifford 對照測試
- [ ] T066 [P] [US4] 新增 CGA(7) 基本功能測試
- [ ] T067 [P] [US4] 新增批次維度測試 (6D+)

### Implementation for User Story 4

- [ ] T068 [US4] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.motor_compose
- [ ] T069 [US4] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.inner_product
- [ ] T070 [US4] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.exp_bivector
- [ ] T071 [US4] 在 fast_clifford/cga/runtime.py 新增 `_embed_motor`, `_extract_motor` 輔助方法
- [ ] T072 [US4] 在 fast_clifford/cga/runtime.py 新增 `_embed_bivector`, `_inner_product_signs` 輔助方法
- [ ] T073 [US4] 在 fast_clifford/cga/runtime.py 新增 `bivector_count` 屬性
- [ ] T074 [US4] 執行 runtime 測試驗證 (T062-T067)

**Checkpoint**: 所有 User Stories 完成

---

## Phase 7: User Story 5 - Outer Product (Priority: P3)

**Goal**: 開發者可計算楔積（外積）

**Independent Test**: 驗證 `outer_product(e1, e2)` 返回 e12 Bivector

### Tests for User Story 5

- [ ] T075 [P] [US5] 建立 fast_clifford/tests/test_outer_product.py 測試框架
- [ ] T076 [P] [US5] 新增正交向量楔積測試：`outer_product(e1, e2) == e12`
- [ ] T077 [P] [US5] 新增自楔積測試：`outer_product(v, v) == 0`
- [ ] T078 [P] [US5] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T079 [P] [US5] 新增批次維度測試
- [ ] T079a [P] [US5] 新增 ONNX 匯出測試

### Implementation for User Story 5

- [ ] T080 [P] [US5] 更新 fast_clifford/algebras/cga{0-5}d/functional.py 加入 `outer_product_full`
- [ ] T081 [US5] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.outer_product
- [ ] T082 [US5] 執行 outer_product 測試驗證 (T075-T079a)

**Checkpoint**: Outer Product 功能完成

---

## Phase 8: User Story 6 - Left/Right Contraction (Priority: P3)

**Goal**: 開發者可計算左縮併和右縮併

**Independent Test**: 驗證向量與 Bivector 縮併返回正確 Grade

### Tests for User Story 6

- [ ] T083 [P] [US6] 建立 fast_clifford/tests/test_contractions.py 測試框架
- [ ] T084 [P] [US6] 新增左縮併 Grade 降低測試
- [ ] T085 [P] [US6] 新增右縮併 Grade 降低測試
- [ ] T086 [P] [US6] 新增同 Grade 縮併為標量測試
- [ ] T087 [P] [US6] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T088 [P] [US6] 新增批次維度測試
- [ ] T088a [P] [US6] 新增 ONNX 匯出測試

### Implementation for User Story 6

- [ ] T089 [P] [US6] 更新 fast_clifford/algebras/cga{0-5}d/functional.py 加入 `left_contraction_full`, `right_contraction_full`
- [ ] T090 [US6] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.left_contraction, right_contraction
- [ ] T091 [US6] 執行 contraction 測試驗證 (T083-T088a)

**Checkpoint**: Left/Right Contraction 功能完成

---

## Phase 9: User Story 7 - Grade Selection (Priority: P3)

**Goal**: 開發者可提取多向量特定 Grade 分量

**Independent Test**: 驗證 `grade_select(mv, 0)` 返回標量分量

### Tests for User Story 7

- [ ] T092 [P] [US7] 建立 fast_clifford/tests/test_grade_select.py 測試框架
- [ ] T093 [P] [US7] 新增 Grade 0 提取測試
- [ ] T094 [P] [US7] 新增 Grade 1 提取測試
- [ ] T095 [P] [US7] 新增無效 Grade 返回零測試
- [ ] T096 [P] [US7] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T097 [P] [US7] 新增批次維度測試

### Implementation for User Story 7

- [ ] T098 [P] [US7] 更新 fast_clifford/algebras/cga{0-5}d/functional.py 加入 `grade_select`
- [ ] T099 [US7] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.grade_select
- [ ] T100 [US7] 執行 grade_select 測試驗證 (T092-T097)

**Checkpoint**: Grade Selection 功能完成

---

## Phase 10: User Story 8 - Dual (Priority: P3)

**Goal**: 開發者可計算多向量對偶

**Independent Test**: 驗證 `dual(scalar)` 返回 Pseudoscalar

### Tests for User Story 8

- [ ] T101 [P] [US8] 建立 fast_clifford/tests/test_dual.py 測試框架
- [ ] T102 [P] [US8] 新增標量對偶測試：`dual(1) == pseudoscalar`
- [ ] T103 [P] [US8] 新增 Pseudoscalar 對偶測試：`dual(I) == ±1`
- [ ] T104 [P] [US8] 新增雙重對偶測試：`dual(dual(mv)) == ±mv`
- [ ] T105 [P] [US8] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T106 [P] [US8] 新增批次維度測試

### Implementation for User Story 8

- [ ] T107 [P] [US8] 更新 fast_clifford/algebras/cga{0-5}d/functional.py 加入 `dual`
- [ ] T108 [US8] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.dual
- [ ] T109 [US8] 執行 dual 測試驗證 (T101-T106)

**Checkpoint**: Dual 功能完成

---

## Phase 11: User Story 9 - Normalize (Priority: P3)

**Goal**: 開發者可正規化多向量為單位範數

**Independent Test**: 驗證 `normalize(v)` 返回單位向量

### Tests for User Story 9

- [ ] T110 [P] [US9] 建立 fast_clifford/tests/test_normalize.py 測試框架
- [ ] T111 [P] [US9] 新增單位化測試：`|normalize(v)| == 1`
- [ ] T112 [P] [US9] 新增零向量穩定性測試：`normalize(0) == 0` (無 NaN)
- [ ] T113 [P] [US9] 新增已正規化向量測試：`normalize(normalize(v)) == normalize(v)`
- [ ] T114 [P] [US9] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T115 [P] [US9] 新增批次維度測試

### Implementation for User Story 9

- [ ] T116 [P] [US9] 更新 fast_clifford/algebras/cga{0-5}d/functional.py 加入 `normalize`
- [ ] T117 [US9] 在 fast_clifford/cga/registry.py 實作 HardcodedCGAWrapper.normalize
- [ ] T118 [US9] 執行 normalize 測試驗證 (T110-T115)

**Checkpoint**: Normalize 功能完成

---

## Phase 12: User Story 10 - Operator Overloading (Priority: P2)

**Goal**: 開發者可使用直觀的 Python 運算子操作多向量

**Independent Test**: 驗證 `a * b` 等同於 `geometric_product(a, b)`

### Operator Mapping Table

| 運算子 | Python 方法 | CGA 操作 |
|--------|------------|----------|
| `a * b` | `__mul__` | 幾何積 (geometric product) |
| `a ^ b` | `__xor__` | 楔積 (outer product) |
| `a \| b` | `__or__` | 內積 (inner product) |
| `a @ b` | `__matmul__` | 左縮併 (left contraction) |
| `a + b` | `__add__` | 加法 |
| `a - b` | `__sub__` | 減法 |
| `-a` | `__neg__` | 取負 |
| `~a` | `__invert__` | 反向 (reverse) |
| `a * s` | `__mul__` | 標量右乘 |
| `s * a` | `__rmul__` | 標量左乘 |
| `a / s` | `__truediv__` | 標量除法 |

### Tests for User Story 10

- [ ] T119 [P] [US10] 建立 fast_clifford/tests/test_operators.py 測試框架
- [ ] T120 [P] [US10] 新增幾何積運算子測試：`a * b == geometric_product(a, b)`
- [ ] T121 [P] [US10] 新增楔積運算子測試：`a ^ b == outer_product(a, b)`
- [ ] T122 [P] [US10] 新增內積運算子測試：`a | b == inner_product(a, b)`
- [ ] T123 [P] [US10] 新增左縮併運算子測試：`a @ b == left_contraction(a, b)`
- [ ] T124 [P] [US10] 新增加減法運算子測試
- [ ] T125 [P] [US10] 新增取負運算子測試：`-a`
- [ ] T126 [P] [US10] 新增反向運算子測試：`~a == reverse(a)`
- [ ] T127 [P] [US10] 新增標量乘除法測試：`a * s`, `s * a`, `a / s`
- [ ] T128 [P] [US10] 新增批次維度測試
- [ ] T129 [P] [US10] 新增 autograd 梯度傳播測試

### Implementation for User Story 10

- [ ] T130 [US10] 在 fast_clifford/cga/ 新增 multivector.py 定義 `Multivector` 類別
- [ ] T131 [US10] 實作 `Multivector.__mul__` 和 `__rmul__` (幾何積/標量乘)
- [ ] T132 [US10] 實作 `Multivector.__xor__` (楔積)
- [ ] T133 [US10] 實作 `Multivector.__or__` (內積)
- [ ] T134 [US10] 實作 `Multivector.__matmul__` (左縮併)
- [ ] T135 [US10] 實作 `Multivector.__add__`, `__sub__`, `__neg__` (加減取負)
- [ ] T136 [US10] 實作 `Multivector.__invert__` (反向)
- [ ] T137 [US10] 實作 `Multivector.__truediv__` (標量除法)
- [ ] T138 [US10] 在 CGAAlgebraBase 新增 `multivector(tensor)` 工廠方法
- [ ] T139 [US10] 更新 fast_clifford/__init__.py 匯出 `Multivector` 類別
- [ ] T140 [US10] 執行 US10 測試驗證 (T119-T129)

**Checkpoint**: Operator Overloading 功能完成

---

## Phase 13: User Story 11 - Unified Layer Naming (Refactor)

**Purpose**: 統一 Layer 命名，移除 CARE 特定名稱（不向後相容）

### 重新命名對照表

| 移除 | 統一後 |
|------|--------|
| `CGA{n}DCareLayer` | `CGATransformLayer` |
| `RuntimeCGACareLayer` | `CGATransformLayer` |
| `UPGC{n}DEncoder` | `CGAEncoder` |
| `UPGC{n}DDecoder` | `CGADecoder` |
| `CGA{n}DTransformPipeline` | `CGAPipeline` |
| `get_care_layer()` | `get_transform_layer()` |

### Tests for User Story 11

- [ ] T141 [P] [US11] 建立 fast_clifford/tests/test_unified_layers.py 測試框架
- [ ] T142 [P] [US11] 新增 CGATransformLayer 實例化測試 (n=0-5)
- [ ] T143 [P] [US11] 新增 CGAEncoder/CGADecoder 輸入輸出形狀測試
- [ ] T144 [P] [US11] 新增 CGAPipeline 端對端測試
- [ ] T145 [P] [US11] 新增 get_transform_layer() 方法測試
- [ ] T146 [P] [US11] 新增運行時 (n≥6) 統一 Layer 測試

### Implementation

- [ ] T147 [P] [US11] 在 fast_clifford/cga/ 新增 layers.py 定義統一介面類別 `CGATransformLayer`, `CGAEncoder`, `CGADecoder`, `CGAPipeline`
- [ ] T148 [P] [US11] 移除 fast_clifford/algebras/cga0d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T149 [P] [US11] 移除 fast_clifford/algebras/cga1d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T150 [P] [US11] 移除 fast_clifford/algebras/cga2d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T151 [P] [US11] 移除 fast_clifford/algebras/cga3d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T152 [P] [US11] 移除 fast_clifford/algebras/cga4d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T153 [P] [US11] 移除 fast_clifford/algebras/cga5d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T154 [US11] 更新 fast_clifford/cga/runtime.py 移除 `RuntimeCGACareLayer`，改用統一 `CGATransformLayer`
- [ ] T155 [US11] 更新 fast_clifford/cga/base.py 將 `get_care_layer()` 改為 `get_transform_layer()`（移除舊方法）
- [ ] T156 [US11] 更新 fast_clifford/cga/registry.py 配合新命名
- [ ] T157 [US11] 執行 US11 測試驗證 (T141-T146)

**Checkpoint**: Layer 命名統一完成

---

## Phase 14: High-Dimensional Runtime for New Operations

**Purpose**: 為新增的代數操作實作 6D+ 運行時支援

### Implementation

- [ ] T158 [US4+] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.outer_product
- [ ] T159 [US4+] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.left_contraction
- [ ] T160 [US4+] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.right_contraction
- [ ] T161 [US4+] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.grade_select
- [ ] T162 [US4+] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.dual
- [ ] T163 [US4+] 在 fast_clifford/cga/runtime.py 實作 RuntimeCGAAlgebra.normalize

### Tests

- [ ] T164 [P] 新增 CGA(6) 新操作的 clifford 對照測試
- [ ] T165 [P] 執行所有運行時操作測試

**Checkpoint**: 所有操作 6D+ 運行時支援完成

---

## Phase 15: Polish & Cross-Cutting Concerns

**Purpose**: 整合、匯出、文檔更新

- [ ] T166 [P] 更新 fast_clifford/__init__.py 匯出新操作、統一 Layer 和 Multivector 類別
- [ ] T167 [P] 更新 README.md 新增 Extended Operations API 文檔、運算子重載和新 Layer 命名
- [ ] T168 執行完整測試套件確認無迴歸
- [ ] T169 執行所有 ONNX 匯出測試驗證無 Loop/If 節點
- [ ] T170 執行 quickstart.md 範例驗證
- [ ] T171 效能基準測試：驗證 SC-001（達完整幾何積 50%+）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: 無依賴 - 可立即開始
- **Phase 2 (Foundational)**: 依賴 Phase 1 完成 - **阻擋所有 User Stories**
- **Phase 3-6 (User Stories 1-4)**: 依賴 Phase 2 完成
  - US1 和 US2 可平行進行（都是 P1）
  - US3 和 US4 可平行進行（都是 P2）
- **Phase 7-11 (User Stories 5-9)**: 代數操作（P3），依賴 Phase 2 完成
  - US5-9 皆可平行進行（US9 依賴 US2）
- **Phase 12 (User Story 10 - Operators)**: 依賴 Phase 3-11 完成（需要所有操作）
- **Phase 13 (User Story 11 - Layer Naming)**: 依賴 Phase 3-11 完成
- **Phase 14 (Runtime for New Ops)**: 依賴 Phase 7-11 完成
- **Phase 15 (Polish)**: 依賴所有 User Stories 完成

### User Story Dependencies

- **US1 (Motor Composition)**: 可於 Phase 2 後立即開始
- **US2 (Inner Product)**: 可於 Phase 2 後立即開始，與 US1 獨立
- **US3 (Exponential Map)**: 可於 Phase 2 後開始，與 US1/US2 獨立
- **US4 (Runtime Core)**: 可於 Phase 2 後開始，但建議在 US1-3 之後（可參考硬編碼實作）
- **US5 (Outer Product)**: 可於 Phase 2 後開始
- **US6 (Contractions)**: 可於 Phase 2 後開始
- **US7 (Grade Selection)**: 可於 Phase 2 後開始
- **US8 (Dual)**: 可於 Phase 2 後開始
- **US9 (Normalize)**: 依賴 US2 (inner_product) 完成（用於計算範數）
- **US10 (Operators)**: 依賴所有操作完成（需要 geometric_product, outer_product 等）
- **US11 (Unified Layers)**: 在其他 User Stories 完成後進行

### Within Each User Story

- Tests (T012-T018 等) 應先撰寫並確認失敗
- functional.py 生成在 registry.py 之前
- 核心實作在整合之前
- Story 完成後再進入下一個

### Parallel Opportunities

- Phase 1: T001-T004f 全部可平行
- Phase 2: T005-T011 依序（有依賴）
- Phase 3: T012-T018 測試可平行，T019-T024 生成可平行
- Phase 4: T028-T035 測試可平行，T036-T041 更新可平行
- Phase 5: T045-T052 測試可平行，T053-T058 更新可平行
- Phase 6: T062-T067 測試可平行
- Phase 7-11: 各 Phase 測試和實作可平行
- Phase 12: T119-T129 測試可平行，T130-T139 實作可平行
- Phase 13: T141-T146 測試可平行，T147-T156 更新可平行

---

## Parallel Example: User Story 1

```bash
# 平行執行所有 US1 測試建立：
Task: "T012 [P] [US1] 建立 test_motor_compose.py 測試框架"
Task: "T013 [P] [US1] 新增單位元測試"
Task: "T014 [P] [US1] 新增結合律測試"
...

# 平行執行所有維度的 functional.py 重新生成：
Task: "T019 [US1] 重新生成 cga0d/functional.py"
Task: "T020 [US1] 重新生成 cga1d/functional.py"
Task: "T021 [US1] 重新生成 cga2d/functional.py"
...
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. 完成 Phase 1: Setup (T001-T004f)
2. 完成 Phase 2: Foundational (T005-T011)
3. 完成 Phase 3: User Story 1 (T012-T027)
4. **驗證**: 測試 motor_compose 功能
5. 可部署 MVP

### Incremental Delivery

**核心操作 (P1-P2)**:
1. Setup + Foundational → codegen 準備完成
2. 加入 US1 (Motor Composition) → 測試 → 交付
3. 加入 US2 (Inner Product) → 測試 → 交付
4. 加入 US3 (Exponential Map) → 測試 → 交付
5. 加入 US4 (Runtime Core) → 測試 → 交付

**代數操作 (P3)**:
6. 加入 US5 (Outer Product) → 測試 → 交付
7. 加入 US6 (Contractions) → 測試 → 交付
8. 加入 US7 (Grade Selection) → 測試 → 交付
9. 加入 US8 (Dual) → 測試 → 交付
10. 加入 US9 (Normalize) → 測試 → 交付

**使用者體驗與重構**:
11. 加入 US10 (Operators) → 測試 → 交付
12. 加入 US11 (Unified Layers) → 測試 → 交付
13. Runtime for New Ops → 測試 → 交付
14. Polish → 最終驗證

---

## Notes

- [P] = 不同檔案，無依賴
- [Story] = 對應 spec.md 的 User Story
- 每個 User Story 應可獨立完成和測試
- 測試失敗後再實作
- 每個任務或邏輯群組後提交 Git
- 任何 Checkpoint 可停下驗證
