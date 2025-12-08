# Tasks: CGA Extended Operations

**Input**: Design documents from `/specs/005-cga-extended-ops/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 包含測試任務（SC-005 要求 90% 覆蓋率）

**Organization**: 任務按 User Story 分組，支援獨立實作和測試

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可平行執行（不同檔案，無依賴）
- **[Story]**: 所屬 User Story (US1, US2, ...)
- 包含確切檔案路徑

## Path Conventions

```text
fast_clifford/
├── cga/base.py, registry.py, runtime.py, multivector.py, layers.py
├── codegen/generate.py, sparse_analysis.py
├── algebras/cga{0-5}d/functional.py, layers.py
└── tests/
```

## Naming Convention

| 舊名稱 | 新名稱 | 說明 |
|--------|--------|------|
| Motor | EvenVersor | 通用 Clifford 代數偶數 Versor |
| motor_compose_sparse | compose_even_versor | 偶數 Versor 組合 |
| sandwich_product_sparse | sandwich_product_even_versor | 偶數 Versor 三明治積 |
| motor_count | even_versor_count | 偶數 Versor 分量數 |
| - | Similitude | CGA 專用子類別（更快） |
| - | compose_similitude | Similitude 組合（更快） |
| - | sandwich_product_similitude | Similitude 三明治積（更快） |

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 擴展 codegen 系統以支援新操作

### 核心操作 codegen (P1-P2)

- [ ] T001 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_compose_even_versor_terms(dim)` 函式
- [ ] T002 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_compose_similitude_terms(dim)` 函式（CGA 專用）
- [ ] T003 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_inner_product_signs(dim)` 函式
- [ ] T004 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_bivector_squared_terms(dim)` 函式
- [ ] T005 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_bivector_indices(dim)` 函式

### 代數操作 codegen (P3)

- [ ] T006 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_outer_product_terms(dim)` 函式
- [ ] T007 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_left_contraction_terms(dim)` 函式
- [ ] T008 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_right_contraction_terms(dim)` 函式
- [ ] T009 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_grade_masks(dim)` 函式
- [ ] T010 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_pseudoscalar_info(dim)` 函式
- [ ] T011 [P] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_norm_squared_terms(dim)` 函式

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 擴展 codegen 生成器和統一介面基礎類別

**⚠️ CRITICAL**: 所有 User Story 依賴此 Phase 完成

### 核心操作 codegen 生成器 (P1-P2)

- [ ] T012 在 `fast_clifford/codegen/generate.py` 新增 `_generate_compose_even_versor()` 方法
- [ ] T013 在 `fast_clifford/codegen/generate.py` 新增 `_generate_compose_similitude()` 方法
- [ ] T014 在 `fast_clifford/codegen/generate.py` 新增 `_generate_sandwich_product_similitude()` 方法
- [ ] T015 在 `fast_clifford/codegen/generate.py` 新增 `_generate_inner_product_full()` 方法
- [ ] T016 在 `fast_clifford/codegen/generate.py` 新增 `_generate_bivector_squared_scalar()` 輔助方法
- [ ] T017 在 `fast_clifford/codegen/generate.py` 新增 `_generate_exp_bivector()` 方法

### 代數操作 codegen 生成器 (P3)

- [ ] T018 在 `fast_clifford/codegen/generate.py` 新增 `_generate_outer_product_full()` 方法
- [ ] T019 在 `fast_clifford/codegen/generate.py` 新增 `_generate_left_contraction_full()` 方法
- [ ] T020 在 `fast_clifford/codegen/generate.py` 新增 `_generate_right_contraction_full()` 方法
- [ ] T021 在 `fast_clifford/codegen/generate.py` 新增 `_generate_grade_select()` 方法
- [ ] T022 在 `fast_clifford/codegen/generate.py` 新增 `_generate_dual()` 方法
- [ ] T023 在 `fast_clifford/codegen/generate.py` 新增 `_generate_normalize()` 方法

### 整合與介面

- [ ] T024 更新 `fast_clifford/codegen/generate.py` 的 `generate_module()` 整合所有新操作
- [ ] T025 在 `fast_clifford/cga/base.py` 新增所有新操作的抽象方法
- [ ] T026 在 `fast_clifford/cga/base.py` 新增統一 API：`compose()`, `sandwich_product()`, `reverse()`
- [ ] T027 在 `fast_clifford/cga/base.py` 新增屬性：`bivector_count`, `max_grade`, `even_versor_count`, `similitude_count`

**Checkpoint**: codegen 和 base.py 準備完成，可開始 User Story 實作

---

## Phase 3: User Story 1 - EvenVersor Composition (Priority: P1) 🎯 MVP

**Goal**: 開發者可組合兩個偶數 Versor 為單一偶數 Versor

**Independent Test**: 驗證 `compose(rotation, translation)` 產生正確複合變換

### Tests for User Story 1

- [ ] T028 [P] [US1] 建立 `fast_clifford/tests/test_compose.py` 測試框架
- [ ] T029 [P] [US1] 新增單位元測試：`compose(identity, V) == V`
- [ ] T030 [P] [US1] 新增結合律測試：`compose(compose(A,B),C) == compose(A,compose(B,C))`
- [ ] T031 [P] [US1] 新增逆元測試：`compose(V, reverse(V)) ≈ identity`
- [ ] T032 [P] [US1] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T033 [P] [US1] 新增批次維度測試
- [ ] T034 [P] [US1] 新增 ONNX 匯出測試 (無 Loop/If 節點)
- [ ] T035 [P] [US1] 新增 autograd 梯度傳播測試
- [ ] T036 [P] [US1] 新增統一 API 路由測試：`compose()` 自動路由

### Implementation for User Story 1

- [ ] T037 [P] [US1] 更新 `fast_clifford/algebras/cga0d/functional.py` 加入 `compose_even_versor`
- [ ] T038 [P] [US1] 更新 `fast_clifford/algebras/cga1d/functional.py` 加入 `compose_even_versor`
- [ ] T039 [P] [US1] 更新 `fast_clifford/algebras/cga2d/functional.py` 加入 `compose_even_versor`
- [ ] T040 [P] [US1] 更新 `fast_clifford/algebras/cga3d/functional.py` 加入 `compose_even_versor`
- [ ] T041 [P] [US1] 更新 `fast_clifford/algebras/cga4d/functional.py` 加入 `compose_even_versor`
- [ ] T042 [P] [US1] 更新 `fast_clifford/algebras/cga5d/functional.py` 加入 `compose_even_versor`
- [ ] T043 [US1] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.compose_even_versor`
- [ ] T044 [US1] 在 `fast_clifford/cga/registry.py` 實作統一 `compose()` API 路由
- [ ] T045 [US1] 更新 `fast_clifford/algebras/cga{0-5}d/__init__.py` 匯出新函式
- [ ] T046 [US1] 執行 compose 測試驗證 (T028-T036)

**Checkpoint**: EvenVersor Composition 功能完成且可獨立測試

---

## Phase 4: User Story 4a - Similitude Accelerated Operations (Priority: P1)

**Goal**: CGA 專用 Similitude 操作提供更高效能

**Independent Test**: 驗證 Similitude 結果與 EvenVersor 一致但更快

### Tests for User Story 4a

- [ ] T047 [P] [US4a] 在 `test_compose.py` 新增 `compose_similitude` 正確性測試
- [ ] T048 [P] [US4a] 新增 Similitude × Similitude 效能比較測試
- [ ] T049 [P] [US4a] 新增 `sandwich_product_similitude` 正確性測試
- [ ] T050 [P] [US4a] 新增 Similitude 三明治積效能比較測試
- [ ] T051 [P] [US4a] 新增 Similitude 約束驗證測試 (排除 transversion)
- [ ] T052 [P] [US4a] 新增靜態路由測試：Similitude × Similitude → `compose_similitude`

### Implementation for User Story 4a

- [ ] T053 [P] [US4a] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `compose_similitude`
- [ ] T054 [P] [US4a] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `sandwich_product_similitude`
- [ ] T055 [US4a] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.compose_similitude`
- [ ] T056 [US4a] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.sandwich_product_similitude`
- [ ] T057 [US4a] 更新 `compose()` 和 `sandwich_product()` 靜態路由邏輯
- [ ] T058 [US4a] 執行 Similitude 測試驗證 (T047-T052)

**Checkpoint**: Similitude 加速功能完成，US1 和 US4a 可獨立運作

---

## Phase 5: User Story 2 - Geometric Inner Product (Priority: P1)

**Goal**: 開發者可計算 CGA 多向量的度規內積

**Independent Test**: 驗證 `inner_product(eo, einf) == -1`

### Tests for User Story 2

- [ ] T059 [P] [US2] 建立 `fast_clifford/tests/test_inner_product.py` 測試框架
- [ ] T060 [P] [US2] 新增 Null Basis 測試：`inner_product(eo, einf) == -1`
- [ ] T061 [P] [US2] 新增對稱性測試：`inner_product(a, b) == inner_product(b, a)`
- [ ] T062 [P] [US2] 新增正交性測試：正交 blade 內積為 0
- [ ] T063 [P] [US2] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T064 [P] [US2] 新增零向量測試：`inner_product(0, 0) == 0`
- [ ] T065 [P] [US2] 新增批次維度測試
- [ ] T066 [P] [US2] 新增 ONNX 匯出測試
- [ ] T067 [P] [US2] 新增 autograd 梯度傳播測試

### Implementation for User Story 2

- [ ] T068 [P] [US2] 更新 `fast_clifford/algebras/cga0d/functional.py` 加入 `inner_product_full` 和 `INNER_PRODUCT_SIGNS`
- [ ] T069 [P] [US2] 更新 `fast_clifford/algebras/cga1d/functional.py` 加入 `inner_product_full`
- [ ] T070 [P] [US2] 更新 `fast_clifford/algebras/cga2d/functional.py` 加入 `inner_product_full`
- [ ] T071 [P] [US2] 更新 `fast_clifford/algebras/cga3d/functional.py` 加入 `inner_product_full`
- [ ] T072 [P] [US2] 更新 `fast_clifford/algebras/cga4d/functional.py` 加入 `inner_product_full`
- [ ] T073 [P] [US2] 更新 `fast_clifford/algebras/cga5d/functional.py` 加入 `inner_product_full`
- [ ] T074 [US2] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.inner_product`
- [ ] T075 [US2] 更新 `fast_clifford/algebras/cga{0-5}d/__init__.py` 匯出 `inner_product_full`
- [ ] T076 [US2] 執行 inner_product 測試驗證 (T059-T067)

**Checkpoint**: User Stories 1, 4a, 2 都可獨立運作

---

## Phase 6: User Story 3 - Exponential Map (Priority: P2)

**Goal**: 開發者可從 Bivector 生成旋轉偶數 Versor

**Independent Test**: 驗證 `exp_bivector(0) == identity` 且 90° 旋轉正確

### Tests for User Story 3

- [ ] T077 [P] [US3] 建立 `fast_clifford/tests/test_exp_bivector.py` 測試框架
- [ ] T078 [P] [US3] 新增零元測試：`exp_bivector(0) == (1, 0, 0, ...)`
- [ ] T079 [P] [US3] 新增 90° 旋轉測試：驗證旋轉結果正確
- [ ] T080 [P] [US3] 新增極小角度穩定性測試：θ < 1e-10 無 NaN/Inf
- [ ] T081 [P] [US3] 新增逆運算測試：`compose(exp(B), exp(-B)) ≈ identity`
- [ ] T082 [P] [US3] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T083 [P] [US3] 新增批次維度測試
- [ ] T084 [P] [US3] 新增 ONNX 匯出測試
- [ ] T085 [P] [US3] 新增 autograd 梯度傳播測試

### Implementation for User Story 3

- [ ] T086 [P] [US3] 更新 `fast_clifford/algebras/cga0d/functional.py` 加入 `exp_bivector`, `BIVECTOR_MASK`
- [ ] T087 [P] [US3] 更新 `fast_clifford/algebras/cga1d/functional.py` 加入 `exp_bivector`
- [ ] T088 [P] [US3] 更新 `fast_clifford/algebras/cga2d/functional.py` 加入 `exp_bivector`
- [ ] T089 [P] [US3] 更新 `fast_clifford/algebras/cga3d/functional.py` 加入 `exp_bivector`
- [ ] T090 [P] [US3] 更新 `fast_clifford/algebras/cga4d/functional.py` 加入 `exp_bivector`
- [ ] T091 [P] [US3] 更新 `fast_clifford/algebras/cga5d/functional.py` 加入 `exp_bivector`
- [ ] T092 [US3] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.exp_bivector`
- [ ] T093 [US3] 更新 `fast_clifford/algebras/cga{0-5}d/__init__.py` 匯出 `exp_bivector`
- [ ] T094 [US3] 執行 exp_bivector 測試驗證 (T077-T085)

**Checkpoint**: User Stories 1, 4a, 2, 3 都可獨立運作

---

## Phase 7: User Story 4 - High-Dimensional Runtime (Priority: P2)

**Goal**: 6D+ 維度使用相同 API，自動切換運行時算法

**Independent Test**: 驗證 CGA(6) 呼叫新操作返回正確結果

### Tests for User Story 4

- [ ] T095 [P] [US4] 建立 `fast_clifford/tests/test_runtime_extended.py` 測試框架
- [ ] T096 [P] [US4] 新增 CGA(6) compose clifford 對照測試
- [ ] T097 [P] [US4] 新增 CGA(6) inner_product clifford 對照測試
- [ ] T098 [P] [US4] 新增 CGA(6) exp_bivector clifford 對照測試
- [ ] T099 [P] [US4] 新增 CGA(7) 基本功能測試
- [ ] T100 [P] [US4] 新增批次維度測試 (6D+)

### Implementation for User Story 4

- [ ] T101 [US4] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.compose_even_versor`
- [ ] T102 [US4] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.inner_product`
- [ ] T103 [US4] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.exp_bivector`
- [ ] T104 [US4] 在 `fast_clifford/cga/runtime.py` 新增 `_embed_even_versor`, `_extract_even_versor` 輔助
- [ ] T105 [US4] 在 `fast_clifford/cga/runtime.py` 新增 `_embed_bivector`, `_inner_product_signs` 輔助
- [ ] T106 [US4] 在 `fast_clifford/cga/runtime.py` 新增 `bivector_count`, `even_versor_count` 屬性
- [ ] T107 [US4] 執行 runtime 測試驗證 (T095-T100)

**Checkpoint**: 所有核心操作 (P1-P2) 完成

---

## Phase 8: User Story 5 - Outer Product (Priority: P3)

**Goal**: 開發者可計算楔積（外積）

**Independent Test**: 驗證 `outer_product(e1, e2)` 返回 e12 Bivector

### Tests for User Story 5

- [ ] T108 [P] [US5] 建立 `fast_clifford/tests/test_outer_product.py` 測試框架
- [ ] T109 [P] [US5] 新增正交向量楔積測試：`outer_product(e1, e2) == e12`
- [ ] T110 [P] [US5] 新增自楔積測試：`outer_product(v, v) == 0`
- [ ] T111 [P] [US5] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T112 [P] [US5] 新增批次維度測試
- [ ] T113 [P] [US5] 新增 ONNX 匯出測試

### Implementation for User Story 5

- [ ] T114 [P] [US5] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `outer_product_full`
- [ ] T115 [US5] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.outer_product`
- [ ] T116 [US5] 執行 outer_product 測試驗證 (T108-T113)

**Checkpoint**: Outer Product 功能完成

---

## Phase 9: User Story 6 - Left/Right Contraction (Priority: P3)

**Goal**: 開發者可計算左縮併和右縮併

**Independent Test**: 驗證向量與 Bivector 縮併返回正確 Grade

### Tests for User Story 6

- [ ] T117 [P] [US6] 建立 `fast_clifford/tests/test_contractions.py` 測試框架
- [ ] T118 [P] [US6] 新增左縮併 Grade 降低測試
- [ ] T119 [P] [US6] 新增右縮併 Grade 降低測試
- [ ] T120 [P] [US6] 新增同 Grade 縮併為標量測試
- [ ] T121 [P] [US6] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T122 [P] [US6] 新增批次維度測試
- [ ] T123 [P] [US6] 新增 ONNX 匯出測試

### Implementation for User Story 6

- [ ] T124 [P] [US6] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `left_contraction_full`, `right_contraction_full`
- [ ] T125 [US6] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.left_contraction`, `right_contraction`
- [ ] T126 [US6] 執行 contraction 測試驗證 (T117-T123)

**Checkpoint**: Left/Right Contraction 功能完成

---

## Phase 10: User Story 7 - Grade Selection (Priority: P3)

**Goal**: 開發者可提取多向量特定 Grade 分量

**Independent Test**: 驗證 `grade_select(mv, 0)` 返回標量分量

### Tests for User Story 7

- [ ] T127 [P] [US7] 建立 `fast_clifford/tests/test_grade_select.py` 測試框架
- [ ] T128 [P] [US7] 新增 Grade 0 提取測試
- [ ] T129 [P] [US7] 新增 Grade 1 提取測試
- [ ] T130 [P] [US7] 新增無效 Grade 返回零測試
- [ ] T131 [P] [US7] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T132 [P] [US7] 新增批次維度測試

### Implementation for User Story 7

- [ ] T133 [P] [US7] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `grade_select`, `GRADE_MASKS`
- [ ] T134 [US7] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.grade_select`
- [ ] T135 [US7] 執行 grade_select 測試驗證 (T127-T132)

**Checkpoint**: Grade Selection 功能完成

---

## Phase 11: User Story 8 - Dual (Priority: P3)

**Goal**: 開發者可計算多向量對偶

**Independent Test**: 驗證 `dual(scalar)` 返回 Pseudoscalar

### Tests for User Story 8

- [ ] T136 [P] [US8] 建立 `fast_clifford/tests/test_dual.py` 測試框架
- [ ] T137 [P] [US8] 新增標量對偶測試：`dual(1) == pseudoscalar`
- [ ] T138 [P] [US8] 新增 Pseudoscalar 對偶測試：`dual(I) == ±1`
- [ ] T139 [P] [US8] 新增雙重對偶測試：`dual(dual(mv)) == ±mv`
- [ ] T140 [P] [US8] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T141 [P] [US8] 新增批次維度測試

### Implementation for User Story 8

- [ ] T142 [P] [US8] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `dual`, `PSEUDOSCALAR_SQUARE`
- [ ] T143 [US8] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.dual`
- [ ] T144 [US8] 執行 dual 測試驗證 (T136-T141)

**Checkpoint**: Dual 功能完成

---

## Phase 12: User Story 9 - Normalize (Priority: P3)

**Goal**: 開發者可正規化多向量為單位範數

**Independent Test**: 驗證 `normalize(v)` 返回單位向量

**Dependency**: 依賴 US2 (inner_product) 完成

### Tests for User Story 9

- [ ] T145 [P] [US9] 建立 `fast_clifford/tests/test_normalize.py` 測試框架
- [ ] T146 [P] [US9] 新增單位化測試：`|normalize(v)| == 1`
- [ ] T147 [P] [US9] 新增零向量穩定性測試：`normalize(0) == 0` (無 NaN)
- [ ] T148 [P] [US9] 新增已正規化向量測試：`normalize(normalize(v)) == normalize(v)`
- [ ] T149 [P] [US9] 新增 clifford 庫對照測試 (n=0-5)
- [ ] T150 [P] [US9] 新增批次維度測試

### Implementation for User Story 9

- [ ] T151 [P] [US9] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `normalize`
- [ ] T152 [US9] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.normalize`
- [ ] T153 [US9] 執行 normalize 測試驗證 (T145-T150)

**Checkpoint**: Normalize 功能完成

---

## Phase 12a: User Story 9a - Structure Normalize (Priority: P2)

**Goal**: 開發者可對 Similitude 進行結構正規化，保持幾何約束

**Independent Test**: 驗證正規化後 Rotor 為單位長，且 `ei+ = ei-`

**Dependency**: 依賴 US4a (Similitude) 完成

### Tests for User Story 9a

- [ ] T153a [P] [US9a] 建立 `fast_clifford/tests/test_structure_normalize.py` 測試框架
- [ ] T153b [P] [US9a] 新增 Rotor 單位化測試：`|rotor_part(structure_normalize(s))| == 1`
- [ ] T153c [P] [US9a] 新增 Similitude 約束測試：正規化後 `ei+ == ei-`
- [ ] T153d [P] [US9a] 新增恆等性測試：已正規化的 Similitude 再次正規化不變
- [ ] T153e [P] [US9a] 新增 soft_structure_normalize 測試：strength=0 返回原值，strength=1 等於 structure_normalize
- [ ] T153f [P] [US9a] 新增 STE 版本梯度測試：確認梯度穿透
- [ ] T153g [P] [US9a] 新增批次維度測試
- [ ] T153h [P] [US9a] 新增 ONNX 匯出測試

### Implementation for User Story 9a

- [ ] T153i [P] [US9a] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_rotor_indices(dim)` 函式
- [ ] T153j [P] [US9a] 在 `fast_clifford/codegen/sparse_analysis.py` 新增 `get_translation_pairs(dim)` 函式
- [ ] T153k [P] [US9a] 在 `fast_clifford/codegen/generate.py` 新增 `_generate_structure_normalize()` 方法
- [ ] T153l [P] [US9a] 更新 `fast_clifford/algebras/cga{0-5}d/functional.py` 加入 `structure_normalize`, `ROTOR_INDICES`, `TRANSLATION_PAIRS`
- [ ] T153m [US9a] 在 `fast_clifford/cga/registry.py` 實作 `HardcodedCGAWrapper.structure_normalize`
- [ ] T153n [US9a] 在 `fast_clifford/cga/registry.py` 實作 `soft_structure_normalize` 和 `structure_normalize_ste`
- [ ] T153o [US9a] 在 `fast_clifford/cga/base.py` 新增 `structure_normalize` 抽象方法
- [ ] T153p [US9a] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.structure_normalize`
- [ ] T153q [US9a] 執行 structure_normalize 測試驗證 (T153a-T153h)

**Checkpoint**: Structure Normalize 功能完成

---

## Phase 13: User Story 10 - Operator Overloading (Priority: P2)

**Goal**: 開發者可使用直觀的 Python 運算子操作多向量

**Independent Test**: 驗證 `a * b` 等同於 `geometric_product(a, b)`

### Operator Mapping Table

| 運算子 | Python 方法 | CGA 操作 |
|--------|------------|----------|
| `a * b` | `__mul__` | 幾何積 / compose (靜態路由) |
| `a ^ b` | `__xor__` | 楔積 (outer product) |
| `a \| b` | `__or__` | 內積 (inner product) |
| `a << b` | `__lshift__` | 左縮併 (left contraction) |
| `a >> b` | `__rshift__` | 右縮併 (right contraction) |
| `m @ x` | `__matmul__` | 三明治積 (sandwich product) |
| `a + b` | `__add__` | 加法 |
| `a - b` | `__sub__` | 減法 |
| `-a` | `__neg__` | 取負 |
| `~a` | `__invert__` | 反向 (reverse) |
| `a / s` | `__truediv__` | 標量除法 / 多向量除法 |
| `a ** n` | `__pow__` | 整數冪次 / 逆元 (`** -1`) |

### Tests for User Story 10

- [ ] T154 [P] [US10] 建立 `fast_clifford/tests/test_operators.py` 測試框架
- [ ] T155 [P] [US10] 新增幾何積運算子測試：`a * b == geometric_product(a, b)`
- [ ] T156 [P] [US10] 新增楔積運算子測試：`a ^ b == outer_product(a, b)`
- [ ] T157 [P] [US10] 新增內積運算子測試：`a | b == inner_product(a, b)`
- [ ] T158 [P] [US10] 新增左縮併運算子測試：`a << b == left_contraction(a, b)`
- [ ] T159 [P] [US10] 新增右縮併運算子測試：`a >> b == right_contraction(a, b)`
- [ ] T160 [P] [US10] 新增三明治積運算子測試：`m @ x == sandwich_product(m, x)`
- [ ] T161 [P] [US10] 新增加減法運算子測試
- [ ] T162 [P] [US10] 新增取負運算子測試：`-a`
- [ ] T163 [P] [US10] 新增反向運算子測試：`~a == reverse(a)`
- [ ] T164 [P] [US10] 新增標量乘除法測試：`a * s`, `s * a`, `a / s`
- [ ] T165 [P] [US10] 新增冪次運算子測試：`a ** 2 == a * a`
- [ ] T166 [P] [US10] 新增逆元冪次測試：`a ** -1 == a.inverse()`
- [ ] T167 [P] [US10] 新增 `exp()` 方法測試：`B.exp() == exp_bivector(B.data)`
- [ ] T168 [P] [US10] 新增多向量除法測試：`a / b == a * b.inverse()`
- [ ] T169 [P] [US10] 新增逆元測試：`a * a.inverse() ≈ identity`
- [ ] T170 [P] [US10] 新增不可逆多向量測試：`null_vector.inverse()` 應處理
- [ ] T171 [P] [US10] 新增類型標記工廠方法測試
- [ ] T172 [P] [US10] 新增靜態路由測試：Similitude × Similitude
- [ ] T173 [P] [US10] 新增批次維度測試
- [ ] T174 [P] [US10] 新增 autograd 梯度傳播測試

### Implementation for User Story 10

- [ ] T175 [US10] 新增 `fast_clifford/cga/multivector.py` 定義 `Multivector` 類別
- [ ] T176 [US10] 在 `multivector.py` 定義 `Versor(Multivector)` 子類別
- [ ] T177 [US10] 在 `multivector.py` 定義 `EvenVersor(Versor)` 子類別
- [ ] T178 [US10] 在 `multivector.py` 定義 `Similitude(EvenVersor)` CGA 專用子類別
- [ ] T179 [US10] 實作 `Multivector.__mul__` 和 `__rmul__` (幾何積/標量乘，含靜態路由)
- [ ] T180 [US10] 實作 `Multivector.__xor__` (楔積)
- [ ] T181 [US10] 實作 `Multivector.__or__` (內積)
- [ ] T182 [US10] 實作 `Multivector.__lshift__` 和 `__rshift__` (左/右縮併)
- [ ] T183 [US10] 實作 `Multivector.__matmul__` (三明治積，含靜態路由)
- [ ] T184 [US10] 實作 `Multivector.__add__`, `__sub__`, `__neg__`
- [ ] T185 [US10] 實作 `Multivector.__invert__` (反向)
- [ ] T186 [US10] 實作 `Multivector.__truediv__` (標量/多向量除法)
- [ ] T187 [US10] 實作 `Multivector.inverse()` 方法
- [ ] T188 [US10] 實作 `Multivector.__pow__` (冪次和 `** -1` 逆元)
- [ ] T189 [US10] 實作 `Multivector.exp()` 方法 (Bivector 指數映射)
- [ ] T190 [US10] 在 `CGAAlgebraBase` 新增工廠方法：`multivector()`, `even_versor()`, `similitude()`, `bivector()`, `point()`
- [ ] T191 [US10] 更新 `fast_clifford/__init__.py` 匯出 `Multivector`, `Versor`, `EvenVersor`, `Similitude`
- [ ] T192 [US10] 執行 US10 測試驗證 (T154-T174)

**Checkpoint**: Operator Overloading 功能完成

---

## Phase 14: User Story 11 - Unified Layer Naming (Priority: P2)

**Goal**: 統一 Layer 命名，移除維度特定名稱

### Rename Table

| 移除 | 統一後 |
|------|--------|
| `CGA{n}DCareLayer` | `CliffordTransformLayer` |
| `RuntimeCGACareLayer` | `CliffordTransformLayer` |
| `UPGC{n}DEncoder` | `CGAEncoder` |
| `UPGC{n}DDecoder` | `CGADecoder` |
| `CGA{n}DTransformPipeline` | `CGAPipeline` |
| `get_care_layer()` | `get_transform_layer()` |

### Tests for User Story 11

- [ ] T193 [P] [US11] 建立 `fast_clifford/tests/test_unified_layers.py` 測試框架
- [ ] T194 [P] [US11] 新增 `CliffordTransformLayer` 實例化測試 (n=0-5)
- [ ] T195 [P] [US11] 新增 `CGAEncoder`/`CGADecoder` 輸入輸出形狀測試
- [ ] T196 [P] [US11] 新增 `CGAPipeline` 端對端測試
- [ ] T197 [P] [US11] 新增 `get_transform_layer()` 方法測試
- [ ] T198 [P] [US11] 新增 `get_transform_layer(versor_type='similitude')` 測試
- [ ] T199 [P] [US11] 新增運行時 (n≥6) 統一 Layer 測試

### Implementation for User Story 11

- [ ] T200 [US11] 新增 `fast_clifford/cga/layers.py` 定義 `CliffordTransformLayer`
- [ ] T201 [US11] 在 `layers.py` 定義 `CGAEncoder`, `CGADecoder`, `CGAPipeline`
- [ ] T202 [P] [US11] 更新 `fast_clifford/algebras/cga0d/layers.py` 改為從 `cga/layers.py` 匯入
- [ ] T203 [P] [US11] 更新 `fast_clifford/algebras/cga1d/layers.py` 改為從 `cga/layers.py` 匯入
- [ ] T204 [P] [US11] 更新 `fast_clifford/algebras/cga2d/layers.py` 改為從 `cga/layers.py` 匯入
- [ ] T205 [P] [US11] 更新 `fast_clifford/algebras/cga3d/layers.py` 改為從 `cga/layers.py` 匯入
- [ ] T206 [P] [US11] 更新 `fast_clifford/algebras/cga4d/layers.py` 改為從 `cga/layers.py` 匯入
- [ ] T207 [P] [US11] 更新 `fast_clifford/algebras/cga5d/layers.py` 改為從 `cga/layers.py` 匯入
- [ ] T208 [US11] 更新 `fast_clifford/cga/runtime.py` 移除 `RuntimeCGACareLayer`
- [ ] T209 [US11] 更新 `fast_clifford/cga/base.py` 將 `get_care_layer()` 改為 `get_transform_layer()`
- [ ] T210 [US11] 更新 `fast_clifford/cga/registry.py` 配合新命名
- [ ] T211 [US11] 執行 US11 測試驗證 (T193-T199)

**Checkpoint**: Layer 命名統一完成

---

## Phase 15: High-Dimensional Runtime for New Operations

**Purpose**: 為新增的代數操作實作 6D+ 運行時支援

### Implementation

- [ ] T212 [US4+] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.outer_product`
- [ ] T213 [US4+] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.left_contraction`
- [ ] T214 [US4+] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.right_contraction`
- [ ] T215 [US4+] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.grade_select`
- [ ] T216 [US4+] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.dual`
- [ ] T217 [US4+] 在 `fast_clifford/cga/runtime.py` 實作 `RuntimeCGAAlgebra.normalize`

### Tests

- [ ] T218 [P] 新增 CGA(6) 新操作的 clifford 對照測試
- [ ] T219 [P] 執行所有運行時操作測試

**Checkpoint**: 所有操作 6D+ 運行時支援完成

---

## Phase 16: Polish & Cross-Cutting Concerns

**Purpose**: 整合、匯出、文檔更新

- [ ] T220 [P] 更新 `fast_clifford/__init__.py` 匯出新操作和類別
- [ ] T221 [P] 更新 README.md 新增 Extended Operations API 文檔
- [ ] T222 執行完整測試套件確認無迴歸
- [ ] T223 執行所有 ONNX 匯出測試驗證無 Loop/If 節點
- [ ] T224 執行 quickstart.md 範例驗證
- [ ] T225 效能基準測試：驗證 SC-001（達完整幾何積 50%+）
- [ ] T226 效能比較測試：Similitude vs EvenVersor 加速效果（SC-001a 30-50% 更快）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: 無依賴 - 可立即開始
- **Phase 2 (Foundational)**: 依賴 Phase 1 完成 - **阻擋所有 User Stories**
- **Phase 3-4 (US1, US4a)**: 依賴 Phase 2，可平行進行（都是 P1）
- **Phase 5 (US2)**: 依賴 Phase 2（P1，可與 Phase 3-4 平行）
- **Phase 6-7 (US3, US4)**: 依賴 Phase 2（P2，可平行進行）
- **Phase 8-12 (US5-US9)**: 依賴 Phase 2（P3，可平行進行）
  - **Phase 12 (US9)**: 額外依賴 Phase 5 (US2) 的 inner_product
- **Phase 13 (US10)**: 依賴 Phase 3-12 完成（需要所有操作）
- **Phase 14 (US11)**: 依賴 Phase 3-12 完成
- **Phase 15 (Runtime)**: 依賴 Phase 8-12 完成
- **Phase 16 (Polish)**: 依賴所有 User Stories 完成

### User Story Dependencies

```
US1 (EvenVersor Composition) ─┬─┬─> US10 (Operators)
US4a (Similitude Acceleration) ─┤ │
US2 (Inner Product) ────────────┼─┼─> US9 (Normalize)
US3 (Exp Map) ──────────────────┤ │
US4 (Runtime Core) ─────────────┤ │
US5 (Outer Product) ────────────┤ │
US6 (Contractions) ─────────────┤ │
US7 (Grade Selection) ──────────┤ │
US8 (Dual) ─────────────────────┤ │
US9 (Normalize) ────────────────┘ │
                                  └─> US11 (Unified Layers)
```

### Parallel Opportunities

- **Phase 1**: T001-T011 全部可平行
- **Phase 2**: T012-T027 依序（有依賴）
- **Phase 3-5**: US1, US4a, US2 可平行（各自獨立）
- **Phase 6-7**: US3, US4 可平行
- **Phase 8-12**: US5-US9 可平行（US9 需等 US2）
- **Phase 13**: T154-T174 測試可平行
- **Phase 14**: T202-T207 更新可平行
- **Phase 15**: T212-T217 可平行
- **Phase 16**: T220-T221 可平行

---

## Implementation Strategy

### MVP First (Phase 3 Only)

1. 完成 Phase 1: Setup (T001-T011)
2. 完成 Phase 2: Foundational (T012-T027)
3. 完成 Phase 3: User Story 1 (T028-T046)
4. **驗證**: 測試 `compose_even_versor` 功能
5. 可部署 MVP

### Incremental Delivery

**核心操作 (P1)**:
1. Setup + Foundational → codegen 準備完成
2. 加入 US1 (EvenVersor Composition) → 測試 → 交付
3. 加入 US4a (Similitude) → 測試 → 交付
4. 加入 US2 (Inner Product) → 測試 → 交付

**核心操作 (P2)**:
5. 加入 US3 (Exponential Map) → 測試 → 交付
6. 加入 US4 (Runtime Core) → 測試 → 交付

**代數操作 (P3)**:
7. 加入 US5 (Outer Product) → 測試 → 交付
8. 加入 US6 (Contractions) → 測試 → 交付
9. 加入 US7 (Grade Selection) → 測試 → 交付
10. 加入 US8 (Dual) → 測試 → 交付
11. 加入 US9 (Normalize) → 測試 → 交付

**使用者體驗與重構**:
12. 加入 US10 (Operators) → 測試 → 交付
13. 加入 US11 (Unified Layers) → 測試 → 交付
14. Runtime for New Ops → 測試 → 交付
15. Polish → 最終驗證

---

## Notes

- **[P]** = 不同檔案，無依賴
- **[Story]** = 對應 spec.md 的 User Story
- 每個 User Story 應可獨立完成和測試
- 測試失敗後再實作
- 每個任務或邏輯群組後提交 Git
- 任何 Checkpoint 可停下驗證
- **新命名**：Motor → EvenVersor，新增 Similitude（CGA 專用加速）
- **統一 API**：`compose()`, `sandwich_product()`, `reverse()` 自動路由到最佳實作
- **Layer 命名**：使用 `CliffordTransformLayer`（非 CGA 專用名稱）
