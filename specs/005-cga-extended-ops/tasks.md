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

- [ ] T001 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_motor_compose_terms(dim)` 函式
- [ ] T002 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_inner_product_signs(dim)` 函式
- [ ] T003 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_bivector_squared_terms(dim)` 函式
- [ ] T004 [P] 在 fast_clifford/codegen/sparse_analysis.py 新增 `get_bivector_indices(dim)` 函式

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 擴展 codegen 生成器和統一介面基礎類別

**⚠️ CRITICAL**: 所有 User Story 依賴此 Phase 完成

- [ ] T005 在 fast_clifford/codegen/generate.py 新增 `_generate_motor_compose_sparse()` 方法
- [ ] T006 在 fast_clifford/codegen/generate.py 新增 `_generate_inner_product_full()` 方法
- [ ] T007 在 fast_clifford/codegen/generate.py 新增 `_generate_bivector_squared_scalar()` 輔助方法
- [ ] T008 在 fast_clifford/codegen/generate.py 新增 `_generate_exp_bivector()` 方法
- [ ] T009 更新 fast_clifford/codegen/generate.py 的 `generate_module()` 和 `generate_sparse_section()` 整合新操作
- [ ] T010 在 fast_clifford/cga/base.py 新增 `motor_compose`, `inner_product`, `exp_bivector` 抽象方法
- [ ] T011 在 fast_clifford/cga/base.py 新增 `bivector_count` 屬性

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

## Phase 7: Layer 重新命名 (Refactor)

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

### Tests for User Story 5

- [ ] T075 [P] [US5] 建立 fast_clifford/tests/test_unified_layers.py 測試框架
- [ ] T076 [P] [US5] 新增 CGATransformLayer 實例化測試 (n=0-5)
- [ ] T077 [P] [US5] 新增 CGAEncoder/CGADecoder 輸入輸出形狀測試
- [ ] T078 [P] [US5] 新增 CGAPipeline 端對端測試
- [ ] T079 [P] [US5] 新增 get_transform_layer() 方法測試
- [ ] T080 [P] [US5] 新增運行時 (n≥6) 統一 Layer 測試

### Implementation

- [ ] T081 [P] [US5] 在 fast_clifford/cga/ 新增 layers.py 定義統一介面類別 `CGATransformLayer`, `CGAEncoder`, `CGADecoder`, `CGAPipeline`
- [ ] T082 [P] [US5] 移除 fast_clifford/algebras/cga0d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T083 [P] [US5] 移除 fast_clifford/algebras/cga1d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T084 [P] [US5] 移除 fast_clifford/algebras/cga2d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T085 [P] [US5] 移除 fast_clifford/algebras/cga3d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T086 [P] [US5] 移除 fast_clifford/algebras/cga4d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T087 [P] [US5] 移除 fast_clifford/algebras/cga5d/layers.py 的舊類別，改為從 cga/layers.py 匯入
- [ ] T088 [US5] 更新 fast_clifford/cga/runtime.py 移除 `RuntimeCGACareLayer`，改用統一 `CGATransformLayer`
- [ ] T089 [US5] 更新 fast_clifford/cga/base.py 將 `get_care_layer()` 改為 `get_transform_layer()`（移除舊方法）
- [ ] T090 [US5] 更新 fast_clifford/cga/registry.py 配合新命名
- [ ] T091 [US5] 執行 US5 測試驗證 (T075-T080)

**Checkpoint**: Layer 命名統一完成

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 整合、匯出、文檔更新

- [ ] T092 [P] 更新 fast_clifford/__init__.py 匯出新操作和統一 Layer 類別
- [ ] T093 [P] 更新 README.md 新增 Extended Operations API 文檔和新 Layer 命名
- [ ] T094 執行完整測試套件確認無迴歸
- [ ] T095 執行所有 ONNX 匯出測試驗證無 Loop/If 節點
- [ ] T096 執行 quickstart.md 範例驗證
- [ ] T097 效能基準測試：驗證 SC-001（達完整幾何積 50%+）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: 無依賴 - 可立即開始
- **Phase 2 (Foundational)**: 依賴 Phase 1 完成 - **阻擋所有 User Stories**
- **Phase 3-6 (User Stories)**: 依賴 Phase 2 完成
  - US1 和 US2 可平行進行（都是 P1）
  - US3 和 US4 可平行進行（都是 P2）
- **Phase 7 (Polish)**: 依賴所有 User Stories 完成

### User Story Dependencies

- **US1 (Motor Composition)**: 可於 Phase 2 後立即開始
- **US2 (Inner Product)**: 可於 Phase 2 後立即開始，與 US1 獨立
- **US3 (Exponential Map)**: 可於 Phase 2 後開始，與 US1/US2 獨立
- **US4 (Runtime)**: 可於 Phase 2 後開始，但建議在 US1-3 之後（可參考硬編碼實作）

### Within Each User Story

- Tests (T012-T018 等) 應先撰寫並確認失敗
- functional.py 生成在 registry.py 之前
- 核心實作在整合之前
- Story 完成後再進入下一個

### Parallel Opportunities

- Phase 1: T001-T004 全部可平行
- Phase 2: T005-T011 依序（有依賴）
- Phase 3: T012-T018 測試可平行，T019-T024 生成可平行
- Phase 4: T028-T035 測試可平行，T036-T041 更新可平行
- Phase 5: T045-T052 測試可平行，T053-T058 更新可平行
- Phase 6: T063-T068 測試可平行
- Phase 7: T076-T077 可平行

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

1. 完成 Phase 1: Setup (T001-T004)
2. 完成 Phase 2: Foundational (T005-T011)
3. 完成 Phase 3: User Story 1 (T012-T027)
4. **驗證**: 測試 motor_compose 功能
5. 可部署 MVP

### Incremental Delivery

1. Setup + Foundational → codegen 準備完成
2. 加入 US1 (Motor Composition) → 測試 → 交付
3. 加入 US2 (Inner Product) → 測試 → 交付
4. 加入 US3 (Exponential Map) → 測試 → 交付
5. 加入 US4 (Runtime) → 測試 → 交付
6. Polish → 最終驗證

---

## Notes

- [P] = 不同檔案，無依賴
- [Story] = 對應 spec.md 的 User Story
- 每個 User Story 應可獨立完成和測試
- 測試失敗後再實作
- 每個任務或邏輯群組後提交 Git
- 任何 Checkpoint 可停下驗證
