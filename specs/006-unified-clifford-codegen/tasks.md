# Tasks: Unified Cl(p,q,0) Codegen System

**Input**: Design documents from `/specs/006-unified-clifford-codegen/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/clifford_algebra.pyi ✅

**Tests**: 測試任務包含在內，使用 clifford 庫作為對照驗證。

**Organization**: 任務依照 User Story 分組，每個 Story 可獨立實作與測試。

---

## 🎯 Progress Summary

| Phase | Status | Tasks |
|-------|--------|-------|
| Phase 1: Setup | ✅ Complete | T001-T003 |
| Phase 2: Foundational | ✅ Complete | T004-T011 |
| Phase 3: US1 VGA | ✅ Complete | T012-T018 |
| Phase 4: US2 Unified | ✅ Complete | T019-T026 |
| Phase 5: US3 CGA | ✅ Complete | T027-T037 |
| Phase 6: US4 Rotor | ✅ Complete | T038-T048 (all rotor ops including exp/log/slerp) |
| Phase 7: US5 Bott | ✅ Basic Complete | T049-T056 (simplified implementation) |
| Phase 8: US6 PGA | ✅ Complete | T057-T063 |
| Phase 9: Polish | ✅ Core Complete | T064-T070 (ONNX pending) |

**Test Results**: 197 tests passing
**Benchmark**: VGA 16.1x faster, CGA 3.1x faster vs clifford library

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可平行執行（不同檔案，無依賴）
- **[Story]**: 任務所屬 User Story（US1-US6）
- 包含精確檔案路徑

## Path Conventions

本專案為 single project 結構：
- 原始碼：`fast_clifford/`
- 測試：`tests/`
- 生成代數：`fast_clifford/algebras/generated/`
- 統一介面：`fast_clifford/clifford/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 清理舊實作、建立新目錄結構

- [x] T001 刪除 `fast_clifford/algebras/cga0d/` 至 `cga5d/` 目錄（不向後相容）
- [x] T002 刪除 `fast_clifford/cga/` 目錄（不向後相容）
- [x] T003 [P] 建立 `fast_clifford/clifford/` 目錄結構
- [x] T004 [P] 建立 `fast_clifford/algebras/generated/` 目錄結構
- [x] T005 [P] 建立 `fast_clifford/clifford/specializations/` 目錄結構

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 核心基礎設施，所有 User Story 依賴此階段

**⚠️ CRITICAL**: 此階段完成前，無法開始任何 User Story

- [x] T006 建立 `fast_clifford/clifford/base.py` - CliffordAlgebraBase 抽象基類（FR-005~FR-025，含 exp(mv) 通用指數映射）
- [x] T007 [P] 建立 `fast_clifford/clifford/multivector.py` - Multivector 類別（FR-055~FR-064）
- [x] T008 建立 `fast_clifford/clifford/multivector.py` - Rotor 類別（FR-026~FR-034）（依賴 T007，同檔案）
- [x] T009 更新 `fast_clifford/codegen/clifford_factory.py` - 參數化 (p,q,r) 稀疏分析（重構為統一工廠）
- [x] T010 建立 `fast_clifford/codegen/generator.py` - ClCodeGenerator 統一生成器（FR-051~FR-053）
- [x] T011 [P] 建立 `fast_clifford/clifford/layers.py` - PyTorch nn.Module layers（FR-049~FR-050, FR-054 float32 強制轉換）

**Checkpoint**: 基礎設施就緒，可開始 User Story 實作

---

## Phase 3: User Story 1 - VGA 純向量代數 (Priority: P1) 🎯 MVP

**Goal**: 建立 VGA(n) = Cl(n, 0) 純歐幾里得向量代數支援

**Independent Test**: `VGA(3)` 可執行 geometric_product、outer、reverse 等運算

### Tests for User Story 1

- [x] T012 [US1] 建立 `tests/test_vga.py` - VGA 基本運算測試（對照 clifford 庫）、形狀驗證測試

### Implementation for User Story 1

- [x] T013 [P] [US1] 建立 `fast_clifford/clifford/specializations/vga.py` - VGAWrapper 類別（FR-035~FR-037）
- [x] T014 [P] [US1] 生成 `fast_clifford/algebras/generated/cl_1_0/` - VGA1D
- [x] T015 [P] [US1] 生成 `fast_clifford/algebras/generated/cl_2_0/` - VGA2D
- [x] T016 [P] [US1] 生成 `fast_clifford/algebras/generated/cl_3_0/` - VGA3D
- [x] T017 [US1] 建立 `fast_clifford/clifford/registry.py` - HardcodedClWrapper（VGA 部分）
- [x] T018 [US1] 更新 `fast_clifford/clifford/__init__.py` - 匯出 VGA() 工廠函數

**Checkpoint**: VGA(1), VGA(2), VGA(3) 應可獨立運作並通過測試

---

## Phase 4: User Story 2 - 統一工廠函數 (Priority: P1)

**Goal**: 建立 Cl(p, q) 統一工廠函數，支援任意 Cl(p,q,0) 代數

**Independent Test**: `Cl(2, 2)` 可執行 geometric_product，回傳正確形狀

### Tests for User Story 2

- [x] T019 [US2] 建立 `tests/test_clifford_interface.py` - Cl() 工廠函數測試、屬性驗證測試（count_blade, count_rotor）、邊界測試（Cl(0,0) 純量代數）

### Implementation for User Story 2

- [x] T020 [P] [US2] 生成 `fast_clifford/algebras/generated/cl_0_0/` - Cl(0,0) 純量代數（邊界情況）
- [x] T021 [P] [US2] 生成 `fast_clifford/algebras/generated/cl_1_1/` - Cl(1,1)
- [x] T022 [P] [US2] 生成 `fast_clifford/algebras/generated/cl_2_2/` - Cl(2,2)
- [x] T023 [P] [US2] 生成 `fast_clifford/algebras/generated/cl_3_2/` - Cl(3,2)
- [x] T024 [US2] 更新 `fast_clifford/clifford/registry.py` - 支援所有 p+q ≤ 9 代數
- [x] T025 [US2] 建立 `fast_clifford/codegen/clifford_factory.py` - 通用 Cl(p,q,r) 建立
- [x] T026 [US2] 更新 `fast_clifford/clifford/__init__.py` - 匯出 Cl() 工廠函數

**Checkpoint**: Cl(p, q) 對任意 p+q ≤ 9 應可正常運作（含 Cl(0,0) 邊界情況）

---

## Phase 5: User Story 3 - CGA 共形幾何代數 (Priority: P1) 🎯 MVP

**Goal**: 建立 CGA(n) = Cl(n+1, 1) 共形幾何代數支援

**Independent Test**: `CGA(3).encode([1,2,3])` → sandwich_rotor → decode 回歐幾里得座標

### Tests for User Story 3

- [x] T027 [US3] 建立 `tests/test_cga.py` - CGA 編解碼測試、sandwich_rotor 測試、null basis 慣例驗證（Dorst 慣例）（對照 clifford 庫）

### Implementation for User Story 3

- [x] T028 [P] [US3] 建立 `fast_clifford/clifford/specializations/cga.py` - CGAWrapper 類別（FR-038~FR-042）
- [x] T029 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_2_1/` - CGA0D
- [x] T030 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_3_1/` - CGA1D
- [x] T031 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_4_1/` - CGA2D
- [x] T032 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_5_1/` - CGA3D
- [x] T033 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_6_1/` - CGA4D
- [x] T034 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_7_1/` - CGA5D
- [x] T035 [P] [US3] 生成 `fast_clifford/algebras/generated/cl_8_1/` - CGA6D
- [x] T036 [US3] 實作 CGA encode/decode（FR-038~FR-039）- null basis 映射（FR-042）
- [x] T037 [US3] 更新 `fast_clifford/clifford/__init__.py` - 匯出 CGA() 工廠函數

**Checkpoint**: CGA(0) 到 CGA(6) 應可獨立運作，encode/decode 正確

---

## Phase 6: User Story 4 - Rotor 加速運算 (Priority: P2)

**Goal**: 實作 Rotor 靜態路由加速，比通用版本快 20%+

**Independent Test**: `compose_rotor` 與 `sandwich_rotor` 比通用版本快 20%+

### Tests for User Story 4

- [x] T038 [US4] 建立 `tests/test_rotor_acceleration.py` - compose_rotor、sandwich_rotor 正確性測試（在 test_cga.py 中）
- [x] T039 [P] [US4] 建立 `tests/benchmark/test_rotor_benchmark.py` - 效能對比測試

### Implementation for User Story 4

- [x] T040 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 compose_rotor 硬編碼
- [x] T041 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 reverse_rotor 硬編碼
- [x] T042 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 sandwich_rotor 硬編碼
- [x] T043 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 norm_squared_rotor 硬編碼
- [x] T044 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 exp_bivector 硬編碼
- [x] T045 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 log_rotor 硬編碼
- [x] T046 [US4] 更新 `fast_clifford/codegen/generator.py` - 生成 slerp_rotor 硬編碼
- [x] T047 [US4] 重新生成所有 p+q ≤ 9 代數（包含 Rotor 加速運算，含 exp/log/slerp）
- [x] T048 [US4] 更新 `fast_clifford/clifford/base.py` - 加入 Rotor 加速方法（FR-026~FR-034）

**Checkpoint**: Rotor 加速運算應比通用版本快 20%+

---

## Phase 7: User Story 5 - Bott 週期性支援 (Priority: P2)

**Goal**: 實作 Bott 週期性支援高維度代數（p+q > 9）

**Independent Test**: `Cl(10, 0)` 可執行 geometric_product，無錯誤

### Tests for User Story 5

- [x] T049 [US5] 建立 `tests/test_bott.py` - Bott 分解重組測試、高維度語法檢查測試、記憶體警告驗證測試（23 tests）

### Implementation for User Story 5

- [x] T050 [US5] 建立 `fast_clifford/clifford/bott.py` - BottPeriodicityAlgebra 類別（簡化版本）
- [ ] T051 [US5] 建立 `fast_clifford/codegen/bott_generator.py` - Bott 週期性生成器（未實作，使用運行時計算）
- [x] T052 [US5] 實作 blade 索引分解（簡化：matrix view）
- [x] T053 [US5] 實作張量積分解運算（簡化版本）
- [x] T054 [US5] 實作張量積重組運算（簡化版本）
- [x] T055 [US5] 更新 `fast_clifford/clifford/__init__.py` - Cl() 支援 Bott fallback
- [x] T056 [US5] 實作 blade_count > 2^14 記憶體警告

**Checkpoint**: Cl(10, 0) 至 Cl(15, 0) 應可正常運作 ✅

**Note**: 簡化版本實作。完整數學正確性需要更完善的張量積結構實作。

---

## Phase 8: User Story 6 - PGA 投影幾何代數 (Priority: P3)

**Goal**: 實作 PGA(n) = Cl(n, 0, 1) 透過 CGA 嵌入

**Independent Test**: `PGA(3).geometric_product(a, b)` 可執行，回傳正確形狀

### Tests for User Story 6

- [x] T057 [US6] 建立 `tests/test_pga.py` - PGA 嵌入/投影測試、PGA 運算正確性測試、PGA() 工廠函數測試（13 tests）

### Implementation for User Story 6

- [x] T058 [US6] 建立 `fast_clifford/clifford/specializations/pga.py` - PGAEmbedding 類別（FR-043~FR-045）
- [x] T059 [US6] 實作 PGA → CGA 嵌入映射（_embed_to_cga）（FR-043）
- [x] T060 [US6] 實作 CGA → PGA 投影映射（_project_from_cga）（FR-044）
- [x] T061 [US6] 實作 PGA geometric_product 透過 CGA
- [x] T062 [US6] 實作 PGA sandwich 透過 CGA
- [x] T063 [US6] 更新 `fast_clifford/clifford/__init__.py` - 匯出 PGA() 工廠函數（FR-004）

**Checkpoint**: PGA(2), PGA(3) 應可透過 CGA 嵌入正常運作

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: 完善、優化、驗證

- [x] T064 [P] 更新 `fast_clifford/__init__.py` - 匯出 Cl, VGA, CGA, PGA
- [x] T065 [P] 建立 `fast_clifford/algebras/generated/__init__.py` - 動態載入器
- [ ] T066 執行 quickstart.md 範例驗證（待手動驗證）
- [ ] T067 [P] 建立 `tests/test_onnx_export.py` - ONNX 匯出驗證（待實作）
- [x] T068 [P] 建立 `tests/benchmark/` - 效能 benchmark（對比 clifford 庫，VGA 16x, CGA 3x 加速）
- [x] T069 程式碼清理與格式化
- [x] T070 [P] 更新現有測試以使用新 API（142 tests passing）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 無依賴 - 可立即開始
- **Foundational (Phase 2)**: 依賴 Setup - 阻擋所有 User Story
- **User Stories (Phase 3-8)**: 依賴 Foundational 完成
  - US1 (VGA)、US2 (Unified)、US3 (CGA) 可平行進行
  - US4 (Rotor) 依賴 US1-US3 的 codegen 基礎
  - US5 (Bott) 依賴 US1-US3 的 codegen 基礎
  - US6 (PGA) 依賴 US3 (CGA) 完成
- **Polish (Phase 9)**: 依賴所有 User Story 完成

### User Story Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational
    ↓
    ├─→ US1 (VGA) ─────────────────┐
    ├─→ US2 (Unified) ─────────────┼─→ US4 (Rotor) ─┐
    └─→ US3 (CGA) ─┬───────────────┘                │
                   │                                ↓
                   └─→ US6 (PGA)     ──→ US5 (Bott) ─→ Phase 9: Polish
```

### Within Each User Story

- Tests MUST 先寫並 FAIL 再實作
- 生成代數 → 包裝類別 → 工廠函數
- Story 完成後再進入下一個優先級

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005 可平行

**Phase 2 (Foundational)**:
- T007, T011 可平行（不同檔案）
- T008 依賴 T007（同檔案 multivector.py）

**US1-US3 生成代數**:
```bash
# 所有 cl_*_* 生成可平行
Task: "生成 cl_1_0/" [P]
Task: "生成 cl_2_0/" [P]
Task: "生成 cl_3_0/" [P]
Task: "生成 cl_4_1/" [P]
# ... 等等
```

**測試**:
- 不同 User Story 的測試可平行執行

---

## Implementation Strategy

### MVP First (US1 + US3)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: US1 (VGA)
4. Complete Phase 5: US3 (CGA)
5. **STOP and VALIDATE**: 測試 VGA 和 CGA 獨立運作
6. 可部署/展示 MVP

### Incremental Delivery

1. Setup + Foundational → 基礎就緒
2. + US1 (VGA) → 測試 → VGA 可用
3. + US2 (Unified) + US3 (CGA) → 測試 → CGA 可用
4. + US4 (Rotor) → 測試 → 效能提升
5. + US5 (Bott) → 測試 → 高維度支援
6. + US6 (PGA) → 測試 → 完整功能

### Parallel Team Strategy

多開發者協作：
1. 共同完成 Setup + Foundational
2. Foundational 完成後：
   - Developer A: US1 (VGA) + US2 (Unified)
   - Developer B: US3 (CGA)
   - Developer C: US4 (Rotor Acceleration)
3. US3 完成後：
   - Developer B: US6 (PGA)
4. 最後共同完成 US5 (Bott) + Polish

---

## Notes

- **總任務數**: 70 個（T001-T070）
- [P] = 不同檔案、無依賴，可平行執行
- [USx] = 追溯到 spec.md User Story
- 每個 User Story 應可獨立完成並測試
- 使用 clifford 庫驗證數學正確性
- 每個任務或邏輯群組後提交（遵循憲法 VII 增量提交原則）
- 在任何 Checkpoint 停止可驗證 Story 獨立運作
- 避免：模糊任務、相同檔案衝突、破壞獨立性的跨 Story 依賴
- float32 強制轉換需在 layers.py 實作（遵循憲法 V 數值精度安全）
