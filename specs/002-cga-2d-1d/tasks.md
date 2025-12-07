# 任務清單：CGA2D 與 CGA1D 支援

**輸入**: 設計文件來自 `/specs/002-cga-2d-1d/`
**前置條件**: plan.md (必要), spec.md (必要), research.md, data-model.md, contracts/

**測試**: 本功能需要驗證測試（數值正確性、ONNX 匯出）

**組織**: 任務按使用者故事分組，以支援獨立實作與測試

## 格式：`[ID] [P?] [Story?] 描述`

- **[P]**: 可並行執行（不同檔案，無依賴）
- **[Story]**: 所屬使用者故事（例如 US1, US2, US3）
- 描述中包含確切檔案路徑

## 路徑慣例

本專案採用按代數類型分資料夾的模組化結構：

```text
fast_clifford/
├── codegen/                    # 通用程式碼生成器框架
│   ├── base.py                 # 基礎代數類別與生成器介面（現有）
│   ├── sparse_analysis.py      # 稀疏性分析工具（需擴展）
│   ├── generate.py             # 生成器主程式（需擴展）
│   └── cga_factory.py          # 新增：通用 CGA 代數工廠
├── algebras/                   # 各代數類型實作
│   ├── cga3d/                  # 現有 3D 實作
│   ├── cga2d/                  # 新增：2D 共形幾何代數 ← US1
│   │   ├── __init__.py
│   │   ├── algebra.py          # CGA2D 代數定義
│   │   ├── functional.py       # 生成的硬編碼函式
│   │   └── layers.py           # CGA2DCareLayer
│   └── cga1d/                  # 新增：1D 共形幾何代數 ← US2
│       ├── __init__.py
│       ├── algebra.py          # CGA1D 代數定義
│       ├── functional.py       # 生成的硬編碼函式
│       └── layers.py           # CGA1DCareLayer
└── tests/
    ├── cga2d/                  # CGA2D 測試 ← US1
    │   ├── __init__.py
    │   ├── test_numerical.py
    │   └── test_onnx.py
    └── cga1d/                  # CGA1D 測試 ← US2
        ├── __init__.py
        ├── test_numerical.py
        └── test_onnx.py

scripts/
├── generate_cga2d.py           # 新增：CGA2D 生成腳本
└── generate_cga1d.py           # 新增：CGA1D 生成腳本
```

---

## Phase 1: Setup（專案初始化）

**目的**: 建立新代數類型的目錄結構

- [x] T001 [P] 建立 fast_clifford/algebras/cga2d/__init__.py
- [x] T002 [P] 建立 fast_clifford/algebras/cga1d/__init__.py
- [x] T003 [P] 建立 fast_clifford/tests/cga2d/__init__.py
- [x] T004 [P] 建立 fast_clifford/tests/cga1d/__init__.py

**檢查點**: 目錄結構就緒

---

## Phase 2: Foundational（基礎建設）

**目的**: 所有使用者故事共用的核心基礎設施

**⚠️ 關鍵**: 此階段完成前，不可開始任何使用者故事

- [ ] T005 實作 fast_clifford/codegen/cga_factory.py - 建立通用 CGA 代數工廠函數 create_cga_algebra(euclidean_dim)
- [ ] T006 實作 fast_clifford/codegen/cga_factory.py - 新增 compute_grade_indices(euclidean_dim) 計算各 grade 的 blade 索引
- [ ] T007 實作 fast_clifford/codegen/cga_factory.py - 新增 compute_reverse_signs(blade_count, grade_indices) 計算反轉符號
- [ ] T008 擴展 fast_clifford/codegen/sparse_analysis.py - 新增 get_upgc_point_pattern(euclidean_dim) 工廠函數
- [ ] T009 擴展 fast_clifford/codegen/sparse_analysis.py - 新增 get_motor_pattern(euclidean_dim, grade_indices) 工廠函數
- [ ] T010 擴展 fast_clifford/codegen/generate.py - 新增 CGANDAlgebra 通用代數定義類別
- [ ] T011 擴展 fast_clifford/codegen/generate.py - 新增 CGANDCodeGenerator 通用代碼生成器類別

**檢查點**: 通用 CGA 生成器框架完成，可支援任意維度

---

## Phase 3: User Story 1 - 2D 幾何變換 (Priority: P1) 🎯 MVP

**目標**: 實作 CGA2D Cl(3,1) 完整支援，包含稀疏三明治積

**獨立測試**: 可透過將 2D 點編碼為 UPGC 表示、透過三明治積套用馬達變換、然後解碼回 2D 座標來測試。可與 clifford 函式庫比對驗證。

### 代數定義 (US1)

- [ ] T012 [US1] 實作 fast_clifford/algebras/cga2d/algebra.py - 使用 clifford 庫定義 CGA2D 代數
- [ ] T013 [US1] 實作 fast_clifford/algebras/cga2d/algebra.py - 提取幾何積乘法表 (16×16)
- [ ] T014 [US1] 實作 fast_clifford/algebras/cga2d/algebra.py - 定義 Null Basis ($n_o$, $n_\infty$) 並驗證性質
- [ ] T015 [P] [US1] 實作 fast_clifford/algebras/cga2d/algebra.py - 定義 16 個 blade 的索引映射與 grade 對應
- [ ] T016 [P] [US1] 實作 fast_clifford/algebras/cga2d/algebra.py - 定義 Reverse 符號表

### 生成器與 functional.py (US1)

- [ ] T017 [US1] 建立 scripts/generate_cga2d.py - 執行 CGA2D 生成器的主腳本
- [ ] T018 [US1] 執行生成器，輸出 fast_clifford/algebras/cga2d/functional.py - 包含常數定義
- [ ] T019 [US1] 驗證 fast_clifford/algebras/cga2d/functional.py - 包含 geometric_product_full() (16×16)
- [ ] T020 [US1] 驗證 fast_clifford/algebras/cga2d/functional.py - 包含 reverse_full() (16 分量)
- [ ] T021 [US1] 驗證 fast_clifford/algebras/cga2d/functional.py - 包含 upgc_encode() 和 upgc_decode()
- [ ] T022 [US1] 驗證 fast_clifford/algebras/cga2d/functional.py - 包含 reverse_motor() (8 分量)
- [ ] T023 [US1] 驗證 fast_clifford/algebras/cga2d/functional.py - 包含 sandwich_product_sparse() (~256 乘法)

### 測試 (US1)

- [ ] T024 [P] [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證幾何積正確性（對比 clifford）
- [ ] T025 [P] [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證 Null Basis 性質 ($n_o^2=0$, $n_\infty^2=0$, $n_o \cdot n_\infty = -1$)
- [ ] T026 [P] [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證幾何積結合律（隨機測試）
- [ ] T027 [P] [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證 Reverse 符號正確性
- [ ] T028 [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證稀疏三明治積正確性（旋轉變換）
- [ ] T029 [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證稀疏三明治積正確性（平移變換）
- [ ] T030 [US1] 實作 fast_clifford/tests/cga2d/test_numerical.py - 邊界案例測試：零向量、單位 Motor、未正規化 Motor

**檢查點**: CGA2D 代數與 functional.py 完成，數值測試通過

---

## Phase 4: User Story 2 - 1D 幾何變換 (Priority: P2)

**目標**: 實作 CGA1D Cl(2,1) 完整支援，包含稀疏三明治積

**獨立測試**: 可透過將 1D 純量編碼為 UPGC 表示、套用馬達變換、然後解碼回來測試。可與 clifford 函式庫比對驗證。

### 代數定義 (US2)

- [ ] T031 [US2] 實作 fast_clifford/algebras/cga1d/algebra.py - 使用 clifford 庫定義 CGA1D 代數
- [ ] T032 [US2] 實作 fast_clifford/algebras/cga1d/algebra.py - 提取幾何積乘法表 (8×8)
- [ ] T033 [US2] 實作 fast_clifford/algebras/cga1d/algebra.py - 定義 Null Basis 並驗證性質
- [ ] T034 [P] [US2] 實作 fast_clifford/algebras/cga1d/algebra.py - 定義 8 個 blade 的索引映射與 grade 對應
- [ ] T035 [P] [US2] 實作 fast_clifford/algebras/cga1d/algebra.py - 定義 Reverse 符號表

### 生成器與 functional.py (US2)

- [ ] T036 [US2] 建立 scripts/generate_cga1d.py - 執行 CGA1D 生成器的主腳本
- [ ] T037 [US2] 執行生成器，輸出 fast_clifford/algebras/cga1d/functional.py - 包含常數定義
- [ ] T038 [US2] 驗證 fast_clifford/algebras/cga1d/functional.py - 包含 geometric_product_full() (8×8)
- [ ] T039 [US2] 驗證 fast_clifford/algebras/cga1d/functional.py - 包含 reverse_full() (8 分量)
- [ ] T040 [US2] 驗證 fast_clifford/algebras/cga1d/functional.py - 包含 upgc_encode() 和 upgc_decode()
- [ ] T041 [US2] 驗證 fast_clifford/algebras/cga1d/functional.py - 包含 reverse_motor() (4 分量)
- [ ] T042 [US2] 驗證 fast_clifford/algebras/cga1d/functional.py - 包含 sandwich_product_sparse() (~72 乘法)

### 測試 (US2)

- [ ] T043 [P] [US2] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證幾何積正確性（對比 clifford）
- [ ] T044 [P] [US2] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證 Null Basis 性質
- [ ] T045 [P] [US2] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證幾何積結合律
- [ ] T046 [P] [US2] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證 Reverse 符號正確性
- [ ] T047 [US2] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證稀疏三明治積正確性（平移變換）
- [ ] T048 [US2] 實作 fast_clifford/tests/cga1d/test_numerical.py - 邊界案例測試

**檢查點**: CGA1D 代數與 functional.py 完成，數值測試通過

---

## Phase 5: User Story 3 - PyTorch 訓練整合 (Priority: P1)

**目標**: 實作 CGA2DCareLayer 和 CGA1DCareLayer，支援 PyTorch 訓練流程

**獨立測試**: 可透過建立使用 CGA 層的簡單神經網路、執行前向/反向傳播、並驗證梯度流動來測試。

### CGA2D 層封裝 (US3)

- [ ] T049 [US3] 實作 fast_clifford/algebras/cga2d/layers.py - 定義 CGA2DCareLayer 類別骨架
- [ ] T050 [US3] 實作 fast_clifford/algebras/cga2d/layers.py - 實作 forward() 方法，包含 fp16→fp32→fp16 轉換
- [ ] T051 [US3] 實作 fast_clifford/algebras/cga2d/layers.py - 整合 sandwich_product_sparse 函式
- [ ] T052 [US3] 更新 fast_clifford/algebras/cga2d/__init__.py - 匯出 CGA2DCareLayer

### CGA1D 層封裝 (US3)

- [ ] T053 [US3] 實作 fast_clifford/algebras/cga1d/layers.py - 定義 CGA1DCareLayer 類別骨架
- [ ] T054 [US3] 實作 fast_clifford/algebras/cga1d/layers.py - 實作 forward() 方法
- [ ] T055 [US3] 實作 fast_clifford/algebras/cga1d/layers.py - 整合 sandwich_product_sparse 函式
- [ ] T056 [US3] 更新 fast_clifford/algebras/cga1d/__init__.py - 匯出 CGA1DCareLayer

### 測試 (US3)

- [ ] T057 [P] [US3] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證 CGA2DCareLayer 數值正確性
- [ ] T058 [P] [US3] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證 CGA2D 精度轉換（fp16→fp32→fp16）
- [ ] T059 [P] [US3] 實作 fast_clifford/tests/cga2d/test_numerical.py - 驗證 CGA2D 梯度流動
- [ ] T060 [P] [US3] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證 CGA1DCareLayer 數值正確性
- [ ] T061 [P] [US3] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證 CGA1D 精度轉換
- [ ] T062 [P] [US3] 實作 fast_clifford/tests/cga1d/test_numerical.py - 驗證 CGA1D 梯度流動

**檢查點**: CGA2DCareLayer 和 CGA1DCareLayer 可用於 PyTorch 訓練

---

## Phase 6: Verification（驗證）

**目的**: 憲法合規驗證 - ONNX 匯出與跨平台測試

### ONNX 匯出測試

- [ ] T063 [P] 實作 fast_clifford/tests/cga2d/test_onnx.py - CGA2D ONNX 匯出測試（opset 17）
- [ ] T064 [P] 實作 fast_clifford/tests/cga2d/test_onnx.py - 驗證 CGA2D ONNX 計算圖無 Loop 節點
- [ ] T065 [P] 實作 fast_clifford/tests/cga2d/test_onnx.py - 驗證 CGA2D ONNX 只有 Add/Mul/Neg 等基本算子
- [ ] T066 [P] 實作 fast_clifford/tests/cga1d/test_onnx.py - CGA1D ONNX 匯出測試（opset 17）
- [ ] T067 [P] 實作 fast_clifford/tests/cga1d/test_onnx.py - 驗證 CGA1D ONNX 計算圖無 Loop 節點
- [ ] T068 [P] 實作 fast_clifford/tests/cga1d/test_onnx.py - 驗證 CGA1D ONNX 只有 Add/Mul/Neg 等基本算子

### 跨平台測試

- [ ] T069 [P] 實作 fast_clifford/tests/cga2d/test_numerical.py - CGA2D 跨平台測試（MPS/CUDA/CPU）
- [ ] T070 [P] 實作 fast_clifford/tests/cga1d/test_numerical.py - CGA1D 跨平台測試（MPS/CUDA/CPU）

**檢查點**: 所有憲法約束驗證通過

---

## Phase 7: Polish & Cross-Cutting Concerns

**目的**: 收尾與整合

- [ ] T071 [P] 應用 torch.jit.script 於 fast_clifford/algebras/cga2d/functional.py
- [ ] T072 [P] 應用 torch.jit.script 於 fast_clifford/algebras/cga1d/functional.py
- [ ] T073 更新 fast_clifford/algebras/__init__.py - 匯出 cga2d, cga1d 模組
- [ ] T074 更新 fast_clifford/__init__.py - 匯出 cga2d, cga1d
- [ ] T075 執行 specs/002-cga-2d-1d/quickstart.md 驗證所有範例可運行
- [ ] T076 [P] 程式碼清理與格式化

---

## 依賴與執行順序

### Phase 依賴

- **Setup (Phase 1)**: 無依賴 - 可立即開始
- **Foundational (Phase 2)**: 依賴 Setup 完成 - 阻塞所有使用者故事
- **User Story 1 (Phase 3)**: 依賴 Foundational 完成
- **User Story 2 (Phase 4)**: 依賴 Foundational 完成，可與 US1 並行
- **User Story 3 (Phase 5)**: 依賴 Phase 3, 4 完成（需要 functional.py）
- **Verification (Phase 6)**: 依賴 Phase 5 完成
- **Polish (Phase 7)**: 依賴 Phase 6 完成

### User Story 依賴

```
                    ┌─────────────────┐
                    │   Foundational  │
                    │    (Phase 2)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  User Story 1   │           │  User Story 2   │
    │  CGA2D (P1)     │           │  CGA1D (P2)     │
    │   (Phase 3)     │           │   (Phase 4)     │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  User Story 3   │
                  │  PyTorch 整合   │
                  │   (Phase 5)     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Verification   │
                  │   (Phase 6)     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Polish      │
                  │   (Phase 7)     │
                  └─────────────────┘
```

### 並行機會

- Phase 1: T001-T004 可並行（不同目錄）
- Phase 3/4: US1 和 US2 可並行執行（不同代數類型）
- Phase 3 內: T015-T016 可並行, T024-T027 可並行
- Phase 4 內: T034-T035 可並行, T043-T046 可並行
- Phase 5 內: T057-T062 可並行
- Phase 6 內: T063-T070 可並行
- Phase 7 內: T071-T072 可並行

---

## 並行範例

### Phase 1 並行任務：
```bash
# 同時執行（四個不同目錄）：
T001: 建立 fast_clifford/algebras/cga2d/__init__.py
T002: 建立 fast_clifford/algebras/cga1d/__init__.py
T003: 建立 fast_clifford/tests/cga2d/__init__.py
T004: 建立 fast_clifford/tests/cga1d/__init__.py
```

### Phase 3 + Phase 4 並行（不同團隊成員）：
```bash
# Developer A: CGA2D (User Story 1)
T012-T030: 完成 CGA2D 代數與測試

# Developer B: CGA1D (User Story 2)
T031-T048: 完成 CGA1D 代數與測試
```

### Phase 6 並行測試：
```bash
# 同時執行所有 ONNX 測試：
T063-T065: CGA2D ONNX 測試
T066-T068: CGA1D ONNX 測試
T069-T070: 跨平台測試
```

---

## 實作策略

### MVP First (User Story 1 Only)

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational
3. 完成 Phase 3: User Story 1 (CGA2D)
4. **停止並驗證**: 測試 CGA2D 功能
5. 可交付 MVP：CGA2D 幾何變換支援

### 增量交付

1. Setup + Foundational → 基礎就緒
2. User Story 1 (CGA2D) → 驗證 → MVP！
3. User Story 2 (CGA1D) → 驗證 → 1D 支援
4. User Story 3 (PyTorch 整合) → 驗證 → 訓練就緒
5. Verification → 憲法合規確認
6. Polish → 生產就緒

### 平行團隊策略

如有多位開發者：
1. 團隊共同完成 Setup + Foundational
2. Foundational 完成後：
   - Developer A: User Story 1 (CGA2D)
   - Developer B: User Story 2 (CGA1D)
3. US1 和 US2 完成後：
   - 共同完成 User Story 3 (PyTorch 整合)
4. 共同完成 Verification 和 Polish

---

## 備註

- [P] 標記 = 不同檔案，無依賴，可並行
- [Story] 標籤 = 對應 spec.md 中的使用者故事
- 每個使用者故事應可獨立完成與測試
- 每個任務或邏輯群組後提交 commit
- 在任何檢查點停止以獨立驗證故事
- 避免：模糊任務、同檔案衝突、破壞獨立性的跨故事依賴

### 成功標準對應

| 成功標準 | 相關任務 |
|----------|----------|
| SC-001 CGA2D <260 乘法 | T023 |
| SC-002 CGA1D <80 乘法 | T042 |
| SC-003 數值誤差 <1e-6 | T024-T030, T043-T048 |
| SC-004 ONNX 無 Loop | T064, T067 |
| SC-005 >100K pts/s | T069, T070 |
| SC-006 測試通過 | Phase 3-6 所有測試 |
| SC-007 可匯入使用 | T074, T075 |
