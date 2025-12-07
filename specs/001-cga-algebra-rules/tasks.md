# 任務清單：CGA 幾何代數規則定義

**輸入**: 設計文件來自 `/specs/001-cga-algebra-rules/`
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
│   ├── base.py                 # 基礎代數類別與生成器介面
│   ├── sparse_analysis.py      # 稀疏性分析工具
│   └── generate.py             # 生成器主程式
├── algebras/                   # 各代數類型實作
│   └── cga3d/                  # 3D 共形幾何代數 Cl(4,1) ← 本功能
│       ├── algebra.py          # CGA 代數定義
│       ├── functional.py       # 生成的硬編碼函式
│       └── layers.py           # CGACareLayer (nn.Module)
└── tests/
    └── cga3d/                  # CGA3D 測試
scripts/
└── generate_cga3d.py           # 執行 CGA3D 生成器
```

---

## Phase 1: Setup（專案初始化）

**目的**: 建立專案結構與基礎設施

- [ ] T001 建立專案目錄結構 per plan.md 的專案結構定義
- [ ] T002 使用 uv 初始化 Python 專案，建立 pyproject.toml 包含依賴：clifford, sympy, torch, onnx, pytest
- [ ] T003 [P] 建立 fast_clifford/__init__.py 模組入口
- [ ] T004 [P] 建立 fast_clifford/codegen/__init__.py
- [ ] T005 [P] 建立 fast_clifford/algebras/__init__.py
- [ ] T006 [P] 建立 fast_clifford/algebras/cga3d/__init__.py
- [ ] T007 [P] 建立 fast_clifford/tests/__init__.py
- [ ] T007.1 [P] 建立 fast_clifford/tests/cga3d/__init__.py

**檢查點**: 專案結構就緒，可執行 `uv sync` 安裝依賴

---

## Phase 2: Foundational（基礎建設）

**目的**: 所有使用者故事共用的核心基礎設施

**⚠️ 關鍵**: 此階段完成前，不可開始任何使用者故事

- [ ] T008 實作 fast_clifford/algebras/cga3d/algebra.py - 使用 clifford 庫定義 CGA 代數
- [ ] T009 實作 fast_clifford/algebras/cga3d/algebra.py - 提取幾何積乘法表 (gmt)
- [ ] T010 實作 fast_clifford/algebras/cga3d/algebra.py - 定義 Null Basis ($n_o$, $n_\infty$) 並驗證性質
- [ ] T011 [P] 實作 fast_clifford/algebras/cga3d/algebra.py - 定義 32 個 blade 的索引映射與 grade 對應
- [ ] T012 [P] 實作 fast_clifford/algebras/cga3d/algebra.py - 定義 Reverse 符號表

**檢查點**: 基礎代數定義完成，可查詢任意 blade 乘積

---

## Phase 3: User Story 1 - 生成器讀取代數規則 (Priority: P1) 🎯 MVP

**目標**: 程式碼生成器能讀取完整的 CGA 代數規則，生成硬編碼幾何積函式

**獨立測試**: 驗證生成器輸出的乘法結果與 clifford 庫計算結果相符（誤差 < 1e-6）

### 實作 User Story 1

- [ ] T013 [US1] 實作 fast_clifford/codegen/base.py - 定義基礎代數類別與生成器介面
- [ ] T014 [US1] 實作 fast_clifford/codegen/generate.py - 定義程式碼生成器類別骨架
- [ ] T015 [US1] 實作 fast_clifford/codegen/generate.py - 生成 blade 索引常數定義
- [ ] T016 [US1] 實作 fast_clifford/codegen/generate.py - 生成 geometric_product_full() 函式（完整 32×32 展開）
- [ ] T017 [US1] 實作 fast_clifford/codegen/generate.py - 生成 reverse() 函式（完整 32 分量版本）
- [ ] T018 [US1] 建立 scripts/generate_cga3d.py - 執行 CGA3D 生成器的主腳本
- [ ] T019 [US1] 執行生成器，輸出 fast_clifford/algebras/cga3d/functional.py
- [ ] T020 [US1] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證幾何積正確性（對比 clifford）
- [ ] T021 [US1] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證 Null Basis 性質 ($n_o \cdot n_\infty = -1$)
- [ ] T021.1 [US1] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證幾何積結合律 $(a \cdot b) \cdot c = a \cdot (b \cdot c)$（隨機測試）
- [ ] T021.2 [US1] 驗證 fast_clifford/algebras/cga3d/algebra.py 可被 codegen/generate.py 直接匯入使用（SC-004）

**檢查點**: User Story 1 完成，生成的幾何積函式數值正確

---

## Phase 4: User Story 2 - 稀疏性假設應用 (Priority: P1)

**目標**: 利用 UPGC 點和 Motor 的稀疏性，生成優化的三明治積函式

**獨立測試**: 驗證 $M \times X \times \widetilde{M}$ 輸出只有 Grade 1 有非零值

### 實作 User Story 2

- [ ] T022 [US2] 實作 fast_clifford/codegen/sparse_analysis.py - 定義 UPGC 點稀疏模式 (Grade 1, 5 個分量)
- [ ] T023 [US2] 實作 fast_clifford/codegen/sparse_analysis.py - 定義 Motor 稀疏模式 (Grade 0,2,4, 16 個分量)
- [ ] T024 [US2] 實作 fast_clifford/codegen/sparse_analysis.py - 分析三明治積輸出稀疏性
- [ ] T025 [US2] 實作 fast_clifford/codegen/generate.py - 生成 sandwich_product_sparse() 函式
- [ ] T026 [US2] 實作 fast_clifford/codegen/generate.py - 生成 upgc_encode() 和 upgc_decode() 函式
- [ ] T027 [US2] 重新執行生成器，更新 fast_clifford/algebras/cga3d/functional.py
- [ ] T028 [US2] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證稀疏三明治積正確性
- [ ] T029 [US2] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證計算量 < 200 次乘法（靜態分析生成程式碼的乘法算子數量）
- [ ] T029.1 [US2] 實作 fast_clifford/tests/cga3d/test_numerical.py - 邊界案例測試：零向量 UPGC 點、單位 Motor、純旋轉 Motor

**檢查點**: User Story 2 完成，稀疏優化的三明治積可用

---

## Phase 5: User Story 3 - Reverse 操作定義 (Priority: P2)

**目標**: 提供完整的 Reverse 操作支援

**獨立測試**: 驗證 $\widetilde{M}$ 的每個 blade 係數符號正確

### 實作 User Story 3

- [ ] T030 [US3] 實作 fast_clifford/codegen/generate.py - 生成 reverse_motor() 函式（稀疏 16 分量版本，用於 sandwich_product_sparse）
- [ ] T031 [US3] 重新執行生成器，更新 fast_clifford/algebras/cga3d/functional.py
- [ ] T032 [US3] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證 Reverse 符號正確性
- [ ] T033 [US3] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證 Motor Reverse 後 Grade 0,4 不變，Grade 2 反號

**檢查點**: User Story 3 完成，Reverse 操作可獨立使用

---

## Phase 6: PyTorch 封裝 (Phase 2 from pipeline)

**目的**: 將生成的函式封裝為 PyTorch nn.Module

- [ ] T034 實作 fast_clifford/algebras/cga3d/layers.py - 定義 CGACareLayer 類別骨架
- [ ] T035 實作 fast_clifford/algebras/cga3d/layers.py - 實作 forward() 方法，包含 fp16→fp32→fp16 轉換
- [ ] T036 實作 fast_clifford/algebras/cga3d/layers.py - 整合 sandwich_product_sparse 函式
- [ ] T037 [P] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證 CGACareLayer 數值正確性
- [ ] T038 [P] 實作 fast_clifford/tests/cga3d/test_numerical.py - 驗證精度轉換不影響結果

**檢查點**: CGACareLayer 可用於 PyTorch 訓練

---

## Phase 7: Verification（驗證）

**目的**: 憲法合規驗證 - ONNX 匯出與跨平台測試

- [ ] T039 實作 fast_clifford/tests/cga3d/test_onnx.py - ONNX 匯出測試（opset 17）
- [ ] T040 實作 fast_clifford/tests/cga3d/test_onnx.py - 驗證 ONNX 計算圖無 Loop 節點
- [ ] T041 實作 fast_clifford/tests/cga3d/test_onnx.py - 驗證 ONNX 計算圖只有 Add/Mul/Neg 等基本算子
- [ ] T042 [P] 實作 fast_clifford/tests/cga3d/test_numerical.py - 跨平台測試（MPS/CUDA/CPU）
- [ ] T043 [P] 實作 fast_clifford/tests/cga3d/test_numerical.py - 精度測試（float32 vs float16）

**檢查點**: 所有憲法約束驗證通過

---

## Phase 8: Polish & Cross-Cutting Concerns

**目的**: 收尾與優化

- [ ] T044 [P] 應用 torch.jit.script 於 fast_clifford/algebras/cga3d/functional.py（MPS 優化）
- [ ] T045 [P] 更新 fast_clifford/algebras/cga3d/__init__.py - 匯出公開 API
- [ ] T046 [P] 更新 fast_clifford/__init__.py - 匯出 cga3d 模組
- [ ] T047 執行 quickstart.md 驗證所有範例可運行
- [ ] T048 [P] 程式碼清理與格式化

---

## 依賴與執行順序

### Phase 依賴

- **Setup (Phase 1)**: 無依賴 - 可立即開始
- **Foundational (Phase 2)**: 依賴 Setup 完成 - 阻塞所有使用者故事
- **User Story 1 (Phase 3)**: 依賴 Foundational 完成
- **User Story 2 (Phase 4)**: 依賴 Phase 3 完成（需要基礎幾何積）
- **User Story 3 (Phase 5)**: 可與 Phase 4 並行（Reverse 獨立於稀疏性）
- **PyTorch 封裝 (Phase 6)**: 依賴 Phase 4, 5 完成
- **Verification (Phase 7)**: 依賴 Phase 6 完成
- **Polish (Phase 8)**: 依賴 Phase 7 完成

### User Story 依賴

```
US1 (代數規則) ──────────────────┐
                                 │
                                 ▼
US2 (稀疏性) ─────────────────> Phase 6 (封裝)
                                 │
US3 (Reverse) ──┘                ▼
              (可並行)        Phase 7 (驗證)
```

### 並行機會

- Phase 1: T003-T007 可並行
- Phase 2: T011-T012 可並行
- Phase 5: 可與 Phase 4 並行
- Phase 7: T041-T042 可並行
- Phase 8: T043-T045, T047 可並行

---

## 並行範例

```bash
# Phase 1 並行任務（同時執行）：
T003: 建立 fast_clifford/__init__.py
T004: 建立 fast_clifford/codegen/__init__.py
T005: 建立 fast_clifford/algebras/__init__.py
T006: 建立 fast_clifford/algebras/cga3d/__init__.py
T007: 建立 fast_clifford/tests/__init__.py
T007.1: 建立 fast_clifford/tests/cga3d/__init__.py

# Phase 2 並行任務：
T011: 定義 blade 索引映射
T012: 定義 Reverse 符號表
```

---

## 實作策略

### MVP First (僅 User Story 1)

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational
3. 完成 Phase 3: User Story 1
4. **停止並驗證**: 測試生成的幾何積函式
5. 可交付 MVP：基礎代數規則生成器

### 增量交付

1. Setup + Foundational → 基礎就緒
2. User Story 1 → 完整幾何積 → 驗證
3. User Story 2 → 稀疏優化 → 驗證
4. User Story 3 → Reverse 支援 → 驗證
5. PyTorch 封裝 → CGACareLayer 可用
6. Verification → 憲法合規確認
7. Polish → 生產就緒

---

## 備註

- [P] 標記 = 不同檔案，無依賴，可並行
- [Story] 標籤 = 對應 spec.md 中的使用者故事
- 每個使用者故事應可獨立完成與測試
- 每個任務或邏輯群組後提交 commit
- 在任何檢查點停止以獨立驗證故事
- 避免：模糊任務、同檔案衝突、破壞獨立性的跨故事依賴

### 術語對照

| tasks.md Phase | plan.md Pipeline | 說明 |
|----------------|------------------|------|
| Phase 1-2 (Setup/Foundational) | — | 專案初始化（plan.md 未涵蓋） |
| Phase 3-5 (User Stories) | Phase 1: Codegen | 程式碼生成器實作 |
| Phase 6 (PyTorch 封裝) | Phase 2: Wrapper | nn.Module 封裝 |
| Phase 7 (Verification) | Phase 4: Verification | ONNX/跨平台驗證 |
| Phase 8 (Polish) | Phase 3: MPS Optimization | 收尾與優化 |
