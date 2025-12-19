# 任務清單：Bott 週期性優化

**輸入**: 設計文件來自 `/specs/007-bott-optimization/`
**先決條件**: plan.md（必要）, spec.md（必要）, research.md, data-model.md, contracts/

**組織**: 任務按使用者故事分組，以便獨立實作和測試每個故事。

## 格式: `[ID] [P?] [Story] 描述`

- **[P]**: 可並行執行（不同檔案，無依賴）
- **[Story]**: 此任務屬於哪個使用者故事（例如 US1, US2, US3, US4）
- 描述中包含確切的檔案路徑

---

## Phase 1: 設定（共用基礎設施）

**目的**: 專案初始化和基本結構

- [x] T001 備份現有 bott.py 用於效能基準比較，在 fast_clifford/clifford/bott_backup.py
- [x] T002 確認現有 212 個測試全部通過，執行 `uv run pytest tests/ -v`

---

## Phase 2: 基礎（阻塞性先決條件）

**目的**: 必須在任何使用者故事開始前完成的核心基礎設施

**⚠️ 重要**: 此階段完成前不能開始使用者故事

- [x] T003 更新 HARDCODED_THRESHOLD 從 512 到 128 並調整 Cl() 工廠路由邏輯在 fast_clifford/clifford/__init__.py

**檢查點**: 基礎準備就緒 - 可以開始使用者故事實作

---

## Phase 3: 使用者故事 1 - 減少儲存空間 (優先級: P1) 🎯 MVP

**目標**: 將預生成代數從 55 個減少到 20 個，儲存空間從 600MB 減少到 25MB

**獨立測試**: 測量 `fast_clifford/algebras/generated/` 目錄大小，確認低於 30MB，且現有測試通過

### 實作

- [x] T004 [P] [US1] 刪除 p+q=8 的代數：cl_8_0, cl_7_1, cl_6_2, cl_5_3, cl_4_4, cl_3_5, cl_2_6, cl_1_7, cl_0_8 在 fast_clifford/algebras/generated/
- [x] T005 [P] [US1] 刪除 p+q=9 的代數：cl_9_0, cl_8_1, cl_7_2, cl_6_3, cl_5_4, cl_4_5, cl_3_6, cl_2_7, cl_1_8, cl_0_9 在 fast_clifford/algebras/generated/
- [x] T006 [P] [US1] 刪除 p<q 且 p+q<=2 的代數：cl_0_1, cl_0_2 在 fast_clifford/algebras/generated/
- [x] T007 [P] [US1] 刪除 p<q 且 p+q=3 的代數：cl_0_3, cl_1_2 在 fast_clifford/algebras/generated/
- [x] T008 [P] [US1] 刪除 p<q 且 p+q=4 的代數：cl_0_4, cl_1_3 在 fast_clifford/algebras/generated/
- [x] T009 [P] [US1] 刪除 p<q 且 p+q=5 的代數：cl_0_5, cl_1_4, cl_2_3 在 fast_clifford/algebras/generated/
- [x] T010 [P] [US1] 刪除 p<q 且 p+q=6 的代數：cl_0_6, cl_1_5, cl_2_4 在 fast_clifford/algebras/generated/
- [x] T011 [P] [US1] 刪除 p<q 且 p+q=7 的代數：cl_0_7, cl_1_6, cl_2_5, cl_3_4 在 fast_clifford/algebras/generated/
- [x] T012 [US1] 驗證保留的 20 個代數完整性：執行 `ls fast_clifford/algebras/generated/ | wc -l` 確認為 20
- [x] T013 [US1] 測量目錄大小確認低於 30MB：執行 `du -sh fast_clifford/algebras/generated/`
- [x] T014 [US1] 執行現有測試確認 VGA/CGA 功能正常：執行 `uv run pytest tests/test_vga.py tests/test_cga.py -v`

**檢查點**: 儲存空間應已減少 95%+，現有硬編碼代數功能正常

---

## Phase 4: 使用者故事 2 - 對稱代數存取 (優先級: P2)

**目標**: ~~實作 SymmetricClWrapper 支援 p < q 的代數透過映射到 Cl(q,p)~~

**⚠️ 修訂**: SymmetricClWrapper 方法經數學分析證明**不可行**：
- Cl(p,q) 和 Cl(q,p) 當 p ≠ q 時不是 Clifford 代數同構
- 度量簽章不同：Cl(0,4) 所有 e_i² = -1，Cl(4,0) 所有 e_i² = +1
- 無法透過索引重排轉換

**解決方案**: 直接生成所有 36 個代數（p+q < 8），儲存空間仍達 95.5% 減少

### 實作（已修訂）

- [x] T015-T022 [US2] **已撤銷** - SymmetricClWrapper 數學上不可行
- [x] T023 [US2] 建立對稱代數測試檔案 tests/test_symmetric.py
- [x] T024 [US2] 測試 Cl(1,3) 建立成功使用直接硬編碼代數
- [x] T025 [US2] 測試 Cl(0,4) 的 e1*e1 = -1（正確負簽章）在 tests/test_symmetric.py
- [x] T026 [US2] 測試所有 p < q 代數直接使用硬編碼版本（50 個測試）

**檢查點**: ✅ 所有 p < q 的代數透過直接硬編碼正確存取（36 個代數，27MB）

---

## Phase 5: 使用者故事 3 - 高維代數運算 (優先級: P2)

**目標**: 重構 BottPeriodicityAlgebra 支援正確的多重週期分解

**獨立測試**: 建立 Cl(10,0) 和 Cl(17,0)，驗證基底向量平方和矩陣大小正確

### 實作

- [x] T027 [US3] 實作 _decompose_signature() 多重週期分解邏輯在 fast_clifford/clifford/bott.py
- [x] T028 [US3] 新增 periods 屬性追蹤週期數在 fast_clifford/clifford/bott.py
- [x] T029 [US3] 更新 __init__() 使用新的分解邏輯在 fast_clifford/clifford/bott.py
- [x] T030 [US3] 確保基底代數使用 p >= q 的版本（若需要則套用對稱性）在 fast_clifford/clifford/bott.py （已有全部 36 個代數）
- [x] T031 [US3] 更新 tests/test_bott.py 測試 Cl(10,0) 分解為 Cl(2,0) + 16×16
- [x] T032 [US3] 新增測試 Cl(17,0) 分解為 Cl(1,0) + 256×256 在 tests/test_bott.py
- [x] T033 [US3] 新增測試 Cl(24,0) 分解為 Cl(0,0) + 4096×4096 在 tests/test_bott.py
- [x] T034 [US3] 測試混合簽章 Cl(10,2) 正確分解在 tests/test_bott.py

**檢查點**: ✅ 高維代數正確分解為基底代數 + 矩陣因子

---

## Phase 6: 使用者故事 4 - 加速的 Bott 運算 (優先級: P3)

**目標**: 使用張量化 einsum 運算取代 Python 迴圈，達成 10x+ 加速

**獨立測試**: Benchmark Cl(10,0) 幾何積，確認比 Python 迴圈實作快 10x+

### 實作

- [x] T035 [P] [US4] 實作 _compute_multiplication_table() 預計算乘法表在 fast_clifford/clifford/bott.py
- [x] T036 [P] [US4] 實作 _compute_outer_table() 預計算外積表在 fast_clifford/clifford/bott.py
- [x] T037 [P] [US4] 實作 _compute_reverse_signs() 預計算反轉符號在 fast_clifford/clifford/bott.py
- [x] T038 [US4] 重寫 geometric_product() 使用 einsum '...ilb, ...ljc, bcd -> ...ijd' 在 fast_clifford/clifford/bott.py
- [x] T039 [P] [US4] 重寫 outer() 使用張量化運算在 fast_clifford/clifford/bott.py
- [x] T040 [P] [US4] 重寫 inner() 使用張量化運算在 fast_clifford/clifford/bott.py
- [x] T041 [P] [US4] 重寫 reverse() 使用張量化運算在 fast_clifford/clifford/bott.py
- [x] T042 [P] [US4] 重寫 conjugate() 使用張量化運算在 fast_clifford/clifford/bott.py
- [x] T043 [US4] 確保所有運算支援批次維度在 fast_clifford/clifford/bott.py
- [x] T044 [US4] 建立效能基準測試 tests/benchmark/test_bott_benchmark.py
- [x] T045 [US4] 測試 Cl(10,0) 幾何積加速 >= 10x 在 tests/benchmark/test_bott_benchmark.py （0.052ms/call）
- [x] T046 [US4] 測試張量化結果與備份版本數值一致在 tests/benchmark/test_bott_benchmark.py
- [x] T047 [US4] 測試批次運算 (100, 1024) 形狀正確處理在 tests/benchmark/test_bott_benchmark.py

**檢查點**: ✅ Bott 運算比 Python 迴圈快 55x+，且結果數值正確

---

## Phase 7: 收尾與跨功能關注點

**目的**: 影響多個使用者故事的改進

- [x] T048 執行完整測試套件確認所有功能正常：275 passed
- [x] T049 [P] 驗證 ONNX 匯出無 Loop 節點：15 passed
- [x] T050 [P] 測試 Bott 運算的 float32/float16 精度一致性（憲法原則 V）在 tests/test_bott.py
- [x] T051 [P] 刪除備份檔案 fast_clifford/clifford/bott_backup.py
- [x] T052 更新 README.md 反映新的儲存空間和效能（36 algebras, 55x faster）
- [x] T053 驗證 quickstart.md 範例仍可執行

---

## 依賴與執行順序

### 階段依賴

- **設定 (Phase 1)**: 無依賴 - 可立即開始
- **基礎 (Phase 2)**: 依賴設定完成 - **阻塞**所有使用者故事
- **使用者故事 1 (Phase 3)**: 依賴基礎完成
- **使用者故事 2 (Phase 4)**: 依賴 Phase 3（因為需要刪除 p<q 代數後才能測試 SymmetricClWrapper）
- **使用者故事 3 (Phase 5)**: 依賴 Phase 4（可能需要 SymmetricClWrapper 處理基底代數）
- **使用者故事 4 (Phase 6)**: 依賴 Phase 5（需要正確的分解邏輯才能加速）
- **收尾 (Phase 7)**: 依賴所有使用者故事完成

### 使用者故事依賴

```
US1 (減少儲存) ─── 獨立，可在 Phase 2 後開始
      │
      ▼
US2 (對稱代數) ─── 依賴 US1（需刪除 p<q 代數後測試）
      │
      ▼
US3 (高維運算) ─── 依賴 US2（基底代數可能需要對稱性）
      │
      ▼
US4 (加速運算) ─── 依賴 US3（需要正確分解邏輯）
```

### 每個使用者故事內

- 核心實作優先
- 測試驗證次之
- 完成一個故事後再進入下一個

### 並行機會

- Phase 1 的備份和測試可並行
- US1 內的刪除任務 (T004-T011) 全部可並行
- US4 內的預計算表 (T035-T037) 可並行
- US4 內的運算重寫 (T039-T042) 可並行
- Phase 7 內 T049-T051 可並行

---

## 並行範例：使用者故事 1

```bash
# 同時刪除所有不需要的代數目錄：
Task: "刪除 p+q=8 的代數"
Task: "刪除 p+q=9 的代數"
Task: "刪除 p<q 且 p+q<=2 的代數"
Task: "刪除 p<q 且 p+q=3 的代數"
Task: "刪除 p<q 且 p+q=4 的代數"
Task: "刪除 p<q 且 p+q=5 的代數"
Task: "刪除 p<q 且 p+q=6 的代數"
Task: "刪除 p<q 且 p+q=7 的代數"
```

---

## 實作策略

### MVP 優先（僅使用者故事 1）

1. 完成 Phase 1: 設定
2. 完成 Phase 2: 基礎
3. 完成 Phase 3: 使用者故事 1
4. **停止並驗證**: 獨立測試 US1 - 儲存空間 < 30MB
5. 提交/展示（MVP 完成）

### 增量交付

1. 完成 設定 + 基礎 → 基礎就緒
2. 新增 US1 → 測試 → 提交（儲存減少 95%）
3. 新增 US2 → 測試 → 提交（對稱代數支援）
4. 新增 US3 → 測試 → 提交（多重週期支援）
5. 新增 US4 → 測試 → 提交（10x 加速）
6. 每個故事都增加價值，不破壞之前的故事

---

## 備註

- [P] 任務 = 不同檔案，無依賴
- [Story] 標籤將任務映射到特定使用者故事以便追蹤
- 每個使用者故事應可獨立完成和測試
- 每個任務或邏輯群組後提交
- 在任何檢查點停止以獨立驗證故事
- 避免：模糊任務、同一檔案衝突、破壞獨立性的跨故事依賴
