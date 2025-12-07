# 任務清單：CGA4D 與 CGA5D 支援

**輸入**: 設計文件來自 `/specs/003-cga4d-cga5d/`
**先決條件**: plan.md, spec.md, research.md, data-model.md, contracts/

**測試**: 包含測試任務（基於規格中的驗收場景）

**組織**: 任務按使用者故事分組，支援獨立實作與測試

## 格式: `[ID] [P?] [Story] 說明`

- **[P]**: 可平行執行（不同檔案，無依賴）
- **[Story]**: 任務所屬的使用者故事（例如 US1, US2, US3）
- 說明中包含確切的檔案路徑

## 路徑慣例

- **專案根目錄**: `fast_clifford/`, `scripts/`
- **測試目錄**: `fast_clifford/tests/`

---

## Phase 1: 基礎設施擴展

**目的**: 擴展代碼生成器支援 CGA4D 和 CGA5D

- [x] T001 擴展 cga_factory.py 的維度範圍檢查，支援 euclidean_dim 4 和 5，檔案路徑：`fast_clifford/codegen/cga_factory.py`
- [x] T002 在 cga_factory.py 新增 CGA4D 和 CGA5D 的預定義常數（BLADE_COUNT, SIGNATURE），檔案路徑：`fast_clifford/codegen/cga_factory.py`
- [x] T003 [P] 驗證 cga_factory.py 對 CGA4D (euclidean_dim=4) 能正確計算 grade 索引和 motor 索引
- [x] T004 [P] 驗證 cga_factory.py 對 CGA5D (euclidean_dim=5) 能正確計算 grade 索引和 motor 索引

**檢查點**: ✅ 代碼生成器基礎設施已擴展，可生成 CGA4D/CGA5D 代數

---

## Phase 2: 使用者故事 1 - 4D 幾何變換 (優先級: P1) 🎯 MVP

**目標**: 實作 CGA4D Cl(5,1) 代數，支援 64 blades、6 分量 UPGC Point、31 分量 Motor

**獨立測試**: 透過將 4D 點編碼為 UPGC 表示、套用三明治積變換、解碼回 4D 座標來驗證。對比 clifford 函式庫結果。

### CGA4D 代數定義

- [x] T005 [US1] 建立 CGA4D 代數定義模組，檔案路徑：`fast_clifford/algebras/cga4d/algebra.py`
  - 使用 clifford.conformalize(Cl(4)) 建立代數
  - 定義 BLADE_COUNT = 64, EUCLIDEAN_DIM = 4
  - 計算並定義 GRADE_0_INDICES 到 GRADE_6_INDICES
  - 定義 UPGC_POINT_MASK = GRADE_1_INDICES (6 分量)
  - 定義 MOTOR_SPARSE_INDICES (31 分量：Grade 0 + 2 + 4)
  - 定義 MOTOR_REVERSE_SIGNS (31 個符號)
  - 定義 get_product_table() 函式提取幾何積規則

### CGA4D 生成腳本

- [x] T006 [US1] 建立 CGA4D 代碼生成腳本，檔案路徑：`scripts/generate_cga4d.py`
  - 參考 generate_cga3d.py 結構
  - 使用 algebra.py 的代數定義
  - 輸出 functional.py 到 fast_clifford/algebras/cga4d/

### CGA4D 硬編碼函式

- [x] T007 [US1] 執行生成腳本產生 CGA4D 硬編碼函式，檔案路徑：`fast_clifford/algebras/cga4d/functional.py`
  - 生成 geometric_product_full (64×64 展開)
  - 生成 sandwich_product_sparse (Motor[31] × Point[6] × Motor_rev[31])
  - 生成 upgc_encode 和 upgc_decode 函式
  - 生成 motor_reverse 函式
  - 所有函式使用 @torch.jit.script 裝飾

### CGA4D PyTorch 層

- [x] T008 [US1] 實作 CGA4D PyTorch 層封裝，檔案路徑：`fast_clifford/algebras/cga4d/layers.py`
  - 實作 CGA4DCareLayer(nn.Module)：fp16→fp32→fp16 精度轉換
  - 實作 UPGC4DEncoder(nn.Module)：4D 座標 → 6 分量 UPGC
  - 實作 UPGC4DDecoder(nn.Module)：6 分量 UPGC → 4D 座標
  - 實作 CGA4DTransformPipeline(nn.Module)：編碼 → 變換 → 解碼

### CGA4D 模組匯出

- [x] T009 [US1] 建立 CGA4D 模組初始化檔案，檔案路徑：`fast_clifford/algebras/cga4d/__init__.py`
  - 匯出 CGA4DCareLayer, UPGC4DEncoder, UPGC4DDecoder, CGA4DTransformPipeline
  - 匯出 functional 模組中的主要函式

### CGA4D 數值測試

- [x] T010 [P] [US1] 建立 CGA4D 測試目錄，檔案路徑：`fast_clifford/tests/cga4d/__init__.py`
- [x] T011 [US1] 實作 CGA4D 數值正確性測試，檔案路徑：`fast_clifford/tests/cga4d/test_numerical.py`
  - 測試幾何積正確性（對比 clifford）
  - 測試 Null basis 性質（eo²=0, einf²=0, eo·einf=-1）
  - 測試結合律
  - 測試 Reverse 運算
  - 測試稀疏三明治積正確性
  - 測試 4D 旋轉變換
  - 測試 4D 平移變換
  - 測試批次處理

### CGA4D ONNX 測試

- [x] T012 [US1] 實作 CGA4D ONNX 匯出測試，檔案路徑：`fast_clifford/tests/cga4d/test_onnx.py`
  - 測試 CGA4DCareLayer ONNX 匯出成功
  - 測試 ONNX 計算圖無 Loop 節點
  - 測試 UPGC4DEncoder/Decoder ONNX 匯出
  - 測試 PyTorch 與 ONNX Runtime 輸出一致

### CGA4D 驗證

- [x] T013 [US1] 執行 CGA4D 測試並驗證全部通過，指令：`uv run pytest fast_clifford/tests/cga4d/ -v`

**檢查點**: ✅ CGA4D 完整可用 - 4D 幾何變換功能完成，可獨立測試與部署

---

## Phase 3: 使用者故事 2 - 5D 幾何變換 (優先級: P2)

**目標**: 實作 CGA5D Cl(6,1) 代數，支援 128 blades、7 分量 UPGC Point、64 分量 Motor

**獨立測試**: 透過將 5D 點編碼為 UPGC 表示、套用三明治積變換、解碼回 5D 座標來驗證。對比 clifford 函式庫結果。

### CGA5D 代數定義

- [x] T014 [US2] 建立 CGA5D 代數定義模組，檔案路徑：`fast_clifford/algebras/cga5d/algebra.py`
  - 使用 clifford.conformalize(Cl(5)) 建立代數
  - 定義 BLADE_COUNT = 128, EUCLIDEAN_DIM = 5
  - 計算並定義 GRADE_0_INDICES 到 GRADE_7_INDICES
  - 定義 UPGC_POINT_MASK = GRADE_1_INDICES (7 分量)
  - 定義 MOTOR_SPARSE_INDICES (64 分量：Grade 0 + 2 + 4 + 6)
  - 定義 MOTOR_REVERSE_SIGNS (64 個符號)
  - 定義 get_product_table() 函式提取幾何積規則

### CGA5D 生成腳本

- [x] T015 [US2] 建立 CGA5D 代碼生成腳本，檔案路徑：`scripts/generate_cga5d.py`
  - 參考 generate_cga4d.py 結構
  - 使用 algebra.py 的代數定義
  - 輸出 functional.py 到 fast_clifford/algebras/cga5d/

### CGA5D 硬編碼函式

- [x] T016 [US2] 執行生成腳本產生 CGA5D 硬編碼函式，檔案路徑：`fast_clifford/algebras/cga5d/functional.py`
  - 生成 geometric_product_full (128×128 展開)
  - 生成 sandwich_product_sparse (Motor[64] × Point[7] × Motor_rev[64])
  - 生成 upgc_encode 和 upgc_decode 函式
  - 生成 motor_reverse 函式
  - 所有函式使用 @torch.jit.script 裝飾

### CGA5D PyTorch 層

- [x] T017 [US2] 實作 CGA5D PyTorch 層封裝，檔案路徑：`fast_clifford/algebras/cga5d/layers.py`
  - 實作 CGA5DCareLayer(nn.Module)：fp16→fp32→fp16 精度轉換
  - 實作 UPGC5DEncoder(nn.Module)：5D 座標 → 7 分量 UPGC
  - 實作 UPGC5DDecoder(nn.Module)：7 分量 UPGC → 5D 座標
  - 實作 CGA5DTransformPipeline(nn.Module)：編碼 → 變換 → 解碼

### CGA5D 模組匯出

- [x] T018 [US2] 建立 CGA5D 模組初始化檔案，檔案路徑：`fast_clifford/algebras/cga5d/__init__.py`
  - 匯出 CGA5DCareLayer, UPGC5DEncoder, UPGC5DDecoder, CGA5DTransformPipeline
  - 匯出 functional 模組中的主要函式

### CGA5D 數值測試

- [x] T019 [P] [US2] 建立 CGA5D 測試目錄，檔案路徑：`fast_clifford/tests/cga5d/__init__.py`
- [x] T020 [US2] 實作 CGA5D 數值正確性測試，檔案路徑：`fast_clifford/tests/cga5d/test_numerical.py`
  - 測試幾何積正確性（對比 clifford）
  - 測試 Null basis 性質
  - 測試結合律
  - 測試 Reverse 運算
  - 測試稀疏三明治積正確性
  - 測試 5D 旋轉變換
  - 測試 5D 平移變換
  - 測試批次處理

### CGA5D ONNX 測試

- [x] T021 [US2] 實作 CGA5D ONNX 匯出測試，檔案路徑：`fast_clifford/tests/cga5d/test_onnx.py`
  - 測試 CGA5DCareLayer ONNX 匯出成功
  - 測試 ONNX 計算圖無 Loop 節點
  - 測試 UPGC5DEncoder/Decoder ONNX 匯出
  - 測試 PyTorch 與 ONNX Runtime 輸出一致

### CGA5D 驗證

- [x] T022 [US2] 執行 CGA5D 測試並驗證全部通過，指令：`uv run pytest fast_clifford/tests/cga5d/ -v`

**檢查點**: ✅ CGA5D 完整可用 - 5D 幾何變換功能完成，可獨立測試與部署

---

## Phase 4: 使用者故事 3 - PyTorch 訓練整合 (優先級: P1)

**目標**: 確保 CGA4D 和 CGA5D 層能無縫整合到 PyTorch 訓練流程中

**獨立測試**: 建立使用 CGA 層的簡單神經網路，執行前向/反向傳播，驗證梯度流動

### 梯度測試

- [x] T023 [US3] 實作 CGA4D 梯度測試，在 `fast_clifford/tests/cga4d/test_numerical.py` 新增
  - 測試 CGA4DCareLayer 梯度傳播正確性
  - 測試 torch.autograd.gradcheck 通過
  - 測試在簡單神經網路中的梯度流動

- [x] T024 [US3] 實作 CGA5D 梯度測試，在 `fast_clifford/tests/cga5d/test_numerical.py` 新增
  - 測試 CGA5DCareLayer 梯度傳播正確性
  - 測試 torch.autograd.gradcheck 通過
  - 測試在簡單神經網路中的梯度流動

### 混合精度測試

- [x] T025 [US3] 實作 CGA4D 混合精度測試，在 `fast_clifford/tests/cga4d/test_numerical.py` 新增
  - 測試 fp16 輸入正確處理
  - 測試輸出 dtype 與輸入一致
  - 測試 fp32 計算的數值穩定性

- [x] T026 [US3] 實作 CGA5D 混合精度測試，在 `fast_clifford/tests/cga5d/test_numerical.py` 新增
  - 測試 fp16 輸入正確處理
  - 測試輸出 dtype 與輸入一致
  - 測試 fp32 計算的數值穩定性

### 批次處理測試

- [x] T027 [US3] 實作 CGA4D 批次處理測試，在 `fast_clifford/tests/cga4d/test_numerical.py` 新增
  - 測試批次大小 1, 16, 64, 256, 1024
  - 測試批次維度正確廣播
  - 驗證各批次大小的輸出一致性

- [x] T028 [US3] 實作 CGA5D 批次處理測試，在 `fast_clifford/tests/cga5d/test_numerical.py` 新增
  - 測試批次大小 1, 16, 64, 256, 1024
  - 測試批次維度正確廣播
  - 驗證各批次大小的輸出一致性

### 裝置測試

- [x] T029 [US3] 實作跨裝置測試，在 `fast_clifford/tests/cga4d/test_numerical.py` 和 `cga5d/test_numerical.py` 新增
  - 測試 CPU 裝置
  - 測試 MPS 裝置（若可用）
  - 測試 CUDA 裝置（若可用）
  - 驗證各裝置輸出一致

**檢查點**: ✅ PyTorch 訓練整合完成 - CGA4D/CGA5D 可用於模型訓練

---

## Phase 5: 整合與收尾

**目的**: 模組整合、效能測試、文件更新

### 模組整合

- [x] T030 更新 algebras 模組初始化檔案匯出 cga4d 和 cga5d，檔案路徑：`fast_clifford/algebras/__init__.py`
- [x] T031 更新主模組初始化檔案匯出 cga4d 和 cga5d，檔案路徑：`fast_clifford/__init__.py`

### JIT 優化驗證

- [x] T032 驗證 CGA4D functional.py 中所有函式套用 @torch.jit.script
- [x] T033 驗證 CGA5D functional.py 中所有函式套用 @torch.jit.script

### 效能測試

- [ ] T034 [P] 執行 CGA4D 效能基準測試，驗證吞吐量 > 500,000 點/秒 (CPU)
- [ ] T035 [P] 執行 CGA5D 效能基準測試，驗證吞吐量 > 100,000 點/秒 (CPU)

### 文件更新

- [ ] T036 [P] 更新效能基準測試報告，檔案路徑：`docs/benchmark.md`
  - 新增 CGA4D 效能數據
  - 新增 CGA5D 效能數據
  - 新增與 CGA1D/2D/3D 的對比表

### 全部測試驗證

- [x] T037 執行所有測試並驗證全部通過，指令：`uv run pytest -v`
  - 結果：204 passed, 3 skipped (CUDA tests)

### Git 提交

- [ ] T038 提交所有變更至 Git，訊息格式遵循憲法原則 VII

**檢查點**: ⏳ CGA4D 和 CGA5D 實作完成，測試通過，待效能測試與提交

---

## 依賴與執行順序

### Phase 依賴

- **Phase 1 (基礎設施)**: 無依賴 - 可立即開始
- **Phase 2 (US1 - CGA4D)**: 依賴 Phase 1 完成
- **Phase 3 (US2 - CGA5D)**: 依賴 Phase 1 完成（可與 Phase 2 平行）
- **Phase 4 (US3 - 整合)**: 依賴 Phase 2 和 Phase 3 完成
- **Phase 5 (收尾)**: 依賴 Phase 4 完成

### 使用者故事依賴

- **US1 (CGA4D)**: Phase 1 完成後可開始 - 不依賴其他故事
- **US2 (CGA5D)**: Phase 1 完成後可開始 - 不依賴其他故事（可與 US1 平行）
- **US3 (整合)**: 依賴 US1 和 US2 完成

### 各故事內部順序

- 代數定義 → 生成腳本 → 執行生成 → 層封裝 → 模組匯出 → 測試

### 平行機會

- T003, T004 可平行執行
- T010, T019 可平行執行
- T034, T035, T036 可平行執行
- US1 (CGA4D) 和 US2 (CGA5D) 可由不同開發者平行執行

---

## 平行執行範例

### 基礎設施驗證（Phase 1 後）

```bash
# 平行驗證 CGA4D 和 CGA5D 的代數規則計算
Task: "驗證 cga_factory.py 對 CGA4D 能正確計算 grade 索引和 motor 索引"
Task: "驗證 cga_factory.py 對 CGA5D 能正確計算 grade 索引和 motor 索引"
```

### CGA4D 和 CGA5D 平行開發

```bash
# 由不同開發者同時進行
Developer A: Phase 2 (CGA4D) - T005 到 T013
Developer B: Phase 3 (CGA5D) - T014 到 T022
```

### 效能測試與文件更新（Phase 5）

```bash
# 平行執行效能測試和文件更新
Task: "執行 CGA4D 效能基準測試"
Task: "執行 CGA5D 效能基準測試"
Task: "更新效能基準測試報告"
```

---

## 實作策略

### MVP 優先（僅 US1 - CGA4D）

1. 完成 Phase 1: 基礎設施擴展
2. 完成 Phase 2: CGA4D 實作
3. **停止並驗證**: 獨立測試 CGA4D
4. 若需要可先部署 CGA4D

### 增量交付

1. 完成 Phase 1 → 基礎設施就緒
2. 完成 Phase 2 (CGA4D) → 獨立測試 → MVP 完成！
3. 完成 Phase 3 (CGA5D) → 獨立測試 → 功能擴展
4. 完成 Phase 4 (整合) → PyTorch 訓練就緒
5. 完成 Phase 5 (收尾) → 效能驗證、文件更新

### 平行團隊策略

有多個開發者時：

1. 團隊共同完成 Phase 1
2. Phase 1 完成後：
   - 開發者 A: Phase 2 (CGA4D)
   - 開發者 B: Phase 3 (CGA5D)
3. 兩者完成後共同完成 Phase 4 和 Phase 5

---

## 進度摘要

| Phase | 任務數 | 狀態 |
|-------|-------|------|
| Phase 1: 基礎設施 | 4 | ✅ 完成 |
| Phase 2: US1 CGA4D | 9 | ✅ 完成 |
| Phase 3: US2 CGA5D | 9 | ✅ 完成 |
| Phase 4: US3 整合 | 7 | ✅ 完成 |
| Phase 5: 收尾 | 9 | ⏳ 進行中 (6/9) |
| **總計** | **38** | **35/38 完成** |

---

## 注意事項

- [P] 任務 = 不同檔案，無依賴，可平行執行
- [Story] 標籤標示任務所屬的使用者故事
- 每個使用者故事應可獨立完成與測試
- 每個任務完成後提交 Git（遵循憲法原則 VII）
- 在任何檢查點停止以獨立驗證故事
- 避免：模糊的任務描述、同檔案衝突、破壞獨立性的跨故事依賴
