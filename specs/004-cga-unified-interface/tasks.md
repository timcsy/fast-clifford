# 任務清單：CGA(n) 統一介面

**輸入**：設計文件來自 `/specs/004-cga-unified-interface/`
**前置條件**：plan.md（必要）、spec.md（必要）、research.md、data-model.md、contracts/

## 格式說明：`[ID] [P?] [Story] 說明`

- **[P]**：可平行執行（不同檔案，無相依性）
- **[Story]**：所屬使用者故事（US1、US2、US3、US4）
- 說明中包含確切檔案路徑

## 路徑慣例

- **原始碼**：`fast_clifford/`
- **測試**：`fast_clifford/tests/`
- **規格**：`specs/004-cga-unified-interface/`

---

## Phase 1：設定（共享基礎設施）

**目的**：建立 cga/ 模組結構和統一介面核心

- [ ] T001 建立 `fast_clifford/cga/` 目錄結構
- [ ] T002 [P] 建立 `fast_clifford/cga/__init__.py` 空模組
- [ ] T003 [P] 建立 `fast_clifford/cga/base.py` 空模組
- [ ] T004 [P] 建立 `fast_clifford/cga/registry.py` 空模組
- [ ] T005 [P] 建立 `fast_clifford/cga/runtime.py` 空模組

---

## Phase 2：基礎架構（阻擋式前置條件）

**目的**：完成所有使用者故事都依賴的核心基礎設施

**⚠️ 關鍵**：在本階段完成前，無法開始任何使用者故事工作

### CGAAlgebraBase 抽象基底類別

- [ ] T006 實作 `CGAAlgebraBase` 抽象類別於 `fast_clifford/cga/base.py`
  - 定義屬性：euclidean_dim、blade_count、point_count、motor_count、signature、clifford_notation
  - 定義抽象方法：upgc_encode、upgc_decode、geometric_product_full、sandwich_product_sparse
  - 定義抽象方法：reverse_full、reverse_motor
  - 定義工廠方法：get_care_layer、get_encoder、get_decoder、get_transform_pipeline

### HardcodedCGAWrapper

- [ ] T007 實作 `HardcodedCGAWrapper` 類別於 `fast_clifford/cga/registry.py`
  - 包裝現有 cga1d-cga5d 模組
  - 實作 CGAAlgebraBase 介面
  - 將模組函式映射至抽象介面方法

### 測試基礎架構

- [ ] T008 [P] 建立 `fast_clifford/tests/test_cga_interface.py` 測試檔案框架
- [ ] T009 [P] 建立 `fast_clifford/tests/cga0d/` 目錄結構

**檢查點**：基礎架構就緒 — 使用者故事實作可開始平行進行

---

## Phase 3：使用者故事 1 - 統一 CGA 維度選擇（優先順序：P1）🎯 MVP

**目標**：透過 `CGA(n)` 介面存取任意 CGA 維度（0-5 硬編碼，6+ 運行時）

**獨立測試**：對 n=0,1,2,3,4,5,6 呼叫 `CGA(n)` 並驗證每個都返回可執行三明治積的代數物件

### 測試 US1

- [ ] T010 [P] [US1] 測試 `CGA(n)` 對 n=1-5 返回正確類型於 `fast_clifford/tests/test_cga_interface.py`
- [ ] T011 [P] [US1] 測試 `CGA(n)` 路由邏輯正確（硬編碼 vs 運行時）

### 實作 US1

- [ ] T012 [US1] 實作 `CGA(n)` 工廠函式於 `fast_clifford/cga/__init__.py`
  - n=0-5 返回 HardcodedCGAWrapper
  - n≥6 返回 RuntimeCGAAlgebra
  - n<0 拋出 ValueError
  - n≥15 發出記憶體警告

- [ ] T013 [US1] 更新 `fast_clifford/__init__.py` 匯出 CGA

- [ ] T014 [US1] 驗證 US1 驗收情境
  - 情境 1：CGA(3) 返回具備所有標準操作的代數物件
  - 情境 2：CGA(0) 返回快速硬編碼算法（需 US2 完成）
  - 情境 3：CGA(6) 返回運行時算法（需 US4 完成）
  - 情境 4：CGA(10) 返回運行時算法

**檢查點**：CGA(n) 對 n=1-5 可正常運作

---

## Phase 4：使用者故事 2 - CGA0D 快速算法（優先順序：P2）

**目標**：完成 0-5D 快速算法覆蓋，提供 CGA0D (Cl(1,1)) 硬編碼實作

**獨立測試**：建立 CGA0D 馬達和點，執行三明治積，對照 clifford 函式庫驗證

### CGA0D 模組結構

- [ ] T015 [P] [US2] 建立 `fast_clifford/algebras/cga0d/__init__.py`
- [ ] T016 [P] [US2] 建立 `fast_clifford/algebras/cga0d/algebra.py` 代數定義
  - EUCLIDEAN_DIM = 0
  - BLADE_COUNT = 4
  - SIGNATURE = (1, -1)
  - GRADE_0_INDICES、GRADE_1_INDICES、GRADE_2_INDICES
  - UPGC_POINT_MASK = (1, 2)
  - MOTOR_MASK = (0, 3)
  - REVERSE_SIGNS、MOTOR_REVERSE_SIGNS

### CGA0D 功能函式

- [ ] T017 [US2] 實作 `geometric_product_full()` 於 `fast_clifford/algebras/cga0d/functional.py`
  - 4×4 完全展開幾何積
  - 無迴圈，ONNX 相容

- [ ] T018 [US2] 實作 `reverse_full()` 和 `reverse_motor()` 於同檔案

- [ ] T019 [US2] 實作 `upgc_encode()` 和 `upgc_decode()` 於同檔案
  - 0D 沒有歐幾里得分量
  - encode 返回原點 n_o = [-0.5, 0.5]
  - decode 返回空張量 shape (..., 0)

- [ ] T020 [US2] 實作 `sandwich_product_sparse()` 於同檔案
  - Motor [2] × Point [2] → Point [2]
  - 利用稀疏性優化

### CGA0D 層封裝

- [ ] T021 [US2] 實作 `CGA0DCareLayer` 於 `fast_clifford/algebras/cga0d/layers.py`
- [ ] T022 [US2] 實作 `UPGC0DEncoder` 和 `UPGC0DDecoder` 於同檔案
- [ ] T023 [US2] 實作 `CGA0DTransformPipeline` 於同檔案

### CGA0D 測試

- [ ] T024 [P] [US2] 數值測試於 `fast_clifford/tests/cga0d/test_numerical.py`
  - 幾何積正確性（對照 clifford 函式庫）
  - 三明治積正確性
  - 編碼/解碼往返

- [ ] T025 [P] [US2] ONNX 測試於 `fast_clifford/tests/cga0d/test_onnx.py`
  - 匯出無 Loop 節點
  - 數值一致性

### CGA0D 整合

- [ ] T026 [US2] 更新 `fast_clifford/algebras/__init__.py` 匯出 cga0d
- [ ] T027 [US2] 更新 HardcodedCGAWrapper 支援 cga0d 模組

**檢查點**：CGA(0) 返回完整功能的 CGA0D 代數

---

## Phase 5：使用者故事 3 - Clifford 簽名表示法（優先順序：P3）

**目標**：使用 `Cl(p, q, r)` 標準表示法建立代數

**獨立測試**：呼叫 `Cl(4, 1, 0)` 驗證返回等同於 CGA3D 的代數

### 測試 US3

- [ ] T028 [P] [US3] 測試 `Cl(p, q, r)` 於 `fast_clifford/tests/test_cga_interface.py`
  - Cl(4, 1, 0) == CGA(3)
  - Cl(4, 1) == CGA(3)（r=0 預設）
  - Cl(1, 1, 0) == CGA(0)
  - Cl(7, 1, 0) == CGA(6)

### 實作 US3

- [ ] T029 [US3] 實作 `Cl(p, q, r=0)` 工廠函式於 `fast_clifford/cga/__init__.py`
  - 識別 CGA 簽名（q==1, r==0）並路由至 CGA(p-1)
  - 非 CGA 簽名發出警告但仍建立代數
  - 支援退化維度 r>0

- [ ] T030 [US3] 實作 `RuntimeCliffordAlgebra` 於 `fast_clifford/cga/runtime.py`（非 CGA 簽名用）

- [ ] T031 [US3] 更新 `fast_clifford/__init__.py` 匯出 Cl

- [ ] T032 [US3] 驗證 US3 驗收情境
  - 情境 1：Cl(4, 1, 0) 返回 CGA3D
  - 情境 2：Cl(4, 1) 返回 CGA3D
  - 情境 3：Cl(3, 0, 0) 發出警告並建立代數
  - 情境 4：Cl(3, 0, 1) 返回帶退化維度的代數

**檢查點**：Cl(p, q, r) 可正常運作

---

## Phase 6：使用者故事 4 - 高維度運行時 CGA（優先順序：P4）

**目標**：支援 CGA6D+ 任意維度，PyTorch 可微分，ONNX 無迴圈匯出

**獨立測試**：建立 CGA6D，驗證基本操作、梯度反向傳播、ONNX 匯出

### RuntimeCGAAlgebra 實作

- [ ] T033 [US4] 實作 `RuntimeCGAAlgebra` 類別於 `fast_clifford/cga/runtime.py`
  - 繼承 CGAAlgebraBase 和 nn.Module
  - 延遲初始化（首次呼叫時計算 Cayley 表）
  - 使用 register_buffer 註冊索引張量

- [ ] T034 [US4] 實作運行時 `_ensure_initialized()` 方法
  - 計算非零乘積的 left_idx、right_idx、result_idx、signs
  - 計算 point_mask、motor_mask、reverse_signs

- [ ] T035 [US4] 實作運行時 `geometric_product_full()`
  - 使用 index_select + scatter_add 張量化批次操作
  - 無 Python 迴圈，ONNX 相容

- [ ] T036 [US4] 實作運行時 `sandwich_product_sparse()`
  - Motor × Point → embedded → gp → gp → extract → Point
  - 無迴圈，ONNX 相容

- [ ] T037 [P] [US4] 實作運行時 `upgc_encode()` 和 `upgc_decode()`
- [ ] T038 [P] [US4] 實作運行時 `reverse_full()` 和 `reverse_motor()`

### 運行時 Layers

- [ ] T039 [P] [US4] 實作 `RuntimeCGACareLayer` 於 `fast_clifford/cga/runtime.py`
- [ ] T040 [P] [US4] 實作 `RuntimeUPGCEncoder` 和 `RuntimeUPGCDecoder`
- [ ] T041 [P] [US4] 實作 `RuntimeCGATransformPipeline`

### 運行時測試（輕量）

- [ ] T042 [US4] 數值測試於 `fast_clifford/tests/test_runtime_cga.py`
  - 僅 CGA6D 作為代表
  - 小 batch size（≤16）
  - 幾何積對照 clifford 驗證
  - 三明治積正確性

- [ ] T043 [US4] 梯度測試於同檔案
  - loss.backward() 梯度傳播

- [ ] T044 [US4] ONNX 測試於同檔案
  - 匯出無 Loop 節點
  - 數值一致性

- [ ] T045 [US4] 驗證 US4 驗收情境
  - 情境 1：CGA(6) 具有 256 個 blades
  - 情境 2：三明治積數值正確
  - 情境 3：梯度正確傳播
  - 情境 4：ONNX 無 Loop 節點

**檢查點**：CGA(6+) 運行時算法完全可用

---

## Phase 7：收尾與跨切面關注點

**目的**：影響多個使用者故事的改進

- [ ] T046 [P] 更新 `README.md` 新增統一介面使用說明
- [ ] T047 [P] 驗證 `quickstart.md` 所有範例可執行
- [ ] T048 執行完整測試套件，確保無退化
- [ ] T049 程式碼清理與重構
- [ ] T050 [P] 更新 `CLAUDE.md` 如需要

---

## 相依性與執行順序

### Phase 相依性

- **Phase 1 設定**：無相依性 — 可立即開始
- **Phase 2 基礎架構**：依賴 Phase 1 — **阻擋所有使用者故事**
- **Phase 3-6 使用者故事**：全部依賴 Phase 2 完成
  - US1 可先開始（僅需 Phase 2）
  - US2 可與 US1 平行（但最終 US1 需要 US2）
  - US3 可在 Phase 2 後開始
  - US4 可在 Phase 2 後開始
- **Phase 7 收尾**：依賴所有必要使用者故事完成

### 使用者故事相依性

```
Phase 2（基礎架構）
    │
    ├──► US1（CGA(n) 工廠）──► US2（CGA0D）──► US1 完整驗證
    │                        └──► US4（運行時）──► US1 完整驗證
    │
    ├──► US3（Cl 表示法）────► US4（RuntimeCliffordAlgebra）
    │
    └──► US4（運行時 CGA）
         │
         ▼
    Phase 7（收尾）
```

### 平行機會

- Phase 1 所有 [P] 任務可平行
- Phase 2 的 T008、T009 可平行
- US2 的 T015、T016、T024、T025 可平行
- US3 的 T028 與實作可平行
- US4 的 T037、T038、T039、T040、T041 可平行
- Phase 7 的 T046、T047、T050 可平行

---

## 實作策略

### MVP 優先（僅 US1 核心功能）

1. 完成 Phase 1：設定
2. 完成 Phase 2：基礎架構
3. 部分完成 Phase 3：US1（對 n=1-5 可用）
4. **停止並驗證**：CGA(n) 對現有模組可用
5. 部署/展示（可用 MVP）

### 增量交付

1. 完成設定 + 基礎架構 → 基礎就緒
2. 新增 US2（CGA0D）→ CGA(0) 可用
3. 新增 US4（運行時）→ CGA(6+) 可用 → US1 完全驗證
4. 新增 US3（Cl 表示法）→ 完整功能
5. 每個故事都增加價值且不破壞先前功能

---

## 備註

- [P] 任務 = 不同檔案，無相依性
- [Story] 標籤將任務映射至特定使用者故事以便追蹤
- 每個使用者故事應可獨立完成和測試
- 在實作前驗證測試失敗
- 每個任務或邏輯群組後提交
- 在任何檢查點停止以獨立驗證故事
- 避免：模糊任務、同檔案衝突、破壞獨立性的跨故事相依性
