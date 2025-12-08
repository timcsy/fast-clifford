# 實作計畫：CGA(n) 統一介面

**分支**：`004-cga-unified-interface` | **日期**：2025-12-08 | **規格**：[spec.md](./spec.md)
**輸入**：功能規格書 `/specs/004-cga-unified-interface/spec.md`

## 摘要

為 fast-clifford 函式庫新增統一的 CGA 代數介面，包含：
1. **CGA(n)** 工廠函式 - 按歐幾里得維度選擇代數
2. **Cl(p, q, r)** 工廠函式 - 按 Clifford 簽名建立代數
3. **CGA0D** 硬編碼快速算法 - 完成 0-5D 快速算法覆蓋
4. **運行時算法** - 支援 CGA6D+ 任意維度，PyTorch 可微分，ONNX 無迴圈匯出

技術方案：使用抽象基底類別 `CGAAlgebraBase` 定義統一介面，透過 `HardcodedCGAWrapper` 包裝現有模組，運行時算法在首次使用時動態展開操作。

## 技術背景

**語言/版本**：Python 3.10+
**主要依賴**：PyTorch 2.0+, NumPy, clifford（測試用）
**儲存**：N/A（純計算函式庫）
**測試**：pytest
**目標平台**：跨平台（CPU、Apple MPS、NVIDIA CUDA）
**專案類型**：單一專案（Python 套件）
**效能目標**：硬編碼算法維持現有效能基準，運行時算法以正確性為優先
**限制**：
- ONNX 匯出必須無 Loop 節點
- 必須支援 PyTorch 自動微分
- 禁止平台特定擴充（純 PyTorch）
**規模**：CGA0D-CGA5D 硬編碼 + CGA6D+ 運行時

## 憲法檢查

*閘門：必須在 Phase 0 研究前通過。Phase 1 設計後再次檢查。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ 通過 | 所有代數操作（含運行時）必須 ONNX 可匯出 |
| II. 平台相容性 | ✅ 通過 | 純 PyTorch 實作，禁止平台特定 API |
| III. 無迴圈前向傳播 | ✅ 通過 | 運行時算法動態展開，不使用迴圈 |
| IV. 硬編碼代數展開 | ✅ 通過 | CGA0D 使用硬編碼展開；運行時算法動態生成展開程式碼 |
| V. 數值精度安全 | ✅ 通過 | 所有層強制 float32 |
| VI. 文件語言規範 | ✅ 通過 | 規格文件使用繁體中文 |
| VII. 增量提交原則 | ✅ 通過 | 按任務分段提交 |

**禁止技術檢查**：
- Triton ❌ 不使用
- CUDA C++ 擴充 ❌ 不使用
- Taichi ❌ 不使用
- 帶迴圈的 torch.jit.script ❌ 不使用

## 專案結構

### 文件（本功能）

```text
specs/004-cga-unified-interface/
├── plan.md              # 本檔案
├── research.md          # Phase 0 輸出
├── data-model.md        # Phase 1 輸出
├── quickstart.md        # Phase 1 輸出
├── contracts/           # Phase 1 輸出
│   ├── cga_base.pyi     # CGAAlgebraBase 型別定義
│   ├── cga0d.pyi        # CGA0D 型別定義
│   └── runtime.pyi      # 運行時代數型別定義
└── tasks.md             # Phase 2 輸出（/speckit.tasks）
```

### 原始碼（專案根目錄）

```text
fast_clifford/
├── __init__.py                    # 更新：匯出 CGA, Cl
├── cga/                           # 新增：統一介面
│   ├── __init__.py                # CGA(n), Cl(p,q,r) 工廠
│   ├── base.py                    # CGAAlgebraBase 抽象類別
│   ├── registry.py                # HardcodedCGAWrapper
│   └── runtime.py                 # RuntimeCGAAlgebra
├── algebras/
│   ├── __init__.py                # 更新：匯出 cga0d
│   ├── cga0d/                     # 新增：CGA0D 快速算法
│   │   ├── __init__.py
│   │   ├── algebra.py
│   │   ├── functional.py
│   │   └── layers.py
│   ├── cga1d/                     # 現有
│   ├── cga2d/                     # 現有
│   ├── cga3d/                     # 現有
│   ├── cga4d/                     # 現有
│   └── cga5d/                     # 現有
└── codegen/                       # 現有：程式碼生成器
    ├── cga_factory.py
    ├── generate.py
    └── sparse_analysis.py

fast_clifford/tests/
├── cga0d/                         # 新增：CGA0D 測試
│   ├── __init__.py
│   ├── test_numerical.py
│   └── test_onnx.py
├── test_cga_interface.py          # 新增：統一介面測試
└── test_runtime_cga.py            # 新增：運行時算法測試
```

**結構決策**：採用單一專案結構，新增 `fast_clifford/cga/` 模組作為統一介面層，包裝現有 `fast_clifford/algebras/` 模組。

## 複雜度追蹤

> 無憲法違規需要說明

## 實作流水線

### Phase 0：大綱與研究

**目標**：研究運行時動態展開的可行性，確認 ONNX 無迴圈匯出方案

**研究任務**：
1. 運行時動態展開技術 - 如何在不使用迴圈的情況下動態生成張量操作
2. PyTorch 張量索引的 ONNX 相容性 - 確認動態索引不產生 Loop 節點
3. 延遲初始化模式 - 首次使用時生成展開程式碼的最佳實踐

**輸出**：`research.md`

### Phase 1：設計與契約

**目標**：定義資料模型、API 契約、快速入門範例

**設計任務**：
1. `data-model.md` - 定義 CGAAlgebraBase 介面、blade 索引表、Motor/Point 遮罩
2. `contracts/` - 型別定義檔（.pyi）
3. `quickstart.md` - 使用範例

**輸出**：`data-model.md`, `contracts/`, `quickstart.md`

### Phase 2：任務分解

**目標**：生成可執行的任務清單

**由 `/speckit.tasks` 指令生成**

## 關鍵設計決策

### D1：統一介面架構

```
CGA(n) ─────────────────────────────────────────────────────┐
                                                            │
Cl(p, q, r) ───► 是否為 CGA 簽名？ ───Yes───► CGA(n-1) ────┤
                        │                                   │
                       No                                   │
                        │                                   │
                        ▼                                   ▼
              RuntimeCliffordAlgebra              n <= 5 ?
                (附警告)                          /        \
                                               Yes          No
                                                │            │
                                                ▼            ▼
                                    HardcodedCGAWrapper  RuntimeCGAAlgebra
                                    (cga0d..cga5d)      (動態展開)
                                                │            │
                                                └────────────┘
                                                       │
                                                       ▼
                                              CGAAlgebraBase 介面
```

### D2：運行時動態展開策略

運行時算法需要在不使用 Python 迴圈的情況下計算幾何積。策略：

1. **首次呼叫時生成張量操作**
   - 使用 `cga_factory` 計算非零乘積係數
   - 生成等效的張量索引和係數列表
   - 快取結果供後續使用

2. **張量化批次操作**
   - 將所有非零乘積的索引打包為張量
   - 使用 `torch.index_select` + 廣播計算
   - 避免 Python 層的迴圈

3. **ONNX 相容性**
   - 索引張量作為常數嵌入模型
   - 使用 Gather 操作（非 Loop）

### D3：CGA0D 代數規格

| 屬性 | 值 |
|------|-----|
| Clifford 簽名 | Cl(1, 1, 0) |
| Blade 數 | 4 |
| 歐幾里得維度 | 0 |
| Point 分量 | 2（e+, e-）|
| Motor 分量 | 2（scalar, e+-）|

Blade 索引表：
| 索引 | Blade | Grade |
|------|-------|-------|
| 0 | 1 | 0 |
| 1 | e+ | 1 |
| 2 | e- | 1 |
| 3 | e+- | 2 |
