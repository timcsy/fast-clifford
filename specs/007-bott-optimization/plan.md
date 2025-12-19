# 實作計畫：Bott 週期性優化

**分支**: `007-bott-optimization` | **日期**: 2025-12-19 | **規格**: [spec.md](./spec.md)
**輸入**: 功能規格來自 `/specs/007-bott-optimization/spec.md`

## 摘要

透過以下優化減少 Clifford 代數函式庫的儲存空間和提升效能：
1. 將預生成代數從 55 個減少到 20 個（只保留 p >= q 且 p+q < 8）
2. 實作 SymmetricClWrapper 支援 p < q 的代數
3. 使用張量化 einsum 運算加速 Bott 週期性的 16×16 矩陣運算

## 技術背景

**語言/版本**: Python 3.11+
**主要依賴**: PyTorch 2.0+, clifford (測試對照)
**儲存**: 檔案系統（預生成 Python 模組）
**測試**: pytest
**目標平台**: 跨平台（MPS 開發 / CUDA 生產）
**專案類型**: 函式庫
**效能目標**: Bott 幾何積加速 10x+（相比 Python 迴圈）
**約束**:
- ONNX 可匯出（無 Loop 節點）
- 純 PyTorch（無平台特定擴充）
- 儲存空間 < 30MB（從 ~600MB 減少）
**規模/範圍**: 20 個預生成代數，支援任意 p+q 維度

## 憲法檢查

*閘門: 必須在 Phase 0 研究前通過。Phase 1 設計後重新檢查。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ 通過 | einsum 可匯出 ONNX，無 Loop 節點 |
| II. 平台相容性 | ✅ 通過 | 純 PyTorch 操作 |
| III. 無迴圈前向傳播 | ✅ 通過 | 使用 einsum 取代 Python 迴圈 |
| IV. 硬編碼代數展開 | ⚠️ 部分 | Bott 使用乘法表張量（已有代數仍用硬編碼） |
| V. 數值精度安全 | ✅ 通過 | 維持 float32 計算 |
| VI. 文件語言規範 | ✅ 通過 | 繁體中文文件 |
| VII. 增量提交原則 | ✅ 通過 | 按 Phase 提交 |

**原則 IV 說明**: Bott 週期性必須使用預計算乘法表張量才能實現張量化運算。這是必要的權衡：
- 乘法表張量大小僅 (n, n, n) 其中 n ≤ 128（最大基底代數 blade 數）
- 換取 10x+ 效能提升和 ONNX 相容性（無 Loop 節點）
- 仍比 Cayley 表方法小且更快

## 專案結構

### 文件（此功能）

```text
specs/007-bott-optimization/
├── plan.md              # 本文件
├── research.md          # Phase 0 輸出
├── data-model.md        # Phase 1 輸出
├── quickstart.md        # Phase 1 輸出
├── contracts/           # Phase 1 輸出
└── tasks.md             # Phase 2 輸出（由 /speckit.tasks 建立）
```

### 原始碼（倉庫根目錄）

```text
fast_clifford/
├── clifford/
│   ├── __init__.py          # 更新：閾值、路由邏輯
│   ├── registry.py          # 新增：SymmetricClWrapper
│   ├── bott.py              # 重寫：張量化運算
│   └── specializations/
│       └── (現有 VGA/CGA/PGA)
└── algebras/generated/
    └── cl_{p}_{q}/          # 保留 20 個，刪除 35 個

tests/
├── test_symmetric.py        # 新增：對稱代數測試
├── test_bott.py             # 更新：效能測試
└── benchmark/
    └── test_bott_benchmark.py  # 新增：效能基準
```

**結構決定**: 修改現有 `fast_clifford/clifford/` 模組，新增 SymmetricClWrapper 到 registry.py，重寫 bott.py 使用張量化運算。

## 複雜度追蹤

| 違規 | 為何需要 | 拒絕更簡單替代方案的原因 |
|------|----------|--------------------------|
| 乘法表張量 (原則 IV) | Bott 張量化運算需要預計算表 | 不使用會產生 Python 迴圈 → ONNX Loop 節點 |
