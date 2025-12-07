# 實作計畫：CGA2D 與 CGA1D 支援

**分支**: `002-cga-2d-1d` | **日期**: 2025-12-07 | **規格**: [spec.md](./spec.md)
**輸入**: 功能規格書 `/specs/002-cga-2d-1d/spec.md`

## 摘要

新增 CGA2D (Cl(3,1)) 和 CGA1D (Cl(2,1)) 支援，利用現有 CGA3D 程式碼生成器架構，為 CARE Transformer 的 2D/1D 版本提供高效能幾何代數運算。實作策略為通用化現有生成器，避免代碼重複。

## 技術脈絡

**語言/版本**: Python 3.11+
**主要依賴**:
- clifford（幾何代數參考實現，用於生成代數規則）
- PyTorch 2.0+（目標輸出）
- onnx（匯出驗證）

**儲存**: N/A（程式碼生成，無持久化需求）
**測試**: pytest
**目標平台**:
- 開發：Apple M3 (MPS)
- 生產：NVIDIA GPU (TensorRT via ONNX)

**專案類型**: 單一專案（程式碼生成器 + 輸出模組）
**效能目標**:
- CGA2D：三明治積計算量 < 250 次乘法（目標 >50% 減少）
- CGA1D：三明治積計算量 < 80 次乘法（目標 >50% 減少）
- 批次吞吐量 > 100,000 點/秒 (CPU)

**約束**:
- ONNX 匯出無 Loop 節點
- 純 PyTorch，禁止平台特定擴充
- CGA 運算強制 float32

**規模/範圍**:

| 屬性 | CGA3D Cl(4,1) | CGA2D Cl(3,1) | CGA1D Cl(2,1) |
|------|---------------|---------------|---------------|
| 基底維度 | 3D (e1,e2,e3) | 2D (e1,e2) | 1D (e1) |
| 簽名 | (+,+,+,+,-) | (+,+,+,-) | (+,+,-) |
| Blade 數量 | 32 | 16 | 8 |
| UPGC Point | 5 分量 | 4 分量 | 3 分量 |
| Motor | 16 分量 | 8 分量 | 4 分量 |

## 憲法檢查

*閘門：必須在 Phase 0 研究前通過。Phase 1 設計後重新檢查。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ 通過 | 所有輸出必須通過 ONNX 匯出驗證 |
| II. 平台相容性 | ✅ 通過 | 純 PyTorch，MPS/CUDA 雙平台支援 |
| III. 無迴圈前向傳播 | ✅ 通過 | 生成器輸出完全展開的算術式 |
| IV. 硬編碼代數展開 | ✅ 通過 | 禁止 Cayley 表，使用符號展開 |
| V. 數值精度安全 | ✅ 通過 | CGA2DCareLayer/CGA1DCareLayer 處理 fp16→fp32→fp16 |
| VI. 文件語言規範 | ✅ 通過 | 規格文件使用繁體中文 |
| VII. 增量提交原則 | ✅ 通過 | 每個任務/檢查點後提交 Git |

## 專案結構

### 文件（本功能）

```text
specs/002-cga-2d-1d/
├── plan.md              # 本文件
├── spec.md              # 功能規格書
├── research.md          # Phase 0 研究輸出
├── data-model.md        # Phase 1 資料模型
├── quickstart.md        # Phase 1 快速入門
└── contracts/           # Phase 1 介面定義
    ├── cga2d_functional.pyi  # CGA2D 生成函式型別定義
    └── cga1d_functional.pyi  # CGA1D 生成函式型別定義
```

### 原始碼（專案根目錄）

```text
fast_clifford/
├── __init__.py                     # 更新匯出 cga2d, cga1d
├── codegen/
│   ├── __init__.py
│   ├── base.py                     # 基礎代數類別（現有）
│   ├── sparse_analysis.py          # 稀疏性分析（現有，需通用化）
│   ├── generate.py                 # 生成器（現有 CGA3D）
│   └── cga_factory.py              # 新增：通用 CGA 代數工廠
├── algebras/
│   ├── __init__.py                 # 更新匯出
│   ├── cga3d/                      # 現有 3D 實作
│   │   └── ...
│   ├── cga2d/                      # 新增：2D 共形幾何代數
│   │   ├── __init__.py
│   │   ├── algebra.py              # CGA2D 代數定義
│   │   ├── functional.py           # 生成的硬編碼函式
│   │   └── layers.py               # CGA2DCareLayer
│   └── cga1d/                      # 新增：1D 共形幾何代數
│       ├── __init__.py
│       ├── algebra.py              # CGA1D 代數定義
│       ├── functional.py           # 生成的硬編碼函式
│       └── layers.py               # CGA1DCareLayer
└── tests/
    ├── cga2d/                      # 新增：CGA2D 測試
    │   ├── __init__.py
    │   ├── test_numerical.py
    │   └── test_onnx.py
    └── cga1d/                      # 新增：CGA1D 測試
        ├── __init__.py
        ├── test_numerical.py
        └── test_onnx.py

scripts/
├── generate_cga3d.py               # 現有
├── generate_cga2d.py               # 新增
└── generate_cga1d.py               # 新增
```

**結構決策**:
- 沿用 CGA3D 的按代數類型分資料夾結構
- 新增 `cga_factory.py` 提供通用 CGA 代數建構能力
- 每個代數類型獨立測試目錄，便於獨立驗證

## 實作策略

### 策略選擇：通用化代碼生成器

分析現有 CGA3D 實作後，決定採用「通用化現有生成器」策略：

1. **新增 CGA 工廠**：建立可參數化的 CGA 代數建構器
2. **重用稀疏分析**：通用化 `sparse_analysis.py` 支援不同維度
3. **重用生成邏輯**：`generate.py` 的核心邏輯不變，僅參數化維度

### 不採用的策略

- ❌ 完全複製 CGA3D 代碼：代碼重複，維護困難
- ❌ 抽象基類繼承：過度工程化，增加複雜度

## 四階段流水線

### Phase 1: Codegen（生成器擴展）

**目標**: 擴展程式碼生成器支援 CGA2D 和 CGA1D

**輸入**:
- CGA2D Cl(3,1) 代數規則
- CGA1D Cl(2,1) 代數規則
- 稀疏性假設（與 CGA3D 相同模式）

**輸出**:
- `fast_clifford/algebras/cga2d/functional.py`
- `fast_clifford/algebras/cga1d/functional.py`

**關鍵函式**:
```python
# CGA2D
def sandwich_product_sparse_2d(
    motor: torch.Tensor,  # (..., 8) - 偶數 grade 分量
    point: torch.Tensor,  # (..., 4) - Grade 1 分量
) -> torch.Tensor:        # (..., 4)

# CGA1D
def sandwich_product_sparse_1d(
    motor: torch.Tensor,  # (..., 4) - 偶數 grade 分量
    point: torch.Tensor,  # (..., 3) - Grade 1 分量
) -> torch.Tensor:        # (..., 3)
```

### Phase 2: Wrapper（封裝）

**目標**: 實作 `CGA2DCareLayer` 和 `CGA1DCareLayer`

**程式碼模式**（與 CGA3D 相同）:
```python
class CGA2DCareLayer(nn.Module):
    def forward(self, motor: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        original_dtype = point.dtype
        motor = motor.to(torch.float32)
        point = point.to(torch.float32)
        result = sandwich_product_sparse_2d(motor, point)
        return result.to(original_dtype)
```

### Phase 3: MPS Optimization（Mac 優化）

**目標**: 確保在 M3 Mac 上啟用 Graph Fusion 加速

**方法**:
- 對生成的函式應用 `torch.jit.script`
- 驗證 MPS 後端效能

### Phase 4: Verification（驗證）

**驗證項目**:

1. **數值正確性**
   - 對比 clifford 庫計算結果
   - CGA2D 和 CGA1D 誤差 < 1e-6

2. **ONNX 匯出**
   - `torch.onnx.export()` 成功
   - ONNX 計算圖無 Loop 節點

3. **跨平台驗證**
   - MPS (Apple M3) 測試通過
   - CPU 測試通過

4. **精度測試**
   - float32 與 float16 輸入的輸出一致

## 複雜度追蹤

> 無憲法違規需要證明

## 研究待辦

以下問題將在 Phase 0 研究中解決：

1. **CGA2D 代數規則**: 驗證 Cl(3,1) 的幾何積規則與 blade 索引
2. **CGA1D 代數規則**: 驗證 Cl(2,1) 的幾何積規則與 blade 索引
3. **稀疏性驗證**: 確認 2D/1D 的 Motor × Point × Motor_rev 輸出僅有 Grade 1
4. **生成器通用化**: 決定最佳的參數化方式
