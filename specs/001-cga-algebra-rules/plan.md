# 實作計畫：CGA 幾何代數規則定義

**分支**: `001-cga-algebra-rules` | **日期**: 2025-12-05 | **規格**: [spec.md](./spec.md)
**輸入**: 功能規格書 `/specs/001-cga-algebra-rules/spec.md`

## 摘要

建立 CGA ($Cl(4,1)$) 幾何代數的程式碼生成器，利用 sympy/clifford 推導數學公式，輸出完全展開、無迴圈的 PyTorch 函式。實作四階段流水線：Codegen → Wrapper → MPS Optimization → Verification。

## 技術脈絡

**語言/版本**: Python 3.11+
**主要依賴**:
- sympy（符號數學推導）
- clifford（幾何代數參考實現）
- PyTorch 2.0+（目標輸出）
- onnx（匯出驗證）

**儲存**: N/A（程式碼生成，無持久化需求）
**測試**: pytest
**目標平台**:
- 開發：Apple M3 (MPS)
- 生產：NVIDIA GPU (TensorRT via ONNX)

**專案類型**: 單一專案（程式碼生成器 + 輸出模組）
**效能目標**: 三明治積計算量 < 200 次乘法（利用稀疏性）
**約束**:
- ONNX 匯出無 Loop 節點
- 純 PyTorch，禁止平台特定擴充
- CGA 運算強制 float32

**規模/範圍**:
- 32 個基底 blade
- 輸入稀疏性：5 個非零分量（UPGC 點）
- Motor 稀疏性：16 個非零分量（偶數 grade）

## 憲法檢查

*閘門：必須在 Phase 0 研究前通過。Phase 1 設計後重新檢查。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ 通過 | 所有輸出必須通過 ONNX 匯出驗證 |
| II. 平台相容性 | ✅ 通過 | 純 PyTorch，MPS/CUDA 雙平台支援 |
| III. 無迴圈前向傳播 | ✅ 通過 | 生成器輸出完全展開的算術式 |
| IV. 硬編碼代數展開 | ✅ 通過 | 禁止 Cayley 表，使用符號展開 |
| V. 數值精度安全 | ✅ 通過 | CGACareLayer 處理 fp16→fp32→fp16 |
| VI. 文件語言規範 | ✅ 通過 | 規格文件使用繁體中文 |

## 專案結構

### 文件（本功能）

```text
specs/001-cga-algebra-rules/
├── plan.md              # 本文件
├── spec.md              # 功能規格書
├── research.md          # Phase 0 研究輸出
├── data-model.md        # Phase 1 資料模型
├── quickstart.md        # Phase 1 快速入門
└── contracts/           # Phase 1 介面定義
    └── cga_functional.pyi  # 生成函式的型別定義
```

### 原始碼（專案根目錄）

```text
cga_care/
├── __init__.py
├── codegen/                    # Phase 1: 程式碼生成器
│   ├── __init__.py
│   ├── algebra.py              # sympy/clifford 代數定義
│   ├── sparse_analysis.py      # 稀疏性分析
│   └── generate.py             # 生成器主程式
├── functional/                 # Phase 1 輸出目標
│   ├── __init__.py
│   └── cga_functional.py       # 生成的硬編碼函式
├── nn/                         # Phase 2: PyTorch 封裝
│   ├── __init__.py
│   └── cga_layer.py            # CGACareLayer (nn.Module)
└── tests/                      # Phase 4: 驗證
    ├── __init__.py
    ├── test_numerical.py       # 數值正確性
    └── test_onnx.py            # ONNX 匯出驗證

scripts/
└── generate_cga.py             # 執行生成器的腳本
```

**結構決策**: 採用單一專案結構，分離生成器（codegen/）與生成輸出（functional/）。生成器使用 sympy/clifford，輸出是純 PyTorch。

## 四階段流水線

### Phase 1: Codegen（生成器）

**目標**: 利用 sympy/clifford 推導 CGA 數學公式，輸出硬編碼 PyTorch 函式

**輸入**:
- CGA 代數規則（spec.md 定義）
- 稀疏性假設（UPGC 點、Motor）

**輸出**:
- `cga_care/functional/cga_functional.py`
- 完全展開、無迴圈的 PyTorch 函式

**關鍵函式**:
```python
def sandwich_product_sparse(
    motor: torch.Tensor,  # (..., 16) - 偶數 grade 分量
    point: torch.Tensor,  # (..., 5) - Grade 1 分量
) -> torch.Tensor:        # (..., 5) - 變換後的 Grade 1 分量
    """
    計算 M × X × M̃，利用稀疏性優化
    所有乘法明確展開，無迴圈
    """
```

### Phase 2: Wrapper（封裝）

**目標**: 實作 `CGACareLayer` (nn.Module)

**職責**:
1. fp16 → fp32 → fp16 精度轉換
2. 將生成的函式與 PyTorch autograd 接軌
3. 提供乾淨的 API 給 Transformer 使用

**程式碼模式**:
```python
class CGACareLayer(nn.Module):
    def forward(self, motor: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        original_dtype = point.dtype
        motor = motor.to(torch.float32)
        point = point.to(torch.float32)

        result = sandwich_product_sparse(motor, point)

        return result.to(original_dtype)
```

### Phase 3: MPS Optimization（Mac 優化）

**目標**: 確保在 M3 Mac 上啟用 Graph Fusion 加速

**方法**:
- 對生成的函式應用 `torch.jit.script`（無迴圈版本安全）
- 驗證 MPS 後端效能

**注意**: 憲法禁止「帶迴圈的 torch.jit.script」，但我們的函式是純算術展開，符合規範。

### Phase 4: Verification（驗證）

**驗證項目**:

1. **數值正確性**
   - 對比 Python clifford 庫計算結果
   - 誤差 < 1e-6

2. **ONNX 匯出**
   - `torch.onnx.export()` 成功
   - ONNX 計算圖無 Loop 節點
   - 只有 Add/Mul/Neg 等基本算子

3. **跨平台驗證**
   - MPS (Apple M3) 測試通過
   - CUDA (如有) 或 CPU 測試通過

4. **精度測試**
   - float32 與 float16 輸入的輸出一致（容差內）

## 複雜度追蹤

> 無憲法違規需要證明

## 研究待辦

以下問題將在 Phase 0 研究中解決：

1. **sympy/clifford 整合**: 如何從 clifford 庫提取乘法規則並轉換為 sympy 符號表達式
2. **稀疏性推導**: 驗證 Motor × Point × Motor_rev 的輸出確實只有 Grade 1
3. **程式碼生成策略**: 確定最佳的 Python AST 生成方法
