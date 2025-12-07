# 實作計畫：CGA4D 與 CGA5D 支援

**分支**: `003-cga4d-cga5d` | **日期**: 2025-12-07 | **規格**: [spec.md](./spec.md)
**輸入**: 功能規格書 `/specs/003-cga4d-cga5d/spec.md`

## 摘要

新增 CGA4D (Cl(5,1)) 和 CGA5D (Cl(6,1)) 支援，利用現有 CGA 程式碼生成器架構，為高維度 CARE Transformer 和幾何深度學習應用提供高效能幾何代數運算。實作策略為擴展現有通用化生成器，支援更高維度代數。

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
- CGA4D：三明治積計算量 ~5,800 次乘法（目標 >95% 減少）
- CGA5D：三明治積計算量 ~28,700 次乘法（目標 >98% 減少）
- CGA4D 批次吞吐量 > 500,000 點/秒 (CPU)
- CGA5D 批次吞吐量 > 100,000 點/秒 (CPU)

**約束**:
- ONNX 匯出無 Loop 節點
- 純 PyTorch，禁止平台特定擴充
- CGA 運算強制 float32

**規模/範圍**:

| 屬性 | CGA3D Cl(4,1) | CGA4D Cl(5,1) | CGA5D Cl(6,1) |
|------|---------------|---------------|---------------|
| 基底維度 | 3D (e1,e2,e3) | 4D (e1,e2,e3,e4) | 5D (e1,e2,e3,e4,e5) |
| 簽名 | (+,+,+,+,-) | (+,+,+,+,+,-) | (+,+,+,+,+,+,-) |
| Blade 數量 | 32 | 64 | 128 |
| UPGC Point | 5 分量 | 6 分量 | 7 分量 |
| Motor | 16 分量 | 31 分量 | 64 分量 |

### Grade 分佈

| Grade | CGA3D (n=5) | CGA4D (n=6) | CGA5D (n=7) |
|-------|-------------|-------------|-------------|
| 0 | 1 | 1 | 1 |
| 1 | 5 | 6 | 7 |
| 2 | 10 | 15 | 21 |
| 3 | 10 | 20 | 35 |
| 4 | 5 | 15 | 35 |
| 5 | 1 | 6 | 21 |
| 6 | - | 1 | 7 |
| 7 | - | - | 1 |

### Motor 分量計算
- CGA4D: 1 (G0) + 15 (G2) + 15 (G4) = **31 分量**
- CGA5D: 1 (G0) + 21 (G2) + 35 (G4) + 7 (G6) = **64 分量**

## 憲法檢查

*閘門：必須在 Phase 0 研究前通過。Phase 1 設計後重新檢查。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ 通過 | 所有輸出必須通過 ONNX 匯出驗證 |
| II. 平台相容性 | ✅ 通過 | 純 PyTorch，MPS/CUDA 雙平台支援 |
| III. 無迴圈前向傳播 | ✅ 通過 | 生成器輸出完全展開的算術式 |
| IV. 硬編碼代數展開 | ✅ 通過 | 禁止 Cayley 表，使用符號展開 |
| V. 數值精度安全 | ✅ 通過 | CGA4DCareLayer/CGA5DCareLayer 處理 fp16→fp32→fp16 |
| VI. 文件語言規範 | ✅ 通過 | 規格文件使用繁體中文 |
| VII. 增量提交原則 | ✅ 通過 | 每個任務/檢查點後提交 Git |

## 專案結構

### 文件（本功能）

```text
specs/003-cga4d-cga5d/
├── plan.md              # 本文件
├── spec.md              # 功能規格書
├── research.md          # Phase 0 研究輸出
├── data-model.md        # Phase 1 資料模型
├── quickstart.md        # Phase 1 快速入門
└── contracts/           # Phase 1 介面定義
    ├── cga4d_functional.pyi  # CGA4D 生成函式型別定義
    └── cga5d_functional.pyi  # CGA5D 生成函式型別定義
```

### 原始碼（專案根目錄）

```text
fast_clifford/
├── __init__.py                     # 更新匯出 cga4d, cga5d
├── codegen/
│   ├── __init__.py
│   ├── base.py                     # 基礎代數類別（現有）
│   ├── sparse_analysis.py          # 稀疏性分析（現有，需擴展支援 4D/5D）
│   ├── generate.py                 # 生成器（現有，需擴展支援 4D/5D）
│   └── cga_factory.py              # CGA 代數工廠（現有，需擴展範圍檢查）
├── algebras/
│   ├── __init__.py                 # 更新匯出
│   ├── cga3d/                      # 現有 3D 實作
│   ├── cga2d/                      # 現有 2D 實作
│   ├── cga1d/                      # 現有 1D 實作
│   ├── cga4d/                      # 新增：4D 共形幾何代數
│   │   ├── __init__.py
│   │   ├── algebra.py              # CGA4D 代數定義
│   │   ├── functional.py           # 生成的硬編碼函式
│   │   └── layers.py               # CGA4DCareLayer
│   └── cga5d/                      # 新增：5D 共形幾何代數
│       ├── __init__.py
│       ├── algebra.py              # CGA5D 代數定義
│       ├── functional.py           # 生成的硬編碼函式
│       └── layers.py               # CGA5DCareLayer
└── tests/
    ├── cga4d/                      # 新增：CGA4D 測試
    │   ├── __init__.py
    │   ├── test_numerical.py
    │   └── test_onnx.py
    └── cga5d/                      # 新增：CGA5D 測試
        ├── __init__.py
        ├── test_numerical.py
        └── test_onnx.py

scripts/
├── generate_cga3d.py               # 現有
├── generate_cga2d.py               # 現有
├── generate_cga1d.py               # 現有
├── generate_cga4d.py               # 新增
└── generate_cga5d.py               # 新增
```

**結構決策**:
- 沿用現有按代數類型分資料夾結構
- 擴展 `cga_factory.py` 支援維度 4 和 5
- 每個代數類型獨立測試目錄，便於獨立驗證

## 實作策略

### 策略選擇：擴展現有通用化生成器

分析現有 CGA1D/2D/3D 實作後，決定採用「擴展現有生成器」策略：

1. **擴展範圍檢查**：修改 `cga_factory.py` 支援 euclidean_dim 4 和 5
2. **重用生成邏輯**：`generate.py` 的核心邏輯不變，僅擴展維度支援
3. **遵循現有模式**：algebra.py、functional.py、layers.py 結構一致

### 不採用的策略

- ❌ 重寫生成器：現有生成器已證明有效
- ❌ 手動編寫硬編碼函式：CGA4D 有 64 blades，CGA5D 有 128 blades，手動編寫不可行

## 四階段流水線

### Phase 1: Codegen（生成器擴展）

**目標**: 擴展程式碼生成器支援 CGA4D 和 CGA5D

**輸入**:
- CGA4D Cl(5,1) 代數規則
- CGA5D Cl(6,1) 代數規則
- 稀疏性假設（與 CGA3D 相同模式）

**輸出**:
- `fast_clifford/algebras/cga4d/functional.py`
- `fast_clifford/algebras/cga5d/functional.py`

**關鍵函式**:
```python
# CGA4D
def sandwich_product_sparse_4d(
    motor: torch.Tensor,  # (..., 31) - 偶數 grade 分量 (G0 + G2 + G4)
    point: torch.Tensor,  # (..., 6) - Grade 1 分量
) -> torch.Tensor:        # (..., 6)

# CGA5D
def sandwich_product_sparse_5d(
    motor: torch.Tensor,  # (..., 64) - 偶數 grade 分量 (G0 + G2 + G4 + G6)
    point: torch.Tensor,  # (..., 7) - Grade 1 分量
) -> torch.Tensor:        # (..., 7)
```

### Phase 2: Wrapper（封裝）

**目標**: 實作 `CGA4DCareLayer` 和 `CGA5DCareLayer`

**程式碼模式**（與 CGA3D 相同）:
```python
class CGA4DCareLayer(nn.Module):
    def forward(self, motor: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        original_dtype = point.dtype
        motor = motor.to(torch.float32)
        point = point.to(torch.float32)
        result = sandwich_product_sparse_4d(motor, point)
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
   - CGA4D 和 CGA5D 誤差 < 1e-6

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

1. **CGA4D 代數規則**: 驗證 Cl(5,1) 的幾何積規則與 blade 索引
2. **CGA5D 代數規則**: 驗證 Cl(6,1) 的幾何積規則與 blade 索引
3. **稀疏性驗證**: 確認 4D/5D 的 Motor × Point × Motor_rev 輸出僅有 Grade 1
4. **生成器擴展**: 確認現有生成器可處理更大的代數結構
5. **效能預估**: 評估 CGA5D 的 128 blade 生成函式的效能影響
