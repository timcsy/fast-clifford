# 快速入門：CGA 幾何代數模組

## 安裝

```bash
# 克隆專案
git clone <repo_url>
cd CGA-CARE

# 使用 uv 安裝依賴
uv sync
```

## 基本使用

### 1. 使用 CGA 三明治積

```python
import torch
from cga_care.functional import sandwich_product_sparse
from cga_care.nn import CGACareLayer

# 建立 Motor（16 個分量）
# 索引: [scalar, e12, e13, e1+, e1-, e23, e2+, e2-, e3+, e3-, e+-, ...]
motor = torch.randn(batch_size, 16)

# 建立 UPGC Point（5 個分量）
# 索引: [e1, e2, e3, e+, e-]
point = torch.randn(batch_size, 5)

# 計算三明治積
result = sandwich_product_sparse(motor, point)
# result.shape == (batch_size, 5)
```

### 2. 使用 PyTorch 封裝層

```python
from cga_care.nn import CGACareLayer

# 建立層（自動處理精度轉換）
layer = CGACareLayer()

# 前向傳播
# 輸入可以是 float16，內部會轉為 float32 計算
motor = motor.half()
point = point.half()
result = layer(motor, point)
# result.dtype == torch.float16
```

### 3. ONNX 匯出

```python
import torch.onnx

# 匯出模型
dummy_motor = torch.randn(1, 16)
dummy_point = torch.randn(1, 5)

torch.onnx.export(
    layer,
    (dummy_motor, dummy_point),
    "cga_layer.onnx",
    opset_version=17,
    input_names=["motor", "point"],
    output_names=["transformed_point"],
    dynamic_axes={
        "motor": {0: "batch_size"},
        "point": {0: "batch_size"},
        "transformed_point": {0: "batch_size"}
    }
)

# 驗證無 Loop 節點
import onnx
model = onnx.load("cga_layer.onnx")
loops = [n for n in model.graph.node if n.op_type == "Loop"]
assert len(loops) == 0, f"發現 {len(loops)} 個 Loop 節點！"
print("驗證通過：無 Loop 節點")
```

## 從 3D 向量建立 UPGC Point

```python
from cga_care.functional import upgc_encode, upgc_decode

# 3D 向量 → UPGC Point
x_3d = torch.tensor([[1.0, 2.0, 3.0]])  # shape: (1, 3)
point = upgc_encode(x_3d)                # shape: (1, 5)

# UPGC Point → 3D 向量
x_recovered = upgc_decode(point)         # shape: (1, 3)
```

## 生成器使用（開發者）

```bash
# 執行程式碼生成器
uv run python scripts/generate_cga.py

# 輸出檔案: cga_care/functional/cga_functional.py
```

## 測試

```bash
# 執行所有測試
uv run pytest cga_care/tests/

# 只執行數值正確性測試
uv run pytest cga_care/tests/test_numerical.py

# 只執行 ONNX 匯出測試
uv run pytest cga_care/tests/test_onnx.py
```

## 目錄結構

```
cga_care/
├── codegen/           # 程式碼生成器（開發時使用）
├── functional/        # 生成的純函式
│   └── cga_functional.py
├── nn/                # PyTorch 封裝層
│   └── cga_layer.py
└── tests/             # 測試
```

## 效能注意事項

1. **精度**: CGA 計算內部強制使用 float32，避免溢位
2. **稀疏性**: 利用 UPGC Point 和 Motor 的稀疏性，計算量約為完整計算的 7%
3. **ONNX**: 匯出的模型只有 Add/Mul/Neg 算子，TensorRT 可完全優化
