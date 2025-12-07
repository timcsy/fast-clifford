# 快速入門：CGA2D 與 CGA1D

本指南說明如何使用 fast-clifford 的 CGA2D 和 CGA1D 模組進行 2D 和 1D 共形幾何代數運算。

## 環境準備

```bash
# 使用 uv 安裝依賴
uv sync
```

## CGA2D 基本使用

### 1. 匯入模組

```python
import torch
from fast_clifford import cga2d

# 或直接匯入層
from fast_clifford.algebras.cga2d import CGA2DCareLayer
from fast_clifford.algebras.cga2d.functional import (
    upgc_encode,
    upgc_decode,
    sandwich_product_sparse,
)
```

### 2. 編碼 2D 點

```python
# 建立 2D 點 (batch_size=4, dim=2)
points_2d = torch.tensor([
    [1.0, 0.0],   # 點 A
    [0.0, 1.0],   # 點 B
    [1.0, 1.0],   # 點 C
    [-1.0, 0.5],  # 點 D
])

# 編碼為 UPGC 表示 (4 分量: e1, e2, e+, e-)
upgc_points = upgc_encode(points_2d)
print(f"UPGC 形狀: {upgc_points.shape}")  # (4, 4)
```

### 3. 建立 2D 馬達（旋轉 + 平移）

```python
import math

# 2D 旋轉馬達: R = cos(θ/2) + sin(θ/2)*e12
theta = math.pi / 4  # 旋轉 45 度
cos_half = math.cos(theta / 2)
sin_half = math.sin(theta / 2)

# Motor 格式: [scalar, e12, e1+, e1-, e2+, e2-, e+-, e12+-]
rotation_motor = torch.tensor([
    cos_half,   # scalar
    sin_half,   # e12
    0.0, 0.0,   # e1+, e1-
    0.0, 0.0,   # e2+, e2-
    0.0,        # e+-
    0.0,        # e12+-
])

# 廣播至批次
motor_batch = rotation_motor.unsqueeze(0).expand(4, -1)  # (4, 8)
```

### 4. 套用三明治積變換

```python
# 使用稀疏三明治積
transformed = sandwich_product_sparse(motor_batch, upgc_points)

# 解碼回 2D 座標
result_2d = upgc_decode(transformed)
print(f"變換後 2D 點: {result_2d}")
```

### 5. 使用 CGA2DCareLayer（PyTorch 整合）

```python
# 建立層（處理精度轉換）
layer = CGA2DCareLayer()

# 使用 fp16 輸入（自動轉換為 fp32 計算）
motor_fp16 = motor_batch.half()
points_fp16 = upgc_points.half()

# 前向傳播（支援 autograd）
output = layer(motor_fp16, points_fp16)
print(f"輸出 dtype: {output.dtype}")  # torch.float16
```

---

## CGA1D 基本使用

### 1. 匯入模組

```python
import torch
from fast_clifford import cga1d

# 或直接匯入層
from fast_clifford.algebras.cga1d import CGA1DCareLayer
from fast_clifford.algebras.cga1d.functional import (
    upgc_encode,
    upgc_decode,
    sandwich_product_sparse,
)
```

### 2. 編碼 1D 點

```python
# 建立 1D 純量 (batch_size=4, dim=1)
points_1d = torch.tensor([
    [0.0],
    [1.0],
    [2.0],
    [-1.0],
])

# 編碼為 UPGC 表示 (3 分量: e1, e+, e-)
upgc_points = upgc_encode(points_1d)
print(f"UPGC 形狀: {upgc_points.shape}")  # (4, 3)
```

### 3. 建立 1D 平移馬達

```python
# 1D 平移馬達: T = 1 - (1/2)*t*einf
# 其中 einf = e+ + e-
translation = 3.0  # 平移 3 單位

# Motor 格式: [scalar, e1+, e1-, e+-]
# T = 1 - 0.5*t*(e+ + e-) 作用在 e1 上產生 e1+ 和 e1- 分量
translation_motor = torch.tensor([
    1.0,           # scalar
    -0.5 * translation,  # e1+ (從 -0.5*t*e1*e+)
    -0.5 * translation,  # e1- (從 -0.5*t*e1*e-)
    0.0,           # e+-
])

# 廣播至批次
motor_batch = translation_motor.unsqueeze(0).expand(4, -1)  # (4, 4)
```

### 4. 套用三明治積變換

```python
# 使用稀疏三明治積
transformed = sandwich_product_sparse(motor_batch, upgc_points)

# 解碼回 1D 座標
result_1d = upgc_decode(transformed)
print(f"變換後 1D 點: {result_1d}")
# 預期: [3.0], [4.0], [5.0], [2.0]
```

### 5. 使用 CGA1DCareLayer（PyTorch 整合）

```python
# 建立層
layer = CGA1DCareLayer()

# 前向傳播
output = layer(motor_batch, upgc_points)
decoded = upgc_decode(output)
print(f"平移結果: {decoded.squeeze()}")
```

---

## ONNX 匯出

### 匯出 CGA2D 模型

```python
import torch.onnx

layer = CGA2DCareLayer()
dummy_motor = torch.randn(1, 8)
dummy_point = torch.randn(1, 4)

torch.onnx.export(
    layer,
    (dummy_motor, dummy_point),
    "cga2d_care_layer.onnx",
    opset_version=17,
    input_names=["motor", "point"],
    output_names=["transformed_point"],
)
```

### 匯出 CGA1D 模型

```python
layer = CGA1DCareLayer()
dummy_motor = torch.randn(1, 4)
dummy_point = torch.randn(1, 3)

torch.onnx.export(
    layer,
    (dummy_motor, dummy_point),
    "cga1d_care_layer.onnx",
    opset_version=17,
    input_names=["motor", "point"],
    output_names=["transformed_point"],
)
```

---

## 效能注意事項

### 批次處理

所有操作都支援批次維度：

```python
# 批次處理 1000 個點
batch_size = 1000
motors = torch.randn(batch_size, 8)   # CGA2D
points = torch.randn(batch_size, 4)   # CGA2D UPGC points

# 平行計算
results = sandwich_product_sparse(motors, points)
```

### 混合精度

CGA 層自動處理精度轉換以確保數值穩定性：

```python
# fp16 輸入 → fp32 計算 → fp16 輸出
layer = CGA2DCareLayer()
output = layer(motor.half(), point.half())  # 內部使用 fp32
```

### 裝置支援

支援 CPU、CUDA 和 MPS（Apple Silicon）：

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
layer = CGA2DCareLayer().to(device)
motor = motor.to(device)
point = point.to(device)
output = layer(motor, point)
```

---

## 與 CGA3D 的對比

| 功能 | CGA3D | CGA2D | CGA1D |
|------|-------|-------|-------|
| 歐幾里得維度 | 3D | 2D | 1D |
| UPGC Point | 5 分量 | 4 分量 | 3 分量 |
| Motor | 16 分量 | 8 分量 | 4 分量 |
| 編碼輸入 | `(..., 3)` | `(..., 2)` | `(..., 1)` |
| 解碼輸出 | `(..., 3)` | `(..., 2)` | `(..., 1)` |

---

## 執行測試

```bash
# 執行 CGA2D 測試
uv run pytest fast_clifford/tests/cga2d/ -v

# 執行 CGA1D 測試
uv run pytest fast_clifford/tests/cga1d/ -v

# 執行所有測試
uv run pytest -v
```
