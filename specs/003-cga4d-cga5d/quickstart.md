# 快速入門：CGA4D 與 CGA5D

本指南說明如何使用 fast-clifford 的 CGA4D 和 CGA5D 模組進行 4D 和 5D 共形幾何代數運算。

## 環境準備

```bash
# 使用 uv 安裝依賴
uv sync
```

## CGA4D 基本使用

### 1. 匯入模組

```python
import torch
from fast_clifford import cga4d

# 或直接匯入層
from fast_clifford.algebras.cga4d import CGA4DCareLayer
from fast_clifford.algebras.cga4d.functional import (
    upgc_encode,
    upgc_decode,
    sandwich_product_sparse,
)
```

### 2. 編碼 4D 點

```python
# 建立 4D 點 (batch_size=4, dim=4)
points_4d = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # 點 A
    [0.0, 1.0, 0.0, 0.0],   # 點 B
    [1.0, 1.0, 1.0, 0.0],   # 點 C
    [-1.0, 0.5, 0.5, 1.0],  # 點 D
])

# 編碼為 UPGC 表示 (6 分量: e1, e2, e3, e4, e+, e-)
upgc_points = upgc_encode(points_4d)
print(f"UPGC 形狀: {upgc_points.shape}")  # (4, 6)
```

### 3. 建立 4D 馬達（旋轉 + 平移）

```python
import math

# 4D 旋轉馬達: 在 e12 平面旋轉
theta = math.pi / 4  # 旋轉 45 度
cos_half = math.cos(theta / 2)
sin_half = math.sin(theta / 2)

# Motor 格式: [scalar, ...Grade 2 (15)..., ...Grade 4 (15)...]
# 簡化範例：僅設定標量和 e12 分量
rotation_motor = torch.zeros(31)
rotation_motor[0] = cos_half    # scalar
rotation_motor[1] = sin_half    # e12 (假設索引 1 對應 e12)

# 廣播至批次
motor_batch = rotation_motor.unsqueeze(0).expand(4, -1)  # (4, 31)
```

### 4. 套用三明治積變換

```python
# 使用稀疏三明治積
transformed = sandwich_product_sparse(motor_batch, upgc_points)

# 解碼回 4D 座標
result_4d = upgc_decode(transformed)
print(f"變換後 4D 點: {result_4d}")
```

### 5. 使用 CGA4DCareLayer（PyTorch 整合）

```python
# 建立層（處理精度轉換）
layer = CGA4DCareLayer()

# 使用 fp16 輸入（自動轉換為 fp32 計算）
motor_fp16 = motor_batch.half()
points_fp16 = upgc_points.half()

# 前向傳播（支援 autograd）
output = layer(motor_fp16, points_fp16)
print(f"輸出 dtype: {output.dtype}")  # torch.float16
```

---

## CGA5D 基本使用

### 1. 匯入模組

```python
import torch
from fast_clifford import cga5d

# 或直接匯入層
from fast_clifford.algebras.cga5d import CGA5DCareLayer
from fast_clifford.algebras.cga5d.functional import (
    upgc_encode,
    upgc_decode,
    sandwich_product_sparse,
)
```

### 2. 編碼 5D 點

```python
# 建立 5D 向量 (batch_size=4, dim=5)
points_5d = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [-1.0, 0.5, 0.5, 0.5, 1.0],
])

# 編碼為 UPGC 表示 (7 分量: e1, e2, e3, e4, e5, e+, e-)
upgc_points = upgc_encode(points_5d)
print(f"UPGC 形狀: {upgc_points.shape}")  # (4, 7)
```

### 3. 建立 5D 平移馬達

```python
# 5D 平移馬達: T = 1 + (1/2)*t*einf*e1
# 其中 einf = e+ + e-
translation = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0])  # 沿 e1 方向平移 3 單位

# Motor 格式: [scalar, ...Grade 2 (21)..., ...Grade 4 (35)..., ...Grade 6 (7)...]
translation_motor = torch.zeros(64)
translation_motor[0] = 1.0  # scalar

# 平移分量需要根據具體的 blade 索引設定
# 這裡是簡化範例，實際使用時請參考 algebra.py 中的索引定義

# 廣播至批次
motor_batch = translation_motor.unsqueeze(0).expand(4, -1)  # (4, 64)
```

### 4. 套用三明治積變換

```python
# 使用稀疏三明治積
transformed = sandwich_product_sparse(motor_batch, upgc_points)

# 解碼回 5D 座標
result_5d = upgc_decode(transformed)
print(f"變換後 5D 點: {result_5d}")
```

### 5. 使用 CGA5DCareLayer（PyTorch 整合）

```python
# 建立層
layer = CGA5DCareLayer()

# 前向傳播
output = layer(motor_batch, upgc_points)
decoded = upgc_decode(output)
print(f"變換結果: {decoded}")
```

---

## ONNX 匯出

### 匯出 CGA4D 模型

```python
import torch.onnx

layer = CGA4DCareLayer()
dummy_motor = torch.randn(1, 31)
dummy_point = torch.randn(1, 6)

torch.onnx.export(
    layer,
    (dummy_motor, dummy_point),
    "cga4d_care_layer.onnx",
    opset_version=17,
    input_names=["motor", "point"],
    output_names=["transformed_point"],
)
```

### 匯出 CGA5D 模型

```python
layer = CGA5DCareLayer()
dummy_motor = torch.randn(1, 64)
dummy_point = torch.randn(1, 7)

torch.onnx.export(
    layer,
    (dummy_motor, dummy_point),
    "cga5d_care_layer.onnx",
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
motors = torch.randn(batch_size, 31)   # CGA4D
points = torch.randn(batch_size, 6)    # CGA4D UPGC points

# 平行計算
results = sandwich_product_sparse(motors, points)
```

### 混合精度

CGA 層自動處理精度轉換以確保數值穩定性：

```python
# fp16 輸入 → fp32 計算 → fp16 輸出
layer = CGA4DCareLayer()
output = layer(motor.half(), point.half())  # 內部使用 fp32
```

### 裝置支援

支援 CPU、CUDA 和 MPS（Apple Silicon）：

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
layer = CGA4DCareLayer().to(device)
motor = motor.to(device)
point = point.to(device)
output = layer(motor, point)
```

---

## 與其他 CGA 代數的對比

| 功能 | CGA3D | CGA4D | CGA5D |
|------|-------|-------|-------|
| 歐幾里得維度 | 3D | 4D | 5D |
| UPGC Point | 5 分量 | 6 分量 | 7 分量 |
| Motor | 16 分量 | 31 分量 | 64 分量 |
| 編碼輸入 | `(..., 3)` | `(..., 4)` | `(..., 5)` |
| 解碼輸出 | `(..., 3)` | `(..., 4)` | `(..., 5)` |

---

## 執行測試

```bash
# 執行 CGA4D 測試
uv run pytest fast_clifford/tests/cga4d/ -v

# 執行 CGA5D 測試
uv run pytest fast_clifford/tests/cga5d/ -v

# 執行所有測試
uv run pytest -v
```
