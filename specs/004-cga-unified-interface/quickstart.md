# 快速入門：CGA(n) 統一介面

## 安裝

```bash
git clone https://github.com/timcsy/fast-clifford.git
cd fast-clifford
uv sync
```

## 基本使用

### 使用 CGA(n) 建立代數

```python
import torch
from fast_clifford import CGA

# 建立 CGA3D 代數（使用硬編碼快速算法）
cga3d = CGA(3)

print(f"維度: {cga3d.euclidean_dim}")      # 3
print(f"Blade 數: {cga3d.blade_count}")    # 32
print(f"Point 分量: {cga3d.point_count}")  # 5
print(f"Motor 分量: {cga3d.motor_count}")  # 16
print(f"簽名: {cga3d.clifford_notation}")  # Cl(4,1,0)

# 建立 CGA0D 代數（新增）
cga0d = CGA(0)
print(f"CGA0D Blade 數: {cga0d.blade_count}")  # 4

# 建立 CGA6D 代數（使用運行時算法）
cga6d = CGA(6)
print(f"CGA6D Blade 數: {cga6d.blade_count}")  # 256
```

### 使用 Cl(p, q, r) 建立代數

```python
from fast_clifford import Cl

# 使用 Clifford 簽名建立 CGA3D
cga3d = Cl(4, 1, 0)  # 等同於 CGA(3)
cga3d = Cl(4, 1)     # r=0 為預設

# 建立非 CGA 代數（會發出警告）
ga3d = Cl(3, 0, 0)   # 純 3D 幾何代數
# Warning: Cl(3,0,0) 不是 CGA 簽名，某些 CGA 特定操作可能不可用

# 建立帶退化維度的代數
pga3d = Cl(3, 0, 1)  # 3D 射影幾何代數
```

### 點編碼與解碼

```python
from fast_clifford import CGA

cga3d = CGA(3)

# 編碼 3D 點
x = torch.tensor([[1.0, 2.0, 3.0]])
point = cga3d.upgc_encode(x)
print(f"UPGC 點 shape: {point.shape}")  # (1, 5)

# 解碼回 3D
x_decoded = cga3d.upgc_decode(point)
print(f"解碼點: {x_decoded}")  # tensor([[1., 2., 3.]])
```

### 三明治積變換

```python
from fast_clifford import CGA
import torch

cga3d = CGA(3)

# 建立 Motor（旋轉 + 平移）
motor = torch.randn(1, 16)  # 16 Motor 分量

# 編碼點
x = torch.tensor([[1.0, 2.0, 3.0]])
point = cga3d.upgc_encode(x)

# 執行三明治積 M × X × M̃
transformed_point = cga3d.sandwich_product_sparse(motor, point)

# 解碼結果
result = cga3d.upgc_decode(transformed_point)
print(f"變換後: {result}")
```

### 使用 PyTorch 層

```python
from fast_clifford import CGA
import torch

cga3d = CGA(3)

# 取得 CareLayer
care_layer = cga3d.get_care_layer()

# 批次處理
batch_size = 1024
motors = torch.randn(batch_size, 16)
points = torch.randn(batch_size, 5)

# 前向傳播
outputs = care_layer(motors, points)
print(f"輸出 shape: {outputs.shape}")  # (1024, 5)
```

### 完整轉換流水線

```python
from fast_clifford import CGA
import torch

cga3d = CGA(3)

# 取得完整流水線（編碼 → 三明治積 → 解碼）
pipeline = cga3d.get_transform_pipeline()

# 輸入 3D 座標
batch_size = 1024
motors = torch.randn(batch_size, 16)
x = torch.randn(batch_size, 3)  # 3D 歐幾里得座標

# 一次完成轉換
y = pipeline(motors, x)  # 輸出 3D 座標
print(f"輸出 shape: {y.shape}")  # (1024, 3)
```

## 高維度運行時代數

```python
from fast_clifford import CGA
import torch

# CGA6D（運行時算法）
cga6d = CGA(6)

print(f"CGA6D Blade 數: {cga6d.blade_count}")    # 256
print(f"CGA6D Point 分量: {cga6d.point_count}")  # 8
print(f"CGA6D Motor 分量: {cga6d.motor_count}")  # 127

# 使用方式與硬編碼版本相同
x = torch.randn(4, 6)  # 6D 座標
point = cga6d.upgc_encode(x)
print(f"UPGC 點 shape: {point.shape}")  # (4, 8)
```

## 訓練範例

```python
from fast_clifford import CGA
import torch
import torch.nn as nn

cga3d = CGA(3)

# 可學習的 Motor 參數
motor = nn.Parameter(torch.randn(1, 16))

# 輸入點
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# 編碼 → 三明治積 → 解碼
point = cga3d.upgc_encode(x)
transformed = cga3d.sandwich_product_sparse(motor, point)
y = cga3d.upgc_decode(transformed)

# 計算損失並反向傳播
loss = y.sum()
loss.backward()

print(f"Motor 梯度: {motor.grad is not None}")  # True
```

## ONNX 匯出

```python
from fast_clifford import CGA
import torch

cga3d = CGA(3)
care_layer = cga3d.get_care_layer()

# 範例輸入
motor = torch.randn(1, 16)
point = torch.randn(1, 5)

# 匯出至 ONNX
torch.onnx.export(
    care_layer,
    (motor, point),
    "cga3d_care.onnx",
    input_names=["motor", "point"],
    output_names=["output"],
    dynamic_axes={
        "motor": {0: "batch"},
        "point": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=17,
)

# 驗證無 Loop 節點
import onnx
model = onnx.load("cga3d_care.onnx")
op_types = {n.op_type for n in model.graph.node}
assert "Loop" not in op_types
print("✓ ONNX 匯出成功，無 Loop 節點")
```

## API 一覽

### CGA 代數方法

| 方法 | 說明 |
|------|------|
| `upgc_encode(x)` | 歐幾里得 → UPGC 點 |
| `upgc_decode(point)` | UPGC 點 → 歐幾里得 |
| `geometric_product_full(a, b)` | 完整幾何積 |
| `sandwich_product_sparse(motor, point)` | 稀疏三明治積 |
| `reverse_full(mv)` | 多向量反轉 |
| `reverse_motor(motor)` | Motor 反轉 |
| `get_care_layer()` | 取得 CareLayer |
| `get_encoder()` | 取得編碼器 |
| `get_decoder()` | 取得解碼器 |
| `get_transform_pipeline()` | 取得完整流水線 |

### 支援的代數

| 代數 | 簽名 | Blade 數 | 算法 |
|------|------|----------|------|
| CGA(0) | Cl(1,1) | 4 | 硬編碼 |
| CGA(1) | Cl(2,1) | 8 | 硬編碼 |
| CGA(2) | Cl(3,1) | 16 | 硬編碼 |
| CGA(3) | Cl(4,1) | 32 | 硬編碼 |
| CGA(4) | Cl(5,1) | 64 | 硬編碼 |
| CGA(5) | Cl(6,1) | 128 | 硬編碼 |
| CGA(6+) | Cl(n+1,1) | 2^(n+2) | 運行時 |
