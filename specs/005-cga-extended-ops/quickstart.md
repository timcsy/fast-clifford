# Quick Start: CGA Extended Operations

## 安裝

```bash
git clone https://github.com/timcsy/fast-clifford.git
cd fast-clifford
git checkout 005-cga-extended-ops
uv sync
```

## Motor Composition (馬達組合)

組合多個幾何變換為單一馬達：

```python
import torch
from fast_clifford import CGA

# 建立 CGA3D 代數
cga = CGA(3)

# 建立兩個馬達（例如：旋轉 + 平移）
# Motor: 16 分量 for CGA3D
motor_rotation = torch.zeros(1, 16)
motor_rotation[0, 0] = 0.707  # cos(45°/2)
motor_rotation[0, 1] = 0.707  # sin(45°/2) * e12 分量

motor_translation = torch.zeros(1, 16)
motor_translation[0, 0] = 1.0
motor_translation[0, 10] = 0.5  # e+- 分量（平移）

# 組合馬達
motor_combined = cga.motor_compose(motor_rotation, motor_translation)
print(f"Combined motor shape: {motor_combined.shape}")  # (1, 16)

# 應用組合變換
point = cga.upgc_encode(torch.tensor([[1.0, 0.0, 0.0]]))
transformed = cga.sandwich_product_sparse(motor_combined, point)
result = cga.upgc_decode(transformed)
print(f"Transformed point: {result}")
```

## Geometric Inner Product (幾何內積)

計算多向量的度規內積，用於 Attention Score：

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 建立兩個完整多向量（32 分量 for CGA3D）
a = torch.randn(1, 32)
b = torch.randn(1, 32)

# 計算內積
score = cga.inner_product(a, b)
print(f"Inner product: {score}")  # shape: (1, 1)

# 驗證 Null Basis 性質
eo = torch.zeros(1, 32)
einf = torch.zeros(1, 32)
# eo = 0.5 * (e- - e+)
eo[0, 4] = -0.5  # e+
eo[0, 5] = 0.5   # e-
# einf = e- + e+
einf[0, 4] = 1.0  # e+
einf[0, 5] = 1.0  # e-

inner = cga.inner_product(eo, einf)
print(f"inner_product(eo, einf) = {inner}")  # 應為 -1
```

## Exponential Map (指數映射)

從 Bivector 生成旋轉馬達：

```python
import torch
import math
from fast_clifford import CGA

cga = CGA(3)

# 建立 Bivector（代表 90° 繞 e12 平面旋轉）
# CGA3D Bivector: 10 分量
# 索引 0 = e12 分量
angle = math.pi / 2  # 90 度
B = torch.zeros(1, 10)
B[0, 0] = angle / 2  # e12 分量（除以 2 因為 sandwich product 會乘兩次）

# 生成旋轉馬達
motor = cga.exp_bivector(B)
print(f"Motor shape: {motor.shape}")  # (1, 16)
print(f"Motor scalar (cos(θ/2)): {motor[0, 0]}")  # ≈ 0.707

# 應用旋轉
point = cga.upgc_encode(torch.tensor([[1.0, 0.0, 0.0]]))
rotated = cga.sandwich_product_sparse(motor, point)
result = cga.upgc_decode(rotated)
print(f"Rotated point: {result}")  # 應為 (0, 1, 0)
```

## 批次處理

所有操作支援任意 batch 維度：

```python
import torch
from fast_clifford import CGA

cga = CGA(3)
batch_size = 1024

# 批次 Motor Composition
motors_a = torch.randn(batch_size, 16)
motors_b = torch.randn(batch_size, 16)
motors_combined = cga.motor_compose(motors_a, motors_b)
print(f"Batch motor compose: {motors_combined.shape}")  # (1024, 16)

# 批次 Inner Product
mvs_a = torch.randn(batch_size, 32)
mvs_b = torch.randn(batch_size, 32)
scores = cga.inner_product(mvs_a, mvs_b)
print(f"Batch inner product: {scores.shape}")  # (1024, 1)

# 批次 Exponential Map
bivectors = torch.randn(batch_size, 10) * 0.1  # 小角度
motors = cga.exp_bivector(bivectors)
print(f"Batch exp_bivector: {motors.shape}")  # (1024, 16)
```

## 高維度支援 (6D+)

相同 API 適用於 6D 及以上：

```python
import torch
from fast_clifford import CGA

# CGA6D (運行時實作)
cga6d = CGA(6)
print(f"CGA6D motor count: {cga6d.motor_count}")  # 127

# 相同 API
m1 = torch.randn(1, cga6d.motor_count)
m2 = torch.randn(1, cga6d.motor_count)
m_combined = cga6d.motor_compose(m1, m2)
print(f"CGA6D motor compose: {m_combined.shape}")
```

## ONNX 匯出

所有硬編碼操作可匯出為 ONNX：

```python
import torch
from fast_clifford.algebras.cga3d import motor_compose_sparse

class MotorComposeModel(torch.nn.Module):
    def forward(self, m1, m2):
        return motor_compose_sparse(m1, m2)

model = MotorComposeModel()
m1 = torch.randn(1, 16)
m2 = torch.randn(1, 16)

torch.onnx.export(
    model,
    (m1, m2),
    "motor_compose.onnx",
    input_names=["m1", "m2"],
    output_names=["result"],
    dynamic_axes={
        "m1": {0: "batch"},
        "m2": {0: "batch"},
        "result": {0: "batch"},
    },
    opset_version=17,
)
print("Exported to motor_compose.onnx")
```

## 統一 Layer 使用

所有維度使用相同的 Layer 類別名稱：

```python
import torch
from fast_clifford import CGA, CGATransformLayer, CGAEncoder, CGADecoder, CGAPipeline

# 建立代數與統一 Layer
cga = CGA(3)
transform_layer = CGATransformLayer(dim=3)  # 統一名稱
encoder = CGAEncoder(dim=3)
decoder = CGADecoder(dim=3)
pipeline = CGAPipeline(dim=3)

# 或透過代數實例取得
transform_layer = cga.get_transform_layer()  # 取代 get_care_layer()

# 使用範例
motor = torch.randn(1, cga.motor_count)
x = torch.tensor([[1.0, 2.0, 3.0]])

# 完整管線
y = pipeline(motor, x)
print(f"Transformed point: {y}")

# 分步操作
point = encoder(x)
transformed_point = transform_layer(motor, point)
result = decoder(transformed_point)
```

## 驗證測試

```bash
# 執行 Extended Operations 測試
uv run pytest fast_clifford/tests/test_motor_compose.py -v
uv run pytest fast_clifford/tests/test_inner_product.py -v
uv run pytest fast_clifford/tests/test_exp_bivector.py -v
uv run pytest fast_clifford/tests/test_unified_layers.py -v
```
