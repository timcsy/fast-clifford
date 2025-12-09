# Quick Start: CGA Extended Operations

## 安裝

```bash
git clone https://github.com/timcsy/fast-clifford.git
cd fast-clifford
git checkout 005-cga-extended-ops
uv sync
```

## EvenVersor Composition (Versor 組合)

組合多個幾何變換為單一 Versor：

```python
import torch
from fast_clifford import CGA

# 建立 CGA3D 代數
cga = CGA(3)

# 建立兩個 EvenVersor（例如：旋轉 + 平移）
# EvenVersor: 16 分量 for CGA3D
versor_rotation = torch.zeros(1, 16)
versor_rotation[0, 0] = 0.707  # cos(45deg/2)
versor_rotation[0, 1] = 0.707  # sin(45deg/2) * e12 分量

versor_translation = torch.zeros(1, 16)
versor_translation[0, 0] = 1.0
versor_translation[0, 10] = 0.5  # e+- 分量（平移）

# 組合 Versor
versor_combined = cga.compose_even_versor(versor_rotation, versor_translation)
print(f"Combined versor shape: {versor_combined.shape}")  # (1, 16)

# 應用組合變換
point = cga.upgc_encode(torch.tensor([[1.0, 0.0, 0.0]]))
transformed = cga.sandwich_product_sparse(versor_combined, point)
result = cga.upgc_decode(transformed)
print(f"Transformed point: {result}")
```

## Similitude 加速

使用 Similitude 專用函式獲得 30-40% 加速：

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# Similitude = 平移 + 旋轉 + 縮放（排除 transversion）
# 使用相同的 EvenVersor 格式，但計算更快
similitude1 = torch.randn(1, 16)  # 假設是有效的 Similitude
similitude2 = torch.randn(1, 16)

# 使用 Similitude 專用組合（更快）
result = cga.compose_similitude(similitude1, similitude2)
print(f"Similitude compose: {result.shape}")

# 或使用一般 EvenVersor 組合（較慢但更通用）
result_general = cga.compose_even_versor(similitude1, similitude2)
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

從 Bivector 生成旋轉 Versor：

```python
import torch
import math
from fast_clifford import CGA

cga = CGA(3)

# 建立 Bivector（代表 90deg 繞 e12 平面旋轉）
# CGA3D Bivector: 10 分量
# 索引 0 = e12 分量
angle = math.pi / 2  # 90 度
B = torch.zeros(1, 10)
B[0, 0] = angle / 2  # e12 分量（除以 2 因為 sandwich product 會乘兩次）

# 生成旋轉 Versor
versor = cga.exp_bivector(B)
print(f"Versor shape: {versor.shape}")  # (1, 16)
print(f"Versor scalar (cos(theta/2)): {versor[0, 0]}")  # ~ 0.707

# 應用旋轉
point = cga.upgc_encode(torch.tensor([[1.0, 0.0, 0.0]]))
rotated = cga.sandwich_product_sparse(versor, point)
result = cga.upgc_decode(rotated)
print(f"Rotated point: {result}")  # 應為 (0, 1, 0)
```

## Outer Product (楔積)

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 兩個 Grade-1 向量
e1 = torch.zeros(1, 32)
e1[0, 1] = 1.0
e2 = torch.zeros(1, 32)
e2[0, 2] = 1.0

# 楔積產生 Grade-2 bivector
e12 = cga.outer_product(e1, e2)
print(f"e1 ^ e2 = bivector at index 6")

# 反對稱性: e2 ^ e1 = -e12
e21 = cga.outer_product(e2, e1)
print(f"e2 ^ e1 = -{e12}")
```

## Left/Right Contraction (縮併)

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 建立 e1 和 e12
e1 = torch.zeros(1, 32)
e1[0, 1] = 1.0
e12 = torch.zeros(1, 32)
e12[0, 6] = 1.0

# 左縮併: e1 ⌋ e12 = e2
result = cga.left_contraction(e1, e12)
print(f"e1 ⌋ e12 = e2")

# Grade 規則: Grade(e1)=1, Grade(e12)=2 -> Grade(result) = 2-1 = 1
```

## Grade Selection

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 混合 Grade 多向量
mv = torch.randn(1, 32)

# 提取 Grade 2 分量（bivectors）
grade2 = cga.grade_select(mv, 2)
print(f"Grade 2 components: {grade2.shape}")  # (1, 10)

# 提取 Grade 0 分量（scalar）
scalar = cga.grade_select(mv, 0)
print(f"Scalar: {scalar}")  # (1, 1)
```

## Dual (對偶)

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 建立 e1
e1 = torch.zeros(1, 32)
e1[0, 1] = 1.0

# 計算對偶
dual_e1 = cga.dual(e1)
print(f"dual(e1) = e2345 (Grade 4)")  # Grade 1 -> Grade 4
```

## Normalize (正規化)

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 非單位向量
v = torch.zeros(1, 32)
v[0, 1] = 3.0  # e1
v[0, 2] = 4.0  # e2

# 正規化
v_unit = cga.normalize(v)
norm = cga.inner_product(v_unit, v_unit)
print(f"Normalized norm: {norm}")  # 應為 1.0

# 零向量處理（無 NaN）
zero = torch.zeros(1, 32)
zero_normalized = cga.normalize(zero)
print(f"Zero normalized: {zero_normalized}")  # 保持為零
```

## TRS 轉換 (Translation-Rotation-Scaling)

從熟悉的 TRS 參數建立 Similitude：

```python
import torch
import math
from fast_clifford import CGA

cga = CGA(3)

# === 從 TRS 建立 Similitude ===

# 平移向量
translation = torch.tensor([[1.0, 2.0, 3.0]])

# 旋轉（四元數 [w, x, y, z]，繞 z 軸旋轉 90 度）
angle = math.pi / 2
rotation = torch.tensor([[
    math.cos(angle / 2),  # w
    0.0,                   # x
    0.0,                   # y
    math.sin(angle / 2)    # z
]])

# 縮放因子
scale = torch.tensor([[1.5]])

# 建立 Similitude
similitude = cga.from_trs(translation, rotation, scale, rotation_format='quaternion')
print(f"Similitude shape: {similitude.shape}")  # (1, 16)

# === 從 Similitude 提取 TRS ===

t_out, r_out, s_out = cga.to_trs(similitude, rotation_format='quaternion')
print(f"Translation: {t_out}")  # [[1.0, 2.0, 3.0]]
print(f"Rotation (quat): {r_out}")  # [[0.707, 0, 0, 0.707]]
print(f"Scale: {s_out}")  # [[1.5]]
```

### 2D 旋轉（角度格式）

```python
cga2d = CGA(2)

# 2D 使用角度而非四元數
translation_2d = torch.tensor([[1.0, 2.0]])
angle_2d = torch.tensor([[math.pi / 4]])  # 45 度
scale_2d = torch.tensor([[2.0]])

similitude_2d = cga2d.from_trs(
    translation_2d, angle_2d, scale_2d,
    rotation_format='angle'
)

# 提取
t, r, s = cga2d.to_trs(similitude_2d, rotation_format='angle')
print(f"Angle: {r}")  # [[0.785...]] (π/4)
```

### Euler 角格式

```python
cga = CGA(3)

# Euler 角 [roll, pitch, yaw]
translation = torch.tensor([[0.0, 0.0, 0.0]])
euler = torch.tensor([[0.0, 0.0, math.pi / 2]])  # 僅 yaw 90 度
scale = torch.tensor([[1.0]])

similitude = cga.from_trs(translation, euler, scale, rotation_format='euler')

# 提取為四元數
t, quat, s = cga.to_trs(similitude, rotation_format='quaternion')
print(f"Quaternion: {quat}")
```

### 單獨建立各分量

```python
cga = CGA(3)

# 只有平移
T = cga.make_translation(torch.tensor([[1.0, 2.0, 3.0]]))

# 只有旋轉
R = cga.make_rotation(
    torch.tensor([[0.707, 0.0, 0.0, 0.707]]),
    rotation_format='quaternion'
)

# 只有縮放
D = cga.make_dilation(torch.tensor([[2.0]]))

# 手動組合 (順序: 先縮放、再旋轉、最後平移)
RD = cga.compose_similitude(R, D)
TRD = cga.compose_similitude(T, RD)

# 等價於 from_trs
similitude = cga.from_trs(
    torch.tensor([[1.0, 2.0, 3.0]]),
    torch.tensor([[0.707, 0.0, 0.0, 0.707]]),
    torch.tensor([[2.0]])
)
```

### 批次 TRS 轉換

```python
cga = CGA(3)
batch_size = 1024

# 批次 TRS
translations = torch.randn(batch_size, 3)
rotations = torch.randn(batch_size, 4)
rotations = rotations / rotations.norm(dim=-1, keepdim=True)  # 正規化四元數
scales = torch.rand(batch_size, 1) + 0.5  # 0.5 ~ 1.5

# 批次建立
similitudes = cga.from_trs(translations, rotations, scales)
print(f"Batch similitudes: {similitudes.shape}")  # (1024, 16)

# 批次提取
t_batch, r_batch, s_batch = cga.to_trs(similitudes)
```

---

## Multivector 運算子重載

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 使用 Multivector 包裝類別
v1 = cga.even_versor(torch.randn(1, 16))
v2 = cga.even_versor(torch.randn(1, 16))

# 運算子重載
v3 = v1 * v2     # 幾何積 / compose
a = cga.multivector(torch.randn(1, 32))
b = cga.multivector(torch.randn(1, 32))

wedge = a ^ b    # 楔積
inner = a | b    # 內積 (返回 Tensor)
lc = a << b      # 左縮併
rc = a >> b      # 右縮併

point = cga.point(torch.tensor([[1.0, 2.0, 3.0]]))
transformed = v1 @ point  # 三明治積

rev = ~v1        # 反向
inv = v1 ** -1   # 逆元

# Bivector 指數映射
bv = cga.bivector(torch.randn(1, 10))
rotor = bv.exp()  # 返回 EvenVersor
```

## 批次處理

所有操作支援任意 batch 維度：

```python
import torch
from fast_clifford import CGA

cga = CGA(3)
batch_size = 1024

# 批次 EvenVersor Composition
versors_a = torch.randn(batch_size, 16)
versors_b = torch.randn(batch_size, 16)
versors_combined = cga.compose_even_versor(versors_a, versors_b)
print(f"Batch compose: {versors_combined.shape}")  # (1024, 16)

# 批次 Inner Product
mvs_a = torch.randn(batch_size, 32)
mvs_b = torch.randn(batch_size, 32)
scores = cga.inner_product(mvs_a, mvs_b)
print(f"Batch inner product: {scores.shape}")  # (1024, 1)

# 批次 Exponential Map
bivectors = torch.randn(batch_size, 10) * 0.1  # 小角度
versors = cga.exp_bivector(bivectors)
print(f"Batch exp_bivector: {versors.shape}")  # (1024, 16)
```

## 高維度支援 (6D+)

相同 API 適用於 6D 及以上：

```python
import torch
from fast_clifford import CGA

# CGA6D (運行時實作)
cga6d = CGA(6)
print(f"CGA6D even_versor count: {cga6d.even_versor_count}")  # 128

# 相同 API
v1 = torch.randn(1, cga6d.even_versor_count)
v2 = torch.randn(1, cga6d.even_versor_count)
v_combined = cga6d.compose_even_versor(v1, v2)
print(f"CGA6D compose: {v_combined.shape}")
```

## 統一 Layer 使用

所有維度使用相同的 Layer 類別名稱：

```python
import torch
from fast_clifford import CGA, CliffordTransformLayer, CGAEncoder, CGADecoder, CGAPipeline

# 建立代數與統一 Layer
cga = CGA(3)
transform_layer = CliffordTransformLayer(dim=3)  # 統一名稱
encoder = CGAEncoder(dim=3)
decoder = CGADecoder(dim=3)
pipeline = CGAPipeline(dim=3)

# 或透過代數實例取得
transform_layer = cga.get_transform_layer()  # 取代 get_care_layer()

# 使用範例
versor = torch.randn(1, cga.even_versor_count)
x = torch.tensor([[1.0, 2.0, 3.0]])

# 完整管線
y = pipeline(versor, x)
print(f"Transformed point: {y}")

# 分步操作
point = encoder(x)
transformed_point = transform_layer(versor, point)
result = decoder(transformed_point)
```

## ONNX 匯出

所有硬編碼操作可匯出為 ONNX：

```python
import torch
from fast_clifford.algebras.cga3d import compose_even_versor_sparse

class EvenVersorComposeModel(torch.nn.Module):
    def forward(self, v1, v2):
        return compose_even_versor_sparse(v1, v2)

model = EvenVersorComposeModel()
v1 = torch.randn(1, 16)
v2 = torch.randn(1, 16)

torch.onnx.export(
    model,
    (v1, v2),
    "even_versor_compose.onnx",
    input_names=["v1", "v2"],
    output_names=["result"],
    dynamic_axes={
        "v1": {0: "batch"},
        "v2": {0: "batch"},
        "result": {0: "batch"},
    },
    opset_version=17,
)
print("Exported to even_versor_compose.onnx")
```

## Structure Normalize (結構正規化)

訓練 Similitude 時維持幾何約束：

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# === 基本 Structure Normalize ===

# 假設神經網路輸出 Similitude 參數（可能違反幾何約束）
raw_similitude = torch.randn(1, 16)

# 正規化：1. Rotor 成為單位四元數 2. 強制 ei+ = ei-
clean_similitude = cga.structure_normalize(raw_similitude)
print(f"Normalized similitude: {clean_similitude.shape}")

# 驗證 Rotor 為單位長度
rotor_part = clean_similitude[..., cga.ROTOR_INDICES]  # scalar, e12, e13, e23
rotor_norm = torch.norm(rotor_part, dim=-1)
print(f"Rotor norm (should be 1.0): {rotor_norm}")

# 驗證 Similitude 約束
for plus_idx, minus_idx in cga.TRANSLATION_PAIRS:
    diff = (clean_similitude[..., plus_idx] - clean_similitude[..., minus_idx]).abs()
    print(f"ei+ - ei- at ({plus_idx}, {minus_idx}): {diff}")  # Should be 0
```

### Soft Structure Normalize (訓練時使用)

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 訓練期間使用 soft normalize，漸進施加約束
raw_params = torch.randn(batch_size, 16, requires_grad=True)

# strength 從 0.0 逐步增加到 1.0
for epoch in range(100):
    strength = min(1.0, epoch / 50)  # 50 epochs 後完全強制

    normalized = cga.soft_structure_normalize(raw_params, strength=strength)

    # 繼續使用 normalized 進行前向傳播
    # loss = ...
    # loss.backward()  # 梯度流經原始參數
```

### Structure Normalize STE (硬約束 + 梯度流)

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# 使用 Straight-Through Estimator：
# 前向：完全正規化（硬約束）
# 反向：梯度直接流過（忽略正規化操作）
raw_params = torch.randn(batch_size, 16, requires_grad=True)

normalized = cga.structure_normalize_ste(raw_params)

# 前向時 normalized 滿足完美約束
# 反向時 raw_params 收到完整梯度（就像沒有正規化一樣）
```

### 作為正則化損失使用

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

def similitude_constraint_loss(params):
    """懲罰違反 Similitude 約束的參數"""
    normalized = cga.structure_normalize(params)
    return torch.nn.functional.mse_loss(params, normalized.detach())

# 在訓練中
raw_params = model_output  # 神經網路輸出
main_loss = compute_main_loss(raw_params)
reg_loss = similitude_constraint_loss(raw_params)

total_loss = main_loss + 0.1 * reg_loss  # λ = 0.1 正則化權重
total_loss.backward()
```

### 多維度支援

```python
from fast_clifford import CGA

# 各維度的 Rotor 索引和平移對
for dim in [1, 2, 3, 4, 5]:
    cga = CGA(dim)
    print(f"CGA{dim}D:")
    print(f"  ROTOR_INDICES: {cga.ROTOR_INDICES}")
    print(f"  TRANSLATION_PAIRS: {cga.TRANSLATION_PAIRS}")
    print(f"  DILATION_INDEX: {cga.DILATION_INDEX}")

# CGA3D 輸出:
#   ROTOR_INDICES: (0, 1, 2, 5)  # scalar, e12, e13, e23
#   TRANSLATION_PAIRS: [(3, 4), (6, 7), (8, 9)]  # (e1+, e1-), (e2+, e2-)...
#   DILATION_INDEX: 10  # e+-
```

---

## 驗證測試

```bash
# 執行 Extended Operations 測試
uv run pytest fast_clifford/tests/test_compose.py -v
uv run pytest fast_clifford/tests/test_inner_product.py -v
uv run pytest fast_clifford/tests/test_exp_bivector.py -v
uv run pytest fast_clifford/tests/test_outer_product.py -v
uv run pytest fast_clifford/tests/test_contractions.py -v
uv run pytest fast_clifford/tests/test_grade_select.py -v
uv run pytest fast_clifford/tests/test_dual.py -v
uv run pytest fast_clifford/tests/test_normalize.py -v
uv run pytest fast_clifford/tests/test_structure_normalize.py -v
uv run pytest fast_clifford/tests/test_operators.py -v
uv run pytest fast_clifford/tests/test_unified_layers.py -v
```
