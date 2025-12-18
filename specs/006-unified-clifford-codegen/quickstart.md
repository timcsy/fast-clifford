# Quickstart: Unified Clifford Algebra System

**Feature**: 006-unified-clifford-codegen
**Date**: 2025-12-18

## 安裝

```bash
git clone https://github.com/timcsy/fast-clifford.git
cd fast-clifford
uv sync
```

## 基本用法

### 1. 建立代數

```python
from fast_clifford import Cl, VGA, CGA, PGA

# 通用工廠
algebra = Cl(4, 1)           # Cl(4,1) = CGA3D
algebra = Cl(3, 0)           # Cl(3,0) = VGA3D
algebra = Cl(2, 2)           # Cl(2,2) - 一般代數

# 便捷工廠
vga3d = VGA(3)               # VGA(3) = Cl(3,0)
cga3d = CGA(3)               # CGA(3) = Cl(4,1)
pga3d = PGA(3)               # PGA(3) = Cl(3,0,1)
```

### 2. 檢查代數屬性

```python
cga = CGA(3)

print(f"Type: {cga.algebra_type}")      # 'cga'
print(f"Signature: Cl({cga.p},{cga.q},{cga.r})")  # Cl(4,1,0)
print(f"Blades: {cga.count_blade}")     # 32
print(f"Rotors: {cga.count_rotor}")     # 16
print(f"Bivectors: {cga.count_bivector}")  # 10
```

### 3. 幾何積和其他積運算

```python
import torch
from fast_clifford import Cl

algebra = Cl(3, 0)  # VGA3D

a = torch.randn(32, algebra.count_blade)  # batch of 32
b = torch.randn(32, algebra.count_blade)

# 積運算
gp = algebra.geometric_product(a, b)  # 幾何積
op = algebra.outer(a, b)              # 外積 a ∧ b
ip = algebra.inner(a, b)              # 內積 <ab>₀
lc = algebra.contract_left(a, b)      # 左縮並 a ⌋ b
rc = algebra.contract_right(a, b)     # 右縮並 a ⌊ b
```

### 4. 單元運算

```python
mv = torch.randn(algebra.count_blade)

rev = algebra.reverse(mv)         # 反轉 m̃
inv = algebra.involute(mv)        # Grade 反演 m̂
conj = algebra.conjugate(mv)      # Clifford 共軛 m†
dual = algebra.dual(mv)           # Poincaré 對偶
norm = algebra.normalize(mv)      # 正規化
inverse = algebra.inverse(mv)     # 乘法逆元
```

### 5. Rotor 加速運算

```python
from fast_clifford import CGA
import torch

cga = CGA(3)

# Rotor 表示（16 個分量）
r1 = torch.randn(cga.count_rotor)
r2 = torch.randn(cga.count_rotor)

# 組合（比 geometric_product 快）
composed = cga.compose_rotor(r1, r2)

# Sandwich product（比通用版本快 8x）
point = cga.encode(torch.tensor([1.0, 2.0, 3.0]))
transformed = cga.sandwich_rotor(r1, point)

# Rotor 正規化
normalized = cga.normalize_rotor(r1)
```

### 6. Bivector 指數映射

```python
# Bivector → Rotor
bivector = torch.randn(cga.count_bivector)  # 10 個分量
rotor = cga.exp_bivector(bivector)          # 16 個分量

# Rotor → Bivector
bivector_back = cga.log_rotor(rotor)

# 球面線性插值
r1 = cga.exp_bivector(torch.randn(10))
r2 = cga.exp_bivector(torch.randn(10))
r_mid = cga.slerp_rotor(r1, r2, 0.5)
```

### 7. CGA 編解碼

```python
from fast_clifford import CGA
import torch

cga = CGA(3)  # 3D 共形幾何代數

# 歐幾里得座標
euclidean = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])

# 編碼為 CGA 點（完整 multivector 表示）
cga_points = cga.encode(euclidean)  # shape: (2, 32)

# 應用變換
rotor = cga.exp_bivector(torch.randn(10))
transformed = cga.sandwich_rotor(rotor, cga_points)

# 解碼回歐幾里得
result = cga.decode(transformed)  # shape: (2, 3)
```

### 8. 運算子重載

```python
from fast_clifford import CGA
import torch

cga = CGA(3)

# Multivector 包裝
a = cga.multivector(torch.randn(32))
b = cga.multivector(torch.randn(32))

# 積運算
c = a * b      # 幾何積
c = a ^ b      # 外積 (wedge)
c = a | b      # 內積 (scalar product)
c = a << b     # 左縮並
c = a >> b     # 右縮並
c = a & b      # meet (regressive product)

# 三明治積
point = cga.multivector(torch.randn(32))
transformed = a @ point  # a * point * ~a

# 單元運算
c = ~a         # 反轉
c = a ** -1    # 逆元
c = a ** 2     # 平方（兩次幾何積）

# Grade 選取
scalar = a(0)      # 提取 grade-0（純量）
bivector = a(2)    # 提取 grade-2（bivector）

# Rotor 包裝
r1 = cga.rotor(torch.randn(16))
r2 = cga.rotor(torch.randn(16))

r3 = r1 * r2          # 組合（使用 compose_rotor）
cga_point = torch.randn(5)
result = r1 @ cga_point  # sandwich product（加速版本）
```

### 9. PyTorch Layers

```python
from fast_clifford import CGA
import torch.nn as nn

cga = CGA(3)

class GeometricNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotor_net = nn.Linear(64, cga.count_rotor)
        self.transform = cga.get_transform_layer()
        self.encoder = cga.get_encoder()
        self.decoder = cga.get_decoder()

    def forward(self, features, points):
        # features: (batch, 64)
        # points: (batch, 3) 歐幾里得座標

        rotors = self.rotor_net(features)
        cga_points = self.encoder(points)
        transformed = self.transform(rotors, cga_points)
        return self.decoder(transformed)
```

### 10. ONNX 導出

```python
import torch
from fast_clifford import CGA

cga = CGA(3)
layer = cga.get_transform_layer()

# 範例輸入
rotor = torch.randn(1, 16)
point = torch.randn(1, 32)  # 完整 multivector 表示

# 導出
torch.onnx.export(
    layer,
    (rotor, point),
    "transform.onnx",
    input_names=["rotor", "point"],
    output_names=["output"],
    dynamic_axes={
        "rotor": {0: "batch"},
        "point": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=17,
)

# 驗證無 Loop 節點
import onnx
model = onnx.load("transform.onnx")
loops = [n for n in model.graph.node if n.op_type == "Loop"]
assert len(loops) == 0, f"Found {len(loops)} Loop nodes!"
```

### 11. 高維度代數（Bott 週期性）

```python
from fast_clifford import Cl

# 高維度代數自動使用 Bott 週期性
high_dim = Cl(10, 0)  # Cl(10,0) - 1024 blades

# 所有運算照常可用
a = torch.randn(high_dim.count_blade)
b = torch.randn(high_dim.count_blade)
c = high_dim.geometric_product(a, b)

# 系統會輸出警告（如果 blade_count > 16384）
very_high = Cl(15, 0)  # 會顯示記憶體警告
```

### 12. VGA 用法

```python
from fast_clifford import VGA
import torch

vga = VGA(3)  # 3D 向量代數

# 向量
v1 = torch.tensor([1.0, 0.0, 0.0])
v2 = torch.tensor([0.0, 1.0, 0.0])

# 嵌入為 Grade-1
mv1 = vga.encode(v1)
mv2 = vga.encode(v2)

# 外積 → bivector
bivector = vga.outer(mv1, mv2)

# Rotor（旋轉）
angle = torch.tensor([0.5])  # radians
axis_bivector = bivector * (angle / 2)
rotor = vga.exp_bivector(axis_bivector)

# 應用旋轉
rotated = vga.sandwich_rotor(rotor, mv1)
```

### 13. PGA 用法

```python
from fast_clifford import PGA
import torch

pga = PGA(3)  # 3D 投影幾何代數

# PGA 運算透過 CGA 嵌入執行
a = torch.randn(pga.count_blade)
b = torch.randn(pga.count_blade)

# 幾何積（內部：嵌入 → CGA 運算 → 投影）
c = pga.geometric_product(a, b)
```

## API 速查表

### 工廠函數

| 函數 | 說明 |
|------|------|
| `Cl(p, q, r=0)` | 任意 Clifford 代數 |
| `VGA(n)` | VGA(n) = Cl(n, 0) |
| `CGA(n)` | CGA(n) = Cl(n+1, 1) |
| `PGA(n)` | PGA(n) = Cl(n, 0, 1) |

### 屬性

| 屬性 | 說明 |
|------|------|
| `count_blade` | 總 blade 數 |
| `count_rotor` | Rotor 分量數 |
| `count_bivector` | Bivector 分量數 |
| `algebra_type` | 'vga', 'cga', 'pga', 'general' |

### 積運算

| 方法 | 說明 |
|------|------|
| `geometric_product(a, b)` | 幾何積 |
| `inner(a, b)` | 內積 |
| `outer(a, b)` | 外積 |
| `contract_left(a, b)` | 左縮並 |
| `contract_right(a, b)` | 右縮並 |
| `sandwich(v, x)` | 三明治積 |

### 單元運算

| 方法 | 說明 |
|------|------|
| `reverse(mv)` | 反轉 |
| `involute(mv)` | Grade 反演 |
| `conjugate(mv)` | Clifford 共軛 |
| `dual(mv)` | Poincaré 對偶 |
| `normalize(mv)` | 正規化 |
| `inverse(mv)` | 逆元 |

### Rotor 加速

| 方法 | 說明 |
|------|------|
| `compose_rotor(r1, r2)` | Rotor 組合 |
| `reverse_rotor(r)` | Rotor 反轉 |
| `sandwich_rotor(r, x)` | Rotor sandwich |
| `exp_bivector(B)` | exp(B) |
| `log_rotor(r)` | log(r) |
| `slerp_rotor(r1, r2, t)` | 球面插值 |

### 運算子（完整對照表）

| 運算子 | Multivector | Rotor | 說明 |
|--------|-------------|-------|------|
| `a * b` | ✅ 幾何積 | ✅ 組合 | 核心運算 |
| `a ^ b` | ✅ 外積 | ❌ | wedge product |
| `a \| b` | ✅ 內積 | ❌ | scalar product |
| `a << b` | ✅ 左縮並 | ❌ | left contraction |
| `a >> b` | ✅ 右縮並 | ❌ | right contraction |
| `m @ x` | ✅ 三明治積 | ✅ 三明治積 | sandwich product |
| `a & b` | ✅ meet | ❌ | regressive product |
| `~a` | ✅ 反轉 | ✅ 反轉 | reversion |
| `a ** -1` | ✅ 逆元 | ✅ 逆元 | inverse |
| `a ** n` | ✅ 冪次 | ✅ 冪次 | power |
| `mv(k)` | ✅ grade 選取 | ❌ | select grade-k |
| `s * a` | ✅ 純量乘 | ✅ 純量乘 | scalar multiply |
| `a / s` | ✅ 純量除 | ✅ 純量除 | scalar divide |
| `a + b` | ✅ 加法 | ✅ 加法 | addition |
| `a - b` | ✅ 減法 | ✅ 減法 | subtraction |
| `-a` | ✅ 負號 | ✅ 負號 | negation |
