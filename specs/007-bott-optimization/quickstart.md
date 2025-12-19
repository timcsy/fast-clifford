# 快速入門：Bott 週期性優化

**功能**: 007-bott-optimization
**日期**: 2025-12-19

## 概述

此功能優化 Clifford 代數函式庫的儲存空間和效能：
- 儲存減少 95%（600MB → 25MB）
- Bott 運算加速 10x+
- 完整支援所有維度的 Clifford 代數

## 基本使用

### 建立代數（無變化）

```python
from fast_clifford import Cl, VGA, CGA

# 常用代數（p+q < 8）- 使用預生成模組
vga3 = VGA(3)           # Cl(3, 0) - 3D 向量代數
cga3 = CGA(3)           # Cl(4, 1) - 3D 共形代數
spacetime = Cl(1, 3)    # Cl(1, 3) - 時空代數（p < q）

# 高維代數（p+q >= 8）- 使用 Bott 週期性
cl10 = Cl(10, 0)        # 自動使用 Bott 分解
cl17 = Cl(17, 0)        # 兩次 Bott 週期
```

### 對稱代數（新功能）

當 p < q 時，系統透明地使用對稱映射：

```python
# 這兩個代數內部共用相同的硬編碼實作
cl13 = Cl(1, 3)  # 自動映射到 Cl(3, 1)
cl31 = Cl(3, 1)  # 直接使用硬編碼

# 運算結果數學上等價
import torch
a = torch.randn(8)
b = torch.randn(8)

# 對應的 blade 索引不同，但代數結構相同
```

### 高維代數（Bott 週期性）

```python
# 建立高維代數
cl10 = Cl(10, 0)

# 查看分解資訊
print(f"基底代數: Cl({cl10.base_algebra.p}, {cl10.base_algebra.q})")
print(f"矩陣大小: {cl10.matrix_size}")
print(f"總 blade 數: {cl10.count_blade}")

# 輸出:
# 基底代數: Cl(2, 0)
# 矩陣大小: 16
# 總 blade 數: 1024

# 執行運算（張量加速）
a = torch.randn(1024)
b = torch.randn(1024)
c = cl10.geometric_product(a, b)  # 使用 einsum，快 10x+
```

### 多重 Bott 週期

```python
# 非常高維的代數
cl17 = Cl(17, 0)

print(f"基底代數: Cl({cl17.base_algebra.p}, {cl17.base_algebra.q})")
print(f"週期數: {cl17.periods}")
print(f"矩陣大小: {cl17.matrix_size}")

# 輸出:
# 基底代數: Cl(1, 0)
# 週期數: 2
# 矩陣大小: 256
```

## 效能比較

```python
import torch
import time

# 建立代數
cl10 = Cl(10, 0)

# 準備資料
a = torch.randn(100, 1024)
b = torch.randn(100, 1024)

# 批次幾何積（張量加速）
start = time.time()
for _ in range(100):
    c = cl10.geometric_product(a, b)
elapsed = time.time() - start

print(f"每次運算: {elapsed/100*1000:.2f} ms")
# 預期: < 5ms（相比舊版 ~50ms）
```

## 路由邏輯

```
Cl(p, q) 請求
    │
    ▼
p + q < 8 ?
    │
   是 ──────────────────────────────────────► 否
    │                                          │
    ▼                                          ▼
p >= q ?                              BottPeriodicityAlgebra
    │                                  (張量化 einsum)
   是                 否
    │                  │
    ▼                  ▼
HardcodedClWrapper   SymmetricClWrapper
(預生成模組)          (索引重排)
```

## 常見問題

### Q: 為什麼 Cl(0,4) 回傳不同類型的物件？

A: 因為 p < q，系統使用 SymmetricClWrapper 包裝 Cl(4,0)。功能完全相同，只是內部實作不同。

### Q: 如何知道使用了哪種實作？

```python
algebra = Cl(10, 0)
print(type(algebra).__name__)  # BottPeriodicityAlgebra

algebra = Cl(1, 3)
print(type(algebra).__name__)  # SymmetricClWrapper

algebra = Cl(3, 0)
print(type(algebra).__name__)  # HardcodedClWrapper (via VGAWrapper)
```

### Q: ONNX 匯出是否支援？

A: 是的。張量化 Bott 運算使用 `torch.einsum`，可匯出 ONNX 且無 Loop 節點。

```python
import torch.onnx

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.algebra = Cl(10, 0)

    def forward(self, a, b):
        return self.algebra.geometric_product(a, b)

model = MyModel()
torch.onnx.export(model, (a, b), "model.onnx", opset_version=17)
```
