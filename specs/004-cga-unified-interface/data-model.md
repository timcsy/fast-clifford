# 資料模型：CGA(n) 統一介面

**日期**：2025-12-08
**功能**：004-cga-unified-interface

## 實體定義

### CGAAlgebraBase（抽象基底類別）

定義所有 CGA 代數的統一介面。

**屬性**：

| 屬性 | 型別 | 說明 |
|------|------|------|
| `euclidean_dim` | `int` | 歐幾里得維度 n（CGA = Cl(n+1, 1, 0)）|
| `blade_count` | `int` | 總 blade 數 = 2^(n+2) |
| `point_count` | `int` | UPGC 點分量數 = n+2（Grade 1）|
| `motor_count` | `int` | Motor 分量數（Grade 0, 2, 4... 偶數級）|
| `signature` | `Tuple[int, ...]` | Clifford 簽名（+1, +1, ..., -1）|
| `clifford_notation` | `str` | Clifford 表示法，如 "Cl(4,1,0)" |

**抽象方法**：

| 方法 | 簽名 | 說明 |
|------|------|------|
| `upgc_encode` | `(x: Tensor) -> Tensor` | 歐幾里得 → UPGC 點 |
| `upgc_decode` | `(point: Tensor) -> Tensor` | UPGC 點 → 歐幾里得 |
| `geometric_product_full` | `(a: Tensor, b: Tensor) -> Tensor` | 完整幾何積 |
| `sandwich_product_sparse` | `(motor: Tensor, point: Tensor) -> Tensor` | 稀疏三明治積 M×X×M̃ |
| `reverse_full` | `(mv: Tensor) -> Tensor` | 多向量反轉 |
| `reverse_motor` | `(motor: Tensor) -> Tensor` | Motor 反轉 |
| `get_care_layer` | `() -> nn.Module` | 取得 CareLayer |
| `get_encoder` | `() -> nn.Module` | 取得 UPGC 編碼器 |
| `get_decoder` | `() -> nn.Module` | 取得 UPGC 解碼器 |
| `get_transform_pipeline` | `() -> nn.Module` | 取得完整轉換流水線 |

---

### HardcodedCGAWrapper

包裝現有 cga0d-cga5d 模組，實作 CGAAlgebraBase 介面。

**屬性**：

| 屬性 | 型別 | 說明 |
|------|------|------|
| `_module` | `ModuleType` | 被包裝的模組（如 `cga3d`）|

**實作說明**：
- 將模組函式映射至抽象介面方法
- 不修改原有模組，僅提供統一存取層

---

### RuntimeCGAAlgebra

使用運行時計算的 CGA 代數實作，支援任意維度。

**屬性**：

| 屬性 | 型別 | 說明 |
|------|------|------|
| `_initialized` | `bool` | 是否已初始化 |
| `left_idx` | `Tensor` (buffer) | 幾何積左運算元索引 |
| `right_idx` | `Tensor` (buffer) | 幾何積右運算元索引 |
| `result_idx` | `Tensor` (buffer) | 幾何積結果索引 |
| `signs` | `Tensor` (buffer) | 幾何積符號係數 |
| `point_mask` | `Tensor` (buffer) | UPGC 點的 blade 遮罩 |
| `motor_mask` | `Tensor` (buffer) | Motor 的 blade 遮罩 |

**狀態轉換**：

```
未初始化 ──首次呼叫──► 已初始化
    │                    │
    │                    ▼
    │              buffers 已註冊
    │              可進行計算
    │              可匯出 ONNX
    │                    │
    └────────────────────┘
         （不可逆）
```

---

### CGA 工廠函式

**CGA(n: int) -> CGAAlgebraBase**

| 輸入 | 輸出 |
|------|------|
| n = 0 | HardcodedCGAWrapper(cga0d) |
| n = 1 | HardcodedCGAWrapper(cga1d) |
| n = 2 | HardcodedCGAWrapper(cga2d) |
| n = 3 | HardcodedCGAWrapper(cga3d) |
| n = 4 | HardcodedCGAWrapper(cga4d) |
| n = 5 | HardcodedCGAWrapper(cga5d) |
| n ≥ 6 | RuntimeCGAAlgebra(n) |

**Cl(p: int, q: int, r: int = 0) -> CliffordAlgebraBase**

| 條件 | 輸出 |
|------|------|
| q == 1, r == 0 | CGA(p - 1)（識別為 CGA）|
| 其他 | RuntimeCliffordAlgebra(p, q, r) + 警告 |

---

## CGA0D 代數規格

### Blade 索引表

| 索引 | Blade | Grade | 二進位 |
|------|-------|-------|--------|
| 0 | 1 | 0 | 00 |
| 1 | e+ | 1 | 01 |
| 2 | e- | 1 | 10 |
| 3 | e+- | 2 | 11 |

### 分量遮罩

| 遮罩 | 索引 | 分量數 |
|------|------|--------|
| UPGC_POINT_MASK | (1, 2) | 2 |
| MOTOR_MASK | (0, 3) | 2 |

### 簽名

```
Cl(1, 1, 0)
e+² = +1
e-² = -1
```

### 幾何積表（Cayley 表）

```
     │  1    e+   e-   e+-
─────┼───────────────────────
  1  │  1    e+   e-   e+-
  e+ │  e+   1    e+-  e-
  e- │  e-  -e+-  -1   e+
 e+- │ e+-  -e-   e+  -1
```

非零乘積項（用於張量化計算）：

| i | j | k | sign |
|---|---|---|------|
| 0 | 0 | 0 | +1 |
| 0 | 1 | 1 | +1 |
| 0 | 2 | 2 | +1 |
| 0 | 3 | 3 | +1 |
| 1 | 0 | 1 | +1 |
| 1 | 1 | 0 | +1 |
| 1 | 2 | 3 | +1 |
| 1 | 3 | 2 | +1 |
| 2 | 0 | 2 | +1 |
| 2 | 1 | 3 | -1 |
| 2 | 2 | 0 | -1 |
| 2 | 3 | 1 | +1 |
| 3 | 0 | 3 | +1 |
| 3 | 1 | 2 | -1 |
| 3 | 2 | 1 | +1 |
| 3 | 3 | 0 | -1 |

### UPGC 編碼公式

對於 0D CGA，只有原點（n_o）和無窮遠點（n_inf）：

```
n_o = 0.5 * (e- - e+)     # 原點
n_inf = e+ + e-           # 無窮遠點

# 編碼（0D 沒有歐幾里得分量，直接返回 n_o）
X = n_o = 0.5 * (e- - e+)
  → e+ = -0.5, e- = 0.5

# 解碼
# 對於 0D，解碼返回空張量 shape (..., 0)
```

### 三明治積稀疏性

Motor 分量：`[scalar, e+-]` (2 分量)
Point 分量：`[e+, e-]` (2 分量)

三明治積 M × X × M̃ 的結果仍為 Point 分量。

---

## 代數維度對照表

| 代數 | 簽名 | Blade 數 | Point | Motor |
|------|------|----------|-------|-------|
| CGA0D | Cl(1,1,0) | 4 | 2 | 2 |
| CGA1D | Cl(2,1,0) | 8 | 3 | 4 |
| CGA2D | Cl(3,1,0) | 16 | 4 | 7 |
| CGA3D | Cl(4,1,0) | 32 | 5 | 16 |
| CGA4D | Cl(5,1,0) | 64 | 6 | 31 |
| CGA5D | Cl(6,1,0) | 128 | 7 | 64 |
| CGA6D | Cl(7,1,0) | 256 | 8 | 127 |

**公式**：
- Blade 數 = 2^(n+2)
- Point 分量 = n + 2
- Motor 分量 = Σ(Grade 0, 2, 4, ...) - 1（排除 pseudoscalar）
