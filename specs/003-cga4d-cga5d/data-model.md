# 資料模型：CGA4D 與 CGA5D

**日期**: 2025-12-07
**分支**: `003-cga4d-cga5d`

## 1. CGA4D Cl(5,1) 資料模型

### 1.1 Blade 索引表

CGA4D 有 64 個 blade (2^6)，使用 clifford 函式庫的標準順序。

| Grade | 範圍 | 數量 | Blade 名稱 |
|-------|------|------|-----------|
| 0 | 0 | 1 | 1 (標量) |
| 1 | 1-6 | 6 | e1, e2, e3, e4, e+, e- |
| 2 | 7-21 | 15 | e12, e13, e14, e1+, e1-, e23, e24, e2+, e2-, e34, e3+, e3-, e4+, e4-, e+- |
| 3 | 22-41 | 20 | (三向量) |
| 4 | 42-56 | 15 | (四向量) |
| 5 | 57-62 | 6 | (五向量) |
| 6 | 63 | 1 | e1234+- (偽標量) |

### 1.2 UPGC Point 格式

```python
# 輸入格式：4D 歐幾里得座標
x: Tensor[..., 4]  # (x1, x2, x3, x4)

# UPGC 編碼格式：6 分量（Grade 1）
point: Tensor[..., 6]  # (e1, e2, e3, e4, e+, e-)

# 稀疏索引對應
UPGC_POINT_INDICES = (1, 2, 3, 4, 5, 6)
```

### 1.3 Motor 格式

```python
# Motor 格式：31 分量（Grade 0 + Grade 2 + Grade 4）
motor: Tensor[..., 31]

# 分量分佈
# [0]: Grade 0 (標量) - 1 分量
# [1-15]: Grade 2 (二向量) - 15 分量
# [16-30]: Grade 4 (四向量) - 15 分量

# Grade 分佈
MOTOR_GRADE_0_COUNT = 1
MOTOR_GRADE_2_COUNT = 15
MOTOR_GRADE_4_COUNT = 15
MOTOR_TOTAL_COUNT = 31
```

### 1.4 Reverse 符號（Motor）

```python
MOTOR_REVERSE_SIGNS = (
    1,                              # Grade 0: 1 分量
    -1, -1, -1, -1, -1, -1, -1,    # Grade 2: 15 分量（全部 -1）
    -1, -1, -1, -1, -1, -1, -1, -1,
    1, 1, 1, 1, 1, 1, 1,           # Grade 4: 15 分量（全部 +1）
    1, 1, 1, 1, 1, 1, 1, 1,
)
```

---

## 2. CGA5D Cl(6,1) 資料模型

### 2.1 Blade 索引表

CGA5D 有 128 個 blade (2^7)，使用 clifford 函式庫的標準順序。

| Grade | 範圍 | 數量 | Blade 名稱 |
|-------|------|------|-----------|
| 0 | 0 | 1 | 1 (標量) |
| 1 | 1-7 | 7 | e1, e2, e3, e4, e5, e+, e- |
| 2 | 8-28 | 21 | (二向量) |
| 3 | 29-63 | 35 | (三向量) |
| 4 | 64-98 | 35 | (四向量) |
| 5 | 99-119 | 21 | (五向量) |
| 6 | 120-126 | 7 | (六向量) |
| 7 | 127 | 1 | e12345+- (偽標量) |

### 2.2 UPGC Point 格式

```python
# 輸入格式：5D 歐幾里得座標
x: Tensor[..., 5]  # (x1, x2, x3, x4, x5)

# UPGC 編碼格式：7 分量（Grade 1）
point: Tensor[..., 7]  # (e1, e2, e3, e4, e5, e+, e-)

# 稀疏索引對應
UPGC_POINT_INDICES = (1, 2, 3, 4, 5, 6, 7)
```

### 2.3 Motor 格式

```python
# Motor 格式：64 分量（Grade 0 + Grade 2 + Grade 4 + Grade 6）
motor: Tensor[..., 64]

# 分量分佈
# [0]: Grade 0 (標量) - 1 分量
# [1-21]: Grade 2 (二向量) - 21 分量
# [22-56]: Grade 4 (四向量) - 35 分量
# [57-63]: Grade 6 (六向量) - 7 分量

# Grade 分佈
MOTOR_GRADE_0_COUNT = 1
MOTOR_GRADE_2_COUNT = 21
MOTOR_GRADE_4_COUNT = 35
MOTOR_GRADE_6_COUNT = 7
MOTOR_TOTAL_COUNT = 64
```

### 2.4 Reverse 符號（Motor）

```python
MOTOR_REVERSE_SIGNS = (
    1,                              # Grade 0: 1 分量
    -1, -1, ..., -1,               # Grade 2: 21 分量（全部 -1）
    1, 1, ..., 1,                  # Grade 4: 35 分量（全部 +1）
    -1, -1, ..., -1,               # Grade 6: 7 分量（全部 -1）
)
```

---

## 3. 共用資料結構

### 3.1 Null Basis

```python
# 對於所有 CGA 代數：
eo = (e- - e+) / 2    # 原點
einf = e+ + e-        # 無窮遠點

# 性質驗證
eo * eo = 0
einf * einf = 0
eo * einf = -1
```

### 3.2 UPGC 編碼公式

```python
def upgc_encode(x: Tensor) -> Tensor:
    """
    將歐幾里得座標編碼為 UPGC 表示。

    公式：X = n_o + x + 0.5|x|² n_inf

    展開後：
    - 歐幾里得分量直接複製
    - e+ = 0.5 * (1 - 0.5|x|²)
    - e- = 0.5 * (1 + 0.5|x|²)
    """
```

### 3.3 UPGC 解碼公式

```python
def upgc_decode(point: Tensor) -> Tensor:
    """
    將 UPGC 表示解碼為歐幾里得座標。

    公式：x = (X ∧ einf ∧ eo) / (X · einf)

    簡化後：直接提取歐幾里得分量並正規化
    """
```

---

## 4. 驗證規則

### 4.1 數值驗證

- 幾何積結果與 clifford 函式庫一致（誤差 < 1e-6）
- Null basis 性質成立
- 結合律成立：(a * b) * c = a * (b * c)
- Reverse 運算正確

### 4.2 ONNX 驗證

- 匯出成功，無錯誤或警告
- 計算圖無 Loop/If/Scan 節點
- PyTorch 與 ONNX Runtime 輸出一致

### 4.3 精度驗證

- fp16 輸入經 fp32 計算後輸出 fp16
- 數值穩定性（無溢位或下溢）

---

## 5. 與其他 CGA 代數對比

| 屬性 | CGA1D | CGA2D | CGA3D | CGA4D | CGA5D |
|------|-------|-------|-------|-------|-------|
| Clifford | Cl(2,1) | Cl(3,1) | Cl(4,1) | Cl(5,1) | Cl(6,1) |
| Blades | 8 | 16 | 32 | 64 | 128 |
| Point 分量 | 3 | 4 | 5 | 6 | 7 |
| Motor 分量 | 4 | 8 | 16 | 31 | 64 |
| 稀疏乘法 | 72 | 256 | 800 | ~5,800 | ~28,700 |
