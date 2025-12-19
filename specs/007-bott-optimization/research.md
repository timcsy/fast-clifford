# 研究文件：Bott 週期性優化

**功能**: 007-bott-optimization
**日期**: 2025-12-19
**狀態**: 完成

## 研究主題

### 1. Bott 週期性數學基礎

**決定**: 使用標準 Bott 週期性定理分解高維 Clifford 代數

**理由**:
- Cl(p+8, q) ≅ Cl(p, q) ⊗ M₁₆(ℝ)
- Cl(p, q+8) ≅ Cl(p, q) ⊗ M₁₆(ℝ)
- 這允許將任意高維代數分解為基底代數（blade_count ≤ 128）加上矩陣因子

**考慮的替代方案**:
- 直接硬編碼所有代數：儲存空間爆炸（p+q=9 時已需 500MB+）
- 使用 clifford 函式庫運行時計算：效能差，無法匯出 ONNX

**分解公式**:
```python
def decompose_signature(p, q):
    periods = 0
    while p + q >= 8:
        if p >= 8:
            p -= 8
        else:
            q -= 8
        periods += 1
    matrix_size = 16 ** periods
    return p, q, matrix_size
```

**範例分解**:

| 輸入代數 | 週期數 | 基底代數 | 矩陣大小 |
|----------|--------|----------|----------|
| Cl(8,0) | 1 | Cl(0,0) | 16 |
| Cl(10,0) | 1 | Cl(2,0) | 16 |
| Cl(10,2) | 1 | Cl(2,2) | 16 |
| Cl(17,0) | 2 | Cl(1,0) | 256 |
| Cl(24,0) | 3 | Cl(0,0) | 4096 |

---

### 2. Cl(p,q) 與 Cl(q,p) 的對稱性

**決定**: 使用基底向量索引重排來實現 Cl(p,q) → Cl(q,p) 映射

**理由**:
- Cl(p,q) 和 Cl(q,p) 作為代數是同構的
- 唯一差異在於基底向量的符號約定：
  - Cl(p,q): e₁²...eₚ² = +1, eₚ₊₁²...eₚ₊q² = -1
  - Cl(q,p): e₁²...eq² = -1, eq₊₁²...eq₊p² = +1
- 映射方法：交換正負簽章基底向量的索引

**實作策略**:
```python
class SymmetricClWrapper(CliffordAlgebraBase):
    def __init__(self, base_algebra, p, q):
        # base_algebra 是 Cl(q, p)，我們提供 Cl(p, q) 介面
        self._base = base_algebra
        self._p = p
        self._q = q
        # 計算索引重排映射
        self._swap_map = self._compute_swap_map()

    def _compute_swap_map(self):
        # 將 Cl(p,q) 的 blade 索引映射到 Cl(q,p) 的 blade 索引
        # 基底向量：[e₁...eₚ, eₚ₊₁...eₚ₊q] → [e₁...eq, eq₊₁...eq₊p]
        pass

    def geometric_product(self, a, b):
        # 1. 用 swap_map 重排 a, b 的分量
        # 2. 在 base algebra 上計算
        # 3. 用 swap_map 重排結果
        pass
```

**考慮的替代方案**:
- 生成所有 Cl(p,q) 和 Cl(q,p)：需要雙倍儲存空間
- 運行時符號計算：效能差，無法 ONNX 匯出

---

### 3. 張量化矩陣運算

**決定**: 使用 `torch.einsum` 實現張量化的 Bott 幾何積

**理由**:
- 現有實作使用三層 Python 迴圈：O(k³) 次迴圈開銷
- einsum 可以將所有運算表達為單一張量操作
- einsum 可匯出 ONNX，無 Loop 節點
- GPU 上可獲得顯著加速

**核心演算法**:

現有實作（慢）:
```python
for i in range(k):
    for j in range(k):
        for l in range(k):
            prod = base.geometric_product(a[i,l], b[l,j])
            result[i,j] += prod
```

張量化實作（快）:
```python
# 預計算乘法表：mult_table[i, j, k] = 當 eᵢ * eⱼ 貢獻到 eₖ 時的係數
# 形狀: (base_blades, base_blades, base_blades)

def geometric_product(self, a, b):
    # a, b: (..., k, k, base_blades)
    # result: (..., k, k, base_blades)

    # 使用 einsum 結合矩陣乘法和基底代數乘法
    # '...ilb, ...ljc, bcd -> ...ijd'
    # i,j: 結果矩陣索引
    # l: 縮並索引（矩陣乘法）
    # b,c: 輸入 blade 索引
    # d: 輸出 blade 索引
    result = torch.einsum(
        '...ilb, ...ljc, bcd -> ...ijd',
        a_mat, b_mat, self._mult_table
    )
    return result
```

**效能預估**:

| 方法 | 複雜度 | 預估加速 |
|------|--------|----------|
| Python 迴圈 | O(k³) × Python 開銷 | 1x（基準） |
| einsum | O(k³) × 張量開銷 | 10-50x |
| einsum + GPU | O(k³) × GPU 開銷 | 50-200x |

**考慮的替代方案**:
- torch.bmm + 迴圈：仍有部分迴圈開銷
- 手寫 CUDA kernel：非跨平台，違反憲法原則 II

---

### 4. 乘法表預計算

**決定**: 在 `__init__` 時預計算基底代數的完整乘法表

**理由**:
- 乘法表大小：(n, n, n) 其中 n = 基底代數 blade 數
- 最大基底代數：Cl(7,0) 有 128 blades → 表大小 128³ × 4 bytes ≈ 8MB
- 實際上大多數 Bott 代數使用更小的基底（n ≤ 16）→ 表大小 < 16KB
- 一次計算，多次使用

**實作**:
```python
def _compute_multiplication_table(self):
    n = self._base.count_blade
    table = torch.zeros(n, n, n, dtype=torch.float32)

    for i in range(n):
        ei = torch.zeros(n)
        ei[i] = 1.0
        for j in range(n):
            ej = torch.zeros(n)
            ej[j] = 1.0
            product = self._base.geometric_product(ei, ej)
            table[i, j, :] = product

    return table
```

**記憶體考量**:

| 基底代數 | Blade 數 | 表大小 |
|----------|----------|--------|
| Cl(0,0) | 1 | 4 bytes |
| Cl(2,0) | 4 | 256 bytes |
| Cl(4,0) | 16 | 16 KB |
| Cl(6,0) | 64 | 1 MB |
| Cl(7,0) | 128 | 8 MB |

---

### 5. 現有測試相容性

**決定**: 維持現有測試通過，新增對稱代數和效能測試

**理由**:
- 現有 212 個測試涵蓋 VGA、CGA、PGA、Bott 功能
- 優化不應破壞現有功能
- 需要新增測試驗證對稱代數正確性

**測試策略**:
1. 現有測試：確保全部通過
2. 新增對稱代數測試：驗證 Cl(p,q) 和 Cl(q,p) 等價性
3. 新增效能測試：驗證 10x+ 加速目標
4. 新增 ONNX 匯出測試：驗證無 Loop 節點

---

## 技術決策摘要

| 主題 | 決定 | 關鍵考量 |
|------|------|----------|
| Bott 分解 | 標準定理 + 迭代分解 | 支援多重週期 |
| 對稱性 | 索引重排包裝器 | 零額外儲存 |
| 張量化 | einsum 單一操作 | ONNX 相容、GPU 加速 |
| 乘法表 | 預計算 3D 張量 | 空間換時間 |
| 測試 | 保留現有 + 新增 | 不破壞功能 |

## 未解決問題

無。所有研究問題已解決。
