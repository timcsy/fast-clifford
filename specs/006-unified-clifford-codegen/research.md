# Research: Unified Cl(p,q,0) Codegen System

**Feature**: 006-unified-clifford-codegen
**Date**: 2025-12-18

## 0. 運算子重載設計

### Decision
完整定義 Python 運算子映射到幾何代數運算。

### Operator Mapping Table

| 運算子 | Python | Multivector | Rotor | 說明 |
|--------|--------|-------------|-------|------|
| 幾何積 | `a * b` | ✅ `geometric_product` | ✅ `compose_rotor` | 核心運算 |
| 外積 | `a ^ b` | ✅ `outer` | ❌ | wedge product |
| 內積 | `a \| b` | ✅ `inner` | ❌ | scalar product |
| 左縮並 | `a << b` | ✅ `contract_left` | ❌ | left contraction |
| 右縮並 | `a >> b` | ✅ `contract_right` | ❌ | right contraction |
| 三明治積 | `m @ x` | ✅ `sandwich` | ✅ `sandwich_rotor` | **注意**: `@` 是三明治積 |
| 反轉 | `~a` | ✅ `reverse` | ✅ `reverse_rotor` | reversion |
| 逆元 | `a ** -1` | ✅ `inverse` | ✅ `inverse_rotor` | multiplicative inverse |
| 正冪次 | `a ** n` | ✅ 重複幾何積 | ✅ 重複組合 | n >= 0 |
| Meet | `a & b` | ✅ `regressive` | ❌ | regressive product (新增) |
| 加法 | `a + b` | ✅ | ✅ | 係數相加 |
| 減法 | `a - b` | ✅ | ✅ | 係數相減 |
| 負號 | `-a` | ✅ | ✅ | 取負 |
| 純量乘 | `s * a` | ✅ | ✅ | 純量乘法 |
| 純量除 | `a / s` | ✅ | ✅ | 純量除法 |
| Grade 選取 | `a(k)` | ✅ `__call__` | ❌ | 選取 grade-k (新增) |

### 新增運算子說明

**1. Meet 運算 (`&`)**
```python
# Regressive product (meet): a ∨ b = (a* ∧ b*)*
# 使用 & 因為它在幾何上表示「相交」
result = a & b  # 等價於 algebra.regressive(a, b)
```

**2. Grade 選取 (`()`)**
```python
# 選取特定 grade
bivector_part = mv(2)  # 等價於 algebra.select_grade(mv, 2)
scalar_part = mv(0)
```

### Rotor 運算子簡化

Rotor 類別**不支援**以下運算子（因為 Rotor 空間不封閉）：
- `^` 外積：結果可能不是 Rotor
- `|` 內積：結果是 scalar
- `<<`, `>>` 縮並：結果可能不是 Rotor
- `&` meet：結果可能不是 Rotor
- `()` grade 選取：Rotor 本身就是特定 grades

### 實作注意事項

```python
class Multivector:
    def __and__(self, other: Multivector) -> Multivector:
        """Meet (regressive product): a & b"""
        result = self.algebra.regressive(self.data, other.data)
        return Multivector(result, self.algebra)

    def __call__(self, grade: int) -> Multivector:
        """Grade selection: mv(k) extracts grade-k"""
        result = self.algebra.select_grade(self.data, grade)
        return Multivector(result, self.algebra)
```

---

## 1. Bott 週期性實作策略

### Decision
採用 Cl(p+8, q) ≅ Cl(p, q) ⊗ M₁₆(ℝ) 張量積分解，base algebra 使用預生成的硬編碼模組。

### Rationale
- **數學正確性**: Bott 週期定理是 Clifford 代數的基本結構定理
- **效能**: 利用 GEMM 高度優化的 16×16 矩陣乘法，避免 O(blade_count²) 的 Cayley 表
- **ONNX 相容**: 可以用 `torch.einsum` 和 `torch.matmul` 實現，無動態迴圈

### Alternatives Considered
| 方案 | 優點 | 缺點 | 結論 |
|------|------|------|------|
| Cayley 表查詢 | 實作簡單 | O(blade_count²) 空間，ONNX 不友好 | ❌ 拒絕 |
| 稀疏矩陣 | 節省空間 | PyTorch sparse 不支援 ONNX | ❌ 拒絕 |
| 遞迴 Bott | 支援任意維度 | 每次遞迴增加 overhead | ✅ 備選（k>2 時使用）|
| **張量積分解** | GEMM 優化，ONNX 相容 | 實作複雜 | ✅ **採用** |

### Implementation Notes
```python
# Blade 索引分解（單層 Bott，k=1）
def decompose_blade_index(I: int, base_blades: int) -> Tuple[int, int, int]:
    """
    將 CGA(n+8) blade 索引分解為 CGA(n) 索引 + 矩陣位置

    CGA(n+8) 有 base_blades * 256 個 blades
    """
    base_idx = I // 256        # [0, base_blades-1]
    matrix_idx = I % 256       # [0, 255]
    row = matrix_idx // 16     # [0, 15]
    col = matrix_idx % 16      # [0, 15]
    return base_idx, row, col
```

---

## 2. PGA 嵌入 CGA 策略

### Decision
PGA(n) = Cl(n, 0, 1) 透過映射到 CGA(n) = Cl(n+1, 1) 實作，利用 null vector 對應。

### Rationale
- **複用優化**: CGA 已有完整的硬編碼優化
- **數學基礎**: PGA 的退化向量 e₀ (e₀² = 0) 可映射到 CGA 的 e_inf (e_inf² = 0)
- **避免重複**: 不需要為 PGA 單獨生成代碼

### Alternatives Considered
| 方案 | 優點 | 缺點 | 結論 |
|------|------|------|------|
| 獨立 PGA codegen | 最優效能 | 退化維度需特殊處理，複雜度高 | ❌ 拒絕 |
| Runtime Cayley | 實作簡單 | 效能差，ONNX 不相容 | ❌ 拒絕 |
| **CGA 嵌入** | 複用現有優化 | 有嵌入/投影開銷 | ✅ **採用** |

### Implementation Notes
```python
class PGAEmbedding:
    """PGA(n) = Cl(n, 0, 1) 嵌入 CGA(n) = Cl(n+1, 1)"""

    def __init__(self, n: int):
        self.cga = Cl(n + 1, 1)  # 複用 CGA 硬編碼

    def embed(self, pga_mv: Tensor) -> Tensor:
        """PGA → CGA: e₀ → e_inf"""
        # 建立映射表：PGA blade → CGA blade
        ...

    def project(self, cga_mv: Tensor) -> Tensor:
        """CGA → PGA: 忽略 e_o 相關分量"""
        ...
```

---

## 3. 代碼生成架構

### Decision
統一使用 `ClCodeGenerator` 生成所有 Cl(p,q,0) 代數，輸出到 `algebras/generated/cl_{p}_{q}/`。

### Rationale
- **一致性**: 所有代數使用相同的生成邏輯
- **可維護**: 修改生成器即可更新所有代數
- **可擴展**: 容易新增運算

### Generated Module Structure
```text
algebras/generated/cl_4_1/
├── __init__.py          # 匯出所有運算
├── functional.py        # 生成的硬編碼運算（geometric_product, reverse, etc.）
├── constants.py         # ROTOR_MASK, BIVECTOR_INDICES, etc.
└── layers.py            # nn.Module 包裝
```

### Key Generated Functions
| 函數 | 輸入 | 輸出 | 說明 |
|------|------|------|------|
| `geometric_product` | (blade, blade) | blade | 完整幾何積 |
| `inner` | (blade, blade) | scalar | 內積 <ab>₀ |
| `outer` | (blade, blade) | blade | 外積 a ∧ b |
| `contract_left` | (blade, blade) | blade | 左縮並 a ⌋ b |
| `contract_right` | (blade, blade) | blade | 右縮並 a ⌊ b |
| `reverse` | blade | blade | 反轉 ã |
| `involute` | blade | blade | Grade 反演 â |
| `conjugate` | blade | blade | Clifford 共軛 a† |
| `dual` | blade | blade | Poincaré 對偶 |
| `select_grade` | (blade, int) | grade_k | 提取 grade-k |
| `compose_rotor` | (rotor, rotor) | rotor | Rotor 組合 |
| `reverse_rotor` | rotor | rotor | Rotor 反轉 |
| `sandwich_rotor` | (rotor, point) | point | 加速 sandwich |
| `exp_bivector` | bivector | rotor | exp(B) |

---

## 4. 命名遷移策略

### Decision
完全採用新命名，不提供向後相容別名。

### Rationale
- **清晰**: 避免兩套命名系統共存的混淆
- **乾淨**: 不累積技術債務
- **一致**: 整個 codebase 使用統一術語

### Migration Table
| 舊名稱 | 新名稱 | 說明 |
|--------|--------|------|
| `EvenVersor` | `Rotor` | 類別名 |
| `Similitude` | *(移除)* | 不再需要 |
| `blade_count` | `count_blade` | 屬性 |
| `even_versor_count` | `count_rotor` | 屬性 |
| `bivector_count` | `count_bivector` | 屬性 |
| `geometric_product_full` | `geometric_product` | 函數 |
| `reverse_full` | `reverse` | 函數 |
| `grade_select` | `select_grade` | 函數 |
| `inner_product` | `inner` | 函數 |
| `outer_product` | `outer` | 函數 |
| `left_contraction` | `contract_left` | 函數 |
| `right_contraction` | `contract_right` | 函數 |
| `compose_even_versor` | `compose_rotor` | 函數 |
| `reverse_even_versor` | `reverse_rotor` | 函數 |
| `sandwich_product_sparse` | `sandwich_rotor` | 函數 |

---

## 5. CGA Null Basis 慣例

### Decision
採用 Dorst 慣例：`e_o = (e_- - e_+)/2`, `e_inf = e_- + e_+`

### Rationale
- **標準**: Dorst《Geometric Algebra for Computer Science》教科書定義
- **相容**: clifford 庫預設使用此慣例，便於驗證
- **直覺**: e_o 表示原點，e_inf 表示無窮遠點

### Mathematical Properties
```
e_o · e_o = 0      (null vector)
e_inf · e_inf = 0  (null vector)
e_o · e_inf = -1   (reciprocal pair)

UPGC Point: X = x + (1/2)|x|² e_inf + e_o
```

---

## 6. 測試策略

### Decision
分層測試：低維度完整驗證，高維度語法檢查。

### Rationale
- **效率**: clifford 庫在高維度極慢（>100ms/op）
- **覆蓋**: 低維度數學邏輯 = 高維度數學邏輯
- **實用**: 大多數應用使用 p+q ≤ 5

### Test Matrix
| 測試類型 | p+q ≤ 5 | 5 < p+q ≤ 9 | p+q > 9 |
|----------|---------|-------------|---------|
| 對照 clifford | ✅ 完整 | ❌ 跳過 | ❌ 跳過 |
| 形狀驗證 | ✅ | ✅ | ✅ |
| ONNX 匯出 | ✅ | ✅ (抽樣) | ❌ (太大) |
| 效能 benchmark | ✅ 對比 | ✅ 計時 | ✅ 計時 |

---

## 7. 檔案大小預估

### Generated File Sizes
| 代數 | Blades | geometric_product 項數 | 預估 functional.py |
|------|--------|------------------------|-------------------|
| Cl(3,0) | 8 | ~64 | ~10 KB |
| Cl(4,1) | 32 | ~1,600 | ~100 KB |
| Cl(6,1) | 128 | ~18,000 | ~1 MB |
| Cl(8,1) | 512 | ~262,000 | ~15 MB |
| Cl(9,0) | 512 | ~262,000 | ~15 MB |

### Total Estimate
- 55 個代數，大多數小於 1 MB
- 最大檔案（Cl(8,1), Cl(9,0)）約 15 MB
- 總計約 50-100 MB（壓縮後約 5-10 MB）

---

## 8. 效能預期

### Benchmark Targets
| 運算 | 相對 clifford | 說明 |
|------|---------------|------|
| geometric_product | 4-10x | 硬編碼 vs 查表 |
| sandwich_rotor | 20-100x | 稀疏優化 + 避免 reverse |
| compose_rotor | 5-15x | 只計算偶數 grade |
| exp_bivector | 10-50x | 解析公式 vs 泰勒展開 |

### Rotor Acceleration
| 運算 | 通用版本複雜度 | Rotor 版本複雜度 | 加速比 |
|------|----------------|------------------|--------|
| compose | O(blade²) | O(rotor²) | ~4x |
| reverse | O(blade) | O(rotor) | ~2x |
| sandwich | O(blade² × point) | O(rotor × point) | ~8x |
