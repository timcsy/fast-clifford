# 研究報告：CGA 幾何代數規則定義

**日期**: 2025-12-05
**功能**: 001-cga-algebra-rules

## 研究摘要

本報告涵蓋三個關鍵研究領域：
1. Python clifford 庫的 CGA 實現
2. ONNX 算子支援與匯出最佳實踐
3. 程式碼生成策略

---

## 1. Clifford 庫 CGA 實現

### 決策：使用 `conformalize()` 函式建立 CGA

**理由**：
- 提供標準的 Null Basis 定義
- 包含 `up()`/`down()` 投影函式
- 自動計算正確的度規簽名

**替代方案**：
- `Cl(4, 1)` 直接建立：需手動定義 Null Basis
- `galgebra`：純符號計算，無數值驗證

### API 使用範例

```python
from clifford import Cl, conformalize

# 建立 CGA
G3, _ = Cl(3)
layout_c, blades_c, stuff = conformalize(G3)

# 存取 Null Basis
eo = stuff['eo']      # 原點: (e_- - e_+) / 2
einf = stuff['einf']  # 無窮遠點: e_- + e_+

# 驗證 Null Basis 性質
# eo * eo = 0, einf * einf = 0, eo * einf = -1
```

### 提取乘法表

```python
# 取得幾何積乘法表（稀疏 COO 格式）
gmt = layout_c.gmt

# 轉換為密集陣列
gmt_dense = gmt.todense()  # 形狀: (32, 32, 32)

# 迭代所有乘法規則
blade_tuples = layout_c.bladeTupList
for i in range(32):
    for j in range(32):
        result = gmt_dense[i, :, j]
        nonzero = np.where(result != 0)[0]
        for k in nonzero:
            coeff = result[k]
            # (i, j) -> k with sign coeff
```

### UPGC 點表示

```python
# 使用 up() 投影函式
x_3d = 2*e1 + 3*e2 + 4*e3
X = stuff['up'](x_3d)  # 自動計算: eo + x + (|x|^2/2)*einf

# 稀疏性：只有 Grade 1 有非零值（5 個分量）
```

### Motor 表示

```python
from clifford.tools.g3c import (
    generate_rotation_rotor,
    generate_translation_rotor
)

# 生成 Motor
rotor_rot = generate_rotation_rotor(angle, axis)
rotor_trans = generate_translation_rotor(translation_vec)
motor = rotor_trans * rotor_rot

# 稀疏性：只有 Grade 0, 2, 4 有非零值（16 個分量）
```

---

## 2. ONNX 算子支援

### 決策：使用 ONNX opset 17

**理由**：
- 包含所有必要的基本算子
- 廣泛的 TensorRT 支援
- 支援 bfloat16 型別

### 關鍵算子清單

| 算子 | PyTorch 對應 | 用途 |
|------|-------------|------|
| Add | `+` | 係數相加 |
| Mul | `*` | 係數相乘 |
| Neg | `-x` | 符號反轉 |
| Sub | `-` | 係數相減 |
| Concat | `torch.cat` | 合併張量 |
| Slice | `x[:]` | 切片操作 |
| Reshape | `torch.reshape` | 形狀變換 |

### PyTorch 匯出配置

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    dynamic_axes={'input': {0: 'batch_size'}},
    do_constant_folding=True
)
```

### Loop 節點避免策略

| 問題 | 解決方案 |
|------|---------|
| Python for 迴圈 | 使用張量廣播 |
| 動態條件 | 避免在 forward 中使用 |
| torch.jit.script 迴圈 | 只用於無迴圈函式 |

### 驗證無 Loop 節點

```python
import onnx

model = onnx.load("model.onnx")
loop_nodes = [n for n in model.graph.node if n.op_type == "Loop"]
assert len(loop_nodes) == 0, f"發現 {len(loop_nodes)} 個 Loop 節點"
```

---

## 3. 程式碼生成策略

### 決策：使用字串模板生成 Python 程式碼

**理由**：
- 簡單直觀
- 易於除錯
- 無需 AST 操作

**替代方案**：
- Python AST：過於複雜
- Jinja2 模板：額外依賴

### 生成器架構

```
codegen/
├── algebra.py          # 從 clifford 提取乘法規則
├── sparse_analysis.py  # 分析稀疏性模式
└── generate.py         # 生成 PyTorch 程式碼
```

### 輸出範例

```python
# 自動生成，禁止手動編輯
def sandwich_product_sparse(motor, point):
    """
    計算 M × X × M̃
    motor: (..., 16) - 偶數 grade 分量
    point: (..., 5) - Grade 1 分量
    返回: (..., 5) - 變換後的 Grade 1 分量
    """
    # 第一步: M × X
    temp_0 = motor[..., 0] * point[..., 0]  # scalar × e1
    temp_1 = motor[..., 0] * point[..., 1]  # scalar × e2
    # ... 展開所有乘積

    # 第二步: (M × X) × M̃
    result_0 = temp_0 * motor[..., 0] + ...
    result_1 = temp_1 * motor[..., 0] + ...
    # ... 展開所有乘積

    return torch.stack([result_0, result_1, result_2, result_3, result_4], dim=-1)
```

### 稀疏性優化

| 操作 | 完整計算量 | 稀疏計算量 | 優化比 |
|------|-----------|-----------|--------|
| M × X | 32 × 32 = 1024 | 16 × 5 = 80 | 12.8x |
| (M×X) × M̃ | 32 × 32 = 1024 | ~20 × 16 = 320 | 3.2x |
| 總計 | 2048 | ~100-150 | ~15x |

**關鍵發現**：三明治積輸出只有 Grade 1（5 個分量），因此最終結果的稀疏性與輸入相同。

---

## 4. 關鍵決策總結

| 決策 | 選擇 | 理由 |
|------|------|------|
| CGA 庫 | clifford + conformalize | 標準 Null Basis，完整 API |
| 符號計算 | 不使用 sympy | 直接從 clifford 提取數值規則 |
| ONNX opset | 17 | 廣泛支援，含 bfloat16 |
| 程式碼生成 | 字串模板 | 簡單，易除錯 |
| 稀疏性 | 編譯時展開 | 符合憲法（無 Cayley 表） |

---

## 5. 待解決問題

1. **Blade 索引順序**：需確認 clifford 的 blade 排序與規格一致
2. **Reverse 符號**：需從 clifford 提取正確的 Reverse 符號表
3. **數值穩定性**：大座標值的 $|x|^2$ 計算可能溢位

---

## 參考資源

- [Clifford CGA 教程](https://clifford.readthedocs.io/en/latest/tutorials/cga/index.html)
- [ONNX 算子文檔](https://onnx.ai/onnx/operators/)
- [PyTorch ONNX 匯出](https://docs.pytorch.org/docs/stable/onnx.html)
