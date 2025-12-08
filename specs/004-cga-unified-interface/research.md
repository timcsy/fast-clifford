# 研究報告：CGA(n) 統一介面

**日期**：2025-12-08
**功能**：004-cga-unified-interface

## 研究任務

### R1：運行時動態展開技術

**問題**：如何在不使用 Python 迴圈的情況下，為任意維度的 Clifford 代數動態生成幾何積操作？

**決策**：使用張量化批次索引操作

**原理**：
1. Clifford 代數的幾何積可表示為：`result[k] = Σ (sign[i,j,k] * a[i] * b[j])`
2. 對於固定的代數（固定 p, q, r），非零項是已知的常數集合
3. 可以將所有非零項的 (i, j, k, sign) 打包為張量，一次性計算

**實作方案**：

```python
# 預計算階段（首次呼叫時）
def _build_product_tensors(self):
    # 從 cga_factory 獲取 Cayley 表
    cayley = self.algebra.cayley_table  # shape: (n, n, n)

    # 找出所有非零項
    nonzero = torch.nonzero(cayley != 0)  # shape: (num_nonzero, 3)

    self.left_indices = nonzero[:, 0]   # i 索引
    self.right_indices = nonzero[:, 1]  # j 索引
    self.result_indices = nonzero[:, 2] # k 索引
    self.signs = cayley[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]]

# 計算階段（每次呼叫）
def geometric_product(self, a, b):
    # 批次取得所有需要的分量
    a_vals = a[..., self.left_indices]    # (..., num_nonzero)
    b_vals = b[..., self.right_indices]   # (..., num_nonzero)

    # 計算所有乘積
    products = self.signs * a_vals * b_vals  # (..., num_nonzero)

    # 累加到結果
    result = torch.zeros((*a.shape[:-1], self.blade_count), device=a.device)
    result.scatter_add_(-1, self.result_indices.expand(*a.shape[:-1], -1), products)

    return result
```

**替代方案考慮**：
1. ❌ Python 迴圈展開 - 會產生 ONNX Loop 節點
2. ❌ torch.jit.script 帶迴圈 - 會產生 ONNX Loop 節點
3. ❌ 動態程式碼生成（exec） - 無法追蹤梯度
4. ✅ 張量化批次操作 - ONNX 相容，可微分

---

### R2：PyTorch 張量索引的 ONNX 相容性

**問題**：使用 `scatter_add_` 和 `index_select` 是否會在 ONNX 中產生 Loop 節點？

**決策**：確認相容，使用 Gather/ScatterElements 操作

**驗證**：

```python
import torch
import torch.onnx

class TestIndexOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('indices', torch.tensor([0, 2, 1, 3]))

    def forward(self, x):
        # index_select -> ONNX Gather
        selected = x[..., self.indices]

        # scatter_add -> ONNX ScatterElements
        result = torch.zeros(x.shape[0], 5)
        result.scatter_add_(1, self.indices.unsqueeze(0).expand(x.shape[0], -1), selected)
        return result

# 匯出並檢查
model = TestIndexOps()
x = torch.randn(4, 4)
torch.onnx.export(model, x, 'test.onnx', opset_version=17)

# 檢查無 Loop 節點
import onnx
onnx_model = onnx.load('test.onnx')
op_types = {n.op_type for n in onnx_model.graph.node}
assert 'Loop' not in op_types  # ✅ 通過
```

**ONNX 操作對應**：
| PyTorch 操作 | ONNX 操作 | Loop 節點 |
|-------------|-----------|-----------|
| `tensor[..., indices]` | Gather | ❌ 無 |
| `scatter_add_` | ScatterElements | ❌ 無 |
| `index_select` | Gather | ❌ 無 |

**結論**：張量化批次索引操作完全相容 ONNX，不會產生 Loop 節點。

---

### R3：延遲初始化模式

**問題**：如何在首次使用時生成展開程式碼，同時保持 ONNX 匯出相容性？

**決策**：使用 `@cached_property` + `register_buffer`

**原理**：
1. 索引張量作為 buffer 註冊，會被包含在 ONNX 模型中
2. 延遲初始化避免未使用代數的計算開銷
3. 快取確保只計算一次

**實作模式**：

```python
class RuntimeCGAAlgebra(nn.Module):
    def __init__(self, euclidean_dim: int):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        # 計算代數參數
        algebra = CGAFactory.create(self.euclidean_dim)

        # 註冊為 buffer（包含在 state_dict 和 ONNX 中）
        self.register_buffer('left_idx', algebra.left_indices)
        self.register_buffer('right_idx', algebra.right_indices)
        self.register_buffer('result_idx', algebra.result_indices)
        self.register_buffer('signs', algebra.signs)

        self._initialized = True

    def forward(self, motor, point):
        self._ensure_initialized()
        # 使用已註冊的 buffer 進行計算
        ...
```

**替代方案考慮**：
1. ❌ 建構時完全初始化 - 對於高維度代數開銷大
2. ❌ 使用 Python dict 快取 - 不會被 ONNX 追蹤
3. ✅ register_buffer + 延遲初始化 - ONNX 相容，高效

---

## 結論

所有研究任務的關鍵問題已解決：

| 任務 | 結論 |
|------|------|
| R1 運行時動態展開 | 使用張量化批次索引操作，無需 Python 迴圈 |
| R2 ONNX 相容性 | Gather/ScatterElements 操作無 Loop 節點 |
| R3 延遲初始化 | register_buffer + 延遲初始化模式 |

**無待釐清項目**，可進入 Phase 1 設計階段。
