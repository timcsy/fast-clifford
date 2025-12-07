<!--
同步影響報告
==================
版本變更: N/A → 0.1.0 (初始版本)
修改原則: N/A (初次建立)
新增章節:
  - I. ONNX 部署優先
  - II. 平台相容性
  - III. 無迴圈前向傳播
  - IV. 硬編碼代數展開
  - V. 數值精度安全
  - VI. 文件語言規範
  - VII. 增量提交原則
  - 禁止技術清單
  - 開發工作流程
移除章節: 無
需更新的範本:
  - .specify/templates/plan-template.md: ⚠ 待更新 (加入 ONNX/TensorRT 檢查)
  - .specify/templates/spec-template.md: ✅ 無需變更
  - .specify/templates/tasks-template.md: ⚠ 待更新 (加入 ONNX 匯出驗證任務)
待辦事項: 無
-->

# fast-clifford 專案憲法

## 核心原則

### I. ONNX 部署優先

**所有實作決策必須優先考慮 ONNX 可匯出性，即使犧牲程式碼美觀度。**

- 模型必須通過 `torch.onnx.export()` 且無錯誤
- 產生的 ONNX 計算圖禁止包含 `Loop` 節點（TensorRT 優化需求）
- ONNX opset 版本：17+（確保更廣泛的算子支援）
- 每個 `nn.Module` 在合併前必須驗證 ONNX 匯出

**理由**：生產環境部署目標為 TensorRT，其對動態控制流的支援有限。Loop 節點會導致編譯失敗或嚴重的效能下降。

### II. 平台相容性

**程式碼必須是純 PyTorch — 禁止平台特定擴充。**

- 開發環境：Apple M3 (MPS 後端)
- 生產環境：NVIDIA GPU (TensorRT)
- 所有程式碼必須在兩個平台上執行結果相同
- 禁止使用平台特定 API（僅 MPS 或僅 CUDA 的操作）

**理由**：跨平台相容性確保開發速度（在 M3 上快速迭代）同時維持生產可部署性（TensorRT 推理）。

### III. 無迴圈前向傳播

**前向傳播禁止包含任何會產生 ONNX Loop 節點的 Python 控制流。**

`forward()` 方法中禁止：
- 對動態維度的 `for` 迴圈
- `while` 迴圈
- 遞迴函式呼叫
- 帶迴圈的 `torch.jit.script`（會產生 Loop 節點）

允許的替代方案：
- 張量廣播
- `torch.einsum()`（僅限靜態下標）
- `torch.stack()`、`torch.cat()`
- 固定張量形狀的向量化操作
- 可在 `__init__` 時預先計算的列表推導式

**理由**：ONNX 追蹤器會將 Python 迴圈轉換為 Loop 節點，TensorRT 無法有效優化這些節點。

### IV. 硬編碼代數展開

**CGA 運算必須使用明確的硬編碼係數計算 — 禁止 Cayley 表查詢。**

禁止：
- 大型預計算 Cayley 表張量（例如 `(32, 32, 32)` 乘積張量）
- 執行時的表格查詢進行 blade 乘法
- `einsum('kij,...i,...j->...k', cayley_table, a, b)` 模式

必須採用的方法：
- 將幾何積展開為明確的係數算術
- 將 32×32 乘法結果硬編碼為個別張量操作
- 使用符號展開：`result[k] = sum of (sign * a[i] * b[j])` 明確寫出

**範例**（grade-0 × grade-1 乘積）：
```python
# 禁止：
result = einsum('kij,i,j->k', cayley[0:6, 0:1, 1:6], scalar, vector)

# 必須：
result_e1 = scalar * vector_e1
result_e2 = scalar * vector_e2
result_e3 = scalar * vector_e3
result_eplus = scalar * vector_eplus
result_eminus = scalar * vector_eminus
```

**理由**：大型常數張量會膨脹 ONNX 模型大小並阻止 TensorRT 核心融合。硬編碼算術可實現最大優化。

### V. 數值精度安全

**CGA 計算層必須強制使用 float32 精度。**

- 所有 CGA 操作必須在層入口將輸入轉型為 `float32`
- 輸出可視需要轉回原始 dtype
- 這可防止幾何積中的溢位（可能產生大的中間值）
- 記錄任何精度敏感的操作

**實作模式**：
```python
def forward(self, x):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    # ... CGA 操作 ...
    return result.to(original_dtype)
```

**理由**：混合精度訓練（float16/bfloat16）可能在 CGA 的鏈式乘法中造成溢位。強制 float32 確保數值穩定性。

### VI. 文件語言規範

**所有規格文件必須使用繁體中文撰寫。**

- 規格書 (spec.md)：繁體中文
- 計畫書 (plan.md)：繁體中文
- 任務清單 (tasks.md)：繁體中文
- 憲法 (constitution.md)：繁體中文
- 程式碼註解：英文（便於國際協作）
- 變數/函式命名：英文（程式碼標準）

**理由**：統一文件語言確保團隊溝通一致性，繁體中文為專案主要語言。

### VII. 增量提交原則

**完成一個邏輯段落後必須立即提交 Git。**

- 每個任務（Task）完成後提交
- 每個檢查點（Checkpoint）通過後提交
- 規格文件修改後提交
- 避免大型、難以審查的提交

**提交時機範例**：
- 完成一個 Phase 的所有任務
- 修正分析報告指出的問題
- 新增或更新設計文件
- 通過測試驗證後

**提交訊息格式**：
```
<type>(<scope>): <subject>

<body>
```

類型：`feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**理由**：頻繁的小提交便於追蹤進度、回滾問題、審查變更。大型提交增加衝突風險且難以理解。

## 禁止技術清單

**以下技術在本專案中嚴格禁止：**

| 技術 | 禁止原因 |
|------|----------|
| Triton | 無法匯出 ONNX，僅限 NVIDIA |
| CUDA C++ 擴充 | 非跨平台（破壞 MPS 開發） |
| Taichi | 無法匯出 ONNX |
| 帶迴圈的 `torch.jit.script` | 產生 ONNX Loop 節點 |
| 自訂 CUDA 核心 | 非跨平台 |

**允許的技術**：
- 純 PyTorch（`torch.*` 操作）
- NumPy（僅用於預處理/測試，禁止在前向傳播中使用）
- `torch.jit.trace`（用於 ONNX 匯出驗證）

## 開發工作流程

### 合併前檢查清單

每個 PR 必須通過：

1. **ONNX 匯出測試**：`torch.onnx.export()` 成功且無警告
2. **Loop 節點檢查**：ONNX 計算圖包含零個 `Loop` 節點
3. **跨平台測試**：測試在 MPS 和 CUDA（或 CPU 備援）上都通過
4. **精度測試**：float32 和 float16 輸入的數值輸出相符（在容差範圍內）

### 驗證指令

```bash
# 匯出至 ONNX 並驗證
python -c "
import torch
from model import YourModule
m = YourModule()
x = torch.randn(1, 32)
torch.onnx.export(m, x, 'test.onnx', opset_version=17)
"

# 檢查 Loop 節點
python -c "
import onnx
model = onnx.load('test.onnx')
loops = [n for n in model.graph.node if n.op_type == 'Loop']
assert len(loops) == 0, f'發現 {len(loops)} 個 Loop 節點！'
"
```

## 治理

本憲法優先於所有其他實作指南。修訂需要：

1. 書面說明變更理由
2. 對現有程式碼的影響分析
3. 若有破壞性變更則需遷移計畫
4. 依照語意版本控制遞增版本號

**合規性**：所有程式碼審查必須驗證憲法合規性。違規必須在合併前解決。

**版本**: 0.1.0 | **批准日期**: 2025-12-05 | **最後修訂**: 2025-12-05
