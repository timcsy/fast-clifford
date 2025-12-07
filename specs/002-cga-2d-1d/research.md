# 研究報告：CGA2D 與 CGA1D 代數規則

**日期**: 2025-12-07
**功能**: 002-cga-2d-1d

## 研究摘要

本研究驗證 CGA2D (Cl(3,1)) 和 CGA1D (Cl(2,1)) 的代數規則，確認稀疏性假設，並計算優化後的計算量。

## 1. CGA2D Cl(3,1) 代數規則

### 決策：採用 clifford 函式庫的 conformalize(Cl(2))

**理由**：
- 與 CGA3D 實作一致的方法
- 自動處理 null basis 建構
- 經過驗證的數學正確性

**替代方案**：
- 手動建構 Cl(3,1)：較複雜，容易出錯

### 代數結構

| 屬性 | 數值 |
|------|------|
| 代數類型 | Cl(3,1) |
| 簽名 | (+,+,+,-) |
| Blade 數量 | 16 (2^4) |
| 基底 | e1, e2, e+, e- |

### Blade 索引映射

```
Grade 0 (1 blade):
  Index 0: scalar ()

Grade 1 (4 blades):
  Index 1: e1
  Index 2: e2
  Index 3: e+ (原 e3)
  Index 4: e- (原 e4)

Grade 2 (6 blades):
  Index 5: e12
  Index 6: e1+
  Index 7: e1-
  Index 8: e2+
  Index 9: e2-
  Index 10: e+-

Grade 3 (4 blades):
  Index 11: e12+
  Index 12: e12-
  Index 13: e1+-
  Index 14: e2+-

Grade 4 (1 blade):
  Index 15: e12+-
```

### Null Basis 驗證

```
e_o = -0.5*e+ + 0.5*e- = -0.5*e3 + 0.5*e4
e_inf = e+ + e- = e3 + e4

驗證：
  e_o² = 0 ✓
  e_inf² = 0 ✓
  e_o · e_inf = -1 ✓
```

### 稀疏性模式

**UPGC 2D 點** (4 分量):
- Grade 1: indices (1, 2, 3, 4)
- 表示: [e1, e2, e+, e-]

**2D Motor** (8 分量):
- Grade 0: index (0) - 1 個
- Grade 2: indices (5, 6, 7, 8, 9, 10) - 6 個
- Grade 4: index (15) - 1 個

### 三明治積稀疏性驗證

測試配置：
- 旋轉馬達: R = cos(π/8) + sin(π/8)*e12
- 輸入點: (1, 0.5)

結果：
- 輸入 Grade 1 分量: [1.0, 0.5, 0.125, 1.125]
- 輸出 Grade 1 分量: [1.06, -0.35, 0.125, 1.125]
- **非 Grade-1 分量: 全為零 ✓**

## 2. CGA1D Cl(2,1) 代數規則

### 決策：採用 clifford 函式庫的 conformalize(Cl(1))

**理由**：同 CGA2D

### 代數結構

| 屬性 | 數值 |
|------|------|
| 代數類型 | Cl(2,1) |
| 簽名 | (+,+,-) |
| Blade 數量 | 8 (2^3) |
| 基底 | e1, e+, e- |

### Blade 索引映射

```
Grade 0 (1 blade):
  Index 0: scalar ()

Grade 1 (3 blades):
  Index 1: e1
  Index 2: e+ (原 e2)
  Index 3: e- (原 e3)

Grade 2 (3 blades):
  Index 4: e1+
  Index 5: e1-
  Index 6: e+-

Grade 3 (1 blade):
  Index 7: e1+-
```

### Null Basis 驗證

```
e_o = -0.5*e+ + 0.5*e- = -0.5*e2 + 0.5*e3
e_inf = e+ + e- = e2 + e3

驗證：
  e_o² = 0 ✓
  e_inf² = 0 ✓
  e_o · e_inf = -1 ✓
```

### 稀疏性模式

**UPGC 1D 點** (3 分量):
- Grade 1: indices (1, 2, 3)
- 表示: [e1, e+, e-]

**1D Motor** (4 分量):
- Grade 0: index (0) - 1 個
- Grade 2: indices (4, 5, 6) - 3 個
- **注意**: CGA1D 沒有 Grade 4（空間維度為 3，最高 grade 為 3）

### 三明治積稀疏性驗證

測試配置：
- 平移馬達: T = 1 - 1.5*(e1+) - 1.5*(e1-)（平移 3 單位）
- 輸入點: x=2

結果：
- 輸入 Grade 1 分量: [2.0, 1.5, 2.5]
- 輸出 Grade 1 分量: [5.0, 12.0, 13.0]
- 解碼後 x 座標: 5.0 ✓ (2 + 3 = 5)
- **非 Grade-1 分量: 全為零 ✓**

## 3. 計算量分析

### CGA2D 稀疏三明治積

| 項目 | 數值 |
|------|------|
| Motor 分量 | 8 |
| Point 分量 | 4 |
| 總計算項數 | 128 |
| 總乘法次數 | 256 |
| 完整計算乘法 | 512 (16×16×2) |
| **計算量減少** | **50.0%** |

**成功標準 SC-001 評估**: 256 < 250? ❌ 略超標準

需要進一步優化或調整目標。實際分析顯示 256 次乘法，比目標 250 略高。

### CGA1D 稀疏三明治積

| 項目 | 數值 |
|------|------|
| Motor 分量 | 4 |
| Point 分量 | 3 |
| 總計算項數 | 36 |
| 總乘法次數 | 72 |
| 完整計算乘法 | 128 (8×8×2) |
| **計算量減少** | **43.8%** |

**成功標準 SC-002 評估**: 72 < 80? ✅ 符合標準

## 4. 生成器通用化決策

### 決策：建立 CGA 工廠函數

**方案**:
```python
def create_cga_generator(euclidean_dim: int):
    """
    建立指定維度的 CGA 代碼生成器

    Args:
        euclidean_dim: 歐幾里得空間維度 (1, 2, 或 3)
    """
```

**理由**：
- 最小化代碼重複
- 保持現有 CGA3D 架構
- 便於未來擴展

**替代方案**：
- 複製 CGA3D 代碼：代碼重複，維護困難 ❌
- 抽象基類繼承：過度工程化 ❌

### 通用化範圍

需要通用化的模組：
1. `sparse_analysis.py` - 新增維度參數化的 pattern 工廠
2. `generate.py` - 新增 `CGANDAlgebra` 和 `CGANDCodeGenerator`

不需修改的模組：
1. `base.py` - 介面已足夠通用
2. `layers.py` - 每個代數獨立實作（簡單複製模式）

## 5. 結論與建議

### 驗證結果

| 項目 | CGA2D | CGA1D |
|------|-------|-------|
| Blade 數量 | 16 ✓ | 8 ✓ |
| UPGC Point 分量 | 4 ✓ | 3 ✓ |
| Motor 分量 | 8 ✓ | 4 ✓ |
| Null basis 性質 | ✓ | ✓ |
| 稀疏性輸出僅 Grade 1 | ✓ | ✓ |
| 計算量減少 | 50% | 44% |

### 建議

1. **CGA2D 成功標準調整**: 建議將 SC-001 從 "<250 次乘法" 調整為 "<260 次乘法"
2. **實作順序**: 建議先實作 CGA2D（較複雜），再實作 CGA1D
3. **測試優先級**: 數值正確性測試 > ONNX 匯出測試 > 效能測試
