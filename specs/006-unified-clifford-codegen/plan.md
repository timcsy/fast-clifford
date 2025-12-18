# Implementation Plan: Unified Cl(p,q,0) Codegen System

**Branch**: `006-unified-clifford-codegen` | **Date**: 2025-12-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/006-unified-clifford-codegen/spec.md`

## Summary

建立統一的 Clifford 代數 Cl(p,q,0) 代碼生成系統，支援：
- VGA(n) = Cl(n, 0) - 純向量代數
- CGA(n) = Cl(n+1, 1) - 共形幾何代數
- 任意 Cl(p, q) 其中 r=0

技術方案：
- 所有 p+q ≤ 9 的代數預生成並提交到 repo（約 55 個）
- p+q > 9 使用 Bott 週期性 Cl(p+8, q) ≅ Cl(p, q) ⊗ M₁₆(ℝ)
- PGA(n) = Cl(n, 0, 1) 透過 CGA 嵌入實作
- 統一 API：Rotor 命名、count_* 屬性、靜態路由加速

## Technical Context

**Language/Version**: Python 3.11+, PyTorch 2.0+
**Primary Dependencies**: PyTorch, NumPy (preprocessing only), clifford (test validation)
**Storage**: N/A（純計算庫，無持久化需求）
**Testing**: pytest, clifford 庫對照驗證
**Target Platform**: CPU, Apple MPS, NVIDIA CUDA/TensorRT
**Project Type**: single（Python 計算庫）
**Performance Goals**:
- sandwich_rotor 相對 clifford 庫加速 > 10x（p+q ≤ 4）
- Rotor 加速運算優於通用 multivector 運算 20%+
**Constraints**:
- 所有運算 loop-free（ONNX 相容）
- 支援 `@torch.jit.script` 優化
- blade_count > 2^14 時輸出記憶體警告
**Scale/Scope**:
- 預生成 55 個代數（p+q ≤ 9）
- 45+ 個功能需求（FR-001 ~ FR-045）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ PASS | FR-041 要求所有運算 loop-free；FR-042 要求靜態路由 |
| II. 平台相容性 | ✅ PASS | 純 PyTorch 實作，支援 MPS/CUDA |
| III. 無迴圈前向傳播 | ✅ PASS | 硬編碼展開 + 張量操作，無動態迴圈 |
| IV. 硬編碼代數展開 | ✅ PASS | FR-043/044 要求 ClCodeGenerator 生成硬編碼係數 |
| V. 數值精度安全 | ✅ PASS | 繼承現有 float32 強制轉換機制 |
| VI. 文件語言規範 | ✅ PASS | 所有規格文件使用繁體中文 |
| VII. 增量提交原則 | ✅ PASS | 按 Phase 分階段提交 |

**禁止技術檢查**:
- ❌ Triton: 不使用
- ❌ CUDA C++ 擴充: 不使用
- ❌ Taichi: 不使用
- ❌ 帶迴圈的 torch.jit.script: 不使用
- ❌ 自訂 CUDA 核心: 不使用

**GATE 結果**: ✅ 通過，可進入 Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/006-unified-clifford-codegen/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API type stubs)
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
fast_clifford/
├── clifford/                       # 新增：統一 Clifford 介面
│   ├── __init__.py                 # Cl(), VGA(), CGA(), PGA() 工廠
│   ├── base.py                     # CliffordAlgebraBase 抽象基類
│   ├── registry.py                 # HardcodedClWrapper
│   ├── runtime.py                  # RuntimeCliffordAlgebra（fallback）
│   ├── bott.py                     # BottPeriodicityAlgebra
│   ├── multivector.py              # Multivector, Rotor 類別
│   ├── layers.py                   # PyTorch nn.Module layers
│   └── specializations/
│       ├── __init__.py
│       ├── vga.py                  # VGA 特化
│       ├── cga.py                  # CGA 特化（encode/decode）
│       └── pga.py                  # PGA 嵌入
├── algebras/
│   └── generated/                  # 新增：自動生成的代數
│       ├── __init__.py             # 動態載入器
│       ├── cl_1_0/                 # VGA1D
│       ├── cl_2_0/                 # VGA2D
│       ├── cl_3_0/                 # VGA3D
│       ├── cl_2_1/                 # CGA0D
│       ├── cl_3_1/                 # CGA1D
│       ├── cl_4_1/                 # CGA2D (原 cga2d)
│       ├── cl_5_1/                 # CGA3D (原 cga3d)
│       ├── cl_6_1/                 # CGA4D (原 cga4d)
│       ├── cl_7_1/                 # CGA5D (原 cga5d)
│       ├── cl_8_1/                 # CGA6D
│       ├── cl_9_0/                 # VGA9D (最大硬編碼)
│       └── ...                     # 其他 p+q ≤ 9 組合
├── codegen/
│   ├── clifford_factory.py         # 新增：通用 Cl(p,q,r) 建立
│   ├── generator.py                # 重構：ClCodeGenerator
│   ├── bott_generator.py           # 新增：Bott 週期性生成器
│   └── sparse_analysis.py          # 重構：參數化 (p,q,r)
└── tests/
    ├── generated/                  # 新增：生成代數的測試
    ├── test_clifford_interface.py  # 新增：統一介面測試
    ├── test_bott.py                # 新增：Bott 週期性測試
    └── test_pga_embedding.py       # 新增：PGA 嵌入測試
```

**Structure Decision**: Single project 結構。新增 `clifford/` 目錄作為統一介面層，`algebras/generated/` 存放自動生成的代數。舊 `cga/` 和 `algebras/cga*d/` 目錄將被刪除（不向後相容）。

## Complexity Tracking

> 無違規需要記錄

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

---

## Post-Design Constitution Check

*Re-evaluated after Phase 1 design completion.*

| 原則 | 狀態 | 驗證項目 |
|------|------|----------|
| I. ONNX 部署優先 | ✅ PASS | contracts/clifford_algebra.pyi 定義 loop-free 運算 |
| II. 平台相容性 | ✅ PASS | 純 PyTorch tensor 操作，無平台特定 API |
| III. 無迴圈前向傳播 | ✅ PASS | ClCodeGenerator 生成硬編碼展開 |
| IV. 硬編碼代數展開 | ✅ PASS | research.md 確認不使用 Cayley 表 |
| V. 數值精度安全 | ✅ PASS | layers 繼承 float32 轉換機制 |
| VI. 文件語言規範 | ✅ PASS | 所有 .md 文件使用繁體中文 |
| VII. 增量提交原則 | ✅ PASS | 按 Phase 分階段實作 |

**Post-Design GATE**: ✅ 通過，可進入 Phase 2 (tasks)

---

## Generated Artifacts

| 檔案 | 狀態 | 說明 |
|------|------|------|
| `plan.md` | ✅ 完成 | 本文件 |
| `research.md` | ✅ 完成 | 技術研究與決策 |
| `data-model.md` | ✅ 完成 | 資料模型定義 |
| `contracts/clifford_algebra.pyi` | ✅ 完成 | API 類型定義 |
| `quickstart.md` | ✅ 完成 | 快速入門指南 |
| `tasks.md` | ⏳ 待生成 | 執行 `/speckit.tasks` |

---

## Next Steps

執行 `/speckit.tasks` 生成任務清單。
