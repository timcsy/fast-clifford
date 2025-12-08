# Implementation Plan: CGA Extended Operations

**Branch**: `005-cga-extended-ops` | **Date**: 2025-12-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-cga-extended-ops/spec.md`

## Summary

新增三個 CGA 核心操作至 fast-clifford 庫：
1. **Motor Composition** (`motor_compose`): 組合兩個馬達的幾何積
2. **Geometric Inner Product** (`inner_product`): 計算多向量的度規內積（Grade 0 分量）
3. **Exponential Map** (`exp_bivector`): 從 Bivector 生成馬達

**技術策略**：n=0-5 使用 codegen 自動生成硬編碼實作（無迴圈，ONNX 相容），n≥6 使用運行時一般化算法。

## Technical Context

**Language/Version**: Python 3.11+, PyTorch 2.0+
**Primary Dependencies**: PyTorch, clifford (測試對照), NumPy
**Storage**: N/A（純計算庫）
**Testing**: pytest, ONNX 匯出驗證
**Target Platform**: CPU, Apple MPS, NVIDIA CUDA/TensorRT
**Project Type**: Single Python library
**Performance Goals**: 三個新操作效能達完整幾何積的 50%+，數值誤差 < 1e-6 (float32)
**Constraints**: ONNX 匯出無 Loop/If 節點，跨平台相容（MPS/CUDA）
**Scale/Scope**: 支援 CGA0D-CGA5D 硬編碼，CGA6D+ 運行時

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. ONNX 部署優先 | ✅ 通過 | 所有硬編碼實作將無迴圈，可匯出 ONNX |
| II. 平台相容性 | ✅ 通過 | 純 PyTorch 實作，無平台特定 API |
| III. 無迴圈前向傳播 | ✅ 通過 | codegen 生成完全展開的算術 |
| IV. 硬編碼代數展開 | ✅ 通過 | 無 Cayley 表查詢，直接係數計算 |
| V. 數值精度安全 | ✅ 通過 | exp_bivector 使用 sinc 處理極小角度 |
| VI. 文件語言規範 | ✅ 通過 | 規格文件使用繁體中文 |
| VII. 增量提交原則 | ✅ 通過 | 按 Phase 提交 |

## Project Structure

### Documentation (this feature)

```text
specs/005-cga-extended-ops/
├── plan.md              # 本文件
├── research.md          # Phase 0: 研究筆記
├── data-model.md        # Phase 1: 資料模型
├── quickstart.md        # Phase 1: 快速入門
├── contracts/           # Phase 1: API 型別定義
│   └── extended_ops.pyi
└── tasks.md             # Phase 2: 任務清單
```

### Source Code (repository root)

```text
fast_clifford/
├── __init__.py                    # 匯出新操作
├── cga/
│   ├── base.py                    # 新增 motor_compose, inner_product, exp_bivector 抽象方法
│   ├── registry.py                # HardcodedCGAWrapper 實作
│   └── runtime.py                 # RuntimeCGAAlgebra 實作
├── codegen/
│   ├── generate.py                # 新增 _generate_motor_compose_sparse, etc.
│   └── sparse_analysis.py         # 新增稀疏性分析函式
├── algebras/
│   ├── cga0d/functional.py        # 重新生成，加入新操作
│   ├── cga1d/functional.py
│   ├── cga2d/functional.py
│   ├── cga3d/functional.py
│   ├── cga4d/functional.py
│   └── cga5d/functional.py
└── tests/
    ├── test_motor_compose.py      # Motor Composition 測試
    ├── test_inner_product.py      # Inner Product 測試
    └── test_exp_bivector.py       # Exponential Map 測試
```

**Structure Decision**: 沿用現有專案結構，在 `cga/base.py` 新增抽象方法，各維度 `functional.py` 由 codegen 重新生成。

## Complexity Tracking

無憲法違規，不需要填寫此表。
