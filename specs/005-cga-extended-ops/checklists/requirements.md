# Requirements Checklist: CGA Extended Operations

## User Stories

- [x] US1: Motor Composition - 有明確的使用場景（變換串接）
- [x] US2: Geometric Inner Product - 有明確的使用場景（Attention/Loss）
- [x] US3: Exponential Map - 有明確的使用場景（馬達生成）
- [x] US4: High-Dimensional Runtime - 有明確的使用場景（高維研究）
- [x] 每個 User Story 都有 Priority 標記
- [x] 每個 User Story 都有獨立可測試的驗收場景

## Functional Requirements

- [x] FR-001 to FR-004: Motor Composition 需求完整
- [x] FR-005 to FR-008: Inner Product 需求完整
- [x] FR-009 to FR-012: Exponential Map 需求完整
- [x] FR-013 to FR-015: 統一介面需求完整
- [x] FR-016 to FR-017: ONNX 相容性需求
- [x] FR-018 to FR-019: PyTorch 整合需求
- [x] 使用 MUST/SHOULD/MAY 清晰標記需求層級

## Success Criteria

- [x] SC-001: 效能標準（達完整幾何積 50%）
- [x] SC-002: 數值精度標準（float32: 1e-6, float64: 1e-10）
- [x] SC-003: 數值穩定性標準（無 NaN/Inf）
- [x] SC-004: ONNX 相容性標準（無 Loop/If）
- [x] SC-005: 測試覆蓋率標準（90%）
- [x] SC-006: API 一致性標準

## Edge Cases

- [x] 零向量輸入處理
- [x] 極小角度處理（θ < 1e-10）
- [x] 非正規化馬達處理
- [x] 混合精度支援（float32/float64）
- [x] 批次維度支援

## Key Entities

- [x] Motor 定義清晰
- [x] Bivector 定義清晰
- [x] Multivector 定義清晰
- [x] Metric Signature 定義清晰

## Assumptions

- [x] 依賴項明確（PyTorch 2.0+, clifford）
- [x] 實作方式明確（codegen 自動生成）
- [x] 運行時策略明確（scatter_add/gather）
