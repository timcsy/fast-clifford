# fast-clifford

High-performance CGA (Conformal Geometric Algebra) code generator for PyTorch with ONNX/TensorRT deployment support.

## Features

- Code generation for Clifford algebra operations
- Optimized sparse implementations for CGA Cl(4,1)
- ONNX-exportable PyTorch modules (no Loop nodes)
- Cross-platform: Apple MPS and NVIDIA CUDA support

## Installation

```bash
git clone <repo_url>
cd fast-clifford
uv sync
```

## Quick Start

```python
import torch
from fast_clifford.algebras.cga3d import sandwich_product_sparse

# Motor (16 components) and UPGC Point (5 components)
motor = torch.randn(1, 16)
point = torch.randn(1, 5)

# Compute sandwich product: M × X × M̃
result = sandwich_product_sparse(motor, point)
```

## License

MIT
