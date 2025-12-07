# fast-clifford

High-performance Conformal Geometric Algebra (CGA) library for PyTorch, optimized for deep learning and ONNX/TensorRT deployment.

## Features

- **Multi-dimensional CGA support**: CGA1D to CGA5D (Cl(2,1) to Cl(6,1))
- **Hardware acceleration**: CPU, Apple MPS, NVIDIA CUDA
- **ONNX compatible**: Loop-free operations for TensorRT deployment
- **High performance**: Up to 284x faster than clifford library

## Supported Algebras

| Algebra | Signature | Blades | Point | Motor | Peak Throughput |
|---------|-----------|--------|-------|-------|-----------------|
| CGA1D | Cl(2,1) | 8 | 3 | 4 | 40M pts/sec |
| CGA2D | Cl(3,1) | 16 | 4 | 7 | 10.8M pts/sec |
| CGA3D | Cl(4,1) | 32 | 5 | 16 | 2.7M pts/sec |
| CGA4D | Cl(5,1) | 64 | 6 | 31 | 521K pts/sec |
| CGA5D | Cl(6,1) | 128 | 7 | 64 | 161K pts/sec |

## Installation

```bash
git clone https://github.com/timcsy/fast-clifford.git
cd fast-clifford
uv sync
```

Or with pip:

```bash
pip install torch numpy clifford
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from fast_clifford.algebras import cga3d

# Create a motor (rotation + translation) and a point
motor = torch.randn(1, 16)  # 16 motor components
point = cga3d.upgc_encode(torch.tensor([[1.0, 2.0, 3.0]]))  # 3D point -> UPGC

# Apply transformation: M × X × M̃
transformed = cga3d.sandwich_product_sparse(motor, point)

# Decode back to 3D
result = cga3d.upgc_decode(transformed)
print(result)  # tensor([[x', y', z']])
```

### Using PyTorch Layers

```python
import torch
from fast_clifford.algebras import cga3d

# Create transformation pipeline
pipeline = cga3d.CGA3DTransformPipeline()

# Batch processing
batch_size = 1024
motors = torch.randn(batch_size, 16)
points_3d = torch.randn(batch_size, 3)

# Transform points
transformed_3d = pipeline(motors, points_3d)
```

### ONNX Export

```python
import torch
from fast_clifford.algebras import cga3d

layer = cga3d.CGA3DCareLayer()
motor = torch.randn(1, 16)
point = torch.randn(1, 5)

torch.onnx.export(
    layer,
    (motor, point),
    "cga3d_transform.onnx",
    input_names=["motor", "point"],
    output_names=["output"],
    dynamic_axes={
        "motor": {0: "batch"},
        "point": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=17,
)
```

## API Reference

### Available Algebras

Each algebra module (`cga1d`, `cga2d`, `cga3d`, `cga4d`, `cga5d`) provides:

**Functions:**
- `geometric_product_full(a, b)` - Full geometric product
- `sandwich_product_sparse(motor, point)` - Optimized M × X × M̃
- `upgc_encode(x)` - Euclidean to UPGC point
- `upgc_decode(point)` - UPGC to Euclidean point
- `reverse_full(mv)` - Multivector reverse
- `reverse_motor(motor)` - Motor reverse

**PyTorch Layers:**
- `CGAxDCareLayer` - Sandwich product layer with precision handling
- `UPGCxDEncoder` - Euclidean to UPGC encoder
- `UPGCxDDecoder` - UPGC to Euclidean decoder
- `CGAxDTransformPipeline` - Complete encode-transform-decode pipeline

### Example: Multi-dimensional

```python
from fast_clifford.algebras import cga1d, cga2d, cga3d, cga4d, cga5d

# 1D transformation
motor_1d = torch.randn(1, 4)
point_1d = cga1d.upgc_encode(torch.tensor([[2.0]]))
result_1d = cga1d.sandwich_product_sparse(motor_1d, point_1d)

# 2D transformation
motor_2d = torch.randn(1, 7)
point_2d = cga2d.upgc_encode(torch.tensor([[1.0, 2.0]]))
result_2d = cga2d.sandwich_product_sparse(motor_2d, point_2d)

# 4D transformation
motor_4d = torch.randn(1, 31)
point_4d = cga4d.upgc_encode(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
result_4d = cga4d.sandwich_product_sparse(motor_4d, point_4d)
```

## Performance

### vs clifford library (batch size 1024)

| Algebra | Geometric Product | Sandwich Product |
|---------|-------------------|------------------|
| CGA1D | 28x faster | **284x faster** |
| CGA2D | 6x faster | **249x faster** |
| CGA3D | 4x faster | **145x faster** |
| CGA4D | - | **24x faster** |
| CGA5D | - | **27x faster** |

### Optimization Techniques

- **Hard-coded operations**: No Cayley table lookups
- **Sparse representation**: Only compute non-zero blade products
- **JIT compilation**: `@torch.jit.script` optimization
- **Loop-free**: All operations unrolled for ONNX compatibility

## Testing

```bash
# Run all tests
uv run pytest fast_clifford/tests/ -v

# Run specific algebra tests
uv run pytest fast_clifford/tests/cga3d/ -v

# Run with coverage
uv run pytest --cov=fast_clifford
```

## Project Structure

```
fast_clifford/
├── algebras/
│   ├── cga1d/          # Cl(2,1) - 1D CGA
│   ├── cga2d/          # Cl(3,1) - 2D CGA
│   ├── cga3d/          # Cl(4,1) - 3D CGA
│   ├── cga4d/          # Cl(5,1) - 4D CGA
│   └── cga5d/          # Cl(6,1) - 5D CGA
├── codegen/            # Code generation tools
│   ├── cga_factory.py  # Algebra factory
│   ├── generate.py     # Code generator
│   └── sparse_analysis.py
└── tests/              # Test suites
```

## Use Cases

- **Robotics**: Rigid body transformations, kinematics
- **Computer Graphics**: 3D rotations, translations, reflections
- **Deep Learning**: Geometric neural networks (CARE Transformer)
- **Physics Simulation**: Conformal transformations

## License

MIT

## References

- [Geometric Algebra for Computer Science](http://geometricalgebra.org/)
- [clifford - Numerical Geometric Algebra Module](https://github.com/pygae/clifford)
