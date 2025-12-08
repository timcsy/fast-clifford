# fast-clifford

High-performance Conformal Geometric Algebra (CGA) library for PyTorch, optimized for deep learning and ONNX/TensorRT deployment.

## Features

- **Unified interface**: `CGA(n)` and `Cl(p,q,r)` factory functions
- **Multi-dimensional CGA support**: CGA0D to CGA5D (hardcoded), CGA6D+ (runtime)
- **Hardware acceleration**: CPU, Apple MPS, NVIDIA CUDA
- **ONNX compatible**: Loop-free operations for TensorRT deployment
- **High performance**: Up to 284x faster than clifford library

## Supported Algebras

| Algebra | Signature | Blades | Point | Motor | Algorithm |
|---------|-----------|--------|-------|-------|-----------|
| CGA0D | Cl(1,1) | 4 | 2 | 2 | Hardcoded |
| CGA1D | Cl(2,1) | 8 | 3 | 4 | Hardcoded |
| CGA2D | Cl(3,1) | 16 | 4 | 7 | Hardcoded |
| CGA3D | Cl(4,1) | 32 | 5 | 16 | Hardcoded |
| CGA4D | Cl(5,1) | 64 | 6 | 31 | Hardcoded |
| CGA5D | Cl(6,1) | 128 | 7 | 64 | Hardcoded |
| CGA6D+ | Cl(n+1,1) | 2^(n+2) | n+2 | varies | Runtime |

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

### Unified Interface (Recommended)

```python
import torch
from fast_clifford import CGA, Cl

# Create CGA by Euclidean dimension
cga3d = CGA(3)  # CGA3D Cl(4,1,0)
print(f"Blades: {cga3d.blade_count}")  # 32

# Or create by Clifford signature
cga3d = Cl(4, 1)  # Same as CGA(3)

# Encode 3D point to UPGC
x = torch.tensor([[1.0, 2.0, 3.0]])
point = cga3d.upgc_encode(x)

# Create motor and apply transformation
motor = torch.randn(1, 16)  # 16 motor components
transformed = cga3d.sandwich_product_sparse(motor, point)

# Decode back to 3D
result = cga3d.upgc_decode(transformed)
print(result)  # tensor([[x', y', z']])
```

### Direct Module Access

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

### Unified Interface

```python
from fast_clifford import CGA, Cl

# Create by Euclidean dimension
cga = CGA(n)  # n=0..5 hardcoded, n>=6 runtime

# Create by Clifford signature
cga = Cl(p, q, r=0)  # Cl(n+1, 1) for CGA(n)
```

**CGAAlgebraBase Properties:**
- `euclidean_dim` - Euclidean dimension n
- `blade_count` - Total blades (2^(n+2))
- `point_count` - UPGC point components (n+2)
- `motor_count` - Motor components
- `clifford_notation` - e.g., "Cl(4,1,0)"

**CGAAlgebraBase Methods:**
- `upgc_encode(x)` - Euclidean to UPGC point
- `upgc_decode(point)` - UPGC to Euclidean point
- `geometric_product_full(a, b)` - Full geometric product
- `sandwich_product_sparse(motor, point)` - Optimized M × X × M̃
- `reverse_full(mv)` - Multivector reverse
- `reverse_motor(motor)` - Motor reverse
- `get_care_layer()` - Get CareLayer module
- `get_encoder()` - Get UPGC encoder module
- `get_decoder()` - Get UPGC decoder module
- `get_transform_pipeline()` - Get complete pipeline

### Direct Module Access

Each algebra module (`cga0d`, `cga1d`, `cga2d`, `cga3d`, `cga4d`, `cga5d`) provides:

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
import torch
from fast_clifford import CGA

# Works uniformly across all dimensions
for n in [0, 1, 2, 3, 4, 5, 6]:
    cga = CGA(n)
    x = torch.randn(1, n) if n > 0 else torch.zeros(1, 0)
    point = cga.upgc_encode(x)
    motor = torch.randn(1, cga.motor_count)
    result = cga.sandwich_product_sparse(motor, point)
    print(f"CGA{n}D: {cga.blade_count} blades, {cga.motor_count} motor components")
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
├── __init__.py         # CGA, Cl unified interface exports
├── algebras/
│   ├── cga0d/          # Cl(1,1) - 0D CGA
│   ├── cga1d/          # Cl(2,1) - 1D CGA
│   ├── cga2d/          # Cl(3,1) - 2D CGA
│   ├── cga3d/          # Cl(4,1) - 3D CGA
│   ├── cga4d/          # Cl(5,1) - 4D CGA
│   └── cga5d/          # Cl(6,1) - 5D CGA
├── cga/                # Unified interface
│   ├── base.py         # CGAAlgebraBase abstract class
│   ├── registry.py     # HardcodedCGAWrapper
│   └── runtime.py      # RuntimeCGAAlgebra (6D+)
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
