# fast-clifford

High-performance Conformal Geometric Algebra (CGA) library for PyTorch, optimized for deep learning and ONNX/TensorRT deployment.

## Features

- **Unified Interface**: `CGA(n)` and `Cl(p,q,r)` factory functions for any dimension
- **Multi-dimensional Support**: CGA0D to CGA5D (hardcoded), CGA6D+ (runtime)
- **Hardware Acceleration**: CPU, Apple MPS, NVIDIA CUDA
- **ONNX Compatible**: Loop-free operations for TensorRT deployment
- **High Performance**: Up to 284x faster than clifford library
- **Operator Overloading**: Intuitive Python operators (`*`, `^`, `|`, `<<`, `>>`, `@`, `~`)
- **Complete Algebra**: Geometric product, wedge, contractions, dual, exponential map, and more

## Supported Algebras

| Algebra | Signature | Blades | Point | EvenVersor | Bivector | Implementation |
|---------|-----------|--------|-------|------------|----------|----------------|
| CGA0D | Cl(1,1) | 4 | 2 | 2 | 1 | Hardcoded |
| CGA1D | Cl(2,1) | 8 | 3 | 4 | 3 | Hardcoded |
| CGA2D | Cl(3,1) | 16 | 4 | 8 | 6 | Hardcoded |
| CGA3D | Cl(4,1) | 32 | 5 | 16 | 10 | Hardcoded |
| CGA4D | Cl(5,1) | 64 | 6 | 32 | 15 | Hardcoded |
| CGA5D | Cl(6,1) | 128 | 7 | 64 | 21 | Hardcoded |
| CGA6D+ | Cl(n+1,1) | 2^(n+2) | n+2 | varies | varies | Runtime |

## Installation

```bash
git clone https://github.com/timcsy/fast-clifford.git
cd fast-clifford
uv sync
```

Or with pip:

```bash
pip install torch numpy
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from fast_clifford import CGA

# Create CGA algebra for 3D Euclidean space
cga = CGA(3)  # CGA3D: Cl(4,1)

# Encode 3D points to conformal representation
points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
conformal_points = cga.cga_encode(points)  # Shape: (2, 5)

# Create an EvenVersor (rotation + translation + scaling)
versor = torch.randn(2, cga.even_versor_count)  # Shape: (2, 16)

# Apply transformation: result = V x point x reverse(V)
transformed = cga.sandwich_product_sparse(versor, conformal_points)

# Decode back to 3D Euclidean space
result = cga.cga_decode(transformed)  # Shape: (2, 3)
```

### Operator Overloading (Multivector Class)

```python
from fast_clifford import CGA
import torch

cga = CGA(3)

# Create Multivector wrappers
a = cga.multivector(torch.randn(32))
b = cga.multivector(torch.randn(32))

# Intuitive operators
c = a * b      # Geometric product
c = a ^ b      # Outer product (wedge)
c = a | b      # Inner product
c = a << b     # Left contraction
c = a >> b     # Right contraction
c = ~a         # Reverse
c = a ** -1    # Inverse

# EvenVersor composition
ev1 = cga.even_versor(torch.randn(16))
ev2 = cga.even_versor(torch.randn(16))
ev3 = ev1 * ev2  # Composition (returns EvenVersor)

# Similitude (optimized for rotation + translation + scaling)
s1 = cga.similitude(torch.randn(16))
s2 = cga.similitude(torch.randn(16))
s3 = s1 * s2  # Composition (returns Similitude)

# Sandwich product
point = cga.cga_encode(torch.tensor([[1.0, 2.0, 3.0]]))
result = ev1 @ point  # ev1 x point x ~ev1
```

### Extended Operations (Functional API)

```python
from fast_clifford import CGA
import torch

cga = CGA(3)

# Full multivectors
a = torch.randn(32)  # 32 blades for CGA3D
b = torch.randn(32)

# Products
inner = cga.inner_product(a, b)        # Scalar product <a*b>_0
outer = cga.outer_product(a, b)        # Wedge product a ^ b
left = cga.left_contraction(a, b)      # Left contraction a << b
right = cga.right_contraction(a, b)    # Right contraction a >> b

# Unary operations
grade2 = cga.grade_select(a, 2)        # Extract bivector component
dual_a = cga.dual(a)                   # Poincare dual
unit_a = cga.normalize(a)              # Unit normalization
rev_a = cga.reverse_full(a)            # Reverse

# Exponential map (bivector -> EvenVersor)
bivector = torch.randn(cga.bivector_count)  # 10 components for CGA3D
rotor = cga.exp_bivector(bivector)     # Returns EvenVersor (16 components)

# EvenVersor operations
ev1 = torch.randn(16)
ev2 = torch.randn(16)
composed = cga.compose_even_versor(ev1, ev2)  # EvenVersor composition
rev_ev = cga.reverse_even_versor(ev1)         # EvenVersor reverse

# Similitude operations (faster, no transversion)
s1 = torch.randn(16)
s2 = torch.randn(16)
composed_s = cga.compose_similitude(s1, s2)
point = cga.cga_encode(torch.tensor([[1.0, 2.0, 3.0]]))
result = cga.sandwich_product_similitude(s1, point)

# Structure normalization for Similitude
normalized = cga.structure_normalize(s1)  # Hard normalization
soft_norm = cga.soft_structure_normalize(s1, strength=0.1)  # Gradient-friendly
ste_norm = cga.structure_normalize_ste(s1)  # Straight-through estimator
```

### PyTorch Layers

```python
import torch
from fast_clifford import CGA

cga = CGA(3)

# Pre-built layers
transform_layer = cga.get_transform_layer()  # Sandwich product layer
encoder = cga.get_encoder()                   # Euclidean -> CGA point
decoder = cga.get_decoder()                   # CGA point -> Euclidean
pipeline = cga.get_transform_pipeline()       # Complete pipeline

# Use in neural network
class GeometricNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = CGA(3).get_transform_pipeline()
        self.versor_net = torch.nn.Linear(64, 16)

    def forward(self, features, points):
        versors = self.versor_net(features)
        return self.pipeline(versors, points)
```

### ONNX Export

```python
import torch
from fast_clifford import CGA

cga = CGA(3)
layer = cga.get_transform_layer()

versor = torch.randn(1, 16)
point = torch.randn(1, 5)

torch.onnx.export(
    layer,
    (versor, point),
    "cga3d_transform.onnx",
    input_names=["versor", "point"],
    output_names=["output"],
    dynamic_axes={
        "versor": {0: "batch"},
        "point": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=17,
)
```

### Direct Module Access

```python
# Access hardcoded modules directly for maximum performance
from fast_clifford.algebras import cga3d

versor = torch.randn(1024, 16)
point = cga3d.cga_encode(torch.randn(1024, 3))

# All operations available as functions
result = cga3d.sandwich_product_sparse(versor, point)
composed = cga3d.compose_even_versor(versor, versor)
inner = cga3d.inner_product_full(torch.randn(32), torch.randn(32))
```

## API Reference

### Factory Functions

```python
from fast_clifford import CGA, Cl

# Create by Euclidean dimension
cga = CGA(n)  # n=0..5 hardcoded, n>=6 runtime

# Create by Clifford signature
cga = Cl(p, q, r=0)  # e.g., Cl(4,1) for CGA3D
```

### CGAAlgebraBase Properties

| Property | Description |
|----------|-------------|
| `euclidean_dim` | Euclidean dimension n |
| `blade_count` | Total blades 2^(n+2) |
| `point_count` | CGA point components (n+2) |
| `even_versor_count` | EvenVersor components |
| `similitude_count` | Similitude components (= even_versor_count) |
| `bivector_count` | Bivector components |
| `max_grade` | Maximum grade (n+2) |
| `signature` | Clifford signature tuple |
| `clifford_notation` | e.g., "Cl(4,1,0)" |

### Core Operations

| Method | Input Shape | Output Shape | Description |
|--------|-------------|--------------|-------------|
| `cga_encode(x)` | (..., n) | (..., n+2) | Euclidean -> CGA point |
| `cga_decode(point)` | (..., n+2) | (..., n) | CGA point -> Euclidean |
| `geometric_product_full(a, b)` | (..., blades) | (..., blades) | Full geometric product |
| `sandwich_product_sparse(ev, point)` | (..., ev), (..., point) | (..., point) | Optimized V x X x ~V |
| `reverse_full(mv)` | (..., blades) | (..., blades) | Multivector reverse |
| `reverse_even_versor(ev)` | (..., ev) | (..., ev) | EvenVersor reverse |

### EvenVersor & Similitude Operations

| Method | Description |
|--------|-------------|
| `compose_even_versor(v1, v2)` | EvenVersor composition v1 x v2 |
| `compose_similitude(s1, s2)` | Similitude composition (optimized) |
| `sandwich_product_even_versor(v, point)` | General versor sandwich |
| `sandwich_product_similitude(s, point)` | Similitude sandwich (optimized) |
| `structure_normalize(s)` | Hard structure normalization |
| `soft_structure_normalize(s, strength)` | Gradient-friendly normalization |
| `structure_normalize_ste(s)` | Straight-through estimator |

### Extended Operations

| Method | Description |
|--------|-------------|
| `inner_product(a, b)` | Scalar product `<a*b>_0` |
| `outer_product(a, b)` | Wedge product `a ^ b` |
| `left_contraction(a, b)` | Left contraction `a << b` |
| `right_contraction(a, b)` | Right contraction `a >> b` |
| `grade_select(mv, k)` | Extract grade-k component |
| `dual(mv)` | Poincare duality |
| `normalize(mv)` | Unit normalization |
| `exp_bivector(B)` | Bivector exponential `exp(B)` |

### Unified API (Static Routing)

| Method | Description |
|--------|-------------|
| `compose(v1, v2, versor_type)` | Routes to compose_even_versor or compose_similitude |
| `sandwich_product(v, point, versor_type)` | Routes to appropriate sandwich |
| `reverse(v, versor_type)` | Routes to reverse_full or reverse_even_versor |

### Multivector Factory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `multivector(data)` | Multivector | Wrap tensor as Multivector |
| `even_versor(data)` | EvenVersor | Wrap tensor as EvenVersor |
| `similitude(data)` | Similitude | Wrap tensor as Similitude |
| `point(x)` | Multivector | Create CGA point from Euclidean |

### Multivector Operators

| Operator | Operation |
|----------|-----------|
| `a * b` | Geometric product |
| `a ^ b` | Outer product (wedge) |
| `a \| b` | Inner product |
| `a << b` | Left contraction |
| `a >> b` | Right contraction |
| `m @ x` | Sandwich product |
| `~a` | Reverse |
| `a ** -1` | Inverse |
| `a + b`, `a - b` | Addition, subtraction |
| `-a` | Negation |
| `a / s` | Scalar division |

### Layer Classes

| Class | Description |
|-------|-------------|
| `CliffordTransformLayer` | Sandwich product layer |
| `CGAEncoder` | Euclidean -> CGA point encoder |
| `CGADecoder` | CGA point -> Euclidean decoder |
| `CGAPipeline` | Complete transform pipeline |

### Layer Factory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_transform_layer()` | nn.Module | Sandwich product layer |
| `get_encoder()` | nn.Module | CGA encoder |
| `get_decoder()` | nn.Module | CGA decoder |
| `get_transform_pipeline()` | nn.Module | Complete pipeline |

## Performance

### Benchmark vs clifford library (batch=1024)

| Algebra | Geometric Product | Sandwich Product |
|---------|-------------------|------------------|
| CGA1D | 28x faster | **284x faster** |
| CGA2D | 6x faster | **249x faster** |
| CGA3D | 4x faster | **145x faster** |
| CGA4D | - | **24x faster** |
| CGA5D | - | **27x faster** |

### CPU Throughput (M1 Pro)

| Algebra | Points/sec (batch=16384) |
|---------|--------------------------|
| CGA1D | 40.2M |
| CGA2D | 10.8M |
| CGA3D | 2.7M |

### Sparsity Optimization

| Algebra | Naive Muls | Sparse Muls | Reduction |
|---------|------------|-------------|-----------|
| CGA1D | 96 | 72 | 25% |
| CGA2D | 512 | 256 | 50% |
| CGA3D | 4096 | 1600 | 61% |

### Optimization Techniques

- **Hardcoded Operations**: No Cayley table lookups for n<=5
- **Sparse Representation**: Only compute non-zero blade products
- **JIT Compilation**: `@torch.jit.script` optimization
- **Loop-free**: All operations unrolled for ONNX compatibility
- **Automatic Sparsity**: 25-61% computation reduction

## Testing

```bash
# Run all tests (588 tests)
uv run pytest fast_clifford/tests/ -v

# Run specific dimension tests
uv run pytest fast_clifford/tests/cga3d/ -v

# Run extended operations tests
uv run pytest fast_clifford/tests/test_extended_ops.py -v

# Run with coverage
uv run pytest --cov=fast_clifford
```

## Project Structure

```
fast_clifford/
├── __init__.py              # CGA, Cl, Multivector, EvenVersor, Similitude
├── algebras/
│   ├── cga0d/               # Cl(1,1) - 0D CGA (4 blades)
│   ├── cga1d/               # Cl(2,1) - 1D CGA (8 blades)
│   ├── cga2d/               # Cl(3,1) - 2D CGA (16 blades)
│   ├── cga3d/               # Cl(4,1) - 3D CGA (32 blades)
│   ├── cga4d/               # Cl(5,1) - 4D CGA (64 blades)
│   └── cga5d/               # Cl(6,1) - 5D CGA (128 blades)
├── cga/
│   ├── __init__.py          # CGA(), Cl() factory functions
│   ├── base.py              # CGAAlgebraBase abstract class
│   ├── registry.py          # HardcodedCGAWrapper (n<=5)
│   ├── runtime.py           # RuntimeCGAAlgebra (n>=6)
│   ├── layers.py            # CliffordTransformLayer, CGAEncoder, etc.
│   └── multivector.py       # Multivector, EvenVersor, Similitude classes
├── codegen/
│   ├── cga_factory.py       # Algebra factory utilities
│   ├── generate.py          # Code generator
│   └── sparse_analysis.py   # Sparsity analysis
└── tests/                   # Test suites (588 tests)
```

## Use Cases

- **Robotics**: Rigid body transformations, kinematics, motion planning
- **Computer Graphics**: 3D rotations, translations, reflections, interpolation
- **Deep Learning**: Geometric neural networks (equivariant networks)
- **Physics Simulation**: Conformal transformations, Minkowski space
- **Computer Vision**: Camera calibration, 3D reconstruction

## License

MIT

## References

- [Geometric Algebra for Computer Science](http://geometricalgebra.org/)
- [clifford - Numerical Geometric Algebra Module](https://github.com/pygae/clifford)
- [A Guided Tour to the Plane-Based Geometric Algebra PGA](https://bivector.net/PGA4CS.html)
