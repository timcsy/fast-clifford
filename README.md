# fast-clifford

High-performance Clifford Algebra library for PyTorch, optimized for deep learning and ONNX/TensorRT deployment.

## Features

- **Unified Interface**: `Cl(p,q,r)`, `VGA(n)`, `CGA(n)`, `PGA(n)` factory functions
- **Complete Coverage**: 55 pre-generated algebras for p+q ≤ 9, Bott periodicity for higher dimensions
- **Hardware Acceleration**: CPU, Apple MPS, NVIDIA CUDA
- **ONNX Compatible**: Loop-free operations for TensorRT deployment
- **High Performance**: Up to 16x faster than clifford library
- **Operator Overloading**: Intuitive Python operators (`*`, `^`, `|`, `<<`, `>>`, `@`, `~`, `&`)
- **Complete Algebra**: Geometric product, wedge, contractions, dual, exponential map, and more

## Supported Algebras

| Type | Factory | Signature | Example | Implementation |
|------|---------|-----------|---------|----------------|
| VGA | `VGA(n)` | Cl(n, 0) | VGA(3) = Cl(3,0) | Hardcoded |
| CGA | `CGA(n)` | Cl(n+1, 1) | CGA(3) = Cl(4,1) | Hardcoded |
| PGA | `PGA(n)` | Cl(n, 0, 1) | PGA(3) = Cl(3,0,1) | Runtime (CGA embedding) |
| General | `Cl(p,q)` | Cl(p, q) | Cl(2,2) | Hardcoded (p+q ≤ 9) |
| High-dim | `Cl(p,q)` | Cl(p, q) | Cl(10,0) | Bott periodicity |

### Pre-generated Algebras (55 total)

All algebras with p+q ≤ 9 (up to 512 blades) are pre-generated with hardcoded operations.

| Blades | Algebras |
|--------|----------|
| 2 | Cl(1,0), Cl(0,1) |
| 4 | Cl(2,0), Cl(1,1), Cl(0,2) |
| 8 | Cl(3,0), Cl(2,1), Cl(1,2), Cl(0,3) |
| 16 | Cl(4,0), Cl(3,1), Cl(2,2), Cl(1,3), Cl(0,4) |
| 32 | Cl(5,0), Cl(4,1), Cl(3,2), Cl(2,3), Cl(1,4), Cl(0,5) |
| 64 | Cl(6,0), Cl(5,1), Cl(4,2), Cl(3,3), Cl(2,4), Cl(1,5), Cl(0,6) |
| 128 | Cl(7,0), Cl(6,1), Cl(5,2), Cl(4,3), Cl(3,4), Cl(2,5), Cl(1,6), Cl(0,7) |
| 256 | Cl(8,0), Cl(7,1), Cl(6,2), Cl(5,3), Cl(4,4), Cl(3,5), Cl(2,6), Cl(1,7), Cl(0,8) |
| 512 | Cl(9,0), Cl(8,1), Cl(7,2), Cl(6,3), Cl(5,4), Cl(4,5), Cl(3,6), Cl(2,7), Cl(1,8), Cl(0,9) |

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
from fast_clifford import Cl, VGA, CGA, PGA

# VGA - Vanilla Geometric Algebra (Euclidean)
vga = VGA(3)  # Cl(3,0) - 8 blades
v = vga.encode(torch.tensor([1.0, 2.0, 3.0]))
print(f"VGA(3): {vga.count_blade} blades")

# CGA - Conformal Geometric Algebra
cga = CGA(3)  # Cl(4,1) - 32 blades
point = cga.encode(torch.tensor([1.0, 2.0, 3.0]))
decoded = cga.decode(point)
print(f"CGA(3): {cga.count_blade} blades")

# PGA - Projective Geometric Algebra
pga = PGA(3)  # Cl(3,0,1) - 16 blades
p = pga.point(torch.tensor([1.0, 2.0, 3.0]))
print(f"PGA(3): {pga.count_blade} blades")

# General Clifford algebra
cl22 = Cl(2, 2)  # 16 blades
print(f"Cl(2,2): {cl22.count_blade} blades")
```

### Geometric Operations

```python
from fast_clifford import VGA
import torch

vga = VGA(3)

# Create basis vectors
e1 = vga.basis_vector(0)  # e1
e2 = vga.basis_vector(1)  # e2
e3 = vga.basis_vector(2)  # e3

# Geometric product
e12 = vga.geometric_product(e1, e2)  # e1 * e2 = e12

# Outer product (wedge)
bivector = vga.outer(e1, e2)  # e1 ^ e2

# Inner product
scalar = vga.inner(e1, e1)  # e1 · e1 = 1

# Reverse
rev = vga.reverse(e12)  # ~e12 = -e12
```

### Operator Overloading (Multivector Class)

```python
from fast_clifford import VGA
import torch

vga = VGA(3)

# Create Multivector wrappers
a = vga.multivector(torch.randn(8))
b = vga.multivector(torch.randn(8))

# Intuitive operators
c = a * b      # Geometric product
c = a ^ b      # Outer product (wedge)
c = a | b      # Inner product
c = a << b     # Left contraction
c = a >> b     # Right contraction
c = a & b      # Regressive product (meet)
c = ~a         # Reverse
c = a ** -1    # Inverse

# Rotor operations
r1 = vga.rotor(torch.randn(4))
r2 = vga.rotor(torch.randn(4))
r3 = r1 * r2   # Rotor composition

# Sandwich product
point = vga.multivector(torch.randn(8))
transformed = r1 @ point  # r1 * point * ~r1
```

### Exponential Map (Bivector -> Rotor)

```python
from fast_clifford import VGA
import torch
import math

vga = VGA(3)

# Create a rotation bivector (90 degrees around e12 plane)
B = torch.zeros(vga.count_bivector)
B[0] = math.pi / 4  # Half angle for rotation

# Exponential map: bivector -> rotor
rotor = vga.exp_bivector(B)

# Logarithm: rotor -> bivector
B_recovered = vga.log_rotor(rotor)

# Spherical interpolation
r1 = vga.exp_bivector(torch.zeros(vga.count_bivector))  # Identity
r2 = vga.exp_bivector(B)
r_mid = vga.slerp_rotor(r1, r2, 0.5)  # Halfway rotation
```

### CGA Transformations

```python
from fast_clifford import CGA
import torch

cga = CGA(3)

# Encode 3D points to conformal representation
points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
conformal_points = cga.encode(points)  # Shape: (2, 32)

# Create a rotor
rotor = torch.randn(2, cga.count_rotor)  # Shape: (2, 16)

# Apply transformation: result = R * point * ~R
transformed = cga.sandwich_rotor(rotor, conformal_points)

# Decode back to 3D Euclidean space
result = cga.decode(transformed)  # Shape: (2, 3)
```

### PGA Primitives

```python
from fast_clifford import PGA
import torch

pga = PGA(3)

# Create geometric primitives
p = pga.point(torch.tensor([1.0, 2.0, 3.0]))
d = pga.direction(torch.tensor([0.0, 0.0, 1.0]))
plane = pga.plane(torch.tensor([0.0, 0.0, 1.0, -5.0]))  # z = 5

# Create a line from two points
p1 = torch.tensor([0.0, 0.0, 0.0])
p2 = torch.tensor([1.0, 0.0, 0.0])
line = pga.line_from_points(p1, p2)
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

# Use in neural network
class GeometricNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cga = CGA(3)
        self.encoder = self.cga.get_encoder()
        self.decoder = self.cga.get_decoder()
        self.transform = self.cga.get_transform_layer()
        self.rotor_net = torch.nn.Linear(64, self.cga.count_rotor)

    def forward(self, features, points):
        rotors = self.rotor_net(features)
        encoded = self.encoder(points)
        transformed = self.transform(rotors, encoded)
        return self.decoder(transformed)
```

### ONNX Export

```python
import torch
from fast_clifford import CGA

cga = CGA(3)
layer = cga.get_transform_layer()

rotor = torch.randn(1, cga.count_rotor)
point = torch.randn(1, cga.count_blade)  # Full multivector

torch.onnx.export(
    layer,
    (rotor, point),
    "cga3d_transform.onnx",
    input_names=["rotor", "point"],
    output_names=["output"],
    dynamic_axes={
        "rotor": {0: "batch"},
        "point": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=17,
)
```

### Bott Periodicity (High Dimensions)

```python
from fast_clifford import Cl

# High-dimensional algebra using Bott periodicity
# Cl(10,0) -> Cl(2,0) ⊗ M_16(R)
cl10 = Cl(10, 0)
print(f"Cl(10,0): {cl10.count_blade} blades")  # 1024

# All operations work the same
e1 = cl10.basis_vector(0)
e2 = cl10.basis_vector(1)
product = cl10.geometric_product(e1, e2)
```

## API Reference

### Factory Functions

```python
from fast_clifford import Cl, VGA, CGA, PGA

# General Clifford algebra
algebra = Cl(p, q, r=0)  # Cl(p, q, r)

# Specialized algebras
vga = VGA(n)   # Cl(n, 0) - Vanilla/Euclidean
cga = CGA(n)   # Cl(n+1, 1) - Conformal
pga = PGA(n)   # Cl(n, 0, 1) - Projective
```

### Algebra Properties

| Property | Description |
|----------|-------------|
| `p` | Positive signature dimension |
| `q` | Negative signature dimension |
| `r` | Degenerate dimension |
| `count_blade` | Total blades 2^(p+q+r) |
| `count_rotor` | Rotor (even grade) components |
| `count_bivector` | Bivector (grade 2) components |
| `algebra_type` | 'vga', 'cga', 'pga', or 'general' |

### Core Operations

| Method | Description |
|--------|-------------|
| `geometric_product(a, b)` | Geometric product a * b |
| `outer(a, b)` | Outer product a ∧ b |
| `inner(a, b)` | Inner product a · b |
| `contract_left(a, b)` | Left contraction a ⌋ b |
| `contract_right(a, b)` | Right contraction a ⌊ b |
| `regressive(a, b)` | Regressive product (meet) a ∨ b |
| `reverse(mv)` | Reverse ~mv |
| `involute(mv)` | Grade involution |
| `conjugate(mv)` | Clifford conjugate |
| `dual(mv)` | Poincaré dual |
| `normalize(mv)` | Unit normalization |
| `inverse(mv)` | Multiplicative inverse |
| `select_grade(mv, k)` | Extract grade-k component |

### Rotor Operations

| Method | Description |
|--------|-------------|
| `compose_rotor(r1, r2)` | Rotor composition r1 * r2 |
| `reverse_rotor(r)` | Rotor reverse ~r |
| `sandwich_rotor(r, x)` | Sandwich r * x * ~r |
| `normalize_rotor(r)` | Normalize to unit rotor |
| `inverse_rotor(r)` | Rotor inverse |
| `exp_bivector(B)` | Bivector → Rotor |
| `log_rotor(r)` | Rotor → Bivector |
| `slerp_rotor(r1, r2, t)` | Spherical linear interpolation |

### VGA/CGA Specializations

| Method | Description |
|--------|-------------|
| `encode(x)` | Euclidean → algebra representation |
| `decode(mv)` | Algebra representation → Euclidean |
| `dim_euclidean` | Euclidean dimension |
| `count_point` | Point representation size (CGA) |

### PGA Primitives

| Method | Description |
|--------|-------------|
| `point(x)` | Create PGA point |
| `direction(v)` | Create ideal point (direction) |
| `plane(coeffs)` | Create plane from [a, b, c, d] |
| `line_from_points(p1, p2)` | Create line through two points |

### Multivector Operators

| Operator | Operation |
|----------|-----------|
| `a * b` | Geometric product |
| `a ^ b` | Outer product (wedge) |
| `a \| b` | Inner product |
| `a << b` | Left contraction |
| `a >> b` | Right contraction |
| `a & b` | Regressive product (meet) |
| `m @ x` | Sandwich product |
| `~a` | Reverse |
| `a ** -1` | Inverse |
| `a(k)` | Grade selection |

## Performance

### Benchmark vs clifford library (batch=1024)

| Algebra | Operation | fast-clifford | clifford | Speedup |
|---------|-----------|---------------|----------|---------|
| VGA(3) | geometric | 0.114ms | 1.841ms | **16.1x** |
| CGA(3) | geometric | 2.050ms | 6.255ms | **3.1x** |
| VGA(3) | rotor compose | 0.037ms | - | 26.8M ops/s |
| CGA(3) | sandwich | 2.490ms | - | 0.4M ops/s |

### Optimization Techniques

- **Hardcoded Operations**: No Cayley table lookups for p+q ≤ 9
- **Sparse Representation**: Only compute non-zero blade products
- **JIT Compilation**: `@torch.jit.script` for p+q ≤ 8
- **Loop-free**: All operations unrolled for ONNX compatibility
- **Bott Periodicity**: Efficient high-dimensional computation

## Testing

```bash
# Run all tests (197 tests)
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_vga.py -v
uv run pytest tests/test_cga.py -v
uv run pytest tests/test_pga.py -v
uv run pytest tests/test_bott.py -v

# Run benchmarks
uv run pytest tests/benchmark/test_benchmark.py -v -s

# Run with coverage
uv run pytest --cov=fast_clifford
```

## Project Structure

```
fast_clifford/
├── __init__.py              # Cl, VGA, CGA, PGA, Multivector, Rotor
├── algebras/
│   ├── __init__.py
│   └── generated/           # 55 pre-generated algebras
│       ├── cl_3_0/          # VGA(3) = Cl(3,0)
│       ├── cl_4_1/          # CGA(3) = Cl(4,1)
│       └── ...
├── clifford/
│   ├── __init__.py          # Factory functions
│   ├── base.py              # CliffordAlgebraBase abstract class
│   ├── registry.py          # HardcodedClWrapper
│   ├── bott.py              # BottPeriodicityAlgebra
│   ├── multivector.py       # Multivector, Rotor classes
│   ├── layers.py            # PyTorch nn.Module layers
│   └── specializations/
│       ├── vga.py           # VGAWrapper
│       ├── cga.py           # CGAWrapper
│       └── pga.py           # PGAEmbedding
├── codegen/
│   ├── __init__.py
│   ├── clifford_factory.py  # Algebra factory utilities
│   └── generator.py         # ClCodeGenerator
└── tests/                   # Test suites (197 tests)
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
