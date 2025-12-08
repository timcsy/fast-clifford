"""
CGA0D Cl(1,1) Functional Operations - Hardcoded

All functions are:
- Loop-free (fully expanded arithmetic)
- ONNX-compatible (only Add/Mul/Neg/Sub operations)
- Hard-coded (no Cayley table lookups)
"""

import torch
from torch import Tensor


# =============================================================================
# Constants
# =============================================================================

BLADE_COUNT = 4
EUCLIDEAN_DIM = 0

# Blade indices by grade
GRADE_0_INDICES = (0,)
GRADE_1_INDICES = (1, 2)
GRADE_2_INDICES = (3,)

# Sparsity masks
UPGC_POINT_MASK = (1, 2)  # 2 components
MOTOR_MASK = (0, 3)  # 2 components

# Reverse signs for all 4 blades
REVERSE_SIGNS = (1, 1, 1, -1)

# Motor-specific reverse signs (2 components)
MOTOR_REVERSE_SIGNS = (1, -1)

# =============================================================================
# Geometric Product (Full 4x4)
# =============================================================================

@torch.jit.script
def geometric_product_full(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the full geometric product of two multivectors.

    Cayley table for Cl(1,1) with signature (+1, -1):
    (Indices: 0=1, 1=e+, 2=e-, 3=e+-)

    From clifford library GMT:
         |  1    e+   e-   e+-
    -----|------------------------
      1  |  1    e+   e-   e+-
      e+ |  e+   1    e+-  e-
      e- |  e-  -e+- -1    e+
     e+- | e+-  -e-  -e+   1

    Key: e+^2 = 1, e-^2 = -1, e+-^2 = 1

    Args:
        a: Left operand, shape (..., 4)
        b: Right operand, shape (..., 4)

    Returns:
        Result multivector, shape (..., 4)

    Note:
        Fully expanded, no loops, ONNX compatible.
    """
    # r0 (scalar): 1*1 + e+*e+ + e-*e- + e+-*e+-
    # From GMT: e[0]*e[0]=1, e[1]*e[1]=1, e[2]*e[2]=-1, e[3]*e[3]=1
    # = a0*b0 + a1*b1 - a2*b2 + a3*b3
    r0 = (
        a[..., 0] * b[..., 0] +
        a[..., 1] * b[..., 1] +
        -a[..., 2] * b[..., 2] +
        a[..., 3] * b[..., 3]
    )

    # r1 (e+): 1*e+ + e+*1 + e-*e+- + e+-*e-
    # From GMT: e[0]*e[1]=e[1], e[1]*e[0]=e[1], e[2]*e[3]=e[1], e[3]*e[2]=-e[1]
    # = a0*b1 + a1*b0 + a2*b3 - a3*b2
    r1 = (
        a[..., 0] * b[..., 1] +
        a[..., 1] * b[..., 0] +
        a[..., 2] * b[..., 3] +
        -a[..., 3] * b[..., 2]
    )

    # r2 (e-): 1*e- + e+*e+- + e-*1 + e+-*e+
    # From GMT: e[0]*e[2]=e[2], e[1]*e[3]=e[2], e[2]*e[0]=e[2], e[3]*e[1]=-e[2]
    # = a0*b2 + a1*b3 + a2*b0 - a3*b1
    r2 = (
        a[..., 0] * b[..., 2] +
        a[..., 1] * b[..., 3] +
        a[..., 2] * b[..., 0] +
        -a[..., 3] * b[..., 1]
    )

    # r3 (e+-): 1*e+- + e+*e- + e-*e+ + e+-*1
    # From GMT: e[0]*e[3]=e[3], e[1]*e[2]=e[3], e[2]*e[1]=-e[3], e[3]*e[0]=e[3]
    # = a0*b3 + a1*b2 - a2*b1 + a3*b0
    r3 = (
        a[..., 0] * b[..., 3] +
        a[..., 1] * b[..., 2] +
        -a[..., 2] * b[..., 1] +
        a[..., 3] * b[..., 0]
    )

    return torch.stack([r0, r1, r2, r3], dim=-1)

# =============================================================================
# Reverse Operation
# =============================================================================

@torch.jit.script
def reverse_full(mv: Tensor) -> Tensor:
    """
    Compute the reverse of a multivector.

    For grade k: coefficient *= (-1)^(k*(k-1)/2)
    Grade 0: +1
    Grade 1: +1
    Grade 2: -1

    Args:
        mv: Input multivector, shape (..., 4)

    Returns:
        Reversed multivector, shape (..., 4)
    """
    r0 = mv[..., 0]    # Grade 0: +1
    r1 = mv[..., 1]    # Grade 1: +1
    r2 = mv[..., 2]    # Grade 1: +1
    r3 = -mv[..., 3]   # Grade 2: -1

    return torch.stack([r0, r1, r2, r3], dim=-1)

# =============================================================================
# Sparse Operations (Motor[2] x Point[2])
# =============================================================================

# Motor sparse indices: (0, 3) - [scalar, e+-]
# Point sparse indices: (1, 2) - [e+, e-]

@torch.jit.script
def upgc_encode(x: Tensor) -> Tensor:
    """
    Encode 0D vector to UPGC point representation.

    For 0D CGA, there is no Euclidean component.
    The point is always the origin: n_o = 0.5 * (e- - e+)

    X = n_o = 0.5 * (e- - e+)
    → e+ = -0.5, e- = 0.5

    Args:
        x: 0D vector, shape (..., 0) or shape (...)

    Returns:
        UPGC point, shape (..., 2) = [e+, e-]
    """
    # For 0D, we ignore the input and return the origin
    # We need to broadcast to match the batch dimensions
    batch_shape = x.shape[:-1] if x.dim() > 0 and x.shape[-1] == 0 else x.shape
    device = x.device
    dtype = x.dtype

    e_plus = torch.full(batch_shape + (1,), -0.5, device=device, dtype=dtype)
    e_minus = torch.full(batch_shape + (1,), 0.5, device=device, dtype=dtype)

    return torch.cat([e_plus, e_minus], dim=-1)


@torch.jit.script
def upgc_decode(point: Tensor) -> Tensor:
    """
    Decode UPGC point to 0D vector.

    For 0D CGA, returns an empty tensor with shape (..., 0).

    Args:
        point: UPGC point, shape (..., 2)

    Returns:
        0D vector, shape (..., 0)
    """
    batch_shape = point.shape[:-1]
    return torch.zeros(batch_shape + (0,), device=point.device, dtype=point.dtype)


@torch.jit.script
def reverse_motor(motor: Tensor) -> Tensor:
    """
    Compute reverse of a motor (sparse 2-component version).

    Motor layout: [scalar, e+-]
    Reverse signs: [+1, -1]

    Args:
        motor: Motor, shape (..., 2)

    Returns:
        Reversed motor, shape (..., 2)
    """
    r0 = motor[..., 0]   # scalar: keep
    r1 = -motor[..., 1]  # e+-: negate

    return torch.stack([r0, r1], dim=-1)


@torch.jit.script
def sandwich_product_sparse(motor: Tensor, point: Tensor) -> Tensor:
    """
    Compute sparse sandwich product: M x X x M~

    Optimized for:
        - Motor M: 2 components [scalar, e+-]
        - Point X: 2 components [e+, e-]
        - Output: 2 components [e+, e-]

    Args:
        motor: Motor, shape (..., 2)
        point: UPGC point, shape (..., 2)

    Returns:
        Transformed point, shape (..., 2)

    Note:
        For CGA0D, the sandwich product has limited effect since
        the motor can only scale/rotate in the conformal plane.

    Cayley table for Cl(1,1) with signature (+1, -1):
        e+*e+- = e-
        e-*e+- = e+
        e+-*e+ = -e-
        e+-*e- = -e+
    """
    # Motor components (sparse)
    m0 = motor[..., 0]  # scalar
    m1 = motor[..., 1]  # e+-

    # Point components (sparse)
    p0 = point[..., 0]  # e+
    p1 = point[..., 1]  # e-

    # Motor reverse (Grade 2 gets negated)
    mr0 = m0
    mr1 = -m1

    # First compute M * X (intermediate result has components at indices 1, 2)
    # M * X where M = m0 + m1*e+- and X = p0*e+ + p1*e-
    #
    # m0 * (p0*e+ + p1*e-) = m0*p0*e+ + m0*p1*e-
    # m1*e+- * (p0*e+ + p1*e-) = m1*p0*(e+-*e+) + m1*p1*(e+-*e-)
    #                         = m1*p0*(-e-) + m1*p1*(-e+)
    #                         = -m1*p0*e- - m1*p1*e+
    #
    # So M*X = (m0*p0 - m1*p1)*e+ + (m0*p1 - m1*p0)*e-
    mx_plus = m0 * p0 - m1 * p1   # e+ coefficient
    mx_minus = m0 * p1 - m1 * p0  # e- coefficient

    # Now compute (M*X) * M~
    # (M*X) * M~ where M~ = mr0 + mr1*e+-
    #
    # (mx_plus*e+ + mx_minus*e-) * (mr0 + mr1*e+-)
    # = mx_plus*e+*mr0 + mx_plus*e+*mr1*e+- + mx_minus*e-*mr0 + mx_minus*e-*mr1*e+-
    # = mx_plus*mr0*e+ + mx_plus*mr1*(e+*e+-) + mx_minus*mr0*e- + mx_minus*mr1*(e-*e+-)
    # = mx_plus*mr0*e+ + mx_plus*mr1*e- + mx_minus*mr0*e- + mx_minus*mr1*e+
    #
    # r0 (e+): mx_plus*mr0 + mx_minus*mr1
    # r1 (e-): mx_plus*mr1 + mx_minus*mr0

    r0 = mx_plus * mr0 + mx_minus * mr1
    r1 = mx_plus * mr1 + mx_minus * mr0

    return torch.stack([r0, r1], dim=-1)
