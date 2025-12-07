"""
Type stubs for CGA2D Cl(3,1) functional operations.

Generated functions in fast_clifford/algebras/cga2d/functional.py
"""

from torch import Tensor

# =============================================================================
# Constants
# =============================================================================

BLADE_COUNT: int  # = 16

# Blade indices by grade
GRADE_0_INDICES: tuple[int, ...]  # (0,)
GRADE_1_INDICES: tuple[int, ...]  # (1, 2, 3, 4)
GRADE_2_INDICES: tuple[int, ...]  # (5, 6, 7, 8, 9, 10)
GRADE_3_INDICES: tuple[int, ...]  # (11, 12, 13, 14)
GRADE_4_INDICES: tuple[int, ...]  # (15,)

# Sparsity masks
UPGC_POINT_MASK: tuple[int, ...]  # = GRADE_1_INDICES, 4 components
MOTOR_MASK: tuple[int, ...]  # Grade 0, 2, 4, 8 components

# Reverse signs for all 16 blades
REVERSE_SIGNS: tuple[int, ...]

# Motor-specific reverse signs (8 components)
MOTOR_REVERSE_SIGNS: tuple[int, ...]

# =============================================================================
# Geometric Product (Full 16x16)
# =============================================================================

def geometric_product_full(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the full geometric product of two multivectors.

    Args:
        a: Left operand, shape (..., 16)
        b: Right operand, shape (..., 16)

    Returns:
        Result multivector, shape (..., 16)

    Note:
        Fully expanded, no loops, ONNX compatible.
    """
    ...

# =============================================================================
# Reverse Operation
# =============================================================================

def reverse_full(mv: Tensor) -> Tensor:
    """
    Compute the reverse of a multivector.

    For grade k: coefficient *= (-1)^(k*(k-1)/2)
    Grade 0, 1, 4: sign = +1
    Grade 2, 3: sign = -1

    Args:
        mv: Input multivector, shape (..., 16)

    Returns:
        Reversed multivector, shape (..., 16)
    """
    ...

# =============================================================================
# Sparse Operations (Motor x Point)
# =============================================================================

def upgc_encode(x: Tensor) -> Tensor:
    """
    Encode 2D vector to UPGC point representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Where:
        n_o = 0.5 * (e- - e+)   -> coefficients: e+ = -0.5, e- = 0.5
        n_inf = e- + e+         -> coefficients: e+ = 1, e- = 1

    Args:
        x: 2D vector, shape (..., 2)

    Returns:
        UPGC point, shape (..., 4) as [e1, e2, e+, e-]
    """
    ...

def upgc_decode(point: Tensor) -> Tensor:
    """
    Decode UPGC point to 2D vector.

    Extracts the e1, e2 components directly.

    Args:
        point: UPGC point, shape (..., 4) as [e1, e2, e+, e-]

    Returns:
        2D vector, shape (..., 2)
    """
    ...

def reverse_motor(motor: Tensor) -> Tensor:
    """
    Compute reverse of a motor (sparse 8-component version).

    Motor layout (8 components):
        [0]: scalar (Grade 0) -> +1
        [1-6]: bivectors (Grade 2) -> -1
        [7]: quadvector (Grade 4) -> +1

    Args:
        motor: Motor, shape (..., 8)

    Returns:
        Reversed motor, shape (..., 8)
    """
    ...

def sandwich_product_sparse(motor: Tensor, point: Tensor) -> Tensor:
    """
    Compute sparse sandwich product: M × X × M̃

    Optimized for:
        - Motor M: 8 components (Grade 0, 2, 4)
        - Point X: 4 components (Grade 1)
        - Output: 4 components (Grade 1)

    Args:
        motor: Motor, shape (..., 8)
               [scalar, e12, e1+, e1-, e2+, e2-, e+-, e12+-]
        point: UPGC point, shape (..., 4)
               [e1, e2, e+, e-]

    Returns:
        Transformed point, shape (..., 4)

    Note:
        Total multiplications: ~256 (vs 512 for full)
    """
    ...
