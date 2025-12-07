"""
Type stubs for CGA1D Cl(2,1) functional operations.

Generated functions in fast_clifford/algebras/cga1d/functional.py
"""

from torch import Tensor

# =============================================================================
# Constants
# =============================================================================

BLADE_COUNT: int  # = 8

# Blade indices by grade
GRADE_0_INDICES: tuple[int, ...]  # (0,)
GRADE_1_INDICES: tuple[int, ...]  # (1, 2, 3)
GRADE_2_INDICES: tuple[int, ...]  # (4, 5, 6)
GRADE_3_INDICES: tuple[int, ...]  # (7,)

# Sparsity masks
UPGC_POINT_MASK: tuple[int, ...]  # = GRADE_1_INDICES, 3 components
MOTOR_MASK: tuple[int, ...]  # Grade 0, 2, 4 components

# Reverse signs for all 8 blades
REVERSE_SIGNS: tuple[int, ...]

# Motor-specific reverse signs (4 components)
MOTOR_REVERSE_SIGNS: tuple[int, ...]

# =============================================================================
# Geometric Product (Full 8x8)
# =============================================================================

def geometric_product_full(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the full geometric product of two multivectors.

    Args:
        a: Left operand, shape (..., 8)
        b: Right operand, shape (..., 8)

    Returns:
        Result multivector, shape (..., 8)

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
    Grade 0, 1: sign = +1
    Grade 2, 3: sign = -1

    Args:
        mv: Input multivector, shape (..., 8)

    Returns:
        Reversed multivector, shape (..., 8)
    """
    ...

# =============================================================================
# Sparse Operations (Motor x Point)
# =============================================================================

def upgc_encode(x: Tensor) -> Tensor:
    """
    Encode 1D scalar to UPGC point representation.

    X = n_o + x + 0.5|x|^2 * n_inf

    Where:
        n_o = 0.5 * (e- - e+)   -> coefficients: e+ = -0.5, e- = 0.5
        n_inf = e- + e+         -> coefficients: e+ = 1, e- = 1

    Args:
        x: 1D scalar, shape (..., 1)

    Returns:
        UPGC point, shape (..., 3) as [e1, e+, e-]
    """
    ...

def upgc_decode(point: Tensor) -> Tensor:
    """
    Decode UPGC point to 1D scalar.

    Extracts the e1 component directly.

    Args:
        point: UPGC point, shape (..., 3) as [e1, e+, e-]

    Returns:
        1D scalar, shape (..., 1)
    """
    ...

def reverse_motor(motor: Tensor) -> Tensor:
    """
    Compute reverse of a motor (sparse 4-component version).

    Motor layout (4 components):
        [0]: scalar (Grade 0) -> +1
        [1-3]: bivectors (Grade 2) -> -1

    Note: CGA1D has no Grade 4 (max grade is 3).

    Args:
        motor: Motor, shape (..., 4)

    Returns:
        Reversed motor, shape (..., 4)
    """
    ...

def sandwich_product_sparse(motor: Tensor, point: Tensor) -> Tensor:
    """
    Compute sparse sandwich product: M × X × M̃

    Optimized for:
        - Motor M: 4 components (Grade 0, 2)
        - Point X: 3 components (Grade 1)
        - Output: 3 components (Grade 1)

    Args:
        motor: Motor, shape (..., 4)
               [scalar, e1+, e1-, e+-]
        point: UPGC point, shape (..., 3)
               [e1, e+, e-]

    Returns:
        Transformed point, shape (..., 3)

    Note:
        Total multiplications: ~72 (vs 128 for full)
    """
    ...
