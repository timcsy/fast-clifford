"""
RuntimeCGAAlgebra - Runtime computed CGA algebra for high dimensions

Supports CGA6D and higher dimensions using tensor batch operations.
All operations are:
- Loop-free (fully vectorized)
- ONNX-compatible (no Loop nodes)
- PyTorch autograd compatible (differentiable)
"""

from typing import Tuple, Optional
import torch
from torch import Tensor, nn
import numpy as np

from .base import CGAAlgebraBase


class RuntimeCGAAlgebra(CGAAlgebraBase, nn.Module):
    """
    Runtime computed CGA algebra for high dimensions (6+).

    Uses tensor batch operations (scatter_add, index_select) for
    ONNX-compatible, loop-free geometric product computation.

    Attributes:
        euclidean_dim: Euclidean dimension n
        blade_count: Total blades = 2^(n+2)
        point_count: UPGC point components = n+2
        motor_count: Motor components (even grades, excluding pseudoscalar)
    """

    def __init__(self, euclidean_dim: int):
        """
        Create a runtime CGA algebra.

        Args:
            euclidean_dim: Euclidean dimension n (typically >= 6)
        """
        super().__init__()

        self._euclidean_dim = euclidean_dim
        self._blade_count = 2 ** (euclidean_dim + 2)
        self._point_count = euclidean_dim + 2

        # Signature: n+1 positive, 1 negative
        self._signature = tuple([1] * (euclidean_dim + 1) + [-1])

        # Lazy initialization flag
        self._initialized = False

        # These will be registered as buffers on first use
        self._left_idx: Optional[Tensor] = None
        self._right_idx: Optional[Tensor] = None
        self._result_idx: Optional[Tensor] = None
        self._signs: Optional[Tensor] = None
        self._point_mask: Optional[Tensor] = None
        self._motor_mask: Optional[Tensor] = None
        self._reverse_signs: Optional[Tensor] = None
        self._motor_reverse_signs: Optional[Tensor] = None

    def _ensure_initialized(self) -> None:
        """
        Ensure algebra parameters are initialized.

        Computes Cayley table and registers buffers on first call.
        This enables lazy initialization to avoid overhead for unused algebras.
        """
        if self._initialized:
            return

        # Import cga_factory for algebra computation
        from fast_clifford.codegen.cga_factory import (
            create_cga_algebra,
            compute_grade_indices,
            compute_reverse_signs,
            get_motor_indices,
        )

        # Get algebra from clifford library
        layout, blades, stuff = create_cga_algebra(self._euclidean_dim)

        # Extract GMT (Geometric Multiplication Table)
        gmt_dense = np.asarray(layout.gmt.todense())

        # Find all non-zero products
        left_indices = []
        right_indices = []
        result_indices = []
        signs = []

        for i in range(self._blade_count):
            for j in range(self._blade_count):
                result_vec = gmt_dense[i, :, j]
                nonzero = np.where(result_vec != 0)[0]
                for k in nonzero:
                    coeff = result_vec[k]
                    if coeff != 0:
                        left_indices.append(i)
                        right_indices.append(j)
                        result_indices.append(k)
                        signs.append(int(np.sign(coeff)))

        # Register as buffers (will be included in ONNX)
        self.register_buffer('left_idx', torch.tensor(left_indices, dtype=torch.long))
        self.register_buffer('right_idx', torch.tensor(right_indices, dtype=torch.long))
        self.register_buffer('result_idx', torch.tensor(result_indices, dtype=torch.long))
        self.register_buffer('signs', torch.tensor(signs, dtype=torch.float32))

        # Compute grade indices
        grade_indices = compute_grade_indices(self._euclidean_dim)

        # Point mask (Grade 1)
        point_indices = list(grade_indices[1])
        self.register_buffer('point_mask', torch.tensor(point_indices, dtype=torch.long))

        # Motor mask (even grades, excluding pseudoscalar)
        motor_indices = list(get_motor_indices(self._euclidean_dim))
        self._motor_count = len(motor_indices)
        self.register_buffer('motor_mask', torch.tensor(motor_indices, dtype=torch.long))

        # Reverse signs
        reverse_signs = compute_reverse_signs(self._euclidean_dim)
        self.register_buffer('reverse_signs', torch.tensor(reverse_signs, dtype=torch.float32))

        # Motor reverse signs
        motor_reverse_signs = [reverse_signs[idx] for idx in motor_indices]
        self.register_buffer('motor_reverse_signs', torch.tensor(motor_reverse_signs, dtype=torch.float32))

        self._initialized = True

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def euclidean_dim(self) -> int:
        return self._euclidean_dim

    @property
    def blade_count(self) -> int:
        return self._blade_count

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def motor_count(self) -> int:
        self._ensure_initialized()
        return self._motor_count

    @property
    def signature(self) -> Tuple[int, ...]:
        return self._signature

    # =========================================================================
    # Core Operations
    # =========================================================================

    def upgc_encode(self, x: Tensor) -> Tensor:
        """
        Encode Euclidean coordinates to UPGC point.

        X = n_o + x + 0.5|x|^2 * n_inf

        where:
            n_o = 0.5 * (e- - e+)
            n_inf = e- + e+

        Args:
            x: Euclidean coordinates, shape (..., n)

        Returns:
            UPGC point, shape (..., n+2)
        """
        self._ensure_initialized()

        n = self._euclidean_dim
        half_norm_sq = 0.5 * (x * x).sum(dim=-1, keepdim=True)

        # Point components: [e1, e2, ..., en, e+, e-]
        # e_i = x_i
        # e+ = -0.5 + 0.5|x|^2
        # e- = 0.5 + 0.5|x|^2
        e_plus = -0.5 + half_norm_sq
        e_minus = 0.5 + half_norm_sq

        return torch.cat([x, e_plus, e_minus], dim=-1)

    def upgc_decode(self, point: Tensor) -> Tensor:
        """
        Decode UPGC point to Euclidean coordinates.

        Args:
            point: UPGC point, shape (..., n+2)

        Returns:
            Euclidean coordinates, shape (..., n)
        """
        n = self._euclidean_dim
        return point[..., :n]

    def geometric_product_full(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute full geometric product using tensor batch operations.

        Uses scatter_add for ONNX-compatible, loop-free computation.

        Args:
            a: Left operand, shape (..., blade_count)
            b: Right operand, shape (..., blade_count)

        Returns:
            Result multivector, shape (..., blade_count)
        """
        self._ensure_initialized()

        # Get values at product indices
        a_vals = a[..., self.left_idx]  # (..., num_nonzero)
        b_vals = b[..., self.right_idx]  # (..., num_nonzero)

        # Compute products with signs
        products = self.signs * a_vals * b_vals  # (..., num_nonzero)

        # Accumulate to result positions using scatter_add
        batch_shape = a.shape[:-1]
        result = torch.zeros(*batch_shape, self._blade_count, device=a.device, dtype=a.dtype)

        # Expand result_idx for batch dimensions
        result_idx_expanded = self.result_idx.expand(*batch_shape, -1)
        result.scatter_add_(-1, result_idx_expanded, products)

        return result

    def sandwich_product_sparse(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        Compute sparse sandwich product M x X x M~.

        Strategy:
        1. Embed motor and point into full multivector space
        2. Compute M x X using geometric product
        3. Compute (M x X) x M~ using geometric product
        4. Extract point components from result

        Args:
            motor: Motor, shape (..., motor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        self._ensure_initialized()

        batch_shape = motor.shape[:-1]
        device = motor.device
        dtype = motor.dtype

        # Embed motor into full space
        motor_full = torch.zeros(*batch_shape, self._blade_count, device=device, dtype=dtype)
        motor_mask_expanded = self.motor_mask.expand(*batch_shape, -1)
        motor_full.scatter_(-1, motor_mask_expanded, motor)

        # Embed point into full space
        point_full = torch.zeros(*batch_shape, self._blade_count, device=device, dtype=dtype)
        point_mask_expanded = self.point_mask.expand(*batch_shape, -1)
        point_full.scatter_(-1, point_mask_expanded, point)

        # Compute motor reverse
        motor_rev = motor * self.motor_reverse_signs

        # Embed reversed motor
        motor_rev_full = torch.zeros(*batch_shape, self._blade_count, device=device, dtype=dtype)
        motor_rev_full.scatter_(-1, motor_mask_expanded, motor_rev)

        # Compute M x X
        mx = self.geometric_product_full(motor_full, point_full)

        # Compute (M x X) x M~
        result_full = self.geometric_product_full(mx, motor_rev_full)

        # Extract point components
        result = result_full.gather(-1, point_mask_expanded)

        return result

    def reverse_full(self, mv: Tensor) -> Tensor:
        """
        Compute reverse of a multivector.

        Args:
            mv: Multivector, shape (..., blade_count)

        Returns:
            Reversed multivector, shape (..., blade_count)
        """
        self._ensure_initialized()
        return mv * self.reverse_signs

    def reverse_motor(self, motor: Tensor) -> Tensor:
        """
        Compute reverse of a motor.

        Args:
            motor: Motor, shape (..., motor_count)

        Returns:
            Reversed motor, shape (..., motor_count)
        """
        self._ensure_initialized()
        return motor * self.motor_reverse_signs

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        """
        Forward pass (sandwich product).

        Can be used directly as nn.Module.

        Args:
            motor: Motor, shape (..., motor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        return self.sandwich_product_sparse(motor, point)

    # =========================================================================
    # Layer Factory Methods
    # =========================================================================

    def get_care_layer(self) -> nn.Module:
        """Get RuntimeCGACareLayer."""
        return RuntimeCGACareLayer(self)

    def get_encoder(self) -> nn.Module:
        """Get RuntimeUPGCEncoder."""
        return RuntimeUPGCEncoder(self)

    def get_decoder(self) -> nn.Module:
        """Get RuntimeUPGCDecoder."""
        return RuntimeUPGCDecoder(self)

    def get_transform_pipeline(self) -> nn.Module:
        """Get RuntimeCGATransformPipeline."""
        return RuntimeCGATransformPipeline(self)


# =============================================================================
# Runtime Layer Classes
# =============================================================================

class RuntimeCGACareLayer(nn.Module):
    """Runtime CGA Care Layer for sandwich product."""

    def __init__(self, algebra: RuntimeCGAAlgebra):
        super().__init__()
        self.algebra = algebra

    def forward(self, motor: Tensor, point: Tensor) -> Tensor:
        original_dtype = point.dtype
        motor_f32 = motor.to(torch.float32)
        point_f32 = point.to(torch.float32)
        result = self.algebra.sandwich_product_sparse(motor_f32, point_f32)
        return result.to(original_dtype)


class RuntimeUPGCEncoder(nn.Module):
    """Runtime UPGC Encoder."""

    def __init__(self, algebra: RuntimeCGAAlgebra):
        super().__init__()
        self.algebra = algebra

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = self.algebra.upgc_encode(x_f32)
        return result.to(original_dtype)


class RuntimeUPGCDecoder(nn.Module):
    """Runtime UPGC Decoder."""

    def __init__(self, algebra: RuntimeCGAAlgebra):
        super().__init__()
        self.algebra = algebra

    def forward(self, point: Tensor) -> Tensor:
        return self.algebra.upgc_decode(point)


class RuntimeCGATransformPipeline(nn.Module):
    """Runtime CGA complete transform pipeline."""

    def __init__(self, algebra: RuntimeCGAAlgebra):
        super().__init__()
        self.encoder = RuntimeUPGCEncoder(algebra)
        self.care_layer = RuntimeCGACareLayer(algebra)
        self.decoder = RuntimeUPGCDecoder(algebra)

    def forward(self, motor: Tensor, x: Tensor) -> Tensor:
        """
        Complete transform: encode -> sandwich -> decode

        Args:
            motor: Motor, shape (..., motor_count)
            x: Euclidean coordinates, shape (..., n)

        Returns:
            Transformed Euclidean coordinates, shape (..., n)
        """
        point = self.encoder(x)
        transformed = self.care_layer(motor, point)
        return self.decoder(transformed)


# =============================================================================
# Non-CGA Clifford Algebra (for Cl(p,q,r) with q != 1 or r != 0)
# =============================================================================

class RuntimeCliffordAlgebra(nn.Module):
    """
    Runtime computed Clifford algebra for non-CGA signatures.

    This is a placeholder for Cl(p,q,r) algebras that are not CGA.
    CGA-specific operations (upgc_encode, upgc_decode) are not available.
    """

    def __init__(self, p: int, q: int, r: int = 0):
        """
        Create a runtime Clifford algebra.

        Args:
            p: Positive-signature dimensions
            q: Negative-signature dimensions
            r: Degenerate dimensions
        """
        super().__init__()
        self._p = p
        self._q = q
        self._r = r
        self._blade_count = 2 ** (p + q + r)
        self._signature = tuple([1] * p + [-1] * q + [0] * r)
        self._initialized = False

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    @property
    def r(self) -> int:
        return self._r

    @property
    def blade_count(self) -> int:
        return self._blade_count

    @property
    def signature(self) -> Tuple[int, ...]:
        return self._signature

    @property
    def clifford_notation(self) -> str:
        return f"Cl({self._p},{self._q},{self._r})"

    def __repr__(self) -> str:
        return f"RuntimeCliffordAlgebra({self.clifford_notation})"

    def _ensure_initialized(self) -> None:
        """Initialize algebra (placeholder for future implementation)."""
        if self._initialized:
            return

        # For non-CGA algebras, we need to use clifford library directly
        # This is a simplified implementation
        from clifford import Cl

        layout, blades = Cl(self._p, self._q)
        gmt_dense = np.asarray(layout.gmt.todense())

        # Find all non-zero products
        left_indices = []
        right_indices = []
        result_indices = []
        signs = []

        for i in range(self._blade_count):
            for j in range(self._blade_count):
                result_vec = gmt_dense[i, :, j]
                nonzero = np.where(result_vec != 0)[0]
                for k in nonzero:
                    coeff = result_vec[k]
                    if coeff != 0:
                        left_indices.append(i)
                        right_indices.append(j)
                        result_indices.append(k)
                        signs.append(int(np.sign(coeff)))

        self.register_buffer('left_idx', torch.tensor(left_indices, dtype=torch.long))
        self.register_buffer('right_idx', torch.tensor(right_indices, dtype=torch.long))
        self.register_buffer('result_idx', torch.tensor(result_indices, dtype=torch.long))
        self.register_buffer('signs', torch.tensor(signs, dtype=torch.float32))

        self._initialized = True

    def geometric_product_full(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute full geometric product."""
        self._ensure_initialized()

        a_vals = a[..., self.left_idx]
        b_vals = b[..., self.right_idx]
        products = self.signs * a_vals * b_vals

        batch_shape = a.shape[:-1]
        result = torch.zeros(*batch_shape, self._blade_count, device=a.device, dtype=a.dtype)
        result_idx_expanded = self.result_idx.expand(*batch_shape, -1)
        result.scatter_add_(-1, result_idx_expanded, products)

        return result
