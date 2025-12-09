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
        even_versor_count: EvenVersor components (even grades)
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
        self._even_versor_mask: Optional[Tensor] = None
        self._reverse_signs: Optional[Tensor] = None
        self._even_versor_reverse_signs: Optional[Tensor] = None

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
            get_even_versor_indices,
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

        # EvenVersor mask (even grades)
        ev_indices = list(get_even_versor_indices(self._euclidean_dim))
        self._even_versor_count = len(ev_indices)
        self.register_buffer('even_versor_mask', torch.tensor(ev_indices, dtype=torch.long))

        # Reverse signs
        reverse_signs = compute_reverse_signs(self._euclidean_dim)
        self.register_buffer('reverse_signs', torch.tensor(reverse_signs, dtype=torch.float32))

        # EvenVersor reverse signs
        even_versor_reverse_signs = [reverse_signs[idx] for idx in ev_indices]
        self.register_buffer('even_versor_reverse_signs', torch.tensor(even_versor_reverse_signs, dtype=torch.float32))

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
    def even_versor_count(self) -> int:
        self._ensure_initialized()
        return self._even_versor_count

    @property
    def signature(self) -> Tuple[int, ...]:
        return self._signature

    @property
    def bivector_count(self) -> int:
        """Number of Bivector components (Grade 2)."""
        self._ensure_initialized()
        if not hasattr(self, '_bivector_count'):
            from fast_clifford.codegen.cga_factory import compute_grade_indices
            grade_indices = compute_grade_indices(self._euclidean_dim)
            self._bivector_count = len(grade_indices[2])
        return self._bivector_count

    @property
    def max_grade(self) -> int:
        """Maximum grade in the algebra (= n+2 for CGA(n))."""
        return self._euclidean_dim + 2

    # =========================================================================
    # Core Operations
    # =========================================================================

    def cga_encode(self, x: Tensor) -> Tensor:
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

    def cga_decode(self, point: Tensor) -> Tensor:
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

    def sandwich_product_sparse(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Compute sparse sandwich product M x X x M~.

        Strategy:
        1. Embed EvenVersor and point into full multivector space
        2. Compute M x X using geometric product
        3. Compute (M x X) x M~ using geometric product
        4. Extract point components from result

        Args:
            ev: EvenVersor, shape (..., even_versor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        self._ensure_initialized()

        batch_shape = ev.shape[:-1]
        device = ev.device
        dtype = ev.dtype

        # Embed EvenVersor into full space
        ev_full = torch.zeros(*batch_shape, self._blade_count, device=device, dtype=dtype)
        even_versor_mask_expanded = self.even_versor_mask.expand(*batch_shape, -1)
        ev_full.scatter_(-1, even_versor_mask_expanded, ev)

        # Embed point into full space
        point_full = torch.zeros(*batch_shape, self._blade_count, device=device, dtype=dtype)
        point_mask_expanded = self.point_mask.expand(*batch_shape, -1)
        point_full.scatter_(-1, point_mask_expanded, point)

        # Compute EvenVersor reverse
        ev_rev = ev * self.even_versor_reverse_signs

        # Embed reversed EvenVersor
        ev_rev_full = torch.zeros(*batch_shape, self._blade_count, device=device, dtype=dtype)
        ev_rev_full.scatter_(-1, even_versor_mask_expanded, ev_rev)

        # Compute M x X
        mx = self.geometric_product_full(ev_full, point_full)

        # Compute (M x X) x M~
        result_full = self.geometric_product_full(mx, ev_rev_full)

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

    def reverse_even_versor(self, ev: Tensor) -> Tensor:
        """
        Compute reverse of an EvenVersor.

        Args:
            ev: EvenVersor, shape (..., even_versor_count)

        Returns:
            Reversed EvenVersor, shape (..., even_versor_count)
        """
        self._ensure_initialized()
        return ev * self.even_versor_reverse_signs

    def forward(self, ev: Tensor, point: Tensor) -> Tensor:
        """
        Forward pass (sandwich product).

        Can be used directly as nn.Module.

        Args:
            ev: EvenVersor, shape (..., even_versor_count)
            point: UPGC point, shape (..., point_count)

        Returns:
            Transformed point, shape (..., point_count)
        """
        return self.sandwich_product_sparse(ev, point)

    # =========================================================================
    # Extended Operations
    # =========================================================================

    def _embed_ev(self, ev: Tensor) -> Tensor:
        """Embed EvenVersor into full multivector space."""
        self._ensure_initialized()
        batch_shape = ev.shape[:-1]
        ev_full = torch.zeros(*batch_shape, self._blade_count, device=ev.device, dtype=ev.dtype)
        even_versor_mask_expanded = self.even_versor_mask.expand(*batch_shape, -1)
        ev_full.scatter_(-1, even_versor_mask_expanded, ev)
        return ev_full

    def _extract_ev(self, mv_full: Tensor) -> Tensor:
        """Extract EvenVersor from full multivector."""
        self._ensure_initialized()
        batch_shape = mv_full.shape[:-1]
        even_versor_mask_expanded = self.even_versor_mask.expand(*batch_shape, -1)
        return mv_full.gather(-1, even_versor_mask_expanded)

    def compose_even_versor(self, v1: Tensor, v2: Tensor) -> Tensor:
        """Compose two EvenVersors via geometric product."""
        v1_full = self._embed_ev(v1)
        v2_full = self._embed_ev(v2)
        result_full = self.geometric_product_full(v1_full, v2_full)
        return self._extract_ev(result_full)

    def compose_similitude(self, s1: Tensor, s2: Tensor) -> Tensor:
        """Compose two Similitudes (uses same method as EvenVersor)."""
        return self.compose_even_versor(s1, s2)

    def sandwich_product_even_versor(self, versor: Tensor, point: Tensor) -> Tensor:
        """Compute sandwich product V x X x ~V for EvenVersor."""
        return self.sandwich_product_sparse(versor, point)

    def sandwich_product_similitude(self, similitude: Tensor, point: Tensor) -> Tensor:
        """Compute sandwich product S x X x ~S for Similitude."""
        return self.sandwich_product_sparse(similitude, point)

    def inner_product(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute geometric inner product (metric inner product)."""
        result_full = self.geometric_product_full(a, b)
        return result_full[..., 0:1]  # Grade 0 (scalar)

    def outer_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute outer product (wedge product).

        The outer product keeps terms where:
        <a ^ b>_k = sum over (r,s) of <a_r * b_s>_k where k = r + s

        For grade-r blade wedged with grade-s blade, result is grade (r+s).
        """
        self._ensure_initialized()
        from fast_clifford.codegen.cga_factory import compute_grade_indices

        grade_indices = compute_grade_indices(self._euclidean_dim)
        max_grade = self._euclidean_dim + 2

        # Initialize result
        result = torch.zeros_like(a)

        # For each pair of grades (r, s), compute a_r * b_s and extract grade (r+s)
        for r in range(max_grade + 1):
            if r not in grade_indices:
                continue
            # Extract grade-r part of a
            a_r = self.grade_select(a, r)

            for s in range(max_grade + 1):
                if s not in grade_indices:
                    continue
                k = r + s  # Result grade
                if k > max_grade or k not in grade_indices:
                    continue

                # Extract grade-s part of b
                b_s = self.grade_select(b, s)

                # Compute geometric product
                prod = self.geometric_product_full(a_r, b_s)

                # Extract and add grade-k part
                prod_k = self.grade_select(prod, k)
                result = result + prod_k

        return result

    def left_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute left contraction a ⌋ b.

        The left contraction keeps terms where:
        <a ⌋ b>_k = sum over (r,s) of <a_r * b_s>_k where k = s - r >= 0

        For grade-r blade contracted with grade-s blade, result is grade (s-r).
        """
        self._ensure_initialized()
        from fast_clifford.codegen.cga_factory import compute_grade_indices

        grade_indices = compute_grade_indices(self._euclidean_dim)
        max_grade = self._euclidean_dim + 2

        # Initialize result
        result = torch.zeros_like(a)

        # For each pair of grades (r, s), compute a_r * b_s and extract grade (s-r)
        for r in range(max_grade + 1):
            if r not in grade_indices:
                continue
            # Extract grade-r part of a
            a_r = self.grade_select(a, r)

            for s in range(max_grade + 1):
                if s not in grade_indices:
                    continue
                k = s - r  # Result grade
                if k < 0 or k > max_grade or k not in grade_indices:
                    continue

                # Extract grade-s part of b
                b_s = self.grade_select(b, s)

                # Compute geometric product
                prod = self.geometric_product_full(a_r, b_s)

                # Extract and add grade-k part
                prod_k = self.grade_select(prod, k)
                result = result + prod_k

        return result

    def right_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute right contraction a ⌊ b.

        The right contraction keeps terms where:
        <a ⌊ b>_k = sum over (r,s) of <a_r * b_s>_k where k = r - s >= 0

        For grade-r blade contracted with grade-s blade, result is grade (r-s).
        """
        self._ensure_initialized()
        from fast_clifford.codegen.cga_factory import compute_grade_indices

        grade_indices = compute_grade_indices(self._euclidean_dim)
        max_grade = self._euclidean_dim + 2

        # Initialize result
        result = torch.zeros_like(a)

        # For each pair of grades (r, s), compute a_r * b_s and extract grade (r-s)
        for r in range(max_grade + 1):
            if r not in grade_indices:
                continue
            # Extract grade-r part of a
            a_r = self.grade_select(a, r)

            for s in range(max_grade + 1):
                if s not in grade_indices:
                    continue
                k = r - s  # Result grade
                if k < 0 or k > max_grade or k not in grade_indices:
                    continue

                # Extract grade-s part of b
                b_s = self.grade_select(b, s)

                # Compute geometric product
                prod = self.geometric_product_full(a_r, b_s)

                # Extract and add grade-k part
                prod_k = self.grade_select(prod, k)
                result = result + prod_k

        return result

    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Compute exponential map from Bivector to EvenVersor.

        exp(B) = cos(theta) + sin(theta)/theta * B
        where theta^2 = -B^2
        """
        self._ensure_initialized()

        # Embed bivector into full space
        from fast_clifford.codegen.cga_factory import compute_grade_indices
        grade_indices = compute_grade_indices(self._euclidean_dim)
        bivector_indices = list(grade_indices[2])

        batch_shape = B.shape[:-1]
        B_full = torch.zeros(*batch_shape, self._blade_count, device=B.device, dtype=B.dtype)
        biv_mask = torch.tensor(bivector_indices, dtype=torch.long, device=B.device)
        biv_mask_expanded = biv_mask.expand(*batch_shape, -1)
        B_full.scatter_(-1, biv_mask_expanded, B)

        # Compute B^2 (scalar part)
        B_sq_full = self.geometric_product_full(B_full, B_full)
        B_sq_scalar = B_sq_full[..., 0]  # Grade 0

        # theta^2 = -B^2, theta = sqrt(max(0, -B^2))
        theta_sq = torch.clamp(-B_sq_scalar, min=1e-12)
        theta = torch.sqrt(theta_sq)

        # Compute cos(theta) and sinc(theta) = sin(theta)/theta
        cos_theta = torch.cos(theta)
        sinc_theta = torch.sinc(theta / torch.pi)  # sinc(x) = sin(pi*x)/(pi*x)

        # Result = cos(theta) * 1 + sinc(theta) * theta / theta * B
        # = cos(theta) + sinc(theta) * B (since sinc already divides by theta)
        # Actually: sin(theta)/theta * B, and sinc(x) = sin(pi*x)/(pi*x)
        # So we need: sin(theta)/theta = sinc(theta/pi) * pi / theta * theta = sinc(theta/pi)
        # Wait, that's wrong. Let me reconsider.
        # sinc(x) = sin(pi*x)/(pi*x), so sin(theta)/theta = sinc(theta/pi)

        # Build result: cos(theta) + sinc(theta) * B
        # Start with zeros and set scalar to cos(theta)
        result_full = torch.zeros_like(B_full)
        result_full[..., 0] = cos_theta
        result_full = result_full + sinc_theta.unsqueeze(-1) * B_full

        return self._extract_ev(result_full)

    def grade_select(self, mv: Tensor, grade: int) -> Tensor:
        """
        Extract components of a specific grade.

        Returns a full multivector with only the specified grade components,
        other components set to zero.

        Args:
            mv: Multivector, shape (..., blade_count)
            grade: Grade to extract (0 to max_grade)

        Returns:
            Multivector with only grade-k components, shape (..., blade_count)
        """
        self._ensure_initialized()
        from fast_clifford.codegen.cga_factory import compute_grade_indices

        grade_indices = compute_grade_indices(self._euclidean_dim)

        # Create mask for the specified grade
        mask = torch.zeros(self._blade_count, device=mv.device, dtype=mv.dtype)
        if grade in grade_indices:
            for idx in grade_indices[grade]:
                mask[idx] = 1.0

        return mv * mask

    def dual(self, mv: Tensor) -> Tensor:
        """Compute the dual of a multivector."""
        self._ensure_initialized()
        # Dual is computed as mv * I^-1 where I is the pseudoscalar
        # For CGA(n), I = e1 ^ e2 ^ ... ^ en ^ e+ ^ e-
        # I^-1 = (-1)^k * I where k depends on the algebra
        # For simplicity, we compute I and use geometric product
        I_full = torch.zeros(self._blade_count, device=mv.device, dtype=mv.dtype)
        I_full[-1] = 1.0  # Pseudoscalar is always the last blade
        I_full = I_full.expand(*mv.shape[:-1], -1)
        return self.geometric_product_full(mv, I_full)

    def normalize(self, mv: Tensor) -> Tensor:
        """Normalize a multivector to unit norm."""
        norm_sq = self.inner_product(mv, mv)
        norm = torch.sqrt(torch.clamp(norm_sq.abs(), min=1e-12))
        return mv / norm

    def structure_normalize(self, similitude: Tensor, eps: float = 1e-8) -> Tensor:
        """Structure normalize a Similitude (returns input for runtime, not implemented)."""
        # Structure normalization is complex and specific to similitude structure
        # For runtime, just return the input as-is
        return similitude

    # =========================================================================
    # Layer Factory Methods (using unified layers)
    # =========================================================================

    def get_transform_layer(self) -> nn.Module:
        """Get CliffordTransformLayer for this algebra."""
        from .layers import CliffordTransformLayer
        return CliffordTransformLayer(self)

    def get_care_layer(self) -> nn.Module:
        """Get CliffordTransformLayer (alias for get_transform_layer, deprecated)."""
        return self.get_transform_layer()

    def get_encoder(self) -> nn.Module:
        """Get CGAEncoder for this algebra."""
        from .layers import CGAEncoder
        return CGAEncoder(self)

    def get_decoder(self) -> nn.Module:
        """Get CGADecoder for this algebra."""
        from .layers import CGADecoder
        return CGADecoder(self)

    def get_transform_pipeline(self) -> nn.Module:
        """Get CGAPipeline for this algebra."""
        from .layers import CGAPipeline
        return CGAPipeline(self)


# =============================================================================
# Non-CGA Clifford Algebra (for Cl(p,q,r) with q != 1 or r != 0)
# =============================================================================

class RuntimeCliffordAlgebra(nn.Module):
    """
    Runtime computed Clifford algebra for non-CGA signatures.

    This is a placeholder for Cl(p,q,r) algebras that are not CGA.
    CGA-specific operations (cga_encode, cga_decode) are not available.
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
