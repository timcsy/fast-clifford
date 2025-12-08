"""
Type stubs for CGA Extended Operations

Defines the API contracts for EvenVersor composition, Similitude,
inner_product, exp_bivector, outer_product, contractions, grade_select, dual, normalize.
"""

from torch import Tensor
from typing import Protocol, runtime_checkable, Literal, Union


Kind = Literal['versor', 'even_versor', 'similitude', 'bivector', 'point', None]


@runtime_checkable
class CGAAlgebraExtendedOps(Protocol):
    """Extended operations protocol for CGA algebras."""

    @property
    def bivector_count(self) -> int:
        """Number of Bivector components (Grade 2)"""
        ...

    @property
    def even_versor_count(self) -> int:
        """Number of EvenVersor components (even grades)"""
        ...

    # === Composition Operations ===

    def compose_even_versor(self, v1: Tensor, v2: Tensor) -> Tensor:
        """
        Compose two EvenVersors via geometric product.

        Computes V_result = V1 * V2

        Args:
            v1: First EvenVersor, shape (..., even_versor_count)
            v2: Second EvenVersor, shape (..., even_versor_count)

        Returns:
            Composed EvenVersor, shape (..., even_versor_count)

        Note:
            - EvenVersor × EvenVersor = EvenVersor (even grades preserved)
            - Non-commutative: compose(v1, v2) != compose(v2, v1)
            - Associative: compose(compose(a,b),c) == compose(a,compose(b,c))
        """
        ...

    def compose_similitude(self, s1: Tensor, s2: Tensor) -> Tensor:
        """
        Compose two Similitudes via optimized geometric product.

        Faster than compose_even_versor by skipping transversion terms.

        Args:
            s1: First Similitude, shape (..., even_versor_count)
            s2: Second Similitude, shape (..., even_versor_count)

        Returns:
            Composed Similitude, shape (..., even_versor_count)

        Note:
            - Similitude × Similitude = Similitude
            - 30-40% faster than compose_even_versor
            - Requires inputs to be valid Similitudes (no transversion)
        """
        ...

    # === Products ===

    def inner_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute geometric inner product (metric inner product).

        Returns the Grade 0 component of the geometric product,
        with proper CGA metric signature handling.

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Scalar inner product, shape (..., 1)

        Note:
            - CGA metric is (+,+,...,+,-), not Euclidean
            - For null basis: inner_product(eo, einf) == -1
            - Symmetric: inner_product(a, b) == inner_product(b, a)
        """
        ...

    def outer_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute outer product (wedge product).

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Wedge product, shape (..., blade_count)

        Note:
            - Grade(a ^ b) = Grade(a) + Grade(b)
            - Anti-symmetric: a ^ b = -(b ^ a)
            - Nilpotent: a ^ a = 0
        """
        ...

    def left_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute left contraction.

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Left contraction result, shape (..., blade_count)

        Note:
            - Grade(a ⌋ b) = Grade(b) - Grade(a), if Grade(a) <= Grade(b)
            - Returns 0 if Grade(a) > Grade(b)
        """
        ...

    def right_contraction(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute right contraction.

        Args:
            a: First multivector, shape (..., blade_count)
            b: Second multivector, shape (..., blade_count)

        Returns:
            Right contraction result, shape (..., blade_count)

        Note:
            - Grade(a ⌊ b) = Grade(a) - Grade(b), if Grade(a) >= Grade(b)
            - Returns 0 if Grade(a) < Grade(b)
        """
        ...

    # === Unary Operations ===

    def exp_bivector(self, B: Tensor) -> Tensor:
        """
        Compute exponential map from Bivector to EvenVersor.

        Implements exp(B) = cos(theta) + sin(theta)/theta * B
        where theta^2 = -B^2

        Args:
            B: Bivector, shape (..., bivector_count)

        Returns:
            EvenVersor, shape (..., even_versor_count)

        Note:
            - exp(0) returns identity (1, 0, 0, ...)
            - Numerically stable for small angles (uses sinc)
            - exp(B) * exp(-B) ~ identity
        """
        ...

    def grade_select(self, mv: Tensor, grade: int) -> Tensor:
        """
        Extract components of a specific grade.

        Args:
            mv: Multivector, shape (..., blade_count)
            grade: Grade to extract (0 to max_grade)

        Returns:
            Grade components, shape (..., grade_count)

        Note:
            - grade_count = C(n+2, grade) for CGA(n)
            - Pure indexing operation, no computation
        """
        ...

    def dual(self, mv: Tensor) -> Tensor:
        """
        Compute the dual of a multivector.

        dual(a) = a * I^{-1}  where I is the pseudoscalar

        Args:
            mv: Multivector, shape (..., blade_count)

        Returns:
            Dual multivector, shape (..., blade_count)

        Note:
            - Grade(dual(a)) = max_grade - Grade(a)
            - dual(dual(a)) = +/- a depending on I^2
        """
        ...

    def normalize(self, mv: Tensor) -> Tensor:
        """
        Normalize a multivector to unit norm.

        normalize(a) = a / |a|  where |a|^2 = <a * ~a>_0

        Args:
            mv: Multivector, shape (..., blade_count)

        Returns:
            Normalized multivector, shape (..., blade_count)

        Note:
            - Returns mv unchanged if norm < eps (avoids NaN)
            - Uses torch.where for ONNX compatibility
        """
        ...

    # === TRS Conversion ===

    def from_trs(
        self,
        translation: Tensor,
        rotation: Tensor,
        scale: Tensor,
        rotation_format: Literal['quaternion', 'euler', 'bivector', 'angle'] = 'quaternion'
    ) -> Tensor:
        """
        Build Similitude from TRS parameters.

        Args:
            translation: Translation vector, shape (..., dim)
            rotation: Rotation parameters
                - quaternion (3D): (..., 4) [w, x, y, z]
                - euler (3D): (..., 3) [roll, pitch, yaw]
                - bivector: (..., bivector_count)
                - angle (2D): (..., 1)
            scale: Uniform scale factor, shape (..., 1)
            rotation_format: Rotation parameter format

        Returns:
            Similitude tensor, shape (..., even_versor_count)

        Note:
            Composition order: S = T * R * D (scale first, then rotate, then translate)
        """
        ...

    def to_trs(
        self,
        similitude: Tensor,
        rotation_format: Literal['quaternion', 'euler', 'bivector', 'angle'] = 'quaternion'
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Extract TRS parameters from Similitude.

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)
            rotation_format: Output rotation format

        Returns:
            Tuple of (translation, rotation, scale):
                - translation: (..., dim)
                - rotation: format-dependent shape
                - scale: (..., 1)
        """
        ...

    def make_translation(self, t: Tensor) -> Tensor:
        """
        Build translation-only Similitude.

        T = 1 + (1/2) * t * einf

        Args:
            t: Translation vector, shape (..., dim)

        Returns:
            Translation Similitude, shape (..., even_versor_count)
        """
        ...

    def make_rotation(self, rotation: Tensor, rotation_format: str = 'quaternion') -> Tensor:
        """
        Build rotation-only Similitude.

        R = exp(B) where B is rotation bivector

        Args:
            rotation: Rotation parameters (format-dependent)
            rotation_format: 'quaternion', 'euler', 'bivector', or 'angle'

        Returns:
            Rotation Similitude, shape (..., even_versor_count)
        """
        ...

    def make_dilation(self, scale: Tensor) -> Tensor:
        """
        Build dilation-only Similitude.

        D = cosh(lambda/2) + sinh(lambda/2) * e+- where s = exp(lambda)

        Args:
            scale: Scale factor, shape (..., 1)

        Returns:
            Dilation Similitude, shape (..., even_versor_count)
        """
        ...

    # === Structure Normalize ===

    def structure_normalize(self, similitude: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Structure normalize a Similitude to maintain geometric constraints.

        Performs:
        1. Normalize Rotor part (maintain unit quaternion)
        2. Enforce Similitude constraint (ei+ = ei-, exclude transversion)
        3. Optionally clamp Dilation range

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)
            eps: Numerical stability constant

        Returns:
            Structure-normalized Similitude, shape (..., even_versor_count)

        Note:
            - ONNX compatible (no loops, no conditionals)
            - Use during training to prevent parameter drift
            - Can also be used as regularization target
        """
        ...

    def soft_structure_normalize(self, similitude: Tensor, strength: float = 0.1) -> Tensor:
        """
        Soft structure normalize (gradient-friendly version).

        Interpolates between original and fully normalized:
        result = similitude + strength * (structure_normalize(similitude) - similitude)

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)
            strength: Interpolation strength (0 = no change, 1 = full normalize)

        Returns:
            Softly normalized Similitude, shape (..., even_versor_count)
        """
        ...

    def structure_normalize_ste(self, similitude: Tensor) -> Tensor:
        """
        Structure normalize with Straight-Through Estimator.

        Forward: uses structure_normalize(similitude)
        Backward: gradients pass through similitude unchanged

        Useful for training with hard constraints while maintaining gradient flow.

        Args:
            similitude: Similitude tensor, shape (..., even_versor_count)

        Returns:
            Structure-normalized Similitude (with STE gradient)
        """
        ...


# Hardcoded functional signatures (per dimension)

# === EvenVersor Composition ===

def compose_even_versor_0d(v1: Tensor, v2: Tensor) -> Tensor:
    """CGA0D EvenVersor composition. v1, v2: (..., 2) -> (..., 2)"""
    ...

def compose_even_versor_1d(v1: Tensor, v2: Tensor) -> Tensor:
    """CGA1D EvenVersor composition. v1, v2: (..., 4) -> (..., 4)"""
    ...

def compose_even_versor_2d(v1: Tensor, v2: Tensor) -> Tensor:
    """CGA2D EvenVersor composition. v1, v2: (..., 8) -> (..., 8)"""
    ...

def compose_even_versor_3d(v1: Tensor, v2: Tensor) -> Tensor:
    """CGA3D EvenVersor composition. v1, v2: (..., 16) -> (..., 16)"""
    ...

def compose_even_versor_4d(v1: Tensor, v2: Tensor) -> Tensor:
    """CGA4D EvenVersor composition. v1, v2: (..., 32) -> (..., 32)"""
    ...

def compose_even_versor_5d(v1: Tensor, v2: Tensor) -> Tensor:
    """CGA5D EvenVersor composition. v1, v2: (..., 64) -> (..., 64)"""
    ...


# === Similitude Composition ===

def compose_similitude_0d(s1: Tensor, s2: Tensor) -> Tensor:
    """CGA0D Similitude composition. s1, s2: (..., 2) -> (..., 2)"""
    ...

def compose_similitude_1d(s1: Tensor, s2: Tensor) -> Tensor:
    """CGA1D Similitude composition. s1, s2: (..., 4) -> (..., 4)"""
    ...

def compose_similitude_2d(s1: Tensor, s2: Tensor) -> Tensor:
    """CGA2D Similitude composition. s1, s2: (..., 8) -> (..., 8)"""
    ...

def compose_similitude_3d(s1: Tensor, s2: Tensor) -> Tensor:
    """CGA3D Similitude composition. s1, s2: (..., 16) -> (..., 16)"""
    ...

def compose_similitude_4d(s1: Tensor, s2: Tensor) -> Tensor:
    """CGA4D Similitude composition. s1, s2: (..., 32) -> (..., 32)"""
    ...

def compose_similitude_5d(s1: Tensor, s2: Tensor) -> Tensor:
    """CGA5D Similitude composition. s1, s2: (..., 64) -> (..., 64)"""
    ...


# === Inner Product ===

def inner_product_full_0d(a: Tensor, b: Tensor) -> Tensor:
    """CGA0D inner product. a, b: (..., 4) -> (..., 1)"""
    ...

def inner_product_full_1d(a: Tensor, b: Tensor) -> Tensor:
    """CGA1D inner product. a, b: (..., 8) -> (..., 1)"""
    ...

def inner_product_full_2d(a: Tensor, b: Tensor) -> Tensor:
    """CGA2D inner product. a, b: (..., 16) -> (..., 1)"""
    ...

def inner_product_full_3d(a: Tensor, b: Tensor) -> Tensor:
    """CGA3D inner product. a, b: (..., 32) -> (..., 1)"""
    ...

def inner_product_full_4d(a: Tensor, b: Tensor) -> Tensor:
    """CGA4D inner product. a, b: (..., 64) -> (..., 1)"""
    ...

def inner_product_full_5d(a: Tensor, b: Tensor) -> Tensor:
    """CGA5D inner product. a, b: (..., 128) -> (..., 1)"""
    ...


# === Exponential Map ===

def exp_bivector_0d(B: Tensor) -> Tensor:
    """CGA0D exp map. B: (..., 1) -> EvenVersor: (..., 2)"""
    ...

def exp_bivector_1d(B: Tensor) -> Tensor:
    """CGA1D exp map. B: (..., 3) -> EvenVersor: (..., 4)"""
    ...

def exp_bivector_2d(B: Tensor) -> Tensor:
    """CGA2D exp map. B: (..., 6) -> EvenVersor: (..., 8)"""
    ...

def exp_bivector_3d(B: Tensor) -> Tensor:
    """CGA3D exp map. B: (..., 10) -> EvenVersor: (..., 16)"""
    ...

def exp_bivector_4d(B: Tensor) -> Tensor:
    """CGA4D exp map. B: (..., 15) -> EvenVersor: (..., 32)"""
    ...

def exp_bivector_5d(B: Tensor) -> Tensor:
    """CGA5D exp map. B: (..., 21) -> EvenVersor: (..., 64)"""
    ...


# === Outer Product ===

def outer_product_full_0d(a: Tensor, b: Tensor) -> Tensor:
    """CGA0D outer product. a, b: (..., 4) -> (..., 4)"""
    ...

def outer_product_full_1d(a: Tensor, b: Tensor) -> Tensor:
    """CGA1D outer product. a, b: (..., 8) -> (..., 8)"""
    ...

def outer_product_full_2d(a: Tensor, b: Tensor) -> Tensor:
    """CGA2D outer product. a, b: (..., 16) -> (..., 16)"""
    ...

def outer_product_full_3d(a: Tensor, b: Tensor) -> Tensor:
    """CGA3D outer product. a, b: (..., 32) -> (..., 32)"""
    ...

def outer_product_full_4d(a: Tensor, b: Tensor) -> Tensor:
    """CGA4D outer product. a, b: (..., 64) -> (..., 64)"""
    ...

def outer_product_full_5d(a: Tensor, b: Tensor) -> Tensor:
    """CGA5D outer product. a, b: (..., 128) -> (..., 128)"""
    ...


# === Left Contraction ===

def left_contraction_full_0d(a: Tensor, b: Tensor) -> Tensor:
    """CGA0D left contraction. a, b: (..., 4) -> (..., 4)"""
    ...

def left_contraction_full_1d(a: Tensor, b: Tensor) -> Tensor:
    """CGA1D left contraction. a, b: (..., 8) -> (..., 8)"""
    ...

def left_contraction_full_2d(a: Tensor, b: Tensor) -> Tensor:
    """CGA2D left contraction. a, b: (..., 16) -> (..., 16)"""
    ...

def left_contraction_full_3d(a: Tensor, b: Tensor) -> Tensor:
    """CGA3D left contraction. a, b: (..., 32) -> (..., 32)"""
    ...

def left_contraction_full_4d(a: Tensor, b: Tensor) -> Tensor:
    """CGA4D left contraction. a, b: (..., 64) -> (..., 64)"""
    ...

def left_contraction_full_5d(a: Tensor, b: Tensor) -> Tensor:
    """CGA5D left contraction. a, b: (..., 128) -> (..., 128)"""
    ...


# === Right Contraction ===

def right_contraction_full_0d(a: Tensor, b: Tensor) -> Tensor:
    """CGA0D right contraction. a, b: (..., 4) -> (..., 4)"""
    ...

def right_contraction_full_1d(a: Tensor, b: Tensor) -> Tensor:
    """CGA1D right contraction. a, b: (..., 8) -> (..., 8)"""
    ...

def right_contraction_full_2d(a: Tensor, b: Tensor) -> Tensor:
    """CGA2D right contraction. a, b: (..., 16) -> (..., 16)"""
    ...

def right_contraction_full_3d(a: Tensor, b: Tensor) -> Tensor:
    """CGA3D right contraction. a, b: (..., 32) -> (..., 32)"""
    ...

def right_contraction_full_4d(a: Tensor, b: Tensor) -> Tensor:
    """CGA4D right contraction. a, b: (..., 64) -> (..., 64)"""
    ...

def right_contraction_full_5d(a: Tensor, b: Tensor) -> Tensor:
    """CGA5D right contraction. a, b: (..., 128) -> (..., 128)"""
    ...


# === Grade Selection ===

def grade_select_0d(mv: Tensor, grade: int) -> Tensor:
    """CGA0D grade selection. mv: (..., 4), grade: 0-2"""
    ...

def grade_select_1d(mv: Tensor, grade: int) -> Tensor:
    """CGA1D grade selection. mv: (..., 8), grade: 0-3"""
    ...

def grade_select_2d(mv: Tensor, grade: int) -> Tensor:
    """CGA2D grade selection. mv: (..., 16), grade: 0-4"""
    ...

def grade_select_3d(mv: Tensor, grade: int) -> Tensor:
    """CGA3D grade selection. mv: (..., 32), grade: 0-5"""
    ...

def grade_select_4d(mv: Tensor, grade: int) -> Tensor:
    """CGA4D grade selection. mv: (..., 64), grade: 0-6"""
    ...

def grade_select_5d(mv: Tensor, grade: int) -> Tensor:
    """CGA5D grade selection. mv: (..., 128), grade: 0-7"""
    ...


# === Dual ===

def dual_0d(mv: Tensor) -> Tensor:
    """CGA0D dual. mv: (..., 4) -> (..., 4)"""
    ...

def dual_1d(mv: Tensor) -> Tensor:
    """CGA1D dual. mv: (..., 8) -> (..., 8)"""
    ...

def dual_2d(mv: Tensor) -> Tensor:
    """CGA2D dual. mv: (..., 16) -> (..., 16)"""
    ...

def dual_3d(mv: Tensor) -> Tensor:
    """CGA3D dual. mv: (..., 32) -> (..., 32)"""
    ...

def dual_4d(mv: Tensor) -> Tensor:
    """CGA4D dual. mv: (..., 64) -> (..., 64)"""
    ...

def dual_5d(mv: Tensor) -> Tensor:
    """CGA5D dual. mv: (..., 128) -> (..., 128)"""
    ...


# === Normalize ===

def normalize_0d(mv: Tensor) -> Tensor:
    """CGA0D normalize. mv: (..., 4) -> (..., 4)"""
    ...

def normalize_1d(mv: Tensor) -> Tensor:
    """CGA1D normalize. mv: (..., 8) -> (..., 8)"""
    ...

def normalize_2d(mv: Tensor) -> Tensor:
    """CGA2D normalize. mv: (..., 16) -> (..., 16)"""
    ...

def normalize_3d(mv: Tensor) -> Tensor:
    """CGA3D normalize. mv: (..., 32) -> (..., 32)"""
    ...

def normalize_4d(mv: Tensor) -> Tensor:
    """CGA4D normalize. mv: (..., 64) -> (..., 64)"""
    ...

def normalize_5d(mv: Tensor) -> Tensor:
    """CGA5D normalize. mv: (..., 128) -> (..., 128)"""
    ...


# === Multivector Wrapper Class ===

class Multivector:
    """
    Multivector wrapper class with operator overloading.

    Attributes:
        data: Underlying tensor, shape (..., blade_count)
        algebra: CGA algebra instance
        kind: Type hint for optimization routing
    """
    data: Tensor
    algebra: 'CGAAlgebraExtendedOps'
    kind: Kind

    def __init__(self, data: Tensor, algebra: 'CGAAlgebraExtendedOps', kind: Kind = None) -> None: ...

    # Operators
    def __mul__(self, other: Union['Multivector', Tensor, float]) -> 'Multivector':
        """Geometric product / compose (routes based on kind)"""
        ...
    def __rmul__(self, other: Union[Tensor, float]) -> 'Multivector': ...
    def __xor__(self, other: 'Multivector') -> 'Multivector':
        """Outer product (wedge)"""
        ...
    def __or__(self, other: 'Multivector') -> Tensor:
        """Inner product (returns scalar tensor)"""
        ...
    def __lshift__(self, other: 'Multivector') -> 'Multivector':
        """Left contraction"""
        ...
    def __rshift__(self, other: 'Multivector') -> 'Multivector':
        """Right contraction"""
        ...
    def __matmul__(self, other: 'Multivector') -> 'Multivector':
        """Sandwich product"""
        ...
    def __invert__(self) -> 'Multivector':
        """Reverse"""
        ...
    def __pow__(self, n: int) -> 'Multivector':
        """Power / inverse (n=-1)"""
        ...
    def __add__(self, other: 'Multivector') -> 'Multivector': ...
    def __sub__(self, other: 'Multivector') -> 'Multivector': ...
    def __neg__(self) -> 'Multivector': ...
    def __truediv__(self, scalar: float) -> 'Multivector': ...

    # Methods
    def grade(self, k: int) -> 'Multivector':
        """Extract grade k components"""
        ...
    def dual(self) -> 'Multivector':
        """Compute dual"""
        ...
    def normalize(self) -> 'Multivector':
        """Normalize to unit norm"""
        ...
    def exp(self) -> 'Multivector':
        """Exponential map (for bivectors)"""
        ...
    def inverse(self) -> 'Multivector':
        """Compute inverse"""
        ...


class EvenVersor(Multivector):
    """Even-grade Versor (rotation, translation, scaling)"""
    ...


class Similitude(EvenVersor):
    """CGA Similitude (translation + rotation + scaling, no transversion)"""
    ...


class Bivector(Multivector):
    """Grade-2 multivector"""
    def exp(self) -> EvenVersor:
        """Exponential map: Bivector -> EvenVersor"""
        ...


# === Unified Layer Classes ===

class CliffordTransformLayer:
    """
    Unified Clifford algebra transform layer.

    Replaces dimension-specific CGA{n}DCareLayer.
    """
    dim: int
    even_versor_count: int
    point_count: int

    def __init__(self, dim: int) -> None: ...
    def forward(self, versor: Tensor, point: Tensor) -> Tensor: ...
    def __call__(self, versor: Tensor, point: Tensor) -> Tensor: ...


class CGAEncoder:
    """
    Unified UPGC encoder.

    Replaces dimension-specific UPGC{n}DEncoder.
    """
    dim: int

    def __init__(self, dim: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Euclidean coordinates, shape (..., dim)
        Returns:
            CGA point representation, shape (..., point_count)
        """
        ...
    def __call__(self, x: Tensor) -> Tensor: ...


class CGADecoder:
    """
    Unified UPGC decoder.

    Replaces dimension-specific UPGC{n}DDecoder.
    """
    dim: int

    def __init__(self, dim: int) -> None: ...
    def forward(self, p: Tensor) -> Tensor:
        """
        Args:
            p: CGA point representation, shape (..., point_count)
        Returns:
            Euclidean coordinates, shape (..., dim)
        """
        ...
    def __call__(self, p: Tensor) -> Tensor: ...


class CGAPipeline:
    """
    Unified transform pipeline: Encoder -> Transform -> Decoder.

    Replaces dimension-specific CGA{n}DTransformPipeline.
    """
    dim: int
    encoder: CGAEncoder
    transform: CliffordTransformLayer
    decoder: CGADecoder

    def __init__(self, dim: int) -> None: ...
    def forward(self, versor: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            versor: EvenVersor, shape (..., even_versor_count)
            x: Euclidean coordinates, shape (..., dim)
        Returns:
            Transformed Euclidean coordinates, shape (..., dim)
        """
        ...
    def __call__(self, versor: Tensor, x: Tensor) -> Tensor: ...
