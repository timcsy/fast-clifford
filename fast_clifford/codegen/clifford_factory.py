"""
Unified Clifford Algebra Factory - General Cl(p, q, r) support

Supports arbitrary Clifford algebra signatures:
- VGA(n) = Cl(n, 0, 0) - Vanilla Geometric Algebra
- CGA(n) = Cl(n+1, 1, 0) - Conformal Geometric Algebra
- PGA(n) = Cl(n, 0, 1) - Projective Geometric Algebra
- General Cl(p, q, r) - Any signature

Signature convention:
- p: number of basis vectors with e_i² = +1
- q: number of basis vectors with e_i² = -1
- r: number of basis vectors with e_i² = 0 (degenerate)

Basis ordering: e1, e2, ..., ep, ep+1, ..., ep+q, ep+q+1, ..., ep+q+r
where:
- e1² = e2² = ... = ep² = +1
- e(p+1)² = ... = e(p+q)² = -1
- e(p+q+1)² = ... = e(p+q+r)² = 0
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from clifford import Cl


def create_clifford_algebra(p: int, q: int = 0, r: int = 0):
    """
    Create a Clifford algebra Cl(p, q, r).

    Args:
        p: Positive signature dimension (e_i² = +1)
        q: Negative signature dimension (e_i² = -1)
        r: Degenerate dimension (e_i² = 0)

    Returns:
        layout: Clifford algebra layout object
        blades: Blade dictionary

    Raises:
        ValueError: If dimensions are negative

    Note:
        For high dimensions (p+q+r >= 10), computation may be slow
        and memory-intensive. Cl(n) has 2^n blades.
    """
    if p < 0 or q < 0 or r < 0:
        raise ValueError(f"Dimensions must be non-negative: p={p}, q={q}, r={r}")

    # Create Clifford algebra with signature
    # clifford library: Cl(p, q) for non-degenerate, Cl(p, q, r) not directly supported
    # For r > 0, we need special handling
    if r > 0:
        raise NotImplementedError(
            f"Degenerate signatures (r={r} > 0) require runtime implementation. "
            "Use RuntimeCliffordAlgebra for PGA."
        )

    layout, blades = Cl(p, q)
    return layout, blades


def compute_blade_count(p: int, q: int = 0, r: int = 0) -> int:
    """
    Compute total blade count for Cl(p, q, r).

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Total blade count = 2^(p+q+r)
    """
    return 2 ** (p + q + r)


def compute_rotor_count(p: int, q: int = 0, r: int = 0) -> int:
    """
    Compute rotor (even-grade) component count.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Sum of even-grade dimensions = 2^(n-1) where n = p+q+r
    """
    n = p + q + r
    if n == 0:
        return 1  # Only scalar
    return 2 ** (n - 1)


def compute_bivector_count(p: int, q: int = 0, r: int = 0) -> int:
    """
    Compute bivector (grade-2) component count.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        C(n, 2) = n*(n-1)/2 where n = p+q+r
    """
    n = p + q + r
    return n * (n - 1) // 2


def compute_grade_indices(p: int, q: int = 0, r: int = 0) -> Dict[int, Tuple[int, ...]]:
    """
    Compute blade indices for each grade.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Dict[grade, tuple of blade indices]
    """
    if r > 0:
        raise NotImplementedError("Degenerate signatures not supported for codegen")

    layout, _ = create_clifford_algebra(p, q, r)
    n = p + q + r
    blade_count = 2 ** n

    grade_indices = {g: [] for g in range(n + 1)}

    for idx in range(blade_count):
        blade_tuple = layout.bladeTupList[idx]
        grade = len(blade_tuple)
        grade_indices[grade].append(idx)

    return {g: tuple(indices) for g, indices in grade_indices.items()}


def compute_reverse_sign(grade: int) -> int:
    """
    Compute reverse sign for a given grade.

    Formula: (-1)^(k*(k-1)/2) where k is grade

    Args:
        grade: Blade grade

    Returns:
        +1 or -1
    """
    exponent = grade * (grade - 1) // 2
    return (-1) ** exponent


def compute_involute_sign(grade: int) -> int:
    """
    Compute grade involution sign for a given grade.

    Formula: (-1)^k where k is grade

    Args:
        grade: Blade grade

    Returns:
        +1 or -1
    """
    return (-1) ** grade


def compute_conjugate_sign(grade: int) -> int:
    """
    Compute Clifford conjugate sign for a given grade.

    Formula: (-1)^(k*(k+1)/2) = reverse * involute

    Args:
        grade: Blade grade

    Returns:
        +1 or -1
    """
    exponent = grade * (grade + 1) // 2
    return (-1) ** exponent


def compute_reverse_signs(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Compute reverse signs for all blades.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of reverse signs for each blade
    """
    grade_indices = compute_grade_indices(p, q, r)
    blade_count = compute_blade_count(p, q, r)

    # Build index -> grade mapping
    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    signs = []
    for idx in range(blade_count):
        grade = index_to_grade[idx]
        signs.append(compute_reverse_sign(grade))

    return tuple(signs)


def compute_involute_signs(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Compute grade involution signs for all blades.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of involution signs for each blade
    """
    grade_indices = compute_grade_indices(p, q, r)
    blade_count = compute_blade_count(p, q, r)

    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    signs = []
    for idx in range(blade_count):
        grade = index_to_grade[idx]
        signs.append(compute_involute_sign(grade))

    return tuple(signs)


def compute_conjugate_signs(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Compute Clifford conjugate signs for all blades.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of conjugate signs for each blade
    """
    grade_indices = compute_grade_indices(p, q, r)
    blade_count = compute_blade_count(p, q, r)

    index_to_grade = {}
    for grade, indices in grade_indices.items():
        for idx in indices:
            index_to_grade[idx] = grade

    signs = []
    for idx in range(blade_count):
        grade = index_to_grade[idx]
        signs.append(compute_conjugate_sign(grade))

    return tuple(signs)


def get_product_table(p: int, q: int = 0, r: int = 0) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Get geometric product multiplication table.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Dict[(left_idx, right_idx), (result_idx, sign)]
    """
    if r > 0:
        raise NotImplementedError("Degenerate signatures not supported for codegen")

    layout, _ = create_clifford_algebra(p, q, r)
    blade_count = compute_blade_count(p, q, r)

    # Extract GMT (Geometric Multiplication Table)
    gmt_dense = np.asarray(layout.gmt.todense())

    table = {}
    for i in range(blade_count):
        for j in range(blade_count):
            result_vec = gmt_dense[i, :, j]
            nonzero_indices = np.where(result_vec != 0)[0]

            for k in nonzero_indices:
                coeff = result_vec[k]
                sign = int(np.sign(coeff))
                if sign != 0:
                    table[(i, j)] = (k, sign)

    return table


def get_rotor_indices(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Get rotor (even-grade) blade indices.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of even-grade blade indices
    """
    grade_indices = compute_grade_indices(p, q, r)
    n = p + q + r

    indices = []
    for grade in range(0, n + 1, 2):
        if grade in grade_indices:
            indices.extend(grade_indices[grade])

    return tuple(sorted(indices))


def get_bivector_indices(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Get bivector (grade-2) blade indices.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of grade-2 blade indices
    """
    grade_indices = compute_grade_indices(p, q, r)
    return grade_indices.get(2, ())


def get_vector_indices(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Get vector (grade-1) blade indices.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of grade-1 blade indices
    """
    grade_indices = compute_grade_indices(p, q, r)
    return grade_indices.get(1, ())


def get_blade_names(p: int, q: int = 0, r: int = 0) -> List[str]:
    """
    Get human-readable names for all blades.

    Naming convention:
    - Positive basis: e1, e2, ..., ep
    - Negative basis: e(p+1), e(p+2), ..., e(p+q) with suffix '-'
    - Degenerate basis: e(p+q+1), ..., e(p+q+r) with suffix '0'

    For VGA: e1, e2, e3, ...
    For CGA: e1, e2, e3, e+, e- (special null basis names)

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        List of blade names
    """
    if r > 0:
        raise NotImplementedError("Degenerate signatures not supported for codegen")

    layout, _ = create_clifford_algebra(p, q, r)
    n = p + q + r

    # Build basis names
    basis_names = {}
    for i in range(p):
        basis_names[i + 1] = f"e{i+1}"
    for i in range(q):
        basis_names[p + i + 1] = f"e{p+i+1}"

    names = []
    for blade_tuple in layout.bladeTupList:
        if len(blade_tuple) == 0:
            names.append("1")  # Scalar
        else:
            blade_name = "".join(basis_names[i] for i in blade_tuple)
            names.append(blade_name)

    return names


def get_metric_signs(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Get metric signature for each basis vector.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of metric signs: +1, -1, or 0 for each basis vector
    """
    signs = [+1] * p + [-1] * q + [0] * r
    return tuple(signs)


def get_inner_product_signs(p: int, q: int = 0, r: int = 0) -> Tuple[int, ...]:
    """
    Get inner product signs for all blades.

    blade_i² depends on the metric signature of the basis vectors.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Tuple of blade² values for each blade
    """
    if r > 0:
        raise NotImplementedError("Degenerate signatures not supported")

    product_table = get_product_table(p, q, r)
    blade_count = compute_blade_count(p, q, r)

    signs = []
    for idx in range(blade_count):
        if (idx, idx) in product_table:
            result_idx, sign = product_table[(idx, idx)]
            if result_idx == 0:  # Result is scalar
                signs.append(sign)
            else:
                signs.append(0)
        else:
            signs.append(0)

    return tuple(signs)


def get_pseudoscalar_info(p: int, q: int = 0, r: int = 0) -> Dict[str, any]:
    """
    Get pseudoscalar information.

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        r: Degenerate dimension

    Returns:
        Dict with 'index', 'grade', 'square' keys
    """
    grade_indices = compute_grade_indices(p, q, r)
    n = p + q + r
    max_grade = n

    if max_grade not in grade_indices or len(grade_indices[max_grade]) == 0:
        return {'index': -1, 'grade': max_grade, 'square': 0}

    pseudoscalar_idx = grade_indices[max_grade][0]

    # Compute pseudoscalar²
    product_table = get_product_table(p, q, r)
    if (pseudoscalar_idx, pseudoscalar_idx) in product_table:
        result_idx, sign = product_table[(pseudoscalar_idx, pseudoscalar_idx)]
        square_value = sign if result_idx == 0 else 0
    else:
        square_value = 0

    return {
        'index': pseudoscalar_idx,
        'grade': max_grade,
        'square': square_value,
    }


# =============================================================================
# Algebra Type Detection
# =============================================================================

def get_algebra_type(p: int, q: int = 0, r: int = 0) -> str:
    """
    Detect algebra type based on signature.

    Returns:
        'vga' if q == 0 and r == 0
        'cga' if q == 1 and r == 0
        'pga' if r > 0
        'general' otherwise
    """
    if r > 0:
        return "pga"
    elif q == 0:
        return "vga"
    elif q == 1:
        return "cga"
    else:
        return "general"


# =============================================================================
# CGA-specific helpers (for backward compatibility)
# =============================================================================

def create_cga_algebra(euclidean_dim: int):
    """
    Create CGA algebra Cl(n+1, 1).

    This is a convenience wrapper for CGA-specific operations.

    Args:
        euclidean_dim: Euclidean space dimension

    Returns:
        layout, blades, stuff (CGA-specific objects)
    """
    from clifford import conformalize

    # Create base Euclidean algebra Cl(n)
    G_n, _ = Cl(euclidean_dim)

    # Conformalize to get CGA Cl(n+1, 1)
    layout, blades, stuff = conformalize(G_n)

    return layout, blades, stuff


def get_cga_point_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    Get CGA Point (grade-1) blade indices.

    Args:
        euclidean_dim: Euclidean space dimension

    Returns:
        Grade-1 blade indices for CGA(n)
    """
    # CGA(n) = Cl(n+1, 1)
    p = euclidean_dim + 1
    q = 1
    return get_vector_indices(p, q, 0)


def get_cga_rotor_indices(euclidean_dim: int) -> Tuple[int, ...]:
    """
    Get CGA Rotor (even-grade) blade indices.

    Args:
        euclidean_dim: Euclidean space dimension

    Returns:
        Even-grade blade indices for CGA(n)
    """
    p = euclidean_dim + 1
    q = 1
    return get_rotor_indices(p, q, 0)


# =============================================================================
# VGA-specific helpers
# =============================================================================

def get_vga_vector_indices(n: int) -> Tuple[int, ...]:
    """
    Get VGA vector (grade-1) blade indices.

    Args:
        n: VGA dimension (VGA(n) = Cl(n, 0))

    Returns:
        Grade-1 blade indices
    """
    return get_vector_indices(n, 0, 0)


def get_vga_rotor_indices(n: int) -> Tuple[int, ...]:
    """
    Get VGA Rotor (even-grade) blade indices.

    Args:
        n: VGA dimension

    Returns:
        Even-grade blade indices
    """
    return get_rotor_indices(n, 0, 0)


if __name__ == "__main__":
    print("=== Clifford Factory Test ===")

    # Test VGA
    for n in [2, 3, 4]:
        print(f"\n--- VGA({n}) = Cl({n}, 0) ---")
        print(f"Blade count: {compute_blade_count(n, 0)}")
        print(f"Rotor count: {compute_rotor_count(n, 0)}")
        print(f"Bivector count: {compute_bivector_count(n, 0)}")
        print(f"Type: {get_algebra_type(n, 0)}")
        print(f"Rotor indices: {get_rotor_indices(n, 0)}")

    # Test CGA (via Cl(p, q))
    for euclidean_dim in [1, 2, 3]:
        p = euclidean_dim + 1
        q = 1
        print(f"\n--- CGA({euclidean_dim}) = Cl({p}, {q}) ---")
        print(f"Blade count: {compute_blade_count(p, q)}")
        print(f"Rotor count: {compute_rotor_count(p, q)}")
        print(f"Bivector count: {compute_bivector_count(p, q)}")
        print(f"Type: {get_algebra_type(p, q)}")
        print(f"Rotor indices: {get_rotor_indices(p, q)}")
        print(f"Point indices: {get_cga_point_indices(euclidean_dim)}")

    # Test general Cl(2, 2)
    print(f"\n--- Cl(2, 2) ---")
    print(f"Blade count: {compute_blade_count(2, 2)}")
    print(f"Rotor count: {compute_rotor_count(2, 2)}")
    print(f"Type: {get_algebra_type(2, 2)}")
