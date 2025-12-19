"""
Unified Clifford Algebra Code Generator - ClCodeGenerator

Generates hard-coded PyTorch functions for any Cl(p, q) algebra with:
- Loop-free (fully expanded arithmetic)
- ONNX-compatible operations
- torch.jit.script compatible

Supports:
- VGA(n) = Cl(n, 0) - Vanilla Geometric Algebra
- CGA(n) = Cl(n+1, 1) - Conformal Geometric Algebra
- General Cl(p, q) - Any non-degenerate signature
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .clifford_factory import (
    compute_blade_count,
    compute_rotor_count,
    compute_bivector_count,
    compute_grade_indices,
    compute_reverse_signs,
    compute_involute_signs,
    compute_conjugate_signs,
    get_product_table,
    get_rotor_indices,
    get_bivector_indices,
    get_vector_indices,
    get_blade_names,
    get_inner_product_signs,
    get_pseudoscalar_info,
    get_algebra_type,
)


class ClCodeGenerator:
    """
    Unified code generator for Clifford algebras Cl(p, q).

    Generates PyTorch code with:
    - Hard-coded arithmetic (no Cayley table lookups)
    - No loops (fully expanded)
    - ONNX-compatible operations only
    - torch.jit.script decorators

    Usage:
        gen = ClCodeGenerator(3, 0)  # VGA(3)
        code = gen.generate_module()
        # Write to file
    """

    # Maximum blade count for torch.jit.script (larger algebras cause recursion errors)
    JIT_BLADE_THRESHOLD = 256  # p+q <= 8

    def __init__(self, p: int, q: int = 0):
        """
        Initialize generator for Cl(p, q).

        Args:
            p: Positive signature dimension
            q: Negative signature dimension
        """
        self.p = p
        self.q = q
        self.n = p + q  # Total dimension

        # Compute algebra properties
        self.blade_count = compute_blade_count(p, q)
        self.rotor_count = compute_rotor_count(p, q)
        self.bivector_count = compute_bivector_count(p, q)

        # Determine if JIT should be used (large algebras cause recursion errors)
        self.use_jit = self.blade_count <= self.JIT_BLADE_THRESHOLD

        # Get algebra data
        self.grade_indices = compute_grade_indices(p, q)
        self.reverse_signs = compute_reverse_signs(p, q)
        self.involute_signs = compute_involute_signs(p, q)
        self.conjugate_signs = compute_conjugate_signs(p, q)
        self.product_table = get_product_table(p, q)
        self.rotor_indices = get_rotor_indices(p, q)
        self.bivector_indices = get_bivector_indices(p, q)
        self.vector_indices = get_vector_indices(p, q)
        self.blade_names = get_blade_names(p, q)
        self.inner_product_signs = get_inner_product_signs(p, q)
        self.pseudoscalar_info = get_pseudoscalar_info(p, q)
        self.algebra_type = get_algebra_type(p, q)

        # Organize products by result for efficient generation
        self._products_by_result = self._organize_products_by_result()

        # Build index mappings
        self._rotor_to_full = {i: idx for i, idx in enumerate(self.rotor_indices)}
        self._full_to_rotor = {idx: i for i, idx in enumerate(self.rotor_indices)}
        self._bivector_to_full = {i: idx for i, idx in enumerate(self.bivector_indices)}
        self._full_to_bivector = {idx: i for i, idx in enumerate(self.bivector_indices)}

    def _jit_decorator(self) -> str:
        """Return JIT decorator line or empty string."""
        return "@torch.jit.script" if self.use_jit else "# @torch.jit.script  # Disabled for large algebras"

    def _organize_products_by_result(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Organize product rules by result index."""
        result = {k: [] for k in range(self.blade_count)}
        for (left, right), (res, sign) in self.product_table.items():
            result[res].append((left, right, sign))
        return result

    def generate_module(self) -> str:
        """Generate complete module code."""
        sections = [
            self.generate_header(),
            self.generate_constants(),
            self.generate_geometric_product(),
            self.generate_reverse(),
            self.generate_involute(),
            self.generate_conjugate(),
            self.generate_select_grade(),
            self.generate_inner_product(),
            self.generate_outer_product(),
            self.generate_left_contraction(),
            self.generate_right_contraction(),
            self.generate_dual(),
            self.generate_norm_squared(),
            self.generate_compose_rotor(),
            self.generate_reverse_rotor(),
            self.generate_sandwich_rotor(),
            self.generate_norm_squared_rotor(),
            self.generate_exp_bivector(),
            self.generate_log_rotor(),
            self.generate_slerp_rotor(),
        ]
        return "\n".join(sections)

    def generate_header(self) -> str:
        """Generate module header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'''"""
Clifford Algebra Cl({self.p}, {self.q}) Functional Operations - Auto-generated

DO NOT EDIT MANUALLY - This file is generated by codegen/generator.py

Generated: {timestamp}
Signature: Cl({self.p}, {self.q})
Blade count: {self.blade_count}
Rotor count: {self.rotor_count}
Bivector count: {self.bivector_count}
Algebra type: {self.algebra_type}

All functions are:
- Loop-free (fully expanded arithmetic)
- ONNX-compatible (only Add/Mul/Neg/Sub operations)
- torch.jit.script compatible
"""

import torch
from torch import Tensor
from typing import Tuple

'''

    def generate_constants(self) -> str:
        """Generate constant definitions."""
        lines = [
            "# =============================================================================",
            "# Constants",
            "# =============================================================================",
            "",
            f"BLADE_COUNT = {self.blade_count}",
            f"ROTOR_COUNT = {self.rotor_count}",
            f"BIVECTOR_COUNT = {self.bivector_count}",
            "",
            "# Blade indices by grade",
        ]

        for grade in range(self.n + 1):
            indices = self.grade_indices.get(grade, ())
            lines.append(f"GRADE_{grade}_INDICES = {indices}")

        lines.extend([
            "",
            "# Rotor (even-grade) indices",
            f"ROTOR_MASK = {self.rotor_indices}",
            "",
            "# Bivector (grade-2) indices",
            f"BIVECTOR_MASK = {self.bivector_indices}",
            "",
            "# Vector (grade-1) indices",
            f"VECTOR_MASK = {self.vector_indices}",
            "",
            "# Reverse signs for all blades",
            f"REVERSE_SIGNS = {self.reverse_signs}",
            "",
            "# Involute signs for all blades",
            f"INVOLUTE_SIGNS = {self.involute_signs}",
            "",
            "# Conjugate signs for all blades",
            f"CONJUGATE_SIGNS = {self.conjugate_signs}",
            "",
            "# Inner product signs (blade² values)",
            f"INNER_PRODUCT_SIGNS = {self.inner_product_signs}",
            "",
            "# Pseudoscalar info",
            f"PSEUDOSCALAR_INDEX = {self.pseudoscalar_info['index']}",
            f"PSEUDOSCALAR_SQUARE = {self.pseudoscalar_info['square']}",
            "",
        ])

        return "\n".join(lines)

    def generate_geometric_product(self) -> str:
        """Generate full geometric product function."""
        lines = [
            "# =============================================================================",
            "# Geometric Product",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def geometric_product(a: Tensor, b: Tensor) -> Tensor:",
            '    """',
            f"    Compute geometric product of two multivectors in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        a: Left operand, shape (..., {self.blade_count})",
            f"        b: Right operand, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Result multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        # Generate computation for each result index
        for result_idx in range(self.blade_count):
            terms = self._products_by_result[result_idx]
            if not terms:
                lines.append(f"    r{result_idx} = torch.zeros_like(a[..., 0])")
            else:
                term_strs = []
                for left, right, sign in terms:
                    if sign == 1:
                        term_strs.append(f"a[..., {left}] * b[..., {right}]")
                    else:
                        term_strs.append(f"-a[..., {left}] * b[..., {right}]")

                if len(term_strs) <= 3:
                    lines.append(f"    r{result_idx} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    r{result_idx} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        # Stack results
        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_reverse(self) -> str:
        """Generate reverse operation."""
        lines = [
            "# =============================================================================",
            "# Reverse Operation",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def reverse(mv: Tensor) -> Tensor:",
            '    """',
            f"    Compute reverse of a multivector in Cl({self.p}, {self.q}).",
            "",
            "    For grade k: coefficient *= (-1)^(k*(k-1)/2)",
            "",
            "    Args:",
            f"        mv: Input multivector, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Reversed multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        for idx in range(self.blade_count):
            sign = self.reverse_signs[idx]
            if sign == 1:
                lines.append(f"    r{idx} = mv[..., {idx}]")
            else:
                lines.append(f"    r{idx} = -mv[..., {idx}]")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_involute(self) -> str:
        """Generate grade involution operation."""
        lines = [
            "# =============================================================================",
            "# Grade Involution",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def involute(mv: Tensor) -> Tensor:",
            '    """',
            f"    Compute grade involution of a multivector in Cl({self.p}, {self.q}).",
            "",
            "    For grade k: coefficient *= (-1)^k",
            "",
            "    Args:",
            f"        mv: Input multivector, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Grade-involuted multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        for idx in range(self.blade_count):
            sign = self.involute_signs[idx]
            if sign == 1:
                lines.append(f"    r{idx} = mv[..., {idx}]")
            else:
                lines.append(f"    r{idx} = -mv[..., {idx}]")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_conjugate(self) -> str:
        """Generate Clifford conjugate operation."""
        lines = [
            "# =============================================================================",
            "# Clifford Conjugate",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def conjugate(mv: Tensor) -> Tensor:",
            '    """',
            f"    Compute Clifford conjugate of a multivector in Cl({self.p}, {self.q}).",
            "",
            "    For grade k: coefficient *= (-1)^(k*(k+1)/2)",
            "",
            "    Args:",
            f"        mv: Input multivector, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Conjugated multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        for idx in range(self.blade_count):
            sign = self.conjugate_signs[idx]
            if sign == 1:
                lines.append(f"    r{idx} = mv[..., {idx}]")
            else:
                lines.append(f"    r{idx} = -mv[..., {idx}]")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_select_grade(self) -> str:
        """Generate grade selection functions."""
        lines = [
            "# =============================================================================",
            "# Grade Selection",
            "# =============================================================================",
            "",
        ]

        # Generate individual grade selection functions
        for grade in range(self.n + 1):
            indices = self.grade_indices.get(grade, ())
            lines.extend([
                self._jit_decorator(),
                f"def select_grade_{grade}(mv: Tensor) -> Tensor:",
                '    """',
                f"    Extract grade-{grade} components from multivector.",
                "",
                "    Args:",
                f"        mv: Input multivector, shape (..., {self.blade_count})",
                "",
                "    Returns:",
                f"        Multivector with only grade-{grade} components, shape (..., {self.blade_count})",
                '    """',
            ])

            for idx in range(self.blade_count):
                if idx in indices:
                    lines.append(f"    r{idx} = mv[..., {idx}]")
                else:
                    lines.append(f"    r{idx} = torch.zeros_like(mv[..., {idx}])")

            lines.append("")
            lines.append("    return torch.stack([")
            chunk_size = 8
            for i in range(0, self.blade_count, chunk_size):
                chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
                lines.append(f"        {chunk},")
            lines.append("    ], dim=-1)")
            lines.append("")
            lines.append("")

        return "\n".join(lines)

    def generate_inner_product(self) -> str:
        """Generate inner product (scalar product)."""
        lines = [
            "# =============================================================================",
            "# Inner Product",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def inner(a: Tensor, b: Tensor) -> Tensor:",
            '    """',
            f"    Compute inner product <ab>_0 in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        a: Left operand, shape (..., {self.blade_count})",
            f"        b: Right operand, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            "        Scalar result, shape (..., 1)",
            '    """',
        ]

        # Inner product: <ab>_0 = sum of products that give scalar
        scalar_terms = []
        for (left, right), (res, sign) in self.product_table.items():
            if res == 0:  # Result is scalar
                if sign == 1:
                    scalar_terms.append(f"a[..., {left}] * b[..., {right}]")
                else:
                    scalar_terms.append(f"-a[..., {left}] * b[..., {right}]")

        if scalar_terms:
            if len(scalar_terms) <= 4:
                lines.append(f"    result = {' + '.join(scalar_terms)}")
            else:
                lines.append("    result = (")
                for i, term in enumerate(scalar_terms):
                    if i < len(scalar_terms) - 1:
                        lines.append(f"        {term} +")
                    else:
                        lines.append(f"        {term}")
                lines.append("    )")
        else:
            lines.append("    result = torch.zeros_like(a[..., 0])")

        lines.append("    return result.unsqueeze(-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_outer_product(self) -> str:
        """Generate outer (wedge) product."""
        lines = [
            "# =============================================================================",
            "# Outer Product (Wedge)",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def outer(a: Tensor, b: Tensor) -> Tensor:",
            '    """',
            f"    Compute outer product a ^ b in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        a: Left operand, shape (..., {self.blade_count})",
            f"        b: Right operand, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Result multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        # Build index -> grade mapping
        index_to_grade = {}
        for grade, indices in self.grade_indices.items():
            for idx in indices:
                index_to_grade[idx] = grade

        # Organize outer product terms by result
        outer_by_result = {k: [] for k in range(self.blade_count)}
        for (left, right), (res, sign) in self.product_table.items():
            grade_left = index_to_grade[left]
            grade_right = index_to_grade[right]
            grade_res = index_to_grade[res]
            if grade_res == grade_left + grade_right:
                outer_by_result[res].append((left, right, sign))

        for result_idx in range(self.blade_count):
            terms = outer_by_result[result_idx]
            if not terms:
                lines.append(f"    r{result_idx} = torch.zeros_like(a[..., 0])")
            else:
                term_strs = []
                for left, right, sign in terms:
                    if sign == 1:
                        term_strs.append(f"a[..., {left}] * b[..., {right}]")
                    else:
                        term_strs.append(f"-a[..., {left}] * b[..., {right}]")

                if len(term_strs) <= 3:
                    lines.append(f"    r{result_idx} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    r{result_idx} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_left_contraction(self) -> str:
        """Generate left contraction."""
        lines = [
            "# =============================================================================",
            "# Left Contraction",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def contract_left(a: Tensor, b: Tensor) -> Tensor:",
            '    """',
            f"    Compute left contraction a << b in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        a: Left operand, shape (..., {self.blade_count})",
            f"        b: Right operand, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Result multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        index_to_grade = {}
        for grade, indices in self.grade_indices.items():
            for idx in indices:
                index_to_grade[idx] = grade

        lc_by_result = {k: [] for k in range(self.blade_count)}
        for (left, right), (res, sign) in self.product_table.items():
            grade_left = index_to_grade[left]
            grade_right = index_to_grade[right]
            grade_res = index_to_grade[res]
            if grade_left <= grade_right and grade_res == grade_right - grade_left:
                lc_by_result[res].append((left, right, sign))

        for result_idx in range(self.blade_count):
            terms = lc_by_result[result_idx]
            if not terms:
                lines.append(f"    r{result_idx} = torch.zeros_like(a[..., 0])")
            else:
                term_strs = []
                for left, right, sign in terms:
                    if sign == 1:
                        term_strs.append(f"a[..., {left}] * b[..., {right}]")
                    else:
                        term_strs.append(f"-a[..., {left}] * b[..., {right}]")

                if len(term_strs) <= 3:
                    lines.append(f"    r{result_idx} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    r{result_idx} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_right_contraction(self) -> str:
        """Generate right contraction."""
        lines = [
            "# =============================================================================",
            "# Right Contraction",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def contract_right(a: Tensor, b: Tensor) -> Tensor:",
            '    """',
            f"    Compute right contraction a >> b in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        a: Left operand, shape (..., {self.blade_count})",
            f"        b: Right operand, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Result multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        index_to_grade = {}
        for grade, indices in self.grade_indices.items():
            for idx in indices:
                index_to_grade[idx] = grade

        rc_by_result = {k: [] for k in range(self.blade_count)}
        for (left, right), (res, sign) in self.product_table.items():
            grade_left = index_to_grade[left]
            grade_right = index_to_grade[right]
            grade_res = index_to_grade[res]
            if grade_left >= grade_right and grade_res == grade_left - grade_right:
                rc_by_result[res].append((left, right, sign))

        for result_idx in range(self.blade_count):
            terms = rc_by_result[result_idx]
            if not terms:
                lines.append(f"    r{result_idx} = torch.zeros_like(a[..., 0])")
            else:
                term_strs = []
                for left, right, sign in terms:
                    if sign == 1:
                        term_strs.append(f"a[..., {left}] * b[..., {right}]")
                    else:
                        term_strs.append(f"-a[..., {left}] * b[..., {right}]")

                if len(term_strs) <= 3:
                    lines.append(f"    r{result_idx} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    r{result_idx} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_dual(self) -> str:
        """Generate dual operation."""
        lines = [
            "# =============================================================================",
            "# Dual (Left Contraction with Pseudoscalar)",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def dual(mv: Tensor) -> Tensor:",
            '    """',
            f"    Compute dual mv* = mv << I in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        mv: Input multivector, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Dual multivector, shape (..., {self.blade_count})",
            '    """',
        ]

        # Dual: mv << I where I is pseudoscalar
        ps_idx = self.pseudoscalar_info['index']

        # For each result, find terms from mv[i] * I[ps_idx] that give grade(result)
        index_to_grade = {}
        for grade, indices in self.grade_indices.items():
            for idx in indices:
                index_to_grade[idx] = grade

        for result_idx in range(self.blade_count):
            # Find (left, ps_idx) -> (result_idx, sign)
            if (result_idx, ps_idx) in self.product_table:
                # We need reverse lookup: what left gives this result when multiplied with ps_idx?
                pass

        # Simplified: compute via left contraction with pseudoscalar
        # result[k] = sum_i mv[i] * I[j] where product gives k and grade condition met
        dual_by_result = {k: [] for k in range(self.blade_count)}
        grade_ps = index_to_grade[ps_idx]

        for left in range(self.blade_count):
            if (left, ps_idx) in self.product_table:
                res, sign = self.product_table[(left, ps_idx)]
                grade_left = index_to_grade[left]
                grade_res = index_to_grade[res]
                # Left contraction condition
                if grade_left <= grade_ps and grade_res == grade_ps - grade_left:
                    dual_by_result[res].append((left, sign))

        for result_idx in range(self.blade_count):
            terms = dual_by_result[result_idx]
            if not terms:
                lines.append(f"    r{result_idx} = torch.zeros_like(mv[..., 0])")
            else:
                term_strs = []
                for left, sign in terms:
                    if sign == 1:
                        term_strs.append(f"mv[..., {left}]")
                    else:
                        term_strs.append(f"-mv[..., {left}]")

                if len(term_strs) == 1:
                    lines.append(f"    r{result_idx} = {term_strs[0]}")
                else:
                    lines.append(f"    r{result_idx} = {' + '.join(term_strs)}")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_norm_squared(self) -> str:
        """Generate norm squared operation."""
        lines = [
            "# =============================================================================",
            "# Norm Squared",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            "def norm_squared(mv: Tensor) -> Tensor:",
            '    """',
            f"    Compute norm squared |mv|^2 = <mv * ~mv>_0 in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        mv: Input multivector, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            "        Norm squared, shape (..., 1)",
            '    """',
        ]

        # |mv|^2 = sum of mv[i]^2 * reverse_sign[i] * inner_sign[i]
        terms = []
        for idx in range(self.blade_count):
            combined = self.reverse_signs[idx] * self.inner_product_signs[idx]
            if combined != 0:
                if combined == 1:
                    terms.append(f"mv[..., {idx}] * mv[..., {idx}]")
                else:
                    terms.append(f"-mv[..., {idx}] * mv[..., {idx}]")

        if terms:
            if len(terms) <= 4:
                lines.append(f"    result = {' + '.join(terms)}")
            else:
                lines.append("    result = (")
                for i, term in enumerate(terms):
                    if i < len(terms) - 1:
                        lines.append(f"        {term} +")
                    else:
                        lines.append(f"        {term}")
                lines.append("    )")
        else:
            lines.append("    result = torch.zeros_like(mv[..., 0])")

        lines.append("    return result.unsqueeze(-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_compose_rotor(self) -> str:
        """Generate rotor composition function."""
        lines = [
            "# =============================================================================",
            "# Rotor Composition",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def compose_rotor(r1: Tensor, r2: Tensor) -> Tensor:",
            '    """',
            f"    Compose two rotors r1 * r2 in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        r1: Left rotor, shape (..., {self.rotor_count})",
            f"        r2: Right rotor, shape (..., {self.rotor_count})",
            "",
            "    Returns:",
            f"        Composed rotor, shape (..., {self.rotor_count})",
            '    """',
        ]

        # Build rotor-only product table
        rotor_set = set(self.rotor_indices)
        rotor_products = {i: [] for i in range(self.rotor_count)}

        for (left, right), (res, sign) in self.product_table.items():
            if left in rotor_set and right in rotor_set and res in rotor_set:
                left_sparse = self._full_to_rotor[left]
                right_sparse = self._full_to_rotor[right]
                res_sparse = self._full_to_rotor[res]
                rotor_products[res_sparse].append((left_sparse, right_sparse, sign))

        # Use 'out' prefix to avoid conflict with input params r1, r2
        for result_sparse in range(self.rotor_count):
            terms = rotor_products[result_sparse]
            if not terms:
                lines.append(f"    out{result_sparse} = torch.zeros_like(r1[..., 0])")
            else:
                term_strs = []
                for left_s, right_s, sign in terms:
                    if sign == 1:
                        term_strs.append(f"r1[..., {left_s}] * r2[..., {right_s}]")
                    else:
                        term_strs.append(f"-r1[..., {left_s}] * r2[..., {right_s}]")

                if len(term_strs) <= 3:
                    lines.append(f"    out{result_sparse} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    out{result_sparse} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.rotor_count, chunk_size):
            chunk = ", ".join(f"out{j}" for j in range(i, min(i + chunk_size, self.rotor_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_reverse_rotor(self) -> str:
        """Generate rotor reverse function."""
        lines = [
            "# =============================================================================",
            "# Rotor Reverse",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def reverse_rotor(r: Tensor) -> Tensor:",
            '    """',
            f"    Compute reverse of a rotor in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        r: Input rotor, shape (..., {self.rotor_count})",
            "",
            "    Returns:",
            f"        Reversed rotor, shape (..., {self.rotor_count})",
            '    """',
        ]

        for sparse_idx in range(self.rotor_count):
            full_idx = self._rotor_to_full[sparse_idx]
            sign = self.reverse_signs[full_idx]
            if sign == 1:
                lines.append(f"    r{sparse_idx} = r[..., {sparse_idx}]")
            else:
                lines.append(f"    r{sparse_idx} = -r[..., {sparse_idx}]")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.rotor_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.rotor_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_sandwich_rotor(self) -> str:
        """Generate rotor sandwich product function."""
        lines = [
            "# =============================================================================",
            "# Rotor Sandwich Product",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def sandwich_rotor(r: Tensor, x: Tensor) -> Tensor:",
            '    """',
            f"    Compute sandwich product r * x * ~r in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        r: Rotor, shape (..., {self.rotor_count})",
            f"        x: Operand multivector, shape (..., {self.blade_count})",
            "",
            "    Returns:",
            f"        Transformed multivector, shape (..., {self.blade_count})",
            "",
            "    Note:",
            "        This is a simplified implementation. Optimized sparse versions",
            "        can be generated for specific input types (e.g., vectors).",
            '    """',
            "    # Expand rotor to full multivector",
        ]

        # Generate rotor expansion to full mv
        for full_idx in range(self.blade_count):
            if full_idx in self._full_to_rotor:
                sparse_idx = self._full_to_rotor[full_idx]
                lines.append(f"    rf{full_idx} = r[..., {sparse_idx}]")
            else:
                lines.append(f"    rf{full_idx} = torch.zeros_like(r[..., 0])")

        lines.append("")
        lines.append("    # Compute r * x")

        # First product: r * x
        for result_idx in range(self.blade_count):
            terms = self._products_by_result[result_idx]
            if not terms:
                lines.append(f"    t{result_idx} = torch.zeros_like(x[..., 0])")
            else:
                term_strs = []
                for left, right, sign in terms:
                    if sign == 1:
                        term_strs.append(f"rf{left} * x[..., {right}]")
                    else:
                        term_strs.append(f"-rf{left} * x[..., {right}]")

                if len(term_strs) <= 3:
                    lines.append(f"    t{result_idx} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    t{result_idx} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        lines.append("")
        lines.append("    # Compute ~r")

        # Reverse rotor
        for full_idx in range(self.blade_count):
            if full_idx in self._full_to_rotor:
                sign = self.reverse_signs[full_idx]
                if sign == 1:
                    lines.append(f"    rr{full_idx} = rf{full_idx}")
                else:
                    lines.append(f"    rr{full_idx} = -rf{full_idx}")
            else:
                lines.append(f"    rr{full_idx} = torch.zeros_like(r[..., 0])")

        lines.append("")
        lines.append("    # Compute (r * x) * ~r")

        # Second product: (r*x) * ~r
        for result_idx in range(self.blade_count):
            terms = self._products_by_result[result_idx]
            if not terms:
                lines.append(f"    s{result_idx} = torch.zeros_like(x[..., 0])")
            else:
                term_strs = []
                for left, right, sign in terms:
                    if sign == 1:
                        term_strs.append(f"t{left} * rr{right}")
                    else:
                        term_strs.append(f"-t{left} * rr{right}")

                if len(term_strs) <= 3:
                    lines.append(f"    s{result_idx} = {' + '.join(term_strs)}")
                else:
                    lines.append(f"    s{result_idx} = (")
                    for i, term in enumerate(term_strs):
                        if i < len(term_strs) - 1:
                            lines.append(f"        {term} +")
                        else:
                            lines.append(f"        {term}")
                    lines.append("    )")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.blade_count, chunk_size):
            chunk = ", ".join(f"s{j}" for j in range(i, min(i + chunk_size, self.blade_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_norm_squared_rotor(self) -> str:
        """Generate rotor norm squared function."""
        lines = [
            "# =============================================================================",
            "# Rotor Norm Squared",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def norm_squared_rotor(r: Tensor) -> Tensor:",
            '    """',
            f"    Compute norm squared of a rotor in Cl({self.p}, {self.q}).",
            "",
            "    Args:",
            f"        r: Input rotor, shape (..., {self.rotor_count})",
            "",
            "    Returns:",
            "        Norm squared, shape (..., 1)",
            '    """',
        ]

        terms = []
        for sparse_idx in range(self.rotor_count):
            full_idx = self._rotor_to_full[sparse_idx]
            combined = self.reverse_signs[full_idx] * self.inner_product_signs[full_idx]
            if combined != 0:
                if combined == 1:
                    terms.append(f"r[..., {sparse_idx}] * r[..., {sparse_idx}]")
                else:
                    terms.append(f"-r[..., {sparse_idx}] * r[..., {sparse_idx}]")

        if terms:
            if len(terms) <= 4:
                lines.append(f"    result = {' + '.join(terms)}")
            else:
                lines.append("    result = (")
                for i, term in enumerate(terms):
                    if i < len(terms) - 1:
                        lines.append(f"        {term} +")
                    else:
                        lines.append(f"        {term}")
                lines.append("    )")
        else:
            lines.append("    result = torch.zeros_like(r[..., 0])")

        lines.append("    return result.unsqueeze(-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_exp_bivector(self) -> str:
        """Generate bivector exponential function (bivector -> rotor)."""
        lines = [
            "# =============================================================================",
            "# Bivector Exponential (exp_bivector)",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def exp_bivector(B: Tensor) -> Tensor:",
            '    """',
            f"    Compute exponential of a bivector in Cl({self.p}, {self.q}).",
            "",
            "    Converts a bivector (generator) to a rotor (rotation).",
            "    Formula: exp(B) = cos(θ) + sin(θ)/θ * B where θ = sqrt(-B²)",
            "",
            "    Args:",
            f"        B: Input bivector, shape (..., {self.bivector_count})",
            "",
            "    Returns:",
            f"        Rotor, shape (..., {self.rotor_count})",
            '    """',
        ]

        # Compute B² (bivector squared -> scalar)
        # B² = sum of B_ij * B_ij * sign(e_ij * e_ij)
        # For bivectors, need to look up self-products
        bivector_sq_terms = []
        for sparse_idx, full_idx in enumerate(self.bivector_indices):
            # Look up e_ij * e_ij in product table
            # product_table[(i,j)] returns (result_index, sign)
            result = self.product_table.get((full_idx, full_idx))
            if result is not None:
                result_idx, sign = result
                # Only include if result is scalar (index 0)
                if result_idx == 0 and sign != 0:
                    if sign == 1:
                        bivector_sq_terms.append(f"B[..., {sparse_idx}] * B[..., {sparse_idx}]")
                    else:
                        bivector_sq_terms.append(f"-B[..., {sparse_idx}] * B[..., {sparse_idx}]")

        if bivector_sq_terms:
            if len(bivector_sq_terms) <= 4:
                lines.append(f"    B_sq = {' + '.join(bivector_sq_terms)}")
            else:
                lines.append("    B_sq = (")
                for i, term in enumerate(bivector_sq_terms):
                    if i < len(bivector_sq_terms) - 1:
                        lines.append(f"        {term} +")
                    else:
                        lines.append(f"        {term}")
                lines.append("    )")
        else:
            lines.append("    B_sq = torch.zeros_like(B[..., 0])")

        lines.extend([
            "",
            "    # θ² = -B² (bivectors square to negative in Euclidean)",
            "    theta_sq = -B_sq",
            "",
            "    # Handle both positive (elliptic) and negative (hyperbolic) cases",
            "    # For numerical stability, use small angle approximation when θ ≈ 0",
            "    eps = 1e-8",
            "",
            "    # Elliptic case: θ² > 0 => exp(B) = cos(θ) + sin(θ)/θ * B",
            "    # Hyperbolic case: θ² < 0 => exp(B) = cosh(θ) + sinh(θ)/θ * B",
            "    theta = torch.sqrt(torch.abs(theta_sq) + eps)",
            "",
            "    # Compute cos/cosh and sinc (sin(θ)/θ or sinh(θ)/θ)",
            "    is_elliptic = theta_sq > 0",
            "    cos_term = torch.where(is_elliptic, torch.cos(theta), torch.cosh(theta))",
            "    sinc_term = torch.where(",
            "        is_elliptic,",
            "        torch.sin(theta) / theta,",
            "        torch.sinh(theta) / theta",
            "    )",
            "",
            "    # Build rotor: scalar part + bivector parts",
        ])

        # Generate rotor output
        # Rotor index 0 is scalar (full index 0)
        # Find which rotor indices correspond to bivector indices
        rotor_output_parts = []
        for rotor_sparse_idx, rotor_full_idx in enumerate(self.rotor_indices):
            if rotor_full_idx == 0:
                # Scalar part
                rotor_output_parts.append((rotor_sparse_idx, "cos_term"))
            elif rotor_full_idx in self.bivector_indices:
                # Bivector part
                biv_sparse_idx = self.bivector_indices.index(rotor_full_idx)
                rotor_output_parts.append((rotor_sparse_idx, f"sinc_term * B[..., {biv_sparse_idx}]"))
            else:
                # Grade 4+ even parts - should be zero for exp(bivector)
                rotor_output_parts.append((rotor_sparse_idx, "torch.zeros_like(B[..., 0])"))

        for rotor_idx, expr in rotor_output_parts:
            lines.append(f"    r{rotor_idx} = {expr}")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.rotor_count, chunk_size):
            chunk = ", ".join(f"r{j}" for j in range(i, min(i + chunk_size, self.rotor_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_log_rotor(self) -> str:
        """Generate rotor logarithm function (rotor -> bivector)."""
        lines = [
            "# =============================================================================",
            "# Rotor Logarithm (log_rotor)",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def log_rotor(r: Tensor) -> Tensor:",
            '    """',
            f"    Compute logarithm of a rotor in Cl({self.p}, {self.q}).",
            "",
            "    Converts a rotor (rotation) to a bivector (generator).",
            "    Formula: log(R) = atan2(|B|, s) / |B| * B where R = s + B",
            "",
            "    Args:",
            f"        r: Input rotor, shape (..., {self.rotor_count})",
            "",
            "    Returns:",
            f"        Bivector, shape (..., {self.bivector_count})",
            '    """',
        ]

        # Extract scalar part (index 0 in rotor)
        lines.append("    s = r[..., 0]  # Scalar part")

        # Extract bivector parts and compute |B|²
        lines.append("")
        lines.append("    # Compute |B|² from bivector parts of rotor")
        bivector_norm_terms = []
        for rotor_sparse_idx, rotor_full_idx in enumerate(self.rotor_indices):
            if rotor_full_idx in self.bivector_indices:
                # This rotor component is a bivector
                # |B|² uses positive contributions (Euclidean norm in bivector space)
                bivector_norm_terms.append(f"r[..., {rotor_sparse_idx}] * r[..., {rotor_sparse_idx}]")

        if bivector_norm_terms:
            if len(bivector_norm_terms) <= 4:
                lines.append(f"    B_norm_sq = {' + '.join(bivector_norm_terms)}")
            else:
                lines.append("    B_norm_sq = (")
                for i, term in enumerate(bivector_norm_terms):
                    if i < len(bivector_norm_terms) - 1:
                        lines.append(f"        {term} +")
                    else:
                        lines.append(f"        {term}")
                lines.append("    )")
        else:
            lines.append("    B_norm_sq = torch.zeros_like(s)")

        lines.extend([
            "",
            "    # Compute angle θ = atan2(|B|, s)",
            "    eps = 1e-8",
            "    B_norm = torch.sqrt(B_norm_sq + eps)",
            "    theta = torch.atan2(B_norm, s)",
            "",
            "    # Scale factor: θ / |B|",
            "    scale = theta / (B_norm + eps)",
            "",
            "    # Extract bivector components from rotor",
        ])

        # Map rotor bivector components to output bivector
        for biv_sparse_idx, biv_full_idx in enumerate(self.bivector_indices):
            # Find corresponding rotor index
            if biv_full_idx in self._full_to_rotor:
                rotor_sparse_idx = self._full_to_rotor[biv_full_idx]
                lines.append(f"    b{biv_sparse_idx} = scale * r[..., {rotor_sparse_idx}]")
            else:
                lines.append(f"    b{biv_sparse_idx} = torch.zeros_like(s)")

        lines.append("")
        lines.append("    return torch.stack([")
        chunk_size = 8
        for i in range(0, self.bivector_count, chunk_size):
            chunk = ", ".join(f"b{j}" for j in range(i, min(i + chunk_size, self.bivector_count)))
            lines.append(f"        {chunk},")
        lines.append("    ], dim=-1)")
        lines.append("")

        return "\n".join(lines)

    def generate_slerp_rotor(self) -> str:
        """Generate rotor spherical linear interpolation function."""
        lines = [
            "# =============================================================================",
            "# Rotor SLERP (Spherical Linear Interpolation)",
            "# =============================================================================",
            "",
            self._jit_decorator(),
            f"def slerp_rotor(r1: Tensor, r2: Tensor, t: Tensor) -> Tensor:",
            '    """',
            f"    Spherical linear interpolation between rotors in Cl({self.p}, {self.q}).",
            "",
            "    Formula: slerp(r1, r2, t) = r1 * exp(t * log(r1~ * r2))",
            "",
            "    Args:",
            f"        r1: Start rotor, shape (..., {self.rotor_count})",
            f"        r2: End rotor, shape (..., {self.rotor_count})",
            "        t: Interpolation parameter [0, 1], shape (...) or scalar",
            "",
            "    Returns:",
            f"        Interpolated rotor, shape (..., {self.rotor_count})",
            '    """',
            "    # Compute r1~ (reverse of r1)",
            "    r1_rev = reverse_rotor(r1)",
            "",
            "    # Compute delta = r1~ * r2",
            "    delta = compose_rotor(r1_rev, r2)",
            "",
            "    # Compute log of delta to get bivector",
            "    log_delta = log_rotor(delta)",
            "",
            "    # Scale by t",
            "    if t.dim() == 0:",
            "        scaled_log = t * log_delta",
            "    else:",
            "        scaled_log = t.unsqueeze(-1) * log_delta",
            "",
            "    # Exponentiate back to rotor",
            "    delta_t = exp_bivector(scaled_log)",
            "",
            "    # Compose with r1",
            "    return compose_rotor(r1, delta_t)",
            "",
        ]

        return "\n".join(lines)


def generate_algebra_module(p: int, q: int = 0, output_path: Optional[str] = None) -> str:
    """
    Generate complete algebra module for Cl(p, q).

    Args:
        p: Positive signature dimension
        q: Negative signature dimension
        output_path: Optional path to write the generated code

    Returns:
        Generated code as string
    """
    gen = ClCodeGenerator(p, q)
    code = gen.generate_module()

    if output_path:
        with open(output_path, "w") as f:
            f.write(code)

    return code


if __name__ == "__main__":
    # Test generation for VGA(3)
    print("Generating VGA(3) = Cl(3, 0)...")
    code = generate_algebra_module(3, 0)
    print(f"Generated {len(code)} characters")
    print("\n--- First 100 lines ---")
    print("\n".join(code.split("\n")[:100]))
