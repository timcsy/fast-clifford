"""
Auto-generated Clifford Algebras

This module contains pre-generated hardcoded implementations for all Cl(p,q)
where p+q <= 9 (approximately 55 algebras).

Structure:
    cl_{p}_{q}/
    ├── __init__.py      # Module exports
    ├── functional.py    # Hardcoded operations
    ├── constants.py     # Index constants (ROTOR_MASK, etc.)
    └── layers.py        # PyTorch nn.Module wrappers
"""

import importlib
from typing import Optional, Any

# Cache for loaded algebra modules
_algebra_cache: dict[tuple[int, int], Any] = {}


def get_algebra_module(p: int, q: int) -> Optional[Any]:
    """
    Dynamically load a generated algebra module.

    Args:
        p: Positive dimension
        q: Negative dimension

    Returns:
        The algebra module, or None if not found
    """
    key = (p, q)
    if key in _algebra_cache:
        return _algebra_cache[key]

    module_name = f"fast_clifford.algebras.generated.cl_{p}_{q}"
    try:
        module = importlib.import_module(module_name)
        _algebra_cache[key] = module
        return module
    except ImportError:
        return None


def list_available_algebras() -> list[tuple[int, int]]:
    """
    List all available pre-generated algebras.

    Returns:
        List of (p, q) tuples for available algebras
    """
    import pkgutil
    import os

    algebras = []
    package_path = os.path.dirname(__file__)

    for _, name, is_pkg in pkgutil.iter_modules([package_path]):
        if is_pkg and name.startswith("cl_"):
            parts = name.split("_")
            if len(parts) == 3:
                try:
                    p, q = int(parts[1]), int(parts[2])
                    algebras.append((p, q))
                except ValueError:
                    pass

    return sorted(algebras)


__all__ = ["get_algebra_module", "list_available_algebras"]
