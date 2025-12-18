"""
Clifford algebra implementations

This module provides access to pre-generated Clifford algebras:

Generated algebras (algebras/generated/):
- cl_p_q/: Pre-generated Cl(p,q) algebras for p+q <= 9

Legacy (removed):
- cga0d-cga5d: Replaced by unified cl_p_q structure
"""

from .generated import get_algebra_module, list_available_algebras

__all__ = ["get_algebra_module", "list_available_algebras"]
