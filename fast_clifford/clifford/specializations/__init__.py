"""
Clifford Algebra Specializations

- VGAWrapper: VGA(n) = Cl(n, 0) specialization
- CGAWrapper: CGA(n) = Cl(n+1, 1) specialization with encode/decode
- PGAEmbedding: PGA(n) = Cl(n, 0, 1) via CGA embedding
"""

from .vga import VGAWrapper
from .cga import CGAWrapper
from .pga import PGAEmbedding

__all__ = ["VGAWrapper", "CGAWrapper", "PGAEmbedding"]
