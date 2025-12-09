"""
Clifford algebra implementations

Each submodule contains a specific algebra type:
- cga0d: 0D Conformal Geometric Algebra Cl(1,1) - 4 blades, 2 Point, 2 EvenVersor
- cga1d: 1D Conformal Geometric Algebra Cl(2,1) - 8 blades, 3 Point, 4 EvenVersor
- cga2d: 2D Conformal Geometric Algebra Cl(3,1) - 16 blades, 4 Point, 8 EvenVersor
- cga3d: 3D Conformal Geometric Algebra Cl(4,1) - 32 blades, 5 Point, 16 EvenVersor
- cga4d: 4D Conformal Geometric Algebra Cl(5,1) - 64 blades, 6 Point, 31 EvenVersor
- cga5d: 5D Conformal Geometric Algebra Cl(6,1) - 128 blades, 7 Point, 64 EvenVersor
"""

from . import cga0d, cga1d, cga2d, cga3d, cga4d, cga5d

__all__ = ["cga0d", "cga1d", "cga2d", "cga3d", "cga4d", "cga5d"]
