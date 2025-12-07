"""
Clifford algebra implementations

Each submodule contains a specific algebra type:
- cga1d: 1D Conformal Geometric Algebra Cl(2,1) - 8 blades, 3 Point, 4 Motor
- cga2d: 2D Conformal Geometric Algebra Cl(3,1) - 16 blades, 4 Point, 8 Motor
- cga3d: 3D Conformal Geometric Algebra Cl(4,1) - 32 blades, 5 Point, 16 Motor
- cga4d: 4D Conformal Geometric Algebra Cl(5,1) - 64 blades, 6 Point, 31 Motor
- cga5d: 5D Conformal Geometric Algebra Cl(6,1) - 128 blades, 7 Point, 64 Motor
"""

from . import cga1d, cga2d, cga3d, cga4d, cga5d

__all__ = ["cga1d", "cga2d", "cga3d", "cga4d", "cga5d"]
