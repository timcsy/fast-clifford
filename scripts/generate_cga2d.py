#!/usr/bin/env python
"""
CGA2D Code Generator Script

Generates the functional.py module with hard-coded CGA2D operations.

CGA2D Cl(3,1) specifications:
- 16 blades (2^4)
- 4 UPGC Point components (Grade 1)
- 7 Motor components (Grade 0 + 2, excluding G4 pseudoscalar)

Usage:
    uv run python scripts/generate_cga2d.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fast_clifford.codegen.generate import generate_cgand_functional


def main():
    # Output path
    output_path = os.path.join(
        project_root,
        "fast_clifford",
        "algebras",
        "cga2d",
        "functional.py"
    )

    print("=" * 60)
    print("CGA2D Code Generator")
    print("=" * 60)
    print(f"Output: {output_path}")
    print()

    # Generate the module for 2D
    generate_cgand_functional(2, output_path)

    print()
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
