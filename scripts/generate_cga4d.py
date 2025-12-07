#!/usr/bin/env python
"""
CGA4D Code Generator Script

Generates the functional.py module with hard-coded CGA4D operations.

CGA4D Cl(5,1) specifications:
- 64 blades (2^6)
- 6 UPGC Point components (Grade 1)
- 31 Motor components (Grade 0 + 2 + 4)

Usage:
    uv run python scripts/generate_cga4d.py
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
        "cga4d",
        "functional.py"
    )

    print("=" * 60)
    print("CGA4D Code Generator")
    print("=" * 60)
    print(f"Output: {output_path}")
    print()

    # Generate the module for 4D
    generate_cgand_functional(4, output_path)

    print()
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
