#!/usr/bin/env python
"""
CGA5D Code Generator Script

Generates the functional.py module with hard-coded CGA5D operations.

CGA5D Cl(6,1) specifications:
- 128 blades (2^7)
- 7 UPGC Point components (Grade 1)
- 64 Motor components (Grade 0 + 2 + 4 + 6)

Usage:
    uv run python scripts/generate_cga5d.py
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
        "cga5d",
        "functional.py"
    )

    print("=" * 60)
    print("CGA5D Code Generator")
    print("=" * 60)
    print(f"Output: {output_path}")
    print()

    # Generate the module for 5D
    generate_cgand_functional(5, output_path)

    print()
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
