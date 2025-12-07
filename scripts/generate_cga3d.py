#!/usr/bin/env python
"""
CGA3D Code Generator Script

Generates the functional.py module with hard-coded CGA operations.

Usage:
    uv run python scripts/generate_cga3d.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fast_clifford.codegen.generate import generate_cga3d_functional


def main():
    # Output path
    output_path = os.path.join(
        project_root,
        "fast_clifford",
        "algebras",
        "cga3d",
        "functional.py"
    )

    print("=" * 60)
    print("CGA3D Code Generator")
    print("=" * 60)
    print(f"Output: {output_path}")
    print()

    # Generate the module
    generate_cga3d_functional(output_path)

    print()
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
