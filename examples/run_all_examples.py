"""
Run all example scripts in sequence.

This script executes all examples and provides a comprehensive demonstration
of the SNN solver capabilities.

Usage:
    python run_all_examples.py
    
    Or from project root:
    python examples/run_all_examples.py
"""

import subprocess
import sys
from pathlib import Path

# Get the examples directory
examples_dir = Path(__file__).parent

examples = [
    ("example1_simple_2d.py", "Simple 2D Quadratic Program"),
    ("example2_3d_polytope.py", "3D QP with Polytope Constraints"),
    ("example3_linear_program.py", "Linear Program with Box Constraints"),
    ("example4_warm_start.py", "Warm Starting (Receding Horizon)"),
    ("example5_infeasible_recovery.py", "Infeasible Initialization Recovery"),
    ("example6_equality_constraint.py", "Equality Constraint Handling"),
    ("example7_svm_dual.py", "SVM Dual (Auto k0 + Box Clipping)"),
]

print("=" * 80)
print("Running All SNN Solver Examples")
print("=" * 80)
print()

for i, (script, description) in enumerate(examples, 1):
    print(f"\n{'=' * 80}")
    print(f"Example {i}/{len(examples)}: {description}")
    print(f"Script: {script}")
    print("=" * 80)
    print()
    
    script_path = examples_dir / script
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print()
        print(f"✓ {script} completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ {script} failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"✗ {script} not found")
        sys.exit(1)
    
    print()
    input("Press Enter to continue to next example...")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print("=" * 80)

