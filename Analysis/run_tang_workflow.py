#!/usr/bin/env python3
"""
Simple wrapper to run Tang reference matrix generation
"""

import sys
import os
sys.path.append('scripts/04_data_preparation')

try:
    from create_tang_reference_matrix import create_tang_reference_matrix
    print("Running Tang reference matrix generation...")
    result = create_tang_reference_matrix()
    print("SUCCESS!")
    print(f"Matrix file: {result['matrix_file']}")
    print(f"Summary file: {result['summary_file']}")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Execution error: {e}") 