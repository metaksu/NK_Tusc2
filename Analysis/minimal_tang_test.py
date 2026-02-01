#!/usr/bin/env python3
"""
Minimal test for Tang data access
"""

import sys
import os

print("=== Basic Environment Test ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Check data file
data_file = "data/processed/comb_CD56_CD16_NK.h5ad"
print(f"\nTesting file access: {data_file}")

if os.path.exists(data_file):
    file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"✓ File exists! Size: {file_size_mb:.1f} MB")
else:
    print(f"✗ File NOT found: {data_file}")
    print("Available files in data/processed/:")
    if os.path.exists("data/processed"):
        for f in os.listdir("data/processed"):
            print(f"  - {f}")
    sys.exit(1)

# Test imports one by one
print("\n=== Testing Imports ===")

modules_to_test = [
    ("os", "os"),
    ("pathlib", "from pathlib import Path"),
    ("numpy", "import numpy as np"),
    ("pandas", "import pandas as pd"),
    ("scipy", "from scipy import sparse"),
    ("scanpy", "import scanpy as sc")
]

for name, import_statement in modules_to_test:
    try:
        exec(import_statement)
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        
print("\n=== Testing scanpy data loading ===")
try:
    import scanpy as sc
    print("Attempting to load just the header info...")
    # Try to load without reading the full data
    adata = sc.read_h5ad(data_file, first_column_names=True)  
    print(f"✓ Basic loading successful!")
except Exception as e:
    print(f"✗ Scanpy loading failed: {e}")

print("\n=== Test Complete ===") 