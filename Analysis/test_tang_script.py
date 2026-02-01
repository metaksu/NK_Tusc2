#!/usr/bin/env python3
"""
Simple test script to debug Tang reference matrix generation
"""

import sys
print("Starting test script...")

try:
    print("Testing imports...")
    import scanpy as sc
    print("✓ scanpy imported")
    
    import pandas as pd
    print("✓ pandas imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import os
    from pathlib import Path
    print("✓ os and pathlib imported")
    
    # Test data file existence
    data_file = "data/processed/comb_CD56_CD16_NK.h5ad"
    if os.path.exists(data_file):
        print(f"✓ Data file exists: {data_file}")
        
        # Get file size
        file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
    else:
        print(f"✗ Data file missing: {data_file}")
        sys.exit(1)
    
    # Try to load just the basic info without full loading
    print("Testing data loading...")
    try:
        adata = sc.read_h5ad(data_file)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {adata.shape}")
        print(f"  Obs columns: {list(adata.obs.columns)[:5]}...")
        
        # Check for key columns
        key_cols = ["meta_tissue_in_paper", "celltype", "Majortype"]
        for col in key_cols:
            if col in adata.obs.columns:
                print(f"  ✓ Found column: {col}")
            else:
                print(f"  ✗ Missing column: {col}")
        
        print("Basic test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Error during imports or setup: {e}")
    sys.exit(1)
    
print("All tests passed!") 