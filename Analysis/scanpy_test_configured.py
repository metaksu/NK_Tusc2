#!/usr/bin/env python3
"""
Test scanpy with proper configuration
"""

import sys
import os
print("=== Testing Scanpy with Proper Configuration ===")

try:
    # Import scanpy
    print("Importing scanpy...")
    import scanpy as sc
    print("✓ Scanpy imported successfully!")
    
    # Configure scanpy settings (mirroring working scripts)
    print("Configuring scanpy settings...")
    sc.settings.verbosity = 1  # Reduce verbosity
    sc.settings.autoshow = False  # Prevent automatic plot display
    print("✓ Scanpy settings configured")
    
    # Test basic functionality
    print(f"Scanpy version: {sc.__version__}")
    
    # Test data loading with the Tang file
    data_file = "data/processed/comb_CD56_CD16_NK.h5ad"
    print(f"\nTesting data loading: {data_file}")
    
    if os.path.exists(data_file):
        file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        
        print("Attempting to load header info...")
        # Try loading just basic info first
        adata = sc.read_h5ad(data_file)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {adata.shape}")
        print(f"  First 5 obs columns: {list(adata.obs.columns)[:5]}")
        
        # Check for specific columns
        key_cols = ["meta_tissue_in_paper", "celltype", "Majortype"]
        for col in key_cols:
            if col in adata.obs.columns:
                print(f"  ✓ Found column: {col}")
            else:
                print(f"  ✗ Missing column: {col}")
        
        print("Basic data validation complete!")
        
    else:
        print(f"✗ File not found: {data_file}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!") 