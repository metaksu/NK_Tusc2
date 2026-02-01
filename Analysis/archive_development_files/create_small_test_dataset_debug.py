#!/usr/bin/env python3
"""
Create Small Test Dataset - Debug Version
========================================

Debug version with better error handling and verification.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
from pathlib import Path

def create_small_test_dataset():
    """Create small test dataset from Tang NK data with debugging."""
    
    print("=== CREATING SMALL TEST DATASET (DEBUG) ===\n")
    
    # Create output directory
    output_dir = Path("tang_test_small")
    output_dir.mkdir(exist_ok=True)
    print(f"Created directory: {output_dir.absolute()}")
    
    try:
        # Load Tang data
        print("1. Loading Tang NK data...")
        adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")
        print(f"   Shape: {adata.shape}")
        print(f"   Cell types: {adata.obs['celltype'].unique()[:5]}...")  # Show first 5
        
        # Sample cells
        print("\n2. Sampling cells...")
        cells_per_type = 10
        sampled_indices = []
        
        for i, celltype in enumerate(adata.obs['celltype'].unique()):
            if i < 5:  # Only show first 5 for debugging
                print(f"   Processing {celltype}...")
            
            celltype_mask = adata.obs['celltype'] == celltype
            celltype_indices = adata.obs[celltype_mask].index.tolist()
            
            n_sample = min(cells_per_type, len(celltype_indices))
            sampled_cells = np.random.choice(celltype_indices, size=n_sample, replace=False)
            sampled_indices.extend(sampled_cells)
        
        print(f"   Total cells sampled: {len(sampled_indices)}")
        
        # Subset data
        adata_small = adata[sampled_indices, :].copy()
        print(f"   Small dataset shape: {adata_small.shape}")
        
        # Convert to format
        print("\n3. Converting format...")
        
        # Handle sparse matrix
        if hasattr(adata_small.X, 'toarray'):
            X_dense = adata_small.X.toarray()
            print("   Converted from sparse to dense")
        else:
            X_dense = adata_small.X
            print("   Already dense")
        
        # Convert to integers
        X_int = np.round(X_dense).astype(int)
        print(f"   Converted to integers, range: {X_int.min()} - {X_int.max()}")
        
        # Create DataFrames
        print("\n4. Creating DataFrames...")
        
        # Counts DataFrame
        counts_df = pd.DataFrame(
            X_int,
            index=range(len(sampled_indices)),
            columns=[f"gene{i}" for i in range(adata_small.n_vars)]
        )
        print(f"   Counts DataFrame: {counts_df.shape}")
        
        # Celltypes DataFrame
        celltypes_df = pd.DataFrame({
            'Celltype': adata_small.obs['celltype'].values
        })
        print(f"   Celltypes DataFrame: {celltypes_df.shape}")
        
        # Save files
        print("\n5. Saving files...")
        
        counts_file = output_dir / "tang_counts.txt"
        celltypes_file = output_dir / "tang_celltypes.txt"
        
        counts_df.to_csv(counts_file, sep="\t", index=True)
        print(f"   Saved counts to: {counts_file}")
        print(f"   File exists: {counts_file.exists()}")
        print(f"   File size: {counts_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        celltypes_df.to_csv(celltypes_file, sep="\t", index=False)
        print(f"   Saved celltypes to: {celltypes_file}")
        print(f"   File exists: {celltypes_file.exists()}")
        print(f"   File size: {celltypes_file.stat().st_size / 1024:.1f} KB")
        
        # Verify files
        print("\n6. Verifying files...")
        
        # Test reading back
        try:
            test_counts = pd.read_csv(counts_file, sep="\t", index_col=0, nrows=5)
            print(f"   Counts file readable: {test_counts.shape}")
            
            test_celltypes = pd.read_csv(celltypes_file, sep="\t", nrows=5)
            print(f"   Celltypes file readable: {test_celltypes.shape}")
            
            print("   ✓ Files verified successfully")
            
        except Exception as e:
            print(f"   ✗ Error reading files: {e}")
            
        print(f"\n=== SUCCESS ===")
        print(f"Files created in: {output_dir.absolute()}")
        
        return output_dir
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    np.random.seed(42)
    create_small_test_dataset() 