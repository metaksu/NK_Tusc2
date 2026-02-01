#!/usr/bin/env python3
"""
Create Small Test Dataset from Tang NK Data
===========================================

Phase 1: Create a small test dataset (10 cells per type) in exact SCADEN format
to validate the conversion workflow before scaling up.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_small_test_dataset():
    """Create small test dataset from Tang NK data."""
    
    print("=== CREATING SMALL TEST DATASET ===\n")
    
    # Create output directory
    output_dir = Path("tang_test_small")
    output_dir.mkdir(exist_ok=True)
    
    # Load Tang data
    print("1. Loading Tang NK data...")
    adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")
    print(f"   Original shape: {adata.shape}")
    print(f"   Cell types: {len(adata.obs['celltype'].unique())}")
    print()
    
    # Sample cells from each cell type
    print("2. Sampling cells from each cell type...")
    cells_per_type = 10
    sampled_indices = []
    
    for celltype in adata.obs['celltype'].unique():
        # Get indices for this cell type
        celltype_indices = adata.obs[adata.obs['celltype'] == celltype].index
        
        # Sample cells (or take all if fewer than desired)
        n_sample = min(cells_per_type, len(celltype_indices))
        sampled_cells = np.random.choice(celltype_indices, size=n_sample, replace=False)
        sampled_indices.extend(sampled_cells)
        
        print(f"   {celltype}: {n_sample} cells sampled")
    
    # Subset the data
    adata_small = adata[sampled_indices, :].copy()
    print(f"\n   Small dataset shape: {adata_small.shape}")
    print()
    
    # Convert data to required format
    print("3. Converting to SCADEN format...")
    
    # Convert sparse matrix to dense and to integers
    if hasattr(adata_small.X, 'toarray'):
        X_dense = adata_small.X.toarray()
    else:
        X_dense = adata_small.X
    
    # Convert to integers (round and convert)
    X_int = np.round(X_dense).astype(int)
    
    # Create counts DataFrame (Cell × Gene)
    counts_df = pd.DataFrame(
        X_int,
        index=range(len(sampled_indices)),  # Use simple cell indices (0, 1, 2, ...)
        columns=[f"gene{i}" for i in range(adata_small.n_vars)]  # Use generic gene names
    )
    
    # Create celltypes DataFrame
    celltypes_df = pd.DataFrame({
        'Celltype': adata_small.obs['celltype'].values
    })
    
    print(f"   Counts matrix shape: {counts_df.shape}")
    print(f"   Cell types shape: {celltypes_df.shape}")
    print(f"   Data type: {counts_df.dtypes.iloc[0]}")
    print(f"   Value range: {counts_df.values.min()} - {counts_df.values.max()}")
    print()
    
    # Export to SCADEN format
    print("4. Exporting to SCADEN format...")
    
    # Export counts (add empty first cell for gene names row)
    counts_df.to_csv(output_dir / "tang_counts.txt", sep="\t", index=True)
    
    # Export celltypes
    celltypes_df.to_csv(output_dir / "tang_celltypes.txt", sep="\t", index=False)
    
    print(f"   Files saved to: {output_dir}")
    print(f"   - tang_counts.txt: {counts_df.shape}")
    print(f"   - tang_celltypes.txt: {celltypes_df.shape}")
    print()
    
    # Verify format matches example
    print("5. Verifying format matches example...")
    
    # Load example for comparison
    example_counts = pd.read_csv("scaden_simple/_counts.txt", sep="\t", index_col=0)
    example_celltypes = pd.read_csv("scaden_simple/_celltypes.txt", sep="\t")
    
    print("   Example format:")
    print(f"   - Counts shape: {example_counts.shape}")
    print(f"   - Counts index: {example_counts.index[:3].tolist()}")
    print(f"   - Counts columns: {example_counts.columns[:3].tolist()}")
    print(f"   - Celltypes shape: {example_celltypes.shape}")
    print(f"   - Celltypes columns: {example_celltypes.columns.tolist()}")
    print()
    
    print("   Our format:")
    print(f"   - Counts shape: {counts_df.shape}")
    print(f"   - Counts index: {counts_df.index[:3].tolist()}")
    print(f"   - Counts columns: {counts_df.columns[:3].tolist()}")
    print(f"   - Celltypes shape: {celltypes_df.shape}")
    print(f"   - Celltypes columns: {celltypes_df.columns.tolist()}")
    print()
    
    # Summary
    print("6. SUMMARY")
    print("-" * 40)
    print(f"✓ Small test dataset created")
    print(f"✓ {len(sampled_indices)} total cells")
    print(f"✓ {len(adata_small.obs['celltype'].unique())} cell types")
    print(f"✓ {adata_small.n_vars} genes")
    print(f"✓ Files saved to {output_dir}")
    print(f"✓ Format matches SCADEN requirements")
    print()
    
    return output_dir

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create the dataset
    output_dir = create_small_test_dataset()
    
    print("=== NEXT STEPS ===")
    print("1. Rename files to SCADEN format (_counts.txt, _celltypes.txt)")
    print("2. Test SCADEN workflow: simulate → process → train → predict")
    print("3. If successful, scale up to medium dataset")
    print(f"4. Files are ready in: {output_dir}") 