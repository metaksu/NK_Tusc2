#!/usr/bin/env python3
"""
Plan Tang to SCADEN Conversion Strategy
======================================

This script analyzes both datasets and creates a detailed plan
for converting Tang NK data to SCADEN format.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
from collections import Counter

print("=== TANG TO SCADEN CONVERSION PLANNING ===\n")

# Load Tang data
print("1. LOADING AND ANALYZING TANG DATA")
print("-" * 50)
adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")

print(f"Tang data shape: {adata.shape}")
print(f"Cell types available: {adata.obs['celltype'].unique()}")
print(f"Cell type counts:")
for celltype, count in adata.obs['celltype'].value_counts().items():
    print(f"  {celltype}: {count:,} cells")
print()

# Load example data for comparison
print("2. LOADING EXAMPLE DATA FOR COMPARISON")
print("-" * 50)
example_counts = pd.read_csv("scaden_simple/_counts.txt", sep="\t", index_col=0)
example_celltypes = pd.read_csv("scaden_simple/_celltypes.txt", sep="\t")

print(f"Example data shape: {example_counts.shape}")
print(f"Example cell types: {example_celltypes['Celltype'].unique()}")
print(f"Example cell type counts:")
for celltype, count in example_celltypes['Celltype'].value_counts().items():
    print(f"  {celltype}: {count} cells")
print()

# Plan conversion strategy
print("3. CONVERSION STRATEGY PLANNING")
print("-" * 50)

# Strategy 1: Small test dataset
print("STRATEGY 1: Small Test Dataset (for initial testing)")
print("Goal: Create a dataset similar in size to the example (10-50 cells)")
test_cells_per_type = 10
test_total_cells = test_cells_per_type * len(adata.obs['celltype'].unique())
print(f"  - {test_cells_per_type} cells per cell type")
print(f"  - Total cells: {test_total_cells}")
print(f"  - Cell types: {len(adata.obs['celltype'].unique())}")
print()

# Strategy 2: Medium dataset for validation
print("STRATEGY 2: Medium Dataset (for validation)")
print("Goal: Balanced dataset with reasonable size")
medium_cells_per_type = 500
medium_total_cells = medium_cells_per_type * len(adata.obs['celltype'].unique())
print(f"  - {medium_cells_per_type} cells per cell type")
print(f"  - Total cells: {medium_total_cells}")
print(f"  - 80% train: {int(medium_total_cells * 0.8)}")
print(f"  - 20% test: {int(medium_total_cells * 0.2)}")
print()

# Strategy 3: Full dataset (proportional)
print("STRATEGY 3: Full Dataset (maintaining proportions)")
print("Goal: Use all data with 80/20 split maintaining proportions")
full_total_cells = adata.shape[0]
print(f"  - Total cells: {full_total_cells:,}")
print(f"  - 80% train: {int(full_total_cells * 0.8):,}")
print(f"  - 20% test: {int(full_total_cells * 0.2):,}")
print("  - Proportions maintained for each cell type")
print()

# Data format analysis
print("4. DATA FORMAT REQUIREMENTS")
print("-" * 50)
print("Tang data current format:")
print(f"  - X data type: {type(adata.X)}")
print(f"  - X is sparse: {hasattr(adata.X, 'toarray')}")
print(f"  - X dtype: {adata.X.dtype}")
print(f"  - Gene names: Real gene symbols (e.g., {adata.var.index[:3].tolist()})")
print()

print("Required SCADEN format:")
print("  - _counts.txt: Cell × Gene matrix with integer counts")
print("  - _celltypes.txt: Single column 'Celltype' with labels")
print("  - Gene names: Can be real names or generic (gene0, gene1, etc.)")
print("  - Data type: Integer counts")
print()

# Conversion steps
print("5. CONVERSION STEPS")
print("-" * 50)
print("Step 1: Data preparation")
print("  - Convert sparse matrix to dense")
print("  - Convert float counts to integers")
print("  - Create stratified train/test split")
print()

print("Step 2: Export to SCADEN format")
print("  - Export counts matrix as _counts.txt")
print("  - Export cell types as _celltypes.txt")
print("  - Use tab-separated format")
print()

print("Step 3: SCADEN workflow")
print("  - scaden simulate (create artificial bulk samples)")
print("  - scaden process (preprocess data)")
print("  - scaden train (train model)")
print("  - scaden predict (test predictions)")
print()

# File size estimates
print("6. FILE SIZE ESTIMATES")
print("-" * 50)
# Estimate based on example data
example_file_size_mb = 2.7 / 1024  # 2.7KB to MB
cells_ratio = test_total_cells / 10
genes_ratio = adata.shape[1] / 100
estimated_test_size_mb = example_file_size_mb * cells_ratio * genes_ratio
print(f"Test dataset estimated size: {estimated_test_size_mb:.1f} MB")

medium_cells_ratio = medium_total_cells / 10
estimated_medium_size_mb = example_file_size_mb * medium_cells_ratio * genes_ratio
print(f"Medium dataset estimated size: {estimated_medium_size_mb:.1f} MB")

full_cells_ratio = full_total_cells / 10
estimated_full_size_mb = example_file_size_mb * full_cells_ratio * genes_ratio
print(f"Full dataset estimated size: {estimated_full_size_mb:.1f} MB")
print()

print("7. RECOMMENDED APPROACH")
print("-" * 50)
print("Phase 1: Test with small dataset")
print("  - Create test dataset with 10 cells per type")
print("  - Verify SCADEN workflow works")
print("  - Debug any format issues")
print()

print("Phase 2: Validate with medium dataset")
print("  - Create medium dataset with 500 cells per type")
print("  - Test complete workflow including training")
print("  - Optimize parameters")
print()

print("Phase 3: Scale to full dataset")
print("  - Use full dataset with 80/20 split")
print("  - Train final model")
print("  - Apply to TCGA data")
print()

print("=== NEXT STEPS ===")
print("1. Create small test dataset script")
print("2. Test SCADEN workflow on small dataset")
print("3. If successful, scale up to medium dataset")
print("4. Finally, scale to full dataset") 