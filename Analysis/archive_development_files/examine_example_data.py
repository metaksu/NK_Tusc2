#!/usr/bin/env python3
"""
Examine SCADEN Example Data Structure
====================================

This script examines the structure of the working SCADEN example data
to understand the exact format requirements.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os

print("=== EXAMINING SCADEN EXAMPLE DATA STRUCTURE ===\n")

# Change to scaden_simple directory
os.chdir("scaden_simple")

# 1. Examine _counts.txt
print("1. EXAMINING _counts.txt")
print("-" * 40)
counts_df = pd.read_csv("_counts.txt", sep="\t", index_col=0)
print(f"Shape: {counts_df.shape}")
print(f"Index (first 5): {counts_df.index[:5].tolist()}")
print(f"Columns (first 5): {counts_df.columns[:5].tolist()}")
print(f"Data type: {counts_df.dtypes.iloc[0]}")
print(f"Value range: {counts_df.values.min()} - {counts_df.values.max()}")
print(f"Sample values (first cell, first 5 genes): {counts_df.iloc[0, :5].tolist()}")
print()

# 2. Examine _celltypes.txt
print("2. EXAMINING _celltypes.txt")
print("-" * 40)
celltypes_df = pd.read_csv("_celltypes.txt", sep="\t")
print(f"Shape: {celltypes_df.shape}")
print(f"Columns: {celltypes_df.columns.tolist()}")
print(f"Unique cell types: {celltypes_df['Celltype'].unique()}")
print(f"Cell type counts: {celltypes_df['Celltype'].value_counts()}")
print(f"First 10 cell types: {celltypes_df['Celltype'].head(10).tolist()}")
print()

# 3. Examine example_bulk_data.txt
print("3. EXAMINING example_bulk_data.txt")
print("-" * 40)
bulk_df = pd.read_csv("example_bulk_data.txt", sep="\t", index_col=0)
print(f"Shape: {bulk_df.shape}")
print(f"Index (first 5): {bulk_df.index[:5].tolist()}")
print(f"Columns: {bulk_df.columns.tolist()}")
print(f"Data type: {bulk_df.dtypes.iloc[0]}")
print(f"Value range: {bulk_df.values.min()} - {bulk_df.values.max()}")
print()

# 4. Examine data.h5ad (created by scaden simulate)
print("4. EXAMINING data.h5ad")
print("-" * 40)
try:
    adata = sc.read_h5ad("data.h5ad")
    print(f"AnnData shape: {adata.shape}")
    print(f"obs columns: {adata.obs.columns.tolist()}")
    print(f"var columns: {adata.var.columns.tolist()}")
    if 'Celltype' in adata.obs.columns:
        print(f"Unique cell types in obs: {adata.obs['Celltype'].unique()}")
        print(f"Cell type counts in obs: {adata.obs['Celltype'].value_counts()}")
    print(f"X data type: {type(adata.X)}")
    print(f"X shape: {adata.X.shape}")
    print(f"X value range: {adata.X.min()} - {adata.X.max()}")
    print()
except Exception as e:
    print(f"Error reading data.h5ad: {e}")

# 5. Examine processed.h5ad (created by scaden process)
print("5. EXAMINING processed.h5ad")
print("-" * 40)
try:
    adata_processed = sc.read_h5ad("processed.h5ad")
    print(f"AnnData shape: {adata_processed.shape}")
    print(f"obs columns: {adata_processed.obs.columns.tolist()}")
    print(f"var columns: {adata_processed.var.columns.tolist()}")
    if 'Celltype' in adata_processed.obs.columns:
        print(f"Unique cell types in obs: {adata_processed.obs['Celltype'].unique()}")
    print(f"X data type: {type(adata_processed.X)}")
    print(f"X shape: {adata_processed.X.shape}")
    print(f"X value range: {adata_processed.X.min()} - {adata_processed.X.max()}")
    print()
except Exception as e:
    print(f"Error reading processed.h5ad: {e}")

# 6. Examine predictions.txt
print("6. EXAMINING predictions.txt")
print("-" * 40)
pred_df = pd.read_csv("predictions.txt", sep="\t", index_col=0)
print(f"Shape: {pred_df.shape}")
print(f"Index: {pred_df.index.tolist()}")
print(f"Columns: {pred_df.columns.tolist()}")
print(f"Sample predictions (first sample): {pred_df.iloc[0].tolist()}")
print(f"Sum of first sample proportions: {pred_df.iloc[0].sum()}")
print()

print("=== SUMMARY ===")
print("✓ _counts.txt: Cell x Gene matrix with integer counts")
print("✓ _celltypes.txt: Single column 'Celltype' with cell type labels")
print("✓ example_bulk_data.txt: Gene x Sample matrix for prediction")
print("✓ data.h5ad: AnnData object with simulated bulk samples")
print("✓ processed.h5ad: Processed training data")
print("✓ predictions.txt: Sample x CellType proportions") 