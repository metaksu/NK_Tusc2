#!/usr/bin/env python3
"""
Examine Tang NK Data Structure
=============================

This script examines the Tang NK data structure to understand
how to convert it to SCADEN format.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os

print("=== EXAMINING TANG NK DATA STRUCTURE ===\n")

# 1. Examine Tang data
print("1. EXAMINING Tang NK h5ad file")
print("-" * 50)
try:
    adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")
    print(f"AnnData shape: {adata.shape}")
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")
    print()
    
    print("obs columns:")
    print(adata.obs.columns.tolist())
    print()
    
    print("var columns:")
    print(adata.var.columns.tolist())
    print()
    
    # Check for cell type information
    print("Cell type columns in obs:")
    celltype_cols = [col for col in adata.obs.columns if 'type' in col.lower() or 'cluster' in col.lower() or 'label' in col.lower()]
    print(celltype_cols)
    print()
    
    # Look at the most promising cell type column
    if celltype_cols:
        for col in celltype_cols:
            print(f"Column '{col}':")
            print(f"  Unique values: {adata.obs[col].unique()}")
            print(f"  Value counts: {adata.obs[col].value_counts()}")
            print()
    
    # Check X data
    print("X data information:")
    print(f"  X data type: {type(adata.X)}")
    print(f"  X shape: {adata.X.shape}")
    if hasattr(adata.X, 'toarray'):
        print(f"  X is sparse: True")
        X_sample = adata.X[:100, :100].toarray()
    else:
        print(f"  X is sparse: False")
        X_sample = adata.X[:100, :100]
    print(f"  X value range (sample): {X_sample.min()} - {X_sample.max()}")
    print(f"  X data type: {X_sample.dtype}")
    print()
    
    # Check gene names
    print("Gene information:")
    print(f"  Gene names (first 10): {adata.var.index[:10].tolist()}")
    print(f"  Gene names (last 10): {adata.var.index[-10:].tolist()}")
    print()
    
    # Check cell barcodes
    print("Cell information:")
    print(f"  Cell barcodes (first 5): {adata.obs.index[:5].tolist()}")
    print(f"  Cell barcodes (last 5): {adata.obs.index[-5:].tolist()}")
    print()
    
except Exception as e:
    print(f"Error reading Tang data: {e}")

print("=== SUMMARY ===")
print("Need to identify:")
print("1. Which column contains cell type labels")
print("2. How many cells per cell type")
print("3. Data format (sparse vs dense)")
print("4. Value range and data type")
print("5. Gene naming convention") 