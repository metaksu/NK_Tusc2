#!/usr/bin/env python3
"""
Inspect Rebuffet H5AD File
=========================

This script inspects the current Rebuffet h5ad file to understand:
1. Data structure and dimensions
2. Available metadata columns
3. Batch information and structure
4. Expression data characteristics
5. What's needed for batch correction pipeline
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=== INSPECTING REBUFFET H5AD FILE ===\n")

# Load the current h5ad file
h5ad_path = r"C:\Users\met-a\Documents\Analysis\data\processed\PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"

try:
    print(f"Loading Rebuffet data from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Successfully loaded data: {adata.shape}")
    print()
    
    # === 1. BASIC DATA STRUCTURE ===
    print("=== 1. BASIC DATA STRUCTURE ===")
    print(f"Shape: {adata.shape} (cells x genes)")
    print(f"Number of cells: {adata.n_obs:,}")
    print(f"Number of genes: {adata.n_vars:,}")
    print()
    
    # Expression data characteristics
    print("Expression data characteristics:")
    print(f"  X data type: {type(adata.X)}")
    print(f"  X dtype: {adata.X.dtype}")
    if hasattr(adata.X, 'toarray'):
        X_sample = adata.X[:100, :100].toarray()
        print(f"  X is sparse: True")
    else:
        X_sample = adata.X[:100, :100]
        print(f"  X is sparse: False")
    
    print(f"  X value range: {adata.X.min():.3f} to {adata.X.max():.3f}")
    print(f"  X mean: {adata.X.mean():.3f}")
    print(f"  X std: {np.std(X_sample):.3f}")
    print()
    
    # Check for layers
    print("Available layers:")
    if adata.layers:
        for layer_name in adata.layers.keys():
            layer_data = adata.layers[layer_name]
            print(f"  {layer_name}: {type(layer_data)}, range: {layer_data.min():.3f} to {layer_data.max():.3f}")
    else:
        print("  No layers found")
    print()
    
    # Check for raw data
    print("Raw data:")
    if hasattr(adata, 'raw') and adata.raw is not None:
        print(f"  Raw available: {adata.raw.shape}")
        print(f"  Raw X range: {adata.raw.X.min():.3f} to {adata.raw.X.max():.3f}")
    else:
        print("  Raw data: Not available")
    print()
    
    # === 2. METADATA COLUMNS ===
    print("=== 2. METADATA COLUMNS ===")
    print(f"Available .obs columns ({len(adata.obs.columns)}):")
    for i, col in enumerate(adata.obs.columns):
        unique_vals = adata.obs[col].nunique()
        data_type = adata.obs[col].dtype
        print(f"  {i+1:2d}. {col:<25} ({data_type}) - {unique_vals} unique values")
    print()
    
    print(f"Available .var columns ({len(adata.var.columns)}):")
    for i, col in enumerate(adata.var.columns):
        print(f"  {i+1:2d}. {col}")
    print()
    
    # === 3. BATCH INFORMATION ANALYSIS ===
    print("=== 3. BATCH INFORMATION ANALYSIS ===")
    
    # Look for potential batch columns
    batch_columns = []
    potential_batch_keywords = ['batch', 'donor', 'sample', 'dataset', 'chemistry', 'platform', 'experiment']
    
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in potential_batch_keywords):
            batch_columns.append(col)
    
    print(f"Potential batch columns found: {batch_columns}")
    print()
    
    # Analyze each potential batch column
    for col in batch_columns:
        print(f"--- Analysis of '{col}' ---")
        value_counts = adata.obs[col].value_counts()
        print(f"  Unique values: {len(value_counts)}")
        print(f"  Value distribution:")
        for val, count in value_counts.head(10).items():
            percentage = (count / len(adata.obs)) * 100
            print(f"    {val}: {count:,} cells ({percentage:.1f}%)")
        if len(value_counts) > 10:
            print(f"    ... and {len(value_counts) - 10} more values")
        print()
    
    # === 4. SUBTYPE INFORMATION ===
    print("=== 4. SUBTYPE INFORMATION ===")
    
    subtype_columns = []
    potential_subtype_keywords = ['ident', 'cluster', 'celltype', 'subtype', 'label']
    
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in potential_subtype_keywords):
            subtype_columns.append(col)
    
    print(f"Potential subtype columns: {subtype_columns}")
    print()
    
    for col in subtype_columns:
        print(f"--- Analysis of '{col}' ---")
        value_counts = adata.obs[col].value_counts()
        print(f"  Unique subtypes: {len(value_counts)}")
        print(f"  Subtype distribution:")
        for val, count in value_counts.items():
            percentage = (count / len(adata.obs)) * 100
            print(f"    {val}: {count:,} cells ({percentage:.1f}%)")
        print()
    
    # === 5. CROSS-TABULATION ANALYSIS ===
    print("=== 5. CROSS-TABULATION ANALYSIS ===")
    
    if len(batch_columns) > 0 and len(subtype_columns) > 0:
        # Pick the most likely batch and subtype columns
        primary_batch_col = batch_columns[0]  # Usually 'donor' or 'batch'
        primary_subtype_col = subtype_columns[0]  # Usually 'ident'
        
        print(f"Cross-tabulation: {primary_subtype_col} vs {primary_batch_col}")
        crosstab = pd.crosstab(adata.obs[primary_subtype_col], adata.obs[primary_batch_col])
        print(crosstab)
        print()
        
        # Check for batch imbalances
        print("Batch imbalance analysis:")
        batch_counts = adata.obs[primary_batch_col].value_counts()
        min_cells = batch_counts.min()
        max_cells = batch_counts.max()
        ratio = max_cells / min_cells
        print(f"  Smallest batch: {min_cells:,} cells")
        print(f"  Largest batch: {max_cells:,} cells")
        print(f"  Imbalance ratio: {ratio:.2f}x")
        
        if ratio > 3:
            print("  ⚠️  SIGNIFICANT batch imbalance detected - batch correction recommended")
        else:
            print("  ✓ Batch sizes are relatively balanced")
        print()
    
    # === 6. GENE INFORMATION ===
    print("=== 6. GENE INFORMATION ===")
    print(f"Gene names (first 10): {adata.var_names[:10].tolist()}")
    print(f"Gene names (last 10): {adata.var_names[-10:].tolist()}")
    print()
    
    # Check for specific genes of interest
    genes_of_interest = ['TUSC2', 'FCGR3A', 'NCAM1', 'KLRC1', 'GZMB', 'PRF1', 'IFNG', 'TNF']
    found_genes = []
    missing_genes = []
    
    for gene in genes_of_interest:
        if gene in adata.var_names:
            found_genes.append(gene)
        else:
            missing_genes.append(gene)
    
    print(f"Genes of interest found: {found_genes}")
    if missing_genes:
        print(f"Genes of interest missing: {missing_genes}")
    print()
    
    # === 7. RECOMMENDATIONS FOR BATCH CORRECTION ===
    print("=== 7. RECOMMENDATIONS FOR BATCH CORRECTION ===")
    
    if len(batch_columns) == 0:
        print("❌ No clear batch columns identified")
        print("   Recommendation: Check original Seurat object for batch information")
    else:
        print("✅ Batch information available")
        print(f"   Primary batch column: {batch_columns[0]}")
        
        # Determine batch correction strategy
        if len(batch_counts) <= 2:
            print("   Recommendation: Simple batch correction (2 batches or fewer)")
        elif len(batch_counts) <= 10:
            print("   Recommendation: Standard batch correction (Harmony/Scanpy)")
        else:
            print("   Recommendation: Advanced batch correction (many batches)")
        
        # Check if correction is needed
        if ratio > 2:
            print("   ⚠️  Batch correction STRONGLY recommended due to imbalance")
        else:
            print("   ℹ️  Batch correction optional but recommended for robustness")
    
    print()
    print("=== SUMMARY ===")
    print(f"Dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Expression type: {'TPM-like' if adata.X.max() > 20 else 'Log-normalized'}")
    print(f"Batch columns: {len(batch_columns)} found")
    print(f"Subtype columns: {len(subtype_columns)} found")
    if len(batch_columns) > 0:
        print(f"Recommended batch column for correction: {batch_columns[0]}")
    
except Exception as e:
    print(f"Error loading file: {e}")
    print("Please check if the file path is correct and the file exists.") 