#!/usr/bin/env python3
"""
Create Compatible Tang Dataset for TCGA Analysis
==============================================

This script creates a Tang training dataset using only genes that exist
in both Tang and TCGA datasets, ensuring perfect compatibility for SCADEN.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_compatible_tang_dataset():
    """Create Tang dataset compatible with TCGA data."""
    
    print("=== CREATING COMPATIBLE TANG DATASET ===\n")
    
    # Create output directory
    output_dir = Path("tang_compatible_train")
    output_dir.mkdir(exist_ok=True)
    
    # Load Tang data
    logger.info("Loading Tang NK data...")
    adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")
    logger.info(f"Original Tang shape: {adata.shape}")
    
    # Load TCGA data to get available genes
    logger.info("Loading TCGA data to find compatible genes...")
    tcga_data = pd.read_csv(r"C:\Users\met-a\Documents\Analysis\TCGAdata\Analysis_Python_Output_v4_SCADEN\GBM\GBM_bulk_for_scaden.txt", 
                           sep='\t', index_col=0)
    tcga_genes = set(tcga_data.index)
    logger.info(f"TCGA genes: {len(tcga_genes)}")
    
    # Find overlapping genes
    tang_genes = adata.var_names.tolist()
    overlap_genes = [g for g in tang_genes if g in tcga_genes]
    logger.info(f"Overlapping genes: {len(overlap_genes)} out of {len(tang_genes)} Tang genes")
    
    # Filter Tang data to only overlapping genes
    adata_filtered = adata[:, overlap_genes].copy()
    logger.info(f"Filtered Tang shape: {adata_filtered.shape}")
    
    # Create 80/20 split
    logger.info("Creating 80/20 stratified split...")
    train_indices = []
    test_indices = []
    
    for celltype in adata_filtered.obs['celltype'].unique():
        celltype_mask = adata_filtered.obs['celltype'] == celltype
        celltype_cells = adata_filtered.obs[celltype_mask].index.tolist()
        
        train_cells, test_cells = train_test_split(
            celltype_cells, test_size=0.2, random_state=42
        )
        train_indices.extend(train_cells)
        test_indices.extend(test_cells)
        
        logger.info(f"  {celltype}: {len(train_cells)} train, {len(test_cells)} test")
    
    # Create training dataset
    adata_train = adata_filtered[train_indices, :].copy()
    logger.info(f"Training set: {adata_train.shape}")
    
    # Convert to SCADEN format
    logger.info("Converting to SCADEN format...")
    
    # Convert sparse to dense and integers
    if hasattr(adata_train.X, 'toarray'):
        X_dense = adata_train.X.toarray()
    else:
        X_dense = adata_train.X
    X_int = np.round(X_dense).astype(int)
    
    # Create counts DataFrame with overlapping genes
    counts_df = pd.DataFrame(
        X_int,
        index=range(adata_train.n_obs),
        columns=adata_train.var_names.tolist()  # Real gene names
    )
    
    # Create celltypes DataFrame
    celltypes_df = pd.DataFrame({
        'Celltype': adata_train.obs['celltype'].values
    })
    
    # Save files
    counts_file = output_dir / "_counts.txt"
    celltypes_file = output_dir / "_celltypes.txt"
    
    logger.info(f"Saving counts to {counts_file}...")
    counts_df.to_csv(counts_file, sep="\t", index=True)
    
    logger.info(f"Saving celltypes to {celltypes_file}...")
    celltypes_df.to_csv(celltypes_file, sep="\t", index=False)
    
    # Create filtered TCGA file with same gene order
    logger.info("Creating compatible TCGA file...")
    tcga_compatible = tcga_data.loc[overlap_genes]
    tcga_file = output_dir / "tcga_compatible.txt"
    tcga_compatible.to_csv(tcga_file, sep="\t")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"✓ Compatible Tang dataset: {adata_train.shape[0]:,} cells × {len(overlap_genes):,} genes")
    print(f"✓ Training data saved to: {output_dir}")
    print(f"✓ Compatible TCGA file: {tcga_file}")
    print(f"✓ Genes compatible: {len(overlap_genes):,} genes")
    print(f"✓ Cell types: {len(adata_train.obs['celltype'].unique())}")
    
    return output_dir

if __name__ == "__main__":
    np.random.seed(42)
    output_dir = create_compatible_tang_dataset()
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. cd {output_dir}")
    print(f"2. scaden simulate --cells 50 --n_samples 1000 --data . --pattern '*_counts.txt'")
    print(f"3. scaden process data.h5ad tcga_compatible.txt")
    print(f"4. scaden train processed.h5ad --steps 5000")
    print(f"5. scaden predict --model_dir . --outname results.txt tcga_compatible.txt")
    print(f"\n🎯 Now Tang and TCGA will have identical gene sets!") 