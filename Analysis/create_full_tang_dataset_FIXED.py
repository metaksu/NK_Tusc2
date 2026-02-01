#!/usr/bin/env python3
"""
FIXED Tang NK Data to SCADEN Format Converter
===========================================

This script converts Tang NK cell data to SCADEN format while PRESERVING
the original HUGO gene symbols instead of replacing them with generic names.

This fixes the major bug where we threw away real gene names.
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

def create_full_tang_dataset_fixed():
    """Create full Tang dataset with 80/20 stratified split - PRESERVING GENE NAMES."""
    
    print("=== CREATING FULL TANG DATASET (FIXED) ===\n")
    
    # Create output directories
    train_dir = Path("tang_full_train_FIXED")
    test_dir = Path("tang_full_test_FIXED")
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created directories: {train_dir}, {test_dir}")
    
    # Load Tang data
    logger.info("Loading full Tang NK data...")
    adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")
    logger.info(f"Original shape: {adata.shape}")
    logger.info(f"Cell types: {len(adata.obs['celltype'].unique())}")
    
    # CHECK GENE NAMES
    logger.info(f"Original gene names (first 10): {adata.var_names[:10].tolist()}")
    logger.info(f"Total genes: {len(adata.var_names)}")
    
    # Display cell type distribution
    print("\nCell type distribution:")
    cell_type_counts = adata.obs['celltype'].value_counts()
    for celltype, count in cell_type_counts.items():
        print(f"  {celltype}: {count:,} cells")
    
    # Create stratified 80/20 split
    logger.info("\nCreating 80/20 stratified split...")
    
    train_indices = []
    test_indices = []
    
    for celltype in adata.obs['celltype'].unique():
        # Get all cells of this type
        celltype_mask = adata.obs['celltype'] == celltype
        celltype_cells = adata.obs[celltype_mask].index.tolist()
        
        # Split 80/20 for this cell type
        train_cells, test_cells = train_test_split(
            celltype_cells, 
            test_size=0.2, 
            random_state=42,
            stratify=None  # No further stratification needed
        )
        
        train_indices.extend(train_cells)
        test_indices.extend(test_cells)
        
        logger.info(f"  {celltype}: {len(train_cells)} train, {len(test_cells)} test")
    
    # Create train and test datasets
    adata_train = adata[train_indices, :].copy()
    adata_test = adata[test_indices, :].copy()
    
    logger.info(f"\nFinal split:")
    logger.info(f"  Training: {adata_train.shape[0]:,} cells")
    logger.info(f"  Testing: {adata_test.shape[0]:,} cells")
    logger.info(f"  Genes: {adata_train.shape[1]:,}")
    
    # Verify proportions are maintained
    print("\nVerifying proportions maintained:")
    train_props = adata_train.obs['celltype'].value_counts(normalize=True)
    test_props = adata_test.obs['celltype'].value_counts(normalize=True)
    original_props = adata.obs['celltype'].value_counts(normalize=True)
    
    for celltype in original_props.index:
        orig_prop = original_props[celltype]
        train_prop = train_props.get(celltype, 0)
        test_prop = test_props.get(celltype, 0)
        print(f"  {celltype}: Original {orig_prop:.3f}, Train {train_prop:.3f}, Test {test_prop:.3f}")
    
    # Convert datasets to SCADEN format
    def convert_to_scaden_format_FIXED(adata_subset, output_dir, dataset_name):
        """Convert AnnData to SCADEN format - PRESERVING GENE NAMES."""
        logger.info(f"\nConverting {dataset_name} to SCADEN format...")
        
        # Convert sparse matrix to dense and to integers
        if hasattr(adata_subset.X, 'toarray'):
            X_dense = adata_subset.X.toarray()
        else:
            X_dense = adata_subset.X
        
        # Convert to integers
        X_int = np.round(X_dense).astype(int)
        logger.info(f"  Data range: {X_int.min()} - {X_int.max()}")
        
        # *** FIXED: USE ORIGINAL GENE NAMES ***
        original_gene_names = adata_subset.var_names.tolist()
        logger.info(f"  Using original gene names: {original_gene_names[:5]}...")
        
        # Create counts DataFrame (Cell × Gene)
        counts_df = pd.DataFrame(
            X_int,
            index=range(adata_subset.n_obs),  # Simple cell indices
            columns=original_gene_names  # *** FIXED: USE REAL GENE NAMES ***
        )
        
        # Create celltypes DataFrame
        celltypes_df = pd.DataFrame({
            'Celltype': adata_subset.obs['celltype'].values
        })
        
        # Save files
        counts_file = output_dir / "_counts.txt"
        celltypes_file = output_dir / "_celltypes.txt"
        
        logger.info(f"  Saving counts to {counts_file}...")
        counts_df.to_csv(counts_file, sep="\t", index=True)
        
        logger.info(f"  Saving celltypes to {celltypes_file}...")
        celltypes_df.to_csv(celltypes_file, sep="\t", index=False)
        
        # Verify files
        logger.info(f"  Counts file size: {counts_file.stat().st_size / 1024 / 1024:.1f} MB")
        logger.info(f"  Celltypes file size: {celltypes_file.stat().st_size / 1024:.1f} KB")
        
        return counts_file, celltypes_file
    
    # Convert training set
    train_counts_file, train_celltypes_file = convert_to_scaden_format_FIXED(
        adata_train, train_dir, "Training set"
    )
    
    # Convert test set  
    test_counts_file, test_celltypes_file = convert_to_scaden_format_FIXED(
        adata_test, test_dir, "Test set"
    )
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"✓ FIXED: Tang dataset converted to SCADEN format WITH ORIGINAL GENE NAMES")
    print(f"✓ Training set: {adata_train.shape[0]:,} cells → {train_dir}")
    print(f"✓ Test set: {adata_test.shape[0]:,} cells → {test_dir}")
    print(f"✓ {adata_train.shape[1]:,} genes in both sets")
    print(f"✓ {len(adata.obs['celltype'].unique())} cell types preserved")
    print(f"✓ Cell type proportions maintained in split")
    print(f"✓ GENE NAMES PRESERVED: {adata_train.var_names[:3].tolist()}...")
    
    return train_dir, test_dir

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create the FIXED datasets
    train_dir, test_dir = create_full_tang_dataset_fixed()
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. cd {train_dir}")
    print(f"2. scaden simulate --cells 50 --n_samples 1000 --data . --pattern '*_counts.txt'")
    print(f"3. scaden process data.h5ad test_bulk_data.txt")
    print(f"4. scaden train processed.h5ad --steps 5000")
    print(f"5. scaden predict --model_dir . --outname predictions.txt <bulk_data.txt>")
    print(f"\nFIXED datasets ready in: {train_dir} and {test_dir}")
    print(f"🎯 Now the model will work with TCGA HUGO gene symbols!") 