#!/usr/bin/env python3
"""
Create Full Tang Dataset for SCADEN
===================================

Phase 2: Convert the complete Tang NK dataset (142,304 cells) to SCADEN format
with proper 80/20 stratified split maintaining cell type proportions.

Based on the successful small test workflow.
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

def create_full_tang_dataset():
    """Create full Tang dataset with 80/20 stratified split."""
    
    print("=== CREATING FULL TANG DATASET ===\n")
    
    # Create output directories
    train_dir = Path("tang_full_train")
    test_dir = Path("tang_full_test")
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created directories: {train_dir}, {test_dir}")
    
    # Load Tang data
    logger.info("Loading full Tang NK data...")
    adata = sc.read_h5ad("data/processed/comb_CD56_CD16_NK.h5ad")
    logger.info(f"Original shape: {adata.shape}")
    logger.info(f"Cell types: {len(adata.obs['celltype'].unique())}")
    
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
    def convert_to_scaden_format(adata_subset, output_dir, dataset_name):
        """Convert AnnData to SCADEN format."""
        logger.info(f"\nConverting {dataset_name} to SCADEN format...")
        
        # Convert sparse matrix to dense and to integers
        if hasattr(adata_subset.X, 'toarray'):
            X_dense = adata_subset.X.toarray()
        else:
            X_dense = adata_subset.X
        
        # Convert to integers
        X_int = np.round(X_dense).astype(int)
        logger.info(f"  Data range: {X_int.min()} - {X_int.max()}")
        
        # Create counts DataFrame (Cell × Gene)
        counts_df = pd.DataFrame(
            X_int,
            index=range(adata_subset.n_obs),  # Simple cell indices
            columns=[f"gene{i}" for i in range(adata_subset.n_vars)]  # Generic gene names
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
    train_counts_file, train_celltypes_file = convert_to_scaden_format(
        adata_train, train_dir, "Training set"
    )
    
    # Convert test set  
    test_counts_file, test_celltypes_file = convert_to_scaden_format(
        adata_test, test_dir, "Test set"
    )
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"✓ Full Tang dataset converted to SCADEN format")
    print(f"✓ Training set: {adata_train.shape[0]:,} cells → {train_dir}")
    print(f"✓ Test set: {adata_test.shape[0]:,} cells → {test_dir}")
    print(f"✓ {adata_train.shape[1]:,} genes in both sets")
    print(f"✓ {len(adata.obs['celltype'].unique())} cell types preserved")
    print(f"✓ Cell type proportions maintained in split")
    
    return train_dir, test_dir

def create_bulk_data_for_testing(train_dir):
    """Create bulk data for testing the trained model."""
    logger.info("\nCreating bulk data for testing...")
    
    # Load the training data to get gene names
    adata = sc.read_h5ad(f"{train_dir}/data.h5ad") if (Path(train_dir) / "data.h5ad").exists() else None
    
    if adata is None:
        # Create a simple bulk data file with same genes as the counts file
        counts_sample = pd.read_csv(f"{train_dir}/_counts.txt", sep="\t", index_col=0, nrows=1)
        gene_names = counts_sample.columns.tolist()
    else:
        gene_names = [f"gene{i}" for i in range(adata.n_vars)]
    
    # Create test bulk data
    np.random.seed(42)
    n_genes = len(gene_names)
    n_samples = 10  # Create 10 test samples
    
    bulk_data = np.random.randint(100, 1000, size=(n_genes, n_samples))
    
    bulk_df = pd.DataFrame(
        bulk_data,
        index=gene_names,
        columns=[f"bulk_sample{i}" for i in range(n_samples)]
    )
    
    bulk_file = train_dir / "test_bulk_data.txt"
    bulk_df.to_csv(bulk_file, sep="\t", index=True)
    
    logger.info(f"  Created bulk data: {bulk_file}")
    logger.info(f"  Shape: {bulk_df.shape}")
    
    return bulk_file

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create the datasets
    train_dir, test_dir = create_full_tang_dataset()
    
    # Create bulk data for testing
    bulk_file = create_bulk_data_for_testing(train_dir)
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. cd {train_dir}")
    print(f"2. scaden simulate --cells 50 --n_samples 1000 --data . --pattern '*_counts.txt'")
    print(f"3. scaden process data.h5ad test_bulk_data.txt")
    print(f"4. scaden train processed.h5ad --model_dir ./model")
    print(f"5. scaden predict test_bulk_data.txt --model_dir ./model")
    print(f"\nDatasets ready in: {train_dir} and {test_dir}") 