#!/usr/bin/env python3
"""
NK Cell Dataset Batch Effect Inspector
=====================================

This script inspects the NK cell scRNA-seq dataset to identify batch/patient information
for batch effect analysis. Based on the data loading approach from create_hybrid_lm22_rebuffet_signatures.py

Dataset: NK cells in healthy human blood with subtypes NKint, NK1a, NK1b, NK1c, NK2, NK3
"""

# === ENVIRONMENT SETUP ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["NUMBA_DISABLE_CUDA"] = "1"   # Disable CUDA for numba
os.environ["OMP_NUM_THREADS"] = "1"      # Prevent threading conflicts

# Standard imports
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging

# Import scanpy with proper error handling
try:
    print("Importing scanpy...")
    import scanpy as sc
    sc.settings.verbosity = 1
    sc.settings.autoshow = False
    print(f"✓ Scanpy {sc.__version__} imported successfully")
except Exception as e:
    print(f"✗ CRITICAL ERROR: Scanpy import failed: {e}")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
# Input file (same as signature script)
NK_DATA_FILE = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"

# Expected NK subtypes from user description
EXPECTED_NK_SUBTYPES = ["NKint", "NK1a", "NK1b", "NK1c", "NK2", "NK3"]

# Common batch/patient metadata column names to look for
BATCH_EFFECT_COLUMNS = [
    # Patient/Donor identifiers
    'patient', 'patient_id', 'donor', 'donor_id', 'subject', 'subject_id', 'individual', 'individual_id',
    'sample', 'sample_id', 'specimen', 'specimen_id',
    
    # Batch identifiers  
    'batch', 'batch_id', 'batch_number', 'processing_batch', 'sequencing_batch', 'library_batch',
    'run', 'run_id', 'sequencing_run', 'plate', 'plate_id', 'well', 'well_id',
    
    # Technical factors
    'library', 'library_id', 'library_prep', 'prep_date', 'seq_date', 'processing_date',
    'protocol', 'method', 'kit', 'chemistry', 'platform',
    
    # Experimental factors
    'experiment', 'experiment_id', 'cohort', 'study', 'study_id', 'dataset', 'dataset_id',
    'condition', 'treatment', 'time_point', 'replicate',
    
    # Demographic/biological
    'age', 'sex', 'gender', 'ethnicity', 'race', 'bmi', 'weight', 'height'
]

def load_nk_data():
    """Load NK dataset and perform basic inspection."""
    logger.info("=" * 70)
    logger.info("LOADING NK DATASET FOR BATCH EFFECT INSPECTION")
    logger.info("=" * 70)
    
    if not os.path.exists(NK_DATA_FILE):
        logger.error(f"NK dataset file not found: {NK_DATA_FILE}")
        logger.info("Please ensure the data file is in the correct location:")
        logger.info(f"Expected: {NK_DATA_FILE}")
        return None
    
    try:
        logger.info(f"Loading NK dataset from: {NK_DATA_FILE}")
        adata = sc.read_h5ad(NK_DATA_FILE)
        logger.info(f"✓ Successfully loaded NK dataset. Shape: {adata.shape}")
        logger.info(f"  Cells: {adata.n_obs:,}")
        logger.info(f"  Genes: {adata.n_vars:,}")
        
        return adata
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR during NK data loading: {e}")
        return None

def inspect_metadata_structure(adata):
    """Comprehensive inspection of metadata structure."""
    logger.info("=" * 70)
    logger.info("METADATA STRUCTURE INSPECTION")
    logger.info("=" * 70)
    
    # Basic info
    logger.info(f"Cell metadata (.obs) columns: {len(adata.obs.columns)}")
    logger.info(f"Gene metadata (.var) columns: {len(adata.var.columns)}")
    
    # List all .obs columns
    logger.info("\nALL CELL METADATA COLUMNS (.obs):")
    logger.info("-" * 40)
    for i, col in enumerate(adata.obs.columns, 1):
        dtype = adata.obs[col].dtype
        nunique = adata.obs[col].nunique()
        logger.info(f"{i:2d}. {col} (dtype: {dtype}, unique values: {nunique})")
    
    # List all .var columns  
    logger.info("\nALL GENE METADATA COLUMNS (.var):")
    logger.info("-" * 40)
    for i, col in enumerate(adata.var.columns, 1):
        dtype = adata.var[col].dtype
        nunique = adata.var[col].nunique()
        logger.info(f"{i:2d}. {col} (dtype: {dtype}, unique values: {nunique})")
    
    # Check for layers
    if hasattr(adata, 'layers') and adata.layers:
        logger.info(f"\nAVAILABLE DATA LAYERS: {list(adata.layers.keys())}")
    else:
        logger.info("\nNO ADDITIONAL DATA LAYERS FOUND")
    
    # Check for .uns (unstructured annotations)
    if hasattr(adata, 'uns') and adata.uns:
        logger.info(f"\nUNSTRUCTURED ANNOTATIONS (.uns): {list(adata.uns.keys())}")
    else:
        logger.info("\nNO UNSTRUCTURED ANNOTATIONS FOUND")

def identify_batch_effect_columns(adata):
    """Identify potential batch effect columns."""
    logger.info("=" * 70)
    logger.info("BATCH EFFECT COLUMN IDENTIFICATION")
    logger.info("=" * 70)
    
    obs_columns = adata.obs.columns.tolist()
    obs_columns_lower = [col.lower() for col in obs_columns]
    
    # Find potential batch effect columns
    potential_batch_columns = []
    
    for batch_term in BATCH_EFFECT_COLUMNS:
        batch_term_lower = batch_term.lower()
        
        # Exact matches
        for i, col_lower in enumerate(obs_columns_lower):
            if col_lower == batch_term_lower:
                potential_batch_columns.append((obs_columns[i], "exact_match", batch_term))
        
        # Partial matches
        for i, col_lower in enumerate(obs_columns_lower):
            if batch_term_lower in col_lower or col_lower in batch_term_lower:
                if obs_columns[i] not in [item[0] for item in potential_batch_columns]:
                    potential_batch_columns.append((obs_columns[i], "partial_match", batch_term))
    
    if potential_batch_columns:
        logger.info("POTENTIAL BATCH EFFECT COLUMNS FOUND:")
        logger.info("-" * 45)
        for col, match_type, matched_term in potential_batch_columns:
            nunique = adata.obs[col].nunique()
            dtype = adata.obs[col].dtype
            logger.info(f"✓ {col} ({match_type} for '{matched_term}')")
            logger.info(f"   - Data type: {dtype}")
            logger.info(f"   - Unique values: {nunique}")
            
            # Show sample values
            sample_values = adata.obs[col].value_counts().head(5)
            logger.info(f"   - Top values: {dict(sample_values)}")
            logger.info("")
    else:
        logger.warning("NO OBVIOUS BATCH EFFECT COLUMNS FOUND")
        logger.info("This could mean:")
        logger.info("- Batch information uses non-standard column names")
        logger.info("- Data has already been batch-corrected")
        logger.info("- All cells are from the same batch/patient")

def inspect_nk_subtypes(adata):
    """Inspect NK cell subtype annotations."""
    logger.info("=" * 70)
    logger.info("NK CELL SUBTYPE INSPECTION")
    logger.info("=" * 70)
    
    # Look for potential cell type columns
    obs_columns = adata.obs.columns.tolist()
    potential_celltype_columns = []
    
    celltype_terms = ['celltype', 'cell_type', 'cluster', 'ident', 'annotation', 'subtype', 'label', 'class']
    
    for term in celltype_terms:
        for col in obs_columns:
            if term in col.lower():
                potential_celltype_columns.append(col)
    
    # Remove duplicates while preserving order
    potential_celltype_columns = list(dict.fromkeys(potential_celltype_columns))
    
    logger.info("POTENTIAL CELL TYPE COLUMNS:")
    logger.info("-" * 35)
    
    for col in potential_celltype_columns:
        nunique = adata.obs[col].nunique()
        logger.info(f"Column: {col} ({nunique} unique values)")
        
        # Check if this column contains expected NK subtypes
        unique_values = set(adata.obs[col].unique())
        expected_nk_set = set(EXPECTED_NK_SUBTYPES)
        
        # Count how many expected NK subtypes are found
        found_nk_subtypes = unique_values.intersection(expected_nk_set)
        
        if found_nk_subtypes:
            logger.info(f"✓ CONTAINS NK SUBTYPES: {sorted(found_nk_subtypes)}")
            
            # Show full distribution
            subtype_counts = adata.obs[col].value_counts()
            logger.info("  Full distribution:")
            for subtype, count in subtype_counts.items():
                percentage = (count / len(adata.obs)) * 100
                logger.info(f"    {subtype}: {count:,} cells ({percentage:.1f}%)")
        else:
            # Show sample values anyway
            sample_values = adata.obs[col].value_counts().head(10)
            logger.info(f"  Sample values: {list(sample_values.index)}")
        
        logger.info("")

def inspect_continuous_variables(adata):
    """Inspect continuous variables that might indicate batch effects."""
    logger.info("=" * 70)
    logger.info("CONTINUOUS VARIABLES INSPECTION")
    logger.info("=" * 70)
    
    continuous_cols = []
    for col in adata.obs.columns:
        if adata.obs[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            nunique = adata.obs[col].nunique()
            if nunique > 10:  # Likely continuous
                continuous_cols.append(col)
    
    if continuous_cols:
        logger.info("CONTINUOUS VARIABLES (potential QC metrics):")
        logger.info("-" * 50)
        for col in continuous_cols:
            values = adata.obs[col]
            logger.info(f"{col}:")
            logger.info(f"  Range: {values.min():.2f} to {values.max():.2f}")
            logger.info(f"  Mean ± SD: {values.mean():.2f} ± {values.std():.2f}")
            logger.info(f"  Median: {values.median():.2f}")
            logger.info("")
    else:
        logger.info("No obvious continuous variables found")

def generate_batch_effect_summary(adata):
    """Generate summary recommendations for batch effect analysis."""
    logger.info("=" * 70)
    logger.info("BATCH EFFECT ANALYSIS RECOMMENDATIONS")
    logger.info("=" * 70)
    
    obs_columns = adata.obs.columns.tolist()
    
    # Check for obvious batch indicators
    high_cardinality_cols = []
    for col in obs_columns:
        nunique = adata.obs[col].nunique()
        if 5 <= nunique <= len(adata.obs) * 0.5:  # Between 5 and 50% of cells
            high_cardinality_cols.append((col, nunique))
    
    logger.info("RECOMMENDED COLUMNS TO INVESTIGATE FOR BATCH EFFECTS:")
    logger.info("-" * 55)
    
    if high_cardinality_cols:
        for col, nunique in sorted(high_cardinality_cols, key=lambda x: x[1]):
            percentage = (nunique / len(adata.obs)) * 100
            logger.info(f"• {col}: {nunique} unique values ({percentage:.1f}% of cells)")
            
            # Show sample values
            sample_values = adata.obs[col].value_counts().head(3)
            logger.info(f"  Top values: {dict(sample_values)}")
    else:
        logger.info("No obvious candidate columns found")
        logger.info("This suggests either:")
        logger.info("- Very clean, single-batch data")
        logger.info("- Batch information encoded in non-obvious ways")
        logger.info("- Already batch-corrected data")
    
    logger.info("\nNEXT STEPS:")
    logger.info("-" * 15)
    logger.info("1. Examine the most promising columns identified above")
    logger.info("2. Look for clustering by potential batch variables")
    logger.info("3. Check if there are technical covariates affecting expression")
    logger.info("4. Consider PCA/UMAP visualization colored by potential batch variables")

def main():
    """Main function to inspect NK dataset for batch effects."""
    print("=" * 80)
    print("NK CELL DATASET BATCH EFFECT INSPECTOR")
    print("=" * 80)
    print()
    print("Inspecting NK cell scRNA-seq dataset for:")
    print("• Patient/donor information")
    print("• Batch/technical variables")
    print("• NK cell subtype annotations")
    print("• Potential confounding factors")
    print()
    
    # Step 1: Load data
    adata = load_nk_data()
    if adata is None:
        logger.error("Failed to load NK data. Exiting.")
        return False
    
    # Step 2: Inspect metadata structure
    inspect_metadata_structure(adata)
    
    # Step 3: Identify potential batch effect columns
    identify_batch_effect_columns(adata)
    
    # Step 4: Inspect NK subtypes
    inspect_nk_subtypes(adata)
    
    # Step 5: Inspect continuous variables
    inspect_continuous_variables(adata)
    
    # Step 6: Generate recommendations
    generate_batch_effect_summary(adata)
    
    logger.info("=" * 70)
    logger.info("BATCH EFFECT INSPECTION COMPLETE")
    logger.info("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ NK dataset batch effect inspection completed!")
        print("📊 Check the output above for batch/patient information")
        print("🔍 Use the recommendations for further batch effect analysis")
    else:
        print("\n❌ NK dataset inspection failed!")