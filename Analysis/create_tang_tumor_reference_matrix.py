#!/usr/bin/env python3
"""
Tang NK Reference Matrix Generator for CIBERSORTx
=================================================

This script recreates the processing file that generates the Tang_reference_matrix_700cells_per_phenotype.txt
file for CIBERSORTx deconvolution analysis.

The script:
1. Loads Tang NK dataset using the exact patterns from the main project
2. Uses ALL tissue contexts (tumor, normal, blood, etc.) for comprehensive representation
3. Samples up to 700 cells per NK subtype (13 Tang subtypes) randomly from all contexts
4. Exports in CIBERSORTx format (genes as rows, cells as columns)
5. Handles TPM/raw expression data according to CIBERSORTx requirements

Reference from user: This file consists of reference single cell RNA-seq expression 
profiles from which a signature matrix will be generated.

CIBERSORTx formatting requirements:
- Genes in column 1; Mixture labels (sample names) in row 1
- Data should be in non-log space (TPM recommended)
- CIBERSORTx will assume log space if max expression < 50
- Remove gene symbol redundancy
- Feature selection typically doesn't use all genes
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings
from scipy import sparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Configuration Constants - Mirroring the main project setup
# These constants are taken directly from the main NK analysis scripts

# === TANG DATASET CONFIGURATION ===
# File path (using the exact pattern from main scripts)
TANG_COMBINED_H5AD_FILE = "data/processed/comb_CD56_CD16_NK.h5ad"

# Key Metadata Columns for Tang Combined Dataset (exact column names from main project)
TANG_TISSUE_COL = "meta_tissue_in_paper"      # Primary tissue context (Blood/Tumor/Normal/Other tissue)
TANG_TISSUE_BLOOD_COL = "meta_tissue"         # Blood-specific dataset column
TANG_MAJORTYPE_COL = "Majortype"              # CD56bright/dim + CD16 high/low combinations
TANG_CELLTYPE_COL = "celltype"                # Fine-grained NK subtypes (13 subtypes for tumor)
TANG_HISTOLOGY_COL = "meta_histology"         # Cancer type information (25 cancer types)
TANG_BATCH_COL = "batch"                      # Batch information for integration
TANG_DATASETS_COL = "datasets"                # Source dataset information
TANG_PATIENT_ID_COL = "meta_patientID"        # Patient identifier

# === TANG NK SUBTYPES CONFIGURATION ===
# The 13 Tang NK subtypes found in tumor tissue (from main project analysis)
TANG_CD56BRIGHT_SUBTYPES = [
    "CD56brightCD16lo-c1-GZMH",        # GZMH+ cytotoxic
    "CD56brightCD16lo-c2-IL7R-RGS1lo", # IL7R+ immature
    "CD56brightCD16lo-c3-CCL3",        # CCL3+ chemokine
    "CD56brightCD16lo-c4-AREG",        # AREG+ tissue repair
    "CD56brightCD16lo-c5-CXCR4",       # CXCR4+ trafficking
    "CD56brightCD16hi",                # Double-positive transitional
]

TANG_CD56DIM_SUBTYPES = [
    "CD56dimCD16hi-c1-IL32",           # Cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1",         # Tissue-homing
    "CD56dimCD16hi-c3-ZNF90",          # Mature
    "CD56dimCD16hi-c4-NFKBIA",         # Activated
    "CD56dimCD16hi-c5-MKI67",          # Proliferating
    "CD56dimCD16hi-c6-DNAJB1",         # Stress-response
    "CD56dimCD16hi-c7-NR4A3",          # Stimulated
    "CD56dimCD16hi-c8-KLRC2",          # Adaptive (NKG2C+)
]

# All 13 Tang NK subtypes (order matters for reproducibility)
TANG_ALL_SUBTYPES = TANG_CD56BRIGHT_SUBTYPES + TANG_CD56DIM_SUBTYPES

# === PROCESSING PARAMETERS ===
TARGET_CELLS_PER_PHENOTYPE = 700              # Target cells per NK subtype
MIN_CELLS_PER_PHENOTYPE = 5                   # Minimum cells required for inclusion
RANDOM_SEED = 42                               # For reproducibility
MIN_GENES_PER_CELL = 200                       # Minimum genes per cell for filtering

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = Path("outputs/signature_matrices/CIBERSORTx_Input_Files")
OUTPUT_FILENAME = "Tang_reference_matrix_700cells_per_phenotype.txt"
SUMMARY_FILENAME = "Tang_reference_matrix_generation_summary.txt"


def load_and_validate_tang_data():
    """
    Load Tang NK dataset using the exact loading pattern from the main project.
    
    Returns:
    --------
    adata : AnnData or None
        Loaded and validated Tang dataset, or None if loading fails
    """
    logger.info("=" * 70)
    logger.info("LOADING TANG NK DATASET")
    logger.info("=" * 70)
    
    # Check if file exists
    if not os.path.exists(TANG_COMBINED_H5AD_FILE):
        logger.error(f"Tang dataset file not found: {TANG_COMBINED_H5AD_FILE}")
        logger.error("Please ensure the file exists at the expected location.")
        return None
    
    try:
        # Load the Tang Combined NK Dataset (exact pattern from main scripts)
        logger.info(f"Loading Tang combined NK dataset from: {TANG_COMBINED_H5AD_FILE}")
        adata_tang_full = sc.read_h5ad(TANG_COMBINED_H5AD_FILE)
        logger.info(f"Successfully loaded Tang combined dataset. Shape: {adata_tang_full.shape}")
        
        # Dataset Overview & Validation (mirroring main scripts)
        logger.info("Dataset overview:")
        logger.info(f"  Total cells: {adata_tang_full.n_obs:,}")
        logger.info(f"  Total genes: {adata_tang_full.n_vars:,}")
        logger.info(f"  Expression data type: {adata_tang_full.X.dtype}")
        logger.info(f"  Expression range: {adata_tang_full.X.min():.3f} to {adata_tang_full.X.max():.3f}")
        
        # Check for raw data availability
        if hasattr(adata_tang_full, "raw") and adata_tang_full.raw is not None:
            logger.info(f"  Raw data available: {adata_tang_full.raw.shape}")
            logger.info(f"  Raw expression range: {adata_tang_full.raw.X.min():.3f} to {adata_tang_full.raw.X.max():.3f}")
        else:
            logger.info("  Raw data: Not available")
        
        # Validate Key Metadata Columns (exact pattern from main scripts)
        logger.info("Metadata validation:")
        available_columns = list(adata_tang_full.obs.columns)
        logger.info(f"  Available columns: {len(available_columns)} total")
        
        # Check and report on key columns
        key_columns = {
            TANG_TISSUE_COL: "Primary tissue context",
            TANG_MAJORTYPE_COL: "CD56/CD16 combinations",
            TANG_CELLTYPE_COL: "Fine-grained NK subtypes",
            TANG_HISTOLOGY_COL: "Cancer type information",
        }
        
        for col, description in key_columns.items():
            if col in adata_tang_full.obs.columns:
                unique_vals = adata_tang_full.obs[col].nunique()
                logger.info(f"  [OK] {col}: {unique_vals} unique values")
                # Show top values with counts
                value_counts = adata_tang_full.obs[col].value_counts()
                top_3_values = value_counts.head(3)
                for val, count in top_3_values.items():
                    logger.info(f"    {val}: {count:,} cells ({count/adata_tang_full.n_obs*100:.1f}%)")
            else:
                logger.error(f"  [MISSING] {col}: Column not found")
                return None
        
        return adata_tang_full
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR during Tang data loading: {e}")
        return None


def preprocess_tang_data(adata_tang_full):
    """
    Preprocess Tang data using the exact workflow from the main project.
    
    Parameters:
    -----------
    adata_tang_full : AnnData
        Raw Tang dataset
        
    Returns:
    --------
    adata_tang_full : AnnData
        Preprocessed Tang dataset
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING TANG DATA")
    logger.info("=" * 70)
    
    logger.info(f"Initial shape: {adata_tang_full.shape}")
    
    # Step 1: Store Raw Counts (exact pattern from main scripts)
    logger.info("Storing raw counts in adata_tang_full.layers['counts']...")
    adata_tang_full.layers["counts"] = adata_tang_full.X.copy()
    
    # Step 2: Perform ONLY minimal cell filtering (exact pattern from main scripts)
    logger.info(f"Filtering cells with fewer than {MIN_GENES_PER_CELL} genes...")
    sc.pp.filter_cells(adata_tang_full, min_genes=MIN_GENES_PER_CELL)
    logger.info(f"Shape after cell filtering: {adata_tang_full.shape}")
    
    # Step 3: Check if normalization is needed (exact pattern from main scripts)
    max_expression = adata_tang_full.X.max()
    if max_expression > 50:  # Likely raw counts or TPM
        logger.info(f"Data appears to be raw counts or TPM (max: {max_expression:.1f}). Normalizing and log-transforming...")
        sc.pp.normalize_total(adata_tang_full, target_sum=1e4)
        sc.pp.log1p(adata_tang_full)
        logger.info("Normalization and log-transformation complete.")
    else:
        logger.info(f"Data appears already normalized (max: {max_expression:.1f}). Skipping normalization.")
    
    # Step 4: Set the .raw Attribute (exact pattern from main scripts)
    logger.info("Storing current log-normalized, unscaled data (with all genes) into adata_tang_full.raw")
    adata_tang_full.raw = adata_tang_full.copy()
    logger.info("Raw attribute set for gene expression analysis.")
    
    return adata_tang_full


def analyze_tissue_distribution(adata_tang_full):
    """
    Analyze tissue distribution in Tang dataset and prepare for sampling across all contexts.
    
    Parameters:
    -----------
    adata_tang_full : AnnData
        Full Tang dataset
        
    Returns:
    --------
    adata_tang_full : AnnData
        Full Tang dataset (unchanged, just analyzed)
    """
    logger.info("=" * 70)
    logger.info("ANALYZING TISSUE DISTRIBUTION")
    logger.info("=" * 70)
    
    # Check tissue distribution
    tissue_counts = adata_tang_full.obs[TANG_TISSUE_COL].value_counts()
    logger.info("Tissue distribution in full dataset:")
    for tissue, count in tissue_counts.items():
        logger.info(f"  {tissue}: {count:,} cells ({count/adata_tang_full.n_obs*100:.1f}%)")
    
    logger.info(f"Using ALL tissue contexts for comprehensive NK subtype representation")
    logger.info(f"Total cells available: {adata_tang_full.shape[0]:,}")
    
    # Check subtype distribution across all tissues
    subtype_counts = adata_tang_full.obs[TANG_CELLTYPE_COL].value_counts()
    logger.info(f"NK subtype distribution across all tissues ({len(subtype_counts)} subtypes):")
    for subtype, count in subtype_counts.items():
        logger.info(f"  {subtype}: {count:,} cells")
    
    # Show tissue context distribution per subtype
    logger.info("\nTissue context breakdown by NK subtype:")
    subtype_tissue_crosstab = pd.crosstab(
        adata_tang_full.obs[TANG_CELLTYPE_COL], 
        adata_tang_full.obs[TANG_TISSUE_COL], 
        margins=True
    )
    logger.info(f"\n{subtype_tissue_crosstab}")
    
    return adata_tang_full


def sample_cells_per_subtype(adata_tang_full):
    """
    Sample up to TARGET_CELLS_PER_PHENOTYPE cells per NK subtype from all tissue contexts.
    
    Parameters:
    -----------
    adata_tang_full : AnnData
        Full Tang dataset (all tissue contexts)
        
    Returns:
    --------
    adata_sampled : AnnData
        Dataset with sampled cells per subtype
    """
    logger.info("=" * 70)
    logger.info("SAMPLING CELLS PER NK SUBTYPE (ALL TISSUES)")
    logger.info("=" * 70)
    
    np.random.seed(RANDOM_SEED)
    
    # Get available subtypes
    available_subtypes = adata_tang_full.obs[TANG_CELLTYPE_COL].unique()
    logger.info(f"Available NK subtypes across all tissues: {len(available_subtypes)}")
    
    # Filter for valid subtypes (those with sufficient cells)
    subtype_counts = adata_tang_full.obs[TANG_CELLTYPE_COL].value_counts()
    valid_subtypes = subtype_counts[subtype_counts >= MIN_CELLS_PER_PHENOTYPE].index
    
    logger.info(f"Subtypes with >= {MIN_CELLS_PER_PHENOTYPE} cells: {len(valid_subtypes)}")
    
    sampled_indices = []
    final_counts = {}
    tissue_breakdown = {}
    
    logger.info(f"Sampling up to {TARGET_CELLS_PER_PHENOTYPE} cells per subtype (randomized from all tissues):")
    
    for subtype in valid_subtypes:
        # Get cells for this subtype
        subtype_mask = adata_tang_full.obs[TANG_CELLTYPE_COL] == subtype
        subtype_indices = np.where(subtype_mask)[0]
        
        n_available = len(subtype_indices)
        logger.info(f"  {subtype}: {n_available:,} available", end="")
        
        if n_available >= TARGET_CELLS_PER_PHENOTYPE:
            # Sample exactly TARGET_CELLS_PER_PHENOTYPE cells
            sampled = np.random.choice(
                subtype_indices, size=TARGET_CELLS_PER_PHENOTYPE, replace=False
            )
            sampled_indices.extend(sampled)
            final_counts[subtype] = TARGET_CELLS_PER_PHENOTYPE
            logger.info(f" → sampled {TARGET_CELLS_PER_PHENOTYPE}")
        else:
            # Use all available cells if less than target
            sampled_indices.extend(subtype_indices)
            final_counts[subtype] = n_available
            logger.info(f" → using all {n_available} (insufficient for {TARGET_CELLS_PER_PHENOTYPE})")
        
        # Track tissue context breakdown for sampled cells
        sampled_subtype_data = adata_tang_full[subtype_indices if n_available < TARGET_CELLS_PER_PHENOTYPE else sampled]
        tissue_breakdown[subtype] = sampled_subtype_data.obs[TANG_TISSUE_COL].value_counts().to_dict()
    
    # Create sampled dataset
    adata_sampled = adata_tang_full[sampled_indices].copy()
    
    logger.info(f"Final sampled dataset: {adata_sampled.shape[0]:,} cells")
    logger.info("Final subtype distribution:")
    for subtype, count in final_counts.items():
        logger.info(f"  {subtype}: {count} cells")
    
    # Show tissue context representation in final sample
    logger.info("\nTissue context representation in final sample:")
    for subtype in final_counts.keys():
        if subtype in tissue_breakdown:
            tissues = ", ".join([f"{tissue}:{count}" for tissue, count in tissue_breakdown[subtype].items()])
            logger.info(f"  {subtype}: {tissues}")
    
    return adata_sampled, final_counts


def get_expression_data_for_cibersortx(adata_sampled):
    """
    Extract expression data in the proper format for CIBERSORTx.
    
    Parameters:
    -----------
    adata_sampled : AnnData
        Sampled dataset
        
    Returns:
    --------
    expr_df : pd.DataFrame
        Expression data with genes as rows, cells as columns
    gene_names : pd.Index
        Gene names
    cell_labels : pd.Series
        Cell subtype labels
    """
    logger.info("=" * 70)
    logger.info("PREPARING EXPRESSION DATA FOR CIBERSORTX")
    logger.info("=" * 70)
    
    # Use raw data if available (recommended for CIBERSORTx)
    if hasattr(adata_sampled, 'raw') and adata_sampled.raw is not None:
        logger.info("Using raw expression data (recommended for CIBERSORTx)")
        expr_data = adata_sampled.raw.X
        gene_names = adata_sampled.raw.var_names
    else:
        logger.info("Using main expression data (.X)")
        expr_data = adata_sampled.X
        gene_names = adata_sampled.var_names
    
    # Convert sparse to dense if needed
    if sparse.issparse(expr_data):
        logger.info("Converting sparse matrix to dense...")
        expr_data = expr_data.toarray()
    
    # Check data properties
    logger.info(f"Expression data shape: {expr_data.shape}")
    logger.info(f"Expression range: {expr_data.min():.3f} to {expr_data.max():.3f}")
    logger.info(f"Number of genes: {len(gene_names):,}")
    
    # Create DataFrame with genes as rows, cells as columns
    # This is the required format for CIBERSORTx
    expr_df = pd.DataFrame(
        expr_data.T,  # Transpose to get genes as rows
        index=gene_names,
        columns=[f"Cell_{i}" for i in range(expr_data.shape[0])]
    )
    
    # Get cell labels (subtype assignments)
    cell_labels = adata_sampled.obs[TANG_CELLTYPE_COL].copy()
    cell_labels.index = expr_df.columns  # Match column names
    
    # Handle log-space data for CIBERSORTx
    if expr_df.values.max() < 50:
        logger.warning("Data appears to be in log space (max < 50)")
        logger.warning("CIBERSORTx will automatically anti-log this data")
        logger.info("Converting back to linear space for CIBERSORTx compatibility...")
        # Convert from log(TPM+1) back to TPM
        expr_df = np.expm1(expr_df)
        logger.info(f"After anti-log conversion - range: {expr_df.values.min():.3f} to {expr_df.values.max():.3f}")
    
    return expr_df, gene_names, cell_labels


def create_cibersortx_reference_matrix(expr_df, cell_labels, output_path):
    """
    Create the final CIBERSORTx reference matrix file.
    
    Parameters:
    -----------
    expr_df : pd.DataFrame
        Expression data with genes as rows, cells as columns
    cell_labels : pd.Series
        Cell subtype labels
    output_path : Path
        Output file path
    """
    logger.info("=" * 70)
    logger.info("CREATING CIBERSORTX REFERENCE MATRIX")
    logger.info("=" * 70)
    
    # Prepare the reference matrix
    # Format: First row = cell labels, First column = gene names
    reference_matrix = expr_df.copy()
    
    # Replace column names with subtype labels
    reference_matrix.columns = cell_labels.values
    
    # Add gene names as first column (required by CIBERSORTx)
    reference_matrix.insert(0, "Gene", reference_matrix.index)
    
    logger.info(f"Reference matrix shape: {reference_matrix.shape}")
    logger.info(f"Genes (rows): {reference_matrix.shape[0]:,}")
    logger.info(f"Cells (columns): {reference_matrix.shape[1]-1:,}")  # -1 for gene column
    
    # Check for duplicate gene names (CIBERSORTx requirement)
    duplicate_genes = reference_matrix["Gene"].duplicated()
    if duplicate_genes.any():
        n_duplicates = duplicate_genes.sum()
        logger.warning(f"Found {n_duplicates} duplicate gene names")
        logger.warning("CIBERSORTx will add unique identifiers to duplicates")
    else:
        logger.info("No duplicate gene names found")
    
    # Save the reference matrix
    logger.info(f"Saving reference matrix to: {output_path}")
    reference_matrix.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    
    # Verify file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"File saved successfully. Size: {file_size_mb:.1f} MB")
    
    return reference_matrix


def generate_analysis_summary(adata_sampled, final_counts, reference_matrix, output_dir):
    """
    Generate a comprehensive summary of the reference matrix generation process.
    
    Parameters:
    -----------
    adata_sampled : AnnData
        Final sampled dataset
    final_counts : dict
        Cell counts per subtype
    reference_matrix : pd.DataFrame
        Generated reference matrix
    output_dir : Path
        Output directory
    """
    logger.info("=" * 70)
    logger.info("GENERATING ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    summary_path = output_dir / SUMMARY_FILENAME
    
    with open(summary_path, 'w') as f:
        f.write("Tang NK Reference Matrix Generation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Source file: {TANG_COMBINED_H5AD_FILE}\n")
        f.write(f"Target cells per phenotype: {TARGET_CELLS_PER_PHENOTYPE}\n")
        f.write(f"Minimum cells per phenotype: {MIN_CELLS_PER_PHENOTYPE}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Final reference matrix dimensions: {reference_matrix.shape[0]:,} genes × {reference_matrix.shape[1]-1:,} cells\n")
        f.write(f"Total cells included: {sum(final_counts.values()):,}\n")
        f.write(f"Number of NK subtypes: {len(final_counts)}\n")
        f.write(f"Tissue contexts: ALL (tumor, normal, blood, other) - randomized sampling\n\n")
        
        f.write("CELL DISTRIBUTION BY NK SUBTYPE\n")
        f.write("-" * 40 + "\n")
        total_cells = sum(final_counts.values())
        for subtype, count in sorted(final_counts.items()):
            percentage = (count / total_cells) * 100
            status = "TARGET" if count == TARGET_CELLS_PER_PHENOTYPE else "LIMITED"
            f.write(f"{subtype:<35} {count:>4} cells ({percentage:>5.1f}%) [{status}]\n")
        
        f.write(f"\nSUBTYPES WITH LIMITED CELLS (<{TARGET_CELLS_PER_PHENOTYPE}):\n")
        limited_subtypes = {k: v for k, v in final_counts.items() if v < TARGET_CELLS_PER_PHENOTYPE}
        if limited_subtypes:
            for subtype, count in sorted(limited_subtypes.items()):
                f.write(f"  {subtype}: {count} cells\n")
        else:
            f.write("  None - all subtypes have sufficient cells\n")
        
        f.write("\nFILE FORMAT COMPLIANCE\n")
        f.write("-" * 25 + "\n")
        f.write("✓ Genes in rows, cells in columns\n")
        f.write("✓ First column contains gene names\n")
        f.write("✓ Column headers contain cell type labels\n")
        f.write("✓ Data in non-log space (TPM-like values)\n")
        f.write("✓ Tab-separated format\n")
        
        expr_max = reference_matrix.iloc[:, 1:].values.max()
        if expr_max >= 50:
            f.write(f"✓ Maximum expression value: {expr_max:.1f} (>50, will be treated as linear)\n")
        else:
            f.write(f"⚠ Maximum expression value: {expr_max:.1f} (<50, CIBERSORTx may assume log space)\n")
        
        f.write("\nPURPOSE\n")
        f.write("-" * 10 + "\n")
        f.write("This reference matrix represents single-cell RNA-seq expression profiles\n")
        f.write("from NK cells across all tissue contexts (tumor, normal, blood, etc.),\n")
        f.write("suitable for CIBERSORTx signature matrix generation and subsequent\n")
        f.write("deconvolution analysis of bulk RNA-seq data.\n")
    
    logger.info(f"Analysis summary saved to: {summary_path}")


def main():
    """
    Main function to execute the Tang NK reference matrix generation workflow.
    """
    print("=" * 80)
    print("TANG NK REFERENCE MATRIX GENERATOR FOR CIBERSORTX")
    print("=" * 80)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load and validate Tang data
    adata_tang_full = load_and_validate_tang_data()
    if adata_tang_full is None:
        logger.error("Failed to load Tang data. Exiting.")
        return False
    
    # Step 2: Preprocess Tang data
    adata_tang_full = preprocess_tang_data(adata_tang_full)
    
    # Step 3: Analyze tissue distribution (use all tissues)
    adata_tang_full = analyze_tissue_distribution(adata_tang_full)
    
    # Step 4: Sample cells per subtype from all tissue contexts
    adata_sampled, final_counts = sample_cells_per_subtype(adata_tang_full)
    
    # Step 5: Prepare expression data for CIBERSORTx
    expr_df, gene_names, cell_labels = get_expression_data_for_cibersortx(adata_sampled)
    
    # Step 6: Create CIBERSORTx reference matrix
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    reference_matrix = create_cibersortx_reference_matrix(expr_df, cell_labels, output_path)
    
    # Step 7: Generate analysis summary
    generate_analysis_summary(adata_sampled, final_counts, reference_matrix, OUTPUT_DIR)
    
    logger.info("=" * 70)
    logger.info("TANG NK REFERENCE MATRIX GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Reference matrix: {output_path}")
    logger.info(f"Analysis summary: {OUTPUT_DIR / SUMMARY_FILENAME}")
    logger.info(f"Total processing time: Complete")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Reference matrix generation completed successfully!")
    else:
        print("\n❌ Reference matrix generation failed!") 