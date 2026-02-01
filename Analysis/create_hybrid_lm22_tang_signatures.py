#!/usr/bin/env python3
"""
Hybrid LM22-Tang NK Signatures Generator for CIBERSORTx (Tumor-Specific)
========================================================================

This script creates a hybrid signature matrix by:
1. Loading the Tang NK core signatures from TUMOR TISSUE ONLY (filtered from Tang dataset)
2. Loading the LM22 signature matrix
3. Replacing LM22's NK signatures with Tang's tumor-specific high-resolution NK signatures
4. Creating a hybrid matrix optimized for tumor deconvolution

The hybrid approach combines:
- LM22's well-validated immune signatures (T cells, B cells, myeloid, etc.)
- Tang's tumor-specific high-resolution NK signatures (Cytotoxic_NK, Bright_NK, Exhausted_TaNK)

Key Feature: Uses ONLY tumor-infiltrating NK cells from Tang dataset to create signatures
that are specifically optimized for tumor microenvironment deconvolution.

Output: A CIBERSORTx-ready hybrid signature matrix with enhanced tumor-specific NK resolution.
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
from datetime import datetime
import warnings
import logging
from scipy.stats import pearsonr, spearmanr

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
# Input files
TANG_DATA_FILE = "data/processed/comb_CD56_CD16_NK.h5ad"
LM22_FILE = "LM22.txt"

# Output configuration
OUTPUT_DIR = Path("outputs/signature_matrices/Hybrid_LM22_Tang")
HYBRID_MATRIX_FILENAME = "Hybrid_LM22_Tang_NK_Signatures_TumorSpecific.txt"
SUMMARY_FILENAME = "Hybrid_matrix_generation_summary_tumor_specific.txt"

# Tang NK signature mapping (same as from create_tang_reference_matrix.py)
TANG_CELLTYPE_COL = "celltype"
TANG_TISSUE_COL = "meta_tissue_in_paper"  # Primary tissue context column

CORE_SIGNATURE_MAPPING = {
    "Cytotoxic_NK": [
        # All CD56dim NK subtypes (mature, cytotoxic NK cells)
        "CD56dimCD16hi-c1-IL32",           # Cytokine-producing
        "CD56dimCD16hi-c2-CX3CR1",         # Tissue-homing
        "CD56dimCD16hi-c3-ZNF90",          # Mature
        "CD56dimCD16hi-c4-NFKBIA",         # Activated
        "CD56dimCD16hi-c5-MKI67",          # Proliferating
        "CD56dimCD16hi-c7-NR4A3",          # Stimulated
        "CD56dimCD16hi-c8-KLRC2",           # Adaptive (NKG2C+)
        "CD56brightCD16hi"                 # Double-positive transitional
    ],
    "Bright_NK": [
        # All CD56bright NK subtypes (regulatory/immature NK cells)
        "CD56brightCD16lo-c1-GZMH",        # GZMH+ cytotoxic
        "CD56brightCD16lo-c2-IL7R-RGS1lo", # IL7R+ immature
        "CD56brightCD16lo-c3-CCL3",        # CCL3+ inflammatory
        "CD56brightCD16lo-c4-IL7R",        # IL7R+ immature
        "CD56brightCD16lo-c5-CREM",        # CREM+ regulatory
        
    ],
    "Exhausted_TaNK": [
        # Tumor-associated exhausted NK cells
        "CD56dimCD16hi-c6-DNAJB1"          # Stress-response/exhausted
    ]
}

# LM22 NK signatures to replace
LM22_NK_COLUMNS_TO_REMOVE = ["NK cells resting", "NK cells activated"]

# Processing parameters
RANDOM_SEED = 42
MIN_GENES_PER_CELL = 200

def load_tang_data():
    """Load Tang NK dataset and create core signatures."""
    logger.info("=" * 70)
    logger.info("LOADING TANG NK DATASET")
    logger.info("=" * 70)
    
    if not os.path.exists(TANG_DATA_FILE):
        logger.error(f"Tang dataset file not found: {TANG_DATA_FILE}")
        return None
    
    try:
        logger.info(f"Loading Tang combined NK dataset from: {TANG_DATA_FILE}")
        adata_tang = sc.read_h5ad(TANG_DATA_FILE)
        logger.info(f"Successfully loaded Tang dataset. Shape: {adata_tang.shape}")
        
        return adata_tang
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR during Tang data loading: {e}")
        return None

def preprocess_tang_data(adata_tang):
    """Preprocess Tang data for signature generation - PRESERVE RAW COUNTS AND FILTER FOR TUMOR ONLY."""
    logger.info("=" * 70)
    logger.info("PREPROCESSING TANG DATA FOR LINEAR SPACE SIGNATURE GENERATION")
    logger.info("=" * 70)
    
    logger.info(f"Initial shape: {adata_tang.shape}")
    
    # STEP 1: Filter for tumor tissue only
    logger.info("=" * 50)
    logger.info("FILTERING FOR TUMOR TISSUE ONLY")
    logger.info("=" * 50)
    
    if TANG_TISSUE_COL not in adata_tang.obs.columns:
        logger.error(f"Tissue column '{TANG_TISSUE_COL}' not found in dataset!")
        logger.error(f"Available columns: {list(adata_tang.obs.columns)}")
        raise ValueError(f"Required tissue column '{TANG_TISSUE_COL}' missing from dataset")
    
    # Check tissue distribution before filtering
    tissue_counts = adata_tang.obs[TANG_TISSUE_COL].value_counts()
    logger.info("Tissue distribution in original dataset:")
    for tissue, count in tissue_counts.items():
        percentage = (count / len(adata_tang.obs)) * 100
        logger.info(f"  {tissue}: {count:,} cells ({percentage:.1f}%)")
    
    # Filter for tumor tissue only
    tumor_mask = adata_tang.obs[TANG_TISSUE_COL] == "Tumor"
    n_tumor_cells = tumor_mask.sum()
    
    if n_tumor_cells == 0:
        logger.error("No tumor tissue cells found in dataset!")
        raise ValueError("No cells with tissue type 'Tumor' found")
    
    logger.info(f"Filtering to {n_tumor_cells:,} tumor tissue cells...")
    adata_tang = adata_tang[tumor_mask, :].copy()
    logger.info(f"Shape after tumor filtering: {adata_tang.shape}")
    
    # Verify filtering worked
    remaining_tissues = adata_tang.obs[TANG_TISSUE_COL].value_counts()
    logger.info("Tissue distribution after filtering:")
    for tissue, count in remaining_tissues.items():
        logger.info(f"  {tissue}: {count:,} cells")
    
    # Store raw counts BEFORE any processing
    logger.info("Storing raw counts in adata.layers['counts'] for signature generation...")
    adata_tang.layers["counts"] = adata_tang.X.copy()
    
    # Minimal cell filtering only
    logger.info(f"Filtering cells with fewer than {MIN_GENES_PER_CELL} genes...")
    sc.pp.filter_cells(adata_tang, min_genes=MIN_GENES_PER_CELL)
    logger.info(f"Shape after cell filtering: {adata_tang.shape}")
    
    # Check data characteristics but DO NOT log-transform for signature generation
    max_expression = adata_tang.X.max()
    logger.info(f"Data characteristics - Max: {max_expression:.1f}")
    
    if max_expression > 50:
        logger.info("Data appears to be raw counts - PERFECT for linear space signature generation")
        logger.info("Skipping normalization to preserve raw counts for TPM calculation")
    else:
        logger.warning(f"Data may already be normalized (max: {max_expression:.1f})")
        logger.warning("This could affect TPM calculation quality")
    
    # Update the raw counts layer after filtering
    adata_tang.layers["counts"] = adata_tang.X.copy()
    
    return adata_tang

def create_tang_signatures(adata_tang):
    """Create average expression signatures for the 3 Tang NK core types using LINEAR SPACE AVERAGING."""
    logger.info("=" * 70)
    logger.info("CREATING TANG NK SIGNATURES (LINEAR SPACE AVERAGING)")
    logger.info("=" * 70)
    
    # Create mapping from subtypes to core signatures
    subtype_to_core = {}
    for core_sig, subtypes in CORE_SIGNATURE_MAPPING.items():
        for subtype in subtypes:
            subtype_to_core[subtype] = core_sig
    
    # Map each cell to its core signature
    adata_tang.obs['core_signature'] = adata_tang.obs[TANG_CELLTYPE_COL].map(subtype_to_core)
    
    # Validate mapping
    core_sig_counts = adata_tang.obs['core_signature'].value_counts()
    logger.info("Core signature distribution:")
    for core_sig, count in core_sig_counts.items():
        logger.info(f"  {core_sig}: {count:,} cells")
    
    # Get raw count data (BEFORE any log transformation)
    if hasattr(adata_tang, 'layers') and 'counts' in adata_tang.layers:
        logger.info("Using stored raw counts from layers['counts']")
        expr_data = adata_tang.layers['counts']
        gene_names = adata_tang.var_names
    elif hasattr(adata_tang, 'raw') and adata_tang.raw is not None:
        logger.info("Using raw expression data")
        expr_data = adata_tang.raw.X
        gene_names = adata_tang.raw.var_names
    else:
        logger.warning("No raw counts found. Using main .X data (may be log-transformed)")
        expr_data = adata_tang.X
        gene_names = adata_tang.var_names
    
    # Convert to dense if sparse
    if hasattr(expr_data, 'toarray'):
        expr_data = expr_data.toarray()
    
    logger.info(f"Expression data range: {expr_data.min():.3f} to {expr_data.max():.3f}")
    
    # Convert to TPM (Transcripts Per Million) in LINEAR space
    logger.info("Converting to TPM in linear space...")
    # Calculate total counts per cell
    total_counts_per_cell = expr_data.sum(axis=1, keepdims=True)
    # Avoid division by zero
    total_counts_per_cell[total_counts_per_cell == 0] = 1
    # Convert to TPM
    expr_data_tpm = (expr_data / total_counts_per_cell) * 1e6
    
    logger.info(f"TPM data range: {expr_data_tpm.min():.3f} to {expr_data_tpm.max():.3f}")
    
    # Create DataFrame with TPM values
    expression_df = pd.DataFrame(
        expr_data_tpm,
        index=adata_tang.obs_names,
        columns=gene_names
    )
    expression_df['core_signature'] = adata_tang.obs['core_signature'].values
    
    # Remove cells without core signature assignment
    expression_df = expression_df.dropna(subset=['core_signature'])
    logger.info(f"Cells after filtering: {expression_df.shape[0]:,}")
    
    # Group by core signature and calculate ARITHMETIC mean in LINEAR space
    logger.info("Calculating arithmetic mean expression per core signature in LINEAR space...")
    numeric_cols = [col for col in expression_df.columns if col != 'core_signature']
    tang_signatures = expression_df.groupby('core_signature')[numeric_cols].mean()
    
    # Transpose to get genes as rows, signatures as columns (required format)
    tang_signatures = tang_signatures.T
    
    logger.info(f"Tang signatures created. Shape: {tang_signatures.shape}")
    logger.info(f"Signatures: {list(tang_signatures.columns)}")
    
    # Report final signature statistics
    for col in tang_signatures.columns:
        sig_values = tang_signatures[col].values
        logger.info(f"  {col}: Min={sig_values.min():.3f}, Max={sig_values.max():.3f}, Median={np.median(sig_values):.3f}")
    
    return tang_signatures

def convert_to_linear_scale(signatures_df, data_type="log1p"):
    """Convert log-transformed data back to linear scale."""
    logger.info("=" * 70)
    logger.info(f"CONVERTING {data_type.upper()} DATA TO LINEAR SCALE")
    logger.info("=" * 70)
    
    max_val = signatures_df.values.max()
    logger.info(f"Maximum value before conversion: {max_val:.3f}")
    
    if data_type == "log1p":
        # Most common: log1p(x) -> expm1
        signatures_linear = np.expm1(signatures_df)
        logger.info("Applied expm1 transformation (reverse of log1p)")
    elif data_type == "log2":
        # log2(x+1) -> (2**x) - 1
        signatures_linear = (2**signatures_df) - 1
        logger.info("Applied 2^x - 1 transformation (reverse of log2(x+1))")
    else:
        logger.warning(f"Unknown data type: {data_type}. Assuming log1p.")
        signatures_linear = np.expm1(signatures_df)
    
    max_val_linear = signatures_linear.values.max()
    logger.info(f"Maximum value after conversion: {max_val_linear:.3f}")
    
    return signatures_linear

def load_and_prepare_lm22():
    """Load LM22 matrix and prepare it for merging."""
    logger.info("=" * 70)
    logger.info("LOADING AND PREPARING LM22 MATRIX")
    logger.info("=" * 70)
    
    if not os.path.exists(LM22_FILE):
        logger.error(f"LM22 file not found: {LM22_FILE}")
        return None, None
    
    try:
        # Load LM22
        logger.info(f"Loading LM22 matrix from: {LM22_FILE}")
        lm22 = pd.read_csv(LM22_FILE, sep='\t', index_col=0)
        logger.info(f"LM22 loaded. Shape: {lm22.shape}")
        logger.info(f"Cell types in LM22: {list(lm22.columns)}")
        
        # Check if LM22 is in log scale
        max_val = lm22.values.max()
        logger.info(f"LM22 maximum value: {max_val:.3f}")
        
        if max_val < 20:  # Likely log2 scale
            logger.info("LM22 appears to be in log2 scale. Converting to linear...")
            lm22_linear = 2**lm22
        else:
            logger.info("LM22 appears to be in linear scale already.")
            lm22_linear = lm22.copy()
        
        # Extract original NK signatures for scaling reference (BEFORE removing them)
        logger.info("Extracting original NK signatures for scaling reference...")
        nk_columns_present = [col for col in LM22_NK_COLUMNS_TO_REMOVE if col in lm22_linear.columns]
        if nk_columns_present:
            lm22_nk_original = lm22_linear[nk_columns_present].copy()
            logger.info(f"Extracted NK signatures: {nk_columns_present}")
            nk_median = np.median(lm22_nk_original.values)
            logger.info(f"Original LM22 NK signatures median: {nk_median:.3f}")
        else:
            logger.warning(f"NK columns not found in LM22: {LM22_NK_COLUMNS_TO_REMOVE}")
            lm22_nk_original = None
        
        # Remove old NK signatures
        logger.info("Removing original NK signatures from LM22...")
        if nk_columns_present:
            lm22_modified = lm22_linear.drop(columns=nk_columns_present)
            logger.info(f"Removed columns: {nk_columns_present}")
        else:
            lm22_modified = lm22_linear.copy()
        
        logger.info(f"LM22 prepared. Final shape: {lm22_modified.shape}")
        logger.info(f"Remaining cell types: {list(lm22_modified.columns)}")
        
        return lm22_modified, lm22_nk_original
        
    except Exception as e:
        logger.error(f"Error loading/preparing LM22: {e}")
        return None, None

def harmonize_signature_scales(tang_signatures, lm22_nk_signatures):
    """
    Harmonize Tang NK signatures to match LM22 NK signatures using gene-wise robust scaling.
    
    This approach scales each gene individually using robust statistics (median and MAD)
    to preserve gene-specific expression patterns while matching the LM22 scale.
    This is more biologically sound than uniform scaling across all genes.
    """
    logger.info("=" * 70)
    logger.info("HARMONIZING TANG NK SIGNATURES USING GENE-WISE ROBUST SCALING")
    logger.info("=" * 70)
    
    # Find common genes between Tang and LM22 NK signatures
    common_genes = tang_signatures.index.intersection(lm22_nk_signatures.index)
    logger.info(f"Genes available for harmonization: {len(common_genes):,}")
    
    if len(common_genes) < 100:
        logger.warning(f"Limited gene overlap ({len(common_genes)}) may affect harmonization quality")
    
    # Initialize harmonized Tang signatures
    tang_harmonized = tang_signatures.copy()
    
    # Track harmonization statistics
    scale_factors = []
    genes_processed = 0
    genes_skipped = 0
    
    logger.info("Performing gene-wise harmonization...")
    
    for gene in common_genes:
        tang_gene_values = tang_signatures.loc[gene].values
        lm22_gene_values = lm22_nk_signatures.loc[gene].values
        
        # Calculate robust statistics for both signatures
        tang_median = np.median(tang_gene_values)
        lm22_median = np.median(lm22_gene_values)
        
        # Calculate Median Absolute Deviation (MAD) for scale
        tang_mad = np.median(np.abs(tang_gene_values - tang_median))
        lm22_mad = np.median(np.abs(lm22_gene_values - lm22_median))
        
        # Skip genes with zero or very low expression in Tang
        if tang_median < 1e-6 or tang_mad < 1e-6:
            # For low-expression genes, just match the LM22 median
            tang_harmonized.loc[gene] = lm22_median
            genes_skipped += 1
            continue
        
        # Calculate scale factor and location adjustment
        scale_factor = lm22_mad / tang_mad if tang_mad > 0 else 1.0
        location_shift = lm22_median - (tang_median * scale_factor)
        
        # Apply gene-wise transformation: new_value = (old_value * scale) + shift
        transformed_values = tang_gene_values * scale_factor + location_shift
        
        # CRITICAL: Ensure no negative values (would cause NaNs in log transformation)
        if transformed_values.min() < 0:
            logger.warning(f"Gene {gene}: Negative values detected after transformation. Applying safety correction.")
            # Shift all values to ensure minimum is small positive value
            min_val = transformed_values.min()
            safety_shift = abs(min_val) + 1e-6  # Small epsilon
            transformed_values = transformed_values + safety_shift
        
        tang_harmonized.loc[gene] = transformed_values
        
        scale_factors.append(scale_factor)
        genes_processed += 1
    
    logger.info(f"Genes successfully harmonized: {genes_processed:,}")
    logger.info(f"Genes with low expression (set to LM22 median): {genes_skipped:,}")
    
    # Calculate overall harmonization statistics
    if scale_factors:
        median_scale_factor = np.median(scale_factors)
        scale_factor_iqr = np.percentile(scale_factors, 75) - np.percentile(scale_factors, 25)
        logger.info(f"Median gene-wise scale factor: {median_scale_factor:.3f}")
        logger.info(f"Scale factor IQR: {scale_factor_iqr:.3f}")
    else:
        median_scale_factor = 1.0
        logger.warning("No scale factors calculated - harmonization may have failed")
    
    # CRITICAL: Validate data integrity for CIBERSORTx compatibility
    data_validation = validate_cibersortx_compatibility(tang_harmonized)
    
    # Validate harmonization quality
    validation_results = validate_harmonization(tang_signatures, tang_harmonized, lm22_nk_signatures, common_genes)
    
    return tang_harmonized, median_scale_factor

def validate_cibersortx_compatibility(harmonized_signatures):
    """
    Validate that harmonized signatures are compatible with CIBERSORTx requirements.
    
    Checks for:
    - No negative values (would cause NaNs in log transformation)
    - No NaN or infinite values
    - All values are finite numbers
    - Minimum expression levels are reasonable
    """
    logger.info("=" * 70)
    logger.info("VALIDATING CIBERSORTX COMPATIBILITY (CRITICAL)")
    logger.info("=" * 70)
    
    issues_found = []
    
    # Check for negative values
    negative_mask = harmonized_signatures.values < 0
    n_negative = negative_mask.sum()
    if n_negative > 0:
        issues_found.append(f"NEGATIVE VALUES: {n_negative} negative values found")
        logger.error(f"❌ Found {n_negative} negative values - will cause NaNs in log transformation!")
        
        # Find genes with negative values
        negative_genes = []
        for gene in harmonized_signatures.index:
            if (harmonized_signatures.loc[gene] < 0).any():
                negative_genes.append(gene)
        logger.error(f"Genes with negative values: {negative_genes[:10]}")  # Show first 10
    else:
        logger.info("✅ No negative values found")
    
    # Check for NaN values
    nan_mask = np.isnan(harmonized_signatures.values)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        issues_found.append(f"NaN VALUES: {n_nan} NaN values found")
        logger.error(f"❌ Found {n_nan} NaN values!")
    else:
        logger.info("✅ No NaN values found")
    
    # Check for infinite values
    inf_mask = np.isinf(harmonized_signatures.values)
    n_inf = inf_mask.sum()
    if n_inf > 0:
        issues_found.append(f"INFINITE VALUES: {n_inf} infinite values found")
        logger.error(f"❌ Found {n_inf} infinite values!")
    else:
        logger.info("✅ No infinite values found")
    
    # Check minimum values
    min_val = harmonized_signatures.values.min()
    if min_val < 1e-10:
        logger.warning(f"⚠️ Very small minimum value: {min_val:.2e} (may cause numerical issues)")
    else:
        logger.info(f"✅ Minimum value reasonable: {min_val:.6f}")
    
    # Check maximum values for extreme outliers
    max_val = harmonized_signatures.values.max()
    median_val = np.median(harmonized_signatures.values)
    if max_val > median_val * 1000:
        logger.warning(f"⚠️ Extreme maximum value: {max_val:.2e} (median: {median_val:.2e})")
    else:
        logger.info(f"✅ Maximum value reasonable: {max_val:.3f}")
    
    # Overall assessment
    if not issues_found:
        logger.info("🎉 ALL CIBERSORTX COMPATIBILITY CHECKS PASSED!")
        logger.info("✅ Signatures are ready for CIBERSORTx deconvolution")
    else:
        logger.error("💥 CIBERSORTX COMPATIBILITY ISSUES DETECTED:")
        for issue in issues_found:
            logger.error(f"   - {issue}")
        logger.error("❌ Signatures may fail in CIBERSORTx - manual review required!")
    
    return len(issues_found) == 0

def validate_harmonization(tang_original, tang_harmonized, lm22_nk_signatures, common_genes):
    """
    Comprehensive validation of harmonization quality.
    
    Checks:
    1. Scale alignment between Tang and LM22
    2. Preservation of gene expression patterns
    3. Distribution similarity
    4. Correlation structure preservation
    """
    logger.info("=" * 70)
    logger.info("VALIDATING HARMONIZATION QUALITY")
    logger.info("=" * 70)
    
    validation_results = {}
    
    # 1. Scale alignment validation
    logger.info("1. Scale Alignment Analysis:")
    
    # Compare medians across common genes
    tang_orig_medians = [np.median(tang_original.loc[gene].values) for gene in common_genes]
    tang_harm_medians = [np.median(tang_harmonized.loc[gene].values) for gene in common_genes]
    lm22_medians = [np.median(lm22_nk_signatures.loc[gene].values) for gene in common_genes]
    
    # Calculate correlation of medians
    # Original Tang vs LM22 correlation
    orig_corr, orig_p = pearsonr(tang_orig_medians, lm22_medians)
    # Harmonized Tang vs LM22 correlation
    harm_corr, harm_p = pearsonr(tang_harm_medians, lm22_medians)
    
    logger.info(f"   Median correlation (Original Tang vs LM22): r={orig_corr:.3f}, p={orig_p:.2e}")
    logger.info(f"   Median correlation (Harmonized Tang vs LM22): r={harm_corr:.3f}, p={harm_p:.2e}")
    
    validation_results['original_correlation'] = orig_corr
    validation_results['harmonized_correlation'] = harm_corr
    validation_results['correlation_improvement'] = harm_corr - orig_corr
    
    # 2. Pattern preservation validation
    logger.info("2. Gene Expression Pattern Preservation:")
    
    # Check if relative ordering of genes is preserved
    pattern_correlations = []
    for col in tang_original.columns:
        orig_pattern = tang_original[col].loc[common_genes].values
        harm_pattern = tang_harmonized[col].loc[common_genes].values
        
        # Calculate rank correlation to check pattern preservation
        rank_corr, _ = spearmanr(orig_pattern, harm_pattern)
        pattern_correlations.append(rank_corr)
    
    avg_pattern_preservation = np.mean(pattern_correlations)
    min_pattern_preservation = np.min(pattern_correlations)
    
    logger.info(f"   Average pattern preservation (rank correlation): {avg_pattern_preservation:.3f}")
    logger.info(f"   Minimum pattern preservation: {min_pattern_preservation:.3f}")
    
    validation_results['avg_pattern_preservation'] = avg_pattern_preservation
    validation_results['min_pattern_preservation'] = min_pattern_preservation
    
    # 3. Distribution similarity
    logger.info("3. Distribution Similarity Analysis:")
    
    # Compare overall distributions
    tang_orig_all = tang_original.loc[common_genes].values.flatten()
    tang_harm_all = tang_harmonized.loc[common_genes].values.flatten()
    lm22_all = lm22_nk_signatures.loc[common_genes].values.flatten()
    
    # Filter out zeros for meaningful comparison
    tang_orig_nonzero = tang_orig_all[tang_orig_all > 0]
    tang_harm_nonzero = tang_harm_all[tang_harm_all > 0]
    lm22_nonzero = lm22_all[lm22_all > 0]
    
    # Calculate distribution statistics
    orig_vs_lm22_median_ratio = np.median(tang_orig_nonzero) / np.median(lm22_nonzero)
    harm_vs_lm22_median_ratio = np.median(tang_harm_nonzero) / np.median(lm22_nonzero)
    
    logger.info(f"   Original Tang/LM22 median ratio: {orig_vs_lm22_median_ratio:.3f}")
    logger.info(f"   Harmonized Tang/LM22 median ratio: {harm_vs_lm22_median_ratio:.3f}")
    
    validation_results['original_median_ratio'] = orig_vs_lm22_median_ratio
    validation_results['harmonized_median_ratio'] = harm_vs_lm22_median_ratio
    
    # 4. Overall quality assessment
    logger.info("4. Overall Quality Assessment:")
    
    # Define quality metrics
    correlation_good = harm_corr > 0.7
    pattern_preserved = avg_pattern_preservation > 0.8
    scale_matched = 0.5 <= harm_vs_lm22_median_ratio <= 2.0
    
    quality_score = sum([correlation_good, pattern_preserved, scale_matched]) / 3
    
    if quality_score >= 0.67:
        quality_status = "✓ GOOD"
        logger.info(f"   Overall Quality: {quality_status} (Score: {quality_score:.2f})")
    elif quality_score >= 0.33:
        quality_status = "⚠ MODERATE"
        logger.info(f"   Overall Quality: {quality_status} (Score: {quality_score:.2f})")
    else:
        quality_status = "✗ POOR"
        logger.info(f"   Overall Quality: {quality_status} (Score: {quality_score:.2f})")
    
    validation_results['quality_score'] = quality_score
    validation_results['quality_status'] = quality_status
    
    # Summary recommendations
    logger.info("5. Validation Summary:")
    if harm_corr > orig_corr:
        logger.info("   ✓ Harmonization improved scale alignment")
    else:
        logger.warning("   ⚠ Harmonization did not improve scale alignment")
    
    if avg_pattern_preservation > 0.8:
        logger.info("   ✓ Gene expression patterns well preserved")
    else:
        logger.warning("   ⚠ Some gene expression patterns may be distorted")
    
    if scale_matched:
        logger.info("   ✓ Overall scale successfully matched to LM22")
    else:
        logger.warning("   ⚠ Overall scale may still differ from LM22")
    
    return validation_results

def create_hybrid_matrix(lm22_prepared, tang_signatures):
    """Merge LM22 and Tang signatures into hybrid matrix."""
    logger.info("=" * 70)
    logger.info("CREATING HYBRID SIGNATURE MATRIX")
    logger.info("=" * 70)
    
    # Get gene lists
    lm22_genes = set(lm22_prepared.index)
    tang_genes = set(tang_signatures.index)
    
    logger.info(f"LM22 genes: {len(lm22_genes):,}")
    logger.info(f"Tang genes: {len(tang_genes):,}")
    
    # Find common genes
    common_genes = list(lm22_genes & tang_genes)
    logger.info(f"Common genes: {len(common_genes):,}")
    
    if len(common_genes) < 1000:
        logger.warning(f"Only {len(common_genes)} common genes found. This may affect deconvolution quality.")
    
    # Filter both matrices to common genes
    lm22_final = lm22_prepared.loc[common_genes].copy()
    tang_final = tang_signatures.loc[common_genes].copy()
    
    # Sort genes alphabetically for consistency
    common_genes_sorted = sorted(common_genes)
    lm22_final = lm22_final.loc[common_genes_sorted]
    tang_final = tang_final.loc[common_genes_sorted]
    
    # Merge matrices
    hybrid_matrix = pd.concat([lm22_final, tang_final], axis=1)
    
    logger.info(f"Hybrid matrix created. Shape: {hybrid_matrix.shape}")
    logger.info(f"Total signatures: {hybrid_matrix.shape[1]}")
    logger.info(f"Final signatures: {list(hybrid_matrix.columns)}")
    
    return hybrid_matrix

def save_hybrid_matrix(hybrid_matrix, output_dir):
    """Save the hybrid matrix in CIBERSORTx format."""
    logger.info("=" * 70)
    logger.info("SAVING HYBRID SIGNATURE MATRIX")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare matrix for CIBERSORTx (add Gene column as first column)
    cibersort_matrix = hybrid_matrix.copy()
    cibersort_matrix.insert(0, "Gene", cibersort_matrix.index)
    
    # Save matrix
    output_path = output_dir / HYBRID_MATRIX_FILENAME
    logger.info(f"Saving hybrid matrix to: {output_path}")
    cibersort_matrix.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    
    # Verify file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"File saved successfully. Size: {file_size_mb:.1f} MB")
    
    return output_path

def generate_summary(hybrid_matrix, tang_signatures, lm22_prepared, output_dir):
    """Generate comprehensive summary of hybrid matrix creation."""
    logger.info("=" * 70)
    logger.info("GENERATING ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    summary_path = output_dir / SUMMARY_FILENAME
    
    with open(summary_path, 'w') as f:
        f.write("Hybrid LM22-Tang NK Signatures Matrix Generation Summary (Tumor-Specific)\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HYBRID APPROACH OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write("This hybrid matrix combines:\n")
        f.write("- LM22's well-validated immune signatures (T cells, B cells, myeloid, etc.)\n")
        f.write("- Tang's TUMOR-SPECIFIC high-resolution NK signatures (Cytotoxic_NK, Bright_NK, Exhausted_TaNK)\n\n")
        
        f.write("KEY FEATURE: TUMOR-SPECIFIC NK SIGNATURES\n")
        f.write("-" * 45 + "\n")
        f.write("Tang NK signatures were derived from TUMOR TISSUE ONLY:\n")
        f.write("- Filtered Tang dataset to include only cells with tissue context = 'Tumor'\n")
        f.write("- This creates signatures optimized for tumor microenvironment deconvolution\n")
        f.write("- Signatures reflect the specific characteristics of tumor-infiltrating NK cells\n")
        f.write("- Particularly important for detecting exhausted TaNK (Tumor-associated NK) cells\n\n")
        
        f.write("ORIGINAL MATRICES\n")
        f.write("-" * 20 + "\n")
        f.write(f"LM22 original signatures: {lm22_prepared.shape[1] + len(LM22_NK_COLUMNS_TO_REMOVE)}\n")
        f.write(f"LM22 after NK removal: {lm22_prepared.shape[1]}\n")
        f.write(f"Tang tumor-specific NK signatures added: {tang_signatures.shape[1]}\n")
        f.write(f"Final hybrid signatures: {hybrid_matrix.shape[1]}\n\n")
        
        f.write("SIGNATURE COMPOSITION\n")
        f.write("-" * 25 + "\n")
        f.write("LM22-derived signatures:\n")
        for col in lm22_prepared.columns:
            f.write(f"  {col}\n")
        f.write("\nTang tumor-specific NK signatures:\n")
        for col in tang_signatures.columns:
            f.write(f"  {col}\n")
        f.write("\n")
        
        f.write("TECHNICAL DETAILS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Common genes used: {hybrid_matrix.shape[0]:,}\n")
        f.write(f"Data scale: Linear (converted from log)\n")
        f.write(f"Scale harmonization: Gene-wise robust scaling (median + MAD)\n")
        f.write(f"  - Each gene scaled individually using robust statistics\n")
        f.write(f"  - Preserves gene-specific expression patterns\n")
        f.write(f"  - More biologically sound than uniform scaling\n")
        f.write(f"Format: CIBERSORTx-compatible (tab-separated)\n")
        f.write(f"Removed from LM22: {LM22_NK_COLUMNS_TO_REMOVE}\n")
        f.write(f"Tissue filtering: Only tumor tissue cells used from Tang dataset\n")
        f.write(f"Tissue column used: {TANG_TISSUE_COL}\n\n")
        
        f.write("HARMONIZATION METHOD\n")
        f.write("-" * 25 + "\n")
        f.write("Gene-wise robust scaling approach:\n")
        f.write("1. For each gene, calculate median and MAD (Median Absolute Deviation)\n")
        f.write("2. Scale factor = LM22_MAD / Tang_MAD\n")
        f.write("3. Location shift = LM22_median - (Tang_median * scale_factor)\n")
        f.write("4. Final value = (Tang_value * scale_factor) + location_shift\n")
        f.write("This preserves gene-specific expression patterns while matching LM22 scale.\n\n")
        
        f.write("TANG NK SIGNATURE DETAILS\n")
        f.write("-" * 30 + "\n")
        for core_sig, subtypes in CORE_SIGNATURE_MAPPING.items():
            f.write(f"{core_sig} (tumor-specific):\n")
            for subtype in subtypes:
                f.write(f"  {subtype}\n")
            f.write("\n")
        
        f.write("USAGE\n")
        f.write("-" * 10 + "\n")
        f.write("Use this hybrid matrix with CIBERSORTx for enhanced tumor-specific NK cell deconvolution.\n")
        f.write("The matrix provides:\n")
        f.write("- Standard immune cell resolution from LM22\n")
        f.write("- High-resolution tumor-specific NK cell subtype analysis\n")
        f.write("- Optimized detection of tumor-infiltrating NK cell states\n")
        f.write("- Enhanced ability to detect exhausted TaNK cells in tumor samples\n")
        f.write("- Signatures specifically trained on tumor microenvironment NK cells\n")
    
    logger.info(f"Analysis summary saved to: {summary_path}")

def main():
    """Main function to create hybrid LM22-Tang NK signatures matrix."""
    print("=" * 80)
    print("HYBRID LM22-TANG NK SIGNATURES GENERATOR (TUMOR-SPECIFIC)")
    print("=" * 80)
    print()
    print("Creating hybrid signature matrix by combining:")
    print("• LM22 immune signatures (minus original NK)")
    print("• Tang tumor-specific high-resolution NK signatures (3 types)")
    print("• Using LINEAR SPACE averaging for proper arithmetic means")
    print("• Scaling Tang NK signatures to match LM22 NK signature scale")
    print("• FILTERING: Using ONLY tumor tissue cells from Tang dataset")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load and process Tang data
    adata_tang = load_tang_data()
    if adata_tang is None:
        logger.error("Failed to load Tang data. Exiting.")
        return False
    
    # Step 2: Preprocess Tang data (includes tumor filtering)
    adata_tang = preprocess_tang_data(adata_tang)
    
    # Step 3: Create Tang signatures (already in linear TPM space)
    tang_signatures_linear = create_tang_signatures(adata_tang)
    
    # Step 4: Load and prepare LM22
    lm22_prepared, lm22_nk_original = load_and_prepare_lm22()
    if lm22_prepared is None or lm22_nk_original is None:
        logger.error("Failed to load/prepare LM22. Exiting.")
        return False
    
    # Step 5: Harmonize scales using gene-wise robust scaling
    tang_signatures_harmonized, median_scale_factor = harmonize_signature_scales(tang_signatures_linear, lm22_nk_original)
    logger.info(f"Applied median gene-wise scale factor: {median_scale_factor:.1f}x to Tang signatures")
    
    # Step 6: Create hybrid matrix
    hybrid_matrix = create_hybrid_matrix(lm22_prepared, tang_signatures_harmonized)
    
    # Step 6.5: Final validation of hybrid matrix
    logger.info("Performing final validation of hybrid matrix...")
    final_validation = validate_cibersortx_compatibility(hybrid_matrix)
    if not final_validation:
        logger.error("CRITICAL: Hybrid matrix failed CIBERSORTx compatibility checks!")
        return False
    
    # Step 7: Save hybrid matrix
    output_path = save_hybrid_matrix(hybrid_matrix, OUTPUT_DIR)
    
    # Step 8: Generate summary
    generate_summary(hybrid_matrix, tang_signatures_harmonized, lm22_prepared, OUTPUT_DIR)
    
    logger.info("=" * 70)
    logger.info("HYBRID MATRIX GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Hybrid matrix: {output_path}")
    logger.info(f"Summary: {OUTPUT_DIR / SUMMARY_FILENAME}")
    logger.info(f"Total signatures: {hybrid_matrix.shape[1]} ({lm22_prepared.shape[1]} LM22 + {tang_signatures_harmonized.shape[1]} Tang tumor-specific)")
    logger.info(f"Tang signatures harmonized using gene-wise robust scaling (median factor: {median_scale_factor:.1f}x)")
    logger.info("IMPORTANT: Tang NK signatures derived from TUMOR TISSUE ONLY")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Hybrid LM22-Tang NK signatures matrix created successfully!")
        print("🔬 Ready for enhanced CIBERSORTx deconvolution with tumor-specific high-resolution NK analysis")
        print("🎯 NK signatures optimized for tumor microenvironment deconvolution")
    else:
        print("\n❌ Hybrid matrix generation failed!") 