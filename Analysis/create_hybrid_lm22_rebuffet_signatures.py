#!/usr/bin/env python3
"""
Hybrid LM22-Rebuffet NK Signatures Generator for CIBERSORTx
===========================================================

This script creates a hybrid signature matrix by:
1. Loading the Rebuffet NK subtypes from PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad
2. Loading the LM22 signature matrix
3. Replacing LM22's NK signatures with Rebuffet's core NK signatures
4. Creating a hybrid matrix optimized for tumor deconvolution

The hybrid approach combines:
- LM22's well-validated immune signatures (T cells, B cells, myeloid, etc.)
- Rebuffet's 3 core NK signatures (NK2, NK1C, NK3) representing key developmental states

Output: A CIBERSORTx-ready hybrid signature matrix with enhanced NK resolution.
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
REBUFFET_DATA_FILE = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
LM22_FILE = "LM22.txt"

# Output configuration
OUTPUT_DIR = Path("outputs/signature_matrices/Hybrid_LM22_Rebuffet")
HYBRID_MATRIX_FILENAME = "Hybrid_LM22_Rebuffet_NK_Signatures.txt"
SUMMARY_FILENAME = "Hybrid_matrix_generation_summary.txt"

# Rebuffet NK subtype mapping - using all 6 main subtypes
REBUFFET_CELLTYPE_COL = "ident"  # Original subtype column from Rebuffet data

# Rebuffet NK Subtypes (functional ordering: immature/regulatory → mature cytotoxic → adaptive/terminal)
# Using 3 core subtypes to avoid redundancy between similar NK1A/B/C subtypes
REBUFFET_SUBTYPES_ORDERED = [
    "NK2",      # Immature/regulatory NK cells (CD56bright-like)
    "NK1C",     # Mature cytotoxic NK cells (most terminal of NK1 series)
    "NK3",      # Adaptive-like NK cells (high NKG2C expression)
]

# LM22 NK signatures to replace
LM22_NK_COLUMNS_TO_REMOVE = ["NK cells resting", "NK cells activated"]

# Processing parameters
RANDOM_SEED = 42
MIN_GENES_PER_CELL = 200

def load_rebuffet_data():
    """Load Rebuffet NK dataset and prepare for signature generation."""
    logger.info("=" * 70)
    logger.info("LOADING REBUFFET NK DATASET")
    logger.info("=" * 70)
    
    if not os.path.exists(REBUFFET_DATA_FILE):
        logger.error(f"Rebuffet dataset file not found: {REBUFFET_DATA_FILE}")
        return None
    
    try:
        logger.info(f"Loading Rebuffet NK dataset from: {REBUFFET_DATA_FILE}")
        adata_rebuffet = sc.read_h5ad(REBUFFET_DATA_FILE)
        logger.info(f"Successfully loaded Rebuffet dataset. Shape: {adata_rebuffet.shape}")
        
        # Check for subtype column
        if REBUFFET_CELLTYPE_COL not in adata_rebuffet.obs.columns:
            logger.error(f"Subtype column '{REBUFFET_CELLTYPE_COL}' not found in Rebuffet data")
            logger.info(f"Available columns: {adata_rebuffet.obs.columns.tolist()}")
            return None
        
        # Check subtype distribution
        subtype_counts = adata_rebuffet.obs[REBUFFET_CELLTYPE_COL].value_counts()
        logger.info("Rebuffet subtype distribution:")
        for subtype, count in subtype_counts.items():
            if subtype in REBUFFET_SUBTYPES_ORDERED:
                logger.info(f"  {subtype}: {count:,} cells")
        
        return adata_rebuffet
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR during Rebuffet data loading: {e}")
        return None

def preprocess_rebuffet_data(adata_rebuffet):
    """Preprocess Rebuffet data for signature generation - PRESERVE RAW COUNTS."""
    logger.info("=" * 70)
    logger.info("PREPROCESSING REBUFFET DATA FOR LINEAR SPACE SIGNATURE GENERATION")
    logger.info("=" * 70)
    
    logger.info(f"Initial shape: {adata_rebuffet.shape}")
    
    # Filter for valid subtypes only
    valid_subtype_mask = adata_rebuffet.obs[REBUFFET_CELLTYPE_COL].isin(REBUFFET_SUBTYPES_ORDERED)
    adata_rebuffet = adata_rebuffet[valid_subtype_mask, :].copy()
    logger.info(f"Shape after filtering for valid subtypes: {adata_rebuffet.shape}")
    
    # Check if data has TPM layer (Rebuffet data is TPM-normalized)
    if hasattr(adata_rebuffet, 'layers') and 'tpm' in adata_rebuffet.layers:
        logger.info("Found TPM layer in Rebuffet data - using for signature generation")
        raw_data = adata_rebuffet.layers['tpm']
    else:
        # Rebuffet data is TPM-normalized by default, need to back-transform from log
        max_expression = adata_rebuffet.X.max()
        logger.info(f"Data characteristics - Max: {max_expression:.1f}")
        
        if max_expression < 20:  # Likely log-transformed TPM
            logger.info("Data appears to be log-transformed TPM - back-transforming to linear TPM")
            raw_data = np.expm1(adata_rebuffet.X)  # Reverse log1p transformation
        else:
            logger.info("Data appears to be linear TPM - using directly")
            raw_data = adata_rebuffet.X
    
    # Store raw TPM data for signature calculation
    adata_rebuffet.layers["tpm_linear"] = raw_data.copy()
    
    # Minimal cell filtering only
    logger.info(f"Filtering cells with fewer than {MIN_GENES_PER_CELL} genes...")
    sc.pp.filter_cells(adata_rebuffet, min_genes=MIN_GENES_PER_CELL)
    logger.info(f"Shape after cell filtering: {adata_rebuffet.shape}")
    
    return adata_rebuffet

def create_rebuffet_signatures(adata_rebuffet):
    """Create average expression signatures for the 3 core Rebuffet NK subtypes using LINEAR SPACE AVERAGING."""
    logger.info("=" * 70)
    logger.info("CREATING REBUFFET NK SIGNATURES (LINEAR SPACE AVERAGING)")
    logger.info("=" * 70)
    
    # Validate subtype distribution
    subtype_counts = adata_rebuffet.obs[REBUFFET_CELLTYPE_COL].value_counts()
    logger.info("Rebuffet subtype distribution for signature generation:")
    for subtype in REBUFFET_SUBTYPES_ORDERED:
        count = subtype_counts.get(subtype, 0)
        logger.info(f"  {subtype}: {count:,} cells")
        if count == 0:
            logger.warning(f"  WARNING: No cells found for {subtype}")
    
    # Get TPM data in linear space
    if hasattr(adata_rebuffet, 'layers') and 'tpm_linear' in adata_rebuffet.layers:
        logger.info("Using stored linear TPM counts from layers['tpm_linear']")
        expr_data = adata_rebuffet.layers['tpm_linear']
        gene_names = adata_rebuffet.var_names
    else:
        logger.warning("No linear TPM counts found. Using main .X data")
        expr_data = adata_rebuffet.X
        gene_names = adata_rebuffet.var_names
    
    # Convert to dense if sparse
    if hasattr(expr_data, 'toarray'):
        expr_data = expr_data.toarray()
    
    logger.info(f"Expression data range: {expr_data.min():.3f} to {expr_data.max():.3f}")
    
    # Create DataFrame with TPM values
    expression_df = pd.DataFrame(
        expr_data,
        index=adata_rebuffet.obs_names,
        columns=gene_names
    )
    expression_df[REBUFFET_CELLTYPE_COL] = adata_rebuffet.obs[REBUFFET_CELLTYPE_COL].values
    
    # Remove cells without valid subtype assignment
    valid_subtypes_mask = expression_df[REBUFFET_CELLTYPE_COL].isin(REBUFFET_SUBTYPES_ORDERED)
    expression_df = expression_df[valid_subtypes_mask]
    logger.info(f"Cells after filtering for valid subtypes: {expression_df.shape[0]:,}")
    
    # Group by subtype and calculate ARITHMETIC mean in LINEAR space
    logger.info("Calculating arithmetic mean expression per Rebuffet subtype in LINEAR space...")
    numeric_cols = [col for col in expression_df.columns if col != REBUFFET_CELLTYPE_COL]
    rebuffet_signatures = expression_df.groupby(REBUFFET_CELLTYPE_COL)[numeric_cols].mean()
    
    # Reorder to match our functional ordering
    rebuffet_signatures = rebuffet_signatures.reindex(REBUFFET_SUBTYPES_ORDERED)
    
    # Transpose to get genes as rows, signatures as columns (required format)
    rebuffet_signatures = rebuffet_signatures.T
    
    logger.info(f"Rebuffet signatures created. Shape: {rebuffet_signatures.shape}")
    logger.info(f"Signatures: {list(rebuffet_signatures.columns)}")
    
    # Report final signature statistics
    for col in rebuffet_signatures.columns:
        if col in rebuffet_signatures.columns:
            sig_values = rebuffet_signatures[col].values
            logger.info(f"  {col}: Min={sig_values.min():.3f}, Max={sig_values.max():.3f}, Median={np.median(sig_values):.3f}")
    
    return rebuffet_signatures

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

def harmonize_signature_scales(rebuffet_signatures, lm22_nk_signatures):
    """
    Harmonize Rebuffet NK signatures to match LM22 NK signatures using gene-wise robust scaling.
    
    This approach scales each gene individually using robust statistics (median and MAD)
    to preserve gene-specific expression patterns while matching the LM22 scale.
    This is more biologically sound than uniform scaling across all genes.
    """
    logger.info("=" * 70)
    logger.info("HARMONIZING REBUFFET NK SIGNATURES USING GENE-WISE ROBUST SCALING")
    logger.info("=" * 70)
    
    # Find common genes between Rebuffet and LM22 NK signatures
    common_genes = rebuffet_signatures.index.intersection(lm22_nk_signatures.index)
    logger.info(f"Genes available for harmonization: {len(common_genes):,}")
    
    if len(common_genes) < 100:
        logger.warning(f"Limited gene overlap ({len(common_genes)}) may affect harmonization quality")
    
    # Initialize harmonized Rebuffet signatures
    rebuffet_harmonized = rebuffet_signatures.copy()
    
    # Track harmonization statistics
    scale_factors = []
    genes_processed = 0
    genes_skipped = 0
    
    logger.info("Performing gene-wise harmonization...")
    
    for gene in common_genes:
        rebuffet_gene_values = rebuffet_signatures.loc[gene].values
        lm22_gene_values = lm22_nk_signatures.loc[gene].values
        
        # Calculate robust statistics for both signatures
        rebuffet_median = np.median(rebuffet_gene_values)
        lm22_median = np.median(lm22_gene_values)
        
        # Calculate Median Absolute Deviation (MAD) for scale
        rebuffet_mad = np.median(np.abs(rebuffet_gene_values - rebuffet_median))
        lm22_mad = np.median(np.abs(lm22_gene_values - lm22_median))
        
        # Skip genes with zero or very low expression in Rebuffet
        if rebuffet_median < 1e-6 or rebuffet_mad < 1e-6:
            # For low-expression genes, just match the LM22 median
            rebuffet_harmonized.loc[gene] = lm22_median
            genes_skipped += 1
            continue
        
        # Calculate scale factor and location adjustment
        scale_factor = lm22_mad / rebuffet_mad if rebuffet_mad > 0 else 1.0
        location_shift = lm22_median - (rebuffet_median * scale_factor)
        
        # Apply gene-wise transformation: new_value = (old_value * scale) + shift
        transformed_values = rebuffet_gene_values * scale_factor + location_shift
        
        # CRITICAL: Ensure no negative values (would cause NaNs in log transformation)
        if transformed_values.min() < 0:
            logger.warning(f"Gene {gene}: Negative values detected after transformation. Applying safety correction.")
            # Shift all values to ensure minimum is small positive value
            min_val = transformed_values.min()
            safety_shift = abs(min_val) + 1e-6  # Small epsilon
            transformed_values = transformed_values + safety_shift
        
        rebuffet_harmonized.loc[gene] = transformed_values
        
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
    data_validation = validate_cibersortx_compatibility(rebuffet_harmonized)
    
    # Validate harmonization quality
    validation_results = validate_harmonization(rebuffet_signatures, rebuffet_harmonized, lm22_nk_signatures, common_genes)
    
    return rebuffet_harmonized, median_scale_factor

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

def validate_harmonization(rebuffet_original, rebuffet_harmonized, lm22_nk_signatures, common_genes):
    """
    Comprehensive validation of harmonization quality.
    
    Checks:
    1. Scale alignment between Rebuffet and LM22
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
    rebuffet_orig_medians = [np.median(rebuffet_original.loc[gene].values) for gene in common_genes]
    rebuffet_harm_medians = [np.median(rebuffet_harmonized.loc[gene].values) for gene in common_genes]
    lm22_medians = [np.median(lm22_nk_signatures.loc[gene].values) for gene in common_genes]
    
    # Calculate correlation of medians
    # Original Rebuffet vs LM22 correlation
    orig_corr, orig_p = pearsonr(rebuffet_orig_medians, lm22_medians)
    # Harmonized Rebuffet vs LM22 correlation
    harm_corr, harm_p = pearsonr(rebuffet_harm_medians, lm22_medians)
    
    logger.info(f"   Median correlation (Original Rebuffet vs LM22): r={orig_corr:.3f}, p={orig_p:.2e}")
    logger.info(f"   Median correlation (Harmonized Rebuffet vs LM22): r={harm_corr:.3f}, p={harm_p:.2e}")
    
    validation_results['original_correlation'] = orig_corr
    validation_results['harmonized_correlation'] = harm_corr
    validation_results['correlation_improvement'] = harm_corr - orig_corr
    
    # 2. Pattern preservation validation
    logger.info("2. Gene Expression Pattern Preservation:")
    
    # Check if relative ordering of genes is preserved
    pattern_correlations = []
    for col in rebuffet_original.columns:
        orig_pattern = rebuffet_original[col].loc[common_genes].values
        harm_pattern = rebuffet_harmonized[col].loc[common_genes].values
        
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
    rebuffet_orig_all = rebuffet_original.loc[common_genes].values.flatten()
    rebuffet_harm_all = rebuffet_harmonized.loc[common_genes].values.flatten()
    lm22_all = lm22_nk_signatures.loc[common_genes].values.flatten()
    
    # Filter out zeros for meaningful comparison
    rebuffet_orig_nonzero = rebuffet_orig_all[rebuffet_orig_all > 0]
    rebuffet_harm_nonzero = rebuffet_harm_all[rebuffet_harm_all > 0]
    lm22_nonzero = lm22_all[lm22_all > 0]
    
    # Calculate distribution statistics
    orig_vs_lm22_median_ratio = np.median(rebuffet_orig_nonzero) / np.median(lm22_nonzero)
    harm_vs_lm22_median_ratio = np.median(rebuffet_harm_nonzero) / np.median(lm22_nonzero)
    
    logger.info(f"   Original Rebuffet/LM22 median ratio: {orig_vs_lm22_median_ratio:.3f}")
    logger.info(f"   Harmonized Rebuffet/LM22 median ratio: {harm_vs_lm22_median_ratio:.3f}")
    
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

def create_hybrid_matrix(lm22_prepared, rebuffet_signatures):
    """Merge LM22 and Rebuffet signatures into hybrid matrix."""
    logger.info("=" * 70)
    logger.info("CREATING HYBRID SIGNATURE MATRIX")
    logger.info("=" * 70)
    
    # Get gene lists
    lm22_genes = set(lm22_prepared.index)
    rebuffet_genes = set(rebuffet_signatures.index)
    
    logger.info(f"LM22 genes: {len(lm22_genes):,}")
    logger.info(f"Rebuffet genes: {len(rebuffet_genes):,}")
    
    # Find common and missing genes
    common_genes = list(lm22_genes & rebuffet_genes)
    missing_in_rebuffet = list(lm22_genes - rebuffet_genes)
    
    logger.info(f"Common genes: {len(common_genes):,}")
    logger.info(f"Missing in Rebuffet (will zero-fill): {len(missing_in_rebuffet):,}")
    
    if len(common_genes) < 1000:
        logger.warning(f"Only {len(common_genes)} common genes found. Zero-filling will help preserve signature quality.")
    
    # Create enhanced Rebuffet signatures by zero-filling missing LM22 genes
    logger.info("Zero-filling missing LM22 genes in Rebuffet signatures...")
    rebuffet_enhanced = rebuffet_signatures.copy()
    
    if missing_in_rebuffet:
        # Create zero matrix for missing genes
        zero_data = pd.DataFrame(
            0.0, 
            index=missing_in_rebuffet, 
            columns=rebuffet_signatures.columns
        )
        
        # Concatenate with existing Rebuffet signatures
        rebuffet_enhanced = pd.concat([rebuffet_enhanced, zero_data], axis=0)
        logger.info(f"Added {len(missing_in_rebuffet)} zero-filled genes to Rebuffet signatures")
    
    # Now use ALL LM22 genes (common + zero-filled)
    all_lm22_genes = sorted(lm22_genes)
    lm22_final = lm22_prepared.loc[all_lm22_genes].copy()
    rebuffet_final = rebuffet_enhanced.loc[all_lm22_genes].copy()
    
    # Merge matrices
    hybrid_matrix = pd.concat([lm22_final, rebuffet_final], axis=1)
    
    logger.info(f"Hybrid matrix created. Shape: {hybrid_matrix.shape}")
    logger.info(f"Total signatures: {hybrid_matrix.shape[1]}")
    logger.info(f"Genes preserved: {hybrid_matrix.shape[0]} (was {len(common_genes)}, gained {len(missing_in_rebuffet)})")
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

def generate_summary(hybrid_matrix, rebuffet_signatures, lm22_prepared, output_dir):
    """Generate comprehensive summary of hybrid matrix creation."""
    logger.info("=" * 70)
    logger.info("GENERATING ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    summary_path = output_dir / SUMMARY_FILENAME
    
    with open(summary_path, 'w') as f:
        f.write("Hybrid LM22-Rebuffet NK Signatures Matrix Generation Summary\n")
        f.write("=" * 68 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HYBRID APPROACH OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write("This hybrid matrix combines:\n")
        f.write("- LM22's well-validated immune signatures (T cells, B cells, myeloid, etc.)\n")
        f.write("- Rebuffet's 3 core NK signatures representing key developmental states\n\n")
        
        f.write("ORIGINAL MATRICES\n")
        f.write("-" * 20 + "\n")
        f.write(f"LM22 original signatures: {lm22_prepared.shape[1] + len(LM22_NK_COLUMNS_TO_REMOVE)}\n")
        f.write(f"LM22 after NK removal: {lm22_prepared.shape[1]}\n")
        f.write(f"Rebuffet NK signatures added: {rebuffet_signatures.shape[1]}\n")
        f.write(f"Final hybrid signatures: {hybrid_matrix.shape[1]}\n\n")
        
        f.write("SIGNATURE COMPOSITION\n")
        f.write("-" * 25 + "\n")
        f.write("LM22-derived signatures:\n")
        for col in lm22_prepared.columns:
            f.write(f"  {col}\n")
        f.write("\nRebuffet NK signatures:\n")
        for col in rebuffet_signatures.columns:
            f.write(f"  {col}\n")
        f.write("\n")
        
        f.write("TECHNICAL DETAILS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Common genes used: {hybrid_matrix.shape[0]:,}\n")
        f.write(f"Data scale: Linear (TPM for Rebuffet, converted for LM22)\n")
        f.write(f"Scale harmonization: Gene-wise robust scaling (median + MAD)\n")
        f.write(f"  - Each gene scaled individually using robust statistics\n")
        f.write(f"  - Preserves gene-specific expression patterns\n")
        f.write(f"  - More biologically sound than uniform scaling\n")
        f.write(f"Format: CIBERSORTx-compatible (tab-separated)\n")
        f.write(f"Removed from LM22: {LM22_NK_COLUMNS_TO_REMOVE}\n\n")
        
        f.write("HARMONIZATION METHOD\n")
        f.write("-" * 25 + "\n")
        f.write("Gene-wise robust scaling approach:\n")
        f.write("1. For each gene, calculate median and MAD (Median Absolute Deviation)\n")
        f.write("2. Scale factor = LM22_MAD / Rebuffet_MAD\n")
        f.write("3. Location shift = LM22_median - (Rebuffet_median * scale_factor)\n")
        f.write("4. Final value = (Rebuffet_value * scale_factor) + location_shift\n")
        f.write("This preserves gene-specific expression patterns while matching LM22 scale.\n\n")
        
        f.write("REBUFFET NK SIGNATURE DETAILS\n")
        f.write("-" * 33 + "\n")
        f.write("Functional ordering (development progression):\n")
        for i, subtype in enumerate(REBUFFET_SUBTYPES_ORDERED, 1):
            f.write(f"{i}. {subtype}\n")
        f.write("\n")
        
        f.write("BIOLOGICAL INTERPRETATION\n")
        f.write("-" * 27 + "\n")
        f.write("NK2: Immature/regulatory NK cells (CD56bright-like)\n")
        f.write("NK1C: Mature cytotoxic NK cells (terminal NK1 subtype)\n")
        f.write("NK3: Adaptive-like NK cells (high NKG2C expression)\n\n")
        
        f.write("USAGE\n")
        f.write("-" * 10 + "\n")
        f.write("Use this hybrid matrix with CIBERSORTx for enhanced NK cell deconvolution.\n")
        f.write("The matrix provides:\n")
        f.write("- Standard immune cell resolution from LM22\n")
        f.write("- High-resolution NK cell subtype analysis from Rebuffet blood data\n")
        f.write("- Developmental progression from immature to adaptive NK cells\n")
    
    logger.info(f"Analysis summary saved to: {summary_path}")

def main():
    """Main function to create hybrid LM22-Rebuffet NK signatures matrix."""
    print("=" * 80)
    print("HYBRID LM22-REBUFFET NK SIGNATURES GENERATOR")
    print("=" * 80)
    print()
    print("Creating hybrid signature matrix by combining:")
    print("• LM22 immune signatures (minus original NK)")
    print("• Rebuffet core NK signatures (3 distinct subtypes)")
    print("• Using LINEAR SPACE averaging for proper arithmetic means")
    print("• Scaling Rebuffet NK signatures to match LM22 NK signature scale")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load and process Rebuffet data
    adata_rebuffet = load_rebuffet_data()
    if adata_rebuffet is None:
        logger.error("Failed to load Rebuffet data. Exiting.")
        return False
    
    # Step 2: Preprocess Rebuffet data
    adata_rebuffet = preprocess_rebuffet_data(adata_rebuffet)
    
    # Step 3: Create Rebuffet signatures (already in linear TPM space)
    rebuffet_signatures_linear = create_rebuffet_signatures(adata_rebuffet)
    
    # Step 4: Load and prepare LM22
    lm22_prepared, lm22_nk_original = load_and_prepare_lm22()
    if lm22_prepared is None or lm22_nk_original is None:
        logger.error("Failed to load/prepare LM22. Exiting.")
        return False
    
    # Step 5: Harmonize scales using gene-wise robust scaling
    rebuffet_signatures_harmonized, median_scale_factor = harmonize_signature_scales(rebuffet_signatures_linear, lm22_nk_original)
    logger.info(f"Applied median gene-wise scale factor: {median_scale_factor:.1f}x to Rebuffet signatures")
    
    # Step 6: Create hybrid matrix
    hybrid_matrix = create_hybrid_matrix(lm22_prepared, rebuffet_signatures_harmonized)
    
    # Step 6.5: Final validation of hybrid matrix
    logger.info("Performing final validation of hybrid matrix...")
    final_validation = validate_cibersortx_compatibility(hybrid_matrix)
    if not final_validation:
        logger.error("CRITICAL: Hybrid matrix failed CIBERSORTx compatibility checks!")
        return False
    
    # Step 7: Save hybrid matrix
    output_path = save_hybrid_matrix(hybrid_matrix, OUTPUT_DIR)
    
    # Step 8: Generate summary
    generate_summary(hybrid_matrix, rebuffet_signatures_harmonized, lm22_prepared, OUTPUT_DIR)
    
    logger.info("=" * 70)
    logger.info("HYBRID MATRIX GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Hybrid matrix: {output_path}")
    logger.info(f"Summary: {OUTPUT_DIR / SUMMARY_FILENAME}")
    logger.info(f"Total signatures: {hybrid_matrix.shape[1]} ({lm22_prepared.shape[1]} LM22 + {rebuffet_signatures_harmonized.shape[1]} Rebuffet)")
    logger.info(f"Rebuffet signatures harmonized using gene-wise robust scaling (median factor: {median_scale_factor:.1f}x)")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Hybrid LM22-Rebuffet NK signatures matrix created successfully!")
        print("🔬 Ready for enhanced CIBERSORTx deconvolution with high-resolution NK analysis")
        print("📊 Features 3 core Rebuffet NK subtypes: NK2, NK1C, NK3")
    else:
        print("\n❌ Hybrid matrix generation failed!") 