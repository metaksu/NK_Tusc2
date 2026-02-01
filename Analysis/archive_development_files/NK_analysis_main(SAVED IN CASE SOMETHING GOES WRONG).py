# %% [markdown]
# # Enhanced Analysis of TUSC2 in the Human Natural Killer (NK) Cell Landscape
#
# **Version:** 3.0 (Enhanced with Modern Best Practices)
# **Date:** December 22, 2024
# **Analyst:** AI Assistant
#
# ---
#
# ## Key Enhancements from v2.1
#
# 1. **Advanced Quality Control** - Adaptive MT filtering, enhanced doublet detection
# 2. **Modern Gene Signatures** - Updated 2024 NK cell signatures while preserving originals
# 3. **Enhanced Statistics** - Effect size calculations, pseudo-bulk analysis
# 4. **Improved Visualization** - Modern plotting aesthetics
#
# ---
#
# ## Objective
#
# This notebook performs a comprehensive analysis of the tumor suppressor gene **TUSC2** within the context of human NK cell biology. The primary goals are to:
# 1.  Characterize the expression of TUSC2 across different NK cell subtypes and biological environments (healthy blood, normal tissue, tumor tissue).
# 2.  Determine the functional impact of TUSC2 expression by comparing TUSC2-expressing and non-expressing cells against established gene programs for cytotoxicity, maturation, and exhaustion.
# 3.  Synthesize findings across contexts to understand how the role of TUSC2 in NK cells may be altered in the tumor microenvironment.
#
# ---
#
# ## Table of Contents
#
# **Part 0: Global Setup & Definitions**
# *   [0.1: Library Imports & Plotting Aesthetics](#27ee6c72)
# *   [0.2: File Paths & Output Directory Structure](#fdc435ee)
# *   [0.3: Core Biological & Analytical Definitions](#4eb7a666)
# *   [0.4: Utility Functions](#44bad9c1)
#
# **Part 1: Data Ingestion, Preprocessing & Cohort Generation**
# *   [1.1: Healthy Blood NK Cohort (`adata_blood`)](#21777f20)
# *   [1.2: Tang et al. Pan-Cancer Dataset (`adata_tang_full`)](#44740648)
# *   [1.3: Context-Specific Cohorts & Subtype Annotation](#cd53838f)
#
# **Part 2: Baseline Characterization of NK Subtypes**
# *   [2.1: Composition and Visual Overview](#1e922005)
# *   [2.2: Context-Specific Transcriptional Markers](#9632f03a)
# *   [2.3: Functional Signature Profiling](#476e8a31)
# *   [2.4: Developmental Marker Profiling](#1fe7093c)
# *   [2.5: Synthesis Blueprint Figure](#6db3d76b)
#
# **Part 3: TUSC2 Analysis - A Layered Investigation**
# *   [3.1: Layer 1 - TUSC2 Across Broad Contexts](#8623913e)
# *   [3.2: Layer 2 - TUSC2 Within Subtypes](#9d234345)
# *   [3.3: Layer 3 - TUSC2 Impact on Functional Signatures](#46b23b06)
# *   [3.5: Layer 5 - DEG Analysis of TUSC2 Groups](#32d238ce)
# *   [3.6: Histology-Specific TUSC2 Analysis](#histology_specific)
#
# **Part 4: Cross-Context Synthesis & Comparative Insights**
# *   [4.1: Comparing Baseline NK Subtype Characteristics](#732681a4)
# *   [4.2: Comparing TUSC2 Biology Across Contexts](#318f4be4)
# *   [4.3: TUSC2 Impact on Subtype Programs](#e270e77a)
# *   [4.4: TUSC2 Impact on Developmental State](#0619a58d)

# %%
# PART 0: Enhanced Global Setup, Data Definitions, and Utility Functions
# Section 0.1: Library Imports & Global Plotting Aesthetics

# --- 0.1.1: Standard Library and Third-Party Imports ---
print("--- Enhanced NK Analysis v3.0: Modern Best Practices Integration ---")
print("--- 0.1.1: Importing Libraries ---")

# Standard Python Libraries
import os
import re
import itertools
import sys
import warnings
from pathlib import Path

# Data Science & Numerical Libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.io import mmread
from scipy.sparse import csr_matrix

# Single-Cell Analysis Libraries
import scanpy as sc

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Specific Statistical/Analysis Tools
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests

# Enhanced utilities import
sys.path.append("scripts/utilities")
try:
    from enhanced_qc_functions import (
        AdaptiveQualityControl,
        UPDATED_NK_SIGNATURES_2024,
        SIGNATURE_MAPPING,
        calculate_effect_sizes,
        pseudo_bulk_differential_expression,
    )

    print("  Enhanced QC functions imported successfully for metabolic signatures.")
    METABOLIC_SIGNATURES_AVAILABLE = True
    # Return to original signatures as base, add metabolic signatures
    ENHANCED_QC_AVAILABLE = False
except ImportError as e:
    print(f"  WARNING: Enhanced QC functions not available: {e}")
    print("  Falling back to standard methods.")
    ENHANCED_QC_AVAILABLE = False
    METABOLIC_SIGNATURES_AVAILABLE = False

# Gene Set Enrichment Analysis (GSEA)
try:
    import gseapy

    print("  gseapy imported successfully.")
    GSEAPY_AVAILABLE = True
except ImportError:
    print(
        "  WARNING: gseapy could not be imported. GSEA functionality will be disabled."
    )
    GSEAPY_AVAILABLE = False

# For explicitly setting inline plot formats in Jupyter
try:
    import matplotlib_inline.backend_inline

    print("  matplotlib_inline.backend_inline imported successfully.")
    MATPLOTLIB_INLINE_AVAILABLE = True
except ImportError:
    print(
        "  WARNING: matplotlib_inline.backend_inline not found. Plotting backend might default."
    )
    MATPLOTLIB_INLINE_AVAILABLE = False

print("All libraries attempted to import.\\n")

# --- 0.1.2: Enhanced Global Plotting Aesthetics & Scanpy Settings ---
print("--- 0.1.2: Configuring Enhanced Plotting Aesthetics & Scanpy Settings ---")

# Configure Matplotlib inline backend for high-resolution outputs in Jupyter
if MATPLOTLIB_INLINE_AVAILABLE:
    matplotlib_inline.backend_inline.set_matplotlib_formats("retina", "png")
    print("  Matplotlib inline formats set to 'retina' and 'png'.")

# Enhanced publication-quality plotting parameters
FIGURE_FORMAT = "png"  # Default format for saving figures (png, svg, pdf)
FIGURE_DPI = 300  # DPI for saved figures

# Enhanced plotting parameters with modern aesthetics
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "savefig.dpi": FIGURE_DPI,
        "savefig.format": FIGURE_FORMAT,
        "savefig.transparent": False,
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

print(f"  Enhanced plotting parameters set for {FIGURE_FORMAT} at {FIGURE_DPI} DPI.")

# Enhanced Scanpy settings
sc.settings.autoshow = False  # Prevents plots from showing automatically.
sc.settings.verbosity = 2  # Reduced verbosity for cleaner output

# Set random seed for reproducibility across all libraries
RANDOM_SEED = 42
sc.settings.seed = RANDOM_SEED
np.random.seed(RANDOM_SEED)
print(f"  Random seed for numpy and scanpy set to: {RANDOM_SEED}")
print("Enhanced plotting aesthetics and Scanpy settings configured.")
print("--- End of Section 0.1 ---")

# %%
# PART 0: Global Setup, Data Definitions, and Utility Functions
# Section 0.2: File Paths & Master Output Directory Structure

print("--- 0.2: Defining File Paths & Master Output Directory Structure ---")

# --- 0.2.1: Define Input File Paths ---
print("  --- 0.2.1: Defining Input File Paths ---")

# Rebuffet et al. (2024) Data - UPDATED TO NEW DATASET (for healthy blood NK cohort and reference markers)
# User: PLEASE VERIFY AND UPDATE THIS PATH if necessary
REBUFFET_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\data\processed\PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
print(f"    REBUFFET_H5AD_FILE set to: {REBUFFET_H5AD_FILE}")

# Tang et al. Combined Dataset (comb_CD56_CD16_NK.h5ad) - UPDATED TO USE COMB DATASET
# User: PLEASE VERIFY AND UPDATE THIS PATH if necessary
TANG_COMBINED_H5AD_FILE = (
    r"C:\Users\met-a\Documents\Analysis\data\processed\comb_CD56_CD16_NK.h5ad"
)
print(f"    TANG_COMBINED_H5AD_FILE set to: {TANG_COMBINED_H5AD_FILE}")

# --- 0.2.2: Define Master Output Directory and Subdirectories ---
print("\n  --- 0.2.2: Defining Master Output Directory and Subdirectories ---")

# User: PLEASE VERIFY AND UPDATE THIS PATH for the main project output
MASTER_OUTPUT_DIR = (
    r"C:\Users\met-a\Documents\Analysis\Combined_NK_TUSC2_Analysis_Output"
)
print(f"    MASTER_OUTPUT_DIR set to: {MASTER_OUTPUT_DIR}")

# Define subdirectory names as per the plan
SUBDIR_NAMES = {
    "setup_figs": "0_Setup_Figs",
    "processed_anndata": "1_Processed_Anndata",
    "blood_nk_char": "2_Blood_NK_Char",
    "normal_tissue_nk_char": "3_NormalTissue_NK_Char",
    "tumor_tissue_nk_char": "4_TumorTissue_NK_Char",
    "tusc2_analysis": "5_TUSC2_Analysis",
    "tusc2_broad_context": "5_TUSC2_Analysis/5A_Broad_Context",
    "tusc2_within_context_subtypes": "5_TUSC2_Analysis/5B_Within_Context_Subtypes",
    "cross_context_synthesis": "6_Cross_Context_Synthesis",
    "histology_deep_dive": "7_Histology_Specific_TUSC2_Analysis",
    # General subdirectories for figures, data, stats within each major part if needed later
    # For now, the structure above is primary. We can add these if the part-specific ones become too cluttered.
    "general_figures": "common_figures",
    "general_data_graphpad": "common_data_graphpad",
    "general_stat_results": "common_stat_results",
    "general_temp_data": "common_temp_data",
}

# Create full paths for subdirectories
OUTPUT_SUBDIRS = {
    key: os.path.join(MASTER_OUTPUT_DIR, name) for key, name in SUBDIR_NAMES.items()
}

# Create all directories
os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
print(f"    Ensured Master Output Directory exists: {MASTER_OUTPUT_DIR}")

for key, path in OUTPUT_SUBDIRS.items():
    os.makedirs(path, exist_ok=True)
    print(f"    Ensured Subdirectory exists for '{key}': {path}")

# --- 0.2.3: Update Scanpy Settings with Master Output Directory ---
print("\n  --- 0.2.3: Updating Scanpy Settings ---")
# Set Scanpy's default figure directory.
# Note: Individual `save_figure_and_data` calls will use more specific subdirectories.
# This is a fallback or for direct sc.pl.savefig calls if any.
sc.settings.figdir = OUTPUT_SUBDIRS["general_figures"]  # or MASTER_OUTPUT_DIR
print(f"    Scanpy figure directory (sc.settings.figdir) set to: {sc.settings.figdir}")

print("--- End of Section 0.2 ---")

# %%
# PART 0: Global Setup, Data Definitions, and Utility Functions
# Section 0.3: Core Biological & Analytical Definitions (Corrected & Expanded)

print(
    "--- 0.3: Defining Core Biological & Analytical Definitions (Corrected & Expanded) ---"
)

# --- 0.3.1: Gene of Interest ---
print("  --- 0.3.1: Gene of Interest ---")
TUSC2_GENE_NAME = "TUSC2"
print(f"    TUSC2_GENE_NAME: {TUSC2_GENE_NAME}")

# --- 0.3.2: NK Subtype Definitions & Key Metadata Columns ---
print("\n  --- 0.3.2: NK Subtype Definitions & Key Metadata Columns ---")

# Rebuffet NK Subtypes (for blood NK data) - functional ordering
REBUFFET_SUBTYPES_ORDERED = [
    "NK2",
    "NKint",
    "NK1A",
    "NK1B",
    "NK1C",
    "NK3",
]  # Functional ordering: immature/regulatory → mature cytotoxic → adaptive/terminal

# Tang NK Subtypes (for tissue NK data) - using original Tang subtypes
TANG_SUBTYPES_ORDERED = [
    "CD56brightCD16lo-c5-CREM",  # Regulatory/immature
    "CD56brightCD16lo-c4-IL7R",  # Immature
    "CD56brightCD16lo-c2-IL7R-RGS1lo",  # Transitional
    "CD56brightCD16lo-c3-CCL3",  # Inflammatory
    "CD56brightCD16lo-c1-GZMH",  # Cytotoxic bright
    "CD56brightCD16hi",  # Double-positive transitional
    "CD56dimCD16hi-c1-IL32",  # Cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1",  # Tissue-homing
    "CD56dimCD16hi-c3-ZNF90",  # Mature
    "CD56dimCD16hi-c4-NFKBIA",  # Activated
    "CD56dimCD16hi-c6-DNAJB1",  # Stress-response
    "CD56dimCD16hi-c7-NR4A3",  # Stimulated
    "CD56dimCD16hi-c8-KLRC2",  # Adaptive (NKG2C+)
    "CD56dimCD16hi-c5-MKI67",  # Proliferating
]  # Functional ordering following developmental progression

# --- Tang Subtype Splits for CD56+CD16- and CD56-CD16+ Analysis ---
print("\n  --- 0.3.2b: Tang Subtype Splits for CD56+CD16- and CD56-CD16+ Analysis ---")

# Tang CD56+CD16- Subtypes (CD56bright) - Regulatory/immature NK cells
TANG_CD56BRIGHT_SUBTYPES = [
    "CD56brightCD16lo-c5-CREM",  # Regulatory/immature
    "CD56brightCD16lo-c4-IL7R",  # Immature
    "CD56brightCD16lo-c2-IL7R-RGS1lo",  # Transitional
    "CD56brightCD16lo-c3-CCL3",  # Inflammatory
    "CD56brightCD16lo-c1-GZMH",  # Cytotoxic bright
    "CD56brightCD16hi",  # Double-positive transitional
]

# Tang CD56-CD16+ Subtypes (CD56dim) - Cytotoxic/mature NK cells
TANG_CD56DIM_SUBTYPES = [
    "CD56dimCD16hi-c1-IL32",  # Cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1",  # Tissue-homing
    "CD56dimCD16hi-c3-ZNF90",  # Mature
    "CD56dimCD16hi-c4-NFKBIA",  # Activated
    "CD56dimCD16hi-c6-DNAJB1",  # Stress-response
    "CD56dimCD16hi-c7-NR4A3",  # Stimulated
    "CD56dimCD16hi-c8-KLRC2",  # Adaptive (NKG2C+)
    "CD56dimCD16hi-c5-MKI67",  # Proliferating
]

# Subset mapping for Tang data
TANG_SUBSETS = {
    "CD56posCD16neg": {
        "name": "CD56posCD16neg",
        "description": "CD56+CD16- (Regulatory/Immature NK cells)",
        "subtypes": TANG_CD56BRIGHT_SUBTYPES,
    },
    "CD56negCD16pos": {
        "name": "CD56negCD16pos",
        "description": "CD56-CD16+ (Cytotoxic/Mature NK cells)",
        "subtypes": TANG_CD56DIM_SUBTYPES,
    },
}

print(f"    Tang CD56bright subtypes: {len(TANG_CD56BRIGHT_SUBTYPES)} subtypes")
print(f"    Tang CD56dim subtypes: {len(TANG_CD56DIM_SUBTYPES)} subtypes")
print(f"    Tang subset names: {list(TANG_SUBSETS.keys())}")

# Standardized column names for different datasets
REBUFFET_SUBTYPE_COL = "Rebuffet_Subtype"  # For Rebuffet blood NK data
TANG_SUBTYPE_COL = "Tang_Subtype"  # For Tang tissue NK data
REBUFFET_ORIG_SUBTYPE_COL = "ident"  # Original column name in Rebuffet data

# Legacy compatibility - will be updated to use dataset-specific columns
NK_SUBTYPE_PROFILED_COL = REBUFFET_SUBTYPE_COL  # Default to Rebuffet for blood data


def get_subtype_column(adata_obj):
    """
    Helper function to determine the correct subtype column for a given dataset.
    Returns the appropriate subtype column name based on dataset characteristics.
    """
    if adata_obj is None or adata_obj.n_obs == 0:
        return None

    # Check if this is Rebuffet blood data (has 'ident' column or Rebuffet subtype column)
    if (
        REBUFFET_ORIG_SUBTYPE_COL in adata_obj.obs.columns
        or REBUFFET_SUBTYPE_COL in adata_obj.obs.columns
    ):
        return REBUFFET_SUBTYPE_COL

    # Check if this is Tang tissue data (has Tang celltype column or Tang subtype column)
    if (
        TANG_CELLTYPE_COL in adata_obj.obs.columns
        or TANG_SUBTYPE_COL in adata_obj.obs.columns
    ):
        return TANG_SUBTYPE_COL

    # Fallback to legacy column
    return NK_SUBTYPE_PROFILED_COL


def get_subtype_categories(adata_obj):
    """
    Helper function to determine the correct subtype categories for a given dataset.
    Returns the appropriate ordered categories based on dataset type.
    """
    subtype_col = get_subtype_column(adata_obj)

    if subtype_col == REBUFFET_SUBTYPE_COL:
        return REBUFFET_SUBTYPES_ORDERED
    elif subtype_col == TANG_SUBTYPE_COL:
        return TANG_SUBTYPES_ORDERED
    else:
        return REBUFFET_SUBTYPES_ORDERED  # Fallback


def should_split_tang_subtypes(adata_obj):
    """
    Determine if the given dataset should be split into Tang subsets.
    Returns True if this is Tang tissue data (not blood).
    """
    if adata_obj is None or adata_obj.n_obs == 0:
        return False

    # Check if this is Tang data with appropriate subtype column
    subtype_col = get_subtype_column(adata_obj)
    if subtype_col != TANG_SUBTYPE_COL:
        return False

    # Check if we have Tang subtypes present
    if subtype_col not in adata_obj.obs.columns:
        return False

    # Check if we have sufficient cells in both CD56bright and CD56dim populations
    available_subtypes = set(adata_obj.obs[subtype_col].unique())
    cd56bright_count = len(available_subtypes.intersection(TANG_CD56BRIGHT_SUBTYPES))
    cd56dim_count = len(available_subtypes.intersection(TANG_CD56DIM_SUBTYPES))

    return cd56bright_count >= 2 and cd56dim_count >= 2


def get_tang_subtype_subsets(adata_obj, context_name):
    """
    Generate Tang subtype subsets for CD56+CD16- and CD56-CD16+ analysis.
    Returns list of (subset_name, adata_subset) tuples.
    """
    if not should_split_tang_subtypes(adata_obj):
        return [(None, adata_obj)]  # Return original data without splitting

    subsets = []
    subtype_col = get_subtype_column(adata_obj)

    for subset_key, subset_info in TANG_SUBSETS.items():
        # Filter cells that belong to this subset
        subset_mask = adata_obj.obs[subtype_col].isin(subset_info["subtypes"])

        if subset_mask.sum() > 0:
            adata_subset = adata_obj[subset_mask, :].copy()

            # Filter subtype categories to only include those present in this subset
            available_subtypes = [
                st
                for st in subset_info["subtypes"]
                if st in adata_subset.obs[subtype_col].unique()
            ]

            if (
                len(available_subtypes) >= 2
            ):  # Need at least 2 subtypes for meaningful analysis
                adata_subset.obs[subtype_col] = pd.Categorical(
                    adata_subset.obs[subtype_col],
                    categories=available_subtypes,
                    ordered=True,
                )

                subset_name = subset_info["name"]
                subsets.append((subset_name, adata_subset))

                print(
                    f"    Created Tang subset '{subset_name}': {adata_subset.n_obs} cells, {len(available_subtypes)} subtypes"
                )

    return subsets if subsets else [(None, adata_obj)]


def get_subtype_color_palette(adata_obj):
    """
    Get the appropriate color palette based on the dataset type.
    Returns the combined palette for maximum compatibility.
    """
    return COMBINED_SUBTYPE_COLOR_PALETTE


print("  Tang subtype splitting functions defined.")

# Key Metadata Columns for Tang Combined Dataset (updated structure)
TANG_TISSUE_COL = (
    "meta_tissue_in_paper"  # Primary tissue context (Blood/Tumor/Normal/Other tissue)
)
TANG_TISSUE_BLOOD_COL = "meta_tissue"  # Blood-specific dataset column
TANG_MAJORTYPE_COL = "Majortype"  # CD56bright/dim + CD16 high/low combinations
TANG_CELLTYPE_COL = "celltype"  # Fine-grained NK subtypes (14 subtypes)
TANG_HISTOLOGY_COL = "meta_histology"  # Cancer type information (25 cancer types)
TANG_BATCH_COL = "batch"  # Batch information for integration
TANG_DATASETS_COL = "datasets"  # Source dataset information
TANG_PATIENT_ID_COL = "meta_patientID"  # Patient identifier

print(
    f"    REBUFFET_SUBTYPES_ORDERED ({len(REBUFFET_SUBTYPES_ORDERED)} subtypes): {REBUFFET_SUBTYPES_ORDERED}"
)
print(
    f"    TANG_SUBTYPES_ORDERED ({len(TANG_SUBTYPES_ORDERED)} subtypes): {TANG_SUBTYPES_ORDERED[:5]}... [showing first 5]"
)
print(f"    Rebuffet Subtype Column: {REBUFFET_SUBTYPE_COL}")
print(f"    Tang Subtype Column: {TANG_SUBTYPE_COL}")
print(f"    Tang Primary Tissue Column: {TANG_TISSUE_COL}")
print(f"    Tang Blood Tissue Column: {TANG_TISSUE_BLOOD_COL}")
print(f"    Tang Majortype Column: {TANG_MAJORTYPE_COL}")
print(f"    Tang Fine Celltype Column: {TANG_CELLTYPE_COL}")
print(f"    Tang Histology Column: {TANG_HISTOLOGY_COL}")

# Legacy GSE212890 column names (kept for backwards compatibility during transition)
METADATA_TISSUE_COLUMN_GSE212890 = TANG_TISSUE_COL  # Primary tissue context
METADATA_CELLTYPE_COLUMN_GSE212890 = (
    TANG_CELLTYPE_COL  # Original Tang fine-grained subtypes
)
METADATA_MAJORTYPE_COLUMN_GSE212890 = TANG_MAJORTYPE_COL  # CD56/CD16 broad categories
METADATA_HISTOLOGY_COLUMN_GSE212890 = TANG_HISTOLOGY_COL  # Cancer type information
METADATA_PATIENT_ID_COLUMN_GSE212890 = TANG_PATIENT_ID_COL  # Patient identifiers

print(f"    Tang Tissue Column: {METADATA_TISSUE_COLUMN_GSE212890}")
print(f"    Tang Majortype Column: {METADATA_MAJORTYPE_COLUMN_GSE212890}")
print(f"    Tang Fine Celltype Column: {METADATA_CELLTYPE_COLUMN_GSE212890}")
print(f"    Tang Histology Column: {METADATA_HISTOLOGY_COLUMN_GSE212890}")

# --- 0.3.2a: Comprehensive Tang Combined Dataset Metadata Values Documentation ---
print("\n  --- 0.3.2a: Tang Combined Dataset Metadata Values Documentation ---")
print("      Source: comb_CD56_CD16_NK.h5ad (142,304 cells, 13,493 genes)")
print(
    "      IMPORTANT: Tang 'Blood' context = peripheral blood (97.9% from cancer patients, 2.1% healthy)"
)
print("                 Rebuffet blood NK cells = 100% healthy donor peripheral blood")

# Primary Tissue Contexts (meta_tissue_in_paper) - 4 contexts
TANG_TISSUE_VALUES = [
    "Blood",  # 67,202 cells (47.2%) - Peripheral blood NK cells (97.9% cancer patients, 2.1% healthy)
    "Tumor",  # 34,900 cells (24.5%) - Tumor-infiltrating NK cells
    "Normal",  # 22,792 cells (16.0%) - Normal adjacent tissue NK cells
    "Other tissue",  # 17,410 cells (12.2%) - Other tissue contexts
]

# Major NK Cell Types (Majortype) - 3 types based on CD56/CD16 expression
TANG_MAJORTYPE_VALUES = [
    "CD56lowCD16high",  # 104,003 cells (73.1%) - Classic mature NK cells
    "CD56highCD16low",  # 35,283 cells (24.8%) - Immature/regulatory NK cells
    "CD56highCD16high",  # 3,018 cells (2.1%) - Transitional NK cells
]

# Fine-grained NK Cell Subtypes (celltype) - 14 subtypes from Tang et al. analysis
TANG_CELLTYPE_VALUES = [
    "CD56dimCD16hi-c3-ZNF90",  # 24,748 cells (17.4%) - ZNF90+ mature NK
    "CD56dimCD16hi-c2-CX3CR1",  # 23,150 cells (16.3%) - CX3CR1+ tissue-homing NK
    "CD56dimCD16hi-c4-NFKBIA",  # 17,177 cells (12.1%) - NFKBIA+ activated NK
    "CD56dimCD16hi-c7-NR4A3",  # 12,079 cells (8.5%) - NR4A3+ stimulated NK
    "CD56brightCD16lo-c3-CCL3",  # 10,373 cells (7.3%) - CCL3+ inflammatory NK
    "CD56dimCD16hi-c8-KLRC2",  # 10,326 cells (7.3%) - KLRC2+/NKG2C+ adaptive NK
    "CD56brightCD16lo-c5-CREM",  # 10,183 cells (7.2%) - CREM+ regulatory NK
    "CD56dimCD16hi-c1-IL32",  # 9,216 cells (6.5%) - IL32+ cytokine-producing NK
    "CD56dimCD16hi-c6-DNAJB1",  # 6,403 cells (4.5%) - DNAJB1+ stress-response NK
    "CD56brightCD16lo-c1-GZMH",  # 5,669 cells (4.0%) - GZMH+ cytotoxic bright NK
    "CD56brightCD16lo-c4-IL7R",  # 5,485 cells (3.9%) - IL7R+ immature NK
    "CD56brightCD16lo-c2-IL7R-RGS1lo",  # 3,573 cells (2.5%) - IL7R+/RGS1lo transitional NK
    "CD56brightCD16hi",  # 2,932 cells (2.1%) - Double-positive NK
    "CD56dimCD16hi-c5-MKI67",  # 990 cells (0.7%) - MKI67+ proliferating NK
]

# Cancer/Histology Types (meta_histology) - 25 cancer types + healthy donors
TANG_HISTOLOGY_VALUES = [
    "Breast Cancer(BRCA)",  # 33,648 cells (23.6%)
    "Lung Cancer(LC)",  # 24,282 cells (17.1%)
    "Melanoma(MELA)",  # 22,571 cells (15.9%)
    "Renal Carcinoma(RC)",  # 12,123 cells (8.5%)
    "Nasopharyngeal Carcinoma(NPC)",  # 10,205 cells (7.2%)
    "Head and Neck Squamous Cell Carcinoma(HNSCC)",  # 7,578 cells (5.3%)
    "Colorectal Cancer(CRC)",  # 6,330 cells (4.4%)
    "Hepatocellular Carcinoma(HCC)",  # 4,529 cells (3.2%)
    "Pancreatic Cancer(PACA)",  # 3,821 cells (2.7%)
    "Esophageal Cancer(ESCA)",  # 3,613 cells (2.5%)
    "Thyroid Carcinoma(THCA)",  # 2,869 cells (2.0%)
    "Gastric Cancer(GC)",  # 2,254 cells (1.6%)
    "Intrahepatic cholangiocarcinoma(ICC)",  # 1,595 cells (1.1%)
    "Healthy donor",  # 1,436 cells (1.0%)
    "Multiple Myeloma(MM)",  # 1,360 cells (1.0%)
    "Uterine Corpus Endometrial Carcinoma(UCEC)",  # 1,347 cells (0.9%)
    "Neuroblastoma(NB)",  # 633 cells (0.4%)
    "Prostate Cancer(PRAD)",  # 492 cells (0.3%)
    "Ovarian Cancer(OV)",  # 416 cells (0.3%)
    "Chronic Lymphocytic Leukemia(CLL)",  # 391 cells (0.3%)
    "Acute Lymphocytic Leukemia(ALL)",  # 307 cells (0.2%)
    "Basal Cell Carinoma(BCC)",  # 181 cells (0.1%)
    "Acute Myeloid Leukemia(AML)",  # 154 cells (0.1%)
    "Fallopian Tube Carcinoma(FTC)",  # 124 cells (0.1%)
    "Squamous Cell carcinoma(SCC)",  # 45 cells (0.0%)
]

# Major Source Datasets (datasets) - Top 20 of 64 total datasets
TANG_MAJOR_DATASETS = [
    "GSE169246",  # 30,016 cells (21.1%) - Largest single dataset
    "GSE139249",  # 21,792 cells (15.3%) - Second largest
    "this study",  # 11,963 cells (8.4%) - Tang et al. original data
    "GSE162025",  # 10,009 cells (7.0%)
    "GSE154826",  # 9,062 cells (6.4%)
    "GSE131907",  # 6,147 cells (4.3%)
    "GSE140228",  # 5,064 cells (3.6%)
    "GSE139324",  # 4,566 cells (3.2%)
    "6_LC",  # 4,187 cells (2.9%)
    "GSE155698",  # 3,738 cells (2.6%)
    "GSE164690",  # 3,715 cells (2.6%)
    "GSE178318",  # 3,481 cells (2.4%)
    "GSE145281",  # 2,146 cells (1.5%)
    "GSE145370",  # 1,810 cells (1.3%)
    "CR",  # 1,486 cells (1.0%)
    "GSE184362",  # 1,443 cells (1.0%)
    "GSE121636",  # 1,440 cells (1.0%)
    "GSE123904",  # 1,399 cells (1.0%)
    "GSE124310",  # 1,360 cells (1.0%)
    "GSE178341",  # 1,332 cells (0.9%)
    # Note: Additional 44 datasets with smaller cell counts
]

# Sequencing Platforms (meta_platform) - 11 platforms
TANG_PLATFORM_VALUES = [
    "10X",  # 133,581 cells (93.9%) - Dominant platform
    "Droplet",  # 4,697 cells (3.3%)
    "Indrop",  # 1,283 cells (0.9%)
    "InDrop",  # 1,250 cells (0.9%)
    "MARS-seq",  # 609 cells (0.4%)
    "SS2",  # 367 cells (0.3%)
    "Smart-seq2",  # 212 cells (0.1%)
    "BD Rahposidy",  # 117 cells (0.1%)
    "Seq-Well",  # 89 cells (0.1%)
    "SeqWell",  # 66 cells (0.0%)
    "Microwell-seq",  # 33 cells (0.0%)
]

print(f"      Documented {len(TANG_TISSUE_VALUES)} tissue contexts")
print(f"      Documented {len(TANG_MAJORTYPE_VALUES)} major NK types")
print(f"      Documented {len(TANG_CELLTYPE_VALUES)} fine-grained NK subtypes")
print(f"      Documented {len(TANG_HISTOLOGY_VALUES)} cancer/histology types")
print(
    f"      Documented {len(TANG_MAJOR_DATASETS)} major source datasets (of 64 total)"
)
print(f"      Documented {len(TANG_PLATFORM_VALUES)} sequencing platforms")
print("      Note: 732 unique batch identifiers available for batch effect analysis")

# Create ordered lists using the documented values
TANG_MAJORTYPE_ORDERED = TANG_MAJORTYPE_VALUES.copy()
TANG_TISSUE_CONTEXTS = TANG_TISSUE_VALUES.copy()

print("    --- End of Section 0.3.2a: Tang Combined Dataset Metadata Documentation ---")

# --- 0.3.2b: Comprehensive Rebuffet Dataset Metadata Values Documentation ---
print("\n  --- 0.3.2b: Rebuffet Dataset Metadata Values Documentation ---")
print(
    "      Source: PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad (35,578 cells, 22,941 genes)"
)
print("      100% healthy donor peripheral blood NK cells (TPM-normalized expression)")

# Primary NK Subtypes (ident) - 6 subtypes from Rebuffet et al. clustering
REBUFFET_SUBTYPE_VALUES = [
    "NK3",  # 11,562 cells (32.5%) - Adaptive-like NK cells (high NKG2C expression)
    "NK1C",  # 8,381 cells (23.6%) - Mature cytotoxic NK cells
    "NK1A",  # 6,098 cells (17.1%) - Early mature NK cells
    "NK1B",  # 3,959 cells (11.1%) - Intermediate mature NK cells
    "NKint",  # 3,567 cells (10.0%) - Intermediate/transitional NK cells
    "NK2",  # 2,011 cells (5.7%) - Immature/regulatory NK cells
]

# Secondary Clustering (SecondClust) - 8 subtypes with NK3 subdivisions
REBUFFET_SUBTYPE_SECONDARY_VALUES = [
    "NK1C",  # 8,381 cells (23.6%) - Mature cytotoxic NK cells
    "NK3A",  # 6,702 cells (18.8%) - NK3 subcluster A (dominant adaptive subset)
    "NK1A",  # 6,098 cells (17.1%) - Early mature NK cells
    "NK3C",  # 4,490 cells (12.6%) - NK3 subcluster C
    "NK1B",  # 3,959 cells (11.1%) - Intermediate mature NK cells
    "NKint",  # 3,567 cells (10.0%) - Intermediate/transitional NK cells
    "NK2",  # 2,011 cells (5.7%) - Immature/regulatory NK cells
    "NK3B",  # 370 cells (1.0%) - NK3 subcluster B (minor adaptive subset)
]

# CMV/NKG2C Status (nkg2c) - 3 categories
REBUFFET_CMV_STATUS_VALUES = [
    "NKG2Cneg",  # 15,093 cells (42.4%) - NKG2C negative (conventional NK cells)
    "unknown",  # 11,617 cells (32.7%) - NKG2C status unknown
    "NKG2Cpos",  # 8,868 cells (24.9%) - NKG2C positive (adaptive-like NK cells)
]

# Source Datasets (Dataset) - 4 integrated datasets
REBUFFET_DATASET_VALUES = [
    "Dataset4",  # 23,961 cells (67.3%) - Largest dataset contribution
    "Dataset3",  # 7,403 cells (20.8%) - Second largest
    "Dataset2",  # 2,768 cells (7.8%) - Third largest
    "Dataset1",  # 1,446 cells (4.1%) - Smallest dataset
]

# Donor Information (donor) - 13 unique healthy donors
REBUFFET_DONOR_VALUES = [
    "CMVpos2_donorC",  # 8,200 cells (23.0%) - CMV positive donor C
    "CMVpos3_donorD",  # 4,952 cells (13.9%) - CMV positive donor D
    "CMVpos4_donorE",  # 3,956 cells (11.1%) - CMV positive donor E
    "CMVneg2_donorB",  # 3,898 cells (11.0%) - CMV negative donor B
    "CMVneg1_donorA",  # 2,955 cells (8.3%) - CMV negative donor A
    "GSM5584156_1",  # 2,145 cells (6.0%) - GEO sample 1
    "GSM5584156_2",  # 1,902 cells (5.3%) - GEO sample 2
    "GSM3738542",  # 1,516 cells (4.3%) - GEO sample
    "GSM3377678",  # 1,446 cells (4.1%) - GEO sample
    "GSM3738543",  # 1,252 cells (3.5%) - GEO sample
    "GSM5584155",  # 1,164 cells (3.3%) - GEO sample
    "GSM5584154",  # 1,160 cells (3.3%) - GEO sample
    "GSM5584156_3",  # 1,032 cells (2.9%) - GEO sample 3
]

# Sequencing Chemistry (Chemistry) - Single chemistry version
REBUFFET_CHEMISTRY_VALUES = ["V2"]  # 35,578 cells (100.0%) - 10x Genomics Chemistry V2

# Quality Control Summary Statistics
REBUFFET_QC_STATS = {
    "nCount_RNA": {"min": 401, "max": 13707, "mean": 2807, "median": 2567},
    "nFeature_RNA": {"min": 226, "max": 3505, "mean": 1093, "median": 1071},
    "percent_mito": {"min": 0.00, "max": 22.34, "mean": 3.59},
    "percent_ribo": {"min": 5.31, "max": 50.12, "mean": 24.67},
    "expression_range": {"min": 0.0, "max": 206896.6, "data_type": "TPM-normalized"},
}

print(f"      Documented {len(REBUFFET_SUBTYPE_VALUES)} primary NK subtypes")
print(
    f"      Documented {len(REBUFFET_SUBTYPE_SECONDARY_VALUES)} secondary NK subtypes (with NK3 subdivisions)"
)
print(f"      Documented {len(REBUFFET_CMV_STATUS_VALUES)} CMV/NKG2C status categories")
print(f"      Documented {len(REBUFFET_DATASET_VALUES)} source datasets")
print(f"      Documented {len(REBUFFET_DONOR_VALUES)} healthy donors")
print(f"      TPM-normalized expression data (range: 0.0 to 206,896.6)")
print(f"      100% 10x Genomics Chemistry V2")

# Key metadata column definitions for the new Rebuffet dataset
# Note: REBUFFET_SUBTYPE_COL is already defined above as "Rebuffet_Subtype"
REBUFFET_SUBTYPE_SECONDARY_COL = (
    "SecondClust"  # Secondary clustering with NK3 subdivisions
)
REBUFFET_CMV_STATUS_COL = "nkg2c"  # CMV/NKG2C status column
REBUFFET_DONOR_COL = "donor"  # Donor information column
REBUFFET_DATASET_COL = "Dataset"  # Source dataset column
REBUFFET_CHEMISTRY_COL = "Chemistry"  # Sequencing chemistry column
REBUFFET_ORIG_IDENT_COL = "orig.ident"  # Original sample identifier column

print("    --- End of Section 0.3.2b: Rebuffet Dataset Metadata Documentation ---")

# --- 0.3.3: Defining Curated Developmental and Functional Gene Sets ---
print("\n  --- 0.3.3: Defining Curated Developmental and Functional Gene Sets ---")

# Developmental Signatures (Maturation Trajectory)
Maturation_NK_Regulatory = [
    # STAGE 1: Corresponds to Human NK2. Early, regulatory, lymphoid-homing state.
    # Genes selected based on their peak and specific expression in the NK2 row of the dot plot.
    "IL2RB",  # Strongest expression in NK2. Ref: Dot Plot. Flow: Yes.
    "SELL",  # Strongest expression in NK2. Ref: Dot Plot, Rebuffet Fig 3a. Flow: No.
    "GATA3",  # Strongest expression in NK2. Ref: Dot Plot. Flow: Yes (intracellular).
    "TCF7",  # Strongest expression in NK2. Ref: Dot Plot, Rebuffet Fig 1b. Flow: Yes (intracellular).
    "KLRC1",  # Hallmark of less mature NK cells, strongest in NK2. Ref: Dot Plot. Flow: Yes.
    "BACH2",  # TF repressing effector function, most prominent in NK2. Ref: Dot Plot, Lit Review II.A. Flow: No.
    "ID2",  # Key developmental TF, highest in NK2. Ref: Dot Plot. Flow: Yes (intracellular).
    "GZMK",  # Data-driven placement: The dot plot clearly shows GZMK is a hallmark of the NK2 subtype in this dataset. Ref: Dot Plot. Flow: No.
]

Maturation_NK_Intermediate = [
    # STAGE 2: Corresponds to Human NKint. The critical transition point.
    # Genes selected for their peak expression in the NKint row, representing the bridge between NK2 and NK1.
    "TBX21",  # T-bet, key TF for cytotoxicity, clearly begins its ramp-up in NKint. Ref: Dot Plot. Flow: Yes (intracellular).
    "ITGAM",  # CD11b, acquisition marks maturation, strong signal in NKint. Ref: Dot Plot. Flow: Yes.
    "KLRB1",  # CD161 (NK1.1 ortholog), marks the transition to the NK1 lineage, strong in NKint. Ref: Dot Plot. Flow: Yes (as NK-1.1).
    "JUNB",  # Key TF that peaks specifically in the NKint population. Ref: Dot Plot, Rebuffet Fig 2e. Flow: No.
    "EOMES",  # TF that drives effector function, ramps up alongside T-bet in NKint. Ref: Dot Plot. Flow: Yes (intracellular).
]

Maturation_NK_Mature_Cytotoxic = [
    # STAGE 3: Corresponds to Human NK1 (specifically NK1C). The canonical cytotoxic effector.
    # Genes selected for their peak expression in the NK1A/B/C rows, especially NK1C.
    "CX3CR1",  # Chemokine receptor for tissue homing, defines the NK1 lineage. Ref: Dot Plot, Rebuffet Fig 1c. Flow: No.
    "CD247",  # CD3-zeta, critical CD16 signaling adaptor, defines NK1. Ref: Dot Plot. Flow: No.
    "GZMB",  # Granzyme B, the quintessential cytotoxic molecule, peaks in NK1C. Ref: Dot Plot, Rebuffet Fig 2e. Flow: Yes (intracellular).
    "FCGR3A",  # CD16, defines CD56dim NK cells, highest expression in NK1. Ref: Dot Plot, Rebuffet Fig 1c. Flow: No.
    "PRF1",  # Perforin, the core cytotoxic molecule, peaks in NK1C. Ref: Dot Plot, Rebuffet Fig 2e. Flow: Yes (intracellular).
    "NKG7",  # Granule protein, its expression pattern perfectly tracks the NK1 maturation axis. Ref: Dot Plot. Flow: No.
    "FCER1G",  # FcR-gamma, critical ITAM adaptor, defines NK1 before loss in some adaptive cells. Ref: Dot Plot. Flow: No.
]

Maturation_NK_Adaptive = [
    # STAGE 4: Corresponds to Human NK3. A distinct, terminally differentiated lineage.
    # Genes are selected for being highly specific to the NK3 row in the dot plot.
    "KLRC2",  # NKG2C, the most specific surface marker for the NK3/adaptive cluster. Ref: Dot Plot, Rebuffet Fig 1b. Flow: No.
    "GZMH",  # Granzyme H, its expression is uniquely and strongly confined to NK3. Ref: Dot Plot, Rebuffet Fig 1b. Flow: No.
    "B3GAT1",  # Encodes CD57, a hallmark of terminal differentiation, peaks in NK3. Ref: Dot Plot. Flow: No.
    "CCL5",  # RANTES, chemokine whose expression is highly specific to NK3. Ref: Dot Plot. Flow: No.
    "IL32",  # Cytokine uniquely and strongly expressed by the NK3 cluster. Ref: Dot Plot, Rebuffet Fig 1b. Flow: No.
    "PRDM1",  # BLIMP-1, TF for terminal differentiation, shows highest specific peak in NK3. Ref: Dot Plot, Rebuffet Fig 1b. Flow: No.
]

# NOTE: DEVELOPMENTAL_GENE_SETS is now generated dynamically in Part 1.4
# based on actual Rebuffet blood NK subtype DEGs for more accurate signatures

# Functional Signatures (Specific Cell Capabilities) - REVERTED TO ORIGINAL
Activating_Receptors_Gene_Set = [
    "IL2RB",
    "IL18R1",
    "IL18RAP",
    "NCR1",
    "NCR2",
    "NCR3",
    "KLRK1",
    "FCGR3A",
    "CD226",
    "KLRC2",
    "CD244",
    "SLAMF6",
    "SLAMF7",
    "CD160",
    "KLRF1",
    "KIR2DS1",
    "KIR2DS2",
    "KIR2DS4",
    "KIR3DS1",
    "ITGAL",
]
Inhibitory_Receptors_Gene_Set = [
    "KLRC1",
    "KIR2DL1",
    "KIR2DL2",
    "KIR2DL3",
    "KIR3DL1",
    "KIR3DL2",
    "LILRB1",
    "PDCD1",
    "TIGIT",
    "CTLA4",
    "HAVCR2",
    "LAG3",
    "SIGLEC7",
    "SIGLEC9",
    "KLRG1",
    "CD300A",
    "LAIR1",
    "CEACAM1",
]
Cytotoxicity_Machinery_Gene_Set = [
    "PRF1",
    "GZMA",
    "GZMB",
    "GZMH",
    "GZMK",
    "GZMM",
    "NKG7",
    "GNLY",
    "SERPINB9",
    "SRGN",
    "FASLG",
    "TNFSF10",
    "LAMP1",
    "CTSC",
    "CTSW",
]
Cytokine_Chemokine_Gene_Set = [
    "IFNG",
    "TNF",
    "LTA",
    "CSF2",
    "IL10",
    "IL32",
    "XCL1",
    "XCL2",
    "CCL3",
    "CCL4",
    "CCL5",
    "CXCL8",
    "CXCL10",
]
Exhaustion_Suppression_Gene_Set = [
    "PDCD1",
    "HAVCR2",
    "LAG3",
    "TIGIT",
    "KLRC1",
    "KLRG1",
    "CD96",
    "LILRB1",
    "ENTPD1",
    "TOX",
    "EGR2",
    "MAF",
    "PRDM1",
    "HSPA1A",
    "DNAJB1",
]

FUNCTIONAL_GENE_SETS = {
    "Activating_Receptors": Activating_Receptors_Gene_Set,
    "Inhibitory_Receptors": Inhibitory_Receptors_Gene_Set,
    "Cytotoxicity_Machinery": Cytotoxicity_Machinery_Gene_Set,
    "Cytokine_Chemokine_Production": Cytokine_Chemokine_Gene_Set,
    "Exhaustion_Suppression_Markers": Exhaustion_Suppression_Gene_Set,
}

Acetylcholine_Receptors_Gene_Set = [
    "CHRNA2",
    "CHRNA3",
    "CHRNA4",
    "CHRNA5",
    "CHRNA7",
    "CHRNB2",
    "CHRNB4",
    "CHRNE",
    "CHRM3",
    "CHRM5",
]
Norepinephrine_Receptors_Gene_Set = ["ADRB2"]
Dopamine_Receptors_Gene_Set = ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5"]
Serotonin_Receptors_Gene_Set = ["HTR1A", "HTR2A", "HTR2C", "HTR7"]
Substance_P_Receptors_Gene_Set = ["TACR1"]
Estrogen_Receptors_Gene_Set = ["ESR1", "ESR2", "GPER1"]
Testosterone_Receptors_Gene_Set = ["NR3C4"]
Glutamate_Receptors_Gene_Set = [
    "GRIA1",
    "GRIA2",
    "GRIA3",
    "GRIA4",
    "GRIN1",
    "GRIN2A",
    "GRIN2B",
    "GRIN2C",
    "GRIN2D",
    "GRIN3A",
    "GRIN3B",
    "GRIK1",
    "GRIK2",
    "GRIK3",
    "GRIK4",
    "GRIK5",
    "GRM1",
    "GRM2",
    "GRM3",
    "GRM4",
    "GRM5",
    "GRM6",
    "GRM7",
    "GRM8",
]
GABA_Receptors_Gene_Set = [
    "GABRA1",
    "GABRA2",
    "GABRA3",
    "GABRA4",
    "GABRA5",
    "GABRA6",
    "GABRB1",
    "GABRB2",
    "GABRB3",
    "GABRG1",
    "GABRG2",
    "GABRG3",
    "GABRD",
    "GABRE",
    "GABRP",
    "GABRR1",
    "GABRR2",
    "GABRR3",
    "GABRQ",
    "GABBR1",
    "GABBR2",
]
Histamine_Receptors_Gene_Set = ["HRH1", "HRH2", "HRH4"]
Cannabinoid_Receptors_Gene_Set = ["CNR1", "CNR2", "GPR18", "GPR55"]
Opioid_Receptors_Gene_Set = ["OPRM1", "OPRD1", "OPRK1"]
Neuropeptide_Y_Receptors_Gene_Set = ["NPY1R", "NPY2R"]
Somatostatin_Receptors_Gene_Set = ["SSTR1", "SSTR2"]
VIP_Receptors_Gene_Set = ["VIPR1", "VIPR2"]
CGRP_Receptors_Gene_Set = ["CALCRL", "RAMP1"]
Purinergic_Receptors_Gene_Set = [
    "P2RX1",
    "P2RX4",
    "P2RY2",
    "P2RY11",
    "ADORA2A",
    "ADORA2B",
]

NEUROTRANSMITTER_RECEPTOR_GENE_SETS = {
    "Acetylcholine_Receptors": Acetylcholine_Receptors_Gene_Set,
    "Norepinephrine_Receptors": Norepinephrine_Receptors_Gene_Set,
    "Dopamine_Receptors": Dopamine_Receptors_Gene_Set,
    "Serotonin_Receptors": Serotonin_Receptors_Gene_Set,
    "Substance_P_Receptors": Substance_P_Receptors_Gene_Set,
    "Estrogen_Receptors": Estrogen_Receptors_Gene_Set,
    "Testosterone_Receptors": Testosterone_Receptors_Gene_Set,
    "Glutamate_Receptors": Glutamate_Receptors_Gene_Set,
    "GABA_Receptors": GABA_Receptors_Gene_Set,
    "Histamine_Receptors": Histamine_Receptors_Gene_Set,
    "Cannabinoid_Receptors": Cannabinoid_Receptors_Gene_Set,
    "Opioid_Receptors": Opioid_Receptors_Gene_Set,
    "Neuropeptide_Y_Receptors": Neuropeptide_Y_Receptors_Gene_Set,
    "Somatostatin_Receptors": Somatostatin_Receptors_Gene_Set,
    "VIP_Receptors": VIP_Receptors_Gene_Set,
    "CGRP_Receptors": CGRP_Receptors_Gene_Set,
    "Purinergic_Receptors": Purinergic_Receptors_Gene_Set,
}

IL15_IL2_Downstream_Gene_Set = [
    "STAT5A",
    "STAT5B",
    "AKT1",
    "MTOR",
    "HIF1A",
    "MYC",
    "BCL2",
    "MCL1",
    "CCND2",
    "ID2",
    "ZEB2",
    "KLF2",
    "GZMB",
    "PRF1",
    "IFNG",
    "CISH",
    "SOCS2",
]
IL12_Downstream_Gene_Set = [
    "STAT4",
    "TBX21",
    "EOMES",
    "IFNG",
    "IRF1",
    "GZMB",
    "PRF1",
    "FASLG",
    "IL12RB2",
    "IL18R1",
    "SOCS1",
]
IL18_Downstream_Gene_Set = [
    "NFKB1",
    "MYD88",
    "IRAK1",
    "IRAK4",
    "TNF",
    "XCL1",
    "CCL3",
    "CCL4",
    "IFNG",
    "PRF1",
    "STX11",
    "NLRP3",
]
IL21_Downstream_Gene_Set = [
    "STAT1",
    "STAT3",
    "BCL6",
    "ID2",
    "SAMHD1",
    "TBX21",
    "EOMES",
    "GZMA",
    "GZMB",
    "PRF1",
    "IFNG",
    "IL21R",
]
IL10_Downstream_Gene_Set = ["STAT3", "SOCS3", "BCL3", "MAF", "TGFB1", "IL10RA"]
IL27_Downstream_Gene_Set = [
    "STAT1",
    "STAT3",
    "ID2",
    "IL10",
    "LAG3",
    "HAVCR2",
    "PDCD1",
    "TIGIT",
]
IL33_Downstream_Gene_Set = [
    "MYD88",
    "IRAK1",
    "IRAK4",
    "NFKB1",
    "GATA3",
    "IL5",
    "IL13",
    "IFNG",
    "GZMB",
    "CSF2",
]

INTERLEUKIN_DOWNSTREAM_GENE_SETS = {
    "IL15_IL2_Downstream": IL15_IL2_Downstream_Gene_Set,
    "IL12_Downstream": IL12_Downstream_Gene_Set,
    "IL18_Downstream": IL18_Downstream_Gene_Set,
    "IL21_Downstream": IL21_Downstream_Gene_Set,
    "IL10_Downstream": IL10_Downstream_Gene_Set,
    "IL27_Downstream": IL27_Downstream_Gene_Set,
    "IL33_Downstream": IL33_Downstream_Gene_Set,
}

# Add metabolic signatures if available
if METABOLIC_SIGNATURES_AVAILABLE:
    print(
        "      Adding NK metabolic signatures (glycolysis and oxidative phosphorylation)..."
    )
    # Define metabolic signatures directly
    NK_Glycolysis_Genes = [
        "HK2",
        "PFKP",
        "ALDOA",
        "TPI1",
        "GAPDH",
        "PGK1",
        "PGAM1",
        "ENO1",
        "PKM",
        "LDHA",
        "SLC2A1",
        "SLC2A3",
    ]
    NK_OxPhos_Genes = [
        "NDUFA4",
        "NDUFB2",
        "SDHB",
        "UQCRB",
        "COX4I1",
        "COX6A1",
        "ATP5F1A",
        "ATP5F1B",
        "ATP5F1D",
        "ATP5PB",
        "IDH2",
        "MDH2",
    ]
    FUNCTIONAL_GENE_SETS["NK_Glycolysis"] = NK_Glycolysis_Genes
    FUNCTIONAL_GENE_SETS["NK_Oxidative_Phosphorylation"] = NK_OxPhos_Genes
    print(f"        Added NK_Glycolysis: {len(NK_Glycolysis_Genes)} genes")
    print(f"        Added NK_Oxidative_Phosphorylation: {len(NK_OxPhos_Genes)} genes")
else:
    print("      Metabolic signatures not available - using original signatures only")

print(f"    Defined {len(FUNCTIONAL_GENE_SETS)} functional gene sets.")

# NOTE: ALL_FUNCTIONAL_GENE_SETS is now created in Part 1.4 after
# DEVELOPMENTAL_GENE_SETS is dynamically generated

# --- 0.3.4: Murine NK Developmental Marker Orthologs (REVISED & EXPANDED) ---
print("\n  --- 0.3.4: Defining Murine NK Developmental Marker Orthologs (Revised) ---")

FLOW_INFORMED_STAGING_NK_LIST = {
    "KIT": "c-Kit (CD117) - Murine NKP. Early progenitor, rare peripheral",
    "SELL": "CD62L - Murine early iNK. Human CD56bright/NK2",
    "CXCR4": "CXCR4 - Human NKint/NK1A (Transitional)",
    "B3GAT1": "CD57 (encoded by B3GAT1) - Human terminal/adaptive marker",
    "KLRG1": "KLRG1 - Human/Murine terminal differentiation marker",
    "PRDM1": "Blimp-1 (TF) - Murine mNK. Human terminal/adaptive TF",
    "ZEB2": "Zeb2 (TF) - Murine mNK. Human maturation TF",
    "IKZF1": "Ikaros (TF) - Early lymphoid development",
    "RUNX3": "Runx3 (TF) - NK lineage specification/maturation",
    "CX3CR1": "CX3CR1 - Human mature CD56dim (Tissue homing)",
    "FCGR3A": "CD16a (FCGR3A) - Human CD56dim effector (ADCC). Murine mNKs Fcgr3+",
    "NCAM1": "CD56 (NCAM1) - Human Pan-NK (Bright vs. Dim stages)",
    "KLRC1": "NKG2A (KLRC1) - Human CD56bright > CD56dim (non-adaptive)",
    "KLRD1": "CD94 (KLRD1) - Partner for NKG2A/C/E",
    "KLRC2": "NKG2C (KLRC2) - Human adaptive NK marker",
    "IL2RB": "CD122 (IL-2Rbeta) - Murine NKP onward -> From FACS Panel",
    "CD19": "CD19 - B-cell marker -> From FACS Panel (Exclusion gate)",
    "ITGA2": "CD49b (DX5) - Murine iNK-a & mNK -> From FACS Panel (Pan-NK)",
    "ITGAM": "CD11b - Murine mNK (late) -> From FACS Panel",
    "NCR1": "NKp46 (CD335) - Murine late iNK-b & mNK -> From FACS Panel",
    "HLA-DRA": "HLA-DR (I-A/I-E) - MHC-II -> From FACS Panel",
    "CD3E": "CD3epsilon - T-cell marker -> From FACS Panel (Exclusion gate)",
    "LY6G6D": "Ly-6G - Granulocyte marker -> From FACS Panel (Exclusion gate, human ortholog complex: LY6G6D)",
    "KLRB1": "NK-1.1/CD161 - Murine iNK-b & mNK -> From FACS Panel",
    "CD27": "CD27 - Murine iNK-a/b, NKP -> From FACS Panel",
    "KLRK1": "NKG2D (CD314) - Broad activating R -> From FACS Panel",
    "GYPA": "TER-119 - Erythroid marker -> From FACS Panel (Exclusion gate)",
    "GZMB": "Granzyme B - Murine mNK -> From FACS Panel (Intracellular)",
    "PRF1": "Perforin - Murine mNK -> From FACS Panel (Intracellular)",
    "IFNG": "IFN-gamma - Murine mNK effector cytokine -> From FACS Panel (Intracellular)",
    "TNF": "TNF-alpha - Murine mNK effector cytokine -> From FACS Panel (Intracellular)",
    "ID2": "Id2 (TF) - Murine NKP & iNK -> From FACS Panel (Intracellular TF)",
    "TCF7": "Tcf1 (TF) - Murine iNK-a & iNK-b -> From FACS Panel (Intracellular TF)",
    "GATA3": "Gata3 (TF) - Murine iNK-a & iNK-b -> From FACS Panel (Intracellular TF)",
    "TBX21": "T-bet (TF) - Murine late iNK-b & mNK -> From FACS Panel (Intracellular TF)",
    "EOMES": "Eomes (TF) - Murine late iNK-b & mNK -> From FACS Panel (Intracellular TF)",
    "NKG7": "NKG7 - Granule protein, often co-expressed with GZMB/PRF1",
}
MURINE_DEV_MARKER_ORTHOLOGS = list(FLOW_INFORMED_STAGING_NK_LIST.keys())
print(
    f"    Defined {len(MURINE_DEV_MARKER_ORTHOLOGS)} murine NK developmental marker orthologs, updated from FACS panel."
)


# --- 0.3.5: Analytical Parameters & Thresholds ---
print("\n  --- 0.3.5: Defining Analytical Parameters & Thresholds ---")
GENE_PATTERNS_TO_EXCLUDE = [
    r"^RPS[0-9L]",
    r"^RPL[0-9L]",
    r"^RPLP[0-9]$",
    r"^RPSA$",
    r"^MT-",
    r"^ACT[BGINR]",
    r"ACTG1",
    r"^MYL[0-9]",
    r"^TPT1$",
    r"^FTL$",
    r"^FTH1$",
    r"^B2M$",
    r"^(HSP90|HSPA|HSPB|HSPD|HSPE|HSPH)[A-Z0-9]+",
    r"^EEF[12][A-Z0-9]*",
    r"^GAPDH$",
    r"^MALAT1$",
    r"^NEAT1$",
]
TOP_N_MARKERS_REF = 50
TOP_N_MARKERS_CONTEXT = 30
LOGFC_THRESHOLD_DEG = 0.25
ADJ_PVAL_THRESHOLD_DEG = 0.05
MIN_PCT_CELLS_FOR_DEG = 0.10
MIN_GENES_FOR_SCORING = 5
SIGNIFICANCE_THRESHOLDS = {"p_value": 0.05, "fdr_q_value": 0.05}
TUSC2_EXPRESSION_THRESHOLD_BINARY = 0.1
TUSC2_BINARY_GROUP_COL = f"{TUSC2_GENE_NAME}_Binary_Group"
TUSC2_BINARY_CATEGORIES = [
    f"{TUSC2_GENE_NAME}_Not_Expressed",
    f"{TUSC2_GENE_NAME}_Expressed",
]
print(
    f"    GENE_PATTERNS_TO_EXCLUDE defined ({len(GENE_PATTERNS_TO_EXCLUDE)} patterns)."
)
print(f"    Analytical thresholds defined (TOP_N_MARKERS, LFC, PVAL, etc.).")

# --- 0.3.6: Color Palettes ---
print("\n  --- 0.3.6: Defining Color Palettes ---")
SUBTYPE_COLOR_PALETTE = {
    "NK2": "#1f77b4",
    "NKint": "#ff7f0e",
    "NK1A": "#2ca02c",
    "NK1B": "#d62728",
    "NK1C": "#9467bd",
    "NK3": "#8c564b",
    "Unassigned": "#bdbdbd",
}

# Tang NK Subtype Color Palette - Comprehensive assignment for all 14 subtypes
TANG_SUBTYPE_COLOR_PALETTE = {
    # CD56bright subtypes (cooler colors - blues/greens/teals)
    "CD56brightCD16lo-c5-CREM": "#2E86AB",  # Deep blue - regulatory/immature
    "CD56brightCD16lo-c4-IL7R": "#A23B72",  # Purple-red - immature
    "CD56brightCD16lo-c2-IL7R-RGS1lo": "#F18F01",  # Orange - transitional
    "CD56brightCD16lo-c3-CCL3": "#C73E1D",  # Red-orange - inflammatory
    "CD56brightCD16lo-c1-GZMH": "#1B998B",  # Teal - cytotoxic bright
    "CD56brightCD16hi": "#4E598C",  # Blue-purple - double-positive
    # CD56dim subtypes (warmer colors - reds/oranges/purples)
    "CD56dimCD16hi-c1-IL32": "#FF6B6B",  # Coral red - cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1": "#4ECDC4",  # Turquoise - tissue-homing
    "CD56dimCD16hi-c3-ZNF90": "#45B7D1",  # Sky blue - mature
    "CD56dimCD16hi-c4-NFKBIA": "#96CEB4",  # Mint green - activated
    "CD56dimCD16hi-c6-DNAJB1": "#FFEAA7",  # Light yellow - stress-response
    "CD56dimCD16hi-c7-NR4A3": "#DDA0DD",  # Plum - stimulated
    "CD56dimCD16hi-c8-KLRC2": "#FF7675",  # Light red - adaptive (NKG2C+)
    "CD56dimCD16hi-c5-MKI67": "#6C5CE7",  # Purple - proliferating
}

# Combined subtype color palette that includes both Rebuffet and Tang subtypes
COMBINED_SUBTYPE_COLOR_PALETTE = {
    **SUBTYPE_COLOR_PALETTE,  # Rebuffet subtypes
    **TANG_SUBTYPE_COLOR_PALETTE,  # Tang subtypes
}

CONTEXT_COLOR_PALETTE = {
    "Blood": "#17becf",
    "NormalTissue": "#7f7f7f",
    "TumorTissue": "#e377c2",
}
TUSC2_BINARY_GROUP_COLORS = {
    TUSC2_BINARY_CATEGORIES[0]: "#aec7e8",
    TUSC2_BINARY_CATEGORIES[1]: "#ff9896",
}
TANG_MAJORTYPE_COLORS = {
    "CD56brightCD16low": "blue",
    "CD56dimCD16hi": "red",
    "CD56highCD16high": "green",
    "Unknown_Majortype": "grey",
}
TANG_META_TISSUE_COLORS = {
    "Tumor": "#e377c2",
    "Normal": "#7f7f7f",
    "Non-tumor": "#7f7f7f",
}
print("    Color palettes defined.")
print(f"      Rebuffet subtype colors: {len(SUBTYPE_COLOR_PALETTE)} subtypes")
print(f"      Tang subtype colors: {len(TANG_SUBTYPE_COLOR_PALETTE)} subtypes")
print(f"      Combined subtype colors: {len(COMBINED_SUBTYPE_COLOR_PALETTE)} subtypes")
print("--- End of Section 0.3 (Corrected & Expanded) ---")

# --- 0.3.7: Gene Name Mapping for Dataset Compatibility ---
print("\n  --- 0.3.7: Gene Name Mapping for Dataset Compatibility ---")

# Maps CD protein names to official gene symbols for compatibility with datasets
GENE_NAME_MAPPING = {
    "CD16": "FCGR3A",  # CD16 -> FCGR3A
    "CD56": "NCAM1",  # CD56 -> NCAM1
    "CD25": "IL2RA",  # CD25 -> IL2RA (already in signatures but keeping for clarity)
    "CD57": "B3GAT1",  # CD57 -> B3GAT1 (already in signatures but keeping for clarity)
    "CD137": "TNFRSF9",  # CD137 -> TNFRSF9 (already in signatures but keeping for clarity)
    "PD1": "PDCD1",  # PD1 -> PDCD1 (already in signatures but keeping for clarity)
    "TIM3": "HAVCR2",  # TIM3 -> HAVCR2 (already in signatures but keeping for clarity)
    "CD49A": "ITGA1",  # CD49A -> ITGA1 (already in signatures but keeping for clarity)
    "CD103": "ITGAE",  # CD103 -> ITGAE (already in signatures but keeping for clarity)
    "CD49B": "ITGA2",  # CD49B -> ITGA2 (already in signatures but keeping for clarity)
    "CD62L": "SELL",  # CD62L -> SELL (already in signatures but keeping for clarity)
    "NKG2C": "KLRC2",  # NKG2C -> KLRC2 (already in signatures but keeping for clarity)
}


def map_gene_names(gene_list, available_genes):
    """
    Map gene names using the mapping dictionary and filter for available genes.

    Parameters:
    - gene_list: List of gene names (may include CD protein names)
    - available_genes: List or Index of available gene names in the dataset

    Returns:
    - List of mapped gene names that are available in the dataset
    """
    mapped_genes = []
    missing_genes = []

    for gene in gene_list:
        # Try original name first
        if gene in available_genes:
            mapped_genes.append(gene)
        # Try mapped name if available
        elif gene in GENE_NAME_MAPPING and GENE_NAME_MAPPING[gene] in available_genes:
            mapped_genes.append(GENE_NAME_MAPPING[gene])
            print(f"      Mapped {gene} -> {GENE_NAME_MAPPING[gene]}")
        else:
            missing_genes.append(gene)

    if missing_genes:
        print(
            f"      Warning: {len(missing_genes)} genes not found: {', '.join(missing_genes)}"
        )

    return mapped_genes


print(
    f"    Gene name mapping dictionary defined with {len(GENE_NAME_MAPPING)} mappings."
)
print("    Function 'map_gene_names' defined for gene signature compatibility.")
print("--- End of Section 0.3.7 ---")

# %%
# PART 0: Global Setup, Data Definitions, and Utility Functions
# Section 0.4: Utility Functions (Final Corrected)

print("--- 0.4: Defining Utility Functions (Final Corrected) ---")


# --- 0.4.1: Function to Save Figures and Data for GraphPad ---
def save_figure_and_data(
    fig_object,
    data_df_for_graphpad,
    plot_basename,  # Base name for the plot/data, should NOT include extension
    figure_subdir,
    data_subdir,
    fig_format_override=None,
    fig_dpi_override=None,
    close_fig=True,
):
    """Saves a figure and an optional DataFrame, constructing paths from a base name."""
    current_fig_format = (
        fig_format_override if fig_format_override else FIGURE_FORMAT
    ).lstrip(".")
    current_fig_dpi = fig_dpi_override if fig_dpi_override else FIGURE_DPI

    # Ensure directories exist
    if figure_subdir:
        os.makedirs(figure_subdir, exist_ok=True)
    if data_subdir:
        os.makedirs(data_subdir, exist_ok=True)

    # Save the figure
    if fig_object and figure_subdir:
        plot_path = os.path.join(figure_subdir, f"{plot_basename}.{current_fig_format}")
        try:
            fig_object.savefig(
                plot_path,
                dpi=current_fig_dpi,
                format=current_fig_format,
                bbox_inches="tight",
            )
            print(f"  SUCCESS: Plot saved to {plot_path}")
        except Exception as e:
            print(f"  ERROR saving plot {plot_path}: {e}")
        finally:
            if close_fig:
                plt.close(fig_object)

    # Save the data
    if data_df_for_graphpad is not None and data_subdir is not None:
        data_path = os.path.join(data_subdir, f"{plot_basename}_data.csv")
        if isinstance(data_df_for_graphpad, pd.DataFrame):
            try:
                data_df_for_graphpad.to_csv(data_path, index=False)
                print(f"  SUCCESS: Data for GraphPad saved to {data_path}")
            except Exception as e:
                print(f"  ERROR saving data {data_path}: {e}")
        else:
            print(
                f"  WARNING: data_df_for_graphpad for '{plot_basename}' not DataFrame. Type: {type(data_df_for_graphpad)}. Not saved."
            )


print("  Function 'save_figure_and_data' defined.")


# --- 0.4.2: Function to Convert P-value to Significance Stars ---
def get_significance_stars(p_value):
    if pd.isna(p_value) or not isinstance(p_value, (int, float)):
        return "ns"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


print("  Function 'get_significance_stars' defined.")


# --- 0.4.2b: Enhanced Statistical Comparison Functions ---
def enhanced_statistical_comparison(group1_expr, group2_expr, group1_name, group2_name):
    """Enhanced statistical comparison with effect sizes and robust methods"""

    # Convert to numpy arrays and handle missing values
    g1 = np.array(group1_expr)
    g2 = np.array(group2_expr)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) == 0 or len(g2) == 0:
        return {
            "p_value": 1.0,
            "u_statistic": np.nan,
            "mean_difference": np.nan,
            "group1_mean": np.nan,
            "group2_mean": np.nan,
            "significance_stars": "ns",
        }

    # Traditional Mann-Whitney U test (keep for backward compatibility)
    try:
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    except Exception as e:
        print(f"        WARNING: Mann-Whitney U test failed: {e}")
        u_stat, p_val = np.nan, 1.0

    # Enhanced statistics with effect sizes
    if ENHANCED_QC_AVAILABLE:
        try:
            # Calculate effect sizes using enhanced functions
            effect_stats = calculate_effect_sizes(g1, g2)

            # Enhanced results dictionary
            results = {
                "p_value": p_val,
                "u_statistic": u_stat,
                "cohens_d": effect_stats["cohens_d"],
                "mean_difference": effect_stats["mean_diff"],
                "group1_mean": effect_stats["group1_mean"],
                "group2_mean": effect_stats["group2_mean"],
                "effect_interpretation": interpret_effect_size(
                    effect_stats["cohens_d"]
                ),
                "significance_stars": get_significance_stars(p_val),
            }

        except Exception as e:
            print(f"        WARNING: Enhanced effect size calculation failed: {e}")
            # Fallback to basic statistics
            results = {
                "p_value": p_val,
                "u_statistic": u_stat,
                "cohens_d": np.nan,
                "mean_difference": np.mean(g1) - np.mean(g2),
                "group1_mean": np.mean(g1),
                "group2_mean": np.mean(g2),
                "effect_interpretation": "unknown",
                "significance_stars": get_significance_stars(p_val),
            }
    else:
        # Fallback to basic statistics when enhanced QC not available
        results = {
            "p_value": p_val,
            "u_statistic": u_stat,
            "mean_difference": np.mean(g1) - np.mean(g2),
            "group1_mean": np.mean(g1),
            "group2_mean": np.mean(g2),
            "significance_stars": get_significance_stars(p_val),
        }

    return results


def interpret_effect_size(cohens_d):
    """Interpret Cohen's d effect size"""
    if np.isnan(cohens_d):
        return "unknown"

    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


print("  Enhanced statistical functions defined.")


# --- 0.4.3: Function to Check if a Gene Matches Exclusion Patterns ---
def is_gene_to_exclude_util(gene_name, patterns=GENE_PATTERNS_TO_EXCLUDE):
    if not isinstance(gene_name, str):
        return False
    for pattern in patterns:
        if re.match(pattern, gene_name, re.IGNORECASE):
            return True
    return False


print("  Function 'is_gene_to_exclude_util' defined.")


# --- 0.4.4: Helper to create descriptive filenames (FINAL) ---
def create_filename(
    base_name,
    context_name=None,
    tusc2_group=None,
    gene_set_name=None,
    plot_type_suffix=None,
    version="v1",
    ext=None,
):  # CORRECTED: ext=None is back
    """Creates a descriptive filename. If ext is provided, appends it."""
    parts = [base_name]
    if context_name:
        parts.append(
            context_name.replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
    if tusc2_group:
        parts.append(
            tusc2_group.replace(" ", "")
            .replace(TUSC2_GENE_NAME + "_", "")
            .replace(TUSC2_GENE_NAME, "TUSC2val")
        )
    if gene_set_name:
        parts.append(
            gene_set_name.replace(" ", "_").replace("/", "_").replace(":", "_")
        )
    if plot_type_suffix:
        parts.append(plot_type_suffix)
    if version:
        parts.append(version)

    cleaned_parts = [str(part) for part in parts if part is not None and str(part)]
    base_filename_out = "_".join(cleaned_parts)

    # If an extension is provided, append it to the base name.
    if ext:
        return f"{base_filename_out}.{ext.lstrip('.')}"
    else:
        return base_filename_out


print("  Function 'create_filename' (Final version, with optional ext) defined.")


# Enhanced function for intelligent marker selection with robust overlap resolution
def select_optimal_subtype_markers(
    adata,
    deg_key,
    subtypes_ordered,
    max_markers_per_subtype=4,
    pval_threshold=0.05,
    logfc_threshold=0.25,
):
    """
    Intelligently select top markers for each NK subtype with robust overlap handling.

    Key Features:
    - Prioritizes shared genes based on composite scores across subtypes
    - Ensures each subtype gets exactly max_markers_per_subtype unique genes
    - Uses sophisticated conflict resolution for overlapping top DEGs

    Parameters:
    -----------
    adata : AnnData
        Annotated data object with DEG results
    deg_key : str
        Key for DEG results in adata.uns
    subtypes_ordered : list
        Ordered list of subtypes to analyze
    max_markers_per_subtype : int
        Exact number of markers per subtype (default: 4)
    pval_threshold : float
        P-value threshold for significance (default: 0.05)
    logfc_threshold : float
        Log fold change threshold (default: 0.25)

    Returns:
    --------
    dict : Dictionary with subtype as key and list of optimal markers as value
    """
    print(
        f"      Selecting {max_markers_per_subtype} optimal markers per subtype with robust overlap resolution..."
    )

    # Step 1: Extract and score all significant DEGs for each subtype
    all_subtype_degs = {}
    gene_scores_by_subtype = {}  # Store scores for conflict resolution

    for subtype in subtypes_ordered:
        try:
            deg_df = sc.get.rank_genes_groups_df(adata, group=subtype, key=deg_key)
            deg_df_filtered = deg_df[
                ~deg_df["names"].apply(is_gene_to_exclude_util)
            ].copy()

            # Filter by significance and effect size
            significant_degs = deg_df_filtered[
                (deg_df_filtered["pvals_adj"] < pval_threshold)
                & (deg_df_filtered["logfoldchanges"] > logfc_threshold)
            ].copy()

            # Enhanced composite score combining multiple factors
            significant_degs["composite_score"] = (
                -np.log10(significant_degs["pvals_adj"] + 1e-300)
                * significant_degs["logfoldchanges"]
                * (1 + significant_degs["scores"])  # Include scanpy's internal score
            )

            # Sort by composite score
            significant_degs = significant_degs.sort_values(
                "composite_score", ascending=False
            )
            all_subtype_degs[subtype] = significant_degs

            # Store gene-score mapping for this subtype
            gene_scores_by_subtype[subtype] = dict(
                zip(significant_degs["names"], significant_degs["composite_score"])
            )

            print(
                f"        {subtype}: {len(significant_degs)} significant DEGs available"
            )

        except Exception as e:
            print(f"        WARNING: Could not extract DEGs for {subtype}: {e}")
            all_subtype_degs[subtype] = pd.DataFrame()
            gene_scores_by_subtype[subtype] = {}

    # Step 2: Build conflict resolution system for overlapping genes
    gene_subtype_conflicts = {}  # gene -> list of (subtype, score) tuples

    for gene in set().union(
        *[set(scores.keys()) for scores in gene_scores_by_subtype.values()]
    ):
        conflicts = []
        for subtype, gene_scores in gene_scores_by_subtype.items():
            if gene in gene_scores:
                conflicts.append((subtype, gene_scores[gene]))

        if len(conflicts) > 1:  # Gene appears in multiple subtypes
            gene_subtype_conflicts[gene] = sorted(
                conflicts, key=lambda x: x[1], reverse=True
            )

    print(f"        Found {len(gene_subtype_conflicts)} genes with subtype conflicts")

    # Step 3: Resolve conflicts by assigning each gene to the subtype with highest score
    assigned_genes = set()  # Track genes that have been assigned
    optimal_markers = {subtype: [] for subtype in subtypes_ordered}

    # Phase 1: Assign high-confidence genes (those with clear winners)
    for gene, conflicts in gene_subtype_conflicts.items():
        if len(conflicts) >= 2:
            winner_subtype, winner_score = conflicts[0]
            runner_up_subtype, runner_up_score = conflicts[1]

            # Only assign if winner has a substantially higher score
            score_ratio = winner_score / (runner_up_score + 1e-6)
            if score_ratio > 1.5:  # Winner has >50% higher score
                if len(optimal_markers[winner_subtype]) < max_markers_per_subtype:
                    optimal_markers[winner_subtype].append(gene)
                    assigned_genes.add(gene)
                    print(
                        f"        Assigned conflict gene {gene} to {winner_subtype} (score: {winner_score:.2f}, ratio: {score_ratio:.2f})"
                    )

    # Phase 2: Fill remaining slots with unique genes for each subtype
    for subtype in subtypes_ordered:
        if subtype not in all_subtype_degs or all_subtype_degs[subtype].empty:
            print(f"        WARNING: No DEGs available for {subtype}")
            continue

        deg_df = all_subtype_degs[subtype]

        # Add unique genes (not in conflicts) first
        for _, row in deg_df.iterrows():
            gene = row["names"]
            if (
                gene not in assigned_genes
                and gene not in gene_subtype_conflicts
                and len(optimal_markers[subtype]) < max_markers_per_subtype
            ):
                optimal_markers[subtype].append(gene)
                assigned_genes.add(gene)
                print(f"        Added unique gene {gene} to {subtype}")

        # If still need more genes, add remaining conflict genes with good scores
        if len(optimal_markers[subtype]) < max_markers_per_subtype:
            for _, row in deg_df.iterrows():
                gene = row["names"]
                if (
                    gene not in assigned_genes
                    and gene in gene_subtype_conflicts
                    and len(optimal_markers[subtype]) < max_markers_per_subtype
                ):
                    # Check if this subtype has a reasonable score for this gene
                    subtype_rank = next(
                        (
                            i
                            for i, (st, _) in enumerate(gene_subtype_conflicts[gene])
                            if st == subtype
                        ),
                        None,
                    )
                    if (
                        subtype_rank is not None and subtype_rank <= 2
                    ):  # Top 3 subtype for this gene
                        optimal_markers[subtype].append(gene)
                        assigned_genes.add(gene)
                        print(
                            f"        Added conflict gene {gene} to {subtype} (rank #{subtype_rank + 1})"
                        )

    # Phase 3: Final pass to ensure every subtype has exactly max_markers_per_subtype genes
    for subtype in subtypes_ordered:
        current_count = len(optimal_markers[subtype])
        if current_count < max_markers_per_subtype:
            shortage = max_markers_per_subtype - current_count
            print(
                f"        {subtype} needs {shortage} more genes, searching alternatives..."
            )

            # Get any remaining unassigned genes for this subtype
            if subtype in all_subtype_degs and not all_subtype_degs[subtype].empty:
                deg_df = all_subtype_degs[subtype]
                for _, row in deg_df.iterrows():
                    gene = row["names"]
                    if (
                        gene not in assigned_genes
                        and gene not in optimal_markers[subtype]
                        and len(optimal_markers[subtype]) < max_markers_per_subtype
                    ):
                        optimal_markers[subtype].append(gene)
                        assigned_genes.add(gene)
                        print(f"        Added fallback gene {gene} to {subtype}")

    # Summary
    print(f"        Final marker assignment summary:")
    total_genes = 0
    for subtype in subtypes_ordered:
        count = len(optimal_markers[subtype])
        total_genes += count
        print(
            f"        {subtype}: {count}/{max_markers_per_subtype} markers -> {optimal_markers[subtype]}"
        )

        if count < max_markers_per_subtype:
            print(
                f"        WARNING: {subtype} has only {count} markers (target: {max_markers_per_subtype})"
            )

    print(f"        Total unique genes selected: {total_genes}")
    return optimal_markers


print("  Function 'select_optimal_subtype_markers' defined.")
print("--- End of Section 0.4 (Final Corrected) ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.1: Healthy Blood NK Cohort (adata_blood)
# 1.1.1: Load Rebuffet PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad

print(
    "--- PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation ---"
)
print("  --- Section 1.1: Healthy Blood NK Cohort (adata_blood) ---")
print(
    "    --- 1.1.1: Load Rebuffet PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad ---"
)

# Initialize adata_blood to None in case of loading failure
adata_blood_source = None
adata_blood = None

try:
    # Load the source AnnData file from Rebuffet et al. (2024)
    # The path REBUFFET_H5AD_FILE was defined in Section 0.2
    print(f"      Attempting to load source Rebuffet data from: {REBUFFET_H5AD_FILE}")
    adata_blood_source = sc.read_h5ad(REBUFFET_H5AD_FILE)
    print(f"      Successfully loaded source Rebuffet file: {REBUFFET_H5AD_FILE}")
    print(
        f"      Original source adata_blood_source shape: (Cells, Genes) = {adata_blood_source.shape}"
    )
    print(
        f"      Original source .var_names example: {adata_blood_source.var_names[:5].tolist()}"
    )
    print(
        f"      Original source .obs columns example: {adata_blood_source.obs.columns[:5].tolist()}"
    )
    if adata_blood_source.raw:
        print(
            f"      Original source .raw available, shape: {adata_blood_source.raw.X.shape}"
        )
    else:
        print(
            "      WARNING: Original source .raw attribute is NOT available. This might affect downstream processing if .raw.X was expected."
        )

except FileNotFoundError:
    print(
        f"      ERROR: AnnData file not found at {REBUFFET_H5AD_FILE}. Cannot create adata_blood."
    )
except Exception as e:
    print(
        f"      ERROR: An unexpected error occurred while loading {REBUFFET_H5AD_FILE}: {e}"
    )

if adata_blood_source is not None:
    print("      adata_blood_source loaded successfully.")
else:
    print(
        "      adata_blood_source could not be loaded. Subsequent steps for adata_blood will be skipped."
    )

print("    --- End of 1.1.1 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.1: Healthy Blood NK Cohort (adata_blood)
# 1.1.2: Preprocessing & QC for adata_blood (Standardized Pipeline)

print(
    "    --- 1.1.2: Enhanced Preprocessing & QC for adata_blood (Modern Pipeline) ---"
)


def enhanced_preprocessing_pipeline(adata_source, context_name):
    """Enhanced preprocessing with modern QC best practices"""
    print(f"      Running enhanced preprocessing for {context_name}...")

    if ENHANCED_QC_AVAILABLE:
        print("        Using enhanced QC framework...")

        # Initialize enhanced QC
        enhanced_qc = AdaptiveQualityControl(
            adata_source,
            sample_key="donor" if context_name == "Blood" else "meta_patientID",
            batch_key="batch" if "batch" in adata_source.obs.columns else None,
        )

        # Calculate enhanced metrics
        print("        Calculating enhanced QC metrics...")
        enhanced_qc.adata.var["mt"] = enhanced_qc.adata.var_names.str.startswith("MT-")
        enhanced_qc.adata.var["ribo"] = enhanced_qc.adata.var_names.str.startswith(
            ("RPS", "RPL")
        )
        enhanced_qc.adata.var["hb"] = enhanced_qc.adata.var_names.str.contains(
            "^HB[^(P)]"
        )

        sc.pp.calculate_qc_metrics(
            enhanced_qc.adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
        )

        # Apply adaptive MT filtering
        enhanced_qc.adaptive_mt_filtering()

        # Enhanced doublet detection
        enhanced_qc.enhanced_doublet_detection()

        # Apply comprehensive filtering
        print("        Applying enhanced filtering criteria...")
        pre_filter_cells = enhanced_qc.adata.n_obs
        pre_filter_genes = enhanced_qc.adata.n_vars

        # Cell filtering: Remove MT outliers and doublets
        cell_filter = ~(
            enhanced_qc.adata.obs.get("mt_outlier", False)
            | enhanced_qc.adata.obs.get("doublet_consensus", False)
        )
        enhanced_qc.adata = enhanced_qc.adata[cell_filter, :].copy()

        # Gene filtering
        sc.pp.filter_genes(enhanced_qc.adata, min_cells=5)

        cells_removed = pre_filter_cells - enhanced_qc.adata.n_obs
        genes_removed = pre_filter_genes - enhanced_qc.adata.n_vars

        print(
            f"        Enhanced QC complete: removed {cells_removed} cells ({cells_removed/pre_filter_cells*100:.1f}%) and {genes_removed} genes"
        )
        print(
            f"        Final shape: {enhanced_qc.adata.n_obs} cells x {enhanced_qc.adata.n_vars} genes"
        )

        return enhanced_qc.adata

    else:
        print("        Using standard preprocessing (enhanced QC not available)...")
        return adata_source.copy()


if adata_blood_source is None:
    print(
        "      ERROR: adata_blood_source is not loaded. Cannot proceed with preprocessing."
    )
else:
    # --- Step 1: Enhanced preprocessing ---
    adata_blood_processed = enhanced_preprocessing_pipeline(adata_blood_source, "Blood")

    # --- Step 2: Handle data transformation ---
    if hasattr(adata_blood_processed, "raw") and adata_blood_processed.raw is not None:
        print(
            f"      Creating adata_blood from processed .raw.X (shape: {adata_blood_processed.raw.X.shape})"
        )
        source_X = adata_blood_processed.raw.X.copy()
        data_type = "log-normalized from .raw"
    else:
        print(
            f"      Creating adata_blood from processed .X (shape: {adata_blood_processed.X.shape})"
        )
        print(
            f"      NOTE: Dataset contains TPM-normalized data, will apply log(TPM+1) transformation"
        )
        source_X = adata_blood_processed.X.copy()
        data_type = "TPM-normalized from .X"

    print(
        f"      Using gene names from processed data (ngenes: {adata_blood_processed.n_vars})"
    )

    # --- Step 3: Create final adata_blood object ---
    adata_blood = sc.AnnData(
        X=source_X,
        obs=adata_blood_processed.obs.copy(),
        var=adata_blood_processed.var.copy(),
    )
    adata_blood.var_names_make_unique()
    adata_blood.obs_names_make_unique()
    print(f"      adata_blood created. Initial Shape: {adata_blood.shape}")

    # --- Step 4: Apply log(TPM+1) transformation if working with TPM data ---
    if data_type == "TPM-normalized from .X":
        print("      Applying log(TPM+1) transformation to TPM-normalized data...")
        adata_blood.layers["tpm"] = adata_blood.X.copy()
        adata_blood.X = np.log1p(adata_blood.X)
        print(
            f"      Log transformation complete. Expression range: {adata_blood.X.min():.3f} to {adata_blood.X.max():.3f}"
        )
        print(f"      Original TPM data stored in adata_blood.layers['tpm']")
    else:
        print(f"      Using {data_type} data as-is")

    # --- Step 5: Standardize Metadata and Filter for Valid Subtypes ---
    if REBUFFET_ORIG_SUBTYPE_COL in adata_blood.obs.columns:
        adata_blood.obs[REBUFFET_SUBTYPE_COL] = adata_blood.obs[
            REBUFFET_ORIG_SUBTYPE_COL
        ]
        adata_blood.obs[REBUFFET_SUBTYPE_COL] = pd.Categorical(
            adata_blood.obs[REBUFFET_SUBTYPE_COL],
            categories=REBUFFET_SUBTYPES_ORDERED,
            ordered=True,
        )
        original_cell_count = adata_blood.n_obs
        adata_blood = adata_blood[
            adata_blood.obs[REBUFFET_SUBTYPE_COL].notna(), :
        ].copy()
        print(f"      Standardized subtype annotations to '{REBUFFET_SUBTYPE_COL}'.")
        print(
            f"      Filtered out {original_cell_count - adata_blood.n_obs} cells with undefined subtypes."
        )
    else:
        print(
            f"      ERROR: Original subtype column '{REBUFFET_ORIG_SUBTYPE_COL}' not found."
        )

    # --- Step 6: Set the .raw attribute BEFORE gene filtering ---
    print(
        "\n      Setting adata_blood.raw to the current state (log-normalized, all genes)."
    )
    adata_blood.raw = adata_blood.copy()
    print(f"      adata_blood.raw is set. Shape: {adata_blood.raw.shape}")

    # --- Step 7: Perform Gene Filtering on the main object (.X) ---
    print(
        "\n      --- Applying Gene Filtering to adata_blood.X for dimensionality reduction ---"
    )
    original_n_vars_blood = adata_blood.n_vars
    sc.pp.filter_genes(adata_blood, min_cells=10)
    print(
        f"        Removed {original_n_vars_blood - adata_blood.n_vars} genes expressed in fewer than 10 cells from .X."
    )
    print(f"        Shape of .X after gene filtering: {adata_blood.shape}")
    print(f"        Shape of .raw remains: {adata_blood.raw.shape}")

print("    --- End of 1.1.2 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.1: Healthy Blood NK Cohort (adata_blood)
# 1.1.3: Calculate PCA, Neighbors, UMAP for adata_blood (Standardized Pipeline)

print(
    "    --- 1.1.3: Calculate PCA, Neighbors, UMAP for adata_blood (Standardized Pipeline) ---"
)

if adata_blood is None:
    print(
        "      ERROR: adata_blood is not available. Cannot perform dimensionality reduction."
    )
elif adata_blood.raw is None:
    print(
        "      ERROR: adata_blood.raw is not set. This is required for state-safe execution."
    )
else:
    # --- CRITICAL State Correction ---
    # This cell must begin with unscaled, log-normalized data in .X for valid HVG selection.
    # The following block ensures that if the notebook is run out of order, the data is reset
    # to its correct, unscaled state before proceeding.
    if adata_blood.X.min() < 0:
        print(
            f"      WARNING: adata_blood.X appears to be scaled (min value is {adata_blood.X.min():.2f})."
        )
        print(
            "      CRITICAL FIX: Restoring unscaled data from .raw for valid HVG calculation."
        )
        # Recreate the filtered, unscaled data matrix by subsetting the raw object
        # to the same genes that are currently in the main object's .var index.
        adata_blood.X = adata_blood.raw[:, adata_blood.var_names].X.copy()
        print("      SUCCESS: adata_blood.X has been restored to its unscaled state.")

    print(f"      Input adata_blood shape (post-gene-filter): {adata_blood.shape}")
    print(f"      adata_blood.raw shape (pre-gene-filter): {adata_blood.raw.shape}")
    print(
        f"      Verified adata_blood.X for HVG selection. Min: {adata_blood.X.min():.2f}, Max: {adata_blood.X.max():.2f}"
    )

    # --- Step 1: Identify Highly Variable Genes (HVGs) ---
    # Performed on the now-guaranteed unscaled, log-normalized data.
    # The 'seurat' flavor is a widely used and robust method for this purpose.
    print(
        "      Identifying Highly Variable Genes for adata_blood (flavor='seurat', subset=False)..."
    )
    sc.pp.highly_variable_genes(
        adata_blood,
        n_top_genes=1000,
        flavor="seurat",
        subset=False,  # This flags genes in .var['highly_variable'] without subsetting the AnnData object.
    )
    n_hvgs_blood = adata_blood.var["highly_variable"].sum()
    print(f"      Flagged {n_hvgs_blood} highly variable genes in adata_blood.var.")

    # --- Step 2: Scale the data in .X ---
    # Data is scaled to a zero mean and unit variance. This is a prerequisite for PCA.
    # The `max_value=10` clips extreme outliers.
    print("      Scaling adata_blood.X for PCA/UMAP steps...")
    sc.pp.scale(adata_blood, max_value=10)
    print("      adata_blood.X has been scaled (mean 0, var 1, capped at 10).")

    # --- Step 3: Principal Component Analysis (PCA) ---
    # The `use_highly_variable` argument is deprecated. The modern approach is to run PCA
    # on a view of the AnnData object containing only the flagged HVGs.
    print("      Running PCA on adata_blood (using flagged HVGs)...\n")
    sc.tl.pca(
        adata_blood,
        svd_solver="arpack",
        random_state=RANDOM_SEED,
        use_highly_variable=True,  # Retaining for this version, but will update in subsequent steps if needed.
    )
    print("      PCA calculation complete for adata_blood.")

    # Corrected plotting call for PCA variance ratio.
    try:
        # The sc.pl.pca_variance_ratio function creates its own figure.
        # We call it with show=False to prevent it from displaying prematurely.
        sc.pl.pca_variance_ratio(adata_blood, log=True, show=False, n_pcs=50)
        fig_pca_var = plt.gcf()  # Get the current figure object created by scanpy.
        ax_pca_var = fig_pca_var.get_axes()[0]  # Get the axis from the figure.
        ax_pca_var.set_title("PCA Variance Ratio (adata_blood)")
        plt.tight_layout()
        plot_basename_pca = create_filename(
            "P0_PCA_Variance_Ratio", context_name="Blood", version="v4_corrected"
        )
        save_figure_and_data(
            fig_pca_var, None, plot_basename_pca, OUTPUT_SUBDIRS["setup_figs"], None
        )
        print(f"      PCA variance ratio plot for adata_blood saved.")
    except Exception as e:
        print(f"      ERROR plotting PCA variance ratio for adata_blood: {e}")
        if "fig_pca_var" in locals() and plt.fignum_exists(fig_pca_var.number):
            plt.close(fig_pca_var)

    # --- Step 4: Compute Nearest Neighbor Graph ---
    # This graph is the foundation for clustering and UMAP.
    # The number of PCs used (N_PCS_BLOOD) is a critical parameter.
    N_PCS_BLOOD = 15
    print(
        f"      Computing nearest neighbor graph for adata_blood using {N_PCS_BLOOD} PCs..."
    )
    sc.pp.neighbors(adata_blood, n_pcs=N_PCS_BLOOD, random_state=RANDOM_SEED)
    print("      Nearest neighbor graph computation complete for adata_blood.")

    # --- Step 5: Uniform Manifold Approximation and Projection (UMAP) ---
    # This computes the 2D embedding for visualization.
    print("      Running UMAP for adata_blood...")
    sc.tl.umap(adata_blood, random_state=RANDOM_SEED, min_dist=0.3)
    print("      UMAP calculation complete for adata_blood.")
    print(
        f"      adata_blood.obsm['X_umap'] shape: {adata_blood.obsm['X_umap'].shape if 'X_umap' in adata_blood.obsm else 'Not found'}"
    )

    # It's crucial to remember that adata_blood.X is now scaled.
    # All subsequent quantitative gene expression analyses MUST use the unscaled data stored in .raw.
    print("      Dimensionality reduction pipeline for adata_blood is complete.")

print("    --- End of 1.1.3 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.2: Tang et al. Combined NK Dataset (Pre-processed Multi-Context Data)
# 1.2.1: Load Tang et al. Combined NK Dataset (comb_CD56_CD16_NK.h5ad)

print("\n  --- Section 1.2: Tang et al. Combined NK Dataset (adata_tang_full) ---")
print("    --- 1.2.1: Load Tang Combined NK Dataset (Pre-processed, Multi-Context) ---")

adata_tang_full = None

try:
    # --- Step 1: Load the Tang Combined NK Dataset ---
    print(f"      Loading Tang combined NK dataset from: {TANG_COMBINED_H5AD_FILE}")
    adata_tang_full = sc.read_h5ad(TANG_COMBINED_H5AD_FILE)
    print(
        f"      Successfully loaded Tang combined dataset. Shape: {adata_tang_full.shape}"
    )

    # --- Step 2: Dataset Overview & Validation ---
    print(f"      Dataset overview:")
    print(f"        Total cells: {adata_tang_full.n_obs:,}")
    print(f"        Total genes: {adata_tang_full.n_vars:,}")
    print(f"        Expression data type: {adata_tang_full.X.dtype}")
    print(
        f"        Expression range: {adata_tang_full.X.min():.3f} to {adata_tang_full.X.max():.3f}"
    )

    # Check for raw data availability
    if hasattr(adata_tang_full, "raw") and adata_tang_full.raw is not None:
        print(f"        Raw data available: {adata_tang_full.raw.shape}")
        print(
            f"        Raw expression range: {adata_tang_full.raw.X.min():.3f} to {adata_tang_full.raw.X.max():.3f}"
        )
    else:
        print(f"        Raw data: Not available")

    # --- Step 3: Validate Key Metadata Columns ---
    print(f"      Metadata validation:")
    available_columns = list(adata_tang_full.obs.columns)
    print(f"        Available columns: {available_columns}")

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
            print(f"        [OK] {col}: {unique_vals} unique values")
            # Show top values with counts
            value_counts = adata_tang_full.obs[col].value_counts()
            top_3_values = value_counts.head(3)
            for val, count in top_3_values.items():
                print(
                    f"          {val}: {count:,} cells ({count/adata_tang_full.n_obs*100:.1f}%)"
                )
        else:
            print(f"        [MISSING] {col}: Column not found")

    # --- FINAL VALIDATION ---
    if TUSC2_GENE_NAME in adata_tang_full.var_names:
        print(f"      SUCCESS: Gene '{TUSC2_GENE_NAME}' found in the dataset.")
    else:
        print(
            f"      FATAL WARNING: '{TUSC2_GENE_NAME}' NOT FOUND. Check gene symbol availability."
        )

except Exception as e:
    print(f"      CRITICAL ERROR during Tang data loading: {e}")
    adata_tang_full = None

if adata_tang_full is not None:
    print("      adata_tang_full successfully loaded.")
else:
    print("      adata_tang_full FAILED to load.")

print("    --- End of 1.2.1 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.2: Tang et al. Pan-Cancer Dataset (Full Dataset Preparation - adata_tang_full)
# 1.2.2: Preprocessing & QC for adata_tang_full

print("    --- 1.2.2: Preprocessing & QC for adata_tang_full (Revised Workflow) ---")

if adata_tang_full is None:
    print(
        "      ERROR: adata_tang_full is not loaded. Cannot proceed with preprocessing."
    )
else:
    print(f"      adata_tang_full initial shape: {adata_tang_full.shape}")

    # --- Step 1: Store Raw Counts ---
    # Store the original, unfiltered counts in a separate layer for safekeeping.
    print("      Storing raw counts in adata_tang_full.layers['counts']...")
    adata_tang_full.layers["counts"] = adata_tang_full.X.copy()

    # --- Step 2: Perform ONLY minimal cell filtering ---
    # We will NOT filter genes at this stage, to ensure all potential markers are available for annotation.
    min_genes_per_cell = 200
    print(f"      Filtering cells with fewer than {min_genes_per_cell} genes...")
    sc.pp.filter_cells(adata_tang_full, min_genes=min_genes_per_cell)
    print(f"        Shape after cell filtering: {adata_tang_full.shape}")

    # --- Step 3: Check if normalization is needed ---
    # The Tang combined dataset may already be processed
    max_expression = adata_tang_full.X.max()
    if max_expression > 50:  # Likely raw counts or TPM
        print(
            "      Data appears to be raw counts or TPM. Normalizing and log-transforming..."
        )
        sc.pp.normalize_total(adata_tang_full, target_sum=1e4)
        sc.pp.log1p(adata_tang_full)
        print(f"        Normalization and log-transformation complete.")
    else:
        print(
            f"      Data appears already normalized (max: {max_expression:.1f}). Skipping normalization."
        )
        print("      Using data as-is for downstream analysis.")

    # --- Step 4: Set the .raw Attribute ---
    # CRUCIAL: Set .raw to the log-normalized data that contains ALL genes.
    # This .raw object will be inherited by subsets and used for all biological scoring and analysis.
    print(
        "      Storing current log-normalized, unscaled data (with all genes) into adata_tang_full.raw"
    )
    adata_tang_full.raw = adata_tang_full.copy()
    print(f"        adata_tang_full.raw created. Shape: {adata_tang_full.raw.shape}")

    if TUSC2_GENE_NAME in adata_tang_full.var_names:
        print(f"        Gene '{TUSC2_GENE_NAME}' is PRESENT in adata_tang_full.")
    else:
        print(
            f"        WARNING: Gene '{TUSC2_GENE_NAME}' is NOT FOUND in adata_tang_full."
        )

print("    --- End of 1.2.2 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.2: Tang et al. Pan-Cancer Dataset (Full Dataset Preparation - adata_tang_full)
# 1.2.3: Finalize adata_tang_full (State-Safe)

print("    --- 1.2.3: Finalizing the master adata_tang_full object ---")

if adata_tang_full is None:
    print("      ERROR: adata_tang_full is not available. Cannot finalize.")
elif adata_tang_full.raw is None:
    print(
        "      ERROR: adata_tang_full.raw is not set. This indicates an issue in the previous step (1.2.2)."
    )
else:
    # ROBUSTNESS FIX: Reset .X from .raw to ensure we start with unscaled data,
    # regardless of the notebook's previous state.
    print("      Ensuring .X contains unscaled data by resetting from .raw...")
    adata_tang_full.X = adata_tang_full.raw.X.copy()

    # Now we can safely check the state.
    print(f"      adata_tang_full shape: {adata_tang_full.shape}")
    print(
        f"      Verified .X is unscaled. Min: {adata_tang_full.X.min():.2f}, Max: {adata_tang_full.X.max():.2f}"
    )

    # We will not run scale(), pca(), or umap() on this master object.
    # We only flag the highly variable genes, which were calculated on the unscaled data in step 1.2.2.
    # The flags are already present from the previous step, so we just confirm.
    if "highly_variable" not in adata_tang_full.var.columns:
        print("      WARNING: HVG information missing. Re-running HVG selection.")
        sc.pp.highly_variable_genes(
            adata_tang_full,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            flavor="seurat",
            subset=False,
        )
        n_hvgs_tang = adata_tang_full.var["highly_variable"].sum()
        print(f"      Flagged {n_hvgs_tang} highly variable genes.")
    else:
        n_hvgs_tang = adata_tang_full.var["highly_variable"].sum()
        print(
            f"      Confirmed {n_hvgs_tang} highly variable genes are flagged in .var."
        )

    print(f"      Master adata_tang_full is finalized and ready for subsetting.")

print("    --- End of 1.2.3 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation
# 1.3.1: Create adata_normal_tissue
# 1.3.2: Create adata_tumor_tissue & Ensure .raw Propagation

print(
    "\n  --- Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation ---"
)

# Initialize context-specific AnnData objects
adata_normal_tissue = None
adata_tumor_tissue = None

if adata_tang_full is None:
    print(
        "      ERROR: adata_tang_full is not available. Cannot create context-specific cohorts."
    )
elif METADATA_TISSUE_COLUMN_GSE212890 not in adata_tang_full.obs.columns:
    print(
        f"      ERROR: Tissue column '{METADATA_TISSUE_COLUMN_GSE212890}' not found in adata_tang_full.obs. Cannot create context-specific cohorts."
    )
else:
    # --- 1.3.1: Create adata_normal_tissue ---
    print("    --- 1.3.1: Creating adata_normal_tissue ---")
    try:
        normal_tissue_mask = (
            adata_tang_full.obs[METADATA_TISSUE_COLUMN_GSE212890] == "Normal"
        )
        if not normal_tissue_mask.any():
            print(
                f"      WARNING: No cells found with tissue type 'Normal' in '{METADATA_TISSUE_COLUMN_GSE212890}'. adata_normal_tissue will be empty or None."
            )
            adata_normal_tissue = sc.AnnData()  # Create an empty AnnData object
        else:
            adata_normal_tissue = adata_tang_full[normal_tissue_mask, :].copy()
            print(
                f"      adata_normal_tissue created. Shape: {adata_normal_tissue.shape}"
            )
            # The .raw attribute is automatically subsetted if it exists in the parent (adata_tang_full)
            if adata_normal_tissue.raw is not None:
                print(
                    f"        adata_normal_tissue.raw successfully propagated. Shape: {adata_normal_tissue.raw.shape}"
                )
                print(
                    f"        adata_normal_tissue.raw.X min: {adata_normal_tissue.raw.X.min():.2f}, max: {adata_normal_tissue.raw.X.max():.2f}"
                )
            else:
                print(
                    "        WARNING: adata_normal_tissue.raw is None. This is unexpected."
                )
    except Exception as e:
        print(f"      ERROR creating adata_normal_tissue: {e}")
        adata_normal_tissue = None

    # --- 1.3.2: Create adata_tumor_tissue ---
    print("\n    --- 1.3.2: Creating adata_tumor_tissue ---")
    try:
        tumor_tissue_mask = (
            adata_tang_full.obs[METADATA_TISSUE_COLUMN_GSE212890] == "Tumor"
        )
        if not tumor_tissue_mask.any():
            print(
                f"      WARNING: No cells found with tissue type 'Tumor' in '{METADATA_TISSUE_COLUMN_GSE212890}'. adata_tumor_tissue will be empty or None."
            )
            adata_tumor_tissue = sc.AnnData()  # Create an empty AnnData object
        else:
            adata_tumor_tissue = adata_tang_full[tumor_tissue_mask, :].copy()
            print(
                f"      adata_tumor_tissue created. Shape: {adata_tumor_tissue.shape}"
            )
            if adata_tumor_tissue.raw is not None:
                print(
                    f"        adata_tumor_tissue.raw successfully propagated. Shape: {adata_tumor_tissue.raw.shape}"
                )
                print(
                    f"        adata_tumor_tissue.raw.X min: {adata_tumor_tissue.raw.X.min():.2f}, max: {adata_tumor_tissue.raw.X.max():.2f}"
                )

            else:
                print(
                    "        WARNING: adata_tumor_tissue.raw is None. This is unexpected."
                )
    except Exception as e:
        print(f"      ERROR creating adata_tumor_tissue: {e}")
        adata_tumor_tissue = None

    # Verify that .raw for subsets contains the unscaled log-normalized data
    if (
        adata_normal_tissue is not None
        and adata_normal_tissue.n_obs > 0
        and adata_normal_tissue.raw is not None
    ):
        # adata_tang_full.X was scaled. adata_tang_full.raw.X was unscaled log-norm.
        # So, adata_normal_tissue.X will be scaled, and adata_normal_tissue.raw.X should be unscaled log-norm.
        print(
            f"      Verification: adata_normal_tissue.X max: {adata_normal_tissue.X.max():.2f} (should be scaled if parent .X was)"
        )
        print(
            f"      Verification: adata_normal_tissue.raw.X max: {adata_normal_tissue.raw.X.max():.2f} (should be unscaled log-norm)"
        )

    if (
        adata_tumor_tissue is not None
        and adata_tumor_tissue.n_obs > 0
        and adata_tumor_tissue.raw is not None
    ):
        print(
            f"      Verification: adata_tumor_tissue.X max: {adata_tumor_tissue.X.max():.2f} (should be scaled if parent .X was)"
        )
        print(
            f"      Verification: adata_tumor_tissue.raw.X max: {adata_tumor_tissue.raw.X.max():.2f} (should be unscaled log-norm)"
        )

print("    --- End of 1.3.1 & 1.3.2 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation
# 1.3.3: Process Tang Data with Original Subtypes (No Reassignment)

print("    --- 1.3.3: Process Tang Data with Original Subtypes (No Reassignment) ---")

# Note: We are no longer extracting reference markers for reassignment.
# Instead, we will use each dataset's original subtypes:
# - Rebuffet data: Keep original Rebuffet subtypes (NK2, NKint, NK1A, NK1B, NK1C, NK3)
# - Tang data: Keep original Tang subtypes (14 fine-grained subtypes)

print("      Using original subtypes for each dataset:")
print(f"        Rebuffet blood NK: {len(REBUFFET_SUBTYPES_ORDERED)} subtypes")
print(f"        Tang tissue NK: {len(TANG_SUBTYPES_ORDERED)} subtypes")
print("      No cross-dataset subtype reassignment will be performed.")

# Initialize ref_rebuffet_markers as empty dict for backward compatibility
# Some downstream sections may reference this variable, so we define it to prevent errors
ref_rebuffet_markers = {}
print("      No reference marker extraction - using original subtypes for each dataset")

# Validate that Tang tissue datasets are available for processing
if adata_normal_tissue is None or adata_normal_tissue.n_obs == 0:
    print("      INFO: adata_normal_tissue not available or empty.")
else:
    print(f"      Tang normal tissue dataset available: {adata_normal_tissue.shape}")

if adata_tumor_tissue is None or adata_tumor_tissue.n_obs == 0:
    print("      INFO: adata_tumor_tissue not available or empty.")
else:
    print(f"      Tang tumor tissue dataset available: {adata_tumor_tissue.shape}")

# Ensure Tang tissue datasets will use original Tang subtypes
for cohort_name, adata_ctx in [
    ("adata_normal_tissue", adata_normal_tissue),
    ("adata_tumor_tissue", adata_tumor_tissue),
]:
    if adata_ctx is None or adata_ctx.n_obs == 0:
        continue

    if TANG_CELLTYPE_COL in adata_ctx.obs.columns:
        # Create standardized Tang subtype column
        adata_ctx.obs[TANG_SUBTYPE_COL] = adata_ctx.obs[TANG_CELLTYPE_COL]

        # Filter for valid Tang subtypes and set as categorical
        valid_subtypes = [
            st
            for st in TANG_SUBTYPES_ORDERED
            if st in adata_ctx.obs[TANG_SUBTYPE_COL].unique()
        ]
        adata_ctx.obs[TANG_SUBTYPE_COL] = pd.Categorical(
            adata_ctx.obs[TANG_SUBTYPE_COL],
            categories=valid_subtypes,
            ordered=True,
        )

        # Filter dataset to only include cells with valid Tang subtypes
        valid_cells = adata_ctx.obs[TANG_SUBTYPE_COL].notna()
        original_count = adata_ctx.n_obs
        adata_ctx = adata_ctx[valid_cells, :].copy()

        print(f"      {cohort_name}: Using original Tang subtypes")
        print(f"        Valid Tang subtypes found: {len(valid_subtypes)}")
        print(
            f"        Kept {adata_ctx.n_obs}/{original_count} cells with valid subtypes"
        )
        print(
            f"        Subtype distribution: {adata_ctx.obs[TANG_SUBTYPE_COL].value_counts().head()}"
        )

        # Update the global variable reference
        if cohort_name == "adata_normal_tissue":
            adata_normal_tissue = adata_ctx
        elif cohort_name == "adata_tumor_tissue":
            adata_tumor_tissue = adata_ctx
    else:
        print(
            f"      WARNING: {cohort_name} missing Tang celltype column '{TANG_CELLTYPE_COL}'"
        )

print("    --- End of 1.3.3 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation
# 1.3.4: Validate Original Subtypes in Tang Data (No Reassignment)

print("    --- 1.3.4: Validate Original Subtypes in Tang Data (No Reassignment) ---")

# Validate that each dataset now has its appropriate subtype annotations
print("      Validating subtype annotations across datasets:")

# Check Rebuffet blood data
if adata_blood is not None and adata_blood.n_obs > 0:
    if REBUFFET_SUBTYPE_COL in adata_blood.obs.columns:
        rebuffet_subtypes = adata_blood.obs[REBUFFET_SUBTYPE_COL].value_counts()
        print(f"      Rebuffet blood NK data ({REBUFFET_SUBTYPE_COL}):")
        for subtype, count in rebuffet_subtypes.items():
            print(
                f"        {subtype}: {count:,} cells ({count/adata_blood.n_obs*100:.1f}%)"
            )
    else:
        print(f"      WARNING: {REBUFFET_SUBTYPE_COL} not found in adata_blood")
else:
    print("      adata_blood not available")

# Check Tang tissue data
for cohort_name, adata_ctx in [
    ("adata_normal_tissue", adata_normal_tissue),
    ("adata_tumor_tissue", adata_tumor_tissue),
]:
    if adata_ctx is None or adata_ctx.n_obs == 0:
        print(f"      {cohort_name}: Not available")
        continue

    if TANG_SUBTYPE_COL in adata_ctx.obs.columns:
        tang_subtypes = adata_ctx.obs[TANG_SUBTYPE_COL].value_counts()
        print(f"      {cohort_name} ({TANG_SUBTYPE_COL}):")
        print(f"        Total subtypes: {len(tang_subtypes)} (showing top 5)")
        for subtype, count in tang_subtypes.head().items():
            print(
                f"        {subtype}: {count:,} cells ({count/adata_ctx.n_obs*100:.1f}%)"
            )
    else:
        print(f"      WARNING: {TANG_SUBTYPE_COL} not found in {cohort_name}")

print("\n      Summary: Each dataset retains its original subtype annotations")
print("        - Rebuffet data: 6 functional NK subtypes")
print("        - Tang data: 14 fine-grained NK subtypes")
print("        - No cross-dataset reassignment performed")

print("\n    --- End of 1.3.4 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3.4a: Validate Tang Adaptive Subtype Distribution

print("    --- 1.3.4a: Validate Tang Adaptive Subtype Distribution ---")

# Validate that Tang's c8-KLRC2 (adaptive) subtype is present in our datasets
# This subtype corresponds to NKG2C+ adaptive NK cells, equivalent to Rebuffet's NK3

tang_adaptive_subtype = "CD56dimCD16hi-c8-KLRC2"
print(f"      Checking for Tang adaptive subtype: {tang_adaptive_subtype}")

for cohort_name, adata_ctx in [
    ("adata_normal_tissue", adata_normal_tissue),
    ("adata_tumor_tissue", adata_tumor_tissue),
]:
    if adata_ctx is None or adata_ctx.n_obs == 0:
        print(f"      {cohort_name}: Not available")
        continue

    if TANG_SUBTYPE_COL in adata_ctx.obs.columns:
        subtype_counts = adata_ctx.obs[TANG_SUBTYPE_COL].value_counts()

        if tang_adaptive_subtype in subtype_counts.index:
            adaptive_count = subtype_counts[tang_adaptive_subtype]
            adaptive_pct = adaptive_count / adata_ctx.n_obs * 100
            print(f"      {cohort_name}: {tang_adaptive_subtype}")
            print(f"        {adaptive_count:,} cells ({adaptive_pct:.1f}%)")

            # Show other major adaptive-related subtypes
            adaptive_related = [
                st for st in subtype_counts.index if "KLRC2" in st or "c8" in st
            ]
            if len(adaptive_related) > 1:
                print(
                    f"        Related adaptive subtypes: {len(adaptive_related)} found"
                )
        else:
            print(f"      {cohort_name}: {tang_adaptive_subtype} not found")
            # Show what adaptive-like subtypes are present
            adaptive_like = [
                st for st in subtype_counts.index if "c8" in st or "KLRC2" in st
            ]
            if adaptive_like:
                print(f"        Similar subtypes found: {adaptive_like}")
    else:
        print(f"      {cohort_name}: Tang subtype column not available")

print(
    "\n      Note: Tang c8-KLRC2 corresponds to adaptive NK cells (similar to Rebuffet NK3)"
)
print(
    "      This validation confirms presence of adaptive NK populations in tissue data"
)

print("\n    --- End of 1.3.4a ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation
# 1.3.5: Calculate PCA, Neighbors, UMAP for Tang et al. Cohorts (Modernized Pipeline v4 - Final)

print(
    "    --- 1.3.5: Calculate PCA, Neighbors, UMAP for Tang et al. Cohorts (Modernized Pipeline v4 - Final) ---"
)


def run_dim_reduction_pipeline(adata_obj, cohort_label, n_pcs_to_use=15, n_hvgs=1000):
    """
    Performs a standard, robust, and idempotent dimensionality reduction workflow on a given AnnData object.
    This modernized pipeline ensures state-safe data handling and uses current best practices.
    """
    if adata_obj is None or adata_obj.n_obs == 0:
        print(
            f"      Skipping dimensionality reduction for '{cohort_label}' as it is None or empty."
        )
        return None

    print(
        f"\n      --- Running Dim Reduction Pipeline for: {cohort_label} (Initial Shape: {adata_obj.shape}) ---"
    )

    # --- Step 0: State-Safe Data Restoration ---
    # Ensures that .X contains unscaled data before any processing. This is critical if the cell
    # has been run before, as it resets the scaled data in .X back to unscaled log-normalized counts.
    if adata_obj.X.min() < -0.001:
        print(
            f"        WARNING: Input .X for {cohort_label} appears scaled. Restoring from .raw..."
        )
        if adata_obj.raw is not None:
            # Correctly subsets the raw matrix to match the genes currently in .var
            adata_obj.X = adata_obj.raw[:, adata_obj.var_names].X.copy()
            print("          SUCCESS: .X restored to unscaled, log-normalized data.")
        else:
            print(
                f"          ERROR: Cannot restore state for {cohort_label} as .raw is missing. Aborting."
            )
            return None

    # --- Step 1: Gene Filtering for this specific cohort ---
    # Reduces dimensionality for subsequent steps. Does not affect the .raw data.
    print("        Filtering genes with low expression in this specific cohort...")
    original_n_vars = adata_obj.n_vars
    sc.pp.filter_genes(adata_obj, min_cells=10)
    print(
        f"        Removed {original_n_vars - adata_obj.n_vars} genes. New shape for .X: {adata_obj.shape}"
    )

    # --- Step 2: Identify Highly Variable Genes ---
    # This must be performed on unscaled, log-normalized data.
    print(f"        Finding top {n_hvgs} HVGs...")
    sc.pp.highly_variable_genes(
        adata_obj,
        n_top_genes=n_hvgs,
        flavor="seurat",
        subset=False,  # Flags genes in .var['highly_variable']
    )
    n_hvgs_found = adata_obj.var["highly_variable"].sum()
    print(f"        Flagged {n_hvgs_found} highly variable genes.")

    # --- Step 3: Scale Data ---
    # Scales the .X matrix to zero mean and unit variance. Required for PCA.
    print("        Scaling data for PCA...")
    sc.pp.scale(adata_obj, max_value=10)

    # --- Step 4: Principal Component Analysis (PCA) ---
    # This is the final corrected call using the 'mask_var' argument as recommended
    # by the FutureWarning. This is the modern replacement for 'use_highly_variable=True'.
    print("        Running PCA on highly variable genes...")
    sc.tl.pca(
        adata_obj,
        svd_solver="arpack",
        random_state=RANDOM_SEED,
        mask_var="highly_variable",  # Correct, modern way to use the HVG boolean mask.
    )

    # --- Step 5: Enhanced PC Selection (Data-Driven) ---
    if n_pcs_to_use is None:
        # Use data-driven PC selection
        variance_ratios = adata_obj.uns["pca"]["variance_ratio"]
        cumsum_var = np.cumsum(variance_ratios)

        # Method 1: 80% variance threshold
        n_pcs_variance = np.argmax(cumsum_var >= 0.80) + 1

        # Method 2: Elbow method (simple version)
        diffs = np.diff(variance_ratios[: min(50, len(variance_ratios))])
        elbow_point = np.argmax(diffs > -0.01) + 2

        # Conservative selection (median of methods)
        optimal_n_pcs = int(
            np.median([n_pcs_variance, elbow_point, 15])
        )  # 15 as fallback
        optimal_n_pcs = max(10, min(optimal_n_pcs, 50))  # Constrain between 10-50

        print(f"        📊 Data-driven PC selection:")
        print(f"          - 80% variance: {n_pcs_variance} PCs")
        print(f"          - Elbow method: {elbow_point} PCs")
        print(
            f"          - Selected: {optimal_n_pcs} PCs ({cumsum_var[optimal_n_pcs-1]:.1%} variance)"
        )
        actual_n_pcs = optimal_n_pcs
    else:
        # Use specified number of PCs
        actual_n_pcs = min(n_pcs_to_use, adata_obj.obsm["X_pca"].shape[1])
        if actual_n_pcs < n_pcs_to_use:
            print(f"          Using {actual_n_pcs} PCs as it's the maximum available.")

    # --- Step 6: Nearest Neighbor Graph ---
    print(f"        Computing nearest neighbor graph using top {actual_n_pcs} PCs...")
    sc.pp.neighbors(adata_obj, n_pcs=actual_n_pcs, random_state=RANDOM_SEED)

    # --- Step 6: Uniform Manifold Approximation and Projection (UMAP) ---
    print("        Running UMAP...")
    sc.tl.umap(adata_obj, random_state=RANDOM_SEED, min_dist=0.3)

    print(f"      --- Pipeline complete for: {cohort_label} ---")
    return adata_obj


# --- Execute the Pipeline on Each Cohort ---
adata_normal_tissue = run_dim_reduction_pipeline(adata_normal_tissue, "NormalTissue")
adata_tumor_tissue = run_dim_reduction_pipeline(adata_tumor_tissue, "TumorTissue")

print("\n    --- End of Section 1.3.5 ---\n")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation
# 1.3.6: Save Processed AnnData Objects

print("    --- 1.3.6: Saving Processed AnnData Objects ---")

# Ensure the output directory for processed AnnData objects exists
processed_anndata_dir = OUTPUT_SUBDIRS["processed_anndata"]  # Defined in 0.2
os.makedirs(processed_anndata_dir, exist_ok=True)
print(f"      Output directory for processed AnnData: {processed_anndata_dir}")

# Save adata_blood
if adata_blood is not None and adata_blood.n_obs > 0:
    try:
        adata_blood_path = os.path.join(
            processed_anndata_dir, "adata_blood_processed.h5ad"
        )
        adata_blood.write_h5ad(adata_blood_path, compression="gzip")
        print(f"      SUCCESS: adata_blood saved to {adata_blood_path}")
        print(
            f"        adata_blood details: Shape={adata_blood.shape}, .raw shape={adata_blood.raw.shape if adata_blood.raw else 'None'}"
        )
        print(
            f"        adata_blood .obs columns example: {adata_blood.obs.columns.tolist()[:5]}"
        )
        subtype_col = get_subtype_column(adata_blood)
        if subtype_col and subtype_col in adata_blood.obs:
            print(
                f"        adata_blood '{subtype_col}' counts:\n{adata_blood.obs[subtype_col].value_counts().sort_index()}"
            )
    except Exception as e:
        print(f"      ERROR saving adata_blood: {e}")
else:
    print("      adata_blood is None or empty. Skipping save.")

# Save adata_normal_tissue
if adata_normal_tissue is not None and adata_normal_tissue.n_obs > 0:
    try:
        adata_normal_tissue_path = os.path.join(
            processed_anndata_dir, "adata_normal_tissue_processed.h5ad"
        )
        adata_normal_tissue.write_h5ad(adata_normal_tissue_path, compression="gzip")
        print(f"      SUCCESS: adata_normal_tissue saved to {adata_normal_tissue_path}")
        print(
            f"        adata_normal_tissue details: Shape={adata_normal_tissue.shape}, .raw shape={adata_normal_tissue.raw.shape if adata_normal_tissue.raw else 'None'}"
        )
        print(
            f"        adata_normal_tissue .obs columns example: {adata_normal_tissue.obs.columns.tolist()[:5]}"
        )
        subtype_col = get_subtype_column(adata_normal_tissue)
        if subtype_col and subtype_col in adata_normal_tissue.obs:
            print(
                f"        adata_normal_tissue '{subtype_col}' counts:\n{adata_normal_tissue.obs[subtype_col].value_counts().sort_index()}"
            )
    except Exception as e:
        print(f"      ERROR saving adata_normal_tissue: {e}")
else:
    print("      adata_normal_tissue is None or empty. Skipping save.")

# Save adata_tumor_tissue
if adata_tumor_tissue is not None and adata_tumor_tissue.n_obs > 0:
    try:
        adata_tumor_tissue_path = os.path.join(
            processed_anndata_dir, "adata_tumor_tissue_processed.h5ad"
        )
        adata_tumor_tissue.write_h5ad(adata_tumor_tissue_path, compression="gzip")
        print(f"      SUCCESS: adata_tumor_tissue saved to {adata_tumor_tissue_path}")
        print(
            f"        adata_tumor_tissue details: Shape={adata_tumor_tissue.shape}, .raw shape={adata_tumor_tissue.raw.shape if adata_tumor_tissue.raw else 'None'}"
        )
        print(
            f"        adata_tumor_tissue .obs columns example: {adata_tumor_tissue.obs.columns.tolist()[:5]}"
        )
        subtype_col = get_subtype_column(adata_tumor_tissue)
        if subtype_col and subtype_col in adata_tumor_tissue.obs:
            print(
                f"        adata_tumor_tissue '{subtype_col}' counts:\n{adata_tumor_tissue.obs[subtype_col].value_counts().sort_index()}"
            )
    except Exception as e:
        print(f"      ERROR saving adata_tumor_tissue: {e}")
else:
    print("      adata_tumor_tissue is None or empty. Skipping save.")

print("    --- End of 1.3.6 ---")
print(
    "\n--- END OF PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation ---"
)

# %%
# PART 1.4: Dynamic Generation of Developmental Signatures from Rebuffet Blood NK Data

print("\n--- PART 1.4: Dynamic Generation of Developmental Signatures ---")


def generate_rebuffet_developmental_signatures(adata_blood_ref, top_n_genes=50):
    """
    Generate developmental gene signatures from Rebuffet blood NK subtypes.

    Parameters:
    -----------
    adata_blood_ref : AnnData
        Blood NK data with Rebuffet subtypes
    top_n_genes : int
        Number of top upregulated genes per subtype to include

    Returns:
    --------
    dict : Dictionary with subtype names as keys and gene lists as values
    """
    print(
        "  Generating dynamic developmental signatures from Rebuffet blood NK subtypes..."
    )

    if adata_blood_ref is None or adata_blood_ref.n_obs == 0:
        print("    ERROR: No valid blood NK data available. Using fallback signatures.")
        return {
            "NK2_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "NKint_Intermediate": ["CD27", "GZMK", "KLRB1", "CD7"],
            "NK1A_Mature": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "NK1B_Cytotoxic": ["GZMA", "GZMH", "KLRD1", "FCGR3A"],
            "NK1C_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "NK3_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }

    # Get the appropriate subtype column
    subtype_col = get_subtype_column(adata_blood_ref)

    if not subtype_col or subtype_col not in adata_blood_ref.obs.columns:
        print(
            "    ERROR: No valid subtype column found in blood data. Using fallback signatures."
        )
        return {
            "NK2_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "NKint_Intermediate": ["CD27", "GZMK", "KLRB1", "CD7"],
            "NK1A_Mature": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "NK1B_Cytotoxic": ["GZMA", "GZMH", "KLRD1", "FCGR3A"],
            "NK1C_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "NK3_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }

    # Create a copy for DEG analysis, filtering to assigned cells only
    if subtype_col == REBUFFET_SUBTYPE_COL:
        assigned_mask = adata_blood_ref.obs[subtype_col] != "Unassigned"
        adata_deg = adata_blood_ref[assigned_mask, :].copy()
    else:
        adata_deg = adata_blood_ref.copy()

    if adata_deg.n_obs == 0 or adata_deg.obs[subtype_col].nunique() < 2:
        print(
            "    ERROR: Not enough cells or subtypes for DEG analysis. Using fallback signatures."
        )
        return {
            "NK2_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "NKint_Intermediate": ["CD27", "GZMK", "KLRB1", "CD7"],
            "NK1A_Mature": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "NK1B_Cytotoxic": ["GZMA", "GZMH", "KLRD1", "FCGR3A"],
            "NK1C_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "NK3_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }

    print(f"    Running DEG analysis on {adata_deg.n_obs} blood NK cells...")
    print(f"    Subtypes found: {adata_deg.obs[subtype_col].value_counts().to_dict()}")

    # Run DEG analysis
    try:
        sc.tl.rank_genes_groups(
            adata_deg,
            groupby=subtype_col,
            method="wilcoxon",
            use_raw=True,
            pts=True,
            corr_method="benjamini-hochberg",
            n_genes=top_n_genes + 100,  # Get extra genes for filtering
            key_added="dynamic_dev_markers",
        )

        print("    DEG analysis completed. Extracting top markers...")

        # Extract top markers for each subtype
        developmental_signatures = {}
        available_subtypes = adata_deg.obs[subtype_col].cat.categories

        for subtype in available_subtypes:
            try:
                # Get DEG results for this subtype
                deg_df = sc.get.rank_genes_groups_df(
                    adata_deg, group=subtype, key="dynamic_dev_markers"
                )

                if deg_df is not None and not deg_df.empty:
                    # Filter genes and select top markers
                    filtered_genes = deg_df[
                        (~deg_df["names"].apply(is_gene_to_exclude_util))
                        & (deg_df["pvals_adj"] < 0.05)
                        & (deg_df["logfoldchanges"] > 0.25)  # Only upregulated genes
                    ]

                    top_genes = filtered_genes.head(top_n_genes)["names"].tolist()

                    # Clean up subtype name for the signature
                    clean_subtype_name = f"{subtype}_Developmental"
                    developmental_signatures[clean_subtype_name] = top_genes

                    print(
                        f"      {subtype}: {len(top_genes)} signature genes extracted"
                    )
                else:
                    print(f"      {subtype}: No DEG results found")
                    developmental_signatures[f"{subtype}_Developmental"] = []

            except Exception as e:
                print(f"      ERROR extracting markers for {subtype}: {e}")
                developmental_signatures[f"{subtype}_Developmental"] = []

        print(
            f"    Successfully generated {len(developmental_signatures)} developmental signatures"
        )
        return developmental_signatures

    except Exception as e:
        print(f"    ERROR during DEG analysis: {e}")
        print("    Using fallback signatures...")
        return {
            "NK2_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "NKint_Intermediate": ["CD27", "GZMK", "KLRB1", "CD7"],
            "NK1A_Mature": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "NK1B_Cytotoxic": ["GZMA", "GZMH", "KLRD1", "FCGR3A"],
            "NK1C_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "NK3_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }


def generate_tang_developmental_signatures(adata_tang_ref, top_n_genes=50):
    """
    Generate developmental gene signatures from Tang tissue NK subtypes.

    Parameters:
    -----------
    adata_tang_ref : AnnData
        Tang NK data with Tang subtypes (either tissue data)
    top_n_genes : int
        Number of top upregulated genes per subtype to include

    Returns:
    --------
    dict : Dictionary with subtype names as keys and gene lists as values
    """
    print(
        "  Generating dynamic developmental signatures from Tang NK subtypes..."
    )

    if adata_tang_ref is None or adata_tang_ref.n_obs == 0:
        print("    ERROR: No valid Tang NK data available. Using fallback signatures.")
        return {
            "Tang_CD56bright_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "Tang_CD56dim_Cytotoxic": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "Tang_CD56dim_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "Tang_CD56dim_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }

    # Get the appropriate subtype column
    subtype_col = get_subtype_column(adata_tang_ref)

    if not subtype_col or subtype_col not in adata_tang_ref.obs.columns:
        print(
            "    ERROR: No valid subtype column found in Tang data. Using fallback signatures."
        )
        return {
            "Tang_CD56bright_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "Tang_CD56dim_Cytotoxic": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "Tang_CD56dim_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "Tang_CD56dim_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }

    # Create a copy for DEG analysis, filtering to assigned cells only
    # For Tang data, we want cells with actual subtype annotations (not "Unassigned")
    assigned_mask = ~adata_tang_ref.obs[subtype_col].isin(["Unassigned", "Unknown", ""])
    adata_deg = adata_tang_ref[assigned_mask, :].copy()

    if adata_deg.n_obs == 0 or adata_deg.obs[subtype_col].nunique() < 2:
        print(
            "    ERROR: Not enough cells or subtypes for DEG analysis. Using fallback signatures."
        )
        return {
            "Tang_CD56bright_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "Tang_CD56dim_Cytotoxic": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "Tang_CD56dim_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "Tang_CD56dim_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }

    print(f"    Running DEG analysis on {adata_deg.n_obs} Tang NK cells...")
    print(f"    Subtypes found: {adata_deg.obs[subtype_col].value_counts().to_dict()}")

    # Run DEG analysis
    try:
        sc.tl.rank_genes_groups(
            adata_deg,
            groupby=subtype_col,
            method="wilcoxon",
            use_raw=True,
            pts=True,
            corr_method="benjamini-hochberg",
            n_genes=top_n_genes + 100,  # Get extra genes for filtering
            key_added="dynamic_tang_markers",
        )

        print("    DEG analysis completed. Extracting top markers...")

        # Extract top markers for each subtype
        tang_signatures = {}
        available_subtypes = adata_deg.obs[subtype_col].cat.categories

        for subtype in available_subtypes:
            try:
                # Get DEG results for this subtype
                deg_df = sc.get.rank_genes_groups_df(
                    adata_deg, group=subtype, key="dynamic_tang_markers"
                )

                if deg_df is not None and not deg_df.empty:
                    # Filter genes and select top markers
                    filtered_genes = deg_df[
                        (~deg_df["names"].apply(is_gene_to_exclude_util))
                        & (deg_df["pvals_adj"] < 0.05)
                        & (deg_df["logfoldchanges"] > 0.25)  # Only upregulated genes
                    ]

                    top_genes = filtered_genes.head(top_n_genes)["names"].tolist()

                    # Clean up subtype name for the signature
                    clean_subtype_name = f"Tang_{subtype}_Developmental"
                    tang_signatures[clean_subtype_name] = top_genes

                    print(
                        f"      {subtype}: {len(top_genes)} signature genes extracted"
                    )
                else:
                    print(f"      {subtype}: No DEG results found")
                    tang_signatures[f"Tang_{subtype}_Developmental"] = []

            except Exception as e:
                print(f"      ERROR extracting markers for {subtype}: {e}")
                tang_signatures[f"Tang_{subtype}_Developmental"] = []

        print(
            f"    Successfully generated {len(tang_signatures)} Tang developmental signatures"
        )
        return tang_signatures

    except Exception as e:
        print(f"    ERROR during DEG analysis: {e}")
        print("    Using fallback signatures...")
        return {
            "Tang_CD56bright_Regulatory": ["SELL", "TCF7", "IL7R", "CCR7"],
            "Tang_CD56dim_Cytotoxic": ["GNLY", "NKG7", "GZMB", "PRF1"],
            "Tang_CD56dim_Terminal": ["KLRC2", "KLRG1", "CX3CR1", "HAVCR2"],
            "Tang_CD56dim_Adaptive": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
        }


# Generate dynamic developmental signatures from blood NK data
if "adata_blood" in locals() and adata_blood is not None:
    DEVELOPMENTAL_GENE_SETS = generate_rebuffet_developmental_signatures(
        adata_blood, top_n_genes=50
    )
    print(f"  Generated {len(DEVELOPMENTAL_GENE_SETS)} dynamic developmental gene sets")
    for sig_name, genes in DEVELOPMENTAL_GENE_SETS.items():
        print(f"    {sig_name}: {len(genes)} genes")
else:
    print(
        "  WARNING: No blood NK data available. Using static developmental signatures."
    )
    # Keep the original static signatures as fallback
    DEVELOPMENTAL_GENE_SETS = {
        "Regulatory_NK": ["SELL", "TCF7", "IL7R", "CCR7"],
        "Intermediate_NK": ["CD27", "GZMK", "KLRB1", "CD7"],
        "Mature_Cytotoxic_NK": ["GNLY", "NKG7", "GZMB", "PRF1"],
        "Adaptive_NK": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
    }

# Generate dynamic Tang developmental signatures from Tang tissue NK data
# Priority: tumor tissue > normal tissue (use whichever is available)
TANG_DEVELOPMENTAL_GENE_SETS = {}
if "adata_tumor_tissue" in locals() and adata_tumor_tissue is not None and adata_tumor_tissue.n_obs > 0:
    if should_split_tang_subtypes(adata_tumor_tissue):
        print("  Using tumor tissue Tang data for signature generation...")
        TANG_DEVELOPMENTAL_GENE_SETS = generate_tang_developmental_signatures(
            adata_tumor_tissue, top_n_genes=50
        )
elif "adata_normal_tissue" in locals() and adata_normal_tissue is not None and adata_normal_tissue.n_obs > 0:
    if should_split_tang_subtypes(adata_normal_tissue):
        print("  Using normal tissue Tang data for signature generation...")
        TANG_DEVELOPMENTAL_GENE_SETS = generate_tang_developmental_signatures(
            adata_normal_tissue, top_n_genes=50
        )

if TANG_DEVELOPMENTAL_GENE_SETS:
    print(f"  Generated {len(TANG_DEVELOPMENTAL_GENE_SETS)} dynamic Tang developmental gene sets")
    for sig_name, genes in TANG_DEVELOPMENTAL_GENE_SETS.items():
        print(f"    {sig_name}: {len(genes)} genes")
else:
    print("  No Tang data available for signature generation.")

# Create combined dictionary for backward compatibility
ALL_FUNCTIONAL_GENE_SETS = {
    **DEVELOPMENTAL_GENE_SETS,
    **TANG_DEVELOPMENTAL_GENE_SETS,
    **FUNCTIONAL_GENE_SETS,
    **NEUROTRANSMITTER_RECEPTOR_GENE_SETS,
    **INTERLEUKIN_DOWNSTREAM_GENE_SETS,
}
print(
    f"  Created combined ALL_FUNCTIONAL_GENE_SETS with {len(ALL_FUNCTIONAL_GENE_SETS)} total gene sets"
)

print("--- END OF PART 1.4 ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# This part iterates for each adata_ctx in [adata_blood, adata_normal_tissue, adata_tumor_tissue].
# All expression analyses use .raw.X to ensure we are using the unscaled, log-normalized data.

print("\n--- PART 2: Baseline Characterization of NK Subtypes within Each Context ---")

# Define the list of AnnData objects and their context names
cohorts_for_characterization = []
if "adata_blood" in locals() and adata_blood is not None and adata_blood.n_obs > 0:
    cohorts_for_characterization.append(
        ("Blood", adata_blood, OUTPUT_SUBDIRS["blood_nk_char"])
    )
if (
    "adata_normal_tissue" in locals()
    and adata_normal_tissue is not None
    and adata_normal_tissue.n_obs > 0
):
    cohorts_for_characterization.append(
        ("NormalTissue", adata_normal_tissue, OUTPUT_SUBDIRS["normal_tissue_nk_char"])
    )
if (
    "adata_tumor_tissue" in locals()
    and adata_tumor_tissue is not None
    and adata_tumor_tissue.n_obs > 0
):
    cohorts_for_characterization.append(
        ("TumorTissue", adata_tumor_tissue, OUTPUT_SUBDIRS["tumor_tissue_nk_char"])
    )

# Loop through each context to perform a standardized set of characterization analyses.
for context_name, adata_ctx, context_output_base_dir in cohorts_for_characterization:
    print(f"\n  --- Processing Context: {context_name} ---")

    # Define output subdirectories.
    ctx_fig_dir = os.path.join(context_output_base_dir, "figures")
    ctx_data_dir = os.path.join(context_output_base_dir, "data_for_graphpad")
    os.makedirs(ctx_fig_dir, exist_ok=True)
    os.makedirs(ctx_data_dir, exist_ok=True)

    # Section 2.1: Composition and Visual Overview
    print(
        f"    --- Section 2.1: Composition and Visual Overview for {context_name} ---"
    )

    # Get the appropriate subtype column for this dataset
    subtype_col = get_subtype_column(adata_ctx)
    subtype_categories = get_subtype_categories(adata_ctx)

    if not subtype_col or subtype_col not in adata_ctx.obs.columns:
        print(
            f"      ERROR: Subtype column '{subtype_col}' not found. Skipping Section 2.1."
        )
        continue

    # --- Create a view of the data containing cells with valid subtype assignments ---
    # For Tang data, we don't have "Unassigned" - all cells should have valid subtypes
    if subtype_col == TANG_SUBTYPE_COL:
        assigned_mask = adata_ctx.obs[subtype_col].notna()
    else:
        assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
    adata_view_assigned = adata_ctx[assigned_mask, :].copy()

    if adata_view_assigned.n_obs == 0:
        print(
            f"      No cells with assigned subtypes found for {context_name}. Skipping visualization."
        )
        continue

    print(
        f"      Analyzing {adata_view_assigned.n_obs} confidently assigned cells (out of {adata_ctx.n_obs} total)."
    )

    # Ensure the subtype column in the view is categorical with the correct order for plotting
    ordered_categories_for_plot = [
        cat
        for cat in subtype_categories
        if cat in adata_view_assigned.obs[subtype_col].unique()
    ]
    adata_view_assigned.obs[subtype_col] = pd.Categorical(
        adata_view_assigned.obs[subtype_col],
        categories=ordered_categories_for_plot,
        ordered=True,
    )

    # 2.1.1: Bar plot of NK_Subtype_Profiled proportions (for assigned cells)
    print(
        f"      --- 2.1.1: Bar plot of assigned subtype proportions for {context_name} ---"
    )
    try:
        subtype_counts = (
            adata_view_assigned.obs[subtype_col].value_counts().sort_index()
        )
        subtype_proportions = (
            adata_view_assigned.obs[subtype_col]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        )

        fig_bar_comp, ax_bar_comp = plt.subplots(figsize=(8, 6))
        colors_for_plot = [
            COMBINED_SUBTYPE_COLOR_PALETTE.get(cat, "#CCCCCC")
            for cat in subtype_proportions.index
        ]
        sns.barplot(
            x=subtype_proportions.index,
            y=subtype_proportions.values,
            ax=ax_bar_comp,
            palette=colors_for_plot,
            hue=subtype_proportions.index,
            legend=False,
        )

        ax_bar_comp.set_title(
            f"Composition of Assigned NK Subtypes in {context_name}", fontsize=14
        )
        ax_bar_comp.set_xlabel(f"Assigned NK Subtype", fontsize=12)
        ax_bar_comp.set_ylabel("Proportion of Assigned Cells (%)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        composition_df_export = pd.DataFrame(
            {
                "Subtype": subtype_counts.index,
                "Cell_Count": subtype_counts.values,
                "Proportion_Pct": subtype_proportions.values,
            }
        )
        plot_basename_bar = create_filename(
            f"P2_1_Barplot_AssignedSubtypeComposition",
            context_name=context_name,
            version="v3_relaxed",
        )
        save_figure_and_data(
            fig_bar_comp,
            composition_df_export,
            plot_basename_bar,
            ctx_fig_dir,
            ctx_data_dir,
        )
    except Exception as e:
        print(f"        ERROR generating subtype composition bar plot: {e}")
        if "fig_bar_comp" in locals() and plt.fignum_exists(fig_bar_comp.number):
            plt.close(fig_bar_comp)

    # 2.1.2: UMAP colored by subtypes
    print(f"\n      --- 2.1.2: UMAP colored by subtypes for {context_name} ---")

    # Get the appropriate subtype column for this dataset
    subtype_col = get_subtype_column(adata_ctx)

    if not subtype_col or subtype_col not in adata_ctx.obs.columns:
        print(
            f"        WARNING: No valid subtype column found for {context_name}. Skipping UMAP plot."
        )
    elif "X_umap" not in adata_ctx.obsm:
        print(
            f"        WARNING: 'X_umap' not found in {context_name}.obsm. Skipping UMAP plot."
        )
    else:
        try:
            # Use the full adata object for the UMAP plot
            fig_umap_subtype, ax_umap_subtype = plt.subplots(figsize=(10, 7))
            sc.pl.umap(
                adata_ctx,
                color=subtype_col,
                ax=ax_umap_subtype,
                show=False,
                legend_loc="right margin",
                legend_fontsize=8,
                title=f"UMAP of {context_name} by Subtype",
                palette=COMBINED_SUBTYPE_COLOR_PALETTE,
            )
            plt.subplots_adjust(right=0.75)  # Adjust plot to make space for the legend

            umap_coords_df = pd.DataFrame(
                adata_ctx.obsm["X_umap"],
                columns=["UMAP1", "UMAP2"],
                index=adata_ctx.obs_names,
            )
            graphpad_umap_data = (
                adata_ctx.obs[[subtype_col]]
                .join(umap_coords_df)
                .reset_index()
                .rename(columns={"index": "CellID"})
            )
            plot_basename_umap = create_filename(
                f"P2_1_UMAP_by_Subtype_with_Unassigned",
                context_name=context_name,
                version="v3_relaxed",
            )
            save_figure_and_data(
                fig_umap_subtype,
                graphpad_umap_data,
                plot_basename_umap,
                ctx_fig_dir,
                ctx_data_dir,
            )
        except Exception as e:
            print(f"        ERROR generating UMAP by subtype for {context_name}: {e}")
            if "fig_umap_subtype" in locals() and plt.fignum_exists(
                fig_umap_subtype.number
            ):
                plt.close(fig_umap_subtype)

    print(f"    --- End of Section 2.1 for {context_name} ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# Section 2.1: Composition and Visual Overview (Continued)
# 2.1.3: Cross-tabulations (vs. Majortype) and Histology Distributions

print(
    "    --- Section 2.1.3: Cross-tabulations (vs. Majortype) and Histology Distributions ---"
)

# This section validates our new subtype assignments against the original annotations from the Tang et al. dataset.
tang_cohorts_for_crosstab = []
if (
    "adata_normal_tissue" in locals()
    and adata_normal_tissue is not None
    and adata_normal_tissue.n_obs > 0
):
    tang_cohorts_for_crosstab.append(
        ("NormalTissue", adata_normal_tissue, OUTPUT_SUBDIRS["normal_tissue_nk_char"])
    )
if (
    "adata_tumor_tissue" in locals()
    and adata_tumor_tissue is not None
    and adata_tumor_tissue.n_obs > 0
):
    tang_cohorts_for_crosstab.append(
        ("TumorTissue", adata_tumor_tissue, OUTPUT_SUBDIRS["tumor_tissue_nk_char"])
    )

for context_name, adata_ctx, context_output_base_dir in tang_cohorts_for_crosstab:
    print(f"\n     --- Processing 2.1.3 for Context: {context_name} ---")

    ctx_fig_dir = os.path.join(context_output_base_dir, "figures")
    ctx_data_dir = os.path.join(context_output_base_dir, "data_for_graphpad")
    ctx_stats_dir = os.path.join(context_output_base_dir, "stat_results_python")
    os.makedirs(ctx_fig_dir, exist_ok=True)
    os.makedirs(ctx_data_dir, exist_ok=True)
    os.makedirs(ctx_stats_dir, exist_ok=True)

    # Get the appropriate subtype column for this dataset
    subtype_col = get_subtype_column(adata_ctx)

    if not subtype_col or subtype_col not in adata_ctx.obs.columns:
        print(
            f"        ERROR: Subtype column '{subtype_col}' not found. Skipping crosstabs."
        )
        continue

    # For Rebuffet data, exclude "Unassigned"; for Tang data, use all cells (no "Unassigned" category)
    if subtype_col == REBUFFET_SUBTYPE_COL:
        adata_view_assigned = adata_ctx[adata_ctx.obs[subtype_col] != "Unassigned", :]
    else:
        # Tang data - all cells should have valid subtypes
        adata_view_assigned = adata_ctx.copy()
    if adata_view_assigned.n_obs == 0:
        print(f"        No cells with assigned subtypes found. Skipping this section.")
        continue
    print(
        f"        Analyzing {adata_view_assigned.n_obs} assigned cells for crosstabulations."
    )

    # --- MODIFIED: Cross-tabulation with original Tang `Majortype` (broad categories) ---
    if METADATA_MAJORTYPE_COLUMN_GSE212890 in adata_view_assigned.obs.columns:
        print(
            f"        Cross-tab: Assigned Subtypes vs. Tang '{METADATA_MAJORTYPE_COLUMN_GSE212890}'"
        )

        # Using the original, robust crosstab creation method
        crosstab_majortype = pd.crosstab(
            adata_view_assigned.obs[subtype_col],
            adata_view_assigned.obs[METADATA_MAJORTYPE_COLUMN_GSE212890]
            .astype(str)
            .fillna("Unknown_Majortype"),
            dropna=False,
        )

        ct_majortype_filename = create_filename(
            "P2_1_Crosstab_vs_TangMajortype",
            context_name=context_name,
            version="v1",
            ext="csv",
        )
        crosstab_majortype.to_csv(os.path.join(ctx_stats_dir, ct_majortype_filename))
        print(f"          Crosstab (vs. Tang Majortype) for {context_name} saved.")

        # Visualize the crosstab as a heatmap
        crosstab_norm = crosstab_majortype.apply(
            lambda x: 100 * x / x.sum() if x.sum() > 0 else 0, axis=0
        ).fillna(0)

        fig_crosstab, ax_crosstab = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            crosstab_norm,
            cmap="viridis",
            linewidths=0.5,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "% of Original Tang Majortype"},
            ax=ax_crosstab,
        )
        ax_crosstab.set_title(
            f"Mapping of Assigned Subtypes onto Original Tang Majortypes\n({context_name} - Assigned Cells Only)",
            fontsize=14,
        )
        ax_crosstab.set_ylabel("Assigned Subtype (This Study)", fontsize=12)
        ax_crosstab.set_xlabel("Original Majortype (Tang et al.)", fontsize=12)
        plt.tight_layout()

        plot_basename_crosstab_hm = create_filename(
            "P2_1_Heatmap_Crosstab_vs_Majortype",
            context_name=context_name,
            version="v1",
        )
        save_figure_and_data(
            fig_crosstab,
            crosstab_norm.reset_index(),
            plot_basename_crosstab_hm,
            ctx_fig_dir,
            ctx_data_dir,
        )
        print(f"          Validation heatmap for {context_name} saved.")

    # --- Distribution of Assigned NK Subtypes across meta_histology (Unchanged from original) ---
    if METADATA_HISTOLOGY_COLUMN_GSE212890 in adata_view_assigned.obs.columns:
        print(
            f"\n        Distribution of Assigned Subtypes across '{METADATA_HISTOLOGY_COLUMN_GSE212890}'"
        )

        histology_subtype_counts = pd.crosstab(
            index=adata_view_assigned.obs[METADATA_HISTOLOGY_COLUMN_GSE212890],
            columns=adata_view_assigned.obs[subtype_col],
            dropna=False,
        )
        histology_subtype_proportions = histology_subtype_counts.apply(
            lambda x: 100 * x / x.sum() if x.sum() > 0 else 0, axis=1
        ).fillna(0)

        if not histology_subtype_proportions.empty:
            fig_hist, ax_hist = plt.subplots(
                figsize=(max(10, histology_subtype_proportions.shape[0] * 0.5 + 2), 7)
            )
            colors_for_hist_plot = [
                COMBINED_SUBTYPE_COLOR_PALETTE.get(cat, "#CCCCCC")
                for cat in histology_subtype_proportions.columns
            ]
            histology_subtype_proportions.plot(
                kind="bar",
                stacked=True,
                color=colors_for_hist_plot,
                width=0.8,
                ax=ax_hist,
            )

            ax_hist.set_title(
                f"Assigned NK Subtype Distribution by Histology in {context_name}",
                fontsize=14,
            )
            ax_hist.set_xlabel(METADATA_HISTOLOGY_COLUMN_GSE212890, fontsize=12)
            ax_hist.set_ylabel("Proportion of Assigned Cells (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            ax_hist.legend(
                title=NK_SUBTYPE_PROFILED_COL,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            plot_basename_hist_dist = create_filename(
                f"P2_1_Dist_Subtype_by_Histology",
                context_name=context_name,
                version="v2_assigned_only",
            )
            save_figure_and_data(
                fig_hist,
                histology_subtype_counts.reset_index(),
                plot_basename_hist_dist,
                ctx_fig_dir,
                ctx_data_dir,
            )
            print(
                f"          Subtype distribution by histology plot for {context_name} saved."
            )

    print(f"      --- End of Section 2.1.3 for {context_name} ---\n")

print("--- End of Section 2.1 (All Contexts) ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# Section 2.2: Transcriptional Definition (Context-Specific Markers)

print("\n  --- Section 2.2: Transcriptional Definition (Context-Specific Markers) ---")

for context_name, adata_ctx, context_output_base_dir in cohorts_for_characterization:
    print(f"\n    --- Processing Context-Specific Markers for: {context_name} ---")

    # Define output directories
    ctx_fig_dir = os.path.join(context_output_base_dir, "figures", "context_markers")
    ctx_data_dir = os.path.join(
        context_output_base_dir, "data_for_graphpad", "context_markers"
    )
    ctx_marker_lists_dir = os.path.join(
        context_output_base_dir, "marker_lists", "context_markers"
    )
    ctx_stats_dir = os.path.join(
        context_output_base_dir, "stat_results_python", "context_markers"
    )
    os.makedirs(ctx_fig_dir, exist_ok=True)
    os.makedirs(ctx_data_dir, exist_ok=True)
    os.makedirs(ctx_marker_lists_dir, exist_ok=True)
    os.makedirs(ctx_stats_dir, exist_ok=True)

    # Get the appropriate subtype column and categories for this dataset
    subtype_col = get_subtype_column(adata_ctx)
    subtype_categories = get_subtype_categories(adata_ctx)

    if not subtype_col or subtype_col not in adata_ctx.obs.columns:
        print(
            f"      ERROR: No valid subtype column found for {context_name}. Skipping."
        )
        continue

    # --- Filter to valid cells for DEG ---
    # For Rebuffet: exclude "Unassigned", for Tang: use all cells
    if subtype_col == REBUFFET_SUBTYPE_COL:
        assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
        adata_view_assigned = adata_ctx[assigned_mask, :].copy()
    else:
        # Tang data - all cells should have valid subtypes
        adata_view_assigned = adata_ctx.copy()

    if (
        adata_view_assigned.n_obs == 0
        or adata_view_assigned.obs[subtype_col].nunique() < 2
    ):
        print(
            f"      Not enough cells or subtypes in {context_name} to perform DEG. Skipping."
        )
        continue

    print(f"      Performing DEG on {adata_view_assigned.n_obs} cells.")

    # Define the subtype categories present in this filtered view
    ordered_categories_for_deg = [
        cat
        for cat in subtype_categories
        if cat in adata_view_assigned.obs[subtype_col].unique()
    ]

    # --- 2.2.1: DEG analysis (sc.tl.rank_genes_groups) ---
    print(
        f"      --- 2.2.1: Running DEG for {context_name} to find context-specific markers ---"
    )
    rank_genes_key_ctx = f"rank_genes_ctx_{context_name}"

    # Run DEG on the filtered view
    sc.tl.rank_genes_groups(
        adata_view_assigned,  # Use the filtered view
        groupby=subtype_col,
        groups=ordered_categories_for_deg,
        method="wilcoxon",
        use_raw=True,  # Still uses the .raw from the original object, which is correct
        pts=True,
        corr_method="benjamini-hochberg",
        n_genes=TOP_N_MARKERS_CONTEXT + 150,
        key_added=rank_genes_key_ctx,
    )
    print(f"        DEG analysis complete for {context_name}.")

    # --- 2.2.2: Extract, Filter, Store, and Visualize Context-Specific Markers ---
    print(
        f"      --- 2.2.2: Processing and visualizing context-specific markers for {context_name} ---"
    )
    context_specific_marker_dict = {}

    for subtype_name in ordered_categories_for_deg:
        try:
            # Extract results from the view object where they were calculated
            deg_df_full_ctx = sc.get.rank_genes_groups_df(
                adata_view_assigned, group=subtype_name, key=rank_genes_key_ctx
            )
            if deg_df_full_ctx is None or deg_df_full_ctx.empty:
                context_specific_marker_dict[subtype_name] = []
                continue

            # Filter and select top markers
            deg_df_pattern_filtered_ctx = deg_df_full_ctx[
                ~deg_df_full_ctx["names"].apply(is_gene_to_exclude_util)
            ].copy()
            markers_df_ctx = deg_df_pattern_filtered_ctx[
                (deg_df_pattern_filtered_ctx["pvals_adj"] < ADJ_PVAL_THRESHOLD_DEG)
                & (deg_df_pattern_filtered_ctx["logfoldchanges"] > LOGFC_THRESHOLD_DEG)
            ].copy()
            markers_df_ctx.sort_values(by="scores", ascending=False, inplace=True)
            top_markers_list_ctx = (
                markers_df_ctx["names"].head(TOP_N_MARKERS_CONTEXT).tolist()
            )
            context_specific_marker_dict[subtype_name] = top_markers_list_ctx

            # Save individual marker lists
            marker_list_filename = create_filename(
                f"CtxMarkers_{subtype_name}", context_name=context_name, version="v2"
            )
            with open(
                os.path.join(ctx_marker_lists_dir, f"{marker_list_filename}.txt"), "w"
            ) as f:
                for gene in top_markers_list_ctx:
                    f.write(f"{gene}\\n")
        except Exception as e:
            print(
                f"        ERROR processing markers for {subtype_name} in {context_name}: {e}"
            )
            context_specific_marker_dict[subtype_name] = []

    # Visualize top markers in a Dot Plot using the filtered view
    n_markers_for_dotplot = 4
    combined_markers_for_dotplot = []
    temp_seen_markers = set()
    for subtype_name in ordered_categories_for_deg:
        markers = context_specific_marker_dict.get(subtype_name, [])
        count = 0
        for gene in markers:
            if gene not in temp_seen_markers:
                combined_markers_for_dotplot.append(gene)
                temp_seen_markers.add(gene)
                count += 1
                if count >= n_markers_for_dotplot:
                    break

    if not combined_markers_for_dotplot:
        print(
            f"        WARNING: No context-specific markers available for dot plot in {context_name}."
        )
    else:
        print(
            f"        Generating dot plot for {len(combined_markers_for_dotplot)} unique top markers in {context_name}."
        )
        try:
            # Plot using the filtered view where categories are already ordered
            sc.pl.dotplot(
                adata_view_assigned,
                var_names=combined_markers_for_dotplot,
                groupby=subtype_col,
                standard_scale="var",
                use_raw=True,
                show=False,
                title=f"Top Context-Specific Markers in {context_name}",
            )
            fig_dot_ctx = plt.gcf()
            plot_basename_dot_ctx = create_filename(
                "P2_2_Dotplot_CtxMarkers", context_name=context_name, version="v2"
            )
            save_figure_and_data(
                fig_dot_ctx, None, plot_basename_dot_ctx, ctx_fig_dir, None
            )
            print(
                f"        Dot plot for context-specific markers in {context_name} saved."
            )
        except Exception as e:
            print(
                f"        ERROR generating dot plot for context-specific markers: {e}"
            )
            if "fig_dot_ctx" in locals() and plt.fignum_exists(fig_dot_ctx.number):
                plt.close(fig_dot_ctx)

    print(f"    --- End of Section 2.2 for {context_name} ---\n")

print("--- End of Section 2.2 (All Contexts) ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# Section 2.3: Developmental and Functional Signature Profiling

print("\n  --- Section 2.3: Developmental and Functional Signature Profiling ---")


def generate_signature_heatmap(
    adata_view,
    context_name,
    gene_sets_dict,
    plot_title,
    base_filename,
    fig_dir,
    data_dir,
    subtype_col=None,
    subset_name=None,
):
    """
    Calculates scores for given gene sets and plots a summary heatmap of the mean scores per subtype.
    """
    print(f"      Calculating and plotting '{plot_title}' for {context_name}...")

    # Determine the correct subtype column if not provided
    if subtype_col is None:
        subtype_col = get_subtype_column(adata_view)

    if not subtype_col or subtype_col not in adata_view.obs.columns:
        print(
            f"        ERROR: No valid subtype column found for {context_name}. Skipping."
        )
        return

    # Calculate scores for the provided gene sets
    score_cols = []
    for set_name, gene_list in gene_sets_dict.items():
        score_col_name = f"{set_name}_Score"
        score_cols.append(score_col_name)
        available_genes = map_gene_names(gene_list, adata_view.raw.var_names)
        if len(available_genes) >= MIN_GENES_FOR_SCORING:
            sc.tl.score_genes(
                adata_view,
                available_genes,
                score_name=score_col_name,
                use_raw=True,
                random_state=RANDOM_SEED,
            )
        else:
            adata_view.obs[score_col_name] = np.nan

    # --- Heatmap of mean scores ---
    valid_score_cols = [
        col
        for col in score_cols
        if col in adata_view.obs.columns and adata_view.obs[col].notna().any()
    ]

    if not valid_score_cols:
        print(f"        No valid scores to plot for '{plot_title}' in {context_name}.")
        return

    mean_scores_df = (
        adata_view.obs.groupby(subtype_col, observed=True)[valid_score_cols].mean().T
    )
    # Clean up index labels for better readability
    mean_scores_df.index = (
        mean_scores_df.index.str.replace("_Score", "")
        .str.replace("Maturation_NK._", "", regex=True)
        .str.replace("_", " ")
    )

    try:
        fig, ax = plt.subplots(
            figsize=(
                max(8, mean_scores_df.shape[1] * 1.2),
                max(5, mean_scores_df.shape[0] * 0.5),
            )
        )
        sns.heatmap(
            mean_scores_df,
            cmap="icefire",
            center=0,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Mean Signature Score"},
            ax=ax,
        )
        ax.set_title(f"{plot_title} in {context_name}", fontsize=14)
        ax.set_xlabel("Assigned NK Subtype", fontsize=14)
        ax.set_ylabel("Signature", fontsize=14)
        plt.tight_layout()

        # Create subset-specific filename if subset_name is provided
        if subset_name:
            subset_base_filename = f"{base_filename}_{subset_name}"
            plot_basename = create_filename(
                subset_base_filename, context_name=context_name, version="v5"
            )
        else:
            plot_basename = create_filename(
                base_filename, context_name=context_name, version="v5"
            )
        save_figure_and_data(
            fig, mean_scores_df.reset_index(), plot_basename, fig_dir, data_dir
        )
        print(f"        {plot_title} heatmap for {context_name} saved.")
    except Exception as e:
        print(f"        ERROR generating heatmap for {plot_title}: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


# --- Main Loop: Generate plots for each context ---
for context_name, adata_ctx, context_output_base_dir in cohorts_for_characterization:
    print(f"\n    --- Characterizing Signatures for: {context_name} ---")

    # Get the appropriate subtype column for this dataset
    subtype_col = get_subtype_column(adata_ctx)

    if (
        not subtype_col
        or subtype_col not in adata_ctx.obs.columns
        or adata_ctx.raw is None
    ):
        print(f"      Prerequisites not met for {context_name}. Skipping.")
        continue

    # Create a view with valid cells for all subsequent plots in this section
    # For Rebuffet: exclude "Unassigned", for Tang: use all cells
    if subtype_col == REBUFFET_SUBTYPE_COL:
        assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
        if not assigned_mask.any():
            print(f"      No assigned cells in {context_name}. Skipping.")
            continue
        adata_view_assigned = adata_ctx[assigned_mask, :].copy()
    else:
        # Tang data - all cells should have valid subtypes
        adata_view_assigned = adata_ctx.copy()

    # Check if we should split Tang data into subsets
    tang_subsets = get_tang_subtype_subsets(adata_view_assigned, context_name)

    for subset_name, adata_subset in tang_subsets:
        # Create subset-specific output directories
        if subset_name:
            ctx_fig_dir = os.path.join(
                context_output_base_dir, "figures", "functional_signatures", subset_name
            )
            ctx_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "functional_signatures",
                subset_name,
            )
            print(f"      Processing Tang subset: {subset_name}")
        else:
            ctx_fig_dir = os.path.join(
                context_output_base_dir, "figures", "functional_signatures"
            )
            ctx_data_dir = os.path.join(
                context_output_base_dir, "data_for_graphpad", "functional_signatures"
            )

        os.makedirs(ctx_fig_dir, exist_ok=True)
        os.makedirs(ctx_data_dir, exist_ok=True)

        # Generate the Developmental Profile heatmap
        generate_signature_heatmap(
            adata_view=adata_subset,
            context_name=context_name,
            gene_sets_dict=DEVELOPMENTAL_GENE_SETS,
            plot_title="Developmental Signature Profiles",
            base_filename="P2_3a_Heatmap_DevProfile",
            fig_dir=ctx_fig_dir,
            data_dir=ctx_data_dir,
            subtype_col=subtype_col,
            subset_name=subset_name,
        )

        # Generate the Functional Profile heatmap
        generate_signature_heatmap(
            adata_view=adata_subset,
            context_name=context_name,
            gene_sets_dict=FUNCTIONAL_GENE_SETS,
            plot_title="Functional Signature Profiles",
            base_filename="P2_3b_Heatmap_FuncProfile",
            fig_dir=ctx_fig_dir,
            data_dir=ctx_data_dir,
            subtype_col=subtype_col,
            subset_name=subset_name,
        )

print("\n--- End of Section 2.3 ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# Section 2.3b: Deep Dive into the Cytotoxicity Signature

print("\n  --- Section 2.3b: Deep Dive into the Cytotoxicity Signature ---")

# Define the most important genes from the Cytotoxicity_Machinery_Gene_Set
key_cytotoxicity_genes = ["PRF1", "GZMB", "GZMA", "NKG7", "GNLY"]

# Use the blood cohort as our clean reference dataset
context_name = "Blood"
adata_ctx = adata_blood
ctx_fig_dir = os.path.join(
    OUTPUT_SUBDIRS["blood_nk_char"], "figures", "functional_signatures"
)
ctx_data_dir = os.path.join(
    OUTPUT_SUBDIRS["blood_nk_char"], "data_for_graphpad", "functional_signatures"
)

print(f"      Generating dot plot for key cytotoxicity genes in {context_name}.")

# Get the appropriate subtype column and categories for this dataset
subtype_col = get_subtype_column(adata_ctx)
subtype_categories = get_subtype_categories(adata_ctx)

if not subtype_col or subtype_col not in adata_ctx.obs.columns:
    print(
        f"      ERROR: No valid subtype column found for {context_name}. Skipping Section 2.3b."
    )
else:
    # Filter to valid cells for a cleaner plot
    if subtype_col == REBUFFET_SUBTYPE_COL:
        # For Rebuffet data, exclude "Unassigned"
        categories_to_plot = [
            cat
            for cat in subtype_categories
            if cat in adata_ctx.obs[subtype_col].cat.categories and cat != "Unassigned"
        ]
        adata_plot_view = adata_ctx[
            adata_ctx.obs[subtype_col].isin(categories_to_plot)
        ].copy()
    else:
        # For Tang data, use all cells
        categories_to_plot = [
            cat
            for cat in subtype_categories
            if cat in adata_ctx.obs[subtype_col].unique()
        ]
        adata_plot_view = adata_ctx.copy()

    adata_plot_view.obs[subtype_col] = adata_plot_view.obs[
        subtype_col
    ].cat.remove_unused_categories()

    try:
        # Create the dot plot
        fig_dot, ax_dot = plt.subplots(figsize=(8, 5))
        sc.pl.dotplot(
            adata_plot_view,
            var_names=key_cytotoxicity_genes,
            groupby=subtype_col,
            categories_order=categories_to_plot,
            use_raw=True,
            standard_scale="var",  # Scale color per gene
            show=False,
            ax=ax_dot,
        )
        ax_dot.set_title("Expression of Key Cytotoxicity Genes by Subtype", fontsize=14)
        plt.tight_layout()

        # Manually create the data for export
        dotplot_data_list = []
        for gene in key_cytotoxicity_genes:
            for subtype in categories_to_plot:
                mask = adata_plot_view.obs[subtype_col] == subtype
                if mask.sum() > 0:
                    expr_vals = adata_plot_view[mask, gene].raw.X.toarray().flatten()
                    mean_expr = expr_vals.mean()
                    frac_expr = np.sum(expr_vals > 0) / len(expr_vals) * 100
                else:
                    mean_expr, frac_expr = 0.0, 0.0
                dotplot_data_list.append(
                    {
                        "Gene": gene,
                        "Subtype": subtype,
                        "Mean_Expression_LogNorm": mean_expr,
                        "Fraction_Expressed_Pct": frac_expr,
                    }
                )
        dotplot_export_df = pd.DataFrame(dotplot_data_list)

        # Save the figure and data
        plot_basename = create_filename(
            "P2_3b_Dotplot_KeyCytoGenes", context_name=context_name, version="v1"
        )
        save_figure_and_data(
            fig_dot, dotplot_export_df, plot_basename, ctx_fig_dir, ctx_data_dir
        )
        print(f"        Dot plot deep dive for Cytotoxicity signature saved.")

    except Exception as e:
        print(f"        ERROR generating cytotoxicity deep dive dot plot: {e}")
        if "fig_dot" in locals() and plt.fignum_exists(fig_dot.number):
            plt.close(fig_dot)

print("\n--- End of Section 2.3b ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# Section 2.3c: Detailed Dot Plots for All Signature Categories

print("\n  --- Section 2.3c: Detailed Dot Plots for All Signature Categories ---")


def create_signature_dotplot(
    adata,
    gene_set,
    set_name,
    context_name,
    fig_dir,
    data_dir,
    subtype_col=None,
    subset_name=None,
):
    """
    Generates and saves a polished dot plot for a specific gene set using a robust plotting pattern.
    """
    print(f"      Generating dot plot for '{set_name}' in {context_name}...")

    available_genes = map_gene_names(gene_set, adata.raw.var_names)
    if not available_genes:
        print(
            f"        No genes from '{set_name}' are available in {context_name}. Skipping."
        )
        return

    # Determine the correct subtype column if not provided
    if subtype_col is None:
        subtype_col = get_subtype_column(adata)

    if not subtype_col or subtype_col not in adata.obs.columns:
        print(
            f"        ERROR: No valid subtype column found for {context_name}. Skipping."
        )
        return

    # Get appropriate categories for this dataset
    subtype_categories = get_subtype_categories(adata)

    # Filter to valid cells for plotting
    # For Rebuffet: exclude "Unassigned", for Tang: use all cells
    if subtype_col == REBUFFET_SUBTYPE_COL:
        adata_view_assigned = adata[adata.obs[subtype_col] != "Unassigned"].copy()
        if adata_view_assigned.n_obs == 0:
            print(f"        No assigned cells to plot for {context_name}. Skipping.")
            return
    else:
        # Tang data - all cells should have valid subtypes
        adata_view_assigned = adata.copy()

    # Ensure categories are ordered correctly for the plot
    categories_to_plot = [
        cat
        for cat in subtype_categories
        if cat in adata_view_assigned.obs[subtype_col].unique()
    ]
    adata_view_assigned.obs[subtype_col] = adata_view_assigned.obs[
        subtype_col
    ].cat.reorder_categories(categories_to_plot, ordered=True)

    # --- Robust Plotting Pattern ---
    # 1. Create Figure and Axes first.
    fig, ax = plt.subplots(figsize=(max(7, len(available_genes) * 0.45), 5.5))

    try:
        # 2. Pass the created axes to the Scanpy plotting function.
        sc.pl.dotplot(
            adata_view_assigned,
            var_names=available_genes,
            groupby=subtype_col,
            use_raw=True,
            standard_scale="var",
            cmap="Reds",
            show=False,
            ax=ax,  # Explicitly draw on our pre-made axes
        )

        # 3. Modify the figure and axes as needed.
        plot_title = set_name.replace("_", " ")
        fig.suptitle(
            f"Expression of {plot_title} Genes\nby Subtype in {context_name}",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # 4. Manually calculate data for export. This is more robust than relying on a returned object.
        dotplot_data_list = []
        for gene in available_genes:
            for subtype in categories_to_plot:
                mask = adata_view_assigned.obs[subtype_col] == subtype
                if mask.sum() > 0:
                    expr_vals = (
                        adata_view_assigned.raw[mask, gene].X.toarray().flatten()
                    )
                    mean_expr = np.mean(expr_vals)
                    frac_expr = np.sum(expr_vals > 0) / len(expr_vals) * 100
                else:
                    mean_expr, frac_expr = 0.0, 0.0
                dotplot_data_list.append(
                    {
                        "Gene": gene,
                        "Subtype": subtype,
                        "Mean_Expression_LogNorm": mean_expr,
                        "Fraction_Expressed_Pct": frac_expr,
                    }
                )
        export_df = pd.DataFrame(dotplot_data_list)

        # 5. Save the final figure and data.
        # Create subset-specific filename if subset_name is provided
        if subset_name:
            base_filename = f"P2_3c_Dotplot_{set_name}_{subset_name}"
        else:
            base_filename = f"P2_3c_Dotplot_{set_name}"

        plot_basename = create_filename(
            base_filename,
            context_name=context_name,
            version="v7_final_robust",
        )
        save_figure_and_data(fig, export_df, plot_basename, fig_dir, data_dir)
        print(f"        Dot plot for '{set_name}' saved.")

    except Exception as e:
        print(f"        ERROR generating dot plot for '{set_name}': {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


# --- Main loop to generate plots, now separated by signature category ---
for context_name, adata_ctx, context_output_base_dir in cohorts_for_characterization:
    if adata_ctx is None or adata_ctx.n_obs == 0:
        continue

    # Get the appropriate subtype column for this dataset
    subtype_col = get_subtype_column(adata_ctx)

    if not subtype_col or subtype_col not in adata_ctx.obs.columns:
        print(
            f"      ERROR: No valid subtype column found for {context_name}. Skipping signature dotplots."
        )
        continue

    # Create a view with valid cells for all subsequent plots in this section
    # For Rebuffet: exclude "Unassigned", for Tang: use all cells
    if subtype_col == REBUFFET_SUBTYPE_COL:
        assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
        if not assigned_mask.any():
            print(f"      No assigned cells in {context_name}. Skipping.")
            continue
        adata_view_assigned = adata_ctx[assigned_mask, :].copy()
    else:
        # Tang data - all cells should have valid subtypes
        adata_view_assigned = adata_ctx.copy()

    # Check if we should split Tang data into subsets
    tang_subsets = get_tang_subtype_subsets(adata_view_assigned, context_name)

    for subset_name, adata_subset in tang_subsets:
        subset_suffix = f" ({subset_name})" if subset_name else ""

        # --- Generate plots for DEVELOPMENTAL Signatures ---
        print(
            f"\n    --- Generating Developmental Signature Dot Plots for: {context_name}{subset_suffix} ---"
        )
        if subset_name:
            dev_fig_dir = os.path.join(
                context_output_base_dir,
                "figures",
                "developmental_signature_dotplots",
                subset_name,
            )
            dev_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "developmental_signature_dotplots",
                subset_name,
            )
        else:
            dev_fig_dir = os.path.join(
                context_output_base_dir, "figures", "developmental_signature_dotplots"
            )
            dev_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "developmental_signature_dotplots",
            )
        os.makedirs(dev_fig_dir, exist_ok=True)
        os.makedirs(dev_data_dir, exist_ok=True)

        for name_of_set, list_of_genes in DEVELOPMENTAL_GENE_SETS.items():
            create_signature_dotplot(
                adata=adata_subset,
                gene_set=list_of_genes,
                set_name=name_of_set,
                context_name=context_name,
                fig_dir=dev_fig_dir,
                data_dir=dev_data_dir,
                subtype_col=subtype_col,
                subset_name=subset_name,
            )

        # --- Generate plots for FUNCTIONAL Signatures ---
        print(
            f"\n    --- Generating Functional Signature Dot Plots for: {context_name}{subset_suffix} ---"
        )
        if subset_name:
            func_fig_dir = os.path.join(
                context_output_base_dir,
                "figures",
                "functional_signature_dotplots",
                subset_name,
            )
            func_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "functional_signature_dotplots",
                subset_name,
            )
        else:
            func_fig_dir = os.path.join(
                context_output_base_dir, "figures", "functional_signature_dotplots"
            )
            func_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "functional_signature_dotplots",
            )
        os.makedirs(func_fig_dir, exist_ok=True)
        os.makedirs(func_data_dir, exist_ok=True)

        for name_of_set, list_of_genes in FUNCTIONAL_GENE_SETS.items():
            create_signature_dotplot(
                adata=adata_subset,
                gene_set=list_of_genes,
                set_name=name_of_set,
                context_name=context_name,
                fig_dir=func_fig_dir,
                data_dir=func_data_dir,
                subtype_col=subtype_col,
                subset_name=subset_name,
            )

        # --- Generate plots for Neurotransmitter Receptor Signatures ---
        print(
            f"\n    --- Generating Neurotransmitter Receptor Signature Dot Plots for: {context_name}{subset_suffix} ---"
        )
        if subset_name:
            neuro_fig_dir = os.path.join(
                context_output_base_dir,
                "figures",
                "neurotransmitter_receptor_signature_dotplots",
                subset_name,
            )
            neuro_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "neurotransmitter_receptor_signature_dotplots",
                subset_name,
            )
        else:
            neuro_fig_dir = os.path.join(
                context_output_base_dir,
                "figures",
                "neurotransmitter_receptor_signature_dotplots",
            )
            neuro_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "neurotransmitter_receptor_signature_dotplots",
            )
        os.makedirs(neuro_fig_dir, exist_ok=True)
        os.makedirs(neuro_data_dir, exist_ok=True)

        for name_of_set, list_of_genes in NEUROTRANSMITTER_RECEPTOR_GENE_SETS.items():
            create_signature_dotplot(
                adata=adata_subset,
                gene_set=list_of_genes,
                set_name=name_of_set,
                context_name=context_name,
                fig_dir=neuro_fig_dir,
                data_dir=neuro_data_dir,
                subtype_col=subtype_col,
                subset_name=subset_name,
            )

        # --- Generate plots for Interleukin Downstream Signatures ---
        print(
            f"\n    --- Generating Interleukin Downstream Signature Dot Plots for: {context_name}{subset_suffix} ---"
        )
        if subset_name:
            il_fig_dir = os.path.join(
                context_output_base_dir,
                "figures",
                "interleukin_downstream_signature_dotplots",
                subset_name,
            )
            il_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "interleukin_downstream_signature_dotplots",
                subset_name,
            )
        else:
            il_fig_dir = os.path.join(
                context_output_base_dir,
                "figures",
                "interleukin_downstream_signature_dotplots",
            )
            il_data_dir = os.path.join(
                context_output_base_dir,
                "data_for_graphpad",
                "interleukin_downstream_signature_dotplots",
            )
        os.makedirs(il_fig_dir, exist_ok=True)
        os.makedirs(il_data_dir, exist_ok=True)

        for name_of_set, list_of_genes in INTERLEUKIN_DOWNSTREAM_GENE_SETS.items():
            create_signature_dotplot(
                adata=adata_subset,
                gene_set=list_of_genes,
                set_name=name_of_set,
                context_name=context_name,
                fig_dir=il_fig_dir,
                data_dir=il_data_dir,
                subtype_col=subtype_col,
                subset_name=subset_name,
            )

print("\n--- End of Section 2.3c ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context
# Section 2.4: Developmental Marker Profile of NK Subtypes

print("  --- Section 2.4: Developmental Marker Profile of NK Subtypes ---")

if "MURINE_DEV_MARKER_ORTHOLOGS" not in locals() or not MURINE_DEV_MARKER_ORTHOLOGS:
    print(
        "      ERROR: MURINE_DEV_MARKER_ORTHOLOGS list not defined. Skipping Section 2.4."
    )
else:
    for (
        context_name,
        adata_ctx,
        context_output_base_dir,
    ) in cohorts_for_characterization:
        print(
            f"    --- Processing Developmental Marker Profile for: {context_name} ---"
        )

        ctx_fig_dir = os.path.join(
            context_output_base_dir, "figures", "developmental_markers"
        )
        ctx_data_dir = os.path.join(
            context_output_base_dir, "data_for_graphpad", "developmental_markers"
        )
        os.makedirs(ctx_fig_dir, exist_ok=True)
        os.makedirs(ctx_data_dir, exist_ok=True)

        # Get the appropriate subtype column for this dataset
        subtype_col = get_subtype_column(adata_ctx)

        if (
            not subtype_col
            or subtype_col not in adata_ctx.obs.columns
            or adata_ctx.raw is None
        ):
            print(f"      ERROR: Prerequisites not met for {context_name}. Skipping.")
            continue

        # Create a view with valid cells for developmental marker analysis
        # For Rebuffet: exclude "Unassigned", for Tang: use all cells
        if subtype_col == REBUFFET_SUBTYPE_COL:
            assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
            if not assigned_mask.any():
                print(f"      No assigned cells in {context_name}. Skipping.")
                continue
            adata_view_assigned = adata_ctx[assigned_mask, :].copy()
        else:
            # Tang data - all cells should have valid subtypes
            adata_view_assigned = adata_ctx.copy()

        available_dev_markers = [
            g
            for g in MURINE_DEV_MARKER_ORTHOLOGS
            if g in adata_view_assigned.raw.var_names
        ]
        if len(available_dev_markers) < 5:
            print(
                f"      Too few developmental markers available in {context_name}. Skipping plots."
            )
            continue

        print(
            "        Identifying the most variable developmental markers for this context..."
        )

        temp_dev_adata = sc.AnnData(
            adata_view_assigned.raw[:, available_dev_markers].X,
            var=pd.DataFrame(index=available_dev_markers),
        )
        sc.pp.highly_variable_genes(
            temp_dev_adata, n_top_genes=20, flavor="seurat", subset=False
        )

        # --- ROBUST FIX for changing scanpy API ---
        # Check for possible HVG score column names in order of likelihood.
        hvg_score_col = None
        possible_score_cols = [
            "dispersions_norm",
            "variances_norm",
            "highly_variable_scores",
        ]
        for col_name in possible_score_cols:
            if col_name in temp_dev_adata.var.columns:
                hvg_score_col = col_name
                print(f"          Found HVG score column: '{hvg_score_col}'")
                break

        if hvg_score_col is None:
            print(
                f"      ERROR: Could not find any expected HVG score column in {possible_score_cols}. Cannot select variable markers."
            )
            continue

        variable_dev_markers = temp_dev_adata.var.nlargest(
            20, hvg_score_col
        ).index.tolist()

        if not variable_dev_markers:
            print(
                f"      Could not identify any variable developmental markers in {context_name}. Skipping plot."
            )
            continue

        print(
            f"      Visualizing {len(variable_dev_markers)} most variable developmental markers for {context_name}."
        )

        print(
            f"        --- 2.4.1: Dot Plot of Most Variable Developmental Markers for {context_name} ---"
        )
        fig, ax = plt.subplots(figsize=(max(8, len(variable_dev_markers) * 0.5), 6))

        try:
            sc.pl.dotplot(
                adata_view_assigned,
                var_names=variable_dev_markers,
                groupby=subtype_col,
                standard_scale="var",
                use_raw=True,
                show=False,
                ax=ax,
            )
            ax.set_title(
                f"Top Variable Developmental Markers in {context_name}", fontsize=14
            )
            plt.tight_layout()

            # --- Manually and Robustly Create Data for Export ---
            dotplot_data_list = []
            for gene in variable_dev_markers:
                for subtype in adata_view_assigned.obs[subtype_col].cat.categories:
                    mask = adata_view_assigned.obs[subtype_col] == subtype
                    if mask.sum() > 0:
                        expr_vals = (
                            adata_view_assigned.raw[mask, gene].X.toarray().flatten()
                        )
                        mean_expr = expr_vals.mean()
                        frac_expr = np.sum(expr_vals > 0) / len(expr_vals) * 100
                    else:
                        mean_expr, frac_expr = 0.0, 0.0
                    dotplot_data_list.append(
                        {
                            "Gene": gene,
                            "Subtype": subtype,
                            "Mean_Expression_LogNorm": mean_expr,
                            "Fraction_Expressed_Pct": frac_expr,
                        }
                    )
            export_df = pd.DataFrame(dotplot_data_list)

            # Use a consistent, corrected filename for saving
            plot_basename_dot_dev = create_filename(
                "P2_4_Dotplot_VariableDevMarkers",
                context_name=context_name,
                version="v3_final",
            )
            save_figure_and_data(
                fig, export_df, plot_basename_dot_dev, ctx_fig_dir, ctx_data_dir
            )
            print(
                f"          Dot plot of developmental markers for {context_name} saved."
            )

        except Exception as e:
            print(
                f"          ERROR generating developmental marker dot plot for {context_name}: {e}"
            )
            if fig is not None and plt.fignum_exists(fig.number):
                plt.close(fig)

        print(f"    --- End of Section 2.4 for {context_name} ---")

print("--- End of Section 2.4 (All Contexts) ---")

# %%
# PART 2: Synthesis Figure
# Section 2.5: Generating Developmental and Functional Blueprint Figures

print(
    "\n--- Section 2.5: Generating Developmental and Functional Blueprint Figures ---"
)


def create_blueprint_dotplot(
    adata_assigned,
    signatures_to_plot,
    gene_sets_dict,
    plot_title,
    base_filename,
    fig_dir,
    data_dir,
    subset_name=None,
):
    """
    Selects, clusters, and plots a categorized dot plot for a given set of gene signatures.
    This version uses the robust plotting and manual data export logic from the original notebook.
    """
    print(f"\n      --- Creating Blueprint: {plot_title} ---")

    # --- 1. Select and Order Genes for Plotting ---
    print("        Selecting and clustering genes...")

    all_signature_genes = list(
        itertools.chain(
            *[gene_sets_dict.get(sig, []) for sig in signatures_to_plot.values()]
        )
    )
    genes_in_adata = [
        g for g in set(all_signature_genes) if g in adata_assigned.raw.var_names
    ]

    mean_expr_df = pd.DataFrame(
        adata_assigned.raw[:, genes_in_adata].X.toarray(),
        columns=genes_in_adata,
        index=adata_assigned.obs.index,
    )
    mean_expr_by_subtype = (
        mean_expr_df.join(adata_assigned.obs[NK_SUBTYPE_PROFILED_COL])
        .groupby(NK_SUBTYPE_PROFILED_COL, observed=True)
        .mean()
    )

    final_plot_genes, group_boundaries, genes_already_added, current_gene_count = (
        [],
        {},
        set(),
        0,
    )
    top_n_per_group = 6

    for display_name, signature_key in signatures_to_plot.items():
        gene_list = gene_sets_dict.get(signature_key, [])
        genes_in_sig = [
            g
            for g in gene_list
            if g in mean_expr_by_subtype.columns and g not in genes_already_added
        ]
        if genes_in_sig:
            group_expr_matrix = mean_expr_by_subtype[genes_in_sig]
            top_genes = group_expr_matrix.var().nlargest(top_n_per_group).index.tolist()
            if len(top_genes) > 1:
                from scipy.cluster.hierarchy import linkage, leaves_list
                from scipy.spatial.distance import pdist

                gene_dist = pdist(group_expr_matrix[top_genes].T, metric="correlation")
                gene_linkage = linkage(gene_dist, method="average")
                ordered_genes = [top_genes[i] for i in leaves_list(gene_linkage)]
            else:
                ordered_genes = top_genes
            final_plot_genes.extend(ordered_genes)
            genes_already_added.update(ordered_genes)
            group_boundaries[display_name] = (
                current_gene_count,
                current_gene_count + len(ordered_genes) - 1,
            )
            current_gene_count += len(ordered_genes)

    print(f"        Selected {len(final_plot_genes)} unique genes for the plot.")

    # --- 2. Generate the Plot and Data for Export ---
    if not final_plot_genes:
        print("        No genes selected for plotting. Skipping.")
        return

    try:
        # Generate the plot using the proven method from the original notebook
        dot_plot = sc.pl.dotplot(
            adata_assigned,
            var_names=final_plot_genes,
            groupby=NK_SUBTYPE_PROFILED_COL,
            use_raw=True,
            standard_scale="var",
            cmap="Reds",
            dot_max=0.8,
            dot_min=0,
            figsize=(max(10, len(final_plot_genes) * 0.48), 5),
            show=False,
            return_fig=True,
        )
        ax = dot_plot.get_axes()["mainplot_ax"]

        # Add custom annotations
        for i, (group_name, (start, end)) in enumerate(group_boundaries.items()):
            if i < len(group_boundaries) - 1:
                ax.axvline(x=end + 1, color="darkgray", linestyle="--", linewidth=1.2)
            mid_point = (start + end) / 2.0
            ax.text(
                mid_point + 0.5,
                ax.get_ylim()[1] + 0.5,
                group_name,
                ha="center",
                va="bottom",
                fontsize=12,
                weight="semibold",
            )

        ax.set_title(plot_title, fontsize=18, weight="bold", pad=15)
        dot_plot.fig.tight_layout(rect=[0, 0, 1, 0.97])

        # Manually calculate the data for export (restoring the working logic)
        print("        Calculating data for export...")
        export_data_list = []
        for gene in final_plot_genes:
            for subtype in mean_expr_by_subtype.index:
                mask = adata_assigned.obs[NK_SUBTYPE_PROFILED_COL] == subtype
                if mask.sum() > 0:
                    expr_vals = adata_assigned.raw[mask, gene].X.toarray().flatten()
                    mean_expr = (
                        np.mean(expr_vals[expr_vals > 0])
                        if (expr_vals > 0).any()
                        else 0.0
                    )
                    frac_expr = np.sum(expr_vals > 0) / len(expr_vals)
                    export_data_list.append(
                        {
                            "Gene": gene,
                            "Subtype": subtype,
                            "Mean_Expression_in_Group": mean_expr,
                            "Fraction_of_Cells_in_Group": frac_expr,
                        }
                    )
        plot_df_for_export = pd.DataFrame(export_data_list)

        # Save the final figure and data
        # Create subset-specific filename if subset_name is provided
        if subset_name:
            subset_base_filename = f"{base_filename}_{subset_name}"
            plot_basename = create_filename(
                subset_base_filename, version="v2_separated"
            )
        else:
            plot_basename = create_filename(base_filename, version="v2_separated")
        save_figure_and_data(
            dot_plot.fig, plot_df_for_export, plot_basename, fig_dir, data_dir
        )
        print(f"        Blueprint figure '{plot_title}' saved successfully.")

    except Exception as e:
        print(f"        ERROR generating blueprint dot plot: {e}")
        if "dot_plot" in locals() and hasattr(dot_plot, "fig"):
            plt.close(dot_plot.fig)


# --- Main Execution ---
if "adata_blood" not in locals() or adata_blood is None:
    print("      ERROR: adata_blood not found. Cannot generate blueprint plots.")
else:
    fig_dir = os.path.join(OUTPUT_SUBDIRS["cross_context_synthesis"])
    data_dir = os.path.join(fig_dir, "data_for_graphpad")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    adata_assigned_view = adata_blood[
        adata_blood.obs[NK_SUBTYPE_PROFILED_COL] != "Unassigned"
    ].copy()

    # Define the signature groups for each plot
    DEV_SIGNATURES_TO_PLOT = {
        "Regulatory NK": "Regulatory_NK",
        "Intermediate NK": "Intermediate_NK",
        "Mature Cytotoxic NK": "Mature_Cytotoxic_NK",
        "Adaptive NK": "Adaptive_NK",
    }

    FUNC_SIGNATURES_TO_PLOT = {
        "Activating Receptors": "Activating_Receptors",
        "Inhibitory Receptors": "Inhibitory_Receptors",
        "Cytotoxicity Machinery": "Cytotoxicity_Machinery",
        "Cytokine Production": "Cytokine_Chemokine_Production",
        "Exhaustion Markers": "Exhaustion_Suppression_Markers",
    }

    # Generate the Developmental Blueprint
    create_blueprint_dotplot(
        adata_assigned=adata_assigned_view,
        signatures_to_plot=DEV_SIGNATURES_TO_PLOT,
        gene_sets_dict=DEVELOPMENTAL_GENE_SETS,
        plot_title="NK Cell Developmental Blueprint",
        base_filename="P2_5a_Dotplot_Developmental_Blueprint",
        fig_dir=fig_dir,
        data_dir=data_dir,
    )

    # Generate the Functional Blueprint
    create_blueprint_dotplot(
        adata_assigned=adata_assigned_view,
        signatures_to_plot=FUNC_SIGNATURES_TO_PLOT,
        gene_sets_dict=FUNCTIONAL_GENE_SETS,
        plot_title="NK Cell Functional Blueprint",
        base_filename="P2_5b_Dotplot_Functional_Blueprint",
        fig_dir=fig_dir,
        data_dir=data_dir,
    )

print("\n--- End of Section 2.5 ---")

# %%
# PART 2: Synthesis Figure
# Section 2.6: Synthesis Dot Plot of Top Subtype-Defining Markers

print("\n--- Section 2.6: Synthesis Dot Plot of Top Subtype-Defining Markers ---")

# --- Prerequisite Check ---
if "adata_blood" not in locals() or adata_blood is None:
    print("      ERROR: adata_blood not found. Cannot generate plot.")
elif "rank_genes_blood_ref_subtypes" not in adata_blood.uns:
    print(
        "      ERROR: DEG results ('rank_genes_blood_ref_subtypes') not found in adata_blood.uns."
    )
    print("      Please ensure Section 1.3.3 has been run successfully.")
else:
    # --- 1. Extract Optimal DEGs for Each Subtype with Intelligent Shared Gene Handling ---
    print(
        "      Extracting optimal DEGs for each NK subtype using enhanced marker selection..."
    )

    # Use the enhanced marker selection function
    top_markers_by_subtype = select_optimal_subtype_markers(
        adata=adata_blood,
        deg_key="rank_genes_blood_ref_subtypes",
        subtypes_ordered=REBUFFET_SUBTYPES_ORDERED,
        max_markers_per_subtype=4,  # Max 4 DEGs per subtype as requested
        pval_threshold=0.05,
        logfc_threshold=0.25,
    )

    # --- 2. Generate the Blueprint-style Dot Plot ---
    fig_dir = os.path.join(OUTPUT_SUBDIRS["cross_context_synthesis"])
    data_dir = os.path.join(fig_dir, "data_for_graphpad")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    adata_assigned_view = adata_blood[
        adata_blood.obs[NK_SUBTYPE_PROFILED_COL] != "Unassigned"
    ].copy()

    print("\n      --- Creating Synthesis Plot: Top Defining Markers ---")

    # --- Prepare Genes for Plotting ---
    print("        Organizing genes for plot layout...")
    final_plot_genes, group_boundaries, genes_already_added, current_gene_count = (
        [],
        {},
        set(),
        0,
    )

    for subtype_name, gene_list in top_markers_by_subtype.items():
        ordered_genes = [g for g in gene_list if g not in genes_already_added]
        if ordered_genes:
            final_plot_genes.extend(ordered_genes)
            genes_already_added.update(ordered_genes)
            group_boundaries[subtype_name] = (
                current_gene_count,
                current_gene_count + len(ordered_genes) - 1,
            )
            current_gene_count += len(ordered_genes)

    print(f"        Selected {len(final_plot_genes)} unique genes for the plot.")

    # --- Generate the Plot and Data for Export ---
    if not final_plot_genes:
        print("        No genes available for plotting. Skipping figure generation.")
    else:
        try:
            dot_plot = sc.pl.dotplot(
                adata_assigned_view,
                var_names=final_plot_genes,
                groupby=NK_SUBTYPE_PROFILED_COL,
                use_raw=True,
                standard_scale="var",
                cmap="Reds",
                dot_max=0.8,
                dot_min=0,
                figsize=(max(10, len(final_plot_genes) * 0.48), 5),
                show=False,
                return_fig=True,
            )
            ax = dot_plot.get_axes()["mainplot_ax"]

            for i, (group_name, (start, end)) in enumerate(group_boundaries.items()):
                if i < len(group_boundaries) - 1:
                    ax.axvline(
                        x=end + 1, color="darkgray", linestyle="--", linewidth=1.2
                    )
                mid_point = (start + end) / 2.0
                ax.text(
                    mid_point + 0.5,
                    ax.get_ylim()[1] + 0.5,
                    group_name,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    weight="semibold",
                )

            ax.set_title(
                "Top Differentially Expressed Markers Defining NK Subtypes",
                fontsize=18,
                weight="bold",
                pad=20,
            )
            dot_plot.fig.tight_layout(rect=[0, 0, 1, 0.97])

            # CORRECTED: Use the robust, manual data calculation loop from the original blueprint cell
            print("        Calculating data for export...")
            export_data_list = []

            # First, calculate mean expression across all genes to get the subtype order
            mean_expr_df = pd.DataFrame(
                adata_assigned_view.raw[:, final_plot_genes].X.toarray(),
                columns=final_plot_genes,
                index=adata_assigned_view.obs.index,
            )
            mean_expr_by_subtype = (
                mean_expr_df.join(adata_assigned_view.obs[NK_SUBTYPE_PROFILED_COL])
                .groupby(NK_SUBTYPE_PROFILED_COL, observed=True)
                .mean()
            )

            for gene in final_plot_genes:
                for subtype in mean_expr_by_subtype.index:
                    mask = adata_assigned_view.obs[NK_SUBTYPE_PROFILED_COL] == subtype
                    if mask.sum() > 0:
                        expr_vals = (
                            adata_assigned_view.raw[mask, gene].X.toarray().flatten()
                        )
                        mean_expr = (
                            np.mean(expr_vals[expr_vals > 0])
                            if (expr_vals > 0).any()
                            else 0.0
                        )
                        frac_expr = np.sum(expr_vals > 0) / len(expr_vals)
                        export_data_list.append(
                            {
                                "Gene": gene,
                                "Subtype": subtype,
                                "Mean_Expression_in_Group": mean_expr,
                                "Fraction_of_Cells_in_Group": frac_expr,
                            }
                        )
            plot_df_for_export = pd.DataFrame(export_data_list)

            # Save the final figure and data
            plot_basename = create_filename(
                "P2_6_Dotplot_Top_DEGs_Blueprint", version="v2_corrected"
            )
            save_figure_and_data(
                dot_plot.fig, plot_df_for_export, plot_basename, fig_dir, data_dir
            )
            print(f"        DEG Blueprint figure saved successfully.")

        except Exception as e:
            print(f"        ERROR generating DEG blueprint dot plot: {e}")
            if "dot_plot" in locals() and hasattr(dot_plot, "fig"):
                plt.close(dot_plot.fig)

print("\n--- End of Section 2.6 ---")

# %%
# PART 3: TUSC2 Analysis - A Layered Approach
# All expression for TUSC2 and other genes is retrieved from .raw.X to ensure the use of
# unscaled, log-normalized data, which is appropriate for differential expression and visualization.

print("\n--- PART 3: TUSC2 Analysis - A Layered Approach ---\n")

# --- Directory Setup for Part 3 ---
layer1_fig_dir = OUTPUT_SUBDIRS["tusc2_broad_context"]
layer1_data_dir = os.path.join(layer1_fig_dir, "data_for_graphpad")
layer1_stats_dir = os.path.join(layer1_fig_dir, "stat_results_python")
os.makedirs(layer1_fig_dir, exist_ok=True)
os.makedirs(layer1_data_dir, exist_ok=True)
os.makedirs(layer1_stats_dir, exist_ok=True)

# --- Section 3.1: Layer 1 - TUSC2 Expression Across Broad Contexts ---
print("\n  --- Section 3.1: Layer 1 - TUSC2 Expression Across Broad Contexts ---\n")

# --- Step 1: Data Preparation for All Cohorts ---
# Define the cohorts to be analyzed in this block. This makes the cell self-contained.
cohorts_for_characterization = []
if "adata_blood" in locals() and adata_blood is not None and adata_blood.n_obs > 0:
    cohorts_for_characterization.append(
        ("Blood", adata_blood, OUTPUT_SUBDIRS["blood_nk_char"])
    )
if (
    "adata_normal_tissue" in locals()
    and adata_normal_tissue is not None
    and adata_normal_tissue.n_obs > 0
):
    cohorts_for_characterization.append(
        ("NormalTissue", adata_normal_tissue, OUTPUT_SUBDIRS["normal_tissue_nk_char"])
    )
if (
    "adata_tumor_tissue" in locals()
    and adata_tumor_tissue is not None
    and adata_tumor_tissue.n_obs > 0
):
    cohorts_for_characterization.append(
        ("TumorTissue", adata_tumor_tissue, OUTPUT_SUBDIRS["tumor_tissue_nk_char"])
    )

all_cohort_obs_list = []
for context_name, adata_ctx, _ in cohorts_for_characterization:
    if adata_ctx.raw is None:
        print(f"    WARNING: {context_name} has no .raw attribute. Skipping.")
        continue
    if TUSC2_GENE_NAME not in adata_ctx.raw.var_names:
        print(
            f"    WARNING: {TUSC2_GENE_NAME} not found in {context_name}.raw.var_names. Skipping."
        )
        continue

    # Extract TUSC2 expression from .raw.X and store it in .obs
    obs_col_tusc2_expr = f"{TUSC2_GENE_NAME}_Expression_Raw"
    adata_ctx.obs[obs_col_tusc2_expr] = (
        adata_ctx.raw[:, TUSC2_GENE_NAME].X.toarray().flatten()
    )

    # Define binary expression groups based on a threshold
    adata_ctx.obs[TUSC2_BINARY_GROUP_COL] = np.where(
        adata_ctx.obs[obs_col_tusc2_expr] > TUSC2_EXPRESSION_THRESHOLD_BINARY,
        TUSC2_BINARY_CATEGORIES[1],  # TUSC2_Expressed
        TUSC2_BINARY_CATEGORIES[0],  # TUSC2_Not_Expressed
    )
    adata_ctx.obs[TUSC2_BINARY_GROUP_COL] = pd.Categorical(
        adata_ctx.obs[TUSC2_BINARY_GROUP_COL],
        categories=TUSC2_BINARY_CATEGORIES,
        ordered=True,
    )
    print(f"    TUSC2 expression and binary groups calculated for {context_name}.")

    all_cohort_obs_list.append(
        adata_ctx.obs[[obs_col_tusc2_expr, TUSC2_BINARY_GROUP_COL]].assign(
            Context=context_name
        )
    )

# --- Step 2: Generate TUSC2 Expression Violin Plot ---
if not all_cohort_obs_list:
    print("\n    ERROR: No valid cohort data found. Cannot generate Layer 1 plots.")
else:
    combined_obs_df = pd.concat(all_cohort_obs_list)
    context_order = [name for name, _, _ in cohorts_for_characterization]
    combined_obs_df["Context"] = pd.Categorical(
        combined_obs_df["Context"], categories=context_order, ordered=True
    )

    print("\n    --- 3.1.1: Visualizing TUSC2 Expression (Normal vs. Tumor Tissue) ---")
    try:
        fig_viol, ax_viol = plt.subplots(figsize=(8, 7))
        sns.violinplot(
            x="Context",
            y=f"{TUSC2_GENE_NAME}_Expression_Raw",
            data=combined_obs_df,
            ax=ax_viol,
            palette=CONTEXT_COLOR_PALETTE,
            hue="Context",
            legend=False,
            inner="quartile",
        )

        ax_viol.set_title(f"TUSC2 Expression Across Biological Contexts", fontsize=16)
        ax_viol.set_xlabel("Biological Context", fontsize=12)
        ax_viol.set_ylabel(f"{TUSC2_GENE_NAME} Expression (Log-Norm)", fontsize=12)
        plt.xticks(rotation=45, ha="right")

        # --- Robust Statistical Annotation ---
        context_cats = ax_viol.get_xticklabels()
        x_coords = {label.get_text(): i for i, label in enumerate(context_cats)}
        x1, x2 = x_coords.get("NormalTissue"), x_coords.get("TumorTissue")

        if x1 is not None and x2 is not None:
            group1 = combined_obs_df[f"{TUSC2_GENE_NAME}_Expression_Raw"][
                combined_obs_df["Context"] == "NormalTissue"
            ].dropna()
            group2 = combined_obs_df[f"{TUSC2_GENE_NAME}_Expression_Raw"][
                combined_obs_df["Context"] == "TumorTissue"
            ].dropna()
            stat, pval = stats.mannwhitneyu(group1, group2, alternative="two-sided")

            y_max = combined_obs_df[f"{TUSC2_GENE_NAME}_Expression_Raw"].max()
            ax_viol.plot(
                [x1, x1, x2, x2],
                [y_max * 1.05, y_max * 1.1, y_max * 1.1, y_max * 1.05],
                lw=1.5,
                c="k",
            )
            ax_viol.text(
                (x1 + x2) / 2,
                y_max * 1.12,
                get_significance_stars(pval),
                ha="center",
                va="bottom",
                fontsize=12,
            )
            ax_viol.set_ylim(top=y_max * 1.25)

        plt.tight_layout()
        plot_basename_viol = create_filename(
            "P3_1_Violin_TUSC2_AcrossContexts", version="v4_robust"
        )
        save_figure_and_data(
            fig_viol,
            combined_obs_df,
            plot_basename_viol,
            layer1_fig_dir,
            layer1_data_dir,
        )
        print(f"      Overall TUSC2 expression violin plot saved.")
    except Exception as e:
        print(f"      ERROR generating TUSC2 expression violin plot: {e}")
        if "fig_viol" in locals() and plt.fignum_exists(fig_viol.number):
            plt.close(fig_viol)

    # --- Step 3: Generate Consolidated Bar Plot for TUSC2 Binary Groups ---
    print("\n    --- 3.1.2: Visualizing TUSC2+ Cell Proportions Across Contexts ---")
    try:
        proportion_df = (
            combined_obs_df.groupby("Context", observed=False)[TUSC2_BINARY_GROUP_COL]
            .value_counts(normalize=True)
            .mul(100)
            .rename("Proportion")
            .reset_index()
        )

        g = sns.catplot(
            data=proportion_df,
            x="Context",
            y="Proportion",
            hue=TUSC2_BINARY_GROUP_COL,
            kind="bar",
            palette=TUSC2_BINARY_GROUP_COLORS,
            height=6,
            aspect=1.2,
        )
        g.fig.suptitle(
            f"Proportion of Cells by TUSC2 Expression Status", y=1.03, fontsize=16
        )
        g.set_axis_labels("Biological Context", "Proportion of Cells (%)")
        g.ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plot_basename_bar = create_filename(
            "P3_1_Barplot_TUSC2Binary_Consolidated", version="v2"
        )
        save_figure_and_data(
            g.fig, proportion_df, plot_basename_bar, layer1_fig_dir, layer1_data_dir
        )
        print(f"      Consolidated TUSC2 binary group bar plot saved.")
    except Exception as e:
        print(f"      ERROR generating consolidated TUSC2 binary bar plot: {e}")
        if "g" in locals():
            plt.close(g.fig)

print(
    "\n  --- Section 3.2: Layer 2 - TUSC2 Expression Across Subtypes Within Each Context (Violin Plot Data Export Fix) ---"
)

layer2_fig_dir = OUTPUT_SUBDIRS["tusc2_within_context_subtypes"]
layer2_data_dir = os.path.join(layer2_fig_dir, "data_for_graphpad")
layer2_stats_dir = os.path.join(layer2_fig_dir, "stat_results_python")
os.makedirs(layer2_fig_dir, exist_ok=True)
os.makedirs(layer2_data_dir, exist_ok=True)
os.makedirs(layer2_stats_dir, exist_ok=True)

for context_name, adata_ctx, _ in cohorts_for_characterization:
    print(
        f"\n    --- Processing Layer 2 TUSC2 Analysis for Context: {context_name} ---"
    )

    if (
        adata_ctx is None
        or adata_ctx.n_obs == 0
        or NK_SUBTYPE_PROFILED_COL not in adata_ctx.obs.columns
        or f"{TUSC2_GENE_NAME}_Expression_Raw" not in adata_ctx.obs.columns
        or TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns
    ):
        print(
            f"      Prerequisites not met for {context_name}. Skipping Layer 2 analysis."
        )
        continue

    obs_col_tusc2_expr = f"{TUSC2_GENE_NAME}_Expression_Raw"
    current_categories_l2 = adata_ctx.obs[
        NK_SUBTYPE_PROFILED_COL
    ].cat.categories.tolist()
    categories_to_plot_l2 = [
        cat
        for cat in REBUFFET_SUBTYPES_ORDERED
        if cat in current_categories_l2 and cat != "Unassigned"
    ]

    if not categories_to_plot_l2:
        print(
            f"      No valid subtypes to plot for {context_name}. Skipping 3.2.1 and 3.2.2."
        )
        continue

    adata_ctx_assigned_only = adata_ctx[
        adata_ctx.obs[NK_SUBTYPE_PROFILED_COL].isin(categories_to_plot_l2)
    ].copy()
    if adata_ctx_assigned_only.n_obs == 0:
        print(f"      No cells with assigned subtypes in {context_name}. Skipping.")
        continue
    adata_ctx_assigned_only.obs[NK_SUBTYPE_PROFILED_COL] = adata_ctx_assigned_only.obs[
        NK_SUBTYPE_PROFILED_COL
    ].cat.reorder_categories(categories_to_plot_l2, ordered=True)

    # --- 3.2.1: Violin/box plots & UMAP: TUSC2 expression across NK_Subtype_Profiled ---
    print(f"      --- 3.2.1: TUSC2 Expression across Subtypes in {context_name} ---")
    try:
        fig_l2_viol, ax_l2_viol = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            x=NK_SUBTYPE_PROFILED_COL,
            y=obs_col_tusc2_expr,
            data=adata_ctx_assigned_only.obs,
            ax=ax_l2_viol,
            palette=COMBINED_SUBTYPE_COLOR_PALETTE,
            hue=NK_SUBTYPE_PROFILED_COL,
            order=categories_to_plot_l2,
            legend=False,
        )
        sns.stripplot(
            x=NK_SUBTYPE_PROFILED_COL,
            y=obs_col_tusc2_expr,
            data=adata_ctx_assigned_only.obs,
            ax=ax_l2_viol,
            color="k",
            alpha=0.1,
            size=1.5,
            jitter=0.2,
            order=categories_to_plot_l2,
        )

        ax_l2_viol.set_title(
            f"{TUSC2_GENE_NAME} Expression by Subtype in {context_name}", fontsize=14
        )
        ax_l2_viol.set_xlabel(NK_SUBTYPE_PROFILED_COL, fontsize=12)
        ax_l2_viol.set_ylabel(
            f"{TUSC2_GENE_NAME} Expression (Log-Norm, Unscaled)", fontsize=12
        )
        plt.xticks(rotation=45, ha="right")

        data_for_stats_l2 = [
            adata_ctx_assigned_only.obs[obs_col_tusc2_expr][
                adata_ctx_assigned_only.obs[NK_SUBTYPE_PROFILED_COL] == subtype
            ]
            .dropna()
            .values
            for subtype in categories_to_plot_l2
        ]
        data_for_stats_l2_filtered = [arr for arr in data_for_stats_l2 if len(arr) >= 3]

        if len(data_for_stats_l2_filtered) >= 2:
            stat_l2, p_val_l2 = stats.kruskal(*data_for_stats_l2_filtered)
            print(
                f"        Kruskal-Wallis for {TUSC2_GENE_NAME} across subtypes in {context_name}: P-value={p_val_l2:.2e}"
            )
            ax_l2_viol.text(
                0.99,
                0.99,
                f"Kruskal-Wallis p = {p_val_l2:.2e}",
                transform=ax_l2_viol.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
            )
        plt.tight_layout()

        # --- Corrected Data Export for Violin Plot ---
        violin_data_to_export = adata_ctx_assigned_only.obs[
            [NK_SUBTYPE_PROFILED_COL, obs_col_tusc2_expr]
        ].copy()
        violin_data_to_export.index.name = "CellID"  # Set the index name
        violin_data_to_export = (
            violin_data_to_export.reset_index()
        )  # Convert index to column
        # Rename the expression column to the simple TUSC2_GENE_NAME for export consistency
        violin_data_to_export = violin_data_to_export.rename(
            columns={obs_col_tusc2_expr: TUSC2_GENE_NAME}
        )

        plot_basename_l2_viol = create_filename(
            f"P3_2_Violin_TUSC2_bySubtype", context_name=context_name, version="v1_fix1"
        )
        save_figure_and_data(
            fig_l2_viol,
            violin_data_to_export[["CellID", NK_SUBTYPE_PROFILED_COL, TUSC2_GENE_NAME]],
            plot_basename_l2_viol,
            layer2_fig_dir,
            layer2_data_dir,
        )
        print(f"        Violin plot of TUSC2 by subtype for {context_name} saved.")

    except Exception as e:
        print(
            f"        ERROR generating TUSC2 violin plot by subtype for {context_name}: {e}"
        )
        if (
            "fig_l2_viol" in locals()
            and fig_l2_viol
            and plt.fignum_exists(fig_l2_viol.number)
        ):
            plt.close(fig_l2_viol)

    # UMAP (This part was working, keeping it the same)
    if "X_umap" in adata_ctx.obsm:
        try:
            fig_l2_umap_tusc2, ax_l2_umap_tusc2 = plt.subplots(figsize=(7, 6))
            sc.pl.umap(
                adata_ctx,
                color=obs_col_tusc2_expr,
                ax=ax_l2_umap_tusc2,
                show=False,
                cmap="viridis",
                size=20,
                legend_loc="on data",
                title=f"{TUSC2_GENE_NAME} Expression in {context_name} (from .raw)",
            )
            plt.tight_layout()
            umap_coords_df_l2 = pd.DataFrame(
                adata_ctx.obsm["X_umap"],
                columns=["UMAP1", "UMAP2"],
                index=adata_ctx.obs_names,
            )
            tusc2_umap_export_df_l2 = adata_ctx.obs[[obs_col_tusc2_expr]].join(
                umap_coords_df_l2
            )
            tusc2_umap_export_df_l2 = tusc2_umap_export_df_l2.reset_index().rename(
                columns={"index": "CellID", obs_col_tusc2_expr: TUSC2_GENE_NAME}
            )
            plot_basename_l2_umap = create_filename(
                f"P3_2_UMAP_TUSC2Expression",
                context_name=context_name,
                version="v1_fix1",
            )
            save_figure_and_data(
                fig_l2_umap_tusc2,
                tusc2_umap_export_df_l2[["CellID", "UMAP1", "UMAP2", TUSC2_GENE_NAME]],
                plot_basename_l2_umap,
                layer2_fig_dir,
                layer2_data_dir,
            )
            print(f"        UMAP of TUSC2 expression for {context_name} saved.")
        except Exception as e:
            print(f"        ERROR generating TUSC2 UMAP for {context_name}: {e}")
            if (
                "fig_l2_umap_tusc2" in locals()
                and fig_l2_umap_tusc2
                and plt.fignum_exists(fig_l2_umap_tusc2.number)
            ):
                plt.close(fig_l2_umap_tusc2)
    else:
        print(
            f"        X_umap not found in {context_name}.obsm. Skipping TUSC2 UMAP plot."
        )

    # --- 3.2.2: Bar plots: Proportion of TUSC2_Binary_Group cells within each NK_Subtype_Profiled ---
    # (This part was working, keeping it the same)
    print(
        f"\n      --- 3.2.2: Proportion of TUSC2 Binary Groups within Subtypes for {context_name} ---"
    )
    try:
        crosstab_binary_subtype = pd.crosstab(
            index=adata_ctx_assigned_only.obs[NK_SUBTYPE_PROFILED_COL],
            columns=adata_ctx_assigned_only.obs[TUSC2_BINARY_GROUP_COL],
            dropna=False,
        )
        crosstab_binary_subtype_prop = crosstab_binary_subtype.apply(
            lambda x: 100 * x / x.sum(), axis=1
        ).fillna(0)

        if crosstab_binary_subtype_prop.empty:
            print(
                f"        Crosstab for TUSC2 binary group proportions in {context_name} is empty. Skipping plot."
            )
        else:
            print(
                f"        Proportions of TUSC2 Binary Groups within Subtypes for {context_name}:\n{crosstab_binary_subtype_prop}"
            )
            fig_l2_bar_binary, ax_l2_bar_binary = plt.subplots(figsize=(10, 7))
            colors_for_binary_bar = [
                TUSC2_BINARY_GROUP_COLORS.get(cat, "#CCCCCC")
                for cat in crosstab_binary_subtype_prop.columns
            ]
            crosstab_binary_subtype_prop.plot(
                kind="bar",
                stacked=True,
                ax=ax_l2_bar_binary,
                color=colors_for_binary_bar,
                width=0.8,
            )
            ax_l2_bar_binary.set_title(
                f"Proportion of {TUSC2_GENE_NAME} Binary Groups\nwithin Subtypes in {context_name}",
                fontsize=14,
            )
            ax_l2_bar_binary.set_xlabel(NK_SUBTYPE_PROFILED_COL, fontsize=12)
            ax_l2_bar_binary.set_ylabel("Proportion of Cells (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            ax_l2_bar_binary.legend(
                title=TUSC2_BINARY_GROUP_COL, bbox_to_anchor=(1.02, 1), loc="upper left"
            )
            plt.tight_layout(rect=[0, 0, 0.85, 0.95])
            plot_basename_l2_bar_binary = create_filename(
                "P3_2_Barplot_TUSC2Binary_by_Subtype",
                context_name=context_name,
                version="v1_fix1",
            )
            save_figure_and_data(
                fig_l2_bar_binary,
                crosstab_binary_subtype.reset_index(),
                plot_basename_l2_bar_binary,
                layer2_fig_dir,
                layer2_data_dir,
            )
            print(
                f"        Bar plot of TUSC2 binary group proportions by subtype for {context_name} saved."
            )
    except Exception as e:
        print(
            f"        ERROR generating TUSC2 binary group proportions plot for {context_name}: {e}"
        )
        if (
            "fig_l2_bar_binary" in locals()
            and fig_l2_bar_binary
            and plt.fignum_exists(fig_l2_bar_binary.number)
        ):
            plt.close(fig_l2_bar_binary)

    print(f"    --- End of Section 3.2 for {context_name} ---")

print("\n--- End of Section 3.2 (All Contexts) ---")

# %%
# PART 3: TUSC2 Analysis - A Layered Approach
# Section 3.3: Layer 3 - Visualizing the Impact of TUSC2 on Functional Signatures

print(
    "\n  --- Section 3.3: Layer 3 - Visualizing the Impact of TUSC2 on Functional Signatures ---"
)

# Define output directories
layer3_fig_dir = os.path.join(
    OUTPUT_SUBDIRS["tusc2_within_context_subtypes"], "TUSC2_Impact_on_Signatures"
)
layer3_data_dir = os.path.join(layer3_fig_dir, "data_for_graphpad")
layer3_stats_dir = os.path.join(layer3_fig_dir, "stat_results_python")
os.makedirs(layer3_fig_dir, exist_ok=True)
os.makedirs(layer3_data_dir, exist_ok=True)
os.makedirs(layer3_stats_dir, exist_ok=True)

score_column_names = [
    f"{set_name}_Score" for set_name in ALL_FUNCTIONAL_GENE_SETS.keys()
]

for context_name, adata_ctx, _ in cohorts_for_characterization:
    print(f"\n    --- Processing Layer 3 TUSC2 Impact for Context: {context_name} ---")

    # --- Step 1: Prerequisite Check ---
    if adata_ctx is None or TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns:
        print(
            f"      SKIPPING {context_name}: AnnData object or TUSC2 binary group column not found."
        )
        continue

    # --- Step 2: Calculate Functional Scores On-Demand ---
    # This is the critical fix: we calculate the scores within this cell to ensure they are always present.
    print("      Calculating functional signature scores...")
    for set_name, gene_list in ALL_FUNCTIONAL_GENE_SETS.items():
        score_col_name = f"{set_name}_Score"
        available_genes = map_gene_names(gene_list, adata_ctx.raw.var_names)
        if len(available_genes) >= MIN_GENES_FOR_SCORING:
            sc.tl.score_genes(
                adata_ctx,
                available_genes,
                score_name=score_col_name,
                use_raw=True,
                random_state=RANDOM_SEED,
            )
        else:
            adata_ctx.obs[score_col_name] = (
                np.nan
            )  # Ensure column exists even if score cannot be calculated

    # --- Step 3: Compare Functional Scores by TUSC2 Group ---
    print(
        f"      Comparing functional scores between TUSC2 expressing and non-expressing cells..."
    )
    stats_list = []
    for score_col in score_column_names:
        if score_col not in adata_ctx.obs.columns:
            continue

        group1 = adata_ctx.obs[score_col][
            adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[1]
        ].dropna()
        group0 = adata_ctx.obs[score_col][
            adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[0]
        ].dropna()

        stat, pval, n1, n0 = np.nan, np.nan, len(group1), len(group0)
        mean1, mean0 = (group1.mean() if n1 > 0 else np.nan), (
            group0.mean() if n0 > 0 else np.nan
        )

        if n1 >= 3 and n0 >= 3:
            stat, pval = stats.mannwhitneyu(group1, group0, alternative="two-sided")

        stats_list.append(
            {
                "Functional_Signature": score_col.replace("_Score", ""),
                "Mean_Score_Diff": mean1 - mean0,
                "P_Value_MWU": pval,
                "N_TUSC2_Expressed": n1,
                "N_TUSC2_Not_Expressed": n0,
            }
        )

    stats_df = pd.DataFrame(stats_list)
    if stats_df.empty or stats_df["P_Value_MWU"].isna().all():
        print(
            "      No valid comparisons possible. Skipping stats and plot generation."
        )
        continue

    # Perform FDR correction
    pvals = stats_df["P_Value_MWU"].dropna()
    if not pvals.empty:
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        stats_df["Q_Value_FDR"] = np.nan
        stats_df.loc[pvals.index, "Q_Value_FDR"] = qvals
        stats_df["Significance"] = stats_df["Q_Value_FDR"].apply(get_significance_stars)
    else:
        stats_df["Q_Value_FDR"] = np.nan
        stats_df["Significance"] = "ns"

    # Save the full statistical results
    stats_filename = create_filename(
        "P3_3_Stats_FuncScores_by_TUSC2Binary_Overall",
        context_name=context_name,
        version="v5_self_contained",
        ext="csv",
    )
    stats_df.to_csv(os.path.join(layer3_stats_dir, stats_filename), index=False)
    print(f"        Statistical results saved to {stats_filename}")

    # --- Step 4: Visualize the Impact with a Heatmap ---
    print(
        "        Generating summary heatmap of TUSC2's impact on functional signatures..."
    )
    try:
        plot_df = stats_df.set_index("Functional_Signature")[
            ["Mean_Score_Diff", "Significance"]
        ]

        fig, ax = plt.subplots(figsize=(7, 8))
        abs_max = (
            plot_df["Mean_Score_Diff"].abs().max()
            if not plot_df["Mean_Score_Diff"].isnull().all()
            else 0.1
        )

        annot_labels = plot_df.apply(
            lambda row: (
                f"{row['Mean_Score_Diff']:.2f}\n{row['Significance']}"
                if pd.notna(row["Mean_Score_Diff"])
                else "N/A"
            ),
            axis=1,
        ).values.reshape(plot_df.shape[0], 1)

        sns.heatmap(
            plot_df[["Mean_Score_Diff"]],
            annot=annot_labels,
            fmt="s",
            cmap="RdBu_r",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={
                "label": "Mean Score Difference\n(TUSC2 Expressed - Not Expressed)"
            },
            ax=ax,
        )
        ax.set_title(
            f"Impact of TUSC2 Expression on NK Functional Programs\n(Context: {context_name})",
            fontsize=14,
        )
        ax.set_ylabel("Functional Signature", fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        plt.tight_layout()

        plot_basename = create_filename(
            "P3_3_Heatmap_TUSC2_Func_Impact",
            context_name=context_name,
            version="v2_self_contained",
        )
        save_figure_and_data(
            fig, stats_df, plot_basename, layer3_fig_dir, layer3_data_dir
        )
        print(f"        Summary heatmap for {context_name} saved.")

    except Exception as e:
        print(f"        ERROR generating heatmap for {context_name}: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

print("\n--- End of Section 3.3 (All Contexts) ---")

# %%
# PART 3: TUSC2 Analysis - A Layered Approach
# Section 3.5: Layer 5 - DEG Analysis and Visualization for TUSC2 Binary Groups (Final)

print(
    "  --- Section 3.5: Layer 5 - DEG Analysis and Visualization for TUSC2 Binary Groups (Final) ---"
)

# Define base output directory
deg_l5_base_dir = os.path.join(OUTPUT_SUBDIRS["tusc2_analysis"], "5C_DEG_TUSC2_Binary")
os.makedirs(deg_l5_base_dir, exist_ok=True)

# Check for text adjustment library
try:
    from adjustText import adjust_text

    ADJUST_TEXT_AVAILABLE = True
    print(
        "      `adjustText` library is available for volcano plot label optimization."
    )
except ImportError:
    print(
        "      WARNING: `adjustText` not found. Volcano plot labels may overlap. `pip install adjustText`"
    )
    ADJUST_TEXT_AVAILABLE = False

for context_name, adata_ctx, _ in cohorts_for_characterization:
    print(
        f"\n    --- Processing DEG and Visuals for TUSC2 Groups in Context: {context_name} ---"
    )

    ctx_deg_fig_dir = os.path.join(deg_l5_base_dir, context_name, "figures")
    ctx_deg_data_dir = os.path.join(deg_l5_base_dir, context_name, "data_for_graphpad")
    ctx_deg_stats_dir = os.path.join(deg_l5_base_dir, context_name, "stat_results")
    os.makedirs(ctx_deg_fig_dir, exist_ok=True)
    os.makedirs(ctx_deg_data_dir, exist_ok=True)
    os.makedirs(ctx_deg_stats_dir, exist_ok=True)

    # --- Step 1: DEG Calculation ---
    group_high, group_low = TUSC2_BINARY_CATEGORIES[1], TUSC2_BINARY_CATEGORIES[0]
    if (
        TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns
        or adata_ctx.obs[TUSC2_BINARY_GROUP_COL].value_counts().get(group_high, 0) < 3
    ):
        print(
            f"      WARNING: Not enough cells in '{group_high}' group for {context_name}. Skipping DEG."
        )
        continue

    print(f"      Running DEG between '{group_high}' and '{group_low}'...")
    rank_genes_key = f"rank_genes_tusc2_binary_{context_name}"
    sc.tl.rank_genes_groups(
        adata_ctx,
        groupby=TUSC2_BINARY_GROUP_COL,
        groups=[group_high],
        reference=group_low,
        method="wilcoxon",
        use_raw=True,
        corr_method="benjamini-hochberg",
        key_added=rank_genes_key,
        n_genes=adata_ctx.raw.n_vars,
    )

    deg_df = sc.get.rank_genes_groups_df(
        adata_ctx, group=group_high, key=rank_genes_key
    )
    deg_df_filtered = deg_df[
        ~deg_df["names"].apply(is_gene_to_exclude_util)
        & (deg_df["names"] != TUSC2_GENE_NAME)
    ].copy()

    significant_degs = deg_df_filtered[
        (deg_df_filtered["pvals_adj"] < ADJ_PVAL_THRESHOLD_DEG)
        & (abs(deg_df_filtered["logfoldchanges"]) > LOGFC_THRESHOLD_DEG)
    ].copy()
    print(f"        Found {len(significant_degs)} significant DEGs.")
    deg_filename = create_filename(
        f"P3_5_Significant_DEG_results", context_name=context_name, ext="csv"
    )
    significant_degs.to_csv(os.path.join(ctx_deg_stats_dir, deg_filename), index=False)
    print(f"        Significant DEG list saved to {deg_filename}")

    # --- Step 2: Volcano Plot Visualization ---
    print("        Generating volcano plot...")
    try:
        volcano_df = deg_df_filtered.copy()
        volcano_df["-log10_pvals_adj"] = -np.log10(volcano_df["pvals_adj"] + 1e-300)

        volcano_df["Significance"] = "Not Significant"
        up_mask = (volcano_df["pvals_adj"] < ADJ_PVAL_THRESHOLD_DEG) & (
            volcano_df["logfoldchanges"] > LOGFC_THRESHOLD_DEG
        )
        down_mask = (volcano_df["pvals_adj"] < ADJ_PVAL_THRESHOLD_DEG) & (
            volcano_df["logfoldchanges"] < -LOGFC_THRESHOLD_DEG
        )
        volcano_df.loc[up_mask, "Significance"] = "Up-regulated"
        volcano_df.loc[down_mask, "Significance"] = "Down-regulated"

        fig, ax = plt.subplots(figsize=(4, 8))
        sns.scatterplot(
            data=volcano_df,
            x="logfoldchanges",
            y="-log10_pvals_adj",
            hue="Significance",
            palette={
                "Not Significant": "lightgray",
                "Up-regulated": "#d62728",
                "Down-regulated": "#1f77b4",
            },
            s=20,
            alpha=0.7,
            ax=ax,
            legend="full",
        )
        ax.set_title(f"TUSC2 Expression DEG in {context_name}", fontsize=16)
        ax.set_xlim([0, 0.5])
        ax.set_xlabel("Log2 Fold Change", fontsize=12)
        ax.set_ylabel("-log10(Adjusted p-value)", fontsize=12)
        ax.axvline(0, c="k", ls="--", lw=1)
        ax.axhline(-np.log10(ADJ_PVAL_THRESHOLD_DEG), c="k", ls="--", lw=1)

        top_genes_to_label = pd.concat(
            [
                significant_degs.sort_values("logfoldchanges", ascending=False).head(
                    10
                ),
                significant_degs.sort_values("logfoldchanges", ascending=True).head(10),
            ]
        ).drop_duplicates()

        texts = [
            ax.text(
                row["logfoldchanges"],
                -np.log10(row["pvals_adj"] + 1e-300),
                row["names"],
                fontsize=9,
            )
            for _, row in top_genes_to_label.iterrows()
        ]
        if ADJUST_TEXT_AVAILABLE:
            adjust_text(
                texts, ax=ax, arrowprops=dict(arrowstyle="-", color="black", lw=0.5)
            )

        plt.tight_layout()
        plot_basename_volcano = create_filename(
            "P3_5_Volcano_Plot_TUSC2_DEG", context_name=context_name, version="v2"
        )
        save_figure_and_data(
            fig, volcano_df, plot_basename_volcano, ctx_deg_fig_dir, ctx_deg_data_dir
        )
        print(f"        Volcano plot saved.")

    except Exception as e:
        print(f"        ERROR generating volcano plot: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

    # --- Step 3: Heatmap of Top DEGs ---
    if not significant_degs.empty:
        print("        Generating heatmap of top significant DEGs...")
        try:
            top_degs_for_heatmap = significant_degs.nlargest(25, "scores")[
                "names"
            ].tolist()

            # CORRECTED HEATMAP LOGIC: Call the plot with show=False, then get the figure object.
            sc.pl.heatmap(
                adata_ctx,
                var_names=top_degs_for_heatmap,
                groupby=TUSC2_BINARY_GROUP_COL,
                cmap="viridis",
                standard_scale="var",
                use_raw=True,
                show=False,
            )
            fig_heatmap = plt.gcf()  # Get the current figure created by scanpy
            fig_heatmap.suptitle(
                f"Top DEGs in {context_name}\n(TUSC2 Expressed vs. Not Expressed)",
                fontsize=14,
            )
            fig_heatmap.tight_layout()

            plot_basename_heatmap = create_filename(
                "P3_5_Heatmap_Top_DEGs", context_name=context_name, version="v3_final"
            )
            save_figure_and_data(
                fig_heatmap, None, plot_basename_heatmap, ctx_deg_fig_dir, None
            )
            print(f"        DEG heatmap saved.")
        except Exception as e:
            print(f"        ERROR generating DEG heatmap: {e}")
            if "fig_heatmap" in locals() and plt.fignum_exists(fig_heatmap.number):
                plt.close(fig_heatmap)

print("\n--- End of Section 3.5 ---")

# %%
# PART 3: TUSC2 Analysis - A Layered Approach
# Section 3.5.2: Overlap Analysis of TUSC2 DEGs with Blood NK Subtype Markers

print(
    "\n  --- Section 3.5.2: Overlap Analysis of TUSC2 DEGs with Blood NK Subtype Markers ---"
)

if "ref_rebuffet_markers" not in locals() or not ref_rebuffet_markers:
    print(
        "      ERROR: `ref_rebuffet_markers` not found. Cannot perform comparison. Please run Part 1.3.3."
    )
else:
    # Get the base directory where the DEG results were saved in the previous step
    deg_l5_base_dir = os.path.join(
        OUTPUT_SUBDIRS["tusc2_analysis"], "5C_DEG_TUSC2_Binary"
    )

    def analyze_and_plot_overlap(
        deg_subset, direction, context, ref_markers, fig_dir, data_dir, stats_dir
    ):
        """
        Analyzes the overlap between a set of DEGs and reference subtype markers.

        Args:
            deg_subset (pd.DataFrame): DataFrame of DEGs (either UP or DOWN).
            direction (str): 'UP' or 'DOWN', for labeling outputs.
            context (str): The context being analyzed (e.g., 'NormalTissue').
            ref_markers (dict): Dictionary of reference markers for each subtype.
            fig_dir (str): Path to save figures.
            data_dir (str): Path to save data for GraphPad.
            stats_dir (str): Path to save the overlap table.
        """
        if deg_subset.empty:
            print(
                f"        No DEGs regulated {direction} in TUSC2_Expressed cells for {context}. Skipping."
            )
            return

        print(
            f"\n        Analyzing overlap for {direction}-regulated DEGs in {context}:"
        )

        # Find which reference subtype markers are present in the DEG list
        overlap_data = []
        for _, row in deg_subset.iterrows():
            gene = row["names"]
            overlapping_subtypes = [
                st for st, markers in ref_markers.items() if gene in markers
            ]
            if overlapping_subtypes:
                row_dict = row.to_dict()
                row_dict["Overlapping_Ref_Blood_Subtypes"] = ", ".join(
                    overlapping_subtypes
                )
                overlap_data.append(row_dict)

        if not overlap_data:
            print(
                f"          No overlap found between {direction}-regulated DEGs and reference markers."
            )
            return

        overlap_df = pd.DataFrame(overlap_data).sort_values(
            by="scores", ascending=(direction == "DOWN")
        )
        print(
            f"          Found {len(overlap_df)} {direction}-regulated DEGs that are also reference markers."
        )

        # Save the detailed table of overlapping genes
        filename = create_filename(
            f"P3_5_Overlap_TUSC2_{direction}DEGs_vs_RefMarkers",
            context_name=context,
            ext="csv",
        )
        overlap_df.to_csv(os.path.join(stats_dir, filename), index=False)
        print(f"          Overlap table saved to {filename}")

        # Enhanced scoring and selection for top 10 overlapping GEPs
        print(f"          Scoring and selecting top 10 overlapping genes...")

        # Calculate composite overlap score for each gene
        overlap_df_scored = overlap_df.copy()
        overlap_scores = []

        for _, row in overlap_df_scored.iterrows():
            gene = row["names"]
            overlapping_subtypes = row["Overlapping_Ref_Blood_Subtypes"].split(", ")

            # Composite scoring formula:
            # - Base scanpy score (importance within analysis)
            # - Number of subtypes overlapped (broader relevance)
            # - Adjusted p-value significance (-log10 for better weighting)
            # - Log fold change magnitude (effect size)

            base_score = abs(row["scores"]) if pd.notna(row.get("scores", 0)) else 0
            overlap_count_bonus = (
                len(overlapping_subtypes) * 2
            )  # Bonus for multiple overlaps
            significance_score = -np.log10(row.get("pvals_adj", 1e-5) + 1e-10)
            effect_size_score = abs(row.get("logfoldchanges", 0))

            # Composite score combines all factors
            composite_score = (
                base_score * 1.0  # Primary importance
                + overlap_count_bonus * 0.5  # Multi-subtype relevance
                + significance_score * 0.3  # Statistical significance
                + effect_size_score * 0.2  # Biological effect size
            )

            overlap_scores.append(composite_score)

        overlap_df_scored["Overlap_Composite_Score"] = overlap_scores

        # Sort by composite score and select top 10
        top_overlapping_genes = overlap_df_scored.nlargest(
            40, "Overlap_Composite_Score"
        )

        print(f"          Selected top 10 genes based on composite overlap scoring:")
        for i, (_, gene_row) in enumerate(top_overlapping_genes.iterrows(), 1):
            gene_name = gene_row["names"]
            score = gene_row["Overlap_Composite_Score"]
            subtypes = gene_row["Overlapping_Ref_Blood_Subtypes"]
            print(
                f"            {i:2d}. {gene_name} (score: {score:.2f}, subtypes: {subtypes})"
            )

        # Create a presence/absence matrix for the top 10 genes
        pivot_df = top_overlapping_genes.copy()
        for subtype in REBUFFET_SUBTYPES_ORDERED:
            pivot_df[subtype] = pivot_df["Overlapping_Ref_Blood_Subtypes"].apply(
                lambda x: 1 if subtype in x else 0
            )

        pivot_heatmap = pivot_df.set_index("names")[REBUFFET_SUBTYPES_ORDERED]

        if not pivot_heatmap.empty:
            # Calculate optimal column width (2/3 of original, rounded to nearest integer)
            original_width = 8
            new_width = int(round(original_width * (2 / 3)))
            print(
                f"          Adjusting plot width: {original_width} -> {new_width} (2/3 width)"
            )

            fig, ax = plt.subplots(
                figsize=(new_width, max(5, pivot_heatmap.shape[0] * 0.4))
            )
            sns.heatmap(
                pivot_heatmap,
                cmap="cividis",
                cbar=False,
                linewidths=0.5,
                linecolor="lightgray",
                ax=ax,
                square=False,  # Allow rectangular cells for better readability
                xticklabels=True,
                yticklabels=True,
            )
            ax.set_title(
                f"Top 10 TUSC2 {direction}-regulated Overlapping DEGs in {context}\nvs. Blood Reference Subtype Markers",
                fontsize=12,
                pad=15,
            )
            ax.set_xlabel("Reference Blood NK Subtype", fontsize=10)
            ax.set_ylabel(f"Top Overlapping DEGs ({direction}-regulated)", fontsize=10)

            # Rotate x-axis labels for better readability in narrower plot
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            plot_filename = create_filename(
                f"P3_5_Heatmap_Overlap_TUSC2_{direction}DEGs",
                context_name=context,
                version="v2_top10",
            )

            # Save both the plot and the enhanced data with scoring
            enhanced_export_data = top_overlapping_genes[
                [
                    "names",
                    "Overlapping_Ref_Blood_Subtypes",
                    "Overlap_Composite_Score",
                    "scores",
                    "logfoldchanges",
                    "pvals_adj",
                ]
            ].copy()
            enhanced_export_data.columns = [
                "Gene",
                "Overlapping_Subtypes",
                "Composite_Score",
                "Scanpy_Score",
                "Log_Fold_Change",
                "Adj_P_Value",
            ]

            save_figure_and_data(
                fig, enhanced_export_data, plot_filename, fig_dir, data_dir
            )
            print(
                f'          Enhanced overlap heatmap saved (top 10 genes, {new_width}" width).'
            )

    for context_name, _, _ in cohorts_for_characterization:
        print(f"\n    --- Testing TUSC2 DEG Overlap for Context: {context_name} ---")

        # --- Robustly load the DEG results from the previous step ---
        ctx_deg_stats_dir = os.path.join(deg_l5_base_dir, context_name, "stat_results")
        # THIS IS THE CRITICAL FIX: Use the same versioning as the saving cell.
        deg_filename = create_filename(
            f"P3_5_Significant_DEG_results", context_name=context_name, ext="csv"
        )
        filepath_deg_ctx = os.path.join(ctx_deg_stats_dir, deg_filename)

        if not os.path.exists(filepath_deg_ctx):
            print(
                f"      WARNING: Significant DEG file not found at '{filepath_deg_ctx}'. Skipping overlap analysis for {context_name}."
            )
            continue

        significant_degs_df = pd.read_csv(filepath_deg_ctx)
        if significant_degs_df.empty:
            print(
                f"      Significant DEG file for {context_name} is empty. No overlap to analyze."
            )
            continue

        print(
            f"      Loaded {len(significant_degs_df)} significant DEGs for {context_name}."
        )
        degs_up = significant_degs_df[significant_degs_df["logfoldchanges"] > 0].copy()
        degs_down = significant_degs_df[
            significant_degs_df["logfoldchanges"] < 0
        ].copy()

        # --- Run the analysis and plotting function ---
        ctx_overlap_fig_dir = os.path.join(
            deg_l5_base_dir, context_name, "figures_deg_overlap"
        )
        ctx_overlap_data_dir = os.path.join(
            deg_l5_base_dir, context_name, "data_for_graphpad_deg_overlap"
        )
        os.makedirs(ctx_overlap_fig_dir, exist_ok=True)
        os.makedirs(ctx_overlap_data_dir, exist_ok=True)

        analyze_and_plot_overlap(
            degs_up,
            "UP",
            context_name,
            ref_rebuffet_markers,
            ctx_overlap_fig_dir,
            ctx_overlap_data_dir,
            ctx_deg_stats_dir,
        )
        analyze_and_plot_overlap(
            degs_down,
            "DOWN",
            context_name,
            ref_rebuffet_markers,
            ctx_overlap_fig_dir,
            ctx_overlap_data_dir,
            ctx_deg_stats_dir,
        )

    print(f"\n    --- End of Overlap Analysis ---")

print("\n--- End of Section 3.5.2 ---")

# %%
# PART 4: Cross-Context Synthesis & Comparative Insights
# Section 4.2.3: Synthesize TUSC2's Impact on Developmental and Functional Signatures

print("\n--- PART 4: Cross-Context Synthesis & Comparative Insights ---")
print(
    "\n    --- 4.2.3: Synthesize TUSC2's Impact on Developmental and Functional Signatures (Heatmap Version) ---"
)

# Define output directories for cross-context synthesis
synthesis_impact_dir = os.path.join(
    OUTPUT_SUBDIRS["cross_context_synthesis"], "TUSC2_Functional_Impact_Comparison"
)
synthesis_impact_data_dir = os.path.join(synthesis_impact_dir, "data_for_graphpad")
os.makedirs(synthesis_impact_dir, exist_ok=True)
os.makedirs(synthesis_impact_data_dir, exist_ok=True)

# Load overall TUSC2 functional impact stats for each context
all_tusc2_impact_stats = []
base_stats_path = os.path.join(
    OUTPUT_SUBDIRS["tusc2_within_context_subtypes"],
    "TUSC2_Impact_on_Signatures",
    "stat_results_python",
)

for context_name, _, _ in cohorts_for_characterization:
    stats_filename = create_filename(
        "P3_3_Stats_FuncScores_by_TUSC2Binary_Overall",
        context_name=context_name,
        version="v5_self_contained",
        ext="csv",
    )
    stats_filepath = os.path.join(base_stats_path, stats_filename)

    print(f"        Attempting to load: {stats_filepath}")
    if os.path.exists(stats_filepath):
        context_stats_df = pd.read_csv(stats_filepath)
        context_stats_df["Context"] = context_name
        all_tusc2_impact_stats.append(context_stats_df)
        print(
            f"          Successfully processed TUSC2 impact stats for {context_name}."
        )
    else:
        print(f"          WARNING: Stats file not found for {context_name}.")

if all_tusc2_impact_stats:
    # Combine all context data
    combined_tusc2_stats = pd.concat(all_tusc2_impact_stats, ignore_index=True)

    # Separate developmental and functional signatures
    dev_signatures = [
        "Regulatory_NK",
        "Intermediate_NK",
        "Mature_Cytotoxic_NK",
        "Adaptive_NK",
    ]
    func_signatures = [
        sig
        for sig in combined_tusc2_stats["Functional_Signature"].unique()
        if sig not in dev_signatures
    ]
    neuro_signatures = [
        sig
        for sig in combined_tusc2_stats["Functional_Signature"].unique()
        if sig in NEUROTRANSMITTER_RECEPTOR_GENE_SETS.keys()
    ]
    il_signatures = [
        sig
        for sig in combined_tusc2_stats["Functional_Signature"].unique()
        if sig in INTERLEUKIN_DOWNSTREAM_GENE_SETS.keys()
    ]

    # Generate heatmap for TUSC2 Impact on Developmental State
    print("\n      Generating heatmap for TUSC2 Impact on Developmental State...")
    dev_stats = combined_tusc2_stats[
        combined_tusc2_stats["Functional_Signature"].isin(dev_signatures)
    ]
    if not dev_stats.empty:
        # Use existing Q_Value_FDR column from the loaded statistics
        dev_pivot_diff = dev_stats.pivot_table(
            index="Functional_Signature", columns="Context", values="Mean_Score_Diff"
        )
        dev_pivot_qval = dev_stats.pivot_table(
            index="Functional_Signature", columns="Context", values="Q_Value_FDR"
        )
        dev_pivot_diff = dev_pivot_diff.reindex(dev_signatures).dropna(how="all")
        dev_pivot_qval = dev_pivot_qval.reindex(dev_pivot_diff.index)

        if not dev_pivot_diff.empty:
            # Create annotation labels with values + significance stars
            annot_labels = dev_pivot_diff.apply(
                lambda x: x.map("{:.3f}".format)
            ) + dev_pivot_qval.apply(lambda x: x.map(get_significance_stars))

            fig, ax = plt.subplots(figsize=(6, 4))
            abs_max = (
                np.nanmax(np.abs(dev_pivot_diff.values))
                if not np.all(np.isnan(dev_pivot_diff.values))
                else 0.1
            )

            sns.heatmap(
                dev_pivot_diff,
                annot=annot_labels,
                fmt="s",
                cmap="RdBu_r",
                center=0,
                vmin=-abs_max,
                vmax=abs_max,
                linewidths=0.5,
                cbar_kws={"label": "Mean Score Difference\n(TUSC2+ vs TUSC2-)"},
                ax=ax,
            )
            ax.set_title("TUSC2 Impact on Developmental State", fontsize=14)
            ax.set_xlabel("Context", fontsize=12)
            ax.set_ylabel("Developmental Signature", fontsize=12)
            plt.tight_layout()

            plot_basename_dev = create_filename(
                "P4_2_Heatmap_TUSC2_Impact_DevelopmentalState", version="v4_final"
            )
            save_figure_and_data(
                fig,
                dev_stats,
                plot_basename_dev,
                synthesis_impact_dir,
                synthesis_impact_data_dir,
            )
            print("        Developmental state heatmap with significance stars saved.")

    # Define signature categories for separate heatmaps
    core_functional_signatures = [
        sig
        for sig in combined_tusc2_stats["Functional_Signature"].unique()
        if sig in FUNCTIONAL_GENE_SETS.keys()
    ]

    tang_derived_signatures = [
        sig
        for sig in combined_tusc2_stats["Functional_Signature"].unique()
        if sig in TANG_DEVELOPMENTAL_GENE_SETS.keys()
    ]

    # Remove overlaps - prioritize by category
    remaining_func_signatures = [
        sig
        for sig in func_signatures
        if sig not in neuro_signatures
        and sig not in il_signatures
        and sig not in core_functional_signatures
        and sig not in tang_derived_signatures
    ]

    # Helper function to create category heatmap
    def create_category_heatmap(
        stats_data, signatures_list, title, basename_suffix, category_name
    ):
        if not signatures_list:
            print(f"        Skipping {category_name} heatmap - no signatures found.")
            return

        category_stats = stats_data[
            stats_data["Functional_Signature"].isin(signatures_list)
        ]
        if category_stats.empty:
            print(f"        Skipping {category_name} heatmap - no data found.")
            return

        pivot_diff = category_stats.pivot_table(
            index="Functional_Signature", columns="Context", values="Mean_Score_Diff"
        )
        pivot_qval = category_stats.pivot_table(
            index="Functional_Signature", columns="Context", values="Q_Value_FDR"
        )

        if pivot_diff.empty:
            print(f"        Skipping {category_name} heatmap - empty pivot table.")
            return

        # Create annotation labels with values + significance stars
        annot_labels = pivot_diff.apply(
            lambda x: x.map("{:.3f}".format)
        ) + pivot_qval.apply(lambda x: x.map(get_significance_stars))

        # Determine figure size based on number of signatures
        fig_height = max(4, pivot_diff.shape[0] * 0.4)
        fig, ax = plt.subplots(figsize=(6, fig_height))

        abs_max = (
            np.nanmax(np.abs(pivot_diff.values))
            if not np.all(np.isnan(pivot_diff.values))
            else 0.1
        )

        sns.heatmap(
            pivot_diff,
            annot=annot_labels,
            fmt="s",
            cmap="RdBu_r",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={"label": "Mean Score Difference\n(TUSC2+ vs TUSC2-)"},
            ax=ax,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Context", fontsize=12)
        ax.set_ylabel(f"{category_name} Signature", fontsize=12)
        plt.tight_layout()

        plot_basename = create_filename(
            f"P4_2_Heatmap_TUSC2_Impact_{basename_suffix}", version="v4_final"
        )
        save_figure_and_data(
            fig,
            category_stats,
            plot_basename,
            synthesis_impact_dir,
            synthesis_impact_data_dir,
        )
        print(f"        {category_name} heatmap with significance stars saved.")

    # Generate separate heatmaps for each category
    print("\n      Generating category-specific TUSC2 impact heatmaps...")

    # 1. Core Functional Signatures
    create_category_heatmap(
        combined_tusc2_stats,
        core_functional_signatures,
        "TUSC2 Impact on Core Functional Capacity",
        "CoreFunctionalCapacity",
        "Core Functional",
    )

    # 2. Interleukin Signatures
    create_category_heatmap(
        combined_tusc2_stats,
        il_signatures,
        "TUSC2 Impact on Interleukin Pathways",
        "InterleukinPathways",
        "Interleukin",
    )

    # 3. Neurotransmitter Signatures
    create_category_heatmap(
        combined_tusc2_stats,
        neuro_signatures,
        "TUSC2 Impact on Neurotransmitter Receptors",
        "NeurotransmitterReceptors",
        "Neurotransmitter",
    )

    # 4. Tang-Derived Signatures (Tissue-Based Developmental)
    create_category_heatmap(
        combined_tusc2_stats,
        tang_derived_signatures,
        "TUSC2 Impact on Tang Developmental Programs",
        "TangDevelopmentalPrograms",
        "Tang Developmental",
    )

    # 5. Remaining Functional Signatures (if any)
    if remaining_func_signatures:
        create_category_heatmap(
            combined_tusc2_stats,
            remaining_func_signatures,
            "TUSC2 Impact on Additional Functional Programs",
            "AdditionalFunctionalPrograms",
            "Additional Functional",
        )


print("\n--- End of Section 4.2.3 ---")

# %%
# PART 4: Cross-Context Synthesis & Comparative Insights
# Section 4.3: Analyzing TUSC2's Impact on NK Subtype Programs (Cross-Context)

print(
    "\n  --- Section 4.3: Analyzing TUSC2's Impact on NK Subtype Programs (Polished) ---"
)

# Define output directories for subtype programs analysis
synthesis_subtype_dir = os.path.join(
    OUTPUT_SUBDIRS["cross_context_synthesis"], "TUSC2_Impact_on_Subtype_Programs"
)
synthesis_subtype_fig_dir = os.path.join(synthesis_subtype_dir, "figures")
synthesis_subtype_data_dir = os.path.join(synthesis_subtype_dir, "data_for_graphpad")
os.makedirs(synthesis_subtype_fig_dir, exist_ok=True)
os.makedirs(synthesis_subtype_data_dir, exist_ok=True)

# Use the blood reference markers to define subtype-specific gene programs
if "ref_rebuffet_markers" not in globals() or not ref_rebuffet_markers:
    print(
        "    ERROR: Blood reference markers not found. Cannot analyze subtype programs."
    )
else:
    all_subtype_stats_results = []

    for context_name, adata_ctx, _ in cohorts_for_characterization:
        print(
            f"\n    --- Analyzing TUSC2 Impact on Subtype Programs for Context: {context_name} ---"
        )
        if TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns or adata_ctx.raw is None:
            continue

        # Calculate signature scores for each NK subtype program
        for subtype in REBUFFET_SUBTYPES_ORDERED:
            if subtype not in ref_rebuffet_markers:
                continue

            subtype_genes = ref_rebuffet_markers[subtype]
            score_name = f"{subtype}_Program_Score"

            # Calculate signature score using scanpy
            sc.tl.score_genes(
                adata_ctx,
                subtype_genes,
                score_name=score_name,
                use_raw=True,
                random_state=42,
                n_bins=50,
            )

            # Extract scores by TUSC2 groups
            tusc2_pos_mask = (
                adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[1]
            )
            tusc2_neg_mask = (
                adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[0]
            )

            scores_pos = adata_ctx.obs.loc[tusc2_pos_mask, score_name]
            scores_neg = adata_ctx.obs.loc[tusc2_neg_mask, score_name]

            p_value, mean_diff = np.nan, np.nan
            if len(scores_pos) >= 3 and len(scores_neg) >= 3:
                mean_diff = scores_pos.mean() - scores_neg.mean()
                _, p_value = stats.mannwhitneyu(
                    scores_pos, scores_neg, alternative="two-sided"
                )

            all_subtype_stats_results.append(
                {
                    "Context": context_name,
                    "Subtype_Program": subtype,
                    "Mean_Score_Diff": mean_diff,
                    "P_Value": p_value,
                }
            )

    # --- Step 2: Assemble results and create a summary heatmap ---
    if all_subtype_stats_results:
        summary_subtype_stats_df = pd.DataFrame(all_subtype_stats_results).dropna()
        pvals_to_correct = summary_subtype_stats_df["P_Value"]
        if not pvals_to_correct.empty:
            from statsmodels.stats.multitest import multipletests

            _, qvals, _, _ = multipletests(pvals_to_correct, method="fdr_bh")
            summary_subtype_stats_df["Q_Value"] = qvals

        # --- Step 3: Polish and Visualize ---
        heatmap_pivot_diff = summary_subtype_stats_df.pivot_table(
            index="Subtype_Program", columns="Context", values="Mean_Score_Diff"
        )
        heatmap_pivot_qval = summary_subtype_stats_df.pivot_table(
            index="Subtype_Program", columns="Context", values="Q_Value"
        )

        heatmap_pivot_diff = heatmap_pivot_diff.reindex(
            REBUFFET_SUBTYPES_ORDERED
        ).dropna(how="all")
        heatmap_pivot_qval = heatmap_pivot_qval.reindex(heatmap_pivot_diff.index)

        annot_labels = heatmap_pivot_diff.apply(
            lambda x: x.map("{:.3f}".format)
        ) + heatmap_pivot_qval.apply(lambda x: x.map(get_significance_stars))

        print(
            "\n      --- Generating Final Summary: Impact of TUSC2 on NK Subtype Programs ---"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        abs_max = (
            np.nanmax(np.abs(heatmap_pivot_diff.values))
            if not np.all(np.isnan(heatmap_pivot_diff.values))
            else 0.1
        )

        sns.heatmap(
            heatmap_pivot_diff,
            annot=annot_labels,
            fmt="s",
            cmap="RdBu_r",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={
                "label": "Mean Program Score Difference\n(TUSC2 Expressed vs. Not Expressed)",
                "shrink": 0.8,
            },
            ax=ax,
        )
        ax.set_title(
            "TUSC2 Expression is Associated with Mature Cytotoxic Subtypes",
            fontsize=14,
            pad=20,
        )
        ax.set_xlabel("Biological Context", fontsize=12)
        ax.set_ylabel("NK Subtype Signature", fontsize=12)
        plt.tight_layout()

        plot_basename_summary = create_filename(
            "P4_3_Heatmap_TUSC2_Impact_on_Subtype_Programs", version="v5_final"
        )
        save_figure_and_data(
            fig,
            summary_subtype_stats_df,
            plot_basename_summary,
            synthesis_subtype_fig_dir,
            synthesis_subtype_data_dir,
        )
        print("\n      Summary heatmap with significance and full stats table saved.")

print("\n--- End of Section 4.3 (Final) ---")

# %%
# PART 4: Cross-Context Synthesis & Comparative Insights
# Section 4.4: Analyzing the Impact of TUSC2 Expression on Developmental Marker Programs

print(
    "\n  --- Section 4.4: Analyzing the Impact of TUSC2 Expression on Developmental State ---"
)

# Define output directories for developmental state analysis
synthesis_dev_dir = os.path.join(
    OUTPUT_SUBDIRS["cross_context_synthesis"], "TUSC2_Impact_on_Dev_State"
)
synthesis_dev_fig_dir = os.path.join(synthesis_dev_dir, "figures")
synthesis_dev_data_dir = os.path.join(synthesis_dev_dir, "data_for_graphpad")
os.makedirs(synthesis_dev_fig_dir, exist_ok=True)
os.makedirs(synthesis_dev_data_dir, exist_ok=True)

if "MURINE_DEV_MARKER_ORTHOLOGS" not in locals():
    print("    ERROR: `MURINE_DEV_MARKER_ORTHOLOGS` not found. Skipping Section 4.4.")
else:
    all_dev_stats_results = []

    for context_name, adata_ctx, _ in cohorts_for_characterization:
        print(
            f"\n    --- Analyzing TUSC2 Impact on Developmental Markers for Context: {context_name} ---"
        )
        if TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns or adata_ctx.raw is None:
            continue

        for marker_gene in MURINE_DEV_MARKER_ORTHOLOGS:
            if marker_gene not in adata_ctx.raw.var_names:
                continue

            gene_expression = adata_ctx.raw[:, marker_gene].X.toarray().flatten()
            scores_pos = gene_expression[
                adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[1]
            ]
            scores_neg = gene_expression[
                adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[0]
            ]

            p_value, mean_diff = np.nan, np.nan
            if len(scores_pos) >= 3 and len(scores_neg) >= 3:
                mean_diff = scores_pos.mean() - scores_neg.mean()
                _, p_value = stats.mannwhitneyu(
                    scores_pos, scores_neg, alternative="two-sided"
                )

            all_dev_stats_results.append(
                {
                    "Context": context_name,
                    "Developmental_Marker": marker_gene,
                    "Mean_Diff_TUSC2pos_vs_neg": mean_diff,
                    "P_Value": p_value,
                }
            )

    if all_dev_stats_results:
        summary_dev_stats_df = pd.DataFrame(all_dev_stats_results).dropna()
        p_vals_to_correct = summary_dev_stats_df["P_Value"]
        if not p_vals_to_correct.empty:
            from statsmodels.stats.multitest import multipletests

            _, qvals, _, _ = multipletests(p_vals_to_correct, method="fdr_bh")
            summary_dev_stats_df["Q_Value_FDR"] = qvals

        heatmap_pivot_diff = summary_dev_stats_df.pivot_table(
            index="Developmental_Marker",
            columns="Context",
            values="Mean_Diff_TUSC2pos_vs_neg",
        )
        heatmap_pivot_qval = summary_dev_stats_df.pivot_table(
            index="Developmental_Marker", columns="Context", values="Q_Value_FDR"
        )

        # Sort markers by developmental order
        dev_order = [
            "KIT",
            "ID2",
            "GATA3",
            "TCF7",
            "IL2RB",
            "SELL",
            "CD27",
            "KLRC1",
            "NCAM1",
            "TBX21",
            "EOMES",
            "ITGAM",
            "FCGR3A",
            "GZMB",
            "PRF1",
            "KLRG1",
            "B3GAT1",
        ]
        ordered_markers = [m for m in dev_order if m in heatmap_pivot_diff.index] + [
            m for m in heatmap_pivot_diff.index if m not in dev_order
        ]

        heatmap_pivot_diff = heatmap_pivot_diff.reindex(ordered_markers).dropna(
            how="all"
        )
        heatmap_pivot_qval = heatmap_pivot_qval.reindex(heatmap_pivot_diff.index)

        annot_labels = heatmap_pivot_diff.apply(
            lambda x: x.map("{:.3f}".format)
        ) + heatmap_pivot_qval.apply(lambda x: x.map(get_significance_stars))

        print(
            "\n      --- Generating Summary Heatmap: Impact of TUSC2 on Developmental Marker Expression ---"
        )

        fig, ax = plt.subplots(figsize=(8, max(6, heatmap_pivot_diff.shape[0] * 0.28)))
        abs_max = (
            np.nanmax(np.abs(heatmap_pivot_diff.values))
            if not np.all(np.isnan(heatmap_pivot_diff.values))
            else 0.1
        )

        sns.heatmap(
            heatmap_pivot_diff,
            annot=annot_labels,
            fmt="s",
            cmap="RdBu_r",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={
                "label": "Mean Expression Difference\n(TUSC2 Expressed vs. Not Expressed)"
            },
            ax=ax,
        )
        ax.set_title(
            "Impact of TUSC2 Expression on Developmental State Markers",
            fontsize=14,
            pad=20,
        )
        ax.set_xlabel("Context", fontsize=12)
        ax.set_ylabel("Developmental Marker", fontsize=12)
        plt.tight_layout()

        plot_basename_summary = create_filename(
            "P4_4_Heatmap_TUSC2_Impact_on_Dev_Markers", version="v2_final"
        )
        save_figure_and_data(
            fig,
            summary_dev_stats_df,
            plot_basename_summary,
            synthesis_dev_fig_dir,
            synthesis_dev_data_dir,
        )
        print(
            "\n      Summary heatmap of TUSC2's impact on developmental markers saved."
        )

print("\n--- End of Section 4.4 ---")

print(
    "\n--- Analysis Complete: NK Cell Transcriptomics and TUSC2 Function Analysis v3.2 (Cleaned) ---"
)
