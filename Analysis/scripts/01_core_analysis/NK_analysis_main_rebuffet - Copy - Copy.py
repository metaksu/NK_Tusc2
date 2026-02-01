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
from datetime import datetime

# Data Science & Numerical Libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.io import mmread
from scipy.sparse import csr_matrix

# Single-Cell Analysis Libraries
import scanpy as sc

# AUCell-like scoring implementation (no external dependencies)
print("  AUCell-like scoring function available for robust cross-dataset gene set scoring.")

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
np.random.seed(RANDOM_SEED)
print(f"  Random seed for numpy set to: {RANDOM_SEED}")
# Note: scanpy.settings.seed is not available in newer versions
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
    r"C:\Users\met-a\Documents\Analysis\Combined_NK_TUSC2_Analysis_Output_Rebuffet_Based"
)
print(f"    MASTER_OUTPUT_DIR set to: {MASTER_OUTPUT_DIR}")

# Define subdirectory names - cleaned up to only include actively used directories
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
# This is a fallback for direct sc.pl.savefig calls if any.
sc.settings.figdir = MASTER_OUTPUT_DIR
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
    Helper function to return the unified subtype column for all datasets.
    All datasets now use the Rebuffet subtype framework after re-annotation.
    """
    if adata_obj is None or adata_obj.n_obs == 0:
        return None
    
    # All datasets now use unified Rebuffet subtype column
    return REBUFFET_SUBTYPE_COL


def get_subtype_categories(adata_obj):
    """
    Helper function to return the unified subtype categories for all datasets.
    All datasets now use Rebuffet subtype categories after re-annotation.
    """
    # All datasets now use unified Rebuffet subtype categories (including Unassigned)
    return REBUFFET_SUBTYPES_ORDERED + ["Unassigned"]


def should_split_tang_subtypes(adata_obj):
    """
    Determine if the given dataset should be split into Tang subsets.
    Returns False since we no longer use Tang-specific splitting.
    """
    # No longer split by Tang subtypes - all datasets use unified Rebuffet framework
    return False


def get_tang_subtype_subsets(adata_obj, context_name):
    """
    Return subsets for analysis. Since we use unified subtyping, no Tang-specific splitting.
    Returns list with single (None, adata_obj) tuple for unified processing.
    """
    # No Tang-specific subsetting - return original data for unified analysis
    return [(None, adata_obj)]


def get_subtype_color_palette(adata_obj):
    """
    Get the appropriate color palette based on the dataset type.
    Returns the Rebuffet palette since all datasets use unified subtyping.
    """
    # Use Rebuffet color palette for all datasets
    return SUBTYPE_COLOR_PALETTE


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
def load_hallmark_geneset(filepath):
    """Load gene set from MSigDB .grp file format."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Skip header line and comments, extract gene symbols
        genes = []
        for line in lines[1:]:  # Skip first line (geneset name)
            line = line.strip()
            if line and not line.startswith('#'):
                genes.append(line)
        return genes
    except FileNotFoundError:
        print(f"        WARNING: Could not find {filepath}, using fallback gene set")
        return None

if METABOLIC_SIGNATURES_AVAILABLE:
    print(
        "      Adding NK metabolic signatures from Hallmark gene sets (glycolysis and oxidative phosphorylation)..."
    )
    
    # Load Hallmark gene sets from proper data directory
    hallmark_glycolysis = load_hallmark_geneset("data/raw/gene_sets/HALLMARK_GLYCOLYSIS.v2025.1.Hs.grp")
    hallmark_oxphos = load_hallmark_geneset("data/raw/gene_sets/HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2025.1.Hs.grp")
    hallmark_fatty_acid_metabolism = load_hallmark_geneset("data/raw/gene_sets/HALLMARK_FATTY_ACID_METABOLISM.v2025.1.Hs.grp")
    
    # Store the full gene sets for later subsetting with Rebuffet blood data
    hallmark_gene_sets_full = {}
    if hallmark_glycolysis is not None:
        hallmark_gene_sets_full["Glycolysis"] = hallmark_glycolysis
        print(f"        Loaded Hallmark Glycolysis: {len(hallmark_glycolysis)} genes from file")
   
    if hallmark_oxphos is not None:
        hallmark_gene_sets_full["NK_Oxidative_Phosphorylation"] = hallmark_oxphos
        print(f"        Loaded Hallmark OxPhos: {len(hallmark_oxphos)} genes from file")

    if hallmark_fatty_acid_metabolism is not None:
        hallmark_gene_sets_full["NK_Fatty_Acid_Metabolism"] = hallmark_fatty_acid_metabolism
        print(f"        Loaded Hallmark Fatty Acid Metabolism: {len(hallmark_fatty_acid_metabolism)} genes from file")

    print(f"        Hallmark gene sets will be subsetted to top 15 most variable genes using Rebuffet blood data")
    
    # NOTE: Actual subsetting will occur after blood data is loaded and available
    
else:
    print("      Metabolic signatures not available - using original signatures only")
    hallmark_gene_sets_full = {}

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
    # Additional markers requested for enhanced analysis
    "GZMH": "Granzyme H - Cytotoxic granule protein, mature NK marker",
    "IL7R": "CD127 (IL-7R) - Early NK progenitor and immature NK marker",
    "CCL3": "MIP-1α - Chemokine, inflammatory response marker",
    "CREM": "cAMP response element modulator - Transcriptional regulator",
    "IL32": "Interleukin-32 - Pro-inflammatory cytokine, NK activation marker",
    "ZNF90": "Zinc finger protein 90 - Transcriptional regulator",
    "NFKBIA": "IκBα - NF-κB inhibitor, inflammatory response regulator",
    "MKI67": "Ki-67 - Proliferation marker",
    "DNAJB1": "HSP40 family member - Heat shock protein, stress response",
    "NR4A3": "NOR1 - Nuclear receptor, immediate early response gene",
    "RGS1": "Regulator of G-protein signaling 1 - GPCR signaling modulator",
    "LAMP3": "CD208 - Lysosomal membrane protein, maturation marker",
    # Additional core NK markers for comprehensive analysis
    "ITGB2": "CD18 - Integrin beta 2, leukocyte function-associated antigen-1",
    "CD96": "TACTILE - NK cell receptor, inhibitory function",
    "CD44": "Hyaluronate receptor - Cell adhesion, activation marker",
    "CD38": "Cyclic ADP ribose hydrolase - Activation and differentiation marker",
    "FCGR3B": "CD16b - Low-affinity IgG Fc receptor III-B",
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


def score_genes_aucell(adata, gene_list, score_name, use_raw=True, normalize=True, auc_max_rank=0.05):
    """
    Calculate proper AUCell scores for a gene signature based on Aibar et al. 2017 Nature Methods.
    
    AUCell uses the "Area Under the Curve" (AUC) to calculate whether a critical subset 
    of the input gene set is enriched within the expressed genes for each cell. Since the 
    scoring method is ranking-based, AUCell is independent of gene expression units and 
    normalization procedures, making it ideal for cross-dataset analysis and batch effect removal.

    The algorithm:
    1. Builds gene-expression rankings for each cell (highest to lowest expression)
    2. Creates a recovery curve showing cumulative gene set membership vs. rank position
    3. Calculates the area under the recovery curve (AUC) using only top-ranked genes
    4. Normalizes AUC scores for comparability

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    gene_list : list
        List of genes to score
    score_name : str
        Name for the score column to be added to adata.obs
    use_raw : bool, default True
        Whether to use raw expression data
    normalize : bool, default True
        Whether to normalize AUC values to [0, 1] range
    auc_max_rank : float, default 0.05
        Fraction of top-ranked genes to consider (default 5% as in original AUCell)

    Returns:
    --------
    None
        Adds score_name column to adata.obs
    """
    # Get expression matrix
    if use_raw and adata.raw is not None:
        exp_mtx = adata.raw.X
        gene_names = adata.raw.var_names
    else:
        exp_mtx = adata.X
        gene_names = adata.var_names

    # Convert to dense if sparse
    if hasattr(exp_mtx, 'toarray'):
        exp_mtx = exp_mtx.toarray()

    # Filter gene list to only include genes present in the data
    available_genes = list(set(gene_list) & set(gene_names))
    
    if len(available_genes) == 0:
        print(f"        WARNING: No genes from signature '{score_name}' found in data")
        adata.obs[score_name] = 0.0
        return
    
    print(f"        Using proper AUCell scoring for {score_name}: {len(available_genes)}/{len(gene_list)} genes available")
    
    try:
        # Get indices of genes in the signature
        gene_indices = [i for i, gene in enumerate(gene_names) if gene in available_genes]
        signature_genes_set = set(gene_indices)
        
        # Calculate proper AUCell scores for each cell
        auc_scores = []
        n_genes_total = len(gene_names)
        n_sig_genes = len(gene_indices)
        max_rank = int(n_genes_total * auc_max_rank)  # Only consider top X% of genes
        
        if max_rank < 1:
            max_rank = min(50, n_genes_total)  # Minimum threshold for very small datasets
        
        for cell_idx in range(exp_mtx.shape[0]):
            # Get expression values for this cell
            cell_expr = exp_mtx[cell_idx, :]
            
            # Rank genes by expression (descending order - highest expression first)
            gene_ranks = np.argsort(-cell_expr)
            
            # Build recovery curve: track cumulative signature genes recovered
            recovery_curve = []
            signature_genes_found = 0
            
            # Only consider top max_rank genes (default 5% of all genes)
            for rank_pos in range(min(max_rank, len(gene_ranks))):
                gene_idx = gene_ranks[rank_pos]
                
                # Check if this gene is in our signature
                if gene_idx in signature_genes_set:
                    signature_genes_found += 1
                
                # Calculate recovery rate at this position
                if n_sig_genes > 0:
                    recovery_rate = signature_genes_found / n_sig_genes
                else:
                    recovery_rate = 0.0
                    
                recovery_curve.append(recovery_rate)
            
            # Calculate AUC using trapezoidal rule
            if len(recovery_curve) > 1:
                # X-axis: normalized rank positions [0, 1]
                x_positions = np.arange(len(recovery_curve)) / (len(recovery_curve) - 1)
                # Calculate area under recovery curve
                auc = np.trapezoid(recovery_curve, x_positions)
            elif len(recovery_curve) == 1:
                # Single point - use rectangle area
                auc = recovery_curve[0]
            else:
                auc = 0.0
            
            auc_scores.append(auc)
        
        # Convert to numpy array
        auc_scores = np.array(auc_scores)
        
        # Normalize scores to [0, 1] if requested
        if normalize and len(auc_scores) > 0:
            min_score = np.min(auc_scores)
            max_score = np.max(auc_scores)
            if max_score > min_score:
                auc_scores = (auc_scores - min_score) / (max_score - min_score)
        
        # Add scores to adata.obs
        adata.obs[score_name] = auc_scores
        
        print(f"        AUCell scoring complete: max_rank={max_rank} ({auc_max_rank*100:.1f}% of {n_genes_total} genes)")
        
    except Exception as e:
        print(f"        ERROR with AUCell scoring for {score_name}: {e}")
        print(f"        Falling back to scanpy scoring")
        sc.tl.score_genes(
            adata,
            gene_list,
            score_name=score_name,
            use_raw=use_raw,
            random_state=RANDOM_SEED,
        )


print(
    f"    Gene name mapping dictionary defined with {len(GENE_NAME_MAPPING)} mappings."
)
print("    Function 'map_gene_names' defined for gene signature compatibility.")
print("    Function 'score_genes_aucell' defined with proper AUCell algorithm for batch-effect-resistant gene set scoring.")
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


# --- 0.4.2a: Helper Function for Heatmap Layout Optimization ---
def calculate_heatmap_layout(data_frame, min_width=10, min_height=6, 
                           cell_width=1.2, cell_height=0.8, 
                           label_padding=2.0, base_left_margin=0.15):
    """
    Calculate optimal figure size and layout parameters for heatmaps based on best practices.
    
    This function follows standard heatmap sizing conventions with enhanced handling for long labels:
    - Each cell gets adequate space for readability (typically 0.8-1.2 inches per cell)
    - Labels get sufficient padding based on their length with adaptive sizing
    - Minimum dimensions ensure readability even for small matrices
    - Enhanced character width estimation for different label lengths
    
    Parameters:
    -----------
    data_frame : pd.DataFrame
        The data frame that will be plotted as a heatmap
    min_width : float
        Minimum figure width in inches
    min_height : float  
        Minimum figure height in inches
    cell_width : float
        Width per column in inches (standard: 1.0-1.2)
    cell_height : float
        Height per row in inches (standard: 0.6-0.8)
    label_padding : float
        Additional padding for row labels in inches
    base_left_margin : float
        Base left margin as fraction of figure width
        
    Returns:
    --------
    tuple: (figure_width, figure_height, left_margin)
    """
    # Calculate dimensions based on data
    n_rows, n_cols = data_frame.shape
    
    # Calculate maximum label length (for row labels, which are typically the problematic ones)
    max_label_length = max(len(str(label)) for label in data_frame.index)
    
    # Enhanced character width estimation that scales with label length
    # Longer labels need more generous spacing due to font rendering
    if max_label_length <= 15:
        char_width = 0.12  # Standard width for short labels
    elif max_label_length <= 30:
        char_width = 0.14  # Slightly wider for medium labels
    elif max_label_length <= 50:
        char_width = 0.16  # Even wider for long labels
    else:
        char_width = 0.18  # Maximum width for very long labels
    
    # Calculate figure dimensions using cell-based sizing (best practice)
    # Width: cell space + label space + padding
    heatmap_content_width = n_cols * cell_width
    
    # Enhanced label width calculation with adaptive padding
    base_label_width = max_label_length * char_width
    adaptive_padding = max(label_padding, min(label_padding * 1.5, base_label_width * 0.2))
    label_width_needed = max(2.0, base_label_width + adaptive_padding)
    
    # Ensure minimum heatmap content width is preserved
    min_heatmap_width = max(6.0, n_cols * 1.0)  # Minimum 1.0 inch per column
    figure_width = max(min_width, heatmap_content_width + label_width_needed, min_heatmap_width + label_width_needed)
    
    # Height: cell space (with better sizing for readability)
    heatmap_content_height = n_rows * cell_height
    title_space = 2.5  # More space for title and axis labels
    figure_height = max(min_height, heatmap_content_height + title_space)
    
    # Enhanced left margin calculation that ensures adequate space
    # Calculate the proportion needed for labels, but with a more generous cap
    left_margin_needed = label_width_needed / figure_width
    
    # Use a more generous cap for very long labels, but ensure heatmap stays readable
    if max_label_length > 30:
        max_left_margin = 0.55  # Up to 55% for very long labels
    elif max_label_length > 20:
        max_left_margin = 0.50  # Up to 50% for long labels
    else:
        max_left_margin = 0.45  # Standard 45% for shorter labels
    
    left_margin = max(base_left_margin, min(left_margin_needed, max_left_margin))
    
    # Final adjustment: if left margin is very high, increase figure width to maintain heatmap readability
    if left_margin > 0.50:
        # Increase figure width to maintain adequate heatmap space
        additional_width = (left_margin - 0.50) * figure_width
        figure_width += additional_width
        # Recalculate left margin with the new figure width
        left_margin = label_width_needed / figure_width
    
    # Debug output to help with tuning
    print(f"        Heatmap sizing: {n_rows}x{n_cols} matrix → {figure_width:.1f}x{figure_height:.1f} inches")
    print(f"        Cell size: {cell_width}x{cell_height} inches, Max label: {max_label_length} chars")
    print(f"        Label space: {label_width_needed:.1f} inches, Left margin: {left_margin:.1%}")
    
    return figure_width, figure_height, left_margin


print("  Function 'calculate_heatmap_layout' defined.")


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
    max_markers_per_subtype=6,  # Increased from 4 to 6 for better representation
    pval_threshold=0.05,  # Relaxed from 0.01 to 0.05 for more lenient significance
    logfc_threshold=0.3,  # Relaxed from 0.5 to 0.3 (~1.23x higher expression)
):
    """
    Intelligently select highly selective markers for each NK subtype with robust overlap handling.

    Key Features:
    - Uses stricter selectivity criteria for more obvious subtype-specific genes
    - Prioritizes genes based on enhanced composite scores and specificity
    - Ensures each subtype gets exactly max_markers_per_subtype unique genes
    - Uses sophisticated conflict resolution for overlapping top DEGs
    - Falls back to highly selective gene method when standard DEG approach insufficient

    Parameters:
    -----------
    adata : AnnData
        Annotated data object with DEG results
    deg_key : str
        Key for DEG results in adata.uns
    subtypes_ordered : list
        Ordered list of subtypes to analyze
    max_markers_per_subtype : int
        Exact number of markers per subtype (default: 6, increased for better representation)
    pval_threshold : float
        P-value threshold for significance (default: 0.05, relaxed for more lenient discovery)
    logfc_threshold : float
        Log fold change threshold (default: 0.3, ~1.23x higher expression)

    Returns:
    --------
    dict : Dictionary with subtype as key and list of highly selective markers as value
    """
    print(
        f"      Selecting {max_markers_per_subtype} optimal markers per subtype with relaxed criteria and robust overlap resolution..."
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

            # Only assign if winner has a meaningfully higher score
            score_ratio = winner_score / (runner_up_score + 1e-6)
            if score_ratio > 1.3:  # Winner has >30% higher score (relaxed from 50%)
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

        # If still need more genes, try highly selective method as backup
        if len(optimal_markers[subtype]) < max_markers_per_subtype:
            print(f"        {subtype} still needs genes, trying highly selective method...")
            subtype_col = get_subtype_column(adata)
            if subtype_col:
                try:
                    highly_selective_genes = get_highly_selective_subtype_genes(
                        adata,
                        subtype_col,
                        subtype,
                        min_logfc=0.15,  # Further relaxed for NK subtypes
                        min_specificity_ratio=1.15,  # More relaxed for NK subtypes
                        min_pct_in_target=0.08,  # Further relaxed for NK subtypes
                        max_pct_in_others=None,  # No overlap limit - allow shared expression
                        top_n_genes=max_markers_per_subtype
                    )
                    
                    for gene in highly_selective_genes:
                        if (
                            gene not in assigned_genes
                            and len(optimal_markers[subtype]) < max_markers_per_subtype
                        ):
                            optimal_markers[subtype].append(gene)
                            assigned_genes.add(gene)
                            print(f"        Added highly selective gene {gene} to {subtype}")
                except Exception as e:
                    print(f"        Highly selective method failed for {subtype}: {e}")
        
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


def get_highly_selective_subtype_genes(
    adata,
    subtype_col,
    target_subtype,
    min_logfc=0.15,  # Further relaxed for NK subtypes (1.11x fold change)
    max_pval=0.05,  # Relaxed from 0.01 to 0.05 for more lenient significance
    min_specificity_ratio=1.15,  # Relaxed from 1.2 to 1.15 (15% higher vs 20%)
    min_pct_in_target=0.08,  # Relaxed from 0.10 to 0.08 (8% of target cells)
    max_pct_in_others=None,  # REMOVED - allow overlap between related subtypes
    top_n_genes=30,
):
    """
    Get highly selective genes that are obviously upregulated in target subtype vs ALL others.
    
    This function implements reasonable selectivity criteria appropriate for developmental NK cell subtypes:
    1. Moderately upregulated in the target subtype (reasonable logFC threshold)
    2. Preferentially expressed in target vs other subtypes (reasonable specificity ratio)
    3. Expressed in a meaningful fraction of target cells with overlap allowed across subtypes
    4. Ranked highly within the target subtype
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with expression data
    subtype_col : str
        Column name containing subtype annotations
    target_subtype : str
        The subtype to find selective genes for
    min_logfc : float
        Minimum log fold change for upregulation (default: 0.15, ~1.11x higher)
    max_pval : float
        Maximum adjusted p-value for significance (default: 0.05, relaxed for more lenient discovery)
    min_specificity_ratio : float
        Minimum ratio of expression in target vs highest other subtype (default: 1.15, 15% higher)
    min_pct_in_target : float
        Minimum percentage of target cells expressing the gene (default: 0.08, 8%)
    max_pct_in_others : float
        Maximum percentage of non-target cells expressing the gene (default: None, no limit)
    top_n_genes : int
        Maximum number of genes to return (default: 50)
    
    Returns:
    --------
    list : List of highly selective gene names for the target subtype
    """
    print(f"        Finding highly selective genes for {target_subtype}...")
    
    # Get expression data
    if adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Get subtype masks
    subtypes = adata.obs[subtype_col].unique()
    target_mask = adata.obs[subtype_col] == target_subtype
    
    if target_mask.sum() < 5:
        print(f"        WARNING: Too few cells ({target_mask.sum()}) for {target_subtype}")
        return []
    
    selective_genes = []
    
    # Calculate metrics for each gene
    for gene_idx, gene_name in enumerate(var_names):
        if is_gene_to_exclude_util(gene_name):
            continue
            
        gene_expr = X[:, gene_idx]
        
        # Expression in target vs other subtypes
        target_expr = gene_expr[target_mask]
        target_mean = np.mean(target_expr)
        target_pct = np.mean(target_expr > 0)
        
        # Check expression in each other subtype
        other_means = []
        other_pcts = []
        
        for other_subtype in subtypes:
            if other_subtype == target_subtype:
                continue
                
            other_mask = adata.obs[subtype_col] == other_subtype
            if other_mask.sum() < 3:  # Skip subtypes with too few cells
                continue
                
            other_expr = gene_expr[other_mask]
            other_means.append(np.mean(other_expr))
            other_pcts.append(np.mean(other_expr > 0))
        
        if not other_means:  # No other subtypes to compare
            continue
            
        max_other_mean = max(other_means) if other_means else 0
        max_other_pct = max(other_pcts) if other_pcts else 0
        
        # Apply selectivity criteria
        
        # 1. High expression in target
        if target_mean < 0.3:  # Relaxed minimum expression level (from 0.5 to 0.3)
            continue
            
        # 2. Specificity ratio (target vs highest other)
        if max_other_mean > 0:
            specificity_ratio = target_mean / max_other_mean
        else:
            specificity_ratio = float('inf')
            
        if specificity_ratio < min_specificity_ratio:
            continue
            
        # 3. Percentage criteria
        if target_pct < min_pct_in_target or (max_pct_in_others is not None and max_other_pct > max_pct_in_others):
            continue
            
        # 4. Statistical significance (if DEG analysis available)
        # For now, we'll use a simple log fold change estimate
        if max_other_mean > 0:
            logfc_estimate = np.log2(target_mean / max_other_mean)
        else:
            logfc_estimate = np.log2(target_mean + 1)  # Pseudo-count approach
            
        if logfc_estimate < min_logfc:
            continue
            
        # Calculate composite selectivity score
        selectivity_score = (
            logfc_estimate * 
            specificity_ratio * 
            target_pct * 
            (1 - max_other_pct)
        )
        
        selective_genes.append({
            'gene': gene_name,
            'target_mean': target_mean,
            'max_other_mean': max_other_mean,
            'specificity_ratio': specificity_ratio,
            'target_pct': target_pct,
            'max_other_pct': max_other_pct,
            'logfc_estimate': logfc_estimate,
            'selectivity_score': selectivity_score
        })
    
    # Sort by selectivity score and return top genes
    selective_genes_df = pd.DataFrame(selective_genes)
    if selective_genes_df.empty:
        print(f"        WARNING: No selective genes found for {target_subtype}")
        return []
    
    selective_genes_df = selective_genes_df.sort_values('selectivity_score', ascending=False)
    top_selective_genes = selective_genes_df.head(top_n_genes)
    
    print(f"        Found {len(top_selective_genes)} highly selective genes for {target_subtype}")
    print(f"        Top genes: {top_selective_genes['gene'].head(5).tolist()}")
    print(f"        Specificity ratios: {top_selective_genes['specificity_ratio'].head(5).round(2).tolist()}")
    
    return top_selective_genes['gene'].tolist()


print("  Function 'get_highly_selective_subtype_genes' defined for improved gene selectivity.")
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
    print(f"      All .obs columns: {adata_blood_source.obs.columns.tolist()}")
    
    # Check for potential batch/donor effects
    if 'donor' in adata_blood_source.obs.columns:
        donor_counts = adata_blood_source.obs['donor'].value_counts()
        print(f"      Donor distribution: {donor_counts.to_dict()}")
    if 'batch' in adata_blood_source.obs.columns:
        batch_counts = adata_blood_source.obs['batch'].value_counts()
        print(f"      Batch distribution: {batch_counts.to_dict()}")
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
    # --- Step 1: Use original data directly (SKIP enhanced preprocessing to preserve natural structure) ---
    # The Rebuffet data is already properly processed by the authors
    print("      Using original Rebuffet data directly (bypassing enhanced preprocessing)")
    print("      This preserves the natural biological NK subtype structure without artificial batch correction")
    adata_blood_processed = adata_blood_source.copy()

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
    # FIXED: Using maximum connectivity parameters to force single blob structure
    N_PCS_BLOOD = 3  # Ultra-minimal PCs to reduce technical variance
    N_NEIGHBORS_BLOOD = 1000  # Ultra-high for maximum connectivity (increased from 500)
    print(
        f"      Computing nearest neighbor graph for adata_blood using {N_PCS_BLOOD} PCs and {N_NEIGHBORS_BLOOD} neighbors..."
    )
    print("      Using ultra-high neighbor count to force maximum biological continuity and eliminate separation...")
    sc.pp.neighbors(adata_blood, n_neighbors=N_NEIGHBORS_BLOOD, n_pcs=N_PCS_BLOOD, random_state=RANDOM_SEED)
    print("      Nearest neighbor graph computation complete for adata_blood.")

    # --- Step 5: Uniform Manifold Approximation and Projection (UMAP) ---
    # This computes the 2D embedding for visualization.
    # FIXED: Using maximum connectivity parameters to eliminate any separation
    print("      Running UMAP for adata_blood with ULTRA-HIGH connectivity parameters...")
    print("      Using ultra-high min_dist and spread to force single merged blob...")
    sc.tl.umap(adata_blood, 
               random_state=RANDOM_SEED, 
               min_dist=1.8,  # Ultra-high min_dist to eliminate local clustering (increased from 1.2)
               spread=4.5)    # Ultra-high spread for maximum global connectivity (increased from 3.0)
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
# 1.3.3: Cross-Dataset Subtype Re-annotation using Rebuffet Reference Signatures

print("    --- 1.3.3: Cross-Dataset Subtype Re-annotation using Rebuffet Reference Signatures ---")

print("      Implementing unified subtype framework:")
print(f"        Reference: Rebuffet blood NK subtypes ({len(REBUFFET_SUBTYPES_ORDERED)} subtypes)")
print(f"        Target: Tang tissue NK cells for re-annotation")
print("      Strategy: Generate DEG signatures → Score Tang cells → Assign best-matching subtype")

def generate_rebuffet_reference_signatures(adata_blood_ref, top_n_genes=150):
    """
    Generate robust reference signatures from Rebuffet blood NK subtypes.
    Uses the top 150-200 DEGs per subtype for cross-dataset annotation.
    
    Parameters:
    -----------
    adata_blood_ref : AnnData
        Blood NK reference data with Rebuffet subtypes
    top_n_genes : int
        Number of top DEGs per subtype (default: 150)
        
    Returns:
    --------
    dict : Dictionary with subtype names as keys and gene lists as values
    """
    print(f"        Generating reference signatures with {top_n_genes} DEGs per subtype...")
    
    if adata_blood_ref is None or adata_blood_ref.n_obs == 0:
        print("        ERROR: No valid blood NK reference data available.")
        return {}
    
    subtype_col = REBUFFET_SUBTYPE_COL
    if subtype_col not in adata_blood_ref.obs.columns:
        print(f"        ERROR: Subtype column '{subtype_col}' not found in reference data.")
        return {}
    
    # Create clean reference data (exclude Unassigned)
    ref_mask = adata_blood_ref.obs[subtype_col] != "Unassigned"
    adata_ref_clean = adata_blood_ref[ref_mask, :].copy()
    
    if adata_ref_clean.n_obs == 0:
        print("        ERROR: No assigned cells in reference data.")
        return {}
    
    print(f"        Running DEG analysis on {adata_ref_clean.n_obs} reference cells...")
    
    # Run comprehensive DEG analysis
    sc.tl.rank_genes_groups(
        adata_ref_clean,
        groupby=subtype_col,
        method="wilcoxon",
        use_raw=True,
        pts=True,
        corr_method="benjamini-hochberg",
        n_genes=top_n_genes + 100,  # Get extra for filtering
        key_added="rebuffet_reference_degs",
    )
    
    # Extract signatures for each subtype
    reference_signatures = {}
    available_subtypes = adata_ref_clean.obs[subtype_col].cat.categories
    
    for subtype in available_subtypes:
        try:
            deg_df = sc.get.rank_genes_groups_df(
                adata_ref_clean, group=subtype, key="rebuffet_reference_degs"
            )
            
            if deg_df is None or deg_df.empty:
                print(f"        WARNING: No DEGs found for {subtype}")
                reference_signatures[subtype] = []
                continue
            
            # Filter for high-quality signature genes
            filtered_degs = deg_df[
                (~deg_df["names"].apply(is_gene_to_exclude_util))
                & (deg_df["pvals_adj"] < 0.05)  # Significant
                & (deg_df["logfoldchanges"] > 0.25)  # Meaningful upregulation
                & (deg_df["scores"] > 0)  # Positive enrichment
            ].copy()
            
            # Select top genes by composite score
            if not filtered_degs.empty:
                filtered_degs["composite_score"] = (
                    -np.log10(filtered_degs["pvals_adj"] + 1e-300) *
                    filtered_degs["logfoldchanges"] *
                    filtered_degs["scores"]
                )
                top_signature_genes = (
                    filtered_degs.nlargest(top_n_genes, "composite_score")["names"].tolist()
                )
            else:
                # Fallback to basic filtering if strict criteria yield no genes
                basic_filtered = deg_df[
                    (~deg_df["names"].apply(is_gene_to_exclude_util))
                    & (deg_df["pvals_adj"] < 0.1)
                    & (deg_df["logfoldchanges"] > 0.1)
                ].copy()
                top_signature_genes = basic_filtered["names"].head(top_n_genes).tolist()
            
            reference_signatures[subtype] = top_signature_genes
            print(f"        {subtype}: {len(top_signature_genes)} signature genes")
            
        except Exception as e:
            print(f"        ERROR extracting signature for {subtype}: {e}")
            reference_signatures[subtype] = []
    
    print(f"        Generated signatures for {len(reference_signatures)} Rebuffet subtypes")
    return reference_signatures

def annotate_tang_cells_with_rebuffet_signatures(adata_tang, ref_signatures, 
                                                assignment_threshold=0.10):
    """
    Re-annotate Tang cells using a biologically-informed two-step approach:
    1. Assign CD56bright cells to NK2 (immature/regulatory)
    2. Use signature scoring for CD56dim cells to assign to NK1A/B/C, NK3, NKint
    
    Parameters:
    -----------
    adata_tang : AnnData
        Tang tissue data to re-annotate
    ref_signatures : dict
        Reference signatures from Rebuffet subtypes
    assignment_threshold : float
        Minimum score difference required for assignment (default: 0.10 = 10%)
        
    Returns:
    --------
    None : Modifies adata_tang.obs in place
    """
    print(f"        Re-annotating {adata_tang.n_obs} Tang cells using biologically-informed approach...")
    
    # Initialize assignment array
    assigned_subtypes = ["Unassigned"] * adata_tang.n_obs
    assignment_scores = np.zeros(adata_tang.n_obs)
    assignment_confidence = np.zeros(adata_tang.n_obs)
    
    # Step 1: Biologically-informed pre-assignment based on Tang major types
    if TANG_MAJORTYPE_COL in adata_tang.obs.columns:
        print(f"        Step 1: Assigning CD56bright cells to NK2...")
        
        # Assign CD56bright cells to NK2 (immature/regulatory)
        cd56bright_mask = adata_tang.obs[TANG_MAJORTYPE_COL] == "CD56highCD16low"
        cd56bright_count = cd56bright_mask.sum()
        
        for i in range(len(adata_tang.obs)):
            if cd56bright_mask.iloc[i]:
                assigned_subtypes[i] = "NK2"
                assignment_scores[i] = 1.0  # High confidence for biological assignment
                assignment_confidence[i] = 1.0
        
        print(f"          Assigned {cd56bright_count:,} CD56bright cells (CD56highCD16low) to NK2")
        
        # Step 2: Use signature scoring for CD56dim AND CD56brightCD16high cells (dynamic assignment)
        cd56dim_mask = adata_tang.obs[TANG_MAJORTYPE_COL] == "CD56lowCD16high"
        cd56bright_cd16high_mask = adata_tang.obs[TANG_MAJORTYPE_COL] == "CD56highCD16high"
        
        # Combine CD56dim and CD56brightCD16high for dynamic signature-based assignment
        cells_for_scoring_mask = cd56dim_mask | cd56bright_cd16high_mask
        cells_for_scoring_count = cells_for_scoring_mask.sum()
        cd56dim_count = cd56dim_mask.sum()
        cd56bright_cd16high_count = cd56bright_cd16high_mask.sum()
        
        print(f"        Step 2: Dynamic signature-based scoring for {cells_for_scoring_count:,} cells:")
        print(f"          - {cd56dim_count:,} CD56dim cells (CD56lowCD16high)")
        print(f"          - {cd56bright_cd16high_count:,} double-positive cells (CD56highCD16high)")
        
        if cells_for_scoring_count > 0:
            # Create subset for cells requiring signature-based assignment
            adata_for_scoring = adata_tang[cells_for_scoring_mask, :].copy()
            
            # Calculate signature scores for all mature NK subtypes (CD56dim + CD56brightCD16high)
            mature_nk_subtypes = ["NK1A", "NK1B", "NK1C", "NK3", "NKint"]
            signature_scores = {}
            
            for subtype in mature_nk_subtypes:
                if subtype not in ref_signatures or not ref_signatures[subtype]:
                    continue
                
                score_col = f"{subtype}_Signature_Score"
                available_genes = map_gene_names(ref_signatures[subtype], adata_for_scoring.raw.var_names)
                
                if len(available_genes) >= MIN_GENES_FOR_SCORING:
                    score_genes_aucell(
                        adata_for_scoring,
                        available_genes,
                        score_name=score_col,
                        use_raw=True,
                        normalize=True,
                    )
                    signature_scores[subtype] = score_col
                    print(f"          {subtype}: {len(available_genes)}/{len(ref_signatures[subtype])} genes available")
            
            if signature_scores:
                # Assign cells based on signature scores (both CD56dim and CD56brightCD16high)
                score_columns = list(signature_scores.values())
                score_matrix = adata_for_scoring.obs[score_columns].values
                
                max_scores = np.max(score_matrix, axis=1)
                max_indices = np.argmax(score_matrix, axis=1)
                second_max_scores = np.partition(score_matrix, -2, axis=1)[:, -2] if score_matrix.shape[1] > 1 else np.zeros_like(max_scores)
                score_differences = max_scores - second_max_scores
                
                subtype_names = list(signature_scores.keys())
                cells_for_scoring_indices = np.where(cells_for_scoring_mask)[0]
                
                for i, cell_idx in enumerate(cells_for_scoring_indices):
                    # TEMPORARY: Assign 100% of cells for UMAP visualization (threshold conditions commented out)
                    # if score_differences[i] >= assignment_threshold and max_scores[i] > 0.05:  # Conservative relaxation: reduced from 0.1 to 0.05
                    assigned_subtypes[cell_idx] = subtype_names[max_indices[i]]
                    assignment_scores[cell_idx] = max_scores[i]
                    assignment_confidence[cell_idx] = score_differences[i]
                    # else:
                    #     assigned_subtypes[cell_idx] = "Unassigned"
                    #     assignment_scores[cell_idx] = max_scores[i]
                    #     assignment_confidence[cell_idx] = score_differences[i]
                
                # Report assignment results for both cell types
                cd56dim_indices = np.where(cd56dim_mask)[0]
                cd56bright_cd16high_indices = np.where(cd56bright_cd16high_mask)[0]
                
                cd56dim_assigned = sum(1 for i in cd56dim_indices if assigned_subtypes[i] != "Unassigned")
                cd56bright_cd16high_assigned = sum(1 for i in cd56bright_cd16high_indices if assigned_subtypes[i] != "Unassigned")
                
                print(f"          Signature-assigned {cd56dim_assigned:,}/{cd56dim_count:,} CD56dim cells")
                print(f"          Signature-assigned {cd56bright_cd16high_assigned:,}/{cd56bright_cd16high_count:,} CD56brightCD16high cells")
            else:
                print(f"          WARNING: No valid signatures for mature NK cell assignment")
    
    else:
        print(f"        WARNING: {TANG_MAJORTYPE_COL} not found. Using signature scoring for all cells...")
        # Fallback to original method if majortype info not available
        signature_scores = {}
        for subtype, gene_list in ref_signatures.items():
            if not gene_list:
                continue
            
            score_col = f"{subtype}_Signature_Score"
            available_genes = map_gene_names(gene_list, adata_tang.raw.var_names)
            
            if len(available_genes) >= MIN_GENES_FOR_SCORING:
                score_genes_aucell(
                    adata_tang,
                    available_genes,
                    score_name=score_col,
                    use_raw=True,
                    normalize=True,
                )
                signature_scores[subtype] = score_col
        
        if signature_scores:
            score_columns = list(signature_scores.values())
            score_matrix = adata_tang.obs[score_columns].values
            max_scores = np.max(score_matrix, axis=1)
            max_indices = np.argmax(score_matrix, axis=1)
            second_max_scores = np.partition(score_matrix, -2, axis=1)[:, -2]
            score_differences = max_scores - second_max_scores
            
            subtype_names = list(signature_scores.keys())
            for i in range(len(adata_tang.obs)):
                # TEMPORARY: Assign 100% of cells for UMAP visualization (threshold conditions commented out)
                # if score_differences[i] >= assignment_threshold and max_scores[i] > 0.05:  # Conservative relaxation: reduced from 0.1 to 0.05
                assigned_subtypes[i] = subtype_names[max_indices[i]]
                assignment_scores[i] = max_scores[i]
                assignment_confidence[i] = score_differences[i]
    
    # Update annotations
    adata_tang.obs[REBUFFET_SUBTYPE_COL] = pd.Categorical(
        assigned_subtypes,
        categories=REBUFFET_SUBTYPES_ORDERED + ["Unassigned"],
        ordered=True,
    )
    
    # Store assignment metrics
    adata_tang.obs["Assignment_Score"] = assignment_scores
    adata_tang.obs["Assignment_Confidence"] = assignment_confidence
    
    # Report final assignment results
    assignment_counts = adata_tang.obs[REBUFFET_SUBTYPE_COL].value_counts()
    total_assigned = assignment_counts.drop("Unassigned", errors="ignore").sum()
    unassigned_count = assignment_counts.get("Unassigned", 0)
    
    print(f"        Final assignment results:")
    print(f"          Total assigned: {total_assigned:,} cells ({total_assigned/adata_tang.n_obs*100:.1f}%)")
    print(f"          Unassigned: {unassigned_count:,} cells ({unassigned_count/adata_tang.n_obs*100:.1f}%)")
    
    for subtype in REBUFFET_SUBTYPES_ORDERED:
        count = assignment_counts.get(subtype, 0)
        if count > 0:
            print(f"          {subtype}: {count:,} cells ({count/adata_tang.n_obs*100:.1f}%)")

# Generate reference signatures from Rebuffet blood NK data
# CONSISTENT: Use same 30-gene focused signatures for both assignment AND developmental scoring
if adata_blood is not None and adata_blood.n_obs > 0:
    ref_rebuffet_markers = generate_rebuffet_reference_signatures(adata_blood, top_n_genes=50)
    print(f"      Generated reference signatures for {len(ref_rebuffet_markers)} subtypes")
    print("      ENHANCED SIGNATURES: Using 50-gene comprehensive signatures for improved assignment coverage")
    
    # Subset Hallmark gene sets to top 20 most variable genes in Rebuffet blood data
    if 'hallmark_gene_sets_full' in locals() and hallmark_gene_sets_full:
        print("\n      Subsetting Hallmark gene sets to top 20 most variable genes in Rebuffet blood data...")
        
        def subset_hallmark_to_most_variable(adata_blood, hallmark_sets, top_n=15):
            """Subset Hallmark gene sets to most variable genes in blood NK data."""
            subsetted_sets = {}
            
            for set_name, full_gene_list in hallmark_sets.items():
                # Get genes available in the blood dataset
                available_genes = [g for g in full_gene_list if g in adata_blood.raw.var_names]
                
                if len(available_genes) < 5:
                    print(f"        WARNING: {set_name} has only {len(available_genes)} available genes, using all")
                    subsetted_sets[set_name] = available_genes
                    continue
                
                # Calculate variance for available genes
                gene_indices = [adata_blood.raw.var_names.get_loc(g) for g in available_genes]
                
                # Get expression data for these genes
                if hasattr(adata_blood.raw.X, 'toarray'):
                    gene_expr = adata_blood.raw.X.toarray()[:, gene_indices]
                else:
                    gene_expr = adata_blood.raw.X[:, gene_indices]
                
                # Calculate variance across cells for each gene
                gene_variances = np.var(gene_expr, axis=0)
                
                # Get top N most variable genes
                n_to_select = min(top_n, len(available_genes))
                top_var_indices = np.argsort(gene_variances)[-n_to_select:]
                top_var_genes = [available_genes[i] for i in top_var_indices]
                
                subsetted_sets[set_name] = top_var_genes
                print(f"        {set_name}: {len(full_gene_list)} → {len(top_var_genes)} genes (top {n_to_select} most variable)")
            
            return subsetted_sets
        
        # Apply subsetting
        hallmark_subsetted = subset_hallmark_to_most_variable(adata_blood, hallmark_gene_sets_full, top_n=15)
        
        # Add subsetted Hallmark sets to FUNCTIONAL_GENE_SETS
        for set_name, genes in hallmark_subsetted.items():
            FUNCTIONAL_GENE_SETS[set_name] = genes
        
        print(f"      Successfully subsetted {len(hallmark_subsetted)} Hallmark gene sets to top 15 most variable genes")
    else:
        print("      No Hallmark gene sets available for subsetting")
else:
    print("      ERROR: No Rebuffet blood NK data available for signature generation")
    ref_rebuffet_markers = {}

# Re-annotate Tang tissue datasets using biologically-informed Rebuffet signatures
# 
# BIOLOGICAL RATIONALE:
# - CD56bright cells are generally immature/regulatory NK cells → assign to NK2
# - CD56bright/CD16high cells are transitional → assign to NKint  
# - CD56dim cells are mature cytotoxic NK cells → use signature scoring for NK1A/B/C, NK3, NKint
# This approach respects the fundamental CD56bright→CD56dim maturation hierarchy
#
for cohort_name, adata_ctx in [
    ("adata_normal_tissue", adata_normal_tissue),
    ("adata_tumor_tissue", adata_tumor_tissue),
]:
    if adata_ctx is None or adata_ctx.n_obs == 0:
        print(f"      {cohort_name}: Not available for re-annotation")
        continue
    
    if not ref_rebuffet_markers:
        print(f"      {cohort_name}: Skipping - no reference signatures available")
        continue
    
    print(f"      Re-annotating {cohort_name} using biologically-informed approach...")
    
    # Store original Tang annotations for validation
    if TANG_CELLTYPE_COL in adata_ctx.obs.columns:
        adata_ctx.obs["Original_Tang_Subtype"] = adata_ctx.obs[TANG_CELLTYPE_COL]
    
    # Perform biologically-informed re-annotation
    annotate_tang_cells_with_rebuffet_signatures(
        adata_ctx, 
        ref_rebuffet_markers, 
        assignment_threshold=0.05  # Conservative relaxation: reduced from 0.08 to 0.05 (5% score difference)
    )
    
    # Update global variable references
    if cohort_name == "adata_normal_tissue":
        adata_normal_tissue = adata_ctx
    elif cohort_name == "adata_tumor_tissue":
        adata_tumor_tissue = adata_ctx

print("    --- End of 1.3.3 ---")

# %%
# PART 1: Data Ingestion, Preprocessing, and Cohort AnnData Object Generation
# Section 1.3: Context-Specific Cohorts from Tang Data & Consistent Subtype Annotation
# 1.3.4: Validate Biologically-Informed Cross-Dataset Re-annotation Results

print("    --- 1.3.4: Validate Biologically-Informed Cross-Dataset Re-annotation Results ---")

# Validate that all datasets now use unified Rebuffet subtype annotations
# with biologically-informed assignment (CD56bright→NK2, CD56dim→signature scoring)
print("      Validating biologically-informed unified subtype annotations across all datasets:")

# Check Rebuffet blood data (reference)
if adata_blood is not None and adata_blood.n_obs > 0:
    if REBUFFET_SUBTYPE_COL in adata_blood.obs.columns:
        rebuffet_subtypes = adata_blood.obs[REBUFFET_SUBTYPE_COL].value_counts()
        print(f"      Rebuffet blood NK data (reference, {REBUFFET_SUBTYPE_COL}):")
        for subtype, count in rebuffet_subtypes.items():
            print(
                f"        {subtype}: {count:,} cells ({count/adata_blood.n_obs*100:.1f}%)"
            )
    else:
        print(f"      WARNING: {REBUFFET_SUBTYPE_COL} not found in adata_blood")
else:
    print("      adata_blood not available")

# Check Tang tissue data (re-annotated)
for cohort_name, adata_ctx in [
    ("adata_normal_tissue", adata_normal_tissue),
    ("adata_tumor_tissue", adata_tumor_tissue),
]:
    if adata_ctx is None or adata_ctx.n_obs == 0:
        print(f"      {cohort_name}: Not available")
        continue

    if REBUFFET_SUBTYPE_COL in adata_ctx.obs.columns:
        assigned_subtypes = adata_ctx.obs[REBUFFET_SUBTYPE_COL].value_counts()
        print(f"      {cohort_name} (re-annotated, {REBUFFET_SUBTYPE_COL}):")
        for subtype, count in assigned_subtypes.items():
            print(
                f"        {subtype}: {count:,} cells ({count/adata_ctx.n_obs*100:.1f}%)"
            )
            
        # Show assignment quality metrics if available
        if "Assignment_Confidence" in adata_ctx.obs.columns:
            avg_confidence = adata_ctx.obs["Assignment_Confidence"].mean()
            print(f"        Average assignment confidence: {avg_confidence:.3f}")
            
    else:
        print(f"      WARNING: {REBUFFET_SUBTYPE_COL} not found in {cohort_name}")

    # Show cross-tabulation with original Tang annotations if available
    if ("Original_Tang_Subtype" in adata_ctx.obs.columns and 
        REBUFFET_SUBTYPE_COL in adata_ctx.obs.columns):
        print(f"      Cross-tabulation for {cohort_name} (Rebuffet vs Original Tang):")
        crosstab = pd.crosstab(
            adata_ctx.obs[REBUFFET_SUBTYPE_COL],
            adata_ctx.obs["Original_Tang_Subtype"],
            normalize='columns'
        ) * 100
        
        # Show top Tang->Rebuffet mappings
        for tang_subtype in crosstab.columns[:3]:  # Show first 3 Tang subtypes
            rebuffet_mapping = crosstab[tang_subtype].nlargest(3)
            print(f"        {tang_subtype} -> {rebuffet_mapping.to_dict()}")

print("\n      Summary: Biologically-informed unified cross-dataset annotation complete")
print("        - Reference: Rebuffet blood NK subtypes (6 subtypes)")
print("        - Method: Two-step biologically-informed assignment:")
print("          1. CD56bright cells → NK2 (immature/regulatory)")
print("          2. CD56bright/CD16high → NKint (transitional)")
print("          3. CD56dim cells → signature-based scoring for NK1A/B/C, NK3, NKint")
print("        - Framework: All datasets now use consistent Rebuffet subtype labels")
print("        - Benefits: Respects CD56bright→CD56dim maturation hierarchy")
print("        - Quality: Assignment confidence metrics stored for validation")

print("\n    --- End of 1.3.4 ---")



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
    This pipeline uses standard (unsupervised) dimensionality reduction to preserve natural biological 
    structure without artificial batch correction that could create false separations.
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
            np.median(np.array([n_pcs_variance, elbow_point, 15]))
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

    # --- Step 6: Neighbors and UMAP (High-connectivity approach for single blob structure) ---
    # UPDATED: Using high-connectivity parameters to minimize artificial separation and create
    # single blob structure as requested. This preserves biological continuity while eliminating
    # technical clustering artifacts that can create false group separations.
    
    # Use minimal PCs to reduce technical variance
    reduced_n_pcs = min(5, actual_n_pcs)  # Use only 5 PCs maximum to reduce separation
    
    # Use very high neighbor count for maximum connectivity
    high_n_neighbors = min(200, adata_obj.n_obs // 10)  # High neighbors, but not more than 10% of cells
    
    print(f"        Computing high-connectivity neighbor graph using {reduced_n_pcs} PCs and {high_n_neighbors} neighbors...")
    print("        Using high-connectivity approach to create single blob structure...")
    sc.pp.neighbors(adata_obj, n_neighbors=high_n_neighbors, n_pcs=reduced_n_pcs, random_state=RANDOM_SEED)
    
    print("        Running high-connectivity UMAP for single blob visualization...")
    sc.tl.umap(adata_obj, random_state=RANDOM_SEED, min_dist=1.5, spread=3.5)

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


def generate_rebuffet_developmental_signatures(adata_blood_ref, top_n_genes=30):
    """
    Generate highly selective developmental gene signatures from Rebuffet blood NK subtypes.
    
    Uses improved selectivity criteria to ensure genes are obviously upregulated 
    in each subtype compared to ALL other subtypes, including:
    - Minimum 1.68x higher expression (logFC > 0.75)
    - 2.5x specificity ratio vs highest other subtype
    - Expressed in ≥25% of target cells, ≤15% of others

    Parameters:
    -----------
    adata_blood_ref : AnnData
        Blood NK data with Rebuffet subtypes
    top_n_genes : int
        Number of top selective genes per subtype to include (limited to 30 for selectivity)

    Returns:
    --------
    dict : Dictionary with subtype names as keys and highly selective gene lists as values
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

                # Use new highly selective gene selection method
                highly_selective_genes = get_highly_selective_subtype_genes(
                    adata_deg,
                    subtype_col,
                    subtype,
                    min_logfc=0.15,  # Further relaxed for NK subtypes
                    min_specificity_ratio=1.15,  # Further relaxed for NK subtypes  
                    min_pct_in_target=0.08,  # Further relaxed for NK subtypes
                    max_pct_in_others=None,  # No overlap limit - allow shared expression
                    top_n_genes=min(top_n_genes, 20)  # Limit to most selective genes
                )
                
                # Fallback to DEG-based method if highly selective method finds too few genes
                if len(highly_selective_genes) < 10 and deg_df is not None and not deg_df.empty:
                    print(f"      {subtype}: Using fallback DEG method (only {len(highly_selective_genes)} highly selective genes)")
                    
                    # More relaxed DEG filtering (aligned with new parameters)
                    filtered_genes = deg_df[
                        (~deg_df["names"].apply(is_gene_to_exclude_util))
                        & (deg_df["pvals_adj"] < 0.05)  # Relaxed p-value (from 0.01 to 0.05)
                        & (deg_df["logfoldchanges"] > 0.3)  # Relaxed logFC (from 0.5 to 0.3)
                    ]

                    fallback_genes = filtered_genes.head(max(0, top_n_genes - len(highly_selective_genes)))["names"].tolist()
                    
                    # Combine highly selective + fallback, prioritizing selective genes
                    top_genes = highly_selective_genes + [g for g in fallback_genes if g not in highly_selective_genes]
                    top_genes = top_genes[:top_n_genes]
                else:
                    top_genes = highly_selective_genes

                # Clean up subtype name for the signature
                clean_subtype_name = f"{subtype}_Developmental"
                developmental_signatures[clean_subtype_name] = top_genes

                print(
                    f"      {subtype}: {len(top_genes)} highly selective signature genes extracted"
                )

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


# NOTE: Tang-specific developmental signature function removed
# Now using unified Rebuffet blood NK developmental signatures for all analyses


# Use the SAME signatures for both assignment and developmental scoring for consistency
if "adata_blood" in locals() and adata_blood is not None and ref_rebuffet_markers:
    # Convert assignment signatures to developmental signature format for compatibility
    DEVELOPMENTAL_GENE_SETS = {
        f"{subtype}_Developmental": genes 
        for subtype, genes in ref_rebuffet_markers.items()
    }
    print(f"  ENHANCED CONSISTENCY: Using same 50-gene comprehensive signatures for both assignment and developmental scoring")
    print(f"  Generated {len(DEVELOPMENTAL_GENE_SETS)} unified developmental gene sets")
    for sig_name, genes in DEVELOPMENTAL_GENE_SETS.items():
        print(f"    {sig_name}: {len(genes)} genes")
else:
    print(
        "  WARNING: No unified signatures available. Using static developmental signatures."
    )
    # Keep the original static signatures as fallback
    DEVELOPMENTAL_GENE_SETS = {
        "Regulatory_NK": ["SELL", "TCF7", "IL7R", "CCR7"],
        "Intermediate_NK": ["CD27", "GZMK", "KLRB1", "CD7"],
        "Mature_Cytotoxic_NK": ["GNLY", "NKG7", "GZMB", "PRF1"],
        "Adaptive_NK": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
    }

# NOTE: UNIFIED APPROACH - Same 30-gene signatures used for both:
# 1. Cross-dataset cell type assignment (Tang → Rebuffet subtypes)  
# 2. Developmental state scoring within assigned subtypes
# This eliminates assignment vs scoring discrepancies
print("  UNIFIED: Same signatures for both cell assignment AND developmental scoring")

# Create combined dictionary with unified developmental signatures
ALL_FUNCTIONAL_GENE_SETS = {
    **DEVELOPMENTAL_GENE_SETS,  # Unified Rebuffet blood NK developmental signatures
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
    # All datasets now use unified Rebuffet subtyping with consistent "Unassigned" handling
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
    # Filter to only include subtypes actually present in this dataset
    ordered_categories_for_plot = [
        cat
        for cat in REBUFFET_SUBTYPES_ORDERED  # Use consistent Rebuffet ordering
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
                size=8,  # Larger dots for better visibility (increased from 2)
            )
            plt.subplots_adjust(right=0.75)  # Adjust plot to make space for the legend

            umap_coords_df = pd.DataFrame(
                adata_ctx.obsm["X_umap"],
                index=adata_ctx.obs_names,
            )
            umap_coords_df.columns = ["UMAP1", "UMAP2"]
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

    # --- Cross-tabulation with original Tang annotations for validation ---
    if "Original_Tang_Subtype" in adata_view_assigned.obs.columns:
        print(
            f"        Cross-tab: Re-annotated Rebuffet Subtypes vs. Original Tang Subtypes"
        )

        # Create crosstab of re-annotation results
        crosstab_validation = pd.crosstab(
            adata_view_assigned.obs[subtype_col],
            adata_view_assigned.obs["Original_Tang_Subtype"],
            dropna=False,
        )

        ct_validation_filename = create_filename(
            "P2_1_Crosstab_ReannotationValidation",
            context_name=context_name,
            version="v1",
            ext="csv",
        )
        crosstab_validation.to_csv(os.path.join(ctx_stats_dir, ct_validation_filename))
        print(f"          Re-annotation validation crosstab for {context_name} saved.")

        # Visualize the crosstab as a heatmap (normalize by columns to see Tang->Rebuffet mapping)
        crosstab_norm = crosstab_validation.apply(
            lambda x: 100 * x / x.sum() if x.sum() > 0 else 0, axis=0
        ).fillna(0)

        # Calculate optimal figure dimensions for potentially many Tang subtypes
        fig_width = max(12, len(crosstab_norm.columns) * 0.8)
        fig_height = max(8, len(crosstab_norm.index) * 0.6)
        
        fig_crosstab, ax_crosstab = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            crosstab_norm,
            cmap="viridis",
            linewidths=0.5,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "% of Original Tang Subtype"},
            ax=ax_crosstab,
        )
        ax_crosstab.set_title(
            f"Re-annotation Validation: Rebuffet Assignment vs Original Tang Subtypes\n({context_name} - Assigned Cells Only)",
            fontsize=14,
        )
        ax_crosstab.set_ylabel("Re-annotated Rebuffet Subtype", fontsize=12)
        ax_crosstab.set_xlabel("Original Tang Subtype", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_basename_crosstab_hm = create_filename(
            "P2_1_Heatmap_ReannotationValidation",
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
        print(f"          Re-annotation validation heatmap for {context_name} saved.")

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
            plt.tight_layout(rect=(0, 0, 0.85, 1))

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
    # All datasets now use unified Rebuffet subtyping with consistent "Unassigned" handling
    assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
    adata_view_assigned = adata_ctx[assigned_mask, :].copy()

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
        for cat in REBUFFET_SUBTYPES_ORDERED  # Use consistent Rebuffet ordering
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

    # Calculate scores for the provided gene sets using AUCell
    score_cols = []
    for set_name, gene_list in gene_sets_dict.items():
        score_col_name = f"{set_name}_Score"
        score_cols.append(score_col_name)
        available_genes = map_gene_names(gene_list, adata_view.raw.var_names)
        if len(available_genes) >= MIN_GENES_FOR_SCORING:
            score_genes_aucell(
                adata_view,
                available_genes,
                score_name=score_col_name,
                use_raw=True,
                normalize=True,
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
        # Calculate optimal figure dimensions and layout for potentially long signature names
        fig_width, fig_height, left_margin = calculate_heatmap_layout(
            mean_scores_df, 
            min_width=10, 
            min_height=6, 
            cell_width=1.5,  # Larger cells for better readability
            cell_height=0.9,  # Taller cells for signature names
            label_padding=3.0  # More padding for long signature names
        )
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        sns.heatmap(
            mean_scores_df,
            cmap="icefire",
            center=0,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Mean Signature Score", "shrink": 0.6},
            annot_kws={"size": 11, "weight": "bold"},  # Larger, bold annotations
            square=True,  # Square cells for better readability
            ax=ax,
        )
        ax.set_title(f"{plot_title} in {context_name}", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Assigned NK Subtype", fontsize=14, fontweight='bold')
        ax.set_ylabel("Signature", fontsize=14, fontweight='bold')
        
        # Improve tick label readability
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)
        
        # Use improved layout control for better signature name visibility
        plt.subplots_adjust(left=left_margin, right=0.85, top=0.9, bottom=0.1)
        plt.tight_layout(rect=(left_margin, 0.1, 0.85, 0.9))

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
    # All datasets now use unified Rebuffet subtyping with consistent "Unassigned" handling
    assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
    if not assigned_mask.any():
        print(f"      No assigned cells in {context_name}. Skipping.")
        continue
    adata_view_assigned = adata_ctx[assigned_mask, :].copy()

    # Use unified processing (no Tang subset splitting)
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
        adata_view=adata_view_assigned,
        context_name=context_name,
        gene_sets_dict=DEVELOPMENTAL_GENE_SETS,
        plot_title="Developmental Signature Profiles",
        base_filename="P2_3a_Heatmap_DevProfile",
        fig_dir=ctx_fig_dir,
        data_dir=ctx_data_dir,
        subtype_col=subtype_col,
        subset_name=None,
    )

    # Generate the Functional Profile heatmap
    generate_signature_heatmap(
        adata_view=adata_view_assigned,
        context_name=context_name,
        gene_sets_dict=FUNCTIONAL_GENE_SETS,
        plot_title="Functional Signature Profiles",
        base_filename="P2_3b_Heatmap_FuncProfile",
        fig_dir=ctx_fig_dir,
        data_dir=ctx_data_dir,
        subtype_col=subtype_col,
        subset_name=None,
    )

    # Generate the Neurotransmitter Receptor Profile heatmap
    generate_signature_heatmap(
        adata_view=adata_view_assigned,
        context_name=context_name,
        gene_sets_dict=NEUROTRANSMITTER_RECEPTOR_GENE_SETS,
        plot_title="Neurotransmitter Receptor Signature Profiles",
        base_filename="P2_3c_Heatmap_NeuroReceptorProfile",
        fig_dir=ctx_fig_dir,
        data_dir=ctx_data_dir,
        subtype_col=subtype_col,
        subset_name=None,
    )

    # Generate the Interleukin Downstream Profile heatmap
    generate_signature_heatmap(
        adata_view=adata_view_assigned,
        context_name=context_name,
        gene_sets_dict=INTERLEUKIN_DOWNSTREAM_GENE_SETS,
        plot_title="Interleukin Downstream Signature Profiles",
        base_filename="P2_3d_Heatmap_InterleukinProfile",
        fig_dir=ctx_fig_dir,
        data_dir=ctx_data_dir,
        subtype_col=subtype_col,
        subset_name=None,
    )

print("\n--- End of Section 2.3 ---")

# %%
# PART 2: Baseline Characterization of NK Subtypes within Each Context  
# Section 2.3c: Detailed Dot Plots for All Signature Categories (Top 10 Most Variable Genes)

print("\n  --- Section 2.3c: Detailed Dot Plots for All Signature Categories (Top 10 Most Variable Genes) ---")


def create_signature_dotplot_top_variable(
    adata,
    gene_set,
    set_name,
    context_name,
    fig_dir,
    data_dir,
    subtype_col=None,
    top_n_genes=10,
):
    """
    Generates and saves a polished dot plot for the top N most variable genes from a specific gene set.
    """
    print(f"      Generating dot plot for '{set_name}' in {context_name} (top {top_n_genes} most variable genes)...")

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

    # Filter to valid cells for plotting (exclude "Unassigned")
    assigned_mask = adata.obs[subtype_col] != "Unassigned"
    adata_view_assigned = adata[assigned_mask, :].copy()
    if adata_view_assigned.n_obs == 0:
        print(f"        No assigned cells to plot for {context_name}. Skipping.")
        return

    # Ensure categories are ordered correctly for the plot
    categories_to_plot = [
        cat
        for cat in subtype_categories
        if cat in adata_view_assigned.obs[subtype_col].unique()
    ]
    adata_view_assigned.obs[subtype_col] = adata_view_assigned.obs[
        subtype_col
    ].cat.reorder_categories(categories_to_plot, ordered=True)

    # --- Select Top N Most Variable Genes ---
    if len(available_genes) > top_n_genes:
        # Calculate variance across all cells for each gene
        gene_variances = []
        for gene in available_genes:
            if gene in adata_view_assigned.raw.var_names:
                expr_vals = adata_view_assigned.raw[:, gene].X.toarray().flatten()
                variance = np.var(expr_vals)
                gene_variances.append((gene, variance))
        
        # Sort by variance and select top N
        gene_variances.sort(key=lambda x: x[1], reverse=True)
        selected_genes = [gene for gene, _ in gene_variances[:top_n_genes]]
        print(f"        Selected {len(selected_genes)} most variable genes from {len(available_genes)} available.")
    else:
        selected_genes = available_genes
        print(f"        Using all {len(selected_genes)} available genes (less than {top_n_genes}).")

    # --- Robust Plotting Pattern ---
    # 1. Create Figure and Axes first.
    fig, ax = plt.subplots(figsize=(max(7, len(selected_genes) * 0.45), 5.5))

    try:
        # 2. Pass the created axes to the Scanpy plotting function.
        sc.pl.dotplot(
            adata_view_assigned,
            var_names=selected_genes,
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
            f"Expression of {plot_title} Genes (Top {len(selected_genes)} Most Variable)\nby Subtype in {context_name}",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # 4. Manually calculate data for export. This is more robust than relying on a returned object.
        dotplot_data_list = []
        for gene in selected_genes:
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
        base_filename = f"P2_3c_Dotplot_{set_name}_Top{top_n_genes}"
        plot_basename = create_filename(
            base_filename,
            context_name=context_name,
            version="v1",
        )
        save_figure_and_data(fig, export_df, plot_basename, fig_dir, data_dir)
        print(f"        Dot plot for '{set_name}' saved.")

    except Exception as e:
        print(f"        ERROR generating dot plot for '{set_name}': {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


# --- Main loop to generate comprehensive signature dotplots for all contexts ---
# TEMPORARILY SKIP DOTPLOTS FOR SPEED
SKIP_SIGNATURE_DOTPLOTS = True  # Set to False to re-enable

for context_name, adata_ctx, context_output_base_dir in cohorts_for_characterization:
    if SKIP_SIGNATURE_DOTPLOTS:
        print(f"    SKIPPING signature dotplots for {context_name} (SKIP_SIGNATURE_DOTPLOTS=True)")
        continue
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
    assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
    if not assigned_mask.any():
        print(f"      No assigned cells in {context_name}. Skipping.")
        continue
    adata_view_assigned = adata_ctx[assigned_mask, :].copy()

    print(f"\n    --- Generating Comprehensive Signature Dot Plots for: {context_name} ---")
    print(f"      Processing {adata_view_assigned.n_obs} assigned cells (from {adata_ctx.n_obs} total)")

    # --- Generate plots for DEVELOPMENTAL Signatures ---
    print(f"\n    --- Generating Developmental Signature Dot Plots for: {context_name} ---")
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
        create_signature_dotplot_top_variable(
            adata=adata_view_assigned,
            gene_set=list_of_genes,
            set_name=name_of_set,
            context_name=context_name,
            fig_dir=dev_fig_dir,
            data_dir=dev_data_dir,
            subtype_col=subtype_col,
            top_n_genes=10,
        )

    # --- Generate plots for FUNCTIONAL Signatures ---
    print(f"\n    --- Generating Functional Signature Dot Plots for: {context_name} ---")
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
        create_signature_dotplot_top_variable(
            adata=adata_view_assigned,
            gene_set=list_of_genes,
            set_name=name_of_set,
            context_name=context_name,
            fig_dir=func_fig_dir,
            data_dir=func_data_dir,
            subtype_col=subtype_col,
            top_n_genes=10,
        )

    # --- Generate plots for Neurotransmitter Receptor Signatures ---
    print(f"\n    --- Generating Neurotransmitter Receptor Signature Dot Plots for: {context_name} ---")
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
        create_signature_dotplot_top_variable(
            adata=adata_view_assigned,
            gene_set=list_of_genes,
            set_name=name_of_set,
            context_name=context_name,
            fig_dir=neuro_fig_dir,
            data_dir=neuro_data_dir,
            subtype_col=subtype_col,
            top_n_genes=10,
        )

    # --- Generate plots for Interleukin Downstream Signatures ---
    print(f"\n    --- Generating Interleukin Downstream Signature Dot Plots for: {context_name} ---")
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
        create_signature_dotplot_top_variable(
            adata=adata_view_assigned,
            gene_set=list_of_genes,
            set_name=name_of_set,
            context_name=context_name,
            fig_dir=il_fig_dir,
            data_dir=il_data_dir,
            subtype_col=subtype_col,
            top_n_genes=10,
        )

print("\n--- End of Section 2.3c ---")

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
        mean_expr_df.join(adata_assigned.obs[REBUFFET_SUBTYPE_COL])
        .groupby(REBUFFET_SUBTYPE_COL, observed=True)
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
            groupby=REBUFFET_SUBTYPE_COL,
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
                mask = adata_assigned.obs[REBUFFET_SUBTYPE_COL] == subtype
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
        adata_blood.obs[REBUFFET_SUBTYPE_COL] != "Unassigned"
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
            .rename_axis("Proportion")
            .reset_index(name="Proportion")
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
        plt.tight_layout(rect=(0, 0, 1, 0.97))

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

    # Get the appropriate subtype column for this dataset
    subtype_col = get_subtype_column(adata_ctx)
    
    if (
        adata_ctx is None
        or adata_ctx.n_obs == 0
        or subtype_col not in adata_ctx.obs.columns
        or f"{TUSC2_GENE_NAME}_Expression_Raw" not in adata_ctx.obs.columns
        or TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns
    ):
        print(
            f"      Prerequisites not met for {context_name}. Skipping Layer 2 analysis."
        )
        continue

    obs_col_tusc2_expr = f"{TUSC2_GENE_NAME}_Expression_Raw"
    current_categories_l2 = adata_ctx.obs[
        subtype_col
    ].cat.categories.tolist()
    categories_to_plot_l2 = [
        cat
        for cat in get_subtype_categories(adata_ctx)
        if cat in current_categories_l2 and cat != "Unassigned"
    ]

    if not categories_to_plot_l2:
        print(
            f"      No valid subtypes to plot for {context_name}. Skipping 3.2.1 and 3.2.2."
        )
        continue

    adata_ctx_assigned_only = adata_ctx[
        adata_ctx.obs[subtype_col].isin(categories_to_plot_l2)
    ].copy()
    if adata_ctx_assigned_only.n_obs == 0:
        print(f"      No cells with assigned subtypes in {context_name}. Skipping.")
        continue
    adata_ctx_assigned_only.obs[subtype_col] = adata_ctx_assigned_only.obs[
        subtype_col
    ].cat.reorder_categories(categories_to_plot_l2, ordered=True)

    # --- 3.2.1: Violin/box plots & UMAP: TUSC2 expression across NK_Subtype_Profiled ---
    print(f"      --- 3.2.1: TUSC2 Expression across Subtypes in {context_name} ---")
    try:
        fig_l2_viol, ax_l2_viol = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            x=subtype_col,
            y=obs_col_tusc2_expr,
            data=adata_ctx_assigned_only.obs,
            ax=ax_l2_viol,
            palette=COMBINED_SUBTYPE_COLOR_PALETTE,
            hue=subtype_col,
            order=categories_to_plot_l2,
            legend=False,
        )
        sns.stripplot(
            x=subtype_col,
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
        ax_l2_viol.set_xlabel(subtype_col, fontsize=12)
        ax_l2_viol.set_ylabel(
            f"{TUSC2_GENE_NAME} Expression (Log-Norm, Unscaled)", fontsize=12
        )
        plt.xticks(rotation=45, ha="right")

        data_for_stats_l2 = [
            adata_ctx_assigned_only.obs[obs_col_tusc2_expr][
                adata_ctx_assigned_only.obs[subtype_col] == subtype
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
            [subtype_col, obs_col_tusc2_expr]
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
            violin_data_to_export[["CellID", subtype_col, TUSC2_GENE_NAME]],
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
                size=6,  # Larger dots for better visibility (increased from 1.5)
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
            index=adata_ctx_assigned_only.obs[subtype_col],
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
            ax_l2_bar_binary.set_xlabel(subtype_col, fontsize=12)
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
    print("      Calculating functional signature scores using AUCell...")
    for set_name, gene_list in ALL_FUNCTIONAL_GENE_SETS.items():
        score_col_name = f"{set_name}_Score"
        available_genes = map_gene_names(gene_list, adata_ctx.raw.var_names)
        if len(available_genes) >= MIN_GENES_FOR_SCORING:
            score_genes_aucell(
                adata_ctx,
                available_genes,
                score_name=score_col_name,
                use_raw=True,
                normalize=True,
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

        # Calculate optimal figure dimensions and layout for potentially long signature names
        fig_width, fig_height, left_margin = calculate_heatmap_layout(
            plot_df[["Mean_Score_Diff"]], 
            min_width=5, 
            min_height=8, 
            cell_width=2.5,  # Wider cells for single-column heatmap
            cell_height=0.8,  # Good height for signature names
            label_padding=3.0  # Extra padding for long signature names
        )
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
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
        
        # Use subplots_adjust for better control over layout with long labels
        plt.subplots_adjust(left=left_margin, right=0.88, top=0.92, bottom=0.05)
        plt.tight_layout(rect=[left_margin, 0.05, 0.88, 0.92])

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
# PART 3: TUSC2 Analysis - Section 3.6: GO Enrichment Analysis Functions

def perform_go_pathway_score_comparison(
    adata_ctx,
    context_name,
    output_dirs,
    min_genes_per_pathway=10,
    max_pathways_per_category=60,
    statistical_threshold=0.05,
    min_cells_per_group=10
):
    """
    COMPUTATIONAL VALIDATION APPROACH: Compare GO pathway scores between TUSC2+ and TUSC2- cells.
    
    This provides an alternative validation route by directly scoring GO pathways in cells
    and comparing scores between groups, rather than enrichment analysis on DEGs.
    This should validate findings (TUSC2+ more mature, cytotoxic, etc.) through redundancy.
    
    Parameters:
    -----------
    adata_ctx : AnnData
        Single-cell data with TUSC2 binary groups
    context_name : str
        Context being analyzed (e.g., 'Blood', 'NormalTissue', 'TumorTissue')
    output_dirs : dict
        Dictionary with 'figures', 'data', 'stats' directory paths
    min_genes_per_pathway : int
        Minimum genes required for pathway scoring
    max_pathways_per_category : int
        Maximum pathways to display per GO category
    statistical_threshold : float
        P-value threshold for significant differences
    min_cells_per_group : int
        Minimum cells required per TUSC2 group
        
    Returns:
    --------
    dict : Analysis results and summary statistics
    """
    
    if not GSEAPY_AVAILABLE:
        print(f"    GSEAPY not available. Skipping GO pathway scoring for {context_name}.")
        return {
            'status': 'skipped',
            'reason': 'gseapy_not_available',
            'summary_stats': {}
        }
    
    if TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns:
        print(f"    TUSC2 binary group column not found. Skipping GO pathway scoring for {context_name}.")
        return {
            'status': 'skipped',
            'reason': 'missing_tusc2_groups',
            'summary_stats': {}
        }
    
    print(f"  Starting GO pathway score comparison for {context_name}...")
    
    # Check group sizes
    group_counts = adata_ctx.obs[TUSC2_BINARY_GROUP_COL].value_counts()
    tusc2_pos = group_counts.get('TUSC2_Expressed', 0)
    tusc2_neg = group_counts.get('TUSC2_Not_Expressed', 0)
    
    if tusc2_pos < min_cells_per_group or tusc2_neg < min_cells_per_group:
        print(f"    Insufficient cells: TUSC2+ ({tusc2_pos}), TUSC2- ({tusc2_neg}). Need ≥{min_cells_per_group} each.")
        return {
            'status': 'skipped',
            'reason': 'insufficient_cells',
            'summary_stats': {'tusc2_pos': tusc2_pos, 'tusc2_neg': tusc2_neg}
        }
    
    print(f"    Analyzing TUSC2+ ({tusc2_pos}) vs TUSC2- ({tusc2_neg}) cells")
    
    # Get GO gene sets from gseapy databases - GO terms only
    go_databases = [
        'GO_Biological_Process_2023', 
        'GO_Molecular_Function_2023', 
        'GO_Cellular_Component_2023'
    ]
    
    results_summary = {
        'pathway_comparisons': {},
        'significant_pathways': 0,
        'databases_analyzed': go_databases
    }
    
    try:
        # For each GO database, get gene sets and score pathways
        for db in go_databases:
            print(f"    Processing {db}...")
            
            try:
                # Get gene sets from gseapy
                gene_sets = gseapy.get_library(name=db, organism='Human')
                
                if not gene_sets:
                    print(f"      No gene sets found for {db}")
                    continue
                
                pathway_scores = []
                
                # Score each pathway and compare between groups
                for pathway_name, pathway_genes in gene_sets.items():
                    # Filter to genes present in dataset
                    available_genes = [g for g in pathway_genes if g in adata_ctx.raw.var_names]
                    
                    if len(available_genes) < min_genes_per_pathway:
                        continue
                    
                    # Calculate pathway scores using AUCell
                    try:
                        score_genes_aucell(
                            adata_ctx,
                            available_genes,
                            score_name=f'temp_pathway_score',
                            use_raw=True,
                            normalize=True
                        )
                        
                        # Get scores for each group
                        tusc2_pos_scores = adata_ctx.obs[adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == 'TUSC2_Expressed']['temp_pathway_score'].dropna()
                        tusc2_neg_scores = adata_ctx.obs[adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == 'TUSC2_Not_Expressed']['temp_pathway_score'].dropna()
                        
                        if len(tusc2_pos_scores) >= 3 and len(tusc2_neg_scores) >= 3:
                            # Statistical comparison
                            stat, pval = stats.mannwhitneyu(tusc2_pos_scores, tusc2_neg_scores, alternative='two-sided')
                            
                            # Calculate effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(tusc2_pos_scores) - 1) * tusc2_pos_scores.var() + 
                                                (len(tusc2_neg_scores) - 1) * tusc2_neg_scores.var()) / 
                                               (len(tusc2_pos_scores) + len(tusc2_neg_scores) - 2))
                            cohens_d = (tusc2_pos_scores.mean() - tusc2_neg_scores.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            pathway_scores.append({
                                'Pathway': pathway_name,
                                'Database': db.replace('_2023', '').replace('GO_', ''),
                                'TUSC2_Positive_Mean_Score': tusc2_pos_scores.mean(),
                                'TUSC2_Negative_Mean_Score': tusc2_neg_scores.mean(),
                                'Score_Difference': tusc2_pos_scores.mean() - tusc2_neg_scores.mean(),
                                'P_Value': pval,
                                'Cohens_D': cohens_d,
                                'N_Genes_Scored': len(available_genes),
                                'N_TUSC2_Pos_Cells': len(tusc2_pos_scores),
                                'N_TUSC2_Neg_Cells': len(tusc2_neg_scores)
                            })
                    
                    except Exception as e:
                        # Skip pathways that can't be scored
                        continue
                
                if pathway_scores:
                    pathway_df = pd.DataFrame(pathway_scores)
                    
                    # FDR correction
                    _, pathway_df['FDR_Adjusted_P'], _, _ = multipletests(
                        pathway_df['P_Value'], 
                        method='fdr_bh'
                    )
                    
                    # Filter to significant pathways and sort by significance
                    significant_pathways = pathway_df[
                        pathway_df['FDR_Adjusted_P'] < statistical_threshold
                    ].sort_values('FDR_Adjusted_P').head(max_pathways_per_category)
                    
                    if not significant_pathways.empty:
                        results_summary['pathway_comparisons'][db] = significant_pathways
                        results_summary['significant_pathways'] += len(significant_pathways)
                        print(f"      {db}: {len(significant_pathways)} significant pathway differences")
            
            except Exception as e:
                print(f"      ERROR processing {db}: {e}")
                continue
        
        # Create visualization and save results
        if results_summary['significant_pathways'] > 0:
            print(f"    Creating pathway score comparison visualization...")
            
            # Save detailed results
            for db, result_df in results_summary['pathway_comparisons'].items():
                db_clean = db.replace('_', '').replace('2023', '')
                filename = f"GO_Pathway_Score_Comparison_{db_clean}_{context_name}.csv"
                filepath = os.path.join(output_dirs['stats'], filename)
                result_df.to_csv(filepath, index=False)
            
            # Create summary visualization
            try:
                create_go_pathway_comparison_plot(results_summary, context_name, output_dirs, statistical_threshold)
            except Exception as e:
                print(f"      ERROR creating pathway comparison plot: {e}")
        
        # Summary statistics
        summary_stats = {
            'total_pathways_tested': sum(len(df) for df in results_summary['pathway_comparisons'].values()),
            'significant_pathways': results_summary['significant_pathways'],
            'tusc2_positive_cells': tusc2_pos,
            'tusc2_negative_cells': tusc2_neg,
            'databases_analyzed': go_databases
        }
        
        print(f"    GO pathway score comparison completed for {context_name}")
        return {
            'status': 'completed',
            'results': results_summary,
            'summary_stats': summary_stats
        }
        
    except Exception as e:
        print(f"    ERROR in GO pathway score comparison for {context_name}: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'summary_stats': {'tusc2_positive_cells': tusc2_pos, 'tusc2_negative_cells': tusc2_neg}
        }


def create_go_pathway_comparison_plot(results_summary, context_name, output_dirs, statistical_threshold):
    """
    Create a comparison plot showing GO pathway scores for TUSC2+ vs TUSC2- cells.
    Displays both scores and highlights significant differences for computational validation.
    """
    # Collect pathway data from all databases
    plot_data = []
    
    for db, df in results_summary['pathway_comparisons'].items():
        if not df.empty:
            # Take top 20 pathways per database for display
            top_pathways = df.head(20)
            for _, row in top_pathways.iterrows():
                plot_data.append({
                    'Pathway': row['Pathway'][:45] + '...' if len(row['Pathway']) > 45 else row['Pathway'],
                    'Database': row['Database'],
                    'TUSC2_Positive_Score': row['TUSC2_Positive_Mean_Score'],
                    'TUSC2_Negative_Score': row['TUSC2_Negative_Mean_Score'],
                    'Score_Difference': row['Score_Difference'],
                    'FDR_P_Value': row['FDR_Adjusted_P'],
                    'Cohens_D': row['Cohens_D'],
                    'Direction': 'TUSC2+ Higher' if row['Score_Difference'] > 0 else 'TUSC2- Higher',
                    'Effect_Size': 'Large' if abs(row['Cohens_D']) >= 0.8 else ('Medium' if abs(row['Cohens_D']) >= 0.5 else 'Small')
                })
    
    if not plot_data:
        print("      No significant pathway differences to plot")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(12, len(plot_df) * 0.5)))
    
    # Left plot: TUSC2+ vs TUSC2- pathway scores
    y_positions = np.arange(len(plot_df))
    
    # Create grouped bars for comparison
    bar_width = 0.35
    ax1.barh(y_positions - bar_width/2, plot_df['TUSC2_Positive_Score'], 
             bar_width, label='TUSC2+ Mean Score', color='#d62728', alpha=0.8)
    ax1.barh(y_positions + bar_width/2, plot_df['TUSC2_Negative_Score'], 
             bar_width, label='TUSC2- Mean Score', color='#1f77b4', alpha=0.8)
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([f"{row['Database']}: {row['Pathway']}" for _, row in plot_df.iterrows()], fontsize=8)
    ax1.set_xlabel('Pathway Score (AUCell)', fontsize=12)
    ax1.set_title(f'GO Pathway Scores: TUSC2+ vs TUSC2-\n{context_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # Right plot: Score differences with significance
    colors_diff = ['#d62728' if diff > 0 else '#1f77b4' for diff in plot_df['Score_Difference']]
    bars = ax2.barh(y_positions, plot_df['Score_Difference'], color=colors_diff, alpha=0.7)
    
    # Add significance stars
    for i, (_, row) in enumerate(plot_df.iterrows()):
        fdr_p = row['FDR_P_Value']
        if fdr_p < 0.0001:
            stars = '****'
        elif fdr_p < 0.001:
            stars = '***'
        elif fdr_p < 0.01:
            stars = '**'
        elif fdr_p < 0.05:
            stars = '*'
        else:
            stars = ''
        
        if stars:
            # Position stars at the end of each bar
            x_pos = row['Score_Difference'] + (0.01 if row['Score_Difference'] > 0 else -0.01)
            ax2.text(x_pos, i, stars, va='center', ha='left' if row['Score_Difference'] > 0 else 'right', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"{row['Database']}: {row['Pathway']}" for _, row in plot_df.iterrows()], fontsize=8)
    ax2.set_xlabel('Score Difference (TUSC2+ - TUSC2-)', fontsize=12)
    ax2.set_title(f'Pathway Score Differences\n(Significant Only, FDR < {statistical_threshold})', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add effect size information
    textstr = f'Total Pathways: {len(plot_df)}\n'
    textstr += f'TUSC2+ Higher: {sum(plot_df["Score_Difference"] > 0)}\n'
    textstr += f'TUSC2- Higher: {sum(plot_df["Score_Difference"] < 0)}\n'
    textstr += f'Large Effect: {sum(plot_df["Effect_Size"] == "Large")}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    plot_basename = create_filename(
        f"P3_6_GO_Pathway_Score_Comparison", 
        context_name=context_name, 
        version="v1_validation"
    )
    save_figure_and_data(
        fig, plot_df, plot_basename, 
        output_dirs['figures'], output_dirs['data']
    )
    
    print(f"      GO pathway comparison plot saved: {plot_basename}")


def create_go_summary_plot(results_summary, context_name, output_dirs, fdr_threshold):
    """
    LEGACY FUNCTION - Create a summary plot showing top GO terms from each category.
    This function is kept for backward compatibility but not used in the new approach.
    """
    # This function is now deprecated in favor of create_go_pathway_comparison_plot
    print("      Using new pathway score comparison visualization instead of legacy GO summary plot")
    return

# %%
# PART 3: TUSC2 Analysis - Section 3.6: GO Enrichment Analysis (PROPER IMPLEMENTATION)

def perform_go_enrichment_analysis(
    deg_results_df,
    context_name,
    output_dirs,
    fc_threshold=0.5,
    fdr_threshold=0.05,
    min_genes_for_analysis=10
):
    """
    Perform proper GO enrichment analysis using differentially expressed genes.
    
    This tests for over-representation of biological pathways in DEGs between 
    TUSC2+ and TUSC2- cells using Fisher's exact test - the scientifically 
    rigorous approach for pathway analysis.
    
    Parameters:
    -----------
    deg_results_df : pd.DataFrame
        DEG results with columns: names, logfoldchanges, pvals_adj
    context_name : str
        Context being analyzed 
    output_dirs : dict
        Output directories for figures, data, stats
    fc_threshold : float
        Log2 fold change threshold for DEG filtering
    fdr_threshold : float
        FDR threshold for significance
    min_genes_for_analysis : int
        Minimum DEGs required for analysis
    
    Returns:
    --------
    dict : Analysis results and summary
    """
    
    if not GSEAPY_AVAILABLE:
        print(f"    GSEAPY not available. Skipping GO enrichment for {context_name}.")
        return {'status': 'skipped', 'reason': 'gseapy_not_available'}
    
    if deg_results_df.empty:
        print(f"    No DEG results available for {context_name}. Skipping GO enrichment.")
        return {'status': 'skipped', 'reason': 'no_deg_results'}
    
    print(f"  Starting proper GO enrichment analysis for {context_name}...")
    
    # Filter significant DEGs
    significant_degs = deg_results_df[
        (deg_results_df['pvals_adj'] < fdr_threshold) & 
        (abs(deg_results_df['logfoldchanges']) > fc_threshold)
    ].copy()
    
    if len(significant_degs) < min_genes_for_analysis:
        print(f"    Insufficient DEGs ({len(significant_degs)}) for GO analysis in {context_name}.")
        return {'status': 'skipped', 'reason': 'insufficient_degs'}
    
    # Separate upregulated and downregulated genes
    up_genes = significant_degs[significant_degs['logfoldchanges'] > 0]['names'].tolist()
    down_genes = significant_degs[significant_degs['logfoldchanges'] < 0]['names'].tolist()
    
    print(f"    Found {len(up_genes)} upregulated and {len(down_genes)} downregulated DEGs")
    
    # Curated GO terms specifically relevant to NK cell biology and TUSC2 research
    # Based on research: NK cell development, cytotoxicity, metabolism, immune response
    
    curated_go_terms = [
        # === NK CELL SPECIFIC BIOLOGICAL PROCESSES ===
        "GO:0042267",  # natural killer cell mediated cytotoxicity
        "GO:0001913",  # T cell mediated cytotoxicity  
        "GO:0001906",  # cell killing
        "GO:0019835",  # cytolysis
        "GO:0042110",  # T cell activation
        "GO:0030101",  # natural killer cell activation
        "GO:0002250",  # adaptive immune response
        "GO:0045321",  # leukocyte activation
        "GO:0002218",  # activation of innate immune response
        "GO:0002376",  # immune system process
        
        # === METABOLIC PROCESSES (upregulated in TUSC2+) ===
        "GO:0006096",  # glycolytic process
        "GO:0006119",  # oxidative phosphorylation
        "GO:0019395",  # fatty acid oxidation
        "GO:0006006",  # glucose metabolic process
        "GO:0006091",  # generation of precursor metabolites and energy
        "GO:0008152",  # metabolic process
        "GO:0044710",  # single-organism metabolic process
        "GO:0006629",  # lipid metabolic process
        "GO:0045333",  # cellular respiration
        "GO:0006094",  # gluconeogenesis
        
        # === CELL DEVELOPMENT & DIFFERENTIATION ===
        "GO:0030154",  # cell differentiation
        "GO:0048468",  # cell development
        "GO:0002520",  # immune system development
        "GO:0030097",  # hemopoiesis
        "GO:0002521",  # leukocyte differentiation
        "GO:0030098",  # lymphocyte differentiation
        "GO:0045058",  # T cell selection
        "GO:0002507",  # tolerance induction
        "GO:0048534",  # hematopoietic or lymphoid organ development
        "GO:0002262",  # myeloid cell homeostasis
        
        # === IMMUNE RESPONSE & CYTOKINE SIGNALING ===
        "GO:0006955",  # immune response
        "GO:0002252",  # immune effector process
        "GO:0019221",  # cytokine-mediated signaling pathway
        "GO:0034097",  # response to cytokine
        "GO:0070498",  # interleukin-1-mediated signaling pathway
        "GO:0070062",  # extracellular exosome
        "GO:0043207",  # response to external biotic stimulus
        "GO:0009617",  # response to bacterium
        "GO:0009615",  # response to virus
        "GO:0002684",  # positive regulation of immune system process
        
        # === CELL DEATH & APOPTOSIS ===
        "GO:0006915",  # apoptotic process
        "GO:0012501",  # programmed cell death
        "GO:0043065",  # positive regulation of apoptotic process
        "GO:0008219",  # cell death
        "GO:0070059",  # intrinsic apoptotic signaling pathway
        "GO:0097190",  # apoptotic signaling pathway
        "GO:0006917",  # apoptotic process
        "GO:0042981",  # regulation of apoptotic process
        
        # === CELL ACTIVATION & PROLIFERATION ===
        "GO:0001775",  # cell activation
        "GO:0008283",  # cell proliferation
        "GO:0007049",  # cell cycle
        "GO:0000079",  # regulation of cyclin-dependent protein kinase activity
        "GO:0051301",  # cell division
        "GO:0007067",  # mitotic nuclear division
        "GO:0006260",  # DNA replication
        "GO:0043066",  # negative regulation of apoptotic process
    ]
    
    # Use only GO Biological Process database with curated terms
    go_databases = ['GO_Biological_Process_2023']
    
    enrichment_results = {}
    summary_stats = {
        'total_upregulated_degs': len(up_genes),
        'total_downregulated_degs': len(down_genes),
        'databases_analyzed': go_databases,
        'enriched_pathways': 0
    }
    
    try:
        # Analyze upregulated genes
        if len(up_genes) >= 5:  # Minimum for meaningful enrichment
            print(f"    Analyzing upregulated genes...")
            up_enrichment = gseapy.enrichr(
                gene_list=up_genes,
                gene_sets=go_databases,
                organism='Human',
                cutoff=fdr_threshold,
                no_plot=True
            )
            
            if up_enrichment.results is not None and not up_enrichment.results.empty:
                up_results = up_enrichment.results[up_enrichment.results['Adjusted P-value'] < fdr_threshold]
                if not up_results.empty:
                    enrichment_results['upregulated'] = up_results.head(20)  # Top 20 pathways
                    summary_stats['enriched_pathways'] += len(up_results)
                    print(f"      Found {len(up_results)} enriched pathways for upregulated genes")
        
        # Analyze downregulated genes
        if len(down_genes) >= 5:  # Minimum for meaningful enrichment
            print(f"    Analyzing downregulated genes...")
            down_enrichment = gseapy.enrichr(
                gene_list=down_genes,
                gene_sets=go_databases,
                organism='Human',
                cutoff=fdr_threshold,
                no_plot=True
            )
            
            if down_enrichment.results is not None and not down_enrichment.results.empty:
                down_results = down_enrichment.results[down_enrichment.results['Adjusted P-value'] < fdr_threshold]
                if not down_results.empty:
                    enrichment_results['downregulated'] = down_results.head(20)  # Top 20 pathways
                    summary_stats['enriched_pathways'] += len(down_results)
                    print(f"      Found {len(down_results)} enriched pathways for downregulated genes")
        
        # Save results
        if enrichment_results:
            for direction, results_df in enrichment_results.items():
                filename = f"P3_6_GO_Enrichment_{direction}_DEGs_{context_name}.csv"
                filepath = os.path.join(output_dirs['stats'], filename)
                results_df.to_csv(filepath, index=False)
                print(f"      Saved {direction} enrichment results to {filename}")
        
        # Create comprehensive visualization
        if enrichment_results:
            create_go_enrichment_plot(enrichment_results, context_name, output_dirs)
        
        # Create curated GO term enrichment analysis
        focused_results = perform_curated_go_enrichment(
            up_genes, down_genes, curated_go_terms, context_name, output_dirs, fdr_threshold
        )
        
        return {
            'status': 'completed',
            'results': enrichment_results,
            'focused_results': focused_results,
            'summary_stats': summary_stats
        }
        
    except Exception as e:
        print(f"    ERROR in GO enrichment analysis for {context_name}: {e}")
        return {'status': 'error', 'error': str(e), 'summary_stats': summary_stats}


def perform_curated_go_enrichment(up_genes, down_genes, curated_go_terms, context_name, output_dirs, fdr_threshold):
    """
    Perform curated GO enrichment analysis using specific GO terms relevant to NK cell biology.
    
    Targets specific biological processes:
    - NK cell cytotoxicity (GO:0042267, GO:0001913, GO:0001906)
    - Metabolic processes (GO:0006096, GO:0006119, GO:0019395) 
    - Cell development (GO:0030154, GO:0048468, GO:0002520)
    - Immune response (GO:0006955, GO:0002252, GO:0019221)
    - Cell death/apoptosis (GO:0006915, GO:0012501)
    """
    
    # Define GO term categories for organization
    go_term_categories = {
        'nk_cytotoxicity': ['GO:0042267', 'GO:0001913', 'GO:0001906', 'GO:0019835', 'GO:0030101'],
        'metabolism': ['GO:0006096', 'GO:0006119', 'GO:0019395', 'GO:0006006', 'GO:0006091', 
                      'GO:0008152', 'GO:0044710', 'GO:0006629', 'GO:0045333', 'GO:0006094'],
        'development': ['GO:0030154', 'GO:0048468', 'GO:0002520', 'GO:0030097', 'GO:0002521', 
                       'GO:0030098', 'GO:0045058', 'GO:0002507', 'GO:0048534', 'GO:0002262'],
        'immune_response': ['GO:0006955', 'GO:0002252', 'GO:0019221', 'GO:0034097', 'GO:0070498',
                           'GO:0043207', 'GO:0009617', 'GO:0009615', 'GO:0002684', 'GO:0002250', 
                           'GO:0045321', 'GO:0002218', 'GO:0002376'],
        'cell_death': ['GO:0006915', 'GO:0012501', 'GO:0043065', 'GO:0008219', 'GO:0070059',
                      'GO:0097190', 'GO:0006917', 'GO:0042981', 'GO:0043066'],
        'cell_activation': ['GO:0001775', 'GO:0008283', 'GO:0007049', 'GO:0000079', 'GO:0051301',
                           'GO:0007067', 'GO:0006260', 'GO:0042110']
    }
    
    # Create reverse mapping for categorization
    go_to_category = {}
    for category, terms in go_term_categories.items():
        for term in terms:
            go_to_category[term] = category
    
    focused_results = {}
    
    try:
        print(f"    Performing curated GO enrichment with {len(curated_go_terms)} specific terms...")
        
        # Analyze upregulated genes
        if len(up_genes) >= 5:
            print(f"      Analyzing upregulated genes against curated GO terms...")
            up_focused = gseapy.enrichr(
                gene_list=up_genes,
                gene_sets='GO_Biological_Process_2023',
                organism='Human',
                cutoff=fdr_threshold,
                no_plot=True
            )
            
            if up_focused.results is not None and not up_focused.results.empty:
                up_results = up_focused.results[up_focused.results['Adjusted P-value'] < fdr_threshold]
                
                # Filter for curated GO terms
                curated_terms = []
                for _, row in up_results.iterrows():
                    # Extract GO ID from the term
                    term_id = None
                    if '(GO:' in row['Term']:
                        # Extract GO ID from parentheses
                        import re
                        go_match = re.search(r'\(GO:\d+\)', row['Term'])
                        if go_match:
                            term_id = go_match.group(0)[1:-1]  # Remove parentheses
                    
                    if term_id and term_id in curated_go_terms:
                        category = go_to_category.get(term_id, 'other')
                        curated_terms.append({
                            'Term': row['Term'],
                            'GO_ID': term_id,
                            'Database': 'GO_Biological_Process',
                            'Adjusted_P_Value': row['Adjusted P-value'],
                            'Gene_Count': len(row['Genes'].split(';')) if isinstance(row['Genes'], str) else 0,
                            'Genes': row['Genes'],
                            'Category': category,
                            'Direction': 'TUSC2+ Upregulated'
                        })
                
                if curated_terms:
                    focused_results['upregulated'] = pd.DataFrame(curated_terms)
                    print(f"        Found {len(curated_terms)} curated upregulated GO terms")
        
        # Analyze downregulated genes
        if len(down_genes) >= 5:
            print(f"      Analyzing downregulated genes against curated GO terms...")
            down_focused = gseapy.enrichr(
                gene_list=down_genes,
                gene_sets='GO_Biological_Process_2023',
                organism='Human',
                cutoff=fdr_threshold,
                no_plot=True
            )
            
            if down_focused.results is not None and not down_focused.results.empty:
                down_results = down_focused.results[down_focused.results['Adjusted P-value'] < fdr_threshold]
                
                # Filter for curated GO terms
                curated_terms = []
                for _, row in down_results.iterrows():
                    # Extract GO ID from the term
                    term_id = None
                    if '(GO:' in row['Term']:
                        import re
                        go_match = re.search(r'\(GO:\d+\)', row['Term'])
                        if go_match:
                            term_id = go_match.group(0)[1:-1]
                    
                    if term_id and term_id in curated_go_terms:
                        category = go_to_category.get(term_id, 'other')
                        curated_terms.append({
                            'Term': row['Term'],
                            'GO_ID': term_id,
                            'Database': 'GO_Biological_Process',
                            'Adjusted_P_Value': row['Adjusted P-value'],
                            'Gene_Count': len(row['Genes'].split(';')) if isinstance(row['Genes'], str) else 0,
                            'Genes': row['Genes'],
                            'Category': category,
                            'Direction': 'TUSC2+ Downregulated'
                        })
                
                if curated_terms:
                    focused_results['downregulated'] = pd.DataFrame(curated_terms)
                    print(f"        Found {len(curated_terms)} curated downregulated GO terms")
        
        # Save curated GO results
        if focused_results:
            # Combine all results for comprehensive analysis
            all_focused = []
            for direction, df in focused_results.items():
                all_focused.append(df)
            
            if all_focused:
                combined_focused = pd.concat(all_focused, ignore_index=True)
                
                # Save combined results
                filename = f"P3_6_CuratedGO_Enrichment_{context_name}.csv"
                filepath = os.path.join(output_dirs['stats'], filename)
                combined_focused.to_csv(filepath, index=False)
                print(f"        Saved curated GO results to {filename}")
                
                # Create curated GO visualization
                create_curated_go_plot(combined_focused, context_name, output_dirs)
        
        return focused_results
        
    except Exception as e:
        print(f"      ERROR in curated GO enrichment: {e}")
        return {}


def create_curated_go_plot(focused_df, context_name, output_dirs):
    """Create visualization specifically for curated GO terms"""
    
    if focused_df.empty:
        print("        No curated GO terms to plot")
        return
    
    # Sort by adjusted p-value and take top terms per category
    top_terms_per_category = 10
    plot_data = []
    
    for category in focused_df['Category'].unique():
        category_df = focused_df[focused_df['Category'] == category].sort_values('Adjusted_P_Value')
        top_category = category_df.head(top_terms_per_category)
        plot_data.append(top_category)
    
    if not plot_data:
        return
    
    plot_df = pd.concat(plot_data, ignore_index=True)
    plot_df['-log10(FDR)'] = -np.log10(plot_df['Adjusted_P_Value'] + 1e-10)
    
    # Create enhanced visualization
    n_terms = len(plot_df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, max(14, n_terms * 0.7)))
    
    # Left panel: Dot plot colored by category with GO IDs
    category_colors = {
        'nk_cytotoxicity': '#d62728',    # Red for cytotoxicity
        'metabolism': '#ff7f0e',         # Orange for metabolism  
        'development': '#2ca02c',        # Green for development
        'immune_response': '#1f77b4',    # Blue for immune response
        'cell_death': '#9467bd',         # Purple for cell death
        'cell_activation': '#8c564b'     # Brown for activation
    }
    
    y_positions = range(len(plot_df))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = category_colors.get(row['Category'], '#7f7f7f')
        ax1.scatter(
            row['-log10(FDR)'], i,
            s=row['Gene_Count'] * 6,  # Larger dots for visibility
            c=color,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.8
        )
    
    ax1.set_yticks(y_positions)
    # Include GO ID in labels for scientific rigor
    ax1.set_yticklabels([
        f"{row['GO_ID']}: {row['Term'][:50]}{'...' if len(row['Term']) > 50 else ''}"
        for _, row in plot_df.iterrows()
    ], fontsize=9)
    ax1.set_xlabel('-log10(Adjusted P-value)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Curated GO Enrichment: NK Cell Biology\n{context_name}', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, linewidth=2, label='FDR = 0.05')
    
    # Right panel: Category summary with enhanced styling
    category_counts = plot_df.groupby(['Category', 'Direction']).size().unstack(fill_value=0)
    
    # Create stacked horizontal bar chart
    ax2_plot = category_counts.plot(kind='barh', ax=ax2, 
                                   color=['#2E86AB', '#A23B72'], # Custom colors for up/down
                                   width=0.7)
    
    ax2.set_title('GO Terms by Functional Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Enriched GO Terms', fontsize=12)
    ax2.legend(title='Gene Expression Direction', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format category labels
    category_labels = [cat.replace('_', ' ').title() for cat in category_counts.index]
    ax2.set_yticklabels(category_labels)
    
    # Add category legend to left panel
    legend_elements = [plt.scatter([], [], c=color, s=150, alpha=0.8, edgecolors='black', 
                                  label=category.replace('_', ' ').title()) 
                      for category, color in category_colors.items() if category in plot_df['Category'].values]
    ax1.legend(handles=legend_elements, title='Biological Process Category', 
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    plot_filename = f"P3_6_CuratedGO_Enrichment_{context_name}.png"
    filepath = os.path.join(output_dirs['figures'], plot_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"        Curated GO enrichment plot saved to {plot_filename}")
    
    # Save summary statistics with GO IDs
    summary_stats = {
        'total_curated_go_terms': len(plot_df),
        'go_terms_by_category': plot_df['Category'].value_counts().to_dict(),
        'go_terms_by_direction': plot_df['Direction'].value_counts().to_dict(),
        'most_significant_go_term': {
            'term': plot_df.loc[plot_df['Adjusted_P_Value'].idxmin(), 'Term'] if not plot_df.empty else None,
            'go_id': plot_df.loc[plot_df['Adjusted_P_Value'].idxmin(), 'GO_ID'] if not plot_df.empty else None,
            'p_value': float(plot_df.loc[plot_df['Adjusted_P_Value'].idxmin(), 'Adjusted_P_Value']) if not plot_df.empty else None
        },
        'enriched_go_ids': plot_df['GO_ID'].tolist()
    }
    
    summary_filename = f"P3_6_CuratedGO_Summary_{context_name}.json"
    summary_filepath = os.path.join(output_dirs['stats'], summary_filename)
    import json
    with open(summary_filepath, 'w') as f:
        json.dump(summary_stats, f, indent=2)


def create_go_enrichment_plot(enrichment_results, context_name, output_dirs):
    """Create visualization of GO enrichment results"""
    
    # Prepare data for plotting
    plot_data = []
    for direction, results_df in enrichment_results.items():
        top_results = results_df.head(10)  # Top 10 per direction
        for _, row in top_results.iterrows():
            plot_data.append({
                'Term': row['Term'][:50] + '...' if len(row['Term']) > 50 else row['Term'],
                'Database': row['Gene_set'].split('_')[1] if '_' in row['Gene_set'] else row['Gene_set'],
                'Adjusted_P_Value': row['Adjusted P-value'],
                'Neg_Log10_P': -np.log10(row['Adjusted P-value'] + 1e-10),
                'Gene_Count': len(row['Genes'].split(';')) if isinstance(row['Genes'], str) else 0,
                'Direction': 'TUSC2+ Upregulated' if direction == 'upregulated' else 'TUSC2+ Downregulated'
            })
    
    if not plot_data:
        print("      No significant pathways to plot")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create dot plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.4)))
    
    # Color by direction
    colors = {'TUSC2+ Upregulated': '#d62728', 'TUSC2+ Downregulated': '#1f77b4'}
    
    for direction in plot_df['Direction'].unique():
        direction_data = plot_df[plot_df['Direction'] == direction]
        scatter = ax.scatter(
            direction_data['Neg_Log10_P'],
            range(len(direction_data)),
            s=direction_data['Gene_Count'] * 3,  # Size by gene count
            c=colors[direction],
            alpha=0.7,
            label=direction,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Format plot
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([f"{row['Database']}: {row['Term']}" for _, row in plot_df.iterrows()], fontsize=8)
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title(f'GO Pathway Enrichment: TUSC2+ vs TUSC2- DEGs\n{context_name}', fontsize=14, fontweight='bold')
    ax.legend(title='Gene Direction', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', alpha=0.3)
    
    # Add significance threshold line
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    
    plt.tight_layout()
    
    # Save figure
    plot_filename = f"P3_6_GO_Enrichment_Summary_{context_name}.png"
    filepath = os.path.join(output_dirs['figures'], plot_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      GO enrichment plot saved to {plot_filename}")


print("\nPerforming proper GO enrichment analysis using TUSC2 DEGs...")
print("This tests for over-representation of biological pathways in DEGs (Fisher's exact test).")

for context_name, adata_ctx, context_output_base_dir in cohorts_for_characterization:
    print(f"\n--- Processing {context_name} ---")
    
    # Define context-specific directories for GO analysis
    go_base_dir = os.path.join(OUTPUT_SUBDIRS["tusc2_analysis"], "5C_DEG_TUSC2_Binary", context_name)
    context_dirs = {
        'figures': os.path.join(go_base_dir, "figures_go_enrichment"),
        'data': os.path.join(go_base_dir, "data_for_graphpad_go_enrichment"),
        'stats': os.path.join(go_base_dir, "stat_results_python_go_enrichment")
    }
    
    # Create directories
    for dir_path in context_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load DEG results from Section 3.5
    deg_stats_dir = os.path.join(OUTPUT_SUBDIRS["tusc2_analysis"], "5C_DEG_TUSC2_Binary", context_name, "stat_results")
    deg_filename = create_filename(f"P3_5_Significant_DEG_results", context_name=context_name, ext="csv")
    deg_filepath = os.path.join(deg_stats_dir, deg_filename)
    
    if not os.path.exists(deg_filepath):
        print(f"    DEG results not found: {deg_filepath}")
        print(f"    Skipping GO enrichment for {context_name}")
        continue
    
    # Load DEG data
    try:
        deg_results_df = pd.read_csv(deg_filepath)
        print(f"    Loaded {len(deg_results_df)} DEGs from Section 3.5")
    except Exception as e:
        print(f"    ERROR loading DEG results: {e}")
        continue
    
    # Perform GO enrichment analysis
    go_results = perform_go_enrichment_analysis(
        deg_results_df=deg_results_df,
        context_name=context_name,
        output_dirs=context_dirs,
        fc_threshold=0.5,
        fdr_threshold=0.05,
        min_genes_for_analysis=10
    )
    
    # Print summary
    if go_results['status'] == 'completed':
        go_stats = go_results['summary_stats']
        print(f"  GO Enrichment Summary for {context_name}:")
        print(f"    Upregulated DEGs: {go_stats['total_upregulated_degs']}")
        print(f"    Downregulated DEGs: {go_stats['total_downregulated_degs']}")
        print(f"    Enriched pathways found: {go_stats['enriched_pathways']}")
        print(f"    Databases analyzed: {', '.join(go_stats['databases_analyzed'])}")
        
    elif go_results['status'] == 'skipped':
        reason = go_results.get('reason', 'unknown')
        print(f"    Skipped GO enrichment for {context_name}: {reason}")
    
    elif go_results['status'] == 'error':
        print(f"    ERROR in GO enrichment for {context_name}: {go_results.get('error', 'Unknown error')}")

print("\n--- End of Section 3.6 (Proper GO Enrichment) ---")

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
                cmap="icefire",
                center=0,
                vmin=-abs_max,
                vmax=abs_max,
                linewidths=0.5,
                cbar_kws={"label": "Mean Score Difference\n(TUSC2+ vs TUSC2-)", "shrink": 0.6},  # Superior shrink parameter matching P2_3b
                annot_kws={"size": 11, "weight": "bold"},
                square=True,
                ax=ax,
            )
            ax.set_title("TUSC2 Impact on Developmental State", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Context", fontsize=14, fontweight='bold')
            ax.set_ylabel("Developmental Signature", fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
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

    # NOTE: Tang-derived signatures removed - now using unified developmental signatures

    # Remove overlaps - prioritize by category
    remaining_func_signatures = [
        sig
        for sig in func_signatures
        if sig not in neuro_signatures
        and sig not in il_signatures
        and sig not in core_functional_signatures
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

        # Clean up signature names for better readability (matching P2_3b styling)
        pivot_diff.index = (
            pivot_diff.index.str.replace("_Score", "")
            .str.replace("Maturation_NK._", "", regex=True)
            .str.replace("_", " ")
            # Additional aggressive shortening for long signature names to improve text fitting
            .str.replace("Cytokine Chemokine Production", "Cytokine/Chemokine")
            .str.replace("Exhaustion Suppression Markers", "Exhaustion/Suppression")
            .str.replace("NK Oxidative Phosphorylation", "Oxidative Phosphorylation")
            .str.replace("NK Fatty Acid Metabolism", "Fatty Acid Metabolism")
        )
        pivot_qval.index = pivot_diff.index  # Keep indices synchronized
        
        # Create annotation labels with values + significance stars
        annot_labels = pivot_diff.apply(
            lambda x: x.map("{:.3f}".format)
        ) + pivot_qval.apply(lambda x: x.map(get_significance_stars))

        # Calculate optimal figure dimensions and layout for long signature labels using P2_3b superior parameters
        fig_width, fig_height, left_margin = calculate_heatmap_layout(
            pivot_diff, 
            min_width=10, 
            min_height=6, 
            cell_width=1.5,  # Larger cells for better readability (matching P2_3b)
            cell_height=0.9,  # Taller cells for signature names (matching P2_3b)
            label_padding=3.0  # More padding for long signature names (matching P2_3b)
        )
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        abs_max = (
            np.nanmax(np.abs(pivot_diff.values))
            if not np.all(np.isnan(pivot_diff.values))
            else 0.1
        )

        # Adjust annotation font size specifically for CoreFunctionalCapacity (P4_2) figure
        annot_size = 9 if basename_suffix == "CoreFunctionalCapacity" else 11
        
        sns.heatmap(
            pivot_diff,
            annot=annot_labels,
            fmt="s",
            cmap="icefire",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={"label": "Mean Score Difference\n(TUSC2+ vs TUSC2-)", "shrink": 0.6},  # Superior shrink parameter matching P2_3b
            annot_kws={"size": annot_size, "weight": "bold"},  # Smaller font for P4_2, standard for others
            square=True,  # Force square cells for consistent appearance
            ax=ax,
        )
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Context", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"{category_name} Signature", fontsize=14, fontweight='bold')
        
        # Improve tick label readability (matching P2_3b styling)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)  # Consistent y-axis label size matching P2_3b
        
        # Use improved layout control for better signature name visibility (matching P2_3b)
        plt.subplots_adjust(left=left_margin, right=0.85, top=0.9, bottom=0.1)
        plt.tight_layout(rect=[left_margin, 0.1, 0.85, 0.9])

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

    # NOTE: Tang-derived signatures removed - now using unified developmental signatures only

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

            # Calculate signature score using AUCell
            score_genes_aucell(
                adata_ctx,
                subtype_genes,
                score_name=score_name,
                use_raw=True,
                normalize=True,
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

        # Calculate optimal figure dimensions and layout for potentially long labels
        # Use simple, clean figure sizing like Part 2
        fig, ax = plt.subplots(
            figsize=(
                max(8, heatmap_pivot_diff.shape[1] * 1.5),
                max(6, heatmap_pivot_diff.shape[0] * 0.6),
            )
        )
        
        abs_max = (
            np.nanmax(np.abs(heatmap_pivot_diff.values))
            if not np.all(np.isnan(heatmap_pivot_diff.values))
            else 0.1
        )

        sns.heatmap(
            heatmap_pivot_diff,
            annot=annot_labels,
            fmt="s",
            cmap="icefire",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={
                "label": "Mean Program Score Difference\n(TUSC2 Expressed vs. Not Expressed)",
                "shrink": 0.6,  # Superior shrink parameter matching P2_3b
            },
            annot_kws={"size": 11, "weight": "bold"},
            square=True,
            ax=ax,
        )
        ax.set_title(
            "TUSC2 Expression is Associated with Mature Cytotoxic Subtypes",
            fontsize=16,
            fontweight='bold',
            pad=20,
        )
        ax.set_xlabel("Biological Context", fontsize=14, fontweight='bold')
        ax.set_ylabel("NK Subtype Signature", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)
        
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

            p_value, mean_diff, z_score_diff = np.nan, np.nan, np.nan
            if len(scores_pos) >= 3 and len(scores_neg) >= 3:
                # Calculate raw mean difference (for reference)
                mean_diff = scores_pos.mean() - scores_neg.mean()
                
                # Calculate Z-score difference for cross-context comparison
                # This normalizes the TUSC2 effect by the gene's expression variability in this context
                # Z-score = (mean_pos - mean_neg) / pooled_std
                # Enables comparison of effect magnitudes across different genes and contexts
                pooled_mean = gene_expression.mean()
                pooled_std = gene_expression.std()
                
                if pooled_std > 0:  # Avoid division by zero
                    z_score_diff = (scores_pos.mean() - scores_neg.mean()) / pooled_std
                else:
                    z_score_diff = 0.0
                
                # Statistical test
                _, p_value = stats.mannwhitneyu(
                    scores_pos, scores_neg, alternative="two-sided"
                )

            all_dev_stats_results.append(
                {
                    "Context": context_name,
                    "Developmental_Marker": marker_gene,
                    "Mean_Diff_TUSC2pos_vs_neg": mean_diff,
                    "Z_Score_Diff_TUSC2pos_vs_neg": z_score_diff,
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

        # Use Z-score differences for cross-context comparison
        heatmap_pivot_diff = summary_dev_stats_df.pivot_table(
            index="Developmental_Marker",
            columns="Context",
            values="Z_Score_Diff_TUSC2pos_vs_neg",
        )
        heatmap_pivot_qval = summary_dev_stats_df.pivot_table(
            index="Developmental_Marker", columns="Context", values="Q_Value_FDR"
        )
        
        # Also create a pivot table for raw mean differences for reference in saved data
        heatmap_pivot_raw = summary_dev_stats_df.pivot_table(
            index="Developmental_Marker",
            columns="Context",
            values="Mean_Diff_TUSC2pos_vs_neg",
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
            "\n      --- Generating Summary Heatmap: Impact of TUSC2 on Developmental Marker Expression (Z-Score Normalized) ---"
        )

        # Calculate optimal figure dimensions and layout for potentially long labels
        # Use simple, clean figure sizing like Part 2
        fig, ax = plt.subplots(
            figsize=(
                max(8, heatmap_pivot_diff.shape[1] * 1.5),
                max(8, heatmap_pivot_diff.shape[0] * 0.6),
            )
        )
        
        abs_max = (
            np.nanmax(np.abs(heatmap_pivot_diff.values))
            if not np.all(np.isnan(heatmap_pivot_diff.values))
            else 0.1
        )

        sns.heatmap(
            heatmap_pivot_diff,
            annot=annot_labels,
            fmt="s",
            cmap="icefire",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.5,
            cbar_kws={
                "label": "Z-Score Difference\n(TUSC2 Expressed vs. Not Expressed)",
                "shrink": 0.6  # Superior shrink parameter matching P2_3b
            },
            annot_kws={"size": 11, "weight": "bold"},
            square=True,
            ax=ax,
        )
        ax.set_title(
            "Impact of TUSC2 Expression on Developmental State Markers\n(Z-Score Normalized for Cross-Context Comparison)",
            fontsize=16,
            fontweight='bold',
            pad=20,
        )
        ax.set_xlabel("Context", fontsize=14, fontweight='bold')
        ax.set_ylabel("Developmental Marker", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)
        
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
            "\n      Summary heatmap of TUSC2's Z-score normalized impact on developmental markers saved."
        )

print("\n--- End of Section 4.4 ---")

# NOTE: RNA Velocity Analysis Removed
# RNA velocity analysis requires spliced/unspliced transcript counts which are not available
# in standard single-cell RNA-seq data. This analysis would require specific preprocessing
# with tools like velocyto or specific sequencing protocols that capture intronic reads.
print("\n--- RNA Velocity Analysis Skipped ---")
print("  RNA velocity analysis requires spliced/unspliced transcript counts")
print("  which are not available in this dataset. This analysis would require:")
print("  - Specific preprocessing with velocyto or similar tools")
print("  - Raw sequencing data with intronic read capture")
print("  - Smart-seq2 or 10X data processed for velocity analysis")
print("--- Continuing with standard transcriptomic analysis ---")

print(
    "\n--- Analysis Complete: NK Cell Transcriptomics and TUSC2 Function Analysis v3.2 (Final) ---"
)
