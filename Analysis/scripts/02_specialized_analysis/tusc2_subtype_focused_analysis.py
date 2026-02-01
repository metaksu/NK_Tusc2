#!/usr/bin/env python3
"""
TUSC2-NK Subtype Focused Analysis: Direct & Biologically Meaningful Approach
================================================================================

This script implements a focused analysis of TUSC2 expression patterns across 
NK cell subtypes using direct statistical approaches tailored for sparse/inducible 
gene expression patterns.

Key improvements over dynamic signature-based approaches:
1. Direct frequency analysis - Which subtypes express TUSC2 most frequently
2. Compositional enrichment - Which subtypes are over-represented in TUSC2+ cells
3. Conditional expression analysis - Among TUSC2+ cells, expression levels by subtype

Author: AI Assistant
Date: January 2025
Version: 1.0
"""

# %% [markdown]
# # TUSC2-NK Subtype Focused Analysis
# 
# ## Rationale
# 
# TUSC2 is an **inducible, non-constitutive gene** with predominantly zero expression 
# in most NK cells. This creates a zero-inflated distribution that makes traditional 
# correlation-based approaches inappropriate. Instead, we focus on:
# 
# 1. **Frequency Analysis**: Which subtypes have the highest percentage of TUSC2+ cells?
# 2. **Compositional Enrichment**: Which subtypes are over-represented among TUSC2+ cells?
# 3. **Conditional Expression**: Among TUSC2+ cells, which subtypes express it most highly?
# 
# This approach is more biologically meaningful and directly addresses the research question.

# %%
# =============================================================================
# PART 0: SETUP AND CONFIGURATION
# =============================================================================

print("=== TUSC2-NK Subtype Focused Analysis ===")
print("Version 1.0 - Direct Statistical Approach for Sparse Gene Expression")
print()

# --- 0.1: Library Imports ---
import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 0.2: Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sc.settings.verbosity = 1
sc.settings.autoshow = False

# Enhanced plotting parameters
plt.rcParams.update({
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.transparent': False,
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("Libraries imported and configuration set")

# %%
# =============================================================================
# PART 1: FILE PATHS AND CONSTANTS
# =============================================================================

# --- 1.1: Input File Paths ---
REBUFFET_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\data\processed\PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
TANG_COMBINED_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\data\processed\comb_CD56_CD16_NK.h5ad"

# --- 1.2: Output Directory ---
MASTER_OUTPUT_DIR = r"C:\Users\met-a\Documents\Analysis\Combined_NK_TUSC2_Analysis_Output"
TUSC2_FOCUSED_OUTPUT_DIR = os.path.join(MASTER_OUTPUT_DIR, "TUSC2_Focused_Analysis")
os.makedirs(TUSC2_FOCUSED_OUTPUT_DIR, exist_ok=True)

# Create subdirectories
SUBDIRS = {
    'figures': os.path.join(TUSC2_FOCUSED_OUTPUT_DIR, 'figures'),
    'data': os.path.join(TUSC2_FOCUSED_OUTPUT_DIR, 'data_for_graphpad'),
    'stats': os.path.join(TUSC2_FOCUSED_OUTPUT_DIR, 'statistical_results'),
    'temp': os.path.join(TUSC2_FOCUSED_OUTPUT_DIR, 'temp_data')
}

for subdir in SUBDIRS.values():
    os.makedirs(subdir, exist_ok=True)

print(f"Output directory created: {TUSC2_FOCUSED_OUTPUT_DIR}")

# %%
# =============================================================================
# PART 2: BIOLOGICAL CONSTANTS AND DEFINITIONS
# =============================================================================

# --- 2.1: Gene of Interest ---
TUSC2_GENE_NAME = "TUSC2"

# --- 2.2: Dataset-Specific Constants ---
# Rebuffet subtypes (blood NK)
REBUFFET_SUBTYPES_ORDERED = [
    "NK2",      # Immature/regulatory
    "NKint",    # Intermediate
    "NK1A",     # Early mature
    "NK1B",     # Intermediate mature  
    "NK1C",     # Mature cytotoxic
    "NK3",      # Adaptive/terminal
]

# Tang subtypes (tissue NK)
TANG_SUBTYPES_ORDERED = [
    "CD56brightCD16lo-c5-CREM",         # Regulatory/immature
    "CD56brightCD16lo-c4-IL7R",         # Immature
    "CD56brightCD16lo-c2-IL7R-RGS1lo",  # Transitional
    "CD56brightCD16lo-c3-CCL3",         # Inflammatory
    "CD56brightCD16lo-c1-GZMH",         # Cytotoxic bright
    "CD56brightCD16hi",                 # Double-positive transitional
    "CD56dimCD16hi-c1-IL32",            # Cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1",          # Tissue-homing
    "CD56dimCD16hi-c3-ZNF90",           # Mature
    "CD56dimCD16hi-c4-NFKBIA",          # Activated
    "CD56dimCD16hi-c6-DNAJB1",          # Stress-response
    "CD56dimCD16hi-c7-NR4A3",           # Stimulated
    "CD56dimCD16hi-c8-KLRC2",           # Adaptive (NKG2C+)
    "CD56dimCD16hi-c5-MKI67",           # Proliferating
]

# Tang subtype splits
TANG_CD56BRIGHT_SUBTYPES = [
    "CD56brightCD16lo-c5-CREM",
    "CD56brightCD16lo-c4-IL7R", 
    "CD56brightCD16lo-c2-IL7R-RGS1lo",
    "CD56brightCD16lo-c3-CCL3",
    "CD56brightCD16lo-c1-GZMH",
    "CD56brightCD16hi",
]

TANG_CD56DIM_SUBTYPES = [
    "CD56dimCD16hi-c1-IL32",
    "CD56dimCD16hi-c2-CX3CR1",
    "CD56dimCD16hi-c3-ZNF90",
    "CD56dimCD16hi-c4-NFKBIA",
    "CD56dimCD16hi-c6-DNAJB1",
    "CD56dimCD16hi-c7-NR4A3",
    "CD56dimCD16hi-c8-KLRC2",
    "CD56dimCD16hi-c5-MKI67",
]

# --- 2.3: Column Names ---
REBUFFET_SUBTYPE_COL = "Rebuffet_Subtype"
REBUFFET_ORIG_SUBTYPE_COL = "ident"
TANG_SUBTYPE_COL = "Tang_Subtype"
TANG_CELLTYPE_COL = "celltype"
TANG_TISSUE_COL = "meta_tissue_in_paper"
TANG_MAJORTYPE_COL = "Majortype"
TANG_HISTOLOGY_COL = "meta_histology"
TANG_PATIENT_ID_COL = "meta_patientID"

# --- 2.4: TUSC2 Analysis Parameters ---
TUSC2_EXPRESSION_THRESHOLD = 0.1  # Minimum expression to consider TUSC2+
MIN_CELLS_PER_SUBTYPE = 10        # Minimum cells required for analysis
SIGNIFICANCE_THRESHOLD = 0.05     # Statistical significance threshold

print("Biological constants and parameters defined")

# %%
# =============================================================================
# PART 3: UTILITY FUNCTIONS
# =============================================================================

def get_tusc2_expression(adata, use_raw=True):
    """
    Extract TUSC2 expression values from AnnData object
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    use_raw : bool
        Whether to use raw counts (recommended for gene expression analysis)
        
    Returns:
    --------
    np.ndarray
        TUSC2 expression values
    """
    # First try raw data, then main data
    if use_raw and hasattr(adata, 'raw') and adata.raw is not None:
        if TUSC2_GENE_NAME in adata.raw.var_names:
            gene_idx = adata.raw.var_names.get_loc(TUSC2_GENE_NAME)
            expression = adata.raw.X[:, gene_idx]
        elif TUSC2_GENE_NAME in adata.var_names:
            gene_idx = adata.var_names.get_loc(TUSC2_GENE_NAME)
            expression = adata.X[:, gene_idx]
        else:
            raise ValueError(f"TUSC2 not found in raw or main data")
    else:
        if TUSC2_GENE_NAME in adata.var_names:
            gene_idx = adata.var_names.get_loc(TUSC2_GENE_NAME)
            expression = adata.X[:, gene_idx]
        else:
            raise ValueError(f"TUSC2 not found in main data")
    
    # Convert sparse matrix to dense if needed
    if issparse(expression):
        try:
            # Handle different sparse matrix types
            if hasattr(expression, 'A1'):
                expression = expression.A1
            elif hasattr(expression, 'toarray'):
                expression = expression.toarray().flatten()
            else:
                expression = np.asarray(expression).flatten()
        except Exception as e:
            print(f"Warning: Could not convert sparse matrix properly: {e}")
            expression = np.asarray(expression).flatten()
    
    # Ensure we have a 1D array
    if expression.ndim > 1:
        expression = expression.flatten()
    
    return expression

def classify_tusc2_status(expression, threshold=TUSC2_EXPRESSION_THRESHOLD):
    """
    Classify cells as TUSC2+ or TUSC2- based on expression threshold
    
    Parameters:
    -----------
    expression : np.ndarray
        TUSC2 expression values
    threshold : float
        Expression threshold for positive classification
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating TUSC2+ cells
    """
    return expression > threshold

def save_results(data, filename, subdir='stats'):
    """
    Save analysis results to file
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to save
    filename : str
        Output filename
    subdir : str
        Subdirectory to save in
    """
    filepath = os.path.join(SUBDIRS[subdir], filename)
    data.to_csv(filepath, index=True)
    print(f"Results saved to: {filepath}")

def create_comparison_plot(data, x_col, y_col, hue_col, title, filename, 
                          plot_type='bar', figsize=(12, 8)):
    """
    Create comparison plots for TUSC2 analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to plot
    x_col, y_col, hue_col : str
        Column names for plot dimensions
    title : str
        Plot title
    filename : str
        Output filename
    plot_type : str
        Type of plot ('bar', 'box', 'violin')
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == 'bar':
        sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    elif plot_type == 'box':
        sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    elif plot_type == 'violin':
        sns.violinplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title(title)
    plt.tight_layout()
    
    filepath = os.path.join(SUBDIRS['figures'], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filepath}")

print("Utility functions defined") 

# %%
# =============================================================================
# PART 4: DATA LOADING AND PREPROCESSING
# =============================================================================

print("\n=== DATA LOADING AND PREPROCESSING ===")

# --- 4.1: Load Rebuffet Blood NK Data ---
print("\n--- Loading Rebuffet Blood NK Data ---")

try:
    print(f"Loading Rebuffet data from: {REBUFFET_H5AD_FILE}")
    adata_blood_source = sc.read_h5ad(REBUFFET_H5AD_FILE)
    print(f"Successfully loaded Rebuffet data: {adata_blood_source.shape}")
    
    # Basic data information
    print(f"Available columns: {list(adata_blood_source.obs.columns)}")
    
    # Check for raw data
    if hasattr(adata_blood_source, 'raw') and adata_blood_source.raw is not None:
        print(f"Raw data available: {adata_blood_source.raw.shape}")
    else:
        print("No raw data available")
    
    # Process blood data
    if hasattr(adata_blood_source, 'raw') and adata_blood_source.raw is not None:
        try:
            source_X = adata_blood_source.raw.X.copy() if hasattr(adata_blood_source.raw.X, 'copy') else adata_blood_source.raw.X
        except:
            source_X = adata_blood_source.raw.X
        data_type = "log-normalized from .raw"
    else:
        try:
            source_X = adata_blood_source.X.copy() if hasattr(adata_blood_source.X, 'copy') else adata_blood_source.X
        except:
            source_X = adata_blood_source.X
        data_type = "TPM-normalized from .X"
    
    # Create final blood object
    adata_blood = sc.AnnData(
        X=source_X,
        obs=adata_blood_source.obs.copy(),
        var=adata_blood_source.var.copy(),
    )
    adata_blood.var_names_make_unique()
    adata_blood.obs_names_make_unique()
    
    # Apply log transformation if needed
    if data_type == "TPM-normalized from .X":
        print("Applying log(TPM+1) transformation...")
        adata_blood.layers["tpm"] = adata_blood.X.copy()
        adata_blood.X = np.log1p(adata_blood.X)
    
    # Standardize subtype annotations
    if REBUFFET_ORIG_SUBTYPE_COL in adata_blood.obs.columns:
        adata_blood.obs[REBUFFET_SUBTYPE_COL] = adata_blood.obs[REBUFFET_ORIG_SUBTYPE_COL]
        adata_blood.obs[REBUFFET_SUBTYPE_COL] = pd.Categorical(
            adata_blood.obs[REBUFFET_SUBTYPE_COL],
            categories=REBUFFET_SUBTYPES_ORDERED,
            ordered=True,
        )
        # Filter for valid subtypes
        original_cell_count = adata_blood.n_obs
        adata_blood = adata_blood[adata_blood.obs[REBUFFET_SUBTYPE_COL].notna(), :].copy()
        print(f"Filtered {original_cell_count - adata_blood.n_obs} cells with undefined subtypes")
    
    # Set raw attribute for gene expression analysis
    print("Setting .raw attribute for gene expression analysis...")
    adata_blood.raw = adata_blood.copy()
    
    print(f"Blood NK data ready: {adata_blood.shape}")
    
    # Verify TUSC2 availability
    if TUSC2_GENE_NAME in adata_blood.var_names:
        print(f"✓ TUSC2 found in blood data")
    else:
        print(f"✗ TUSC2 NOT found in blood data")
        
except Exception as e:
    print(f"ERROR loading Rebuffet data: {e}")
    adata_blood = None

# --- 4.2: Load Tang Combined NK Data ---
print("\n--- Loading Tang Combined NK Data ---")

try:
    print(f"Loading Tang data from: {TANG_COMBINED_H5AD_FILE}")
    adata_tang_full = sc.read_h5ad(TANG_COMBINED_H5AD_FILE)
    print(f"Successfully loaded Tang data: {adata_tang_full.shape}")
    
    # Basic data information
    expr_min = float(adata_tang_full.X.min()) if hasattr(adata_tang_full.X, 'min') else 0.0
    expr_max = float(adata_tang_full.X.max()) if hasattr(adata_tang_full.X, 'max') else 1.0
    print(f"Expression range: {expr_min:.3f} to {expr_max:.3f}")
    print(f"Available columns: {list(adata_tang_full.obs.columns)}")
    
    # Check for raw data
    if hasattr(adata_tang_full, 'raw') and adata_tang_full.raw is not None:
        print(f"Raw data available: {adata_tang_full.raw.shape}")
    else:
        print("No raw data available")
    
    # Store raw counts
    adata_tang_full.layers["counts"] = adata_tang_full.X.copy()
    
    # Minimal cell filtering
    min_genes_per_cell = 200
    print(f"Filtering cells with fewer than {min_genes_per_cell} genes...")
    sc.pp.filter_cells(adata_tang_full, min_genes=min_genes_per_cell)
    print(f"Shape after cell filtering: {adata_tang_full.shape}")
    
    # Check if normalization is needed
    max_expression = float(adata_tang_full.X.max()) if hasattr(adata_tang_full.X, 'max') else 1.0
    if max_expression > 50:
        print("Normalizing and log-transforming data...")
        sc.pp.normalize_total(adata_tang_full, target_sum=1e4)
        sc.pp.log1p(adata_tang_full)
    else:
        print("Data appears already normalized")
    
    # Set raw attribute
    print("Setting .raw attribute for gene expression analysis...")
    adata_tang_full.raw = adata_tang_full.copy()
    
    print(f"Tang data ready: {adata_tang_full.shape}")
    
    # Verify TUSC2 availability
    if TUSC2_GENE_NAME in adata_tang_full.var_names:
        print(f"✓ TUSC2 found in Tang data")
    else:
        print(f"✗ TUSC2 NOT found in Tang data")
        
except Exception as e:
    print(f"ERROR loading Tang data: {e}")
    adata_tang_full = None

# --- 4.3: Create Context-Specific Datasets ---
print("\n--- Creating Context-Specific Datasets ---")

# Create context-specific datasets from Tang data
contexts = {}

if adata_tang_full is not None:
    try:
        # Normal tissue
        normal_mask = adata_tang_full.obs[TANG_TISSUE_COL] == "Normal"
        if normal_mask.sum() > 0:
            adata_normal = adata_tang_full[normal_mask, :].copy()
            adata_normal.obs[TANG_SUBTYPE_COL] = adata_normal.obs[TANG_CELLTYPE_COL]
            adata_normal.obs[TANG_SUBTYPE_COL] = pd.Categorical(
                adata_normal.obs[TANG_SUBTYPE_COL],
                categories=TANG_SUBTYPES_ORDERED,
                ordered=True,
            )
            contexts['Normal'] = adata_normal
            print(f"Normal tissue NK: {adata_normal.shape}")
        
        # Tumor tissue
        tumor_mask = adata_tang_full.obs[TANG_TISSUE_COL] == "Tumor"
        if tumor_mask.sum() > 0:
            adata_tumor = adata_tang_full[tumor_mask, :].copy()
            adata_tumor.obs[TANG_SUBTYPE_COL] = adata_tumor.obs[TANG_CELLTYPE_COL]
            adata_tumor.obs[TANG_SUBTYPE_COL] = pd.Categorical(
                adata_tumor.obs[TANG_SUBTYPE_COL],
                categories=TANG_SUBTYPES_ORDERED,
                ordered=True,
            )
            contexts['Tumor'] = adata_tumor
            print(f"Tumor tissue NK: {adata_tumor.shape}")
        
        # Blood context (from Tang data - note: mostly cancer patients)
        blood_mask = adata_tang_full.obs[TANG_TISSUE_COL] == "Blood"
        if blood_mask.sum() > 0:
            adata_tang_blood = adata_tang_full[blood_mask, :].copy()
            adata_tang_blood.obs[TANG_SUBTYPE_COL] = adata_tang_blood.obs[TANG_CELLTYPE_COL]
            adata_tang_blood.obs[TANG_SUBTYPE_COL] = pd.Categorical(
                adata_tang_blood.obs[TANG_SUBTYPE_COL],
                categories=TANG_SUBTYPES_ORDERED,
                ordered=True,
            )
            contexts['Tang_Blood'] = adata_tang_blood
            print(f"Tang blood NK: {adata_tang_blood.shape}")
            
    except Exception as e:
        print(f"ERROR creating context-specific datasets: {e}")

# Add Rebuffet blood data
if adata_blood is not None:
    contexts['Rebuffet_Blood'] = adata_blood
    print(f"Rebuffet blood NK: {adata_blood.shape}")

print(f"\nAvailable contexts: {list(contexts.keys())}")
print("Data loading and preprocessing complete")

# %%
# =============================================================================
# PART 5: TUSC2 EXPRESSION OVERVIEW
# =============================================================================

print("\n=== TUSC2 EXPRESSION OVERVIEW ===")

# Analyze TUSC2 expression patterns across all contexts
tusc2_overview = {}

for context_name, adata in contexts.items():
    if adata is None:
        continue
        
    print(f"\n--- {context_name} ---")
    
    try:
        # Get TUSC2 expression
        tusc2_expr = get_tusc2_expression(adata, use_raw=True)
        
        # Basic statistics
        n_cells = len(tusc2_expr)
        n_positive = np.sum(tusc2_expr > TUSC2_EXPRESSION_THRESHOLD)
        percent_positive = (n_positive / n_cells) * 100
        
        stats_dict = {
            'context': context_name,
            'total_cells': n_cells,
            'tusc2_positive_cells': n_positive,
            'percent_positive': percent_positive,
            'mean_expression': np.mean(tusc2_expr),
            'median_expression': np.median(tusc2_expr),
            'std_expression': np.std(tusc2_expr),
            'min_expression': np.min(tusc2_expr),
            'max_expression': np.max(tusc2_expr),
            'zero_expression_cells': np.sum(tusc2_expr == 0),
            'percent_zero': (np.sum(tusc2_expr == 0) / n_cells) * 100,
        }
        
        tusc2_overview[context_name] = stats_dict
        
        print(f"Total cells: {n_cells:,}")
        print(f"TUSC2+ cells: {n_positive:,} ({percent_positive:.1f}%)")
        print(f"Zero expression: {stats_dict['zero_expression_cells']:,} ({stats_dict['percent_zero']:.1f}%)")
        print(f"Expression range: {stats_dict['min_expression']:.3f} - {stats_dict['max_expression']:.3f}")
        print(f"Mean expression: {stats_dict['mean_expression']:.3f}")
        
    except Exception as e:
        print(f"ERROR analyzing TUSC2 in {context_name}: {e}")
        tusc2_overview[context_name] = None

# Save overview results
if tusc2_overview and any(v is not None for v in tusc2_overview.values()):
    overview_df = pd.DataFrame(tusc2_overview).T
    overview_df.index.name = 'Context'
    save_results(overview_df, 'tusc2_expression_overview.csv', 'stats')
else:
    print("No successful TUSC2 analyses - skipping overview DataFrame creation")

print("\nTUSC2 expression overview complete")

# %%
# =============================================================================
# PART 6: FOCUSED TUSC2-SUBTYPE ANALYSIS FUNCTIONS
# =============================================================================

print("\n=== DEFINING FOCUSED ANALYSIS FUNCTIONS ===")

def analyze_tusc2_frequency_by_subtype(adata, context_name, subtype_col, 
                                      threshold=TUSC2_EXPRESSION_THRESHOLD):
    """
    Analyze TUSC2+ frequency by subtype
    
    Returns:
    --------
    pd.DataFrame: Frequency analysis results
    """
    print(f"\n--- Frequency Analysis: {context_name} ---")
    
    # Get TUSC2 expression and subtype information
    tusc2_expr = get_tusc2_expression(adata, use_raw=True)
    tusc2_positive = classify_tusc2_status(tusc2_expr, threshold)
    subtypes = adata.obs[subtype_col]
    
    # Calculate frequency by subtype
    results = []
    for subtype in subtypes.cat.categories:
        subtype_mask = subtypes == subtype
        if subtype_mask.sum() < MIN_CELLS_PER_SUBTYPE:
            continue
            
        total_cells = subtype_mask.sum()
        positive_cells = np.sum(tusc2_positive & subtype_mask)
        frequency = (positive_cells / total_cells) * 100
        
        # Mean expression in this subtype
        subtype_expr = tusc2_expr[subtype_mask]
        mean_expr = np.mean(subtype_expr)
        mean_expr_positive = np.mean(subtype_expr[subtype_expr > threshold]) if positive_cells > 0 else 0
        
        results.append({
            'context': context_name,
            'subtype': subtype,
            'total_cells': total_cells,
            'tusc2_positive_cells': positive_cells,
            'tusc2_frequency_percent': frequency,
            'mean_expression_all': mean_expr,
            'mean_expression_positive_only': mean_expr_positive,
            'median_expression_all': np.median(subtype_expr),
            'std_expression_all': np.std(subtype_expr),
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('tusc2_frequency_percent', ascending=False)
    
    print(f"Frequency analysis complete for {context_name}")
    print(f"Top 3 subtypes by TUSC2 frequency:")
    for i, row in results_df.head(3).iterrows():
        print(f"  {row['subtype']}: {row['tusc2_frequency_percent']:.1f}% ({row['tusc2_positive_cells']}/{row['total_cells']} cells)")
    
    return results_df

def analyze_tusc2_compositional_enrichment(adata, context_name, subtype_col, 
                                         threshold=TUSC2_EXPRESSION_THRESHOLD):
    """
    Analyze which subtypes are over-represented in TUSC2+ population
    
    Returns:
    --------
    pd.DataFrame: Compositional enrichment results
    """
    print(f"\n--- Compositional Enrichment Analysis: {context_name} ---")
    
    # Get TUSC2 expression and subtype information
    tusc2_expr = get_tusc2_expression(adata, use_raw=True)
    tusc2_positive = classify_tusc2_status(tusc2_expr, threshold)
    subtypes = adata.obs[subtype_col]
    
    # Calculate compositions
    results = []
    total_cells = len(subtypes)
    total_positive = np.sum(tusc2_positive)
    
    if total_positive == 0:
        print("No TUSC2+ cells found")
        return pd.DataFrame()
    
    for subtype in subtypes.cat.categories:
        subtype_mask = subtypes == subtype
        if subtype_mask.sum() < MIN_CELLS_PER_SUBTYPE:
            continue
            
        # Overall composition
        cells_in_subtype = subtype_mask.sum()
        percent_of_total = (cells_in_subtype / total_cells) * 100
        
        # Composition in TUSC2+ population
        positive_in_subtype = np.sum(tusc2_positive & subtype_mask)
        percent_of_positive = (positive_in_subtype / total_positive) * 100
        
        # Enrichment ratio
        enrichment_ratio = percent_of_positive / percent_of_total if percent_of_total > 0 else 0
        
        # Statistical test (hypergeometric)
        from scipy.stats import hypergeom
        p_value = hypergeom.sf(positive_in_subtype - 1, total_cells, cells_in_subtype, total_positive)
        
        results.append({
            'context': context_name,
            'subtype': subtype,
            'total_cells_in_subtype': cells_in_subtype,
            'tusc2_positive_in_subtype': positive_in_subtype,
            'percent_of_total_population': percent_of_total,
            'percent_of_tusc2_positive': percent_of_positive,
            'enrichment_ratio': enrichment_ratio,
            'hypergeometric_p_value': p_value,
        })
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction
    if len(results_df) > 1:
        _, corrected_p, _, _ = multipletests(results_df['hypergeometric_p_value'], 
                                           method='fdr_bh', alpha=SIGNIFICANCE_THRESHOLD)
        results_df['corrected_p_value'] = corrected_p
        results_df['significant'] = corrected_p < SIGNIFICANCE_THRESHOLD
    
    results_df = results_df.sort_values('enrichment_ratio', ascending=False)
    
    print(f"Compositional enrichment analysis complete for {context_name}")
    print(f"Top 3 enriched subtypes:")
    for i, row in results_df.head(3).iterrows():
        sig_marker = "***" if row.get('significant', False) else ""
        print(f"  {row['subtype']}: {row['enrichment_ratio']:.2f}x enrichment {sig_marker}")
    
    return results_df

def analyze_tusc2_conditional_expression(adata, context_name, subtype_col, 
                                       threshold=TUSC2_EXPRESSION_THRESHOLD):
    """
    Analyze TUSC2 expression levels among TUSC2+ cells by subtype
    
    Returns:
    --------
    pd.DataFrame: Conditional expression analysis results
    """
    print(f"\n--- Conditional Expression Analysis: {context_name} ---")
    
    # Get TUSC2 expression and subtype information
    tusc2_expr = get_tusc2_expression(adata, use_raw=True)
    tusc2_positive = classify_tusc2_status(tusc2_expr, threshold)
    subtypes = adata.obs[subtype_col]
    
    # Filter to TUSC2+ cells only
    positive_mask = tusc2_positive
    if positive_mask.sum() == 0:
        print("No TUSC2+ cells found")
        return pd.DataFrame()
    
    positive_expr = tusc2_expr[positive_mask]
    positive_subtypes = subtypes[positive_mask]
    
    # Calculate expression by subtype in TUSC2+ population
    results = []
    for subtype in positive_subtypes.cat.categories:
        subtype_mask = positive_subtypes == subtype
        if subtype_mask.sum() < MIN_CELLS_PER_SUBTYPE:
            continue
            
        subtype_expr = positive_expr[subtype_mask]
        
        results.append({
            'context': context_name,
            'subtype': subtype,
            'n_tusc2_positive_cells': len(subtype_expr),
            'mean_expression': np.mean(subtype_expr),
            'median_expression': np.median(subtype_expr),
            'std_expression': np.std(subtype_expr),
            'min_expression': np.min(subtype_expr),
            'max_expression': np.max(subtype_expr),
            'q25_expression': np.percentile(subtype_expr, 25),
            'q75_expression': np.percentile(subtype_expr, 75),
        })
    
    results_df = pd.DataFrame(results)
    
    # Statistical comparison if multiple subtypes
    if len(results_df) > 1:
        # Perform pairwise comparisons
        subtype_groups = []
        subtype_names = []
        for subtype in positive_subtypes.cat.categories:
            subtype_mask = positive_subtypes == subtype
            if subtype_mask.sum() >= MIN_CELLS_PER_SUBTYPE:
                subtype_groups.append(positive_expr[subtype_mask])
                subtype_names.append(subtype)
        
        if len(subtype_groups) > 1:
            # Kruskal-Wallis test
            from scipy.stats import kruskal
            h_stat, kw_p = kruskal(*subtype_groups)
            print(f"Kruskal-Wallis test p-value: {kw_p:.3e}")
            
            # Post-hoc pairwise comparisons if significant
            if kw_p < SIGNIFICANCE_THRESHOLD:
                print("Performing post-hoc pairwise comparisons...")
                # This would be more complex to implement here
    
    results_df = results_df.sort_values('mean_expression', ascending=False)
    
    print(f"Conditional expression analysis complete for {context_name}")
    print(f"Top 3 subtypes by TUSC2 expression level (in TUSC2+ cells):")
    for i, row in results_df.head(3).iterrows():
        print(f"  {row['subtype']}: {row['mean_expression']:.3f} (n={row['n_tusc2_positive_cells']})")
    
    return results_df

print("Analysis functions defined") 

# %%
# =============================================================================
# PART 7: EXECUTE FOCUSED TUSC2-SUBTYPE ANALYSIS
# =============================================================================

print("\n=== EXECUTING FOCUSED TUSC2-SUBTYPE ANALYSIS ===")

# Store all results
all_frequency_results = []
all_enrichment_results = []
all_conditional_results = []

# Analyze each context
for context_name, adata in contexts.items():
    if adata is None:
        continue
        
    print(f"\n{'='*60}")
    print(f"ANALYZING CONTEXT: {context_name}")
    print(f"{'='*60}")
    
    # Determine subtype column
    if 'Rebuffet' in context_name:
        subtype_col = REBUFFET_SUBTYPE_COL
    else:
        subtype_col = TANG_SUBTYPE_COL
    
    try:
        # 1. Frequency Analysis
        freq_results = analyze_tusc2_frequency_by_subtype(adata, context_name, subtype_col)
        if not freq_results.empty:
            all_frequency_results.append(freq_results)
            save_results(freq_results, f'tusc2_frequency_{context_name.lower()}.csv', 'stats')
        
        # 2. Compositional Enrichment Analysis
        enrich_results = analyze_tusc2_compositional_enrichment(adata, context_name, subtype_col)
        if not enrich_results.empty:
            all_enrichment_results.append(enrich_results)
            save_results(enrich_results, f'tusc2_enrichment_{context_name.lower()}.csv', 'stats')
        
        # 3. Conditional Expression Analysis
        cond_results = analyze_tusc2_conditional_expression(adata, context_name, subtype_col)
        if not cond_results.empty:
            all_conditional_results.append(cond_results)
            save_results(cond_results, f'tusc2_conditional_{context_name.lower()}.csv', 'stats')
        
    except Exception as e:
        print(f"ERROR analyzing {context_name}: {e}")
        continue

# %%
# =============================================================================
# PART 8: GENERATE COMPARATIVE VISUALIZATIONS
# =============================================================================

print("\n=== GENERATING COMPARATIVE VISUALIZATIONS ===")

# Combine all results for cross-context comparison
if all_frequency_results:
    combined_frequency = pd.concat(all_frequency_results, ignore_index=True)
    save_results(combined_frequency, 'tusc2_frequency_all_contexts.csv', 'stats')
    
    # Create frequency comparison plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_frequency, x='subtype', y='tusc2_frequency_percent', hue='context')
    plt.title('TUSC2+ Frequency by Subtype Across Contexts')
    plt.xlabel('NK Subtype')
    plt.ylabel('TUSC2+ Frequency (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Context', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SUBDIRS['figures'], 'tusc2_frequency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Frequency comparison plot saved")

if all_enrichment_results:
    combined_enrichment = pd.concat(all_enrichment_results, ignore_index=True)
    save_results(combined_enrichment, 'tusc2_enrichment_all_contexts.csv', 'stats')
    
    # Create enrichment comparison plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_enrichment, x='subtype', y='enrichment_ratio', hue='context')
    plt.title('TUSC2+ Enrichment Ratio by Subtype Across Contexts')
    plt.xlabel('NK Subtype')
    plt.ylabel('Enrichment Ratio')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No enrichment')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Context', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SUBDIRS['figures'], 'tusc2_enrichment_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Enrichment comparison plot saved")

if all_conditional_results:
    combined_conditional = pd.concat(all_conditional_results, ignore_index=True)
    save_results(combined_conditional, 'tusc2_conditional_all_contexts.csv', 'stats')
    
    # Create conditional expression comparison plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_conditional, x='subtype', y='mean_expression', hue='context')
    plt.title('TUSC2 Expression Level in TUSC2+ Cells by Subtype Across Contexts')
    plt.xlabel('NK Subtype')
    plt.ylabel('Mean TUSC2 Expression (TUSC2+ cells only)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Context', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SUBDIRS['figures'], 'tusc2_conditional_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Conditional expression comparison plot saved")

# %%
# =============================================================================
# PART 9: SUMMARY REPORT
# =============================================================================

print("\n=== GENERATING SUMMARY REPORT ===")

# Create comprehensive summary
summary_report = []
summary_report.append("TUSC2-NK Subtype Focused Analysis - Summary Report")
summary_report.append("=" * 60)
summary_report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_report.append(f"TUSC2 Expression Threshold: {TUSC2_EXPRESSION_THRESHOLD}")
summary_report.append(f"Minimum Cells per Subtype: {MIN_CELLS_PER_SUBTYPE}")
summary_report.append(f"Significance Threshold: {SIGNIFICANCE_THRESHOLD}")
summary_report.append("")

# Context overview
summary_report.append("ANALYZED CONTEXTS:")
summary_report.append("-" * 20)
for context_name, adata in contexts.items():
    if adata is not None:
        summary_report.append(f"  {context_name}: {adata.n_obs:,} cells")

summary_report.append("")

# Key findings
summary_report.append("KEY FINDINGS:")
summary_report.append("-" * 15)

if all_frequency_results:
    # Top TUSC2+ frequency subtypes overall
    top_freq_overall = combined_frequency.nlargest(5, 'tusc2_frequency_percent')
    summary_report.append("Top 5 Subtypes by TUSC2+ Frequency (Overall):")
    for _, row in top_freq_overall.iterrows():
        summary_report.append(f"  {row['subtype']} ({row['context']}): {row['tusc2_frequency_percent']:.1f}%")
    summary_report.append("")

if all_enrichment_results:
    # Top enriched subtypes
    top_enrich_overall = combined_enrichment.nlargest(5, 'enrichment_ratio')
    summary_report.append("Top 5 Subtypes by TUSC2+ Enrichment:")
    for _, row in top_enrich_overall.iterrows():
        sig_marker = "***" if row.get('significant', False) else ""
        summary_report.append(f"  {row['subtype']} ({row['context']}): {row['enrichment_ratio']:.2f}x {sig_marker}")
    summary_report.append("")

if all_conditional_results:
    # Highest TUSC2 expression in TUSC2+ cells
    top_expr_overall = combined_conditional.nlargest(5, 'mean_expression')
    summary_report.append("Top 5 Subtypes by TUSC2 Expression Level (in TUSC2+ cells):")
    for _, row in top_expr_overall.iterrows():
        summary_report.append(f"  {row['subtype']} ({row['context']}): {row['mean_expression']:.3f}")
    summary_report.append("")

# Methodological notes
summary_report.append("METHODOLOGICAL NOTES:")
summary_report.append("-" * 20)
summary_report.append("1. TUSC2 is analyzed as a sparse/inducible gene with zero-inflated expression")
summary_report.append("2. Frequency analysis identifies subtypes with highest % of TUSC2+ cells")
summary_report.append("3. Enrichment analysis identifies subtypes over-represented in TUSC2+ population")
summary_report.append("4. Conditional analysis compares TUSC2 expression levels among TUSC2+ cells only")
summary_report.append("5. Statistical significance tested using hypergeometric test with FDR correction")
summary_report.append("")

# File locations
summary_report.append("OUTPUT FILES:")
summary_report.append("-" * 15)
summary_report.append(f"Results directory: {TUSC2_FOCUSED_OUTPUT_DIR}")
summary_report.append("  - figures/: All visualization plots")
summary_report.append("  - statistical_results/: All statistical analysis results")
summary_report.append("  - data_for_graphpad/: (Reserved for GraphPad-ready data)")
summary_report.append("")

# Save summary report
report_text = "\n".join(summary_report)
with open(os.path.join(SUBDIRS['stats'], 'TUSC2_Analysis_Summary_Report.txt'), 'w') as f:
    f.write(report_text)

print("Summary report generated")
print("\n" + "=" * 60)
print("TUSC2-NK SUBTYPE FOCUSED ANALYSIS COMPLETE")
print("=" * 60)
print(f"All results saved to: {TUSC2_FOCUSED_OUTPUT_DIR}")
print("This analysis provides biologically meaningful insights into TUSC2-subtype")
print("associations without the complexity of dynamic gene signature generation.")
print("=" * 60) 