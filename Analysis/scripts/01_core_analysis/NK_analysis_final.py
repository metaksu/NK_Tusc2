# %% [markdown]
# # Streamlined NK Cell TUSC2 Analysis - Final Version
#
# **Outputs:**
# 1. P2_1_UMAP_by_Subtype_with_Unassigned_Blood_v3_relaxed
# 2. P2_3b_Heatmap_FuncProfile_Blood_v5
# 3. P2_3b_Heatmap_FuncProfile_NormalTissue_v5
# 4. P2_3b_Heatmap_FuncProfile_TumorTissue_v5
# 5. P4_2_Heatmap_TUSC2_Impact_CoreFunctionalCapacity_v4_final
# 6. P4_3_Heatmap_TUSC2_Impact_on_Subtype_Programs_v5_final

# %%
# PART 0: Global Setup
# Section 0.1: Library Imports

print("--- Streamlined NK Analysis: Final Version ---")
print("--- 0.1.1: Importing Libraries ---")

# Standard Python Libraries
import os
import re
import sys
import warnings
from pathlib import Path

# Data Science & Numerical Libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix

# Single-Cell Analysis Libraries
import scanpy as sc

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Specific Statistical/Analysis Tools
from statsmodels.stats.multitest import multipletests

print("All libraries imported.\n")

# --- 0.1.2: Plotting Aesthetics & Scanpy Settings ---
print("--- 0.1.2: Configuring Plotting Aesthetics ---")

FIGURE_FORMAT = "png"
FIGURE_DPI = 300
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
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
})

sc.settings.autoshow = False
sc.settings.verbosity = 2

print("Plotting aesthetics configured.")
print("--- End of Section 0.1 ---")

# %%
# Section 0.2: File Paths & Output Directory Structure

print("--- 0.2: Defining File Paths & Output Directories ---")

# === INPUT FILES ===
REBUFFET_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\outputs\harmony_batch_correction\PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad"
TANG_COMBINED_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\data\processed\comb_CD56_CD16_NK.h5ad"

# === OUTPUT DIRECTORY - NEW CLEAN STRUCTURE ===
MASTER_OUTPUT_DIR = r"C:\Users\met-a\Documents\Analysis\Final_NK_TUSC2_Analysis"

# Logical folder structure for 6 final outputs:
# Final_NK_TUSC2_Analysis/
# ├── 1_NK_Subtype_Characterization/
# │   ├── figures/          <- UMAP + Functional heatmaps (all contexts)
# │   └── data/             <- Corresponding CSVs for GraphPad
# ├── 2_TUSC2_Impact_Analysis/
# │   ├── figures/          <- TUSC2 impact heatmaps  
# │   └── data/             <- Corresponding CSVs
# └── _intermediate/        <- Stats files used between sections
#     └── tusc2_stats/

OUTPUT_SUBDIRS = {
    # Part 1: NK Subtype Characterization (UMAP + Functional Profiles)
    "characterization_figs": os.path.join(MASTER_OUTPUT_DIR, "1_NK_Subtype_Characterization", "figures"),
    "characterization_data": os.path.join(MASTER_OUTPUT_DIR, "1_NK_Subtype_Characterization", "data"),
    
    # Part 2: TUSC2 Impact Analysis
    "tusc2_impact_figs": os.path.join(MASTER_OUTPUT_DIR, "2_TUSC2_Impact_Analysis", "figures"),
    "tusc2_impact_data": os.path.join(MASTER_OUTPUT_DIR, "2_TUSC2_Impact_Analysis", "data"),
    
    # Intermediate files (stats used between sections)
    "intermediate_stats": os.path.join(MASTER_OUTPUT_DIR, "_intermediate", "tusc2_stats"),
}

# Create all directories
os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
for key, path in OUTPUT_SUBDIRS.items():
    os.makedirs(path, exist_ok=True)
    print(f"    Created: {path}")

sc.settings.figdir = MASTER_OUTPUT_DIR
print("--- End of Section 0.2 ---")

# %%
# Section 0.3: Core Biological Definitions

print("--- 0.3: Defining Core Biological Definitions ---")

TUSC2_GENE_NAME = "TUSC2"

# Rebuffet NK Subtypes
REBUFFET_SUBTYPES_ORDERED = ["NK2", "NKint", "NK1A", "NK1B", "NK1C", "NK3"]
REBUFFET_SUBTYPE_COL = "Rebuffet_Subtype"
REBUFFET_ORIG_SUBTYPE_COL = "ident"

# Tang metadata columns
TANG_TISSUE_COL = "meta_tissue_in_paper"
TANG_MAJORTYPE_COL = "Majortype"
TANG_CELLTYPE_COL = "celltype"
METADATA_TISSUE_COLUMN_GSE212890 = TANG_TISSUE_COL

def get_subtype_column(adata_obj):
    if adata_obj is None or adata_obj.n_obs == 0:
        return None
    return REBUFFET_SUBTYPE_COL

def get_subtype_categories(adata_obj):
    return REBUFFET_SUBTYPES_ORDERED + ["Unassigned"]

# Functional Gene Sets
Activating_Receptors_Gene_Set = [
    "IL2RB", "IL18R1", "IL18RAP", "NCR1", "NCR2", "NCR3", "KLRK1", "FCGR3A",
    "CD226", "KLRC2", "CD244", "SLAMF6", "SLAMF7", "CD160", "KLRF1",
    "KIR2DS1", "KIR2DS2", "KIR2DS4", "KIR3DS1", "ITGAL",
]
Inhibitory_Receptors_Gene_Set = [
    "KLRC1", "KIR2DL1", "KIR2DL2", "KIR2DL3", "KIR3DL1", "KIR3DL2",
    "LILRB1", "PDCD1", "TIGIT", "CTLA4", "HAVCR2", "LAG3",
    "SIGLEC7", "SIGLEC9", "KLRG1", "CD300A", "LAIR1", "CEACAM1",
]
Cytotoxicity_Machinery_Gene_Set = [
    "PRF1", "GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "NKG7", "GNLY",
    "SERPINB9", "SRGN", "FASLG", "TNFSF10", "LAMP1", "CTSC", "CTSW",
]
Cytokine_Chemokine_Gene_Set = [
    "IFNG", "TNF", "LTA", "CSF2", "IL10", "IL32", "XCL1", "XCL2",
    "CCL3", "CCL4", "CCL5", "CXCL8", "CXCL10",
]
Exhaustion_Suppression_Gene_Set = [
    "PDCD1", "HAVCR2", "LAG3", "TIGIT", "KLRC1", "KLRG1", "CD96",
    "LILRB1", "ENTPD1", "TOX", "EGR2", "MAF", "PRDM1", "HSPA1A", "DNAJB1",
]

FUNCTIONAL_GENE_SETS = {
    "Activating_Receptors": Activating_Receptors_Gene_Set,
    "Inhibitory_Receptors": Inhibitory_Receptors_Gene_Set,
    "Cytotoxicity_Machinery": Cytotoxicity_Machinery_Gene_Set,
    "Cytokine_Chemokine_Production": Cytokine_Chemokine_Gene_Set,
    "Exhaustion_Suppression_Markers": Exhaustion_Suppression_Gene_Set,
}

# --- Load Metabolic Signatures from Hallmark Gene Sets ---
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
        print(f"        WARNING: Could not find {filepath}")
        return None

print("  Loading metabolic signatures from Hallmark gene sets...")
HALLMARK_GENE_SET_DIR = r"data\raw\gene_sets"

hallmark_glycolysis = load_hallmark_geneset(
    os.path.join(HALLMARK_GENE_SET_DIR, "HALLMARK_GLYCOLYSIS.v2025.1.Hs.grp")
)
hallmark_oxphos = load_hallmark_geneset(
    os.path.join(HALLMARK_GENE_SET_DIR, "HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2025.1.Hs.grp")
)
hallmark_fatty_acid = load_hallmark_geneset(
    os.path.join(HALLMARK_GENE_SET_DIR, "HALLMARK_FATTY_ACID_METABOLISM.v2025.1.Hs.grp")
)

# Store full Hallmark gene sets - will filter to expressed genes after data is loaded
HALLMARK_GENE_SETS_FULL = {}
if hallmark_glycolysis is not None:
    HALLMARK_GENE_SETS_FULL["Glycolysis"] = hallmark_glycolysis
    print(f"    Loaded Glycolysis: {len(hallmark_glycolysis)} genes (will filter to expressed)")
    
if hallmark_oxphos is not None:
    HALLMARK_GENE_SETS_FULL["Oxidative_Phosphorylation"] = hallmark_oxphos
    print(f"    Loaded Oxidative Phosphorylation: {len(hallmark_oxphos)} genes (will filter to expressed)")
    
if hallmark_fatty_acid is not None:
    HALLMARK_GENE_SETS_FULL["Fatty_Acid_Metabolism"] = hallmark_fatty_acid
    print(f"    Loaded Fatty Acid Metabolism: {len(hallmark_fatty_acid)} genes (will filter to expressed)")


def filter_hallmark_to_expressed(adata_ref, hallmark_sets, min_pct_cells=0.01):
    """
    Filter Hallmark gene sets to genes expressed in the reference dataset.
    
    This is a non-circular approach that simply removes genes not detected
    in the dataset, while preserving the full pathway character.
    
    Parameters
    ----------
    adata_ref : AnnData
        Reference dataset (e.g., blood NK cells)
    hallmark_sets : dict
        Dictionary of {set_name: gene_list}
    min_pct_cells : float
        Minimum fraction of cells a gene must be expressed in (default 1%)
    
    Returns
    -------
    dict : Filtered gene sets
    """
    filtered_sets = {}
    
    for set_name, full_gene_list in hallmark_sets.items():
        # Get genes available in the dataset
        available_genes = [g for g in full_gene_list if g in adata_ref.raw.var_names]
        
        if len(available_genes) == 0:
            print(f"      WARNING: No genes from {set_name} found in dataset")
            filtered_sets[set_name] = []
            continue
        
        # Further filter to genes expressed in at least min_pct_cells
        expressed_genes = []
        for gene in available_genes:
            gene_idx = list(adata_ref.raw.var_names).index(gene)
            gene_expr = adata_ref.raw.X[:, gene_idx]
            if hasattr(gene_expr, 'toarray'):
                gene_expr = gene_expr.toarray().flatten()
            pct_expressing = np.sum(gene_expr > 0) / len(gene_expr)
            if pct_expressing >= min_pct_cells:
                expressed_genes.append(gene)
        
        filtered_sets[set_name] = expressed_genes
        print(f"      {set_name}: {len(full_gene_list)} → {len(expressed_genes)} genes (expressed in ≥{min_pct_cells*100:.0f}% cells)")
    
    return filtered_sets

# Gene patterns to exclude from DEG analysis
GENE_PATTERNS_TO_EXCLUDE = [
    r"^RPS[0-9L]", r"^RPL[0-9L]", r"^RPLP[0-9]$", r"^RPSA$", r"^MT-",
    r"^ACT[BGINR]", r"ACTG1", r"^MYL[0-9]", r"^TPT1$", r"^FTL$", r"^FTH1$",
    r"^B2M$", r"^(HSP90|HSPA|HSPB|HSPD|HSPE|HSPH)[A-Z0-9]+",
    r"^EEF[12][A-Z0-9]*", r"^GAPDH$", r"^MALAT1$", r"^NEAT1$",
]

MIN_GENES_FOR_SCORING = 5
TUSC2_EXPRESSION_THRESHOLD_BINARY = 0.1
TUSC2_BINARY_GROUP_COL = f"{TUSC2_GENE_NAME}_Binary_Group"
TUSC2_BINARY_CATEGORIES = [f"{TUSC2_GENE_NAME}_Not_Expressed", f"{TUSC2_GENE_NAME}_Expressed"]

# Color Palettes
SUBTYPE_COLOR_PALETTE = {
    "NK2": "#1f77b4", "NKint": "#ff7f0e", "NK1A": "#2ca02c",
    "NK1B": "#d62728", "NK1C": "#9467bd", "NK3": "#8c564b", "Unassigned": "#bdbdbd",
}
COMBINED_SUBTYPE_COLOR_PALETTE = SUBTYPE_COLOR_PALETTE.copy()

GENE_NAME_MAPPING = {
    "CD16": "FCGR3A", "CD56": "NCAM1", "CD25": "IL2RA", "CD57": "B3GAT1",
    "CD137": "TNFRSF9", "PD1": "PDCD1", "TIM3": "HAVCR2", "CD49A": "ITGA1",
    "CD103": "ITGAE", "CD49B": "ITGA2", "CD62L": "SELL", "NKG2C": "KLRC2",
}

print("--- End of Section 0.3 ---")

# %%
# Section 0.4: Utility Functions

print("--- 0.4: Defining Utility Functions ---")

# Heatmap defaults
HEATMAP_CBAR_SHRINK = 0.6
HEATMAP_LINEWIDTHS = 0.5
HEATMAP_ANNOT_SIZE = 11
HEATMAP_ANNOT_WEIGHT = "bold"
HEATMAP_SQUARE_CELLS = True


def save_figure_and_data(fig_object, data_df_for_graphpad, plot_basename, figure_subdir,
                         data_subdir, fig_format_override=None, fig_dpi_override=None, close_fig=True):
    """Saves a figure and an optional DataFrame."""
    current_fig_format = (fig_format_override if fig_format_override else FIGURE_FORMAT).lstrip(".")
    current_fig_dpi = fig_dpi_override if fig_dpi_override else FIGURE_DPI

    if figure_subdir:
        os.makedirs(figure_subdir, exist_ok=True)
    if data_subdir:
        os.makedirs(data_subdir, exist_ok=True)

    if fig_object and figure_subdir:
        plot_path = os.path.join(figure_subdir, f"{plot_basename}.{current_fig_format}")
        try:
            fig_object.savefig(plot_path, dpi=current_fig_dpi, format=current_fig_format, bbox_inches="tight")
            print(f"  SUCCESS: Plot saved to {plot_path}")
        except Exception as e:
            print(f"  ERROR saving plot {plot_path}: {e}")
        finally:
            if close_fig:
                plt.close(fig_object)

    if data_df_for_graphpad is not None and data_subdir is not None:
        data_path = os.path.join(data_subdir, f"{plot_basename}_data.csv")
        if isinstance(data_df_for_graphpad, pd.DataFrame):
            try:
                data_df_for_graphpad.to_csv(data_path, index=False)
                print(f"  SUCCESS: Data saved to {data_path}")
            except Exception as e:
                print(f"  ERROR saving data {data_path}: {e}")


def export_raw_cell_data_for_graphpad(adata_view, score_columns, group_column, data_dir,
                                       filename_base, include_cell_id=True, format_type="long"):
    """
    Exports raw cell-level data for GraphPad Prism statistical analysis.
    
    Parameters
    ----------
    adata_view : AnnData
        The AnnData object containing cell data.
    score_columns : list
        List of column names in obs to export (e.g., signature scores).
    group_column : str
        Column name for grouping variable (e.g., 'Assigned_NK_Subtype' or 'TUSC2_Binary_Group').
    data_dir : str
        Directory to save the CSV file.
    filename_base : str
        Base name for the output file (without extension).
    include_cell_id : bool
        Whether to include cell barcodes in output.
    format_type : str
        'long' - One row per cell-score pair (for multi-variable comparisons)
        'wide' - One row per cell, scores as columns (for simple group comparisons)
    
    Returns
    -------
    pd.DataFrame : The exported DataFrame
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Validate columns
    valid_score_cols = [col for col in score_columns if col in adata_view.obs.columns]
    if not valid_score_cols:
        print(f"    WARNING: No valid score columns found for export")
        return None
    
    if group_column not in adata_view.obs.columns:
        print(f"    WARNING: Group column '{group_column}' not found")
        return None
    
    if format_type == "wide":
        # Wide format: One row per cell, columns for each score + group
        cols_to_export = [group_column] + valid_score_cols
        if include_cell_id:
            export_df = adata_view.obs[cols_to_export].copy()
            export_df.insert(0, "Cell_ID", adata_view.obs_names)
        else:
            export_df = adata_view.obs[cols_to_export].copy().reset_index(drop=True)
        
        # Clean column names for GraphPad
        export_df.columns = [col.replace("_Score", "") for col in export_df.columns]
        
    elif format_type == "long":
        # Long format: Melt scores into separate rows for stacked analyses
        wide_df = adata_view.obs[[group_column] + valid_score_cols].copy()
        wide_df["Cell_ID"] = adata_view.obs_names
        
        export_df = pd.melt(
            wide_df,
            id_vars=["Cell_ID", group_column] if include_cell_id else [group_column],
            value_vars=valid_score_cols,
            var_name="Signature",
            value_name="Score"
        )
        export_df["Signature"] = export_df["Signature"].str.replace("_Score", "")
    
    else:
        print(f"    WARNING: Unknown format_type '{format_type}'")
        return None
    
    # Export
    output_path = os.path.join(data_dir, f"{filename_base}_raw_data.csv")
    try:
        export_df.to_csv(output_path, index=False)
        print(f"    SUCCESS: Raw cell data exported to {output_path} ({len(export_df)} rows)")
    except Exception as e:
        print(f"    ERROR exporting raw data: {e}")
        return None
    
    return export_df


def export_grouped_comparison_data(adata_view, score_column, group_column, data_dir,
                                    filename_base, max_cells_per_group=None):
    """
    Exports data in GraphPad column format for two-group comparisons.
    
    Creates a CSV where each column represents a group, with individual cell scores
    as rows. This is the ideal format for GraphPad t-tests or Mann-Whitney tests.
    
    Parameters
    ----------
    adata_view : AnnData
        The AnnData object.
    score_column : str
        The score column to compare between groups.
    group_column : str
        Column containing group assignments.
    data_dir : str
        Output directory.
    filename_base : str
        Base filename.
    max_cells_per_group : int, optional
        If set, randomly sample this many cells per group (for very large datasets).
    
    Returns
    -------
    pd.DataFrame : The pivoted DataFrame (groups as columns)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if score_column not in adata_view.obs.columns or group_column not in adata_view.obs.columns:
        print(f"    WARNING: Required columns not found for grouped export")
        return None
    
    # Get unique groups
    groups = adata_view.obs[group_column].dropna().unique()
    
    # Create dict of score arrays per group
    group_data = {}
    for grp in groups:
        mask = adata_view.obs[group_column] == grp
        scores = adata_view.obs.loc[mask, score_column].dropna().values
        
        if max_cells_per_group and len(scores) > max_cells_per_group:
            rng = np.random.default_rng(42)
            scores = rng.choice(scores, size=max_cells_per_group, replace=False)
        
        group_data[str(grp)] = pd.Series(scores)
    
    # Combine into DataFrame (unequal lengths handled by pandas)
    export_df = pd.DataFrame(group_data)
    
    # Export
    output_path = os.path.join(data_dir, f"{filename_base}_grouped.csv")
    try:
        export_df.to_csv(output_path, index=False)
        print(f"    SUCCESS: Grouped data exported to {output_path}")
    except Exception as e:
        print(f"    ERROR exporting grouped data: {e}")
        return None
    
    return export_df


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


def calculate_heatmap_layout(data_frame, min_width=10, min_height=6, cell_width=1.2,
                             cell_height=0.8, label_padding=2.0, base_left_margin=0.15):
    """Calculate optimal figure size and layout parameters for heatmaps."""
    n_rows, n_cols = data_frame.shape
    max_label_length = max(len(str(label)) for label in data_frame.index)

    if max_label_length <= 15:
        char_width = 0.12
    elif max_label_length <= 30:
        char_width = 0.14
    elif max_label_length <= 50:
        char_width = 0.16
    else:
        char_width = 0.18

    heatmap_content_width = n_cols * cell_width
    base_label_width = max_label_length * char_width
    adaptive_padding = max(label_padding, min(label_padding * 1.5, base_label_width * 0.2))
    label_width_needed = max(2.0, base_label_width + adaptive_padding)
    min_heatmap_width = max(6.0, n_cols * 1.0)
    figure_width = max(min_width, heatmap_content_width + label_width_needed, min_heatmap_width + label_width_needed)

    heatmap_content_height = n_rows * cell_height
    title_space = 2.5
    figure_height = max(min_height, heatmap_content_height + title_space)

    left_margin_needed = label_width_needed / figure_width
    if max_label_length > 30:
        max_left_margin = 0.55
    elif max_label_length > 20:
        max_left_margin = 0.50
    else:
        max_left_margin = 0.45

    left_margin = max(base_left_margin, min(left_margin_needed, max_left_margin))

    if left_margin > 0.50:
        additional_width = (left_margin - 0.50) * figure_width
        figure_width += additional_width
        left_margin = label_width_needed / figure_width

    return figure_width, figure_height, left_margin


def is_gene_to_exclude_util(gene_name, patterns=GENE_PATTERNS_TO_EXCLUDE):
    if not isinstance(gene_name, str):
        return False
    for pattern in patterns:
        if re.match(pattern, gene_name, re.IGNORECASE):
            return True
    return False


def create_filename(base_name, context_name=None, tusc2_group=None, gene_set_name=None,
                    plot_type_suffix=None, version="v1", ext=None):
    """Creates a descriptive filename."""
    parts = [base_name]
    if context_name:
        parts.append(context_name.replace(" ", "").replace("(", "").replace(")", "").replace("/", "_"))
    if tusc2_group:
        parts.append(tusc2_group.replace(" ", "").replace(TUSC2_GENE_NAME + "_", "").replace(TUSC2_GENE_NAME, "TUSC2val"))
    if gene_set_name:
        parts.append(gene_set_name.replace(" ", "_").replace("/", "_").replace(":", "_"))
    if plot_type_suffix:
        parts.append(plot_type_suffix)
    if version:
        parts.append(version)

    cleaned_parts = [str(part) for part in parts if part is not None and str(part)]
    base_filename_out = "_".join(cleaned_parts)

    if ext:
        return f"{base_filename_out}.{ext.lstrip('.')}"
    return base_filename_out


def map_gene_names(gene_list, available_genes):
    """Map gene names using the mapping dictionary and filter for available genes."""
    mapped_genes = []
    for gene in gene_list:
        if gene in available_genes:
            mapped_genes.append(gene)
        elif gene in GENE_NAME_MAPPING and GENE_NAME_MAPPING[gene] in available_genes:
            mapped_genes.append(GENE_NAME_MAPPING[gene])
    return mapped_genes


def score_genes_aucell(adata, gene_list, score_name, use_raw=True, normalize=True, auc_max_rank=0.05):
    """Calculate proper AUCell scores for a gene signature."""
    if use_raw and adata.raw is not None:
        exp_mtx = adata.raw.X
        gene_names = adata.raw.var_names
    else:
        exp_mtx = adata.X
        gene_names = adata.var_names

    if hasattr(exp_mtx, 'toarray'):
        exp_mtx = exp_mtx.toarray()

    available_genes = list(set(gene_list) & set(gene_names))

    if len(available_genes) == 0:
        adata.obs[score_name] = 0.0
        return

    gene_indices = [i for i, gene in enumerate(gene_names) if gene in available_genes]
    signature_genes_set = set(gene_indices)

    auc_scores = []
    n_genes_total = len(gene_names)
    n_sig_genes = len(gene_indices)
    max_rank = int(n_genes_total * auc_max_rank)

    if max_rank < 1:
        max_rank = min(50, n_genes_total)

    for cell_idx in range(exp_mtx.shape[0]):
        cell_expr = exp_mtx[cell_idx, :]
        gene_ranks = np.argsort(-cell_expr)

        recovery_curve = []
        signature_genes_found = 0

        for rank_pos in range(min(max_rank, len(gene_ranks))):
            gene_idx = gene_ranks[rank_pos]
            if gene_idx in signature_genes_set:
                signature_genes_found += 1
            recovery_rate = signature_genes_found / n_sig_genes if n_sig_genes > 0 else 0.0
            recovery_curve.append(recovery_rate)

        if len(recovery_curve) > 1:
            x_positions = np.arange(len(recovery_curve)) / (len(recovery_curve) - 1)
            auc = np.trapezoid(recovery_curve, x_positions)
        elif len(recovery_curve) == 1:
            auc = recovery_curve[0]
        else:
            auc = 0.0

        auc_scores.append(auc)

    auc_scores = np.array(auc_scores)

    if normalize and len(auc_scores) > 0:
        min_score = np.min(auc_scores)
        max_score = np.max(auc_scores)
        if max_score > min_score:
            auc_scores = (auc_scores - min_score) / (max_score - min_score)

    adata.obs[score_name] = auc_scores


print("--- End of Section 0.4 ---")

# %%
# PART 1: Data Ingestion and Preprocessing
# Section 1.1: Load Rebuffet Blood NK Data

print("--- PART 1: Data Ingestion ---")
print("  --- Section 1.1: Loading Blood NK Data ---")

adata_blood_source = None
adata_blood = None

try:
    adata_blood_source = sc.read_h5ad(REBUFFET_H5AD_FILE)
    print(f"      Loaded Rebuffet blood data: {adata_blood_source.shape}")
except Exception as e:
    print(f"      ERROR loading Rebuffet data: {e}")

if adata_blood_source is not None:
    adata_blood = adata_blood_source.copy()

    if REBUFFET_ORIG_SUBTYPE_COL in adata_blood.obs.columns and REBUFFET_SUBTYPE_COL not in adata_blood.obs.columns:
        adata_blood.obs[REBUFFET_SUBTYPE_COL] = adata_blood.obs[REBUFFET_ORIG_SUBTYPE_COL]
        adata_blood.obs[REBUFFET_SUBTYPE_COL] = pd.Categorical(
            adata_blood.obs[REBUFFET_SUBTYPE_COL],
            categories=REBUFFET_SUBTYPES_ORDERED,
            ordered=True,
        )

    if REBUFFET_SUBTYPE_COL in adata_blood.obs.columns:
        adata_blood = adata_blood[adata_blood.obs[REBUFFET_SUBTYPE_COL].notna(), :].copy()

    # Use pre-computed Harmony embeddings
    if 'X_pca_harmony' in adata_blood.obsm:
        adata_blood.obsm['X_pca'] = adata_blood.obsm['X_pca_harmony'].copy()
        sc.pp.neighbors(adata_blood, use_rep='X_pca_harmony', n_neighbors=30, random_state=RANDOM_SEED)

    if 'X_umap_harmony' in adata_blood.obsm:
        adata_blood.obsm['X_umap'] = adata_blood.obsm['X_umap_harmony'].copy()
    else:
        sc.tl.umap(adata_blood, random_state=RANDOM_SEED, min_dist=0.5, spread=1.0)

    print(f"      adata_blood ready: {adata_blood.shape}")
    
    # Filter Hallmark gene sets to expressed genes (non-circular approach)
    if HALLMARK_GENE_SETS_FULL:
        print("\n      Filtering Hallmark gene sets to expressed genes...")
        hallmark_filtered = filter_hallmark_to_expressed(
            adata_blood, HALLMARK_GENE_SETS_FULL, min_pct_cells=0.01
        )
        
        # Add filtered Hallmark sets to FUNCTIONAL_GENE_SETS
        for set_name, genes in hallmark_filtered.items():
            if genes:  # Only add if we got valid genes
                FUNCTIONAL_GENE_SETS[set_name] = genes
        
        print(f"      Added {len(hallmark_filtered)} metabolic signatures (filtered to expressed genes)")

print("    --- End of 1.1 ---")

# %%
# Section 1.2: Load Tang Combined Dataset

print("  --- Section 1.2: Loading Tang Combined Dataset ---")

adata_tang_full = None

try:
    adata_tang_full = sc.read_h5ad(TANG_COMBINED_H5AD_FILE)
    print(f"      Loaded Tang data: {adata_tang_full.shape}")

    if adata_tang_full.raw is None:
        adata_tang_full.raw = adata_tang_full.copy()

    # Log-normalize if needed
    if adata_tang_full.X.max() > 100:
        sc.pp.normalize_total(adata_tang_full, target_sum=1e4)
        sc.pp.log1p(adata_tang_full)

except Exception as e:
    print(f"      ERROR loading Tang data: {e}")

print("    --- End of 1.2 ---")

# %%
# Section 1.3: Create Context-Specific Cohorts

print("  --- Section 1.3: Creating Context-Specific Cohorts ---")

adata_normal_tissue = None
adata_tumor_tissue = None

if adata_tang_full is not None and METADATA_TISSUE_COLUMN_GSE212890 in adata_tang_full.obs.columns:
    # Create Normal Tissue cohort
    normal_mask = adata_tang_full.obs[METADATA_TISSUE_COLUMN_GSE212890] == "Normal"
    if normal_mask.any():
        adata_normal_tissue = adata_tang_full[normal_mask, :].copy()
        print(f"      adata_normal_tissue created: {adata_normal_tissue.shape}")

    # Create Tumor Tissue cohort
    tumor_mask = adata_tang_full.obs[METADATA_TISSUE_COLUMN_GSE212890] == "Tumor"
    if tumor_mask.any():
        adata_tumor_tissue = adata_tang_full[tumor_mask, :].copy()
        print(f"      adata_tumor_tissue created: {adata_tumor_tissue.shape}")

print("    --- End of 1.3 ---")

# %%
# Section 1.3.3: Cross-Dataset Re-annotation

print("  --- Section 1.3.3: Cross-Dataset Re-annotation ---")


def generate_rebuffet_reference_signatures(adata_blood_ref, top_n_genes=50):
    """Generate robust reference signatures from Rebuffet blood NK subtypes."""
    if adata_blood_ref is None or adata_blood_ref.n_obs == 0:
        return {}

    subtype_col = REBUFFET_SUBTYPE_COL
    if subtype_col not in adata_blood_ref.obs.columns:
        return {}

    ref_mask = adata_blood_ref.obs[subtype_col] != "Unassigned"
    adata_ref_clean = adata_blood_ref[ref_mask, :].copy()

    if adata_ref_clean.n_obs == 0:
        return {}

    sc.tl.rank_genes_groups(
        adata_ref_clean, groupby=subtype_col, method="wilcoxon",
        use_raw=True, pts=True, corr_method="benjamini-hochberg",
        n_genes=top_n_genes + 100, key_added="rebuffet_reference_degs",
    )

    reference_signatures = {}
    available_subtypes = adata_ref_clean.obs[subtype_col].cat.categories

    for subtype in available_subtypes:
        try:
            deg_df = sc.get.rank_genes_groups_df(adata_ref_clean, group=subtype, key="rebuffet_reference_degs")
            if deg_df is None or deg_df.empty:
                reference_signatures[subtype] = []
                continue

            filtered_degs = deg_df[
                (~deg_df["names"].apply(is_gene_to_exclude_util))
                & (deg_df["pvals_adj"] < 0.05)
                & (deg_df["logfoldchanges"] > 0.25)
                & (deg_df["scores"] > 0)
            ].copy()

            if not filtered_degs.empty:
                filtered_degs["composite_score"] = (
                    -np.log10(filtered_degs["pvals_adj"] + 1e-300) *
                    filtered_degs["logfoldchanges"] * filtered_degs["scores"]
                )
                top_genes = filtered_degs.nlargest(top_n_genes, "composite_score")["names"].tolist()
            else:
                basic_filtered = deg_df[
                    (~deg_df["names"].apply(is_gene_to_exclude_util))
                    & (deg_df["pvals_adj"] < 0.1)
                    & (deg_df["logfoldchanges"] > 0.1)
                ].copy()
                top_genes = basic_filtered["names"].head(top_n_genes).tolist()

            reference_signatures[subtype] = top_genes
            print(f"        {subtype}: {len(top_genes)} signature genes")

        except Exception as e:
            print(f"        ERROR for {subtype}: {e}")
            reference_signatures[subtype] = []

    return reference_signatures


def annotate_tang_cells_with_rebuffet_signatures(adata_tang, ref_signatures, assignment_threshold=0.10):
    """Re-annotate Tang cells using Rebuffet signatures."""
    assigned_subtypes = ["Unassigned"] * adata_tang.n_obs

    if TANG_MAJORTYPE_COL in adata_tang.obs.columns:
        # Assign CD56bright cells to NK2
        cd56bright_mask = adata_tang.obs[TANG_MAJORTYPE_COL] == "CD56highCD16low"
        for i in range(len(adata_tang.obs)):
            if cd56bright_mask.iloc[i]:
                assigned_subtypes[i] = "NK2"

        # Use signature scoring for remaining cells
        cd56dim_mask = adata_tang.obs[TANG_MAJORTYPE_COL] == "CD56lowCD16high"
        cd56bright_cd16high_mask = adata_tang.obs[TANG_MAJORTYPE_COL] == "CD56highCD16high"
        cells_for_scoring_mask = cd56dim_mask | cd56bright_cd16high_mask

        if cells_for_scoring_mask.sum() > 0:
            adata_for_scoring = adata_tang[cells_for_scoring_mask, :].copy()
            mature_nk_subtypes = ["NK1A", "NK1B", "NK1C", "NK3", "NKint"]
            signature_scores = {}

            for subtype in mature_nk_subtypes:
                if subtype not in ref_signatures or not ref_signatures[subtype]:
                    continue
                score_col = f"{subtype}_Signature_Score"
                available_genes = map_gene_names(ref_signatures[subtype], adata_for_scoring.raw.var_names)
                if len(available_genes) >= MIN_GENES_FOR_SCORING:
                    score_genes_aucell(adata_for_scoring, available_genes, score_name=score_col, use_raw=True, normalize=True)
                    signature_scores[subtype] = score_col

            if signature_scores:
                score_columns = list(signature_scores.values())
                score_matrix = adata_for_scoring.obs[score_columns].values
                max_indices = np.argmax(score_matrix, axis=1)
                subtype_names = list(signature_scores.keys())
                cells_for_scoring_indices = np.where(cells_for_scoring_mask)[0]

                for i, cell_idx in enumerate(cells_for_scoring_indices):
                    assigned_subtypes[cell_idx] = subtype_names[max_indices[i]]

    adata_tang.obs[REBUFFET_SUBTYPE_COL] = pd.Categorical(
        assigned_subtypes,
        categories=REBUFFET_SUBTYPES_ORDERED + ["Unassigned"],
        ordered=True,
    )


# Generate reference signatures
ref_rebuffet_markers = {}
if adata_blood is not None and adata_blood.n_obs > 0:
    ref_rebuffet_markers = generate_rebuffet_reference_signatures(adata_blood, top_n_genes=50)
    print(f"      Generated reference signatures for {len(ref_rebuffet_markers)} subtypes")

# Re-annotate Tang tissue datasets
for cohort_name, adata_ctx in [("NormalTissue", adata_normal_tissue), ("TumorTissue", adata_tumor_tissue)]:
    if adata_ctx is not None and adata_ctx.n_obs > 0 and ref_rebuffet_markers:
        annotate_tang_cells_with_rebuffet_signatures(adata_ctx, ref_rebuffet_markers)
        print(f"      Re-annotated {cohort_name}: {adata_ctx.n_obs} cells")

# Run dimensionality reduction for Tang cohorts
for cohort_name, adata_ctx in [("NormalTissue", adata_normal_tissue), ("TumorTissue", adata_tumor_tissue)]:
    if adata_ctx is not None and adata_ctx.n_obs > 0:
        if "X_umap" not in adata_ctx.obsm:
            sc.pp.highly_variable_genes(adata_ctx, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=False)
            sc.pp.scale(adata_ctx, max_value=10)
            sc.tl.pca(adata_ctx, n_comps=30, random_state=RANDOM_SEED)
            sc.pp.neighbors(adata_ctx, n_neighbors=15, n_pcs=30, random_state=RANDOM_SEED)
            sc.tl.umap(adata_ctx, random_state=RANDOM_SEED)
            print(f"      Computed UMAP for {cohort_name}")

print("    --- End of 1.3.3 ---")

# %%
# Section 1.4: Create Combined Gene Sets

print("  --- Section 1.4: Creating Combined Gene Sets ---")

if ref_rebuffet_markers:
    DEVELOPMENTAL_GENE_SETS = {
        f"{subtype}_Developmental": genes
        for subtype, genes in ref_rebuffet_markers.items()
    }
else:
    DEVELOPMENTAL_GENE_SETS = {
        "Regulatory_NK": ["SELL", "TCF7", "IL7R", "CCR7"],
        "Intermediate_NK": ["CD27", "GZMK", "KLRB1", "CD7"],
        "Mature_Cytotoxic_NK": ["GNLY", "NKG7", "GZMB", "PRF1"],
        "Adaptive_NK": ["KLRC2", "KLRG1", "FGFBP2", "ZEB2"],
    }

ALL_FUNCTIONAL_GENE_SETS = {
    **DEVELOPMENTAL_GENE_SETS,
    **FUNCTIONAL_GENE_SETS,
}

print(f"  Created ALL_FUNCTIONAL_GENE_SETS with {len(ALL_FUNCTIONAL_GENE_SETS)} gene sets")
print("--- END OF PART 1 ---")

# %%
# PART 2: Baseline Characterization
# Define cohorts for characterization

print("\n--- PART 2: Baseline Characterization ---")

# Define cohorts (adata objects only - output dirs are now centralized)
cohorts_for_characterization = []
if adata_blood is not None and adata_blood.n_obs > 0:
    cohorts_for_characterization.append(("Blood", adata_blood))
if adata_normal_tissue is not None and adata_normal_tissue.n_obs > 0:
    cohorts_for_characterization.append(("NormalTissue", adata_normal_tissue))
if adata_tumor_tissue is not None and adata_tumor_tissue.n_obs > 0:
    cohorts_for_characterization.append(("TumorTissue", adata_tumor_tissue))

# %%
# Section 2.1: UMAP by Subtype (Blood only for P2_1_UMAP output)

print("  --- Section 2.1: UMAP by Subtype ---")

for context_name, adata_ctx in cohorts_for_characterization:
    # Only generate for Blood context as per requirements
    if context_name != "Blood":
        continue

    print(f"\n    --- Processing UMAP for: {context_name} ---")

    # Use new centralized output directories
    ctx_fig_dir = OUTPUT_SUBDIRS["characterization_figs"]
    ctx_data_dir = OUTPUT_SUBDIRS["characterization_data"]

    subtype_col = get_subtype_column(adata_ctx)

    if subtype_col and subtype_col in adata_ctx.obs.columns and "X_umap" in adata_ctx.obsm:
        try:
            fig_umap_subtype, ax_umap_subtype = plt.subplots(figsize=(10, 7))
            sc.pl.umap(
                adata_ctx, color=subtype_col, ax=ax_umap_subtype, show=False,
                legend_loc="right margin", legend_fontsize=8,
                title=f"UMAP of {context_name} by Subtype",
                palette=COMBINED_SUBTYPE_COLOR_PALETTE, size=8,
            )
            plt.subplots_adjust(right=0.75)

            umap_coords_df = pd.DataFrame(adata_ctx.obsm["X_umap"], index=adata_ctx.obs_names)
            umap_coords_df.columns = ["UMAP1", "UMAP2"]
            graphpad_umap_data = (
                adata_ctx.obs[[subtype_col]].join(umap_coords_df).reset_index().rename(columns={"index": "CellID"})
            )
            plot_basename_umap = create_filename(
                "P2_1_UMAP_by_Subtype_with_Unassigned", context_name=context_name, version="v3_relaxed"
            )
            save_figure_and_data(fig_umap_subtype, graphpad_umap_data, plot_basename_umap, ctx_fig_dir, ctx_data_dir)

        except Exception as e:
            print(f"        ERROR generating UMAP: {e}")

print("  --- End of Section 2.1 ---")

# %%
# Section 2.3: Functional Signature Heatmaps (P2_3b output)

print("\n  --- Section 2.3: Functional Signature Heatmaps ---")


def generate_signature_heatmap(adata_view, context_name, gene_sets_dict, plot_title,
                               base_filename, fig_dir, data_dir, subtype_col=None):
    """Calculates scores for given gene sets and plots a summary heatmap."""
    print(f"      Generating '{plot_title}' for {context_name}...")

    if subtype_col is None:
        subtype_col = get_subtype_column(adata_view)

    if not subtype_col or subtype_col not in adata_view.obs.columns:
        print(f"        ERROR: No valid subtype column for {context_name}. Skipping.")
        return

    # Calculate scores
    score_cols = []
    for set_name, gene_list in gene_sets_dict.items():
        score_col_name = f"{set_name}_Score"
        score_cols.append(score_col_name)
        available_genes = map_gene_names(gene_list, adata_view.raw.var_names)
        if len(available_genes) >= MIN_GENES_FOR_SCORING:
            score_genes_aucell(adata_view, available_genes, score_name=score_col_name, use_raw=True, normalize=True)
        else:
            adata_view.obs[score_col_name] = np.nan

    # Create heatmap
    valid_score_cols = [col for col in score_cols if col in adata_view.obs.columns and adata_view.obs[col].notna().any()]
    if not valid_score_cols:
        print(f"        No valid scores for {context_name}.")
        return

    mean_scores_df = adata_view.obs.groupby(subtype_col, observed=True)[valid_score_cols].mean().T
    mean_scores_df.index = (
        mean_scores_df.index.str.replace("_Score", "")
        .str.replace("Maturation_NK._", "", regex=True)
        .str.replace("_", " ")
    )

    try:
        fig_width, fig_height, left_margin = calculate_heatmap_layout(
            mean_scores_df, min_width=10, min_height=6, cell_width=1.5, cell_height=0.9, label_padding=3.0
        )
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        abs_max = np.nanmax(np.abs(mean_scores_df.values)) if not np.all(np.isnan(mean_scores_df.values)) else 0.1

        sns.heatmap(
            mean_scores_df, annot=True, fmt=".3f", cmap="icefire", center=0,
            vmin=-abs_max, vmax=abs_max, linewidths=HEATMAP_LINEWIDTHS,
            cbar_kws={"label": "Mean Signature Score", "shrink": HEATMAP_CBAR_SHRINK},
            annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": HEATMAP_ANNOT_WEIGHT},
            square=HEATMAP_SQUARE_CELLS, ax=ax,
        )
        ax.set_title(f"{plot_title} in {context_name}", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Assigned NK Subtype", fontsize=14, fontweight="bold")
        ax.set_ylabel("Signature", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12, rotation=45)
        ax.tick_params(axis="y", labelsize=12)

        plt.subplots_adjust(left=left_margin, right=0.85, top=0.9, bottom=0.1)
        plt.tight_layout(rect=[left_margin, 0.1, 0.85, 0.9])

        plot_basename = create_filename(base_filename, context_name=context_name, version="v5")
        
        # Save summary heatmap data
        save_figure_and_data(fig, mean_scores_df.reset_index(), plot_basename, fig_dir, data_dir)
        
        # === GRAPHPAD RAW DATA EXPORT ===
        # Export cell-level scores for independent statistical analysis
        # Wide format: each cell as a row with subtype + all signature scores
        export_raw_cell_data_for_graphpad(
            adata_view=adata_view,
            score_columns=valid_score_cols,
            group_column=subtype_col,
            data_dir=data_dir,
            filename_base=f"{plot_basename}_by_subtype",
            include_cell_id=True,
            format_type="wide"
        )

    except Exception as e:
        print(f"        ERROR generating heatmap: {e}")


# Generate functional profile heatmaps for all contexts
for context_name, adata_ctx in cohorts_for_characterization:
    print(f"\n    --- Generating Functional Heatmap for: {context_name} ---")

    subtype_col = get_subtype_column(adata_ctx)
    if not subtype_col or subtype_col not in adata_ctx.obs.columns or adata_ctx.raw is None:
        print(f"      Prerequisites not met for {context_name}. Skipping.")
        continue

    assigned_mask = adata_ctx.obs[subtype_col] != "Unassigned"
    if not assigned_mask.any():
        print(f"      No assigned cells in {context_name}. Skipping.")
        continue

    adata_view_assigned = adata_ctx[assigned_mask, :].copy()

    # Use new centralized output directories
    ctx_fig_dir = OUTPUT_SUBDIRS["characterization_figs"]
    ctx_data_dir = OUTPUT_SUBDIRS["characterization_data"]

    generate_signature_heatmap(
        adata_view=adata_view_assigned, context_name=context_name,
        gene_sets_dict=FUNCTIONAL_GENE_SETS, plot_title="Functional Signature Profiles",
        base_filename="P2_3b_Heatmap_FuncProfile", fig_dir=ctx_fig_dir, data_dir=ctx_data_dir,
        subtype_col=subtype_col,
    )

print("\n--- End of PART 2 ---")

# %%
# PART 3: TUSC2 Analysis - Calculate Binary Groups and Impact Stats

print("\n--- PART 3: TUSC2 Analysis ---")
print("  --- Section 3.1: Calculate TUSC2 Binary Groups ---")

for context_name, adata_ctx in cohorts_for_characterization:
    if adata_ctx is None or adata_ctx.raw is None:
        continue
    if TUSC2_GENE_NAME not in adata_ctx.raw.var_names:
        print(f"      {TUSC2_GENE_NAME} not found in {context_name}. Skipping.")
        continue

    obs_col_tusc2_expr = f"{TUSC2_GENE_NAME}_Expression_Raw"
    adata_ctx.obs[obs_col_tusc2_expr] = adata_ctx.raw[:, TUSC2_GENE_NAME].X.toarray().flatten()

    adata_ctx.obs[TUSC2_BINARY_GROUP_COL] = np.where(
        adata_ctx.obs[obs_col_tusc2_expr] > TUSC2_EXPRESSION_THRESHOLD_BINARY,
        TUSC2_BINARY_CATEGORIES[1],
        TUSC2_BINARY_CATEGORIES[0],
    )
    adata_ctx.obs[TUSC2_BINARY_GROUP_COL] = pd.Categorical(
        adata_ctx.obs[TUSC2_BINARY_GROUP_COL], categories=TUSC2_BINARY_CATEGORIES, ordered=True
    )
    print(f"    TUSC2 binary groups calculated for {context_name}.")

# %%
# Section 3.3: TUSC2 Impact on Functional Signatures

print("\n  --- Section 3.3: TUSC2 Impact on Functional Signatures ---")

# Use new centralized intermediate stats directory
layer3_stats_dir = OUTPUT_SUBDIRS["intermediate_stats"]

score_column_names = [f"{set_name}_Score" for set_name in ALL_FUNCTIONAL_GENE_SETS.keys()]

# Collect stats in memory for Part 4 (avoid unnecessary disk reload)
all_tusc2_impact_stats_memory = []

for context_name, adata_ctx in cohorts_for_characterization:
    print(f"\n    --- Processing TUSC2 Impact for: {context_name} ---")

    if adata_ctx is None or TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns:
        continue

    # Calculate functional scores
    for set_name, gene_list in ALL_FUNCTIONAL_GENE_SETS.items():
        score_col_name = f"{set_name}_Score"
        available_genes = map_gene_names(gene_list, adata_ctx.raw.var_names)
        if len(available_genes) >= MIN_GENES_FOR_SCORING:
            score_genes_aucell(adata_ctx, available_genes, score_name=score_col_name, use_raw=True, normalize=True)
        else:
            adata_ctx.obs[score_col_name] = np.nan

    # Compare scores by TUSC2 group
    stats_list = []
    for score_col in score_column_names:
        if score_col not in adata_ctx.obs.columns:
            continue

        group1 = adata_ctx.obs[score_col][adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[1]].dropna()
        group0 = adata_ctx.obs[score_col][adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[0]].dropna()

        stat, pval, n1, n0 = np.nan, np.nan, len(group1), len(group0)
        mean1, mean0 = (group1.mean() if n1 > 0 else np.nan), (group0.mean() if n0 > 0 else np.nan)

        if n1 >= 3 and n0 >= 3:
            stat, pval = stats.mannwhitneyu(group1, group0, alternative="two-sided")

        stats_list.append({
            "Functional_Signature": score_col.replace("_Score", ""),
            "Mean_Score_Diff": mean1 - mean0,
            "P_Value_MWU": pval,
            "N_TUSC2_Expressed": n1,
            "N_TUSC2_Not_Expressed": n0,
        })

    stats_df = pd.DataFrame(stats_list)
    if stats_df.empty or stats_df["P_Value_MWU"].isna().all():
        continue

    # FDR correction
    pvals = stats_df["P_Value_MWU"].dropna()
    if not pvals.empty:
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        stats_df["Q_Value_FDR"] = np.nan
        stats_df.loc[pvals.index, "Q_Value_FDR"] = qvals
        stats_df["Significance"] = stats_df["Q_Value_FDR"].apply(get_significance_stars)
    else:
        stats_df["Q_Value_FDR"] = np.nan
        stats_df["Significance"] = "ns"

    # Save stats to intermediate directory (for archival/debugging)
    stats_filename = create_filename(
        "TUSC2_Impact_Stats", context_name=context_name, version="v1", ext="csv"
    )
    stats_df.to_csv(os.path.join(layer3_stats_dir, stats_filename), index=False)
    print(f"      Stats saved: {stats_filename}")
    
    # Keep in memory for Part 4 (avoid unnecessary disk reload)
    stats_df_with_context = stats_df.copy()
    stats_df_with_context["Context"] = context_name
    all_tusc2_impact_stats_memory.append(stats_df_with_context)
    
    # === GRAPHPAD RAW DATA EXPORT ===
    # Export cell-level scores for independent TUSC2 group comparisons
    valid_score_cols_for_export = [col for col in score_column_names if col in adata_ctx.obs.columns]
    
    # Wide format: each cell with TUSC2 group and all signature scores
    export_raw_cell_data_for_graphpad(
        adata_view=adata_ctx,
        score_columns=valid_score_cols_for_export,
        group_column=TUSC2_BINARY_GROUP_COL,
        data_dir=layer3_stats_dir,
        filename_base=f"TUSC2_Impact_{context_name}_by_group",
        include_cell_id=True,
        format_type="wide"
    )
    
    # Also export grouped format for each signature (ideal for GraphPad t-tests)
    for score_col in valid_score_cols_for_export[:5]:  # Core signatures only
        sig_name = score_col.replace("_Score", "")
        export_grouped_comparison_data(
            adata_view=adata_ctx,
            score_column=score_col,
            group_column=TUSC2_BINARY_GROUP_COL,
            data_dir=layer3_stats_dir,
            filename_base=f"TUSC2_vs_{sig_name}_{context_name}"
        )

print("\n--- End of PART 3 ---")

# %%
# PART 4: Cross-Context Synthesis
# Section 4.2.3: TUSC2 Impact on Core Functional Capacity

print("\n--- PART 4: Cross-Context Synthesis ---")
print("  --- Section 4.2.3: TUSC2 Impact Heatmaps ---")

# Use new centralized output directories
tusc2_impact_fig_dir = OUTPUT_SUBDIRS["tusc2_impact_figs"]
tusc2_impact_data_dir = OUTPUT_SUBDIRS["tusc2_impact_data"]

# Use stats already in memory from Part 3 (no disk reload needed)
if all_tusc2_impact_stats_memory:
    print(f"      Using {len(all_tusc2_impact_stats_memory)} context stats from memory")
    all_tusc2_impact_stats = all_tusc2_impact_stats_memory
else:
    # Fallback: load from disk if running Part 4 separately
    print("      Loading stats from disk (fallback mode)...")
    all_tusc2_impact_stats = []
    base_stats_path = OUTPUT_SUBDIRS["intermediate_stats"]
    for context_name, _ in cohorts_for_characterization:
        stats_filename = create_filename(
            "TUSC2_Impact_Stats", context_name=context_name, version="v1", ext="csv"
        )
        stats_filepath = os.path.join(base_stats_path, stats_filename)
        if os.path.exists(stats_filepath):
            context_stats_df = pd.read_csv(stats_filepath)
            context_stats_df["Context"] = context_name
            all_tusc2_impact_stats.append(context_stats_df)
            print(f"        Loaded stats for {context_name}")

if all_tusc2_impact_stats:
    combined_tusc2_stats = pd.concat(all_tusc2_impact_stats, ignore_index=True)

    # Extract core functional signatures
    core_functional_signatures = [
        sig for sig in combined_tusc2_stats["Functional_Signature"].unique()
        if sig in FUNCTIONAL_GENE_SETS.keys()
    ]

    if core_functional_signatures:
        category_stats = combined_tusc2_stats[
            combined_tusc2_stats["Functional_Signature"].isin(core_functional_signatures)
        ]

        if not category_stats.empty:
            pivot_diff = category_stats.pivot_table(
                index="Functional_Signature", columns="Context", values="Mean_Score_Diff"
            )
            pivot_qval = category_stats.pivot_table(
                index="Functional_Signature", columns="Context", values="Q_Value_FDR"
            )

            # Clean up labels
            pivot_diff.index = (
                pivot_diff.index.str.replace("_Score", "")
                .str.replace("_", " ")
                .str.replace("Cytokine Chemokine Production", "Cytokine/Chemokine")
                .str.replace("Exhaustion Suppression Markers", "Exhaustion/Suppression")
            )
            pivot_qval.index = pivot_diff.index

            # Enforce order
            desired_order = [
                "Activating Receptors", "Inhibitory Receptors", "Exhaustion/Suppression",
                "Cytotoxicity Machinery", "Cytokine/Chemokine",
            ]
            present = [name for name in desired_order if name in pivot_diff.index]
            remaining = [name for name in list(pivot_diff.index) if name not in present]
            new_order = present + remaining
            pivot_diff = pivot_diff.reindex(new_order)
            pivot_qval = pivot_qval.reindex(new_order)

            # Create annotation labels
            annot_labels = pivot_diff.apply(lambda x: x.map("{:.3f}".format)) + pivot_qval.apply(lambda x: x.map(get_significance_stars))

            fig_width, fig_height, left_margin = calculate_heatmap_layout(
                pivot_diff, min_width=10, min_height=6, cell_width=1.5, cell_height=0.9, label_padding=3.0
            )
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            abs_max = np.nanmax(np.abs(pivot_diff.values)) if not np.all(np.isnan(pivot_diff.values)) else 0.1

            sns.heatmap(
                pivot_diff, annot=annot_labels, fmt="s", cmap="icefire", center=0,
                vmin=-abs_max, vmax=abs_max, linewidths=HEATMAP_LINEWIDTHS,
                cbar_kws={"label": "Mean Score Difference\n(TUSC2+ vs TUSC2-)", "shrink": HEATMAP_CBAR_SHRINK},
                annot_kws={"size": 9, "weight": HEATMAP_ANNOT_WEIGHT},
                square=HEATMAP_SQUARE_CELLS, ax=ax,
            )
            ax.set_title("TUSC2+ Cells Associated Core Functional Capacities", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Context", fontsize=14, fontweight="bold")
            ax.set_ylabel("Core Functional Signature", fontsize=14, fontweight="bold")
            ax.tick_params(axis="x", labelsize=12, rotation=45)
            ax.tick_params(axis="y", labelsize=12)

            plt.subplots_adjust(left=left_margin, right=0.85, top=0.9, bottom=0.1)
            plt.tight_layout(rect=[left_margin, 0.1, 0.85, 0.9])

            plot_basename = create_filename("P4_2_Heatmap_TUSC2_Impact_CoreFunctionalCapacity", version="v4_final")
            save_figure_and_data(fig, category_stats, plot_basename, tusc2_impact_fig_dir, tusc2_impact_data_dir)
            
            # === GRAPHPAD RAW DATA EXPORT ===
            # For cross-context synthesis, export pooled raw cell data from all contexts
            print("\n      Exporting raw cell data for P4_2 cross-context analysis...")
            all_contexts_raw_data = []
            core_score_cols = [f"{sig}_Score" for sig in core_functional_signatures]
            
            for ctx_name, adata_ctx in cohorts_for_characterization:
                if TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns:
                    continue
                valid_cols = [c for c in core_score_cols if c in adata_ctx.obs.columns]
                if not valid_cols:
                    continue
                
                ctx_raw = adata_ctx.obs[[TUSC2_BINARY_GROUP_COL] + valid_cols].copy()
                ctx_raw["Context"] = ctx_name
                ctx_raw["Cell_ID"] = adata_ctx.obs_names
                all_contexts_raw_data.append(ctx_raw)
            
            if all_contexts_raw_data:
                combined_raw_df = pd.concat(all_contexts_raw_data, ignore_index=True)
                # Rename columns for clarity
                combined_raw_df.columns = [
                    col.replace("_Score", "") for col in combined_raw_df.columns
                ]
                raw_export_path = os.path.join(
                    tusc2_impact_data_dir, 
                    f"{plot_basename}_raw_data.csv"
                )
                combined_raw_df.to_csv(raw_export_path, index=False)
                print(f"      SUCCESS: Raw data for P4_2 exported ({len(combined_raw_df)} cells)")

print("  --- End of Section 4.2.3 ---")

# %%
# Section 4.3: TUSC2 Impact on NK Subtype Programs

print("\n  --- Section 4.3: TUSC2 Impact on NK Subtype Programs ---")

# Use same centralized TUSC2 impact directories
if ref_rebuffet_markers:
    all_subtype_stats_results = []

    for context_name, adata_ctx in cohorts_for_characterization:
        print(f"\n    --- Analyzing Subtype Programs for: {context_name} ---")

        if TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns or adata_ctx.raw is None:
            continue

        for subtype in REBUFFET_SUBTYPES_ORDERED:
            if subtype not in ref_rebuffet_markers:
                continue

            subtype_genes = ref_rebuffet_markers[subtype]
            score_name = f"{subtype}_Program_Score"

            score_genes_aucell(adata_ctx, subtype_genes, score_name=score_name, use_raw=True, normalize=True)

            tusc2_pos_mask = adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[1]
            tusc2_neg_mask = adata_ctx.obs[TUSC2_BINARY_GROUP_COL] == TUSC2_BINARY_CATEGORIES[0]

            scores_pos = adata_ctx.obs.loc[tusc2_pos_mask, score_name]
            scores_neg = adata_ctx.obs.loc[tusc2_neg_mask, score_name]

            p_value, mean_diff = np.nan, np.nan
            if len(scores_pos) >= 3 and len(scores_neg) >= 3:
                mean_diff = scores_pos.mean() - scores_neg.mean()
                _, p_value = stats.mannwhitneyu(scores_pos, scores_neg, alternative="two-sided")

            all_subtype_stats_results.append({
                "Context": context_name,
                "Subtype_Program": subtype,
                "Mean_Score_Diff": mean_diff,
                "P_Value": p_value,
            })

    # Generate heatmap
    if all_subtype_stats_results:
        summary_subtype_stats_df = pd.DataFrame(all_subtype_stats_results).dropna()
        pvals_to_correct = summary_subtype_stats_df["P_Value"]

        if not pvals_to_correct.empty:
            _, qvals, _, _ = multipletests(pvals_to_correct, method="fdr_bh")
            summary_subtype_stats_df["Q_Value"] = qvals

        heatmap_pivot_diff = summary_subtype_stats_df.pivot_table(
            index="Subtype_Program", columns="Context", values="Mean_Score_Diff"
        )
        heatmap_pivot_qval = summary_subtype_stats_df.pivot_table(
            index="Subtype_Program", columns="Context", values="Q_Value"
        )

        heatmap_pivot_diff = heatmap_pivot_diff.reindex(REBUFFET_SUBTYPES_ORDERED).dropna(how="all")
        heatmap_pivot_qval = heatmap_pivot_qval.reindex(heatmap_pivot_diff.index)

        annot_labels = heatmap_pivot_diff.apply(lambda x: x.map("{:.3f}".format)) + heatmap_pivot_qval.apply(lambda x: x.map(get_significance_stars))

        fig, ax = plt.subplots(figsize=(max(8, heatmap_pivot_diff.shape[1] * 1.5), max(6, heatmap_pivot_diff.shape[0] * 0.6)))

        abs_max = np.nanmax(np.abs(heatmap_pivot_diff.values)) if not np.all(np.isnan(heatmap_pivot_diff.values)) else 0.1

        sns.heatmap(
            heatmap_pivot_diff, annot=annot_labels, fmt="s", cmap="icefire", center=0,
            vmin=-abs_max, vmax=abs_max, linewidths=HEATMAP_LINEWIDTHS,
            cbar_kws={"label": "Mean Program Score Difference\n(TUSC2 Expressed vs. Not Expressed)", "shrink": HEATMAP_CBAR_SHRINK},
            annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": HEATMAP_ANNOT_WEIGHT},
            square=HEATMAP_SQUARE_CELLS, ax=ax,
        )
        ax.set_title("TUSC2 Expression is Associated with Mature Cytotoxic Subtypes", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Biological Context", fontsize=14, fontweight="bold")
        ax.set_ylabel("NK Subtype Signature", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12, rotation=45)
        ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()

        plot_basename_summary = create_filename("P4_3_Heatmap_TUSC2_Impact_on_Subtype_Programs", version="v5_final")
        save_figure_and_data(fig, summary_subtype_stats_df, plot_basename_summary, tusc2_impact_fig_dir, tusc2_impact_data_dir)
        
        # === GRAPHPAD RAW DATA EXPORT ===
        # Export cell-level subtype program scores for independent analysis
        print("\n      Exporting raw cell data for P4_3 subtype program analysis...")
        all_contexts_subtype_raw = []
        subtype_score_cols = [f"{st}_Program_Score" for st in REBUFFET_SUBTYPES_ORDERED]
        
        for ctx_name, adata_ctx in cohorts_for_characterization:
            if TUSC2_BINARY_GROUP_COL not in adata_ctx.obs.columns:
                continue
            
            valid_cols = [c for c in subtype_score_cols if c in adata_ctx.obs.columns]
            if not valid_cols:
                continue
            
            ctx_raw = adata_ctx.obs[[TUSC2_BINARY_GROUP_COL] + valid_cols].copy()
            ctx_raw["Context"] = ctx_name
            ctx_raw["Cell_ID"] = adata_ctx.obs_names
            all_contexts_subtype_raw.append(ctx_raw)
        
        if all_contexts_subtype_raw:
            combined_subtype_raw = pd.concat(all_contexts_subtype_raw, ignore_index=True)
            # Rename columns for clarity
            combined_subtype_raw.columns = [
                col.replace("_Program_Score", "_Program") for col in combined_subtype_raw.columns
            ]
            raw_export_path = os.path.join(
                tusc2_impact_data_dir, 
                f"{plot_basename_summary}_raw_data.csv"
            )
            combined_subtype_raw.to_csv(raw_export_path, index=False)
            print(f"      SUCCESS: Raw data for P4_3 exported ({len(combined_subtype_raw)} cells)")

        print("\n      Subtype programs heatmap saved.")

print("\n--- End of Section 4.3 ---")
print("\n--- ANALYSIS COMPLETE ---")
