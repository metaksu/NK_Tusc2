# %%
# ==============================================================================
# --- I. Initial Setup, Configuration, and Utility Functions ---
# ==============================================================================

all_group_comparison_stats_list = []
all_survival_results_list = []

# --- A. Standard Library Imports ---
import os
import re
import xml.etree.ElementTree as ET
from itertools import combinations

# --- B. Third-Party Library Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import statsmodels.api as sm # For potential regression, if kept

from scipy.stats import spearmanr, mannwhitneyu, kruskal, fisher_exact
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# --- C. Global Configuration & Paths ---

# !!! USER: DEFINE THESE VARIABLES FOR EACH RUN !!!
CANCER_TYPE_ABBREV = "GBM"  # e.g., "BRCA", "LUAD", "GBM"
BASE_DATA_DIR = r"C:\Users\met-a\Documents\Analysis\TCGAdata" # Root for your data
BASE_OUTPUT_DIR = r"C:\Users\met-a\Documents\Analysis\TCGAdata\Analysis_Python_Output_v3" # Changed to v3 for new run

# --- Construct Cancer-Specific Paths ---
CANCER_SPECIFIC_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, CANCER_TYPE_ABBREV)
os.makedirs(CANCER_SPECIFIC_OUTPUT_DIR, exist_ok=True)

# Input file/directory paths (these will be used by the loading sections)
CLINICAL_XML_DIR = os.path.join(BASE_DATA_DIR, "xml")
CONSOLIDATED_SAMPLE_SHEET_PATH = os.path.join(BASE_DATA_DIR, "gdc_sample_sheet.2025-06-20.tsv") # Path from your original notebook
GENERAL_RNA_SEQ_DIR = os.path.join(BASE_DATA_DIR, "rna") # Path from your original notebook

# CIBERSORTx and HiRes paths (using CANCER_TYPE_ABBREV for cancer-specificity)
CIBERSORT_LM22_PATH = os.path.join(BASE_DATA_DIR, f"CIBERSORTx_Adjusted_{CANCER_TYPE_ABBREV}_LM22_Fractions.txt") # Placeholder, ensure actual name
CIBERSORT_REBUFFET_PATH = os.path.join(BASE_DATA_DIR, f"CIBERSORTx_Adjusted_{CANCER_TYPE_ABBREV}_Rebuffet_Fractions.txt") # Placeholder
HIRES_NK_ACT_PATH = os.path.join(BASE_DATA_DIR, f"CIBERSORTxHiRes_{CANCER_TYPE_ABBREV}_NKcellsactivated_Window8.txt") # Adjusted original name
HIRES_NK_REST_PATH = os.path.join(BASE_DATA_DIR, f"CIBERSORTxHiRes_{CANCER_TYPE_ABBREV}_NKcellsresting_Window8.txt") # Adjusted original name

# RNA-seq Count Column Preference
PREFERRED_RNA_COUNT_COLUMN = "tpm_unstranded" # As per your usage and confirmation
# ALTERNATIVE_RNA_COUNT_COLUMNS = ["fpkm_unstranded", "unstranded"] # Optional for more complex fallback

# CIBERSORTx Filtering Thresholds
P_VALUE_THRESHOLD_CIBERSORT = 0.05
CORRELATION_THRESHOLD_CIBERSORT = 0.30
RMSE_PERCENTILE_THRESHOLD_CIBERSORT = 0.90
FDR_SIGNIFICANCE_THRESHOLD = 0.05 # For FDR filtering in correlation results

# --- D. Pandas and Matplotlib/Seaborn Display Options ---
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)
pd.set_option('display.precision', 3)
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
plt.rcParams['figure.dpi'] = 100        # For display in notebook
plt.rcParams['savefig.dpi'] = 300     # For saved figures
plt.rcParams['savefig.format'] = 'tiff' # Default save format for publication quality

print(f"--- Configuration for: {CANCER_TYPE_ABBREV} ---")
print(f"Output directory: {CANCER_SPECIFIC_OUTPUT_DIR}")
print(f"Preferred RNA count column: {PREFERRED_RNA_COUNT_COLUMN}")

# ==============================================================================
# --- E. Utility Functions Definition ---
# ==============================================================================

def aggregate_correlation_results(source_data_path, analysis_name, output_dir,
                                  id_col2_name, # This is the varying feature (e.g., Immune_Cell_Type)
                                  r_col="Spearman_R", p_col="p_value", q_col="FDR_q_value",
                                  id_col1_name=None, # Make this optional; it's often implied
                                  cancer_abbrev=CANCER_TYPE_ABBREV):      # Assumes global var
    """
    Loads a correlation data CSV, filters for significance, and saves the summary.
    id_col1_name is the 'constant' variable for this set of correlations (e.g., TUSC2).
    id_col2_name is the column listing the features correlated against id_col1_name.
    """
    print(f"\n  Aggregating: {analysis_name} (Correlated with {id_col1_name if id_col1_name else 'implied primary variable'})")
    if os.path.exists(source_data_path):
        try:
            corr_df = pd.read_csv(source_data_path)
            print(f"    Loaded {analysis_name} data: {corr_df.shape}")

            # Expected columns from the CSV: id_col2_name, r_col, p_col, q_col
            # id_col1_name is conceptual for naming the analysis.
            required_csv_cols = [id_col2_name, r_col, p_col, q_col]
            
            # Handle potential "Unnamed: 0" if CSVs were saved with index
            if 'Unnamed: 0' in corr_df.columns:
                corr_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')


            if not all(col in corr_df.columns for col in required_csv_cols):
                # Try to infer id_col2_name if it wasn't perfectly matched
                # This happens if the barplot y-axis name was different from id_col2_name parameter
                if id_col2_name not in corr_df.columns:
                    # Attempt to find a suitable candidate for id_col2_name
                    # Typically the first non-stat column
                    candidate_cols = [c for c in corr_df.columns if c not in [r_col, p_col, q_col, "Annot_Text", "Annot_Text_Heatmap"]]
                    if candidate_cols:
                        inferred_id_col2 = candidate_cols[0]
                        print(f"    Info: '{id_col2_name}' not found. Inferred '{inferred_id_col2}' as the identifier column for {analysis_name}.")
                        corr_df.rename(columns={inferred_id_col2: id_col2_name}, inplace=True)
                    else:
                        print(f"    ERROR: Could not find or infer '{id_col2_name}' in {source_data_path}.")
                        print(f"    Available columns: {list(corr_df.columns)}")
                        return pd.DataFrame()
                
                # Re-check after potential rename
                if not all(col in corr_df.columns for col in required_csv_cols):
                    print(f"    ERROR: Still missing required columns in {source_data_path} after inference attempts. Expected: {required_csv_cols}. Found: {list(corr_df.columns)}")
                    return pd.DataFrame()


            # Filter for significant correlations
            if q_col not in corr_df.columns:
                print(f"    WARNING: FDR q-value column '{q_col}' not found. Cannot filter by FDR for {analysis_name}.")
                significant_df = pd.DataFrame() # Or filter by p_value as a fallback
            else:
                significant_df = corr_df[corr_df[q_col] < FDR_SIGNIFICANCE_THRESHOLD].copy()
                significant_df.sort_values(by=r_col, ascending=False, inplace=True)
            
            print(f"    Significant Correlations (FDR < {FDR_SIGNIFICANCE_THRESHOLD}) for {analysis_name}:")
            if not significant_df.empty:
                # Display: Variable1 (Implied), Variable2 (id_col2_name), R, P, Q
                display_cols = [id_col2_name, r_col, p_col, q_col]
                # For HiRes data, "Gene_Symbol" and "NK_State" might be more informative than "HiRes_Gene_Full" alone
                if analysis_name == "TUSC2_vs_HiRes_NK_Genes" and all(c in significant_df.columns for c in ["Gene_Symbol", "NK_State"]):
                    display_cols = ["Gene_Symbol", "NK_State", r_col, p_col, q_col]
                
                display(significant_df[display_cols].head(20))
                
                summary_filename = f"{cancer_abbrev}_{analysis_name.replace(' ', '_')}_SignificantCorrelations.csv"
                significant_df.to_csv(os.path.join(output_dir, summary_filename), index=False)
                print(f"    Saved significant correlations to: {summary_filename}")
            else:
                print(f"    No significant correlations found for {analysis_name} at FDR < {FDR_SIGNIFICANCE_THRESHOLD}.")
            return corr_df # Return the full (non-significant filtered) df for other potential uses
        except Exception as e:
            print(f"    ERROR processing {source_data_path}: {e}")
            return pd.DataFrame()
    else:
        print(f"    WARNING: Data file not found: {source_data_path}")
        return pd.DataFrame()

def save_plot_and_data(fig, plot_data_df, output_dir, filename_base, is_summary_data=False):
    """
    Saves the Matplotlib figure and the DataFrame.
    Exports DataFrame as CSV. If it's summary data (like means for a bar plot),
    it might not have a per-sample index.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        plot_data_df (pd.DataFrame): DataFrame used for the plot.
        output_dir (str): Directory to save the files.
        filename_base (str): Base name for the output files (without extension).
        is_summary_data (bool): If True, saves index if meaningful (e.g. for matrices).
                                If False (default, for sample-level data), usually saves without index
                                unless index is meaningful (like Sample_ID).
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{filename_base}.tiff")
    data_path = os.path.join(output_dir, f"{filename_base}_data.csv")

    fig.savefig(plot_path, dpi=300, format="tiff", bbox_inches='tight')

    if is_summary_data: # e.g. correlation matrix, mean compositions
        plot_data_df.to_csv(data_path, index=True)
    else: # e.g. long-form data for boxplots/scatterplots
        # Try to save with a meaningful index if it's not just a range index
        if plot_data_df.index.name or not isinstance(plot_data_df.index, pd.RangeIndex):
            plot_data_df.to_csv(data_path, index=True)
        else:
            plot_data_df.to_csv(data_path, index=False)

    print(f"  Plot saved: {plot_path}")
    print(f"  Data saved: {data_path}")

def get_significance_stars(p_value, threshold_type='p_value'):
    """Returns significance stars based on p-value or q-value."""
    if not pd.notna(p_value): return "ns" # Not significant or not tested
    # Standard thresholds
    if p_value < 0.0001: return "****"
    if p_value < 0.001: return "***"
    if p_value < 0.01: return "**"
    if p_value < 0.05: return "*"
    return "ns"

def add_stat_annotation_bar(ax, idx1, idx2, y_pos, p_value_text, sig_stars_text):
    """Helper to draw a single significance bar with text between two x-positions."""
    bar_height_factor = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    if bar_height_factor <= 0: bar_height_factor = 0.05 * abs(y_pos) if y_pos!=0 else 0.05

    line_y_coords = [y_pos, y_pos + bar_height_factor, y_pos + bar_height_factor, y_pos]
    ax.plot([idx1, idx1, idx2, idx2], line_y_coords, lw=1.5, c='black')
    ax.text((idx1 + idx2) / 2.0, line_y_coords[1] + bar_height_factor * 0.2,
            f"{sig_stars_text}\n(p={p_value_text})",
            ha='center', va='bottom', fontsize=8, color='black') # Adjusted fontsize


      
def plot_box_violin_with_stats(
    df, x_col, y_col, group_order, title, xlab, ylab,
    output_dir, plot_filename_base, palette=None, plot_type='box',
    test_type='mannwhitneyu_2group', significance_threshold=0.05, add_stripplot=True,
    y_limit=None
):
    # ... (initial setup, plot_df creation, group_counts, fig, ax, plotting as before) ...
    # Ensure this initial part is identical to your last working version that produced the plots.
    # The part that generates the plot visuals.
    if not all(c in df.columns for c in [x_col, y_col]):
        print(f"⚠️ plot_box_violin: Missing columns '{x_col}' or '{y_col}' in DataFrame for '{title}'. Skipping.")
        return pd.DataFrame() 

    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    plot_df.dropna(subset=[x_col, y_col], inplace=True)

    if not (isinstance(plot_df[x_col].dtype, pd.CategoricalDtype) and
            list(plot_df[x_col].cat.categories) == group_order and
            plot_df[x_col].cat.ordered):
        plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=group_order, ordered=True)
    plot_df = plot_df[plot_df[x_col].isin(group_order) & plot_df[x_col].notna()] 

    if plot_df.empty or plot_df[x_col].nunique() < 1:
        print(f"⚠️ plot_box_violin: Not enough data or groups for '{title}' after filtering. Skipping.")
        return pd.DataFrame()

    group_counts = plot_df[x_col].value_counts().reindex(group_order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(max(6, len(group_order) * 1.8), 6.5))
    
    hue_order_param = group_order 
    
    if plot_type == 'violin':
        sns.violinplot(x=x_col, y=y_col, data=plot_df, order=group_order, 
                       hue=x_col if palette else None, 
                       hue_order=hue_order_param if palette else None,
                       palette=palette, ax=ax, inner=None, cut=0, 
                       legend=False if palette else "auto") 
        if add_stripplot:
            sns.stripplot(x=x_col, y=y_col, data=plot_df, order=group_order, color='dimgray', alpha=0.4, jitter=True, size=3.5, ax=ax)
    else: 
        sns.boxplot(x=x_col, y=y_col, data=plot_df, order=group_order, 
                    hue=x_col if palette else None, 
                    hue_order=hue_order_param if palette else None,
                    palette=palette, ax=ax, showfliers=False, width=0.6, 
                    legend=False if palette else "auto") 
        if add_stripplot:
            sns.stripplot(x=x_col, y=y_col, data=plot_df, order=group_order, color='dimgray', alpha=0.4, jitter=True, size=3.5, ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_xticks(range(len(group_order))) 
    ax.set_xticklabels([f"{label}\n(n={group_counts.get(label, 0)})" for label in group_order], rotation=0, ha="center")
    if y_limit: ax.set_ylim(y_limit)

    # --- STATISTICAL RESULTS COLLECTION ---
    stats_results_list = [] # Changed name to avoid conflict with outer scope if any

    y_max_data = plot_df[y_col].max(skipna=True)
    y_min_data = plot_df[y_col].min(skipna=True)
    plot_range = y_max_data - y_min_data if pd.notna(y_max_data) and pd.notna(y_min_data) and (y_max_data - y_min_data)>0 else (abs(y_max_data)*0.2 if pd.notna(y_max_data) and y_max_data !=0 else 0.2)
    
    annotation_y_start = (y_max_data if pd.notna(y_max_data) else 0) + plot_range * 0.05
    annotation_step = plot_range * 0.10 if plot_range > 0 else 0.1
    current_annotation_level = 0

    if test_type == 'mannwhitneyu_2group' and len(group_order) == 2:
        g1_vals = plot_df[plot_df[x_col] == group_order[0]][y_col].dropna()
        g2_vals = plot_df[plot_df[x_col] == group_order[1]][y_col].dropna()
        if len(g1_vals) >= 2 and len(g2_vals) >= 2: # Scipy MWU needs at least 1 in each, but practically more for meaningful p
            stat, p_val = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
            stats_results_list.append({
                'test_type': 'Mann-Whitney U', 'group1': group_order[0], 'group2': group_order[1], 
                'statistic': stat, 'p_value': p_val, 'q_value_fdr': np.nan 
            })
            sig_text = get_significance_stars(p_val, significance_threshold)
            if sig_text != "ns":
                y_annot_pos = annotation_y_start + current_annotation_level * annotation_step
                add_stat_annotation_bar(ax, 0, 1, y_annot_pos, f"{p_val:.2e}", sig_text)
                current_annotation_level +=1
    
    elif test_type == 'kruskal_multigroup' and len(group_order) > 1:
        groups_data = [plot_df[plot_df[x_col] == g][y_col].dropna() for g in group_order if group_counts.get(g,0) >= 2]
        valid_groups_for_test = [g for g in groups_data if len(g) >=1] # Kruskal needs at least one observation per group
        if len(valid_groups_for_test) >= 2: # Kruskal needs at least two groups
            try:
                stat, p_val = kruskal(*valid_groups_for_test)
                stats_results_list.append({
                    'test_type': 'Kruskal-Wallis', 'group1': "Overall", 'group2': str(group_order), 
                    'statistic': stat, 'p_value': p_val, 'q_value_fdr': np.nan 
                })
                if p_val < significance_threshold:
                     ax.text(0.98, 0.98, f"Kruskal-Wallis\np={p_val:.2e}", transform=ax.transAxes, ha='right', va='top',
                            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
            except ValueError as e: print(f"  Kruskal-Wallis error for '{title}': {e}")
    
    elif test_type == 'mannwhitneyu_pairwise_fdr' and len(group_order) > 1:
        # Overall Kruskal as context
        groups_data_kw_pair = [plot_df[plot_df[x_col] == g][y_col].dropna() for g in group_order if group_counts.get(g,0) >=2]
        valid_groups_for_kw_pair = [g for g in groups_data_kw_pair if len(g) >=1]
        if len(valid_groups_for_kw_pair) >= 2:
            try:
                stat_kw, p_kw = kruskal(*valid_groups_for_kw_pair)
                stats_results_list.append({'test_type': 'Kruskal-Wallis (Overall Context)', 'group1': "Overall", 'group2': str(group_order), 
                                      'statistic': stat_kw, 'p_value': p_kw, 'q_value_fdr': np.nan})
                if p_kw < significance_threshold:
                     ax.text(0.98, 0.98, f"Overall K-W p={p_kw:.2e}", transform=ax.transAxes, ha='right', va='top',
                            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
            except ValueError as e: print(f"  Kruskal-Wallis (pairwise context) error for '{title}': {e}")

        pairs = list(combinations(group_order, 2))
        raw_p_values_for_fdr = []
        valid_pairs_for_stats = []
        pair_stats_list_temp = [] # Store (stat, p) for valid pairs

        for g1_name, g2_name in pairs:
            g1_data = plot_df[plot_df[x_col] == g1_name][y_col].dropna()
            g2_data = plot_df[plot_df[x_col] == g2_name][y_col].dropna()
            if len(g1_data) >= 1 and len(g2_data) >= 1: # MWU can run with 1 obs, but p-val might not be ideal
                stat_mw_pair, p_mw_pair = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                raw_p_values_for_fdr.append(p_mw_pair)
                valid_pairs_for_stats.append((g1_name, g2_name))
                pair_stats_list_temp.append({'statistic': stat_mw_pair, 'p_value': p_mw_pair})
            else: # Keep order for later mapping if a pair is invalid
                raw_p_values_for_fdr.append(np.nan) 
                valid_pairs_for_stats.append((g1_name, g2_name)) 
                pair_stats_list_temp.append({'statistic': np.nan, 'p_value': np.nan})
        
        q_values_fdr_corrected = np.full(len(raw_p_values_for_fdr), np.nan)
        pvals_to_correct_actual = [p for p in raw_p_values_for_fdr if pd.notna(p)]
        if pvals_to_correct_actual:
            reject, q_values_temp, _, _ = multipletests(pvals_to_correct_actual, method='fdr_bh', alpha=0.05)
            # Map back
            fdr_idx = 0
            for i in range(len(raw_p_values_for_fdr)):
                if pd.notna(raw_p_values_for_fdr[i]):
                    q_values_fdr_corrected[i] = q_values_temp[fdr_idx]
                    fdr_idx +=1
        
        for i, (g1_name, g2_name) in enumerate(valid_pairs_for_stats):
            stats_results_list.append({
                'test_type': 'Mann-Whitney U (Pairwise)', 'group1': g1_name, 'group2': g2_name,
                'statistic': pair_stats_list_temp[i]['statistic'], 
                'p_value': pair_stats_list_temp[i]['p_value'], 
                'q_value_fdr': q_values_fdr_corrected[i]
            })
            q_val_current_pair = q_values_fdr_corrected[i]
            if pd.notna(q_val_current_pair) and q_val_current_pair < significance_threshold:
                sig_text = get_significance_stars(q_val_current_pair, significance_threshold)
                idx1 = group_order.index(g1_name)
                idx2 = group_order.index(g2_name)
                min_idx, max_idx = min(idx1,idx2), max(idx1,idx2)
                y_annot_pos = annotation_y_start + current_annotation_level * annotation_step
                add_stat_annotation_bar(ax, min_idx, max_idx, y_annot_pos, f"{q_val_current_pair:.2e}", sig_text)
                current_annotation_level += 1
    
    if current_annotation_level > 2: # Adjust ylim if many annotations
        final_y_top = annotation_y_start + current_annotation_level * annotation_step + annotation_step * 0.5
        if ax.get_ylim()[1] < final_y_top : ax.set_ylim(top=final_y_top)

    plt.tight_layout(pad=0.5)
    
    export_data = plot_df[[x_col, y_col]].copy()
    if df.index.name and df.index.name not in export_data.columns:
        try: export_data[df.index.name] = plot_df.loc[export_data.index, df.index.name] # Get index from plot_df
        except: pass # In case plot_df index changed
    elif not isinstance(plot_df.index, pd.RangeIndex) and 'Sample_ID' not in export_data.columns:
        export_data['Sample_ID_from_Index'] = plot_df.index
        
    save_plot_and_data(fig, export_data, output_dir, plot_filename_base, is_summary_data=False)
    plt.close(fig)
    
    return pd.DataFrame(stats_results_list) # Return the collected stats

    


def plot_correlation_scatter(
    df, x_col, y_col, title, xlab, ylab,
    output_dir, plot_filename_base, hue_col=None, palette=None, add_reg_line=True,
    text_pos=(0.05, 0.95) # Default top-left
):
    """Generates scatter plot with optional regression line and Spearman correlation."""
    if not all(c in df.columns for c in [x_col, y_col]):
        print(f"⚠️ plot_correlation_scatter: Missing '{x_col}' or '{y_col}' for '{title}'. Skipping.")
        return
    if hue_col and hue_col not in df.columns:
        print(f"⚠️ plot_correlation_scatter: Hue column '{hue_col}' for '{title}' not found. Plotting without hue.")
        hue_col = None

    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    
    # Drop NaNs essential for plotting and correlation
    essential_cols_for_dropna = [x_col, y_col]
    if hue_col: essential_cols_for_dropna.append(hue_col)
    plot_df.dropna(subset=essential_cols_for_dropna, inplace=True)


    if len(plot_df) < 3: # Need at least 3 for meaningful correlation/regression
        print(f"⚠️ plot_correlation_scatter: Not enough data for '{title}' (n={len(plot_df)}). Skipping.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.scatterplot(x=x_col, y=y_col, data=plot_df, hue=hue_col, palette=palette,
                    alpha=0.6, edgecolor='k', s=40, ax=ax, legend='auto' if hue_col else False)

    r_val, p_val = np.nan, np.nan
    if len(plot_df[x_col].unique()) > 1 and len(plot_df[y_col].unique()) > 1:
        r_val, p_val = spearmanr(plot_df[x_col], plot_df[y_col], nan_policy='omit') # Add nan_policy
        annotation = f"Spearman R = {r_val:.3f}\np = {p_val:.2e}"
        ax.text(text_pos[0], text_pos[1], annotation, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.8))
        if add_reg_line:
            sns.regplot(x=x_col, y=y_col, data=plot_df, scatter=False, ax=ax, color='firebrick',
                        line_kws={'linewidth': 2, 'linestyle': '--'})
    else:
        ax.text(text_pos[0], text_pos[1], "Not enough variance for correlation", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.8))

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    if hue_col and ax.get_legend() is not None:
        ax.legend(title=str(hue_col).replace("_"," "), loc='best', frameon=True, edgecolor='darkgrey')

    plt.tight_layout()
    
    export_cols = [x_col, y_col]
    if hue_col: export_cols.append(hue_col)
    if df.index.name and df.index.name not in export_cols: # Add index if named
        export_df = plot_df[export_cols].set_index(df.index.loc[plot_df.index])
    elif not isinstance(df.index, pd.RangeIndex) and 'Sample_ID' not in export_cols:
        export_df = plot_df[export_cols].copy()
        export_df['Sample_ID_from_Index'] = export_df.index
    else:
        export_df = plot_df[export_cols].copy()
        
    save_plot_and_data(fig, export_df, output_dir, plot_filename_base, is_summary_data=False)
    plt.close(fig)


def plot_kaplan_meier(
    df, time_col, event_col, group_col, title,
    output_dir, plot_filename_base, group_order=None, palette=None,
    legend_loc='best'
):
    """
    Generates Kaplan-Meier survival plots for different groups.
    Returns the logrank_test result object if a 2-group comparison is made and successful.
    """
    if not all(c in df.columns for c in [time_col, event_col, group_col]):
        print(f"⚠️ plot_kaplan_meier: Missing columns for '{title}'. Required: {time_col}, {event_col}, {group_col}. Skipping.")
        return None # Return None if setup fails

    plot_df = df.copy() # Work on a copy to avoid modifying original df
    plot_df[time_col] = pd.to_numeric(plot_df[time_col], errors='coerce')
    plot_df[event_col] = pd.to_numeric(plot_df[event_col], errors='coerce') # Assuming 0/1
    # group_col might already be categorical, handle NaNs before astype(str)
    plot_df.dropna(subset=[time_col, event_col, group_col], inplace=True)

    if plot_df.empty or plot_df[group_col].nunique() == 0:
        print(f"⚠️ plot_kaplan_meier: Not enough data for '{title}' after dropping NaNs or no groups. Skipping.")
        return None

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 6.5)) 

    # Determine unique groups present in the data for this plot
    unique_groups_in_data = sorted(plot_df[group_col].astype(str).unique().tolist())
    
    if group_order is None: # If no specific order provided, use unique groups from data
        plot_group_order = unique_groups_in_data
    else: # Filter provided group_order to only those present in current plot_df
        plot_group_order = [g for g in group_order if str(g) in unique_groups_in_data]
    
    if not plot_group_order or len(plot_group_order) < 1 : # Need at least one group to plot
        print(f"⚠️ plot_kaplan_meier: No valid groups found to plot for '{title}' based on group_order. Skipping.")
        return None

    color_map = None
    if palette: # Create color map if palette is provided
        if isinstance(palette, dict): # If palette is already a map
            color_map = {str(k): v for k,v in palette.items()}
        elif isinstance(palette, list) and len(palette) >= len(plot_group_order):
            color_map = dict(zip(map(str, plot_group_order), palette))
        else:
            print(f"  Warning: Palette length mismatch for KM plot '{title}'. Using default colors.")


    durations_for_logrank = []
    events_for_logrank = []
    group_labels_for_logrank = [] # For log message if needed

    for i, group_name_orig in enumerate(plot_group_order):
        group_name_str = str(group_name_orig) # Ensure string for comparison and labeling
        # Filter data for the current group
        group_data = plot_df[plot_df[group_col].astype(str) == group_name_str]
        
        if not group_data.empty:
            n_group = len(group_data)
            # Sanitize label for plot legend (replace underscores, etc.)
            clean_label_group_name = group_name_str.replace('_', ' ')
            label_for_plot = f"{clean_label_group_name} (n={n_group})"
            
            current_color = None
            if color_map:
                current_color = color_map.get(group_name_str)
            elif isinstance(palette, list): # Fallback to list indexing if map failed but palette list exists
                 current_color = palette[i % len(palette)]


            kmf.fit(group_data[time_col], event_observed=group_data[event_col], label=label_for_plot)
            kmf.plot_survival_function(ax=ax, color=current_color, ci_show=False, linewidth=2)
            
            # Collect data for log-rank test if the group has variance in time and events
            if len(group_data[time_col].unique()) > 0: # Ensure there are events/censoring
                durations_for_logrank.append(group_data[time_col])
                events_for_logrank.append(group_data[event_col])
                group_labels_for_logrank.append(group_name_str) # Store the actual group name used
        else:
            print(f"  Note: Group '{group_name_str}' is empty for KM plot '{title}'.")


    plot_title_text = title
    log_rank_results_obj = None 

    # Perform and annotate log-rank test if exactly two valid groups were prepared for it
    if len(durations_for_logrank) == 2 and len(events_for_logrank) == 2:
        # Ensure there are enough events in each group for a meaningful test
        # logrank_test can fail if one group has no events, or if times are identical.
        if (events_for_logrank[0].sum() > 0 or events_for_logrank[1].sum() > 0) and \
           (len(durations_for_logrank[0]) > 0 and len(durations_for_logrank[1]) > 0) :
            try:
                log_rank_results_obj = logrank_test(
                    durations_A=durations_for_logrank[0],
                    durations_B=durations_for_logrank[1],
                    event_observed_A=events_for_logrank[0],
                    event_observed_B=events_for_logrank[1]
                )
                if log_rank_results_obj:
                    plot_title_text += f"\nLog-rank p = {log_rank_results_obj.p_value:.3g}"
            except Exception as e_lr:
                print(f"    Error during 2-group logrank test for '{title}': {e_lr}")
                plot_title_text += f"\nLog-rank: Error" # Indicate test error
        else:
            print(f"    Log-rank test skipped for '{title}': one or both groups lack events or samples.")
            plot_title_text += f"\nLog-rank: N/A (insufficient events/samples)"

    elif len(durations_for_logrank) > 2:
        print(f"    Note: More than 2 groups ({len(group_labels_for_logrank)}) found for '{title}'. Plotting all, but log-rank p-value on plot is not for overall comparison. Consider multivariate logrank separately.")
        # You could attempt a multivariate logrank here and store its result if you have a specific way to display it
        # e.g., log_rank_multivariate = logrank_test(*durations_for_logrank, event_observed_A=events_for_logrank[0], ...)
        # For now, the returned log_rank_results_obj will be None if not a 2-group test.
    elif len(durations_for_logrank) < 2:
         print(f"    Note: Less than 2 groups with sufficient data for log-rank test in '{title}'.")


    ax.set_title(plot_title_text, fontsize=14, pad=10)
    ax.set_xlabel("Time (Days or Months - Check Data Source)", fontsize=12) # Remind to check unit
    ax.set_ylabel("Survival Probability", fontsize=12)
    if ax.lines: # If lines were plotted, a legend will be useful
        ax.legend(loc=legend_loc, frameon=True, edgecolor='darkgrey', fontsize=9)
    else: # No lines plotted, likely no groups had enough data
        print(f"  Note: No lines plotted for KM '{title}', legend skipped.")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(pad=0.5) 
    
    # Prepare data for export
    export_df = plot_df[[time_col, event_col, group_col]].copy()
    if df.index.name and df.index.name not in export_df.columns:
        # If plot_df was from df.copy(), its index is same as df.
        # If df.index is multi-index, reset_index might be complex. Assume simple index.
        try:
            export_df = export_df.reset_index() # Bring index to column if it was named
        except Exception as e_reset:
            print(f"  Warning: Could not reset_index for KM data export: {e_reset}")
    elif not isinstance(df.index, pd.RangeIndex) and 'Sample_ID' not in export_df.columns and \
         (df.index.name is None or df.index.name not in export_df.columns): # If index is not default range and no Sample_ID exists and index is not already a col
        # Use plot_df's index which is from the filtered df
        try:
            export_df['Sample_ID_from_Index'] = plot_df.index 
        except Exception as e_idx_assign:
             print(f"  Warning: Could not assign Sample_ID_from_Index for KM data export: {e_idx_assign}")
        
    save_plot_and_data(fig, export_df, output_dir, plot_filename_base, is_summary_data=False)
    plt.close(fig)

    return log_rank_results_obj


# (Inside Utility Functions Definition - Section I.E)

def plot_correlation_heatmap(
    corr_matrix_df, title, output_dir, plot_filename_base,
    annot_text_matrix_df=None, # Pre-formatted text for annotation (e.g. "0.75***")
    cmap="vlag", center=0,
    figsize=None, cbar_label='Spearman R',
    sort_rows_by_column=None, # New: Column name or index in corr_matrix_df to sort rows by
    sort_rows_ascending=True    # New: Direction of sort (True for lowest to highest R)
):
    """
    Generates a heatmap for a given correlation matrix.
    Annotations are from annot_text_matrix_df if provided.
    Optionally sorts rows based on values in a specified column of corr_matrix_df.
    If external clustering is applied to corr_matrix_df before calling this function,
    sort_rows_by_column should typically be None to avoid overriding that clustering.
    """
    if corr_matrix_df.empty:
        print(f"⚠️ plot_correlation_heatmap: Correlation matrix for '{title}' is empty. Skipping.")
        return

    # Use copies for plotting, especially if sorting is applied
    plot_corr_matrix = corr_matrix_df.copy()
    if annot_text_matrix_df is not None:
        plot_annot_text_matrix = annot_text_matrix_df.copy()
    else:
        plot_annot_text_matrix = None

    if sort_rows_by_column is not None:
        valid_column_for_sort = False
        target_sort_col_name = None

        if isinstance(sort_rows_by_column, str) and sort_rows_by_column in plot_corr_matrix.columns:
            target_sort_col_name = sort_rows_by_column
            valid_column_for_sort = True
        elif isinstance(sort_rows_by_column, int) and 0 <= sort_rows_by_column < plot_corr_matrix.shape[1]:
            target_sort_col_name = plot_corr_matrix.columns[sort_rows_by_column]
            valid_column_for_sort = True
        
        if valid_column_for_sort and target_sort_col_name is not None:
            print(f"  Sorting heatmap rows by column: '{target_sort_col_name}', ascending={sort_rows_ascending}")
            # Sort the correlation matrix
            plot_corr_matrix.sort_values(by=target_sort_col_name, ascending=sort_rows_ascending, inplace=True)
            # If annotation matrix exists, reindex it to match the sorted correlation matrix
            if plot_annot_text_matrix is not None and isinstance(plot_annot_text_matrix, pd.DataFrame):
                try:
                    plot_annot_text_matrix = plot_annot_text_matrix.reindex(plot_corr_matrix.index)
                except Exception as e_reindex:
                    print(f"    ⚠️ Warning: Could not reindex annotation matrix for sorting: {e_reindex}. Annotations might be misaligned.")
        else:
            print(f"⚠️ plot_correlation_heatmap: sort_rows_by_column '{sort_rows_by_column}' not found or invalid. Plotting without this specific sorting.")

    annotate_values = True
    fmt_for_annot = ""

    if plot_annot_text_matrix is None:
        # If no specific text annotation, use the correlation values formatted to 2 decimal places
        plot_annot_text_matrix = plot_corr_matrix.round(2)
        fmt_for_annot = ".2f"
    else:
        if not isinstance(plot_annot_text_matrix, pd.DataFrame) or plot_annot_text_matrix.shape != plot_corr_matrix.shape:
            print(f"⚠️ plot_correlation_heatmap: Annot text matrix shape mismatch or not a DataFrame for '{title}'. Using R-values for annotation.")
            plot_annot_text_matrix = plot_corr_matrix.round(2)
            fmt_for_annot = ".2f"
        # If plot_annot_text_matrix is already strings (e.g. "0.75***"), fmt should be ""

    if figsize is None:
        figsize = (max(8, plot_corr_matrix.shape[1] * 0.8, 6), # Min width 8, min height 6
                   max(6, plot_corr_matrix.shape[0] * 0.7, 5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(plot_corr_matrix,
                annot=plot_annot_text_matrix if annotate_values else None, # Pass the (potentially sorted) annotation matrix
                fmt=fmt_for_annot,  # Use fmt only if annotating with numbers
                cmap=cmap,
                center=center,
                linewidths=.5,
                linecolor='gainsboro',
                cbar_kws={'label': cbar_label, 'shrink': 0.8}, # Added shrink for better layout
                ax=ax,
                robust=True) # robust=True can help with outliers in colormap scaling
    
    ax.set_title(title, fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0, rect=[0, 0.03, 1, 0.95]) # Added rect for title space

    save_plot_and_data(fig, plot_corr_matrix, output_dir, plot_filename_base, is_summary_data=True)
    
    # Save the annotation text matrix if it was custom (i.e., likely contains stars or specific formatting)
    if fmt_for_annot == "" and plot_annot_text_matrix is not None and isinstance(plot_annot_text_matrix, pd.DataFrame):
        annot_data_path = os.path.join(output_dir, f"{plot_filename_base}_annotation_text_data.csv")
        try:
            plot_annot_text_matrix.to_csv(annot_data_path, index=True)
            print(f"  Annotation text matrix data saved: {annot_data_path}")
        except Exception as e_anotsave:
            print(f"    Warning: Could not save custom annotation text matrix: {e_anotsave}")
             
    plt.close(fig)


def plot_stacked_bar_composition(
    df_means_composition, # Should have groups as index, components as columns
    title, xlab, ylab, output_dir, plot_filename_base,
    group_counts_series=None, # Series with group names as index, counts as values
    legend_title='Component', colormap="Spectral", figsize=(10,7),
    bar_width=0.8, y_lim_max_factor=1.05 # Factor to extend y-axis beyond 100% if needed
):
    """Generates a stacked bar plot from a DataFrame of mean compositions."""
    if df_means_composition.empty:
        print(f"⚠️ plot_stacked_bar: df_means_composition for '{title}' empty. Skipping.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    df_means_composition.plot(kind='bar', stacked=True, ax=ax, colormap=colormap,
                              edgecolor="black", width=bar_width)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)

    # Prepare x-tick labels
    x_tick_labels = []
    plotted_groups = df_means_composition.index.tolist() # Groups that are actually plotted
    for group_name in plotted_groups:
        count_str = ""
        if group_counts_series is not None and group_name in group_counts_series:
            count_str = f"\n(n={int(group_counts_series.get(group_name, 0))})"
        x_tick_labels.append(f"{str(group_name).replace('_',' ')}{count_str}")
    
    ax.set_xticks(range(len(plotted_groups)))
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")

    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Sensible y-limits, assuming percentages sum to 100
    if "100%" in ylab or "Fraction (%)" in ylab or df_means_composition.sum(axis=1).max() <= 105: # Check if data is scaled to 100
        ax.set_ylim(0, 100 * y_lim_max_factor if y_lim_max_factor > 1 else 105) # e.g. 0 to 105 for %
    else: # Auto-scale if not percentage
        current_ylim_top = ax.get_ylim()[1]
        ax.set_ylim(0, current_ylim_top * 1.1)


    plt.tight_layout(rect=[0, 0, 0.82, 1] if legend_title else None) # Adjust for legend

    save_plot_and_data(fig, df_means_composition, output_dir, plot_filename_base, is_summary_data=True)
    plt.close(fig)


# --- F. Statistical Utility Functions ---

def apply_fdr_correction_df(df, p_value_col='p_value', q_value_col='FDR_q_value'):
    """Applies FDR (Benjamini-Hochberg) to a p_value column in a DataFrame."""
    if p_value_col not in df.columns or df[p_value_col].isna().all():
        df[q_value_col] = np.nan
        return df
    
    not_na_mask = df[p_value_col].notna()
    p_values_to_correct = df.loc[not_na_mask, p_value_col]
    
    if len(p_values_to_correct) == 0:
        df[q_value_col] = np.nan
        return df
        
    reject, q_values, _, _ = multipletests(p_values_to_correct, method='fdr_bh')
    
    df[q_value_col] = np.nan
    df.loc[not_na_mask, q_value_col] = q_values
    return df


# --- G. Signature Scoring Utility ---
def calculate_signature_score(adata_obj, gene_list, score_name, ctrl_size=50, use_raw=False, random_state=0, inplace=True):
    """Calculates signature score using scanpy.tl.score_genes with robust error/input handling."""
    available_genes = [gene for gene in gene_list if gene in adata_obj.var_names]
    
    if not available_genes:
        print(f"  ⚠️ No genes from list for '{score_name}' found in AnnData. Assigning NaN.")
        if inplace: adata_obj.obs[score_name] = np.nan
        else: return pd.Series(np.nan, index=adata_obj.obs_names)
        return

    effective_ctrl_size = 0 # Default for 1 gene or if issues
    if len(available_genes) > 1 :
        effective_ctrl_size = min(ctrl_size, len(available_genes) -1, 50) # Max 50, ensure less than available
        if effective_ctrl_size <= 0 and ctrl_size > 0 : effective_ctrl_size = 1 # Ensure at least 1 if original ctrl_size wasn't 0

    print(f"  Calculating '{score_name}' with {len(available_genes)} genes (ctrl_size={effective_ctrl_size}).")
    try:
        sc.tl.score_genes(adata_obj, gene_list=available_genes, score_name=score_name,
                          ctrl_size=effective_ctrl_size, use_raw=use_raw, random_state=random_state)
        if not inplace: return adata_obj.obs[score_name].copy()
    except Exception as e:
        print(f"  ⚠️ Error scoring '{score_name}': {e}. Assigning NaN.")
        if inplace: adata_obj.obs[score_name] = np.nan
        else: return pd.Series(np.nan, index=adata_obj.obs_names)


# --- H. TCGA Clinical Data Parsing Utilities (from original notebook, minor refinements) ---
NS_TCGA = {
    "admin": "http://tcga.nci/bcr/xml/administration/2.7", "shared": "http://tcga.nci/bcr/xml/shared/2.7",
    "clin_shared": "http://tcga.nci/bcr/xml/clinical/shared/2.7", "shared_stage": "http://tcga.nci/bcr/xml/clinical/shared/stage/2.7",
    "luad": "http://tcga.nci/bcr/xml/clinical/luad/2.7", "lung_shared": "http://tcga.nci/bcr/xml/clinical/shared/lung/2.7",
    "brca": "http://tcga.nci/bcr/xml/clinical/brca/2.7", "brca_shared": "http://tcga.nci/bcr/xml/clinical/brca/shared/2.7",
    "prad": "http://tcga.nci/bcr/xml/clinical/prad/2.7", "prad_shared": "http://tcga.nci/bcr/xml/clinical/prad/shared/2.7",
    "gbm": "http://tcga.nci/bcr/xml/clinical/gbm/2.7", "gbm_shared": "http://tcga.nci/bcr/xml/clinical/gbm/shared/2.7",
    "lgg": "http://tcga.nci/bcr/xml/clinical/lgg/2.7", "ov": "http://tcga.nci/bcr/xml/clinical/ov/2.7",
    "lihc": "http://tcga.nci/bcr/xml/clinical/lihc/2.7", "chol_lihc_shared": "http://tcga.nci/bcr/xml/clinical/shared/chol_lihc/2.7",
    "ucs": "http://tcga.nci/bcr/xml/clinical/ucs/2.7", "ucec_ucs_shared": "http://tcga.nci/bcr/xml/clinical/shared/ucec_ucs/2.7"
}
DISEASE_CONFIG_TCGA = {
    "LUAD": {"histology_path": ".//lung_shared:diagnosis", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": ".//shared:tobacco_smoking_history"},
    "BRCA": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "PRAD": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "GBM": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "LGG": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "OV": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "LIHC": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "UCS": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None},
    "DEFAULT": {"histology_path": ".//shared:histological_type", "tumor_site_path": ".//clin_shared:tumor_tissue_site", "smoking_history_path": None} # Changed default key
}

def _get_xml_text(root, xpath, namespaces, default_val="N/A"):
    """Internal helper for XML text extraction."""
    try:
        element = root.find(xpath, namespaces)
        if element is not None and element.text:
            return element.text.strip()
    except Exception: pass
    return default_val

def parse_tcga_clinical_xml_file(xml_file_path, ns_map=NS_TCGA, disease_config=DISEASE_CONFIG_TCGA):
    """Parses a single TCGA clinical XML file and returns a dictionary of data."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  ERROR: Could not parse XML: {os.path.basename(xml_file_path)}")
        return None

    disease_code_val = _get_xml_text(root, ".//admin:disease_code", ns_map, "UNKNOWN")
    cfg = disease_config.get(disease_code_val.upper(), disease_config["DEFAULT"])

    clinical_record = {
        "Patient_Barcode": _get_xml_text(root, ".//shared:bcr_patient_barcode", ns_map),
        "Patient_UUID": _get_xml_text(root, ".//shared:bcr_patient_uuid", ns_map),
        "Disease_Code": disease_code_val,
        "Gender": _get_xml_text(root, ".//shared:gender", ns_map),
        "Race": _get_xml_text(root, ".//clin_shared:race", ns_map),
        "Ethnicity": _get_xml_text(root, ".//clin_shared:ethnicity", ns_map),
        "Vital_Status": _get_xml_text(root, ".//clin_shared:vital_status", ns_map),
        "Age_at_Diagnosis": _get_xml_text(root, ".//clin_shared:age_at_initial_pathologic_diagnosis", ns_map),
        "Days_to_Birth": _get_xml_text(root, ".//clin_shared:days_to_birth", ns_map),
        "Days_to_Death": _get_xml_text(root, ".//clin_shared:days_to_death", ns_map),
        "Days_to_Last_Followup": _get_xml_text(root, ".//clin_shared:days_to_last_followup", ns_map),
        "Histology": _get_xml_text(root, cfg["histology_path"], ns_map),
        "Tumor_Site": _get_xml_text(root, cfg["tumor_site_path"], ns_map),
        "Pathologic_Stage_Raw": _get_xml_text(root, ".//shared_stage:pathologic_stage", ns_map), # Keep raw
        "Neoplasm_Status": _get_xml_text(root, ".//clin_shared:person_neoplasm_cancer_status", ns_map),
        "Informed_Consent": _get_xml_text(root, ".//clin_shared:informed_consent_verified", ns_map),
        "Neoadjuvant_Tx_History": _get_xml_text(root, ".//shared:history_of_neoadjuvant_treatment", ns_map),
        "XML_Filename": os.path.basename(xml_file_path)
    }
    if cfg.get("smoking_history_path"):
        clinical_record["Smoking_History"] = _get_xml_text(root, cfg["smoking_history_path"], ns_map)
    return clinical_record

print("--- Utility functions and TCGA parsing helpers defined. ---")

# %%
# ==============================================================================
# --- II. Data Loading and Initial Preprocessing (Revised for Consolidated Files v2) ---
# ==============================================================================
# This section loads ALL TCGA clinical data, a CONSOLIDATED sample sheet, and then
# filters this combined metadata for the specified CANCER_TYPE_ABBREV.
# RNA-seq counts are then loaded only for the samples relevant to this CANCER_TYPE_ABBREV.

# --- A. Load ALL Clinical Data from XML Files ---
print(f"\n--- II.A. Loading ALL Clinical Data from XMLs in {CLINICAL_XML_DIR} ---")
# This loads all clinical data available. Filtering by cancer type happens after merging with the sample sheet.

all_clinical_records = []
clinical_df_all_cancers = pd.DataFrame() # Initialize

if os.path.exists(CLINICAL_XML_DIR) and os.path.isdir(CLINICAL_XML_DIR):
    xml_files = [f for f in os.listdir(CLINICAL_XML_DIR) if f.lower().endswith(".xml")]
    if not xml_files:
        print(f"  WARNING: No XML files found in {CLINICAL_XML_DIR}.")
    else:
        print(f"  Found {len(xml_files)} XML files. Parsing (this may take a while)...")
        for i, filename in enumerate(xml_files):
            if (i + 1) % 200 == 0: print(f"    Parsed {i+1}/{len(xml_files)} XML files...")
            xml_path = os.path.join(CLINICAL_XML_DIR, filename)
            record = parse_tcga_clinical_xml_file(xml_path) # Utility from setup block
            if record: all_clinical_records.append(record)
        
        if all_clinical_records:
            clinical_df_all_cancers = pd.DataFrame(all_clinical_records)
            print(f"  Successfully parsed {len(clinical_df_all_cancers)} clinical records from all XML files.")
            print(f"  Overall Clinical DataFrame shape: {clinical_df_all_cancers.shape}")
            
            cols_to_numeric_clinical = ['Age_at_Diagnosis', 'Days_to_Birth', 'Days_to_Death', 'Days_to_Last_Followup']
            for col in cols_to_numeric_clinical:
                if col in clinical_df_all_cancers.columns:
                    clinical_df_all_cancers[col] = pd.to_numeric(clinical_df_all_cancers[col], errors='coerce')
            
            if 'Patient_Barcode' not in clinical_df_all_cancers.columns or clinical_df_all_cancers['Patient_Barcode'].isna().all():
                print("  CRITICAL WARNING: 'Patient_Barcode' is missing or all NaN in clinical_df_all_cancers.")
            else:
                # Remove potential duplicates based on Patient_Barcode from the clinical XMLs, keeping the first.
                # This can happen if multiple XML versions exist for the same patient.
                initial_clinical_rows = len(clinical_df_all_cancers)
                clinical_df_all_cancers.drop_duplicates(subset=['Patient_Barcode'], keep='first', inplace=True)
                if len(clinical_df_all_cancers) < initial_clinical_rows:
                    print(f"  Dropped {initial_clinical_rows - len(clinical_df_all_cancers)} duplicate Patient_Barcodes from clinical data.")
                print(f"  Unique Patient_Barcode count in clinical data: {clinical_df_all_cancers['Patient_Barcode'].notna().sum()}")
        else:
            print("  No clinical records were successfully parsed.")
else:
    print(f"  WARNING: Clinical XML directory not found: {CLINICAL_XML_DIR}")

# --- B. Load CONSOLIDATED Sample Sheet Metadata ---
print(f"\n--- II.B. Loading CONSOLIDATED Sample Sheet Metadata ---")
CONSOLIDATED_SAMPLE_SHEET_PATH = os.path.join(BASE_DATA_DIR, "gdc_sample_sheet.2025-06-20.tsv") # Path from your original notebook

sample_sheet_df_raw_unfiltered = pd.DataFrame() # Initialize
if os.path.exists(CONSOLIDATED_SAMPLE_SHEET_PATH):
    try:
        sample_sheet_df_raw_unfiltered = pd.read_csv(CONSOLIDATED_SAMPLE_SHEET_PATH, sep="\t", low_memory=False)
        print(f"  Successfully loaded consolidated sample sheet: {CONSOLIDATED_SAMPLE_SHEET_PATH}")
        print(f"  Raw unfiltered sample sheet shape: {sample_sheet_df_raw_unfiltered.shape}")

        # Basic processing of the raw sample sheet
        tissue_type_col_name = "Tissue Type" if "Tissue Type" in sample_sheet_df_raw_unfiltered.columns else "Sample Type"
        required_ss_cols_raw = {"Case ID", "File Name", tissue_type_col_name}

        if not required_ss_cols_raw.issubset(sample_sheet_df_raw_unfiltered.columns):
            print(f"  ERROR: Consolidated sample sheet missing required columns. Need: {list(required_ss_cols_raw)}.")
            sample_sheet_df_raw_unfiltered = pd.DataFrame() # Invalidate if critical cols missing
        else:
            # Select and rename essential columns
            sample_sheet_df_processed_unfiltered = sample_sheet_df_raw_unfiltered[list(required_ss_cols_raw) + ["Project ID"]].copy() # Keep Project ID
            sample_sheet_df_processed_unfiltered.rename(columns={
                "Case ID": "Patient_ID_from_SampleSheet", # This is typically TCGA-XX-YYYY
                tissue_type_col_name: "Tissue_Type",
                "File Name": "Original_File_Name", # Keep original for reference
                "Project ID": "Project_ID" # e.g., TCGA-BRCA
            }, inplace=True)
            sample_sheet_df_processed_unfiltered["File_Name_Root"] = sample_sheet_df_processed_unfiltered["Original_File_Name"].apply(lambda x: str(x).split(".")[0])
            
            # Drop duplicates based on File_Name_Root to ensure one metadata entry per unique RNA-seq file
            initial_rows_ss = len(sample_sheet_df_processed_unfiltered)
            sample_sheet_df_processed_unfiltered.drop_duplicates(subset=["File_Name_Root"], keep='first', inplace=True)
            if len(sample_sheet_df_processed_unfiltered) < initial_rows_ss:
                 print(f"  Dropped {initial_rows_ss - len(sample_sheet_df_processed_unfiltered)} duplicate File_Name_Root entries from the consolidated sample sheet.")
            print(f"  Processed unfiltered sample sheet shape: {sample_sheet_df_processed_unfiltered.shape}")
            sample_sheet_df_raw_unfiltered = sample_sheet_df_processed_unfiltered # Update variable name
            
    except Exception as e:
        print(f"  ERROR loading/processing consolidated sample sheet {CONSOLIDATED_SAMPLE_SHEET_PATH}: {e}")
else:
    print(f"  WARNING: Consolidated sample sheet file not found: {CONSOLIDATED_SAMPLE_SHEET_PATH}")


# --- C. Create Master Metadata: Merge ALL Clinical and ALL Sample Sheet Data ---
print(f"\n--- II.C. Creating Master Metadata (All Samples) ---")
master_metadata_df = pd.DataFrame() # Initialize
if not clinical_df_all_cancers.empty and not sample_sheet_df_raw_unfiltered.empty:
    # Merge based on patient identifiers.
    # Patient_ID_from_SampleSheet (from "Case ID") should be TCGA-XX-YYYY
    # Patient_Barcode from clinical_df_all_cancers is also TCGA-XX-YYYY
    master_metadata_df = pd.merge(
        sample_sheet_df_raw_unfiltered, # Contains all samples from all projects in the sheet
        clinical_df_all_cancers,      # Contains all clinical data parsed
        left_on="Patient_ID_from_SampleSheet",
        right_on="Patient_Barcode",
        how="left" # Keep all samples from the sample sheet, add clinical data if available
    )
    print(f"  Master metadata_df (all samples, pre-filtering) shape: {master_metadata_df.shape}")
    
    # Define a primary Patient_ID (preferring from sample sheet as it's the link to files)
    if 'Patient_ID_from_SampleSheet' in master_metadata_df.columns:
         master_metadata_df.rename(columns={'Patient_ID_from_SampleSheet': 'Patient_ID'}, inplace=True)
    elif 'Patient_Barcode' in master_metadata_df.columns: # Fallback if merge happened differently
         master_metadata_df.rename(columns={'Patient_Barcode': 'Patient_ID'}, inplace=True)

    # Create a 'Cancer_Type_Derived' column for filtering.
    # Prioritize Project_ID from sample sheet if it exists and is reliable.
    # Fallback to Disease_Code from clinical data if Project_ID is missing/unreliable.
    if 'Project_ID' in master_metadata_df.columns:
        # Assuming Project_ID is like 'TCGA-BRCA', extract 'BRCA'
        master_metadata_df['Cancer_Type_Derived'] = master_metadata_df['Project_ID'].str.split('-').str[1].str.upper()
        print("  Derived 'Cancer_Type_Derived' from 'Project_ID'.")
    elif 'Disease_Code' in master_metadata_df.columns: # Disease_Code comes from clinical_df
        master_metadata_df['Cancer_Type_Derived'] = master_metadata_df['Disease_Code'].str.upper()
        print("  Derived 'Cancer_Type_Derived' from 'Disease_Code' (clinical XML).")
    else:
        print("  WARNING: Neither 'Project_ID' nor 'Disease_Code' available to derive Cancer_Type reliably for all samples.")
        master_metadata_df['Cancer_Type_Derived'] = 'UNKNOWN'
    
    print("  'Cancer_Type_Derived' distribution in master_metadata_df:")
    print(master_metadata_df['Cancer_Type_Derived'].value_counts(dropna=False).head(10)) # Show top 10

    if 'File_Name_Root' not in master_metadata_df.columns or master_metadata_df['File_Name_Root'].isna().all():
        print("  CRITICAL WARNING: 'File_Name_Root' missing or all NaN in master_metadata_df.")
    else:
        # Check for duplicates in File_Name_Root *before* setting index, though earlier drop should handle it.
        if master_metadata_df['File_Name_Root'].duplicated().any():
            print("  WARNING: Duplicate 'File_Name_Root' in master_metadata_df after merge. Deduplicating again (should not happen if prior steps correct).")
            master_metadata_df.drop_duplicates(subset=['File_Name_Root'], keep='first', inplace=True)
        master_metadata_df.set_index('File_Name_Root', inplace=True, verify_integrity=True)
        print("  Set 'File_Name_Root' as index for master_metadata_df.")

elif not sample_sheet_df_raw_unfiltered.empty: # Only sample sheet data available
    print("  Using only sample sheet data for master_metadata as clinical data was empty.")
    master_metadata_df = sample_sheet_df_raw_unfiltered.copy()
    if 'Patient_ID_from_SampleSheet' in master_metadata_df.columns:
         master_metadata_df.rename(columns={'Patient_ID_from_SampleSheet': 'Patient_ID'}, inplace=True)
    if 'Project_ID' in master_metadata_df.columns:
        master_metadata_df['Cancer_Type_Derived'] = master_metadata_df['Project_ID'].str.split('-').str[1].str.upper()
    else: master_metadata_df['Cancer_Type_Derived'] = 'UNKNOWN'
    
    if 'File_Name_Root' in master_metadata_df.columns:
        if master_metadata_df['File_Name_Root'].duplicated().any():
             master_metadata_df.drop_duplicates(subset=['File_Name_Root'], keep='first', inplace=True)
        master_metadata_df.set_index('File_Name_Root', inplace=True, verify_integrity=True)
    else: print("  CRITICAL WARNING: 'File_Name_Root' not in master_metadata_df (from sample_sheet only).")
else:
    print("  ERROR: Both all_clinical_df and unfiltered_sample_sheet are empty. Cannot create master metadata.")


# --- D. Filter Master Metadata for the Target Cancer Type ---
print(f"\n--- II.D. Filtering Master Metadata for {CANCER_TYPE_ABBREV} ---")
metadata_df = pd.DataFrame() # This will be the cancer-specific metadata
if not master_metadata_df.empty:
    if 'Cancer_Type_Derived' in master_metadata_df.columns:
        metadata_df = master_metadata_df[master_metadata_df['Cancer_Type_Derived'] == CANCER_TYPE_ABBREV.upper()].copy()
        if metadata_df.empty:
            print(f"  WARNING: No records found in master_metadata_df for Cancer_Type_Derived = '{CANCER_TYPE_ABBREV.upper()}'.")
            print(f"  Available cancer types in master metadata: {master_metadata_df['Cancer_Type_Derived'].unique()}")
        else:
            print(f"  Filtered metadata for {CANCER_TYPE_ABBREV}. Shape: {metadata_df.shape}")
            print(metadata_df.head(3))
    else:
        print(f"  WARNING: 'Cancer_Type_Derived' column not present in master_metadata_df. Cannot filter for {CANCER_TYPE_ABBREV}.")
else:
    print(f"  Master metadata is empty, cannot filter for {CANCER_TYPE_ABBREV}.")


# --- E. Load RNA-Seq Gene Count Data (Filtered by metadata_df for the current CANCER_TYPE_ABBREV) ---
print(f"\n--- II.E. Loading RNA-Seq Gene Count Data for {CANCER_TYPE_ABBREV} (Filtered) ---")
GENERAL_RNA_SEQ_DIR = os.path.join(BASE_DATA_DIR, "rna") # From your original notebook path

rna_counts_df = pd.DataFrame() # Initialize
if not metadata_df.empty and metadata_df.index.name == 'File_Name_Root':
    valid_file_roots_for_this_cancer = set(metadata_df.index) # These are the File_Name_Root for the current cancer
    
    if not valid_file_roots_for_this_cancer:
        print(f"  No valid File_Name_Roots identified in metadata_df for {CANCER_TYPE_ABBREV} to select RNA-seq files.")
    elif os.path.exists(GENERAL_RNA_SEQ_DIR) and os.path.isdir(GENERAL_RNA_SEQ_DIR):
        all_sample_counts_list = []
        all_rna_files_in_general_dir = [f for f in os.listdir(GENERAL_RNA_SEQ_DIR) if f.lower().endswith((".tsv", ".txt", ".gz"))]
        
        print(f"  Found {len(all_rna_files_in_general_dir)} total files in general RNA directory: {GENERAL_RNA_SEQ_DIR}.")
        print(f"  Filtering these for the {len(valid_file_roots_for_this_cancer)} samples relevant to {CANCER_TYPE_ABBREV}...")
        
        rna_files_to_process_for_cancer = []
        for filename in all_rna_files_in_general_dir:
            sample_name_from_file_root = filename.split(".")[0] # Assumes root of filename matches File_Name_Root
            if sample_name_from_file_root in valid_file_roots_for_this_cancer:
                rna_files_to_process_for_cancer.append(filename)
        
        if not rna_files_to_process_for_cancer:
            print(f"  WARNING: No RNA-seq files in {GENERAL_RNA_SEQ_DIR} match the File_Name_Roots for {CANCER_TYPE_ABBREV} samples.")
        else:
            print(f"  Identified {len(rna_files_to_process_for_cancer)} RNA-seq files for {CANCER_TYPE_ABBREV} to process.")
            processed_count = 0
            for filename in rna_files_to_process_for_cancer:
                file_path = os.path.join(GENERAL_RNA_SEQ_DIR, filename)
                sample_name_from_file_root = filename.split(".")[0]
                try:
                    df_sample = pd.read_csv(file_path, sep="\t", comment="#", header=0, low_memory=False)
                    if "gene_name" not in df_sample.columns: 
                        print(f"    Skipping {filename}: 'gene_name' col missing.")
                        continue
                    
                    count_col_to_use = None
                    PREFERRED_COUNT_COLUMN = "tpm_unstranded"
                    if PREFERRED_COUNT_COLUMN in df_sample.columns: 
                        count_col_to_use = PREFERRED_COUNT_COLUMN
                    else:
                        numeric_cols_cand = df_sample.select_dtypes(include=np.number).columns
                        if len(numeric_cols_cand) > 0: count_col_to_use = numeric_cols_cand[0]
                    
                    if not count_col_to_use: 
                        print(f"    Skipping {filename}: No suitable count col found.")
                        continue
                        
                    df_sample = df_sample[df_sample["gene_name"].notna() & ~df_sample["gene_name"].astype(str).str.upper().str.startswith("N_")]
                    if df_sample.empty:
                        print(f"    Skipping {filename}: Empty after gene filtering.")
                        continue
                        
                    df_sample = df_sample.set_index("gene_name")[[count_col_to_use]]
                    df_sample.rename(columns={count_col_to_use: sample_name_from_file_root}, inplace=True)
                    all_sample_counts_list.append(df_sample)
                    processed_count += 1
                    if processed_count % 50 == 0 or processed_count == len(rna_files_to_process_for_cancer):
                        print(f"    Processed {processed_count}/{len(rna_files_to_process_for_cancer)} RNA files for {CANCER_TYPE_ABBREV}...")
                except Exception as e: print(f"    ERROR processing RNA file {filename}: {e}")
            
            if all_sample_counts_list:
                # 1. Concatenate all sample data into a single DataFrame
                raw_rna_counts_df = pd.concat(all_sample_counts_list, axis=1, join='outer').fillna(0)
                raw_rna_counts_df.index.name = "Gene_Symbol"
                
                # 2. (NEW) Check for and aggregate duplicate gene symbols
                if raw_rna_counts_df.index.duplicated().any():
                    num_duplicates = raw_rna_counts_df.index.duplicated().sum()
                    print(f"  Found {num_duplicates} duplicate gene symbols in the source RNA data. Aggregating by mean expression.")
                    print(f"    Original shape with duplicates: {raw_rna_counts_df.shape}")
                    
                    # The key fix: Group by gene symbol (index) and take the mean
                    rna_counts_df = raw_rna_counts_df.groupby(raw_rna_counts_df.index).mean()
                    
                    print(f"    New shape after aggregation: {rna_counts_df.shape}")
                else:
                    # If no duplicates, just use the original dataframe
                    print("  No duplicate gene symbols found in the source RNA data.")
                    rna_counts_df = raw_rna_counts_df
                    
                print(f"  Final RNA counts matrix for {CANCER_TYPE_ABBREV} shape: {rna_counts_df.shape} (Genes x Samples)")
            else: 
                print(f"  No RNA-seq data successfully processed for {CANCER_TYPE_ABBREV}.")
    elif metadata_df.empty:
        print(f"  Skipping RNA-seq loading as filtered metadata for {CANCER_TYPE_ABBREV} is empty.")
    else: # Should not happen if metadata_df index set correctly
        print(f"  WARNING: General RNA-seq directory not found ({GENERAL_RNA_SEQ_DIR}) or metadata_df index issue.")


# --- F. Create Final AnnData Object for the specific CANCER_TYPE_ABBREV ---
print(f"\n--- II.F. Creating Final AnnData Object for {CANCER_TYPE_ABBREV} ---")
adata = None # Initialize the cancer-specific AnnData object
if not rna_counts_df.empty and not metadata_df.empty: # metadata_df is now cancer-specific
    # Align rna_counts_df columns (samples) with metadata_df index (File_Name_Root)
    # Both should now only contain samples relevant to CANCER_TYPE_ABBREV
    common_samples_final = metadata_df.index.intersection(rna_counts_df.columns)
    print(f"  Found {len(common_samples_final)} common samples between {CANCER_TYPE_ABBREV}-specific metadata and RNA counts.")

    if len(common_samples_final) == 0:
        print(f"  CRITICAL ERROR: No common samples for {CANCER_TYPE_ABBREV}. Cannot create AnnData object.")
    else:
        # Ensure order matches for AnnData creation
        rna_counts_final_aligned = rna_counts_df[common_samples_final].copy() # Genes x Samples
        metadata_final_aligned = metadata_df.loc[common_samples_final].copy() # Samples x Features

        adata = sc.AnnData(X=rna_counts_final_aligned.T.astype(np.float32)) # Samples x Genes
        adata.obs = metadata_final_aligned
        # adata.var.index.name = "Gene_Symbol" # var_names are gene symbols from rna_counts_final_aligned.index
        
        print(f"  Successfully created AnnData object (adata) for {CANCER_TYPE_ABBREV}.")
        print(f"  adata shape: {adata.shape} (Samples x Genes)")
        adata.var_names_make_unique()
        print("  Made AnnData.var_names unique.")
        if "TUSC2" in adata.var_names: print("  Gene 'TUSC2' is present in adata.")
        else: print("  WARNING: Gene 'TUSC2' is NOT present in adata.var_names for current cancer type.")
elif rna_counts_df.empty: print(f"  ERROR: RNA counts for {CANCER_TYPE_ABBREV} empty. AnnData not created.")
elif metadata_df.empty: print(f"  ERROR: Filtered metadata for {CANCER_TYPE_ABBREV} empty. AnnData not created.")
else: print(f"  ERROR: Unknown state for {CANCER_TYPE_ABBREV}. AnnData not created.")

print(f"\n--- End of Section II: Data Loading and Initial Preprocessing for {CANCER_TYPE_ABBREV} ---")

# %%
# ==============================================================================
# --- III. AnnData Object Core Preprocessing & Filtering (for cancer-specific adata) ---
# ==============================================================================
# This section performs log transformation, gene filtering, and creates tumor_adata.

if adata is not None:
    print(f"\n--- III.A. Log Transformation for {CANCER_TYPE_ABBREV} adata ---")
    # Data is typically TPM. Log-transforming (e.g., log2(TPM+1)) is common.
    # Check if data looks like it's already logged (e.g. max value, distribution)
    # For now, assuming it needs transformation as per original notebook's later steps.
    if adata.X.max() > 25: # Heuristic: if max TPM is high, it's likely not logged
        print("  Applying log2(X+1) transformation to adata.X")
        sc.pp.log1p(adata, base=2)
        # Optional: store the unlogged data if needed for some tools, though this uses more memory
        # if 'counts_raw' not in adata.layers:
        #     print("  Raw data not previously stored in layer. If needed, store before log transform.")
    else:
        print("  Data in adata.X seems to be on a smaller scale (max <= 100). Assuming already transformed or نیازی به تبدیل ندارد.")

    print(f"\n--- III.B. Gene Filtering for {CANCER_TYPE_ABBREV} adata ---")
    # Original notebook used min_cells=5. This is a common simple filter.
    # Highly variable gene selection is another option for dimensionality reduction if needed later.
    n_genes_before_filter = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=5) # Keep genes expressed in at least 5 samples
    print(f"  Filtered genes: kept {adata.n_vars} out of {n_genes_before_filter} (min 5 cells).")
    if "TUSC2" not in adata.var_names:
        print(f"  WARNING: TUSC2 was filtered out during gene filtering for {CANCER_TYPE_ABBREV}! Check expression levels if this is unexpected.")
    
    print(f"\n--- III.C. Creating tumor_adata for {CANCER_TYPE_ABBREV} ---")
    # Filter adata for tumor samples only, based on 'Tissue_Type'
    # Ensure 'Tissue_Type' column exists.
    if 'Tissue_Type' in adata.obs.columns:
        # Common TCGA tissue types for tumor: "Primary Tumor", "Solid Tissue Normal", "Metastatic"
        # We are interested in "Tumor" samples. The original notebook used .str.contains("Tumor", case=False).
        # Let's be a bit more specific if possible, e.g., "Primary Tumor".
        # For now, using the original logic for broader match.
        
        # Make a copy for tumor_adata to avoid modifying adata directly if adata might be used for other purposes.
        tumor_adata = adata[adata.obs['Tissue_Type'].astype(str).str.contains("Tumor", case=False, na=False)].copy()
        
        if tumor_adata.n_obs > 0:
            print(f"  Created tumor_adata with {tumor_adata.n_obs} samples (from {adata.n_obs} total samples in adata).")
            
            # Standardize cancer type column
            # Prefer 'Cancer_Type_Derived' if it exists, then 'Disease_Code', then fall back to CANCER_TYPE_ABBREV
            authoritative_cancer_col = None
            if 'Cancer_Type_Derived' in tumor_adata.obs.columns and tumor_adata.obs['Cancer_Type_Derived'].notna().any():
                authoritative_cancer_col = 'Cancer_Type_Derived'
            elif 'Disease_Code' in tumor_adata.obs.columns and tumor_adata.obs['Disease_Code'].notna().any(): # From clinical XML
                authoritative_cancer_col = 'Disease_Code'
            
            if authoritative_cancer_col:
                tumor_adata.obs['Cancer_Type'] = tumor_adata.obs[authoritative_cancer_col].astype(str).str.strip().str.upper()
                print(f"  Standardized 'Cancer_Type' in tumor_adata.obs using column '{authoritative_cancer_col}'.")
                # If the original authoritative column is different and no longer needed as primary, it could be dropped or kept.
                # if authoritative_cancer_col != 'Cancer_Type':
                #     tumor_adata.obs.drop(columns=[authoritative_cancer_col], inplace=True, errors='ignore')

            elif 'Cancer_Type' in tumor_adata.obs.columns: # If 'Cancer_Type' already exists somehow
                 tumor_adata.obs['Cancer_Type'] = tumor_adata.obs['Cancer_Type'].astype(str).str.strip().str.upper()
                 print("  Standardized existing 'Cancer_Type' in tumor_adata.obs.")
            else: # Fallback if no clear source, assign based on the script's context
                tumor_adata.obs['Cancer_Type'] = CANCER_TYPE_ABBREV.upper()
                print(f"  Assigned '{CANCER_TYPE_ABBREV.upper()}' to 'Cancer_Type' in tumor_adata.obs as a fallback.")

            print(f"  Final 'Cancer_Type' distribution in tumor_adata for {CANCER_TYPE_ABBREV}:")
            print(tumor_adata.obs['Cancer_Type'].value_counts(dropna=False))
            
            # Verification
            if not (tumor_adata.obs['Cancer_Type'] == CANCER_TYPE_ABBREV.upper()).all():
                print(f"  WARNING: tumor_adata's 'Cancer_Type' column contains values other than {CANCER_TYPE_ABBREV.upper()}. This might indicate upstream data mixing or incorrect filtering if only one cancer type is expected at this stage.")
        
        else:
            print(f"  WARNING: No tumor samples found for {CANCER_TYPE_ABBREV}. tumor_adata is empty.")
            tumor_adata = None 
    else:
        print("  WARNING: 'Tissue_Type' column not found in adata.obs. Cannot create tumor_adata.")
        tumor_adata = None
else:
    print(f"  adata object for {CANCER_TYPE_ABBREV} is None. Skipping Section III.")
    tumor_adata = None # Ensure tumor_adata is None if adata is None

print(f"\n--- End of Section III: AnnData Core Preprocessing for {CANCER_TYPE_ABBREV} ---")

# ==============================================================================
# --- IV. CIBERSORTx Data Loading and Integration with tumor_adata ---
# ==============================================================================
# This section loads CIBERSORTx LM22 and Rebuffet NK fractions, and HiRes gene expression,
# applies quality filters, and merges them into tumor_adata.obs.

if tumor_adata is not None:
    print(f"\n--- IV.A. Loading and Filtering CIBERSORTx LM22 Cell Fractions for {CANCER_TYPE_ABBREV} ---")
    # CIBERSORT_LM22_PATH is defined in the config section using CANCER_TYPE_ABBREV
    
    cibersort_lm22_df_filtered = pd.DataFrame() # Initialize
    if os.path.exists(CIBERSORT_LM22_PATH):
        try:
            cibersort_lm22_df_raw = pd.read_csv(CIBERSORT_LM22_PATH, sep="\t", index_col="Mixture") # Mixture = Sample ID
            print(f"  Loaded CIBERSORTx LM22 fractions from: {CIBERSORT_LM22_PATH} (Shape: {cibersort_lm22_df_raw.shape})")
            initial_lm22_samples = len(cibersort_lm22_df_raw)
            cibersort_lm22_df_filtered = cibersort_lm22_df_raw.copy()

            # 1. P-value filter
            if "P-value" in cibersort_lm22_df_filtered.columns:
                pval_mask_lm22 = cibersort_lm22_df_filtered["P-value"] < P_VALUE_THRESHOLD_CIBERSORT
                cibersort_lm22_df_filtered = cibersort_lm22_df_filtered[pval_mask_lm22]
                print(f"    LM22: Kept {len(cibersort_lm22_df_filtered)} of {initial_lm22_samples} samples after P-value < {P_VALUE_THRESHOLD_CIBERSORT} filter.")
            else: print("    LM22: 'P-value' column not found. Skipping P-value filter.")

            # 2. Correlation filter
            if "Correlation" in cibersort_lm22_df_filtered.columns and CORRELATION_THRESHOLD_CIBERSORT is not None:
                samples_before_r_lm22 = len(cibersort_lm22_df_filtered)
                corr_mask_lm22 = cibersort_lm22_df_filtered["Correlation"] > CORRELATION_THRESHOLD_CIBERSORT
                cibersort_lm22_df_filtered = cibersort_lm22_df_filtered[corr_mask_lm22]
                print(f"    LM22: Kept {len(cibersort_lm22_df_filtered)} of {samples_before_r_lm22} samples after Correlation > {CORRELATION_THRESHOLD_CIBERSORT} filter.")
            elif "Correlation" not in cibersort_lm22_df_filtered.columns: print("    LM22: 'Correlation' column not found. Skipping Correlation filter.")

            # 3. RMSE filter
            if "RMSE" in cibersort_lm22_df_filtered.columns and not cibersort_lm22_df_filtered.empty:
                samples_before_rmse_lm22 = len(cibersort_lm22_df_filtered)
                if samples_before_rmse_lm22 > 0 and cibersort_lm22_df_filtered["RMSE"].notna().any():
                    rmse_thresh_val_lm22 = cibersort_lm22_df_filtered["RMSE"].quantile(RMSE_PERCENTILE_THRESHOLD_CIBERSORT)
                    rmse_mask_lm22 = cibersort_lm22_df_filtered["RMSE"] < rmse_thresh_val_lm22
                    cibersort_lm22_df_filtered = cibersort_lm22_df_filtered[rmse_mask_lm22]
                    print(f"    LM22: Kept {len(cibersort_lm22_df_filtered)} of {samples_before_rmse_lm22} samples after RMSE < {rmse_thresh_val_lm22:.4f} ({RMSE_PERCENTILE_THRESHOLD_CIBERSORT*100:.0f}th percentile) filter.")
                else: print("    LM22: Not enough data or no valid RMSE values for RMSE threshold calculation.")
            elif "RMSE" not in cibersort_lm22_df_filtered.columns: print("    LM22: 'RMSE' column not found. Skipping RMSE filter.")
            
            # Drop metric columns
            lm22_cols_to_drop = ["P-value", "Correlation", "RMSE", "Absolute score (sig.score)"]
            existing_lm22_cols_to_drop = [col for col in lm22_cols_to_drop if col in cibersort_lm22_df_filtered.columns]
            if existing_lm22_cols_to_drop:
                cibersort_lm22_df_filtered = cibersort_lm22_df_filtered.drop(columns=existing_lm22_cols_to_drop)
                print(f"    LM22: Dropped metric columns: {existing_lm22_cols_to_drop}")
            
            if cibersort_lm22_df_filtered.empty:
                print(f"    LM22: No samples remaining after filtering for {CANCER_TYPE_ABBREV}.")

        except FileNotFoundError:
            print(f"  ERROR: CIBERSORTx LM22 file not found: {CIBERSORT_LM22_PATH}")
        except Exception as e:
            print(f"  ERROR: Could not load or process CIBERSORTx LM22 file {CIBERSORT_LM22_PATH}: {e}")
    else:
        print(f"  WARNING: CIBERSORTx LM22 file path does not exist: {CIBERSORT_LM22_PATH}")


    print(f"\n--- IV.B. Loading and Filtering CIBERSORTx Rebuffet NK Phenotype Fractions for {CANCER_TYPE_ABBREV} ---")
    # CIBERSORT_REBUFFET_PATH defined in config
    cibersort_rebuffet_df_filtered = pd.DataFrame() # Initialize
    if os.path.exists(CIBERSORT_REBUFFET_PATH):
        try:
            cibersort_rebuffet_df_raw = pd.read_csv(CIBERSORT_REBUFFET_PATH, sep="\t", index_col="Mixture")
            print(f"  Loaded CIBERSORTx Rebuffet NK fractions from: {CIBERSORT_REBUFFET_PATH} (Shape: {cibersort_rebuffet_df_raw.shape})")
            initial_rebuffet_samples = len(cibersort_rebuffet_df_raw)
            cibersort_rebuffet_df_filtered = cibersort_rebuffet_df_raw.copy()

            # Apply filters (similar to LM22)
            if "P-value" in cibersort_rebuffet_df_filtered.columns:
                pval_mask_reb = cibersort_rebuffet_df_filtered["P-value"] < P_VALUE_THRESHOLD_CIBERSORT
                cibersort_rebuffet_df_filtered = cibersort_rebuffet_df_filtered[pval_mask_reb]
                print(f"    Rebuffet: Kept {len(cibersort_rebuffet_df_filtered)} of {initial_rebuffet_samples} samples after P-value filter.")
            if "Correlation" in cibersort_rebuffet_df_filtered.columns and CORRELATION_THRESHOLD_CIBERSORT is not None:
                samples_before_r_reb = len(cibersort_rebuffet_df_filtered)
                corr_mask_reb = cibersort_rebuffet_df_filtered["Correlation"] > CORRELATION_THRESHOLD_CIBERSORT
                cibersort_rebuffet_df_filtered = cibersort_rebuffet_df_filtered[corr_mask_reb]
                print(f"    Rebuffet: Kept {len(cibersort_rebuffet_df_filtered)} of {samples_before_r_reb} samples after Correlation filter.")
            if "RMSE" in cibersort_rebuffet_df_filtered.columns and not cibersort_rebuffet_df_filtered.empty:
                samples_before_rmse_reb = len(cibersort_rebuffet_df_filtered)
                if samples_before_rmse_reb > 0 and cibersort_rebuffet_df_filtered["RMSE"].notna().any():
                    rmse_thresh_val_reb = cibersort_rebuffet_df_filtered["RMSE"].quantile(RMSE_PERCENTILE_THRESHOLD_CIBERSORT)
                    rmse_mask_reb = cibersort_rebuffet_df_filtered["RMSE"] < rmse_thresh_val_reb
                    cibersort_rebuffet_df_filtered = cibersort_rebuffet_df_filtered[rmse_mask_reb]
                    print(f"    Rebuffet: Kept {len(cibersort_rebuffet_df_filtered)} of {samples_before_rmse_reb} samples after RMSE filter.")
            
            rebuffet_cols_to_drop = ["P-value", "Correlation", "RMSE", "Absolute score (sig.score)"]
            existing_reb_cols_to_drop = [col for col in rebuffet_cols_to_drop if col in cibersort_rebuffet_df_filtered.columns]
            if existing_reb_cols_to_drop:
                cibersort_rebuffet_df_filtered = cibersort_rebuffet_df_filtered.drop(columns=existing_reb_cols_to_drop)
                print(f"    Rebuffet: Dropped metric columns: {existing_reb_cols_to_drop}")

            # Rename Rebuffet columns to avoid clashes if any shared names (e.g. "NK cells")
            rename_dict_reb = {col: f"Rebuffet_{col}" for col in cibersort_rebuffet_df_filtered.columns}
            cibersort_rebuffet_df_filtered.rename(columns=rename_dict_reb, inplace=True)
            print(f"    Rebuffet: Renamed columns with 'Rebuffet_' prefix. New example columns: {list(cibersort_rebuffet_df_filtered.columns[:3])}")
            
            if cibersort_rebuffet_df_filtered.empty:
                print(f"    Rebuffet: No samples remaining after filtering for {CANCER_TYPE_ABBREV}.")

        except FileNotFoundError: print(f"  ERROR: CIBERSORTx Rebuffet file not found: {CIBERSORT_REBUFFET_PATH}")
        except Exception as e: print(f"  ERROR: Could not load/process CIBERSORTx Rebuffet file {CIBERSORT_REBUFFET_PATH}: {e}")
    else: print(f"  WARNING: CIBERSORTx Rebuffet file path does not exist: {CIBERSORT_REBUFFET_PATH}")


    print(f"\n--- IV.C. Merging Filtered CIBERSORTx Fractions into tumor_adata.obs ---")
    # The index of CIBERSORTx dataframes should be Sample IDs that match tumor_adata.obs_names (File_Name_Root)
    # If CIBERSORTx used different sample IDs (e.g. full TCGA barcodes), alignment step would be needed here.
    # Assuming CIBERSORTx 'Mixture' column (index) matches `tumor_adata.obs_names`.

    # Align CIBERSORTx data with tumor_adata samples
    # Align CIBERSORTx data with tumor_adata samples
    common_obs_names = tumor_adata.obs_names
    
    # Initialize DataFrames that will be merged
    lm22_to_merge = pd.DataFrame()
    if not cibersort_lm22_df_filtered.empty:
        lm22_to_merge = cibersort_lm22_df_filtered.reindex(common_obs_names).dropna(how='all')

    rebuffet_to_merge = pd.DataFrame()
    if not cibersort_rebuffet_df_filtered.empty:
        rebuffet_to_merge = cibersort_rebuffet_df_filtered.reindex(common_obs_names).dropna(how='all')

    # --- LM22 Merge and Verification ---
    if not lm22_to_merge.empty:
        original_lm22_cols_in_merge = lm22_to_merge.columns.tolist() # Get cols before join
        tumor_adata.obs = tumor_adata.obs.join(lm22_to_merge, how='left')
        print(f"  Joined LM22 fractions to tumor_adata.obs. Added columns: {original_lm22_cols_in_merge[:3]}...")

        # VERIFY LM22 PER-SAMPLE SUMS
        if original_lm22_cols_in_merge:
            for col_sum_check in original_lm22_cols_in_merge:
                if col_sum_check in tumor_adata.obs:
                    tumor_adata.obs[col_sum_check] = pd.to_numeric(tumor_adata.obs[col_sum_check], errors='coerce')
            
            sum_check_df_lm22 = tumor_adata.obs[original_lm22_cols_in_merge].dropna(how='all')
            if not sum_check_df_lm22.empty:
                sum_check_series_lm22 = sum_check_df_lm22.sum(axis=1)
                print("  LM22 Fraction Sum Check (for samples with LM22 data, sums of original fractions):")
                print(sum_check_series_lm22.describe())
                if not np.allclose(sum_check_series_lm22.dropna(), 1.0, atol=0.01):
                    print("  WARNING: Sum of original LM22 fractions per sample is not consistently close to 1.0!")
                else:
                    print("  Original LM22 fractions per sample (where present) appear to sum to ~1.0.")
            else:
                print("  No samples with non-NaN LM22 data in tumor_adata.obs to perform sum check.")
        else:
            print("  No LM22 columns were identified from lm22_to_merge for sum check.")
    else: 
        print("  LM22 data (lm22_to_merge) was empty after aligning. Nothing to join or verify for LM22.")

    # --- Rebuffet Merge and Verification ---
    if not rebuffet_to_merge.empty:
        # These are the columns that will be joined (already prefixed)
        rebuffet_cols_to_be_joined = rebuffet_to_merge.columns.tolist() 
            
        tumor_adata.obs = tumor_adata.obs.join(rebuffet_to_merge, how='left') 
        print(f"  Joined Rebuffet NK fractions to tumor_adata.obs. Added columns: {rebuffet_cols_to_be_joined[:3]}...")
            
        # VERIFY REBUFFET PER-SAMPLE SUMS (using the actual joined columns)
        if rebuffet_cols_to_be_joined: # Check if there are columns to sum
            for col_sum_check in rebuffet_cols_to_be_joined:
                 if col_sum_check in tumor_adata.obs: # Check if join was successful for this col in tumor_adata
                    tumor_adata.obs[col_sum_check] = pd.to_numeric(tumor_adata.obs[col_sum_check], errors='coerce')
            
            sum_check_df_reb = tumor_adata.obs[rebuffet_cols_to_be_joined].dropna(how='all')
            if not sum_check_df_reb.empty:
                sum_check_series_reb = sum_check_df_reb.sum(axis=1)
                print("  Rebuffet NK Fraction Sum Check per sample (expect ~1.0 if relative mode across these phenotypes):")
                print(sum_check_series_reb.describe())
                if not np.allclose(sum_check_series_reb.dropna(), 1.0, atol=0.01): 
                    print("  WARNING: Sum of original Rebuffet NK fractions per sample is not consistently close to 1.0!")
                else:
                    print("  Original Rebuffet NK fractions per sample (where present) appear to sum to ~1.0.")
            else:
                print("  No samples with non-NaN Rebuffet data in tumor_adata.obs to perform sum check.")
        else:
            print("  No Rebuffet columns were identified from rebuffet_to_merge for sum check.")
    else: 
        print("  Rebuffet data (rebuffet_to_merge) was empty after aligning. Nothing to join or verify for Rebuffet.")

    if not cibersort_rebuffet_df_filtered.empty: # Check if Rebuffet data was loaded and filtered
        # Note: Rebuffet columns were already prefixed with "Rebuffet_"
        original_rebuffet_cols_in_merge_post_prefix = [col for col in tumor_adata.obs.columns if col.startswith("Rebuffet_") and col in rebuffet_to_merge.columns]

        if not rebuffet_to_merge.empty: # rebuffet_to_merge is the aligned data
            # ... existing join code for Rebuffet ...
            # tumor_adata.obs = tumor_adata.obs.join(rebuffet_to_merge, how='left') 
            # print(f"  Joined Rebuffet NK fractions to tumor_adata.obs. Added columns: {list(rebuffet_to_merge.columns[:3])}...")
            
            # --- VERIFY REBUFFET PER-SAMPLE SUMS ---
            if original_rebuffet_cols_in_merge_post_prefix: # These are the columns that were just joined
                for col_sum_check in original_rebuffet_cols_in_merge_post_prefix:
                    if col_sum_check in tumor_adata.obs:
                        tumor_adata.obs[col_sum_check] = pd.to_numeric(tumor_adata.obs[col_sum_check], errors='coerce')
                
                sum_check_df_reb = tumor_adata.obs[original_rebuffet_cols_in_merge_post_prefix].dropna(how='all')
                if not sum_check_df_reb.empty:
                    sum_check_series_reb = sum_check_df_reb.sum(axis=1)
                    print("  Rebuffet NK Fraction Sum Check per sample (expect ~1.0 if relative mode across these phenotypes):")
                    print(sum_check_series_reb.describe())
                    if not np.allclose(sum_check_series_reb.dropna(), 1.0, atol=0.01):
                        print("  WARNING: Sum of original Rebuffet NK fractions per sample is not consistently close to 1.0!")
                    else:
                        print("  Original Rebuffet NK fractions per sample (where present) appear to sum to ~1.0.")
                else:
                    print("  No samples with non-NaN Rebuffet data to perform sum check.")
            else:
                print("  No Rebuffet columns identified for sum check after prefixing and join.")
            # --- END VERIFICATION ---
        else:
            print("  Rebuffet data empty after aligning. Not joined or verified.")
    else:
        print("  Filtered Rebuffet DataFrame was empty initially. Nothing to join or verify for Rebuffet.")


    print(f"\n--- IV.D. Loading and Merging CIBERSORTx HiRes Gene Expression for {CANCER_TYPE_ABBREV} ---")
    # HIRES_NK_ACT_PATH and HIRES_NK_REST_PATH defined in config

    def load_and_prep_hires_v2(hires_path, prefix_str, target_obs_names):
        if not os.path.exists(hires_path):
            print(f"    WARNING: HiRes file not found: {hires_path}")
            return pd.DataFrame()
        try:
            df_hires = pd.read_csv(hires_path, sep="\t", index_col="GeneSymbol") # Genes as rows, Samples as columns
            df_hires = df_hires.T # Samples as rows, Genes as columns (to match .obs structure for join)
            
            # Align with target_obs_names (tumor_adata.obs_names)
            df_hires_aligned = df_hires.reindex(target_obs_names).dropna(how='all') # Keep only samples in tumor_adata
            if df_hires_aligned.empty:
                print(f"    HiRes: No common samples found in {os.path.basename(hires_path)} after aligning with tumor_adata.obs_names.")
                return pd.DataFrame()

            df_hires_aligned.columns = [f"{prefix_str}{gene}" for gene in df_hires_aligned.columns]
            print(f"    Loaded and processed HiRes from {os.path.basename(hires_path)}. Shape: {df_hires_aligned.shape}. Prefixed columns with '{prefix_str}'.")
            return df_hires_aligned
        except Exception as e:
            print(f"    ERROR loading/processing HiRes file {hires_path}: {e}")
            return pd.DataFrame()

    hires_nk_act_df = load_and_prep_hires_v2(HIRES_NK_ACT_PATH, "NKact_", tumor_adata.obs_names)
    hires_nk_rest_df = load_and_prep_hires_v2(HIRES_NK_REST_PATH, "NKrest_", tumor_adata.obs_names)

    if not hires_nk_act_df.empty:
        tumor_adata.obs = tumor_adata.obs.join(hires_nk_act_df, how='left')
        print(f"  Joined HiRes NK-Activated gene expression to tumor_adata.obs. Added columns like: {list(hires_nk_act_df.columns[:2])}...")
    else: print("  HiRes NK-Activated data is empty or failed to load. Not joined.")
    
    if not hires_nk_rest_df.empty:
        tumor_adata.obs = tumor_adata.obs.join(hires_nk_rest_df, how='left')
        print(f"  Joined HiRes NK-Resting gene expression to tumor_adata.obs. Added columns like: {list(hires_nk_rest_df.columns[:2])}...")
    else: print("  HiRes NK-Resting data is empty or failed to load. Not joined.")
    
    print(f"  tumor_adata.obs shape after CIBERSORTx merges: {tumor_adata.obs.shape}")
    print(tumor_adata.obs.head(2))

else:
    print(f"  tumor_adata for {CANCER_TYPE_ABBREV} is None. Skipping Section IV (CIBERSORTx data integration).")

print(f"\n--- End of Section IV: CIBERSORTx Data Integration for {CANCER_TYPE_ABBREV} ---")

# %%
# ==============================================================================
# --- (NEW) Section III-bis: Exporting Linear TPM Expression Matrix (Tumor Only) ---
# ==============================================================================
# This block exports the compiled linear (non-log-transformed) TPM data for the 
# TUMOR SAMPLES ONLY for the current cancer type. This is useful for external tools 
# or for sharing data.
# NOTE: This block should be placed AFTER Section III, where `tumor_adata` is created.

print(f"\n--- III-bis. Exporting Tumor-Only Linear TPM Data for {CANCER_TYPE_ABBREV} ---")

# This code requires both the original linear counts and the filtered tumor_adata object.
if 'rna_counts_df' in locals() and not rna_counts_df.empty and \
   'tumor_adata' in locals() and tumor_adata is not None and tumor_adata.n_obs > 0:
    
    try:
        # Define a specific directory for these data exports
        export_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "Exported_Data_Matrices")
        os.makedirs(export_dir, exist_ok=True)
        
        # --- Export the tumor-only matrix ---
        print(f"\n  Preparing to export linear TPM matrix for the {tumor_adata.n_obs} tumor samples.")
        
        # Filter the original linear TPM dataframe for the tumor samples present in `tumor_adata`
        tumor_sample_ids = tumor_adata.obs_names
        # Ensure we only try to select columns that exist in the original rna_counts_df
        valid_tumor_ids = [tid for tid in tumor_sample_ids if tid in rna_counts_df.columns]
        
        if not valid_tumor_ids:
            print("  ERROR: No tumor sample IDs from `tumor_adata` were found in the original `rna_counts_df`.")
        else:
            tumor_rna_counts_df = rna_counts_df[valid_tumor_ids]
            
            # Define the uncompressed TSV filename
            tumor_only_filename = f"{CANCER_TYPE_ABBREV}_TPM_Linear_TumorOnly.tsv"
            tumor_only_path = os.path.join(export_dir, tumor_only_filename)
            
            print(f"    Shape: {tumor_rna_counts_df.shape} (Genes x Samples)")
            print(f"    Saving uncompressed TSV file to: {tumor_only_path}")
            
            # Save to a tab-separated value (TSV) file without compression
            tumor_rna_counts_df.to_csv(
                tumor_only_path, 
                sep='\t',          # Use tab as the separator
                index=True,        # Keep the gene symbols as the index column
                header=True        # Keep the sample IDs as the header row
            )
            print("    Export complete.")

    except Exception as e:
        print(f"  ERROR during data export: {e}")
else:
    print("  WARNING: Could not export tumor-only TPM data.")
    if 'rna_counts_df' not in locals() or rna_counts_df.empty:
        print("    Reason: `rna_counts_df` is not available or is empty.")
    if 'tumor_adata' not in locals() or tumor_adata is None or tumor_adata.n_obs == 0:
        print("    Reason: `tumor_adata` is not available or is empty. Please run this cell after Section III.")

print(f"\n--- End of Section III-bis ---")

# %%
# ==============================================================================
# --- V.E. (MODIFIED) Stratified Survival Analysis: TUSC2 Tertile Effect by Age Group ---
# ==============================================================================
# This section investigates if the relationship between TUSC2 expression (Top vs. Bottom 33%)
# and survival is different for younger vs. older patients.

from lifelines.statistics import multivariate_logrank_test, logrank_test

print(f"\n--- V.E. Stratified Survival Analysis: TUSC2 Tertiles vs. Survival by Age Group ({CANCER_TYPE_ABBREV}) ---")

# Define the stratification variable and cutoff
age_col = "Age_at_Diagnosis"
age_cutoff = 50

# Ensure the necessary columns are present (using the new tertile group column)
required_cols = ["Survival_Time_KM", "Event_Observed_KM", "TUSC2_Group_Tertile", age_col]
if tumor_adata is not None and all(col in tumor_adata.obs.columns for col in required_cols):
    
    # We only need the top and bottom tertiles for this analysis
    tusc2_tertile_group_order = ["Low TUSC2 (Bottom 33%)", "High TUSC2 (Top 33%)"]
    
    # Create a DataFrame for analysis, already filtered for the groups of interest
    stratified_df = tumor_adata.obs[tumor_adata.obs["TUSC2_Group_Tertile"].isin(tusc2_tertile_group_order)][required_cols].copy().dropna()
    
    # Create the age stratification group
    stratified_df['Age_Group'] = np.where(stratified_df[age_col] >= age_cutoff, 
                                          f"Age >= {age_cutoff}", 
                                          f"Age < {age_cutoff}")
    
    # --- 1. Stratified Kaplan-Meier Plots & Statistical Tests ---
    print("\n  1. Generating Stratified Kaplan-Meier Plots (using Tertiles) and running tests...")
    
    stratified_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_Clinical", "Survival_Stratified_by_Age")
    os.makedirs(stratified_output_dir, exist_ok=True)
    
    for age_group_name in stratified_df['Age_Group'].unique():
        print(f"    - Analyzing group: {age_group_name}")
        
        df_subset = stratified_df[stratified_df['Age_Group'] == age_group_name]
        
        if len(df_subset) >= 20 and df_subset['TUSC2_Group_Tertile'].nunique() == 2:
            # The plot function now uses the tertile group column
            lr_result_stratified = plot_kaplan_meier(
                df=df_subset,
                time_col="Survival_Time_KM",
                event_col="Event_Observed_KM",
                group_col="TUSC2_Group_Tertile",
                title=f"TUSC2 Tertiles and Survival in Patients ({age_group_name})",
                output_dir=stratified_output_dir,
                plot_filename_base=f"KM_TUSC2_Tertiles_in_{age_group_name.replace(' ', '').replace('<', 'lt').replace('>=', 'gte')}",
                group_order=tusc2_tertile_group_order,
                palette=['dodgerblue', 'orangered']
            )
            
            T = df_subset['Survival_Time_KM']
            E = df_subset['Event_Observed_KM']
            G = df_subset['TUSC2_Group_Tertile']
            
            weighted_lr_result = multivariate_logrank_test(T, G, E, weightings='peto')
            standard_p_value = lr_result_stratified.p_value if lr_result_stratified else float('nan')
            
            print(f"      Standard Log-Rank p-value: {standard_p_value:.4f}")
            print(f"      Weighted Log-Rank p-value (Peto): {weighted_lr_result.p_value:.4f}")

            if lr_result_stratified:
                all_survival_results_list.append({
                    "Analysis_Name": f"Survival_by_TUSC2_Group_Stratified",
                    "Cancer_Type": CANCER_TYPE_ABBREV,
                    "Grouping_Variable": "TUSC2_Group_Tertile",
                    "Metric_Split_By": "TUSC2 (Top vs Bottom 33%)",
                    "Stratification": age_group_name,
                    "LogRank_P_Value": standard_p_value,
                    "Weighted_LogRank_P_Value": weighted_lr_result.p_value,
                    "LogRank_Statistic": lr_result_stratified.test_statistic,
                    "Groups_Compared": str(tusc2_tertile_group_order)
                })
        else:
            print(f"      Skipping analysis for '{age_group_name}': Not enough data or only one TUSC2 group present.")

    # --- 2. Formal Interaction Test with Cox Proportional Hazards Model (using Tertiles) ---
    print("\n  2. Performing CoxPH Analysis with a TUSC2 Tertile x Age Interaction Term...")
    
    cox_interaction_df = stratified_df.copy()
    cox_interaction_df['TUSC2_High'] = (cox_interaction_df['TUSC2_Group_Tertile'] == tusc2_tertile_group_order[1]).astype(int)
    cox_interaction_df['Age_GTE_50'] = (cox_interaction_df['Age_Group'] == f"Age >= {age_cutoff}").astype(int)
    
    cph_interaction = CoxPHFitter()
    try:
        cph_interaction.fit(
            cox_interaction_df[['Survival_Time_KM', 'Event_Observed_KM', 'TUSC2_High', 'Age_GTE_50']],
            duration_col='Survival_Time_KM',
            event_col='Event_Observed_KM',
            formula="TUSC2_High + Age_GTE_50 + TUSC2_High * Age_GTE_50"
        )
        
        print("\n    --- Cox Model with TUSC2 Tertile x Age Interaction ---")
        cph_interaction.print_summary(decimals=3)
        
        interaction_p_value = cph_interaction.summary.loc['TUSC2_High:Age_GTE_50', 'p']
        
        print("\n    --- Interpretation ---")
        print(f"    The p-value for the TUSC2:Age interaction is: {interaction_p_value:.4f}")
        if interaction_p_value < 0.05:
            print("    CONCLUSION: The p-value is significant (< 0.05).")
            print("    This provides strong evidence that the relationship between TUSC2 (Top/Bottom 33%) and survival IS DIFFERENT for patients above and below age 50.")
        else:
            print("    CONCLUSION: The p-value is not significant (>= 0.05).")
            print("    We do not have strong statistical evidence to say that the effect of TUSC2 (Top/Bottom 33%) on survival depends on the patient's age group.")

    except Exception as e:
        print(f"    ERROR running Cox model with interaction term: {e}")

else:
    print("  Skipping stratified analysis: Required columns for survival, TUSC2 tertile group, or age are missing.")

# %%
# ==============================================================================
# --- VI. Bulk TUSC2 and the Immune Microenvironment Landscape (Revised for Stats Collection) ---
# ==============================================================================
# This section explores how bulk TUSC2 expression relates to broad immune cell
# infiltration (LM22 fractions) for the specified CANCER_TYPE_ABBREV.

if tumor_adata is not None and "TUSC2_Expression_Bulk" in tumor_adata.obs:
    print(f"\n--- VI. Analyzing Bulk TUSC2 vs. Immune Landscape for {CANCER_TYPE_ABBREV} ---")

    lm22_cell_types = [
        "B cells naive", "B cells memory", "Plasma cells", "T cells CD8",
        "T cells CD4 naive", "T cells CD4 memory resting", "T cells CD4 memory activated",
        "T cells follicular helper", "T cells regulatory (Tregs)", "T cells gamma delta",
        "NK cells resting", "NK cells activated", "Monocytes", "Macrophages M0",
        "Macrophages M1", "Macrophages M2", "Dendritic cells resting",
        "Dendritic cells activated", "Mast cells resting", "Mast cells activated",
        "Eosinophils", "Neutrophils"
    ]
    available_lm22_cols = [col for col in lm22_cell_types if col in tumor_adata.obs.columns]
    
    if not available_lm22_cols:
        print("  WARNING: No LM22 cell type columns found. Skipping Section VI.")
    else:
        print(f"  Using {len(available_lm22_cols)} LM22 cell types for analysis.")
        for col in available_lm22_cols: # Ensure numeric
            tumor_adata.obs[col] = pd.to_numeric(tumor_adata.obs[col], errors='coerce')

        # --- VI.A. Correlation of Bulk TUSC2 with Overall LM22 Immune Cell Fractions ---
        # This part primarily generates correlation data tables and a summary plot;
        # individual group comparison stats are not generated here.
        print(f"\n--- VI.A. Correlation of Bulk TUSC2 with LM22 Fractions for {CANCER_TYPE_ABBREV} ---")
        tusc2_lm22_corr_data = []
        for cell_type_col in available_lm22_cols:
            temp_df = tumor_adata.obs[["TUSC2_Expression_Bulk", cell_type_col]].dropna()
            if len(temp_df) > 2 and temp_df[cell_type_col].nunique() > 1 and temp_df["TUSC2_Expression_Bulk"].nunique() > 1 :
                r, p = spearmanr(temp_df["TUSC2_Expression_Bulk"], temp_df[cell_type_col])
                tusc2_lm22_corr_data.append({"Immune_Cell_Type": cell_type_col, "Spearman_R": r, "p_value": p})
            else:
                tusc2_lm22_corr_data.append({"Immune_Cell_Type": cell_type_col, "Spearman_R": np.nan, "p_value": np.nan})
        
        tusc2_lm22_corr_df = pd.DataFrame(tusc2_lm22_corr_data)
        tusc2_lm22_corr_df = apply_fdr_correction_df(tusc2_lm22_corr_df, p_value_col='p_value', q_value_col='FDR_q_value')
        tusc2_lm22_corr_df["Annot_Text"] = tusc2_lm22_corr_df.apply(
            lambda row: f"{row['Spearman_R']:.2f}{get_significance_stars(row['FDR_q_value'])}" if pd.notna(row['Spearman_R']) else "N/A", axis=1
        )
        tusc2_lm22_corr_df_sorted = tusc2_lm22_corr_df.sort_values("Spearman_R", ascending=False).dropna(subset=['Spearman_R'])
        
        if not tusc2_lm22_corr_df_sorted.empty:
            lm22_corr_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_Immune_Landscape", "LM22_Correlations")
            fig_corr_bar, ax_corr_bar = plt.subplots(figsize=(8, max(6, len(tusc2_lm22_corr_df_sorted) * 0.3)))
            sns.barplot(x="Spearman_R", y="Immune_Cell_Type", data=tusc2_lm22_corr_df_sorted,
                        palette="coolwarm_r", edgecolor="black", ax=ax_corr_bar, hue="Immune_Cell_Type", legend=False) # Added hue, legend
            ax_corr_bar.set_title(f"Bulk TUSC2 vs. LM22 Fractions ({CANCER_TYPE_ABBREV})", fontsize=14)
            ax_corr_bar.set_xlabel("Spearman Correlation (R)", fontsize=12)
            ax_corr_bar.set_ylabel("LM22 Immune Cell Type", fontsize=12)
            ax_corr_bar.axvline(0, color='grey', linestyle='--', linewidth=0.8)
            for i, bar_plot in enumerate(ax_corr_bar.patches):
                r_val_text = tusc2_lm22_corr_df_sorted.iloc[i]["Annot_Text"]
                r_val_num = tusc2_lm22_corr_df_sorted.iloc[i]["Spearman_R"]
                ax_corr_bar.text(bar_plot.get_width() + (0.01 if r_val_num >=0 else -0.08), 
                                 bar_plot.get_y() + bar_plot.get_height() / 2,
                                 r_val_text, va='center', ha='left' if r_val_num >=0 else 'right', fontsize=8)
            plt.tight_layout()
            save_plot_and_data(fig_corr_bar, tusc2_lm22_corr_df_sorted, lm22_corr_output_dir, "TUSC2_vs_LM22_Corr_Barplot", is_summary_data=True)
            plt.close(fig_corr_bar)
            print(f"  TUSC2 vs LM22 Correlation Barplot and data saved to {lm22_corr_output_dir}")
            print("  Top/Bottom TUSC2-LM22 Correlations (FDR corrected):")
            display(tusc2_lm22_corr_df.sort_values("FDR_q_value").head(5))
            display(tusc2_lm22_corr_df.sort_values("Spearman_R", ascending=False).head(5))
            display(tusc2_lm22_corr_df.sort_values("Spearman_R", ascending=True).head(5))
        else:
            print("  No valid correlation data to plot for TUSC2 vs LM22 fractions.")

        # --- VI.B. LM22 Immune Cell Composition by Bulk TUSC2 Status (High/Low) ---
        print(f"\n--- VI.B. LM22 Composition by Bulk TUSC2 High/Low Groups for {CANCER_TYPE_ABBREV} ---")
        
        tusc2_group_col_composition = "TUSC2_Group_Composition"
        if tusc2_group_col_composition not in tumor_adata.obs or tumor_adata.obs[tusc2_group_col_composition].isna().all():
            median_tusc2_comp = tumor_adata.obs["TUSC2_Expression_Bulk"].median()
            tumor_adata.obs[tusc2_group_col_composition] = np.where(tumor_adata.obs["TUSC2_Expression_Bulk"] >= median_tusc2_comp,
                                                                  "High TUSC2", "Low TUSC2")
            tumor_adata.obs.loc[tumor_adata.obs["TUSC2_Expression_Bulk"].isna(), tusc2_group_col_composition] = np.nan
            print(f"  Created '{tusc2_group_col_composition}' based on median TUSC2 expression ({median_tusc2_comp:.3f}).")
        
        tusc2_group_order_comp = ["Low TUSC2", "High TUSC2"]
        tumor_adata.obs[tusc2_group_col_composition] = pd.Categorical(
            tumor_adata.obs[tusc2_group_col_composition], categories=tusc2_group_order_comp, ordered=True
        )
        
        # 1. Stacked bar plot
        lm22_norm_cols_comp = [f"{col}_norm_comp_vi" for col in available_lm22_cols] # Unique suffix
        temp_lm22_fractions_comp = tumor_adata.obs[available_lm22_cols].copy()
        temp_lm22_fractions_comp[temp_lm22_fractions_comp < 0] = 0 
        sample_sums_comp = temp_lm22_fractions_comp.sum(axis=1)
        
        # Create normalized DataFrame separately to avoid fragmentation warning on tumor_adata.obs directly
        normalized_lm22_data_comp = temp_lm22_fractions_comp.div(sample_sums_comp.replace(0, np.nan), axis=0).fillna(0) * 100
        normalized_lm22_data_comp.columns = lm22_norm_cols_comp # Assign new column names
        
        # Join to a temporary DataFrame for groupby, or join to tumor_adata.obs if these cols are needed later
        plotting_obs_lm22_comp = tumor_adata.obs.join(normalized_lm22_data_comp, how='left')

        lm22_means_by_tusc2_group = plotting_obs_lm22_comp.groupby(tusc2_group_col_composition, observed=False)[lm22_norm_cols_comp].mean()
        lm22_means_by_tusc2_group.columns = [col.replace("_norm_comp_vi", "") for col in lm22_norm_cols_comp]
        
        lm22_comp_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_Immune_Landscape", "LM22_Composition_by_TUSC2_Group")
        group_counts_comp = tumor_adata.obs[tusc2_group_col_composition].value_counts().reindex(tusc2_group_order_comp).fillna(0)

        if not lm22_means_by_tusc2_group.empty and not lm22_means_by_tusc2_group.isna().all().all():
            plot_stacked_bar_composition(
                df_means_composition=lm22_means_by_tusc2_group,
                title=f"Mean LM22 Immune Composition by TUSC2 Group ({CANCER_TYPE_ABBREV})",
                xlab="TUSC2 Expression Group", ylab="Mean Immune Cell Fraction (%)", # Corrected label
                output_dir=lm22_comp_output_dir, plot_filename_base="LM22_StackedBar_by_TUSC2_Group",
                group_counts_series=group_counts_comp, legend_title="LM22 Cell Type", colormap="tab20"
            )
        else:
            print("  Skipping LM22 stacked bar plot: mean compositions are empty or all NaN.")

        # 2. Individual boxplots for key/significant LM22 fractions
        key_lm22_for_boxplot = [
            "T cells CD8", "T cells regulatory (Tregs)", "NK cells activated", 
            "NK cells resting", "Macrophages M1", "Macrophages M2", "Mast cells resting",
            "T cells CD4 memory activated" # Added from your previous significant list
        ]
        if not tusc2_lm22_corr_df.empty:
            sig_pos_corr = tusc2_lm22_corr_df[(tusc2_lm22_corr_df['FDR_q_value'] < 0.05) & (tusc2_lm22_corr_df['Spearman_R'] > 0)].nsmallest(2, 'FDR_q_value')['Immune_Cell_Type'].tolist()
            sig_neg_corr = tusc2_lm22_corr_df[(tusc2_lm22_corr_df['FDR_q_value'] < 0.05) & (tusc2_lm22_corr_df['Spearman_R'] < 0)].nsmallest(2, 'FDR_q_value')['Immune_Cell_Type'].tolist()
            key_lm22_for_boxplot.extend(sig_pos_corr)
            key_lm22_for_boxplot.extend(sig_neg_corr)
        key_lm22_for_boxplot = sorted(list(set(key_lm22_for_boxplot))) 
        
        print(f"\n  Plotting individual boxplots for selected LM22 fractions vs TUSC2 group: {key_lm22_for_boxplot}")
        lm22_indiv_boxplot_dir = os.path.join(lm22_comp_output_dir, "Individual_Boxplots")

        for cell_type in key_lm22_for_boxplot:
            if cell_type in tumor_adata.obs.columns:
                stats_df_lm22_tusc2 = plot_box_violin_with_stats( # Capture stats
                    df=tumor_adata.obs, x_col=tusc2_group_col_composition, y_col=cell_type, 
                    group_order=tusc2_group_order_comp,
                    title=f"{cell_type} Fraction by TUSC2 Group ({CANCER_TYPE_ABBREV})",
                    xlab="TUSC2 Expression Group", ylab=f"{cell_type} Fraction (CIBERSORTx)",
                    output_dir=lm22_indiv_boxplot_dir,
                    plot_filename_base=f"{cell_type.replace(' ', '_').replace('/', '_')}_by_TUSC2_Group",
                    plot_type='box', test_type='mannwhitneyu_2group', palette=['skyblue', 'salmon']
                )
                if stats_df_lm22_tusc2 is not None and not stats_df_lm22_tusc2.empty: # Append stats
                    all_group_comparison_stats_list.append(
                        stats_df_lm22_tusc2.assign(
                            Analysis_Context="LM22_vs_TUSC2_Group",
                            Variable_Compared=cell_type,
                            Grouping_Variable=tusc2_group_col_composition
                        )
                    )
            else:
                print(f"    Skipping boxplot for {cell_type} as it's not in tumor_adata.obs")
        
        # Clean up temporary normalized columns from plotting_obs_lm22_comp if they were added to tumor_adata.obs directly.
        # Since we used a temporary join for plotting_obs_lm22_comp, tumor_adata.obs itself wasn't modified with _norm_comp_vi columns.
        # If you had assigned them directly to tumor_adata.obs earlier, you would drop them here:
        # tumor_adata.obs.drop(columns=lm22_norm_cols_comp, inplace=True, errors='ignore')

else:
    print(f"  tumor_adata is None or TUSC2_Expression_Bulk not in tumor_adata.obs. Skipping Section VI for {CANCER_TYPE_ABBREV}.")

print(f"\n--- End of Section VI: Bulk TUSC2 and Immune Microenvironment Landscape for {CANCER_TYPE_ABBREV} ---")

# %%
# ==============================================================================
# --- VII. Deep Dive: Bulk TUSC2 and NK Cell Populations (Revised for Stats Collection) ---
# ==============================================================================

if tumor_adata is not None and "TUSC2_Expression_Bulk" in tumor_adata.obs:
    print(f"\n--- VII. Deep Dive: Bulk TUSC2 and NK Cells for {CANCER_TYPE_ABBREV} ---")

    tusc2_group_col = "TUSC2_Group_Composition" 
    if tusc2_group_col not in tumor_adata.obs or tumor_adata.obs[tusc2_group_col].isna().all():
        print(f"  '{tusc2_group_col}' not found or all NaN. Defining TUSC2 High/Low groups.")
        median_tusc2_val = tumor_adata.obs["TUSC2_Expression_Bulk"].median()
        tumor_adata.obs[tusc2_group_col] = np.where(
            tumor_adata.obs["TUSC2_Expression_Bulk"] >= median_tusc2_val, "High TUSC2", "Low TUSC2"
        )
        tumor_adata.obs.loc[tumor_adata.obs["TUSC2_Expression_Bulk"].isna(), tusc2_group_col] = np.nan
    tusc2_group_order = ["Low TUSC2", "High TUSC2"] # Ensure this order is used consistently
    tumor_adata.obs[tusc2_group_col] = pd.Categorical(tumor_adata.obs[tusc2_group_col], categories=tusc2_group_order, ordered=True)

    # --- VII.A. Bulk TUSC2 vs. LM22 NK Cell Fractions (Specific Focus) ---
    print(f"\n--- VII.A. Bulk TUSC2 vs. LM22 NK Cell Fractions for {CANCER_TYPE_ABBREV} ---")
    nk_output_dir_lm22 = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_NK_Cells", "LM22_NK_Focus")

    if "NK cells activated" in tumor_adata.obs.columns and "NK cells resting" in tumor_adata.obs.columns:
        tumor_adata.obs["NK cells activated"] = pd.to_numeric(tumor_adata.obs["NK cells activated"], errors='coerce')
        tumor_adata.obs["NK cells resting"] = pd.to_numeric(tumor_adata.obs["NK cells resting"], errors='coerce')

        if "NK_Cells_Total_LM22" not in tumor_adata.obs: # Calculate if not already present
             tumor_adata.obs["NK_Cells_Total_LM22"] = tumor_adata.obs["NK cells activated"] + tumor_adata.obs["NK cells resting"]
        
        stats_df_totalnk_lm22_tusc2 = plot_box_violin_with_stats(
            df=tumor_adata.obs, x_col=tusc2_group_col, y_col="NK_Cells_Total_LM22",
            group_order=tusc2_group_order,
            title=f"Total LM22 NK Cells by TUSC2 Group ({CANCER_TYPE_ABBREV})",
            xlab="TUSC2 Expression Group", ylab="Total LM22 NK Cell Fraction",
            output_dir=nk_output_dir_lm22, plot_filename_base="LM22_TotalNK_by_TUSC2_Group",
            plot_type='box', test_type='mannwhitneyu_2group', palette=['lightsteelblue', 'salmon']
        )
        if stats_df_totalnk_lm22_tusc2 is not None and not stats_df_totalnk_lm22_tusc2.empty:
            all_group_comparison_stats_list.append(
                stats_df_totalnk_lm22_tusc2.assign(
                    Analysis_Context="LM22_NK_Focus_vs_TUSC2_Group",
                    Variable_Compared="NK_Cells_Total_LM22",
                    Grouping_Variable=tusc2_group_col 
                )
            )

        epsilon_lm22 = 1e-6 
        nk_resting_numeric = pd.to_numeric(tumor_adata.obs["NK cells resting"], errors='coerce').fillna(0)
        nk_activated_numeric = pd.to_numeric(tumor_adata.obs["NK cells activated"], errors='coerce').fillna(0)
        
        nk_ratio_lm22_values = nk_activated_numeric / (nk_resting_numeric + epsilon_lm22)
        nk_ratio_lm22_values.replace([np.inf, -np.inf], np.nan, inplace=True) # Inplace on Series is fine before assignment
        tumor_adata.obs["NK_Ratio_LM22"] = nk_ratio_lm22_values

        stats_df_nkratio_lm22_tusc2 = plot_box_violin_with_stats(
            df=tumor_adata.obs, x_col=tusc2_group_col, y_col="NK_Ratio_LM22",
            group_order=tusc2_group_order,
            title=f"LM22 NK Active/Resting Ratio by TUSC2 Group ({CANCER_TYPE_ABBREV})",
            xlab="TUSC2 Expression Group", ylab="NK Active/Resting Ratio (LM22)",
            output_dir=nk_output_dir_lm22, plot_filename_base="LM22_NK_Ratio_by_TUSC2_Group",
            plot_type='box', test_type='mannwhitneyu_2group', palette=['lightsteelblue', 'salmon']
        )
        if stats_df_nkratio_lm22_tusc2 is not None and not stats_df_nkratio_lm22_tusc2.empty:
            all_group_comparison_stats_list.append(
                stats_df_nkratio_lm22_tusc2.assign(
                    Analysis_Context="LM22_NK_Focus_vs_TUSC2_Group",
                    Variable_Compared="NK_Ratio_LM22",
                    Grouping_Variable=tusc2_group_col
                )
            )
    else:
        print("  'NK cells activated' or 'NK cells resting' (LM22) not found. Skipping LM22 NK specific analysis.")

    # --- VII.B. Bulk TUSC2 vs. Rebuffet NK Phenotype Fractions ---
    print(f"\n--- VII.B. Bulk TUSC2 vs. Rebuffet NK Phenotypes for {CANCER_TYPE_ABBREV} ---")
    
    original_rebuffet_cols = [col for col in tumor_adata.obs.columns if col.startswith("Rebuffet_") and 
                              not col.endswith(("_norm_comp_nk", "_norm_stacked", "_norm"))] # Ensure we get base Rebuffet cols

    if not original_rebuffet_cols:
        print("  WARNING: No original Rebuffet NK phenotype columns found. Skipping this sub-section.")
    else:
        print(f"  Using {len(original_rebuffet_cols)} original Rebuffet NK phenotypes: {original_rebuffet_cols}")
        for col in original_rebuffet_cols: 
            tumor_adata.obs[col] = pd.to_numeric(tumor_adata.obs[col], errors='coerce')

        nk_output_dir_rebuffet = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_NK_Cells", "Rebuffet_NK_Focus")

        # 1. Spearman correlation of TUSC2 with each Rebuffet NK phenotype
        tusc2_rebuffet_corr_data = []
        for cell_type_col_rb in original_rebuffet_cols: 
            temp_df_rb = tumor_adata.obs[["TUSC2_Expression_Bulk", cell_type_col_rb]].dropna()
            if len(temp_df_rb) > 2 and temp_df_rb[cell_type_col_rb].nunique() > 1 and temp_df_rb["TUSC2_Expression_Bulk"].nunique() > 1:
                r, p = spearmanr(temp_df_rb["TUSC2_Expression_Bulk"], temp_df_rb[cell_type_col_rb])
                tusc2_rebuffet_corr_data.append({"Rebuffet_NK_Phenotype": cell_type_col_rb, "Spearman_R": r, "p_value": p})
            else:
                tusc2_rebuffet_corr_data.append({"Rebuffet_NK_Phenotype": cell_type_col_rb, "Spearman_R": np.nan, "p_value": np.nan})
        
        tusc2_rebuffet_corr_df = pd.DataFrame(tusc2_rebuffet_corr_data)
        if not tusc2_rebuffet_corr_df.empty:
            tusc2_rebuffet_corr_df = apply_fdr_correction_df(tusc2_rebuffet_corr_df, p_value_col='p_value', q_value_col='FDR_q_value')
            tusc2_rebuffet_corr_df["Annot_Text"] = tusc2_rebuffet_corr_df.apply(
                lambda row: f"{row['Spearman_R']:.2f}{get_significance_stars(row['FDR_q_value'])}" if pd.notna(row['Spearman_R']) else "N/A", axis=1
            )
            tusc2_rebuffet_corr_df_sorted = tusc2_rebuffet_corr_df.sort_values("Spearman_R", ascending=False).dropna(subset=['Spearman_R'])

            if not tusc2_rebuffet_corr_df_sorted.empty:
                fig_reb_corr_bar, ax_reb_corr_bar = plt.subplots(figsize=(8, max(4, len(tusc2_rebuffet_corr_df_sorted) * 0.4)))
                sns.barplot(x="Spearman_R", y="Rebuffet_NK_Phenotype", data=tusc2_rebuffet_corr_df_sorted,
                            palette="viridis", edgecolor="black", ax=ax_reb_corr_bar, hue="Rebuffet_NK_Phenotype", legend=False) 
                ax_reb_corr_bar.set_title(f"Bulk TUSC2 vs. Rebuffet NK Phenotypes ({CANCER_TYPE_ABBREV})", fontsize=14)
                ax_reb_corr_bar.set_xlabel("Spearman Correlation (R)", fontsize=12)
                ax_reb_corr_bar.set_ylabel("Rebuffet NK Phenotype", fontsize=12)
                ax_reb_corr_bar.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                for i, bar_plot in enumerate(ax_reb_corr_bar.patches):
                    r_val_text = tusc2_rebuffet_corr_df_sorted.iloc[i]["Annot_Text"]
                    r_val_num = tusc2_rebuffet_corr_df_sorted.iloc[i]["Spearman_R"]
                    ax_reb_corr_bar.text(bar_plot.get_width() + (0.01 if r_val_num >=0 else -0.08),
                                     bar_plot.get_y() + bar_plot.get_height() / 2,
                                     r_val_text, va='center', ha='left' if r_val_num >=0 else 'right', fontsize=9)
                plt.tight_layout()
                save_plot_and_data(fig_reb_corr_bar, tusc2_rebuffet_corr_df_sorted, nk_output_dir_rebuffet, "TUSC2_vs_RebuffetNK_Corr_Barplot", is_summary_data=True)
                plt.close(fig_reb_corr_bar)
                print(f"  TUSC2 vs Rebuffet NK Correlation Barplot and data saved to {nk_output_dir_rebuffet}")
            else:
                print("  No valid correlation data for TUSC2 vs Rebuffet NK phenotypes after sorting/dropping NaN R values.")
        else:
            print("  No correlation data generated for TUSC2 vs Rebuffet NK (dataframe empty).")


        # 2. Stacked bar plot for Rebuffet NK phenotype composition by TUSC2 group
        rebuffet_norm_cols_stacked = [f"{col}_norm_stacked_vii" for col in original_rebuffet_cols] # Unique suffix
        temp_rebuffet_fractions_stacked = tumor_adata.obs[original_rebuffet_cols].copy()
        temp_rebuffet_fractions_stacked[temp_rebuffet_fractions_stacked < 0] = 0
        sample_sums_rebuffet_stacked = temp_rebuffet_fractions_stacked.sum(axis=1)
        
        normalized_rebuffet_df_stacked = temp_rebuffet_fractions_stacked.div(sample_sums_rebuffet_stacked.replace(0, np.nan), axis=0).fillna(0) * 100
        normalized_rebuffet_df_stacked.columns = rebuffet_norm_cols_stacked
        
        temp_plotting_obs_stacked_vii = tumor_adata.obs.join(normalized_rebuffet_df_stacked, how='left') # Use unique name
        
        rebuffet_means_by_tusc2_group = temp_plotting_obs_stacked_vii.groupby(tusc2_group_col, observed=False)[rebuffet_norm_cols_stacked].mean()
        rebuffet_means_by_tusc2_group.columns = [col.replace("_norm_stacked_vii", "").replace("Rebuffet_","") for col in rebuffet_norm_cols_stacked]
        group_counts_reb_comp = tumor_adata.obs[tusc2_group_col].value_counts().reindex(tusc2_group_order).fillna(0)

        if not rebuffet_means_by_tusc2_group.empty and not rebuffet_means_by_tusc2_group.isna().all().all():
            plot_stacked_bar_composition(
                df_means_composition=rebuffet_means_by_tusc2_group,
                title=f"Mean Rebuffet NK Phenotype Composition by TUSC2 Group ({CANCER_TYPE_ABBREV})",
                xlab="TUSC2 Expression Group", ylab="Mean NK Phenotype Fraction (%)", # Corrected Y-axis label
                output_dir=nk_output_dir_rebuffet, plot_filename_base="RebuffetNK_StackedBar_by_TUSC2_Group",
                group_counts_series=group_counts_reb_comp, legend_title="Rebuffet NK Phenotype", colormap="Set3"
            )
        else: print("  Skipping Rebuffet NK stacked bar plot: mean compositions empty/NaN.")

        # 3. Individual boxplots (use original_rebuffet_cols)
        rebuffet_indiv_boxplot_dir = os.path.join(nk_output_dir_rebuffet, "Individual_Boxplots")
        for cell_type_col_rb_box in original_rebuffet_cols: 
            stats_df_rebuffet_tusc2 = plot_box_violin_with_stats( # Capture stats
                df=tumor_adata.obs, x_col=tusc2_group_col, y_col=cell_type_col_rb_box,
                group_order=tusc2_group_order,
                title=f"{cell_type_col_rb_box.replace('Rebuffet_','')} Fraction by TUSC2 Group ({CANCER_TYPE_ABBREV})",
                xlab="TUSC2 Expression Group", ylab=f"{cell_type_col_rb_box.replace('Rebuffet_','')} Fraction",
                output_dir=rebuffet_indiv_boxplot_dir, 
                plot_filename_base=f"{cell_type_col_rb_box.replace('Rebuffet_','')}_by_TUSC2_Group",
                plot_type='box', test_type='mannwhitneyu_2group', palette=['lightcyan', 'lightcoral']
            )
            if stats_df_rebuffet_tusc2 is not None and not stats_df_rebuffet_tusc2.empty: # Append
                all_group_comparison_stats_list.append(
                    stats_df_rebuffet_tusc2.assign(
                        Analysis_Context="Rebuffet_NK_vs_TUSC2_Group",
                        Variable_Compared=cell_type_col_rb_box,
                        Grouping_Variable=tusc2_group_col
                    )
                )
    
    # --- VII.C. Bulk TUSC2 vs. HiRes Expression of OTHER NK-Related Genes ---
    # (Code for VII.C - HiRes correlations - remains the same as it doesn't generate group comparison stats for the list)
    # ... (Keep the existing VII.C code block here, starting from print(f"\n--- VII.C. Bulk TUSC2 vs. HiRes...))
    # --- VII.C. Bulk TUSC2 vs. HiRes Expression of OTHER NK-Related Genes ---
    print(f"\n--- VII.C. Bulk TUSC2 vs. HiRes Expression of other NK-related Genes for {CANCER_TYPE_ABBREV} ---")
    # (genes_of_interest definition from previous cell)
    nk_genes_of_interest_hires = [
    # === I. Core NK Cell Lineage & Subset-Defining Surface Markers ===
    "NCAM1",     # (CD56) Defines CD56bright (NK2) vs CD56dim (NK1, NK3) subsets. (Rebuffet et al. Nat Immunol 2024)
    "FCGR3A",    # (CD16a) Primarily on CD56dim (NK1, NK3), key for ADCC. (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024)
    "NCR1",      # (NKp46) Broadly on mature NK cells; key activating receptor. (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024 - NK1/NK3 protein)
    "SELL",      # (L-selectin/CD62L) High on CD56bright/NK2. (Rebuffet et al. Nat Immunol 2024 - NK2 RNA high)
    "CCR7",      # LN homing; primarily on CD56bright/NK2. (DOI: 10.1038/ni.1810)
    "ITGAM",     # (CD11b) Marks more mature/effector CD56dim NK cells (NK1B, NK1C). (DOI: 10.1182/blood-2005-03-1203)
    "B3GAT1",    # (CD57) Marks terminally differentiated, "adaptive" NK cells (NK3). (Rebuffet et al. Nat Immunol 2024 - NK3 protein high)
    "KLRG1",     # Inhibitory receptor; marker of terminally differentiated effector NK cells. (DOI: 10.4049/jimmunol.176.5.2853)
    "CD27",      # Expression decreases with maturation (high on CD56bright/NK2). (Rebuffet et al. Nat Immunol 2024 - NK2 protein high)
    "CX3CR1",    # High on terminally differentiated effector/tissue-homing NK (NK1C, some NK3). (Rebuffet et al. Nat Immunol 2024 - NK1C protein high)

    # === II. Key NK Cell Education & Inhibitory/Activating Receptor Systems ===
    # KIRs (highly clonal, but presence/absence of expression is NK-intrinsic state)
    "KIR2DL1", "KIR2DL3", "KIR3DL1", # Key inhibitory KIRs. (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024 - NK3 protein high)
    "KIR2DS4", "KIR3DS1", # Representative activating KIRs.
    "KLRD1",     # (CD94) Forms heterodimers with NKG2A/C/E. (Rebuffet et al. Nat Immunol 2024)
    "KLRC1",     # (NKG2A) Inhibitory, high on immature NK (NK2). (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024 - NK2 RNA high)
    "KLRC2",     # (NKG2C) Activating, marker for "adaptive" NK cells (NK3). (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024 - NK3 RNA high)
    "KLRK1",     # (NKG2D) Activating receptor for stress ligands. (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024 - NK2 protein high)
    "NCR3",      # (NKp30) Key activating Natural Cytotoxicity Receptor. (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024 - NK1/NK3 protein)
    "CD226",     # (DNAM-1) Activating receptor. (Wolf et al. Nat Rev Immunol 2023)
    "CD244",     # (2B4) SLAM family receptor, context-dependent. (Rebuffet et al. Nat Immunol 2024)
    "SIGLEC7",   # Inhibitory receptor. (DOI: 10.1038/nri2609)
    "TIGIT",     # Inhibitory checkpoint receptor on NK cells. (Wolf et al. Nat Rev Immunol 2023)
    "HAVCR2",    # (TIM-3) Inhibitory checkpoint receptor on NK cells. (Wolf et al. Nat Rev Immunol 2023)
    "PDCD1",     # (PD-1) Inhibitory checkpoint receptor on activated/exhausted NK cells. (Wolf et al. Nat Rev Immunol 2023)

    # === III. Cytotoxic Machinery & Degranulation ===
    "PRF1", "GZMB", "GZMH", "GZMK", "NKG7", "GNLY", # Core cytotoxic molecules (GZMA/MM less specific or dynamic). (Rebuffet et al. Nat Immunol 2024 - shows differential GZMs by subset)
    "LAMP1",     # (CD107a) Degranulation marker. (DOI: 10.1016/j.jim.2004.09.015)
    "FASLG",     # Fas Ligand. (Wolf et al. Nat Rev Immunol 2023)
    "TNFSF10",   # (TRAIL) Cytotoxic ligand. (Wolf et al. Nat Rev Immunol 2023)

    # === IV. Key NK-Cell Intrinsic Transcription Factors ===
    "EOMES", "ID2", "TBX21", "GATA3", "RUNX3", "IKZF2", "PRDM1", "ZEB2", "BACH2", "TOX", # Cover development, maturation, effector function, adaptive states, exhaustion. (Wolf et al. Nat Rev Immunol 2023; Rebuffet et al. Nat Immunol 2024)
    "NFATC1", "NFATC2", # Ca2+-responsive TFs important for NK effector function.

    # === V. NK-Cell Intrinsic Signaling Adaptors/Kinases (More specific or highly regulated in NKs) ===
    "FCER1G", "CD247", "TYROBP", "HCST", # ITAM-bearing adaptors crucial for NK activation.
    "SYK", "ZAP70", # Key kinases in activation pathways.
    "SH2D1A",    # (SAP) Adaptor for SLAM receptors.
    "VAV1",      # GEF crucial for actin dynamics and signaling.

    # === VI. Cytokines Produced BY NK Cells (Reflecting NK functional state) ===
    "IFNG", "TNF", "CSF2", "XCL1", "CCL5", # Key examples. (Rebuffet et al. Nat Immunol 2024)

    # === VII. Receptors for Key NK-Modulating Cytokines (Reflecting NK responsiveness) ===
    "IL2RB", "IL2RG", "IL15RA", "IL12RB1", "IL12RB2", "IL18R1", "IL18RAP", "IL21R", "IFNGR1", "IFNGR2", "IL7R",

    # === VIII. Metabolic Regulators (Key TFs, Transporters, Rate-limiting/Regulatory Enzymes for NK states) ===
    "HIF1A", "MYC", # Master metabolic TFs.
    "SLC2A1", "SLC2A3", "SLC7A5", "SLC1A5", # Key nutrient transporters.
    "PFKFB3", # Glycolysis regulator.
    "NAMPT",  # NAD salvage.
    "CPT1A",  # Fatty Acid Oxidation regulator.

    # === IX. Your Gene of Interest ===
    "TUSC2",
]
    nk_genes_of_interest_hires = sorted(list(set(nk_genes_of_interest_hires)))

    hires_corr_results = []
    hires_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_NK_Cells", "HiRes_NK_Gene_Correlations")

    for gene_state_prefix in ["NKact_", "NKrest_"]:
        for gene_symbol in nk_genes_of_interest_hires:
            hires_gene_col = f"{gene_state_prefix}{gene_symbol}"
            if hires_gene_col in tumor_adata.obs.columns:
                tumor_adata.obs[hires_gene_col] = pd.to_numeric(tumor_adata.obs[hires_gene_col], errors='coerce')
                temp_df = tumor_adata.obs[["TUSC2_Expression_Bulk", hires_gene_col]].dropna()
                if len(temp_df) > 2 and temp_df[hires_gene_col].nunique() > 1 and temp_df["TUSC2_Expression_Bulk"].nunique() > 1: # Check TUSC2 variance too
                    r, p = spearmanr(temp_df["TUSC2_Expression_Bulk"], temp_df[hires_gene_col])
                    hires_corr_results.append({
                        "HiRes_Gene_Full": hires_gene_col, "Gene_Symbol": gene_symbol, 
                        "NK_State": gene_state_prefix.replace("_",""), "Spearman_R": r, "p_value": p
                    })
                else:
                    hires_corr_results.append({
                        "HiRes_Gene_Full": hires_gene_col, "Gene_Symbol": gene_symbol, "NK_State": gene_state_prefix.replace("_",""),
                        "Spearman_R": np.nan, "p_value": np.nan
                    })
    
    if hires_corr_results:
        hires_corr_df = pd.DataFrame(hires_corr_results)
        if not hires_corr_df.empty:
            hires_corr_df = apply_fdr_correction_df(hires_corr_df, p_value_col='p_value', q_value_col='FDR_q_value')
            hires_corr_df["Annot_Text_Heatmap"] = hires_corr_df.apply(
                lambda row: f"{row['Spearman_R']:.2f}{get_significance_stars(row['FDR_q_value'])}" if pd.notna(row['Spearman_R']) else "", axis=1
            )
            heatmap_r_df = hires_corr_df.pivot_table(index="Gene_Symbol", columns="NK_State", values="Spearman_R")
            heatmap_annot_df = hires_corr_df.pivot_table(index="Gene_Symbol", columns="NK_State", values="Annot_Text_Heatmap", aggfunc='first') # 'first' should be fine
            
            state_order_heatmap = [s.replace("_","") for s in ["NKact_", "NKrest_"] if s.replace("_","") in heatmap_r_df.columns]
            
            if state_order_heatmap: # Ensure columns exist before trying to subset
                heatmap_r_df = heatmap_r_df[state_order_heatmap]
                # Align annotation DF to R DF index and columns AFTER R DF is finalized
                heatmap_annot_df = heatmap_annot_df.reindex(index=heatmap_r_df.index, columns=state_order_heatmap)


            if not heatmap_r_df.dropna(how='all').empty:
                from scipy.cluster import hierarchy
                from scipy.spatial.distance import pdist

                heatmap_r_df_for_plot = heatmap_r_df.dropna(how='all') # Use this for plotting
                heatmap_annot_df_for_plot = heatmap_annot_df.reindex(index=heatmap_r_df_for_plot.index)


                if len(heatmap_r_df_for_plot) > 1: # Only cluster if there's more than one gene
                    try:
                        row_linkage = hierarchy.linkage(pdist(heatmap_r_df_for_plot.fillna(0), metric='correlation'), method='average')
                        row_order_idx = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']
                        
                        heatmap_r_df_clustered = heatmap_r_df_for_plot.iloc[row_order_idx, :]
                        heatmap_annot_df_clustered = heatmap_annot_df_for_plot.iloc[row_order_idx, :] # Cluster annotation accordingly
                        
                        print("  Applied row clustering to HiRes correlation heatmap.")
                        plot_title_hm = f"Bulk TUSC2 vs. HiRes NK Gene Expression (Row Clustered)\n({CANCER_TYPE_ABBREV})"
                        plot_fn_base_hm = "TUSC2_vs_HiRes_NKGenes_CorrHeatmap_Clustered"
                        df_to_plot_hm = heatmap_r_df_clustered
                        annot_to_plot_hm = heatmap_annot_df_clustered
                    except Exception as e_cluster:
                        print(f"  ERROR during HiRes heatmap clustering: {e_cluster}. Plotting without clustering.")
                        plot_title_hm = f"Bulk TUSC2 vs. HiRes NK Gene Expression ({CANCER_TYPE_ABBREV})"
                        plot_fn_base_hm = "TUSC2_vs_HiRes_NKGenes_CorrHeatmap_Unclustered"
                        df_to_plot_hm = heatmap_r_df_for_plot
                        annot_to_plot_hm = heatmap_annot_df_for_plot
                else: 
                     print("  Not enough rows for HiRes heatmap clustering, plotting without clustering.")
                     plot_title_hm = f"Bulk TUSC2 vs. HiRes NK Gene Expression ({CANCER_TYPE_ABBREV})"
                     plot_fn_base_hm = "TUSC2_vs_HiRes_NKGenes_CorrHeatmap_Unclustered"
                     df_to_plot_hm = heatmap_r_df_for_plot
                     annot_to_plot_hm = heatmap_annot_df_for_plot
                
                plot_correlation_heatmap( 
                    corr_matrix_df=df_to_plot_hm, 
                    annot_text_matrix_df=annot_to_plot_hm, 
                    title=plot_title_hm,
                    output_dir=hires_output_dir,
                    plot_filename_base=plot_fn_base_hm,
                    cmap="vlag", center=0,
                    figsize=(7, max(10, len(df_to_plot_hm) * 0.25)),
                    sort_rows_by_column='NKact',
                    sort_rows_ascending=True
                )
                
                hires_corr_df.sort_values(["NK_State", "FDR_q_value"], inplace=True)
                hires_corr_df.to_csv(os.path.join(hires_output_dir, "TUSC2_vs_HiRes_NKGenes_FullCorrData.csv"), index=False)
                print(f"  HiRes NK Gene correlation data saved to {hires_output_dir}")
            else:
                print("  No data for HiRes NK gene correlation heatmap after processing and dropping all-NaN rows.")
    else:
        print("  No HiRes gene correlation results generated (initial list empty or DataFrame empty).")

else:
    print(f"  tumor_adata is None or TUSC2_Expression_Bulk not in tumor_adata.obs. Skipping Section VII for {CANCER_TYPE_ABBREV}.")

print(f"\n--- End of Section VII: Deep Dive into TUSC2 and NK Cells for {CANCER_TYPE_ABBREV} ---")

# %%
# ==============================================================================
# --- VIII. Contextualizing NK Cell States and Phenotypes (Revised for Stats Collection) ---
# ==============================================================================
# This section explores inter-relationships between different NK cell fraction estimates
# and their associations with clinical variables and survival.

if tumor_adata is not None:
    print(f"\n--- VIII. Contextualizing NK Cell States for {CANCER_TYPE_ABBREV} ---")

    # Define relevant column sets
    lm22_nk_cols_for_context = ["NK cells activated", "NK cells resting", "NK_Cells_Total_LM22", "NK_Ratio_LM22"]
    lm22_nk_cols_present = [col for col in lm22_nk_cols_for_context if col in tumor_adata.obs.columns]
    
    original_rebuffet_cols_for_context = [ # Renamed for clarity within this section
        col for col in tumor_adata.obs.columns if col.startswith("Rebuffet_") and 
        not col.endswith(("_norm_comp_nk", "_norm_stacked", "_norm")) # From VII.B
    ]

    if not lm22_nk_cols_present and not original_rebuffet_cols_for_context:
        print("  WARNING: No LM22 or Rebuffet NK columns found. Skipping Section VIII.")
    else:
        nk_context_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "NK_Cell_Contextualization")
        os.makedirs(nk_context_output_dir, exist_ok=True)

        # --- VIII.A. Inter-Correlations: Rebuffet NK vs. LM22 NK ---
        print(f"\n--- VIII.A. Inter-Correlations of Rebuffet and LM22 NK Estimates ---")
        if lm22_nk_cols_present and original_rebuffet_cols_for_context:
            all_nk_context_cols_intercorr = lm22_nk_cols_present + original_rebuffet_cols_for_context # Use specific list name
            
            for col in all_nk_context_cols_intercorr: # Ensure numeric
                if col in tumor_adata.obs:
                    tumor_adata.obs[col] = pd.to_numeric(tumor_adata.obs[col], errors='coerce')
            
            nk_inter_corr_df = tumor_adata.obs[all_nk_context_cols_intercorr].corr(method='spearman')
            nk_inter_corr_annot_text = nk_inter_corr_df.round(2).astype(str)
            
            plot_correlation_heatmap(
                corr_matrix_df=nk_inter_corr_df,
                annot_text_matrix_df=nk_inter_corr_annot_text,
                title=f"Inter-Correlations of LM22 & Rebuffet NK Estimates ({CANCER_TYPE_ABBREV})",
                output_dir=os.path.join(nk_context_output_dir, "Inter_NK_Correlations"),
                plot_filename_base="LM22_vs_Rebuffet_NK_CorrelationHeatmap",
                cmap="coolwarm", center=0, 
                figsize=(max(8, len(all_nk_context_cols_intercorr)*0.7), max(6, len(all_nk_context_cols_intercorr)*0.6))
            )
        else:
            print("  Skipping Inter-NK Correlations: Missing either LM22 NK or Rebuffet NK columns.")

        # --- VIII.B. NK Cell Fractions/Phenotypes vs. Clinical Variables ---
        print(f"\n--- VIII.B. NK Cell Metrics vs. Clinical Variables ---")
        
        all_nk_metrics_to_test_clin = lm22_nk_cols_present + original_rebuffet_cols_for_context # Use specific list name
        
        # 1. vs. Age at Diagnosis
        age_col_clin = "Age_at_Diagnosis" # Use specific var name
        if age_col_clin in tumor_adata.obs:
            print(f"  VIII.B.1. NK Metrics vs. {age_col_clin}")
            nk_vs_age_output_dir = os.path.join(nk_context_output_dir, "NK_vs_Age")
            tumor_adata.obs[age_col_clin] = pd.to_numeric(tumor_adata.obs[age_col_clin], errors='coerce')

            for nk_metric_col_age in all_nk_metrics_to_test_clin: # Use specific var name
                if nk_metric_col_age in tumor_adata.obs:
                    plot_correlation_scatter( # This plot func doesn't return p-values for list collection
                        df=tumor_adata.obs, x_col=age_col_clin, y_col=nk_metric_col_age,
                        title=f"{nk_metric_col_age.replace('_',' ')} vs. Age ({CANCER_TYPE_ABBREV})",
                        xlab="Age at Diagnosis (Years)", ylab=nk_metric_col_age.replace('_',' '),
                        output_dir=nk_vs_age_output_dir,
                        plot_filename_base=f"{nk_metric_col_age.replace(' ','_')}_vs_Age_Scatter" # Safe filename
                    )
        else:
            print(f"  '{age_col_clin}' not found. Skipping NK Metrics vs. Age.")

        # 2. vs. Collapsed Pathologic Stage
        collapsed_stage_col_clin = "Pathologic_Stage_Collapsed" # From Section V
        if collapsed_stage_col_clin in tumor_adata.obs:
            print(f"  VIII.B.2. NK Metrics vs. {collapsed_stage_col_clin}")
            nk_vs_stage_output_dir = os.path.join(nk_context_output_dir, "NK_vs_Stage")
            stage_order_plot_clin = ["Stage I", "Stage II", "Stage III", "Stage IV"] # From Section V

            for nk_metric_col_stage in all_nk_metrics_to_test_clin: # Use specific var name
                if nk_metric_col_stage in tumor_adata.obs:
                    plot_df_nk_stage_loop = tumor_adata.obs[ # Use specific var name
                        tumor_adata.obs[collapsed_stage_col_clin].isin(stage_order_plot_clin) &
                        tumor_adata.obs[nk_metric_col_stage].notna()
                    ].copy()
                    plot_df_nk_stage_loop[collapsed_stage_col_clin] = pd.Categorical(
                        plot_df_nk_stage_loop[collapsed_stage_col_clin], categories=stage_order_plot_clin, ordered=True
                    )
                    stage_val_counts_loop = plot_df_nk_stage_loop[collapsed_stage_col_clin].value_counts()
                    valid_stages_for_plot_loop = [s for s in stage_order_plot_clin if stage_val_counts_loop.get(s, 0) >= 2]

                    if len(valid_stages_for_plot_loop) > 1:
                        stats_df_nk_stage = plot_box_violin_with_stats( # Capture stats
                            df=plot_df_nk_stage_loop, x_col=collapsed_stage_col_clin, y_col=nk_metric_col_stage,
                            group_order=valid_stages_for_plot_loop,
                            title=f"{nk_metric_col_stage.replace('_',' ')} by Stage ({CANCER_TYPE_ABBREV})",
                            xlab="Pathologic Stage (Collapsed)", ylab=nk_metric_col_stage.replace('_',' '),
                            output_dir=nk_vs_stage_output_dir,
                            plot_filename_base=f"{nk_metric_col_stage.replace(' ','_')}_by_Stage_Boxplot", # Safe filename
                            plot_type='box', test_type='kruskal_multigroup' # Or 'mannwhitneyu_pairwise_fdr'
                        )
                        if stats_df_nk_stage is not None and not stats_df_nk_stage.empty: # Append stats
                            all_group_comparison_stats_list.append(
                                stats_df_nk_stage.assign(
                                    Analysis_Context="NK_Metrics_vs_Stage",
                                    Variable_Compared=nk_metric_col_stage,
                                    Grouping_Variable=collapsed_stage_col_clin
                                )
                            )
                    else:
                        print(f"    Skipping {nk_metric_col_stage} by Stage: Not enough groups with data.")
        else:
            print(f"  '{collapsed_stage_col_clin}' not found. Skipping NK Metrics vs. Stage.")

        # 3. vs. Vital Status
        vital_status_col_clin = next((col for col in ["Vital_Status", "Vital_Status_Clinical"] if col in tumor_adata.obs.columns), None)
        if vital_status_col_clin:
            print(f"  VIII.B.3. NK Metrics vs. {vital_status_col_clin}")
            nk_vs_vital_output_dir = os.path.join(nk_context_output_dir, "NK_vs_VitalStatus")
            vital_order_plot_clin = ["ALIVE", "DEAD"] # Use specific var name

            for nk_metric_col_vital in all_nk_metrics_to_test_clin: # Use specific var name
                if nk_metric_col_vital in tumor_adata.obs:
                    plot_df_nk_vital_loop = tumor_adata.obs[ # Use specific var name
                        tumor_adata.obs[vital_status_col_clin].astype(str).str.upper().isin(vital_order_plot_clin) &
                        tumor_adata.obs[nk_metric_col_vital].notna()
                    ].copy()
                    plot_df_nk_vital_loop[vital_status_col_clin] = plot_df_nk_vital_loop[vital_status_col_clin].astype(str).str.upper()
                    plot_df_nk_vital_loop[vital_status_col_clin] = pd.Categorical(
                        plot_df_nk_vital_loop[vital_status_col_clin], categories=vital_order_plot_clin, ordered=True
                    )
                    
                    if plot_df_nk_vital_loop[vital_status_col_clin].nunique() == 2 and \
                       all(plot_df_nk_vital_loop[vital_status_col_clin].value_counts().get(g,0) >= 2 for g in vital_order_plot_clin):
                        stats_df_nk_vital = plot_box_violin_with_stats( # Capture stats
                            df=plot_df_nk_vital_loop, x_col=vital_status_col_clin, y_col=nk_metric_col_vital,
                            group_order=vital_order_plot_clin,
                            title=f"{nk_metric_col_vital.replace('_',' ')} by Vital Status ({CANCER_TYPE_ABBREV})",
                            xlab="Vital Status", ylab=nk_metric_col_vital.replace('_',' '),
                            output_dir=nk_vs_vital_output_dir,
                            plot_filename_base=f"{nk_metric_col_vital.replace(' ','_')}_by_VitalStatus_Boxplot", # Safe filename
                            plot_type='box', test_type='mannwhitneyu_2group'
                        )
                        if stats_df_nk_vital is not None and not stats_df_nk_vital.empty: # Append stats
                             all_group_comparison_stats_list.append(
                                stats_df_nk_vital.assign(
                                    Analysis_Context="NK_Metrics_vs_VitalStatus",
                                    Variable_Compared=nk_metric_col_vital,
                                    Grouping_Variable=vital_status_col_clin
                                )
                            )
                    else:
                        print(f"    Skipping {nk_metric_col_vital} by Vital Status: Need both 'ALIVE' and 'DEAD' with sufficient data.")
        else:
            print("  Vital status column not found. Skipping NK Metrics vs. Vital Status.")

        # --- VIII.C. Survival Analysis for Key NK Populations ---
        print(f"\n--- VIII.C. Survival Analysis for Key NK Populations ---")
        nk_survival_output_dir = os.path.join(nk_context_output_dir, "NK_Survival_Analysis")
        
        # Re-fetch actual survival column names (as done in Section V.C)
        actual_event_col_surv = next((col for col in ["Vital_Status", "Vital_Status_Clinical"] if col in tumor_adata.obs.columns), None)
        actual_time_death_col_surv = next((col for col in ["Days_to_Death", "Days_to_Death_Clinical"] if col in tumor_adata.obs.columns), None)
        actual_time_followup_col_surv = next((col for col in ["Days_to_Last_Followup", "Days_to_Last_Followup_Clinical"] if col in tumor_adata.obs.columns), None)

        if actual_event_col_surv and (actual_time_death_col_surv or actual_time_followup_col_surv):
            nk_metrics_for_survival_map = { # Renamed variable for clarity
                "LM22_Total_NK": "NK_Cells_Total_LM22" if "NK_Cells_Total_LM22" in tumor_adata.obs else None,
                "LM22_NK_Ratio": "NK_Ratio_LM22" if "NK_Ratio_LM22" in tumor_adata.obs else None,
                "Rebuffet_NK1A": "Rebuffet_NK1A" if "Rebuffet_NK1A" in tumor_adata.obs else None, # Example
                "Rebuffet_NK2": "Rebuffet_NK2" if "Rebuffet_NK2" in tumor_adata.obs else None,   # Example
                # Add other specific Rebuffet phenotypes you want to test for survival
                 "Rebuffet_NK1B": "Rebuffet_NK1B" if "Rebuffet_NK1B" in tumor_adata.obs else None,
                 "Rebuffet_NK1C": "Rebuffet_NK1C" if "Rebuffet_NK1C" in tumor_adata.obs else None,
                 "Rebuffet_NK3": "Rebuffet_NK3" if "Rebuffet_NK3" in tumor_adata.obs else None,
                 "Rebuffet_NKint": "Rebuffet_NKint" if "Rebuffet_NKint" in tumor_adata.obs else None,
            }
            
            for km_label_nk, nk_metric_col_km_val in nk_metrics_for_survival_map.items(): # Use specific var name
                if nk_metric_col_km_val and nk_metric_col_km_val in tumor_adata.obs:
                    print(f"  Running survival analysis for: {km_label_nk} (using column: {nk_metric_col_km_val})")
                    # Create a fresh survival_df for each metric to avoid carrying over group columns
                    survival_df_nk_loop = tumor_adata.obs[[nk_metric_col_km_val, actual_event_col_surv, actual_time_death_col_surv, actual_time_followup_col_surv]].copy()
                    survival_df_nk_loop[nk_metric_col_km_val] = pd.to_numeric(survival_df_nk_loop[nk_metric_col_km_val], errors='coerce')
                    survival_df_nk_loop.dropna(subset=[nk_metric_col_km_val, actual_event_col_surv], inplace=True)

                    survival_df_nk_loop["Survival_Time_KM_NK"] = np.where( # Unique time/event col names
                        survival_df_nk_loop[actual_event_col_surv].astype(str).str.upper() == 'DEAD',
                        pd.to_numeric(survival_df_nk_loop[actual_time_death_col_surv], errors='coerce'),
                        pd.to_numeric(survival_df_nk_loop[actual_time_followup_col_surv], errors='coerce')
                    )
                    survival_df_nk_loop["Event_Observed_KM_NK"] = (survival_df_nk_loop[actual_event_col_surv].astype(str).str.upper() == 'DEAD').astype(int)
                    survival_df_nk_loop.dropna(subset=["Survival_Time_KM_NK", "Event_Observed_KM_NK"], inplace=True)
                    survival_df_nk_loop = survival_df_nk_loop[survival_df_nk_loop["Survival_Time_KM_NK"] > 0]

                    if len(survival_df_nk_loop) >= 20: # Min samples for meaningful analysis
                        median_nk_metric_val = survival_df_nk_loop[nk_metric_col_km_val].median()
                        group_col_km_current_nk = f"{km_label_nk}_Group_KM" # Unique group col name
                        survival_df_nk_loop[group_col_km_current_nk] = np.where(
                            survival_df_nk_loop[nk_metric_col_km_val] >= median_nk_metric_val, 
                            f"High {km_label_nk}", f"Low {km_label_nk}"
                        )
                        
                        group_order_km_current_nk = [f"Low {km_label_nk}", f"High {km_label_nk}"]
                        survival_df_nk_loop[group_col_km_current_nk] = pd.Categorical(
                            survival_df_nk_loop[group_col_km_current_nk], 
                            categories=group_order_km_current_nk, ordered=True
                        )

                        if survival_df_nk_loop[group_col_km_current_nk].nunique() == 2 and \
                           all(survival_df_nk_loop[group_col_km_current_nk].value_counts().get(g,0) >=1 for g in group_order_km_current_nk):
                            lr_result_nk_km = plot_kaplan_meier( # Capture result
                                df=survival_df_nk_loop, time_col="Survival_Time_KM_NK", event_col="Event_Observed_KM_NK",
                                group_col=group_col_km_current_nk,
                                title=f"Kaplan-Meier Survival by {km_label_nk.replace('_',' ')} ({CANCER_TYPE_ABBREV})",
                                output_dir=nk_survival_output_dir,
                                plot_filename_base=f"KM_{km_label_nk.replace(' ','_')}", # Safe filename
                                group_order=group_order_km_current_nk,
                                palette=['darkcyan', 'coral'] # Example different palette
                            )
                            if lr_result_nk_km: # Append result
                                all_survival_results_list.append({
                                    "Analysis_Name": f"Survival_by_{km_label_nk}_Group",
                                    "Cancer_Type": CANCER_TYPE_ABBREV,
                                    "Grouping_Variable": group_col_km_current_nk,
                                    "Metric_Split_By": f"{nk_metric_col_km_val} (Median)",
                                    "LogRank_P_Value": lr_result_nk_km.p_value,
                                    "LogRank_Statistic": lr_result_nk_km.test_statistic,
                                    "Groups_Compared": str(group_order_km_current_nk) 
                                })
                        else: 
                            print(f"    Skipping {km_label_nk} KM plot: Need two distinct groups with data after splitting.")
                    else: 
                        print(f"    Not enough valid samples (n={len(survival_df_nk_loop)}) for {km_label_nk} KM analysis.")
                else:
                    print(f"  Skipping survival for {km_label_nk}: column '{nk_metric_col_km_val}' not found or undefined.")
        else:
            print("  Required columns for survival analysis not all present. Skipping NK Population KM.")
else:
    print(f"  tumor_adata is None. Skipping Section VIII for {CANCER_TYPE_ABBREV}.")

print(f"\n--- End of Section VIII: Contextualizing NK Cell States and Phenotypes for {CANCER_TYPE_ABBREV} ---")

# %%
# ==============================================================================
# --- IX. Functional Correlates and Integrative TME Score (Corrected and Complete for Stats Collection) ---
# ==============================================================================
# This section investigates correlations of TUSC2 with checkpoint genes and
# defines and analyzes a TME Net Inflammatory Score.

if tumor_adata is not None and "TUSC2_Expression_Bulk" in tumor_adata.obs:
    print(f"\n--- IX. Functional Correlates & TME Score for {CANCER_TYPE_ABBREV} ---")
    tme_score_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TME_Scores_and_Correlates")
    os.makedirs(tme_score_output_dir, exist_ok=True)

    # --- IX.A. Bulk TUSC2 vs. Checkpoint Gene Expression ---
    # (This part was correct and remains the same)
    print(f"\n--- IX.A. Bulk TUSC2 vs. Checkpoint Gene Expression ---")
    checkpoint_genes_list = [
    # === I. PD-1 Pathway (Inhibitory) ===
    "PDCD1",     # (PD-1/CD279) Programmed Cell Death 1: Key inhibitory receptor on T cells, B cells, NK cells, macrophages. (DOI: 10.1038/nri3591; Wolf et al. Nat Rev Immunol 2023 - Fig 2f)
    "CD274",     # (PD-L1/B7-H1) Programmed Death-Ligand 1: Primary ligand for PD-1, expressed on tumor cells, APCs, other immune cells. (DOI: 10.1038/nri3591)
    "PDCD1LG2",  # (PD-L2/B7-DC) Programmed Death-Ligand 2: Second ligand for PD-1, mainly on APCs, some tumors. (DOI: 10.1038/nri3591)

    # === II. CTLA-4 Pathway (Inhibitory) ===
    "CTLA4",     # (CD152) Cytotoxic T-Lymphocyte Associated Protein 4: Inhibitory receptor on T cells (mainly regulatory T cells, activated conventional T cells). Competes with CD28 for B7 ligands. (DOI: 10.1126/science.271.5256.1734)
    "CD80",      # (B7-1) Co-stimulatory ligand for CD28 (activating) and CTLA-4 (inhibitory). Expressed on APCs. (NK cells can express upon strong activation).
    "CD86",      # (B7-2) Co-stimulatory ligand for CD28 (activating) and CTLA-4 (inhibitory). Expressed on APCs. (NK cells can express upon strong activation).

    # === III. TIM-3 Pathway (Inhibitory) ===
    "HAVCR2",    # (TIM-3/CD366) Hepatitis A Virus Cellular Receptor 2: Inhibitory receptor on T cells (Th1, CTLs), NK cells, Tregs, myeloid cells. Ligands include Galectin-9, CEACAM1, HMGB1. (DOI: 10.1038/nri3591; Wolf et al. Nat Rev Immunol 2023 - Fig 2f)
    "LGALS9",    # (Galectin-9) Ligand for TIM-3. Expressed by tumor cells, endothelial cells, APCs. (DOI: 10.1038/ni.1980)
    "CEACAM1",   # (CD66a) Carcinoembryonic Antigen-Related Cell Adhesion Molecule 1: Ligand for TIM-3, also homophilic interactions, complex roles. (DOI: 10.1038/s41467-018-04659-0)

    # === IV. LAG-3 Pathway (Inhibitory) ===
    "LAG3",      # (CD223) Lymphocyte Activating Gene 3: Inhibitory receptor on activated T cells, NK cells, Tregs. Binds MHC Class II, LSECtin, FGL1. (DOI: 10.1038/nri3591; Wolf et al. Nat Rev Immunol 2023 - Fig 2f)
    # "FGL1",    # Fibrinogen Like Protein 1: A major ligand for LAG-3. Expressed by some cancer cells. (DOI: 10.1016/j.cell.2018.11.010) - *Keep if broad TME profiling is desired*
    # MHC Class II genes (e.g., "HLA-DRA", "HLA-DPA1", "HLA-DQB1") are also LAG-3 ligands but are too broad for a "checkpoint" list.

    # === V. TIGIT/CD226/PVR Axis (Complex: Inhibitory/Activating) ===
    "TIGIT",     # T cell Immunoreceptor with Ig and ITIM domains: Inhibitory receptor on T cells and NK cells. Competes with CD226 for PVR/PVRL2. (DOI: 10.1038/nri3591; Wolf et al. Nat Rev Immunol 2023 - Fig 1 & 2f)
    "CD226",     # (DNAM-1) Activating receptor on T cells, NK cells, monocytes. Ligands PVR (CD155) and PVRL2 (Nectin-2/CD112). (DOI: 10.1038/nri1507)
    "PVR",       # (CD155) Poliovirus Receptor: Ligand for TIGIT, CD226, CD96. Expressed on tumor cells, APCs. (Wolf et al. Nat Rev Immunol 2023 - Fig 1)
    "PVRL2",     # (Nectin-2/CD112) PVR Related 2: Ligand for TIGIT, CD226. (Wolf et al. Nat Rev Immunol 2023 - Fig 1)
    "CD96",      # (TACTILE) Co-receptor, interacts with PVR, can be inhibitory or activating depending on context. (DOI: 10.1038/ni.2514)
    "PVRIG",     # Poliovirus Receptor Related Immunoglobulin Domain Containing: Inhibitory receptor, binds PVRL2 (Nectin-2/CD112). (DOI: 10.1016/j.cell.2016.10.047; Wolf et al. Nat Rev Immunol 2023 - Fig 1)

    # === VI. VISTA (Inhibitory) ===
    "VSIR",      # (VISTA/PD-1H/CD276-like) V-Set Immunoregulatory Receptor: Inhibitory checkpoint, expressed on myeloid cells, T cells. Acts as both ligand and receptor. (DOI: 10.1038/nature10347)

    # === VII. B7 Family (Beyond PD-L1/L2 & B7-1/2) ===
    "CD276",     # (B7-H3) Broadly expressed on tumors, APCs. Complex roles, often co-inhibitory, but activating contexts exist. Receptor largely unknown. (DOI: 10.1038/nrc.2016.140)
    "VTCN1",     # (B7-H4) V-Set C-Type Transmembrane Domain Containing 1: Inhibitory ligand, expressed on tumor cells, APCs. Receptor unknown. (DOI: 10.1038/ni1136)
    "HHLA2",     # (B7-H7/B7y) HERV-H LTR-Associating 2: Expressed on tumor cells, APCs. Interacts with CD28H (activating) and KIR3DL3 (inhibitory). (DOI: 10.1038/s41467-020-18965-9)
    "CD28H",     # (TMIGD2) CD28 Homolog: Activating receptor for HHLA2.

    # === VIII. Adenosine Pathway (Metabolic Checkpoint - Inhibitory) ===
    "ENTPD1",    # (CD39) Ectonucleoside Triphosphate Diphosphohydrolase 1: Converts ATP/ADP to AMP. On Tregs, endothelial cells, some NK/T cells, tumor cells. (DOI: 10.1038/s41467-019-13767-3)
    "NT5E",      # (CD73) 5'-Nucleotidase Ecto: Converts AMP to adenosine. On Tregs, endothelial cells, some NK/T cells, tumor cells. (DOI: 10.1038/s41467-019-13767-3)
    "ADORA2A",   # Adenosine A2a Receptor: GPCR for adenosine, mediates immunosuppression on T cells, NK cells. (DOI: 10.1038/nri.2017.86)
    "ADORA2B",   # Adenosine A2b Receptor: Another adenosine receptor with immunosuppressive roles.

    # === IX. Co-stimulatory TNF Receptor Superfamily (Activating) ===
    "TNFRSF4",   # (OX40/CD134) Expressed on activated T cells (and some NK). Ligand is TNFSF4 (OX40L). (DOI: 10.1038/nri3538)
    "TNFSF4",    # (OX40L/CD252) Ligand for OX40, expressed on APCs, endothelial cells.
    "TNFRSF9",   # (4-1BB/CD137) Expressed on activated T cells, NK cells. Ligand is TNFSF9 (4-1BBL). (DOI: 10.1038/nri3538; Wolf et al. Nat Rev Immunol 2023 - Fig 3a)
    "TNFSF9",    # (4-1BBL) Ligand for 4-1BB, expressed on APCs.
    "TNFRSF18",  # (GITR) Glucocorticoid-Induced TNFR-Related Protein: Expressed on T cells (esp. Tregs), NK cells. Ligand TNFSF18 (GITRL). (DOI: 10.1038/nri3538)
    "TNFSF18",   # (GITRL) Ligand for GITR.
    "CD27",      # TNF Receptor Superfamily Member 7: Co-stimulatory receptor on T cells, B cells, NK cells. Ligand CD70. (DOI: 10.1038/nri1602)
    "CD70",      # Ligand for CD27, expressed on activated APCs, lymphocytes, some tumors.
    "CD40",      # TNF Receptor Superfamily Member 5: On APCs, B cells, some tumor cells. (Interaction is CD40 on APC/tumor with CD40L on T cell)
    "CD40LG",    # (CD154) CD40 Ligand: Primarily on activated CD4+ T cells, crucial for B cell and APC activation.

    # === X. ICOS Pathway (Co-stimulatory) ===
    "ICOS",      # Inducible T-cell COStimulator (CD278): Expressed on activated T cells, some NK subsets. (DOI: 10.1038/nri3538)
    "ICOSLG",    # (B7-H2/CD275) ICOS Ligand: Expressed on APCs, some tumor cells. (DOI: 10.1038/nri3538)

    # === XI. Other Emerging/Contextual Checkpoints ===
    "SIGLEC7",   # Sialic Acid Binding Ig Like Lectin 7: Can be inhibitory on NK cells, myeloid cells. (DOI: 10.1038/nri2609)
    "SIGLEC9",   # Sialic Acid Binding Ig Like Lectin 9: Similar to SIGLEC7, inhibitory on myeloid, NK.
    "SIGLEC10",  # Sialic Acid Binding Ig Like Lectin 10: Binds CD24, can be inhibitory. (DOI: 10.1038/s43018-021-00216-z)
    "CD24",      # Ligand for SIGLEC10, often overexpressed on cancer cells.
    "SIGLEC15",  # Sialic Acid Binding Ig Like Lectin 15: Expressed on macrophages, tumor cells, potential immune suppressor. (DOI: 10.1038/s41591-019-0374-x)
    "NKG2A",     # (KLRC1) If not already covered elsewhere, inhibitory receptor on NK and some T cells. (Included as KLRC1 in NK lists, but good checkpoint context)
    "CD160",     # Co-receptor on NK cells and some T cells, binds HVEM, context-dependent. (Also on NK lists)
    "BTLA",      # (CD272) B And T Lymphocyte Attenuator: Inhibitory receptor, binds HVEM (TNFRSF14). (DOI: 10.1038/nri.2009.90)
    "TNFRSF14",  # (HVEM) Herpesvirus Entry Mediator: Ligand for BTLA, CD160, LIGHT. Complex roles.
    "IDO1",      # Indoleamine 2,3-Dioxygenase 1: Metabolic enzyme, depletes tryptophan, generates kynurenines, immunosuppressive. (DOI: 10.1038/nri3179)
    "IDO2",      # Indoleamine 2,3-Dioxygenase 2: Similar to IDO1.
    "TDO2",      # Tryptophan 2,3-Dioxygenase: Similar to IDO1.
    "ARG1",      # Arginase 1: Metabolic enzyme, depletes arginine, immunosuppressive, often in MDSCs.
    "ARG2"       # Arginase 2.
    ]
    available_checkpoint_genes = [gene for gene in checkpoint_genes_list if gene in tumor_adata.var_names]
    if not available_checkpoint_genes:
        print("  WARNING: No defined checkpoint genes found. Skipping TUSC2 vs. Checkpoints.")
    else:
        print(f"  Found {len(available_checkpoint_genes)} checkpoint genes for analysis: {available_checkpoint_genes}")
        tusc2_checkpoint_corr_data = []
        tumor_adata.obs["TUSC2_Expression_Bulk"] = pd.to_numeric(tumor_adata.obs["TUSC2_Expression_Bulk"], errors='coerce')
        for gene_symbol in available_checkpoint_genes:
            try:
                gene_expr_data = tumor_adata[:, gene_symbol].X
                checkpoint_expr = gene_expr_data.toarray().flatten() if hasattr(gene_expr_data, "toarray") else np.array(gene_expr_data).flatten()
            except Exception as e_gene_extract:
                print(f"    Could not extract expression for checkpoint {gene_symbol}: {e_gene_extract}. Skipping.")
                tusc2_checkpoint_corr_data.append({"Checkpoint_Gene": gene_symbol, "Spearman_R": np.nan, "p_value": np.nan})
                continue
            temp_df = pd.DataFrame({"TUSC2_Expression_Bulk": tumor_adata.obs["TUSC2_Expression_Bulk"], gene_symbol: checkpoint_expr}).dropna()
            if len(temp_df) > 2 and temp_df[gene_symbol].nunique() > 1 and temp_df["TUSC2_Expression_Bulk"].nunique() > 1:
                r, p = spearmanr(temp_df["TUSC2_Expression_Bulk"], temp_df[gene_symbol])
                tusc2_checkpoint_corr_data.append({"Checkpoint_Gene": gene_symbol, "Spearman_R": r, "p_value": p})
            else:
                tusc2_checkpoint_corr_data.append({"Checkpoint_Gene": gene_symbol, "Spearman_R": np.nan, "p_value": np.nan})
        tusc2_checkpoint_corr_df = pd.DataFrame(tusc2_checkpoint_corr_data)
        if not tusc2_checkpoint_corr_df.empty:
            tusc2_checkpoint_corr_df = apply_fdr_correction_df(tusc2_checkpoint_corr_df, p_value_col='p_value', q_value_col='FDR_q_value')
            tusc2_checkpoint_corr_df["Annot_Text"] = tusc2_checkpoint_corr_df.apply(lambda row: f"{row['Spearman_R']:.2f}{get_significance_stars(row['FDR_q_value'])}" if pd.notna(row['Spearman_R']) else "N/A", axis=1)
            tusc2_checkpoint_corr_df_sorted = tusc2_checkpoint_corr_df.sort_values("Spearman_R", ascending=False).dropna(subset=['Spearman_R'])
            if not tusc2_checkpoint_corr_df_sorted.empty:
                checkpoint_corr_output_dir = os.path.join(tme_score_output_dir, "TUSC2_vs_Checkpoints")
                fig_ckpt_corr_bar, ax_ckpt_corr_bar = plt.subplots(figsize=(8, max(5, len(tusc2_checkpoint_corr_df_sorted) * 0.3)))
                sns.barplot(x="Spearman_R", y="Checkpoint_Gene", data=tusc2_checkpoint_corr_df_sorted, palette="coolwarm", edgecolor="black", ax=ax_ckpt_corr_bar, hue="Checkpoint_Gene", legend=False)
                ax_ckpt_corr_bar.set_title(f"Bulk TUSC2 vs. Checkpoint Gene Expression ({CANCER_TYPE_ABBREV})", fontsize=14)
                ax_ckpt_corr_bar.set_xlabel("Spearman Correlation (R)", fontsize=12); ax_ckpt_corr_bar.set_ylabel("Checkpoint Gene", fontsize=12)
                ax_ckpt_corr_bar.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                for i, bar_plot in enumerate(ax_ckpt_corr_bar.patches):
                    r_val_text = tusc2_checkpoint_corr_df_sorted.iloc[i]["Annot_Text"]; r_val_num = tusc2_checkpoint_corr_df_sorted.iloc[i]["Spearman_R"]
                    ax_ckpt_corr_bar.text(bar_plot.get_width() + (0.01 if r_val_num >=0 else -0.08), bar_plot.get_y() + bar_plot.get_height() / 2, r_val_text, va='center', ha='left' if r_val_num >=0 else 'right', fontsize=8)
                plt.tight_layout()
                save_plot_and_data(fig_ckpt_corr_bar, tusc2_checkpoint_corr_df_sorted, checkpoint_corr_output_dir, "TUSC2_vs_Checkpoints_Corr_Barplot", is_summary_data=True)
                plt.close(fig_ckpt_corr_bar)
                print(f"  TUSC2 vs Checkpoints Correlation Barplot and data saved to {checkpoint_corr_output_dir}")
                print("  Top TUSC2-Checkpoint Correlations (FDR corrected):"); display(tusc2_checkpoint_corr_df.sort_values("FDR_q_value").head(10))
            else: print("  No valid correlation data for TUSC2 vs. Checkpoint genes after sorting.")
        else: print("  Correlation DataFrame for TUSC2 vs Checkpoints is empty.")

    # --- IX.B. TME Net Inflammatory Score ---
    # IX.B.1-3 (Define Signatures and Calculate Scores - this part was correct)
    print(f"\n--- IX.B. Calculating TME Net Inflammatory Score ---")
    inflammatory_signature_genes_raw = [
    "IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "CCL5",
    "CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "NKG7", "GNLY", "EOMES", "TBX21",
    "CD3D", "CD3E", "CD3G", "LCK", "CD2", "CD27", "CD69",
    "TNF", "IL1B", "IL2", "IL12A", "IL12B", "IL15", "IL18", "STAT4",
    "CD80", "CD86", "CD40", "ICOSLG",
    "NOS2", "FCGR1A"
    ]
    immunosuppressive_signature_genes_raw = [
    "FOXP3", "IL2RA", "CTLA4", "IKZF2", # IKZF2 kept here for Treg association
    "IL10", "TGFB1", "TGFBR1", "TGFBR2", "IL6", "IL4R",
    "CD163", "MRC1", "CCL18", "CCL22", "ARG1", "CSF1R", "CD209",
    "FAP", "ACTA2", "COL1A1", "COL1A2", "POSTN", "PDGFRA", "PDGFRB",
    "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CD274", "PDCD1LG2", "VSIR",
    "CD24", "SIGLEC10", # CD24/SIGLEC10 axis
    "ENTPD1", "NT5E", "ADORA2A", "IDO1",
    "VEGFA", "HIF1A",
    "CCL2"
    ]
    available_inflammatory_genes = [gene for gene in inflammatory_signature_genes_raw if gene in tumor_adata.var_names]
    available_immunosuppressive_genes = [gene for gene in immunosuppressive_signature_genes_raw if gene in tumor_adata.var_names]
    print(f"  Using {len(available_inflammatory_genes)}/{len(inflammatory_signature_genes_raw)} genes for Inflammatory component.")
    print(f"  Using {len(available_immunosuppressive_genes)}/{len(immunosuppressive_signature_genes_raw)} genes for Immunosuppressive component.")
    inflammatory_score_col_name = "Inflammatory_Score"; immunosuppressive_score_col_name = "Immunosuppressive_Score"
    if available_inflammatory_genes: calculate_signature_score(tumor_adata, available_inflammatory_genes, inflammatory_score_col_name)
    else: tumor_adata.obs[inflammatory_score_col_name] = np.nan
    if available_immunosuppressive_genes: calculate_signature_score(tumor_adata, available_immunosuppressive_genes, immunosuppressive_score_col_name)
    else: tumor_adata.obs[immunosuppressive_score_col_name] = np.nan
    net_tme_score_col_name = "TME_Net_Inflammation_Score"
    if inflammatory_score_col_name in tumor_adata.obs and immunosuppressive_score_col_name in tumor_adata.obs:
        tumor_adata.obs[inflammatory_score_col_name] = pd.to_numeric(tumor_adata.obs[inflammatory_score_col_name], errors='coerce')
        tumor_adata.obs[immunosuppressive_score_col_name] = pd.to_numeric(tumor_adata.obs[immunosuppressive_score_col_name], errors='coerce')
        tumor_adata.obs[net_tme_score_col_name] = tumor_adata.obs[inflammatory_score_col_name] - tumor_adata.obs[immunosuppressive_score_col_name]
        if tumor_adata.obs[net_tme_score_col_name].notna().any():
            print(f"\n  Calculated '{net_tme_score_col_name}'. Summary:"); print(tumor_adata.obs[net_tme_score_col_name].describe())
            fig_net_tme_dist, ax_net_tme_dist = plt.subplots(figsize=(6,4))
            sns.histplot(tumor_adata.obs[net_tme_score_col_name].dropna(), kde=True, ax=ax_net_tme_dist, color="teal")
            ax_net_tme_dist.set_title(f"Distribution of {net_tme_score_col_name} ({CANCER_TYPE_ABBREV})"); ax_net_tme_dist.set_xlabel("Score Value"); ax_net_tme_dist.set_ylabel("Frequency")
            plt.tight_layout()
            save_plot_and_data(fig_net_tme_dist, tumor_adata.obs[[net_tme_score_col_name]].dropna(), tme_score_output_dir, f"{net_tme_score_col_name}_Distribution", is_summary_data=False); plt.close(fig_net_tme_dist)
        else: print(f"  '{net_tme_score_col_name}' is all NaN."); tumor_adata.obs[net_tme_score_col_name] = np.nan
    else: print(f"  Could not calculate '{net_tme_score_col_name}'."); tumor_adata.obs[net_tme_score_col_name] = np.nan
        
    # --- IX.B.4. Characterize TME Net Inflammation Score ---
    if net_tme_score_col_name in tumor_adata.obs and tumor_adata.obs[net_tme_score_col_name].notna().any():
        print(f"\n--- IX.B.4. Characterizing '{net_tme_score_col_name}' for {CANCER_TYPE_ABBREV} ---")
        net_score_char_output_dir = os.path.join(tme_score_output_dir, "Net_Score_Characterization")
        os.makedirs(net_score_char_output_dir, exist_ok=True)

        # Correlation with Bulk TUSC2
        if "TUSC2_Expression_Bulk" in tumor_adata.obs:
            print(f"\n  Net TME Score vs. Bulk TUSC2 Expression...")
            plot_correlation_scatter(
                df=tumor_adata.obs, x_col="TUSC2_Expression_Bulk", y_col=net_tme_score_col_name,
                title=f"{net_tme_score_col_name.replace('_',' ')} vs. Bulk TUSC2 ({CANCER_TYPE_ABBREV})", # Nicer title
                xlab="Bulk TUSC2 Expression (log2(TPM+1))", ylab=net_tme_score_col_name.replace("_"," "),
                output_dir=net_score_char_output_dir, plot_filename_base=f"{net_tme_score_col_name}_vs_TUSC2_Bulk"
            )
        
        # --- INSERTED: Correlation with LM22 Immune Cell Fractions ---
        print(f"\n  Net TME Score vs. LM22 Immune Cell Fractions...")
        if 'lm22_cell_types' not in locals(): 
            lm22_cell_types = ["B cells naive", "B cells memory", "Plasma cells", "T cells CD8", "T cells CD4 naive", 
                               "T cells CD4 memory resting", "T cells CD4 memory activated", "T cells follicular helper", 
                               "T cells regulatory (Tregs)", "T cells gamma delta", "NK cells resting", "NK cells activated", 
                               "Monocytes", "Macrophages M0", "Macrophages M1", "Macrophages M2", "Dendritic cells resting",
                               "Dendritic cells activated", "Mast cells resting", "Mast cells activated", "Eosinophils", "Neutrophils"]
        available_lm22_cols_for_nettme_corr = [col for col in lm22_cell_types if col in tumor_adata.obs.columns]
        
        if available_lm22_cols_for_nettme_corr:
            net_score_lm22_corr_data = []
            for cell_type_col in available_lm22_cols_for_nettme_corr:
                tumor_adata.obs[cell_type_col] = pd.to_numeric(tumor_adata.obs[cell_type_col], errors='coerce')
                temp_df = tumor_adata.obs[[net_tme_score_col_name, cell_type_col]].dropna()
                if len(temp_df) > 2 and temp_df[cell_type_col].nunique() > 1 and temp_df[net_tme_score_col_name].nunique() > 1:
                    r, p = spearmanr(temp_df[net_tme_score_col_name], temp_df[cell_type_col])
                    net_score_lm22_corr_data.append({"Immune_Cell_Type": cell_type_col, "Spearman_R": r, "p_value": p})
                else:
                    net_score_lm22_corr_data.append({"Immune_Cell_Type": cell_type_col, "Spearman_R": np.nan, "p_value": np.nan})
            net_score_lm22_corr_df = pd.DataFrame(net_score_lm22_corr_data)
            if not net_score_lm22_corr_df.empty:
                net_score_lm22_corr_df = apply_fdr_correction_df(net_score_lm22_corr_df, p_value_col='p_value', q_value_col='FDR_q_value')
                net_score_lm22_corr_df["Annot_Text"] = net_score_lm22_corr_df.apply(lambda r_row: f"{r_row['Spearman_R']:.2f}{get_significance_stars(r_row['FDR_q_value'])}" if pd.notna(r_row['Spearman_R']) else "N/A", axis=1)
                net_score_lm22_corr_df_sorted = net_score_lm22_corr_df.sort_values("Spearman_R", ascending=False).dropna(subset=['Spearman_R'])
                if not net_score_lm22_corr_df_sorted.empty:
                    fig_ns_lm22, ax_ns_lm22 = plt.subplots(figsize=(8, max(6, len(net_score_lm22_corr_df_sorted) * 0.3)))
                    sns.barplot(x="Spearman_R", y="Immune_Cell_Type", data=net_score_lm22_corr_df_sorted, palette="PRGn", edgecolor="black", ax=ax_ns_lm22, hue="Immune_Cell_Type", legend=False)
                    ax_ns_lm22.set_title(f"{net_tme_score_col_name.replace('_',' ')} vs. LM22 Fractions ({CANCER_TYPE_ABBREV})", fontsize=14)
                    ax_ns_lm22.set_xlabel("Spearman Correlation (R)"); ax_ns_lm22.set_ylabel("LM22 Immune Cell Type")
                    ax_ns_lm22.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                    for i, bar in enumerate(ax_ns_lm22.patches): text = net_score_lm22_corr_df_sorted.iloc[i]["Annot_Text"]; val = net_score_lm22_corr_df_sorted.iloc[i]["Spearman_R"]; ax_ns_lm22.text(bar.get_width() + (0.01 if val >=0 else -0.08), bar.get_y() + bar.get_height()/2, text, va='center', ha='left' if val>=0 else 'right', fontsize=8)
                    plt.tight_layout()
                    save_plot_and_data(fig_ns_lm22, net_score_lm22_corr_df_sorted, net_score_char_output_dir, f"{net_tme_score_col_name}_vs_LM22_Corr_Barplot", is_summary_data=True)
                    plt.close(fig_ns_lm22)
                    print(f"  Net TME Score vs LM22 Correlation Barplot/data saved to {net_score_char_output_dir}")
        else: print("    No LM22 columns for Net TME Score correlation.")

        # --- INSERTED: Correlation with Rebuffet NK Phenotype Fractions ---
        print(f"\n  Net TME Score vs. Rebuffet NK Phenotypes...")
        if 'original_rebuffet_cols' not in locals() or not original_rebuffet_cols: original_rebuffet_cols = [col for col in tumor_adata.obs.columns if col.startswith("Rebuffet_") and not col.endswith(("_norm_comp_nk", "_norm_stacked", "_norm"))]
        if original_rebuffet_cols:
            net_score_reb_corr_data = []
            for cell_type_col_rb in original_rebuffet_cols:
                tumor_adata.obs[cell_type_col_rb] = pd.to_numeric(tumor_adata.obs[cell_type_col_rb], errors='coerce')
                temp_df_rb = tumor_adata.obs[[net_tme_score_col_name, cell_type_col_rb]].dropna()
                if len(temp_df_rb) > 2 and temp_df_rb[cell_type_col_rb].nunique() > 1 and temp_df_rb[net_tme_score_col_name].nunique() > 1:
                    r, p = spearmanr(temp_df_rb[net_tme_score_col_name], temp_df_rb[cell_type_col_rb])
                    net_score_reb_corr_data.append({"Rebuffet_NK_Phenotype": cell_type_col_rb, "Spearman_R": r, "p_value": p})
                else:
                    net_score_reb_corr_data.append({"Rebuffet_NK_Phenotype": cell_type_col_rb, "Spearman_R": np.nan, "p_value": np.nan})
            net_score_reb_corr_df = pd.DataFrame(net_score_reb_corr_data)
            if not net_score_reb_corr_df.empty:
                net_score_reb_corr_df = apply_fdr_correction_df(net_score_reb_corr_df, p_value_col='p_value', q_value_col='FDR_q_value')
                net_score_reb_corr_df["Annot_Text"] = net_score_reb_corr_df.apply(lambda r_row: f"{r_row['Spearman_R']:.2f}{get_significance_stars(r_row['FDR_q_value'])}" if pd.notna(r_row['Spearman_R']) else "N/A", axis=1)
                net_score_reb_corr_df_sorted = net_score_reb_corr_df.sort_values("Spearman_R", ascending=False).dropna(subset=['Spearman_R'])
                if not net_score_reb_corr_df_sorted.empty:
                    fig_ns_reb, ax_ns_reb = plt.subplots(figsize=(8, max(4, len(net_score_reb_corr_df_sorted) * 0.4)))
                    sns.barplot(x="Spearman_R", y="Rebuffet_NK_Phenotype", data=net_score_reb_corr_df_sorted, palette="PiYG", edgecolor="black", ax=ax_ns_reb, hue="Rebuffet_NK_Phenotype", legend=False)
                    ax_ns_reb.set_title(f"{net_tme_score_col_name.replace('_',' ')} vs. Rebuffet NK Phenotypes ({CANCER_TYPE_ABBREV})", fontsize=14)
                    ax_ns_reb.set_xlabel("Spearman Correlation (R)"); ax_ns_reb.set_ylabel("Rebuffet NK Phenotype")
                    ax_ns_reb.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                    for i, bar in enumerate(ax_ns_reb.patches): text = net_score_reb_corr_df_sorted.iloc[i]["Annot_Text"]; val = net_score_reb_corr_df_sorted.iloc[i]["Spearman_R"]; ax_ns_reb.text(bar.get_width() + (0.01 if val >=0 else -0.08), bar.get_y() + bar.get_height()/2, text, va='center', ha='left' if val>=0 else 'right', fontsize=8)
                    plt.tight_layout()
                    save_plot_and_data(fig_ns_reb, net_score_reb_corr_df_sorted, net_score_char_output_dir, f"{net_tme_score_col_name}_vs_Rebuffet_Corr_Barplot", is_summary_data=True)
                    plt.close(fig_ns_reb)
                    print(f"  Net TME Score vs Rebuffet Correlation Barplot/data saved to {net_score_char_output_dir}")
        else: print("    No Rebuffet NK columns for Net TME Score correlation.")

        # Distribution by Clinical Variables (Age, Stage, Vital Status)
        # (This part with capturing stats was already correct in the previous version you ran)
        print(f"\n  Net TME Score vs. Clinical Variables...")
        age_col_nettme_char = "Age_at_Diagnosis"
        if age_col_nettme_char in tumor_adata.obs and tumor_adata.obs[age_col_nettme_char].notna().any():
            temp_age_bin_col_nettme_char = "Age_Bin_NetTME_Char_Plot"
            age_bins_nettme_char = [0, 40, 50, 60, 70, 120]; age_labels_nettme_char = ["<40", "40-49", "50-59", "60-69", "≥70"]
            if len(age_labels_nettme_char) != len(age_bins_nettme_char)-1: age_labels_nettme_char = [f"AgeBin{i+1}" for i in range(len(age_bins_nettme_char)-1)]
            tumor_adata.obs[temp_age_bin_col_nettme_char] = pd.cut(tumor_adata.obs[age_col_nettme_char], bins=age_bins_nettme_char, labels=age_labels_nettme_char[:len(age_bins_nettme_char)-1], right=False, include_lowest=True)
            age_val_counts_nettme_char = tumor_adata.obs[temp_age_bin_col_nettme_char].value_counts()
            valid_age_groups_nettme_char = [g for g in age_labels_nettme_char[:len(age_bins_nettme_char)-1] if age_val_counts_nettme_char.get(g,0) >=2]
            if len(valid_age_groups_nettme_char) > 1 :
                stats_df_nettme_age = plot_box_violin_with_stats(df=tumor_adata.obs, x_col=temp_age_bin_col_nettme_char, y_col=net_tme_score_col_name, group_order=valid_age_groups_nettme_char, title=f"{net_tme_score_col_name} by Age Bin ({CANCER_TYPE_ABBREV})", xlab="Age Bin", ylab=net_tme_score_col_name.replace("_"," "), output_dir=net_score_char_output_dir, plot_filename_base=f"{net_tme_score_col_name}_by_AgeBin", test_type='kruskal_multigroup', palette='viridis')
                if stats_df_nettme_age is not None and not stats_df_nettme_age.empty: all_group_comparison_stats_list.append(stats_df_nettme_age.assign(Analysis_Context="NetTMEScore_vs_Clinical_AgeBin", Variable_Compared=net_tme_score_col_name, Grouping_Variable=temp_age_bin_col_nettme_char))
            if temp_age_bin_col_nettme_char in tumor_adata.obs.columns: tumor_adata.obs.drop(columns=[temp_age_bin_col_nettme_char], inplace=True, errors='ignore')
        else: print(f"  Age column '{age_col_nettme_char}' not suitable for Net TME Score by Age.")

        collapsed_stage_col_nettme_char = "Pathologic_Stage_Collapsed" 
        if collapsed_stage_col_nettme_char in tumor_adata.obs:
            stage_order_plot_nettme_char = ["Stage I", "Stage II", "Stage III", "Stage IV"]
            plot_df_nettme_stage = tumor_adata.obs[tumor_adata.obs[collapsed_stage_col_nettme_char].isin(stage_order_plot_nettme_char) & tumor_adata.obs[net_tme_score_col_name].notna()].copy()
            plot_df_nettme_stage[collapsed_stage_col_nettme_char] = pd.Categorical(plot_df_nettme_stage[collapsed_stage_col_nettme_char], categories=stage_order_plot_nettme_char, ordered=True)
            stage_val_counts_nettme_char = plot_df_nettme_stage[collapsed_stage_col_nettme_char].value_counts()
            valid_stages_nettme_char = [s for s in stage_order_plot_nettme_char if stage_val_counts_nettme_char.get(s,0) >=2]
            if len(valid_stages_nettme_char) > 1:
                stats_df_nettme_stage = plot_box_violin_with_stats(df=plot_df_nettme_stage, x_col=collapsed_stage_col_nettme_char, y_col=net_tme_score_col_name, group_order=valid_stages_nettme_char, title=f"{net_tme_score_col_name} by Pathologic Stage ({CANCER_TYPE_ABBREV})", xlab="Pathologic Stage", ylab=net_tme_score_col_name.replace("_"," "), output_dir=net_score_char_output_dir, plot_filename_base=f"{net_tme_score_col_name}_by_Stage", test_type='kruskal_multigroup', palette='plasma')
                if stats_df_nettme_stage is not None and not stats_df_nettme_stage.empty: all_group_comparison_stats_list.append(stats_df_nettme_stage.assign(Analysis_Context="NetTMEScore_vs_Clinical_Stage", Variable_Compared=net_tme_score_col_name, Grouping_Variable=collapsed_stage_col_nettme_char))
        else: print(f"  Collapsed stage column '{collapsed_stage_col_nettme_char}' not available for Net TME Score by Stage.")
        
        vital_status_col_nettme_char = next((col for col in ["Vital_Status", "Vital_Status_Clinical"] if col in tumor_adata.obs.columns), None)
        if vital_status_col_nettme_char:
            plot_df_nettme_vital = tumor_adata.obs[tumor_adata.obs[vital_status_col_nettme_char].astype(str).str.upper().isin(["ALIVE", "DEAD"]) & tumor_adata.obs[net_tme_score_col_name].notna()].copy()
            plot_df_nettme_vital[vital_status_col_nettme_char] = plot_df_nettme_vital[vital_status_col_nettme_char].astype(str).str.upper()
            vital_order_nettme_char = ["ALIVE", "DEAD"]
            plot_df_nettme_vital[vital_status_col_nettme_char] = pd.Categorical(plot_df_nettme_vital[vital_status_col_nettme_char], categories=vital_order_nettme_char, ordered=True)
            if plot_df_nettme_vital[vital_status_col_nettme_char].nunique() == 2 and all(plot_df_nettme_vital[vital_status_col_nettme_char].value_counts().get(g,0)>=2 for g in vital_order_nettme_char):
                stats_df_nettme_vital = plot_box_violin_with_stats(df=plot_df_nettme_vital, x_col=vital_status_col_nettme_char, y_col=net_tme_score_col_name, group_order=vital_order_nettme_char, title=f"{net_tme_score_col_name} by Vital Status ({CANCER_TYPE_ABBREV})", xlab="Vital Status", ylab=net_tme_score_col_name.replace("_"," "), output_dir=net_score_char_output_dir, plot_filename_base=f"{net_tme_score_col_name}_by_VitalStatus", test_type='mannwhitneyu_2group', palette='coolwarm')
                if stats_df_nettme_vital is not None and not stats_df_nettme_vital.empty: all_group_comparison_stats_list.append(stats_df_nettme_vital.assign(Analysis_Context="NetTMEScore_vs_Clinical_VitalStatus", Variable_Compared=net_tme_score_col_name, Grouping_Variable=vital_status_col_nettme_char))
        else: print("  Vital Status column not available for Net TME Score by Vital Status.")

        # Survival Analysis based on Net TME Score
        print(f"\n  Survival Analysis for {net_tme_score_col_name}...")
        if 'actual_event_col' in locals() and actual_event_col and ('actual_time_death_col' in locals() and actual_time_death_col or 'actual_time_followup_col' in locals() and actual_time_followup_col):
            survival_df_net_tme = tumor_adata.obs[[net_tme_score_col_name, actual_event_col, actual_time_death_col, actual_time_followup_col]].copy()
            survival_df_net_tme.dropna(subset=[net_tme_score_col_name, actual_event_col], inplace=True)
            survival_df_net_tme["Survival_Time_KM_NetTME"] = np.where(survival_df_net_tme[actual_event_col].astype(str).str.upper() == 'DEAD', pd.to_numeric(survival_df_net_tme[actual_time_death_col], errors='coerce'), pd.to_numeric(survival_df_net_tme[actual_time_followup_col], errors='coerce'))
            survival_df_net_tme["Event_Observed_KM_NetTME"] = (survival_df_net_tme[actual_event_col].astype(str).str.upper() == 'DEAD').astype(int)
            survival_df_net_tme.dropna(subset=["Survival_Time_KM_NetTME", "Event_Observed_KM_NetTME"], inplace=True)
            survival_df_net_tme = survival_df_net_tme[survival_df_net_tme["Survival_Time_KM_NetTME"] > 0]
            if len(survival_df_net_tme) >= 20:
                median_net_score_km_val = survival_df_net_tme[net_tme_score_col_name].median()
                net_score_group_km_col_name = f"{net_tme_score_col_name}_Group_KM"
                survival_df_net_tme[net_score_group_km_col_name] = np.where(survival_df_net_tme[net_tme_score_col_name] >= median_net_score_km_val, "High Net Score", "Low Net Score")
                net_score_group_order_km_val = ["Low Net Score", "High Net Score"]
                survival_df_net_tme[net_score_group_km_col_name] = pd.Categorical(survival_df_net_tme[net_score_group_km_col_name], categories=net_score_group_order_km_val, ordered=True)
                if survival_df_net_tme[net_score_group_km_col_name].nunique() == 2 and all(survival_df_net_tme[net_score_group_km_col_name].value_counts().get(g,0) >=1 for g in net_score_group_order_km_val):
                    lr_result_nettme_km = plot_kaplan_meier(df=survival_df_net_tme, time_col="Survival_Time_KM_NetTME", event_col="Event_Observed_KM_NetTME", group_col=net_score_group_km_col_name, title=f"KM Survival by {net_tme_score_col_name.replace('_',' ')} ({CANCER_TYPE_ABBREV})", output_dir=net_score_char_output_dir, plot_filename_base=f"KM_{net_tme_score_col_name.replace(' ','_')}", group_order=net_score_group_order_km_val, palette=['cornflowerblue', 'firebrick'])
                    if lr_result_nettme_km: all_survival_results_list.append({"Analysis_Name": f"Survival_by_NetTMEScore_Group", "Cancer_Type": CANCER_TYPE_ABBREV, "Grouping_Variable": net_score_group_km_col_name, "Metric_Split_By": f"{net_tme_score_col_name} (Median)", "LogRank_P_Value": lr_result_nettme_km.p_value, "LogRank_Statistic": lr_result_nettme_km.test_statistic, "Groups_Compared": str(net_score_group_order_km_val)})
                else: print(f"    Skipping {net_tme_score_col_name} KM plot: Need two distinct groups after median split with data.")
            else: print(f"    Not enough valid samples (n={len(survival_df_net_tme)}) for {net_tme_score_col_name} KM analysis.")
        else: print("    Required columns for survival (actual_event_col etc.) not found. Skipping KM for Net TME Score.")
    else:
        print(f"  '{net_tme_score_col_name}' not found or all NaN in tumor_adata.obs. Skipping Characterization.")
else:
    print(f"  tumor_adata is None or TUSC2_Expression_Bulk not in tumor_adata.obs. Skipping Section IX for {CANCER_TYPE_ABBREV}.")

print(f"\n--- End of Section IX: Functional Correlates and Integrative TME Score for {CANCER_TYPE_ABBREV} ---")

# %%
# ==============================================================================
# --- X. Results Aggregation and Significance Highlights (Full Implementation - Corrected Paths) ---
# ==============================================================================
print(f"\n--- X. Aggregating and Highlighting Significant Results for {CANCER_TYPE_ABBREV} ---")

# Base directory for these new aggregated result tables
aggregated_results_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "Aggregated_Analysis_Results")
os.makedirs(aggregated_results_output_dir, exist_ok=True) # Ensure it's created here

FDR_SIGNIFICANCE_THRESHOLD = 0.05
P_VALUE_SIGNIFICANCE_THRESHOLD = 0.05 

# --- X.A. Aggregate Correlation Results ---
print(f"\n--- X.A. Aggregating Correlation Results ---")

# 1. TUSC2 vs. LM22 Fractions (from Section VI.A)
path_tusc2_lm22 = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_Immune_Landscape", "LM22_Correlations", "TUSC2_vs_LM22_Corr_Barplot_data.csv")
_ = aggregate_correlation_results(
    source_data_path=path_tusc2_lm22, analysis_name="TUSC2_vs_LM22_Fractions", 
    id_col1_name="TUSC2_Expression_Bulk", id_col2_name="Immune_Cell_Type",
    output_dir=aggregated_results_output_dir # Pass the directory
)

# 2. TUSC2 vs. Rebuffet NK Phenotypes (from Section VII.B)
path_tusc2_reb = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_NK_Cells", "Rebuffet_NK_Focus", "TUSC2_vs_RebuffetNK_Corr_Barplot_data.csv")
_ = aggregate_correlation_results(
    source_data_path=path_tusc2_reb, analysis_name="TUSC2_vs_Rebuffet_NK", 
    id_col1_name="TUSC2_Expression_Bulk", id_col2_name="Rebuffet_NK_Phenotype",
    output_dir=aggregated_results_output_dir
)

# 3. TUSC2 vs. HiRes NK Gene Expression (from Section VII.C)
path_tusc2_hires = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TUSC2_vs_NK_Cells", "HiRes_NK_Gene_Correlations", "TUSC2_vs_HiRes_NKGenes_FullCorrData.csv")
_ = aggregate_correlation_results(
    source_data_path=path_tusc2_hires, analysis_name="TUSC2_vs_HiRes_NK_Genes", 
    id_col1_name="TUSC2_Expression_Bulk", id_col2_name="HiRes_Gene_Full",
    output_dir=aggregated_results_output_dir
)

# 4. TUSC2 vs. Checkpoint Gene Expression (from Section IX.A)
# Ensure tme_score_output_dir is defined (it was at the start of Section IX)
if 'tme_score_output_dir' not in locals(): # Fallback if running cells out of order
    tme_score_output_dir = os.path.join(CANCER_SPECIFIC_OUTPUT_DIR, "TME_Scores_and_Correlates")
path_tusc2_ckpt = os.path.join(tme_score_output_dir, "TUSC2_vs_Checkpoints", "TUSC2_vs_Checkpoints_Corr_Barplot_data.csv")
_ = aggregate_correlation_results(
    source_data_path=path_tusc2_ckpt, analysis_name="TUSC2_vs_Checkpoint_Genes", 
    id_col1_name="TUSC2_Expression_Bulk", id_col2_name="Checkpoint_Gene",
    output_dir=aggregated_results_output_dir
)

# 5. Net TME Score vs. TUSC2 (from Section IX.B.4)
print(f"\n  Aggregating: Net TME Score vs. TUSC2 Correlation")
if 'net_score_char_output_dir' not in locals() and 'tme_score_output_dir' in locals(): # Ensure path components are defined
    net_score_char_output_dir = os.path.join(tme_score_output_dir, "Net_Score_Characterization")

if 'net_score_char_output_dir' in locals() and 'net_tme_score_col_name' in locals():
    path_nettme_tusc2_scatter_data = os.path.join(net_score_char_output_dir, f"{net_tme_score_col_name}_vs_TUSC2_Bulk_data.csv")
    if os.path.exists(path_nettme_tusc2_scatter_data):
        try:
            nettme_tusc2_scatter_df = pd.read_csv(path_nettme_tusc2_scatter_data)
            if len(nettme_tusc2_scatter_df) > 2 and net_tme_score_col_name in nettme_tusc2_scatter_df.columns and "TUSC2_Expression_Bulk" in nettme_tusc2_scatter_df.columns and nettme_tusc2_scatter_df[net_tme_score_col_name].nunique() > 1 and nettme_tusc2_scatter_df["TUSC2_Expression_Bulk"].nunique() > 1:
                r_nt, p_nt = spearmanr(nettme_tusc2_scatter_df[net_tme_score_col_name], nettme_tusc2_scatter_df["TUSC2_Expression_Bulk"])
                print(f"    Correlation Net TME Score vs. TUSC2: R={r_nt:.3f}, p={p_nt:.2e}")
                single_corr_df = pd.DataFrame([{"Variable1": net_tme_score_col_name, "Variable2": "TUSC2_Expression_Bulk", "Spearman_R": r_nt, "p_value": p_nt}])
                single_corr_df.to_csv(os.path.join(aggregated_results_output_dir, f"{CANCER_TYPE_ABBREV}_NetTMEScore_vs_TUSC2_Correlation.csv"), index=False)
                print(f"    Saved Net TME Score vs TUSC2 correlation to {aggregated_results_output_dir}")
            else: print(f"    Not enough data/variance in {path_nettme_tusc2_scatter_data} for Net TME Score vs TUSC2 correlation.")
        except Exception as e_nettme_tusc2: print(f"    ERROR processing Net TME Score vs TUSC2 data from {path_nettme_tusc2_scatter_data}: {e_nettme_tusc2}")
    else: print(f"    WARNING: Data file not found for NetTMEScore vs TUSC2 scatter: {path_nettme_tusc2_scatter_data}")
else: print("    WARNING: Path components for Net TME Score vs TUSC2 aggregation missing.")

# 6. Net TME Score vs. LM22 Fractions (from Section IX.B.4)
if 'net_score_char_output_dir' in locals() and 'net_tme_score_col_name' in locals():
    path_nettme_lm22 = os.path.join(net_score_char_output_dir, f"{net_tme_score_col_name}_vs_LM22_Corr_Barplot_data.csv")
    _ = aggregate_correlation_results(
        source_data_path=path_nettme_lm22, analysis_name="NetTMEScore_vs_LM22_Fractions", 
        id_col1_name=net_tme_score_col_name, id_col2_name="Immune_Cell_Type",
        output_dir=aggregated_results_output_dir
    )
else: print("    WARNING: Path components for NetTMEScore vs LM22 aggregation missing.")

# 7. Net TME Score vs. Rebuffet NK Phenotypes (from Section IX.B.4)
if 'net_score_char_output_dir' in locals() and 'net_tme_score_col_name' in locals():
    path_nettme_reb = os.path.join(net_score_char_output_dir, f"{net_tme_score_col_name}_vs_Rebuffet_Corr_Barplot_data.csv")
    _ = aggregate_correlation_results(
        source_data_path=path_nettme_reb, analysis_name="NetTMEScore_vs_Rebuffet_NK", 
        id_col1_name=net_tme_score_col_name, id_col2_name="Rebuffet_NK_Phenotype",
        output_dir=aggregated_results_output_dir
    )
else: print("    WARNING: Path components for NetTMEScore vs Rebuffet aggregation missing.")

# --- X.B. Aggregate Group Comparison Statistics (from Boxplots/Violins) ---
print(f"\n--- X.B. Aggregating Group Comparison Statistics (from Boxplots/Violins) ---")
if 'all_group_comparison_stats_list' in globals() and isinstance(all_group_comparison_stats_list, list) and all_group_comparison_stats_list:
    try:
        compiled_group_stats_df = pd.concat(all_group_comparison_stats_list, ignore_index=True)
        desired_col_order = ["Analysis_Context", "Variable_Compared", "Grouping_Variable", "group1", "group2", "test_type", "statistic", "p_value", "q_value_fdr"]
        existing_cols_in_df = [col for col in desired_col_order if col in compiled_group_stats_df.columns]
        remaining_cols = [col for col in compiled_group_stats_df.columns if col not in existing_cols_in_df]
        final_col_order = existing_cols_in_df + remaining_cols
        compiled_group_stats_df = compiled_group_stats_df[final_col_order]
        compiled_group_stats_df.to_csv(os.path.join(aggregated_results_output_dir, f"{CANCER_TYPE_ABBREV}_Compiled_AllGroupComparison_Stats.csv"), index=False)
        print(f"  Compiled all group comparison statistics ({len(compiled_group_stats_df)} tests) saved.")
        print(f"\n  Significant Group Comparisons (FDR/P-value < {P_VALUE_SIGNIFICANCE_THRESHOLD}):")
        significant_group_comparisons = pd.DataFrame()
        if 'q_value_fdr' in compiled_group_stats_df.columns and compiled_group_stats_df['q_value_fdr'].notna().any():
            significant_group_comparisons = compiled_group_stats_df[compiled_group_stats_df['q_value_fdr'] < FDR_SIGNIFICANCE_THRESHOLD].copy()
            significant_group_comparisons.sort_values(by=['Analysis_Context', 'q_value_fdr'], inplace=True)
            print(f"  (Filtered by FDR q-value < {FDR_SIGNIFICANCE_THRESHOLD})")
        elif 'p_value' in compiled_group_stats_df.columns and compiled_group_stats_df['p_value'].notna().any(): 
            significant_group_comparisons = compiled_group_stats_df[compiled_group_stats_df['p_value'] < P_VALUE_SIGNIFICANCE_THRESHOLD].copy()
            significant_group_comparisons.sort_values(by=['Analysis_Context', 'p_value'], inplace=True)
            print(f"  (Filtered by p-value < {P_VALUE_SIGNIFICANCE_THRESHOLD} as FDR q-value was not available/applicable for all entries)")
        else: print("    Neither 'q_value_fdr' nor 'p_value' found or usable for significance filtering in compiled group stats.")
        if not significant_group_comparisons.empty:
            display(significant_group_comparisons)
            significant_group_comparisons.to_csv(os.path.join(aggregated_results_output_dir, f"{CANCER_TYPE_ABBREV}_Compiled_SignificantGroupComparison_Stats.csv"), index=False)
            print(f"  Saved significant group comparison statistics.")
        else: print("    No significant group comparisons found based on available p/q values and threshold.")
    except Exception as e_group_concat: print(f"  ERROR concatenating or processing group comparison stats: {e_group_concat}")
else:
    print("  'all_group_comparison_stats_list' not found, not a list, or empty. Cannot compile group comparison stats.")
    print("  Please ensure plotting functions in Sections V-IX were correctly modified to append their stats DataFrames to this list.")

# --- X.C. Aggregate Survival Analysis Results ---
print(f"\n--- X.C. Aggregating Survival Analysis (Log-Rank P-values) ---")
if 'all_survival_results_list' in globals() and isinstance(all_survival_results_list, list) and all_survival_results_list:
    try:
        compiled_survival_df = pd.DataFrame(all_survival_results_list)
        desired_survival_cols = ["Analysis_Name", "Cancer_Type", "Grouping_Variable", "Metric_Split_By", "LogRank_P_Value", "LogRank_Statistic", "Groups_Compared"]
        existing_survival_cols = [col for col in desired_survival_cols if col in compiled_survival_df.columns]
        remaining_survival_cols = [col for col in compiled_survival_df.columns if col not in existing_survival_cols]
        final_survival_col_order = existing_survival_cols + remaining_survival_cols
        compiled_survival_df = compiled_survival_df[final_survival_col_order]
        compiled_survival_df.sort_values(by="LogRank_P_Value", inplace=True)
        compiled_survival_df.to_csv(os.path.join(aggregated_results_output_dir, f"{CANCER_TYPE_ABBREV}_Compiled_AllSurvival_LogRank_Stats.csv"), index=False)
        print(f"  Compiled all survival log-rank statistics ({len(compiled_survival_df)} tests) saved.")
        print(f"\n  Survival Analyses with LogRank P-value < {P_VALUE_SIGNIFICANCE_THRESHOLD}:")
        significant_survival = compiled_survival_df[compiled_survival_df['LogRank_P_Value'] < P_VALUE_SIGNIFICANCE_THRESHOLD].copy()
        if not significant_survival.empty:
            display(significant_survival)
            significant_survival.to_csv(os.path.join(aggregated_results_output_dir, f"{CANCER_TYPE_ABBREV}_Compiled_SignificantSurvival_Stats.csv"), index=False)
            print(f"  Saved significant survival statistics.")
        else: print(f"    No significant survival analyses found at p < {P_VALUE_SIGNIFICANCE_THRESHOLD}.")
    except Exception as e_survival_concat: print(f"  ERROR creating DataFrame from survival results: {e_survival_concat}")
else:
    print("  'all_survival_results_list' not found, not a list, or empty. Cannot compile survival stats.")
    print("  Please ensure plot_kaplan_meier function returns log-rank results and calls are modified to append to this list.")

print(f"\n--- End of Section X: Results Aggregation for {CANCER_TYPE_ABBREV} ---")

# --- XI. Summary, Conclusions, and Next Steps (Placeholder) ---
print(f"\n--- XI. Summary, Conclusions, and Next Steps for {CANCER_TYPE_ABBREV} (Placeholder) ---")
print("Review aggregated CSV files in:", aggregated_results_output_dir)
print("Identify key patterns, concordant findings, and formulate biological hypotheses.")


