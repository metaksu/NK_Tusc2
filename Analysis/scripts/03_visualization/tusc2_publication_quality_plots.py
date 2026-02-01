#!/usr/bin/env python3
"""
TUSC2 Publication-Quality Visualization Script
==============================================

Creates comprehensive, publication-ready visualizations of TUSC2-NK subtype analysis
with proper statistical context and biological interpretation.

Key Features:
- Modern plotting with seaborn/matplotlib
- Proper statistical annotation
- Biological insight focus
- Publication-quality aesthetics
- Organized output structure

Author: AI Assistant
Date: January 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'Arial',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.dpi': 300
})

# %% 
# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================

# Base paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "Combined_NK_TUSC2_Analysis_Output" / "TUSC2_Focused_Analysis" / "statistical_results"
OUTPUT_DIR = BASE_DIR / "Combined_NK_TUSC2_Analysis_Output" / "TUSC2_Focused_Analysis" / "publication_figures"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files
FREQUENCY_FILE = DATA_DIR / "tusc2_frequency_all_contexts.csv"
ENRICHMENT_FILE = DATA_DIR / "tusc2_enrichment_all_contexts.csv"
CONDITIONAL_FILE = DATA_DIR / "tusc2_conditional_all_contexts.csv"
OVERVIEW_FILE = DATA_DIR / "tusc2_expression_overview.csv"

# Color schemes
CONTEXT_COLORS = {
    'Rebuffet_Blood': '#1f77b4',    # Blue
    'Normal': '#2ca02c',            # Green  
    'Tumor': '#d62728',             # Red
    'Tang_Blood': '#ff7f0e'         # Orange (excluded but defined)
}

SUBTYPE_COLORS = {
    # Rebuffet subtypes
    'NK1A': '#1f77b4', 'NK1B': '#aec7e8', 'NK1C': '#ff7f0e',
    'NK2': '#ffbb78', 'NK3': '#2ca02c', 'NKint': '#98df8a',
    
    # Tang subtypes - organized by CD56/CD16 expression
    'CD56brightCD16hi': '#8c564b',
    'CD56brightCD16lo-c1-GZMH': '#e377c2', 'CD56brightCD16lo-c2-IL7R-RGS1lo': '#f7b6d3',
    'CD56brightCD16lo-c3-CCL3': '#c5b0d5', 'CD56brightCD16lo-c4-IL7R': '#9467bd',
    'CD56brightCD16lo-c5-CREM': '#ff9896',
    'CD56dimCD16hi-c1-IL32': '#17becf', 'CD56dimCD16hi-c2-CX3CR1': '#9edae5',
    'CD56dimCD16hi-c3-ZNF90': '#98df8a', 'CD56dimCD16hi-c4-NFKBIA': '#c7c7c7',
    'CD56dimCD16hi-c5-MKI67': '#d62728', 'CD56dimCD16hi-c6-DNAJB1': '#ff9896',
    'CD56dimCD16hi-c7-NR4A3': '#ffbb78', 'CD56dimCD16hi-c8-KLRC2': '#bcbd22'
}

# %% 
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_data():
    """Load all analysis results into pandas DataFrames"""
    print("Loading analysis data...")
    
    data = {}
    files = {
        'frequency': FREQUENCY_FILE,
        'enrichment': ENRICHMENT_FILE,
        'conditional': CONDITIONAL_FILE,
        'overview': OVERVIEW_FILE
    }
    
    for key, file_path in files.items():
        if file_path.exists():
            data[key] = pd.read_csv(file_path, index_col=0)
            print(f"✓ Loaded {key}: {data[key].shape}")
        else:
            print(f"✗ Missing {key} file: {file_path}")
    
    return data

def filter_contexts(df, exclude_tang_blood=True):
    """Filter DataFrame to exclude Tang_Blood context"""
    if exclude_tang_blood:
        return df[df['context'] != 'Tang_Blood'].copy()
    return df.copy()

def save_figure(fig, filename, dpi=300):
    """Save figure with publication quality settings"""
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")

def add_significance_bars(ax, data, x_col, y_col, pval_col, 
                         significance_threshold=0.05, bar_height=0.02):
    """Add significance bars to bar plots"""
    significant_data = data[data[pval_col] < significance_threshold]
    
    if len(significant_data) > 0:
        max_y = ax.get_ylim()[1]
        for i, (idx, row) in enumerate(significant_data.iterrows()):
            x_pos = row[x_col] if isinstance(row[x_col], (int, float)) else i
            y_pos = max_y * (1 + bar_height)
            
            # Add significance star
            ax.text(x_pos, y_pos, '*', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='red')

def format_subtype_names(subtype_name):
    """Format subtype names for better readability"""
    # Handle Tang subtypes
    if 'CD56' in subtype_name:
        # Split and format Tang subtypes
        parts = subtype_name.split('-')
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}\n{parts[2]}"
        return subtype_name
    
    # Handle Rebuffet subtypes (already short)
    return subtype_name

# %% 
# =============================================================================
# FIGURE 1: TUSC2 EXPRESSION OVERVIEW
# =============================================================================

def create_figure_1_overview(data):
    """Create Figure 1: TUSC2 Expression Overview"""
    print("\n=== Creating Figure 1: TUSC2 Expression Overview ===")
    
    overview_df = filter_contexts(data['overview'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('TUSC2 Expression Overview Across NK Cell Contexts', fontsize=16, fontweight='bold')
    
    # Panel A: Overall TUSC2+ frequency
    ax1 = axes[0]
    bars = ax1.bar(overview_df['context'], overview_df['percent_positive'], 
                   color=[CONTEXT_COLORS[ctx] for ctx in overview_df['context']])
    
    ax1.set_title('A. TUSC2+ Cell Frequency by Context', fontweight='bold')
    ax1.set_xlabel('Context')
    ax1.set_ylabel('TUSC2+ Cells (%)')
    ax1.set_ylim(0, max(overview_df['percent_positive']) * 1.1)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, overview_df['percent_positive']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Panel B: Expression distribution
    ax2 = axes[1]
    contexts = overview_df['context'].tolist()
    means = overview_df['mean_expression'].tolist()
    stds = overview_df['std_expression'].tolist()
    
    bars = ax2.bar(contexts, means, yerr=stds, capsize=5,
                   color=[CONTEXT_COLORS[ctx] for ctx in contexts],
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    ax2.set_title('B. Mean TUSC2 Expression by Context', fontweight='bold')
    ax2.set_xlabel('Context')
    ax2.set_ylabel('Mean Expression Level')
    ax2.set_ylim(0, max(means) * 1.2)
    
    # Add cell count annotations
    for i, (ctx, total) in enumerate(zip(contexts, overview_df['total_cells'])):
        ax2.text(i, means[i] + stds[i] + 0.05, f'n={total:,}', 
                ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    save_figure(fig, 'Figure_1_TUSC2_Expression_Overview.png')
    plt.close()

# %% 
# =============================================================================
# FIGURE 2: SUBTYPE-SPECIFIC TUSC2 FREQUENCY ANALYSIS
# =============================================================================

def create_figure_2_frequency(data):
    """Create Figure 2: Subtype-Specific TUSC2 Frequency Analysis"""
    print("\n=== Creating Figure 2: Subtype-Specific TUSC2 Frequency ===")
    
    freq_df = filter_contexts(data['frequency'])
    
    # Create separate plots for each context
    contexts = ['Rebuffet_Blood', 'Normal', 'Tumor']
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('TUSC2+ Cell Frequency by NK Cell Subtype', fontsize=16, fontweight='bold')
    
    for i, context in enumerate(contexts):
        ax = axes[i]
        context_data = freq_df[freq_df['context'] == context].copy()
        context_data = context_data.sort_values('tusc2_frequency_percent', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(context_data)), context_data['tusc2_frequency_percent'],
                      color=[SUBTYPE_COLORS.get(subtype, '#808080') for subtype in context_data['subtype']])
        
        # Format y-axis with subtype names
        ax.set_yticks(range(len(context_data)))
        ax.set_yticklabels([format_subtype_names(s) for s in context_data['subtype']], 
                          fontsize=10)
        
        # Formatting
        ax.set_xlabel('TUSC2+ Cells (%)')
        ax.set_title(f'{context.replace("_", " ")} NK Subtypes', fontweight='bold')
        ax.set_xlim(0, max(context_data['tusc2_frequency_percent']) * 1.1)
        
        # Add percentage labels
        for j, (bar, pct) in enumerate(zip(bars, context_data['tusc2_frequency_percent'])):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        # Add cell count labels
        for j, (bar, total, tusc2_pos) in enumerate(zip(bars, context_data['total_cells'], 
                                                        context_data['tusc2_positive_cells'])):
            ax.text(0.5, bar.get_y() + bar.get_height()/2,
                   f'{tusc2_pos}/{total}', va='center', ha='left', 
                   fontsize=8, style='italic', color='white' if bar.get_width() > 10 else 'black')
    
    plt.tight_layout()
    save_figure(fig, 'Figure_2_Subtype_TUSC2_Frequency.png')
    plt.close()

# %% 
# =============================================================================
# FIGURE 3: TUSC2 ENRICHMENT ANALYSIS
# =============================================================================

def create_figure_3_enrichment(data):
    """Create Figure 3: TUSC2 Enrichment Analysis"""
    print("\n=== Creating Figure 3: TUSC2 Enrichment Analysis ===")
    
    enrich_df = filter_contexts(data['enrichment'])
    
    # Create separate plots for each context
    contexts = ['Rebuffet_Blood', 'Normal', 'Tumor']
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('TUSC2 Enrichment Analysis by NK Cell Subtype', fontsize=16, fontweight='bold')
    
    for i, context in enumerate(contexts):
        ax = axes[i]
        context_data = enrich_df[enrich_df['context'] == context].copy()
        context_data = context_data.sort_values('enrichment_ratio', ascending=True)
        
        # Color bars by significance
        bar_colors = ['red' if sig else 'lightgray' for sig in context_data['significant']]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(context_data)), context_data['enrichment_ratio'],
                      color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add reference line at enrichment ratio = 1
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        # Format y-axis with subtype names
        ax.set_yticks(range(len(context_data)))
        ax.set_yticklabels([format_subtype_names(s) for s in context_data['subtype']], 
                          fontsize=10)
        
        # Formatting
        ax.set_xlabel('Enrichment Ratio (TUSC2+ / Expected)')
        ax.set_title(f'{context.replace("_", " ")} NK Subtypes', fontweight='bold')
        ax.set_xlim(0, max(context_data['enrichment_ratio']) * 1.1)
        
        # Add enrichment ratio labels
        for j, (bar, ratio, sig) in enumerate(zip(bars, context_data['enrichment_ratio'], 
                                                  context_data['significant'])):
            label_color = 'white' if sig else 'black'
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{ratio:.2f}', va='center', fontweight='bold', fontsize=9, color=label_color)
        
        # Add significance stars
        for j, (bar, sig) in enumerate(zip(bars, context_data['significant'])):
            if sig:
                ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                       '*', va='center', fontweight='bold', fontsize=12, color='red')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='lightgray', alpha=0.7, label='Not Significant')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    save_figure(fig, 'Figure_3_TUSC2_Enrichment_Analysis.png')
    plt.close()

# %% 
# =============================================================================
# FIGURE 4: CONDITIONAL EXPRESSION IN TUSC2+ CELLS
# =============================================================================

def create_figure_4_conditional(data):
    """Create Figure 4: Conditional Expression in TUSC2+ Cells"""
    print("\n=== Creating Figure 4: Conditional Expression in TUSC2+ Cells ===")
    
    cond_df = filter_contexts(data['conditional'])
    
    # Create separate plots for each context
    contexts = ['Rebuffet_Blood', 'Normal', 'Tumor']
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('TUSC2 Expression Levels in TUSC2+ Cells Only', fontsize=16, fontweight='bold')
    
    for i, context in enumerate(contexts):
        ax = axes[i]
        context_data = cond_df[cond_df['context'] == context].copy()
        context_data = context_data.sort_values('mean_expression', ascending=True)
        
        # Create horizontal bar plot with error bars
        bars = ax.barh(range(len(context_data)), context_data['mean_expression'],
                      xerr=context_data['std_expression'], capsize=3,
                      color=[SUBTYPE_COLORS.get(subtype, '#808080') for subtype in context_data['subtype']],
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        # Format y-axis with subtype names
        ax.set_yticks(range(len(context_data)))
        ax.set_yticklabels([format_subtype_names(s) for s in context_data['subtype']], 
                          fontsize=10)
        
        # Formatting
        ax.set_xlabel('Mean TUSC2 Expression (TUSC2+ cells only)')
        ax.set_title(f'{context.replace("_", " ")} NK Subtypes', fontweight='bold')
        ax.set_xlim(0, max(context_data['mean_expression'] + context_data['std_expression']) * 1.1)
        
        # Add expression level labels
        for j, (bar, mean_expr, n_cells) in enumerate(zip(bars, context_data['mean_expression'], 
                                                          context_data['n_tusc2_positive_cells'])):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{mean_expr:.2f}', va='center', fontweight='bold', fontsize=9)
            
            # Add cell count
            ax.text(0.05, bar.get_y() + bar.get_height()/2,
                   f'n={n_cells}', va='center', ha='left', 
                   fontsize=8, style='italic', color='white' if bar.get_width() > 1 else 'black')
    
    plt.tight_layout()
    save_figure(fig, 'Figure_4_Conditional_Expression.png')
    plt.close()

# %% 
# =============================================================================
# FIGURE 5: COMPARATIVE ANALYSIS - KEY BIOLOGICAL INSIGHTS
# =============================================================================

def create_figure_5_comparative(data):
    """Create Figure 5: Comparative Analysis - Key Biological Insights"""
    print("\n=== Creating Figure 5: Comparative Analysis ===")
    
    freq_df = filter_contexts(data['frequency'])
    enrich_df = filter_contexts(data['enrichment'])
    cond_df = filter_contexts(data['conditional'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Comparative Analysis: Key TUSC2-NK Subtype Associations', fontsize=16, fontweight='bold')
    
    # Panel A: MKI67 subtype focus across contexts
    ax1 = plt.subplot(2, 2, 1)
    mki67_data = freq_df[freq_df['subtype'].str.contains('MKI67', na=False)]
    
    bars = ax1.bar(mki67_data['context'], mki67_data['tusc2_frequency_percent'],
                   color=[CONTEXT_COLORS[ctx] for ctx in mki67_data['context']])
    
    ax1.set_title('A. MKI67+ Subtypes: TUSC2+ Frequency', fontweight='bold')
    ax1.set_ylabel('TUSC2+ Cells (%)')
    ax1.set_ylim(0, max(mki67_data['tusc2_frequency_percent']) * 1.1)
    
    # Add labels
    for bar, pct in zip(bars, mki67_data['tusc2_frequency_percent']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Panel B: Top enriched subtypes comparison
    ax2 = plt.subplot(2, 2, 2)
    top_enriched = enrich_df[enrich_df['significant']].groupby('context').apply(
        lambda x: x.nlargest(2, 'enrichment_ratio')
    ).reset_index(drop=True)
    
    # Create grouped bar plot
    contexts = top_enriched['context'].unique()
    x_pos = np.arange(len(contexts))
    
    for i, context in enumerate(contexts):
        context_top = top_enriched[top_enriched['context'] == context]
        for j, (_, row) in enumerate(context_top.iterrows()):
            ax2.bar(x_pos[i] + j*0.4 - 0.2, row['enrichment_ratio'], 
                   width=0.35, color=CONTEXT_COLORS[context], alpha=0.7)
            
            # Add subtype label
            ax2.text(x_pos[i] + j*0.4 - 0.2, row['enrichment_ratio'] + 0.03,
                    row['subtype'].split('-')[-1] if '-' in row['subtype'] else row['subtype'],
                    ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax2.set_title('B. Top Enriched Subtypes (Significant)', fontweight='bold')
    ax2.set_ylabel('Enrichment Ratio')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(contexts, rotation=45)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Panel C: Expression range comparison by data source
    ax3 = plt.subplot(2, 1, 2)
    
    # Separate by data source
    rebuffet_cond = cond_df[cond_df['context'] == 'Rebuffet_Blood']
    tang_cond = cond_df[cond_df['context'].isin(['Normal', 'Tumor'])]
    
    # Create violin plots
    data_for_violin = []
    labels = []
    
    # Rebuffet data
    for _, row in rebuffet_cond.iterrows():
        data_for_violin.append([row['mean_expression']] * int(row['n_tusc2_positive_cells']))
        labels.append(f"Rebuffet\n{row['subtype']}")
    
    # Tang data
    for _, row in tang_cond.iterrows():
        data_for_violin.append([row['mean_expression']] * int(row['n_tusc2_positive_cells']))
        labels.append(f"{row['context']}\n{row['subtype'].split('-')[-1] if '-' in row['subtype'] else row['subtype']}")
    
    # Create box plot instead of violin for cleaner visualization
    means = [np.mean(data) for data in data_for_violin]
    stds = [np.std(data) for data in data_for_violin]
    
    # Separate Rebuffet and Tang data
    rebuffet_means = means[:len(rebuffet_cond)]
    rebuffet_stds = stds[:len(rebuffet_cond)]
    tang_means = means[len(rebuffet_cond):]
    tang_stds = stds[len(rebuffet_cond):]
    
    # Plot
    x_reb = np.arange(len(rebuffet_means))
    x_tang = np.arange(len(tang_means)) + len(rebuffet_means) + 1
    
    ax3.bar(x_reb, rebuffet_means, yerr=rebuffet_stds, capsize=3,
           color='#1f77b4', alpha=0.7, label='Rebuffet Blood')
    ax3.bar(x_tang, tang_means, yerr=tang_stds, capsize=3,
           color='#2ca02c', alpha=0.7, label='Tang (Normal/Tumor)')
    
    ax3.set_title('C. TUSC2 Expression Range by Data Source', fontweight='bold')
    ax3.set_ylabel('Mean Expression in TUSC2+ Cells')
    ax3.set_xlabel('Subtypes')
    ax3.legend()
    
    # Add divider line
    ax3.axvline(x=len(rebuffet_means) + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_5_Comparative_Analysis.png')
    plt.close()

# %% 
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=== TUSC2 PUBLICATION-QUALITY VISUALIZATION SCRIPT ===")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    data = load_data()
    
    if not data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Generate figures
    try:
        create_figure_1_overview(data)
        create_figure_2_frequency(data)
        create_figure_3_enrichment(data)
        create_figure_4_conditional(data)
        create_figure_5_comparative(data)
        
        print("\n✅ All figures created successfully!")
        print(f"📁 Output location: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error creating figures: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 