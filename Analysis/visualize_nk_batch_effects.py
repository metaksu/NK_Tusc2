#!/usr/bin/env python3
"""
NK Cell Dataset Batch Effect Visualization
==========================================

Creates visualizations to show the identified batch effects:
1. Dataset imbalance  
2. Technical covariate differences
3. NK subtype distribution across batches
4. Donor contribution patterns

Based on the batch effect analysis results.
"""

# === ENVIRONMENT SETUP ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["NUMBA_DISABLE_CUDA"] = "1"   # Disable CUDA for numba
os.environ["OMP_NUM_THREADS"] = "1"      # Prevent threading conflicts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Import scanpy
try:
    import scanpy as sc
    sc.settings.verbosity = 0
    sc.settings.autoshow = False
    print(f"✓ Scanpy {sc.__version__} imported successfully")
except Exception as e:
    print(f"✗ CRITICAL ERROR: Scanpy import failed: {e}")
    exit(1)

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
NK_DATA_FILE = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
OUTPUT_DIR = Path("outputs/batch_effect_analysis/figures")

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_data_with_cmv():
    """Load data and add CMV status."""
    print("Loading NK dataset for visualization...")
    adata = sc.read_h5ad(NK_DATA_FILE)
    
    # Extract CMV status
    def get_cmv_status(donor_name):
        if pd.isna(donor_name):
            return "Unknown"
        donor_str = str(donor_name)
        if "CMVpos" in donor_str:
            return "CMV_Positive"
        elif "CMVneg" in donor_str:
            return "CMV_Negative"
        else:
            return "CMV_Unknown"
    
    adata.obs['CMV_Status'] = adata.obs['donor'].apply(get_cmv_status)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    return adata

def plot_dataset_imbalance(adata):
    """Visualize severe dataset imbalance."""
    print("Creating dataset imbalance visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of dataset sizes
    dataset_counts = adata.obs['Dataset'].value_counts().sort_index()
    bars = ax1.bar(dataset_counts.index, dataset_counts.values, 
                   color=['#ff4757', '#ffa502', '#2ed573', '#5352ed'])
    ax1.set_title('SEVERE Dataset Imbalance\n(16.6:1 ratio)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Cells')
    ax1.set_ylim(0, dataset_counts.max() * 1.1)
    
    # Add count labels on bars
    for bar, count in zip(bars, dataset_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{count:,}\n({count/len(adata.obs)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%',
            colors=['#ff4757', '#ffa502', '#2ed573', '#5352ed'], startangle=90)
    ax2.set_title('Dataset Distribution\n(HIGHLY UNBALANCED)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_imbalance.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR / 'dataset_imbalance.png'}")
    plt.close()

def plot_technical_batch_effects(adata):
    """Visualize technical batch effects."""
    print("Creating technical batch effects visualization...")
    
    technical_vars = ['nCount_RNA', 'nFeature_RNA', 'percent.mito', 'percent.ribo']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(technical_vars):
        ax = axes[i]
        
        # Violin plot by Dataset
        sns.violinplot(data=adata.obs, x='Dataset', y=var, ax=ax, palette='Set2')
        ax.set_title(f'{var} by Dataset\n(Strong Batch Effect)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Calculate and display CV
        dataset_means = adata.obs.groupby('Dataset')[var].mean()
        cv = dataset_means.std() / dataset_means.mean()
        ax.text(0.02, 0.98, f'CV = {cv:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                verticalalignment='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'technical_batch_effects.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR / 'technical_batch_effects.png'}")
    plt.close()

def plot_nk_subtype_batch_bias(adata):
    """Visualize NK subtype distribution bias across batches."""
    print("Creating NK subtype batch bias visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. NK subtypes by Dataset (stacked bar)
    ax1 = axes[0, 0]
    crosstab_dataset = pd.crosstab(adata.obs['Dataset'], adata.obs['ident'], normalize='index')
    crosstab_dataset.plot(kind='bar', stacked=True, ax=ax1, width=0.8)
    ax1.set_title('NK Subtype Distribution by Dataset\n(BIASED - Not Even)', fontweight='bold')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Proportion')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=0)
    
    # 2. NK subtypes by CMV Status (stacked bar)
    ax2 = axes[0, 1]
    crosstab_cmv = pd.crosstab(adata.obs['CMV_Status'], adata.obs['ident'], normalize='index')
    crosstab_cmv.plot(kind='bar', stacked=True, ax=ax2, width=0.8)
    ax2.set_title('NK Subtype Distribution by CMV Status\n(STRONG BIAS)', fontweight='bold')
    ax2.set_xlabel('CMV Status')
    ax2.set_ylabel('Proportion')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Heatmap of NK subtypes vs top donors
    ax3 = axes[1, 0]
    top_donors = adata.obs['donor'].value_counts().head(8).index
    subset_data = adata.obs[adata.obs['donor'].isin(top_donors)]
    crosstab_donor = pd.crosstab(subset_data['donor'], subset_data['ident'], normalize='index')
    sns.heatmap(crosstab_donor, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax3)
    ax3.set_title('NK Subtype Proportions by Top Donors\n(HIGHLY VARIABLE)', fontweight='bold')
    ax3.set_xlabel('NK Subtype')
    ax3.set_ylabel('Donor')
    
    # 4. NK3 proportion by donor (since it's most variable)
    ax4 = axes[1, 1]
    nk3_props = []
    donor_names = []
    for donor in adata.obs['donor'].value_counts().index:
        donor_data = adata.obs[adata.obs['donor'] == donor]
        nk3_prop = (donor_data['ident'] == 'NK3').mean()
        nk3_props.append(nk3_prop)
        donor_names.append(donor)
    
    bars = ax4.bar(range(len(donor_names)), nk3_props, 
                   color=['red' if prop > 0.4 else 'orange' if prop > 0.2 else 'green' 
                          for prop in nk3_props])
    ax4.set_title('NK3 Proportion by Donor\n(Range: 12% to 58%!)', fontweight='bold')
    ax4.set_xlabel('Donor')
    ax4.set_ylabel('NK3 Proportion')
    ax4.set_xticks(range(len(donor_names)))
    ax4.set_xticklabels(donor_names, rotation=45, ha='right')
    
    # Add proportion labels
    for i, prop in enumerate(nk3_props):
        ax4.text(i, prop + 0.01, f'{prop:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nk_subtype_batch_bias.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR / 'nk_subtype_batch_bias.png'}")
    plt.close()

def plot_donor_contribution_analysis(adata):
    """Visualize donor contribution patterns."""
    print("Creating donor contribution analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Donor contribution sizes
    donor_counts = adata.obs['donor'].value_counts()
    colors = ['red' if count > 5000 else 'orange' if count > 2000 else 'green' 
              for count in donor_counts.values]
    
    bars = ax1.bar(range(len(donor_counts)), donor_counts.values, color=colors)
    ax1.set_title('Donor Contribution Imbalance\n(7.9:1 ratio)', fontweight='bold')
    ax1.set_xlabel('Donor (ordered by contribution)')
    ax1.set_ylabel('Number of Cells')
    ax1.set_xticks(range(len(donor_counts)))
    ax1.set_xticklabels(donor_counts.index, rotation=45, ha='right')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, donor_counts.values)):
        ax1.text(i, count + 100, f'{count:,}', ha='center', va='bottom', 
                rotation=90, fontweight='bold')
    
    # CMV status distribution
    cmv_counts = adata.obs['CMV_Status'].value_counts()
    wedges, texts, autotexts = ax2.pie(cmv_counts.values, labels=cmv_counts.index, 
                                       autopct='%1.1f%%', startangle=90,
                                       colors=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_title('CMV Status Distribution\n(Potential Biological Confounder)', fontweight='bold')
    
    # Make text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'donor_contribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR / 'donor_contribution_analysis.png'}")
    plt.close()

def create_batch_effect_summary_plot(adata):
    """Create a comprehensive summary of all batch effects."""
    print("Creating comprehensive batch effect summary...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('NK Cell Dataset: CRITICAL BATCH EFFECTS REQUIRING CORRECTION', 
                 fontsize=20, fontweight='bold', color='red')
    
    # 1. Dataset imbalance (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    dataset_counts = adata.obs['Dataset'].value_counts().sort_index()
    bars = ax1.bar(dataset_counts.index, dataset_counts.values, 
                   color=['#ff4757', '#ffa502', '#2ed573', '#5352ed'])
    ax1.set_title('🚨 SEVERE Dataset Imbalance\n(16.6:1 ratio)', fontweight='bold', color='red')
    ax1.set_ylabel('Cells')
    for bar, count in zip(bars, dataset_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Technical CV summary (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    tech_vars = ['nCount_RNA', 'nFeature_RNA', 'percent.mito']
    cvs = []
    for var in tech_vars:
        dataset_means = adata.obs.groupby('Dataset')[var].mean()
        cv = dataset_means.std() / dataset_means.mean()
        cvs.append(cv)
    
    bars = ax2.bar(tech_vars, cvs, color=['red' if cv > 0.25 else 'orange' for cv in cvs])
    ax2.set_title('🚨 Strong Technical Batch Effects\n(CV > 10% = Problem)', fontweight='bold', color='red')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.axhline(y=0.1, color='black', linestyle='--', alpha=0.7, label='10% threshold')
    ax2.tick_params(axis='x', rotation=45)
    for bar, cv in zip(bars, cvs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. NK3 proportion variability (top right)
    ax3 = fig.add_subplot(gs[0, 2:])
    nk3_props = []
    donor_names = []
    for donor in adata.obs['donor'].value_counts().head(10).index:
        donor_data = adata.obs[adata.obs['donor'] == donor]
        nk3_prop = (donor_data['ident'] == 'NK3').mean()
        nk3_props.append(nk3_prop)
        donor_names.append(donor.replace('_', '\n'))
    
    bars = ax3.bar(range(len(donor_names)), nk3_props, 
                   color=['red' if prop > 0.4 else 'orange' if prop > 0.2 else 'green' 
                          for prop in nk3_props])
    ax3.set_title('🚨 NK Subtype Bias Across Donors\n(NK3: 12% to 58%!)', fontweight='bold', color='red')
    ax3.set_ylabel('NK3 Proportion')
    ax3.set_xticks(range(len(donor_names)))
    ax3.set_xticklabels(donor_names, rotation=45, ha='right', fontsize=8)
    
    # 4. CMV status bias (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    crosstab_cmv = pd.crosstab(adata.obs['CMV_Status'], adata.obs['ident'], normalize='index')
    crosstab_cmv.plot(kind='bar', stacked=True, ax=ax4, width=0.8, legend=False)
    ax4.set_title('🧬 CMV Status Bias\n(Biological Confounder)', fontweight='bold', color='orange')
    ax4.set_xlabel('CMV Status')
    ax4.set_ylabel('Proportion')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Dataset bias heatmap (middle)
    ax5 = fig.add_subplot(gs[1, 1:3])
    crosstab_dataset = pd.crosstab(adata.obs['Dataset'], adata.obs['ident'], normalize='index')
    sns.heatmap(crosstab_dataset, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax5, cbar_kws={'shrink': 0.5})
    ax5.set_title('🚨 NK Subtype Distribution by Dataset\n(HIGHLY BIASED)', fontweight='bold', color='red')
    ax5.set_xlabel('NK Subtype')
    ax5.set_ylabel('Dataset')
    
    # 6. Recommendations (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    recommendations = """
CRITICAL FINDINGS & RECOMMENDATIONS:

🚨 SEVERE ISSUES REQUIRING IMMEDIATE CORRECTION:
   • Dataset imbalance: 16.6:1 ratio (Dataset4 dominates with 67% of cells)
   • Strong technical batch effects: nCount_RNA CV = 33%, nFeature_RNA CV = 23%
   • NK subtype distribution bias: NK3 ranges from 12% to 58% across donors
   • Moderate donor imbalance: 7.9:1 ratio

🔧 RECOMMENDED BATCH CORRECTION STRATEGY:
   1. PRIMARY: Correct for Dataset effects (highest priority)
   2. SECONDARY: Correct for Donor effects  
   3. CONSIDER: Whether to preserve or correct CMV effects (biological decision)
   
💡 SUGGESTED METHODS:
   • Harmony: Excellent for multiple batch variables (donor + dataset)
   • scvi-tools: Advanced deep learning approach with GPU acceleration
   • Seurat CCA/RPCA: If using Seurat workflow
   
⚠️  CRITICAL: Validate that NK subtype biology is preserved after correction!
"""
    ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    plt.savefig(OUTPUT_DIR / 'BATCH_EFFECTS_COMPREHENSIVE_SUMMARY.png', 
                dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR / 'BATCH_EFFECTS_COMPREHENSIVE_SUMMARY.png'}")
    plt.close()

def main():
    """Main function for batch effect visualization."""
    print("=" * 80)
    print("NK CELL DATASET - BATCH EFFECT VISUALIZATION")
    print("=" * 80)
    print("Creating visualizations of identified batch effects:")
    print("• Severe dataset imbalance (16.6:1)")
    print("• Strong technical batch effects (CV > 25%)")
    print("• NK subtype distribution bias")
    print("• Donor contribution patterns")
    print()
    
    # Load data
    adata = load_data_with_cmv()
    
    # Create visualizations
    plot_dataset_imbalance(adata)
    plot_technical_batch_effects(adata)
    plot_nk_subtype_batch_bias(adata)
    plot_donor_contribution_analysis(adata)
    create_batch_effect_summary_plot(adata)
    
    print("\n" + "=" * 50)
    print("BATCH EFFECT VISUALIZATION COMPLETE")
    print("=" * 50)
    print(f"📁 All plots saved to: {OUTPUT_DIR}")
    print("🔍 Review the comprehensive summary plot for key findings")
    print("🚨 CRITICAL: This dataset requires batch correction before analysis!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Batch effect visualization completed!")
        print("📊 Critical batch effects clearly identified")
        print("🔧 Proceed with batch correction using recommended methods")
    else:
        print("\n❌ Batch effect visualization failed!")