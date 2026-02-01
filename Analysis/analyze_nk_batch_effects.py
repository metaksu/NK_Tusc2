#!/usr/bin/env python3
"""
NK Cell Dataset Detailed Batch Effect Analysis
==============================================

Based on the inspection results, this script performs detailed analysis of:
1. Donor effects (13 donors)
2. Dataset effects (4 datasets) 
3. CMV status effects
4. Technical covariates

Dataset: NK cells with batch variables in 'donor', 'Dataset', and CMV status
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
    sc.settings.verbosity = 1
    sc.settings.autoshow = False
    print(f"✓ Scanpy {sc.__version__} imported successfully")
except Exception as e:
    print(f"✗ CRITICAL ERROR: Scanpy import failed: {e}")
    exit(1)

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
NK_DATA_FILE = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
OUTPUT_DIR = Path("outputs/batch_effect_analysis")

def load_and_setup_data():
    """Load NK dataset and prepare for batch analysis."""
    print("=" * 70)
    print("LOADING NK DATASET FOR BATCH EFFECT ANALYSIS")
    print("=" * 70)
    
    adata = sc.read_h5ad(NK_DATA_FILE)
    print(f"✓ Loaded dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    return adata

def extract_cmv_status(adata):
    """Extract CMV status from donor names."""
    print("\n" + "=" * 50)
    print("EXTRACTING CMV STATUS FROM DONOR NAMES")
    print("=" * 50)
    
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
    
    cmv_counts = adata.obs['CMV_Status'].value_counts()
    print("CMV Status Distribution:")
    for status, count in cmv_counts.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"  {status}: {count:,} cells ({percentage:.1f}%)")
    
    return adata

def analyze_batch_variables(adata):
    """Comprehensive analysis of batch variables."""
    print("\n" + "=" * 50)
    print("BATCH VARIABLE ANALYSIS")
    print("=" * 50)
    
    batch_vars = ['donor', 'Dataset', 'CMV_Status']
    
    for var in batch_vars:
        print(f"\n📊 {var.upper()} ANALYSIS:")
        print("-" * 30)
        
        counts = adata.obs[var].value_counts()
        print(f"Number of {var}s: {len(counts)}")
        print("Distribution:")
        
        for value, count in counts.items():
            percentage = (count / len(adata.obs)) * 100
            print(f"  {value}: {count:,} cells ({percentage:.1f}%)")

def analyze_subtype_by_batch(adata):
    """Analyze NK subtype distribution across batch variables."""
    print("\n" + "=" * 50)
    print("NK SUBTYPE vs BATCH VARIABLE ANALYSIS")
    print("=" * 50)
    
    batch_vars = ['donor', 'Dataset', 'CMV_Status']
    
    for batch_var in batch_vars:
        print(f"\n🔬 NK SUBTYPES vs {batch_var.upper()}:")
        print("-" * 40)
        
        # Create crosstab
        crosstab = pd.crosstab(adata.obs['ident'], adata.obs[batch_var], normalize='columns')
        crosstab_counts = pd.crosstab(adata.obs['ident'], adata.obs[batch_var])
        
        print("Proportions (normalized by column):")
        print(crosstab.round(3))
        print("\nCounts:")
        print(crosstab_counts)
        
        # Save crosstab
        output_file = OUTPUT_DIR / f"crosstab_NK_subtypes_vs_{batch_var}.csv"
        crosstab_counts.to_csv(output_file)
        print(f"💾 Saved crosstab to: {output_file}")

def analyze_technical_covariates(adata):
    """Analyze technical covariates by batch variables."""
    print("\n" + "=" * 50)
    print("TECHNICAL COVARIATE ANALYSIS")
    print("=" * 50)
    
    technical_vars = ['nCount_RNA', 'nFeature_RNA', 'percent.mito', 'percent.ribo']
    batch_vars = ['donor', 'Dataset', 'CMV_Status']
    
    summary_data = []
    
    for tech_var in technical_vars:
        print(f"\n📈 {tech_var} by batch variables:")
        print("-" * 35)
        
        for batch_var in batch_vars:
            print(f"\n  {tech_var} vs {batch_var}:")
            
            # Calculate summary statistics by batch
            summary = adata.obs.groupby(batch_var)[tech_var].agg(['mean', 'std', 'median'])
            print(summary.round(2))
            
            # Calculate coefficient of variation across batches
            batch_means = summary['mean']
            cv = batch_means.std() / batch_means.mean()
            print(f"  Coefficient of variation across {batch_var}s: {cv:.3f}")
            
            summary_data.append({
                'technical_var': tech_var,
                'batch_var': batch_var,
                'cv_across_batches': cv
            })
    
    # Save technical summary
    summary_df = pd.DataFrame(summary_data)
    output_file = OUTPUT_DIR / "technical_covariate_batch_summary.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n💾 Technical covariate summary saved to: {output_file}")
    
    return summary_df

def identify_strong_batch_effects(adata, summary_df):
    """Identify which batch variables show strongest effects."""
    print("\n" + "=" * 50)
    print("BATCH EFFECT STRENGTH ASSESSMENT")
    print("=" * 50)
    
    # Identify high coefficient of variation (CV > 0.1 = 10%)
    high_cv = summary_df[summary_df['cv_across_batches'] > 0.1]
    
    if len(high_cv) > 0:
        print("🚨 STRONG BATCH EFFECTS DETECTED (CV > 10%):")
        print("-" * 45)
        for _, row in high_cv.iterrows():
            print(f"  {row['technical_var']} vs {row['batch_var']}: CV = {row['cv_across_batches']:.3f}")
    else:
        print("✅ No strong technical batch effects detected (all CV < 10%)")
    
    # Dataset imbalance analysis
    print(f"\n📊 DATASET IMBALANCE ANALYSIS:")
    print("-" * 35)
    dataset_counts = adata.obs['Dataset'].value_counts()
    largest = dataset_counts.max()
    smallest = dataset_counts.min()
    imbalance_ratio = largest / smallest
    
    print(f"Largest dataset: {largest:,} cells")
    print(f"Smallest dataset: {smallest:,} cells") 
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("🚨 SEVERE dataset imbalance detected!")
    elif imbalance_ratio > 3:
        print("⚠️ Moderate dataset imbalance detected")
    else:
        print("✅ Datasets reasonably balanced")
    
    # Donor imbalance analysis
    print(f"\n👥 DONOR IMBALANCE ANALYSIS:")
    print("-" * 30)
    donor_counts = adata.obs['donor'].value_counts()
    largest_donor = donor_counts.max()
    smallest_donor = donor_counts.min()
    donor_imbalance = largest_donor / smallest_donor
    
    print(f"Largest donor contribution: {largest_donor:,} cells")
    print(f"Smallest donor contribution: {smallest_donor:,} cells")
    print(f"Donor imbalance ratio: {donor_imbalance:.1f}:1")
    
    if donor_imbalance > 10:
        print("🚨 SEVERE donor imbalance detected!")
    elif donor_imbalance > 3:
        print("⚠️ Moderate donor imbalance detected")
    else:
        print("✅ Donor contributions reasonably balanced")

def generate_batch_effect_recommendations(adata):
    """Generate specific recommendations for batch effect correction."""
    print("\n" + "=" * 50)
    print("BATCH EFFECT CORRECTION RECOMMENDATIONS")
    print("=" * 50)
    
    print("Based on the analysis, here are recommendations:")
    print("\n1. 🎯 PRIMARY BATCH VARIABLES TO CORRECT:")
    print("   • donor (13 levels) - Patient/individual effects")
    print("   • Dataset (4 levels) - Study/batch effects")
    
    print("\n2. 🧬 BIOLOGICAL VARIABLES TO CONSIDER:")
    print("   • CMV_Status - May be biological, not technical")
    print("   • Consider if CMV should be corrected or preserved")
    
    print("\n3. 🔧 RECOMMENDED CORRECTION METHODS:")
    print("   • Harmony: Good for donor + dataset correction")
    print("   • scvi-tools: Advanced deep learning approach") 
    print("   • Seurat CCA/RPCA: If using Seurat workflow")
    print("   • ComBat: For simpler linear correction")
    
    print("\n4. ⚠️ IMPORTANT CONSIDERATIONS:")
    print("   • Preserve NK subtype biology during correction")
    print("   • Decide whether to correct CMV effects (biological)")
    print("   • Consider donor as random effect vs fixed effect")
    print("   • Validate that correction doesn't remove biology")
    
    print("\n5. 📋 NEXT STEPS:")
    print("   • Visualize data with PCA/UMAP colored by batch variables")
    print("   • Apply batch correction method")
    print("   • Validate NK subtype markers are preserved")
    print("   • Check that known biology is maintained")

def save_batch_metadata_summary(adata):
    """Save comprehensive batch metadata for further analysis."""
    print("\n" + "=" * 50)
    print("SAVING BATCH METADATA SUMMARY")
    print("=" * 50)
    
    # Create summary of cell metadata
    batch_metadata = adata.obs[['ident', 'donor', 'Dataset', 'CMV_Status', 
                               'nCount_RNA', 'nFeature_RNA', 'percent.mito', 'percent.ribo']].copy()
    
    # Add some derived statistics
    batch_metadata['log_nCount_RNA'] = np.log10(batch_metadata['nCount_RNA'])
    batch_metadata['log_nFeature_RNA'] = np.log10(batch_metadata['nFeature_RNA'])
    
    output_file = OUTPUT_DIR / "batch_metadata_for_analysis.csv"
    batch_metadata.to_csv(output_file)
    print(f"💾 Batch metadata saved to: {output_file}")
    print(f"   Contains {len(batch_metadata)} cells with batch information")
    
    return batch_metadata

def main():
    """Main function for detailed batch effect analysis."""
    print("=" * 80)
    print("NK CELL DATASET - DETAILED BATCH EFFECT ANALYSIS")
    print("=" * 80)
    print("Analyzing identified batch variables:")
    print("• Donor effects (13 donors)")
    print("• Dataset effects (4 datasets)")
    print("• CMV status effects")
    print("• Technical covariates")
    print()
    
    # Load data
    adata = load_and_setup_data()
    
    # Extract CMV status
    adata = extract_cmv_status(adata)
    
    # Analyze batch variables
    analyze_batch_variables(adata)
    
    # Analyze NK subtypes vs batch variables
    analyze_subtype_by_batch(adata)
    
    # Analyze technical covariates
    summary_df = analyze_technical_covariates(adata)
    
    # Assess batch effect strength
    identify_strong_batch_effects(adata, summary_df)
    
    # Generate recommendations
    generate_batch_effect_recommendations(adata)
    
    # Save metadata for further analysis
    batch_metadata = save_batch_metadata_summary(adata)
    
    print("\n" + "=" * 50)
    print("BATCH EFFECT ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("📊 Ready for batch effect visualization and correction")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Detailed batch effect analysis completed!")
        print("🔍 Check the outputs directory for detailed results")
        print("📈 Proceed with visualization and batch correction")
    else:
        print("\n❌ Batch effect analysis failed!")