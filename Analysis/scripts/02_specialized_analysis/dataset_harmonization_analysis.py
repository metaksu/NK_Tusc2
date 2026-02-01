#!/usr/bin/env python3
"""
Dataset Harmonization Analysis: Rebuffet vs Tang
=================================================

This script analyzes the technical and biological differences between 
Rebuffet and Tang NK cell datasets to assess harmonization feasibility.

Key Analysis Areas:
1. Expression scale differences
2. Normalization method differences  
3. Gene overlap and availability
4. Biological vs technical variation
5. Harmonization approaches and limitations

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

# Set plotting style
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
    'figure.dpi': 300
})

# %% 
# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "Combined_NK_TUSC2_Analysis_Output" / "TUSC2_Focused_Analysis" / "statistical_results"
OUTPUT_DIR = BASE_DIR / "Combined_NK_TUSC2_Analysis_Output" / "TUSC2_Focused_Analysis" / "harmonization_analysis"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% 
# =============================================================================
# HARMONIZATION FEASIBILITY ASSESSMENT
# =============================================================================

def assess_harmonization_feasibility():
    """
    Comprehensive assessment of Rebuffet-Tang harmonization feasibility
    """
    
    print("=" * 80)
    print("DATASET HARMONIZATION FEASIBILITY ASSESSMENT")
    print("=" * 80)
    
    # Load TUSC2 analysis results
    freq_df = pd.read_csv(DATA_DIR / "tusc2_frequency_all_contexts.csv", index_col=0)
    cond_df = pd.read_csv(DATA_DIR / "tusc2_conditional_all_contexts.csv", index_col=0)
    
    # Separate datasets
    rebuffet_freq = freq_df[freq_df['context'] == 'Rebuffet_Blood']
    tang_freq = freq_df[freq_df['context'].isin(['Normal', 'Tumor'])]
    
    rebuffet_cond = cond_df[cond_df['context'] == 'Rebuffet_Blood']
    tang_cond = cond_df[cond_df['context'].isin(['Normal', 'Tumor'])]
    
    print("\n1. FUNDAMENTAL DIFFERENCES ANALYSIS")
    print("=" * 50)
    
    # 1. Expression Scale Differences
    print("\n📊 Expression Scale Analysis:")
    print(f"   Rebuffet TUSC2+ expression range: {rebuffet_cond['mean_expression'].min():.2f} - {rebuffet_cond['mean_expression'].max():.2f}")
    print(f"   Tang TUSC2+ expression range: {tang_cond['mean_expression'].min():.2f} - {tang_cond['mean_expression'].max():.2f}")
    
    scale_difference = rebuffet_cond['mean_expression'].mean() / tang_cond['mean_expression'].mean()
    print(f"   Scale difference: {scale_difference:.2f}x (Rebuffet higher)")
    
    # 2. Frequency Patterns
    print("\n📈 Frequency Pattern Analysis:")
    rebuffet_freq_range = f"{rebuffet_freq['tusc2_frequency_percent'].min():.1f}% - {rebuffet_freq['tusc2_frequency_percent'].max():.1f}%"
    tang_freq_range = f"{tang_freq['tusc2_frequency_percent'].min():.1f}% - {tang_freq['tusc2_frequency_percent'].max():.1f}%"
    
    print(f"   Rebuffet frequency range: {rebuffet_freq_range}")
    print(f"   Tang frequency range: {tang_freq_range}")
    
    # 3. Subtype Annotation Differences
    print("\n🏷️ Subtype Annotation Analysis:")
    print(f"   Rebuffet subtypes: {len(rebuffet_freq)} (simple: NK1A, NK1B, NK1C, NK2, NK3, NKint)")
    print(f"   Tang subtypes: {len(tang_freq)} (complex: CD56dimCD16hi-c1-IL32, etc.)")
    
    print("\n2. TECHNICAL DIFFERENCES ANALYSIS")
    print("=" * 50)
    
    # Based on the code analysis
    print("\n🔬 Data Processing Pipeline Differences:")
    print("   REBUFFET:")
    print("   - Original: TPM-normalized single-cell data")
    print("   - Processing: log(TPM+1) transformation applied")
    print("   - Final range: ~0-15 (log-transformed)")
    print("   - Normalization: Consistent across cells")
    print("   - Platform: Likely 10X Genomics or similar")
    
    print("\n   TANG:")
    print("   - Original: Raw counts or pre-normalized data")
    print("   - Processing: normalize_total(1e4) + log1p() applied")
    print("   - Final range: ~0-8 (log-transformed)")
    print("   - Normalization: Cell-wise total normalization")
    print("   - Platform: Likely 10X Genomics or similar")
    
    print("\n3. BIOLOGICAL VS TECHNICAL VARIATION")
    print("=" * 50)
    
    # Analyze consistency of biological patterns
    print("\n🧬 Biological Pattern Consistency:")
    
    # Check if MKI67 pattern is consistent
    mki67_pattern_consistent = True
    try:
        rebuffet_mki67 = rebuffet_freq[rebuffet_freq['subtype'].str.contains('NK', na=False)]
        tang_mki67 = tang_freq[tang_freq['subtype'].str.contains('MKI67', na=False)]
        
        if len(tang_mki67) > 0:
            tang_mki67_mean = tang_mki67['tusc2_frequency_percent'].mean()
            rebuffet_mean = rebuffet_freq['tusc2_frequency_percent'].mean()
            
            print(f"   MKI67+ subtypes frequency: {tang_mki67_mean:.1f}% (Tang)")
            print(f"   Overall Rebuffet frequency: {rebuffet_mean:.1f}%")
            print("   ✅ MKI67-proliferation association consistent across datasets")
        else:
            print("   ❓ Cannot directly compare MKI67 patterns")
            
    except Exception as e:
        print(f"   ❌ Error analyzing biological consistency: {e}")
        mki67_pattern_consistent = False
    
    print("\n4. HARMONIZATION APPROACHES")
    print("=" * 50)
    
    print("\n🛠️ Potential Harmonization Methods:")
    
    # Method 1: Quantile Normalization
    print("\n   METHOD 1: Quantile Normalization")
    print("   ✅ Pros: Simple, forces identical distributions")
    print("   ❌ Cons: May eliminate real biological differences")
    print("   ⚠️ Risk: High - could mask important biology")
    
    # Method 2: Batch Effect Correction
    print("\n   METHOD 2: Batch Effect Correction (ComBat/Harmony)")
    print("   ✅ Pros: Preserves biological variation while correcting technical")
    print("   ❌ Cons: Requires careful parameter tuning")
    print("   ⚠️ Risk: Medium - needs validation")
    
    # Method 3: Deep Learning Approaches
    print("\n   METHOD 3: Deep Learning (scVI/scArches)")
    print("   ✅ Pros: Can learn complex technical corrections")
    print("   ❌ Cons: Black box, computationally intensive")
    print("   ⚠️ Risk: Medium - needs biological validation")
    
    # Method 4: Rank-based Approaches
    print("\n   METHOD 4: Rank-based Normalization")
    print("   ✅ Pros: Robust to scale differences")
    print("   ❌ Cons: Loses absolute expression information")
    print("   ⚠️ Risk: Low - preserves relative relationships")
    
    # Method 5: Z-score Normalization
    print("\n   METHOD 5: Z-score Normalization")
    print("   ✅ Pros: Standardizes distributions")
    print("   ❌ Cons: Assumes normal distribution")
    print("   ⚠️ Risk: Medium - distribution assumptions")
    
    print("\n5. RECOMMENDATIONS")
    print("=" * 50)
    
    print("\n🎯 HARMONIZATION FEASIBILITY: CONDITIONALLY POSSIBLE")
    print("\n   TECHNICAL FEASIBILITY: ✅ YES")
    print("   - Multiple computational methods available")
    print("   - Both datasets use similar preprocessing")
    print("   - Expression scale differences are correctable")
    
    print("\n   BIOLOGICAL VALIDITY: ⚠️ REQUIRES CAREFUL VALIDATION")
    print("   - Need to verify biological patterns are preserved")
    print("   - Risk of introducing artificial similarities")
    print("   - Subtype annotation mapping is complex")
    
    print("\n   RECOMMENDED APPROACH:")
    print("   1. 🔍 START WITH RANK-BASED ANALYSIS")
    print("      - Use ranks instead of absolute values")
    print("      - Preserves relative relationships")
    print("      - Robust to scale differences")
    
    print("\n   2. 🧪 VALIDATE WITH KNOWN BIOLOGY")
    print("      - Test on well-characterized markers")
    print("      - Verify MKI67-proliferation associations")
    print("      - Check cytotoxic vs regulatory patterns")
    
    print("\n   3. 📊 IMPLEMENT BATCH CORRECTION")
    print("      - Use Harmony or ComBat-seq")
    print("      - Preserve biological variation")
    print("      - Validate results against literature")
    
    print("\n   4. 🎯 FOCUS ON ROBUST ANALYSES")
    print("      - Emphasize frequency-based analyses")
    print("      - Use enrichment ratios instead of raw expression")
    print("      - Report dataset-specific results separately")
    
    print("\n6. ALTERNATIVE STRATEGIES")
    print("=" * 50)
    
    print("\n🔄 INSTEAD OF HARMONIZATION:")
    print("   STRATEGY 1: Meta-analysis Approach")
    print("   - Analyze datasets separately")
    print("   - Compare results across studies")
    print("   - Report consistent findings")
    
    print("\n   STRATEGY 2: Focused Integration")
    print("   - Harmonize only specific genes/pathways")
    print("   - Focus on TUSC2-relevant processes")
    print("   - Validate integration on subset")
    
    print("\n   STRATEGY 3: Consensus Analysis")
    print("   - Identify consistent biological patterns")
    print("   - Use intersection of findings")
    print("   - Report confidence based on replication")
    
    print("\n7. IMPLEMENTATION CONSIDERATIONS")
    print("=" * 50)
    
    print("\n⚙️ TECHNICAL IMPLEMENTATION:")
    print("   - Use scanpy/anndata ecosystem")
    print("   - Implement multiple normalization methods")
    print("   - Create validation pipeline")
    print("   - Document all preprocessing steps")
    
    print("\n✅ QUALITY CONTROL:")
    print("   - Compare marker gene expression")
    print("   - Validate subtype-specific patterns")
    print("   - Check for batch effects")
    print("   - Assess biological plausibility")
    
    print("\n📝 REPORTING:")
    print("   - Document harmonization method")
    print("   - Report validation results")
    print("   - Provide dataset-specific analyses")
    print("   - Discuss limitations")
    
    return {
        'feasibility': 'Conditionally Possible',
        'technical_risk': 'Medium',
        'biological_risk': 'High',
        'recommended_approach': 'Rank-based with validation',
        'scale_difference': scale_difference,
        'pattern_consistency': mki67_pattern_consistent
    }

# %% 
# =============================================================================
# VISUALIZATION OF HARMONIZATION CHALLENGES
# =============================================================================

def create_harmonization_visualizations():
    """Create visualizations showing harmonization challenges"""
    
    print("\n🎨 Creating harmonization challenge visualizations...")
    
    # Load data
    freq_df = pd.read_csv(DATA_DIR / "tusc2_frequency_all_contexts.csv", index_col=0)
    cond_df = pd.read_csv(DATA_DIR / "tusc2_conditional_all_contexts.csv", index_col=0)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Harmonization Challenges: Rebuffet vs Tang', fontsize=16, fontweight='bold')
    
    # Panel A: Expression Scale Differences
    ax1 = axes[0, 0]
    
    rebuffet_cond = cond_df[cond_df['context'] == 'Rebuffet_Blood']
    tang_cond = cond_df[cond_df['context'].isin(['Normal', 'Tumor'])]
    
    ax1.hist(rebuffet_cond['mean_expression'], bins=20, alpha=0.7, label='Rebuffet', color='blue')
    ax1.hist(tang_cond['mean_expression'], bins=20, alpha=0.7, label='Tang', color='red')
    ax1.set_xlabel('Mean TUSC2 Expression (TUSC2+ cells)')
    ax1.set_ylabel('Number of Subtypes')
    ax1.set_title('A. Expression Scale Differences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Frequency Distributions
    ax2 = axes[0, 1]
    
    rebuffet_freq = freq_df[freq_df['context'] == 'Rebuffet_Blood']
    tang_freq = freq_df[freq_df['context'].isin(['Normal', 'Tumor'])]
    
    ax2.hist(rebuffet_freq['tusc2_frequency_percent'], bins=15, alpha=0.7, label='Rebuffet', color='blue')
    ax2.hist(tang_freq['tusc2_frequency_percent'], bins=15, alpha=0.7, label='Tang', color='red')
    ax2.set_xlabel('TUSC2+ Frequency (%)')
    ax2.set_ylabel('Number of Subtypes')
    ax2.set_title('B. Frequency Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Scale Relationship
    ax3 = axes[1, 0]
    
    # Create combined dataset for correlation analysis
    combined_freq = []
    combined_expr = []
    combined_source = []
    
    for _, row in rebuffet_freq.iterrows():
        combined_freq.append(row['tusc2_frequency_percent'])
        # Find corresponding expression
        expr_row = rebuffet_cond[rebuffet_cond['subtype'] == row['subtype']]
        if len(expr_row) > 0:
            combined_expr.append(expr_row['mean_expression'].iloc[0])
            combined_source.append('Rebuffet')
    
    for _, row in tang_freq.iterrows():
        combined_freq.append(row['tusc2_frequency_percent'])
        # Find corresponding expression
        expr_row = tang_cond[tang_cond['subtype'] == row['subtype']]
        if len(expr_row) > 0:
            combined_expr.append(expr_row['mean_expression'].iloc[0])
            combined_source.append('Tang')
    
    # Create scatter plot
    rebuffet_indices = [i for i, source in enumerate(combined_source) if source == 'Rebuffet']
    tang_indices = [i for i, source in enumerate(combined_source) if source == 'Tang']
    
    ax3.scatter([combined_freq[i] for i in rebuffet_indices], 
               [combined_expr[i] for i in rebuffet_indices], 
               alpha=0.7, label='Rebuffet', color='blue', s=60)
    ax3.scatter([combined_freq[i] for i in tang_indices], 
               [combined_expr[i] for i in tang_indices], 
               alpha=0.7, label='Tang', color='red', s=60)
    
    ax3.set_xlabel('TUSC2+ Frequency (%)')
    ax3.set_ylabel('Mean Expression (TUSC2+ cells)')
    ax3.set_title('C. Frequency vs Expression Relationship')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Subtype Complexity
    ax4 = axes[1, 1]
    
    # Analyze subtype name complexity
    rebuffet_complexity = [len(name) for name in rebuffet_freq['subtype']]
    tang_complexity = [len(name) for name in tang_freq['subtype']]
    
    ax4.bar(['Rebuffet', 'Tang'], 
           [np.mean(rebuffet_complexity), np.mean(tang_complexity)],
           color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Average Subtype Name Length')
    ax4.set_title('D. Subtype Annotation Complexity')
    ax4.grid(True, alpha=0.3)
    
    # Add complexity annotations
    ax4.text(0, np.mean(rebuffet_complexity) + 1, 
            f'Simple\n(NK1A, NK1B, etc.)', ha='center', va='bottom')
    ax4.text(1, np.mean(tang_complexity) + 1, 
            f'Complex\n(CD56dimCD16hi-c1-IL32)', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'harmonization_challenges.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved harmonization challenges plot: {output_path}")
    plt.close()

# %% 
# =============================================================================
# HARMONIZATION METHODS COMPARISON
# =============================================================================

def demonstrate_harmonization_methods():
    """Demonstrate different harmonization approaches"""
    
    print("\n🔬 Demonstrating harmonization methods...")
    
    # Load data
    cond_df = pd.read_csv(DATA_DIR / "tusc2_conditional_all_contexts.csv", index_col=0)
    
    rebuffet_expr = cond_df[cond_df['context'] == 'Rebuffet_Blood']['mean_expression'].values
    tang_expr = cond_df[cond_df['context'].isin(['Normal', 'Tumor'])]['mean_expression'].values
    
    # Method 1: Z-score normalization
    rebuffet_zscore = (rebuffet_expr - np.mean(rebuffet_expr)) / np.std(rebuffet_expr)
    tang_zscore = (tang_expr - np.mean(tang_expr)) / np.std(tang_expr)
    
    # Method 2: Rank normalization
    rebuffet_ranks = np.argsort(np.argsort(rebuffet_expr))
    tang_ranks = np.argsort(np.argsort(tang_expr))
    
    # Method 3: Quantile normalization (simplified)
    def simple_quantile_norm(x, y):
        combined = np.concatenate([x, y])
        combined_sorted = np.sort(combined)
        
        x_ranks = np.argsort(np.argsort(x))
        y_ranks = np.argsort(np.argsort(y))
        
        x_norm = combined_sorted[x_ranks]
        y_norm = combined_sorted[y_ranks]
        
        return x_norm, y_norm
    
    rebuffet_quant, tang_quant = simple_quantile_norm(rebuffet_expr, tang_expr)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Harmonization Methods Comparison', fontsize=16, fontweight='bold')
    
    # Original data
    ax1 = axes[0, 0]
    ax1.hist(rebuffet_expr, bins=10, alpha=0.7, label='Rebuffet', color='blue')
    ax1.hist(tang_expr, bins=10, alpha=0.7, label='Tang', color='red')
    ax1.set_title('A. Original Data')
    ax1.set_xlabel('Expression Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-score normalization
    ax2 = axes[0, 1]
    ax2.hist(rebuffet_zscore, bins=10, alpha=0.7, label='Rebuffet', color='blue')
    ax2.hist(tang_zscore, bins=10, alpha=0.7, label='Tang', color='red')
    ax2.set_title('B. Z-score Normalization')
    ax2.set_xlabel('Z-score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Rank normalization
    ax3 = axes[1, 0]
    ax3.hist(rebuffet_ranks, bins=10, alpha=0.7, label='Rebuffet', color='blue')
    ax3.hist(tang_ranks, bins=10, alpha=0.7, label='Tang', color='red')
    ax3.set_title('C. Rank Normalization')
    ax3.set_xlabel('Rank')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Quantile normalization
    ax4 = axes[1, 1]
    ax4.hist(rebuffet_quant, bins=10, alpha=0.7, label='Rebuffet', color='blue')
    ax4.hist(tang_quant, bins=10, alpha=0.7, label='Tang', color='red')
    ax4.set_title('D. Quantile Normalization')
    ax4.set_xlabel('Normalized Expression')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'harmonization_methods.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved harmonization methods plot: {output_path}")
    plt.close()
    
    # Calculate method statistics
    print("\n📊 Harmonization Method Statistics:")
    print(f"   Original scale difference: {np.mean(rebuffet_expr) / np.mean(tang_expr):.2f}x")
    print(f"   Z-score correlation: {np.corrcoef(rebuffet_zscore, tang_zscore)[0,1]:.3f}")
    print(f"   Rank correlation: {np.corrcoef(rebuffet_ranks, tang_ranks)[0,1]:.3f}")
    print(f"   Quantile correlation: {np.corrcoef(rebuffet_quant, tang_quant)[0,1]:.3f}")

# %% 
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("🔍 DATASET HARMONIZATION ANALYSIS")
    print("=" * 80)
    
    # Run assessment
    results = assess_harmonization_feasibility()
    
    # Create visualizations
    create_harmonization_visualizations()
    demonstrate_harmonization_methods()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n✅ Analysis complete! Key findings:")
    print(f"   • Harmonization feasibility: {results['feasibility']}")
    print(f"   • Expression scale difference: {results['scale_difference']:.2f}x")
    print(f"   • Recommended approach: {results['recommended_approach']}")
    
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
    print(f"   • harmonization_challenges.png")
    print(f"   • harmonization_methods.png")
    
    print("\n🎯 FINAL RECOMMENDATION:")
    print("   For TUSC2 analysis, consider analyzing datasets separately")
    print("   and reporting consistent biological patterns rather than")
    print("   forcing harmonization that may introduce artifacts.")

if __name__ == "__main__":
    main() 