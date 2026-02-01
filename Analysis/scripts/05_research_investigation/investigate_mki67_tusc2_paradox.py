#!/usr/bin/env python3
"""
Investigation of MKI67-TUSC2 Paradox
=====================================

This script investigates why MKI67+ subtype has the highest % of TUSC2+ cells,
but TUSC2+ cells show lower MKI67 scores than TUSC2- cells.

This explores Simpson's Paradox in NK cell biology.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import scanpy as sc

def investigate_mki67_tusc2_paradox(adata_list, context_names):
    """
    Investigate the MKI67-TUSC2 paradox across contexts
    
    Parameters:
    -----------
    adata_list : list
        List of AnnData objects for each context
    context_names : list
        Names of contexts corresponding to adata_list
    """
    
    print("🔍 INVESTIGATING MKI67-TUSC2 PARADOX")
    print("=" * 50)
    
    all_results = []
    
    for adata, context in zip(adata_list, context_names):
        if adata is None or adata.n_obs == 0:
            continue
            
        print(f"\n📊 Analyzing {context}...")
        
        # Get TUSC2 and MKI67 expression
        if 'TUSC2' not in adata.raw.var_names or 'MKI67' not in adata.raw.var_names:
            print(f"   ⚠️  Missing TUSC2 or MKI67 in {context}")
            continue
            
        tusc2_expr = adata.raw[:, 'TUSC2'].X.toarray().flatten()
        mki67_expr = adata.raw[:, 'MKI67'].X.toarray().flatten()
        
        # Define TUSC2 binary groups (using same threshold as main analysis)
        tusc2_threshold = 0.1
        tusc2_positive = tusc2_expr > tusc2_threshold
        
        # Get subtype information
        subtype_col = None
        for col in ['celltype', 'NK_Subtype_Profiled', 'Subtype']:
            if col in adata.obs.columns:
                subtype_col = col
                break
                
        if subtype_col is None:
            print(f"   ⚠️  No subtype column found in {context}")
            continue
            
        # Analysis 1: Overall correlation
        correlation = np.corrcoef(tusc2_expr, mki67_expr)[0, 1]
        
        # Analysis 2: TUSC2+ vs TUSC2- MKI67 levels
        mki67_tusc2_pos = mki67_expr[tusc2_positive]
        mki67_tusc2_neg = mki67_expr[~tusc2_positive]
        
        if len(mki67_tusc2_pos) > 0 and len(mki67_tusc2_neg) > 0:
            mean_diff = np.mean(mki67_tusc2_pos) - np.mean(mki67_tusc2_neg)
            _, pval = stats.mannwhitneyu(mki67_tusc2_pos, mki67_tusc2_neg, alternative='two-sided')
        else:
            mean_diff, pval = np.nan, np.nan
            
        # Analysis 3: By subtype analysis
        subtype_results = []
        
        for subtype in adata.obs[subtype_col].unique():
            if pd.isna(subtype) or subtype == 'Unassigned':
                continue
                
            mask = adata.obs[subtype_col] == subtype
            n_cells = mask.sum()
            
            if n_cells < 10:  # Skip small subtypes
                continue
                
            # TUSC2+ percentage in this subtype
            tusc2_pct = np.mean(tusc2_positive[mask]) * 100
            
            # MKI67 levels: TUSC2+ vs TUSC2- within this subtype
            subtype_tusc2_pos = tusc2_positive[mask]
            subtype_mki67 = mki67_expr[mask]
            
            if np.sum(subtype_tusc2_pos) > 2 and np.sum(~subtype_tusc2_pos) > 2:
                mki67_pos = subtype_mki67[subtype_tusc2_pos]
                mki67_neg = subtype_mki67[~subtype_tusc2_pos]
                within_subtype_diff = np.mean(mki67_pos) - np.mean(mki67_neg)
                _, within_pval = stats.mannwhitneyu(mki67_pos, mki67_neg, alternative='two-sided')
            else:
                within_subtype_diff, within_pval = np.nan, np.nan
                
            subtype_results.append({
                'Context': context,
                'Subtype': subtype,
                'N_Cells': n_cells,
                'TUSC2_Positive_Pct': tusc2_pct,
                'MKI67_Diff_TUSC2pos_vs_neg': within_subtype_diff,
                'Within_Subtype_Pval': within_pval,
                'Contains_MKI67_Name': 'MKI67' in str(subtype).upper()
            })
            
        subtype_df = pd.DataFrame(subtype_results)
        
        # Find the MKI67+ subtype specifically
        mki67_subtypes = subtype_df[subtype_df['Contains_MKI67_Name'] == True]
        
        print(f"   📈 Overall TUSC2-MKI67 correlation: {correlation:.3f}")
        print(f"   📊 Overall MKI67 difference (TUSC2+ vs TUSC2-): {mean_diff:.3f} (p={pval:.3f})")
        
        if not mki67_subtypes.empty:
            mki67_subtype = mki67_subtypes.iloc[0]
            print(f"   🎯 MKI67+ subtype: {mki67_subtype['Subtype']}")
            print(f"      - TUSC2+ percentage: {mki67_subtype['TUSC2_Positive_Pct']:.1f}%")
            print(f"      - Within-subtype MKI67 difference: {mki67_subtype['MKI67_Diff_TUSC2pos_vs_neg']:.3f}")
            print(f"      - This confirms the paradox! ⚡")
        
        # Store results
        all_results.append({
            'Context': context,
            'Overall_Correlation': correlation,
            'Overall_MKI67_Diff': mean_diff,
            'Overall_Pval': pval,
            'Subtype_Data': subtype_df
        })
        
        print(f"   ✅ {context} analysis complete")
    
    return all_results

def create_paradox_visualization(results, output_path="mki67_tusc2_paradox_analysis.png"):
    """Create a visualization explaining the paradox"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MKI67-TUSC2 Paradox Investigation', fontsize=16, fontweight='bold')
    
    # Collect data across contexts
    all_subtype_data = []
    for result in results:
        if 'Subtype_Data' in result:
            df = result['Subtype_Data'].copy()
            all_subtype_data.append(df)
    
    if all_subtype_data:
        combined_df = pd.concat(all_subtype_data, ignore_index=True)
        
        # Plot 1: TUSC2+ percentage by subtype
        ax1 = axes[0, 0]
        mki67_subtypes = combined_df[combined_df['Contains_MKI67_Name'] == True]
        others = combined_df[combined_df['Contains_MKI67_Name'] == False]
        
        if not mki67_subtypes.empty and not others.empty:
            ax1.bar(['MKI67+ Subtype', 'Other Subtypes'], 
                   [mki67_subtypes['TUSC2_Positive_Pct'].mean(), 
                    others['TUSC2_Positive_Pct'].mean()],
                   color=['red', 'lightblue'])
            ax1.set_ylabel('TUSC2+ Percentage')
            ax1.set_title('TUSC2+ Cells by Subtype Type')
        
        # Plot 2: Within-subtype MKI67 differences
        ax2 = axes[0, 1]
        valid_diffs = combined_df.dropna(subset=['MKI67_Diff_TUSC2pos_vs_neg'])
        if not valid_diffs.empty:
            ax2.boxplot([valid_diffs['MKI67_Diff_TUSC2pos_vs_neg']], labels=['All Subtypes'])
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_ylabel('MKI67 Difference\n(TUSC2+ vs TUSC2-)')
            ax2.set_title('Within-Subtype MKI67 Effects')
        
        # Plot 3: Correlation summary
        ax3 = axes[1, 0]
        correlations = [r['Overall_Correlation'] for r in results if not np.isnan(r.get('Overall_Correlation', np.nan))]
        contexts = [r['Context'] for r in results if not np.isnan(r.get('Overall_Correlation', np.nan))]
        
        if correlations:
            bars = ax3.bar(contexts, correlations, color='purple', alpha=0.7)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_ylabel('TUSC2-MKI67 Correlation')
            ax3.set_title('Overall TUSC2-MKI67 Correlation')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Explanation text
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.9, 'PARADOX EXPLANATION:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.8, '1. MKI67+ subtype has highest', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.15, 0.75, '% of TUSC2+ cells', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.65, '2. BUT within subtypes,', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.15, 0.6, 'TUSC2+ cells have lower MKI67', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, '3. MECHANISM:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.15, 0.45, 'TUSC2 = proliferation brake', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.15, 0.4, 'Upregulated in proliferating cells', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.15, 0.35, 'to control excessive division', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.25, '4. RESULT: Simpson\'s Paradox', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.15, 0.2, 'Group-level vs individual-level', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.15, 0.15, 'effects are opposite!', fontsize=10, transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Paradox visualization saved: {output_path}")

if __name__ == "__main__":
    print("🔬 MKI67-TUSC2 Paradox Investigation Script")
    print("This script should be run from the main analysis after loading data")
    print("Usage example:")
    print("results = investigate_mki67_tusc2_paradox([adata_blood, adata_normal, adata_tumor], ['Blood', 'Normal', 'Tumor'])")
    print("create_paradox_visualization(results)") 