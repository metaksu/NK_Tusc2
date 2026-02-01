#!/usr/bin/env python3
"""
Comprehensive CIBERSORTx Results Analysis
Evaluate quality, robustness, and reliability of deconvolution results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_cibersortx_results(results_file):
    """
    Comprehensive analysis of CIBERSORTx deconvolution results.
    
    Parameters:
    -----------
    results_file : str
        Path to CIBERSORTx results CSV file
    """
    
    print("🔬 CIBERSORTx Results Quality Assessment")
    print("=" * 60)
    
    # Load the results
    print("\n📊 Loading CIBERSORTx results...")
    try:
        df = pd.read_csv(results_file)
        print(f"✅ Successfully loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Identify column types
    mixture_col = 'Mixture'
    quality_cols = ['P-value', 'Correlation', 'RMSE', 'Absolute score (sig.score)']
    
    # Find cell type columns (exclude mixture and quality columns)
    cell_type_cols = [col for col in df.columns 
                     if col not in [mixture_col] + quality_cols]
    
    print(f"📈 Identified {len(cell_type_cols)} cell types")
    print(f"🎯 Quality metrics: {quality_cols}")
    
    # Basic data structure analysis
    print(f"\n🔍 Data Structure Analysis:")
    print(f"   Total samples: {len(df)}")
    print(f"   Cell types: {len(cell_type_cols)}")
    print(f"   Complete cases: {df.dropna().shape[0]}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Quality metrics analysis
    print(f"\n📊 Quality Metrics Summary:")
    for col in quality_cols:
        if col in df.columns:
            values = df[col].dropna()
            print(f"   {col}:")
            print(f"     Mean: {values.mean():.4f}")
            print(f"     Median: {values.median():.4f}")
            print(f"     Range: {values.min():.4f} - {values.max():.4f}")
            print(f"     Missing: {df[col].isnull().sum()}")
    
    # CIBERSORTx quality thresholds
    print(f"\n🎯 CIBERSORTx Quality Assessment:")
    
    # P-value analysis (< 0.05 generally considered significant)
    if 'P-value' in df.columns:
        p_significant = (df['P-value'] < 0.05).sum()
        p_total = df['P-value'].notna().sum()
        print(f"   Significant results (p < 0.05): {p_significant}/{p_total} ({p_significant/p_total*100:.1f}%)")
        
        # Very strict threshold
        p_very_sig = (df['P-value'] < 0.01).sum()
        print(f"   Very significant (p < 0.01): {p_very_sig}/{p_total} ({p_very_sig/p_total*100:.1f}%)")
    
    # Correlation analysis (> 0.7 generally considered good)
    if 'Correlation' in df.columns:
        corr_good = (df['Correlation'] > 0.7).sum()
        corr_excellent = (df['Correlation'] > 0.9).sum()
        corr_total = df['Correlation'].notna().sum()
        print(f"   Good correlation (> 0.7): {corr_good}/{corr_total} ({corr_good/corr_total*100:.1f}%)")
        print(f"   Excellent correlation (> 0.9): {corr_excellent}/{corr_total} ({corr_excellent/corr_total*100:.1f}%)")
    
    # RMSE analysis (lower is better)
    if 'RMSE' in df.columns:
        rmse_values = df['RMSE'].dropna()
        rmse_low = (rmse_values < 0.5).sum()
        print(f"   Low RMSE (< 0.5): {rmse_low}/{len(rmse_values)} ({rmse_low/len(rmse_values)*100:.1f}%)")
    
    # Cell type proportion analysis
    print(f"\n🧬 Cell Type Proportion Analysis:")
    cell_proportions = df[cell_type_cols].copy()
    
    # Check if proportions sum to ~1 (indicating proper normalization)
    row_sums = cell_proportions.sum(axis=1)
    print(f"   Row sum statistics (should be ~1.0 for proportions):")
    print(f"     Mean: {row_sums.mean():.4f}")
    print(f"     Std: {row_sums.std():.4f}")
    print(f"     Range: {row_sums.min():.4f} - {row_sums.max():.4f}")
    
    # Check for negative values (shouldn't exist)
    negative_values = (cell_proportions < 0).sum().sum()
    print(f"   Negative values: {negative_values} (should be 0)")
    
    # Cell type summary statistics
    print(f"\n📈 Cell Type Summary Statistics:")
    cell_stats = cell_proportions.describe()
    print("   Top 5 most abundant cell types (by mean):")
    mean_abundances = cell_proportions.mean().sort_values(ascending=False)
    for i, (cell_type, abundance) in enumerate(mean_abundances.head().items()):
        print(f"     {i+1}. {cell_type}: {abundance:.4f} ({abundance*100:.2f}%)")
    
    # NK cell specific analysis
    nk_cols = [col for col in cell_type_cols if 'NK' in col.upper()]
    if nk_cols:
        print(f"\n🔥 NK Cell Subtype Analysis:")
        print(f"   NK subtypes identified: {nk_cols}")
        
        nk_data = df[nk_cols].copy()
        total_nk = nk_data.sum(axis=1)
        
        print(f"   Total NK cell statistics:")
        print(f"     Mean: {total_nk.mean():.4f} ({total_nk.mean()*100:.2f}%)")
        print(f"     Median: {total_nk.median():.4f}")
        print(f"     Range: {total_nk.min():.4f} - {total_nk.max():.4f}")
        
        for nk_type in nk_cols:
            nk_values = df[nk_type]
            detected_samples = (nk_values > 0).sum()
            print(f"   {nk_type}:")
            print(f"     Detected in: {detected_samples}/{len(df)} samples ({detected_samples/len(df)*100:.1f}%)")
            print(f"     Mean: {nk_values.mean():.4f}")
            print(f"     Max: {nk_values.max():.4f}")
    
    # Data quality assessment
    print(f"\n🔬 Overall Data Quality Assessment:")
    
    # Calculate quality score
    quality_score = 0
    max_score = 0
    
    # P-value component (25% of score)
    if 'P-value' in df.columns:
        p_score = (df['P-value'] < 0.05).sum() / len(df) * 25
        quality_score += p_score
        max_score += 25
    
    # Correlation component (25% of score)
    if 'Correlation' in df.columns:
        corr_score = (df['Correlation'] > 0.7).sum() / len(df) * 25
        quality_score += corr_score
        max_score += 25
    
    # Row sum component (25% of score) - penalize if sums deviate too much from 1
    sum_score = (np.abs(row_sums - 1) < 0.1).sum() / len(df) * 25
    quality_score += sum_score
    max_score += 25
    
    # Negative values component (25% of score) - penalize negative values
    no_negative_score = 25 if negative_values == 0 else 0
    quality_score += no_negative_score
    max_score += 25
    
    final_quality = quality_score / max_score * 100
    
    print(f"   Quality Score: {final_quality:.1f}/100")
    if final_quality >= 80:
        print("   ✅ EXCELLENT - Highly reliable results")
    elif final_quality >= 60:
        print("   ⚠️  GOOD - Generally reliable with some caution")
    elif final_quality >= 40:
        print("   ⚠️  FAIR - Use with caution, some quality issues")
    else:
        print("   ❌ POOR - Significant quality concerns")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if 'P-value' in df.columns:
        non_sig = (df['P-value'] >= 0.05).sum()
        if non_sig > 0:
            print(f"   • Consider filtering {non_sig} samples with p-value ≥ 0.05")
    
    if 'Correlation' in df.columns:
        low_corr = (df['Correlation'] <= 0.7).sum()
        if low_corr > 0:
            print(f"   • Consider filtering {low_corr} samples with correlation ≤ 0.7")
    
    if negative_values > 0:
        print(f"   • Investigate {negative_values} negative values (possible processing error)")
    
    if np.abs(row_sums.mean() - 1) > 0.05:
        print(f"   • Check normalization - row sums should be ~1.0")
    
    print(f"   • Total recommended samples for analysis: {len(df)} → {len(df) - max(non_sig if 'P-value' in df.columns else 0, low_corr if 'Correlation' in df.columns else 0)}")
    
    # Create summary dictionary
    summary = {
        'total_samples': len(df),
        'cell_types': len(cell_type_cols),
        'quality_score': final_quality,
        'significant_samples': p_significant if 'P-value' in df.columns else None,
        'good_correlation_samples': corr_good if 'Correlation' in df.columns else None,
        'nk_subtypes': nk_cols,
        'mean_total_nk': total_nk.mean() if nk_cols else None,
        'negative_values': negative_values,
        'row_sum_check': row_sums.mean()
    }
    
    return df, summary

if __name__ == "__main__":
    # Analyze the CIBERSORTx results
    results_file = "CIBERSORTx_Job1_Results.csv"
    
    if Path(results_file).exists():
        df, summary = analyze_cibersortx_results(results_file)
        
        print(f"\n📋 Analysis Complete!")
        print(f"   File: {results_file}")
        print(f"   Quality Score: {summary['quality_score']:.1f}/100")
        print(f"   Recommendation: {'✅ Ready for analysis' if summary['quality_score'] >= 60 else '⚠️ Use with caution'}")
        
    else:
        print(f"❌ File not found: {results_file}") 