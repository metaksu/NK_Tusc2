#!/usr/bin/env python3
"""
Publication-Ready Forest Plot for TCGA NK Survival Analysis
Uses actual results from BRCA_NK_Survival_Results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import forestplot as fp
import os

def create_nk_forest_plot():
    """
    Create publication-ready forest plot using actual NK survival results
    """
    
    # Load the actual results
    results_file = "TCGAdata/Simple_Analysis_Output/BRCA_NK_Survival_Results.csv"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the NK survival analysis first.")
        return
    
    # Read the results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} results from {results_file}")
    print(f"Columns: {list(df.columns)}")
    
    # Prepare data for forestplot
    forest_data = df.copy()
    
    # Rename columns to match forestplot expectations
    forest_data = forest_data.rename(columns={
        'Variable': 'variable',
        'HR': 'hr', 
        'HR_CI_Lower': 'ci_lower',
        'HR_CI_Upper': 'ci_upper',
        'P_Value': 'p_value',
        'Scenario': 'scenario',
        'Events': 'n_events'
    })
    
    # Add significance indicators
    forest_data['significance'] = forest_data['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # Format estimates with significance
    forest_data['estimate_formatted'] = forest_data.apply(
        lambda row: f"{row['hr']:.3f} ({row['ci_lower']:.3f}-{row['ci_upper']:.3f}){row['significance']}", 
        axis=1
    )
    
    # Add groups for better organization
    forest_data['group'] = forest_data['variable'].apply(
        lambda x: 'Individual NK Subtypes' if x in ['Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK'] else 'Combined NK Total'
    )
    
    # Create display labels
    forest_data['display_variable'] = forest_data.apply(
        lambda row: f"{row['variable']} ({row['scenario']})", axis=1
    )
    
    # Sort for better presentation
    forest_data = forest_data.sort_values(['group', 'scenario', 'variable'])
    
    print("\nForest plot data preview:")
    print(forest_data[['display_variable', 'hr', 'ci_lower', 'ci_upper', 'p_value']].head())
    
    # Create the forest plot
    fig = fp.forestplot(
        forest_data,
        estimate="hr",
        ll="ci_lower", 
        hl="ci_upper",
        varlabel="display_variable",
        groupvar="group",
        xlabel="Hazard Ratio (95% CI)",
        ylabel="NK Cell Variables by Analysis Group",
        annote=["n_events", "estimate_formatted"],
        annoteheaders=["Events", "HR (95% CI)"],
        rightannote=["p_value"],
        right_annoteheaders=["P-Value"],
        **{
            "marker": "D",              # Diamond markers
            "markersize": 40,           # Larger markers  
            "xlabel_size": 14,          # X-axis label size
            "color_alt_rows": True,     # Alternating row colors
            "figsize": (12, 8),         # Larger figure
            "flush": True,              # Left-align text
            "decimal_precision": 3,     # 3 decimal places
            "sort": False,              # Keep our custom sort
        }
    )
    
    # Add reference line at HR = 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add title and annotations
    plt.title('TCGA BRCA NK Cell Survival Analysis\n10-Year Follow-up', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add protective/harmful regions
    plt.text(0.5, -0.5, 'Protective\n(Better Survival)', 
             ha='center', va='top', fontsize=10, style='italic',
             transform=plt.gca().transData)
    plt.text(1.5, -0.5, 'Harmful\n(Worse Survival)', 
             ha='center', va='top', fontsize=10, style='italic',
             transform=plt.gca().transData)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = "TCGAdata/Simple_Analysis_Output/BRCA_NK_Publication_Forest_Plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Publication-ready forest plot saved: {output_file}")
    
    # Show summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total analyses: {len(forest_data)}")
    sig_results = forest_data[forest_data['p_value'] < 0.05]
    print(f"   Significant results (p<0.05): {len(sig_results)}")
    
    if len(sig_results) > 0:
        print(f"\n🎯 SIGNIFICANT FINDINGS:")
        for _, row in sig_results.iterrows():
            direction = "Protective" if row['hr'] < 1 else "Harmful"
            print(f"   {row['display_variable']}: HR={row['hr']:.3f} [{row['ci_lower']:.3f}-{row['ci_upper']:.3f}], p={row['p_value']:.3f} ({direction})")
    
    # Display the plot
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("🎯 Creating Publication-Ready NK Forest Plot...")
    print("=" * 60)
    
    # Create the forest plot
    fig = create_nk_forest_plot()
    
    print("\n" + "=" * 60)
    print("✅ Forest plot creation complete!")
    print("\nTip: The plot shows:")
    print("  • Diamond markers for hazard ratios")
    print("  • Confidence intervals as horizontal lines") 
    print("  • Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("  • Red dashed line at HR=1.0 (no effect)")
    print("  • Left side: Protective (better survival)")
    print("  • Right side: Harmful (worse survival)")