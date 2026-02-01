#!/usr/bin/env python3
"""
Modern Forest Plot Implementation for Hazard Ratio Visualization
Using the forestplot package - publication-ready HR plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# First install the package:
# pip install forestplot

import forestplot as fp

def create_publication_ready_forest_plot(hr_results):
    """
    Create a modern, publication-ready forest plot for hazard ratios
    
    Parameters:
    hr_results: DataFrame with columns ['Variable', 'HR', 'CI_Lower', 'CI_Upper', 'P_Value', 'Group']
    """
    
    # Example data structure for your TCGA results
    hr_data = pd.DataFrame({
        'variable': [
            'NK_Total (Quartile)',
            'Bright_NK', 
            'Cytotoxic_NK',
            'Exhausted_TaNK',
            'NK_CD56bright',
            'NK_CD56dim'
        ],
        'hr': [0.488, 1.096, 0.582, 0.284, 0.756, 0.623],
        'ci_lower': [0.247, 0.654, 0.345, 0.142, 0.445, 0.387],
        'ci_upper': [0.965, 1.838, 0.982, 0.568, 1.285, 1.003],
        'p_value': [0.045, 0.523, 0.105, 0.098, 0.334, 0.051],
        'group': ['NK Subtypes'] * 6,
        'scenario': ['Overall'] * 6,
        'n_events': [45, 38, 42, 35, 41, 39]
    })
    
    # Add significance indicators
    hr_data['significance'] = hr_data['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # Add formatted estimates
    hr_data['estimate_formatted'] = hr_data.apply(
        lambda row: f"{row['hr']:.3f} ({row['ci_lower']:.3f}-{row['ci_upper']:.3f}){row['significance']}", 
        axis=1
    )
    
    # Create the forest plot
    fig = fp.forestplot(
        hr_data,
        estimate="hr",
        ll="ci_lower", 
        hl="ci_upper",
        varlabel="variable",
        groupvar="group",
        xlabel="Hazard Ratio (95% CI)",
        ylabel="NK Cell Variables",
        annote=["n_events", "estimate_formatted"],
        annoteheaders=["Events", "HR (95% CI)"],
        rightannote=["p_value", "scenario"],
        right_annoteheaders=["P-Value", "Analysis"],
        **{
            "marker": "D",           # Diamond markers
            "markersize": 35,        # Larger markers
            "xlabel_size": 12,       # X-axis label size
            "color_alt_rows": True,  # Alternating row colors
            "figsize": (8, 6),      # Figure dimensions
            "flush": True,          # Left-align text
            "decimal_precision": 3,  # 3 decimal places
        }
    )
    
    # Add reference line at HR = 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    return fig

# Example usage for your TCGA data
def create_tcga_specific_forest_plot():
    """Create forest plot specifically for your TCGA NK analysis results"""
    
    # Use your actual results here
    tcga_results = pd.DataFrame({
        'variable': [
            'NK_Total (Quartile)',
            'NK_Total (Continuous)', 
            'Bright_NK',
            'Cytotoxic_NK',
            'Exhausted_TaNK',
            'NK_CD56bright',
            'NK_CD56dim'
        ],
        'hr': [0.34, 0.456, 1.096, 0.582, 0.284, 0.756, 0.623],
        'ci_lower': [0.124, 0.223, 0.654, 0.345, 0.142, 0.445, 0.387],
        'ci_upper': [0.932, 0.934, 1.838, 0.982, 0.568, 1.285, 1.003],
        'p_value': [0.030, 0.045, 0.523, 0.105, 0.098, 0.334, 0.051],
        'group': ['Primary Analysis', 'Primary Analysis'] + ['NK Subtypes'] * 5,
        'scenario': ['Overall'] * 7,
        'description': [
            'Quartile-based grouping',
            'Per 1% increase',
            'CD56bright NK cells',
            'Cytotoxic NK cells', 
            'Exhausted TaNK cells',
            'CD56bright subset',
            'CD56dim subset'
        ]
    })
    
    # Add clinical interpretation
    tcga_results['interpretation'] = tcga_results['hr'].apply(
        lambda hr: 'Protective' if hr < 1 else 'Risk Factor'
    )
    
    # Create the plot
    fig = fp.forestplot(
        tcga_results,
        estimate="hr",
        ll="ci_lower",
        hl="ci_upper", 
        varlabel="variable",
        groupvar="group",
        xlabel="Hazard Ratio (95% CI)",
        ylabel="NK Infiltration Variables",
        annote=["description"],
        annoteheaders=["Description"],
        rightannote=["p_value", "interpretation"],
        right_annoteheaders=["P-Value", "Effect"],
        color_alt_rows=True,
        **{
            "marker": "s",          # Square markers
            "markersize": 40,
            "xlabel_size": 14,
            "figsize": (10, 8),
            "xticks": [0.1, 0.5, 1.0, 2.0, 5.0],  # Custom x-axis ticks
        }
    )
    
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.title('NK Cell Infiltration and Overall Survival\nBRCA Cohort (n=506)', 
              fontsize=16, fontweight='bold', pad=20)
    
    return fig

if __name__ == "__main__":
    # Create the modern forest plot
    fig = create_tcga_specific_forest_plot()
    plt.savefig('modern_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.show()