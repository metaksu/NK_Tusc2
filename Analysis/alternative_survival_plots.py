#!/usr/bin/env python3
"""
Alternative Modern Survival Analysis Visualizations
Multiple high-quality options for HR and survival visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set modern styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_modern_seaborn_forest_plot(hr_data):
    """
    Create a publication-quality forest plot using seaborn
    More customizable than forestplot package
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    hr_data = hr_data.copy()
    hr_data['y_pos'] = range(len(hr_data))
    hr_data['is_significant'] = hr_data['p_value'] < 0.05
    
    # Color scheme
    colors = ['#d62728' if sig else '#1f77b4' for sig in hr_data['is_significant']]
    
    # Plot confidence intervals
    for i, row in hr_data.iterrows():
        ax.plot([row['ci_lower'], row['ci_upper']], 
                [row['y_pos'], row['y_pos']], 
                color=colors[i], linewidth=3, alpha=0.7)
    
    # Plot point estimates
    scatter = ax.scatter(hr_data['hr'], hr_data['y_pos'], 
                        c=colors, s=150, marker='D', 
                        edgecolors='black', linewidth=1, zorder=5)
    
    # Add reference line
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.8, linewidth=2)
    
    # Customize axes
    ax.set_yticks(hr_data['y_pos'])
    ax.set_yticklabels(hr_data['variable'])
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Variables', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add annotations
    for i, row in hr_data.iterrows():
        # HR and CI text
        text = f"{row['hr']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f})"
        if row['p_value'] < 0.05:
            text += "*"
        
        ax.text(max(hr_data['ci_upper']) * 1.1, row['y_pos'], text, 
                va='center', fontsize=10, fontweight='bold' if row['is_significant'] else 'normal')
    
    # Title
    ax.set_title('NK Cell Infiltration and Overall Survival\nModern Forest Plot Visualization', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='#d62728', s=150, marker='D', label='Significant (p<0.05)'),
        plt.scatter([], [], c='#1f77b4', s=150, marker='D', label='Non-significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def create_interactive_plotly_forest_plot(hr_data):
    """
    Create an interactive forest plot using Plotly
    Great for presentations and web display
    """
    fig = go.Figure()
    
    # Add confidence intervals
    for i, row in hr_data.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['ci_lower'], row['ci_upper']], 
            y=[i, i],
            mode='lines',
            line=dict(color='rgba(0,100,80,0.7)', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add point estimates
    colors = ['red' if p < 0.05 else 'blue' for p in hr_data['p_value']]
    
    fig.add_trace(go.Scatter(
        x=hr_data['hr'],
        y=list(range(len(hr_data))),
        mode='markers',
        marker=dict(
            color=colors,
            size=15,
            symbol='diamond',
            line=dict(color='black', width=1)
        ),
        text=[f"{var}<br>HR: {hr:.3f}<br>95% CI: ({ci_l:.3f}-{ci_u:.3f})<br>p-value: {p:.3f}" 
              for var, hr, ci_l, ci_u, p in zip(hr_data['variable'], hr_data['hr'], 
                                                hr_data['ci_lower'], hr_data['ci_upper'], 
                                                hr_data['p_value'])],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    # Add reference line
    fig.add_shape(
        type="line",
        x0=1, x1=1,
        y0=-0.5, y1=len(hr_data)-0.5,
        line=dict(color="gray", width=2, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Interactive NK Cell Survival Analysis<br><sub>Hover over points for details</sub>",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Hazard Ratio (95% CI)",
            type="log",
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Variables",
            tickmode='array',
            tickvals=list(range(len(hr_data))),
            ticktext=hr_data['variable'],
            showgrid=True,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        hovermode='closest'
    )
    
    return fig

def create_maraca_plot_style(survival_data):
    """
    Create a Maraca-style plot for hierarchical composite endpoints
    Based on recent medical literature for survival visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 2]})
    
    # Left panel: Time-to-event outcomes
    survival_times = np.random.exponential(2, 1000)  # Example data
    censoring = np.random.binomial(1, 0.7, 1000)
    
    # Kaplan-Meier style curves
    time_points = np.linspace(0, max(survival_times), 100)
    
    # High-risk group
    high_risk_surv = np.exp(-0.8 * time_points)
    # Low-risk group  
    low_risk_surv = np.exp(-0.4 * time_points)
    
    ax1.plot(time_points, high_risk_surv, 'r-', linewidth=3, label='High NK Infiltration')
    ax1.plot(time_points, low_risk_surv, 'b-', linewidth=3, label='Low NK Infiltration')
    ax1.fill_between(time_points, high_risk_surv, alpha=0.3, color='red')
    ax1.fill_between(time_points, low_risk_surv, alpha=0.3, color='blue')
    
    ax1.set_xlabel('Time (Months)', fontsize=12)
    ax1.set_ylabel('Overall Survival Probability', fontsize=12)
    ax1.set_title('Kaplan-Meier Survival Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Win odds/ratio visualization
    outcomes = ['Death', 'Progression', 'Stable', 'Response']
    high_risk_pct = [25, 35, 30, 10]
    low_risk_pct = [10, 25, 40, 25]
    
    x_pos = np.arange(len(outcomes))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, high_risk_pct, width, 
                    label='High NK Infiltration', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, low_risk_pct, width,
                    label='Low NK Infiltration', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Outcomes', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Outcome Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(outcomes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Modern Survival Analysis Visualization\n(Maraca-Style Plot)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_shap_style_survival_plot(feature_importance_data):
    """
    Create SHAP-style visualization for survival analysis
    Shows feature importance in survival prediction
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by importance
    feature_importance_data = feature_importance_data.sort_values('importance', ascending=True)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(feature_importance_data)), 
                   feature_importance_data['importance'],
                   color=['red' if x < 0 else 'blue' for x in feature_importance_data['importance']])
    
    # Customize
    ax.set_yticks(range(len(feature_importance_data)))
    ax.set_yticklabels(feature_importance_data['feature'])
    ax.set_xlabel('SHAP Value (Impact on Survival)', fontsize=12)
    ax.set_title('Feature Importance in Survival Prediction\n(SHAP-Style Visualization)', 
                 fontsize=14, fontweight='bold')
    
    # Add reference line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, feature_importance_data['importance'])):
        ax.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', 
                va='center', ha='left' if value > 0 else 'right', fontweight='bold')
    
    # Add legend
    ax.text(0.02, 0.98, 'Blue: Protective\nRed: Risk Factor', 
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Example data for your TCGA analysis
    hr_data = pd.DataFrame({
        'variable': ['NK_Total', 'Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK', 'NK_CD56bright'],
        'hr': [0.488, 1.096, 0.582, 0.284, 0.756],
        'ci_lower': [0.247, 0.654, 0.345, 0.142, 0.445],
        'ci_upper': [0.965, 1.838, 0.982, 0.568, 1.285],
        'p_value': [0.045, 0.523, 0.105, 0.098, 0.334]
    })
    
    # Feature importance data
    feature_importance = pd.DataFrame({
        'feature': ['Age', 'NK_Total', 'TUSC2_Expression', 'Tumor_Stage', 'BMI'],
        'importance': [0.23, -0.45, 0.12, 0.34, 0.08]
    })
    
    # Create all visualizations
    print("Creating modern seaborn forest plot...")
    fig1 = create_modern_seaborn_forest_plot(hr_data)
    fig1.savefig('seaborn_forest_plot.png', dpi=300, bbox_inches='tight')
    
    print("Creating interactive plotly forest plot...")
    fig2 = create_interactive_plotly_forest_plot(hr_data)
    fig2.write_html('interactive_forest_plot.html')
    
    print("Creating maraca-style plot...")
    fig3 = create_maraca_plot_style(None)  # Uses synthetic data
    fig3.savefig('maraca_style_plot.png', dpi=300, bbox_inches='tight')
    
    print("Creating SHAP-style survival plot...")
    fig4 = create_shap_style_survival_plot(feature_importance)
    fig4.savefig('shap_style_survival.png', dpi=300, bbox_inches='tight')
    
    plt.show()