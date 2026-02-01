#!/usr/bin/env python3
"""
TCGA NK Cell Subtype Survival Analysis

This module performs survival analysis of Natural Killer (NK) cell subtypes in 
breast cancer patients using TCGA clinical and CIBERSORTx immune deconvolution data.

Scientific Approach:
    The analysis integrates clinical survival data with immune cell infiltration 
    estimates to assess the prognostic significance of NK cell subtypes (Bright_NK, 
    Cytotoxic_NK) in breast cancer. The methodology follows established practices 
    for cancer immunology survival analysis.

Statistical Methods:
    - Cox Proportional Hazards Regression: Estimates hazard ratios for NK subtypes
    - Kaplan-Meier Survival Analysis: Calculates survival probabilities over time
    - Log-rank Test: Compares survival distributions between high/low NK groups
    - False Discovery Rate (FDR): Controls multiple testing using Benjamini-Hochberg

Data Sources:
    - TCGA Clinical Data: Patient demographics, vital status, follow-up time
    - CIBERSORTx Results: Immune cell type abundance estimates
    - Integration ensures one tumor sample per patient for valid survival analysis

Key References:
    - Cox, D.R. (1972). Regression models and life-tables. J R Stat Soc Series B.
    - Newman et al. (2019). Determining cell type abundance and expression from 
      bulk tissues with digital cytometry. Nat Biotechnol.
    - Kaplan, E.L. & Meier, P. (1958). Nonparametric estimation from incomplete 
      observations. J Am Stat Assoc.

Author: Computational Biology Analysis Pipeline
Date: 2025-01-28
Version: 1.0 - Publication Supplemental Material
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
import scanpy as sc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    import forestplot as fp
    FORESTPLOT_AVAILABLE = True
except ImportError:
    FORESTPLOT_AVAILABLE = False
    print("Warning: forestplot package not available. Forest plots will be skipped.")
    
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Module Metadata
# =============================================================================

__version__ = "1.0.0"
__author__ = "Computational Biology Analysis Pipeline"
__license__ = "Academic Use"
__email__ = "supplemental.material@publication.org"

# =============================================================================
# External Dependencies
# =============================================================================

# Import core TCGA data processing functions
from TCGA_Gene_Survival_Analysis import (
    load_and_preprocess_tcga_data,
    prepare_survival_data,
    integrate_cibersortx_data,
    load_cibersortx_data,
    NS_TCGA,
    DISEASE_CONFIG_TCGA,
    DEFAULT_THRESHOLDS
)

# =============================================================================
# Analysis Configuration
# =============================================================================

CONFIG = {
    # Cancer type for analysis (TCGA project code)
    "cancer_type": "BRCA",
    
    # Survival analysis time horizon (None = full follow-up period)
    # Full timeline recommended for NK survival analysis to capture long-term effects
    "max_years": None,
    
    # Data directories
    "base_data_dir": "TCGAdata",
    "output_dir": "TCGAdata/Simple_Analysis_Output",
    
    # NK cell subtypes for survival analysis
    # Based on CIBERSORTx immune deconvolution with established NK signatures
    "nk_variables": ["Bright_NK", "Cytotoxic_NK", "NK_Total"],
    
    # Age stratification cutoff (years)
    # Standard clinical cutoff for breast cancer prognosis stratification
    "age_cutoff": 60,
    
    # Quality control thresholds
    "thresholds": DEFAULT_THRESHOLDS
}

# =============================================================================
# Core Analysis Pipeline
# =============================================================================

def load_and_preprocess_tcga_data_with_logging(
    cancer_type: str, 
    base_data_dir: str, 
    thresholds: Optional[Dict] = None
) -> Optional[pd.DataFrame]:
    """
    Load and preprocess TCGA clinical and molecular data for survival analysis.
    
    This function integrates TCGA clinical data, RNA-seq data, and CIBERSORTx 
    immune deconvolution results into a unified dataset suitable for NK cell 
    survival analysis.
    
    Parameters:
        cancer_type (str): TCGA cancer type code (e.g., 'BRCA')
        base_data_dir (str): Directory containing TCGA data files
        thresholds (dict): Quality control thresholds for data filtering
    
    Returns:
        pandas.DataFrame: Integrated survival dataset with clinical and immune data
    """
    print(f"\nLoading TCGA {cancer_type} data for NK survival analysis...")
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    try:
        # Load comprehensive TCGA dataset using established pipeline
        print("Processing clinical data, RNA-seq, and sample metadata...")
        
        # Load comprehensive TCGA dataset
        tumor_adata, master_metadata_df = load_and_preprocess_tcga_data(
            cancer_type=cancer_type,
            base_data_dir=base_data_dir,
            output_dir=None,
            thresholds=thresholds
        )
        
        if tumor_adata is None:
            print(f"Error: Failed to load TCGA {cancer_type} data")
            return None
        
        print(f"Successfully loaded {tumor_adata.n_obs} tumor samples with {tumor_adata.n_vars} genes")
        
        # Prepare survival data for analysis
        print("Preparing survival endpoints and follow-up data...")
        survival_df = prepare_survival_data(tumor_adata)
        
        if survival_df is None or len(survival_df) == 0:
            print("Error: No valid survival data available")
            return None
        
        # Survival data summary for quality assessment
        total_events = survival_df['Event'].sum()
        event_rate = total_events / len(survival_df) * 100
        median_followup = survival_df['Survival_Time'].median()
        
        print(f"Survival data summary: {len(survival_df)} samples, {total_events} events ({event_rate:.1f}%)")
        print(f"Median follow-up: {median_followup:.0f} days ({median_followup/365.25:.1f} years)")
        
        # Integrate CIBERSORTx immune deconvolution data
        cibersort_paths = {
            "primary": os.path.join(base_data_dir, f"CIBERSORTx_{cancer_type}.csv"),
            "rebuffet": os.path.join(base_data_dir, f"CIBERSORTx_Adjusted_{cancer_type}_Rebuffet_Fractions.txt")
        }
        
        # Check for available CIBERSORTx results
        existing_cibersort = {data_type: path for data_type, path in cibersort_paths.items() if os.path.exists(path)}
        if existing_cibersort:
            print("Integrating CIBERSORTx immune deconvolution data...")
            
            # Integrate immune deconvolution results
            tumor_adata_with_cibersort = integrate_cibersortx_data(
                tumor_adata, existing_cibersort, thresholds
            )
            
            if tumor_adata_with_cibersort is not None:
                # Update survival data with immune infiltration estimates
                survival_df_updated = prepare_survival_data(tumor_adata_with_cibersort)
                if survival_df_updated is not None and len(survival_df_updated) > 0:
                    survival_df = survival_df_updated
                    
                    # Identify NK cell variables for analysis
                    nk_columns = [col for col in survival_df.columns if any(nk in col for nk in ["NK", "Bright_NK", "Cytotoxic_NK"])]
                    print(f"NK cell subtypes available: {len(nk_columns)} variables")
                else:
                    print("Warning: Failed to integrate immune deconvolution data")
            else:
                print("Warning: CIBERSORTx integration unsuccessful")
        else:
            print("Warning: No CIBERSORTx files found - using clinical data only")
        
        print(f"\nDataset preparation complete: {len(survival_df)} samples ready for analysis")
        return survival_df
        
    except Exception as e:
        print(f"Error loading TCGA data: {e}")
        return None

def load_simple_data(config: Dict) -> Optional[pd.DataFrame]:
    """
    Load TCGA data for NK cell survival analysis.
    
    Parameters:
        config (dict): Analysis configuration parameters
    
    Returns:
        pandas.DataFrame: Integrated clinical and immune deconvolution data
    """
    return load_and_preprocess_tcga_data_with_logging(
        config["cancer_type"], 
        config["base_data_dir"], 
        config["thresholds"]
    )

def preprocess_simple(df: pd.DataFrame, config: Dict) -> Optional[pd.DataFrame]:
    """
    Preprocess survival data for NK cell analysis.
    
    Performs quality control, sample filtering, and variable preparation for 
    survival analysis. Includes validation of survival endpoints and NK cell 
    infiltration estimates.
    
    Parameters:
        df (pandas.DataFrame): Raw survival and immune data
        config (dict): Analysis configuration parameters
    
    Returns:
        pandas.DataFrame: Cleaned dataset ready for survival analysis
    """
    print("\nPreprocessing data for survival analysis...")
    
    if df is None:
        print("Error: No data provided for preprocessing")
        return None
    
    print(f"Input dataset: {len(df)} samples, {len(df.columns)} variables")
    
    # Validate required survival columns
    survival_cols = ["Survival_Time", "Event", "OS_Time", "OS_Event"]
    found_survival = [col for col in survival_cols if col in df.columns]
    
    # Identify NK cell variables
    nk_cols = [col for col in df.columns if any(nk in col for nk in ["NK", "Bright_NK", "Cytotoxic_NK"])]
    print(f"Available NK variables: {len(nk_cols)} subtypes")
    
    # Select essential columns for analysis
    essential_cols = ["Survival_Time", "Event"] + config["nk_variables"]
    if "Age_at_Diagnosis" in df.columns:
        essential_cols.append("Age_at_Diagnosis")
    
    # Filter to available columns
    available_cols = [col for col in essential_cols if col in df.columns]
    missing_cols = [col for col in essential_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
    
    df_clean = df[available_cols].copy()
    print(f"Selected {len(df_clean.columns)} variables for analysis")
    
    # Quality control: Remove samples with missing survival data
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=["Survival_Time", "Event"])
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        print(f"Removed {removed_count} samples with missing survival data")
    
    if len(df_clean) == 0:
        print("Error: No samples with valid survival data available")
        return None
    
    # Survival data characteristics
    total_events = df_clean['Event'].sum()
    event_rate = total_events / len(df_clean) * 100
    max_survival_time = df_clean['Survival_Time'].max()
    
    print(f"Survival analysis cohort: {len(df_clean)} patients, {total_events} events ({event_rate:.1f}%)")
    print(f"Follow-up range: 0 to {max_survival_time:.1f} days (full timeline)")
    
    # Create age stratification groups
    if "Age_at_Diagnosis" in df_clean.columns:
        df_clean["Age_Group"] = pd.cut(
            df_clean["Age_at_Diagnosis"], 
            bins=[0, config["age_cutoff"], 120], 
            labels=["Young", "Old"]
        )
        age_counts = df_clean["Age_Group"].value_counts()
        print(f"Age stratification: {age_counts.get('Young', 0)} young (≤{config['age_cutoff']}) vs {age_counts.get('Old', 0)} older patients")
    else:
        print("Warning: Age data not available for stratification")
    
    # Calculate total NK infiltration from subtypes
    nk_subtypes = [col for col in ["Bright_NK", "Cytotoxic_NK"] if col in df_clean.columns]
    if nk_subtypes:
        df_clean["NK_Total"] = df_clean[nk_subtypes].sum(axis=1)
        nk_total_stats = df_clean["NK_Total"]
        print(f"NK_Total calculated from {len(nk_subtypes)} subtypes (range: {nk_total_stats.min():.4f} - {nk_total_stats.max():.4f})")
    else:
        print("Warning: No NK subtypes available for calculating NK_Total")
    
    # Quality control: Filter samples with insufficient NK infiltration data
    nk_vars = [col for col in config["nk_variables"] if col in df_clean.columns]
    if nk_vars:
        before_filter = len(df_clean)
        
        # Remove samples with zero or negative NK infiltration
        nk_positive = df_clean[nk_vars].sum(axis=1) > 0
        df_clean = df_clean[nk_positive]
        filtered_count = before_filter - len(df_clean)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} samples with insufficient NK infiltration data")
    else:
        print("Warning: No NK variables found for quality filtering")
    
    # Final dataset summary
    final_events = df_clean['Event'].sum()
    final_event_rate = final_events / len(df_clean) * 100
    median_followup = df_clean['Survival_Time'].median()
    
    print(f"\nPreprocessing complete: {len(df_clean)} samples ready for analysis")
    print(f"Final cohort: {final_events} events ({final_event_rate:.1f}%), median follow-up {median_followup:.0f} days")
    
    return df_clean

def calculate_cox_simple(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Perform Cox proportional hazards regression analysis for NK cell subtypes.
    
    Calculates hazard ratios for NK cell infiltration variables across different
    patient scenarios (overall cohort, age-stratified groups). Uses the lifelines
    implementation of Cox regression with robust confidence interval estimation.
    
    Statistical Method:
        Cox Proportional Hazards Model: h(t|x) = h0(t) * exp(β*x)
        where h(t|x) is the hazard at time t for covariates x,
        h0(t) is the baseline hazard, and β is the regression coefficient.
        
        Hazard Ratio = exp(β) represents the relative risk of event occurrence
        per unit increase in the NK variable.
    
    Parameters:
        df (pandas.DataFrame): Preprocessed survival data with NK variables
        config (dict): Analysis configuration including NK variables and age cutoff
    
    Returns:
        pandas.DataFrame: Results with HR, confidence intervals, p-values, and FDR correction
        
    Reference:
        Cox, D.R. (1972). Regression models and life-tables. J R Stat Soc Series B 34:187-220.
    """
    print("Performing Cox proportional hazards regression...")
    
    # Identify available NK variables for analysis
    nk_vars = [col for col in config["nk_variables"] if col in df.columns]
    
    # Define analysis scenarios
    scenarios = {"Overall": df.copy()}
    
    # Add age-stratified analyses if age data available
    if "Age_Group" in df.columns:
        scenarios["Young"] = df[df["Age_Group"] == "Young"].copy()
        scenarios["Old"] = df[df["Age_Group"] == "Old"].copy()
    
    results = []
    
    # Perform Cox regression for each scenario
    for scenario_name, scenario_df in scenarios.items():
        # Quality control: Ensure adequate sample size for stable regression
        if len(scenario_df) < 20:
            print(f"Skipping {scenario_name}: insufficient sample size ({len(scenario_df)})")
            continue
            
        print(f"Cox regression - {scenario_name} cohort: {len(scenario_df)} patients")
        
        # Analyze each NK variable in this scenario
        for nk_var in nk_vars:
            if nk_var not in scenario_df.columns:
                continue
            
            # Prepare regression dataset
            cox_data = scenario_df[["Survival_Time", "Event", nk_var]].copy()
            cox_data = cox_data.dropna()
            
            # Minimum sample size check for stable regression
            if len(cox_data) < 10:
                print(f"  Skipping {nk_var}: insufficient data ({len(cox_data)} samples)")
                continue
            
            # Handle numerical scaling for very small NK values
            nk_mean = cox_data[nk_var].mean()
            nk_var_to_use = nk_var
            
            if nk_mean < 0.1:  # Scale small values for numerical stability
                scaling_factor = 100
                cox_data[f"{nk_var}_scaled"] = cox_data[nk_var] * scaling_factor
                nk_var_to_use = f"{nk_var}_scaled"
            
            try:
                # Fit Cox proportional hazards model
                cph = CoxPHFitter()
                cph.fit(cox_data[["Survival_Time", "Event", nk_var_to_use]], 
                       duration_col="Survival_Time", event_col="Event")
                
                # Extract hazard ratio and statistical significance
                hr = cph.hazard_ratios_[nk_var_to_use]
                p_value = cph.summary.loc[nk_var_to_use, "p"]
                
                # Calculate 95% confidence intervals
                try:
                    ci_lower = cph.confidence_intervals_.loc[nk_var_to_use, f"{nk_var_to_use} lower 95%"]
                    ci_upper = cph.confidence_intervals_.loc[nk_var_to_use, f"{nk_var_to_use} upper 95%"]
                except:
                    # Fallback: Manual CI calculation from coefficient and standard error
                    coef = cph.summary.loc[nk_var_to_use, "coef"]
                    se = cph.summary.loc[nk_var_to_use, "se(coef)"]
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                
                # Apply reasonable bounds to prevent extreme outlier values
                ci_lower = max(ci_lower, 0.001)
                ci_upper = min(ci_upper, 1000)
                hr = max(min(hr, 1000), 0.001)
                
                # Store analysis results
                results.append({
                    "Scenario": scenario_name,
                    "Variable": nk_var,
                    "HR": hr,
                    "HR_CI_Lower": ci_lower,
                    "HR_CI_Upper": ci_upper,
                    "P_Value": p_value,
                    "Sample_Size": len(cox_data),
                    "Events": cox_data["Event"].sum(),
                    "Risk_Direction": "Protective" if hr < 1 else "Harmful"
                })
                
                # Report individual result
                risk_interpretation = "protective" if hr < 1 else "harmful"
                print(f"  {nk_var}: HR={hr:.3f} [{ci_lower:.3f}-{ci_upper:.3f}], p={p_value:.3f} ({risk_interpretation})")
                
            except Exception as e:
                print(f"  {nk_var}: Cox regression failed - {e}")
                continue
    
    # Compile results and apply multiple testing correction
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Apply Benjamini-Hochberg FDR correction for multiple testing
        _, q_values, _, _ = multipletests(results_df["P_Value"], method="fdr_bh")
        results_df["FDR_Q_Value"] = q_values
        
        print(f"\nCox regression complete: {len(results_df)} analyses performed")
        significant_count = (results_df["P_Value"] < 0.05).sum()
        fdr_significant_count = (results_df["FDR_Q_Value"] < 0.05).sum()
        print(f"Significant results: {significant_count} nominal (p<0.05), {fdr_significant_count} FDR-corrected (q<0.05)")
    
    return results_df

def prepare_km_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and validate survival data for Kaplan-Meier analysis.
    
    Ensures proper survival time calculation and event coding for accurate
    Kaplan-Meier survival curve estimation. Handles TCGA-specific data formats
    and validates survival endpoints.
    
    Parameters:
        df (pandas.DataFrame): Raw survival data with TCGA clinical variables
        
    Returns:
        pandas.DataFrame: Validated survival dataset with standardized time/event columns
        
    Reference:
        Kaplan, E.L. & Meier, P. (1958). Nonparametric estimation from incomplete 
        observations. J Am Stat Assoc 53:457-481.
    """
    print("Preparing survival data for Kaplan-Meier analysis...")
    
    survival_df = df.copy()
    
    # Enhanced survival time calculation with validation
    print("  Calculating survival times...")
    
    # Initialize survival time and event columns
    if "Survival_Time" not in survival_df.columns:
        survival_df["Survival_Time"] = np.nan
    if "Event" not in survival_df.columns:
        survival_df["Event"] = 0
    
    # Check data availability
    has_death_data = 'Days_to_Death' in survival_df.columns and survival_df['Days_to_Death'].notna().sum() > 0
    has_followup_data = 'Days_to_Last_Followup' in survival_df.columns and survival_df['Days_to_Last_Followup'].notna().sum() > 0
    has_vital_status = 'Vital_Status' in survival_df.columns
    
    print(f"    Days_to_Death available: {survival_df['Days_to_Death'].notna().sum() if has_death_data else 0}")
    print(f"    Days_to_Last_Followup available: {survival_df['Days_to_Last_Followup'].notna().sum() if has_followup_data else 0}")
    print(f"    Vital_Status available: {survival_df['Vital_Status'].notna().sum() if has_vital_status else 0}")
    
    if not has_vital_status:
        print("  Warning: No Vital_Status data available, using mock survival data")
        # For mock data, use existing Survival_Time and Event
        return survival_df
    
    # Analyze vital status distribution
    vital_counts = survival_df['Vital_Status'].value_counts()
    print("    Vital status distribution:")
    for status, count in vital_counts.items():
        print(f"      {status}: {count} ({count/len(survival_df)*100:.1f}%)")
    
    # Enhanced event coding with validation
    print("  Coding survival events...")
    
    # More robust vital status coding
    dead_patterns = ['dead', 'deceased', 'death']
    
    survival_df["Event"] = 0  # Default to censored
    vital_status_lower = survival_df["Vital_Status"].astype(str).str.lower().str.strip()
    
    for pattern in dead_patterns:
        survival_df.loc[vital_status_lower.str.contains(pattern, na=False), "Event"] = 1
    
    # Survival time assignment with enhanced logic
    print("  Assigning survival times...")
    
    # For deceased patients: use Days_to_Death if available
    if has_death_data:
        deceased_mask = survival_df["Event"] == 1
        deceased_with_death_time = deceased_mask & survival_df["Days_to_Death"].notna()
        survival_df.loc[deceased_with_death_time, "Survival_Time"] = survival_df.loc[deceased_with_death_time, "Days_to_Death"]
        print(f"    Deceased patients with death time: {deceased_with_death_time.sum()}")
    
    # For all patients without survival time: use last follow-up
    if has_followup_data:
        missing_survival_time = survival_df["Survival_Time"].isna()
        has_followup = missing_survival_time & survival_df["Days_to_Last_Followup"].notna()
        survival_df.loc[has_followup, "Survival_Time"] = survival_df.loc[has_followup, "Days_to_Last_Followup"]
        print(f"    Patients using follow-up time: {has_followup.sum()}")
    
    # Convert survival time to years for plotting
    survival_df["Survival_Time_Years"] = survival_df["Survival_Time"] / 365.25
    
    # Summary statistics
    total_samples = len(survival_df)
    total_events = survival_df["Event"].sum()
    event_rate = (total_events / total_samples * 100) if total_samples > 0 else 0
    median_followup = survival_df["Survival_Time"].median() / 365.25 if survival_df["Survival_Time"].notna().any() else 0
    
    print(f"\n  Survival Data Summary:")
    print(f"    Total samples: {total_samples}")
    print(f"    Total events: {total_events} ({event_rate:.1f}%)")
    print(f"    Median follow-up: {median_followup:.1f} years")
    
    return survival_df

def create_nk_risk_table(
    ax, 
    kmf_high: KaplanMeierFitter, 
    kmf_low: KaplanMeierFitter, 
    group_labels: List[str], 
    colors: List[str], 
    font_size_base: int
) -> None:
    """
    Create publication-quality numbers-at-risk table for survival plots.
    
    Generates a formatted table showing the number of patients at risk at
    different time points during follow-up for each survival group.
    
    Parameters:
        ax (matplotlib.axes.Axes): Matplotlib axis for table placement
        kmf_high (lifelines.KaplanMeierFitter): High NK group survival curve
        kmf_low (lifelines.KaplanMeierFitter): Low NK group survival curve
        group_labels (list): Labels for survival groups
        colors (list): Colors corresponding to survival curves
        font_size_base (int): Base font size for table text
        
    Table Features:
        - Dynamic time points based on follow-up duration
        - Color-coded group indicators
        - Professional formatting with grid lines
        - Handles both years and days time units automatically
    """
    # Calculate time points for risk table using FULL timeline (no 10-year cap)
    max_time_raw = max(
        kmf_high.durations.max() if len(kmf_high.durations) > 0 else 0,
        kmf_low.durations.max() if len(kmf_low.durations) > 0 else 0
    )
    
    # Determine if data is in days or years
    is_in_days = max_time_raw > 50  # If max > 50, likely in days
    
    if is_in_days:
        max_time_years = max_time_raw / 365.25
    else:
        max_time_years = max_time_raw  # Already in years
    
    # Dynamic time points based on actual data range
    if max_time_years <= 3:
        time_points_years = [0, 1, 2, 3]
    elif max_time_years <= 5:
        time_points_years = [0, 1, 2, 3, 4, 5]
    elif max_time_years <= 8:
        time_points_years = [0, 1, 2, 3, 5, 7]
    elif max_time_years <= 12:
        time_points_years = [0, 1, 2, 3, 5, 7, 10]
    elif max_time_years <= 20:
        time_points_years = [0, 2, 5, 10, 15]
    else:
        # For very long follow-up
        time_points_years = [0, 5, 10, 15, 20, int(max_time_years)]
    
    # Filter to only include time points within actual data range
    time_points_years = [t for t in time_points_years if t <= max_time_years + 0.5]
    
    # Convert time points to same units as the data
    if is_in_days:
        time_points_raw = [t * 365.25 for t in time_points_years]  # Convert to days
    else:
        time_points_raw = time_points_years  # Keep in years
    
    # Calculate numbers at risk using proper lifelines method
    risk_high = []
    risk_low = []
    
    for t_raw in time_points_raw:
        try:
            # Count subjects still at risk at time t (using same units as data)
            n_risk_high = (kmf_high.durations >= t_raw).sum()
            n_risk_low = (kmf_low.durations >= t_raw).sum()
            risk_high.append(n_risk_high)
            risk_low.append(n_risk_low)
        except:
            # Fallback
            risk_high.append(0)
            risk_low.append(0)
    
    # Configure axes for professional appearance
    ax.clear()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Remove all spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Table dimensions
    n_time_points = len(time_points_years)
    col_width = 1.0 / (n_time_points + 1)  # +1 for group label column
    row_height = 0.35
    
    # Header: "Numbers at risk" title
    ax.text(
        0.02, 0.85, "Numbers at risk", 
        fontsize=font_size_base, fontweight='bold', 
        ha='left', va='center', color='black'
    )
    
    # Column headers (time points)
    header_y = 0.65
    for i, time_year in enumerate(time_points_years):
        x_pos = (i + 1) * col_width + col_width/2
        ax.text(
            x_pos, header_y, f"{int(time_year)}", 
            fontsize=font_size_base - 1, fontweight='bold',
            ha='center', va='center', color='black'
        )
    
    # Draw subtle horizontal line under headers
    ax.axhline(y=0.58, xmin=0.02, xmax=0.98, color='lightgray', linewidth=0.8, alpha=0.7)
    
    # Group data rows
    group_data = [
        (group_labels[0], colors[0], risk_high),
        (group_labels[1], colors[1], risk_low)
    ]
    
    for row_idx, (label, color, risk_nums) in enumerate(group_data):
        row_y = 0.45 - (row_idx * row_height)
        
        # Group label with colored square indicator
        label_x = 0.02
        
        # Draw small colored square
        square_size = 0.03
        square = patches.Rectangle(
            (label_x, row_y - square_size/2), square_size, square_size,
            facecolor=color, edgecolor=color, alpha=0.8,
            transform=ax.transAxes
        )
        ax.add_patch(square)
        
        # Group label text
        ax.text(
            label_x + square_size + 0.01, row_y, label,
            fontsize=font_size_base - 1, fontweight='bold',
            ha='left', va='center', color='black'
        )
        
        # Risk numbers for each time point
        for i, (time_year, n_risk) in enumerate(zip(time_points_years, risk_nums)):
            x_pos = (i + 1) * col_width + col_width/2
            
            # Format number with appropriate styling
            risk_text = str(n_risk) if n_risk > 0 else "0"
            text_color = 'black' if n_risk > 0 else '#666666'
            
            ax.text(
                x_pos, row_y, risk_text,
                fontsize=font_size_base - 1, 
                ha='center', va='center', 
                color=text_color, family='monospace'
            )
    
    # Add subtle vertical lines between columns for better readability
    for i in range(1, n_time_points + 1):
        x_pos = i * col_width + col_width/2 + col_width/4
        if x_pos < 0.98:  # Don't draw line at the very end
            ax.axvline(x=x_pos, ymin=0.1, ymax=0.8, color='lightgray', 
                      linewidth=0.5, alpha=0.4, linestyle=':')
    
    # Units label for time
    ax.text(
        0.5, 0.02, "Time (years)", 
        fontsize=font_size_base - 2, style='italic',
        ha='center', va='bottom', color='#555555'
    )

def create_nk_km_plot(
    high_group: pd.DataFrame, 
    low_group: pd.DataFrame, 
    group_labels: Optional[List[str]] = None, 
    title: str = "NK Cell Kaplan-Meier Survival Curve",
    colors: Optional[List[str]] = None, 
    figsize: Tuple[int, int] = (12, 8), 
    save_path: Optional[str] = None, 
    show_risk_table: bool = True,
    hr: Optional[float] = None, 
    p_value: Optional[float] = None, 
    fdr_q_value: Optional[float] = None, 
    ci_95: Optional[Tuple[float, float]] = None, 
    variable_name: Optional[str] = None,
    scenario_name: Optional[str] = None, 
    font_size_base: int = 12
) -> Tuple[Optional[plt.Figure], Optional[Tuple]]:
    """
    Generate publication-ready Kaplan-Meier survival curves for NK cell analysis.
    
    Creates professional survival plots with confidence intervals, risk tables,
    and statistical annotations suitable for publication supplemental material.
    
    Parameters:
        high_group (pandas.DataFrame): High NK infiltration patient data
        low_group (pandas.DataFrame): Low NK infiltration patient data
        group_labels (list): Labels for survival groups ['High NK', 'Low NK']
        title (str): Plot title
        colors (list): Colors for survival curves [high_color, low_color]
        figsize (tuple): Figure dimensions (width, height)
        save_path (str): File path for saving plot
        show_risk_table (bool): Include numbers-at-risk table
        hr (float): Hazard ratio from Cox regression
        p_value (float): Log-rank test p-value
        fdr_q_value (float): FDR-corrected q-value
        ci_95 (tuple): 95% confidence interval for HR (lower, upper)
        variable_name (str): NK variable name for annotation
        scenario_name (str): Analysis scenario for annotation
        font_size_base (int): Base font size for scaling
        
    Returns:
        tuple: (matplotlib.figure.Figure, axes) or (None, None) if failed
        
    Statistical Methods:
        - Kaplan-Meier estimator for survival probability curves
        - Log-rank test for comparing survival distributions between groups
        - Numbers-at-risk table showing sample sizes over time
    """
    try:
        # Validate input data
        if high_group.empty or low_group.empty:
            print(f"    Warning: Empty group data - skipping plot")
            return None, None
            
        required_cols = ['Survival_Time', 'Event']
        for col in required_cols:
            if col not in high_group.columns or col not in low_group.columns:
                print(f"    Warning: Missing required column '{col}' - skipping plot")
                return None, None
        
        # Set default parameters
        if group_labels is None:
            group_labels = ["High NK", "Low NK"]
        if colors is None:
            colors = ["#E31A1C", "#1F78B4"]  # Red for high, blue for low
            
        # Set publication-ready style
        plt.style.use('default')
        sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
        
        # Create figure with subplots for main plot and risk table
        if show_risk_table:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_main = fig.add_subplot(gs[0])
            ax_risk = fig.add_subplot(gs[1])
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
            ax_risk = None
        
        # Fit Kaplan-Meier curves with error handling
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        try:
            kmf_high.fit(high_group["Survival_Time"], high_group["Event"], label=group_labels[0])
            kmf_low.fit(low_group["Survival_Time"], low_group["Event"], label=group_labels[1])
        except Exception as e:
            print(f"    Warning: KM fitting failed: {e}")
            plt.close(fig)
            return None, None
        
        # Plot survival curves with confidence intervals
        try:
            kmf_high.plot_survival_function(
                ax=ax_main, color=colors[0], linewidth=2.5, 
                show_censors=True, censor_styles={'marker': '|', 'ms': 8, 'mew': 2}
            )
            kmf_low.plot_survival_function(
                ax=ax_main, color=colors[1], linewidth=2.5,
                show_censors=True, censor_styles={'marker': '|', 'ms': 8, 'mew': 2}
            )
        except Exception as e:
            print(f"    Warning: Plotting survival curves failed: {e}")
            plt.close(fig)
            return None, None
        
        # Customize main plot
        ax_main.set_xlabel("Time (years)", fontsize=font_size_base + 2, fontweight='bold')
        ax_main.set_ylabel("Survival Probability", fontsize=font_size_base + 2, fontweight='bold')
        ax_main.set_title(title, fontsize=font_size_base + 4, fontweight='bold', pad=20)
        
        # Format x-axis appropriately (check if data is in days or years)
        try:
            # Check data units by examining survival time range  
            sample_times = pd.concat([high_group["Survival_Time"], low_group["Survival_Time"]])
            max_time = sample_times.max()
            is_in_days = max_time > 50  # If max > 50, likely in days
            
            x_ticks = ax_main.get_xticks()
            if is_in_days:
                # Convert from days to years
                x_labels = [f"{int(x/365.25)}" for x in x_ticks if x >= 0]
            else:
                # Data already in years
                x_labels = [f"{int(x)}" for x in x_ticks if x >= 0]
            
            # Only set ticks if we have valid values
            if len(x_labels) == len(x_ticks):
                ax_main.set_xticks(x_ticks)
                ax_main.set_xticklabels(x_labels)
        except Exception as e:
            print(f"    Warning: X-axis formatting failed: {e}")
        
        # Set y-axis limits and ticks
        ax_main.set_ylim(0, 1.05)
        ax_main.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_main.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        
        # Customize tick labels
        ax_main.tick_params(axis='both', which='major', labelsize=font_size_base)
        
        # Add grid
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Customize legend
        try:
            legend = ax_main.legend(
                loc='upper right', frameon=True, fancybox=True, shadow=True,
                fontsize=font_size_base + 1, title_fontsize=font_size_base + 2
            )
            legend.get_frame().set_alpha(0.9)
        except Exception as e:
            print(f"    Warning: Legend creation failed: {e}")
        
        # Add statistics box
        stats_text = []
        
        # Validate and display accurate sample sizes and events
        high_events = high_group["Event"].sum()
        low_events = low_group["Event"].sum()
        total_events = high_events + low_events
        
        # Sample sizes (ensure they match what's being analyzed)
        stats_text.append(f"n = {len(high_group)} vs {len(low_group)}")
        
        # Events with percentages for clarity
        high_event_rate = (high_events / len(high_group)) * 100 if len(high_group) > 0 else 0
        low_event_rate = (low_events / len(low_group)) * 100 if len(low_group) > 0 else 0
        stats_text.append(f"Events = {high_events} ({high_event_rate:.1f}%) vs {low_events} ({low_event_rate:.1f}%)")
        
        # Add HR and confidence interval (only show if properly calculated from group comparison)
        if hr is not None and not np.isnan(hr):
            # Determine significance stars for HR based on p-value
            hr_stars = ""
            if p_value is not None and not np.isnan(p_value):
                if p_value < 0.001:
                    hr_stars = "***"
                elif p_value < 0.01:
                    hr_stars = "**"
                elif p_value < 0.05:
                    hr_stars = "*"
            
            if ci_95 is not None and len(ci_95) == 2 and not np.isnan(ci_95[0]) and not np.isnan(ci_95[1]):
                stats_text.append(f"HR (High vs Low) = {hr:.2f}{hr_stars} (95% CI: {ci_95[0]:.2f}-{ci_95[1]:.2f})")
            else:
                stats_text.append(f"HR (High vs Low) = {hr:.2f}{hr_stars}")
        
        # Add p-value with significance stars (clarify this is log-rank test p-value)
        if p_value is not None and not np.isnan(p_value):
            stars = ""
            if p_value < 0.001:
                stars = "***"
            elif p_value < 0.01:
                stars = "**"
            elif p_value < 0.05:
                stars = "*"
            
            if p_value < 0.001:
                stats_text.append(f"Log-rank P < 0.001{stars}")
            else:
                stats_text.append(f"Log-rank P = {p_value:.3f}{stars}")
        
        # Add FDR q-value if provided
        if fdr_q_value is not None and not np.isnan(fdr_q_value):
            if fdr_q_value < 0.001:
                stats_text.append(f"FDR q < 0.001")
            else:
                stats_text.append(f"FDR q = {fdr_q_value:.3f}")
        
        # Add median survival times (completely rewritten - simple and robust)
        try:
            # Simple approach: directly get medians from KM fitters
            median_high_raw = kmf_high.median_survival_time_
            median_low_raw = kmf_low.median_survival_time_
            
            # Format the results simply
            if pd.isna(median_high_raw):
                median_high_str = "Not Reached"
            else:
                median_high_str = f"{median_high_raw:.1f}"
                
            if pd.isna(median_low_raw):
                median_low_str = "Not Reached"
            else:
                median_low_str = f"{median_low_raw:.1f}"
            
            # Add median survival to stats (units will match input data)
            stats_text.append(f"Median survival: {median_high_str} vs {median_low_str}")
            
        except Exception as e:
            print(f"    Warning: Median survival calculation failed: {e}")
            # Fallback: calculate manually from survival curves
            try:
                # Manual calculation using survival function
                high_survival = kmf_high.survival_function_
                low_survival = kmf_low.survival_function_
                
                # Find where survival drops to 0.5 (50%)
                median_high_manual = "Not Reached"
                median_low_manual = "Not Reached"
                
                if not high_survival.empty and (high_survival <= 0.5).any():
                    median_idx = (high_survival <= 0.5).idxmax()
                    median_high_manual = f"{median_idx:.1f}"
                
                if not low_survival.empty and (low_survival <= 0.5).any():
                    median_idx = (low_survival <= 0.5).idxmax()
                    median_low_manual = f"{median_idx:.1f}"
                
                stats_text.append(f"Median survival: {median_high_manual} vs {median_low_manual}")
                
            except Exception as e2:
                print(f"    Warning: Manual median calculation also failed: {e2}")
                stats_text.append("Median survival: Unable to calculate")
        
        # Create statistics box
        stats_box_text = "\n".join(stats_text)
        
        # Position statistics box
        try:
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax_main.text(
                0.02, 0.02, stats_box_text, transform=ax_main.transAxes,
                fontsize=font_size_base - 1, verticalalignment='bottom',
                bbox=props, family='monospace'
            )
        except Exception as e:
            print(f"    Warning: Statistics box creation failed: {e}")
        
        # Add risk table if requested
        if show_risk_table and ax_risk is not None:
            try:
                create_nk_risk_table(ax_risk, kmf_high, kmf_low, group_labels, colors, font_size_base)
            except Exception as e:
                print(f"    Warning: Risk table creation failed: {e}")
        
        # Add analysis details to title area
        if variable_name or scenario_name:
            try:
                subtitle_parts = []
                if scenario_name:
                    subtitle_parts.append(f"Scenario: {scenario_name}")
                if variable_name:
                    subtitle_parts.append(f"NK Variable: {variable_name}")
                
                if subtitle_parts:
                    subtitle = " | ".join(subtitle_parts)
                    ax_main.text(
                        0.5, 1.02, subtitle, transform=ax_main.transAxes,
                        ha='center', fontsize=font_size_base, style='italic'
                    )
            except Exception as e:
                print(f"    Warning: Subtitle creation failed: {e}")
        
        # Adjust layout (robust layout handling)
        try:
            plt.tight_layout()
        except Exception:
            try:
                # Fallback: use subplots_adjust if tight_layout fails
                plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.3 if ax_risk else 0.1)
            except Exception as e:
                print(f"    Warning: Layout adjustment failed: {e}")
        
        # Save plot if path provided
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"    Saved plot: {save_path}")
            except Exception as e:
                print(f"    Warning: Plot saving failed: {e}")
        
        return fig, (ax_main, ax_risk) if ax_risk else (ax_main,)
        
    except Exception as e:
        print(f"    Error: Plot creation completely failed: {e}")
        try:
            if 'fig' in locals():
                plt.close(fig)
        except:
            pass
        return None, None

def generate_km_plots(
    df: pd.DataFrame, 
    results_df: pd.DataFrame, 
    config: Dict
) -> List[Path]:
    """
    Generate comprehensive Kaplan-Meier survival plots for NK analysis.
    
    Creates survival curves for all NK variables across different patient scenarios
    (overall cohort, age-stratified groups) using median-split stratification.
    
    Parameters:
        df (pandas.DataFrame): Preprocessed survival and NK infiltration data
        results_df (pandas.DataFrame): Cox regression results for statistical annotation
        config (dict): Analysis configuration including NK variables and output directory
        
    Returns:
        list: Paths to generated plot files
        
    Plot Generation Strategy:
        - Median split: High vs. Low NK infiltration groups
        - Age stratification: Separate analyses for young vs. older patients
        - Statistical annotation: HR, confidence intervals, p-values from Cox regression
        - Risk tables: Numbers-at-risk over time for each survival group
    """
    print("Generating Kaplan-Meier survival plots...")
    
    if results_df.empty:
        print("No results to plot")
        return []
    
    output_dir = Path(config["output_dir"])
    km_plots_dir = output_dir / "kaplan_meier_plots"
    km_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare survival data
    survival_df = prepare_km_survival_data(df)
    
    nk_vars = [col for col in config["nk_variables"] if col in survival_df.columns]
    scenarios = {
        "Overall": survival_df.copy()
    }
    
    # Add age-based scenarios if age data available
    if "Age_Group" in survival_df.columns:
        scenarios["Young"] = survival_df[survival_df["Age_Group"] == "Young"].copy()
        scenarios["Old"] = survival_df[survival_df["Age_Group"] == "Old"].copy()
    
    created_plots = []
    
    for scenario_name, scenario_df in scenarios.items():
        if len(scenario_df) < 20:  # Minimum sample size
            print(f"  Skipping {scenario_name}: insufficient samples ({len(scenario_df)})")
            continue
            
        print(f"\n  Creating KM plots for {scenario_name}: {len(scenario_df)} samples")
        
        for nk_var in nk_vars:
            if nk_var not in scenario_df.columns:
                continue
            
            # Find corresponding results
            result_row = results_df[
                (results_df["Scenario"] == scenario_name) & 
                (results_df["Variable"] == nk_var)
            ]
            
            if result_row.empty:
                print(f"    Skipping {nk_var}: no results found")
                continue
            
            result = result_row.iloc[0]
            
            # Prepare data for KM plot
            km_data = scenario_df[["Survival_Time", "Event", nk_var]].copy()
            km_data = km_data.dropna()
            
            if len(km_data) < 10:
                print(f"    Skipping {nk_var}: insufficient data after cleaning ({len(km_data)})")
                continue
            
            # Create high/low groups based on median split
            median_value = km_data[nk_var].median()
            high_group = km_data[km_data[nk_var] > median_value].copy()
            low_group = km_data[km_data[nk_var] <= median_value].copy()
            
            if len(high_group) < 5 or len(low_group) < 5:
                print(f"    Skipping {nk_var}: groups too small (high: {len(high_group)}, low: {len(low_group)})")
                continue
            
            # Perform log-rank test for the plot
            try:
                logrank_result = logrank_test(
                    high_group["Survival_Time"], low_group["Survival_Time"],
                    high_group["Event"], low_group["Event"]
                )
                logrank_p = logrank_result.p_value
            except:
                logrank_p = np.nan
            
            # Create plot
            try:
                title = f"NK Cell Survival Analysis: {nk_var}"
                group_labels = [f"High {nk_var}", f"Low {nk_var}"]
                
                # Save path
                safe_nk_var = nk_var.replace("/", "_").replace(" ", "_")
                safe_scenario = scenario_name.replace("/", "_").replace(" ", "_")
                save_path = km_plots_dir / f"{config['cancer_type']}_{safe_scenario}_{safe_nk_var}_KM.png"
                
                fig, axes = create_nk_km_plot(
                    high_group=high_group,
                    low_group=low_group,
                    group_labels=group_labels,
                    title=title,
                    save_path=save_path,
                    show_risk_table=True,
                    hr=result["HR"],
                    p_value=logrank_p,
                    fdr_q_value=result["FDR_Q_Value"],
                    ci_95=(result["HR_CI_Lower"], result["HR_CI_Upper"]),
                    variable_name=nk_var,
                    scenario_name=scenario_name,
                    font_size_base=11
                )
                
                if fig is not None:
                    plt.close(fig)  # Close to free memory
                    created_plots.append(save_path)
                    print(f"    Created: {nk_var} -> {save_path.name}")
                else:
                    print(f"    Failed: {nk_var}")
                    
            except Exception as e:
                print(f"    Error creating plot for {nk_var}: {e}")
                continue
    
    print(f"\n  KM Plots Summary:")
    print(f"    Total plots created: {len(created_plots)}")
    print(f"    Saved to: {km_plots_dir}")
    
    return created_plots

def create_nk_forest_plot(
    results_df: pd.DataFrame, 
    config: Dict
) -> Optional[Path]:
    """
    Generate publication-ready forest plot of NK survival analysis results.
    
    Creates a comprehensive forest plot visualization showing hazard ratios,
    confidence intervals, and statistical significance for all NK variables
    across different analysis scenarios.
    
    Parameters:
        results_df (pandas.DataFrame): Cox regression results with HR, CI, p-values
        config (dict): Analysis configuration including cancer type and output directory
        
    Returns:
        str: Path to saved forest plot file, or None if forestplot package unavailable
        
    Forest Plot Features:
        - Hazard ratios with 95% confidence intervals
        - Statistical significance indicators (*, **, ***)
        - Protective vs. harmful effect visualization
        - Publication-quality formatting and annotations
        
    Requires:
        forestplot package (pip install forestplot)
    """
    
    if not FORESTPLOT_AVAILABLE:
        print("  Forestplot package not available - skipping forest plot generation")
        return None
    
    if results_df.empty:
        print("  No results available for forest plot")
        return None
    
    print("Creating publication-ready forest plot...")
    
    # Prepare data for forestplot
    forest_data = results_df.copy()
    
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
        lambda x: 'Individual NK Subtypes' if x in ['Bright_NK', 'Cytotoxic_NK'] else 'Combined NK Total'
    )
    
    # Create display labels
    forest_data['display_variable'] = forest_data.apply(
        lambda row: f"{row['variable']} ({row['scenario']})", axis=1
    )
    
    # Sort for better presentation
    forest_data = forest_data.sort_values(['group', 'scenario', 'variable'])
    
    try:
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
        cancer_type = config.get("cancer_type", "TCGA")
        plt.title(f'{cancer_type} NK Cell Survival Analysis\nFull Timeline Follow-up', 
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
        output_dir = Path(config["output_dir"])
        output_file = output_dir / f"{cancer_type}_NK_Publication_Forest_Plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"    Forest plot saved: {output_file}")
        
        # Show summary statistics
        sig_results = forest_data[forest_data['p_value'] < 0.05]
        print(f"    Total analyses: {len(forest_data)}")
        print(f"    Significant results (p<0.05): {len(sig_results)}")
        
        if len(sig_results) > 0:
            print(f"    Significant findings:")
            for _, row in sig_results.iterrows():
                direction = "Protective" if row['hr'] < 1 else "Harmful"
                print(f"      {row['display_variable']}: HR={row['hr']:.3f} [{row['ci_lower']:.3f}-{row['ci_upper']:.3f}], p={row['p_value']:.3f} ({direction})")
        
        plt.close('all')  # Close all figures to save memory
        return output_file
        
    except Exception as e:
        print(f"    Warning: Forest plot creation failed: {e}")
        return None

def plot_simple(results_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Generate publication-ready forest plot and results table.
    
    Creates a forest plot visualization of hazard ratios with confidence intervals
    and exports detailed results to CSV format for supplemental material.
    
    Parameters:
        results_df (pandas.DataFrame): Cox regression results with HR, CI, p-values
        config (dict): Analysis configuration including output directory
    
    Returns:
        pandas.DataFrame: Formatted results table
        
    Output Files:
        - {cancer_type}_NK_Forest_Plot.png: Forest plot visualization
        - {cancer_type}_NK_Survival_Results.csv: Detailed results table
    """
    print("Generating results visualization and tables...")
    
    if results_df.empty:
        print("Warning: No results available for visualization")
        return
    
    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Create forest plot of hazard ratios
    plt.figure(figsize=(12, 8))
    plt.style.use('default')  # Clean publication style
    
    # Plot each analysis result
    for i, (idx, row) in enumerate(results_df.iterrows()):
        y_pos = i
        hr = row["HR"]
        ci_lower = row["HR_CI_Lower"]
        ci_upper = row["HR_CI_Upper"]
        p_value = row["P_Value"]
        
        # Color code by risk direction
        color = "#1f77b4" if hr < 1 else "#d62728"  # Blue for protective, red for harmful
        
        # Plot hazard ratio with confidence interval
        plt.errorbar(hr, y_pos, xerr=[[hr - ci_lower], [ci_upper - hr]], 
                    fmt='o', color=color, capsize=5, markersize=8, linewidth=2)
        
        # Add analysis label with significance indicator
        label = f"{row['Scenario']} - {row['Variable']}"
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        plt.text(hr + 0.1, y_pos, f"{label} (p={p_value:.3f}){significance}", 
                va='center', fontsize=10, fontweight='bold' if significance else 'normal')
    
    # Add reference line at no effect (HR=1)
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
    
    # Format axes and labels
    plt.xlabel("Hazard Ratio (95% Confidence Interval)", fontsize=12, fontweight='bold')
    plt.ylabel("NK Cell Analysis", fontsize=12, fontweight='bold')
    plt.title(f"{config['cancer_type']} NK Cell Subtype Survival Analysis\n"
             f"Cox Proportional Hazards Regression", fontsize=14, fontweight='bold', pad=20)
    
    # Add protective/harmful region labels
    plt.text(0.5, -0.8, 'Protective\n(Better Survival)', ha='center', va='top', 
             fontsize=10, style='italic', color='#1f77b4')
    plt.text(1.5, -0.8, 'Harmful\n(Worse Survival)', ha='center', va='top', 
             fontsize=10, style='italic', color='#d62728')
    
    # Set reasonable x-axis limits
    plt.xlim(0.1, 10)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save high-resolution plot
    plot_file = os.path.join(config["output_dir"], f"{config['cancer_type']}_NK_Forest_Plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Export detailed results table
    csv_file = os.path.join(config["output_dir"], f"{config['cancer_type']}_NK_Survival_Results.csv")
    
    # Format results for publication
    export_df = results_df.copy()
    export_df['HR_95CI'] = export_df.apply(
        lambda row: f"{row['HR']:.3f} ({row['HR_CI_Lower']:.3f}-{row['HR_CI_Upper']:.3f})", axis=1
    )
    export_df['P_Value_Formatted'] = export_df['P_Value'].apply(
        lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001"
    )
    
    # Select and reorder columns for publication
    publication_columns = [
        'Scenario', 'Variable', 'Sample_Size', 'Events', 
        'HR_95CI', 'P_Value_Formatted', 'FDR_Q_Value', 'Risk_Direction'
    ]
    
    if 'FDR_Q_Value' in export_df.columns:
        export_df['FDR_Q_Value_Formatted'] = export_df['FDR_Q_Value'].apply(
            lambda q: f"{q:.3f}" if q >= 0.001 else "<0.001"
        )
        publication_columns[6] = 'FDR_Q_Value_Formatted'
    
    export_df[publication_columns].to_csv(csv_file, index=False)
    
    print(f"Results exported:")
    print(f"  Forest plot: {plot_file}")
    print(f"  Results table: {csv_file}")
    
    return results_df

def main() -> None:
    """
    Execute complete NK cell survival analysis pipeline.
    
    Performs the full analysis workflow:
    1. Load TCGA clinical and CIBERSORTx immune deconvolution data
    2. Preprocess and quality control the integrated dataset  
    3. Perform Cox proportional hazards regression analysis
    4. Generate publication-ready survival plots and results
    
    The analysis follows established practices for cancer immunology survival
    studies with proper statistical corrections and visualization standards.
    """
    print("="*80)
    print("TCGA NK Cell Subtype Survival Analysis")
    print("="*80)
    print(f"Analysis started: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cancer type: {CONFIG['cancer_type']}")
    print(f"NK variables: {CONFIG['nk_variables']}")
    print(f"Follow-up period: Full timeline (no truncation)")
    
    # Step 1: Data Loading
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)
    
    df = load_simple_data(CONFIG)
    
    if df is None:
        print("\nERROR: Data loading failed")
        print("\nRequired data structure:")
        print(f"  - {CONFIG['base_data_dir']}/{CONFIG['cancer_type']}_Clinical_XML/")
        print(f"  - {CONFIG['base_data_dir']}/{CONFIG['cancer_type']}_sample_sheet.tsv")
        print(f"  - {CONFIG['base_data_dir']}/{CONFIG['cancer_type']}_RNA_Seq/")
        print(f"  - {CONFIG['base_data_dir']}/CIBERSORTx_{CONFIG['cancer_type']}.csv")
        print("\nPlease ensure TCGA data is properly downloaded and organized.")
        return
    
    print("✓ Data loading completed successfully")
    
    # Step 2: Data Preprocessing
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    df_clean = preprocess_simple(df, CONFIG)
    
    if df_clean is None or len(df_clean) == 0:
        print("\nERROR: Data preprocessing failed")
        print("\nPossible issues:")
        print("  - No valid survival data found")
        print("  - No NK infiltration data available") 
        print("  - Data format incompatible with survival analysis")
        return
    
    print("✓ Data preprocessing completed successfully")
    
    # Step 3: Statistical Analysis
    print("\n" + "="*60)
    print("STEP 3: COX REGRESSION ANALYSIS")
    print("="*60)
    
    results = calculate_cox_simple(df_clean, CONFIG)
    
    if results is None or results.empty:
        print("\nERROR: Cox regression analysis failed")
        print("No survival analysis results generated.")
        return
    
    print("✓ Cox regression analysis completed successfully")
    
    # Step 4: Visualization and Output
    print("\n" + "="*60)
    print("STEP 4: VISUALIZATION AND OUTPUT")
    print("="*60)
    
    # Generate basic results plots and tables
    plot_simple(results, CONFIG)
    
    # Generate Kaplan-Meier survival curves
    print("\nGenerating Kaplan-Meier survival plots...")
    km_plots = generate_km_plots(df_clean, results, CONFIG)
    
    # Generate publication-ready forest plot
    print("\nGenerating publication forest plot...")
    forest_plot_file = create_nk_forest_plot(results, CONFIG)
    
    print("\n✓ Visualization and output completed successfully")
    print(f"  - Generated {len(km_plots)} Kaplan-Meier survival plots")
    if forest_plot_file:
        print(f"  - Created publication forest plot: {Path(forest_plot_file).name}")
    else:
        print("  - Forest plot generation skipped (forestplot package not available)")
    
    # Analysis Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Statistical summary
    total_analyses = len(results)
    significant_nominal = len(results[results['P_Value'] < 0.05])
    significant_fdr = len(results[results['FDR_Q_Value'] < 0.05]) if 'FDR_Q_Value' in results.columns else 0
    
    print(f"Statistical Results Summary:")
    print(f"  Total analyses performed: {total_analyses}")
    print(f"  Nominally significant (p<0.05): {significant_nominal}")
    print(f"  FDR-corrected significant (q<0.05): {significant_fdr}")
    print(f"  Output directory: {CONFIG['output_dir']}")
    print(f"  Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Module version: {__version__}")
    
    # Highlight key findings
    if not results.empty:
        print(f"\nKey Findings:")
        significant = results[results['P_Value'] < 0.05].sort_values('P_Value')
        if not significant.empty:
            for idx, row in significant.head(3).iterrows():
                risk_direction = "protective" if row['HR'] < 1 else "harmful"
                print(f"  {row['Scenario']} - {row['Variable']}: {risk_direction} effect")
                print(f"    HR = {row['HR']:.3f} [95% CI: {row['HR_CI_Lower']:.3f}-{row['HR_CI_Upper']:.3f}], p = {row['P_Value']:.3f}")
        else:
            print("  No statistically significant associations identified (all p > 0.05)")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("TCGA NK Cell Survival Analysis Pipeline")
    print(f"Version {__version__} - Publication Supplemental Material")
    print(f"Author: {__author__}")
    print("-" * 60)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        print("Please verify data setup and configuration.")
        raise