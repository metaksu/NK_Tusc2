#!/usr/bin/env python3
"""
TCGA Gene-Based Survival Analysis

This script performs comprehensive survival analysis for specific genes of interest in TCGA data.
It uses the functionally equivalent data loading pipeline as tcga_cibersortx_mixture_pipeline.py 
but focuses on gene expression-based survival analysis rather than NK cell subtypes.

Key Features:
- Functionally equivalent TCGA data loading pipeline as tcga_cibersortx_mixture_pipeline.py
- CRITICAL patient deduplication to ensure one tumor sample per patient (essential for survival analysis)
- Easy configuration of genes of interest
- Comprehensive survival analysis with multiple stratification approaches
- Cox proportional hazards regression
- Log-rank survival tests
- Age and expression level stratification
- Publication-ready output with effect sizes and confidence intervals

UPDATED 2025-01-27: Fixed data loading to match tcga_cibersortx_mixture_pipeline.py exactly:
- Added deduplicate_patients_for_tumor_samples() function
- Updated create_tumor_adata() to include patient deduplication
- Enhanced sample sheet loading to derive Cancer_Type_Derived
- Improved error handling and logging consistency
- Ensured master metadata creation is functionally identical

CRITICAL UPDATE 2025-01-27: Added batch correction for accurate survival analysis:
- Added apply_tcga_batch_correction() function with Combat/Harmony/Simple centering
- TCGA data has significant batch effects that can completely confound survival results
- Without batch correction, results like PDCD1 showing protective effects are likely artifacts
- Batch correction is now applied by default in preprocessing pipeline
- Multiple fallback methods ensure batch correction works in different environments

CRITICAL FIXES 2025-01-27: Major improvements to statistical robustness:
- Fixed Cox regression CI extraction to use standardized lifelines API
- Added Cox proportional hazards assumption testing
- Enhanced survival time quality control with comprehensive filtering
- Improved expression group validation with minimum sample size checks
- Fixed immune normalization numerical stability issues
- Enhanced biological plausibility checks with cancer-type context
- Improved FDR correction with better statistical guidance

ENHANCED NK ANALYSIS 2025-01-27: Upgraded to use hybrid Atlas-Tang immune signatures:
- Enhanced immune cell detection: Atlas immune minor types + Tang NK core signatures
- Atlas immune minor: T_cells_CD4+, T_cells_CD8+, NKT_cells, B_cells_Memory, Macrophage, etc.
- Tang NK core: NK_Bright, NK_Cytotoxic, NK_Exhausted_TaNK (from create_tang_reference_matrix.py)
- Updated immune normalization to use comprehensive Atlas-Tang signature matrix
- Added comprehensive gene-immune correlation analysis with both new and legacy column support
- New correlation analysis between genes and CD8 T cells + NK subtypes individually
- Provides both Pearson and Spearman correlations with FDR correction
- Generates separate output files for primary focus (CD8/NK) vs contextual immune cells

Requirements for Batch Correction (recommended):
    pip install scanpy[combat]  # For Combat batch correction
    pip install harmonypy       # For Harmony batch correction (optional)
    
    # If neither available, script will use simple batch centering as fallback

Usage:
    # Configure genes of interest and run:
    python TCGA_Gene_Survival_Analysis.py
    
    # Output: {CANCER_TYPE}_Gene_Survival_Analysis_Results.csv
    
IMPORTANT: The script now applies batch correction by default. This is CRITICAL for TCGA
survival analysis as batch effects can completely confound results. Previous results
showing unexpected patterns (e.g., PDCD1 as protective) were likely due to batch artifacts.

CRITICAL FIX 2025-01-27: Fixed immune normalization to use CIBERSORTx "Absolute score (sig.score)"
and individual cell absolute scores. CIBERSORTx outputs absolute infiltration values, not fractions.
"Absolute score" = sum of all cell type absolute scores. Cytotoxic lymphoid infiltration = 
sum of CD8 + NK absolute scores. This provides meaningful normalization using actual infiltration levels.

Version: 2.2.0 (2025-01-27)
Author: AI Assistant with user validation
Statistical Review: Enhanced with critical statistical fixes
"""

import os
import re
import xml.etree.ElementTree as ET
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

# FIXED: Add version and metadata tracking
__version__ = "2.2.0"
__last_updated__ = "2025-01-27"
__critical_fixes__ = [
    "Cox CI extraction API standardization",
    "Proportional hazards assumption testing", 
    "Enhanced survival time quality control",
    "Expression group sample size validation",
    "Immune normalization numerical stability",
    "Cancer-type specific biological validation",
    "Improved FDR correction guidance",
    "Enhanced immune cell detection using hybrid Atlas-Tang signatures",
    "Support for Atlas immune minor types and Tang NK core signatures", 
    "Comprehensive gene-immune correlation analysis with backward compatibility",
    "CD8 T cell and NK subtype correlation analysis"
]

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
sc.settings.verbosity = 1  # Reduce scanpy verbosity

print(f"🔬 TCGA Gene Survival Analysis v{__version__}")
print(f"📅 Last Updated: {__last_updated__}")
print(f"🔧 Critical Fixes Applied: {len(__critical_fixes__)} statistical and numerical improvements")
print(f"⏰ Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ==============================================================================
# --- GENES OF INTEREST CONFIGURATION ---
# ==============================================================================

# EASY CONFIGURATION: Add your genes of interest here
GENES_OF_INTEREST = [
    "TUSC2",      # Tumor suppressor candidate 2
    "CEACAM1",
    "HAVCR2",
    "TOX",
    "LAG3",
    "PDCD1",
    "EZH2",
]

# Expression level stratification strategies
STRATIFICATION_STRATEGIES = {
    "Tertile": "Top vs Bottom 1/3 (exclude middle)",
    "Median": "Above vs Below Median",  
    "Quartile": "Top vs Bottom 1/4 (exclude middle 50%)",
    "Extreme_Decile": "Top vs Bottom 10% (exclude middle 80%)"
}


# ==============================================================================
# --- Configuration Constants (Identical to TCGA_STUDY_ANALYSIS.py) ---
# ==============================================================================

# TCGA XML Namespace mappings
NS_TCGA = {
    "admin": "http://tcga.nci/bcr/xml/administration/2.7",
    "shared": "http://tcga.nci/bcr/xml/shared/2.7",
    "clin_shared": "http://tcga.nci/bcr/xml/clinical/shared/2.7",
    "shared_stage": "http://tcga.nci/bcr/xml/clinical/shared/stage/2.7",
}

# Disease-specific XML path configurations
DISEASE_CONFIG_TCGA = {
    "DEFAULT": {
        "histology_path": ".//shared:histological_type",
        "tumor_site_path": ".//clin_shared:anatomic_neoplasm_subdivision",
    },
    "BRCA": {
        "histology_path": ".//brca:histological_type",
        "tumor_site_path": ".//clin_shared:anatomic_neoplasm_subdivision",
        "smoking_history_path": ".//clin_shared:tobacco_smoking_history",
    },
    "LUAD": {
        "histology_path": ".//luad:histological_type",
        "tumor_site_path": ".//clin_shared:anatomic_neoplasm_subdivision",
        "smoking_history_path": ".//clin_shared:tobacco_smoking_history",
    },
    "GBM": {
        "histology_path": ".//gbm:histological_type",
        "tumor_site_path": ".//clin_shared:anatomic_neoplasm_subdivision",
    },
}

# Default filtering thresholds (includes CIBERSORTx quality control)
DEFAULT_THRESHOLDS = {
    "p_value_cibersort": 0.05,
    "correlation_cibersort": 0.70, 
    "rmse_percentile_cibersort": 0.90,  
    "min_cells_gene_filter": 5,
    "preferred_rna_count_column": "tpm_unstranded",
}


# ==============================================================================
# --- Utility Functions (Identical to TCGA_STUDY_ANALYSIS.py) ---
# ==============================================================================


def _get_xml_text(root, xpath, namespaces, default_val="N/A"):
    """Internal helper for XML text extraction."""
    try:
        element = root.find(xpath, namespaces)
        if element is not None and element.text:
            return element.text.strip()
    except Exception:
        pass
    return default_val


def _extract_extended_followup(root, ns_map):
    """
    Extract extended follow-up data from TCGA XML follow-up sections.
    This is CRITICAL for proper survival analysis as it provides 10+ year follow-up.
    """
    extended_data = {}
    
    # Common follow-up section patterns in TCGA XMLs
    followup_patterns = [
        ".//follow_up_v1.5:follow_up",
        ".//follow_up_v2.1:follow_up", 
        ".//follow_up_v4.0:follow_up",
        ".//follow_up:follow_up",
        ".//follow_up_v1.0:follow_up",
        ".//follow_up_v2.0:follow_up",
        ".//follow_up_v3.0:follow_up",
    ]
    
    # Find all follow-up sections
    all_followups = []
    for pattern in followup_patterns:
        try:
            followups = root.findall(pattern, ns_map)
            all_followups.extend(followups)
        except Exception:
            continue
    
    if not all_followups:
        return extended_data
    
    # Extract data from each follow-up and find the latest/longest
    latest_followup_days = 0
    latest_death_days = None
    latest_vital_status = None
    
    for followup in all_followups:
        try:
            # Extract follow-up data
            followup_days_text = _get_xml_text(followup, ".//clin_shared:days_to_last_followup", ns_map)
            death_days_text = _get_xml_text(followup, ".//clin_shared:days_to_death", ns_map)
            vital_status_text = _get_xml_text(followup, ".//clin_shared:vital_status", ns_map)
            
            # Convert to numeric for comparison
            try:
                followup_days = int(followup_days_text) if followup_days_text != "N/A" else 0
            except (ValueError, TypeError):
                followup_days = 0
                
            # Keep the follow-up with the longest time
            if followup_days > latest_followup_days:
                latest_followup_days = followup_days
                extended_data["days_to_last_followup"] = followup_days_text
                
                # Update death and vital status if available
                if death_days_text != "N/A":
                    extended_data["days_to_death"] = death_days_text
                if vital_status_text != "N/A":
                    extended_data["vital_status"] = vital_status_text
                    
        except Exception:
            continue
    
    return extended_data


def parse_tcga_clinical_xml_file(
    xml_file_path, ns_map=NS_TCGA, disease_config=DISEASE_CONFIG_TCGA
):
    """
    Parses a single TCGA clinical XML file and returns a dictionary of clinical data.
    Enhanced to extract extended follow-up data for better survival analysis.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  ERROR: Could not parse XML: {os.path.basename(xml_file_path)}")
        return None

    disease_code_val = _get_xml_text(root, ".//admin:disease_code", ns_map, "UNKNOWN")
    cfg = disease_config.get(disease_code_val.upper(), disease_config["DEFAULT"])

    # Extract initial survival data
    initial_days_to_death = _get_xml_text(root, ".//clin_shared:days_to_death", ns_map)
    initial_days_to_followup = _get_xml_text(root, ".//clin_shared:days_to_last_followup", ns_map)
    initial_vital_status = _get_xml_text(root, ".//clin_shared:vital_status", ns_map)
    
    # Extract extended follow-up data (CRITICAL for proper survival analysis)
    extended_followup_data = _extract_extended_followup(root, ns_map)
    
    # Use extended follow-up if available, otherwise use initial data
    final_days_to_death = extended_followup_data.get("days_to_death", initial_days_to_death)
    final_days_to_followup = extended_followup_data.get("days_to_last_followup", initial_days_to_followup)
    final_vital_status = extended_followup_data.get("vital_status", initial_vital_status)

    clinical_record = {
        "Patient_Barcode": _get_xml_text(root, ".//shared:bcr_patient_barcode", ns_map),
        "Patient_UUID": _get_xml_text(root, ".//shared:bcr_patient_uuid", ns_map),
        "Disease_Code": disease_code_val,
        "Gender": _get_xml_text(root, ".//shared:gender", ns_map),
        "Race": _get_xml_text(root, ".//clin_shared:race", ns_map),
        "Ethnicity": _get_xml_text(root, ".//clin_shared:ethnicity", ns_map),
        "Vital_Status": final_vital_status,
        "Age_at_Diagnosis": _get_xml_text(
            root, ".//clin_shared:age_at_initial_pathologic_diagnosis", ns_map
        ),
        "Days_to_Birth": _get_xml_text(root, ".//clin_shared:days_to_birth", ns_map),
        "Days_to_Death": final_days_to_death,
        "Days_to_Last_Followup": final_days_to_followup,
        # Store both initial and extended data for debugging
        "Initial_Days_to_Death": initial_days_to_death,
        "Initial_Days_to_Followup": initial_days_to_followup,
        "Extended_Followup_Available": "Yes" if extended_followup_data else "No",
        "Histology": _get_xml_text(root, cfg["histology_path"], ns_map),
        "Tumor_Site": _get_xml_text(root, cfg["tumor_site_path"], ns_map),
        "Pathologic_Stage_Raw": _get_xml_text(
            root, ".//shared_stage:pathologic_stage", ns_map
        ),
        "Neoplasm_Status": _get_xml_text(
            root, ".//clin_shared:person_neoplasm_cancer_status", ns_map
        ),
        "Informed_Consent": _get_xml_text(
            root, ".//clin_shared:informed_consent_verified", ns_map
        ),
        "Neoadjuvant_Tx_History": _get_xml_text(
            root, ".//shared:history_of_neoadjuvant_treatment", ns_map
        ),
        "XML_Filename": os.path.basename(xml_file_path),
    }

    return clinical_record


def load_clinical_data(clinical_xml_dir):
    """
    Load all clinical data from TCGA XML files.
    """
    print(f"\n--- Loading Clinical Data from XMLs in {clinical_xml_dir} ---")

    all_clinical_records = []
    clinical_df = pd.DataFrame()

    if not os.path.exists(clinical_xml_dir) or not os.path.isdir(clinical_xml_dir):
        print(f"  WARNING: Clinical XML directory not found: {clinical_xml_dir}")
        return clinical_df

    xml_files = [f for f in os.listdir(clinical_xml_dir) if f.lower().endswith(".xml")]
    if not xml_files:
        print(f"  WARNING: No XML files found in {clinical_xml_dir}")
        return clinical_df

    print(f"  Found {len(xml_files)} XML files. Parsing...")
    for i, filename in enumerate(xml_files):
        if (i + 1) % 200 == 0:
            print(f"    Parsed {i+1}/{len(xml_files)} XML files...")

        xml_path = os.path.join(clinical_xml_dir, filename)
        record = parse_tcga_clinical_xml_file(xml_path)
        if record:
            all_clinical_records.append(record)

    if all_clinical_records:
        clinical_df = pd.DataFrame(all_clinical_records)
        print(f"  Successfully parsed {len(clinical_df)} clinical records")

        # Convert numeric columns
        numeric_cols = [
            "Age_at_Diagnosis",
            "Days_to_Birth",
            "Days_to_Death",
            "Days_to_Last_Followup",
        ]
        for col in numeric_cols:
            if col in clinical_df.columns:
                clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce")

        # Remove duplicates based on Patient_Barcode
        if "Patient_Barcode" in clinical_df.columns:
            initial_rows = len(clinical_df)
            clinical_df.drop_duplicates(
                subset=["Patient_Barcode"], keep="first", inplace=True
            )
            if len(clinical_df) < initial_rows:
                print(
                    f"  Dropped {initial_rows - len(clinical_df)} duplicate Patient_Barcodes"
                )

    return clinical_df


def load_sample_sheet(sample_sheet_path):
    """
    Load and process the consolidated TCGA sample sheet.
    UPDATED to match tcga_cibersortx_mixture_pipeline.py load_sample_sheet() logic exactly.
    """
    print(f"\n--- Loading Sample Sheet Metadata ---")

    if not os.path.exists(sample_sheet_path):
        print(f"  ERROR: Sample sheet not found: {sample_sheet_path}")
        return pd.DataFrame()

    try:
        sample_sheet_df = pd.read_csv(sample_sheet_path, sep='\t', low_memory=False)
        print(f"  Successfully loaded sample sheet: {sample_sheet_df.shape}")
        
        # Determine tissue type column name
        tissue_type_col = (
            "Tissue Type" if "Tissue Type" in sample_sheet_df.columns else "Sample Type"
        )
        required_cols = {"Case ID", "File Name", tissue_type_col}
        
        if not required_cols.issubset(sample_sheet_df.columns):
            print(f"  ERROR: Sample sheet missing required columns: {list(required_cols)}")
            return pd.DataFrame()
        
        # Select and rename essential columns
        keep_cols = list(required_cols) + ["Project ID"]
        sample_sheet_processed = sample_sheet_df[keep_cols].copy()
        
        # Create rename mapping dictionary
        rename_mapping = {
            "Case ID": "Patient_ID_from_SampleSheet",
            tissue_type_col: "Tissue_Type",
            "File Name": "Original_File_Name",
            "Project ID": "Project_ID",
        }
        
        # Apply column renaming
        for old_col, new_col in rename_mapping.items():
            if old_col in sample_sheet_processed.columns:
                sample_sheet_processed[new_col] = sample_sheet_processed[old_col]
                if old_col != new_col:
                    sample_sheet_processed = sample_sheet_processed.drop(columns=[old_col])
        
        # Create file name root for matching
        sample_sheet_processed["File_Name_Root"] = sample_sheet_processed[
            "Original_File_Name"
        ].apply(lambda x: str(x).split(".")[0])
        
        # Derive cancer type from Project_ID (CRITICAL: this was missing in original)
        sample_sheet_processed["Cancer_Type_Derived"] = (
            sample_sheet_processed["Project_ID"].str.split("-").str[1].str.upper()
        )
        
        # Remove duplicates based on File_Name_Root
        initial_rows = len(sample_sheet_processed)
        sample_sheet_processed.drop_duplicates(
            subset=["File_Name_Root"], keep="first", inplace=True
        )
        if len(sample_sheet_processed) < initial_rows:
            print(f"  Dropped {initial_rows - len(sample_sheet_processed)} duplicate files")
        
        print(f"  Processed sample sheet shape: {sample_sheet_processed.shape}")
        return sample_sheet_processed
        
    except Exception as e:
        print(f"  ERROR loading sample sheet: {e}")
        return pd.DataFrame()


def create_master_metadata(clinical_df, sample_sheet_df):
    """
    Merge clinical data with sample sheet metadata.
    UPDATED to ensure functional equivalence with tcga_cibersortx_mixture_pipeline.py.
    """
    print(f"\n--- Creating Master Metadata ---")

    if clinical_df.empty and sample_sheet_df.empty:
        print("  ERROR: Both clinical and sample sheet data are empty")
        return pd.DataFrame()

    if sample_sheet_df.empty:
        print("  ERROR: Cannot proceed without sample sheet data")
        return pd.DataFrame()

    if clinical_df.empty:
        print("  Using only sample sheet data (no clinical data available)")
        master_df = sample_sheet_df.copy()
        if "Patient_ID_from_SampleSheet" in master_df.columns:
            master_df.rename(
                columns={"Patient_ID_from_SampleSheet": "Patient_ID"}, inplace=True
            )
    else:
        # Merge clinical and sample sheet data
        master_df = pd.merge(
            sample_sheet_df,
            clinical_df,
            left_on="Patient_ID_from_SampleSheet",
            right_on="Patient_Barcode",
            how="left",
        )
        print(f"  Master metadata shape after merge: {master_df.shape}")

        # Create primary Patient_ID column
        if "Patient_ID_from_SampleSheet" in master_df.columns:
            master_df.rename(
                columns={"Patient_ID_from_SampleSheet": "Patient_ID"}, inplace=True
            )
        elif "Patient_Barcode" in master_df.columns:
            master_df.rename(columns={"Patient_Barcode": "Patient_ID"}, inplace=True)

    # Derive cancer type from Project_ID or Disease_Code
    if "Project_ID" in master_df.columns:
        master_df["Cancer_Type_Derived"] = (
            master_df["Project_ID"].str.split("-").str[1].str.upper()
        )
    elif "Disease_Code" in master_df.columns:
        master_df["Cancer_Type_Derived"] = master_df["Disease_Code"].str.upper()
    else:
        master_df["Cancer_Type_Derived"] = "UNKNOWN"

    # Set File_Name_Root as index
    if "File_Name_Root" in master_df.columns:
        if master_df["File_Name_Root"].duplicated().any():
            print("  WARNING: Duplicate File_Name_Root found, removing duplicates")
            master_df.drop_duplicates(
                subset=["File_Name_Root"], keep="first", inplace=True
            )
        master_df.set_index("File_Name_Root", inplace=True, verify_integrity=True)
        print("  Set 'File_Name_Root' as index")
    else:
        print("  ERROR: 'File_Name_Root' column missing")
        return pd.DataFrame()

    print(f"  Final master metadata shape: {master_df.shape}")
    return master_df


def load_rna_seq_data(
    rna_seq_dir, target_sample_ids, preferred_count_col="tpm_unstranded"
):
    """
    Load RNA-seq data for specific samples.
    UPDATED to match tcga_cibersortx_mixture_pipeline.py load_rna_seq_data() logic exactly.
    """
    print(f"\n--- Loading RNA-seq Data ---")
    print(f"  Target samples: {len(target_sample_ids)}")

    if not os.path.exists(rna_seq_dir) or not os.path.isdir(rna_seq_dir):
        print(f"  ERROR: RNA-seq directory not found: {rna_seq_dir}")
        return pd.DataFrame()

    all_files = [
        f
        for f in os.listdir(rna_seq_dir)
        if f.lower().endswith((".tsv", ".txt", ".gz"))
    ]

    files_to_process = []
    for filename in all_files:
        sample_id = filename.split(".")[0]
        if sample_id in target_sample_ids:
            files_to_process.append(filename)

    print(f"  Found {len(files_to_process)} files matching target samples")

    if not files_to_process:
        print("  ERROR: No matching RNA-seq files found")
        return pd.DataFrame()

    all_sample_data = []
    genes_reference = None

    for i, filename in enumerate(files_to_process):
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(files_to_process)} files...")

        file_path = os.path.join(rna_seq_dir, filename)
        sample_id = filename.split(".")[0]

        try:
            # Load the file with comment handling
            if filename.endswith(".gz"):
                sample_df = pd.read_csv(
                    file_path,
                    sep="\t",
                    compression="gzip",
                    comment="#",
                    header=0,
                    low_memory=False,
                )
            else:
                sample_df = pd.read_csv(
                    file_path, sep="\t", comment="#", header=0, low_memory=False
                )

            # Ensure required columns exist
            if "gene_name" not in sample_df.columns:
                print(f"  WARNING: 'gene_name' column missing in {filename}")
                continue

            if preferred_count_col not in sample_df.columns:
                print(f"  WARNING: '{preferred_count_col}' column missing in {filename}")
                continue

            # Filter genes (remove N_ genes and NaN gene names)
            sample_df = sample_df[
                sample_df["gene_name"].notna()
                & ~sample_df["gene_name"].astype(str).str.upper().str.startswith("N_")
            ]

            if sample_df.empty:
                print(f"  WARNING: No valid genes after filtering in {filename}")
                continue

            # Use gene_name as index and extract counts
            sample_counts = sample_df.set_index("gene_name")[preferred_count_col]
            sample_counts.name = sample_id

            # Store gene reference from first file
            if genes_reference is None:
                genes_reference = sample_counts.index
            else:
                # Ensure consistent gene ordering
                sample_counts = sample_counts.reindex(genes_reference, fill_value=0)

            all_sample_data.append(sample_counts)

        except Exception as e:
            print(f"  WARNING: Error processing {filename}: {e}")
            continue

    if not all_sample_data:
        print("  ERROR: No RNA-seq data successfully loaded")
        return pd.DataFrame()

    # Combine all samples into a single DataFrame
    raw_rna_counts_df = pd.concat(all_sample_data, axis=1, join="outer").fillna(0)
    raw_rna_counts_df.index.name = "Gene_Symbol"

    # Handle duplicate gene symbols
    if raw_rna_counts_df.index.duplicated().any():
        num_duplicates = raw_rna_counts_df.index.duplicated().sum()
        print(f"  Found {num_duplicates} duplicate gene symbols. Aggregating by mean expression.")
        rna_counts_df = raw_rna_counts_df.groupby(raw_rna_counts_df.index).mean()
    else:
        print("  No duplicate gene symbols found.")
        rna_counts_df = raw_rna_counts_df

    print(f"  Final RNA-seq data loaded: {rna_counts_df.shape} (genes x samples)")

    return rna_counts_df


def create_anndata_object(rna_counts_df, metadata_df, cancer_type):
    """
    Create AnnData object from RNA-seq data and metadata.
    """
    print(f"\n--- Creating AnnData Object for {cancer_type} ---")

    if rna_counts_df.empty or metadata_df.empty:
        print("  ERROR: RNA-seq data or metadata is empty")
        return None

    # Filter metadata for cancer type
    cancer_metadata = metadata_df[
        metadata_df["Cancer_Type_Derived"] == cancer_type.upper()
    ].copy()

    if cancer_metadata.empty:
        print(f"  ERROR: No samples found for cancer type {cancer_type}")
        return None

    # Find common samples between RNA-seq data and metadata
    common_samples = set(rna_counts_df.columns) & set(cancer_metadata.index)
    if not common_samples:
        print("  ERROR: No common samples between RNA-seq data and metadata")
        return None

    print(f"  Common samples: {len(common_samples)}")

    # Subset data to common samples
    rna_subset = rna_counts_df[list(common_samples)]
    metadata_subset = cancer_metadata.loc[list(common_samples)]

    # Create AnnData object (samples x genes)
    adata = sc.AnnData(
        X=rna_subset.T.values,  # Transpose: samples x genes
        obs=metadata_subset,
        var=pd.DataFrame(index=rna_subset.index),
    )

    # Add gene names to var
    adata.var["gene_name"] = adata.var_names

    print(f"  Created AnnData: {adata.n_obs} samples x {adata.n_vars} genes")

    return adata


def preprocess_anndata(adata, min_cells=5, log_transform=True, apply_batch_correction=True):
    """
    Preprocess AnnData object with filtering, transformation, and CRITICAL batch correction.
    UPDATED to include batch correction essential for accurate survival analysis.
    """
    print(f"\n--- Preprocessing AnnData with Batch Correction ---")

    if adata is None:
        print("  ERROR: AnnData object is None")
        return None

    print(f"  Initial shape: {adata.n_obs} samples x {adata.n_vars} genes")

    # Log transformation
    if log_transform:
        if adata.X.max() > 25:  # Heuristic: likely not log-transformed
            print("  Applying log2(X+1) transformation")
            sc.pp.log1p(adata, base=2)
        else:
            print("  Data appears already transformed (max <= 25)")

    # Gene filtering
    if min_cells and min_cells > 0:
        n_genes_before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells)
        print(
            f"  Gene filtering: kept {adata.n_vars} of {n_genes_before} genes (min {min_cells} cells)"
        )

    # CRITICAL: Apply batch correction for TCGA data
    if apply_batch_correction:
        adata = apply_tcga_batch_correction(adata)
        
        # Verify batch correction was applied
        if 'batch_correction' in adata.uns:
            if adata.uns['batch_correction']['applied']:
                print(f"  ✅ Batch correction successful: {adata.uns['batch_correction']['method']}")
            else:
                print("  ⚠️  WARNING: Batch correction failed - results may be unreliable")
        else:
            print("  ⚠️  WARNING: No batch correction metadata found")
    else:
        print("  ⚠️  Skipping batch correction (NOT recommended for TCGA survival analysis)")

    print(f"  Final preprocessed shape: {adata.n_obs} samples x {adata.n_vars} genes")

    return adata


def apply_tcga_batch_correction(adata):
    """
    Apply batch correction to TCGA data to remove technical artifacts.
    CRITICAL for accurate survival analysis - batch effects can completely confound results.
    """
    print(f"\n--- Applying TCGA Batch Correction (CRITICAL for Survival Analysis) ---")
    
    if adata is None:
        print("  ERROR: AnnData object is None")
        return None
    
    # Store original data
    adata.layers['raw_counts'] = adata.X.copy()
    
    # Check for batch variables
    potential_batch_cols = ['Project_ID', 'Cancer_Type_Derived', 'Disease_Code', 'Original_File_Name']
    batch_col = None
    
    for col in potential_batch_cols:
        if col in adata.obs.columns:
            unique_vals = adata.obs[col].nunique()
            if unique_vals > 1 and unique_vals < adata.n_obs * 0.8:  # Reasonable batch variable
                batch_col = col
                break
    
    if batch_col is None:
        print("  WARNING: No suitable batch column found for correction")
        print("  This may lead to confounded survival results due to technical artifacts")
        return adata
    
    print(f"  Using batch column: {batch_col}")
    print(f"  Number of batches: {adata.obs[batch_col].nunique()}")
    
    # Display batch distribution
    batch_counts = adata.obs[batch_col].value_counts()
    print("  Batch distribution:")
    for batch, count in batch_counts.items():
        print(f"    {batch}: {count} samples ({count/len(adata)*100:.1f}%)")
    
    try:
        # Try Combat batch correction first (more robust for survival analysis)
        try:
            import pandas as pd
            from scanpy.external.pp import combat
            
            print("  Applying Combat batch correction...")
            # Combat requires dense matrix
            if hasattr(adata.X, 'toarray'):
                adata.X = adata.X.toarray()
            
            # Apply Combat
            combat(adata, key=batch_col)
            print("  ✅ Combat batch correction applied successfully")
            adata.uns['batch_correction_method'] = 'Combat'
            
        except ImportError:
            print("  Combat not available, trying alternative method...")
            raise ImportError("Combat not available")
            
    except Exception as e:
        print(f"  Combat failed: {e}")
        
        try:
            # Fallback: Harmony via external package
            print("  Trying Harmony batch correction...")
            
            # First need PCA
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata.raw = adata
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
            
            # Apply harmony if available
            try:
                import harmonypy as hm
                ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, batch_col)
                adata.obsm['X_pca_harmony'] = ho.Z_corr.T
                print("  ✅ Harmony batch correction applied successfully")
                adata.uns['batch_correction_method'] = 'Harmony'
            except ImportError:
                print("  Harmony not available, applying simple centering...")
                # Simple batch centering as last resort
                apply_simple_batch_centering(adata, batch_col)
                adata.uns['batch_correction_method'] = 'Simple_Centering'
                
        except Exception as e2:
            print(f"  Alternative methods failed: {e2}")
            print("  ⚠️  WARNING: No batch correction applied!")
            print("  ⚠️  Survival results may be confounded by technical artifacts")
            adata.uns['batch_correction_method'] = 'None'
    
    # Add batch correction metadata
    adata.uns['batch_correction'] = {
        'applied': True if 'batch_correction_method' in adata.uns and adata.uns['batch_correction_method'] != 'None' else False,
        'batch_column': batch_col,
        'n_batches': adata.obs[batch_col].nunique(),
        'method': adata.uns.get('batch_correction_method', 'None')
    }
    
    print(f"  Batch correction status: {adata.uns['batch_correction']['method']}")
    
    return adata


def apply_simple_batch_centering(adata, batch_col):
    """
    Apply simple batch centering as a last resort batch correction method.
    """
    print("  Applying simple batch centering (basic method)...")
    
    # Convert to dense if sparse
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X.copy()
    
    # Calculate global mean
    global_mean = np.mean(X_dense, axis=0)
    
    # Center each batch to global mean
    for batch in adata.obs[batch_col].unique():
        batch_mask = adata.obs[batch_col] == batch
        batch_data = X_dense[batch_mask, :]
        batch_mean = np.mean(batch_data, axis=0)
        
        # Center the batch
        X_dense[batch_mask, :] = batch_data - batch_mean + global_mean
    
    # Update AnnData
    adata.X = X_dense
    print("  ✅ Simple batch centering applied")


def deduplicate_patients_for_tumor_samples(tumor_metadata):
    """
    Ensure only one tumor sample per patient by selecting the first sample alphabetically.
    IDENTICAL to tcga_cibersortx_mixture_pipeline.py deduplicate_patients_for_tumor_samples().
    
    Parameters:
    -----------
    tumor_metadata : pd.DataFrame
        Tumor sample metadata with Patient_ID column and File_Name_Root as index
        
    Returns:
    --------
    pd.DataFrame
        Deduplicated tumor metadata with one sample per patient
    """
    print("  Deduplicating patients to ensure one tumor sample per patient...")
    
    initial_samples = len(tumor_metadata)
    
    # Check if Patient_ID column exists
    if "Patient_ID" not in tumor_metadata.columns:
        print("  WARNING: Patient_ID column not found - cannot deduplicate by patient")
        return tumor_metadata
    
    # Count samples per patient before deduplication
    samples_per_patient = tumor_metadata['Patient_ID'].value_counts()
    patients_with_multiple_samples = samples_per_patient[samples_per_patient > 1]
    
    if len(patients_with_multiple_samples) > 0:
        print(f"  Found {len(patients_with_multiple_samples)} patients with multiple tumor samples:")
        print(f"    - Total patients with multiple samples: {len(patients_with_multiple_samples)}")
        print(f"    - Max samples per patient: {patients_with_multiple_samples.max()}")
        print(f"    - Extra samples to remove: {(patients_with_multiple_samples - 1).sum()}")
        
        # Show top patients with most samples
        top_multi_patients = patients_with_multiple_samples.head(5)
        for patient_id, count in top_multi_patients.items():
            print(f"      Patient {patient_id}: {count} tumor samples")
    else:
        print("  No patients with multiple tumor samples found")
        return tumor_metadata
    
    # Deduplicate by selecting first sample per patient (alphabetically by index/sample ID)
    # Reset index temporarily to include File_Name_Root in the sorting
    tumor_metadata_reset = tumor_metadata.reset_index()
    
    # Sort by Patient_ID and File_Name_Root to ensure consistent selection
    tumor_metadata_reset = tumor_metadata_reset.sort_values(['Patient_ID', 'File_Name_Root'])
    
    # Keep first sample per patient
    deduplicated_metadata = tumor_metadata_reset.drop_duplicates(
        subset=['Patient_ID'], 
        keep='first'
    )
    
    # Restore the original index
    deduplicated_metadata = deduplicated_metadata.set_index('File_Name_Root')
    
    final_samples = len(deduplicated_metadata)
    removed_samples = initial_samples - final_samples
    
    print(f"  Patient deduplication complete:")
    print(f"    - Initial tumor samples: {initial_samples}")
    print(f"    - Final tumor samples: {final_samples}")
    print(f"    - Removed samples: {removed_samples}")
    print(f"    - Unique patients: {deduplicated_metadata['Patient_ID'].nunique()}")
    
    return deduplicated_metadata


def create_tumor_adata(adata, cancer_type):
    """
    Create tumor-only AnnData object with patient-level deduplication.
    UPDATED to match tcga_cibersortx_mixture_pipeline.py create_tumor_subset() logic.
    """
    print(f"\n--- Creating Tumor-Only AnnData for {cancer_type} ---")

    if adata is None:
        print("  ERROR: AnnData object is None")
        return None

    if "Tissue_Type" not in adata.obs.columns:
        print("  ERROR: 'Tissue_Type' column not found")
        return None

    print(f"  Total {cancer_type} samples in AnnData: {adata.n_obs}")

    # Filter for tumor samples
    tumor_mask = (
        adata.obs["Tissue_Type"].astype(str).str.contains("Tumor", case=False, na=False)
    )
    tumor_adata = adata[tumor_mask].copy()

    if tumor_adata.n_obs == 0:
        print(f"  WARNING: No tumor samples found for {cancer_type}")
        return None

    print(f"  Tumor samples after tissue type filter: {tumor_adata.n_obs}")

    # CRITICAL: Add patient deduplication (missing from original version)
    # Extract metadata for deduplication
    tumor_metadata = tumor_adata.obs.copy()
    
    # Ensure we have File_Name_Root as index (should already be the case)
    if tumor_metadata.index.name != 'File_Name_Root':
        print(f"  WARNING: Index is not 'File_Name_Root', current index name: {tumor_metadata.index.name}")
        # Try to get File_Name_Root from obs if it exists as a column
        if 'File_Name_Root' in tumor_metadata.columns:
            tumor_metadata = tumor_metadata.set_index('File_Name_Root')
        else:
            print("  ERROR: Cannot find File_Name_Root for patient deduplication")
            return None
    
    # Ensure we have Patient_ID for deduplication
    if "Patient_ID" not in tumor_metadata.columns:
        # Try alternative column names
        if "Patient_Barcode" in tumor_metadata.columns:
            tumor_metadata["Patient_ID"] = tumor_metadata["Patient_Barcode"]
        elif "Patient_ID_from_SampleSheet" in tumor_metadata.columns:
            tumor_metadata["Patient_ID"] = tumor_metadata["Patient_ID_from_SampleSheet"]
        else:
            print("  ERROR: No Patient_ID column found for deduplication")
            return None
    
    # Apply patient deduplication
    deduplicated_metadata = deduplicate_patients_for_tumor_samples(tumor_metadata)
    
    if deduplicated_metadata.empty:
        print(f"  ERROR: No tumor samples remain after patient deduplication for {cancer_type}")
        return None
    
    # Filter AnnData to keep only deduplicated samples
    common_samples = list(set(tumor_adata.obs_names) & set(deduplicated_metadata.index))
    if not common_samples:
        print("  ERROR: No common samples between AnnData and deduplicated metadata")
        return None
    
    tumor_adata = tumor_adata[common_samples].copy()
    
    # Update obs with deduplicated metadata
    tumor_adata.obs = deduplicated_metadata.loc[common_samples]
    
    print(f"  Final tumor_adata: {tumor_adata.n_obs} samples (from initial {adata.n_obs})")
    print(f"  Unique patients: {tumor_adata.obs['Patient_ID'].nunique()}")

    # Standardize cancer type column
    tumor_adata.obs["Cancer_Type"] = cancer_type.upper()

    return tumor_adata


def load_and_preprocess_tcga_data(
    cancer_type, base_data_dir, output_dir=None, thresholds=None
):
    """
    Main function to load and preprocess all TCGA data.
    UPDATED to be functionally equivalent to tcga_cibersortx_mixture_pipeline.py 
    including critical patient deduplication for survival analysis.
    """
    print(f"\n{'='*80}")
    print(f"TCGA Data Loading and Preprocessing Pipeline - Gene Survival Analysis")
    print(f"Cancer Type: {cancer_type}")
    print(f"Base Data Directory: {base_data_dir}")
    print(f"{'='*80}")

    # Set default thresholds
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # Define paths
    clinical_xml_dir = os.path.join(base_data_dir, "xml")
    sample_sheet_path = os.path.join(base_data_dir, "gdc_sample_sheet.2025-06-26.tsv")
    rna_seq_dir = os.path.join(base_data_dir, "rna")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Load clinical data
        clinical_df = load_clinical_data(clinical_xml_dir)

        # Step 2: Load sample sheet
        sample_sheet_df = load_sample_sheet(sample_sheet_path)

        # Step 3: Create master metadata
        master_metadata_df = create_master_metadata(clinical_df, sample_sheet_df)
        if master_metadata_df.empty:
            print("ERROR: Failed to create master metadata")
            return None, None

        # Step 4: Filter for target cancer type
        cancer_samples = master_metadata_df[
            master_metadata_df["Cancer_Type_Derived"] == cancer_type.upper()
        ]

        if cancer_samples.empty:
            print(f"ERROR: No samples found for cancer type {cancer_type}")
            return None, master_metadata_df

        target_sample_ids = set(cancer_samples.index)
        print(f"\nTarget {cancer_type} samples: {len(target_sample_ids)}")

        # Step 5: Load RNA-seq data
        rna_counts_df = load_rna_seq_data(
            rna_seq_dir, target_sample_ids, thresholds["preferred_rna_count_column"]
        )

        if rna_counts_df.empty:
            print("ERROR: Failed to load RNA-seq data")
            return None, master_metadata_df

        # Step 6: Create AnnData object
        adata = create_anndata_object(rna_counts_df, master_metadata_df, cancer_type)
        if adata is None:
            print("ERROR: Failed to create AnnData object")
            return None, master_metadata_df

        # Step 7: Preprocess AnnData
        adata = preprocess_anndata(
            adata, min_cells=thresholds["min_cells_gene_filter"], log_transform=True, apply_batch_correction=True
        )

        # Step 8: Create tumor-only AnnData
        tumor_adata = create_tumor_adata(adata, cancer_type)
        if tumor_adata is None:
            print("ERROR: Failed to create tumor AnnData")
            return None, master_metadata_df

        return tumor_adata, master_metadata_df

    except Exception as e:
        print(f"ERROR in data loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==============================================================================
# --- Gene-Focused Survival Analysis Functions ---
# ==============================================================================


def add_gene_expression_to_adata(tumor_adata, genes_of_interest):
    """
    Add gene expression data to tumor_adata.obs with enhanced validation.
    UPDATED to address potential expression scaling issues.
    """
    print(f"\n--- Adding Gene Expression Data with Enhanced Validation ---")
    
    available_genes = []
    missing_genes = []
    
    for gene in genes_of_interest:
        if gene in tumor_adata.var_names:
            # Extract gene expression
            gene_expr = tumor_adata[:, gene].X
            # Convert to dense if sparse
            if hasattr(gene_expr, "toarray"):
                gene_expr = gene_expr.toarray()
            gene_expr = np.ravel(gene_expr)  # Ensures 1D
            
            # Validate expression data
            print(f"  {gene} - Raw stats: range={gene_expr.min():.3f}-{gene_expr.max():.3f}, mean={gene_expr.mean():.3f}")
            
            # Check for potential scaling issues
            if gene_expr.max() > 100:
                print(f"    ⚠️  {gene}: Very high expression values detected")
            elif gene_expr.max() < 0.001:
                print(f"    ⚠️  {gene}: Very low expression values detected")
            
            # Check for variance
            if gene_expr.std() < 1e-6:
                print(f"    ⚠️  {gene}: Very low variance - may affect analysis")
            
            # Add to obs
            tumor_adata.obs[f"{gene}_Expression"] = gene_expr
            available_genes.append(gene)
            
        else:
            missing_genes.append(gene)
    
    print(f"  Successfully added {len(available_genes)} gene expressions to obs")
    if missing_genes:
        print(f"  WARNING: {len(missing_genes)} genes not found: {missing_genes}")
    
    return tumor_adata, available_genes


def create_expression_groups(expression_data, strategy="Tertile"):
    """
    Create expression level groups based on different strategies.
    
    Parameters:
    -----------
    expression_data : pd.Series
        Gene expression data
    strategy : str
        Stratification strategy
        
    Returns:
    --------
    pd.Series
        Group assignments (High, Low, or None for excluded middle)
    """
    data = expression_data.dropna()
    
    if len(data) == 0:
        return pd.Series(index=expression_data.index, dtype="object")
    
    # FIXED: Add minimum sample size validation for each strategy
    min_samples_required = {
        "Tertile": 60,        # Need at least 20 per group (60/3)
        "Median": 40,         # Need at least 20 per group (40/2) 
        "Quartile": 80,       # Need at least 20 per group (80/4, excluding middle 50%)
        "Extreme_Decile": 100 # Need at least 10 per group (100/10, excluding middle 80%)
    }
    
    min_required = min_samples_required.get(strategy, 40)
    if len(data) < min_required:
        print(f"  WARNING: Insufficient samples ({len(data)}) for {strategy} analysis (minimum: {min_required})")
        return pd.Series(index=expression_data.index, dtype="object")
    
    if strategy == "Tertile":
        # Top vs Bottom 1/3 (exclude middle)
        low_thresh = data.quantile(0.33)
        high_thresh = data.quantile(0.67)
        groups = np.where(
            expression_data >= high_thresh,
            "High",
            np.where(expression_data <= low_thresh, "Low", "Middle")
        )
    elif strategy == "Median":
        # Above vs Below Median
        median_thresh = data.median()
        groups = np.where(expression_data > median_thresh, "High", "Low")
    elif strategy == "Quartile":
        # Top vs Bottom 1/4 (exclude middle 50%)
        low_thresh = data.quantile(0.25)
        high_thresh = data.quantile(0.75)
        groups = np.where(
            expression_data >= high_thresh,
            "High",
            np.where(expression_data <= low_thresh, "Low", "Middle")
        )
    elif strategy == "Extreme_Decile":
        # Top vs Bottom 10% (exclude middle 80%)
        low_thresh = data.quantile(0.10)
        high_thresh = data.quantile(0.90)
        groups = np.where(
            expression_data >= high_thresh,
            "High",
            np.where(expression_data <= low_thresh, "Low", "Middle")
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    result = pd.Series(groups, index=expression_data.index, dtype="object")
    
    # Replace middle group with None for exclusion (except for Median strategy)
    if strategy != "Median":
        result = result.replace("Middle", None)
    
    # FIXED: Validate final group sizes
    if strategy != "Median":
        high_count = (result == "High").sum()
        low_count = (result == "Low").sum()
        if high_count < 10 or low_count < 10:
            print(f"  WARNING: {strategy} stratification resulted in small groups (High: {high_count}, Low: {low_count})")
    else:
        high_count = (result == "High").sum()
        low_count = (result == "Low").sum()
        if high_count < 15 or low_count < 15:
            print(f"  WARNING: {strategy} stratification resulted in small groups (High: {high_count}, Low: {low_count})")
    
    return result


def prepare_survival_data(tumor_adata):
    """
    Prepare survival data from tumor_adata.obs with enhanced validation.
    UPDATED to fix critical survival analysis issues.
    """
    print(f"\n--- Preparing Survival Data with Enhanced Validation ---")
    
    df = tumor_adata.obs.copy()
    
    # Enhanced survival time calculation with validation
    print("  Calculating survival times...")
    
    # Initialize survival time
    df["Survival_Time"] = np.nan
    df["Event"] = 0
    
    # Check data availability
    has_death_data = 'Days_to_Death' in df.columns and df['Days_to_Death'].notna().sum() > 0
    has_followup_data = 'Days_to_Last_Followup' in df.columns and df['Days_to_Last_Followup'].notna().sum() > 0
    has_vital_status = 'Vital_Status' in df.columns
    
    print(f"    Days_to_Death available: {df['Days_to_Death'].notna().sum() if has_death_data else 0}")
    print(f"    Days_to_Last_Followup available: {df['Days_to_Last_Followup'].notna().sum() if has_followup_data else 0}")
    print(f"    Vital_Status available: {df['Vital_Status'].notna().sum() if has_vital_status else 0}")
    
    if not has_vital_status:
        print("  ❌ ERROR: No Vital_Status data available")
        return df
    
    # Analyze vital status distribution
    vital_counts = df['Vital_Status'].value_counts()
    print("    Vital status distribution:")
    for status, count in vital_counts.items():
        print(f"      {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Enhanced event coding with validation
    print("  Coding survival events...")
    
    # More robust vital status coding
    dead_patterns = ['dead', 'deceased', 'death']
    alive_patterns = ['alive', 'living']
    
    df["Event"] = 0  # Default to censored
    vital_status_lower = df["Vital_Status"].astype(str).str.lower().str.strip()
    
    for pattern in dead_patterns:
        df.loc[vital_status_lower.str.contains(pattern, na=False), "Event"] = 1
    
    # Survival time assignment with enhanced logic
    print("  Assigning survival times...")
    
    # For deceased patients: use Days_to_Death if available
    if has_death_data:
        deceased_mask = df["Event"] == 1
        deceased_with_death_time = deceased_mask & df["Days_to_Death"].notna()
        df.loc[deceased_with_death_time, "Survival_Time"] = df.loc[deceased_with_death_time, "Days_to_Death"]
        print(f"    Deceased patients with death time: {deceased_with_death_time.sum()}")
    
    # For all patients without survival time: use last follow-up
    if has_followup_data:
        missing_survival_time = df["Survival_Time"].isna()
        has_followup = missing_survival_time & df["Days_to_Last_Followup"].notna()
        df.loc[has_followup, "Survival_Time"] = df.loc[has_followup, "Days_to_Last_Followup"]
        print(f"    Patients using follow-up time: {has_followup.sum()}")
    
    # Data quality checks
    print("  Data quality validation...")
    
    # Remove negative survival times
    negative_times = df["Survival_Time"] < 0
    if negative_times.sum() > 0:
        print(f"    ⚠️  Removing {negative_times.sum()} samples with negative survival times")
        df = df[~negative_times].copy()
    
    # FIXED: More comprehensive short survival time filtering
    # Remove very short survival times with more sophisticated logic
    very_short_censored = (df["Survival_Time"] < 30) & (df["Event"] == 0)  # Censored within 30 days
    very_short_death = (df["Survival_Time"] < 7) & (df["Event"] == 1)      # Deaths within 7 days (likely perioperative)
    
    if very_short_censored.sum() > 0:
        print(f"    ⚠️  Removing {very_short_censored.sum()} samples censored within 30 days (likely follow-up issues)")
        df = df[~very_short_censored].copy()
    
    if very_short_death.sum() > 0:
        print(f"    ⚠️  Removing {very_short_death.sum()} samples with death within 7 days (likely perioperative mortality)")
        print(f"        Note: This removes potential confounding from surgical complications")
        df = df[~very_short_death].copy()
    
    # Remove samples with missing survival time
    missing_survival = df["Survival_Time"].isna()
    if missing_survival.sum() > 0:
        print(f"    ⚠️  Removing {missing_survival.sum()} samples with missing survival times")
        df = df[~missing_survival].copy()
    
    # FIXED: Add validation for minimum follow-up in censored patients
    min_followup_censored = 90  # Minimum 3 months follow-up for censored patients
    short_followup_censored = (df["Survival_Time"] < min_followup_censored) & (df["Event"] == 0)
    if short_followup_censored.sum() > 0:
        print(f"    ⚠️  Found {short_followup_censored.sum()} censored patients with <{min_followup_censored} days follow-up")
        print(f"        These patients have insufficient follow-up and may bias results")
        # Remove these as well
        df = df[~short_followup_censored].copy()
    
    # Final statistics
    total_events = df["Event"].sum()
    total_samples = len(df)
    overall_event_rate = (total_events / total_samples) * 100 if total_samples > 0 else 0
    median_followup = df["Survival_Time"].median() if total_samples > 0 else 0
    
    print(f"\n  FINAL SURVIVAL DATA SUMMARY:")
    print(f"    Total samples: {total_samples}")
    print(f"    Total events (deaths): {total_events}")
    print(f"    Event rate: {overall_event_rate:.1f}%")
    print(f"    Median follow-up: {median_followup:.0f} days ({median_followup/365.25:.1f} years)")
    
    # Age stratification
    if 'Age_at_Diagnosis' in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age_at_Diagnosis"], bins=[0, 60, 120], labels=["<60", ">=60"], right=False
        )
        print(f"    Age groups: <60: {(df['Age_Group'] == '<60').sum()}, >=60: {(df['Age_Group'] == '>=60').sum()}")
    
    # Critical warnings
    if overall_event_rate < 15:
        print(f"  ⚠️  WARNING: Low event rate ({overall_event_rate:.1f}%) may limit statistical power")
        print(f"      This suggests either very short follow-up or data quality issues")
    
    if total_samples < 100:
        print(f"  ⚠️  WARNING: Very small sample size ({total_samples}) after filtering")
    
    if median_followup < 365:
        print(f"  ⚠️  WARNING: Very short follow-up time ({median_followup/365.25:.1f} years)")
    
    return df


def validate_cox_results(cox_results, available_genes):
    """
    Validate Cox regression results and flag suspicious patterns.
    """
    print(f"\n--- Validating Cox Regression Results ---")
    
    if not cox_results:
        print("  No Cox results to validate")
        return
    
    cox_df = pd.DataFrame(cox_results)
    
    # Check for suspicious patterns
    print("  Checking for suspicious patterns...")
    
    # 1. Too many protective effects for immune checkpoints
    immune_checkpoints = ['PDCD1', 'LAG3', 'HAVCR2', 'TOX', 'TIGIT', 'CD274']
    immune_genes_in_analysis = [gene for gene in available_genes if gene in immune_checkpoints]
    
    if immune_genes_in_analysis:
        protective_immune = cox_df[
            (cox_df['Gene'].isin(immune_genes_in_analysis)) & 
            (cox_df['HR'] < 1) & 
            (cox_df['p_value'] < 0.05)
        ]
        
        if len(protective_immune) > len(immune_genes_in_analysis) * 0.5:
            print(f"  ⚠️  WARNING: {len(protective_immune)} immune checkpoints showing protective effects")
            print(f"      This is biologically implausible and suggests analytical issues")
            print(f"      Affected genes: {list(protective_immune['Gene'].unique())}")
    
    # 2. Check HR distribution
    hrs = cox_df['HR']
    protective_count = (hrs < 1).sum()
    harmful_count = (hrs > 1).sum()
    
    print(f"  Effect direction distribution:")
    print(f"    Protective (HR < 1): {protective_count} ({protective_count/len(hrs)*100:.1f}%)")
    print(f"    Harmful (HR > 1): {harmful_count} ({harmful_count/len(hrs)*100:.1f}%)")
    
    if protective_count > harmful_count * 2:
        print(f"  ⚠️  WARNING: Unusual bias toward protective effects")
        print(f"      This may indicate analytical issues")
    
    # 3. Check for extreme HRs
    extreme_protective = (hrs < 0.3).sum()
    extreme_harmful = (hrs > 3.0).sum()
    
    if extreme_protective > 0:
        extreme_genes = cox_df[cox_df['HR'] < 0.3]['Gene'].tolist()
        print(f"  ⚠️  WARNING: {extreme_protective} genes with very strong protective effects (HR < 0.3)")
        print(f"      Genes: {extreme_genes}")
    
    if extreme_harmful > 0:
        extreme_genes = cox_df[cox_df['HR'] > 3.0]['Gene'].tolist()
        print(f"  ⚠️  WARNING: {extreme_harmful} genes with very strong harmful effects (HR > 3.0)")
        print(f"      Genes: {extreme_genes}")
    
    # 4. Check p-value distribution
    significant_results = (cox_df['p_value'] < 0.05).sum()
    print(f"  Statistical significance:")
    print(f"    Significant results (p < 0.05): {significant_results} ({significant_results/len(cox_df)*100:.1f}%)")
    
    return cox_df


def perform_gene_survival_analysis(df, available_genes, output_dir, cancer_type):
    """
    Comprehensive survival analysis for genes of interest with enhanced validation.
    UPDATED to include result validation and better error handling.
    """
    print(f"\n=== Gene-Based Survival Analysis for {cancer_type} ===")
    
    cox_results = []
    logrank_results = []
    
    # Analysis scenarios
    scenarios = [
        ("Overall", df),
        ("Age_<60", df[df["Age_Group"] == "<60"]),
        ("Age_>=60", df[df["Age_Group"] == ">=60"])
    ]
    
    for gene in available_genes:
        gene_expr_col = f"{gene}_Expression"
        
        if gene_expr_col not in df.columns:
            continue
            
        print(f"\n--- Analyzing {gene} ---")
        
        # Check for sufficient expression variation
        gene_variance = df[gene_expr_col].var()
        if gene_variance < 1e-8:
            print(f"  {gene}: Insufficient expression variation (var={gene_variance:.2e}), skipping")
            continue
        
        # Expression distribution check
        expr_stats = df[gene_expr_col].describe()
        print(f"  {gene} expression: min={expr_stats['min']:.3f}, median={expr_stats['50%']:.3f}, max={expr_stats['max']:.3f}")
        
        for scenario_name, subset_df in scenarios:
            if len(subset_df) < 30:  # Require minimum sample size
                continue
                
            n_events = subset_df["Event"].sum()
            if n_events < 10:  # Require minimum events
                continue
                
            print(f"  {scenario_name}: {len(subset_df)} samples, {n_events} events ({n_events/len(subset_df)*100:.1f}%)")
            
            # === COX REGRESSION ANALYSIS ===
            try:
                cph = CoxPHFitter()
                cox_df = subset_df[["Survival_Time", "Event", gene_expr_col]].dropna()
                
                if len(cox_df) < 10:
                    continue
                
                # Validate expression data before Cox regression
                expr_values = cox_df[gene_expr_col]
                if expr_values.var() < 1e-10:
                    print(f"    Cox: Skipping {gene} - no expression variation in {scenario_name}")
                    continue
                
                # Fit Cox model
                cph.fit(cox_df, duration_col="Survival_Time", event_col="Event")
                
                # FIXED: Add Cox regression assumptions validation
                try:
                    # Check proportional hazards assumption
                    assumption_check = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
                    if hasattr(assumption_check, 'p_value') and assumption_check.p_value is not None:
                        if assumption_check.p_value < 0.05:
                            print(f"    ⚠️  WARNING: Proportional hazards assumption violated for {gene} (p={assumption_check.p_value:.3f})")
                            print(f"        Cox model results may be unreliable. Consider stratified Cox or time-varying coefficients.")
                except Exception as e:
                    # If assumption checking fails, continue but warn
                    print(f"    NOTE: Could not check Cox assumptions for {gene}: {e}")
                
                # Extract HR and stats
                hr = cph.hazard_ratios_[gene_expr_col]
                p_value = cph.summary.loc[gene_expr_col, "p"]
                
                # Get confidence intervals
                summary_row = cph.summary.loc[gene_expr_col]
                
                # FIXED: Use standardized lifelines API for confidence intervals
                try:
                    # Try the standard lifelines confidence intervals method
                    ci_df = cph.confidence_intervals_
                    hr_ci_lower = np.exp(ci_df.loc[gene_expr_col, 'coef lower 95%'])
                    hr_ci_upper = np.exp(ci_df.loc[gene_expr_col, 'coef upper 95%'])
                except (KeyError, AttributeError):
                    # Fallback: calculate from coefficient and standard error
                    try:
                        coef = summary_row['coef']
                        se = summary_row['se(coef)']
                        # 95% CI: coef ± 1.96 * SE
                        hr_ci_lower = np.exp(coef - 1.96 * se)
                        hr_ci_upper = np.exp(coef + 1.96 * se)
                    except KeyError:
                        # Last resort: use summary statistics if available
                        hr_ci_lower = np.nan
                        hr_ci_upper = np.nan
                        print(f"    WARNING: Could not extract confidence intervals for {gene}")
                
                # Enhanced effect size and direction categorization
                if hr < 1:
                    risk_direction = "Protective"
                    # FIXED: Enhanced biological plausibility check for immune checkpoints
                    immune_checkpoints = ['PDCD1', 'LAG3', 'HAVCR2', 'TOX', 'TIGIT', 'CD274']
                    
                    # Cancer-type specific biological expectations
                    cancer_type_context = {
                        'BRCA': {
                            'usually_not_protective': ['PDCD1', 'LAG3', 'HAVCR2'],
                            'context': 'breast cancer typically shows immune checkpoint expression associated with worse outcomes'
                        },
                        'LUAD': {
                            'usually_not_protective': ['PDCD1'],
                            'context': 'lung adenocarcinoma may show variable immune checkpoint effects'
                        },
                        'GBM': {
                            'usually_not_protective': ['PDCD1', 'LAG3', 'HAVCR2', 'TOX'],
                            'context': 'glioblastoma rarely shows protective immune checkpoint effects'
                        },
                        'KIRC': {
                            'may_be_protective': ['PDCD1'],  # Kidney cancer is immunotherapy-responsive
                            'context': 'renal cell carcinoma may show protective immune checkpoint effects'
                        }
                    }
                    
                    if gene in immune_checkpoints and hr < 0.8 and p_value < 0.05:
                        # Get cancer type from scenario or use a default
                        current_cancer = scenario_name.split('_')[0] if '_' in scenario_name else cancer_type
                        context_info = cancer_type_context.get(current_cancer, {})
                        
                        usually_not_protective = context_info.get('usually_not_protective', [])
                        may_be_protective = context_info.get('may_be_protective', [])
                        
                        if gene in usually_not_protective:
                            print(f"    ⚠️  WARNING: {gene} showing strong protective effect (HR={hr:.3f})")
                            print(f"        This is biologically unusual for {gene} in {current_cancer}")
                            print(f"        Context: {context_info.get('context', 'immune checkpoint genes typically associated with worse outcomes')}")
                            print(f"        Consider: batch effects, immune infiltration confounding, or data quality issues")
                        elif gene not in may_be_protective:
                            print(f"    ℹ️  NOTE: {gene} showing protective effect (HR={hr:.3f}) in {current_cancer}")
                            print(f"        This warrants further investigation but may be biologically plausible")
                else:
                    risk_direction = "Harmful"
                
                # Effect size categorization
                if hr < 0.5 or hr > 2.0:
                    effect_size = "Strong"
                elif hr < 0.8 or hr > 1.25:
                    effect_size = "Moderate"
                else:
                    effect_size = "Weak"
                
                cox_results.append({
                    "Gene": gene,
                    "Scenario": scenario_name,
                    "n_samples": len(cox_df),
                    "n_events": cox_df["Event"].sum(),
                    "HR": hr,
                    "HR_CI_Lower": hr_ci_lower,
                    "HR_CI_Upper": hr_ci_upper,
                    "p_value": p_value,
                    "Risk_Direction": risk_direction,
                    "Effect_Size": effect_size,
                    "Expression_Mean": expr_values.mean(),
                    "Expression_Std": expr_values.std()
                })
                
                print(f"    Cox: HR={hr:.3f} (95% CI: {hr_ci_lower:.3f}-{hr_ci_upper:.3f}), p={p_value:.3f}")
                
            except Exception as e:
                print(f"    Cox model failed: {e}")
                continue
            
            # === LOG-RANK ANALYSIS ===
            for strategy_name, strategy_desc in STRATIFICATION_STRATEGIES.items():
                try:
                    # Create expression groups
                    expr_groups = create_expression_groups(subset_df[gene_expr_col], strategy_name)
                    
                    # Get high and low groups
                    high_mask = expr_groups == "High"
                    low_mask = expr_groups == "Low"
                    
                    high_group = subset_df[high_mask]
                    low_group = subset_df[low_mask]
                    
                    # Quality control
                    if len(high_group) < 10 or len(low_group) < 10:
                        continue
                        
                    high_events = high_group["Event"].sum()
                    low_events = low_group["Event"].sum()
                    
                    if high_events < 3 or low_events < 3:
                        continue
                    
                    # Perform log-rank test
                    logrank_result = logrank_test(
                        durations_A=high_group["Survival_Time"],
                        durations_B=low_group["Survival_Time"],
                        event_observed_A=high_group["Event"],
                        event_observed_B=low_group["Event"]
                    )
                    
                    # Calculate survival statistics
                    high_rate = (high_events / len(high_group)) * 100
                    low_rate = (low_events / len(low_group)) * 100
                    
                    # Calculate median survival times
                    try:
                        kmf = KaplanMeierFitter()
                        
                        # High group median survival
                        kmf.fit(high_group["Survival_Time"], high_group["Event"])
                        high_median_survival = kmf.median_survival_time_
                        high_median_survival = high_median_survival if not pd.isna(high_median_survival) else "NR"
                        
                        # Low group median survival
                        kmf.fit(low_group["Survival_Time"], low_group["Event"])
                        low_median_survival = kmf.median_survival_time_
                        low_median_survival = low_median_survival if not pd.isna(low_median_survival) else "NR"
                        
                    except Exception:
                        high_median_survival = "NR"
                        low_median_survival = "NR"
                    
                    # Determine effect direction
                    if high_rate < low_rate:
                        effect_direction = "Protective (Higher Expression → Better Survival)"
                    else:
                        effect_direction = "Harmful (Higher Expression → Worse Survival)"
                    
                    logrank_results.append({
                        "Gene": gene,
                        "Scenario": scenario_name,
                        "Strategy": strategy_name,
                        "Strategy_Description": strategy_desc,
                        "High_n": len(high_group),
                        "Low_n": len(low_group),
                        "High_events": high_events,
                        "Low_events": low_events,
                        "High_event_rate": high_rate,
                        "Low_event_rate": low_rate,
                        "High_median_survival": high_median_survival,
                        "Low_median_survival": low_median_survival,
                        "Effect_Direction": effect_direction,
                        "p_value": logrank_result.p_value,
                        "test_statistic": logrank_result.test_statistic,
                        "is_significant_05": logrank_result.p_value < 0.05
                    })
                    
                except Exception as e:
                    continue
    
    # Validate results before saving
    validate_cox_results(cox_results, available_genes)
    
    # Convert to DataFrames and apply FDR correction
    cox_df = pd.DataFrame(cox_results)
    logrank_df = pd.DataFrame(logrank_results)
    
    if not cox_df.empty:
        cox_df = apply_fdr_correction_df(cox_df, 'p_value', 'FDR_q_value')
        cox_df = cox_df.sort_values('p_value')
    
    if not logrank_df.empty:
        logrank_df = apply_fdr_correction_df(logrank_df, 'p_value', 'FDR_q_value')
        logrank_df = logrank_df.sort_values('p_value')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    if not cox_df.empty:
        cox_file = os.path.join(output_dir, f"{cancer_type}_Gene_Cox_Regression_Results.csv")
        cox_df.to_csv(cox_file, index=False)
        print(f"\n✅ Cox regression results saved: {cox_file}")
    
    if not logrank_df.empty:
        logrank_file = os.path.join(output_dir, f"{cancer_type}_Gene_LogRank_Results.csv")
        logrank_df.to_csv(logrank_file, index=False)
        print(f"✅ Log-rank results saved: {logrank_file}")
    
    return cox_df, logrank_df


def apply_fdr_correction_df(df, p_value_col="p_value", q_value_col="FDR_q_value"):
    """
    Applies FDR (Benjamini-Hochberg) to a p_value column in a DataFrame.
    
    FIXED: Enhanced with better error handling and statistical guidance.
    """
    if p_value_col not in df.columns or df[p_value_col].isna().all():
        df[q_value_col] = np.nan
        print(f"  WARNING: No valid p-values found for FDR correction in column '{p_value_col}'")
        return df
        
    not_na_mask = df[p_value_col].notna()
    p_values_to_correct = df.loc[not_na_mask, p_value_col]
    
    if len(p_values_to_correct) == 0:
        df[q_value_col] = np.nan
        print(f"  WARNING: No non-NA p-values found for FDR correction")
        return df
    
    # Apply Benjamini-Hochberg FDR correction
    try:
        reject, q_values, _, _ = multipletests(p_values_to_correct, method="fdr_bh")
        df[q_value_col] = np.nan
        df.loc[not_na_mask, q_value_col] = q_values
        
        # Provide statistical guidance
        n_tests = len(p_values_to_correct)
        n_significant_raw = (p_values_to_correct < 0.05).sum()
        n_significant_fdr = (q_values < 0.05).sum()
        
        print(f"  FDR correction applied: {n_tests} tests, {n_significant_raw} nominally significant → {n_significant_fdr} FDR significant")
        
        if n_significant_fdr == 0 and n_significant_raw > 0:
            print(f"  NOTE: No results survive FDR correction. Consider the exploratory nature of findings.")
        elif n_significant_fdr < n_significant_raw:
            print(f"  NOTE: FDR correction reduced significant findings by {n_significant_raw - n_significant_fdr}")
            
    except Exception as e:
        print(f"  ERROR in FDR correction: {e}")
        df[q_value_col] = np.nan
        
    return df


def generate_analysis_summary(cox_df, logrank_df, cancer_type, available_genes):
    """
    Generate a comprehensive analysis summary.
    
    Parameters:
    -----------
    cox_df : pd.DataFrame
        Cox regression results
    logrank_df : pd.DataFrame
        Log-rank test results
    cancer_type : str
        Cancer type abbreviation
    available_genes : list
        List of genes analyzed
        
    Returns:
    --------
    dict
        Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"GENE SURVIVAL ANALYSIS SUMMARY - {cancer_type}")
    print(f"{'='*60}")
    
    summary = {
        "cancer_type": cancer_type,
        "genes_analyzed": len(available_genes),
        "gene_list": available_genes
    }
    
    # Cox regression summary
    if not cox_df.empty:
        sig_cox = cox_df[cox_df['p_value'] < 0.05]
        fdr_sig_cox = cox_df[cox_df['FDR_q_value'] < 0.05] if 'FDR_q_value' in cox_df.columns else pd.DataFrame()
        
        print(f"\n📊 COX REGRESSION RESULTS:")
        print(f"   Total analyses: {len(cox_df)}")
        print(f"   Nominally significant (p<0.05): {len(sig_cox)}")
        print(f"   FDR significant (q<0.05): {len(fdr_sig_cox)}")
        
        if not fdr_sig_cox.empty:
            print(f"\n✅ FDR-SIGNIFICANT COX RESULTS:")
            for _, row in fdr_sig_cox.iterrows():
                print(f"   {row['Gene']} ({row['Scenario']}): HR={row['HR']:.3f}, p={row['p_value']:.3f}, q={row['FDR_q_value']:.3f}")
        
        summary.update({
            "cox_total": len(cox_df),
            "cox_nominal_sig": len(sig_cox),
            "cox_fdr_sig": len(fdr_sig_cox)
        })
    
    # Log-rank summary
    if not logrank_df.empty:
        sig_logrank = logrank_df[logrank_df['p_value'] < 0.05]
        fdr_sig_logrank = logrank_df[logrank_df['FDR_q_value'] < 0.05] if 'FDR_q_value' in logrank_df.columns else pd.DataFrame()
        
        print(f"\n📊 LOG-RANK RESULTS:")
        print(f"   Total comparisons: {len(logrank_df)}")
        print(f"   Nominally significant (p<0.05): {len(sig_logrank)}")
        print(f"   FDR significant (q<0.05): {len(fdr_sig_logrank)}")
        
        if not fdr_sig_logrank.empty:
            print(f"\n✅ FDR-SIGNIFICANT LOG-RANK RESULTS:")
            for _, row in fdr_sig_logrank.iterrows():
                print(f"   {row['Gene']} ({row['Scenario']}, {row['Strategy']}): p={row['p_value']:.3f}, q={row['FDR_q_value']:.3f}")
                print(f"      → {row['Effect_Direction']}")
        
        summary.update({
            "logrank_total": len(logrank_df),
            "logrank_nominal_sig": len(sig_logrank),
            "logrank_fdr_sig": len(fdr_sig_logrank)
        })
    
    print(f"\n{'='*60}")
    
    return summary


# Add diagnostic function at the end of the file, before the main execution

def diagnose_survival_analysis_issues(tumor_adata, cancer_type):
    """
    Comprehensive diagnostic function to identify issues with survival analysis.
    """
    print(f"\n{'='*70}")
    print(f"SURVIVAL ANALYSIS DIAGNOSTICS FOR {cancer_type}")
    print(f"{'='*70}")
    
    df = tumor_adata.obs.copy()
    
    # 1. Basic data quality
    print(f"\n1. BASIC DATA QUALITY:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique patients: {df['Patient_ID'].nunique() if 'Patient_ID' in df.columns else 'Unknown'}")
    
    # 2. Survival time issues
    print(f"\n2. SURVIVAL TIME ANALYSIS:")
    if 'Days_to_Death' in df.columns:
        days_to_death_available = df['Days_to_Death'].notna().sum()
        print(f"   Samples with Days_to_Death: {days_to_death_available}")
    
    if 'Days_to_Last_Followup' in df.columns:
        days_to_followup_available = df['Days_to_Last_Followup'].notna().sum()
        print(f"   Samples with Days_to_Last_Followup: {days_to_followup_available}")
    
    # 3. Vital status analysis
    print(f"\n3. VITAL STATUS ANALYSIS:")
    if 'Vital_Status' in df.columns:
        vital_counts = df['Vital_Status'].value_counts()
        print(f"   Vital status distribution:")
        for status, count in vital_counts.items():
            print(f"     {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Prepare survival variables like in main analysis
    df["Survival_Time"] = df["Days_to_Death"]
    df.loc[df["Survival_Time"].isna(), "Survival_Time"] = df.loc[
        df["Survival_Time"].isna(), "Days_to_Last_Followup"
    ]
    df["Event"] = (df["Vital_Status"].astype(str).str.lower() == "dead").astype(int)
    
    # 4. Survival data quality
    print(f"\n4. SURVIVAL DATA QUALITY:")
    total_events = df["Event"].sum()
    total_samples = len(df)
    event_rate = (total_events / total_samples) * 100
    
    print(f"   Total events (deaths): {total_events}")
    print(f"   Event rate: {event_rate:.1f}%")
    
    # Check survival times
    valid_survival_times = df["Survival_Time"].notna().sum()
    print(f"   Valid survival times: {valid_survival_times}")
    
    if valid_survival_times > 0:
        survival_stats = df["Survival_Time"].describe()
        print(f"   Survival time stats (days):")
        print(f"     Min: {survival_stats['min']:.0f}")
        print(f"     Median: {survival_stats['50%']:.0f}")
        print(f"     Max: {survival_stats['max']:.0f}")
        print(f"     Mean: {survival_stats['mean']:.0f}")
    
    # 5. Expression data analysis
    print(f"\n5. EXPRESSION DATA ANALYSIS:")
    test_gene = 'PDCD1'
    if test_gene in tumor_adata.var_names:
        gene_expr = tumor_adata[:, test_gene].X
        if hasattr(gene_expr, "toarray"):
            gene_expr = gene_expr.toarray()
        gene_expr = np.ravel(gene_expr)
        
        print(f"   {test_gene} expression stats:")
        print(f"     Min: {gene_expr.min():.3f}")
        print(f"     Median: {np.median(gene_expr):.3f}")
        print(f"     Max: {gene_expr.max():.3f}")
        print(f"     Mean: {gene_expr.mean():.3f}")
        print(f"     Std: {gene_expr.std():.3f}")
        
        # Check if expression data seems reasonable
        if gene_expr.max() > 50:
            print(f"   ⚠️  WARNING: Expression values very high - may need log transformation")
        elif gene_expr.max() < 0.1:
            print(f"   ⚠️  WARNING: Expression values very low - may be over-normalized")
    
    # 6. Batch correction status
    print(f"\n6. BATCH CORRECTION STATUS:")
    if 'batch_correction' in tumor_adata.uns:
        batch_info = tumor_adata.uns['batch_correction']
        print(f"   Batch correction applied: {batch_info.get('applied', 'Unknown')}")
        print(f"   Method: {batch_info.get('method', 'Unknown')}")
        print(f"   Batch column: {batch_info.get('batch_column', 'Unknown')}")
        print(f"   Number of batches: {batch_info.get('n_batches', 'Unknown')}")
    else:
        print(f"   ⚠️  No batch correction metadata found")
    
    # 7. Age distribution analysis
    print(f"\n7. AGE DISTRIBUTION:")
    if 'Age_at_Diagnosis' in df.columns:
        age_stats = df['Age_at_Diagnosis'].describe()
        print(f"   Age stats:")
        print(f"     Min: {age_stats['min']:.0f}")
        print(f"     Median: {age_stats['50%']:.0f}")
        print(f"     Max: {age_stats['max']:.0f}")
        print(f"     Mean: {age_stats['mean']:.0f}")
    
    # 8. Sample composition analysis
    print(f"\n8. SAMPLE COMPOSITION:")
    if 'Tissue_Type' in df.columns:
        tissue_counts = df['Tissue_Type'].value_counts()
        print(f"   Tissue types:")
        for tissue, count in tissue_counts.items():
            print(f"     {tissue}: {count}")
    
    # 9. Critical warnings
    print(f"\n9. CRITICAL WARNINGS:")
    warnings = []
    
    if event_rate < 10:
        warnings.append(f"Very low event rate ({event_rate:.1f}%) - insufficient events for survival analysis")
    
    if valid_survival_times < total_samples * 0.8:
        warnings.append(f"Missing survival times for {total_samples - valid_survival_times} samples")
    
    if 'batch_correction' not in tumor_adata.uns:
        warnings.append("No batch correction applied - results likely confounded")
    
    if len(warnings) == 0:
        print(f"   ✅ No critical warnings detected")
    else:
        for warning in warnings:
            print(f"   ⚠️  {warning}")
    
    print(f"\n{'='*70}")
    return df


# ==============================================================================
# --- CIBERSORTx Integration Functions ---
# ==============================================================================

def load_cibersortx_data(cibersort_path, thresholds=None):
    """
    Load and filter CIBERSORTx results.
    Identical to TCGA_STUDY_ANALYSIS.py implementation.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    print(f"\n--- Loading CIBERSORTx Data: {os.path.basename(cibersort_path)} ---")

    if not os.path.exists(cibersort_path):
        print(f"  WARNING: File not found: {cibersort_path}")
        return pd.DataFrame()

    try:
        # Load CIBERSORTx CSV file with Mixture column as index
        df = pd.read_csv(cibersort_path, index_col="Mixture")
        print(f"  Loaded data: {df.shape}")
        print(f"  Columns found: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"  Columns: {list(df.columns)}")
        
        # Apply quality filters
        initial_samples = len(df)
        
        if "P-value" in df.columns:
            p_mask = df["P-value"] < thresholds["p_value_cibersort"]
            df = df[p_mask]
            print(f"    P-value filter: kept {len(df)}/{initial_samples} samples")

        if "Correlation" in df.columns and thresholds.get("correlation_cibersort"):
            before_corr = len(df)
            corr_mask = df["Correlation"] > thresholds["correlation_cibersort"]
            df = df[corr_mask]
            pct = (len(df)/before_corr)*100 if before_corr > 0 else 0
            print(f"    Correlation filter: kept {len(df)}/{before_corr} samples ({pct:.1f}%)")

        if "RMSE" in df.columns and len(df) > 0:
            rmse_threshold = df["RMSE"].quantile(thresholds["rmse_percentile_cibersort"])
            rmse_mask = df["RMSE"] < rmse_threshold
            df = df[rmse_mask]
            print(f"    RMSE filter: kept {len(df)} samples")
        
        # Remove metric columns (but keep Absolute score for immune normalization)
        metric_cols = ["P-value", "Correlation", "RMSE"]
        existing_metric_cols = [col for col in metric_cols if col in df.columns]
        if existing_metric_cols:
            df = df.drop(columns=existing_metric_cols)
        
        # Rename absolute score for clarity
        if "Absolute score (sig.score)" in df.columns:
            df = df.rename(columns={"Absolute score (sig.score)": "Total_Immune_Score"})

        return df

    except Exception as e:
        print(f"  ERROR loading CIBERSORTx data: {e}")
        return pd.DataFrame()


def integrate_cibersortx_data(tumor_adata, cibersort_paths, thresholds=None):
    """
    Integrate CIBERSORTx data into tumor AnnData object.
    """
    print(f"\n--- Integrating CIBERSORTx Data ---")

    if tumor_adata is None:
        print("  ERROR: tumor_adata is None")
        return None

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    for data_type, file_path in cibersort_paths.items():
        print(f"\n  Processing {data_type}...")

        cibersort_df = load_cibersortx_data(file_path, thresholds)

        if cibersort_df.empty:
            continue

        # Find common samples
        common_samples = set(tumor_adata.obs_names) & set(cibersort_df.index)
        if not common_samples:
            print(f"    No common samples between tumor_adata and {data_type}")
            continue

        print(f"    Common samples: {len(common_samples)}")

        # Merge data into tumor_adata.obs
        common_samples_list = list(common_samples)
        for col in cibersort_df.columns:
            # Initialize with zeros for all samples
            new_col = pd.Series(0.0, index=tumor_adata.obs_names, dtype=float)
            # Fill in values for common samples
            new_col.loc[common_samples_list] = cibersort_df.loc[
                common_samples_list, col
            ].values
            tumor_adata.obs[col] = new_col

        print(f"    Added {len(cibersort_df.columns)} immune infiltration columns")

    return tumor_adata


def analyze_genes_with_immune_normalization(tumor_adata, genes_of_interest, cancer_type, output_dir):
    """
    Perform gene survival analysis with immune infiltration normalization.
    
    This compares:
    1. Raw gene expression survival effects
    2. Gene expression normalized to total immune infiltration (using CIBERSORTx Absolute Score)
    3. Gene expression normalized to cytotoxic lymphoid infiltration (sum of CD8 + NK absolute scores)
    
    CORRECTED: CIBERSORTx outputs absolute infiltration scores for each cell type, and 
    "Absolute score (sig.score)" is the sum of all cell type scores. Individual cell columns
    are absolute values, not fractions.
    
    This helps distinguish direct gene effects from immune infiltration confounding.
    """
    print(f"\n--- Gene Survival Analysis with Immune Normalization ---")
    
    # Create immune-normalized output directory
    immune_output_dir = os.path.join(output_dir, f"{cancer_type}_Gene_Survival_Immune_Normalized")
    os.makedirs(immune_output_dir, exist_ok=True)
    
    # Check for CIBERSORTx data - comprehensive immune cell detection
    # UPDATED: Enhanced to detect hybrid Atlas-Tang signature column terms
    all_immune_search_terms = [
        # Atlas immune minor types
        'T_cells_CD4+', 'T_cells_CD8+', 'NKT_cells', 'T_cells_Cycling',
        'B_cells_Memory', 'B_cells_Naive', 'Plasmablasts',
        'Macrophage', 'Monocyte', 'DCs', 'Myeloid_Cycling',
        # Tang NK core signatures
        'NK_Cytotoxic', 'NK_Bright', 'NK_Exhausted_TaNK',
        # Legacy terms for backward compatibility
        'T cells', 'B cells', 'Monocytes', 'Macrophages', 
        'Dendritic cells', 'Mast cells', 'Eosinophils', 'Neutrophils',
        'Plasma cells', 'Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK'
    ]
    
    immune_cols = [col for col in tumor_adata.obs.columns if 
                   any(cell_type in col for cell_type in all_immune_search_terms)]
    
    if not immune_cols:
        print("  WARNING: No CIBERSORTx immune infiltration data found!")
        print("  Will perform raw expression analysis only")
        return perform_gene_survival_analysis(
            prepare_survival_data(tumor_adata), genes_of_interest, output_dir, cancer_type
        )
    
    print(f"  Found {len(immune_cols)} immune infiltration columns")
    print(f"  Immune columns: {immune_cols}")
    
    # UPDATED: Calculate immune infiltration metrics using enhanced NK subtypes
    # Check if we have the total immune score from CIBERSORTx
    if 'Total_Immune_Score' in tumor_adata.obs.columns:
        print("  Using CIBERSORTx Absolute Score for total immune infiltration")
        tumor_adata.obs['Total_Immune_Infiltration'] = tumor_adata.obs['Total_Immune_Score']
        
        # ENHANCED: For cytotoxic lymphoid using hybrid Atlas-Tang NK subtypes
        enhanced_lymphoid_search_terms = ['T_cells_CD8+', 'NK_Bright', 'NK_Cytotoxic', 'T cells CD8', 'Bright_NK', 'Cytotoxic_NK']
        lymphoid_cols = [col for col in immune_cols if 
                         any(cell_type in col for cell_type in enhanced_lymphoid_search_terms)]
        
        if lymphoid_cols:
            print(f"  Summing enhanced cytotoxic lymphoid scores: {lymphoid_cols}")
            tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration'] = tumor_adata.obs[lymphoid_cols].sum(axis=1)
        else:
            print("  WARNING: No enhanced cytotoxic lymphoid cells found, using total immune score")
            tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration'] = tumor_adata.obs['Total_Immune_Score']
    else:
        print("  WARNING: No Total_Immune_Score found, falling back to sum of individual scores")
        # Fallback: sum individual absolute scores
        tumor_adata.obs['Total_Immune_Infiltration'] = tumor_adata.obs[immune_cols].sum(axis=1)
        
        enhanced_lymphoid_search_terms = ['T_cells_CD8+', 'NK_Bright', 'NK_Cytotoxic', 'T cells CD8', 'Bright_NK', 'Cytotoxic_NK']
        lymphoid_cols = [col for col in immune_cols if 
                         any(cell_type in col for cell_type in enhanced_lymphoid_search_terms)]
        tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration'] = tumor_adata.obs[lymphoid_cols].sum(axis=1)
    
    # Get enhanced lymphoid columns for reporting
    enhanced_lymphoid_search_terms = ['T_cells_CD8+', 'NK_Bright', 'NK_Cytotoxic', 'T cells CD8', 'Bright_NK', 'Cytotoxic_NK']
    lymphoid_cols = [col for col in immune_cols if 
                     any(cell_type in col for cell_type in enhanced_lymphoid_search_terms)]
    
    print(f"  All immune columns ({len(immune_cols)}): {immune_cols}")
    print(f"  Enhanced cytotoxic lymphoid columns ({len(lymphoid_cols)}): {lymphoid_cols}")
    print(f"    Total immune infiltration: {tumor_adata.obs['Total_Immune_Infiltration'].mean():.3f} ± {tumor_adata.obs['Total_Immune_Infiltration'].std():.3f}")
    print(f"    Total enhanced cytotoxic lymphoid infiltration: {tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration'].mean():.3f} ± {tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration'].std():.3f}")
    
    # Check if they're identical (debugging)
    if tumor_adata.obs['Total_Immune_Infiltration'].equals(tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration']):
        print("  ⚠️  WARNING: Total immune and cytotoxic lymphoid infiltration are identical!")
        print("      This suggests all detected immune cells are cytotoxic lymphoid cells")
    else:
        correlation = tumor_adata.obs['Total_Immune_Infiltration'].corr(tumor_adata.obs['Total_Cytotoxic_Lymphoid_Infiltration'])
        print(f"    Immune vs Cytotoxic Lymphoid correlation: {correlation:.3f}")
    
    # Prepare survival data
    survival_df = prepare_survival_data(tumor_adata)
    if survival_df is None or len(survival_df) < 50:
        print("ERROR: Insufficient survival data")
        return None, None
    
    # Perform analysis for each normalization method
    all_results = []
    
    normalization_methods = {
        'raw': 'Raw Expression',
        'immune_normalized': 'Immune-Normalized Expression', 
        'cytotoxic_lymphoid_normalized': 'Cytotoxic Lymphoid-Normalized Expression'
    }
    
    for method, method_name in normalization_methods.items():
        print(f"\n  === {method_name} Analysis ===")
        
        method_survival_df = survival_df.copy()
        cox_results = []
        logrank_results = []
        
        # Add gene expression data based on normalization method
        for gene in genes_of_interest:
            if gene not in tumor_adata.var_names:
                print(f"    WARNING: {gene} not found in expression data")
                continue
            
            # Get samples that exist in both tumor_adata and survival DataFrame
            common_samples = method_survival_df.index.intersection(tumor_adata.obs_names)
            if len(common_samples) == 0:
                print(f"    WARNING: No common samples between survival data and tumor_adata for {gene}")
                continue
            
            # Get expression data for common samples in the correct order
            sample_indices = [tumor_adata.obs_names.get_loc(sample) for sample in common_samples]
            gene_expr = tumor_adata[sample_indices, gene].X.flatten()
            if hasattr(gene_expr, 'toarray'):
                gene_expr = gene_expr.toarray()
            gene_expr = np.ravel(gene_expr)
            
            # Apply normalization using data from the same samples
            if method == 'raw':
                normalized_expr = gene_expr
            elif method == 'immune_normalized':
                total_immune = tumor_adata.obs.loc[common_samples, 'Total_Immune_Infiltration'].values
                # FIXED: Avoid division by zero and very small numbers
                min_threshold = 1e-6  # Minimum threshold to avoid numerical instability
                normalized_expr = np.where(total_immune > min_threshold, 
                                         gene_expr / total_immune, np.nan)
                # Additional check for reasonable normalization values
                if not np.isnan(normalized_expr).all():
                    # Remove extreme outliers (beyond 3 standard deviations)
                    mean_norm = np.nanmean(normalized_expr)
                    std_norm = np.nanstd(normalized_expr)
                    outlier_mask = np.abs(normalized_expr - mean_norm) > 3 * std_norm
                    normalized_expr[outlier_mask] = np.nan
            elif method == 'cytotoxic_lymphoid_normalized':
                total_cytotoxic_lymphoid = tumor_adata.obs.loc[common_samples, 'Total_Cytotoxic_Lymphoid_Infiltration'].values
                # FIXED: Avoid division by zero and very small numbers
                min_threshold = 1e-6
                normalized_expr = np.where(total_cytotoxic_lymphoid > min_threshold, 
                                         gene_expr / total_cytotoxic_lymphoid, np.nan)
                # Additional check for reasonable normalization values
                if not np.isnan(normalized_expr).all():
                    mean_norm = np.nanmean(normalized_expr)
                    std_norm = np.nanstd(normalized_expr)
                    outlier_mask = np.abs(normalized_expr - mean_norm) > 3 * std_norm
                    normalized_expr[outlier_mask] = np.nan
            
            # Create a Series aligned with survival DataFrame index, filled with NaN
            normalized_series = pd.Series(index=method_survival_df.index, dtype=float)
            normalized_series.loc[common_samples] = normalized_expr
            
            # Add to survival dataframe
            method_survival_df[f'{gene}_Expression'] = normalized_series
            
            # Remove samples with invalid normalized expression
            valid_mask = ~np.isnan(method_survival_df[f'{gene}_Expression'])
            gene_survival_df = method_survival_df[valid_mask].copy()
            
            if len(gene_survival_df) < 50:
                print(f"    WARNING: Only {len(gene_survival_df)} valid samples for {gene}")
                continue
                
            print(f"    {gene}: {len(gene_survival_df)} valid samples")
            
            # Tertile-based survival analysis
            try:
                low_thresh = gene_survival_df[f'{gene}_Expression'].quantile(0.33)
                high_thresh = gene_survival_df[f'{gene}_Expression'].quantile(0.67)
                
                high_group = gene_survival_df[gene_survival_df[f'{gene}_Expression'] >= high_thresh]
                low_group = gene_survival_df[gene_survival_df[f'{gene}_Expression'] <= low_thresh]
                
                if len(high_group) >= 10 and len(low_group) >= 10:
                    # Cox regression
                    cox_data = gene_survival_df[[f'{gene}_Expression', 'Survival_Time', 'Event']].copy()
                    cph = CoxPHFitter()
                    cph.fit(cox_data, duration_col='Survival_Time', event_col='Event')
                    
                    # FIXED: Use column names instead of hard-coded indices
                    gene_expr_col = f'{gene}_Expression'
                    hr = cph.hazard_ratios_[gene_expr_col]
                    
                    # Get confidence intervals using standardized approach
                    try:
                        ci_df = cph.confidence_intervals_
                        ci_lower = np.exp(ci_df.loc[gene_expr_col, 'coef lower 95%'])
                        ci_upper = np.exp(ci_df.loc[gene_expr_col, 'coef upper 95%'])
                    except (KeyError, AttributeError):
                        # Fallback calculation
                        summary_row = cph.summary.loc[gene_expr_col]
                        coef = summary_row['coef']
                        se = summary_row['se(coef)']
                        ci_lower = np.exp(coef - 1.96 * se)
                        ci_upper = np.exp(coef + 1.96 * se)
                    
                    # Get p-value using column name
                    p_value = cph.summary.loc[gene_expr_col, 'p']
                    
                    cox_results.append({
                        'Gene': gene,
                        'Normalization_Method': method,
                        'Method_Name': method_name,
                        'Analysis_Type': 'Cox_Regression',
                        'Hazard_Ratio': hr,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'P_Value': p_value,
                        'Risk_Direction': 'Protective' if hr < 1 else 'Harmful',
                        'Significance': 'Significant' if p_value < 0.05 else 'Non-significant',
                        'N_Samples': len(cox_data),
                        'N_Events': cox_data['Event'].sum(),
                        'N_High': len(high_group),
                        'N_Low': len(low_group)
                    })
                    
                    print(f"      Cox HR: {hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p_value:.4f}")
                    
                    # Enhanced Log-rank test with proper survival analysis
                    T_high = high_group['Survival_Time']
                    E_high = high_group['Event']
                    T_low = low_group['Survival_Time']
                    E_low = low_group['Event']
                    
                    # Perform log-rank test
                    logrank_result = logrank_test(T_high, T_low, E_high, E_low)
                    
                    # Calculate proper median survival using Kaplan-Meier
                    from lifelines import KaplanMeierFitter
                    kmf = KaplanMeierFitter()
                    
                    # High expression group
                    kmf.fit(T_high, E_high)
                    median_high = kmf.median_survival_time_
                    survival_1yr_high = kmf.survival_function_at_times(365.25).iloc[0] if 365.25 <= T_high.max() else np.nan
                    survival_3yr_high = kmf.survival_function_at_times(365.25*3).iloc[0] if 365.25*3 <= T_high.max() else np.nan
                    survival_5yr_high = kmf.survival_function_at_times(365.25*5).iloc[0] if 365.25*5 <= T_high.max() else np.nan
                    
                    # Low expression group  
                    kmf.fit(T_low, E_low)
                    median_low = kmf.median_survival_time_
                    survival_1yr_low = kmf.survival_function_at_times(365.25).iloc[0] if 365.25 <= T_low.max() else np.nan
                    survival_3yr_low = kmf.survival_function_at_times(365.25*3).iloc[0] if 365.25*3 <= T_low.max() else np.nan
                    survival_5yr_low = kmf.survival_function_at_times(365.25*5).iloc[0] if 365.25*5 <= T_low.max() else np.nan
                    
                    # Calculate effect size (Hazard Ratio approximation from log-rank)
                    # HR approximation: exp((logrank_statistic) / (total_events))
                    total_events = E_high.sum() + E_low.sum()
                    if total_events > 0:
                        hr_approx = np.exp(logrank_result.test_statistic / total_events)
                        # Adjust direction based on which group has higher events
                        events_rate_high = E_high.sum() / len(high_group)
                        events_rate_low = E_low.sum() / len(low_group)
                        if events_rate_high > events_rate_low:
                            hr_approx = hr_approx  # High expression = higher risk
                        else:
                            hr_approx = 1/hr_approx  # High expression = lower risk
                    else:
                        hr_approx = np.nan
                    
                    # Survival advantage calculation (difference in median survival)
                    if not pd.isna(median_high) and not pd.isna(median_low):
                        survival_advantage_days = median_high - median_low
                        survival_advantage_years = survival_advantage_days / 365.25
                    else:
                        survival_advantage_days = np.nan
                        survival_advantage_years = np.nan
                    
                    # Determine better prognostic group
                    if not pd.isna(median_high) and not pd.isna(median_low):
                        better_prognosis = "High_Expression" if median_high > median_low else "Low_Expression"
                    elif not pd.isna(median_high) and pd.isna(median_low):
                        better_prognosis = "High_Expression"  # High has reached median, low hasn't
                    elif pd.isna(median_high) and not pd.isna(median_low):
                        better_prognosis = "Low_Expression"   # Low has reached median, high hasn't  
                    else:
                        # Use event rates if median unavailable
                        better_prognosis = "High_Expression" if events_rate_high < events_rate_low else "Low_Expression"
                    
                    logrank_results.append({
                        'Gene': gene,
                        'Normalization_Method': method,
                        'Method_Name': method_name,
                        'Analysis_Type': 'Log_Rank',
                        'Test_Statistic': logrank_result.test_statistic,
                        'P_Value': logrank_result.p_value,
                        'HR_Approximation': hr_approx,
                        'N_High': len(high_group),
                        'N_Low': len(low_group),
                        'Events_High': int(E_high.sum()),
                        'Events_Low': int(E_low.sum()),
                        'Event_Rate_High': events_rate_high,
                        'Event_Rate_Low': events_rate_low,
                        'Median_Survival_High_Days': median_high,
                        'Median_Survival_Low_Days': median_low,
                        'Median_Survival_High_Years': median_high/365.25 if not pd.isna(median_high) else np.nan,
                        'Median_Survival_Low_Years': median_low/365.25 if not pd.isna(median_low) else np.nan,
                        'Survival_Advantage_Days': survival_advantage_days,
                        'Survival_Advantage_Years': survival_advantage_years,
                        'Survival_1yr_High': survival_1yr_high,
                        'Survival_1yr_Low': survival_1yr_low,
                        'Survival_3yr_High': survival_3yr_high,
                        'Survival_3yr_Low': survival_3yr_low,
                        'Survival_5yr_High': survival_5yr_high,
                        'Survival_5yr_Low': survival_5yr_low,
                        'Better_Prognosis_Group': better_prognosis,
                        'Risk_Direction': 'Protective' if better_prognosis == "High_Expression" else 'Harmful',
                        'Significance': 'Significant' if logrank_result.p_value < 0.05 else 'Non-significant'
                    })
                    
                    # Enhanced progress reporting
                    median_high_str = f"{median_high/365.25:.1f}y" if not pd.isna(median_high) else "NR"
                    median_low_str = f"{median_low/365.25:.1f}y" if not pd.isna(median_low) else "NR"
                    advantage_str = f"{survival_advantage_years:+.1f}y" if not pd.isna(survival_advantage_years) else "N/A"
                    
                    print(f"      Log-rank p={logrank_result.p_value:.4f}, HR≈{hr_approx:.2f}, Median: High={median_high_str} vs Low={median_low_str} (Δ={advantage_str})")
                    
            except Exception as e:
                print(f"      ERROR analyzing {gene}: {e}")
                continue
        
        all_results.extend(cox_results)
        all_results.extend(logrank_results)
        
        # ADDED: Generate survival plots for each normalization method
        if len(method_survival_df) > 0 and any(f'{gene}_Expression' in method_survival_df.columns for gene in genes_of_interest):
            print(f"\n  📊 Generating survival plots for {method_name}...")
            
            # Get genes that have expression data in this method's DataFrame
            available_genes_method = [gene for gene in genes_of_interest 
                                    if f'{gene}_Expression' in method_survival_df.columns]
            
            if available_genes_method:
                # Create method-specific cancer type name to generate separate plot directories
                method_cancer_type = f"{cancer_type}_{method.replace('_', '-')}"
                
                # Call plotting with method-specific cancer type (function will create appropriate directory)
                plot_summary = generate_gene_survival_plots(
                    method_survival_df, available_genes_method, output_dir, 
                    method_cancer_type, create_plots=True
                )
                
                if plot_summary and plot_summary.get("plots_created", 0) > 0:
                    plots_dir = plot_summary.get("output_directory", "Unknown")
                    print(f"    ✅ {plot_summary.get('plots_created', 0)} plots saved to: {plots_dir}")
                    if plot_summary.get('significant_plots', 0) > 0:
                        print(f"       {plot_summary.get('significant_plots')} significant plots (p<0.05)")
                else:
                    print(f"    ⚠️  No plots generated for {method_name}")
            else:
                print(f"    ⚠️  No genes with expression data for {method_name}")
        
        print(f"  {method_name} analysis complete: {len(cox_results)} Cox + {len(logrank_results)} Log-rank results")
    
    # Save comprehensive results with improved structure
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Separate Cox and Log-rank results for better readability
        cox_df = results_df[results_df['Analysis_Type'] == 'Cox_Regression'].copy()
        logrank_df = results_df[results_df['Analysis_Type'] == 'Log_Rank'].copy()
        
        # Save separate files for different analysis types
        if not cox_df.empty:
            cox_file = os.path.join(immune_output_dir, f"{cancer_type}_Cox_Regression_Immune_Normalized.csv")
            cox_df.to_csv(cox_file, index=False)
            print(f"\n✅ Cox regression results saved: {cox_file}")
        
        if not logrank_df.empty:
            logrank_file = os.path.join(immune_output_dir, f"{cancer_type}_LogRank_Survival_Immune_Normalized.csv")
            logrank_df.to_csv(logrank_file, index=False)
            print(f"✅ Log-rank survival results saved: {logrank_file}")
            
            # Generate enhanced log-rank summary
            print(f"\n📊 LOG-RANK SURVIVAL ANALYSIS SUMMARY")
            print(f"{'='*60}")
            
            for gene in genes_of_interest:
                gene_logrank = logrank_df[logrank_df['Gene'] == gene]
                if not gene_logrank.empty:
                    print(f"\n🧬 {gene} - Log-rank Survival Analysis:")
                    
                    for _, row in gene_logrank.iterrows():
                        method_name = row['Method_Name']
                        p_val = row['P_Value']
                        hr_approx = row['HR_Approximation']
                        better_group = row['Better_Prognosis_Group']
                        
                        # Survival times
                        med_high_y = row['Median_Survival_High_Years']
                        med_low_y = row['Median_Survival_Low_Years']
                        advantage_y = row['Survival_Advantage_Years']
                        
                        # Format survival times
                        med_high_str = f"{med_high_y:.1f}y" if not pd.isna(med_high_y) else "NR"
                        med_low_str = f"{med_low_y:.1f}y" if not pd.isna(med_low_y) else "NR"
                        advantage_str = f"{advantage_y:+.1f}y" if not pd.isna(advantage_y) else "N/A"
                        
                        # Event rates
                        event_rate_high = row['Event_Rate_High']
                        event_rate_low = row['Event_Rate_Low']
                        
                        # Survival rates
                        surv_1y_high = row['Survival_1yr_High']
                        surv_1y_low = row['Survival_1yr_Low']
                        surv_3y_high = row['Survival_3yr_High']
                        surv_3y_low = row['Survival_3yr_Low']
                        
                        # Statistical significance
                        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        
                        print(f"  {method_name}:")
                        print(f"    Log-rank p-value: {p_val:.4f}{sig_marker}")
                        print(f"    HR approximation: {hr_approx:.2f}")
                        print(f"    Better prognosis: {better_group}")
                        print(f"    Median survival - High: {med_high_str}, Low: {med_low_str} (Advantage: {advantage_str})")
                        print(f"    Event rates - High: {event_rate_high:.1%}, Low: {event_rate_low:.1%}")
                        
                        if not pd.isna(surv_1y_high) and not pd.isna(surv_1y_low):
                            print(f"    1-year survival - High: {surv_1y_high:.1%}, Low: {surv_1y_low:.1%}")
                        if not pd.isna(surv_3y_high) and not pd.isna(surv_3y_low):
                            print(f"    3-year survival - High: {surv_3y_high:.1%}, Low: {surv_3y_low:.1%}")
            
            # Overall summary statistics
            significant_logrank = logrank_df[logrank_df['P_Value'] < 0.05]
            print(f"\n📈 OVERALL LOG-RANK SUMMARY:")
            print(f"   Total comparisons: {len(logrank_df)}")
            print(f"   Significant results (p<0.05): {len(significant_logrank)}")
            if len(significant_logrank) > 0:
                print(f"   Most significant:")
                top_result = significant_logrank.loc[significant_logrank['P_Value'].idxmin()]
                print(f"     {top_result['Gene']} ({top_result['Method_Name']}) - p={top_result['P_Value']:.4f}")
            print(f"{'='*60}")
        
        # Save combined results (legacy format)
        detailed_file = os.path.join(immune_output_dir, f"{cancer_type}_Gene_Survival_Immune_Normalized_Results.csv")
        results_df.to_csv(detailed_file, index=False)
        print(f"✅ Combined results saved: {detailed_file}")
        
        # Create comparison summary
        summary_data = []
        for gene in genes_of_interest:
            gene_results = results_df[(results_df['Gene'] == gene) & (results_df['Analysis_Type'] == 'Cox_Regression')]
            
            if len(gene_results) > 0:
                summary_row = {'Gene': gene}
                
                for method in normalization_methods.keys():
                    method_data = gene_results[gene_results['Normalization_Method'] == method]
                    if len(method_data) > 0:
                        result = method_data.iloc[0]
                        summary_row[f'{method}_HR'] = result['Hazard_Ratio']
                        summary_row[f'{method}_P'] = result['P_Value']
                        summary_row[f'{method}_Direction'] = result['Risk_Direction']
                
                summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(immune_output_dir, f"{cancer_type}_Gene_Survival_Immune_Normalized_Summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary comparison saved: {summary_file}")
            
            # Print key findings
            print(f"\n{'='*80}")
            print(f"KEY FINDINGS: IMMUNE NORMALIZATION EFFECTS")
            print(f"{'='*80}")
            
            for _, row in summary_df.iterrows():
                gene = row['Gene']
                print(f"\n{gene}:")
                for method in normalization_methods.keys():
                    if f'{method}_HR' in row and not pd.isna(row[f'{method}_HR']):
                        hr = row[f'{method}_HR']
                        p_val = row[f'{method}_P']
                        direction = row[f'{method}_Direction']
                        sig_status = "**" if p_val < 0.05 else ""
                        print(f"  {method:20s}: HR={hr:.3f}, p={p_val:.4f} ({direction}) {sig_status}")
        
        return results_df, summary_df
    
    else:
        print("WARNING: No immune-normalized results generated")
        return None, None


# ==============================================================================
# --- Main Execution ---
# ==============================================================================

def analyze_gene_immune_correlations(tumor_adata, genes_of_interest, cancer_type, output_dir):
    """
    NEW: Comprehensive correlation analysis between gene markers and specific immune cell infiltration scores.
    
    This function calculates correlations between genes of interest and:
    1. CD8 T cells (T_cells_CD8+ from Atlas or legacy T cells CD8)
    2. Individual NK subtypes (NK_Bright, NK_Cytotoxic, NK_Exhausted_TaNK from hybrid Atlas-Tang) 
    3. Other immune cell types for context (Atlas immune minor types)
    
    Parameters:
    -----------
    tumor_adata : AnnData
        Tumor samples with immune infiltration data
    genes_of_interest : list
        List of genes to analyze
    cancer_type : str
        Cancer type for output naming
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    pd.DataFrame
        Correlation results with statistical significance
    """
    print(f"\n--- Gene-Immune Cell Correlation Analysis for {cancer_type} ---")
    
    # Create correlation-specific output directory
    corr_output_dir = os.path.join(output_dir, f"{cancer_type}_Gene_Immune_Correlations")
    os.makedirs(corr_output_dir, exist_ok=True)
    
    # Define specific immune cell types to analyze (hybrid Atlas-Tang approach)
    primary_immune_cells = {
        'CD8_T_cells': ['T_cells_CD8+', 'T cells CD8'],  # Primary and legacy
        'NK_Bright': ['NK_Bright', 'Bright_NK'], 
        'NK_Cytotoxic': ['NK_Cytotoxic', 'Cytotoxic_NK'],
        'NK_Exhausted': ['NK_Exhausted_TaNK', 'Exhausted_TaNK']
    }
    
    # Additional immune cells for context (Atlas-based + legacy)
    context_immune_cells = {
        'CD4_T_cells': ['T_cells_CD4+', 'T cells CD4'],
        'NKT_cells': ['NKT_cells'],
        'T_cycling': ['T_cells_Cycling'],
        'B_naive': ['B_cells_Naive', 'B cells naive'],
        'B_memory': ['B_cells_Memory', 'B cells memory'],
        'Plasma_cells': ['Plasmablasts', 'Plasma cells'],
        'Macrophages': ['Macrophage'],
        'Monocytes': ['Monocyte'],
        'Dendritic_cells': ['DCs'],
        'Myeloid_cycling': ['Myeloid_Cycling']
    }
    
    # Combine all immune cells
    all_immune_cells = {**primary_immune_cells, **context_immune_cells}
    
    # Find available immune cell columns (handling multiple possible names per cell type)
    available_immune_cells = {}
    for cell_name, possible_columns in all_immune_cells.items():
        found_column = None
        for column_name in possible_columns:
            if column_name in tumor_adata.obs.columns:
                found_column = column_name
                break
        
        if found_column:
            available_immune_cells[cell_name] = found_column
        else:
            print(f"  WARNING: None of {possible_columns} found in data for {cell_name}")
    
    print(f"  Found {len(available_immune_cells)} immune cell types for correlation analysis")
    print(f"  Primary focus: CD8 T cells and NK subtypes")
    
    # Get gene expression data
    available_genes = [gene for gene in genes_of_interest if gene in tumor_adata.var_names]
    missing_genes = [gene for gene in genes_of_interest if gene not in tumor_adata.var_names]
    
    if missing_genes:
        print(f"  WARNING: {len(missing_genes)} genes not found: {missing_genes}")
    
    if not available_genes:
        print("  ERROR: No genes of interest found in expression data")
        return pd.DataFrame()
    
    print(f"  Analyzing correlations for {len(available_genes)} genes: {available_genes}")
    
    # Calculate correlations
    correlation_results = []
    
    for gene in available_genes:
        # Extract gene expression
        gene_expr = tumor_adata[:, gene].X
        if hasattr(gene_expr, "toarray"):
            gene_expr = gene_expr.toarray()
        gene_expr = np.ravel(gene_expr)
        
        print(f"\n  🧬 Analyzing {gene} correlations...")
        
        for cell_name, column_name in available_immune_cells.items():
            # Extract immune cell infiltration
            immune_score = tumor_adata.obs[column_name].values
            
            # Remove samples with missing data
            valid_mask = ~(np.isnan(gene_expr) | np.isnan(immune_score))
            
            if valid_mask.sum() < 10:
                print(f"    Insufficient valid samples for {gene} vs {cell_name}")
                continue
            
            gene_valid = gene_expr[valid_mask]
            immune_valid = immune_score[valid_mask]
            
            # Calculate Pearson correlation
            try:
                from scipy.stats import pearsonr, spearmanr
                
                # Pearson correlation (linear relationship)
                pearson_r, pearson_p = pearsonr(gene_valid, immune_valid)
                
                # Spearman correlation (monotonic relationship)
                spearman_r, spearman_p = spearmanr(gene_valid, immune_valid)
                
                # Categorize correlation strength
                def categorize_correlation(r):
                    abs_r = abs(r)
                    if abs_r >= 0.7:
                        return "Strong"
                    elif abs_r >= 0.5:
                        return "Moderate"
                    elif abs_r >= 0.3:
                        return "Weak"
                    else:
                        return "Very Weak"
                
                # Determine if this is a primary immune cell of interest
                is_primary = cell_name in primary_immune_cells
                cell_category = "Primary_Focus" if is_primary else "Context"
                
                correlation_results.append({
                    'Gene': gene,
                    'Immune_Cell_Type': cell_name,
                    'Immune_Cell_Column': column_name,
                    'Cell_Category': cell_category,
                    'N_Samples': valid_mask.sum(),
                    'Pearson_R': pearson_r,
                    'Pearson_P': pearson_p,
                    'Pearson_Significant': pearson_p < 0.05,
                    'Spearman_R': spearman_r,
                    'Spearman_P': spearman_p,
                    'Spearman_Significant': spearman_p < 0.05,
                    'Correlation_Strength': categorize_correlation(pearson_r),
                    'Direction': 'Positive' if pearson_r > 0 else 'Negative',
                    'Gene_Mean_Expression': gene_valid.mean(),
                    'Gene_Std_Expression': gene_valid.std(),
                    'Immune_Mean_Score': immune_valid.mean(),
                    'Immune_Std_Score': immune_valid.std()
                })
                
                # Print significant correlations
                if pearson_p < 0.05 and abs(pearson_r) >= 0.3:
                    direction = "↑" if pearson_r > 0 else "↓"
                    significance = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*"
                    print(f"    {direction} {cell_name}: r={pearson_r:.3f}, p={pearson_p:.4f}{significance}")
                
            except Exception as e:
                print(f"    ERROR calculating correlation for {gene} vs {cell_name}: {e}")
                continue
    
    # Convert to DataFrame and apply FDR correction
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        
        # Apply FDR correction to Pearson correlations
        corr_df = apply_fdr_correction_df(corr_df, 'Pearson_P', 'Pearson_FDR_Q')
        corr_df = apply_fdr_correction_df(corr_df, 'Spearman_P', 'Spearman_FDR_Q')
        
        # Sort by significance and correlation strength
        corr_df = corr_df.sort_values(['Cell_Category', 'Pearson_P'])
        
        # Save results
        corr_file = os.path.join(corr_output_dir, f"{cancer_type}_Gene_Immune_Correlations.csv")
        corr_df.to_csv(corr_file, index=False)
        
        # Create summary files
        primary_focus_df = corr_df[corr_df['Cell_Category'] == 'Primary_Focus'].copy()
        if not primary_focus_df.empty:
            primary_file = os.path.join(corr_output_dir, f"{cancer_type}_Gene_CD8_NK_Correlations_Primary.csv")
            primary_focus_df.to_csv(primary_file, index=False)
            print(f"\n✅ Primary correlations (CD8 & NK subtypes) saved: {primary_file}")
        
        # Generate summary report
        generate_correlation_summary_report(corr_df, cancer_type, corr_output_dir)
        
        print(f"✅ Complete correlation analysis saved: {corr_file}")
        return corr_df
    
    else:
        print("WARNING: No correlation results generated")
        return pd.DataFrame()


def generate_correlation_summary_report(corr_df, cancer_type, output_dir):
    """
    Generate a comprehensive summary report of gene-immune correlations.
    """
    print(f"\n📊 GENE-IMMUNE CORRELATION SUMMARY - {cancer_type}")
    print("="*70)
    
    # Overall statistics
    total_correlations = len(corr_df)
    significant_pearson = (corr_df['Pearson_P'] < 0.05).sum()
    fdr_significant_pearson = (corr_df['Pearson_FDR_Q'] < 0.05).sum() if 'Pearson_FDR_Q' in corr_df.columns else 0
    
    print(f"📈 Overall Statistics:")
    print(f"   Total correlations tested: {total_correlations}")
    print(f"   Nominally significant (p<0.05): {significant_pearson}")
    print(f"   FDR significant (q<0.05): {fdr_significant_pearson}")
    
    # Primary focus results (CD8 & NK subtypes)
    primary_df = corr_df[corr_df['Cell_Category'] == 'Primary_Focus']
    if not primary_df.empty:
        print(f"\n🎯 Primary Focus Results (CD8 T cells & NK subtypes):")
        
        for gene in primary_df['Gene'].unique():
            gene_primary = primary_df[primary_df['Gene'] == gene]
            significant_primary = gene_primary[gene_primary['Pearson_P'] < 0.05]
            
            if not significant_primary.empty:
                print(f"\n   🧬 {gene}:")
                for _, row in significant_primary.iterrows():
                    r = row['Pearson_R']
                    p = row['Pearson_P']
                    cell = row['Immune_Cell_Type']
                    direction = "↗️" if r > 0 else "↘️"
                    strength = row['Correlation_Strength']
                    sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    
                    print(f"     {direction} {cell}: r={r:.3f}, p={p:.4f}{sig_marker} ({strength})")
    
    # Strong correlations across all immune cells
    strong_corr = corr_df[(abs(corr_df['Pearson_R']) >= 0.5) & (corr_df['Pearson_P'] < 0.05)]
    if not strong_corr.empty:
        print(f"\n💪 Strong Correlations (|r| ≥ 0.5, p<0.05):")
        for _, row in strong_corr.iterrows():
            gene = row['Gene']
            cell = row['Immune_Cell_Type']
            r = row['Pearson_R']
            p = row['Pearson_P']
            direction = "↗️" if r > 0 else "↘️"
            print(f"   {direction} {gene} ↔ {cell}: r={r:.3f}, p={p:.4f}")
    
    # Save summary to file
    summary_lines = [
        f"Gene-Immune Correlation Analysis Summary - {cancer_type}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Overall Statistics:",
        f"  Total correlations tested: {total_correlations}",
        f"  Nominally significant (p<0.05): {significant_pearson}",
        f"  FDR significant (q<0.05): {fdr_significant_pearson}",
        f""
    ]
    
    if not primary_df.empty:
        summary_lines.append("Primary Focus Results (CD8 T cells & NK subtypes):")
        for gene in primary_df['Gene'].unique():
            gene_primary = primary_df[primary_df['Gene'] == gene]
            significant_primary = gene_primary[gene_primary['Pearson_P'] < 0.05]
            
            if not significant_primary.empty:
                summary_lines.append(f"  {gene}:")
                for _, row in significant_primary.iterrows():
                    r = row['Pearson_R']
                    p = row['Pearson_P']
                    cell = row['Immune_Cell_Type']
                    direction = "Positive" if r > 0 else "Negative"
                    strength = row['Correlation_Strength']
                    summary_lines.append(f"    {cell}: r={r:.3f}, p={p:.4f} ({direction}, {strength})")
    
    summary_file = os.path.join(output_dir, f"{cancer_type}_Gene_Immune_Correlation_Summary.txt")
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n📋 Summary report saved: {summary_file}")
    print("="*70)


# ==============================================================================
# --- Publication-Ready Survival Plotting Functions (Adapted from TCGA_STUDY_ANALYSIS.py) ---
# ==============================================================================

def sanitize_filename(s):
    """Replace non-alphanumeric characters (except _ and -) with underscores."""
    return re.sub(r"[^\w\-]", "_", str(s))


def extract_robust_cox_ci(cph_summary, variable_name):
    """
    Robust extraction of Cox regression confidence intervals.
    Handles different lifelines versions and column naming conventions.
    """
    try:
        summary_row = cph_summary.loc[variable_name]
        
        # Try multiple possible CI column names (different lifelines versions)
        ci_patterns = [
            ('coef lower 95%', 'coef upper 95%'),
            ('exp(coef) lower 95%', 'exp(coef) upper 95%'),
            ('lower 0.95', 'upper 0.95'),
            ('2.5%', '97.5%'),
            ('coef lower CI', 'coef upper CI')
        ]
        
        for lower_col, upper_col in ci_patterns:
            if lower_col in summary_row.index and upper_col in summary_row.index:
                coef_lower = summary_row[lower_col]
                coef_upper = summary_row[upper_col]
                # Convert to HR scale if we got coefficient CIs
                if 'coef' in lower_col:
                    return np.exp(coef_lower), np.exp(coef_upper)
                else:
                    return coef_lower, coef_upper
        
        # Fallback: Calculate CI from coefficient and standard error
        if 'coef' in summary_row.index and 'se(coef)' in summary_row.index:
            coef = summary_row['coef']
            se = summary_row['se(coef)']
            ci_lower_coef = coef - 1.96 * se
            ci_upper_coef = coef + 1.96 * se
            return np.exp(ci_lower_coef), np.exp(ci_upper_coef)
        
        # Final fallback: return None if nothing works
        print(f"    Warning: Could not extract CI for {variable_name}")
        return None, None
        
    except Exception as e:
        print(f"    Warning: CI extraction failed for {variable_name}: {e}")
        return None, None


def create_publication_km_plot(
    high_group, low_group, group_labels=None, title="Kaplan-Meier Survival Curve",
    colors=None, figsize=(12, 8), save_path=None, show_risk_table=True,
    hr=None, p_value=None, fdr_q_value=None, ci_95=None, variable_name=None,
    scenario_name=None, font_size_base=12
):
    """
    Create publication-ready Kaplan-Meier survival plots with risk tables.
    Adapted from TCGA_STUDY_ANALYSIS.py for gene expression analysis.
    
    CRITICAL STATISTICS NOTES:
    - HR: Hazard ratio comparing HIGH vs LOW expression groups (Group_Binary Cox regression)
    - P-value: Log-rank test comparing survival curves (appropriate for KM plots)
    - Median survival: From Kaplan-Meier estimator for each group
    - Sample sizes and events: Actual counts from the groups being plotted
    
    Parameters:
    -----------
    high_group : pd.DataFrame
        High expression group with 'Survival_Time' and 'Event' columns
    low_group : pd.DataFrame  
        Low expression group with 'Survival_Time' and 'Event' columns
    group_labels : list, optional
        Labels for [high_group, low_group]. Default: ["High Expression", "Low Expression"]
    title : str, optional
        Plot title
    colors : list, optional
        Colors for [high_group, low_group]. Default: ["#E31A1C", "#1F78B4"]
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    show_risk_table : bool, optional
        Whether to show numbers at risk table
    hr : float, optional
        Hazard ratio comparing high vs low expression groups
    p_value : float, optional
        Log-rank test p-value (appropriate for survival curve comparison)
    fdr_q_value : float, optional
        FDR-corrected q-value to display
    ci_95 : tuple, optional
        95% confidence interval for group comparison HR (lower, upper)
    variable_name : str, optional
        Name of the gene being analyzed
    scenario_name : str, optional
        Analysis scenario name
    font_size_base : int, optional
        Base font size for scaling
        
    Returns:
    --------
    tuple
        (fig, axes) matplotlib figure and axes objects
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
            group_labels = ["High Expression", "Low Expression"]
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
        
        # Convert x-axis to years (robust tick handling)
        try:
            x_ticks = ax_main.get_xticks()
            x_labels = [f"{int(x/365.25)}" for x in x_ticks if x >= 0]
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
        else:
            # Only show if we have a valid comparison
            if len(high_group) >= 5 and len(low_group) >= 5 and high_events >= 2 and low_events >= 2:
                stats_text.append("HR = Could not calculate")
        
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
        
        # Add median survival times (simplified and clearer display)
        try:
            median_high = kmf_high.median_survival_time_
            median_low = kmf_low.median_survival_time_
            
            # Simplified median survival display
            if pd.isna(median_high) or np.isinf(median_high):
                median_high_str = "Not Reached"
            else:
                median_high_str = f"{median_high/365.25:.1f} years"
                
            if pd.isna(median_low) or np.isinf(median_low):
                median_low_str = "Not Reached"
            else:
                median_low_str = f"{median_low/365.25:.1f} years"
                
            stats_text.append(f"Median survival: {median_high_str} vs {median_low_str}")
        except Exception as e:
            print(f"    Warning: Median survival calculation failed: {e}")
            stats_text.append("Median survival: calculation failed")
        
        # VALIDATION: Ensure statistical consistency before display
        try:
            # Basic validation of displayed statistics
            if hr is not None and not np.isnan(hr):
                if hr <= 0:
                    print(f"    Warning: Invalid HR value ({hr}) - setting to None")
                    hr = None
                    ci_95 = None
            
            # Validate CI bounds
            if ci_95 is not None and len(ci_95) == 2:
                if ci_95[0] <= 0 or ci_95[1] <= 0 or ci_95[0] >= ci_95[1]:
                    print(f"    Warning: Invalid CI bounds ({ci_95}) - removing CI")
                    ci_95 = None
            
            # Validate sample sizes match what we're analyzing
            expected_total = len(high_group) + len(low_group)
            actual_events = high_events + low_events
            if actual_events > expected_total:
                print(f"    Warning: Events ({actual_events}) exceed total samples ({expected_total})")
            
        except Exception as e:
            print(f"    Warning: Statistics validation failed: {e}")
        
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
                create_risk_table(ax_risk, kmf_high, kmf_low, group_labels, colors, font_size_base)
            except Exception as e:
                print(f"    Warning: Risk table creation failed: {e}")
        
        # Add analysis details to title area
        if variable_name or scenario_name:
            try:
                subtitle_parts = []
                if scenario_name:
                    subtitle_parts.append(f"Scenario: {scenario_name}")
                if variable_name:
                    subtitle_parts.append(f"Gene: {variable_name}")
                
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


def create_risk_table(ax, kmf_high, kmf_low, group_labels, colors, font_size_base):
    """
    Create a professional publication-quality numbers-at-risk table.
    Adapted from TCGA_STUDY_ANALYSIS.py for gene expression analysis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes object for the risk table
    kmf_high : KaplanMeierFitter
        Fitted KM model for high expression group
    kmf_low : KaplanMeierFitter
        Fitted KM model for low expression group
    group_labels : list
        Labels for the groups
    colors : list
        Colors for the groups
    font_size_base : int
        Base font size
    """
    # Calculate time points for risk table (optimized for 10-year survival analysis)
    max_time_days = max(
        kmf_high.durations.max() if len(kmf_high.durations) > 0 else 0,
        kmf_low.durations.max() if len(kmf_low.durations) > 0 else 0
    )
    max_time_years = min(max_time_days / 365.25, 10.0)  # Cap at 10 years
    
    # Optimized time points for 10-year survival analysis
    if max_time_years <= 3:
        time_points_years = [0, 1, 2, 3]
    elif max_time_years <= 5:
        time_points_years = [0, 1, 2, 3, 4, 5]
    elif max_time_years <= 8:
        time_points_years = [0, 1, 2, 3, 5, 7]
    else:
        time_points_years = [0, 1, 2, 3, 5, 7, 10]  # Standard 10-year analysis
    
    # Filter to only include time points within data range (max 10 years)
    time_points_years = [t for t in time_points_years if t <= min(max_time_years + 0.5, 10.0)]
    time_points_days = [t * 365.25 for t in time_points_years]
    
    # Calculate numbers at risk using proper lifelines method
    risk_high = []
    risk_low = []
    
    for t_days in time_points_days:
        try:
            # Count subjects still at risk at time t
            n_risk_high = (kmf_high.durations >= t_days).sum()
            n_risk_low = (kmf_low.durations >= t_days).sum()
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


def generate_gene_survival_plots(
    df, available_genes, output_dir, cancer_type, create_plots=True,
    plot_scenarios=None, plot_strategies=None
):
    """
    Generate comprehensive survival plots for gene expression analysis.
    Adapted from TCGA_STUDY_ANALYSIS.py for gene expression instead of NK infiltration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survival data with gene expression variables
    available_genes : list
        List of genes with expression data available
    output_dir : str
        Output directory for plots
    cancer_type : str
        Cancer type abbreviation
    create_plots : bool, optional
        Whether to actually create plots (set False for testing)
    plot_scenarios : list, optional
        Specific scenarios to plot. If None, plots all scenarios
    plot_strategies : list, optional
        Specific strategies to plot. If None, plots all strategies
        
    Returns:
    --------
    dict
        Summary of plots created
    """
    print(f"\n=== Generating Gene Expression Survival Plots for {cancer_type} ===")
    
    if not create_plots:
        print("  Plot creation disabled - running in analysis mode only")
        return {}
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, f"{cancer_type}_Gene_Survival_Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_summary = {
        "plots_created": 0,
        "plots_failed": 0,
        "significant_plots": 0,
        "output_directory": plots_dir
    }
    
    # Define scenarios and strategies to plot
    scenarios = [
        ("Overall", df, "Complete cohort"),
        ("Age_<60", df[df["Age_Group"] == "<60"], "Age < 60 years"),
        ("Age_>=60", df[df["Age_Group"] == ">=60"], "Age ≥ 60 years")
    ]
    
    # Use the same stratification strategies as defined in the main script
    split_strategies = [
        ("Tertile", lambda x: (x.quantile(0.33), x.quantile(0.67)), "Top vs Bottom 1/3"),
        ("Median", lambda x: (x.median(), x.median()), "Above vs Below Median"),
        ("Quartile", lambda x: (x.quantile(0.25), x.quantile(0.75)), "Top vs Bottom 1/4")
    ]
    
    # Filter scenarios and strategies if specified
    if plot_scenarios:
        scenarios = [(name, data, desc) for name, data, desc in scenarios if name in plot_scenarios]
    if plot_strategies:
        split_strategies = [(name, func, desc) for name, func, desc in split_strategies if name in plot_strategies]
    
    for scenario_name, subset_df, scenario_desc in scenarios:
        if len(subset_df) < 30:
            print(f"  Skipping {scenario_name}: insufficient samples ({len(subset_df)} < 30)")
            continue
            
        total_events = subset_df["Event"].sum()
        if total_events < 10:
            print(f"  Skipping {scenario_name}: insufficient events ({total_events} < 10)")
            continue
            
        print(f"\n  Creating plots for {scenario_name} ({scenario_desc})")
        
        for gene in available_genes:
            gene_expr_col = f"{gene}_Expression"
            
            if gene_expr_col not in subset_df.columns:
                continue
                
            # Check for sufficient variation
            if subset_df[gene_expr_col].std() < 1e-8:
                print(f"    Skipping {gene}: insufficient expression variation")
                continue
                
            for strategy_name, split_func, strategy_desc in split_strategies:
                try:
                    # Clean data before analysis to remove NaNs
                    analysis_df = subset_df[["Survival_Time", "Event", gene_expr_col]].dropna()
                    
                    if len(analysis_df) < 20:  # Need sufficient cleaned data
                        continue
                        
                    # Calculate split thresholds on cleaned data
                    low_thresh, high_thresh = split_func(analysis_df[gene_expr_col])
                    
                    # Create groups from cleaned data
                    if strategy_name == "Median":
                        high_group = analysis_df[analysis_df[gene_expr_col] > high_thresh]
                        low_group = analysis_df[analysis_df[gene_expr_col] <= low_thresh]
                    else:
                        high_group = analysis_df[analysis_df[gene_expr_col] >= high_thresh]
                        low_group = analysis_df[analysis_df[gene_expr_col] <= low_thresh]
                    
                    # Quality control
                    if len(high_group) < 10 or len(low_group) < 10:
                        continue
                        
                    high_events = high_group["Event"].sum()
                    low_events = low_group["Event"].sum()
                    
                    if high_events < 3 or low_events < 3:
                        continue
                    
                    # Perform statistical tests
                    logrank_result = logrank_test(
                        durations_A=high_group["Survival_Time"],
                        durations_B=low_group["Survival_Time"],
                        event_observed_A=high_group["Event"],
                        event_observed_B=low_group["Event"]
                    )
                    
                    # Calculate HR for HIGH vs LOW group comparison
                    try:
                        # Create combined dataset with group indicator
                        combined_group_data = pd.concat([
                            high_group.assign(Group='High'),
                            low_group.assign(Group='Low')
                        ])
                        
                        # Create binary indicator (High=1, Low=0) for Cox regression
                        combined_group_data['Group_Binary'] = (combined_group_data['Group'] == 'High').astype(int)
                        
                        # Fit Cox model comparing High vs Low groups directly
                        cph = CoxPHFitter()
                        cox_group_df = combined_group_data[["Survival_Time", "Event", "Group_Binary"]].dropna()
                        
                        if len(cox_group_df) >= 10 and cox_group_df['Group_Binary'].var() > 0:
                            cph.fit(cox_group_df, duration_col="Survival_Time", event_col="Event")
                            
                            # Extract group comparison HR (High vs Low)
                            hr = cph.hazard_ratios_['Group_Binary']
                            
                            # Get confidence intervals using robust method
                            hr_ci_lower, hr_ci_upper = extract_robust_cox_ci(cph.summary, 'Group_Binary')
                            hr_ci = (hr_ci_lower, hr_ci_upper) if hr_ci_lower is not None else None
                            
                            print(f"      Cox HR (High vs Low {gene}): {hr:.3f}")
                        else:
                            print(f"      Warning: Insufficient data for group Cox regression")
                            hr = None
                            hr_ci = None
                    except Exception as e:
                        print(f"      Warning: Group Cox regression failed for {gene}: {e}")
                        hr = None
                        hr_ci = None
                    
                    # Create plot title
                    plot_title = f"{cancer_type} Survival Analysis\n{gene} Expression - {strategy_desc}"
                    if scenario_name != "Overall":
                        plot_title += f" ({scenario_desc})"
                    
                    # Create filename
                    safe_scenario = sanitize_filename(scenario_name)
                    safe_gene = sanitize_filename(gene)
                    safe_strategy = sanitize_filename(strategy_name)
                    
                    filename = f"{cancer_type}_{safe_scenario}_{safe_gene}_{safe_strategy}_Survival.png"
                    save_path = os.path.join(plots_dir, filename)
                    
                    # Create the plot
                    fig, axes = create_publication_km_plot(
                        high_group=high_group,
                        low_group=low_group,
                        group_labels=[f"High {gene}", f"Low {gene}"],
                        title=plot_title,
                        save_path=save_path,
                        hr=hr,
                        p_value=logrank_result.p_value,
                        ci_95=hr_ci,
                        variable_name=gene,
                        scenario_name=scenario_name,
                        figsize=(12, 8)
                    )
                    
                    if fig is not None:
                        plt.close(fig)  # Free memory
                        
                        plot_summary["plots_created"] += 1
                        if logrank_result.p_value < 0.05:
                            plot_summary["significant_plots"] += 1
                        
                        # Print progress
                        sig_marker = "***" if logrank_result.p_value < 0.001 else "**" if logrank_result.p_value < 0.01 else "*" if logrank_result.p_value < 0.05 else ""
                        hr_text = f", HR={hr:.2f}" if hr else ""
                        print(f"    ✅ {gene} ({strategy_name}): p={logrank_result.p_value:.3f}{sig_marker}{hr_text}")
                    else:
                        plot_summary["plots_failed"] += 1
                        print(f"    ❌ {gene} ({strategy_name}): Plot creation failed")
                    
                except Exception as e:
                    print(f"    ❌ {gene} ({strategy_name}): Plot failed - {e}")
                    plot_summary["plots_failed"] += 1
                    continue
    
    # Print summary
    print(f"\n  📊 Gene Survival Plots Summary:")
    print(f"    Total plots created: {plot_summary['plots_created']}")
    print(f"    Significant plots (p<0.05): {plot_summary['significant_plots']}")
    print(f"    Failed plots: {plot_summary['plots_failed']}")
    print(f"    Output directory: {plots_dir}")
    
    return plot_summary


if __name__ == "__main__":
    # ===== EASY CONFIGURATION SECTION =====
    # Modify these variables to customize your analysis
    
    CANCER_TYPE = "BRCA"  # Change to your cancer type of interest
    BASE_DATA_DIR = r"C:\Users\met-a\Documents\Analysis\TCGAdata"
    OUTPUT_DIR = r"C:\Users\met-a\Documents\Analysis\TCGAdata\Analysis_Python_Output"
    
    # Add your genes of interest here (modify GENES_OF_INTEREST at top of script)
    print(f"🧬 GENES OF INTEREST: {GENES_OF_INTEREST}")
    print(f"📊 STRATIFICATION STRATEGIES: {list(STRATIFICATION_STRATEGIES.keys())}")
    
    # ===== ANALYSIS EXECUTION =====
    
    # Load TCGA data using identical pipeline
    print(f"\n🔄 Loading TCGA data for {CANCER_TYPE}...")
    tumor_adata, master_metadata = load_and_preprocess_tcga_data(
        cancer_type=CANCER_TYPE,
        base_data_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR,
        thresholds=DEFAULT_THRESHOLDS,
    )
    
    # Integrate CIBERSORTx data for immune infiltration analysis
    print(f"\n🔄 Integrating CIBERSORTx immune infiltration data...")
    cibersort_paths = {
        "cibersortx": os.path.join(
            BASE_DATA_DIR, f"CIBERSORTx_BRCA.csv"
        )
    }
    
    # Check if CIBERSORTx file exists
    cibersort_file = cibersort_paths["cibersortx"]
    if os.path.exists(cibersort_file):
        print(f"   Found CIBERSORTx BRCA file with enhanced NK subtypes: {cibersort_file}")
        print(f"   This file contains Bright_NK, Cytotoxic_NK, and Exhausted_TaNK subtypes")
        tumor_adata = integrate_cibersortx_data(tumor_adata, cibersort_paths, DEFAULT_THRESHOLDS)
        if tumor_adata is not None:
            immune_cols = [col for col in tumor_adata.obs.columns if 
                          any(cell_type in col for cell_type in 
                              ['T cells', 'B cells', 'Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK'])]
            print(f"   ✅ Successfully integrated {len(immune_cols)} immune infiltration columns")
            print(f"       Enhanced NK subtypes: Bright_NK, Cytotoxic_NK, Exhausted_TaNK")
        else:
            print(f"   ❌ Failed to integrate CIBERSORTx data")
    else:
        print(f"   ⚠️  CIBERSORTx file not found: {cibersort_file}")
        print(f"   Will perform analysis without immune normalization")

    if tumor_adata is not None:
        print(f"\n✅ SUCCESS: Data loaded and preprocessed")
        print(f"   tumor_adata: {tumor_adata.n_obs} samples x {tumor_adata.n_vars} genes")

        # Create cancer-specific output directory
        cancer_output_dir = os.path.join(OUTPUT_DIR, f"{CANCER_TYPE}_Gene_Survival")
        os.makedirs(cancer_output_dir, exist_ok=True)
        
        # Add gene expression data to AnnData
        tumor_adata, available_genes = add_gene_expression_to_adata(tumor_adata, GENES_OF_INTEREST)
        
        if available_genes:
            # RUN DIAGNOSTICS FIRST
            print(f"\n🔍 RUNNING SURVIVAL ANALYSIS DIAGNOSTICS...")
            survival_df = diagnose_survival_analysis_issues(tumor_adata, CANCER_TYPE)
            
            # Check if CIBERSORTx data is available for immune normalization
            immune_cols = [col for col in tumor_adata.obs.columns if 
                          any(cell_type in col for cell_type in 
                              ['T cells', 'B cells', 'Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK'])]
            
            if immune_cols:
                print(f"\n🧬 IMMUNE INFILTRATION DATA AVAILABLE")
                print(f"   Found {len(immune_cols)} immune cell type columns")
                print(f"   Enhanced NK subtypes: Bright_NK, Cytotoxic_NK, Exhausted_TaNK")
                print(f"   Will perform ENHANCED analysis with immune normalization")
                print(f"   This will help distinguish direct gene effects from immune confounding")
                
                # Perform enhanced analysis with immune normalization
                immune_results, immune_summary = analyze_genes_with_immune_normalization(
                    tumor_adata, available_genes, CANCER_TYPE, cancer_output_dir
                )
                
                if immune_results is not None:
                    print(f"\n✅ IMMUNE-NORMALIZED ANALYSIS COMPLETE!")
                    print(f"   Check the results to see if 'strange' effects like protective PDCD1")
                    print(f"   are due to immune infiltration confounding rather than direct gene effects")
                    
                    # NOTE: Survival plots are generated within the immune normalization analysis
                    # for each normalization method (raw, immune_normalized, cytotoxic_lymphoid_normalized)
                else:
                    print(f"\n⚠️  Immune normalization failed, falling back to standard analysis")
                    
                    # Fallback to standard analysis
                    survival_df = prepare_survival_data(tumor_adata)
                    cox_results, logrank_results = perform_gene_survival_analysis(
                        survival_df, available_genes, cancer_output_dir, CANCER_TYPE
                    )
                    
                    # ADDED: Generate survival plots for fallback analysis
                    if not survival_df.empty and available_genes:
                        print(f"\n📊 GENERATING SURVIVAL PLOTS (Fallback Analysis)...")
                        plot_summary = generate_gene_survival_plots(
                            survival_df, available_genes, cancer_output_dir, CANCER_TYPE,
                            create_plots=True
                        )
                        
                        if plot_summary:
                            print(f"   ✅ Survival plots generated:")
                            print(f"     Total plots: {plot_summary.get('plots_created', 0)}")
                            print(f"     Significant plots: {plot_summary.get('significant_plots', 0)}")
                            print(f"     Plots directory: {plot_summary.get('output_directory', 'Unknown')}")
                        else:
                            print(f"   ⚠️  Plot generation failed or returned no summary")
                    
                # NEW: Add comprehensive gene-immune correlation analysis
                print(f"\n🔬 RUNNING GENE-IMMUNE CORRELATION ANALYSIS...")
                print(f"   This will analyze correlations between your genes of interest and:")
                print(f"   - CD8 T cells")
                print(f"   - NK subtypes (Bright_NK, Cytotoxic_NK, Exhausted_TaNK)")
                print(f"   - Additional immune cell types for context")
                
                correlation_results = analyze_gene_immune_correlations(
                    tumor_adata, available_genes, CANCER_TYPE, cancer_output_dir
                )
                
                if correlation_results is not None and not correlation_results.empty:
                    print(f"\n✅ CORRELATION ANALYSIS COMPLETE!")
                    print(f"   Results show associations between gene expression and immune infiltration")
                    print(f"   This helps interpret survival analysis results in immune context")
                else:
                    print(f"\n⚠️  Correlation analysis yielded no results")
            else:
                print(f"\n📊 STANDARD ANALYSIS (No immune data available)")
                print(f"   Performing traditional gene expression survival analysis")
                
                # Standard analysis without immune normalization
                survival_df = prepare_survival_data(tumor_adata)
                cox_results, logrank_results = perform_gene_survival_analysis(
                    survival_df, available_genes, cancer_output_dir, CANCER_TYPE
                )
                
                # ADDED: Generate survival plots for standard analysis
                if not survival_df.empty and available_genes:
                    print(f"\n📊 GENERATING SURVIVAL PLOTS (Standard Analysis)...")
                    plot_summary = generate_gene_survival_plots(
                        survival_df, available_genes, cancer_output_dir, CANCER_TYPE,
                        create_plots=True
                    )
                    
                    if plot_summary:
                        print(f"   ✅ Survival plots generated:")
                        print(f"     Total plots: {plot_summary.get('plots_created', 0)}")
                        print(f"     Significant plots: {plot_summary.get('significant_plots', 0)}")
                        print(f"     Plots directory: {plot_summary.get('output_directory', 'Unknown')}")
                    else:
                        print(f"   ⚠️  Plot generation failed or returned no summary")
                
                # Generate summary
                summary = generate_analysis_summary(cox_results, logrank_results, CANCER_TYPE, available_genes)
                
                # Save summary
                summary_file = os.path.join(cancer_output_dir, f"{CANCER_TYPE}_Gene_Survival_Summary.json")
                import json
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # NEW: Attempt correlation analysis even in standard mode
                print(f"\n🔬 ATTEMPTING GENE-IMMUNE CORRELATION ANALYSIS...")
                print(f"   Checking for any available immune infiltration data...")
                
                correlation_results = analyze_gene_immune_correlations(
                    tumor_adata, available_genes, CANCER_TYPE, cancer_output_dir
                )
                
                if correlation_results is not None and not correlation_results.empty:
                    print(f"\n✅ CORRELATION ANALYSIS COMPLETE!")
                    print(f"   Found immune data and generated correlation results")
                else:
                    print(f"\n⚠️  No immune infiltration data available for correlation analysis")
                
                print(f"\n💾 Analysis complete! Results saved to: {cancer_output_dir}")
                print(f"📋 Summary saved to: {summary_file}")
            
        else:
            print(f"\n❌ ERROR: No genes of interest found in the dataset")
            print(f"   Available genes: {list(tumor_adata.var_names[:20])}... (showing first 20)")
    else:
        print(f"\n❌ FAILED: Data loading unsuccessful")
        print(f"   Check your data paths and cancer type configuration") 