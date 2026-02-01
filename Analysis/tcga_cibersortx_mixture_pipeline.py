#!/usr/bin/env python3
"""
Comprehensive TCGA Raw Data to CIBERSORTx Mixture Pipeline

This script processes raw TCGA data directly from XML clinical files and TSV RNA-seq files
to create CIBERSORTx-compatible mixture files for NK cell deconvolution analysis.

Features:
- Parses TCGA XML clinical files using the same approach as TCGA_STUDY_ANALYSIS.py
- Loads GDC sample sheet metadata
- Processes raw RNA-seq TSV files
- Creates CIBERSORTx-ready tumor mixture files
- Comprehensive quality control and filtering
- Automated cancer type detection and processing

Author: AI Assistant
Date: 2025
"""

import os
import re
import json
import logging
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import sparse
import scanpy as sc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class TCGACIBERSORTxProcessor:
    """
    Comprehensive TCGA raw data processor for CIBERSORTx mixture file creation.
    """
    
    def __init__(self, 
                 tcga_base_dir: str = "TCGAdata",
                 output_dir: str = "outputs/tcga_cibersortx_mixtures",
                 sample_sheet_filename: str = "gdc_sample_sheet.2025-06-26.tsv"):
        """
        Initialize the TCGA CIBERSORTx processor.
        
        Parameters:
        -----------
        tcga_base_dir : str
            Base directory containing TCGA data
        output_dir : str
            Output directory for CIBERSORTx mixture files
        sample_sheet_filename : str
            Name of the GDC sample sheet file
        """
        self.tcga_base_dir = Path(tcga_base_dir)
        self.output_dir = Path(output_dir)
        self.sample_sheet_filename = sample_sheet_filename
        
        # Define paths
        self.xml_dir = self.tcga_base_dir / "xml"
        self.rna_dir = self.tcga_base_dir / "rna"
        self.sample_sheet_path = self.tcga_base_dir / sample_sheet_filename
        
        # Initialize data containers
        self.clinical_data = None
        self.sample_sheet_data = None
        self.cancer_types = None
        
        # TCGA XML namespaces (same as TCGA_STUDY_ANALYSIS.py)
        self.xml_namespaces = {
            'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
            'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
            'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
            'shared_stage': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7',
        }
        
        # Disease-specific XML path configurations (from TCGA_STUDY_ANALYSIS.py)
        self.disease_config = {
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
        
        # CIBERSORTx filtering thresholds
        self.cibersortx_thresholds = {
            "min_expression_threshold": 0.1,  # Minimum mean expression for gene inclusion
            "min_nonzero_samples": 10,         # Minimum samples with non-zero expression
            "min_variance_threshold": 0.01,    # Minimum variance across samples
            "max_zero_fraction": 0.8,          # Maximum fraction of zero values per gene
            "preferred_rna_count_column": "tpm_unstranded",
            "min_cells_gene_filter": 5,
        }
        
        # Cancer type mappings
        self.cancer_type_mappings = {
            'BRCA': 'Breast Invasive Carcinoma',
            'GBM': 'Glioblastoma Multiforme', 
            'LUAD': 'Lung Adenocarcinoma',
            'LUSC': 'Lung Squamous Cell Carcinoma',
            'KIRC': 'Kidney Renal Clear Cell Carcinoma',
            'HNSC': 'Head and Neck Squamous Cell Carcinoma',
            'SKCM': 'Skin Cutaneous Melanoma',
            'OV': 'Ovarian Serous Cystadenocarcinoma',
            'PRAD': 'Prostate Adenocarcinoma',
            'THCA': 'Thyroid Carcinoma',
            'COAD': 'Colon Adenocarcinoma',
            'BLCA': 'Bladder Urothelial Carcinoma',
            'LIHC': 'Liver Hepatocellular Carcinoma',
            'STAD': 'Stomach Adenocarcinoma',
            'LGG': 'Lower Grade Glioma'
        }
        
        # Patient deduplication ensures one sample per patient - no need for validation limits
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_xml_text(self, root, xpath: str, namespaces: dict, default_val: str = "N/A") -> str:
        """
        Extract text from XML element using XPath (from TCGA_STUDY_ANALYSIS.py).
        
        Parameters:
        -----------
        root : xml.etree.ElementTree.Element
            XML root element
        xpath : str
            XPath to target element
        namespaces : dict
            XML namespaces
        default_val : str
            Default value if element not found
            
        Returns:
        --------
        str
            Extracted text or default value
        """
        try:
            element = root.find(xpath, namespaces)
            if element is not None and element.text:
                return element.text.strip()
        except Exception:
            pass
        return default_val
    
    def parse_tcga_clinical_xml_file(self, xml_file_path: Path) -> Optional[Dict]:
        """
        Parse a single TCGA clinical XML file (adapted from TCGA_STUDY_ANALYSIS.py).
        
        Parameters:
        -----------
        xml_file_path : Path
            Path to XML file
            
        Returns:
        --------
        Optional[Dict]
            Dictionary of clinical data or None if parsing failed
        """
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
        except ET.ParseError:
            logger.warning(f"Could not parse XML: {xml_file_path.name}")
            return None
        
        disease_code_val = self._get_xml_text(root, ".//admin:disease_code", self.xml_namespaces, "UNKNOWN")
        cfg = self.disease_config.get(disease_code_val.upper(), self.disease_config["DEFAULT"])
        
        clinical_record = {
            "Patient_Barcode": self._get_xml_text(root, ".//shared:bcr_patient_barcode", self.xml_namespaces),
            "Patient_UUID": self._get_xml_text(root, ".//shared:bcr_patient_uuid", self.xml_namespaces),
            "Disease_Code": disease_code_val,
            "Gender": self._get_xml_text(root, ".//shared:gender", self.xml_namespaces),
            "Race": self._get_xml_text(root, ".//clin_shared:race", self.xml_namespaces),
            "Ethnicity": self._get_xml_text(root, ".//clin_shared:ethnicity", self.xml_namespaces),
            "Vital_Status": self._get_xml_text(root, ".//clin_shared:vital_status", self.xml_namespaces),
            "Age_at_Diagnosis": self._get_xml_text(root, ".//clin_shared:age_at_initial_pathologic_diagnosis", self.xml_namespaces),
            "Days_to_Birth": self._get_xml_text(root, ".//clin_shared:days_to_birth", self.xml_namespaces),
            "Days_to_Death": self._get_xml_text(root, ".//clin_shared:days_to_death", self.xml_namespaces),
            "Days_to_Last_Followup": self._get_xml_text(root, ".//clin_shared:days_to_last_followup", self.xml_namespaces),
            "Histology": self._get_xml_text(root, cfg["histology_path"], self.xml_namespaces),
            "Tumor_Site": self._get_xml_text(root, cfg["tumor_site_path"], self.xml_namespaces),
            "Pathologic_Stage_Raw": self._get_xml_text(root, ".//shared_stage:pathologic_stage", self.xml_namespaces),
            "Neoplasm_Status": self._get_xml_text(root, ".//clin_shared:person_neoplasm_cancer_status", self.xml_namespaces),
            "Informed_Consent": self._get_xml_text(root, ".//clin_shared:informed_consent_verified", self.xml_namespaces),
            "Neoadjuvant_Tx_History": self._get_xml_text(root, ".//shared:history_of_neoadjuvant_treatment", self.xml_namespaces),
            "XML_Filename": xml_file_path.name,
        }
        
        return clinical_record
    
    def load_clinical_data(self) -> pd.DataFrame:
        """
        Load all clinical data from TCGA XML files (adapted from TCGA_STUDY_ANALYSIS.py).
        
        Returns:
        --------
        pd.DataFrame
            Clinical data DataFrame
        """
        logger.info(f"Loading clinical data from XML files in {self.xml_dir}")
        
        if not self.xml_dir.exists():
            logger.warning(f"XML directory not found: {self.xml_dir}")
            return pd.DataFrame()
        
        xml_files = list(self.xml_dir.glob("*.xml"))
        if not xml_files:
            logger.warning(f"No XML files found in {self.xml_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(xml_files)} XML files. Parsing...")
        
        all_clinical_records = []
        for i, xml_file in enumerate(xml_files):
            if (i + 1) % 200 == 0:
                logger.info(f"Parsed {i+1}/{len(xml_files)} XML files...")
            
            record = self.parse_tcga_clinical_xml_file(xml_file)
            if record:
                all_clinical_records.append(record)
        
        if not all_clinical_records:
            logger.warning("No clinical records were successfully parsed")
            return pd.DataFrame()
        
        clinical_df = pd.DataFrame(all_clinical_records)
        logger.info(f"Successfully parsed {len(clinical_df)} clinical records")
        
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
                logger.info(f"Dropped {initial_rows - len(clinical_df)} duplicate Patient_Barcodes")
        
        logger.info(f"Final clinical DataFrame shape: {clinical_df.shape}")
        return clinical_df
    
    def load_sample_sheet(self) -> pd.DataFrame:
        """
        Load and process the consolidated TCGA sample sheet (adapted from TCGA_STUDY_ANALYSIS.py).
        
        Returns:
        --------
        pd.DataFrame
            Processed sample sheet DataFrame
        """
        logger.info(f"Loading sample sheet from {self.sample_sheet_path}")
        
        if not self.sample_sheet_path.exists():
            logger.error(f"Sample sheet not found: {self.sample_sheet_path}")
            return pd.DataFrame()
        
        try:
            sample_sheet_df = pd.read_csv(self.sample_sheet_path, sep='\t', low_memory=False)
            logger.info(f"Successfully loaded sample sheet: {sample_sheet_df.shape}")
            
            # Determine tissue type column name
            tissue_type_col = (
                "Tissue Type" if "Tissue Type" in sample_sheet_df.columns else "Sample Type"
            )
            required_cols = {"Case ID", "File Name", tissue_type_col}
            
            if not required_cols.issubset(sample_sheet_df.columns):
                logger.error(f"Sample sheet missing required columns: {list(required_cols)}")
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
            
            # Derive cancer type from Project_ID
            sample_sheet_processed["Cancer_Type_Derived"] = (
                sample_sheet_processed["Project_ID"].str.split("-").str[1].str.upper()
            )
            
            # Remove duplicates based on File_Name_Root
            initial_rows = len(sample_sheet_processed)
            sample_sheet_processed.drop_duplicates(
                subset=["File_Name_Root"], keep="first", inplace=True
            )
            if len(sample_sheet_processed) < initial_rows:
                logger.info(f"Dropped {initial_rows - len(sample_sheet_processed)} duplicate files")
            
            logger.info(f"Processed sample sheet shape: {sample_sheet_processed.shape}")
            return sample_sheet_processed
            
        except Exception as e:
            logger.error(f"Error loading sample sheet: {e}")
            return pd.DataFrame()
    
    def create_master_metadata(self) -> pd.DataFrame:
        """
        Merge clinical data with sample sheet metadata (adapted from TCGA_STUDY_ANALYSIS.py).
        
        Returns:
        --------
        pd.DataFrame
            Master metadata DataFrame with File_Name_Root as index
        """
        logger.info("Creating master metadata...")
        
        if self.clinical_data.empty and self.sample_sheet_data.empty:
            logger.error("Both clinical and sample sheet data are empty")
            return pd.DataFrame()
        
        if self.sample_sheet_data.empty:
            logger.error("Cannot proceed without sample sheet data")
            return pd.DataFrame()
        
        if self.clinical_data.empty:
            logger.warning("Using only sample sheet data (no clinical data available)")
            master_df = self.sample_sheet_data.copy()
            if "Patient_ID_from_SampleSheet" in master_df.columns:
                master_df.rename(
                    columns={"Patient_ID_from_SampleSheet": "Patient_ID"}, inplace=True
                )
        else:
            # Merge clinical and sample sheet data
            master_df = pd.merge(
                self.sample_sheet_data,
                self.clinical_data,
                left_on="Patient_ID_from_SampleSheet",
                right_on="Patient_Barcode",
                how="left",
            )
            logger.info(f"Master metadata shape after merge: {master_df.shape}")
            
            # Create primary Patient_ID column
            if "Patient_ID_from_SampleSheet" in master_df.columns:
                master_df.rename(
                    columns={"Patient_ID_from_SampleSheet": "Patient_ID"}, inplace=True
                )
            elif "Patient_Barcode" in master_df.columns:
                master_df.rename(columns={"Patient_Barcode": "Patient_ID"}, inplace=True)
        
        # Set File_Name_Root as index
        if "File_Name_Root" in master_df.columns:
            if master_df["File_Name_Root"].duplicated().any():
                logger.warning("Duplicate File_Name_Root found, removing duplicates")
                master_df.drop_duplicates(
                    subset=["File_Name_Root"], keep="first", inplace=True
                )
            master_df.set_index("File_Name_Root", inplace=True, verify_integrity=True)
            logger.info("Set 'File_Name_Root' as index")
        else:
            logger.error("'File_Name_Root' column missing")
            return pd.DataFrame()
        
        logger.info(f"Final master metadata shape: {master_df.shape}")
        return master_df
    
    def detect_available_cancer_types(self) -> List[str]:
        """
        Detect available cancer types from the sample sheet.
        
        Returns:
        --------
        List[str]
            List of available cancer type codes
        """
        if self.sample_sheet_data.empty:
            logger.error("Sample sheet data not loaded")
            return []
        
        cancer_types = self.sample_sheet_data['Cancer_Type_Derived'].dropna().unique().tolist()
        cancer_types = [ct for ct in cancer_types if ct != "UNKNOWN"]
        
        logger.info(f"Detected {len(cancer_types)} cancer types: {cancer_types}")
        return cancer_types
    
    def load_rna_seq_data(self, target_sample_ids: Set[str], 
                         preferred_count_col: str = "tpm_unstranded") -> pd.DataFrame:
        """
        Load RNA-seq data for specific samples (adapted from TCGA_STUDY_ANALYSIS.py).
        
        Parameters:
        -----------
        target_sample_ids : Set[str]
            Set of sample IDs (File_Name_Root) to load
        preferred_count_col : str
            Preferred count column to use
            
        Returns:
        --------
        pd.DataFrame
            RNA-seq counts DataFrame (genes x samples)
        """
        logger.info(f"Loading RNA-seq data for {len(target_sample_ids)} target samples")
        
        if not self.rna_dir.exists() or not self.rna_dir.is_dir():
            logger.error(f"RNA-seq directory not found: {self.rna_dir}")
            return pd.DataFrame()
        
        all_files = [
            f for f in self.rna_dir.iterdir() 
            if f.suffix.lower() in [".tsv", ".txt", ".gz"]
        ]
        
        files_to_process = []
        for file_path in all_files:
            sample_id = file_path.stem.split(".")[0]
            if sample_id in target_sample_ids:
                files_to_process.append(file_path)
        
        logger.info(f"Found {len(files_to_process)} files matching target samples")
        
        if not files_to_process:
            logger.error("No matching RNA-seq files found")
            return pd.DataFrame()
        
        all_sample_data = []
        genes_reference = None
        
        for i, file_path in enumerate(files_to_process):
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(files_to_process)} files...")
            
            sample_id = file_path.stem.split(".")[0]
            
            try:
                # Load the file with comment handling
                if file_path.suffix == ".gz":
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
                    logger.warning(f"'gene_name' column missing in {file_path.name}")
                    continue
                
                if preferred_count_col not in sample_df.columns:
                    logger.warning(f"'{preferred_count_col}' column missing in {file_path.name}")
                    continue
                
                # Filter genes (remove N_ genes and NaN gene names)
                sample_df = sample_df[
                    sample_df["gene_name"].notna()
                    & ~sample_df["gene_name"].astype(str).str.upper().str.startswith("N_")
                ]
                
                if sample_df.empty:
                    logger.warning(f"No valid genes after filtering in {file_path.name}")
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
                logger.warning(f"Error processing {file_path.name}: {e}")
                continue
        
        if not all_sample_data:
            logger.error("No RNA-seq data successfully loaded")
            return pd.DataFrame()
        
        # Combine all samples into a single DataFrame
        raw_rna_counts_df = pd.concat(all_sample_data, axis=1, join="outer").fillna(0)
        raw_rna_counts_df.index.name = "Gene_Symbol"
        
        # Handle duplicate gene symbols
        if raw_rna_counts_df.index.duplicated().any():
            num_duplicates = raw_rna_counts_df.index.duplicated().sum()
            logger.info(f"Found {num_duplicates} duplicate gene symbols. Aggregating by mean expression.")
            rna_counts_df = raw_rna_counts_df.groupby(raw_rna_counts_df.index).mean()
        else:
            logger.info("No duplicate gene symbols found.")
            rna_counts_df = raw_rna_counts_df
        
        logger.info(f"Final RNA-seq data loaded: {rna_counts_df.shape} (genes x samples)")
        return rna_counts_df
    
    def filter_genes_for_cibersortx(self, rna_counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply CIBERSORTx-specific gene filtering.
        
        Parameters:
        -----------
        rna_counts_df : pd.DataFrame
            RNA-seq counts (genes x samples)
            
        Returns:
        --------
        pd.DataFrame
            Filtered RNA-seq data
        """
        logger.info("Applying CIBERSORTx-specific gene filtering...")
        
        initial_genes = len(rna_counts_df)
        logger.info(f"Initial genes: {initial_genes}")
        
        # Filter 1: Remove genes with too many zeros
        max_zeros = int(self.cibersortx_thresholds["max_zero_fraction"] * rna_counts_df.shape[1])
        zero_counts = (rna_counts_df == 0).sum(axis=1)
        genes_to_keep_zeros = zero_counts <= max_zeros
        rna_counts_df = rna_counts_df[genes_to_keep_zeros]
        logger.info(f"After zero filter: {len(rna_counts_df)} genes (removed {initial_genes - len(rna_counts_df)})")
        
        # Filter 2: Minimum expression threshold
        mean_expression = rna_counts_df.mean(axis=1)
        genes_to_keep_expr = mean_expression >= self.cibersortx_thresholds["min_expression_threshold"]
        rna_counts_df = rna_counts_df[genes_to_keep_expr]
        logger.info(f"After expression filter: {len(rna_counts_df)} genes")
        
        # Filter 3: Minimum variance
        gene_variance = rna_counts_df.var(axis=1)
        genes_to_keep_var = gene_variance >= self.cibersortx_thresholds["min_variance_threshold"]
        rna_counts_df = rna_counts_df[genes_to_keep_var]
        logger.info(f"After variance filter: {len(rna_counts_df)} genes")
        
        # Filter 4: Minimum non-zero samples
        nonzero_counts = (rna_counts_df > 0).sum(axis=1)
        genes_to_keep_nonzero = nonzero_counts >= self.cibersortx_thresholds["min_nonzero_samples"]
        rna_counts_df = rna_counts_df[genes_to_keep_nonzero]
        
        final_genes = len(rna_counts_df)
        logger.info(f"Final genes after all filters: {final_genes} ({initial_genes - final_genes} removed)")
        
        return rna_counts_df
    
    
    
    def deduplicate_patients_for_tumor_samples(self, tumor_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure only one tumor sample per patient by selecting the first sample alphabetically.
        
        Parameters:
        -----------
        tumor_metadata : pd.DataFrame
            Tumor sample metadata with Patient_ID column
            
        Returns:
        --------
        pd.DataFrame
            Deduplicated tumor metadata with one sample per patient
        """
        logger.info("Deduplicating patients to ensure one tumor sample per patient...")
        
        initial_samples = len(tumor_metadata)
        
        # Check if Patient_ID column exists
        if "Patient_ID" not in tumor_metadata.columns:
            logger.warning("Patient_ID column not found - cannot deduplicate by patient")
            return tumor_metadata
        
        # Count samples per patient before deduplication
        samples_per_patient = tumor_metadata['Patient_ID'].value_counts()
        patients_with_multiple_samples = samples_per_patient[samples_per_patient > 1]
        
        if len(patients_with_multiple_samples) > 0:
            logger.info(f"Found {len(patients_with_multiple_samples)} patients with multiple tumor samples:")
            logger.info(f"  - Total patients with multiple samples: {len(patients_with_multiple_samples)}")
            logger.info(f"  - Max samples per patient: {patients_with_multiple_samples.max()}")
            logger.info(f"  - Extra samples to remove: {(patients_with_multiple_samples - 1).sum()}")
            
            # Show top patients with most samples
            top_multi_patients = patients_with_multiple_samples.head(5)
            for patient_id, count in top_multi_patients.items():
                logger.info(f"    Patient {patient_id}: {count} tumor samples")
        else:
            logger.info("No patients with multiple tumor samples found")
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
        
        logger.info(f"Patient deduplication complete:")
        logger.info(f"  - Initial tumor samples: {initial_samples}")
        logger.info(f"  - Final tumor samples: {final_samples}")
        logger.info(f"  - Removed samples: {removed_samples}")
        logger.info(f"  - Unique patients: {deduplicated_metadata['Patient_ID'].nunique()}")
        
        return deduplicated_metadata

    def create_tumor_subset(self, rna_counts_df: pd.DataFrame, 
                           metadata_df: pd.DataFrame, cancer_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create tumor-only subset of data with patient-level deduplication.
        
        Parameters:
        -----------
        rna_counts_df : pd.DataFrame
            RNA-seq counts (genes x samples)
        metadata_df : pd.DataFrame
            Sample metadata
        cancer_type : str
            Target cancer type
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tumor RNA-seq data and tumor metadata (one sample per patient)
        """
        logger.info(f"Creating tumor-only subset for {cancer_type}")
        
        # Filter for cancer type
        cancer_metadata = metadata_df[
            metadata_df["Cancer_Type_Derived"] == cancer_type.upper()
        ].copy()
        
        if cancer_metadata.empty:
            logger.error(f"No samples found for cancer type {cancer_type}")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Total {cancer_type} samples in metadata: {len(cancer_metadata)}")
        
        # Filter for tumor samples only
        if "Tissue_Type" in cancer_metadata.columns:
            tumor_mask = (
                cancer_metadata["Tissue_Type"].astype(str).str.contains("Tumor", case=False, na=False)
            )
            tumor_metadata = cancer_metadata[tumor_mask].copy()
            logger.info(f"Tumor samples after tissue type filter: {len(tumor_metadata)}")
        else:
            logger.warning("'Tissue_Type' column not found, using all samples")
            tumor_metadata = cancer_metadata.copy()
        
        if tumor_metadata.empty:
            logger.error(f"No tumor samples found for {cancer_type}")
            return pd.DataFrame(), pd.DataFrame()
        
        # NEW: Deduplicate patients to ensure one tumor sample per patient
        tumor_metadata = self.deduplicate_patients_for_tumor_samples(tumor_metadata)
        
        if tumor_metadata.empty:
            logger.error(f"No tumor samples remain after patient deduplication for {cancer_type}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Find common samples between RNA-seq data and tumor metadata
        common_samples = list(set(rna_counts_df.columns) & set(tumor_metadata.index))
        if not common_samples:
            logger.error("No common samples between RNA-seq data and tumor metadata")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Common samples with RNA-seq data: {len(common_samples)}")
        
        # Subset data to common tumor samples
        tumor_rna = rna_counts_df[common_samples].copy()
        tumor_metadata = tumor_metadata.loc[common_samples].copy()
        
        # Add cancer type info
        tumor_metadata["Cancer_Type"] = cancer_type.upper()
        
        logger.info(f"Final tumor subset: {len(common_samples)} samples x {tumor_rna.shape[0]} genes")
        logger.info(f"Final unique patients: {tumor_metadata['Patient_ID'].nunique()}")
        
        return tumor_rna, tumor_metadata
    
    def create_cibersortx_mixture_file(self, tumor_rna_df: pd.DataFrame, 
                                     tumor_metadata_df: pd.DataFrame, 
                                     cancer_type: str) -> str:
        """
        Create CIBERSORTx mixture file for a cancer type.
        
        Parameters:
        -----------
        tumor_rna_df : pd.DataFrame
            Tumor RNA-seq data (genes x samples)
        tumor_metadata_df : pd.DataFrame
            Tumor metadata
        cancer_type : str
            Cancer type code
            
        Returns:
        --------
        str
            Path to created mixture file
        """
        logger.info(f"Creating CIBERSORTx mixture file for {cancer_type}")
        
        # Prepare mixture data in CIBERSORTx format
        # CIBERSORTx expects: samples as columns, genes as rows, with gene symbols as first column
        mixture_df = tumor_rna_df.copy()
        
        # Data is already in correct format: genes as rows, samples as columns
        
        # Add gene symbols as first column (required by CIBERSORTx)
        mixture_df.insert(0, "GeneSymbol", mixture_df.index)
        
        # Create output filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{cancer_type}_tumor_mixture_for_cibersortx_harmonized_{timestamp}.txt"
        output_path = self.output_dir / output_filename
        
        # Save mixture file
        logger.info(f"Saving mixture file: {output_path}")
        mixture_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
        
        # Verify file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Mixture file saved successfully. Size: {file_size_mb:.1f} MB")
        
        # Create metadata summary file
        metadata_summary_path = self.output_dir / f"{cancer_type}_mixture_metadata_summary_{timestamp}.txt"
        with open(metadata_summary_path, 'w') as f:
            f.write(f"CIBERSORTx Mixture File Summary: {cancer_type}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Mixture file: {output_filename}\n")
            f.write(f"Genes: {mixture_df.shape[0]-1:,}\n")  # -1 for header
            f.write(f"Samples: {mixture_df.shape[1]-1:,}\n")  # -1 for gene column
            f.write(f"File size: {file_size_mb:.1f} MB\n\n")
            
            # Data summary section
            f.write("Data Summary:\n")
            f.write(f"  Total tumor samples (one per patient): {len(tumor_metadata_df)}\n")
            f.write(f"  Unique patients: {tumor_metadata_df['Patient_ID'].nunique()}\n\n")
            
            f.write("Sample metadata summary:\n")
            f.write(f"  Total tumor samples: {len(tumor_metadata_df)}\n")
            if "Gender" in tumor_metadata_df.columns:
                gender_counts = tumor_metadata_df["Gender"].value_counts()
                f.write(f"  Gender distribution: {dict(gender_counts)}\n")
            if "Vital_Status" in tumor_metadata_df.columns:
                vital_counts = tumor_metadata_df["Vital_Status"].value_counts()
                f.write(f"  Vital status: {dict(vital_counts)}\n")
            if "Age_at_Diagnosis" in tumor_metadata_df.columns:
                age_stats = tumor_metadata_df["Age_at_Diagnosis"].describe()
                f.write(f"  Age at diagnosis: {age_stats['mean']:.1f} +- {age_stats['std']:.1f} years\n")
            
            f.write(f"\nExpression data summary:\n")
            f.write(f"  Expression range: {tumor_rna_df.values.min():.3f} to {tumor_rna_df.values.max():.3f}\n")
            f.write(f"  Mean expression: {tumor_rna_df.values.mean():.3f}\n")
            f.write(f"  Non-zero entries: {(tumor_rna_df != 0).sum().sum():,}\n")
            
            f.write(f"\nCIBERSORTx Analysis Instructions:\n")
            f.write(f"1. Upload {output_filename} to CIBERSORTx as mixture file\n")
            f.write(f"2. Use your NK cell signature matrix\n")
            f.write(f"3. Run deconvolution analysis\n")
            f.write(f"4. Download results for downstream analysis\n")
        
        logger.info(f"Metadata summary saved: {metadata_summary_path}")
        
        return str(output_path)
    
    def process_cancer_type(self, cancer_type: str) -> Optional[str]:
        """
        Process a single cancer type end-to-end.
        
        Parameters:
        -----------
        cancer_type : str
            Cancer type code
            
        Returns:
        --------
        Optional[str]
            Path to mixture file or None if failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {cancer_type}")
        logger.info(f"{'='*60}")
        
        try:
            # Get samples for this cancer type from master metadata
            cancer_samples = self.master_metadata[
                self.master_metadata["Cancer_Type_Derived"] == cancer_type.upper()
            ]
            
            if cancer_samples.empty:
                logger.warning(f"No samples found for cancer type {cancer_type}")
                return None
            
            target_sample_ids = set(cancer_samples.index)
            logger.info(f"Found {len(target_sample_ids)} samples for {cancer_type}")
            
            # Load RNA-seq data
            rna_data = self.load_rna_seq_data(
                target_sample_ids, 
                self.cibersortx_thresholds["preferred_rna_count_column"]
            )
            if rna_data.empty:
                logger.error(f"Failed to load RNA-seq data for {cancer_type}")
                return None
            
            # Apply CIBERSORTx-specific filtering
            filtered_rna_data = self.filter_genes_for_cibersortx(rna_data)
            if filtered_rna_data.empty:
                logger.error(f"No genes remain after filtering for {cancer_type}")
                return None
            
            # Create tumor-only subset
            tumor_rna, tumor_metadata = self.create_tumor_subset(
                filtered_rna_data, self.master_metadata, cancer_type
            )
            if tumor_rna.empty:
                logger.error(f"Failed to create tumor subset for {cancer_type}")
                return None
            
            # Create CIBERSORTx mixture file
            mixture_file_path = self.create_cibersortx_mixture_file(
                tumor_rna, tumor_metadata, cancer_type
            )
            
            logger.info(f"Successfully processed {cancer_type}")
            logger.info(f"   Mixture file: {mixture_file_path}")
            logger.info(f"   Genes: {tumor_rna.shape[0]:,}")
            logger.info(f"   Tumor samples: {tumor_rna.shape[1]:,}")
            logger.info(f"   Unique patients: {tumor_metadata['Patient_ID'].nunique()}")
            
            return mixture_file_path
            
        except Exception as e:
            logger.error(f"Failed to process {cancer_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_full_pipeline(self) -> Dict[str, str]:
        """
        Run the complete raw data to CIBERSORTx mixture pipeline.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping cancer types to mixture file paths
        """
        logger.info("Starting TCGA Raw Data to CIBERSORTx Mixture Pipeline")
        logger.info("=" * 70)
        
        # Step 1: Load clinical data
        logger.info("\nStep 1: Loading clinical data from XML files...")
        self.clinical_data = self.load_clinical_data()
        
        # Step 2: Load sample sheet
        logger.info("\nStep 2: Loading GDC sample sheet...")
        self.sample_sheet_data = self.load_sample_sheet()
        if self.sample_sheet_data.empty:
            logger.error("Failed to load sample sheet")
            return {}
        
        # Step 3: Create master metadata
        logger.info("\nStep 3: Creating master metadata...")
        self.master_metadata = self.create_master_metadata()
        if self.master_metadata.empty:
            logger.error("Failed to create master metadata")
            return {}
        
        # Step 4: Detect available cancer types
        logger.info("\nStep 4: Detecting available cancer types...")
        self.cancer_types = self.detect_available_cancer_types()
        if not self.cancer_types:
            logger.error("No cancer types detected")
            return {}
        
        # Step 5: Process each cancer type
        logger.info(f"\nStep 5: Processing {len(self.cancer_types)} cancer types...")
        results = {}
        failed_types = []
        
        for cancer_type in self.cancer_types:
            result = self.process_cancer_type(cancer_type)
            if result:
                results[cancer_type] = result
            else:
                failed_types.append(cancer_type)
        
        # Summary
        logger.info(f"\nPipeline Complete!")
        logger.info(f"Successfully processed: {len(results)} cancer types")
        logger.info(f"Failed: {len(failed_types)} cancer types")
        
        if results:
            logger.info(f"\nCIBERSORTx Mixture Files Ready:")
            for cancer_type, mixture_file in results.items():
                logger.info(f"  {cancer_type}: {Path(mixture_file).name}")
        
        if failed_types:
            logger.info(f"\nFailed Cancer Types: {', '.join(failed_types)}")
        
        # Collect sample count information
        sample_count_summary = {}
        
        for cancer_type in results.keys():
            mixture_file = results[cancer_type]
            try:
                # Read mixture file to get actual sample count
                import pandas as pd
                df = pd.read_csv(mixture_file, sep='\t', nrows=1)
                actual_samples = df.shape[1] - 1  # -1 for gene column
                
                sample_count_summary[cancer_type] = {
                    'actual_samples': actual_samples,
                    'deduplication_status': 'completed'
                }
                
            except Exception as e:
                logger.warning(f"Could not read sample count from {mixture_file}: {e}")
                sample_count_summary[cancer_type] = {
                    'actual_samples': "Unknown",
                    'deduplication_status': 'unknown'
                }
        
        # Save pipeline summary
        summary_file = self.output_dir / "pipeline_summary.json"
        pipeline_summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'successful_cancer_types': list(results.keys()),
            'failed_cancer_types': failed_types,
            'total_cancer_types_detected': len(self.cancer_types),
            'clinical_records_count': len(self.clinical_data) if not self.clinical_data.empty else 0,
            'sample_sheet_records_count': len(self.sample_sheet_data) if not self.sample_sheet_data.empty else 0,
            'cibersortx_thresholds': self.cibersortx_thresholds,
            'sample_count_summary': sample_count_summary,
            'patient_deduplication': 'enabled - one tumor sample per patient'
        }
        
        with open(summary_file, 'w') as f:
            json.dump(pipeline_summary, f, indent=2)
        
        logger.info(f"\nPipeline summary saved to: {summary_file}")
        
        logger.info("\n" + "=" * 70)
        logger.info("NEXT STEPS FOR CIBERSORTX ANALYSIS:")
        logger.info("=" * 70)
        logger.info("1. Upload mixture files to CIBERSORTx online platform")
        logger.info("2. Use your NK cell signature matrix (e.g., Tang-derived)")
        logger.info("3. Run deconvolution analysis")
        logger.info("4. Download results for downstream analysis")
        logger.info("5. Correlate NK infiltration with clinical outcomes")
        
        return results

def main():
    """Main execution function."""
    
    # Configuration
    TCGA_BASE_DIR = "TCGAdata"
    OUTPUT_DIR = "outputs/tcga_cibersortx_mixtures"
    SAMPLE_SHEET_FILENAME = "gdc_sample_sheet.2025-06-26.tsv"
    
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir=TCGA_BASE_DIR,
        output_dir=OUTPUT_DIR,
        sample_sheet_filename=SAMPLE_SHEET_FILENAME
    )
    
    results = processor.run_full_pipeline()
    
    if results:
        print(f"\nPipeline completed successfully!")
        print(f"Processed {len(results)} cancer types")
        print(f"Mixture files saved to: {OUTPUT_DIR}")
    else:
        print(f"\nPipeline failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 