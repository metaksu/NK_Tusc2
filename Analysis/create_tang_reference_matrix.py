#!/usr/bin/env python3
"""
Hybrid Atlas-Tang Reference Matrix Generator for CIBERSORTx
============================================================

This script generates a comprehensive reference matrix combining breast cancer atlas
and Tang NK datasets for CIBERSORTx deconvolution analysis.

The script creates:
- 11 Atlas immune minor subtypes (T/B/Myeloid cells) 
- 2 Tang NK core signatures (Cytotoxic including TaNK/Bright)
- 5 Atlas non-immune major subtypes (Cancer/Normal Epithelial/Endothelial/CAFs/PVL)

Total: 18 signatures × 400 cells = 7,200 cells

The script:
1. Loads and harmonizes both Atlas and Tang datasets (gene intersection)
2. Maps Atlas cell types to unified signatures (immune minor + non-immune major)
3. Maps Tang NK subtypes to 2 biologically meaningful core signatures
4. Samples 400 cells per unified signature (prevalence-weighted for merged types)
5. Exports in CIBERSORTx format (genes as rows, cells as columns)

Reference: This hybrid approach combines Atlas tissue diversity with Tang NK expertise
for comprehensive breast cancer tumor microenvironment deconvolution.
"""

# === ENVIRONMENT SETUP FOR GPU CONFLICTS ===
# Fix GPU/CUDA conflicts that can cause scanpy hanging issues
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA for scanpy
os.environ["NUMBA_DISABLE_CUDA"] = "1"   # Disable CUDA for numba
os.environ["OMP_NUM_THREADS"] = "1"      # Prevent threading conflicts

# Standard imports
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from scipy import sparse
import logging

# Import scanpy with proper error handling
try:
    print("Importing scanpy...")
    import scanpy as sc
    
    # Configure scanpy settings immediately (mirroring working NK analysis scripts)
    sc.settings.verbosity = 1  # Reduce verbosity
    sc.settings.autoshow = False  # Prevent automatic plot display
    print(f"✓ Scanpy {sc.__version__} imported and configured successfully")
    
except Exception as e:
    print(f"✗ CRITICAL ERROR: Scanpy import failed: {e}")
    print("This may be due to GPU/CUDA conflicts or dependency issues.")
    print("Please check your environment and try again.")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Configuration Constants - Mirroring the main project setup
# These constants are taken directly from the main NK analysis scripts

# === DATASET CONFIGURATION ===
# File paths for both datasets
TANG_COMBINED_H5AD_FILE = "data/processed/comb_CD56_CD16_NK.h5ad"
ATLAS_METADATA_FILE = "data/raw/breast_cancer_atlas/Whole_miniatlas_meta.csv"
ATLAS_MATRIX_DIR = "data/raw/breast_cancer_atlas/"

# Key Metadata Columns for Tang Combined Dataset (exact column names from main project)
TANG_TISSUE_COL = "meta_tissue_in_paper"      # Primary tissue context (Blood/Tumor/Normal/Other tissue)
TANG_TISSUE_BLOOD_COL = "meta_tissue"         # Blood-specific dataset column
TANG_MAJORTYPE_COL = "Majortype"              # CD56bright/dim + CD16 high/low combinations
TANG_CELLTYPE_COL = "celltype"                # Fine-grained NK subtypes (14 subtypes total)
TANG_HISTOLOGY_COL = "meta_histology"         # Cancer type information (25 cancer types)
TANG_BATCH_COL = "batch"                      # Batch information for integration
TANG_DATASETS_COL = "datasets"                # Source dataset information
TANG_PATIENT_ID_COL = "meta_patientID"        # Patient identifier

# === UNIFIED SIGNATURE CONFIGURATION ===
# Atlas immune cell types to keep as minor subtypes (using exact names from Atlas metadata)
ATLAS_IMMUNE_MINOR_TYPES = {
    "T cells CD4+": "T_cells_CD4+",
    "T cells CD8+": "T_cells_CD8+", 
    "NKT cells": "NKT_cells",
    "Cycling T-cells": "T_cells_Cycling",
    "B cells Memory": "B_cells_Memory",
    "B cells Naive": "B_cells_Naive", 
    "Plasmablasts": "Plasmablasts",
    "Macrophage": "Macrophage",
    "Monocyte": "Monocyte",
    "DCs": "DCs",
    "Cycling_Myeloid": "Myeloid_Cycling"
}

# Atlas non-immune cell types to merge as major subtypes
ATLAS_NONIMMUNE_MAJOR_MAPPING = {
    "Cancer_Epithelial": ["Cancer Basal SC", "Cancer Cycling", "Cancer Her2 SC", "Cancer LumA SC", "Cancer LumB SC"],
    "Normal_Epithelial": ["Luminal Progenitors", "Mature Luminal", "Myoepithelial"],
    "Endothelial": ["Endothelial ACKR1", "Endothelial CXCL12", "Endothelial RGS5", "Endothelial Lymphatic LYVE1"],
    "CAFs": ["CAFs myCAF-like", "CAFs MSC iCAF-like"],
    "PVL": ["PVL Differentiated", "PVL Immature", "Cycling PVL"]
}

# Tang NK signatures (from existing Tang subtypes)
TANG_NK_MAPPING = {
    "NK_Cytotoxic": ["CD56dimCD16hi-c3-ZNF90", "CD56dimCD16hi-c2-CX3CR1", "CD56dimCD16hi-c4-NFKBIA", 
                     "CD56dimCD16hi-c7-NR4A3", "CD56dimCD16hi-c8-KLRC2", "CD56dimCD16hi-c1-IL32", 
                     "CD56dimCD16hi-c5-MKI67", "CD56dimCD16hi-c6-DNAJB1"],  # Added DNAJB1+ TaNK cells
    "NK_Bright": ["CD56brightCD16lo-c3-CCL3", "CD56brightCD16lo-c5-CREM", "CD56brightCD16lo-c1-GZMH",
                  "CD56brightCD16lo-c4-IL7R", "CD56brightCD16lo-c2-IL7R-RGS1lo", "CD56brightCD16hi"]
}

# Combined unified signature mapping
UNIFIED_SIGNATURE_MAPPING = {**ATLAS_IMMUNE_MINOR_TYPES, **ATLAS_NONIMMUNE_MAJOR_MAPPING, **TANG_NK_MAPPING}

# === UNIFIED SIGNATURE DESCRIPTIONS ===
UNIFIED_SIGNATURE_DESCRIPTIONS = {
    # Atlas immune minor types
    "T_cells_CD4+": "CD4+ T helper cells from Atlas",
    "T_cells_CD8+": "CD8+ cytotoxic T cells from Atlas",
    "NKT_cells": "Natural killer T cells from Atlas", 
    "T_cells_Cycling": "Proliferating T cells from Atlas",
    "B_cells_Memory": "Memory B cells from Atlas",
    "B_cells_Naive": "Naive B cells from Atlas",
    "Plasmablasts": "Antibody-secreting plasma cells from Atlas",
    "Macrophage": "Tissue macrophages from Atlas",
    "Monocyte": "Circulating monocytes from Atlas", 
    "DCs": "Dendritic cells from Atlas",
    "Myeloid_Cycling": "Proliferating myeloid cells from Atlas",
    # Tang NK signatures
    "NK_Cytotoxic": "CD56dimCD16hi NK cells including cytotoxic and exhausted/TaNK subtypes from Tang",
    "NK_Bright": "Cytokine-producing CD56brightCD16lo NK cells from Tang",
    # Atlas non-immune major types  
    "Cancer_Epithelial": "Malignant epithelial cells (all subtypes) from Atlas",
    "Normal_Epithelial": "Normal epithelial cells (all subtypes) from Atlas",
    "Endothelial": "Vascular endothelial cells (all subtypes) from Atlas",
    "CAFs": "Cancer-associated fibroblasts (all subtypes) from Atlas",
    "PVL": "Perivascular-like cells (all subtypes) from Atlas"
}

# === PROCESSING PARAMETERS ===
TARGET_CELLS_PER_SIGNATURE = 400              # Target cells per unified signature
MIN_CELLS_PER_SIGNATURE = 50                  # Minimum cells required for inclusion (higher for combined signatures)
RANDOM_SEED = 42                               # For reproducibility
MIN_GENES_PER_CELL = 200                       # Minimum genes per cell for filtering

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = Path("outputs/signature_matrices/CIBERSORTx_Input_Files")
OUTPUT_FILENAME = "Hybrid_Atlas_Tang_signatures_18types_400cells_per_signature.txt"
SUMMARY_FILENAME = "Hybrid_Atlas_Tang_generation_summary.txt"


def load_and_harmonize_datasets():
    """
    Load and harmonize both Atlas and Tang datasets for unified reference matrix.
    
    Returns:
    --------
    adata_combined : AnnData or None
        Combined harmonized dataset, or None if loading fails
    """
    logger.info("=" * 70)
    logger.info("LOADING AND HARMONIZING ATLAS + TANG DATASETS")
    logger.info("=" * 70)
    
    # Load Atlas dataset (10X format + metadata)
    try:
        logger.info("Loading Atlas breast cancer dataset...")
        
        # Load 10X format data
        adata_atlas = sc.read_10x_mtx(
            ATLAS_MATRIX_DIR,
            var_names='gene_symbols',
            cache=True
        )
        # Make variable names unique (handle duplicate gene names)
        if hasattr(adata_atlas, 'var_names_unique'):
            adata_atlas.var_names_unique()
        else:
            # Alternative method for older scanpy versions
            adata_atlas.var_names = pd.Index(adata_atlas.var_names).astype(str)
            adata_atlas.var_names = pd.Index([f"{name}_{i}" if name in adata_atlas.var_names[:i] else name 
                                            for i, name in enumerate(adata_atlas.var_names)])
        logger.info(f"Atlas expression data shape: {adata_atlas.shape}")
        
        # Load metadata
        metadata_df = pd.read_csv(ATLAS_METADATA_FILE, low_memory=False)
        metadata_df = metadata_df[metadata_df['NAME'] != 'TYPE'].copy()  # Remove header row
        logger.info(f"Atlas metadata shape: {metadata_df.shape}")
        
        # Match cells and add metadata
        adata_atlas.obs_names = [name.replace('-', '_') for name in adata_atlas.obs_names]
        metadata_df['NAME'] = metadata_df['NAME'].str.replace('-', '_')
        
        # Add metadata to Atlas dataset
        for col in ['Patient', 'celltype_major', 'celltype_minor']:
            if col in metadata_df.columns:
                cell_to_meta = dict(zip(metadata_df['NAME'], metadata_df[col]))
                adata_atlas.obs[col] = [cell_to_meta.get(cell, 'Unknown') for cell in adata_atlas.obs_names]
        
        # Add dataset source
        adata_atlas.obs['dataset_source'] = 'Atlas'
        logger.info(f"Atlas dataset loaded: {adata_atlas.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load Atlas dataset: {e}")
        return None
    
    # Load Tang dataset
    try:
        logger.info(f"Loading Tang NK dataset from: {TANG_COMBINED_H5AD_FILE}")
        if not os.path.exists(TANG_COMBINED_H5AD_FILE):
            logger.error(f"Tang dataset file not found: {TANG_COMBINED_H5AD_FILE}")
            return None
            
        adata_tang = sc.read_h5ad(TANG_COMBINED_H5AD_FILE)
        adata_tang.obs['dataset_source'] = 'Tang'
        logger.info(f"Tang dataset loaded: {adata_tang.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load Tang dataset: {e}")
        return None
    
    # Harmonize datasets (gene intersection)
    logger.info("Harmonizing datasets (finding gene intersection)...")
    atlas_genes = set(adata_atlas.var_names)
    tang_genes = set(adata_tang.var_names)
    common_genes = atlas_genes.intersection(tang_genes)
    
    logger.info(f"Atlas genes: {len(atlas_genes):,}")
    logger.info(f"Tang genes: {len(tang_genes):,}")
    logger.info(f"Common genes: {len(common_genes):,} ({len(common_genes)/min(len(atlas_genes), len(tang_genes))*100:.1f}% overlap)")
    
    if len(common_genes) < 5000:
        logger.warning(f"Low gene overlap ({len(common_genes)}). Results may be suboptimal.")
    
    # Subset to common genes
    common_genes_list = sorted(list(common_genes))
    adata_atlas = adata_atlas[:, common_genes_list].copy()
    adata_tang = adata_tang[:, common_genes_list].copy()
    
    # Normalize both datasets to same scale (log-normalized TPM)
    logger.info("Normalizing datasets to same scale...")
    for adata, name in [(adata_atlas, 'Atlas'), (adata_tang, 'Tang')]:
        if adata.X.max() > 50:  # Raw counts or high TPM
            logger.info(f"  {name}: Normalizing and log-transforming")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            logger.info(f"  {name}: Already normalized (max: {adata.X.max():.1f})")
    
    # Concatenate datasets
    logger.info("Concatenating datasets...")
    adata_combined = adata_atlas.concatenate(adata_tang, batch_key='dataset', index_unique=None)
    
    logger.info(f"Combined dataset shape: {adata_combined.shape}")
    logger.info(f"Dataset composition:")
    dataset_counts = adata_combined.obs['dataset_source'].value_counts()
    for dataset, count in dataset_counts.items():
        logger.info(f"  {dataset}: {count:,} cells ({count/adata_combined.n_obs*100:.1f}%)")
    
    return adata_combined


def preprocess_combined_data(adata_combined):
    """
    Preprocess combined Atlas-Tang data for reference matrix generation.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined harmonized dataset
        
    Returns:
    --------
    adata_combined : AnnData
        Preprocessed combined dataset
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING COMBINED ATLAS-TANG DATA")
    logger.info("=" * 70)
    
    logger.info(f"Initial combined shape: {adata_combined.shape}")
    
    # Store normalized data as raw (already log-normalized from harmonization)
    logger.info("Storing current log-normalized data as .raw for downstream analysis...")
    adata_combined.raw = adata_combined.copy()
    
    # Basic cell filtering
    logger.info(f"Filtering cells with fewer than {MIN_GENES_PER_CELL} genes...")
    sc.pp.filter_cells(adata_combined, min_genes=MIN_GENES_PER_CELL)
    logger.info(f"Shape after cell filtering: {adata_combined.shape}")
    
    # Show dataset composition after filtering
    logger.info("Dataset composition after preprocessing:")
    dataset_counts = adata_combined.obs['dataset_source'].value_counts()
    for dataset, count in dataset_counts.items():
        logger.info(f"  {dataset}: {count:,} cells ({count/adata_combined.n_obs*100:.1f}%)")
    
    return adata_combined


def create_unified_signatures(adata_combined):
    """
    Create unified signature labels by mapping Atlas and Tang cell types to final signatures.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined Atlas-Tang dataset
        
    Returns:
    --------
    adata_combined : AnnData
        Combined dataset with new 'unified_signature' column added
    """
    logger.info("=" * 70)
    logger.info("CREATING UNIFIED SIGNATURES")
    logger.info("=" * 70)
    
    # Initialize unified signature column
    adata_combined.obs['unified_signature'] = 'Unmapped'
    
    # Process Atlas cells
    atlas_mask = adata_combined.obs['dataset_source'] == 'Atlas'
    atlas_cells = adata_combined.obs[atlas_mask]
    logger.info(f"Processing {atlas_mask.sum():,} Atlas cells...")
    
    # Map Atlas immune minor types (keep as-is)
    for atlas_type, unified_sig in ATLAS_IMMUNE_MINOR_TYPES.items():
        type_mask = atlas_cells['celltype_minor'] == atlas_type
        if type_mask.any():
            count = type_mask.sum()
            adata_combined.obs.loc[atlas_mask & (adata_combined.obs['celltype_minor'] == atlas_type), 'unified_signature'] = unified_sig
            logger.info(f"  {atlas_type} → {unified_sig}: {count:,} cells")
    
    # Map Atlas non-immune to major types (merge multiple subtypes)
    for major_sig, minor_types in ATLAS_NONIMMUNE_MAJOR_MAPPING.items():
        total_count = 0
        for minor_type in minor_types:
            type_mask = atlas_cells['celltype_minor'] == minor_type
            if type_mask.any():
                count = type_mask.sum()
                total_count += count
                adata_combined.obs.loc[atlas_mask & (adata_combined.obs['celltype_minor'] == minor_type), 'unified_signature'] = major_sig
        if total_count > 0:
            logger.info(f"  {major_sig} (merged): {total_count:,} cells from {len(minor_types)} subtypes")
    
    # Process Tang cells
    tang_mask = adata_combined.obs['dataset_source'] == 'Tang'
    tang_cells = adata_combined.obs[tang_mask]
    logger.info(f"Processing {tang_mask.sum():,} Tang cells...")
    
    # Map Tang NK subtypes to core NK signatures
    for nk_sig, tang_subtypes in TANG_NK_MAPPING.items():
        total_count = 0
        for tang_subtype in tang_subtypes:
            if TANG_CELLTYPE_COL in tang_cells.columns:
                type_mask = tang_cells[TANG_CELLTYPE_COL] == tang_subtype
                if type_mask.any():
                    count = type_mask.sum()
                    total_count += count
                    adata_combined.obs.loc[tang_mask & (adata_combined.obs[TANG_CELLTYPE_COL] == tang_subtype), 'unified_signature'] = nk_sig
        if total_count > 0:
            logger.info(f"  {nk_sig}: {total_count:,} cells from {len(tang_subtypes)} Tang subtypes")
    
    # Report final unified signature distribution
    unified_counts = adata_combined.obs['unified_signature'].value_counts()
    unmapped_count = (adata_combined.obs['unified_signature'] == 'Unmapped').sum()
    
    logger.info("\nUnified signature distribution:")
    total_mapped = len(adata_combined) - unmapped_count
    
    for sig_type in ['Atlas Immune Minor', 'Tang NK', 'Atlas Non-Immune Major']:
        logger.info(f"\n{sig_type}:")
        
        if sig_type == 'Atlas Immune Minor':
            sigs = list(ATLAS_IMMUNE_MINOR_TYPES.values())
        elif sig_type == 'Tang NK':
            sigs = list(TANG_NK_MAPPING.keys())
        else:  # Atlas Non-Immune Major
            sigs = list(ATLAS_NONIMMUNE_MAJOR_MAPPING.keys())
        
        for sig in sigs:
            if sig in unified_counts:
                count = unified_counts[sig]
                percentage = (count / total_mapped) * 100 if total_mapped > 0 else 0
                logger.info(f"  {sig}: {count:,} cells ({percentage:.1f}%)")
    
    if unmapped_count > 0:
        logger.warning(f"\nUnmapped cells: {unmapped_count:,} ({unmapped_count/len(adata_combined)*100:.1f}%)")
        # Show unmapped cell types for debugging
        unmapped_mask = adata_combined.obs['unified_signature'] == 'Unmapped'
        if atlas_mask.any():
            unmapped_atlas = adata_combined.obs[unmapped_mask & atlas_mask]['celltype_minor'].value_counts()
            if len(unmapped_atlas) > 0:
                logger.warning(f"  Unmapped Atlas types: {dict(unmapped_atlas.head())}")
        if tang_mask.any() and TANG_CELLTYPE_COL in adata_combined.obs.columns:
            unmapped_tang = adata_combined.obs[unmapped_mask & tang_mask][TANG_CELLTYPE_COL].value_counts()
            if len(unmapped_tang) > 0:
                logger.warning(f"  Unmapped Tang types: {dict(unmapped_tang.head())}")
    
    return adata_combined


def analyze_dataset_composition(adata_combined):
    """
    Analyze composition of combined Atlas-Tang dataset by unified signatures.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined dataset with unified signatures
        
    Returns:
    --------
    adata_combined : AnnData
        Combined dataset (unchanged, just analyzed)
    """
    logger.info("=" * 70)
    logger.info("ANALYZING COMBINED DATASET COMPOSITION")
    logger.info("=" * 70)
    
    # Dataset source distribution
    dataset_counts = adata_combined.obs['dataset_source'].value_counts()
    logger.info("Dataset source distribution:")
    for dataset, count in dataset_counts.items():
        logger.info(f"  {dataset}: {count:,} cells ({count/adata_combined.n_obs*100:.1f}%)")
    
    # Unified signature distribution by dataset source
    logger.info("\nUnified signature breakdown by dataset source:")
    sig_dataset_crosstab = pd.crosstab(
        adata_combined.obs['unified_signature'], 
        adata_combined.obs['dataset_source'], 
        margins=True
    )
    logger.info(f"\n{sig_dataset_crosstab}")
    
    # For Tang cells, show tissue distribution if available
    tang_mask = adata_combined.obs['dataset_source'] == 'Tang'
    if tang_mask.any() and TANG_TISSUE_COL in adata_combined.obs.columns:
        logger.info(f"\nTang cells tissue distribution:")
        tang_tissue_counts = adata_combined.obs[tang_mask][TANG_TISSUE_COL].value_counts()
        for tissue, count in tang_tissue_counts.items():
            logger.info(f"  {tissue}: {count:,} cells ({count/tang_mask.sum()*100:.1f}% of Tang)")
    
    logger.info(f"\nTotal cells available for sampling: {adata_combined.shape[0]:,}")
    
    return adata_combined


def sample_cells_per_unified_signature(adata_combined):
    """
    Sample up to TARGET_CELLS_PER_SIGNATURE cells per unified signature from combined dataset.
    For Tang NK signatures, sampling is weighted by subtype prevalence to maintain biological composition.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined Atlas-Tang dataset with unified signatures
        
    Returns:
    --------
    adata_sampled : AnnData
        Dataset with sampled cells per unified signature
    final_counts : dict
        Cell counts per unified signature
    """
    logger.info("=" * 70)
    logger.info("SAMPLING CELLS PER UNIFIED SIGNATURE (PREVALENCE-WEIGHTED)")
    logger.info("=" * 70)
    
    np.random.seed(RANDOM_SEED)
    
    # Get available unified signatures (filter out unmapped)
    available_signatures = adata_combined.obs['unified_signature'].dropna().unique()
    available_signatures = [sig for sig in available_signatures if sig != 'Unmapped']
    logger.info(f"Available unified signatures: {len(available_signatures)} total")
    
    # Filter for valid signatures (those with sufficient cells)
    signature_counts = adata_combined.obs['unified_signature'].value_counts()
    valid_signatures = signature_counts[signature_counts >= MIN_CELLS_PER_SIGNATURE].index
    valid_signatures = [sig for sig in valid_signatures if sig != 'Unmapped']
    
    logger.info(f"Signatures with >= {MIN_CELLS_PER_SIGNATURE} cells: {len(valid_signatures)}")
    
    sampled_indices = []
    final_counts = {}
    dataset_breakdown = {}
    subtype_breakdown = {}
    
    logger.info(f"Sampling up to {TARGET_CELLS_PER_SIGNATURE} cells per unified signature:")
    logger.info("Note: Merged signatures (Tang NK + Atlas non-immune major) use prevalence-weighted sampling")
    
    for unified_sig in sorted(valid_signatures):
        # Get cells for this unified signature
        signature_mask = adata_combined.obs['unified_signature'] == unified_sig
        signature_indices = np.where(signature_mask)[0]
        
        n_available = len(signature_indices)
        
        if n_available >= TARGET_CELLS_PER_SIGNATURE:
            # Use prevalence-weighted sampling for merged signatures (Tang NK + Atlas non-immune major)
            if unified_sig in TANG_NK_MAPPING:
                sampled = sample_tang_nk_weighted(adata_combined, unified_sig, signature_indices, TARGET_CELLS_PER_SIGNATURE)
                logger.info(f"  {unified_sig}: {n_available:,} available → sampled {TARGET_CELLS_PER_SIGNATURE} (Tang NK prevalence-weighted)")
            elif unified_sig in ATLAS_NONIMMUNE_MAJOR_MAPPING:
                sampled = sample_atlas_major_weighted(adata_combined, unified_sig, signature_indices, TARGET_CELLS_PER_SIGNATURE)
                logger.info(f"  {unified_sig}: {n_available:,} available → sampled {TARGET_CELLS_PER_SIGNATURE} (Atlas major prevalence-weighted)")
            else:
                # Standard random sampling for Atlas immune minor types (no merging)
                sampled = np.random.choice(
                    signature_indices, size=TARGET_CELLS_PER_SIGNATURE, replace=False
                )
                logger.info(f"  {unified_sig}: {n_available:,} available → sampled {TARGET_CELLS_PER_SIGNATURE} (random)")
            
            sampled_indices.extend(sampled)
            final_counts[unified_sig] = TARGET_CELLS_PER_SIGNATURE
        else:
            # Use all available cells if less than target
            sampled_indices.extend(signature_indices)
            final_counts[unified_sig] = n_available
            logger.info(f"  {unified_sig}: {n_available:,} available → using all {n_available} (insufficient for {TARGET_CELLS_PER_SIGNATURE})")
        
        # Track dataset source breakdown for sampled cells
        sampled_signature_data = adata_combined[signature_indices if n_available < TARGET_CELLS_PER_SIGNATURE else sampled]
        dataset_breakdown[unified_sig] = sampled_signature_data.obs['dataset_source'].value_counts().to_dict()
        
        # Track subtype breakdown for merged signatures (Tang NK + Atlas non-immune major)
        if unified_sig in TANG_NK_MAPPING and TANG_CELLTYPE_COL in sampled_signature_data.obs.columns:
            subtype_breakdown[unified_sig] = sampled_signature_data.obs[TANG_CELLTYPE_COL].value_counts().to_dict()
        elif unified_sig in ATLAS_NONIMMUNE_MAJOR_MAPPING and 'celltype_minor' in sampled_signature_data.obs.columns:
            subtype_breakdown[unified_sig] = sampled_signature_data.obs['celltype_minor'].value_counts().to_dict()
    
    # Create sampled dataset
    adata_sampled = adata_combined[sampled_indices].copy()
    
    logger.info(f"\nFinal sampled dataset: {adata_sampled.shape[0]:,} cells")
    logger.info("Final unified signature distribution:")
    
    # Group signatures by type for cleaner reporting
    atlas_immune = [sig for sig in final_counts.keys() if sig in ATLAS_IMMUNE_MINOR_TYPES.values()]
    tang_nk = [sig for sig in final_counts.keys() if sig in TANG_NK_MAPPING.keys()]
    atlas_nonimmune = [sig for sig in final_counts.keys() if sig in ATLAS_NONIMMUNE_MAJOR_MAPPING.keys()]
    
    for sig_group, sigs, title in [(atlas_immune, atlas_immune, "Atlas Immune Minor"),
                                   (tang_nk, tang_nk, "Tang NK Signatures"),
                                   (atlas_nonimmune, atlas_nonimmune, "Atlas Non-Immune Major")]:
        if sigs:
            logger.info(f"\n{title}:")
            for sig in sorted(sigs):
                count = final_counts[sig]
                logger.info(f"  {sig}: {count} cells")
                if sig in UNIFIED_SIGNATURE_DESCRIPTIONS:
                    logger.info(f"    {UNIFIED_SIGNATURE_DESCRIPTIONS[sig]}")
    
    # Show dataset source representation in final sample
    logger.info("\nDataset source representation in final sample:")
    total_atlas = sum(1 for sig in final_counts.keys() if sig in list(ATLAS_IMMUNE_MINOR_TYPES.values()) + list(ATLAS_NONIMMUNE_MAJOR_MAPPING.keys()))
    total_tang = sum(1 for sig in final_counts.keys() if sig in TANG_NK_MAPPING.keys())
    logger.info(f"  Atlas-derived signatures: {total_atlas}")
    logger.info(f"  Tang-derived signatures: {total_tang}")
    
    for unified_sig in sorted(final_counts.keys()):
        if unified_sig in dataset_breakdown:
            sources = ", ".join([f"{source}:{count}" for source, count in dataset_breakdown[unified_sig].items()])
            logger.info(f"  {unified_sig}: {sources}")
    
    # Show subtype representation in final sample (prevalence-weighted for merged signatures)
    tang_nk_sigs = [sig for sig in final_counts.keys() if sig in TANG_NK_MAPPING]
    atlas_major_sigs = [sig for sig in final_counts.keys() if sig in ATLAS_NONIMMUNE_MAJOR_MAPPING]
    
    if tang_nk_sigs:
        logger.info("\nTang NK subtype representation (prevalence-weighted sampling):")
        for unified_sig in tang_nk_sigs:
            if unified_sig in subtype_breakdown:
                logger.info(f"  {unified_sig}:")
                total_sampled = sum(subtype_breakdown[unified_sig].values())
                for subtype, count in sorted(subtype_breakdown[unified_sig].items()):
                    percentage = (count / total_sampled) * 100 if total_sampled > 0 else 0
                    logger.info(f"    {subtype}: {count} cells ({percentage:.1f}% of sampled)")
    
    if atlas_major_sigs:
        logger.info("\nAtlas non-immune major subtype representation (prevalence-weighted sampling):")
        for unified_sig in atlas_major_sigs:
            if unified_sig in subtype_breakdown:
                logger.info(f"  {unified_sig}:")
                total_sampled = sum(subtype_breakdown[unified_sig].values())
                for subtype, count in sorted(subtype_breakdown[unified_sig].items()):
                    percentage = (count / total_sampled) * 100 if total_sampled > 0 else 0
                    logger.info(f"    {subtype}: {count} cells ({percentage:.1f}% of sampled)")
    
    return adata_sampled, final_counts


def sample_tang_nk_weighted(adata_combined, unified_sig, signature_indices, target_cells):
    """
    Sample Tang NK cells with prevalence-weighted sampling to maintain biological composition.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined dataset
    unified_sig : str
        Tang NK signature name
    signature_indices : np.array
        Indices of cells belonging to this signature
    target_cells : int
        Number of cells to sample
        
    Returns:
    --------
    sampled_indices : np.array
        Indices of sampled cells
    """
    # Get the Tang subtypes for this unified signature
    tang_subtypes = TANG_NK_MAPPING[unified_sig]
    
    # Get cells data for this signature
    signature_data = adata_combined[signature_indices]
    
    # Calculate prevalence of each Tang subtype
    subtype_counts = signature_data.obs[TANG_CELLTYPE_COL].value_counts()
    total_cells = len(signature_data)
    
    # Calculate how many cells to sample from each subtype (proportional to prevalence)
    sampled_indices = []
    remaining_target = target_cells
    
    # Sort subtypes by count (largest first) to handle rounding better
    sorted_subtypes = [(subtype, count) for subtype, count in subtype_counts.items() if subtype in tang_subtypes]
    sorted_subtypes.sort(key=lambda x: x[1], reverse=True)
    
    for i, (subtype, count) in enumerate(sorted_subtypes):
        if i == len(sorted_subtypes) - 1:
            # Last subtype gets all remaining cells to ensure exact target
            cells_to_sample = remaining_target
        else:
            # Calculate proportional sampling
            proportion = count / total_cells
            cells_to_sample = int(np.round(proportion * target_cells))
            cells_to_sample = min(cells_to_sample, remaining_target, count)
        
        if cells_to_sample > 0:
            # Get indices for this subtype
            subtype_mask = signature_data.obs[TANG_CELLTYPE_COL] == subtype
            subtype_indices = signature_indices[subtype_mask]
            
            # Sample from this subtype
            if len(subtype_indices) >= cells_to_sample:
                subtype_sampled = np.random.choice(subtype_indices, size=cells_to_sample, replace=False)
            else:
                # If not enough cells, take all available
                subtype_sampled = subtype_indices
                
            sampled_indices.extend(subtype_sampled)
            remaining_target -= len(subtype_sampled)
    
    return np.array(sampled_indices)


def sample_atlas_major_weighted(adata_combined, unified_sig, signature_indices, target_cells):
    """
    Sample Atlas non-immune major type cells with prevalence-weighted sampling to maintain biological composition.
    
    Parameters:
    -----------
    adata_combined : AnnData
        Combined dataset
    unified_sig : str
        Atlas non-immune major signature name
    signature_indices : np.array
        Indices of cells belonging to this signature
    target_cells : int
        Number of cells to sample
        
    Returns:
    --------
    sampled_indices : np.array
        Indices of sampled cells
    """
    # Get the Atlas minor subtypes for this unified signature
    atlas_subtypes = ATLAS_NONIMMUNE_MAJOR_MAPPING[unified_sig]
    
    # Get cells data for this signature
    signature_data = adata_combined[signature_indices]
    
    # Calculate prevalence of each Atlas subtype
    subtype_counts = signature_data.obs['celltype_minor'].value_counts()
    total_cells = len(signature_data)
    
    # Calculate how many cells to sample from each subtype (proportional to prevalence)
    sampled_indices = []
    remaining_target = target_cells
    
    # Sort subtypes by count (largest first) to handle rounding better
    sorted_subtypes = [(subtype, count) for subtype, count in subtype_counts.items() if subtype in atlas_subtypes]
    sorted_subtypes.sort(key=lambda x: x[1], reverse=True)
    
    for i, (subtype, count) in enumerate(sorted_subtypes):
        if i == len(sorted_subtypes) - 1:
            # Last subtype gets all remaining cells to ensure exact target
            cells_to_sample = remaining_target
        else:
            # Calculate proportional sampling
            proportion = count / total_cells
            cells_to_sample = int(np.round(proportion * target_cells))
            cells_to_sample = min(cells_to_sample, remaining_target, count)
        
        if cells_to_sample > 0:
            # Get indices for this subtype
            subtype_mask = signature_data.obs['celltype_minor'] == subtype
            subtype_indices = signature_indices[subtype_mask]
            
            # Sample from this subtype
            if len(subtype_indices) >= cells_to_sample:
                subtype_sampled = np.random.choice(subtype_indices, size=cells_to_sample, replace=False)
            else:
                # If not enough cells, take all available
                subtype_sampled = subtype_indices
                
            sampled_indices.extend(subtype_sampled)
            remaining_target -= len(subtype_sampled)
    
    return np.array(sampled_indices)


def get_expression_data_for_cibersortx(adata_sampled):
    """
    Extract expression data in the proper format for CIBERSORTx.
    
    Parameters:
    -----------
    adata_sampled : AnnData
        Sampled dataset
        
    Returns:
    --------
    expr_df : pd.DataFrame
        Expression data with genes as rows, cells as columns
    gene_names : pd.Index
        Gene names
    cell_labels : pd.Series
        Cell subtype labels
    """
    logger.info("=" * 70)
    logger.info("PREPARING EXPRESSION DATA FOR CIBERSORTX")
    logger.info("=" * 70)
    
    # Use raw data if available (recommended for CIBERSORTx)
    if hasattr(adata_sampled, 'raw') and adata_sampled.raw is not None:
        logger.info("Using raw expression data (recommended for CIBERSORTx)")
        expr_data = adata_sampled.raw.X
        gene_names = adata_sampled.raw.var_names
    else:
        logger.info("Using main expression data (.X)")
        expr_data = adata_sampled.X
        gene_names = adata_sampled.var_names
    
    # Convert sparse to dense if needed
    if sparse.issparse(expr_data):
        logger.info("Converting sparse matrix to dense...")
        expr_data = expr_data.toarray()
    
    # Check data properties
    logger.info(f"Expression data shape: {expr_data.shape}")
    logger.info(f"Expression range: {expr_data.min():.3f} to {expr_data.max():.3f}")
    logger.info(f"Number of genes: {len(gene_names):,}")
    
    # Create DataFrame with genes as rows, cells as columns
    # This is the required format for CIBERSORTx
    expr_df = pd.DataFrame(
        expr_data.T,  # Transpose to get genes as rows
        index=gene_names,
        columns=[f"Cell_{i}" for i in range(expr_data.shape[0])]
    )
    
    # Get cell labels (unified signature assignments)
    cell_labels = adata_sampled.obs['unified_signature'].copy()
    cell_labels.index = expr_df.columns  # Match column names
    
    # Handle log-space data for CIBERSORTx
    if expr_df.values.max() < 50:
        logger.warning("Data appears to be in log space (max < 50)")
        logger.warning("CIBERSORTx will automatically anti-log this data")
        logger.info("Converting back to linear space for CIBERSORTx compatibility...")
        # Convert from log(TPM+1) back to TPM
        expr_df = np.expm1(expr_df)
        logger.info(f"After anti-log conversion - range: {expr_df.values.min():.3f} to {expr_df.values.max():.3f}")
    
    return expr_df, gene_names, cell_labels


def create_cibersortx_reference_matrix(expr_df, cell_labels, output_path):
    """
    Create the final CIBERSORTx reference matrix file.
    
    Parameters:
    -----------
    expr_df : pd.DataFrame
        Expression data with genes as rows, cells as columns
    cell_labels : pd.Series
        Cell subtype labels
    output_path : Path
        Output file path
    """
    logger.info("=" * 70)
    logger.info("CREATING CIBERSORTX REFERENCE MATRIX")
    logger.info("=" * 70)
    
    # Prepare the reference matrix
    # Format: First row = cell labels, First column = gene names
    reference_matrix = expr_df.copy()
    
    # Replace column names with subtype labels
    reference_matrix.columns = cell_labels.values
    
    # Add gene names as first column (required by CIBERSORTx)
    reference_matrix.insert(0, "Gene", reference_matrix.index)
    
    logger.info(f"Reference matrix shape: {reference_matrix.shape}")
    logger.info(f"Genes (rows): {reference_matrix.shape[0]:,}")
    logger.info(f"Cells (columns): {reference_matrix.shape[1]-1:,}")  # -1 for gene column
    
    # Check for duplicate gene names (CIBERSORTx requirement)
    duplicate_genes = reference_matrix["Gene"].duplicated()
    if duplicate_genes.any():
        n_duplicates = duplicate_genes.sum()
        logger.warning(f"Found {n_duplicates} duplicate gene names")
        logger.warning("CIBERSORTx will add unique identifiers to duplicates")
    else:
        logger.info("No duplicate gene names found")
    
    # Save the reference matrix
    logger.info(f"Saving reference matrix to: {output_path}")
    reference_matrix.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    
    # Verify file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"File saved successfully. Size: {file_size_mb:.1f} MB")
    
    return reference_matrix


def generate_analysis_summary(adata_sampled, final_counts, reference_matrix, output_dir):
    """
    Generate a comprehensive summary of the hybrid Atlas-Tang reference matrix generation.
    
    Parameters:
    -----------
    adata_sampled : AnnData
        Final sampled dataset
    final_counts : dict
        Cell counts per unified signature
    reference_matrix : pd.DataFrame
        Generated reference matrix
    output_dir : Path
        Output directory
    """
    logger.info("=" * 70)
    logger.info("GENERATING ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    summary_path = output_dir / SUMMARY_FILENAME
    
    # Categorize signatures
    atlas_immune = [sig for sig in final_counts.keys() if sig in ATLAS_IMMUNE_MINOR_TYPES.values()]
    tang_nk = [sig for sig in final_counts.keys() if sig in TANG_NK_MAPPING.keys()]
    atlas_nonimmune = [sig for sig in final_counts.keys() if sig in ATLAS_NONIMMUNE_MAJOR_MAPPING.keys()]
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Hybrid Atlas-Tang Reference Matrix Generation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HYBRID APPROACH\n")
        f.write("-" * 20 + "\n")
        f.write("This matrix combines breast cancer atlas and Tang NK datasets:\n")
        f.write(f"  • Atlas Immune Minor Types: {len(atlas_immune)} signatures (T/B/Myeloid cells)\n")
        f.write(f"  • Tang NK Core Signatures: {len(tang_nk)} signatures (cytotoxic including TaNK + bright)\n")
        f.write(f"  • Atlas Non-Immune Major Types: {len(atlas_nonimmune)} signatures (epithelial/stromal)\n")
        f.write(f"  • Total: {len(final_counts)} unified signatures (target: 18)\n")
        f.write("  • Merged signature sampling: Prevalence-weighted to maintain biological composition\n")
        f.write("    (applies to Tang NK core signatures and Atlas non-immune major types)\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 15 + "\n")
        f.write(f"Atlas source: {ATLAS_MATRIX_DIR} + {ATLAS_METADATA_FILE}\n")
        f.write(f"Tang source: {TANG_COMBINED_H5AD_FILE}\n")
        f.write(f"Target cells per signature: {TARGET_CELLS_PER_SIGNATURE}\n")
        f.write(f"Minimum cells per signature: {MIN_CELLS_PER_SIGNATURE}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 18 + "\n")
        f.write(f"Final reference matrix dimensions: {reference_matrix.shape[0]:,} genes × {reference_matrix.shape[1]-1:,} cells\n")
        f.write(f"Total cells included: {sum(final_counts.values()):,}\n")
        f.write(f"Number of unified signatures: {len(final_counts)}\n")
        f.write(f"Gene harmonization: Intersection of Atlas and Tang gene sets\n\n")
        
        f.write("CELL DISTRIBUTION BY SIGNATURE TYPE\n")
        f.write("-" * 38 + "\n")
        total_cells = sum(final_counts.values())
        
        for sig_group, title in [(atlas_immune, "Atlas Immune Minor Types"),
                                (tang_nk, "Tang NK Core Signatures"),
                                (atlas_nonimmune, "Atlas Non-Immune Major Types")]:
            if sig_group:
                f.write(f"\n{title}:\n")
                for sig in sorted(sig_group):
                    count = final_counts[sig]
            percentage = (count / total_cells) * 100
            status = "TARGET" if count == TARGET_CELLS_PER_SIGNATURE else "LIMITED"
            f.write(f"  {sig:<20} {count:>3} cells ({percentage:>5.1f}%) [{status}]\n")
        
        f.write(f"\nSIGNATURES WITH LIMITED CELLS (<{TARGET_CELLS_PER_SIGNATURE}):\n")
        limited_signatures = {k: v for k, v in final_counts.items() if v < TARGET_CELLS_PER_SIGNATURE}
        if limited_signatures:
            for sig, count in sorted(limited_signatures.items()):
                f.write(f"  {sig}: {count} cells\n")
        else:
            f.write("  None - all signatures have sufficient cells\n")
        
        # Show signature mapping details
        f.write("\nSIGNATURE MAPPING DETAILS\n")
        f.write("-" * 27 + "\n")
        
        f.write("Atlas Immune Minor (preserved as-is):\n")
        for atlas_type, unified_sig in ATLAS_IMMUNE_MINOR_TYPES.items():
            if unified_sig in final_counts:
                f.write(f"  '{atlas_type}' -> {unified_sig}: {final_counts[unified_sig]} cells\n")
        
        f.write("\nTang NK (merged into core signatures with prevalence weighting):\n")
        for nk_sig, tang_subtypes in TANG_NK_MAPPING.items():
            if nk_sig in final_counts:
                f.write(f"  {nk_sig}: {final_counts[nk_sig]} cells (prevalence-weighted sampling)\n")
                for subtype in tang_subtypes:
                    if subtype == "CD56dimCD16hi-c6-DNAJB1":
                        f.write(f"    <- {subtype} (TaNK/exhausted - merged with cytotoxic)\n")
                    else:
                        f.write(f"    <- {subtype}\n")
        
        f.write("\nAtlas Non-Immune Major (multiple subtypes merged with prevalence weighting):\n")
        for major_sig, minor_types in ATLAS_NONIMMUNE_MAJOR_MAPPING.items():
            if major_sig in final_counts:
                f.write(f"  {major_sig}: {final_counts[major_sig]} cells (prevalence-weighted sampling)\n")
                for minor_type in minor_types:
                    f.write(f"    <- {minor_type}\n")
        
        f.write("\nFILE FORMAT COMPLIANCE\n")
        f.write("-" * 23 + "\n")
        f.write("✓ Genes in rows, cells in columns\n")
        f.write("✓ First column contains gene names\n")
        f.write("✓ Column headers contain unified signature labels\n")
        f.write("✓ Data in non-log space (TPM-like values)\n")
        f.write("✓ Tab-separated format\n")
        
        expr_max = reference_matrix.iloc[:, 1:].values.max()
        if expr_max >= 50:
            f.write(f"✓ Maximum expression value: {expr_max:.1f} (>50, will be treated as linear)\n")
        else:
            f.write(f"⚠ Maximum expression value: {expr_max:.1f} (<50, CIBERSORTx may assume log space)\n")
        
        f.write("\nPURPOSE\n")
        f.write("-" * 10 + "\n")
        f.write("This hybrid reference matrix combines the comprehensive cell type coverage\n")
        f.write("of the breast cancer atlas with the specialized NK cell expertise from Tang\n")
        f.write("et al. The matrix is optimized for CIBERSORTx deconvolution of breast cancer\n")
        f.write("bulk RNA-seq data, providing both immune granularity and tumor microenvironment\n")
        f.write("context. Atlas immune cells preserve biological diversity while Tang NK cells\n")
        f.write("add functional specialization for tumor immunity analysis.\n\n")
        f.write("IMPORTANT: TaNK (DNAJB1+) cells were merged into the NK_Cytotoxic signature\n")
        f.write("because they represent a dysfunctional subset of cytotoxic NK cells that share\n")
        f.write("core cytotoxic machinery but with exhaustion markers. This prevents CIBERSORTx\n")
        f.write("confusion between highly similar cytotoxic and exhausted NK signatures while\n")
        f.write("maintaining the overall cytotoxic NK cell representation in the matrix.\n\n")
        f.write("Both Tang NK signatures and Atlas non-immune major types use prevalence-weighted\n")
        f.write("sampling to maintain the biological composition and relative abundance of\n")
        f.write("constituent subtypes found in the original datasets.\n")
    
    logger.info(f"Analysis summary saved to: {summary_path}")


def main():
    """
    Main function to execute the Tang NK core signatures reference matrix generation workflow.
    """
    print("=" * 80)
    print("HYBRID ATLAS-TANG REFERENCE MATRIX GENERATOR FOR CIBERSORTX")
    print("=" * 80)
    print()
    print("Combining breast cancer atlas and Tang NK datasets:")
    print("• Atlas immune minor types (T/B/Myeloid cells) - preserved granularity")
    print("• Tang NK core signatures (2 specialized subtypes) - functional expertise")
    print("• Atlas non-immune major types (epithelial/stromal) - simplified for focus")
    print(f"• Target: 18 signatures × {TARGET_CELLS_PER_SIGNATURE} cells = {18*TARGET_CELLS_PER_SIGNATURE:,} total cells")
    print("• Prevalence-weighted sampling for merged signatures (Tang NK + Atlas non-immune major)")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load and harmonize both datasets
    adata_combined = load_and_harmonize_datasets()
    if adata_combined is None:
        logger.error("Failed to load and harmonize datasets. Exiting.")
        return False
    
    # Step 2: Preprocess combined data
    adata_combined = preprocess_combined_data(adata_combined)
    
    # Step 3: Create unified signatures from Atlas and Tang cell types
    adata_combined = create_unified_signatures(adata_combined)
    
    # Step 4: Analyze combined dataset composition
    adata_combined = analyze_dataset_composition(adata_combined)
    
    # Step 5: Sample cells per unified signature from combined dataset
    adata_sampled, final_counts = sample_cells_per_unified_signature(adata_combined)
    
    # Step 6: Prepare expression data for CIBERSORTx
    expr_df, gene_names, cell_labels = get_expression_data_for_cibersortx(adata_sampled)
    
    # Step 7: Create CIBERSORTx reference matrix
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    reference_matrix = create_cibersortx_reference_matrix(expr_df, cell_labels, output_path)
    
    # Step 8: Generate analysis summary
    generate_analysis_summary(adata_sampled, final_counts, reference_matrix, OUTPUT_DIR)
    
    logger.info("=" * 70)
    logger.info("HYBRID ATLAS-TANG REFERENCE MATRIX GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Reference matrix: {output_path}")
    logger.info(f"Analysis summary: {OUTPUT_DIR / SUMMARY_FILENAME}")
    logger.info("Unified signatures created:")
    for unified_sig, count in final_counts.items():
        logger.info(f"  {unified_sig}: {count} cells")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Hybrid Atlas-Tang reference matrix generation completed successfully!")
        print("🔬 Created comprehensive breast cancer reference with specialized NK signatures")
    else:
        print("\n❌ Hybrid reference matrix generation failed!") 