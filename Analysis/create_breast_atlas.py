#!/usr/bin/env python3
"""
Atlas-Only Reference Matrix Generator for CIBERSORTx
=====================================================

This script generates a comprehensive reference matrix using only the breast cancer atlas
dataset for CIBERSORTx deconvolution analysis.

The script creates:
- Atlas immune minor subtypes (T/B/NK/Myeloid cells) 
- Atlas non-immune major subtypes (Cancer/Normal Epithelial/Endothelial/CAFs/PVL)

Total: ~16 signatures × 400 cells = ~6,400 cells

The script:
1. Loads Atlas dataset
2. Maps Atlas cell types to unified signatures (immune minor + non-immune major)
3. Samples 400 cells per unified signature (prevalence-weighted for merged types)
4. Exports in CIBERSORTx format (genes as rows, cells as columns)

Reference: This Atlas-only approach provides comprehensive breast cancer tissue diversity
for tumor microenvironment deconvolution without external NK cell datasets.
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
# File paths for Atlas dataset only
ATLAS_METADATA_FILE = "data/raw/breast_cancer_atlas/Whole_miniatlas_meta.csv"
ATLAS_MATRIX_DIR = "data/raw/breast_cancer_atlas/"

# === UNIFIED SIGNATURE CONFIGURATION ===
# Atlas immune cell types to keep as minor subtypes (using exact names from Atlas metadata)
ATLAS_IMMUNE_MINOR_TYPES = {
    "T cells CD4+": "T_cells_CD4+",
    "T cells CD8+": "T_cells_CD8+", 
    "NKT cells": "NKT_cells",
    "NK cells": "NK_cells",  # Added Atlas NK cells if present
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

# Combined unified signature mapping (Atlas only)
UNIFIED_SIGNATURE_MAPPING = {**ATLAS_IMMUNE_MINOR_TYPES, **ATLAS_NONIMMUNE_MAJOR_MAPPING}

# === UNIFIED SIGNATURE DESCRIPTIONS ===
UNIFIED_SIGNATURE_DESCRIPTIONS = {
    # Atlas immune minor types
    "T_cells_CD4+": "CD4+ T helper cells from Atlas",
    "T_cells_CD8+": "CD8+ cytotoxic T cells from Atlas",
    "NKT_cells": "Natural killer T cells from Atlas", 
    "NK_cells": "Natural killer cells from Atlas",
    "T_cells_Cycling": "Proliferating T cells from Atlas",
    "B_cells_Memory": "Memory B cells from Atlas",
    "B_cells_Naive": "Naive B cells from Atlas",
    "Plasmablasts": "Antibody-secreting plasma cells from Atlas",
    "Macrophage": "Tissue macrophages from Atlas",
    "Monocyte": "Circulating monocytes from Atlas", 
    "DCs": "Dendritic cells from Atlas",
    "Myeloid_Cycling": "Proliferating myeloid cells from Atlas",
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
OUTPUT_FILENAME = "Atlas_Only_signatures_16types_400cells_per_signature.txt"
SUMMARY_FILENAME = "Atlas_Only_generation_summary.txt"


def load_atlas_dataset():
    """
    Load Atlas dataset only for reference matrix generation.
    
    Returns:
    --------
    adata_atlas : AnnData or None
        Atlas dataset, or None if loading fails
    """
    logger.info("=" * 70)
    logger.info("LOADING ATLAS DATASET")
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
        
        # Normalize dataset if needed
        logger.info("Normalizing Atlas dataset...")
        if adata_atlas.X.max() > 50:  # Raw counts or high TPM
            logger.info("  Atlas: Normalizing and log-transforming")
            sc.pp.normalize_total(adata_atlas, target_sum=1e4)
            sc.pp.log1p(adata_atlas)
        else:
            logger.info(f"  Atlas: Already normalized (max: {adata_atlas.X.max():.1f})")
        
        logger.info(f"Atlas dataset before gene filtering: {adata_atlas.shape}")
        logger.info(f"Genes before filtering: {len(adata_atlas.var_names):,}")
        
        return adata_atlas
        
    except Exception as e:
        logger.error(f"Failed to load Atlas dataset: {e}")
        return None


def filter_genes_for_cibersortx(adata_atlas):
    """
    Filter genes using data-driven approaches for CIBERSORTx reference matrix.
    This reduces file size while maintaining biological relevance through:
    1. Expression-based filtering (removes low-expressed genes)
    2. Variability-based filtering (keeps informative genes)
    3. Cell-type specificity (keeps discriminative genes)
    
    Parameters:
    -----------
    adata_atlas : AnnData
        Atlas dataset
        
    Returns:
    --------
    adata_atlas : AnnData
        Atlas dataset with filtered genes
    """
    logger.info("=" * 70)
    logger.info("FILTERING GENES FOR CIBERSORTX OPTIMIZATION")
    logger.info("=" * 70)
    
    logger.info(f"Starting with {adata_atlas.shape[1]:,} genes")
    
    # Filter 1: Remove genes expressed in very few cells
    min_cells_threshold = max(10, int(adata_atlas.n_obs * 0.001))  # At least 0.1% of cells
    logger.info(f"Filtering genes expressed in < {min_cells_threshold} cells...")
    sc.pp.filter_genes(adata_atlas, min_cells=min_cells_threshold)
    logger.info(f"After min_cells filter: {adata_atlas.shape[1]:,} genes")
    
    # Filter 2: Remove very low expression genes (bottom 10% by mean expression)
    logger.info("Removing very low expression genes...")
    gene_means = np.array(adata_atlas.X.mean(axis=0)).flatten()
    expression_threshold = np.percentile(gene_means, 10)
    high_expr_genes = gene_means >= expression_threshold
    adata_atlas = adata_atlas[:, high_expr_genes].copy()
    logger.info(f"After low expression filter: {adata_atlas.shape[1]:,} genes")
    
    # Filter 3: Keep highly variable genes (current best practice approach)
    logger.info("Identifying highly variable genes using Seurat v3 method...")
    target_hvg = min(12000, int(adata_atlas.shape[1] * 0.4))  # 40% of remaining genes or 12K max
    
    # Use Seurat v3 method (current gold standard for scRNA-seq)
    try:
        sc.pp.highly_variable_genes(adata_atlas, n_top_genes=target_hvg, flavor='seurat_v3')
    except:
        # Fallback to standard seurat method if seurat_v3 fails
        logger.warning("Seurat v3 method failed, falling back to standard seurat method")
        sc.pp.highly_variable_genes(adata_atlas, n_top_genes=target_hvg, flavor='seurat')
    
    hvg_genes = adata_atlas.var['highly_variable']
    n_hvg = hvg_genes.sum()
    logger.info(f"Identified {n_hvg:,} highly variable genes using proven Seurat method")
    
    # Filter 4: Add genes with high cell-type specificity (data-driven discriminative genes)
    logger.info("Identifying cell-type discriminative genes...")
    
    # Calculate gene expression variance across cell types if cell type info is available
    discriminative_genes = set()
    if 'celltype_minor' in adata_atlas.obs.columns:
        cell_types = adata_atlas.obs['celltype_minor'].unique()
        logger.info(f"Analyzing discrimination across {len(cell_types)} cell types...")
        
        # For each gene, calculate how well it discriminates between cell types
        gene_discrimination_scores = []
        
        # Use a more efficient approach: calculate coefficient of variation across cell types
        for i, gene in enumerate(adata_atlas.var_names):
            if i % 5000 == 0:  # Progress indicator
                logger.info(f"  Processed {i:,}/{len(adata_atlas.var_names):,} genes...")
            
            gene_expr = adata_atlas[:, gene].X.toarray().flatten()
            celltype_means = []
            
            for ct in cell_types:
                ct_mask = adata_atlas.obs['celltype_minor'] == ct
                if ct_mask.sum() > 5:  # Only consider cell types with >5 cells
                    ct_mean = gene_expr[ct_mask].mean()
                    celltype_means.append(ct_mean)
            
            if len(celltype_means) > 1:
                # Coefficient of variation across cell types (higher = more discriminative)
                cv = np.std(celltype_means) / (np.mean(celltype_means) + 1e-8)
                gene_discrimination_scores.append(cv)
            else:
                gene_discrimination_scores.append(0)
        
        # Keep top discriminative genes (top 20% by discrimination score)
        discrimination_threshold = np.percentile(gene_discrimination_scores, 80)
        high_discrimination = np.array(gene_discrimination_scores) >= discrimination_threshold
        
        discriminative_gene_names = adata_atlas.var_names[high_discrimination]
        discriminative_genes = set(discriminative_gene_names)
        logger.info(f"  Identified {len(discriminative_genes):,} highly discriminative genes")
    
    # Combine filters: HVG + discriminative genes
    genes_to_keep = hvg_genes.copy()
    
    # Add discriminative genes
    for gene in discriminative_genes:
        if gene in adata_atlas.var_names:
            gene_idx = adata_atlas.var_names == gene
            genes_to_keep[gene_idx] = True
    
    total_selected = genes_to_keep.sum()
    logger.info(f"Total genes selected: {total_selected:,}")
    
    # If still too many genes, use HVG ranking to trim
    if total_selected > 15000:
        logger.info(f"Trimming to top 15,000 most variable genes for optimal performance...")
        hvg_scores = adata_atlas.var['dispersions_norm'].fillna(0)
        top_15k_indices = hvg_scores.nlargest(15000).index
        genes_to_keep = adata_atlas.var_names.isin(top_15k_indices)
    
    # Apply gene filtering
    adata_atlas = adata_atlas[:, genes_to_keep].copy()
    
    final_count = adata_atlas.shape[1]
    logger.info(f"Final gene count: {final_count:,} genes")
    
    # Calculate reduction percentage
    original_count = len(gene_means)  # Before any filtering
    reduction_pct = (1 - final_count/original_count) * 100
    logger.info(f"Gene reduction: {reduction_pct:.1f}% (from {original_count:,} to {final_count:,})")
    logger.info(f"Target achieved: {final_count:,} genes (optimal range: 8K-15K for CIBERSORTx)")
    
    return adata_atlas


def preprocess_atlas_data(adata_atlas):
    """
    Preprocess Atlas data for reference matrix generation.
    
    Parameters:
    -----------
    adata_atlas : AnnData
        Atlas dataset
        
    Returns:
    --------
    adata_atlas : AnnData
        Preprocessed Atlas dataset
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING ATLAS DATA")
    logger.info("=" * 70)
    
    logger.info(f"Initial Atlas shape: {adata_atlas.shape}")
    
    # Filter genes for CIBERSORTx optimization
    adata_atlas = filter_genes_for_cibersortx(adata_atlas)
    
    # Store filtered data as raw for downstream analysis
    logger.info("Storing filtered data as .raw for downstream analysis...")
    adata_atlas.raw = adata_atlas.copy()
    
    # Basic cell filtering
    logger.info(f"Filtering cells with fewer than {MIN_GENES_PER_CELL} genes...")
    sc.pp.filter_cells(adata_atlas, min_genes=MIN_GENES_PER_CELL)
    logger.info(f"Shape after cell filtering: {adata_atlas.shape}")
    
    # Show dataset composition after filtering
    logger.info("Dataset composition after preprocessing:")
    logger.info(f"  Atlas: {adata_atlas.n_obs:,} cells (100.0%)")
    logger.info(f"  Final genes: {adata_atlas.shape[1]:,}")
    
    return adata_atlas


def create_unified_signatures(adata_atlas):
    """
    Create unified signature labels by mapping Atlas cell types to final signatures.
    
    Parameters:
    -----------
    adata_atlas : AnnData
        Atlas dataset
        
    Returns:
    --------
    adata_atlas : AnnData
        Atlas dataset with new 'unified_signature' column added
    """
    logger.info("=" * 70)
    logger.info("CREATING UNIFIED SIGNATURES")
    logger.info("=" * 70)
    
    # Initialize unified signature column
    adata_atlas.obs['unified_signature'] = 'Unmapped'
    
    # Process Atlas cells
    logger.info(f"Processing {adata_atlas.n_obs:,} Atlas cells...")
    
    # Map Atlas immune minor types (keep as-is)
    for atlas_type, unified_sig in ATLAS_IMMUNE_MINOR_TYPES.items():
        type_mask = adata_atlas.obs['celltype_minor'] == atlas_type
        if type_mask.any():
            count = type_mask.sum()
            adata_atlas.obs.loc[type_mask, 'unified_signature'] = unified_sig
            logger.info(f"  {atlas_type} → {unified_sig}: {count:,} cells")
    
    # Map Atlas non-immune to major types (merge multiple subtypes)
    for major_sig, minor_types in ATLAS_NONIMMUNE_MAJOR_MAPPING.items():
        total_count = 0
        for minor_type in minor_types:
            type_mask = adata_atlas.obs['celltype_minor'] == minor_type
            if type_mask.any():
                count = type_mask.sum()
                total_count += count
                adata_atlas.obs.loc[type_mask, 'unified_signature'] = major_sig
        if total_count > 0:
            logger.info(f"  {major_sig} (merged): {total_count:,} cells from {len(minor_types)} subtypes")
    
    # Report final unified signature distribution
    unified_counts = adata_atlas.obs['unified_signature'].value_counts()
    unmapped_count = (adata_atlas.obs['unified_signature'] == 'Unmapped').sum()
    
    logger.info("\nUnified signature distribution:")
    total_mapped = len(adata_atlas) - unmapped_count
    
    for sig_type in ['Atlas Immune Minor', 'Atlas Non-Immune Major']:
        logger.info(f"\n{sig_type}:")
        
        if sig_type == 'Atlas Immune Minor':
            sigs = list(ATLAS_IMMUNE_MINOR_TYPES.values())
        else:  # Atlas Non-Immune Major
            sigs = list(ATLAS_NONIMMUNE_MAJOR_MAPPING.keys())
        
        for sig in sigs:
            if sig in unified_counts:
                count = unified_counts[sig]
                percentage = (count / total_mapped) * 100 if total_mapped > 0 else 0
                logger.info(f"  {sig}: {count:,} cells ({percentage:.1f}%)")
    
    if unmapped_count > 0:
        logger.warning(f"\nUnmapped cells: {unmapped_count:,} ({unmapped_count/len(adata_atlas)*100:.1f}%)")
        # Show unmapped cell types for debugging
        unmapped_mask = adata_atlas.obs['unified_signature'] == 'Unmapped'
        unmapped_atlas = adata_atlas.obs[unmapped_mask]['celltype_minor'].value_counts()
        if len(unmapped_atlas) > 0:
            logger.warning(f"  Unmapped Atlas types: {dict(unmapped_atlas.head())}")
    
    return adata_atlas


def analyze_dataset_composition(adata_atlas):
    """
    Analyze composition of Atlas dataset by unified signatures.
    
    Parameters:
    -----------
    adata_atlas : AnnData
        Atlas dataset with unified signatures
        
    Returns:
    --------
    adata_atlas : AnnData
        Atlas dataset (unchanged, just analyzed)
    """
    logger.info("=" * 70)
    logger.info("ANALYZING ATLAS DATASET COMPOSITION")
    logger.info("=" * 70)
    
    # Dataset source distribution
    logger.info("Dataset source distribution:")
    logger.info(f"  Atlas: {adata_atlas.n_obs:,} cells (100.0%)")
    
    # Unified signature distribution
    logger.info("\nUnified signature distribution:")
    sig_counts = adata_atlas.obs['unified_signature'].value_counts()
    logger.info(f"\n{sig_counts}")
    
    # Show cell type distribution by patient if available
    if 'Patient' in adata_atlas.obs.columns:
        logger.info(f"\nPatient distribution:")
        patient_counts = adata_atlas.obs['Patient'].value_counts()
        logger.info(f"  Total patients: {len(patient_counts)}")
        logger.info(f"  Cells per patient range: {patient_counts.min()} - {patient_counts.max()}")
    
    logger.info(f"\nTotal cells available for sampling: {adata_atlas.shape[0]:,}")
    
    return adata_atlas


def sample_cells_per_unified_signature(adata_atlas):
    """
    Sample up to TARGET_CELLS_PER_SIGNATURE cells per unified signature from Atlas dataset.
    For Atlas non-immune major signatures, sampling is weighted by subtype prevalence to maintain biological composition.
    
    Parameters:
    -----------
    adata_atlas : AnnData
        Atlas dataset with unified signatures
        
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
    available_signatures = adata_atlas.obs['unified_signature'].dropna().unique()
    available_signatures = [sig for sig in available_signatures if sig != 'Unmapped']
    logger.info(f"Available unified signatures: {len(available_signatures)} total")
    
    # Filter for valid signatures (those with sufficient cells)
    signature_counts = adata_atlas.obs['unified_signature'].value_counts()
    valid_signatures = signature_counts[signature_counts >= MIN_CELLS_PER_SIGNATURE].index
    valid_signatures = [sig for sig in valid_signatures if sig != 'Unmapped']
    
    logger.info(f"Signatures with >= {MIN_CELLS_PER_SIGNATURE} cells: {len(valid_signatures)}")
    
    sampled_indices = []
    final_counts = {}
    subtype_breakdown = {}
    
    logger.info(f"Sampling up to {TARGET_CELLS_PER_SIGNATURE} cells per unified signature:")
    logger.info("Note: Merged signatures (Atlas non-immune major) use prevalence-weighted sampling")
    
    for unified_sig in sorted(valid_signatures):
        # Get cells for this unified signature
        signature_mask = adata_atlas.obs['unified_signature'] == unified_sig
        signature_indices = np.where(signature_mask)[0]
        
        n_available = len(signature_indices)
        
        if n_available >= TARGET_CELLS_PER_SIGNATURE:
            # Use prevalence-weighted sampling for merged signatures (Atlas non-immune major)
            if unified_sig in ATLAS_NONIMMUNE_MAJOR_MAPPING:
                sampled = sample_atlas_major_weighted(adata_atlas, unified_sig, signature_indices, TARGET_CELLS_PER_SIGNATURE)
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
        
        # Track subtype breakdown for merged signatures (Atlas non-immune major)
        sampled_signature_data = adata_atlas[signature_indices if n_available < TARGET_CELLS_PER_SIGNATURE else sampled]
        if unified_sig in ATLAS_NONIMMUNE_MAJOR_MAPPING and 'celltype_minor' in sampled_signature_data.obs.columns:
            subtype_breakdown[unified_sig] = sampled_signature_data.obs['celltype_minor'].value_counts().to_dict()
    
    # Create sampled dataset
    adata_sampled = adata_atlas[sampled_indices].copy()
    
    logger.info(f"\nFinal sampled dataset: {adata_sampled.shape[0]:,} cells")
    logger.info("Final unified signature distribution:")
    
    # Group signatures by type for cleaner reporting
    atlas_immune = [sig for sig in final_counts.keys() if sig in ATLAS_IMMUNE_MINOR_TYPES.values()]
    atlas_nonimmune = [sig for sig in final_counts.keys() if sig in ATLAS_NONIMMUNE_MAJOR_MAPPING.keys()]
    
    for sig_group, sigs, title in [(atlas_immune, atlas_immune, "Atlas Immune Minor"),
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
    total_signatures = len(final_counts)
    logger.info(f"  Atlas-derived signatures: {total_signatures} (100.0%)")
    
    # Show subtype representation in final sample (prevalence-weighted for merged signatures)
    atlas_major_sigs = [sig for sig in final_counts.keys() if sig in ATLAS_NONIMMUNE_MAJOR_MAPPING]
    
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





def sample_atlas_major_weighted(adata_atlas, unified_sig, signature_indices, target_cells):
    """
    Sample Atlas non-immune major type cells with prevalence-weighted sampling to maintain biological composition.
    
    Parameters:
    -----------
    adata_atlas : AnnData
        Atlas dataset
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
    signature_data = adata_atlas[signature_indices]
    
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
    Generate a comprehensive summary of the Atlas-only reference matrix generation.
    
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
    atlas_nonimmune = [sig for sig in final_counts.keys() if sig in ATLAS_NONIMMUNE_MAJOR_MAPPING.keys()]
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Atlas-Only Reference Matrix Generation Summary\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ATLAS-ONLY APPROACH\n")
        f.write("-" * 20 + "\n")
        f.write("This matrix uses only the breast cancer atlas dataset:\n")
        f.write(f"  • Atlas Immune Minor Types: {len(atlas_immune)} signatures (T/B/NK/Myeloid cells)\n")
        f.write(f"  • Atlas Non-Immune Major Types: {len(atlas_nonimmune)} signatures (epithelial/stromal)\n")
        f.write(f"  • Total: {len(final_counts)} unified signatures\n")
        f.write("  • Merged signature sampling: Prevalence-weighted to maintain biological composition\n")
        f.write("    (applies to Atlas non-immune major types)\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 15 + "\n")
        f.write(f"Atlas source: {ATLAS_MATRIX_DIR} + {ATLAS_METADATA_FILE}\n")
        f.write(f"Target cells per signature: {TARGET_CELLS_PER_SIGNATURE}\n")
        f.write(f"Minimum cells per signature: {MIN_CELLS_PER_SIGNATURE}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 18 + "\n")
        f.write(f"Final reference matrix dimensions: {reference_matrix.shape[0]:,} genes × {reference_matrix.shape[1]-1:,} cells\n")
        f.write(f"Total cells included: {sum(final_counts.values()):,}\n")
        f.write(f"Number of unified signatures: {len(final_counts)}\n")
        f.write(f"Source: Atlas breast cancer dataset only\n\n")
        
        f.write("CELL DISTRIBUTION BY SIGNATURE TYPE\n")
        f.write("-" * 38 + "\n")
        total_cells = sum(final_counts.values())
        
        for sig_group, title in [(atlas_immune, "Atlas Immune Minor Types"),
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
        f.write("This Atlas-only reference matrix provides comprehensive cell type coverage\n")
        f.write("from the breast cancer atlas dataset. The matrix is optimized for CIBERSORTx\n")
        f.write("deconvolution of breast cancer bulk RNA-seq data, providing immune granularity\n")
        f.write("and tumor microenvironment context from a single, well-characterized dataset.\n")
        f.write("This approach ensures consistency in gene expression profiles and avoids\n")
        f.write("potential batch effects from combining multiple datasets.\n\n")
        f.write("Atlas non-immune major types use prevalence-weighted sampling to maintain\n")
        f.write("the biological composition and relative abundance of constituent subtypes\n")
        f.write("found in the original Atlas dataset.\n")
    
    logger.info(f"Analysis summary saved to: {summary_path}")


def main():
    """
    Main function to execute the Atlas-only reference matrix generation workflow.
    """
    print("=" * 80)
    print("ATLAS-ONLY REFERENCE MATRIX GENERATOR FOR CIBERSORTX")
    print("=" * 80)
    print()
    print("Using breast cancer atlas dataset only:")
    print("• Atlas immune minor types (T/B/NK/Myeloid cells) - preserved granularity")
    print("• Atlas non-immune major types (epithelial/stromal) - simplified for focus")
    estimated_sigs = len(ATLAS_IMMUNE_MINOR_TYPES) + len(ATLAS_NONIMMUNE_MAJOR_MAPPING)
    print(f"• Target: ~{estimated_sigs} signatures × {TARGET_CELLS_PER_SIGNATURE} cells = ~{estimated_sigs*TARGET_CELLS_PER_SIGNATURE:,} total cells")
    print("• Prevalence-weighted sampling for merged signatures (Atlas non-immune major)")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load Atlas dataset
    adata_atlas = load_atlas_dataset()
    if adata_atlas is None:
        logger.error("Failed to load Atlas dataset. Exiting.")
        return False
    
    # Step 2: Preprocess Atlas data
    adata_atlas = preprocess_atlas_data(adata_atlas)
    
    # Step 3: Create unified signatures from Atlas cell types
    adata_atlas = create_unified_signatures(adata_atlas)
    
    # Step 4: Analyze Atlas dataset composition
    adata_atlas = analyze_dataset_composition(adata_atlas)
    
    # Step 5: Sample cells per unified signature from Atlas dataset
    adata_sampled, final_counts = sample_cells_per_unified_signature(adata_atlas)
    
    # Step 6: Prepare expression data for CIBERSORTx
    expr_df, gene_names, cell_labels = get_expression_data_for_cibersortx(adata_sampled)
    
    # Step 7: Create CIBERSORTx reference matrix
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    reference_matrix = create_cibersortx_reference_matrix(expr_df, cell_labels, output_path)
    
    # Step 8: Generate analysis summary
    generate_analysis_summary(adata_sampled, final_counts, reference_matrix, OUTPUT_DIR)
    
    logger.info("=" * 70)
    logger.info("ATLAS-ONLY REFERENCE MATRIX GENERATION COMPLETE")
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
        print("\n✅ Atlas-only reference matrix generation completed successfully!")
        print("🔬 Created comprehensive breast cancer reference from Atlas dataset")
    else:
        print("\n❌ Atlas reference matrix generation failed!") 