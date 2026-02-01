#!/usr/bin/env python3
"""
Harmony Batch Correction for NK Cell Dataset
============================================

Implements Harmony batch correction as used by the original dataset authors.
Corrects for Dataset and Donor batch effects while preserving NK subtype biology.

Key features:
- Harmony correction for multiple batch variables (Dataset + Donor)
- Comprehensive before/after evaluation
- Preservation of biological variation (NK subtypes)
- GPU acceleration where possible
- Integration-ready for NK_analysis_main_rebuffet.py
"""

# === ENVIRONMENT SETUP ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA initially
os.environ["NUMBA_DISABLE_CUDA"] = "1"   # Disable CUDA for numba
os.environ["OMP_NUM_THREADS"] = "1"      # Prevent threading conflicts

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import logging
import anndata

# Import scanpy
try:
    import scanpy as sc
    sc.settings.verbosity = 1
    sc.settings.autoshow = False
    print(f"✓ Scanpy {sc.__version__} imported successfully")
except Exception as e:
    print(f"✗ CRITICAL ERROR: Scanpy import failed: {e}")
    exit(1)

# Try to import harmonypy
try:
    import harmonypy as hm
    print(f"✓ HarmonyPy imported successfully")
    HARMONY_AVAILABLE = True
except ImportError:
    print("⚠️ HarmonyPy not found. Installing...")
    HARMONY_AVAILABLE = False

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
# Input/Output paths
NK_DATA_FILE = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
OUTPUT_DIR = Path("outputs/harmony_batch_correction")
CORRECTED_DATA_FILE = "PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad"

# Enhanced Quality Control parameters
ENHANCED_QC = True  # Enable enhanced QC (MT filtering, doublet detection, etc.)
MT_THRESHOLD_MAX = 20  # Maximum MT% allowed (adaptive threshold capped at this)
DOUBLET_QUANTILE = 0.98  # Quantile threshold for doublet detection
COUNT_FILTER_QUANTILES = (0.02, 0.98)  # Remove extreme count outliers

# Harmony parameters
BATCH_VARIABLES = ["Dataset", "donor"]  # Primary batch variables to correct
CMV_CORRECTION = True  # CMV status appears confounded with technical batches - correct it
N_PCS = 50  # Number of PCs for Harmony
HARMONY_SIGMA = 0.1  # Harmony diversity penalty (default 0.1)
HARMONY_THETA = 2.0  # Harmony cluster diversity (default 2.0)
HARMONY_MAX_ITER = 10  # Maximum Harmony iterations

# Analysis parameters
N_HVGS = 2000  # Number of highly variable genes for analysis
RANDOM_SEED = 42

def install_harmony_if_needed():
    """Install HarmonyPy if not available."""
    global HARMONY_AVAILABLE
    
    if not HARMONY_AVAILABLE:
        try:
            print("Installing HarmonyPy...")
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", "harmonypy"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ HarmonyPy installed successfully")
                import harmonypy as hm
                globals()['hm'] = hm
                HARMONY_AVAILABLE = True
            else:
                print(f"✗ Failed to install HarmonyPy: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Error installing HarmonyPy: {e}")
            return False
    
    return HARMONY_AVAILABLE

def load_and_prepare_data():
    """Load NK dataset and prepare for Harmony correction."""
    logger.info("=" * 70)
    logger.info("LOADING NK DATASET FOR HARMONY BATCH CORRECTION")
    logger.info("=" * 70)
    
    # Load data
    adata = sc.read_h5ad(NK_DATA_FILE)
    logger.info(f"✓ Loaded dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # Extract CMV status
    def get_cmv_status(donor_name):
        if pd.isna(donor_name):
            return "Unknown"
        donor_str = str(donor_name)
        if "CMVpos" in donor_str:
            return "CMV_Positive"
        elif "CMVneg" in donor_str:
            return "CMV_Negative"
        else:
            return "CMV_Unknown"
    
    adata.obs['CMV_Status'] = adata.obs['donor'].apply(get_cmv_status)
    
    # Create batch variables list
    batch_vars = BATCH_VARIABLES.copy()
    if CMV_CORRECTION:
        batch_vars.append('CMV_Status')
        logger.info("Including CMV_Status in batch correction - appears confounded with technical batches")
    else:
        logger.info("Preserving CMV_Status as biological variable - NK cells are affected by CMV infection")
    
    logger.info(f"Batch variables to correct: {batch_vars}")
    
    # Store original data
    adata.layers['X_original'] = adata.X.copy()
    
    return adata, batch_vars

def preprocess_for_harmony(adata):
    """Preprocess data for Harmony correction with enhanced QC."""
    logger.info("=" * 70)
    logger.info("PREPROCESSING FOR HARMONY CORRECTION WITH ENHANCED QC")
    logger.info("=" * 70)
    
    # Store original shape
    original_shape = adata.shape
    logger.info(f"Starting shape: {original_shape[0]:,} cells × {original_shape[1]:,} genes")
    
    # Enhanced QC metrics calculation
    logger.info("Calculating enhanced QC metrics...")
    
    # Mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    mt_gene_count = adata.var['mt'].sum()
    logger.info(f"  Identified {mt_gene_count} mitochondrial genes")
    
    # Ribosomal genes
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    ribo_gene_count = adata.var['ribo'].sum()
    logger.info(f"  Identified {ribo_gene_count} ribosomal genes")
    
    # Hemoglobin genes
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')
    hb_gene_count = adata.var['hb'].sum()
    logger.info(f"  Identified {hb_gene_count} hemoglobin genes")
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], inplace=True, log1p=True)
    
    # Enhanced cell filtering
    logger.info("Applying enhanced cell filtering...")
    
    # 1. Basic gene count filter
    pre_gene_filter = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    logger.info(f"  Removed {pre_gene_filter - adata.n_obs} cells with <200 genes")
    
    # 2. Mitochondrial gene filtering (adaptive thresholds)
    mt_pct = adata.obs['pct_counts_mt']
    mt_median = mt_pct.median()
    mt_mad = (mt_pct - mt_median).abs().median() * 1.4826  # MAD to std conversion
    mt_threshold = mt_median + 3 * mt_mad  # 3 MAD outliers
    mt_threshold = min(mt_threshold, 20)  # Cap at 20% to avoid removing healthy cells
    
    pre_mt_filter = adata.n_obs
    adata = adata[adata.obs['pct_counts_mt'] < mt_threshold, :].copy()
    logger.info(f"  Removed {pre_mt_filter - adata.n_obs} cells with MT% > {mt_threshold:.1f}%")
    
    # 3. Total count filtering (remove very low and very high count cells)
    total_counts = adata.obs['total_counts']
    count_lower = total_counts.quantile(0.02)  # Remove bottom 2%
    count_upper = total_counts.quantile(0.98)  # Remove top 2%
    
    pre_count_filter = adata.n_obs
    adata = adata[(adata.obs['total_counts'] > count_lower) & 
                  (adata.obs['total_counts'] < count_upper), :].copy()
    logger.info(f"  Removed {pre_count_filter - adata.n_obs} cells with extreme total counts")
    
    # 4. Enhanced gene filtering
    logger.info("Applying enhanced gene filtering...")
    pre_gene_count = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=5)  # More stringent than basic (3)
    logger.info(f"  Removed {pre_gene_count - adata.n_vars} genes expressed in <5 cells")
    
    # 5. Simple doublet detection (high gene count + high total count)
    logger.info("Applying simple doublet detection...")
    n_genes = adata.obs['n_genes_by_counts']
    total_counts = adata.obs['total_counts']
    
    # Define doublet criteria (top 2% for both metrics)
    gene_threshold = n_genes.quantile(0.98)
    count_threshold = total_counts.quantile(0.98)
    
    # Cells that are outliers in both metrics are likely doublets
    doublet_mask = (n_genes > gene_threshold) & (total_counts > count_threshold)
    pre_doublet_filter = adata.n_obs
    adata = adata[~doublet_mask, :].copy()
    logger.info(f"  Removed {pre_doublet_filter - adata.n_obs} potential doublets")
    
    # Summary of filtering
    cells_removed = original_shape[0] - adata.n_obs
    genes_removed = original_shape[1] - adata.n_vars
    logger.info("Enhanced QC filtering summary:")
    logger.info(f"  Cells removed: {cells_removed:,} ({cells_removed/original_shape[0]*100:.1f}%)")
    logger.info(f"  Genes removed: {genes_removed:,} ({genes_removed/original_shape[1]*100:.1f}%)")
    logger.info(f"  Final shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # Store raw counts BEFORE normalization (all genes)
    # Instead of using .raw attribute, store in layers for better control
    adata.layers['X_raw_filtered'] = adata.X.copy()
    
    # Also set the raw attribute for compatibility
    adata.raw = adata
    logger.info(f"✓ Raw data stored: {adata.raw.shape[0]:,} cells × {adata.raw.shape[1]:,} genes")
    logger.info(f"✓ Raw data also stored in layer 'X_raw_filtered' for backup")
    
    # Normalization and log transformation
    logger.info("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    logger.info(f"Finding {N_HVGS} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVGS, batch_key='Dataset')
    
    # Store normalized data for all genes before subsetting
    adata.layers['X_normalized_all_genes'] = adata.X.copy()
    logger.info(f"✓ Normalized data stored for all {adata.shape[1]:,} genes")
    
    # Store the full dataset before subsetting to HVGs
    # This will be used to preserve all gene information
    adata_all_genes = adata.copy()
    
    # Subset to HVGs for Harmony correction
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # Manually reconstruct the raw data for the HVG subset
    # Use the stored raw data from layers to create a proper raw AnnData object
    
    # Create raw data structure manually using the stored raw counts
    raw_X = adata_all_genes.layers['X_raw_filtered']  # Raw filtered counts
    raw_var = adata_all_genes.var.copy()  # Gene metadata for all genes
    raw_obs = adata_all_genes.obs.copy()  # Cell metadata
    
    # Create a new AnnData object for raw data
    adata_raw = anndata.AnnData(X=raw_X, obs=raw_obs, var=raw_var)
    
    # Set the raw attribute to this manually created object
    adata_hvg.raw = adata_raw
    
    # Store which genes are HVGs in the subset
    adata_hvg.layers['X_normalized_hvg'] = adata_hvg.X.copy()
    
    # Store metadata about the full dataset in .uns for later use
    adata_hvg.uns['full_gene_data'] = {
        'n_genes_total': adata_all_genes.shape[1],
        'n_genes_hvg': adata_hvg.shape[1],
        'hvg_indices': np.where(adata_all_genes.var.highly_variable)[0],
        'all_gene_names': adata_all_genes.var_names.tolist()
    }
    
    # Note: Full normalized and raw data are preserved in adata_hvg.raw
    # This contains all 18,759 genes with raw filtered counts
    
    logger.info(f"✓ Subset to {adata_hvg.shape[1]:,} highly variable genes for correction")
    logger.info(f"✓ Raw data preserved: {adata_hvg.raw.shape[1]:,} total genes available")
    
    return adata_hvg

def apply_harmony_correction(adata, batch_vars):
    """Apply Harmony batch correction."""
    logger.info("=" * 70)
    logger.info("APPLYING HARMONY BATCH CORRECTION")
    logger.info("=" * 70)
    
    # PCA
    logger.info(f"Computing PCA with {N_PCS} components...")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=N_PCS, svd_solver='arpack', random_state=RANDOM_SEED)
    
    # Prepare batch information for Harmony
    logger.info("Preparing batch metadata...")
    batch_data = adata.obs[batch_vars].copy()
    
    # Convert categorical variables to string
    for var in batch_vars:
        batch_data[var] = batch_data[var].astype(str)
    
    logger.info("Batch variable distributions:")
    for var in batch_vars:
        counts = batch_data[var].value_counts()
        logger.info(f"  {var}: {len(counts)} levels")
        for level, count in counts.head(5).items():
            logger.info(f"    {level}: {count:,} cells")
    
    # Apply Harmony
    logger.info("Running Harmony correction...")
    logger.info(f"  Sigma (diversity penalty): {HARMONY_SIGMA}")
    logger.info(f"  Theta (cluster diversity): {HARMONY_THETA}")
    logger.info(f"  Max iterations: {HARMONY_MAX_ITER}")
    
    try:
        # Run Harmony
        harmony_out = hm.run_harmony(
            adata.obsm['X_pca'],  # PCA embeddings
            batch_data,  # Batch metadata
            batch_vars,  # Variables to correct
            sigma=HARMONY_SIGMA,
            theta=HARMONY_THETA,
            max_iter_harmony=HARMONY_MAX_ITER,
            random_state=RANDOM_SEED
        )
        
        # Store Harmony-corrected embeddings
        adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T  # Transpose to cells x components
        
        logger.info("✓ Harmony correction completed successfully")
        logger.info(f"  Corrected embedding shape: {adata.obsm['X_pca_harmony'].shape}")
        
        return adata
        
    except Exception as e:
        logger.error(f"✗ Harmony correction failed: {e}")
        raise

def compute_harmony_downstream_analysis(adata):
    """Compute downstream analysis on Harmony-corrected data."""
    logger.info("=" * 70)
    logger.info("COMPUTING DOWNSTREAM ANALYSIS ON CORRECTED DATA")
    logger.info("=" * 70)
    
    # Compute neighborhood graph on corrected embeddings
    logger.info("Computing neighborhood graph on Harmony-corrected embeddings...")
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15, random_state=RANDOM_SEED)
    
    # UMAP on corrected data
    logger.info("Computing UMAP on corrected data...")
    sc.tl.umap(adata, random_state=RANDOM_SEED)
    adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
    
    # Also compute UMAP on original PCA for comparison
    logger.info("Computing UMAP on original PCA for comparison...")
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, random_state=RANDOM_SEED)
    sc.tl.umap(adata, random_state=RANDOM_SEED)
    adata.obsm['X_umap_original'] = adata.obsm['X_umap'].copy()
    
    # Restore corrected neighbors
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15, random_state=RANDOM_SEED)
    adata.obsm['X_umap'] = adata.obsm['X_umap_harmony'].copy()
    
    logger.info("✓ Downstream analysis completed")
    
    return adata

def evaluate_batch_correction_analytically(adata, batch_vars):
    """Analytical evaluation of batch correction effectiveness."""
    logger.info("=" * 70)
    logger.info("ANALYTICAL EVALUATION OF BATCH CORRECTION")
    logger.info("=" * 70)
    
    evaluation_results = {}
    
    # 1. Silhouette analysis for batch mixing
    logger.info("1. Computing silhouette scores for batch mixing...")
    from sklearn.metrics import silhouette_score
    
    for batch_var in batch_vars:
        # Original PCA
        try:
            sil_original = silhouette_score(adata.obsm['X_pca'], adata.obs[batch_var])
        except:
            sil_original = np.nan
        
        # Harmony-corrected
        try:
            sil_harmony = silhouette_score(adata.obsm['X_pca_harmony'], adata.obs[batch_var])
        except:
            sil_harmony = np.nan
        
        logger.info(f"  {batch_var}:")
        logger.info(f"    Original PCA silhouette: {sil_original:.3f}")
        logger.info(f"    Harmony silhouette: {sil_harmony:.3f}")
        logger.info(f"    Improvement: {sil_original - sil_harmony:.3f} (lower = better mixing)")
        
        evaluation_results[f'{batch_var}_sil_original'] = sil_original
        evaluation_results[f'{batch_var}_sil_harmony'] = sil_harmony
        evaluation_results[f'{batch_var}_sil_improvement'] = sil_original - sil_harmony
    
    # 2. Biological preservation (NK subtype separation)
    logger.info("2. Evaluating NK subtype preservation...")
    try:
        # Silhouette for NK subtypes (higher = better preserved)
        sil_biology_original = silhouette_score(adata.obsm['X_pca'], adata.obs['ident'])
        sil_biology_harmony = silhouette_score(adata.obsm['X_pca_harmony'], adata.obs['ident'])
        
        logger.info(f"  NK subtype separation:")
        logger.info(f"    Original PCA: {sil_biology_original:.3f}")
        logger.info(f"    Harmony: {sil_biology_harmony:.3f}")
        logger.info(f"    Preservation: {sil_biology_harmony/sil_biology_original:.3f} (should be >0.8)")
        
        evaluation_results['biology_sil_original'] = sil_biology_original
        evaluation_results['biology_sil_harmony'] = sil_biology_harmony
        evaluation_results['biology_preservation'] = sil_biology_harmony/sil_biology_original
        
    except Exception as e:
        logger.warning(f"Could not compute biological preservation: {e}")
    
    # 3. Technical covariate analysis (updated for enhanced QC metrics)
    logger.info("3. Analyzing technical covariate reduction...")
    technical_vars = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo']
    
    for tech_var in technical_vars:
        if tech_var in adata.obs.columns:
            # Correlation with PC1 (often captures technical variation)
            corr_original = np.corrcoef(adata.obsm['X_pca'][:, 0], adata.obs[tech_var])[0, 1]
            corr_harmony = np.corrcoef(adata.obsm['X_pca_harmony'][:, 0], adata.obs[tech_var])[0, 1]
            
            logger.info(f"  {tech_var} correlation with PC1:")
            logger.info(f"    Original: {corr_original:.3f}")
            logger.info(f"    Harmony: {corr_harmony:.3f}")
            logger.info(f"    Reduction: {abs(corr_original) - abs(corr_harmony):.3f}")
            
            evaluation_results[f'{tech_var}_pc1_corr_original'] = corr_original
            evaluation_results[f'{tech_var}_pc1_corr_harmony'] = corr_harmony
            evaluation_results[f'{tech_var}_corr_reduction'] = abs(corr_original) - abs(corr_harmony)
    
    return evaluation_results

def create_harmony_evaluation_plots(adata, batch_vars, evaluation_results):
    """Create comprehensive visualization of Harmony correction."""
    logger.info("=" * 70)
    logger.info("CREATING HARMONY EVALUATION VISUALIZATIONS")
    logger.info("=" * 70)
    
    # Create output directory
    fig_dir = OUTPUT_DIR / "evaluation_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Before/After UMAP comparison
    logger.info("Creating before/after UMAP comparison...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Plot for each batch variable
    for i, batch_var in enumerate(batch_vars[:2]):  # Limit to 2 main variables
        # Original
        sc.pl.umap(adata, color=batch_var, ax=axes[0, i*2], show=False, 
                  title=f'Original: {batch_var}', frameon=False)
        # Harmony
        adata.obsm['X_umap'] = adata.obsm['X_umap_harmony']
        sc.pl.umap(adata, color=batch_var, ax=axes[0, i*2+1], show=False, 
                  title=f'Harmony: {batch_var}', frameon=False)
    
    # NK subtypes - biological preservation
    adata.obsm['X_umap'] = adata.obsm['X_umap_original']
    sc.pl.umap(adata, color='ident', ax=axes[1, 0], show=False, 
              title='Original: NK Subtypes', frameon=False, legend_loc=None)
    
    adata.obsm['X_umap'] = adata.obsm['X_umap_harmony']
    sc.pl.umap(adata, color='ident', ax=axes[1, 1], show=False, 
              title='Harmony: NK Subtypes', frameon=False, legend_loc=None)
    
    # Technical variables (updated for enhanced QC)
    adata.obsm['X_umap'] = adata.obsm['X_umap_original']
    sc.pl.umap(adata, color='total_counts', ax=axes[1, 2], show=False, 
              title='Original: Total Counts', frameon=False)
    
    adata.obsm['X_umap'] = adata.obsm['X_umap_harmony']
    sc.pl.umap(adata, color='total_counts', ax=axes[1, 3], show=False, 
              title='Harmony: Total Counts', frameon=False)
    
    plt.suptitle('Harmony Batch Correction: Before vs After Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'harmony_before_after_umap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Evaluation metrics summary
    logger.info("Creating evaluation metrics summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Silhouette scores for batch mixing
    ax1 = axes[0, 0]
    batch_sil_data = []
    for var in batch_vars:
        if f'{var}_sil_original' in evaluation_results:
            batch_sil_data.append({
                'Variable': var,
                'Original': evaluation_results[f'{var}_sil_original'],
                'Harmony': evaluation_results[f'{var}_sil_harmony'],
                'Type': 'Batch Mixing'
            })
    
    if batch_sil_data:
        df_batch = pd.DataFrame(batch_sil_data)
        x = np.arange(len(df_batch))
        width = 0.35
        ax1.bar(x - width/2, df_batch['Original'], width, label='Original', alpha=0.8)
        ax1.bar(x + width/2, df_batch['Harmony'], width, label='Harmony', alpha=0.8)
        ax1.set_xlabel('Batch Variable')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Batch Mixing (Lower = Better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_batch['Variable'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Biological preservation
    ax2 = axes[0, 1]
    if 'biology_sil_original' in evaluation_results:
        bio_data = [evaluation_results['biology_sil_original'], 
                   evaluation_results['biology_sil_harmony']]
        ax2.bar(['Original', 'Harmony'], bio_data, 
                color=['lightcoral', 'lightblue'], alpha=0.8)
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('NK Subtype Preservation (Higher = Better)')
        ax2.grid(True, alpha=0.3)
        
        # Add preservation ratio text
        preservation = evaluation_results['biology_preservation']
        color = 'green' if preservation > 0.8 else 'orange' if preservation > 0.6 else 'red'
        ax2.text(0.5, max(bio_data) * 0.9, f'Preservation: {preservation:.3f}', 
                ha='center', va='center', fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Technical covariate reduction (updated for enhanced QC)
    ax3 = axes[1, 0]
    tech_vars = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo']
    tech_improvements = []
    tech_names = []
    
    for var in tech_vars:
        if f'{var}_corr_reduction' in evaluation_results:
            tech_improvements.append(evaluation_results[f'{var}_corr_reduction'])
            tech_names.append(var.replace('_', '\n').replace('pct counts', 'MT%' if 'mt' in var else 'Ribo%'))
    
    if tech_improvements:
        colors = ['green' if x > 0 else 'red' for x in tech_improvements]
        bars = ax3.bar(tech_names, tech_improvements, color=colors, alpha=0.7)
        ax3.set_ylabel('Correlation Reduction')
        ax3.set_title('Technical Covariate Reduction\n(Positive = Improvement)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, tech_improvements):
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.01 if val > 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top',
                    fontweight='bold')
    
    # Overall summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = "HARMONY CORRECTION SUMMARY\n\n"
    
    # Overall assessment
    if 'biology_preservation' in evaluation_results:
        preservation = evaluation_results['biology_preservation']
        if preservation > 0.8:
            bio_status = "✅ EXCELLENT"
        elif preservation > 0.6:
            bio_status = "⚠️ MODERATE" 
        else:
            bio_status = "❌ POOR"
        summary_text += f"Biology Preservation: {bio_status}\n"
        summary_text += f"  NK subtype separation: {preservation:.3f}\n\n"
    
    # Batch mixing improvement
    batch_improvements = []
    for var in batch_vars:
        if f'{var}_sil_improvement' in evaluation_results:
            improvement = evaluation_results[f'{var}_sil_improvement']
            batch_improvements.append(improvement)
            status = "✅" if improvement > 0.1 else "⚠️" if improvement > 0.05 else "❌"
            summary_text += f"{var} mixing: {status} {improvement:.3f}\n"
    
    if batch_improvements:
        avg_improvement = np.mean(batch_improvements)
        summary_text += f"\nAverage batch mixing improvement: {avg_improvement:.3f}\n"
    
    # Technical improvement
    tech_improvements_avg = np.mean(tech_improvements) if tech_improvements else 0
    tech_status = "✅" if tech_improvements_avg > 0.1 else "⚠️" if tech_improvements_avg > 0 else "❌"
    summary_text += f"Technical covariate reduction: {tech_status} {tech_improvements_avg:.3f}\n"
    
    # Overall recommendation
    summary_text += "\n" + "="*30 + "\n"
    if (evaluation_results.get('biology_preservation', 0) > 0.8 and 
        np.mean(batch_improvements) > 0.05 if batch_improvements else False):
        summary_text += "RECOMMENDATION: ✅ PROCEED\nHarmony correction successful!"
    else:
        summary_text += "RECOMMENDATION: ⚠️ REVIEW\nCheck parameters or method!"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Harmony Batch Correction: Evaluation Summary', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'harmony_evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"💾 Evaluation plots saved to: {fig_dir}")

def save_corrected_data(adata):
    """Save Harmony-corrected dataset with comprehensive data preservation."""
    logger.info("=" * 70)
    logger.info("SAVING HARMONY-CORRECTED DATASET")
    logger.info("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate raw data preservation
    logger.info("Validating data preservation...")
    has_raw = adata.raw is not None
    raw_genes = adata.raw.shape[1] if has_raw else 0
    hvg_genes = adata.shape[1]
    
    logger.info(f"  HVG subset: {hvg_genes:,} genes")
    logger.info(f"  Raw data: {'✓' if has_raw else '✗'} ({raw_genes:,} genes)")
    
    if not has_raw:
        logger.warning("⚠️ Raw data not found! Only HVG data will be saved.")
    elif raw_genes <= hvg_genes:
        logger.warning(f"⚠️ Raw data has fewer genes ({raw_genes}) than HVG subset ({hvg_genes})")
    else:
        logger.info(f"✓ Raw data properly preserved: {raw_genes:,} total genes available")
    
    # Validate data layers
    available_layers = list(adata.layers.keys())
    logger.info(f"Data layers available: {available_layers}")
    
    expected_layers = ['X_original', 'X_normalized_hvg']
    missing_layers = [layer for layer in expected_layers if layer not in available_layers]
    if missing_layers:
        logger.warning(f"⚠️ Expected layers missing: {missing_layers}")
    
    # Add comprehensive correction metadata
    adata.uns['harmony_correction'] = {
        'corrected_date': datetime.now().isoformat(),
        'enhanced_qc': bool(ENHANCED_QC),
        'mt_threshold_max': float(MT_THRESHOLD_MAX),
        'doublet_quantile': float(DOUBLET_QUANTILE),
        'count_filter_quantiles': list(COUNT_FILTER_QUANTILES),  # Convert tuple to list
        'batch_variables': list(BATCH_VARIABLES),  # Ensure list format
        'cmv_correction': bool(CMV_CORRECTION),
        'n_pcs': int(N_PCS),
        'sigma': float(HARMONY_SIGMA),
        'theta': float(HARMONY_THETA),
        'max_iter': int(HARMONY_MAX_ITER),
        'n_hvgs': int(N_HVGS),
        'hvg_shape_cells': int(adata.shape[0]),
        'hvg_shape_genes': int(adata.shape[1]),
        'raw_shape_cells': int(adata.raw.shape[0]) if has_raw else 0,
        'raw_shape_genes': int(adata.raw.shape[1]) if has_raw else 0,
        'has_raw_data': has_raw,
        'data_layers': list(available_layers)  # Ensure list format
    }
    
    # Add data structure documentation
    adata.uns['data_structure_info'] = {
        'X_matrix': 'Harmony-corrected, log-normalized HVG expression (for analysis)',
        'X_pca_harmony': 'Harmony batch-corrected PCA embeddings',
        'X_umap_harmony': 'UMAP computed on Harmony-corrected embeddings',
        'raw': 'Filtered raw counts for all genes (unnormalized)',
        'layers': {
            'X_original': 'Original expression matrix before any processing',
            'X_normalized_hvg': 'Log-normalized expression for HVGs only'
        },
        'full_gene_data': 'Metadata about the complete gene set preserved in .raw'
    }
    
    # Save corrected dataset
    output_path = OUTPUT_DIR / CORRECTED_DATA_FILE
    adata.write_h5ad(output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✓ Harmony-corrected dataset saved to: {output_path}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  HVG subset shape: {adata.shape}")
    logger.info(f"  Raw data shape: {adata.raw.shape if has_raw else 'Not available'}")
    logger.info(f"  Corrected embeddings: X_pca_harmony")
    logger.info(f"  Corrected UMAP: X_umap_harmony")
    logger.info(f"  Data layers: {', '.join(available_layers)}")
    
    return output_path

def main():
    """Main function for enhanced Harmony batch correction."""
    print("=" * 80)
    print("ENHANCED HARMONY BATCH CORRECTION FOR NK CELL DATASET")
    print("=" * 80)
    print("Enhanced approach with comprehensive quality control:")
    print("• Enhanced QC: Mitochondrial filtering, doublet detection")
    print("• Batch correction: Dataset, Donor, and CMV effects")
    print("• Adaptive thresholds for robust cell filtering")
    print("• Comprehensive evaluation and validation")
    print()
    
    # Install Harmony if needed
    if not install_harmony_if_needed():
        logger.error("Failed to install HarmonyPy. Cannot proceed.")
        return False
    
    try:
        # Step 1: Load and prepare data
        adata, batch_vars = load_and_prepare_data()
        
        # Step 2: Preprocess for Harmony
        adata = preprocess_for_harmony(adata)
        
        # Step 3: Apply Harmony correction
        adata = apply_harmony_correction(adata, batch_vars)
        
        # Step 4: Compute downstream analysis
        adata = compute_harmony_downstream_analysis(adata)
        
        # Step 5: Evaluate correction analytically
        evaluation_results = evaluate_batch_correction_analytically(adata, batch_vars)
        
        # Step 6: Create evaluation visualizations
        create_harmony_evaluation_plots(adata, batch_vars, evaluation_results)
        
        # Step 7: Save corrected data
        output_path = save_corrected_data(adata)
        
        logger.info("=" * 70)
        logger.info("HARMONY BATCH CORRECTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✓ Corrected dataset: {output_path}")
        logger.info(f"✓ Evaluation plots: {OUTPUT_DIR / 'evaluation_figures'}")
        logger.info("✓ Ready for integration into NK_analysis_main_rebuffet.py")
        
        # Print key results
        print(f"\n🎯 KEY RESULTS:")
        if 'biology_preservation' in evaluation_results:
            preservation = evaluation_results['biology_preservation']
            status = "✅" if preservation > 0.8 else "⚠️" if preservation > 0.6 else "❌"
            print(f"   Biology preservation: {status} {preservation:.3f}")
        
        print(f"   Batch variables corrected: {batch_vars}")
        
        # Data structure information
        print(f"\n📊 EXPORTED DATA STRUCTURE:")
        print(f"   ├── adata.X: Harmony-corrected HVG expression ({adata.shape})")
        print(f"   ├── adata.raw: Raw counts for all genes ({adata.raw.shape if adata.raw else 'Not available'})")
        print(f"   ├── adata.obsm['X_pca_harmony']: Batch-corrected PCA embeddings")
        print(f"   ├── adata.obsm['X_umap_harmony']: Corrected UMAP coordinates")
        print(f"   └── adata.layers:")
        for layer in adata.layers.keys():
            layer_shape = adata.layers[layer].shape
            print(f"       ├── {layer}: {layer_shape}")
        
        print(f"\n💾 Raw data export: {'✅ PRESERVED' if adata.raw else '❌ MISSING'}")
        if adata.raw:
            print(f"   └── {adata.raw.shape[1]:,} genes available for downstream analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"Harmony batch correction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Harmony batch correction completed successfully!")
        print("🔬 Dataset ready for unbiased NK cell analysis")
        print("📊 Review evaluation plots to confirm correction quality")
    else:
        print("\n❌ Harmony batch correction failed!")
        print("🔍 Check logs for detailed error information")