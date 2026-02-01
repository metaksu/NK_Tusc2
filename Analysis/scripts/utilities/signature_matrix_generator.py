#!/usr/bin/env python3
"""
Robust Signature Matrix Generation for CIBERSORTx Deconvolution
Using Linear TPM Data from Rebuffet et al. 2024

This script creates high-quality signature matrices for NK cell subtype deconvolution.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("ROBUST SIGNATURE MATRIX GENERATION FOR CIBERSORTx")
    print("="*80)
    
    # Configuration
    TPM_DATA_PATH = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
    OUTPUT_DIR = "outputs/signature_matrices/Robust_Signature_Matrix_Output_TPM"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # NK subtype definitions
    NK_SUBTYPES = ["NK1A", "NK1B", "NK1C", "NKint", "NK2", "NK3"]
    
    # Gene exclusion patterns for robust signatures
    GENE_EXCLUSION_PATTERNS = [
        r"^RPS[0-9L]", r"^RPL[0-9L]",  # Ribosomal proteins
        r"^RPLP[0-9]$", r"^RPSA$",     # Additional ribosomal
        r"^MT-",                       # Mitochondrial genes
        r"^ACT[BGINR]", r"^MYL[0-9]",  # Actin/Myosin
        r"^TPT1$", r"^FTL$", r"^FTH1$", r"^B2M$",  # Housekeeping
        r"^(HSP90|HSPA|HSPB|HSPD|HSPE|HSPH)[A-Z0-9]+",  # Heat shock
        r"^EEF[12]",                   # Translation factors
        r"^GAPDH$", r"^MALAT1$",       # Common housekeeping
    ]
    
    # Analysis parameters
    N_GENES_PER_SUBTYPE = 200
    LOGFC_THRESHOLD = 0.25
    PVAL_THRESHOLD = 0.05
    
    # Set scanpy settings
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # Set random seed for reproducibility
    RANDOM_SEED = 42
    sc.settings.seed = RANDOM_SEED
    np.random.seed(RANDOM_SEED)
    
    print(f"📁 TPM data path: {TPM_DATA_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎯 Target subtypes: {NK_SUBTYPES}")
    print(f"🔬 Analysis parameters: {N_GENES_PER_SUBTYPE} genes/subtype, logFC>{LOGFC_THRESHOLD}, p<{PVAL_THRESHOLD}")
    
    # Step 1: Load and validate data
    print("\n" + "="*60)
    print("STEP 1: LOADING TPM DATA")
    print("="*60)
    
    if not os.path.exists(TPM_DATA_PATH):
        raise FileNotFoundError(f"TPM data file not found: {TPM_DATA_PATH}")
    
    adata = sc.read_h5ad(TPM_DATA_PATH)
    print(f"✅ Data loaded: {adata.shape}")
    print(f"📊 Data type: {adata.X.dtype}")
    print(f"📈 Data range: {adata.X.min():.3f} to {adata.X.max():.3f}")
    
    # Validate NK subtype annotations
    if 'ident' not in adata.obs.columns:
        raise ValueError("'ident' column not found in metadata")
    
    # Check available subtypes
    available_subtypes = adata.obs['ident'].unique()
    print(f"\n📋 Available subtypes: {available_subtypes}")
    
    # Show subtype distribution
    subtype_counts = adata.obs['ident'].value_counts()
    print("\n📊 Subtype distribution:")
    for subtype in NK_SUBTYPES:
        if subtype in subtype_counts:
            count = subtype_counts[subtype]
            print(f"  {subtype}: {count:,} cells")
        else:
            print(f"  {subtype}: 0 cells (missing)")
    
    # Step 2: Preprocessing
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING & QUALITY CONTROL")
    print("="*60)
    
    # Filter for NK subtypes only
    nk_mask = adata.obs['ident'].isin(NK_SUBTYPES)
    adata_nk = adata[nk_mask].copy()
    print(f"✅ Filtered to NK cells: {adata_nk.shape}")
    
    # Verify TPM normalization
    print(f"\n🔍 TPM data validation:")
    tpm_sums = adata_nk.X.sum(axis=1)
    print(f"  - TPM sums per cell: {tpm_sums.mean():.0f} ± {tpm_sums.std():.0f}")
    print(f"  - Expected TPM sum: ~1,000,000")
    print(f"  - Data range: {adata_nk.X.min():.3f} to {adata_nk.X.max():.3f}")
    
    # Step 3: Marker gene identification
    print("\n" + "="*60)
    print("STEP 3: MARKER GENE IDENTIFICATION")
    print("="*60)
    
    # Create log-transformed copy for DEG analysis
    adata_log = adata_nk.copy()
    sc.pp.log1p(adata_log)
    print(f"✅ Created log-transformed copy for DEG analysis")
    
    # Run differential expression analysis
    print("\n🔬 Running differential expression analysis...")
    sc.tl.rank_genes_groups(
        adata_log, 
        groupby='ident',
        groups=NK_SUBTYPES,
        reference='rest',
        method='wilcoxon',
        n_genes=N_GENES_PER_SUBTYPE * 2,  # Get more to filter
        pts=True,
        corr_method='benjamini-hochberg'
    )
    print("✅ Differential expression analysis complete")
    
    # Extract and filter marker genes
    marker_genes = {}
    all_markers = set()
    
    def is_gene_to_exclude(gene_name):
        """Check if gene should be excluded based on patterns"""
        for pattern in GENE_EXCLUSION_PATTERNS:
            if re.match(pattern, gene_name, re.IGNORECASE):
                return True
        return False
    
    print("\n📋 Processing marker genes for each subtype...")
    # Vectorized mean expression calculation for all genes and subtypes
    expr_df = pd.DataFrame(adata_nk.X.toarray(), columns=adata_nk.var_names, index=adata_nk.obs_names)
    expr_df['subtype'] = adata_nk.obs['ident'].values
    mean_expr_df = expr_df.groupby('subtype').mean().T  # genes x subtypes

    initial_marker_genes = {}
    for subtype in NK_SUBTYPES:
        print(f"\n🔍 Processing {subtype}...")
        deg_df = sc.get.rank_genes_groups_df(adata_log, group=subtype)
        significant = deg_df[
            (deg_df['pvals_adj'] < PVAL_THRESHOLD) &
            (deg_df['logfoldchanges'] > LOGFC_THRESHOLD)
        ].copy()
        significant = significant[~significant['names'].apply(is_gene_to_exclude)].copy()
        significant = significant.sort_values('scores', ascending=False)
        top_genes = significant['names'].tolist()
        initial_marker_genes[subtype] = top_genes
        all_markers.update(top_genes)
        print(f"  {subtype}: {len(top_genes)} initial marker genes")

    # Filter for specificity: keep only genes with >=2x expression in target subtype vs mean of others
    specificity_threshold = 2.0
    filtered_marker_genes = {}
    for subtype in NK_SUBTYPES:
        specific = []
        for gene in initial_marker_genes[subtype]:
            if gene in mean_expr_df.index:
                expr_in_subtype = mean_expr_df.at[gene, subtype]
                expr_in_others = mean_expr_df.loc[gene, [s for s in NK_SUBTYPES if s != subtype]].mean()
            else:
                expr_in_subtype = 0.0
                expr_in_others = 0.0
            specificity = expr_in_subtype / (expr_in_others + 1e-6)
            if specificity >= specificity_threshold:
                specific.append((gene, specificity))
        # Sort by specificity and keep top 100
        specific = sorted(specific, key=lambda x: -x[1])[:100]
        filtered_marker_genes[subtype] = [g for g, s in specific]
        print(f"  {subtype}: {len(filtered_marker_genes[subtype])} specific marker genes after filtering")

    # Remove genes that appear in more than one subtype's marker list
    from collections import Counter
    all_filtered = [g for genes in filtered_marker_genes.values() for g in genes]
    marker_counts = Counter(all_filtered)
    unique_marker_genes = {
        subtype: [g for g in genes if marker_counts[g] == 1]
        for subtype, genes in filtered_marker_genes.items()
    }
    for subtype in NK_SUBTYPES:
        print(f"  {subtype}: {len(unique_marker_genes[subtype])} unique, specific marker genes")

    marker_genes = unique_marker_genes
    all_markers = set([g for genes in marker_genes.values() for g in genes])
    
    print(f"\n📊 Summary:")
    print(f"  - Total unique marker genes: {len(all_markers)}")
    print(f"  - Average markers per subtype: {len(all_markers) / len(NK_SUBTYPES):.1f}")
    
    # Step 4: Create signature matrix
    print("\n" + "="*60)
    print("STEP 4: SIGNATURE MATRIX CREATION")
    print("="*60)
    
    # Create signature matrix using TPM data
    print("🔬 Creating signature matrix from TPM data...")
    
    # Initialize signature matrix
    signature_matrix = pd.DataFrame(index=list(all_markers), columns=NK_SUBTYPES)
    signature_matrix.fillna(0, inplace=True)
    
    # Calculate mean TPM expression for each subtype
    for subtype in NK_SUBTYPES:
        if subtype in adata_nk.obs['ident'].unique():
            # Get cells for this subtype
            subtype_mask = adata_nk.obs['ident'] == subtype
            subtype_data = adata_nk[subtype_mask]
            
            # Calculate mean expression for marker genes
            for gene in all_markers:
                if gene in subtype_data.var_names:
                    gene_idx = list(subtype_data.var_names).index(gene)
                    mean_expr = subtype_data.X[:, gene_idx].mean()
                    signature_matrix.loc[gene, subtype] = mean_expr
    
    print(f"✅ Signature matrix created: {signature_matrix.shape}")
    print(f"📊 Matrix statistics:")
    print(f"  - Mean expression range: {signature_matrix.values.min():.3f} to {signature_matrix.values.max():.3f}")
    print(f"  - Genes with zero expression: {(signature_matrix == 0).sum().sum()}")
    print(f"  - Non-zero entries: {(signature_matrix != 0).sum().sum()}")
    
    # Step 5: Validation
    print("\n" + "="*60)
    print("STEP 5: SIGNATURE MATRIX VALIDATION")
    print("="*60)
    
    # Check for zero columns
    zero_cols = (signature_matrix == 0).all()
    if zero_cols.any():
        print(f"⚠️ Warning: Zero expression in columns: {zero_cols[zero_cols].index.tolist()}")
    else:
        print("✅ All subtypes have expression data")
    
    # Check for zero rows
    zero_rows = (signature_matrix == 0).all(axis=1)
    print(f"📊 Genes with zero expression across all subtypes: {zero_rows.sum()}")
    
    # Calculate subtype-specificity scores
    specificity_scores = {}
    print("\n🎯 Calculating subtype specificity scores...")
    
    for subtype in NK_SUBTYPES:
        if subtype in signature_matrix.columns:
            # Calculate how specific each gene is to this subtype
            subtype_expr = signature_matrix[subtype]
            other_expr = signature_matrix.drop(columns=[subtype]).mean(axis=1)
            specificity = subtype_expr / (other_expr + 1e-6)  # Avoid division by zero
            specificity_scores[subtype] = specificity
    
    # Show top specific genes for each subtype
    print("\n🏆 Top 5 most specific genes per subtype:")
    for subtype, scores in specificity_scores.items():
        top_genes = scores.nlargest(5)
        print(f"\n{subtype}:")
        for gene, score in top_genes.items():
            expr = signature_matrix.loc[gene, subtype]
            print(f"  {gene}: specificity={score:.2f}, expr={expr:.3f}")
    
    # Step 6: Save results
    print("\n" + "="*60)
    print("STEP 6: SAVING SIGNATURE MATRIX")
    print("="*60)
    
    # Save signature matrix as TSV (CIBERSORTx format)
    output_path = os.path.join(OUTPUT_DIR, "NK_signature_matrix_TPM.tsv")
    signature_matrix.to_csv(output_path, sep='\t')
    print(f"✅ Signature matrix saved to: {output_path}")
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "signature_matrix_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("NK SUBTYPE SIGNATURE MATRIX METADATA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Creation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source data: {TPM_DATA_PATH}\n")
        f.write(f"Matrix dimensions: {signature_matrix.shape}\n")
        f.write(f"Subtypes: {list(signature_matrix.columns)}\n")
        f.write(f"Analysis parameters:\n")
        f.write(f"  - Genes per subtype: {N_GENES_PER_SUBTYPE}\n")
        f.write(f"  - LogFC threshold: {LOGFC_THRESHOLD}\n")
        f.write(f"  - P-value threshold: {PVAL_THRESHOLD}\n\n")
        
        # Subtype cell counts
        f.write("CELL COUNTS PER SUBTYPE:\n")
        for subtype in NK_SUBTYPES:
            if subtype in adata_nk.obs['ident'].value_counts():
                count = adata_nk.obs['ident'].value_counts()[subtype]
                f.write(f"  {subtype}: {count:,} cells\n")
    
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Save marker gene lists
    markers_path = os.path.join(OUTPUT_DIR, "marker_genes_by_subtype.txt")
    with open(markers_path, 'w') as f:
        f.write("MARKER GENES BY SUBTYPE\n")
        f.write("=" * 30 + "\n\n")
        for subtype, genes in marker_genes.items():
            f.write(f"{subtype} ({len(genes)} genes):\n")
            f.write(", ".join(genes) + "\n\n")
    
    print(f"✅ Marker gene lists saved to: {markers_path}")
    
    # Step 7: Create visualizations
    print("\n" + "="*60)
    print("STEP 7: CREATING QUALITY CONTROL VISUALIZATIONS")
    print("="*60)
    
    # Set up plotting
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Subtype distribution
    subtype_counts = adata_nk.obs['ident'].value_counts()
    axes[0,0].bar(subtype_counts.index, subtype_counts.values, color='skyblue')
    axes[0,0].set_title('NK Subtype Distribution', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Number of Cells', fontsize=12)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. TPM sums distribution
    axes[0,1].hist(tpm_sums, bins=50, alpha=0.7, color='lightcoral')
    axes[0,1].axvline(tpm_sums.mean(), color='red', linestyle='--', label=f'Mean: {tpm_sums.mean():.0f}')
    axes[0,1].set_xlabel('TPM Sum per Cell', fontsize=12)
    axes[0,1].set_ylabel('Number of Cells', fontsize=12)
    axes[0,1].set_title('TPM Normalization Validation', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    
    # 3. TPM distribution
    tpm_values = adata_nk.X.toarray().flatten()
    tpm_values = tpm_values[tpm_values > 0]  # Only non-zero values
    axes[1,0].hist(tpm_values, bins=50, alpha=0.7, color='lightgreen')
    axes[1,0].set_xlabel('TPM Values', fontsize=12)
    axes[1,0].set_ylabel('Frequency', fontsize=12)
    axes[1,0].set_title('TPM Distribution (Non-zero)', fontsize=14, fontweight='bold')
    axes[1,0].set_xscale('log')
    
    # 4. Signature matrix heatmap (top genes)
    top_genes_per_subtype = 10
    genes_for_heatmap = []
    for subtype in NK_SUBTYPES:
        if subtype in signature_matrix.columns:
            # Get top genes for this subtype
            subtype_expr = signature_matrix[subtype]
            top_genes = subtype_expr.nlargest(top_genes_per_subtype).index
            genes_for_heatmap.extend(top_genes)
    
    # Remove duplicates and limit
    genes_for_heatmap = list(set(genes_for_heatmap))[:30]
    
    if genes_for_heatmap:
        heatmap_data = signature_matrix.loc[genes_for_heatmap]
        sns.heatmap(heatmap_data, ax=axes[1,1], cmap='viridis', 
                   xticklabels=True, yticklabels=True, cbar_kws={'label': 'TPM'})
        axes[1,1].set_title('Top Marker Genes Heatmap', fontsize=14, fontweight='bold')
    else:
        axes[1,1].text(0.5, 0.5, 'No marker genes available', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Top Marker Genes Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "quality_control_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Quality control plots saved to: {plot_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 SIGNATURE MATRIX GENERATION COMPLETE!")
    print("="*80)
    
    print(f"📊 Final Results:")
    print(f"  - Signature matrix: {signature_matrix.shape}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Files created:")
    print(f"    • NK_signature_matrix_TPM.tsv (CIBERSORTx format)")
    print(f"    • signature_matrix_metadata.txt")
    print(f"    • marker_genes_by_subtype.txt")
    print(f"    • quality_control_plots.png")
    
    print(f"\n🔬 Matrix Quality:")
    print(f"  - Mean expression: {signature_matrix.values.mean():.3f}")
    print(f"  - Expression range: {signature_matrix.values.min():.3f} to {signature_matrix.values.max():.3f}")
    print(f"  - Non-zero entries: {(signature_matrix != 0).sum().sum():,}")
    
    print(f"\n📋 Subtype Summary:")
    for subtype in NK_SUBTYPES:
        if subtype in signature_matrix.columns:
            cell_count = adata_nk.obs['ident'].value_counts().get(subtype, 0)
            marker_count = len(marker_genes.get(subtype, []))
            mean_expr = signature_matrix[subtype].mean()
            print(f"  {subtype}: {cell_count:,} cells, {marker_count} markers, mean TPM={mean_expr:.3f}")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Upload NK_signature_matrix_TPM.tsv to CIBERSORTx")
    print(f"  2. Set parameters: 1000 permutations, quantile normalization disabled")
    print(f"  3. Upload your bulk RNA-seq data (TPM or raw counts)")
    print(f"  4. Run deconvolution and download results")
    
    print(f"\n✅ Your TPM-based signature matrix is ready for CIBERSORTx deconvolution!")
    
    return signature_matrix, marker_genes, OUTPUT_DIR

if __name__ == "__main__":
    signature_matrix, marker_genes, output_dir = main() 