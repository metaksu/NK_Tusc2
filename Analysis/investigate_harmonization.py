import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the output directory for figures
FIG_DIR = "harmonization_analysis_figs"
os.makedirs(FIG_DIR, exist_ok=True)

# Load the Tang combined dataset
try:
    adata_tang_full = sc.read_h5ad(r"C:\Users\met-a\Documents\Analysis\data\processed\comb_CD56_CD16_NK.h5ad")
    print("--- Successfully loaded Tang dataset ---")

    # --- Analytical Investigation ---
    print("\n--- Starting Analytical Investigation for Tang Dataset ---")
    
    # 1. Preprocessing (using the same logic as the main script)
    print("Step 1: Running basic preprocessing...")
    # Ensure we have unscaled counts for HVG detection
    if 'counts' in adata_tang_full.layers:
        adata_tang_full.X = adata_tang_full.layers['counts'].copy()
    
    sc.pp.normalize_total(adata_tang_full, target_sum=1e4)
    sc.pp.log1p(adata_tang_full)
    
    # 2. Find Highly Variable Genes
    print("Step 2: Identifying Highly Variable Genes...")
    sc.pp.highly_variable_genes(adata_tang_full, n_top_genes=2000, flavor='seurat', subset=True)
    
    # 3. Dimensionality Reduction
    print("Step 3: Running PCA and UMAP...")
    sc.pp.scale(adata_tang_full, max_value=10)
    sc.tl.pca(adata_tang_full, svd_solver='arpack')
    sc.pp.neighbors(adata_tang_full, n_pcs=30)
    sc.tl.umap(adata_tang_full)
    
    # 4. Visualization
    print("Step 4: Generating UMAP visualizations...")
    
    # Plot UMAP colored by 'datasets'
    plt.figure(figsize=(12, 10))
    sc.pl.umap(adata_tang_full, color='datasets', show=False, legend_loc='on data', frameon=False)
    plt.title('Tang Dataset UMAP Colored by Source Dataset', fontsize=16)
    plt.savefig(os.path.join(FIG_DIR, 'tang_umap_by_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved UMAP colored by 'datasets' to {os.path.join(FIG_DIR, 'tang_umap_by_dataset.png')}")

    # Plot UMAP colored by 'cell_type_major' for comparison
    cell_type_col = 'initial_clustering' # Using a likely cell type column
    if cell_type_col in adata_tang_full.obs.columns:
        plt.figure(figsize=(12, 10))
        sc.pl.umap(adata_tang_full, color=cell_type_col, show=False, legend_loc='on data', frameon=False)
        plt.title(f'Tang Dataset UMAP Colored by Cell Type ({cell_type_col})', fontsize=16)
        plt.savefig(os.path.join(FIG_DIR, f'tang_umap_by_{cell_type_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Saved UMAP colored by '{cell_type_col}' to {os.path.join(FIG_DIR, f'tang_umap_by_{cell_type_col}.png')}")
    else:
        print(f"  - Could not find '{cell_type_col}' column for comparison plot.")

    print("\n--- Analytical Investigation Complete ---")

except FileNotFoundError:
    print("ERROR: Could not find the Tang dataset file at the specified path.")
except Exception as e:
    print(f"An error occurred: {e}") 