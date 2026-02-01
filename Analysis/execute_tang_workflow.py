#!/usr/bin/env python3
"""
Direct execution of Tang reference matrix generation
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
from pathlib import Path
from datetime import datetime

def main():
    """Execute Tang reference matrix generation workflow."""
    
    print("=== Tang CIBERSORTx Reference Matrix Generation ===")
    
    # Setup paths
    tang_data_path = "data/processed/comb_CD56_CD16_NK.h5ad"
    output_dir = Path("outputs/cibersortx_reference_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory created: {output_dir}")
    
    try:
        # Load Tang data
        print(f"Loading Tang NK data from: {tang_data_path}")
        adata = sc.read_h5ad(tang_data_path)
        print(f"✓ Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
        
        # Check metadata
        print(f"Available columns: {list(adata.obs.columns)[:10]}...")
        
        # Check for celltype column
        celltype_col = "celltype"
        if celltype_col not in adata.obs.columns:
            print(f"ERROR: '{celltype_col}' column not found")
            return False
            
        # Analyze cell types
        celltype_counts = adata.obs[celltype_col].value_counts()
        print(f"Found {len(celltype_counts)} cell types")
        print("Top 10 cell types:")
        for ct, count in celltype_counts.head(10).items():
            print(f"  {ct}: {count:,} cells")
        
        # Filter cell types with sufficient cells (>=5)
        min_cells = 5
        valid_celltypes = celltype_counts[celltype_counts >= min_cells].index
        print(f"Cell types with >= {min_cells} cells: {len(valid_celltypes)}")
        
        # Filter data
        celltype_mask = adata.obs[celltype_col].isin(valid_celltypes)
        adata_filtered = adata[celltype_mask, :].copy()
        print(f"✓ Filtered dataset: {adata_filtered.shape[0]:,} cells")
        
        # Get expression data
        if hasattr(adata_filtered, 'raw') and adata_filtered.raw is not None:
            expr_data = adata_filtered.raw.X
            gene_names = adata_filtered.raw.var_names
            print("✓ Using raw expression data")
        else:
            expr_data = adata_filtered.X
            gene_names = adata_filtered.var_names
            print("✓ Using main expression data")
            
        # Convert sparse to dense
        if hasattr(expr_data, 'toarray'):
            print("Converting sparse matrix to dense...")
            expr_data = expr_data.toarray()
            print("✓ Conversion complete")
            
        # Balance cell types (max 500 cells per type)
        cell_types = adata_filtered.obs[celltype_col].values
        max_cells_per_type = 500
        print(f"Balancing to max {max_cells_per_type} cells per type...")
        
        balanced_indices = []
        np.random.seed(42)
        final_counts = {}
        
        for celltype in np.unique(cell_types):
            celltype_indices = np.where(cell_types == celltype)[0]
            
            if len(celltype_indices) > max_cells_per_type:
                selected = np.random.choice(celltype_indices, max_cells_per_type, replace=False)
                final_counts[celltype] = max_cells_per_type
            else:
                selected = celltype_indices
                final_counts[celltype] = len(celltype_indices)
                
            balanced_indices.extend(selected)
            
        balanced_indices = np.array(balanced_indices)
        expr_balanced = expr_data[balanced_indices, :]
        celltypes_balanced = cell_types[balanced_indices]
        
        print(f"✓ Balanced dataset: {len(balanced_indices):,} cells")
        
        # Create reference matrix
        print("Creating reference matrix (genes x cells)...")
        ref_matrix = pd.DataFrame(
            expr_balanced.T,  # Transpose to genes x cells
            index=gene_names,
            columns=celltypes_balanced
        )
        
        # Remove zero-expression genes
        gene_sums = ref_matrix.sum(axis=1)
        non_zero_genes = gene_sums > 0
        ref_matrix = ref_matrix.loc[non_zero_genes]
        removed_genes = (~non_zero_genes).sum()
        
        print(f"✓ Removed {removed_genes:,} zero-expression genes")
        print(f"✓ Final matrix: {ref_matrix.shape[0]:,} genes x {ref_matrix.shape[1]:,} cells")
        
        # Calculate statistics
        max_expr = ref_matrix.values.max()
        min_expr = ref_matrix.values.min()
        mean_expr = ref_matrix.values.mean()
        
        print(f"Expression statistics:")
        print(f"  Range: {min_expr:.3f} to {max_expr:.3f}")
        print(f"  Mean: {mean_expr:.3f}")
        
        # Save reference matrix
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"Tang_NK_reference_matrix_max{max_cells_per_type}cells_{date_str}.txt"
        output_path = output_dir / filename
        
        print(f"Saving reference matrix to: {output_path}")
        ref_matrix.to_csv(output_path, sep='\t', index=True, header=True)
        
        # Create summary
        celltype_dist = ref_matrix.columns.value_counts().sort_values(ascending=False)
        summary_path = output_dir / f"Tang_NK_reference_matrix_max{max_cells_per_type}cells_{date_str}_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Tang NK Cell CIBERSORTx Reference Matrix Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Creation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Matrix dimensions: {ref_matrix.shape[0]:,} genes x {ref_matrix.shape[1]:,} cells\n")
            f.write(f"Cell types: {len(celltype_dist)}\n\n")
            
            f.write("PHENOTYPE DISTRIBUTION:\n")
            for celltype, count in celltype_dist.items():
                f.write(f"  {celltype}: {count} cells\n")
                
            f.write(f"\nEXPRESSION STATISTICS:\n")
            f.write(f"  Range: {min_expr:.3f} to {max_expr:.3f}\n")
            f.write(f"  Mean: {mean_expr:.3f}\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
        print(f"\n{'='*60}")
        print("TANG CIBERSORTX REFERENCE MATRIX GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Reference matrix: {output_path}")
        print(f"✓ Summary report: {summary_path}")
        print(f"✓ Dimensions: {ref_matrix.shape[0]:,} genes × {ref_matrix.shape[1]:,} cells")
        print(f"✓ Cell types: {len(celltype_dist)}")
        print(f"✓ Expression range: {min_expr:.3f} to {max_expr:.3f}")
        print("\nReady for CIBERSORTx signature matrix generation!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 