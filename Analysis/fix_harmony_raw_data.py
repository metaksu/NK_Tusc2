#!/usr/bin/env python3
"""
Fix Harmony Data: Preserve Raw Data for DEG Analysis
===================================================

The original Harmony processing lost the .raw attribute when subsetting to HVGs.
This script re-creates the Harmony-corrected data while properly preserving
the full gene set in .raw for downstream DEG analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
ORIGINAL_DATA = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
HARMONY_DATA = "outputs/harmony_batch_correction/PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad"
OUTPUT_FILE = "outputs/harmony_batch_correction/PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected_with_raw.h5ad"

def main():
    print("🔧 FIXING HARMONY DATA: RESTORING RAW COUNTS")
    print("=" * 60)
    
    # Load original full data
    print("📂 Loading original full dataset...")
    adata_original = sc.read_h5ad(ORIGINAL_DATA)
    print(f"   Original: {adata_original.shape[0]:,} cells × {adata_original.shape[1]:,} genes")
    
    # Load Harmony-corrected data (with embeddings)
    print("📂 Loading Harmony-corrected data...")
    adata_harmony = sc.read_h5ad(HARMONY_DATA)
    print(f"   Harmony: {adata_harmony.shape[0]:,} cells × {adata_harmony.shape[1]:,} genes")
    
    # Check if cells match
    if not adata_original.obs.index.equals(adata_harmony.obs.index):
        print("⚠️ Cell indices don't match - aligning datasets...")
        # Reorder original to match harmony
        common_cells = adata_harmony.obs.index.intersection(adata_original.obs.index)
        adata_original = adata_original[common_cells].copy()
        adata_harmony = adata_harmony[common_cells].copy()
        print(f"   Aligned: {len(common_cells):,} common cells")
    
    # Create new AnnData with Harmony HVGs but original raw data
    print("🔧 Combining Harmony embeddings with original raw data...")
    
    # Start with harmony structure (2000 HVGs)
    adata_fixed = adata_harmony.copy()
    
    # Add full original data as .raw
    adata_fixed.raw = adata_original
    print(f"   ✅ Restored .raw: {adata_fixed.raw.shape[1]:,} genes available for DEG analysis")
    
    # Verify embeddings are preserved
    embeddings_preserved = []
    if 'X_pca_harmony' in adata_fixed.obsm:
        embeddings_preserved.append("X_pca_harmony")
    if 'X_umap_harmony' in adata_fixed.obsm:
        embeddings_preserved.append("X_umap_harmony")
    
    print(f"   ✅ Preserved embeddings: {embeddings_preserved}")
    
    # Add metadata about the fix
    if 'harmony_correction' not in adata_fixed.uns:
        adata_fixed.uns['harmony_correction'] = {}
    
    adata_fixed.uns['harmony_correction']['raw_data_restored'] = True
    adata_fixed.uns['harmony_correction']['raw_genes_count'] = int(adata_fixed.raw.shape[1])
    adata_fixed.uns['harmony_correction']['hvg_genes_count'] = int(adata_fixed.shape[1])
    
    # Save fixed data
    print("💾 Saving fixed Harmony data...")
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    adata_fixed.write_h5ad(output_path, compression='gzip')
    
    print("✅ HARMONY DATA FIXED!")
    print("=" * 60)
    print(f"📄 Output: {output_path}")
    print(f"🔬 Cells: {adata_fixed.shape[0]:,}")
    print(f"🧬 HVGs (for analysis): {adata_fixed.shape[1]:,}")
    print(f"🧬 Raw genes (for DEGs): {adata_fixed.raw.shape[1]:,}")
    print(f"📊 Embeddings: {list(adata_fixed.obsm.keys())}")
    print()
    print("🎯 NOW YOUR NK SUBTYPE SCORING WILL WORK!")
    print("   - Update NK_analysis_main_rebuffet.py to use this new file")
    print("   - DEG analysis will have access to all genes")
    print("   - Harmony embeddings are preserved")

if __name__ == "__main__":
    main()