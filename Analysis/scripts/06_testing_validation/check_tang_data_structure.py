#!/usr/bin/env python3
"""
Check Tang Dataset Structure for Raw/Linear Data
Investigate what data formats are available in the Tang combined dataset
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from scipy import sparse


def main():
    print("🔬 INVESTIGATING TANG DATASET STRUCTURE")
    print("=" * 60)

    # Load Tang dataset
    tang_path = "../../data/processed/comb_CD56_CD16_NK.h5ad"
    print(f"📁 Loading: {tang_path}")

    if not os.path.exists(tang_path):
        print(f"❌ File not found: {tang_path}")
        return

    adata = sc.read_h5ad(tang_path)
    print(f"✅ Data loaded: {adata.shape}")

    print("\n📊 MAIN DATA MATRIX (.X):")
    print(f"  - Type: {type(adata.X).__name__}")
    print(f"  - Shape: {adata.X.shape}")

    # Get sample of main data
    if sparse.issparse(adata.X):
        sample_data = adata.X[:1000, :100].toarray()
    else:
        sample_data = adata.X[:1000, :100]

    print(f"  - Range: {sample_data.min():.3f} to {sample_data.max():.3f}")
    print(f"  - Mean: {sample_data.mean():.3f}")
    print(f"  - Std: {sample_data.std():.3f}")

    # Check for typical log-normalized vs linear patterns
    print(f"  - Max value analysis:")
    if sample_data.max() < 15:
        print(f"    → Likely LOG-NORMALIZED (max < 15)")
    elif sample_data.max() > 1000:
        print(f"    → Likely RAW COUNTS (max > 1000)")
    else:
        print(f"    → Likely LINEAR NORMALIZED (15 < max < 1000)")

    print("\n🗂️ AVAILABLE LAYERS:")
    if adata.layers:
        for layer_name in adata.layers.keys():
            layer_data = adata.layers[layer_name]
            if sparse.issparse(layer_data):
                sample_layer = layer_data[:1000, :100].toarray()
            else:
                sample_layer = layer_data[:1000, :100]

            print(f"  - {layer_name}:")
            print(f"    Type: {type(layer_data).__name__}")
            print(f"    Range: {sample_layer.min():.3f} to {sample_layer.max():.3f}")
            print(f"    Mean: {sample_layer.mean():.3f}")

            # Analyze what this layer might be
            if sample_layer.max() < 15:
                data_type = "LOG-NORMALIZED"
            elif sample_layer.max() > 1000:
                data_type = "RAW COUNTS"
            else:
                data_type = "LINEAR NORMALIZED (TPM/CPM)"
            print(f"    → Likely: {data_type}")
    else:
        print("  - No layers found")

    print("\n🔬 RAW DATA (.raw):")
    if adata.raw is not None:
        print(f"  - Available: Yes")
        print(f"  - Shape: {adata.raw.X.shape}")
        print(f"  - Type: {type(adata.raw.X).__name__}")

        # Sample raw data
        if sparse.issparse(adata.raw.X):
            sample_raw = adata.raw.X[:1000, :100].toarray()
        else:
            sample_raw = adata.raw.X[:1000, :100]

        print(f"  - Range: {sample_raw.min():.3f} to {sample_raw.max():.3f}")
        print(f"  - Mean: {sample_raw.mean():.3f}")

        if sample_raw.max() < 15:
            data_type = "LOG-NORMALIZED"
        elif sample_raw.max() > 1000:
            data_type = "RAW COUNTS"
        else:
            data_type = "LINEAR NORMALIZED (TPM/CPM)"
        print(f"  - → Likely: {data_type}")
    else:
        print("  - Available: No")

    print("\n📋 METADATA SAMPLE:")
    print(f"  - Columns: {list(adata.obs.columns)}")

    # Check a few cells from tumor tissue
    tumor_mask = adata.obs["meta_tissue_in_paper"] == "Tumor"
    tumor_sample = adata[tumor_mask][:5]

    print(f"\n🎯 TUMOR TISSUE SAMPLE (5 cells):")
    print(f"  - Subtypes: {list(tumor_sample.obs['celltype'].values)}")

    # Look at a few genes in these cells
    test_genes = ["GZMB", "GNLY", "IFNG", "TNF", "IL2"]
    available_genes = [g for g in test_genes if g in adata.var_names]

    if available_genes:
        print(f"\n🧬 EXPRESSION VALUES FOR {available_genes} in tumor sample:")

        print("  Main matrix (.X):")
        for gene in available_genes:
            gene_idx = list(adata.var_names).index(gene)
            if sparse.issparse(tumor_sample.X):
                values = tumor_sample.X[:, gene_idx].toarray().flatten()
            else:
                values = tumor_sample.X[:, gene_idx]
            print(f"    {gene}: {values}")

        if tumor_sample.raw is not None:
            print("  Raw matrix (.raw.X):")
            for gene in available_genes:
                gene_idx = list(tumor_sample.raw.var_names).index(gene)
                if sparse.issparse(tumor_sample.raw.X):
                    values = tumor_sample.raw.X[:, gene_idx].toarray().flatten()
                else:
                    values = tumor_sample.raw.X[:, gene_idx]
                print(f"    {gene}: {values}")

    print("\n💡 RECOMMENDATIONS:")
    print("Based on the analysis above:")

    # Check what we found
    main_max = sample_data.max()
    if main_max < 15:
        print("❌ Main matrix (.X) appears to be LOG-NORMALIZED")
        print("   → NOT suitable for CIBERSORTx signature matrix")

        if adata.raw is not None:
            if sparse.issparse(adata.raw.X):
                raw_sample = adata.raw.X[:1000, :100].toarray()
            else:
                raw_sample = adata.raw.X[:1000, :100]
            raw_max = raw_sample.max()

            if raw_max > 1000:
                print("✅ Raw matrix (.raw.X) appears to be RAW COUNTS")
                print("   → Can be converted to TPM for CIBERSORTx")
            elif 15 < raw_max < 1000:
                print("✅ Raw matrix (.raw.X) appears to be LINEAR NORMALIZED")
                print("   → Suitable for CIBERSORTx signature matrix")
            else:
                print("❌ Raw matrix (.raw.X) also appears log-normalized")

        # Check layers
        if adata.layers:
            print("\n🔍 Checking layers for linear data:")
            for layer_name in adata.layers.keys():
                layer_data = adata.layers[layer_name]
                if sparse.issparse(layer_data):
                    sample_layer = layer_data[:1000, :100].toarray()
                else:
                    sample_layer = layer_data[:1000, :100]

                if sample_layer.max() > 15:
                    print(f"✅ Layer '{layer_name}' might contain linear data")
                    print(f"   → Consider using for CIBERSORTx")

    elif main_max > 1000:
        print("✅ Main matrix (.X) appears to be RAW COUNTS")
        print("   → Can be converted to TPM for CIBERSORTx")
    else:
        print("✅ Main matrix (.X) appears to be LINEAR NORMALIZED")
        print("   → Suitable for CIBERSORTx signature matrix")


if __name__ == "__main__":
    main()
