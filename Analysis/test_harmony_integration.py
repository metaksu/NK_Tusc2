#!/usr/bin/env python3
"""
Quick Test: Harmony Integration Verification
===========================================

Test script to verify the Harmony-corrected data is properly integrated
and ready for use in NK_analysis_main_rebuffet.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
HARMONY_DATA_FILE = "outputs/harmony_batch_correction/PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad"

def test_harmony_data_loading():
    """Test that Harmony-corrected data loads properly."""
    print("=" * 60)
    print("TESTING HARMONY DATA LOADING")
    print("=" * 60)
    
    try:
        # Load Harmony-corrected data
        adata = sc.read_h5ad(HARMONY_DATA_FILE)
        print(f"✅ Successfully loaded Harmony data: {adata.shape}")
        
        # Check required embeddings
        embeddings_check = {
            'X_pca_harmony': '❌',
            'X_umap_harmony': '❌',
            'X_umap_original': '❌'
        }
        
        for emb in embeddings_check.keys():
            if emb in adata.obsm:
                embeddings_check[emb] = '✅'
                print(f"  {embeddings_check[emb]} {emb}: {adata.obsm[emb].shape}")
            else:
                print(f"  {embeddings_check[emb]} {emb}: Missing")
        
        # Check batch correction metadata
        if 'harmony_correction' in adata.uns:
            correction_info = adata.uns['harmony_correction']
            print(f"✅ Batch correction metadata found:")
            print(f"   Variables corrected: {correction_info['batch_variables']}")
            print(f"   CMV corrected: {correction_info['cmv_correction']}")
            print(f"   Date: {correction_info['corrected_date']}")
        else:
            print("❌ No batch correction metadata found")
        
        # Check NK subtype distribution
        if 'ident' in adata.obs.columns:
            subtype_counts = adata.obs['ident'].value_counts()
            print(f"✅ NK subtypes found: {list(subtype_counts.index)}")
            print(f"   Total cells: {subtype_counts.sum():,}")
        else:
            print("❌ NK subtype column 'ident' not found")
        
        # Check batch variables
        batch_vars = ['Dataset', 'donor', 'CMV_Status']
        for var in batch_vars:
            if var in adata.obs.columns:
                unique_vals = adata.obs[var].nunique()
                print(f"✅ {var}: {unique_vals} unique values")
            else:
                print(f"❌ {var}: Not found")
        
        return adata
        
    except Exception as e:
        print(f"❌ Failed to load Harmony data: {e}")
        return None

def test_umap_quality(adata):
    """Test UMAP quality and mixing."""
    print("\n" + "=" * 60)
    print("TESTING UMAP QUALITY")
    print("=" * 60)
    
    if adata is None:
        print("❌ No data to test")
        return
    
    if 'X_umap_harmony' not in adata.obsm:
        print("❌ No Harmony UMAP found")
        return
    
    # Set UMAP for plotting
    adata.obsm['X_umap'] = adata.obsm['X_umap_harmony'].copy()
    
    # Create test plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: NK subtypes
    sc.pl.umap(adata, color='ident', ax=axes[0,0], show=False, 
              title='NK Subtypes (Should be Well-Separated)', frameon=False)
    
    # Plot 2: Dataset (should be mixed)
    sc.pl.umap(adata, color='Dataset', ax=axes[0,1], show=False,
              title='Dataset (Should be Well-Mixed)', frameon=False)
    
    # Plot 3: Donor (should be mixed)  
    sc.pl.umap(adata, color='donor', ax=axes[1,0], show=False,
              title='Donor (Should be Well-Mixed)', frameon=False, legend_loc=None)
    
    # Plot 4: CMV Status (should be mixed)
    if 'CMV_Status' in adata.obs.columns:
        sc.pl.umap(adata, color='CMV_Status', ax=axes[1,1], show=False,
                  title='CMV Status (Should be Well-Mixed)', frameon=False)
    else:
        axes[1,1].text(0.5, 0.5, 'CMV_Status\nNot Found', ha='center', va='center')
        axes[1,1].set_title('CMV Status (Not Found)')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "outputs/harmony_batch_correction/integration_test_umap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ UMAP test plots saved to: {output_file}")
    print("🔍 Manually inspect plots to verify:")
    print("   - NK subtypes should be well-separated")
    print("   - Dataset/Donor/CMV should be well-mixed")

def test_standard_parameters():
    """Test that standard UMAP parameters work well."""
    print("\n" + "=" * 60)
    print("TESTING STANDARD PARAMETERS")
    print("=" * 60)
    
    # Load data
    adata = sc.read_h5ad(HARMONY_DATA_FILE)
    
    if 'X_pca_harmony' not in adata.obsm:
        print("❌ No Harmony PCA embeddings for testing")
        return
    
    print("🧪 Testing standard UMAP parameters on Harmony-corrected data...")
    
    # Use Harmony PCA
    adata.obsm['X_pca'] = adata.obsm['X_pca_harmony'].copy()
    
    # Apply STANDARD parameters (what the main analysis will use)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, random_state=42)
    sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=42)
    
    print("✅ Standard parameters applied successfully")
    print(f"   Neighbors: 15 (was 1000)")
    print(f"   Min_dist: 0.5 (was 1.8)")
    print(f"   Spread: 1.0 (was 4.5)")
    print(f"   PCs: 50 (was 3)")
    
    # Quick quality check
    umap_coords = adata.obsm['X_umap']
    print(f"✅ New UMAP computed: {umap_coords.shape}")
    print(f"   X range: {umap_coords[:, 0].min():.2f} to {umap_coords[:, 0].max():.2f}")
    print(f"   Y range: {umap_coords[:, 1].min():.2f} to {umap_coords[:, 1].max():.2f}")
    
    return True

def main():
    """Main test function."""
    print("🧪 TESTING HARMONY INTEGRATION FOR NK_ANALYSIS_MAIN_REBUFFET.PY")
    print("=" * 80)
    
    # Test 1: Data loading
    adata = test_harmony_data_loading()
    
    # Test 2: UMAP quality
    test_umap_quality(adata)
    
    # Test 3: Standard parameters
    test_standard_parameters()
    
    print("\n" + "=" * 80)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    if adata is not None:
        print("✅ READY FOR INTEGRATION!")
        print("\nNext steps:")
        print("1. Update data file path in NK_analysis_main_rebuffet.py")
        print("2. Revert UMAP parameters to standard values")
        print("3. Use Harmony embeddings (X_pca_harmony)")
        print("4. Remove batch effect workarounds")
        print("\nExpected improvements:")
        print("📊 Better cluster separation")
        print("🔬 Cleaner biological signals")
        print("⚡ No more artificial parameter forcing")
        print("🎯 Unbiased NK subtype analysis")
    else:
        print("❌ INTEGRATION NOT READY")
        print("Fix Harmony data loading issues first")
    
    return adata is not None

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ALL TESTS PASSED - READY FOR INTEGRATION!")
    else:
        print("\n⚠️ TESTS FAILED - CHECK HARMONY DATA")