#!/usr/bin/env python3
"""
Integration Guide: Harmony-Corrected Data into NK_analysis_main_rebuffet.py
===========================================================================

This script shows exactly how to:
1. Load Harmony-corrected data instead of original data
2. Revert UMAP parameters to standard settings
3. Update the analysis pipeline to use corrected embeddings

Key changes needed in NK_analysis_main_rebuffet.py:
"""

# === PATH UPDATES ===
# Replace the original data loading path:

# OLD (in NK_analysis_main_rebuffet.py):
# REBUFFET_DATA_FILE = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"

# NEW (update to use Harmony-corrected data):
REBUFFET_DATA_FILE = "outputs/harmony_batch_correction/PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad"

# === UMAP PARAMETER CHANGES ===
# In the run_dim_reduction_pipeline function, REVERT to standard parameters:

def run_dim_reduction_pipeline_CORRECTED(adata_obj, cohort_label, n_pcs_to_use=15, n_hvgs=1000):
    """
    Updated dimension reduction pipeline for Harmony-corrected data.
    
    MAJOR CHANGE: Now using STANDARD UMAP parameters since batch effects are corrected!
    """
    
    # Use Harmony-corrected embeddings if available
    if 'X_pca_harmony' in adata_obj.obsm:
        print("🎯 Using Harmony-corrected embeddings for analysis")
        use_rep = 'X_pca_harmony'
    else:
        print("⚠️ No Harmony embeddings found, using original PCA")
        use_rep = 'X_pca'
    
    # === UMAP WITH STANDARD PARAMETERS ===
    # REMOVE any artificial modifications that were compensating for batch effects
    
    # STANDARD UMAP parameters (no more forcing groups together):
    sc.pp.neighbors(
        adata_obj, 
        use_rep=use_rep,  # Use Harmony-corrected embeddings
        n_neighbors=15,   # Standard (was probably modified before)
        n_pcs=n_pcs_to_use,
        random_state=42
    )
    
    sc.tl.umap(
        adata_obj,
        min_dist=0.5,     # STANDARD (was probably lower to force closer)
        spread=1.0,       # STANDARD (was probably modified)
        n_components=2,
        random_state=42
    )
    
    print(f"✅ Standard UMAP computed on {use_rep} embeddings")
    return adata_obj

# === DATA LOADING MODIFICATIONS ===
def load_harmony_corrected_data():
    """
    Load Harmony-corrected NK data with proper embeddings.
    """
    
    # Load the corrected data
    adata = sc.read_h5ad(REBUFFET_DATA_FILE)
    print(f"✅ Loaded Harmony-corrected data: {adata.shape}")
    
    # Verify Harmony embeddings are present
    if 'X_pca_harmony' in adata.obsm:
        print("✅ Harmony-corrected PCA embeddings found")
    else:
        print("❌ WARNING: No Harmony embeddings found!")
    
    if 'X_umap_harmony' in adata.obsm:
        print("✅ Harmony-corrected UMAP found")
        # Use the pre-computed Harmony UMAP
        adata.obsm['X_umap'] = adata.obsm['X_umap_harmony'].copy()
    
    # Check batch correction metadata
    if 'harmony_correction' in adata.uns:
        correction_info = adata.uns['harmony_correction']
        print(f"✅ Batch correction applied on: {correction_info['corrected_date']}")
        print(f"   Variables corrected: {correction_info['batch_variables']}")
        print(f"   CMV corrected: {correction_info['cmv_correction']}")
    
    return adata

# === SPECIFIC CHANGES FOR NK_ANALYSIS_MAIN_REBUFFET.PY ===

# 1. UPDATE DATA LOADING SECTION (around line 100-200):
"""
# OLD:
adata_blood = sc.read_h5ad(REBUFFET_DATA_FILE)

# NEW:
adata_blood = sc.read_h5ad("outputs/harmony_batch_correction/PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad")

# Ensure we use Harmony embeddings
if 'X_pca_harmony' in adata_blood.obsm:
    adata_blood.obsm['X_pca'] = adata_blood.obsm['X_pca_harmony'].copy()
    print("🎯 Using Harmony-corrected PCA embeddings")

if 'X_umap_harmony' in adata_blood.obsm:
    adata_blood.obsm['X_umap'] = adata_blood.obsm['X_umap_harmony'].copy()
    print("🎯 Using Harmony-corrected UMAP")
"""

# 2. UPDATE run_dim_reduction_pipeline FUNCTION:
"""
FIND lines that look like:

# Potentially modified parameters to compensate for batch effects:
sc.pp.neighbors(adata_obj, n_neighbors=XX, ...)  # XX might be non-standard
sc.tl.umap(adata_obj, min_dist=XX, spread=XX, ...)  # XX might be modified

REPLACE WITH STANDARD:
sc.pp.neighbors(adata_obj, use_rep='X_pca_harmony', n_neighbors=15, random_state=42)
sc.tl.umap(adata_obj, min_dist=0.5, spread=1.0, random_state=42)
"""

# 3. REMOVE ANY BATCH EFFECT WORKAROUNDS:
"""
Look for and REMOVE any lines that were added to compensate for batch effects:
- Manual cluster adjustments
- Artificial parameter modifications
- Special handling for dataset mixing
"""

# === VALIDATION STEPS ===
def validate_integration():
    """
    Steps to validate the integration worked correctly.
    """
    
    validation_steps = """
    After making the changes, validate by:
    
    1. 📊 Check UMAP plots colored by:
       - NK subtypes (should be well-separated)
       - Dataset (should be well-mixed)
       - Donor (should be well-mixed)
       - CMV status (should be well-mixed)
    
    2. 🔍 Verify embeddings:
       - adata.obsm['X_pca'] should be Harmony-corrected
       - adata.obsm['X_umap'] should show good mixing
    
    3. ✅ Confirm analysis quality:
       - Differential expression should be cleaner
       - No artificial clustering by technical variables
       - NK subtype markers should be preserved
    
    4. 📈 Compare before/after:
       - Clustering should be more biologically meaningful
       - Less technical noise in gene expression patterns
    """
    
    print(validation_steps)

if __name__ == "__main__":
    print("=" * 80)
    print("HARMONY INTEGRATION GUIDE FOR NK_ANALYSIS_MAIN_REBUFFET.PY")
    print("=" * 80)
    print()
    print("Key changes to make:")
    print("1. Update data file path to Harmony-corrected version")
    print("2. Use Harmony embeddings (X_pca_harmony)")
    print("3. Revert UMAP to standard parameters")
    print("4. Remove batch effect workarounds")
    print()
    validate_integration()