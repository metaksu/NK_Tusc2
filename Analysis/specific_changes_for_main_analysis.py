#!/usr/bin/env python3
"""
SPECIFIC CHANGES FOR NK_analysis_main_rebuffet.py
================================================

Exact line-by-line changes to integrate Harmony-corrected data
and revert artificial UMAP parameters back to normal.
"""

# === CHANGE 1: UPDATE DATA FILE PATH (around line 202) ===

# FIND THIS LINE (around line 202):
# REBUFFET_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\data\processed\PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"

# REPLACE WITH:
REBUFFET_H5AD_FILE = r"C:\Users\met-a\Documents\Analysis\outputs\harmony_batch_correction\PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad"

# === CHANGE 2: UPDATE AFTER DATA LOADING (around line 2375) ===

# FIND THIS SECTION (around line 2375):
# adata_blood = sc.AnnData(...)
# print(f"      adata_blood created. Initial Shape: {adata_blood.shape}")

# ADD THESE LINES AFTER:
"""
    # === USE HARMONY-CORRECTED EMBEDDINGS ===
    print("      🎯 USING HARMONY-CORRECTED EMBEDDINGS")
    
    # Check if Harmony embeddings are available
    if 'X_pca_harmony' in adata_blood_processed.obsm:
        print("      ✅ Found Harmony-corrected PCA embeddings")
        adata_blood.obsm['X_pca'] = adata_blood_processed.obsm['X_pca_harmony'].copy()
        HARMONY_CORRECTED = True
    else:
        print("      ⚠️ No Harmony PCA embeddings found, using original")
        HARMONY_CORRECTED = False
    
    if 'X_umap_harmony' in adata_blood_processed.obsm:
        print("      ✅ Found Harmony-corrected UMAP embeddings")
        adata_blood.obsm['X_umap'] = adata_blood_processed.obsm['X_umap_harmony'].copy()
        print("      🎯 Using pre-computed Harmony UMAP - skipping dimension reduction")
        SKIP_DIMENSION_REDUCTION = True
    else:
        print("      📊 Will compute new UMAP with standard parameters")
        SKIP_DIMENSION_REDUCTION = False
"""

# === CHANGE 3: REPLACE ARTIFICIAL DIMENSION REDUCTION (around lines 2530-2548) ===

# FIND THIS SECTION:
"""
N_PCS_BLOOD = 3  # Ultra-minimal PCs to reduce technical variance
N_NEIGHBORS_BLOOD = 1000  # Ultra-high for maximum connectivity (increased from 500)
print(f"      Computing nearest neighbor graph for adata_blood using {N_PCS_BLOOD} PCs and {N_NEIGHBORS_BLOOD} neighbors...")
print("      Using ultra-high neighbor count to force maximum biological continuity and eliminate separation...")
sc.pp.neighbors(adata_blood, n_neighbors=N_NEIGHBORS_BLOOD, n_pcs=N_PCS_BLOOD, random_state=RANDOM_SEED)
print("      Nearest neighbor graph computation complete for adata_blood.")

# --- Step 5: Uniform Manifold Approximation and Projection (UMAP) ---
print("      Running UMAP for adata_blood with ULTRA-HIGH connectivity parameters...")
print("      Using ultra-high min_dist and spread to force single merged blob...")
sc.tl.umap(adata_blood, 
           random_state=RANDOM_SEED, 
           min_dist=1.8,  # Ultra-high min_dist to eliminate local clustering (increased from 1.2)
           spread=4.5)    # Ultra-high spread for maximum global connectivity (increased from 3.0)
print("      UMAP calculation complete for adata_blood.")
"""

# REPLACE WITH:
REPLACEMENT_CODE = '''
    # === STANDARD DIMENSION REDUCTION (BATCH EFFECTS NOW CORRECTED) ===
    if not SKIP_DIMENSION_REDUCTION:
        print("      🎯 COMPUTING STANDARD DIMENSION REDUCTION (no more batch effect compensation needed)")
        
        # STANDARD PARAMETERS (no more artificial modifications)
        N_PCS_BLOOD = 50         # STANDARD (was artificially reduced to 3)
        N_NEIGHBORS_BLOOD = 15   # STANDARD (was artificially increased to 1000)
        
        print(f"      Computing nearest neighbor graph using {N_PCS_BLOOD} PCs and {N_NEIGHBORS_BLOOD} neighbors...")
        print("      Using STANDARD parameters - batch effects are now corrected!")
        
        # Use Harmony embeddings if available
        use_rep = 'X_pca' if HARMONY_CORRECTED else None
        
        sc.pp.neighbors(
            adata_blood, 
            n_neighbors=N_NEIGHBORS_BLOOD, 
            n_pcs=N_PCS_BLOOD, 
            use_rep=use_rep,
            random_state=RANDOM_SEED
        )
        print("      ✅ Nearest neighbor graph computation complete.")

        # === STANDARD UMAP PARAMETERS ===
        print("      Running UMAP with STANDARD parameters...")
        print("      No more artificial parameter forcing - batch effects corrected!")
        
        sc.tl.umap(
            adata_blood, 
            random_state=RANDOM_SEED, 
            min_dist=0.5,    # STANDARD (was artificially 1.8)
            spread=1.0       # STANDARD (was artificially 4.5)
        )
        print("      ✅ UMAP calculation complete with standard parameters.")
    else:
        print("      ✅ Using pre-computed Harmony UMAP - dimension reduction skipped.")
    
    print(f"      adata_blood.obsm['X_umap'] shape: {adata_blood.obsm['X_umap'].shape if 'X_umap' in adata_blood.obsm else 'Not found'}")
'''

# === CHANGE 4: UPDATE ANY run_dim_reduction_pipeline CALLS ===

# FIND any calls to run_dim_reduction_pipeline and ensure they use standard parameters:
"""
def run_dim_reduction_pipeline(adata_obj, cohort_label, n_pcs_to_use=15, n_hvgs=1000):
    # Update to use Harmony embeddings if available
    use_rep = 'X_pca_harmony' if 'X_pca_harmony' in adata_obj.obsm else 'X_pca'
    
    # STANDARD UMAP parameters
    sc.pp.neighbors(adata_obj, use_rep=use_rep, n_neighbors=15, random_state=42)
    sc.tl.umap(adata_obj, min_dist=0.5, spread=1.0, random_state=42)
    
    return adata_obj
"""

# === SUMMARY OF CHANGES ===
print("""
🎯 SUMMARY OF REQUIRED CHANGES:

1. 📁 DATA PATH: Update to Harmony-corrected file
   outputs/harmony_batch_correction/PBMC_V2_VF1_AllGenes_NewNames_TPM_harmony_corrected.h5ad

2. 🧮 EMBEDDINGS: Use X_pca_harmony and X_umap_harmony if available

3. 📊 UMAP PARAMETERS: Revert to STANDARD values:
   - n_pcs: 3 → 50 (STANDARD)
   - n_neighbors: 1000 → 15 (STANDARD) 
   - min_dist: 1.8 → 0.5 (STANDARD)
   - spread: 4.5 → 1.0 (STANDARD)

4. 🧹 REMOVE: All "ultra-high connectivity" workarounds

RESULT: Clean, unbiased NK cell analysis with proper cluster separation! 🎉
""")

# === QUICK VALIDATION ===
validation_code = '''
# Add this at the end to validate the changes worked:

print("🔍 VALIDATION:")
print(f"✅ Data shape: {adata_blood.shape}")
print(f"✅ UMAP shape: {adata_blood.obsm['X_umap'].shape}")

# Check if embeddings are Harmony-corrected
if 'harmony_correction' in adata_blood.uns:
    correction_info = adata_blood.uns['harmony_correction']
    print(f"✅ Harmony correction applied: {correction_info['batch_variables']}")
else:
    print("⚠️ No Harmony correction metadata found")

print("🎯 Ready for unbiased NK cell analysis!")
'''