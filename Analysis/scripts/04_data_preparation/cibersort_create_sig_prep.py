#!/usr/bin/env python3
"""
CIBERSORTx Reference Matrix Preparation
Creates a reference matrix file for CIBERSORTx's online "create signature" function.

The output file format:
- Rows: genes
- Columns: single cells (400 cells per phenotype)
- First row: cell phenotype labels
- Data: TPM expression values
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
from scipy import sparse

warnings.filterwarnings("ignore")


def sample_cells_per_phenotype(
    adata, phenotype_col="ident", n_cells_per_phenotype=400, random_seed=42
):
    """
    Sample a specified number of cells per phenotype from the dataset.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    phenotype_col : str
        Column name containing phenotype annotations
    n_cells_per_phenotype : int
        Number of cells to sample per phenotype
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    AnnData : Subsampled data with equal cells per phenotype
    """
    np.random.seed(random_seed)

    # Get unique phenotypes
    phenotypes = adata.obs[phenotype_col].unique()
    sampled_indices = []

    print(f"📊 Sampling {n_cells_per_phenotype} cells per phenotype:")

    for phenotype in phenotypes:
        # Get cells for this phenotype
        phenotype_mask = adata.obs[phenotype_col] == phenotype
        phenotype_indices = np.where(phenotype_mask)[0]

        n_available = len(phenotype_indices)
        print(f"  {phenotype}: {n_available:,} available cells", end="")

        if n_available >= n_cells_per_phenotype:
            # Sample exactly n_cells_per_phenotype cells
            sampled = np.random.choice(
                phenotype_indices, size=n_cells_per_phenotype, replace=False
            )
            sampled_indices.extend(sampled)
            print(f" → sampled {n_cells_per_phenotype}")
        else:
            # Use all available cells if less than required
            sampled_indices.extend(phenotype_indices)
            print(
                f" → using all {n_available} cells (insufficient for {n_cells_per_phenotype})"
            )

    # Return subsampled data
    return adata[sampled_indices].copy()


def create_cibersortx_reference_matrix(adata, phenotype_col="ident", output_path=None):
    """
    Create a reference matrix file for CIBERSORTx in the correct format.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object with sampled cells
    phenotype_col : str
        Column name containing phenotype annotations
    output_path : str
        Path to save the reference matrix file

    Returns:
    --------
    pd.DataFrame : Reference matrix with genes as rows, cells as columns
    """
    print("\n🔬 Creating CIBERSORTx reference matrix...")

    # Get expression data and convert to DataFrame
    # Genes as rows, cells as columns
    if sparse.issparse(adata.X):
        X_data = adata.X.T.toarray()
    else:
        X_data = np.asarray(adata.X).T

    expr_matrix = pd.DataFrame(
        X_data,  # Genes as rows, cells as columns
        index=adata.var_names,  # Gene names as row indices
        columns=adata.obs_names,  # Cell barcodes as column names
    )

    # Get phenotype labels for each cell
    phenotype_labels = adata.obs[phenotype_col].values

    # Replace column names with phenotype labels
    expr_matrix.columns = phenotype_labels

    print(
        f"✅ Reference matrix created: {expr_matrix.shape[0]} genes × {expr_matrix.shape[1]} cells"
    )

    # Show phenotype distribution
    phenotype_counts = pd.Series(phenotype_labels).value_counts()
    print(f"\n📋 Final phenotype distribution:")
    for phenotype, count in phenotype_counts.items():
        print(f"  {phenotype}: {count:,} cells")

    # Add gene names as first column (required for CIBERSORTx)
    expr_matrix.index.name = "GeneSymbol"

    if output_path:
        # Save to file
        expr_matrix.to_csv(output_path, sep="\t")
        print(f"✅ Reference matrix saved to: {output_path}")

        # Create summary file
        summary_path = output_path.replace(".txt", "_summary.txt").replace(
            ".tsv", "_summary.txt"
        )
        with open(summary_path, "w") as f:
            f.write("CIBERSORTx REFERENCE MATRIX SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Creation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Matrix dimensions: {expr_matrix.shape[0]} genes × {expr_matrix.shape[1]} cells\n"
            )
            f.write(f"Data type: TPM expression values\n\n")

            f.write("PHENOTYPE DISTRIBUTION:\n")
            for phenotype, count in phenotype_counts.items():
                f.write(f"  {phenotype}: {count:,} cells\n")

            f.write(f"\nEXPRESSION STATISTICS:\n")
            f.write(f"  Mean expression: {expr_matrix.values.mean():.3f}\n")
            f.write(
                f"  Expression range: {expr_matrix.values.min():.3f} to {expr_matrix.values.max():.3f}\n"
            )
            f.write(f"  Non-zero entries: {(expr_matrix != 0).sum().sum():,}\n")

            f.write(f"\nCIBERSORTx UPLOAD INSTRUCTIONS:\n")
            f.write(f"1. Go to CIBERSORTx online platform\n")
            f.write(f"2. Select 'Create Signature Matrix'\n")
            f.write(f"3. Upload this reference matrix file\n")
            f.write(f"4. Set 'Single Cell Input Options':\n")
            f.write(f"   - Replicates: 5 (default) - all phenotypes have ≥400 cells\n")
            f.write(f"   - Sampling: 100 (default)\n")
            f.write(f"   - kappa: 999 (default)\n")
            f.write(f"5. Run signature matrix creation\n")

        print(f"✅ Summary saved to: {summary_path}")

    return expr_matrix


def main():
    print("=" * 80)
    print("CIBERSORTx REFERENCE MATRIX PREPARATION")
    print("=" * 80)

    # Configuration
    TPM_DATA_PATH = (
        "../../data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
    )
    OUTPUT_DIR = "../../outputs/signature_matrices/CIBERSORTx_Input_Files"
    N_CELLS_PER_PHENOTYPE = 500
    RANDOM_SEED = 42

    # NK subtypes to include
    NK_SUBTYPES = ["NK1A", "NK1B", "NK1C", "NKint", "NK2", "NK3"]

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📁 TPM data path: {TPM_DATA_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎯 Target phenotypes: {NK_SUBTYPES}")
    print(f"🔢 Cells per phenotype: {N_CELLS_PER_PHENOTYPE}")
    print(f"🎲 Random seed: {RANDOM_SEED}")

    # Step 1: Load data
    print("\n" + "=" * 60)
    print("STEP 1: LOADING TPM DATA")
    print("=" * 60)

    if not os.path.exists(TPM_DATA_PATH):
        raise FileNotFoundError(f"TPM data file not found: {TPM_DATA_PATH}")

    adata = sc.read_h5ad(TPM_DATA_PATH)
    print(f"✅ Data loaded: {adata.shape}")

    # Handle different data types safely
    if sparse.issparse(adata.X):
        X_array = adata.X.toarray()
    else:
        X_array = np.asarray(adata.X)
    print(f"📊 Data type: {type(adata.X).__name__}")
    print(f"📈 Data range: {X_array.min():.3f} to {X_array.max():.3f}")

    # Validate phenotype annotations
    if "ident" not in adata.obs.columns:
        raise ValueError("'ident' column not found in metadata")

    # Check available phenotypes
    available_phenotypes = adata.obs["ident"].unique()
    print(f"\n📋 Available phenotypes: {available_phenotypes}")

    # Show phenotype distribution
    phenotype_counts = adata.obs["ident"].value_counts()
    print("\n📊 Original phenotype distribution:")
    for phenotype in NK_SUBTYPES:
        if phenotype in phenotype_counts:
            count = phenotype_counts[phenotype]
            status = "✅" if count >= N_CELLS_PER_PHENOTYPE else "⚠️"
            print(f"  {status} {phenotype}: {count:,} cells")
        else:
            print(f"  ❌ {phenotype}: 0 cells (missing)")

    # Step 2: Filter for NK subtypes
    print("\n" + "=" * 60)
    print("STEP 2: FILTERING FOR NK SUBTYPES")
    print("=" * 60)

    # Filter for NK subtypes only
    nk_mask = adata.obs["ident"].isin(NK_SUBTYPES)
    adata_nk = adata[nk_mask].copy()
    print(f"✅ Filtered to NK cells: {adata_nk.shape}")

    # Verify TPM data quality
    print(f"\n🔍 TPM data validation:")
    tpm_sums = adata_nk.X.sum(axis=1)
    print(f"  - TPM sums per cell: {tpm_sums.mean():.0f} ± {tpm_sums.std():.0f}")
    print(f"  - Expected TPM sum: ~1,000,000")
    print(f"  - Data range: {adata_nk.X.min():.3f} to {adata_nk.X.max():.3f}")

    # Step 3: Sample cells per phenotype
    print("\n" + "=" * 60)
    print("STEP 3: SAMPLING CELLS PER PHENOTYPE")
    print("=" * 60)

    adata_sampled = sample_cells_per_phenotype(
        adata_nk,
        phenotype_col="ident",
        n_cells_per_phenotype=N_CELLS_PER_PHENOTYPE,
        random_seed=RANDOM_SEED,
    )

    print(f"\n✅ Sampling complete: {adata_sampled.shape}")

    # Step 4: Create reference matrix
    print("\n" + "=" * 60)
    print("STEP 4: CREATING REFERENCE MATRIX")
    print("=" * 60)

    output_filename = (
        f"NK_reference_matrix_{N_CELLS_PER_PHENOTYPE}cells_per_phenotype.txt"
    )
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    ref_matrix = create_cibersortx_reference_matrix(
        adata_sampled, phenotype_col="ident", output_path=output_path
    )

    # Step 5: Final validation
    print("\n" + "=" * 60)
    print("STEP 5: FINAL VALIDATION")
    print("=" * 60)

    print(f"📊 Reference matrix validation:")
    print(f"  - Dimensions: {ref_matrix.shape[0]} genes × {ref_matrix.shape[1]} cells")
    print(
        f"  - Expression range: {ref_matrix.values.min():.3f} to {ref_matrix.values.max():.3f}"
    )
    print(f"  - Mean expression: {ref_matrix.values.mean():.3f}")
    print(
        f"  - Zero values: {(ref_matrix == 0).sum().sum():,} ({(ref_matrix == 0).sum().sum() / ref_matrix.size * 100:.1f}%)"
    )

    # Check minimum cells per phenotype
    final_phenotype_counts = pd.Series(ref_matrix.columns).value_counts()
    min_cells = final_phenotype_counts.min()
    print(f"\n✅ Minimum cells per phenotype: {min_cells}")
    if min_cells >= 5:  # CIBERSORTx default minimum
        print(f"✅ All phenotypes meet CIBERSORTx minimum requirement (≥5 cells)")
    else:
        print(f"⚠️ Some phenotypes below CIBERSORTx minimum requirement (≥5 cells)")

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 REFERENCE MATRIX PREPARATION COMPLETE!")
    print("=" * 80)

    print(f"📊 Final Results:")
    print(
        f"  - Reference matrix: {ref_matrix.shape[0]} genes × {ref_matrix.shape[1]} cells"
    )
    print(f"  - Output file: {output_path}")
    print(f"  - Summary file: {output_path.replace('.txt', '_summary.txt')}")

    print(f"\n📋 Phenotype Summary:")
    for phenotype, count in final_phenotype_counts.items():
        print(f"  {phenotype}: {count:,} cells")

    print(f"\n🚀 Next Steps:")
    print(f"  1. Go to CIBERSORTx online platform (https://cibersortx.stanford.edu/)")
    print(f"  2. Select 'Create Signature Matrix'")
    print(f"  3. Upload: {output_filename}")
    print(f"  4. Use default Single Cell Input Options:")
    print(f"     - Replicates: 5 (all phenotypes have ≥{min_cells} cells)")
    print(f"     - Sampling: 100")
    print(f"     - kappa: 999")
    print(f"  5. Run signature matrix creation")
    print(f"  6. Download the generated signature matrix for deconvolution")

    print(f"\n✅ Your reference matrix is ready for CIBERSORTx signature creation!")

    return ref_matrix, output_path


if __name__ == "__main__":
    ref_matrix, output_path = main()
