#!/usr/bin/env python3
"""
TUSC2 Analysis Using Tang et al. Original NK Subtypes
Focus on functional vs dysfunctional NK cells in tumor microenvironment
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set up scanpy
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor="white")


def load_tang_data():
    """Load the Tang dataset with original annotations"""
    print("🔬 Loading Tang et al. dataset...")

    tang_path = "../../data/processed/comb_CD56_CD16_NK.h5ad"
    if not os.path.exists(tang_path):
        print(f"❌ Tang dataset not found: {tang_path}")
        return None

    adata = sc.read_h5ad(tang_path)
    print(f"✅ Tang dataset loaded: {adata.shape}")

    # Check available annotations
    print(f"📋 Available annotations:")
    for col in ["celltype", "Majortype", "meta_tissue_in_paper"]:
        if col in adata.obs.columns:
            print(f"   {col}: {adata.obs[col].nunique()} unique values")
            if adata.obs[col].nunique() < 20:  # Show values if not too many
                print(f"      Values: {list(adata.obs[col].unique())}")

    # Check for TUSC2
    if "TUSC2" in adata.var_names:
        tusc2_expr = adata[:, "TUSC2"].X.toarray().flatten()
        print(
            f"✅ TUSC2 found - Expression range: {tusc2_expr.min():.3f} to {tusc2_expr.max():.3f}"
        )
        print(
            f"   TUSC2+ cells (>0): {(tusc2_expr > 0).sum():,} ({(tusc2_expr > 0).mean()*100:.1f}%)"
        )
    else:
        print("❌ TUSC2 not found in dataset")
        return None

    return adata


def define_functional_signatures():
    """Define key functional signatures for NK cell analysis"""

    signatures = {
        # Cytotoxicity (what we want to see in functional NK cells)
        "Cytotoxicity": ["GZMB", "PRF1", "GNLY", "GZMA", "GZMH"],
        # Dysfunction/Exhaustion (TaNK signature - what we don't want)
        "Dysfunction": ["DNAJB1", "HSPA1A", "BAG3", "NR4A1", "TIGIT", "LAG3"],
        # Inflammatory (can be good or bad depending on context)
        "Inflammatory": ["CCL3", "CCL4", "NFKBIA", "TNF", "IL1B"],
        # Adaptive/Memory (specialized functional state)
        "Adaptive": ["KLRC2", "CD57", "FCGR3A", "KLRG1"],
        # Proliferation (active response)
        "Proliferation": ["MKI67", "STMN1", "PCNA", "TOP2A"],
        # Stress Response (negative)
        "Stress": ["HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "BAG3"],
    }

    return signatures


def calculate_signature_scores(adata, signatures):
    """Calculate signature scores for each cell"""
    print("📊 Calculating functional signature scores...")

    for sig_name, genes in signatures.items():
        # Find available genes
        available_genes = [g for g in genes if g in adata.var_names]

        if len(available_genes) >= 2:  # Need at least 2 genes
            print(f"   {sig_name}: {len(available_genes)}/{len(genes)} genes available")

            # Calculate signature score
            sc.tl.score_genes(
                adata, available_genes, score_name=f"{sig_name}_Score", use_raw=False
            )
        else:
            print(
                f"   ⚠️  {sig_name}: Only {len(available_genes)} genes available, skipping"
            )

    return adata


def analyze_tusc2_by_subtype(adata):
    """Analyze TUSC2 expression patterns by NK subtype"""
    print("🔍 Analyzing TUSC2 patterns by NK subtype...")

    # Get TUSC2 expression
    tusc2_expr = adata[:, "TUSC2"].X.toarray().flatten()
    adata.obs["TUSC2_expression"] = tusc2_expr
    adata.obs["TUSC2_positive"] = tusc2_expr > 0

    # Use celltype as the primary annotation
    subtype_col = "celltype"
    if subtype_col not in adata.obs.columns:
        print(f"❌ Column '{subtype_col}' not found")
        return None

    # Calculate TUSC2+ frequency by subtype
    tusc2_by_subtype = (
        adata.obs.groupby(subtype_col)
        .agg(
            {
                "TUSC2_positive": ["count", "sum", "mean"],
                "TUSC2_expression": ["mean", "std"],
            }
        )
        .round(3)
    )

    # Flatten column names
    tusc2_by_subtype.columns = [
        "Total_Cells",
        "TUSC2_Positive_Count",
        "TUSC2_Positive_Fraction",
        "TUSC2_Mean_Expression",
        "TUSC2_Std_Expression",
    ]

    # Sort by TUSC2+ fraction
    tusc2_by_subtype = tusc2_by_subtype.sort_values(
        "TUSC2_Positive_Fraction", ascending=False
    )

    print("📈 TUSC2+ frequency by NK subtype:")
    print(tusc2_by_subtype)

    return tusc2_by_subtype


def compare_functional_signatures(adata):
    """Compare functional signatures between TUSC2+ and TUSC2- cells"""
    print("⚖️  Comparing functional signatures: TUSC2+ vs TUSC2-...")

    # Get signature score columns
    signature_cols = [col for col in adata.obs.columns if col.endswith("_Score")]

    if not signature_cols:
        print("❌ No signature scores found")
        return None

    results = []

    for sig_col in signature_cols:
        sig_name = sig_col.replace("_Score", "")

        # Get scores for TUSC2+ and TUSC2- cells
        tusc2_pos_scores = adata.obs[adata.obs["TUSC2_positive"]][sig_col]
        tusc2_neg_scores = adata.obs[~adata.obs["TUSC2_positive"]][sig_col]

        # Statistical test
        stat, pval = stats.mannwhitneyu(
            tusc2_pos_scores, tusc2_neg_scores, alternative="two-sided"
        )

        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt(
            (
                (len(tusc2_pos_scores) - 1) * tusc2_pos_scores.std() ** 2
                + (len(tusc2_neg_scores) - 1) * tusc2_neg_scores.std() ** 2
            )
            / (len(tusc2_pos_scores) + len(tusc2_neg_scores) - 2)
        )
        cohens_d = (tusc2_pos_scores.mean() - tusc2_neg_scores.mean()) / pooled_std

        results.append(
            {
                "Signature": sig_name,
                "TUSC2_Pos_Mean": tusc2_pos_scores.mean(),
                "TUSC2_Neg_Mean": tusc2_neg_scores.mean(),
                "Difference": tusc2_pos_scores.mean() - tusc2_neg_scores.mean(),
                "Cohens_D": cohens_d,
                "P_Value": pval,
                "Significant": pval < 0.05,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Cohens_D", ascending=False)

    print("📊 Functional signature comparison (TUSC2+ vs TUSC2-):")
    print(results_df.round(4))

    return results_df


def identify_key_subtypes(adata):
    """Identify the key functional vs dysfunctional subtypes"""
    print("🎯 Identifying key NK subtypes of interest...")

    # Look for Tang's key subtypes in the celltype column
    celltype_counts = adata.obs["celltype"].value_counts()
    print("📋 Available NK subtypes:")
    for subtype, count in celltype_counts.items():
        pct = count / len(adata) * 100
        print(f"   {subtype}: {count:,} cells ({pct:.1f}%)")

    # Try to identify key subtypes based on naming patterns
    key_subtypes = {
        "TaNK_like": [],  # Dysfunctional
        "Healthy_Cytotoxic": [],  # Functional
        "Adaptive": [],  # Specialized functional
        "Inflammatory": [],  # Context-dependent
    }

    for subtype in celltype_counts.index:
        subtype_lower = subtype.lower()

        # Look for TaNK-like (dysfunctional) patterns
        if any(
            marker in subtype_lower
            for marker in ["dnajb1", "stress", "exhausted", "dysfunction"]
        ):
            key_subtypes["TaNK_like"].append(subtype)

        # Look for healthy cytotoxic patterns
        elif any(
            marker in subtype_lower for marker in ["nr4a3", "cytotoxic", "healthy"]
        ):
            key_subtypes["Healthy_Cytotoxic"].append(subtype)

        # Look for adaptive patterns
        elif any(marker in subtype_lower for marker in ["klrc2", "adaptive", "memory"]):
            key_subtypes["Adaptive"].append(subtype)

        # Look for inflammatory patterns
        elif any(
            marker in subtype_lower for marker in ["nfkbia", "ccl3", "inflammatory"]
        ):
            key_subtypes["Inflammatory"].append(subtype)

    print("🔍 Categorized subtypes:")
    for category, subtypes in key_subtypes.items():
        if subtypes:
            print(f"   {category}: {subtypes}")

    return key_subtypes


def create_tusc2_subtype_plot(adata, output_dir="../../outputs"):
    """Create visualization of TUSC2 patterns by subtype"""
    print("📊 Creating TUSC2 subtype visualization...")

    os.makedirs(output_dir, exist_ok=True)

    # Calculate TUSC2+ frequency by subtype
    tusc2_freq = adata.obs.groupby("celltype")["TUSC2_positive"].agg(
        ["count", "sum", "mean"]
    )
    tusc2_freq.columns = ["Total_Cells", "TUSC2_Pos_Count", "TUSC2_Pos_Freq"]
    tusc2_freq = tusc2_freq.sort_values("TUSC2_Pos_Freq", ascending=True)

    # Create horizontal bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(tusc2_freq)), tusc2_freq["TUSC2_Pos_Freq"] * 100)

    # Color bars based on frequency (red = low, green = high)
    colors = plt.cm.RdYlGn(tusc2_freq["TUSC2_Pos_Freq"])
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.yticks(range(len(tusc2_freq)), tusc2_freq.index, fontsize=10)
    plt.xlabel("TUSC2+ Frequency (%)", fontsize=12)
    plt.title(
        "TUSC2+ Frequency by NK Subtype\n(Tang et al. Original Annotations)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(axis="x", alpha=0.3)

    # Add frequency labels on bars
    for i, (idx, row) in enumerate(tusc2_freq.iterrows()):
        plt.text(
            row["TUSC2_Pos_Freq"] * 100 + 0.5,
            i,
            f"{row['TUSC2_Pos_Freq']*100:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/TUSC2_frequency_by_NK_subtype.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return tusc2_freq


def main():
    """Main analysis function"""
    print("🚀 TUSC2 Analysis Using Tang et al. Original NK Subtypes")
    print("=" * 70)

    # Load data
    adata = load_tang_data()
    if adata is None:
        return

    # Define signatures
    signatures = define_functional_signatures()

    # Calculate signature scores
    adata = calculate_signature_scores(adata, signatures)

    # Analyze TUSC2 by subtype
    tusc2_by_subtype = analyze_tusc2_by_subtype(adata)

    # Compare functional signatures
    signature_comparison = compare_functional_signatures(adata)

    # Identify key subtypes
    key_subtypes = identify_key_subtypes(adata)

    # Create visualization
    tusc2_freq = create_tusc2_subtype_plot(adata)

    # Summary
    print("\n🎯 KEY FINDINGS:")
    print("=" * 50)

    if tusc2_by_subtype is not None:
        print("📈 Top 3 subtypes with highest TUSC2+ frequency:")
        top_3 = tusc2_by_subtype.head(3)
        for idx, row in top_3.iterrows():
            print(
                f"   {idx}: {row['TUSC2_Positive_Fraction']*100:.1f}% TUSC2+ ({row['TUSC2_Positive_Count']}/{row['Total_Cells']} cells)"
            )

    if signature_comparison is not None:
        print("\n📊 Functional signatures most enhanced in TUSC2+ cells:")
        enhanced = signature_comparison[signature_comparison["Difference"] > 0].head(3)
        for _, row in enhanced.iterrows():
            print(
                f"   {row['Signature']}: +{row['Difference']:.3f} (Cohen's D: {row['Cohens_D']:.2f})"
            )

        print("\n📉 Functional signatures most reduced in TUSC2+ cells:")
        reduced = signature_comparison[signature_comparison["Difference"] < 0].head(3)
        for _, row in reduced.iterrows():
            print(
                f"   {row['Signature']}: {row['Difference']:.3f} (Cohen's D: {row['Cohens_D']:.2f})"
            )

    print("\n✅ Analysis complete! Check outputs directory for visualizations.")

    return adata, tusc2_by_subtype, signature_comparison


if __name__ == "__main__":
    adata, tusc2_results, sig_results = main()
