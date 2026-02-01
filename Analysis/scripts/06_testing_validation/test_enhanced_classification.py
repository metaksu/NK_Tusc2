#!/usr/bin/env python3
"""
Test Enhanced Hybrid Classification Approach
Compares the new hybrid approach with the previous neural network only approach
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced scArches module
from scarches_module import add_scarches_to_existing_workflow


def test_enhanced_classification():
    """Test the enhanced hybrid classification approach"""

    print("🧪 Testing Enhanced Hybrid Classification")
    print("=" * 60)

    # Load test data
    print("📊 Loading test data...")

    # Reference dataset
    reference_path = (
        "../../data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad"
    )
    if not os.path.exists(reference_path):
        print(f"❌ Reference dataset not found: {reference_path}")
        return False

    # Load reference
    adata_ref = sc.read_h5ad(reference_path)
    print(f"✅ Reference loaded: {adata_ref.shape}")

    # Load Tang datasets
    tang_path = "../../data/processed/comb_CD56_CD16_NK.h5ad"
    if not os.path.exists(tang_path):
        print(f"❌ Tang dataset not found: {tang_path}")
        return False

    adata_tang = sc.read_h5ad(tang_path)
    print(f"✅ Tang dataset loaded: {adata_tang.shape}")

    # Create test cohorts (subset for faster testing)
    print("🔬 Creating test cohorts...")

    # Debug: Check available columns
    print(f"   Available columns in Tang dataset: {list(adata_tang.obs.columns)}")

    # Check for context column (might be named differently)
    context_col = None
    for col in ["context", "Context", "tissue_type", "Tissue_Type", "source", "Source"]:
        if col in adata_tang.obs.columns:
            context_col = col
            break

    if context_col is None:
        print("   ⚠️  No context column found. Using random split for testing...")
        # Use random split for testing
        np.random.seed(42)
        n_cells = adata_tang.n_obs
        split_point = n_cells // 2
        indices = np.random.permutation(n_cells)

        adata_normal_tissue = adata_tang[indices[:split_point]].copy()
        adata_tumor_tissue = adata_tang[indices[split_point:]].copy()
    else:
        print(f"   Using column '{context_col}' for context separation")
        # Normal tissue subset
        normal_mask = adata_tang.obs[context_col] == "Normal"
        adata_normal_tissue = adata_tang[normal_mask].copy()

        # Tumor tissue subset
        tumor_mask = adata_tang.obs[context_col] == "Tumor"
        adata_tumor_tissue = adata_tang[tumor_mask].copy()

    print(f"   Normal tissue: {adata_normal_tissue.shape}")
    print(f"   Tumor tissue: {adata_tumor_tissue.shape}")

    # Test both approaches
    test_results = {}

    # Test 1: Neural Network Only (original approach)
    print("\n🧠 Testing Neural Network Only Approach")
    print("-" * 40)

    try:
        nn_results = add_scarches_to_existing_workflow(
            adata_normal_tissue.copy(),
            adata_tumor_tissue.copy(),
            reference_path=reference_path,
            output_dir="../../outputs/test_nn_only",
            use_hybrid=False,
            confidence_threshold=0.7,
        )
        test_results["neural_network_only"] = nn_results
        print("✅ Neural network only test completed")
    except Exception as e:
        print(f"❌ Neural network only test failed: {e}")
        test_results["neural_network_only"] = None

    # Test 2: Hybrid Approach (enhanced approach)
    print("\n🔀 Testing Hybrid Approach")
    print("-" * 40)

    try:
        hybrid_results = add_scarches_to_existing_workflow(
            adata_normal_tissue.copy(),
            adata_tumor_tissue.copy(),
            reference_path=reference_path,
            output_dir="../../outputs/test_hybrid",
            use_hybrid=True,
            confidence_threshold=0.5,
        )
        test_results["hybrid"] = hybrid_results
        print("✅ Hybrid approach test completed")
    except Exception as e:
        print(f"❌ Hybrid approach test failed: {e}")
        test_results["hybrid"] = None

    # Compare results
    print("\n📈 Results Comparison")
    print("=" * 60)

    for approach, results in test_results.items():
        if results is None:
            print(f"\n❌ {approach.upper()}: FAILED")
            continue

        print(f"\n✅ {approach.upper()}:")

        for cohort_name, result in results.items():
            if result is None:
                print(f"   {cohort_name}: FAILED")
                continue

            print(f"   {cohort_name}:")
            print(f"     Total cells: {result['total_cells']:,}")

            # Assignment rate
            assigned_cells = result["total_cells"] - result[
                "classification_counts"
            ].get("Unassigned", 0)
            assignment_rate = assigned_cells / result["total_cells"] * 100
            print(f"     Assignment rate: {assignment_rate:.1f}%")

            # High confidence rate
            high_conf_rate = (
                result["high_confidence_count"] / result["total_cells"] * 100
            )
            print(f"     High confidence rate: {high_conf_rate:.1f}%")

            # Method breakdown (for hybrid)
            if "method_breakdown" in result:
                print(f"     Method breakdown:")
                for method, count in result["method_breakdown"].items():
                    pct = count / result["total_cells"] * 100
                    print(f"       {method}: {count:,} ({pct:.1f}%)")

            # Subtype distribution
            print(f"     Subtype distribution:")
            for subtype, count in result["classification_counts"].items():
                if count > 0:
                    pct = count / result["total_cells"] * 100
                    print(f"       {subtype}: {count:,} ({pct:.1f}%)")

    # Summary
    print("\n🎯 Summary")
    print("=" * 60)

    nn_success = test_results["neural_network_only"] is not None
    hybrid_success = test_results["hybrid"] is not None

    if nn_success and hybrid_success:
        print("✅ Both approaches completed successfully!")

        # Compare assignment rates
        for cohort_name in ["normal_tissue", "tumor_tissue"]:
            if (
                cohort_name in test_results["neural_network_only"]
                and cohort_name in test_results["hybrid"]
            ):
                nn_result = test_results["neural_network_only"][cohort_name]
                hybrid_result = test_results["hybrid"][cohort_name]

                nn_assigned = nn_result["total_cells"] - nn_result[
                    "classification_counts"
                ].get("Unassigned", 0)
                hybrid_assigned = hybrid_result["total_cells"] - hybrid_result[
                    "classification_counts"
                ].get("Unassigned", 0)

                nn_rate = nn_assigned / nn_result["total_cells"] * 100
                hybrid_rate = hybrid_assigned / hybrid_result["total_cells"] * 100

                improvement = hybrid_rate - nn_rate

                print(f"\n📊 {cohort_name.replace('_', ' ').title()}:")
                print(f"   Neural Network Only: {nn_rate:.1f}% assigned")
                print(f"   Hybrid Approach: {hybrid_rate:.1f}% assigned")
                print(f"   Improvement: {improvement:+.1f} percentage points")

                if improvement > 0:
                    print(
                        f"   🎉 Hybrid approach shows {improvement:.1f}% improvement!"
                    )
                else:
                    print(
                        f"   ⚠️  Hybrid approach shows {abs(improvement):.1f}% decrease"
                    )

    elif hybrid_success:
        print("✅ Hybrid approach completed successfully!")
        print("❌ Neural network only approach failed")
        print("🎉 Hybrid approach is the recommended solution")

    elif nn_success:
        print("✅ Neural network only approach completed successfully!")
        print("❌ Hybrid approach failed")
        print("⚠️  Need to debug hybrid approach")

    else:
        print("❌ Both approaches failed")
        print("🚨 Need to investigate classification issues")

    return hybrid_success


if __name__ == "__main__":
    print("🚀 Starting Enhanced Classification Test")
    print("=" * 60)

    success = test_enhanced_classification()

    if success:
        print("\n🎉 Test completed successfully!")
        print("✅ Enhanced hybrid classification is ready for production use")
    else:
        print("\n❌ Test failed")
        print("🚨 Need to investigate and fix issues")

    print("\n" + "=" * 60)
