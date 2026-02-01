#!/usr/bin/env python3
"""
Example: How to Use TCGA Gene Survival Analysis

This script demonstrates different ways to configure and run gene-based survival analysis.
Simply modify the gene lists below and run this script to perform your analysis.

Usage:
    python example_gene_analysis.py
"""

import os
import sys

# Add the current directory to path so we can import the main analysis script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main analysis functions
from TCGA_Gene_Survival_Analysis import (
    load_and_preprocess_tcga_data,
    add_gene_expression_to_adata,
    prepare_survival_data,
    perform_gene_survival_analysis,
    generate_analysis_summary,
    DEFAULT_THRESHOLDS
)

# ==============================================================================
# === EASY CONFIGURATION - MODIFY THESE SECTIONS ===
# ==============================================================================

# Cancer type and data paths
CANCER_TYPE = "BRCA"  # Change to your cancer type
BASE_DATA_DIR = r"C:\Users\met-a\Documents\Analysis\TCGAdata"
OUTPUT_DIR = r"C:\Users\met-a\Documents\Analysis\TCGAdata\Analysis_Python_Output"

# ==============================================================================
# === EXAMPLE 1: TUMOR SUPPRESSOR GENES ===
# ==============================================================================

TUMOR_SUPPRESSOR_GENES = [
    "TP53",     # Guardian of the genome
    "RB1",      # Retinoblastoma protein
    "PTEN",     # Phosphatase and tensin homolog
    "APC",      # Adenomatous polyposis coli
    "CDKN2A",   # Cyclin-dependent kinase inhibitor 2A (p16)
    "VHL",      # von Hippel-Lindau tumor suppressor
    "NF1",      # Neurofibromin 1
    "TUSC2",    # Tumor suppressor candidate 2
]

# ==============================================================================
# === EXAMPLE 2: ONCOGENES ===
# ==============================================================================

ONCOGENES = [
    "MYC",      # MYC proto-oncogene
    "KRAS",     # KRAS proto-oncogene
    "PIK3CA",   # Phosphatidylinositol-4,5-bisphosphate 3-kinase
    "EGFR",     # Epidermal growth factor receptor
    "ERBB2",    # erb-b2 receptor tyrosine kinase 2 (HER2)
    "BRAF",     # B-Raf proto-oncogene
    "AKT1",     # AKT serine/threonine kinase 1
    "CCND1",    # Cyclin D1
]

# ==============================================================================
# === EXAMPLE 3: DNA REPAIR GENES ===
# ==============================================================================

DNA_REPAIR_GENES = [
    "BRCA1",    # BRCA1 DNA repair associated
    "BRCA2",    # BRCA2 DNA repair associated
    "ATM",      # ATM serine/threonine kinase
    "CHEK2",    # Checkpoint kinase 2
    "PALB2",    # Partner and localizer of BRCA2
    "RAD51",    # RAD51 recombinase
    "MLH1",     # mutL homolog 1
    "MSH2",     # mutS homolog 2
]

# ==============================================================================
# === EXAMPLE 4: IMMUNE CHECKPOINT GENES ===
# ==============================================================================

IMMUNE_CHECKPOINT_GENES = [
    "PDCD1",    # Programmed cell death 1 (PD-1)
    "CD274",    # CD274 molecule (PD-L1)
    "CTLA4",    # Cytotoxic T-lymphocyte associated protein 4
    "LAG3",     # Lymphocyte activating 3
    "HAVCR2",   # Hepatitis A virus cellular receptor 2 (TIM-3)
    "TIGIT",    # T cell immunoreceptor with Ig and ITIM domains
    "IDO1",     # Indoleamine 2,3-dioxygenase 1
]

# ==============================================================================
# === EXAMPLE 5: CUSTOM PATHWAY GENES ===
# ==============================================================================

# Add your own genes of interest here
CUSTOM_GENES = [
    "TUSC2",    # Your gene of interest
    "TP53",     # Add any genes you want to analyze
    # "YOUR_GENE_HERE",  # Uncomment and add your genes
]


def run_gene_analysis(gene_list, analysis_name, cancer_type, base_data_dir, output_dir):
    """
    Run gene survival analysis for a specific gene list.
    
    Parameters:
    -----------
    gene_list : list
        List of gene symbols to analyze
    analysis_name : str
        Name for this analysis (used in output directory)
    cancer_type : str
        Cancer type abbreviation
    base_data_dir : str
        Base TCGA data directory
    output_dir : str
        Output directory
    """
    print(f"\n{'='*80}")
    print(f"🧬 RUNNING GENE SURVIVAL ANALYSIS: {analysis_name}")
    print(f"🎯 Genes: {gene_list}")
    print(f"📊 Cancer Type: {cancer_type}")
    print(f"{'='*80}")
    
    # Load TCGA data (this step is identical across all analyses)
    tumor_adata, master_metadata = load_and_preprocess_tcga_data(
        cancer_type=cancer_type,
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        thresholds=DEFAULT_THRESHOLDS,
    )
    
    if tumor_adata is None:
        print(f"❌ FAILED: Could not load TCGA data for {cancer_type}")
        return
    
    print(f"\n✅ SUCCESS: Data loaded - {tumor_adata.n_obs} samples x {tumor_adata.n_vars} genes")
    
    # Create analysis-specific output directory
    analysis_output_dir = os.path.join(output_dir, f"{cancer_type}_{analysis_name}_Gene_Survival")
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    # Add gene expressions to AnnData
    tumor_adata, available_genes = add_gene_expression_to_adata(tumor_adata, gene_list)
    
    if not available_genes:
        print(f"❌ ERROR: None of the specified genes found in dataset")
        print(f"   Available genes (first 20): {list(tumor_adata.var_names[:20])}")
        return
    
    print(f"📊 Found {len(available_genes)}/{len(gene_list)} genes in dataset")
    
    # Prepare survival data
    survival_df = prepare_survival_data(tumor_adata)
    
    # Perform survival analysis
    cox_results, logrank_results = perform_gene_survival_analysis(
        survival_df, available_genes, analysis_output_dir, cancer_type
    )
    
    # Generate and save summary
    summary = generate_analysis_summary(cox_results, logrank_results, cancer_type, available_genes)
    summary["analysis_name"] = analysis_name
    summary["gene_list_input"] = gene_list
    
    import json
    summary_file = os.path.join(analysis_output_dir, f"{cancer_type}_{analysis_name}_Summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Analysis complete! Results saved to: {analysis_output_dir}")
    print(f"📋 Summary: {summary_file}")
    
    return cox_results, logrank_results


def main():
    """
    Main function - run different gene analyses.
    
    CUSTOMIZE THIS SECTION: Comment/uncomment the analyses you want to run
    """
    
    print("🚀 Starting TCGA Gene Survival Analysis Examples")
    print(f"   Cancer Type: {CANCER_TYPE}")
    print(f"   Data Directory: {BASE_DATA_DIR}")
    print(f"   Output Directory: {OUTPUT_DIR}")
    
    # ==============================================================================
    # === RUN ANALYSES - MODIFY THIS SECTION ===
    # ==============================================================================
    
    # EXAMPLE 1: Analyze tumor suppressor genes
    print("\n" + "="*50)
    print("🔬 EXAMPLE 1: TUMOR SUPPRESSOR GENES")
    run_gene_analysis(
        gene_list=TUMOR_SUPPRESSOR_GENES,
        analysis_name="TumorSuppressors",
        cancer_type=CANCER_TYPE,
        base_data_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # EXAMPLE 2: Analyze oncogenes (UNCOMMENT TO RUN)
    # print("\n" + "="*50)
    # print("🔬 EXAMPLE 2: ONCOGENES")
    # run_gene_analysis(
    #     gene_list=ONCOGENES,
    #     analysis_name="Oncogenes",
    #     cancer_type=CANCER_TYPE,
    #     base_data_dir=BASE_DATA_DIR,
    #     output_dir=OUTPUT_DIR
    # )
    
    # EXAMPLE 3: Analyze DNA repair genes (UNCOMMENT TO RUN)
    # print("\n" + "="*50)
    # print("🔬 EXAMPLE 3: DNA REPAIR GENES")
    # run_gene_analysis(
    #     gene_list=DNA_REPAIR_GENES,
    #     analysis_name="DNARepair",
    #     cancer_type=CANCER_TYPE,
    #     base_data_dir=BASE_DATA_DIR,
    #     output_dir=OUTPUT_DIR
    # )
    
    # EXAMPLE 4: Analyze immune checkpoint genes (UNCOMMENT TO RUN)
    # print("\n" + "="*50)
    # print("🔬 EXAMPLE 4: IMMUNE CHECKPOINT GENES")
    # run_gene_analysis(
    #     gene_list=IMMUNE_CHECKPOINT_GENES,
    #     analysis_name="ImmuneCheckpoint",
    #     cancer_type=CANCER_TYPE,
    #     base_data_dir=BASE_DATA_DIR,
    #     output_dir=OUTPUT_DIR
    # )
    
    # EXAMPLE 5: Analyze your custom genes (MODIFY CUSTOM_GENES LIST ABOVE)
    # print("\n" + "="*50)
    # print("🔬 EXAMPLE 5: CUSTOM GENES")
    # run_gene_analysis(
    #     gene_list=CUSTOM_GENES,
    #     analysis_name="Custom",
    #     cancer_type=CANCER_TYPE,
    #     base_data_dir=BASE_DATA_DIR,
    #     output_dir=OUTPUT_DIR
    # )
    
    print(f"\n🎉 All analyses complete!")
    print(f"📁 Check results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 