#!/usr/bin/env python3
"""
Diagnostic script to identify sample ID mismatch between TCGA and CIBERSORTx data.
"""

import pandas as pd
import os
from pathlib import Path

def diagnose_sample_mismatch():
    """Compare sample IDs between TCGA and CIBERSORTx datasets."""
    
    print("🔍 Diagnosing Sample ID Mismatch Between TCGA and CIBERSORTx")
    print("=" * 70)
    
    # Load CIBERSORTx results
    cibersortx_file = "TCGAdata/CIBERSORTx_Job1_Results.csv"
    if not os.path.exists(cibersortx_file):
        print(f"❌ CIBERSORTx file not found: {cibersortx_file}")
        return
    
    print(f"\n📊 Loading CIBERSORTx Results: {cibersortx_file}")
    cibersortx_df = pd.read_csv(cibersortx_file, index_col="Mixture")
    cibersortx_samples = set(cibersortx_df.index)
    print(f"   CIBERSORTx samples: {len(cibersortx_samples)}")
    print(f"   First 5 CIBERSORTx samples: {list(cibersortx_samples)[:5]}")
    
    # Check TCGA RNA-seq directory
    rna_dir = "TCGAdata/rna"
    if not os.path.exists(rna_dir):
        print(f"❌ TCGA RNA directory not found: {rna_dir}")
        return
    
    print(f"\n📁 Scanning TCGA RNA-seq Directory: {rna_dir}")
    
    # Get all RNA-seq files
    rna_files = [f for f in os.listdir(rna_dir) if f.endswith('.tsv')]
    tcga_sample_ids = set()
    
    for filename in rna_files[:10]:  # Sample first 10 files for quick check
        sample_id = filename.split(".")[0]  # Extract UUID
        tcga_sample_ids.add(sample_id)
    
    print(f"   TCGA RNA files found: {len(rna_files)}")
    print(f"   Sample of TCGA sample IDs: {list(tcga_sample_ids)[:5]}")
    
    # Find overlaps
    common_samples = cibersortx_samples & tcga_sample_ids
    print(f"\n🔍 Sample Overlap Analysis:")
    print(f"   CIBERSORTx samples: {len(cibersortx_samples)}")
    print(f"   TCGA samples (sampled): {len(tcga_sample_ids)}")
    print(f"   Common samples: {len(common_samples)}")
    
    if common_samples:
        print(f"   ✅ Found overlapping samples: {list(common_samples)[:5]}")
    else:
        print("   ❌ NO OVERLAPPING SAMPLES FOUND!")
        
        print(f"\n📋 Sample ID Comparison:")
        print("   CIBERSORTx sample examples:")
        for i, sample in enumerate(list(cibersortx_samples)[:5]):
            print(f"     {i+1}: {sample}")
        
        print("   TCGA sample examples:")
        for i, sample in enumerate(list(tcga_sample_ids)[:5]):
            print(f"     {i+1}: {sample}")
    
    # Check if CIBERSORTx samples exist as files
    print(f"\n📂 Checking if CIBERSORTx samples exist as TCGA files:")
    
    existing_files = 0
    missing_files = 0
    
    for sample_id in list(cibersortx_samples)[:10]:  # Check first 10
        expected_file = f"{sample_id}.rna_seq.augmented_star_gene_counts.tsv"
        file_path = os.path.join(rna_dir, expected_file)
        
        if os.path.exists(file_path):
            existing_files += 1
            print(f"     ✅ {expected_file}")
        else:
            missing_files += 1
            print(f"     ❌ {expected_file}")
    
    print(f"\n📊 File Existence Summary (first 10 checked):")
    print(f"   Files found: {existing_files}")
    print(f"   Files missing: {missing_files}")
    
    if missing_files > existing_files:
        print("\n⚠️  DIAGNOSIS: CIBERSORTx results appear to be from a different dataset")
        print("   The sample IDs in your CIBERSORTx results don't correspond to")
        print("   RNA-seq files in your current TCGA directory.")
        print("\n💡 SOLUTION: You need to either:")
        print("   1. Re-run CIBERSORTx with the current TCGA samples, OR")
        print("   2. Use TCGA samples that match your CIBERSORTx results")
    
    # Check if LUAD samples specifically are in CIBERSORTx
    print(f"\n🔬 Checking TCGA sample sheet for LUAD samples...")
    sample_sheet_file = "TCGAdata/gdc_sample_sheet.2025-06-26.tsv"
    
    if os.path.exists(sample_sheet_file):
        sample_sheet = pd.read_csv(sample_sheet_file, sep='\t')
        luad_samples = sample_sheet[sample_sheet['Project ID'] == 'TCGA-LUAD']
        
        # Extract file UUIDs from LUAD samples
        luad_file_ids = set()
        for filename in luad_samples['File Name']:
            if filename.endswith('.tsv'):
                file_id = filename.split('.')[0]
                luad_file_ids.add(file_id)
        
        print(f"   LUAD samples in sample sheet: {len(luad_file_ids)}")
        print(f"   LUAD sample examples: {list(luad_file_ids)[:5]}")
        
        # Check overlap with CIBERSORTx
        luad_cibersortx_overlap = cibersortx_samples & luad_file_ids
        print(f"   LUAD-CIBERSORTx overlap: {len(luad_cibersortx_overlap)}")
        
        if luad_cibersortx_overlap:
            print(f"   ✅ Found LUAD samples in CIBERSORTx: {list(luad_cibersortx_overlap)[:5]}")
        else:
            print("   ❌ No LUAD samples found in CIBERSORTx results")

if __name__ == "__main__":
    diagnose_sample_mismatch() 