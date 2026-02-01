#!/usr/bin/env python3
"""
Fix CIBERSORTx-TCGA Integration by using exact sample matching.
This script modifies the TCGA analysis to work with your existing CIBERSORTx results.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def load_cibersortx_samples():
    """Load sample IDs from CIBERSORTx results."""
    cibersortx_file = "TCGAdata/CIBERSORTx_Job1_Results.csv"
    
    if not os.path.exists(cibersortx_file):
        print(f"❌ CIBERSORTx file not found: {cibersortx_file}")
        return set()
    
    print(f"📊 Loading CIBERSORTx sample IDs from: {cibersortx_file}")
    df = pd.read_csv(cibersortx_file, index_col="Mixture")
    
    # Apply quality filters (same as TCGA analysis)
    print(f"   Initial samples: {len(df)}")
    
    # P-value filter
    if 'P-value' in df.columns:
        df = df[df['P-value'] < 0.05]
        print(f"   After P-value filter: {len(df)}")
    
    # Correlation filter  
    if 'Correlation' in df.columns:
        df = df[df['Correlation'] > 0.1]
        print(f"   After Correlation filter: {len(df)}")
    
    # RMSE filter
    if 'RMSE' in df.columns:
        df = df[df['RMSE'] < 2.0]
        print(f"   After RMSE filter: {len(df)}")
    
    sample_ids = set(df.index)
    print(f"   Final high-quality samples: {len(sample_ids)}")
    
    return sample_ids

def get_cancer_types_from_sample_sheet(sample_ids):
    """Get cancer types for the CIBERSORTx sample IDs."""
    sample_sheet_file = "TCGAdata/gdc_sample_sheet.2025-06-26.tsv"
    
    if not os.path.exists(sample_sheet_file):
        print(f"❌ Sample sheet not found: {sample_sheet_file}")
        return {}
    
    print(f"📋 Loading sample sheet: {sample_sheet_file}")
    sample_sheet = pd.read_csv(sample_sheet_file, sep='\t')
    
    # Create mapping from file ID to cancer type
    cancer_mapping = {}
    
    for _, row in sample_sheet.iterrows():
        file_id = row['File Name'].split('.')[0]  # Extract UUID
        cancer_type = row['Project ID'].replace('TCGA-', '')  # Remove TCGA- prefix
        cancer_mapping[file_id] = cancer_type
    
    # Find cancer types for CIBERSORTx samples
    cibersortx_cancer_types = {}
    for sample_id in sample_ids:
        if sample_id in cancer_mapping:
            cancer_type = cancer_mapping[sample_id]
            if cancer_type not in cibersortx_cancer_types:
                cibersortx_cancer_types[cancer_type] = []
            cibersortx_cancer_types[cancer_type].append(sample_id)
    
    print(f"🔬 Cancer types in CIBERSORTx results:")
    for cancer_type, samples in cibersortx_cancer_types.items():
        print(f"   {cancer_type}: {len(samples)} samples")
    
    return cibersortx_cancer_types

def create_targeted_tcga_analysis():
    """Create a simplified TCGA analysis script that works with CIBERSORTx results."""
    
    script_content = '''#!/usr/bin/env python3
"""
Targeted TCGA Analysis for CIBERSORTx Integration
Works with existing CIBERSORTx results by using exact sample matching.
"""

import pandas as pd
import numpy as np
import os
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_rna_seq_for_samples(sample_ids, rna_dir="TCGAdata/rna"):
    """Load RNA-seq data for specific sample IDs."""
    print(f"\\n📊 Loading RNA-seq data for {len(sample_ids)} samples...")
    
    rna_data = {}
    genes_reference = None
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(sample_ids)} files...")
        
        filename = f"{sample_id}.rna_seq.augmented_star_gene_counts.tsv"
        filepath = os.path.join(rna_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"    ⚠️  File not found: {filename}")
            continue
        
        try:
            df = pd.read_csv(filepath, sep='\\t', comment='#', header=0, low_memory=False)
            
            if 'gene_name' not in df.columns or 'tpm_unstranded' not in df.columns:
                continue
            
            # Filter and process
            df = df[df['gene_name'].notna() & 
                   ~df['gene_name'].astype(str).str.upper().str.startswith('N_')]
            
            if df.empty:
                continue
                
            sample_counts = df.set_index('gene_name')['tpm_unstranded']
            sample_counts.name = sample_id
            
            # Establish gene reference from first sample
            if genes_reference is None:
                genes_reference = sample_counts.index
                print(f"    Using {len(genes_reference)} genes as reference")
            
            # Align to reference genes
            sample_counts = sample_counts.reindex(genes_reference, fill_value=0)
            rna_data[sample_id] = sample_counts
            
        except Exception as e:
            print(f"    ❌ Error loading {filename}: {e}")
    
    if not rna_data:
        print("❌ No RNA-seq data loaded")
        return None
    
    # Combine into DataFrame
    rna_df = pd.DataFrame(rna_data)
    print(f"✅ RNA-seq data loaded: {rna_df.shape[0]} genes x {rna_df.shape[1]} samples")
    
    # Handle duplicate genes
    if rna_df.index.duplicated().any():
        print("   Aggregating duplicate genes by mean...")
        rna_df = rna_df.groupby(rna_df.index).mean()
    
    return rna_df

def load_cibersortx_results(cibersortx_file="TCGAdata/CIBERSORTx_Job1_Results.csv"):
    """Load and filter CIBERSORTx results."""
    print(f"\\n🔬 Loading CIBERSORTx results: {cibersortx_file}")
    
    df = pd.read_csv(cibersortx_file, index_col="Mixture")
    print(f"   Raw results: {df.shape}")
    
    # Apply quality filters
    if 'P-value' in df.columns:
        df = df[df['P-value'] < 0.05]
        print(f"   After P-value filter: {df.shape[0]} samples")
    
    if 'Correlation' in df.columns:
        df = df[df['Correlation'] > 0.1] 
        print(f"   After Correlation filter: {df.shape[0]} samples")
        
    if 'RMSE' in df.columns:
        df = df[df['RMSE'] < 2.0]
        print(f"   After RMSE filter: {df.shape[0]} samples")
    
    # Remove quality metric columns
    quality_cols = ['P-value', 'Correlation', 'RMSE', 'Absolute score (sig.score)']
    for col in quality_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    print(f"✅ Final CIBERSORTx data: {df.shape}")
    return df

def analyze_cibersortx_tcga_integration():
    """Main analysis function."""
    print("🚀 Starting CIBERSORTx-TCGA Integration Analysis")
    print("=" * 60)
    
    # Load CIBERSORTx results
    cibersortx_df = load_cibersortx_results()
    if cibersortx_df is None or cibersortx_df.empty:
        print("❌ Failed to load CIBERSORTx results")
        return
    
    sample_ids = list(cibersortx_df.index)
    
    # Load RNA-seq data for these specific samples
    rna_df = load_rna_seq_for_samples(sample_ids)
    if rna_df is None:
        print("❌ Failed to load RNA-seq data")
        return
    
    # Find common samples
    common_samples = list(set(cibersortx_df.index) & set(rna_df.columns))
    print(f"\\n🔍 Sample Matching:")
    print(f"   CIBERSORTx samples: {len(cibersortx_df)}")
    print(f"   RNA-seq samples: {len(rna_df.columns)}")
    print(f"   Common samples: {len(common_samples)}")
    
    if not common_samples:
        print("❌ No common samples found!")
        return
    
    # Subset to common samples
    cibersortx_final = cibersortx_df.loc[common_samples]
    rna_final = rna_df[common_samples]
    
    print(f"\\n✅ Integration successful!")
    print(f"   Final dataset: {len(common_samples)} samples")
    print(f"   NK subtypes available: {['Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK']}")
    
    # Calculate NK totals
    nk_cols = ['Bright_NK', 'Cytotoxic_NK', 'Exhausted_TaNK']
    cibersortx_final['NK_Total'] = cibersortx_final[nk_cols].sum(axis=1)
    
    # Basic statistics
    print(f"\\n📊 NK Cell Infiltration Summary:")
    print(f"   Mean total NK: {cibersortx_final['NK_Total'].mean():.4f}")
    print(f"   NK range: {cibersortx_final['NK_Total'].min():.4f} - {cibersortx_final['NK_Total'].max():.4f}")
    
    for nk_col in nk_cols:
        detected = (cibersortx_final[nk_col] > 0).sum()
        print(f"   {nk_col}: detected in {detected}/{len(common_samples)} samples ({detected/len(common_samples)*100:.1f}%)")
    
    # Save results
    output_dir = Path("TCGAdata/Analysis_Python_Output")
    output_dir.mkdir(exist_ok=True)
    
    # Save integrated dataset
    cibersortx_final.to_csv(output_dir / "CIBERSORTx_TCGA_Integrated_Results.csv")
    print(f"\\n💾 Results saved to: {output_dir / 'CIBERSORTx_TCGA_Integrated_Results.csv'}")
    
    return cibersortx_final, rna_final

if __name__ == "__main__":
    results = analyze_cibersortx_tcga_integration()
'''
    
    # Save the script
    with open("targeted_tcga_cibersortx_analysis.py", "w") as f:
        f.write(script_content)
    
    print("📄 Created: targeted_tcga_cibersortx_analysis.py")

def main():
    """Main function to diagnose and fix CIBERSORTx-TCGA integration."""
    
    print("🔧 CIBERSORTx-TCGA Integration Fix")
    print("=" * 50)
    
    # Step 1: Load CIBERSORTx sample IDs
    cibersortx_samples = load_cibersortx_samples()
    if not cibersortx_samples:
        print("❌ No CIBERSORTx samples found")
        return
    
    # Step 2: Map to cancer types
    cancer_mapping = get_cancer_types_from_sample_sheet(cibersortx_samples)
    if not cancer_mapping:
        print("❌ No cancer type mapping found")
        return
    
    # Step 3: Create targeted analysis script
    create_targeted_tcga_analysis()
    
    print(f"\n✅ Integration fix complete!")
    print(f"📋 Summary:")
    print(f"   • CIBERSORTx samples: {len(cibersortx_samples)}")
    print(f"   • Cancer types: {len(cancer_mapping)}")
    print(f"   • Analysis script: targeted_tcga_cibersortx_analysis.py")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Run: python targeted_tcga_cibersortx_analysis.py")
    print(f"   2. Check results in TCGAdata/Analysis_Python_Output/")

if __name__ == "__main__":
    main() 