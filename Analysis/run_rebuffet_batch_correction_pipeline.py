#!/usr/bin/env python3
"""
Rebuffet Batch Correction Pipeline Runner
========================================

Simple script to run the complete Rebuffet Seurat → h5ad pipeline with batch correction.

This script:
1. Checks for required dependencies
2. Runs the batch correction pipeline
3. Validates the output
4. Provides usage instructions
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'scanpy', 'pandas', 'numpy', 'scipy', 'matplotlib', 
        'seaborn', 'sklearn', 'anndata'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    # Check for optional but recommended packages
    optional_packages = ['harmonypy', 'leidenalg']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️  {package} (optional, recommended for batch correction)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_batch_correction.txt")
        return False
    
    print("✅ All required dependencies found")
    return True

def check_input_files():
    """Check if required input files exist"""
    print("\nChecking input files...")
    
    required_files = [
        "data/raw/rebuffet_counts.csv",
        "data/raw/rebuffet_metadata.csv", 
        "data/raw/rebuffet_genes.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✅ {file_path} ({file_size:.1f} MB)")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing input files. Please run the R export script first:")
        print("  Rscript export_seurat_to_csv.R")
        return False
    
    print("✅ All input files found")
    return True

def run_pipeline():
    """Run the batch correction pipeline"""
    print("\n" + "="*60)
    print("RUNNING BATCH CORRECTION PIPELINE")
    print("="*60)
    
    try:
        # Import and run the pipeline
        from create_rebuffet_h5ad_with_batch_correction import RebuffetPipelineWithBatchCorrection
        
        # Initialize pipeline
        pipeline = RebuffetPipelineWithBatchCorrection()
        
        # Run pipeline
        adata = pipeline.run_pipeline()
        
        return adata
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None

def validate_output():
    """Validate the output h5ad file"""
    print("\nValidating output...")
    
    output_file = "data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_BatchCorrected.h5ad"
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    try:
        import scanpy as sc
        adata = sc.read_h5ad(output_file)
        
        print(f"✅ Output file validated:")
        print(f"  📊 Shape: {adata.shape}")
        print(f"  🧬 Raw data: {adata.raw.shape if adata.raw else 'Not available'}")
        print(f"  🎯 Batch correction: {'Applied' if 'batch_correction' in adata.uns else 'Not applied'}")
        
        # Check for essential components
        if hasattr(adata, 'raw') and adata.raw is not None:
            print("  ✅ Raw data preserved for gene expression analysis")
        
        if 'highly_variable' in adata.var:
            n_hvgs = adata.var['highly_variable'].sum()
            print(f"  ✅ Highly variable genes: {n_hvgs}")
        
        if 'X_pca' in adata.obsm:
            print("  ✅ PCA computed")
        
        if 'X_umap' in adata.obsm:
            print("  ✅ UMAP computed")
        
        # Check for layers
        if adata.layers:
            print(f"  ✅ Data layers: {list(adata.layers.keys())}")
        
        print(f"\n📁 Output file ready: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Output validation failed: {e}")
        return False

def print_usage_instructions():
    """Print instructions for using the batch-corrected data"""
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n📖 How to use the batch-corrected data:")
    print("")
    print("1. In your NK analysis scripts, update the file path:")
    print("   REBUFFET_H5AD_FILE = 'data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_BatchCorrected.h5ad'")
    print("")
    print("2. For gene expression analysis, use adata.raw.X:")
    print("   gene_expr = adata.raw[:, 'TUSC2'].X")
    print("")
    print("3. For dimensionality reduction/clustering, use adata.X:")
    print("   sc.tl.leiden(adata)  # Uses batch-corrected data")
    print("")
    print("4. Original TPM data (if available) is in adata.layers['tpm']")
    print("")
    print("5. Check the batch correction report:")
    print("   cat data/processed/batch_correction_report.txt")
    print("")
    print("🔬 The data is now ready for NK analysis with reduced batch effects!")

def main():
    """Main execution function"""
    print("="*60)
    print("REBUFFET BATCH CORRECTION PIPELINE RUNNER")
    print("="*60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return 1
    
    # Step 2: Check input files
    if not check_input_files():
        return 1
    
    # Step 3: Run pipeline
    adata = run_pipeline()
    if adata is None:
        return 1
    
    # Step 4: Validate output
    if not validate_output():
        return 1
    
    # Step 5: Print usage instructions
    print_usage_instructions()
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 