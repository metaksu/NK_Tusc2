#!/usr/bin/env python3
"""
Test Script for TCGA CIBERSORTx Mixture Pipeline

This script demonstrates how to use the TCGACIBERSORTxProcessor
to create CIBERSORTx mixture files from raw TCGA data.

Usage:
    python test_cibersortx_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor

def test_single_cancer_type():
    """Test processing a single cancer type."""
    print("=" * 70)
    print("TESTING SINGLE CANCER TYPE PROCESSING")
    print("=" * 70)
    
    # Initialize processor
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir="TCGAdata",
        output_dir="outputs/tcga_cibersortx_mixtures_test",
        sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
    )
    
    # Load initial data
    print("\n1. Loading clinical data...")
    processor.clinical_data = processor.load_clinical_data()
    
    print("\n2. Loading sample sheet...")
    processor.sample_sheet_data = processor.load_sample_sheet()
    
    print("\n3. Creating master metadata...")
    processor.master_metadata = processor.create_master_metadata()
    
    print("\n4. Detecting cancer types...")
    processor.cancer_types = processor.detect_available_cancer_types()
    
    if processor.cancer_types:
        # Test with first available cancer type
        test_cancer = processor.cancer_types[0]
        print(f"\n5. Testing with {test_cancer}...")
        
        result = processor.process_cancer_type(test_cancer)
        
        if result:
            print(f"\n✅ SUCCESS: Created mixture file for {test_cancer}")
            print(f"   File: {result}")
        else:
            print(f"\n❌ FAILED: Could not process {test_cancer}")
    else:
        print("\n❌ No cancer types detected")

def test_full_pipeline():
    """Test the complete pipeline."""
    print("\n" + "=" * 70)
    print("TESTING FULL PIPELINE")
    print("=" * 70)
    
    # Initialize processor with test output directory
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir="TCGAdata",
        output_dir="outputs/tcga_cibersortx_mixtures_test",
        sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
    )
    
    # Run full pipeline
    results = processor.run_full_pipeline()
    
    # Print results
    if results:
        print(f"\n✅ PIPELINE SUCCESS: Processed {len(results)} cancer types")
        for cancer_type, mixture_file in results.items():
            file_size = Path(mixture_file).stat().st_size / (1024 * 1024)
            print(f"   {cancer_type}: {Path(mixture_file).name} ({file_size:.1f} MB)")
    else:
        print(f"\n❌ PIPELINE FAILED: No cancer types processed successfully")

def validate_output_files():
    """Validate the created output files."""
    print("\n" + "=" * 70)
    print("VALIDATING OUTPUT FILES")
    print("=" * 70)
    
    output_dir = Path("outputs/tcga_cibersortx_mixtures_test")
    
    if not output_dir.exists():
        print("❌ Output directory does not exist")
        return
    
    # Check for mixture files
    mixture_files = list(output_dir.glob("*_tumor_mixture_for_cibersortx_*.txt"))
    summary_files = list(output_dir.glob("*_mixture_metadata_summary_*.txt"))
    
    print(f"Found {len(mixture_files)} mixture files:")
    for mixture_file in mixture_files:
        file_size = mixture_file.stat().st_size / (1024 * 1024)
        print(f"  {mixture_file.name} ({file_size:.1f} MB)")
        
        # Quick validation of file format
        try:
            import pandas as pd
            df = pd.read_csv(mixture_file, sep='\t', nrows=5)
            print(f"    Format: {df.shape[0]} genes (sample) x {df.shape[1]} columns")
            print(f"    First column: {df.columns[0]} (should be 'GeneSymbol')")
            print(f"    Sample columns: {df.shape[1] - 1}")
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")
    
    print(f"\nFound {len(summary_files)} summary files:")
    for summary_file in summary_files:
        print(f"  {summary_file.name}")

def main():
    """Main test function."""
    print("🧪 TCGA CIBERSORTx Mixture Pipeline Test Suite")
    print("=" * 70)
    
    # Check if data directory exists
    tcga_dir = Path("TCGAdata")
    if not tcga_dir.exists():
        print(f"❌ TCGA data directory not found: {tcga_dir}")
        print("Please ensure TCGAdata directory exists with:")
        print("  - xml/ (clinical XML files)")
        print("  - rna/ (RNA-seq TSV files)")
        print("  - gdc_sample_sheet.2025-06-26.tsv")
        return 1
    
    try:
        # Test 1: Single cancer type processing
        test_single_cancer_type()
        
        # Test 2: Full pipeline (comment out if you want to test just one cancer type)
        test_full_pipeline()
        
        # Test 3: Validate output files
        validate_output_files()
        
        print("\n🎯 ALL TESTS COMPLETED")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 