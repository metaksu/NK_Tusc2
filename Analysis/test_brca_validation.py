#!/usr/bin/env python3
"""
BRCA Validation Test for TCGA CIBERSORTx Mixture Pipeline

This script specifically tests BRCA processing to ensure sample counts
don't exceed the ground truth limit of 881 samples.

Ground Truth Data:
- 1,098 total BRCA cases
- 1,097 with clinical data
- 881 with proteome profiling data
- Therefore maximum 881 final merged files for BRCA mixture

Usage:
    python test_brca_validation.py
"""

import sys
import os
from pathlib import Path
import json

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor

def test_brca_sample_limits():
    """Test BRCA specifically to validate sample count limits."""
    print("=" * 70)
    print("BRCA SAMPLE COUNT VALIDATION TEST")
    print("=" * 70)
    print("Ground Truth: BRCA should have ≤881 tumor samples")
    print("=" * 70)
    
    # Initialize processor
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir="TCGAdata",
        output_dir="outputs/tcga_cibersortx_mixtures_brca_test",
        sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
    )
    
    try:
        # Load initial data
        print("\n1. Loading clinical data...")
        processor.clinical_data = processor.load_clinical_data()
        print(f"   Loaded {len(processor.clinical_data)} clinical records")
        
        print("\n2. Loading sample sheet...")
        processor.sample_sheet_data = processor.load_sample_sheet()
        print(f"   Loaded {len(processor.sample_sheet_data)} sample sheet records")
        
        print("\n3. Creating master metadata...")
        processor.master_metadata = processor.create_master_metadata()
        print(f"   Created master metadata: {processor.master_metadata.shape}")
        
        print("\n4. Detecting cancer types...")
        processor.cancer_types = processor.detect_available_cancer_types()
        print(f"   Detected cancer types: {processor.cancer_types}")
        
        if 'BRCA' not in processor.cancer_types:
            print("\n❌ BRCA not found in detected cancer types!")
            print(f"Available types: {processor.cancer_types}")
            return False
        
        # Analyze BRCA data step by step
        print(f"\n5. Analyzing BRCA data in detail...")
        
        # Filter for BRCA samples in metadata
        brca_metadata = processor.master_metadata[
            processor.master_metadata["Cancer_Type_Derived"] == "BRCA"
        ]
        print(f"   Total BRCA samples in metadata: {len(brca_metadata)}")
        
        # Check tissue types
        if "Tissue_Type" in brca_metadata.columns:
            tissue_counts = brca_metadata["Tissue_Type"].value_counts()
            print(f"   BRCA tissue type distribution:")
            for tissue, count in tissue_counts.items():
                print(f"     {tissue}: {count}")
            
            # Filter for tumor samples
            tumor_mask = brca_metadata["Tissue_Type"].astype(str).str.contains("Tumor", case=False, na=False)
            brca_tumor_metadata = brca_metadata[tumor_mask]
            print(f"   BRCA tumor samples in metadata: {len(brca_tumor_metadata)}")
        else:
            print("   WARNING: No 'Tissue_Type' column found")
            brca_tumor_metadata = brca_metadata
        
        # Load RNA-seq data for BRCA samples
        print(f"\n6. Loading RNA-seq data for BRCA...")
        target_sample_ids = set(brca_metadata.index)
        rna_data = processor.load_rna_seq_data(
            target_sample_ids, 
            processor.cibersortx_thresholds["preferred_rna_count_column"]
        )
        
        if rna_data.empty:
            print("   ❌ No RNA-seq data loaded for BRCA")
            return False
        
        print(f"   RNA-seq data shape: {rna_data.shape}")
        print(f"   RNA-seq samples: {rna_data.shape[1]}")
        
        # Find common samples between RNA-seq and tumor metadata
        common_samples = list(set(rna_data.columns) & set(brca_tumor_metadata.index))
        print(f"   Common tumor samples with RNA-seq: {len(common_samples)}")
        
        # Validate against ground truth
        expected_limit = 881
        print(f"\n7. Validation Results:")
        print(f"   Expected maximum: {expected_limit}")
        print(f"   Actual samples: {len(common_samples)}")
        
        if len(common_samples) > expected_limit:
            print(f"   ❌ VALIDATION FAILED: {len(common_samples)} > {expected_limit}")
            print(f"   This suggests data quality issues or incorrect filtering")
            validation_passed = False
        elif len(common_samples) > expected_limit * 0.9:
            print(f"   ⚠️  WARNING: {len(common_samples)} approaching limit of {expected_limit}")
            validation_passed = True
        else:
            print(f"   ✅ VALIDATION PASSED: {len(common_samples)} within expected limit")
            validation_passed = True
        
        # Additional analysis
        print(f"\n8. Additional Analysis:")
        
        # Check for duplicates
        if len(set(common_samples)) != len(common_samples):
            print("   ⚠️  WARNING: Duplicate sample IDs found")
        else:
            print("   ✅ No duplicate sample IDs")
        
        # Check sample ID format
        sample_formats = {}
        for sample in common_samples[:10]:  # Check first 10
            if '-' in sample:
                parts = sample.split('-')
                format_key = f"{len(parts)} parts"
            else:
                format_key = "No dashes"
            sample_formats[format_key] = sample_formats.get(format_key, 0) + 1
        
        print(f"   Sample ID formats (first 10): {sample_formats}")
        
        # Now test full processing
        print(f"\n9. Testing full BRCA processing...")
        result = processor.process_cancer_type('BRCA')
        
        if result:
            print(f"   ✅ BRCA processing completed successfully")
            print(f"   Mixture file: {result}")
            
            # Validate final file
            try:
                import pandas as pd
                df = pd.read_csv(result, sep='\t', nrows=1)
                final_samples = df.shape[1] - 1  # -1 for gene column
                print(f"   Final samples in mixture file: {final_samples}")
                
                if final_samples > expected_limit:
                    print(f"   ❌ FINAL VALIDATION FAILED: {final_samples} > {expected_limit}")
                    validation_passed = False
                else:
                    print(f"   ✅ FINAL VALIDATION PASSED: {final_samples} ≤ {expected_limit}")
                    
            except Exception as e:
                print(f"   ❌ Could not validate final file: {e}")
                validation_passed = False
        else:
            print(f"   ❌ BRCA processing failed")
            validation_passed = False
        
        return validation_passed
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_all_cancer_types():
    """Run validation on all cancer types to check sample limits."""
    print("\n" + "=" * 70)
    print("VALIDATION TEST FOR ALL CANCER TYPES")
    print("=" * 70)
    
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir="TCGAdata",
        output_dir="outputs/tcga_cibersortx_mixtures_validation_test",
        sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
    )
    
    # Run full pipeline
    results = processor.run_full_pipeline()
    
    if results:
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"✅ Successfully processed: {len(results)} cancer types")
        
        # Read validation results from summary file
        summary_file = processor.output_dir / "pipeline_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            if 'sample_count_summary' in summary:
                print(f"\n📈 Sample Count Validation:")
                validation_failures = 0
                for cancer_type, info in summary['sample_count_summary'].items():
                    actual = info['actual_samples']
                    expected = info['expected_limit']
                    passed = info['validation_passed']
                    
                    status = "✅ PASS" if passed else "❌ FAIL"
                    print(f"  {cancer_type}: {actual}/{expected} samples - {status}")
                    
                    if not passed:
                        validation_failures += 1
                
                print(f"\n🎯 Overall Validation Result:")
                if validation_failures == 0:
                    print(f"✅ ALL VALIDATIONS PASSED")
                else:
                    print(f"❌ {validation_failures} VALIDATION FAILURES")
                
                return validation_failures == 0
            else:
                print("⚠️  No validation summary found in pipeline results")
                return False
        else:
            print("⚠️  Pipeline summary file not found")
            return False
    else:
        print("❌ Pipeline failed completely")
        return False

def main():
    """Main test function."""
    print("🧪 BRCA Validation Test Suite")
    print("Ground Truth: BRCA ≤881 tumor samples")
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
        # Test 1: BRCA-specific validation
        brca_validation = test_brca_sample_limits()
        
        # Test 2: All cancer types validation
        all_validation = validate_all_cancer_types()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"BRCA validation: {'PASSED' if brca_validation else 'FAILED'}")
        print(f"All cancer types validation: {'PASSED' if all_validation else 'FAILED'}")
        
        if brca_validation and all_validation:
            print(f"\n✅ ALL TESTS PASSED")
            return 0
        else:
            print(f"\n❌ SOME TESTS FAILED")
            return 1
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 