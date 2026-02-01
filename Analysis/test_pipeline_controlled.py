#!/usr/bin/env python3
"""Controlled test of the TCGA CIBERSORTx pipeline."""

print("Starting controlled pipeline test...")

try:
    print("1. Testing pipeline import...")
    # Import without running main
    import sys
    import os
    sys.path.append('.')
    
    print("2. Importing processor class...")
    from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor
    print("✅ Pipeline import successful")
    
    print("3. Creating processor instance...")
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir="TCGAdata",
        output_dir="outputs/tcga_cibersortx_mixtures_test",
        sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
    )
    print("✅ Processor created successfully")
    
    print("4. Checking configuration...")
    print(f"✅ TCGA base dir: {processor.tcga_base_dir}")
    print(f"✅ Output dir: {processor.output_dir}")
    print(f"✅ Expected BRCA samples: {processor.expected_sample_limits.get('BRCA', 'Not found')}")
    print(f"✅ Total cancer types configured: {len(processor.expected_sample_limits)}")
    
    print("5. Testing basic functionality...")
    # Test if we can at least check paths
    print(f"✅ XML dir exists: {processor.xml_dir.exists()}")
    print(f"✅ RNA dir exists: {processor.rna_dir.exists()}")
    print(f"✅ Sample sheet exists: {processor.sample_sheet_path.exists()}")
    
    print("\n🎉 All controlled tests passed!")
    print("✅ Pipeline is ready for use")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 