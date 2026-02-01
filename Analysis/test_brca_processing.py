#!/usr/bin/env python3
"""Test BRCA processing specifically."""

print("Starting BRCA-focused test...")

try:
    print("1. Importing pipeline...")
    from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor
    print("✅ Pipeline imported")
    
    print("2. Creating processor...")
    processor = TCGACIBERSORTxProcessor(
        tcga_base_dir="TCGAdata",
        output_dir="outputs/tcga_cibersortx_mixtures_test",
        sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
    )
    print("✅ Processor created")
    
    print("3. Loading basic data...")
    # Load basic data first
    processor.load_clinical_data()
    print(f"✅ Clinical data loaded: {len(processor.clinical_data)} records")
    
    processor.load_sample_sheet()
    print(f"✅ Sample sheet loaded: {len(processor.sample_sheet_data)} records")
    
    # Find available cancer types
    processor.detect_cancer_types()
    print(f"✅ Cancer types detected: {list(processor.cancer_types)}")
    
    print("4. Testing BRCA processing...")
    if 'BRCA' in processor.cancer_types:
        print("🎯 BRCA data found - attempting to process...")
        
        # Process just BRCA
        result = processor.process_cancer_type('BRCA')
        
        if result:
            print("✅ BRCA processing completed successfully!")
            print(f"✅ Final sample count: {result.get('final_sample_count', 'Unknown')}")
            print(f"✅ Validation status: {result.get('validation_passed', 'Unknown')}")
            print(f"✅ Output file: {result.get('output_file', 'Unknown')}")
        else:
            print("❌ BRCA processing failed")
    else:
        print("❌ BRCA data not found in cancer types")
    
    print("\n🎉 BRCA test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 