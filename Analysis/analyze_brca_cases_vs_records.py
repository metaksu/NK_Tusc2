#!/usr/bin/env python3
"""Analyze BRCA cases vs records discrepancy."""

import pandas as pd

try:
    print("Loading TCGA sample sheet...")
    sample_sheet = pd.read_csv("TCGAdata/gdc_sample_sheet.2025-06-26.tsv", sep='\t')
    
    # Filter for BRCA only
    brca_data = sample_sheet[sample_sheet['Project ID'] == 'TCGA-BRCA']
    print(f"✅ Total BRCA records: {len(brca_data)}")
    
    # Check unique cases
    unique_cases = brca_data['Case ID'].nunique()
    print(f"✅ Unique BRCA cases (patients): {unique_cases}")
    
    # Check unique samples
    unique_samples = brca_data['Sample ID'].nunique()
    print(f"✅ Unique BRCA samples: {unique_samples}")
    
    # Check unique files
    unique_files = brca_data['File ID'].nunique()
    print(f"✅ Unique BRCA files: {unique_files}")
    
    print(f"\n📊 Summary:")
    print(f"  - Records in sample sheet: {len(brca_data)}")
    print(f"  - Unique cases (patients): {unique_cases}")
    print(f"  - Unique samples: {unique_samples}")
    print(f"  - Unique files: {unique_files}")
    print(f"  - GDC website cases: 1,098")
    print(f"  - Pipeline expected samples: 881")
    
    # Check what types of data we have
    print(f"\n🔍 Data breakdown:")
    print("Data Categories:")
    print(brca_data['Data Category'].value_counts())
    
    print("\nData Types:")
    print(brca_data['Data Type'].value_counts())
    
    print("\nTissue Types:")
    print(brca_data['Tissue Type'].value_counts())
    
    # Check for cases with multiple records
    case_counts = brca_data['Case ID'].value_counts()
    cases_with_multiple_records = case_counts[case_counts > 1]
    
    print(f"\n📋 Cases with multiple records: {len(cases_with_multiple_records)}")
    print(f"Max records per case: {case_counts.max()}")
    print(f"Average records per case: {case_counts.mean():.2f}")
    
    if len(cases_with_multiple_records) > 0:
        print(f"\nTop 5 cases with most records:")
        print(cases_with_multiple_records.head())

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 