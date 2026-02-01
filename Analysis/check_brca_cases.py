#!/usr/bin/env python3
"""Check BRCA-specific cases in the sample sheet."""

import pandas as pd

try:
    print("Loading TCGA sample sheet...")
    sample_sheet = pd.read_csv("TCGAdata/gdc_sample_sheet.2025-06-26.tsv", sep='\t')
    print(f"✅ Total sample sheet records: {len(sample_sheet)}")
    
    print("\nColumn names in sample sheet:")
    print(sample_sheet.columns.tolist())
    
    # Look for cancer type information in various possible columns
    possible_cancer_cols = ['Project ID', 'project_id', 'cancer_type', 'Cancer_Type', 'disease', 'Disease']
    
    for col in possible_cancer_cols:
        if col in sample_sheet.columns:
            print(f"\n✅ Found cancer type column: '{col}'")
            
            # Get unique values
            unique_values = sample_sheet[col].unique()
            print(f"Unique values in {col}:")
            for val in sorted(unique_values):
                count = len(sample_sheet[sample_sheet[col] == val])
                print(f"  {val}: {count} records")
                
            # Check specifically for BRCA
            brca_records = sample_sheet[sample_sheet[col].str.contains('BRCA', na=False)]
            print(f"\n🎯 BRCA-specific records: {len(brca_records)}")
            break
    else:
        print("❌ No cancer type column found")
        print("First few rows of sample sheet:")
        print(sample_sheet.head())

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 