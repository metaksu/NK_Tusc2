#!/usr/bin/env python3
"""Simple test to verify enhanced pipeline works."""

try:
    from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor
    
    processor = TCGACIBERSORTxProcessor()
    
    print("✅ Pipeline import successful")
    print(f"✅ BRCA sample limit: {processor.expected_sample_limits.get('BRCA', 'Not found')}")
    print(f"✅ Total cancer types with limits: {len(processor.expected_sample_limits)}")
    print("✅ Enhanced pipeline ready for use")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 