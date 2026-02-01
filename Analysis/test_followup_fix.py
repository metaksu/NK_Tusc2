#!/usr/bin/env python3
"""
Test script to validate extended follow-up parsing fix
"""

import sys
import os
sys.path.append('.')

# Import the parsing function
try:
    from scripts.tcga_analysis.TCGA_Gene_Survival_Analysis import parse_tcga_clinical_xml_file
except ImportError:
    try:
        sys.path.append('scripts/07_tcga_analysis')
        from TCGA_Gene_Survival_Analysis import parse_tcga_clinical_xml_file
    except ImportError:
        print("Failed to import parsing function")

# Test on a single BRCA file
test_xml = "TCGAdata/xml/nationwidechildrens.org_clinical.TCGA-A7-A13E.xml"

if os.path.exists(test_xml):
    print("🔍 Testing Enhanced XML Parsing...")
    print(f"File: {test_xml}")
    
    try:
        result = parse_tcga_clinical_xml_file(test_xml)
        
        if result:
            print("\n📊 SURVIVAL DATA COMPARISON:")
            print(f"Initial Follow-up: {result.get('Initial_Days_to_Followup', 'N/A')} days")
            print(f"Final Follow-up: {result.get('Days_to_Last_Followup', 'N/A')} days")
            print(f"Extended Follow-up Available: {result.get('Extended_Followup_Available', 'N/A')}")
            
            # Calculate years
            try:
                initial_days = int(result.get('Initial_Days_to_Followup', '0'))
                final_days = int(result.get('Days_to_Last_Followup', '0'))
                print(f"Initial Follow-up: {initial_days/365.25:.1f} years")
                print(f"Final Follow-up: {final_days/365.25:.1f} years")
                
                if final_days > initial_days:
                    print(f"✅ SUCCESS: Extended follow-up found (+{final_days-initial_days} days)")
                else:
                    print("⚠️  No extended follow-up found - may need different XML namespace")
                    
            except (ValueError, TypeError):
                print("⚠️  Could not parse follow-up days")
                
        else:
            print("❌ Failed to parse XML file")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"❌ Test file not found: {test_xml}")
    print("Available XML files:")
    for f in os.listdir("TCGAdata/xml/")[:5]:
        if "TCGA-A" in f:
            print(f"  {f}") 