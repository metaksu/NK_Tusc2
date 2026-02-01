#!/usr/bin/env python3
"""Minimal test to isolate pipeline import issues."""

print("Starting minimal pipeline test...")

try:
    print("1. Testing basic imports...")
    import os
    import json
    import logging
    from pathlib import Path
    import pandas as pd
    import numpy as np
    print("✅ Basic imports OK")
    
    print("2. Testing XML import...")
    import xml.etree.ElementTree as ET
    print("✅ XML import OK")
    
    print("3. Creating simple class...")
    class SimpleProcessor:
        def __init__(self):
            self.tcga_base_dir = Path("TCGAdata")
            self.output_dir = Path("outputs/tcga_cibersortx_mixtures")
            print("Simple processor initialized")
    
    print("4. Creating instance...")
    processor = SimpleProcessor()
    print("✅ Simple processor created successfully")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 