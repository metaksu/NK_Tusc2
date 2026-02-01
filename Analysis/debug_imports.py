#!/usr/bin/env python3
"""Debug script to test imports individually."""

print("Testing basic imports...")

try:
    print("1. Testing os, json, logging...")
    import os
    import json
    import logging
    print("✅ Basic standard library imports OK")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    exit(1)

try:
    print("2. Testing pathlib...")
    from pathlib import Path
    print("✅ pathlib import OK")
except Exception as e:
    print(f"❌ pathlib import failed: {e}")
    exit(1)

try:
    print("3. Testing pandas...")
    import pandas as pd
    print("✅ pandas import OK")
except Exception as e:
    print(f"❌ pandas import failed: {e}")
    exit(1)

try:
    print("4. Testing numpy...")
    import numpy as np
    print("✅ numpy import OK")
except Exception as e:
    print(f"❌ numpy import failed: {e}")
    exit(1)

try:
    print("5. Testing scipy...")
    from scipy import sparse
    print("✅ scipy import OK")
except Exception as e:
    print(f"❌ scipy import failed: {e}")
    exit(1)

try:
    print("6. Testing scanpy...")
    import scanpy as sc
    print("✅ scanpy import OK")
except Exception as e:
    print(f"❌ scanpy import failed: {e}")
    exit(1)

print("All imports successful!") 