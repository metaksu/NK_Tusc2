#!/usr/bin/env python3
"""
Create Test Bulk Data
====================

Create a simple bulk data file for testing SCADEN workflow.
"""

import pandas as pd
import numpy as np
import scanpy as sc

# Load the simulated data to get gene names
adata = sc.read_h5ad("data.h5ad")
print(f"Simulated data shape: {adata.shape}")
print(f"Gene names: {adata.var.index[:10].tolist()}")

# Create simple bulk data (Gene × Sample format)
# Use the same genes as in the simulated data
n_genes = adata.n_vars
n_samples = 5  # Create 5 test samples

# Generate random bulk data
np.random.seed(42)
bulk_data = np.random.randint(50, 500, size=(n_genes, n_samples))

# Create DataFrame
bulk_df = pd.DataFrame(
    bulk_data,
    index=adata.var.index,  # Use same gene names
    columns=[f"sample{i}" for i in range(n_samples)]
)

print(f"Bulk data shape: {bulk_df.shape}")
print(f"Value range: {bulk_df.values.min()} - {bulk_df.values.max()}")

# Save bulk data
bulk_df.to_csv("test_bulk_data.txt", sep="\t", index=True)
print("Saved bulk data to: test_bulk_data.txt")

# Verify
test_df = pd.read_csv("test_bulk_data.txt", sep="\t", index_col=0)
print(f"Verification: {test_df.shape}")
print("✓ Test bulk data created successfully") 