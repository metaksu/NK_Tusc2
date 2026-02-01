#!/usr/bin/env python3
"""
Analyze Tang Tumor Reference Matrix Structure - Memory Efficient Version
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

def analyze_tang_reference_matrix():
    """Analyze the structure of Tang_Tumor_reference_matrix_700cells_per_phenotype.txt efficiently"""
    
    file_path = "outputs/signature_matrices/CIBERSORTx_Input_Files/Tang_Tumor_reference_matrix_700cells_per_phenotype.txt"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    print("\n--- Memory-efficient analysis ---")
    
    with open(file_path, 'r') as f:
        # Read header line
        header_line = f.readline().strip()
        columns = header_line.split('\t')
        
        print(f"Number of columns: {len(columns)}")
        print(f"First column (gene symbol header): '{columns[0]}'")
        
        # Analyze cell type columns (skip first column which is gene names)
        cell_type_columns = columns[1:]
        print(f"Number of cells: {len(cell_type_columns)}")
        
        # Count unique cell types
        cell_type_counts = Counter(cell_type_columns)
        
        print(f"\nCell type distribution:")
        for cell_type, count in sorted(cell_type_counts.items()):
            print(f"  {cell_type}: {count} cells")
        
        print(f"\nSample cell type columns (first 10): {cell_type_columns[:10]}")
        print(f"Sample cell type columns (last 10): {cell_type_columns[-10:]}")
        
        # Read just a few data lines to understand format
        print(f"\n--- Sample data analysis ---")
        data_lines = []
        for i in range(5):  # Read 5 data rows
            line = f.readline().strip()
            if line:
                data_lines.append(line)
            else:
                break
        
        # Analyze data format
        if data_lines:
            print(f"Sample data rows (first 5 genes):")
            for i, line in enumerate(data_lines):
                parts = line.split('\t')
                gene_name = parts[0]
                first_values = parts[1:6]  # First 5 expression values
                print(f"  Gene {i+1}: {gene_name} -> {first_values}")
            
            # Analyze expression values from first gene
            sample_values = data_lines[0].split('\t')[1:]  # All expression values for first gene
            try:
                float_values = [float(v) for v in sample_values]
                print(f"\nExpression value analysis (first gene across all cells):")
                print(f"  Number of values: {len(float_values)}")
                print(f"  Min value: {min(float_values):.3f}")
                print(f"  Max value: {max(float_values):.3f}")
                print(f"  Mean value: {np.mean(float_values):.3f}")
                print(f"  Zero values: {sum(1 for v in float_values if v == 0.0)}")
                print(f"  Non-zero values: {sum(1 for v in float_values if v > 0.0)}")
                
                # Check if data looks like TPM/expression data
                max_val = max(float_values)
                has_high_values = max_val > 100
                print(f"  Data characteristics: {'Likely TPM/expression data' if has_high_values else 'Possibly normalized data'}")
                
            except ValueError as e:
                print(f"  WARNING: Could not parse values as floats: {e}")

    # Count total lines (genes) efficiently
    print(f"\n--- Counting total genes ---")
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    
    gene_count = line_count - 1  # Subtract header
    print(f"Total genes: {gene_count}")
    print(f"Matrix dimensions: {gene_count} genes x {len(cell_type_columns)} cells")
    
    print(f"\n--- File structure summary ---")
    print(f"• Tab-separated format")
    print(f"• First column: Gene symbols")
    print(f"• Remaining columns: Expression values per cell")
    print(f"• {len(set(cell_type_columns))} unique cell types")
    print(f"• Mostly 700 cells per cell type (some have fewer)")
    print(f"• Expression values appear to be TPM-normalized")

if __name__ == "__main__":
    analyze_tang_reference_matrix() 