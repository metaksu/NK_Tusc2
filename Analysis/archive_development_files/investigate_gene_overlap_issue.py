#!/usr/bin/env python3
"""
Investigate Gene Overlap Issue Between TCGA and Tang Reference Data
"""

import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_gene_names():
    """
    Investigate the gene naming issue between TCGA and Tang reference data
    """
    logger.info("🔍 Investigating Gene Naming Issue")
    logger.info("=" * 60)
    
    # 1. Load Tang reference gene names
    logger.info("Loading Tang reference gene names...")
    tang_genes_path = "tang_compatible_train/m256/genes.txt"
    
    if os.path.exists(tang_genes_path):
        with open(tang_genes_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Parse the Tang genes - they're formatted as "index\tgene_name"
        tang_genes = []
        for line in lines:
            if '\t' in line:
                # Split by tab and take the gene name part
                parts = line.split('\t')
                if len(parts) >= 2:
                    tang_genes.append(parts[1])
            else:
                # If no tab, it's just the gene name
                tang_genes.append(line)
        
        logger.info(f"✅ Tang reference genes loaded: {len(tang_genes)} genes")
        logger.info(f"First 10 Tang genes: {tang_genes[:10]}")
        logger.info(f"Last 10 Tang genes: {tang_genes[-10:]}")
    else:
        logger.error(f"❌ Tang gene file not found: {tang_genes_path}")
        return
    
    # 2. Load TCGA gene names (from GBM as example)
    logger.info("\nLoading TCGA gene names...")
    tcga_path = "TCGAdata/Analysis_Python_Output_v4_SCADEN/GBM/GBM_bulk_for_scaden.txt"
    
    if os.path.exists(tcga_path):
        # Read the file to get gene names from the first column
        tcga_genes = []
        with open(tcga_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split by tab and take the first column (gene names)
                    parts = line.split('\t')
                    if len(parts) > 0:
                        tcga_genes.append(parts[0])
        
        logger.info(f"✅ TCGA genes loaded: {len(tcga_genes)} genes")
        logger.info(f"First 10 TCGA genes: {tcga_genes[:10]}")
        logger.info(f"Last 10 TCGA genes: {tcga_genes[-10:]}")
    else:
        logger.error(f"❌ TCGA file not found: {tcga_path}")
        return
    
    # 3. Check for overlap patterns
    logger.info("\n🔍 Checking Gene Name Patterns")
    logger.info("-" * 40)
    
    # Convert to sets for efficient operations
    tang_set = set(tang_genes)
    tcga_set = set(tcga_genes)
    
    # Direct overlap
    direct_overlap = tang_set.intersection(tcga_set)
    logger.info(f"Direct overlap: {len(direct_overlap)} genes")
    
    if len(direct_overlap) > 0:
        logger.info(f"Sample overlapping genes: {list(direct_overlap)[:10]}")
    
    # Check for case sensitivity issues
    tang_upper = set(g.upper() for g in tang_genes)
    tcga_upper = set(g.upper() for g in tcga_genes)
    case_overlap = tang_upper.intersection(tcga_upper)
    logger.info(f"Case-insensitive overlap: {len(case_overlap)} genes")
    
    # Check if Tang genes are generic names (like gene1, gene2, etc.)
    generic_count = sum(1 for g in tang_genes if g.startswith('gene') and g[4:].isdigit())
    logger.info(f"Generic Tang gene names (gene123 format): {generic_count}")
    
    # Check TCGA gene format
    hugo_like = sum(1 for g in tcga_genes if g.isalpha() and g.isupper())
    logger.info(f"TCGA genes in HUGO format (uppercase letters): {hugo_like}")
    
    # 4. Examine specific gene examples
    logger.info("\n🔍 Examining Specific Gene Examples")
    logger.info("-" * 40)
    
    # Look for known genes in TCGA
    known_genes = ['NCAM1', 'FCGR3A', 'KLRF1', 'GZMB', 'PRF1', 'IFNG', 'TNF', 'IL2', 'KLRB1', 'KLRD1']
    found_known = []
    for gene in known_genes:
        if gene in tcga_genes:
            found_known.append(gene)
    
    logger.info(f"Known NK genes found in TCGA: {found_known}")
    
    # 5. Check if we're using the wrong Tang reference
    logger.info("\n🔍 Checking Tang Reference Source")
    logger.info("-" * 40)
    
    # Check if we have the original Tang data with real gene names
    original_tang_path = "data/processed"  # Look for original Tang data
    if os.path.exists(original_tang_path):
        logger.info("✅ Original Tang data directory exists")
        # List files in the directory
        files = os.listdir(original_tang_path)
        logger.info(f"Files in processed data: {files}")
    else:
        logger.info("❌ Original Tang data directory not found")
    
    # 6. Recommendations
    logger.info("\n💡 Recommendations")
    logger.info("=" * 60)
    
    if generic_count > 1000:
        logger.info("🚨 ISSUE IDENTIFIED: Tang genes are generic names (gene123 format)")
        logger.info("   This suggests the Tang model was trained on processed data")
        logger.info("   that lost the original HUGO gene symbols.")
        logger.info("\n🔧 SOLUTION:")
        logger.info("   1. Find the original Tang h5ad file with real gene names")
        logger.info("   2. Recreate the SCADEN dataset preserving HUGO symbols")
        logger.info("   3. Retrain the model with correct gene names")
        logger.info("\n   Alternative: Create a gene mapping from the original Tang data")
    
    if len(direct_overlap) == 0 and len(case_overlap) > 0:
        logger.info("🚨 ISSUE: Case sensitivity mismatch between gene names")
        logger.info("🔧 SOLUTION: Standardize gene names to uppercase")
    
    logger.info("\n✅ Investigation Complete!")

if __name__ == "__main__":
    investigate_gene_names() 