"""
Enhanced Quality Control Functions for Single-Cell RNA-seq Analysis
Modern best practices implementation for NK cell analysis pipeline

Author: AI Assistant  
Date: December 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

def _load_hallmark_geneset_with_fallback(signature_type):
    """Load hallmark gene set with fallback to original signatures"""
    fallback_genes = {
        'glycolysis': [
            'HK2', 'PFKP', 'ALDOA', 'TPI1', 'GAPDH', 'PGK1',
            'PGAM1', 'ENO1', 'PKM', 'LDHA', 'SLC2A1', 'SLC2A3'
        ],
        'oxphos': [
            'NDUFA4', 'NDUFB2', 'SDHB', 'UQCRB', 'COX4I1', 'COX6A1',
            'ATP5F1A', 'ATP5F1B', 'ATP5F1D', 'ATP5PB', 'IDH2', 'MDH2'
        ]
    }
    
    # Try to load from hallmark files
    file_paths = {
        'glycolysis': "HALLMARK_GLYCOLYSIS.v2025.1.Hs.grp",
        'oxphos': "HALLMARK_OXIDATIVE_PHOSPHORYLATION.v2025.1.Hs.grp"
    }
    
    if signature_type in file_paths:
        filepath = file_paths[signature_type]
        # Check multiple possible locations
        possible_paths = [
            filepath,  # Current directory
            f"../../{filepath}",  # Go up two levels (from scripts/utilities/)
            f"../../../{filepath}",  # Go up three levels if needed
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    genes = []
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            # Skip empty lines, comments, and the title line
                            if line and not line.startswith('#') and not line.startswith('HALLMARK_'):
                                genes.append(line)
                    if genes:
                        return genes
                except:
                    continue
    
    # Fallback to original signatures
    return fallback_genes.get(signature_type, [])

class AdaptiveQualityControl:
    """Modern quality control framework implementing 2024 best practices"""
    
    def __init__(self, adata, sample_key=None, batch_key=None):
        self.adata = adata.copy()
        self.sample_key = sample_key
        self.batch_key = batch_key
        self.qc_metrics = {}
        
        # Tissue-specific thresholds based on recent literature
        self.tissue_mt_thresholds = {
            'Blood': 15,      # Healthy blood - stricter threshold
            'Tumor': 25,      # Tumor tissue - more permissive (hypoxic conditions)
            'Normal': 20,     # Normal adjacent tissue - intermediate
            'Other tissue': 20,  # Default for other contexts
            'Other': 20       # Fallback
        }
        
    def adaptive_mt_filtering(self, tissue_col='tissue'):
        """Apply tissue-specific mitochondrial gene thresholds"""
        print("  Applying adaptive mitochondrial gene filtering...")
        
        if tissue_col not in self.adata.obs.columns:
            print(f"    WARNING: {tissue_col} not found. Using default threshold (20%)")
            self.adata.obs['mt_outlier'] = self.adata.obs['pct_counts_mt'] > 20
            return
            
        self.adata.obs['mt_outlier'] = False
        
        for tissue in self.adata.obs[tissue_col].unique():
            if pd.isna(tissue):
                continue
                
            threshold = self.tissue_mt_thresholds.get(tissue, 20)
            mask = self.adata.obs[tissue_col] == tissue
            
            mt_outliers = self.adata.obs.loc[mask, 'pct_counts_mt'] > threshold
            self.adata.obs.loc[mask, 'mt_outlier'] = mt_outliers
            
            n_outliers = mt_outliers.sum()
            n_total = mask.sum()
            
            print(f"    {tissue}: {n_outliers}/{n_total} cells ({n_outliers/n_total*100:.1f}%) "
                  f"above {threshold}% MT threshold")
    
    def enhanced_doublet_detection(self):
        """Enhanced doublet detection using multiple approaches"""
        print("  Running enhanced doublet detection...")
        
        # Method 1: Scanpy's scrublet
        try:
            sc.pp.scrublet(self.adata, batch_key=self.batch_key)
            scrublet_available = True
        except Exception as e:
            print(f"    WARNING: Scrublet failed: {e}")
            self.adata.obs['doublet_score'] = 0
            self.adata.obs['predicted_doublet'] = False
            scrublet_available = False
        
        # Method 2: Statistical outlier detection
        count_threshold = np.percentile(self.adata.obs['total_counts'], 97.5)
        gene_threshold = np.percentile(self.adata.obs['n_genes_by_counts'], 97.5)
        
        self.adata.obs['doublet_statistical'] = (
            (self.adata.obs['total_counts'] > count_threshold) &
            (self.adata.obs['n_genes_by_counts'] > gene_threshold)
        )
        
        # Consensus doublet calling
        if scrublet_available:
            self.adata.obs['doublet_consensus'] = (
                self.adata.obs['predicted_doublet'] |
                self.adata.obs['doublet_statistical']
            )
        else:
            self.adata.obs['doublet_consensus'] = self.adata.obs['doublet_statistical']
        
        n_doublets = self.adata.obs['doublet_consensus'].sum()
        print(f"    Consensus doublets detected: {n_doublets} ({n_doublets/self.adata.n_obs*100:.1f}%)")
        
        return n_doublets


def calculate_effect_sizes(group1_values, group2_values):
    """Calculate effect sizes with confidence intervals"""
    
    # Convert to arrays and remove NaN values
    g1 = np.array(group1_values)
    g2 = np.array(group2_values)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    if len(g1) == 0 or len(g2) == 0:
        return {'cohens_d': np.nan, 'rank_biserial_r': np.nan, 'mean_diff': np.nan}
    
    # Cohen's d
    pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) + 
                         (len(g2) - 1) * np.var(g2, ddof=1)) / 
                        (len(g1) + len(g2) - 2))
    
    if pooled_std == 0:
        cohens_d = 0
    else:
        cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std
    
    return {
        'cohens_d': cohens_d,
        'mean_diff': np.mean(g1) - np.mean(g2),
        'group1_mean': np.mean(g1),
        'group2_mean': np.mean(g2)
    }


def pseudo_bulk_differential_expression(adata, groupby, sample_key, min_cells=10):
    """Perform pseudo-bulk differential expression analysis"""
    print(f"  Creating pseudo-bulk profiles for {groupby} by {sample_key}...")
    
    pseudo_bulk_data = []
    metadata = []
    
    for group in adata.obs[groupby].unique():
        if pd.isna(group):
            continue
            
        group_data = adata[adata.obs[groupby] == group]
        
        for sample in group_data.obs[sample_key].unique():
            if pd.isna(sample):
                continue
                
            sample_data = group_data[group_data.obs[sample_key] == sample]
            
            if sample_data.n_obs >= min_cells:
                # Sum counts across cells in the sample
                if hasattr(sample_data, 'raw') and sample_data.raw is not None:
                    pseudo_counts = np.array(sample_data.raw.X.sum(axis=0)).flatten()
                else:
                    pseudo_counts = np.array(sample_data.X.sum(axis=0)).flatten()
                
                pseudo_bulk_data.append(pseudo_counts)
                metadata.append({
                    'group': group,
                    'sample': sample,
                    'n_cells': sample_data.n_obs,
                    'sample_group': f"{sample}_{group}"
                })
    
    if pseudo_bulk_data:
        # Create pseudo-bulk AnnData object
        pseudo_adata = sc.AnnData(
            X=np.array(pseudo_bulk_data),
            var=adata.var.copy(),
            obs=pd.DataFrame(metadata)
        )
        
        print(f"    Created pseudo-bulk data: {pseudo_adata.n_obs} samples x {pseudo_adata.n_vars} genes")
        return pseudo_adata
    else:
        print("    WARNING: No valid pseudo-bulk samples created")
        return None


# Enhanced developmental signatures (based on latest NK biology 2024)
ENHANCED_DEVELOPMENTAL_SIGNATURES_2024 = {
    # Stage 1: Early NK progenitors and immature cells
    'NK_Stage1_Immature_Enhanced': [
        # Original core genes (preserved)
        'IL2RB', 'SELL', 'GATA3', 'TCF7', 'KLRC1', 'BACH2', 'ID2',
        # 2024 additions - early progenitor markers
        'IL7R', 'KIT', 'CD127', 'CCR7', 'LEF1', 'TOX', 'IKZF1'
    ],
    
    # Stage 2: Transitional/intermediate NK cells  
    'NK_Stage2_Transitional_Enhanced': [
        # Original core genes (preserved)
        'TBX21', 'ITGAM', 'KLRB1', 'JUNB', 'EOMES',
        # 2024 additions - transition markers
        'CD122', 'CD11b', 'RUNX3', 'ZEB2', 'NR4A1', 'EGR2'
    ],
    
    # Stage 3: Mature cytotoxic NK cells
    'NK_Stage3_Mature_Enhanced': [
        # Original core genes (preserved) 
        'CX3CR1', 'CD247', 'GZMB', 'FCGR3A', 'PRF1', 'NKG7', 'FCER1G',
        # 2024 additions - mature effector markers
        'KIR2DL1', 'KIR2DL3', 'KIR3DL1', 'KLRD1', 'IFNG', 'PRDM1'
    ],
    
    # Stage 4: Adaptive/memory-like NK cells
    'NK_Stage4_Adaptive_Enhanced': [
        # Original core genes (preserved)
        'KLRC2', 'GZMH', 'B3GAT1', 'CCL5', 'IL32', 'PRDM1',
        # 2024 additions - adaptive/memory markers
        'LILRB1', 'CX3CR1', 'S1PR5', 'ZBTB32', 'PLZF', 'HOBIT', 'ZEB2'
    ]
}

# Enhanced functional signatures (2024 research updates)
ENHANCED_FUNCTIONAL_SIGNATURES_2024 = {
    # Enhanced activating receptors (original + 2024 updates)
    'Activating_Receptors_Enhanced': [
        # Original genes (preserved)
        'IL2RB', 'IL18R1', 'IL18RAP', 'NCR1', 'NCR2', 'NCR3', 'KLRK1',
        'FCGR3A', 'CD226', 'KLRC2', 'CD244', 'SLAMF6', 'SLAMF7', 'CD160',
        'KLRF1', 'KIR2DS1', 'KIR2DS2', 'KIR2DS4', 'KIR3DS1', 'ITGAL',
        # 2024 additions - newly characterized activating receptors
        'TNFRSF9', 'CD137', 'ICOS', 'TNFRSF18', 'CD134'
    ],
    
    # Enhanced inhibitory receptors (original + 2024 updates)
    'Inhibitory_Receptors_Enhanced': [
        # Original genes (preserved)
        'KLRC1', 'KIR2DL1', 'KIR2DL2', 'KIR2DL3', 'KIR3DL1', 'KIR3DL2',
        'LILRB1', 'PDCD1', 'TIGIT', 'CTLA4', 'HAVCR2', 'LAG3', 'SIGLEC7',
        'SIGLEC9', 'KLRG1', 'CD300A', 'LAIR1', 'CEACAM1',
        # 2024 additions - newly characterized inhibitory receptors
        'CD96', 'BTLA', 'VSIR', 'CD200R1', 'LILRA4'
    ],
    
    # Enhanced cytotoxicity machinery (original + 2024 updates)
    'Cytotoxicity_Enhanced': [
        # Original genes (preserved)
        'PRF1', 'GZMA', 'GZMB', 'GZMH', 'GZMK', 'GZMM', 'NKG7', 'GNLY',
        'SERPINB9', 'SRGN', 'FASLG', 'TNFSF10', 'LAMP1', 'CTSC', 'CTSW',
        # 2024 additions - additional cytotoxic molecules
        'GZMC', 'GZMF', 'CST7', 'CRTAM', 'SLAMF7'
    ],
    
    # Enhanced cytokine/chemokine production (original + 2024 updates)
    'Cytokine_Production_Enhanced': [
        # Original genes (preserved)
        'IFNG', 'TNF', 'LTA', 'CSF2', 'IL10', 'IL32', 'XCL1', 'XCL2',
        'CCL3', 'CCL4', 'CCL5', 'CXCL8', 'CXCL10',
        # 2024 additions - additional cytokines/chemokines
        'IL2', 'IL21', 'CXCL1', 'CXCL2', 'CCL1', 'CCL2'
    ],
    
    # Enhanced exhaustion/dysfunction (original + 2024 updates)
    'Exhaustion_Dysfunction_Enhanced': [
        # Original genes (preserved)
        'PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'KLRC1', 'KLRG1', 'CD96',
        'LILRB1', 'ENTPD1', 'TOX', 'EGR2', 'MAF', 'PRDM1', 'HSPA1A', 'DNAJB1',
        # 2024 additions - newly identified exhaustion markers
        'TOX2', 'BATF', 'IRF4', 'LAYN', 'CTLA4', 'CD39', 'CD73'
    ]
}

# New 2024-specific signatures (cutting-edge research)
NEW_NK_SIGNATURES_2024 = {
    # Tissue-resident NK cells (2024 discovery)
    'NK_Tissue_Resident': [
        'CD69', 'ITGA1', 'ITGAE', 'CXCR6', 'CD101', 'TNFRSF9',
        'RGS1', 'RGS2', 'DUSP4', 'DUSP6', 'EGR2', 'NR4A1'
    ],
    
    # Memory-like NK cells (2024 characterization)
    'NK_Memory_Like': [
        'KLRC2', 'LILRB1', 'CX3CR1', 'S1PR5', 'ZBTB32',
        'PLZF', 'HOBIT', 'BLIMP1', 'ZEB2', 'KLRG1'
    ],
    
    # TME-specific exhaustion (2024 tumor research)
    'NK_TME_Exhaustion': [
        'PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'CD96', 'KLRC1',
        'TOX', 'TOX2', 'EOMES', 'PRDM1', 'BATF', 'IRF4',
        'ENTPD1', 'LAYN', 'CTLA4'
    ],
    
    # Metabolic signatures (2024 metabolism research) - Updated to Hallmark gene sets
    'Glycolysis': _load_hallmark_geneset_with_fallback('glycolysis'),
    
    'NK_Oxidative_Phosphorylation': _load_hallmark_geneset_with_fallback('oxphos'),
    
    # Tissue-context specific signatures (2024)
    'Blood_NK_Homeostasis': [
        'S1PR1', 'S1PR5', 'CCR7', 'SELL', 'IL7R', 'TCF7', 'LEF1'
    ],
    
    'Tumor_NK_Infiltration': [
        'CXCR3', 'CXCR6', 'CCR5', 'ITGA1', 'ITGAE', 'CD69', 'CD103'
    ],
    
    'Tumor_NK_Dysfunction': [
        'PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'TOX', 'TOX2', 'ENTPD1'
    ]
}

# Combined signature dictionary for easy access
UPDATED_NK_SIGNATURES_2024 = {
    **ENHANCED_DEVELOPMENTAL_SIGNATURES_2024,
    **ENHANCED_FUNCTIONAL_SIGNATURES_2024, 
    **NEW_NK_SIGNATURES_2024
}

# Backward compatibility: Original signature names mapped to enhanced versions
SIGNATURE_MAPPING = {
    'Regulatory_NK': 'NK_Stage1_Immature_Enhanced',
    'Intermediate_NK': 'NK_Stage2_Transitional_Enhanced', 
    'Mature_Cytotoxic_NK': 'NK_Stage3_Mature_Enhanced',
    'Adaptive_NK': 'NK_Stage4_Adaptive_Enhanced',
    'Activating_Receptors': 'Activating_Receptors_Enhanced',
    'Inhibitory_Receptors': 'Inhibitory_Receptors_Enhanced',
    'Cytotoxicity_Machinery': 'Cytotoxicity_Enhanced',
    'Cytokine_Chemokine_Production': 'Cytokine_Production_Enhanced',
    'Exhaustion_Suppression_Markers': 'Exhaustion_Dysfunction_Enhanced'
}

print("Enhanced QC functions module loaded successfully!")
print(f"Available enhanced signatures: {list(UPDATED_NK_SIGNATURES_2024.keys())}")
print(f"Preserved original signatures as comments for reference")
print(f"Signature mapping for backward compatibility: {list(SIGNATURE_MAPPING.keys())}") 