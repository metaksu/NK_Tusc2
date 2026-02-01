# NK Cell Analysis Technical Documentation
## Comprehensive Methods Documentation for NK_analysis_main.py

**Version:** 3.0 (Enhanced with Modern Best Practices)  
**Date:** December 22, 2024  
**Analysis Script:** `NK_analysis_main.py`  

---

## Overview

This document provides detailed technical documentation of the Natural Killer (NK) cell analysis pipeline used to investigate TUSC2 expression across human NK cell subtypes in different biological contexts (healthy blood, normal tissue, and tumor tissue). The analysis integrates data from Rebuffet et al. (healthy blood NK cells) and Tang et al. (pan-cancer tissue NK cells) to provide comprehensive characterization of TUSC2's role in NK cell biology.

---

## Table of Contents

### Part 0: Global Setup & Definitions
- [0.1: Library Imports & Plotting Aesthetics](#01-library-imports--plotting-aesthetics)
- [0.2: File Paths & Output Directory Structure](#02-file-paths--output-directory-structure)
- [0.3: Core Biological & Analytical Definitions](#03-core-biological--analytical-definitions)

### Part 1: Data Ingestion, Preprocessing & Cohort Generation ✓ DOCUMENTED
### Part 2: Baseline Characterization of NK Subtypes
### Part 3: TUSC2 Analysis - A Layered Investigation  
### Part 4: Cross-Context Synthesis & Comparative Insights

---

## Summary of Documented Sections

**Completed Documentation:**
- **Part 0:** Complete technical setup, gene signatures, utility functions (41 sections documented)
- **Part 1:** Complete data ingestion pipeline (1.1-1.4 documented)

**Key Technical Parameters Captured:**
- All library dependencies and versions
- Complete gene signature definitions (84+ gene sets)
- Statistical thresholds and parameters
- Dimensionality reduction parameters (PCA: ARPACK, 15 PCs; UMAP: min_dist=0.3)
- File paths and directory structure
- Quality control metrics and thresholds

**Completed Documentation:**
- **Part 2:** ✅ COMPLETE - NK subtype characterization methods  
- **Part 3:** ✅ COMPLETE - TUSC2 analysis methodology
- **Part 4:** ✅ COMPLETE - Cross-context synthesis approaches

**Documentation Status:** 🎯 **COMPLETE** - All parts documented with technical precision

---

## Additional Technical Notes for Methods Section

### Reproducibility Parameters
- **Random Seed:** 42 (applied to numpy, scanpy, PCA, UMAP, neighbors)
- **Software Versions:** Python 3.13.2, Scanpy (latest), PyTorch 2.6.0+cu124
- **Hardware:** NVIDIA RTX 4070 GPU with CUDA 12.4 support
- **Memory:** 12.9 GB GPU memory available for deep learning acceleration

### Statistical Methods Summary
- **DEG Analysis:** Mann-Whitney U test, Bonferroni correction
- **Effect Sizes:** Cohen's d with interpretation (negligible/small/medium/large)
- **Gene Set Scoring:** Scanpy score_genes function
- **Multiple Testing:** Benjamini-Hochberg FDR correction
- **Thresholds:** adj_p < 0.05, |logFC| > 0.25, min_pct > 0.10

### Data Processing Pipeline Summary
1. **Quality Control:** Adaptive MT filtering, doublet detection, gene filtering
2. **Normalization:** TPM → log(TPM+1) for Rebuffet; total normalization for Tang
3. **Feature Selection:** Seurat HVG method, 1000 genes
4. **Dimensionality Reduction:** ARPACK PCA, 15 PCs, UMAP embedding
5. **Subtype Annotation:** Original subtype preservation per dataset

---

## Part 0: Global Setup & Definitions

### 0.1: Library Imports & Plotting Aesthetics

#### 0.1.1: Standard Library and Third-Party Imports

**Core Dependencies:**
- **Python Standard Libraries:**
  - `os`, `re`, `itertools`, `sys`, `warnings`: System operations and utilities
  - `pathlib.Path`: Cross-platform path handling

- **Data Science & Numerical Libraries:**
  - `pandas` (pd): Data manipulation and analysis
  - `numpy` (np): Numerical computing with array operations
  - `scipy.stats`: Statistical functions and tests
  - `scipy.io.mmread`: Matrix Market file reading for sparse matrices
  - `scipy.sparse.csr_matrix`: Compressed sparse row matrix format

- **Single-Cell Analysis Libraries:**
  - `scanpy` (sc): Single-cell RNA sequencing analysis toolkit

- **Plotting Libraries:**
  - `matplotlib.pyplot` (plt): Primary plotting interface
  - `seaborn` (sns): Statistical data visualization

- **Statistical Analysis Tools:**
  - `scikit_posthocs` (sp): Post-hoc statistical tests
  - `statsmodels.stats.multitest.multipletests`: Multiple testing correction

**Enhanced Utilities (Custom):**
- **Module:** `enhanced_qc_functions` (scripts/utilities)
  - `AdaptiveQualityControl`: Adaptive quality control methods
  - `UPDATED_NK_SIGNATURES_2024`: Modern NK cell gene signatures
  - `SIGNATURE_MAPPING`: Gene signature mapping utilities
  - `calculate_effect_sizes`: Effect size calculation functions
  - `pseudo_bulk_differential_expression`: Pseudo-bulk DEG analysis

**Optional Dependencies:**
- **gseapy:** Gene Set Enrichment Analysis (GSEA) functionality
  - Status: Conditionally imported with fallback
  - Used for: Pathway enrichment analysis
- **matplotlib_inline.backend_inline:** High-resolution inline plotting in Jupyter
  - Status: Conditionally imported with fallback

#### 0.1.2: Enhanced Global Plotting Aesthetics & Scanpy Settings

**Matplotlib Configuration:**
- **Backend Configuration:**
  - Format: 'retina' and 'png' for high-resolution displays
  - Implemented via `matplotlib_inline.backend_inline.set_matplotlib_formats()`

**Figure Export Settings:**
- **Default Format:** PNG
- **Resolution:** 300 DPI for publication-quality figures
- **Parameters:**
  ```python
  FIGURE_FORMAT = "png"
  FIGURE_DPI = 300
  ```

**Enhanced Plotting Parameters:**
```python
plt.rcParams.update({
    "figure.dpi": 100,                    # Display resolution
    "figure.facecolor": "white",          # White background
    "savefig.dpi": 300,                   # Publication DPI
    "savefig.format": "png",              # Default save format
    "savefig.transparent": False,         # Solid background
    "font.family": "Arial",               # Publication font
    "font.size": 11,                      # Base font size
    "axes.labelsize": 12,                 # Axis label size
    "axes.titlesize": 14,                 # Title size
    "xtick.labelsize": 10,                # X-tick label size
    "ytick.labelsize": 10,                # Y-tick label size
    "legend.fontsize": 10,                # Legend font size
    "axes.spines.top": False,             # Remove top spine
    "axes.spines.right": False            # Remove right spine
})
```

**Scanpy Configuration:**
- **Verbosity Level:** 2 (moderate output)
- **Auto-show:** Disabled (`sc.settings.autoshow = False`)
- **Reproducibility:** Random seed set to 42 for numpy and scanpy
  ```python
  RANDOM_SEED = 42
  sc.settings.seed = RANDOM_SEED
  np.random.seed(RANDOM_SEED)
  ```

### 0.2: File Paths & Output Directory Structure

#### 0.2.1: Input File Paths

**Primary Datasets:**
1. **Rebuffet et al. (2024) - Healthy Blood NK Cells:**
   - **File:** `PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad`
   - **Purpose:** Reference dataset for healthy blood NK cell subtypes
   - **Format:** AnnData HDF5 format

2. **Tang et al. - Pan-Cancer Tissue NK Cells:**
   - **File:** `comb_CD56_CD16_NK.h5ad`
   - **Purpose:** Combined dataset for normal and tumor tissue NK cells
   - **Format:** AnnData HDF5 format

#### 0.2.2: Master Output Directory Structure

**Hierarchical Organization:**
```
Combined_NK_TUSC2_Analysis_Output/
├── 0_Setup_Figs/                           # Setup and QC figures
├── 1_Processed_Anndata/                    # Processed AnnData objects
├── 2_Blood_NK_Char/                        # Blood NK characterization
├── 3_NormalTissue_NK_Char/                 # Normal tissue NK analysis
├── 4_TumorTissue_NK_Char/                  # Tumor tissue NK analysis
├── 5_TUSC2_Analysis/                       # TUSC2-focused analysis
│   ├── 5A_Broad_Context/                   # Cross-context TUSC2 analysis
│   └── 5B_Within_Context_Subtypes/         # Subtype-specific TUSC2
├── 6_Cross_Context_Synthesis/              # Comparative analysis
├── 7_Histology_Specific_TUSC2_Analysis/    # Histology-stratified analysis
├── common_figures/                         # Shared figure outputs
├── common_data_graphpad/                   # Data for statistical software
├── common_stat_results/                    # Statistical analysis results
└── common_temp_data/                       # Temporary data files
```

**Directory Creation:**
- All directories created programmatically using `os.makedirs(exist_ok=True)`
- Ensures reproducible output structure across different systems

#### 0.2.3: Scanpy Integration

**Figure Directory Configuration:**
- Scanpy default figure directory: `common_figures/`
- Individual analyses use specific subdirectories via `save_figure_and_data()`

### 0.3: Core Biological & Analytical Definitions

#### 0.3.1: Gene of Interest

**Primary Target:**
- **Gene Symbol:** TUSC2 (Tumor Suppressor Candidate 2)
- **Purpose:** Central gene for functional impact analysis across NK cell subtypes

#### 0.3.2: NK Subtype Definitions & Metadata Columns

**Rebuffet NK Subtypes (Blood NK Cells):**
Functionally ordered from immature/regulatory to mature/cytotoxic:
```python
REBUFFET_SUBTYPES_ORDERED = [
    "NK2",      # Immature/regulatory
    "NKint",    # Intermediate
    "NK1A",     # Mature cytotoxic (early)
    "NK1B",     # Mature cytotoxic (mid)
    "NK1C",     # Mature cytotoxic (late)
    "NK3"       # Adaptive/terminal
]
```

**Tang NK Subtypes (Tissue NK Cells):**
Developmentally ordered following maturation progression:
```python
TANG_SUBTYPES_ORDERED = [
    "CD56brightCD16lo-c5-CREM",        # Regulatory/immature
    "CD56brightCD16lo-c4-IL7R",        # Immature
    "CD56brightCD16lo-c2-IL7R-RGS1lo", # Transitional
    "CD56brightCD16lo-c3-CCL3",        # Inflammatory
    "CD56brightCD16lo-c1-GZMH",        # Cytotoxic bright
    "CD56brightCD16hi",                # Double-positive transitional
    "CD56dimCD16hi-c1-IL32",           # Cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1",         # Tissue-homing
    "CD56dimCD16hi-c3-ZNF90",          # Mature
    "CD56dimCD16hi-c4-NFKBIA",         # Activated
    "CD56dimCD16hi-c6-DNAJB1",         # Stress-response
    "CD56dimCD16hi-c7-NR4A3",          # Stimulated
    "CD56dimCD16hi-c8-KLRC2",          # Adaptive (NKG2C+)
    "CD56dimCD16hi-c5-MKI67"           # Proliferating
]
```

#### 0.3.2b: Tang Subtype Splits for Functional Analysis

**CD56+CD16- Subtypes (Regulatory/Immature NK cells):**
```python
TANG_CD56BRIGHT_SUBTYPES = [
    "CD56brightCD16lo-c5-CREM",        # Regulatory/immature
    "CD56brightCD16lo-c4-IL7R",        # Immature
    "CD56brightCD16lo-c2-IL7R-RGS1lo", # Transitional
    "CD56brightCD16lo-c3-CCL3",        # Inflammatory
    "CD56brightCD16lo-c1-GZMH",        # Cytotoxic bright
    "CD56brightCD16hi"                 # Double-positive transitional
]
```

**CD56-CD16+ Subtypes (Cytotoxic/Mature NK cells):**
```python
TANG_CD56DIM_SUBTYPES = [
    "CD56dimCD16hi-c1-IL32",           # Cytokine-producing
    "CD56dimCD16hi-c2-CX3CR1",         # Tissue-homing
    "CD56dimCD16hi-c3-ZNF90",          # Mature
    "CD56dimCD16hi-c4-NFKBIA",         # Activated
    "CD56dimCD16hi-c6-DNAJB1",         # Stress-response
    "CD56dimCD16hi-c7-NR4A3",          # Stimulated
    "CD56dimCD16hi-c8-KLRC2",          # Adaptive (NKG2C+)
    "CD56dimCD16hi-c5-MKI67"           # Proliferating
]
```

**Subset Organization:**
```python
TANG_SUBSETS = {
    "CD56posCD16neg": {
        "name": "CD56posCD16neg",
        "description": "CD56+CD16- (Regulatory/Immature NK cells)",
        "subtypes": TANG_CD56BRIGHT_SUBTYPES
    },
    "CD56negCD16pos": {
        "name": "CD56negCD16pos", 
        "description": "CD56-CD16+ (Cytotoxic/Mature NK cells)",
        "subtypes": TANG_CD56DIM_SUBTYPES
    }
}
```

**Column Name Standardization:**
- **Rebuffet Data:** `Rebuffet_Subtype` (standardized from original `ident`)
- **Tang Data:** `Tang_Subtype` (standardized subtype annotations)

**Helper Functions for Dataset Compatibility:**
1. **`get_subtype_column(adata_obj)`:** Auto-detects appropriate subtype column
2. **`get_subtype_categories(adata_obj)`:** Returns correct ordered categories
3. **`should_split_tang_subtypes(adata_obj)`:** Determines if Tang subset analysis needed
4. **`get_tang_subtype_subsets(adata_obj, context_name)`:** Generates CD56+CD16- and CD56-CD16+ subsets

#### 0.3.2a: Tang Combined Dataset Metadata Documentation

**Dataset Overview:**
- **Source:** `comb_CD56_CD16_NK.h5ad`
- **Cells:** 142,304 NK cells
- **Genes:** 13,493 genes
- **Important Note:** Tang 'Blood' context = peripheral blood (97.9% from cancer patients, 2.1% healthy)
- **Comparison:** Rebuffet blood NK cells = 100% healthy donor peripheral blood

**Tissue Contexts (meta_tissue_in_paper) - 4 categories:**
```python
TANG_TISSUE_VALUES = [
    "Blood",        # 67,202 cells (47.2%) - Peripheral blood NK
    "Tumor",        # 34,900 cells (24.5%) - Tumor-infiltrating NK
    "Normal",       # 22,792 cells (16.0%) - Normal adjacent tissue NK  
    "Other tissue"  # 17,410 cells (12.2%) - Other tissue contexts
]
```

**Major NK Cell Types (Majortype) - CD56/CD16 expression patterns:**
```python
TANG_MAJORTYPE_VALUES = [
    "CD56lowCD16high",  # 104,003 cells (73.1%) - Classic mature NK
    "CD56highCD16low",  # 35,283 cells (24.8%) - Immature/regulatory NK
    "CD56highCD16high"  # 3,018 cells (2.1%) - Transitional NK
]
```

**Fine-grained NK Subtypes (celltype) - 14 Tang subtypes:**
- Most abundant: CD56dimCD16hi-c3-ZNF90 (24,748 cells, 17.4%)
- Least abundant: CD56dimCD16hi-c5-MKI67 (990 cells, 0.7%)

**Cancer Types:** 25 cancer types + healthy donors
- Top 3: Breast Cancer (23.6%), Lung Cancer (17.1%), Melanoma (15.9%)

**Source Datasets:** 64 total datasets
- Major contributors: GSE169246 (21.1%), GSE139249 (15.3%), Tang original (8.4%)

**Sequencing Platforms:** 11 platforms
- Dominant: 10X (93.9% of cells)

#### 0.3.2b: Rebuffet Dataset Metadata Documentation

**Dataset Overview:**
- **Source:** `PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad`
- **Cells:** 35,578 NK cells
- **Genes:** 22,941 genes
- **Population:** 100% healthy donor peripheral blood NK cells
- **Normalization:** TPM (Transcripts Per Million)

**Primary NK Subtypes - 6 Rebuffet subtypes:**
```python
REBUFFET_SUBTYPES_ORDERED = [
    "NK2",   # 2,011 cells (5.7%) - Immature/regulatory
    "NKint", # 3,567 cells (10.0%) - Intermediate/transitional
    "NK1A",  # 6,098 cells (17.1%) - Early mature
    "NK1B",  # 3,959 cells (11.1%) - Intermediate mature
    "NK1C",  # 8,381 cells (23.6%) - Mature cytotoxic
    "NK3"    # 11,562 cells (32.5%) - Adaptive-like NK
]
```

**CMV/NKG2C Status:**
- NKG2Cneg: 42.4% (conventional NK cells)
- NKG2Cpos: 24.9% (adaptive-like NK cells)
- Unknown: 32.7%

**Quality Control Metrics:**
- **UMI count:** 401-13,707 (mean: 2,807)
- **Gene count:** 226-3,505 (mean: 1,093)
- **Mitochondrial %:** 0.00-22.34% (mean: 3.59%)
- **Ribosomal %:** 5.31-50.12% (mean: 24.67%)

### 0.3.3: Curated Developmental and Functional Gene Sets

#### 0.3.3a: Developmental Signatures (Maturation Trajectory)

**Stage 1 - Regulatory NK (NK2-like):**
```python
Maturation_NK_Regulatory = [
    "IL2RB",   # IL-2 receptor beta (CD122)
    "SELL",    # L-selectin (CD62L)
    "GATA3",   # GATA3 transcription factor
    "TCF7",    # TCF7/TCF-1 transcription factor
    "KLRC1",   # NKG2A inhibitory receptor
    "BACH2",   # BACH2 transcription factor
    "ID2",     # ID2 transcription factor
    "GZMK"     # Granzyme K
]
```

**Stage 2 - Intermediate NK (NKint-like):**
```python
Maturation_NK_Intermediate = [
    "TBX21",   # T-bet transcription factor
    "ITGAM",   # CD11b integrin
    "KLRB1",   # CD161/NK1.1
    "JUNB",    # JunB transcription factor
    "EOMES"    # Eomesodermin transcription factor
]
```

**Stage 3 - Mature Cytotoxic NK (NK1-like):**
```python
Maturation_NK_Mature_Cytotoxic = [
    "CX3CR1",  # CX3CR1 chemokine receptor
    "CD247",   # CD3-zeta signaling adaptor
    "GZMB",    # Granzyme B
    "FCGR3A",  # CD16a Fc receptor
    "PRF1",    # Perforin 1
    "NKG7",    # NKG7 granule protein
    "FCER1G"   # FcR-gamma adaptor
]
```

**Stage 4 - Adaptive NK (NK3-like):**
```python
Maturation_NK_Adaptive = [
    "KLRC2",   # NKG2C activating receptor
    "GZMH",    # Granzyme H
    "B3GAT1",  # CD57 biosynthesis enzyme
    "CCL5",    # RANTES chemokine
    "IL32",    # Interleukin-32
    "PRDM1"    # BLIMP-1 transcription factor
]
```

#### 0.3.3b: Functional Gene Signatures

**Core NK Functions:**
1. **Activating Receptors (20 genes):** IL2RB, IL18R1, NCR1-3, KLRK1, FCGR3A, etc.
2. **Inhibitory Receptors (18 genes):** KLRC1, KIR2DL1-3, PDCD1, TIGIT, LAG3, etc.
3. **Cytotoxicity Machinery (15 genes):** PRF1, GZMA/B/H/K/M, NKG7, GNLY, FASLG, etc.
4. **Cytokine/Chemokine Production (13 genes):** IFNG, TNF, IL10, IL32, CCL3-5, etc.
5. **Exhaustion/Suppression Markers (15 genes):** PDCD1, HAVCR2, LAG3, TIGIT, TOX, etc.

**Specialized Receptor Systems:**
1. **Neurotransmitter Receptors (15 categories):**
   - Acetylcholine Receptors (10 genes): CHRNA2-7, CHRNB2/4, CHRNE, CHRM3/5
   - Dopamine Receptors (5 genes): DRD1-5
   - Serotonin Receptors (4 genes): HTR1A, HTR2A/C, HTR7
   - GABA Receptors (21 genes): GABRA1-6, GABRB1-3, GABRG1-3, etc.
   - Additional systems: Glutamate, Histamine, Cannabinoid, Opioid, etc.

2. **Interleukin Downstream Pathways (7 pathways):**
   - IL15/IL2 Downstream (17 genes): STAT5A/B, AKT1, MTOR, BCL2, etc.
   - IL12 Downstream (11 genes): STAT4, TBX21, EOMES, IFNG, etc.
   - IL18 Downstream (12 genes): NFKB1, MYD88, TNF, XCL1, etc.
   - Additional pathways: IL21, IL10, IL27, IL33

**Metabolic Signatures:**
- **NK Glycolysis:** Hallmark Glycolysis gene set (if available) or custom 12-gene set
- **NK Oxidative Phosphorylation:** Hallmark OxPhos gene set (if available) or custom 12-gene set

#### 0.3.3c: Murine NK Developmental Orthologs

**Flow Cytometry-Informed Staging List (35 markers):**
Comprehensive set including surface markers (CD117, CD62L, CD16a, CD56, NKG2A/C), transcription factors (ID2, TCF7, GATA3, T-bet, Eomes), and functional markers (Granzyme B, Perforin, IFN-γ)

### 0.3.4: Analytical Parameters & Thresholds

**Gene Filtering Parameters:**
```python
GENE_PATTERNS_TO_EXCLUDE = [
    r"^RPS[0-9L]",      # Ribosomal protein small subunit
    r"^RPL[0-9L]",      # Ribosomal protein large subunit
    r"^MT-",            # Mitochondrial genes
    r"^HSP",            # Heat shock proteins
    r"^EEF[12]",        # Translation elongation factors
    # Additional patterns for housekeeping genes
]
```

**Statistical Thresholds:**
- **DEG Analysis:** LogFC ≥ 0.25, Adjusted p-value ≤ 0.05
- **Marker Selection:** Top 50 genes for reference, Top 30 for context-specific
- **Cell Frequency:** Minimum 10% for DEG inclusion
- **Gene Set Scoring:** Minimum 5 genes required
- **TUSC2 Binary Classification:** Expression threshold = 0.1

**TUSC2-Specific Parameters:**
```python
TUSC2_GENE_NAME = "TUSC2"
TUSC2_EXPRESSION_THRESHOLD_BINARY = 0.1
TUSC2_BINARY_GROUP_COL = "TUSC2_Binary_Group"
TUSC2_BINARY_CATEGORIES = [
    "TUSC2_Not_Expressed",
    "TUSC2_Expressed"
]
```

### 0.3.5: Color Palettes & Visualization

**Subtype Color Schemes:**
- **Rebuffet Subtypes:** 6-color palette (blues to browns)
- **Tang Subtypes:** 14-color comprehensive palette
  - CD56bright: Cooler colors (blues/greens/teals)
  - CD56dim: Warmer colors (reds/oranges/purples)
- **Combined Palette:** Merges both systems for compatibility

**Context Color Scheme:**
- Blood: #17becf (cyan)
- Normal Tissue: #7f7f7f (gray) 
- Tumor Tissue: #e377c2 (pink)

**TUSC2 Binary Groups:**
- Not Expressed: #aec7e8 (light blue)
- Expressed: #ff9896 (light red)

---

## 0.4: Utility Functions

### 0.4.1: Core Utility Functions

#### 0.4.1a: Figure and Data Saving

**`save_figure_and_data()` Function:**
- **Purpose:** Standardized saving of figures and associated data for GraphPad analysis
- **Parameters:**
  - `fig_object`: Matplotlib figure object
  - `data_df_for_graphpad`: DataFrame for statistical software export
  - `plot_basename`: Base filename (without extension)
  - `figure_subdir`, `data_subdir`: Output directories
  - `fig_format_override`, `fig_dpi_override`: Optional format/resolution overrides
  - `close_fig`: Whether to close figure after saving (default: True)
- **Output Format:** PNG at 300 DPI with tight bounding box
- **Data Export:** CSV format for GraphPad compatibility

#### 0.4.1b: Statistical Utilities

**`get_significance_stars()` Function:**
- **P-value to Stars Conversion:**
  - p < 0.0001: ****
  - p < 0.001: ***
  - p < 0.01: **
  - p < 0.05: *
  - p ≥ 0.05: ns (not significant)

**`enhanced_statistical_comparison()` Function:**
- **Purpose:** Comprehensive statistical comparison between two groups
- **Tests Performed:**
  - Mann-Whitney U test (primary)
  - Effect size calculation (Cohen's d)
  - Mean difference calculation
- **Output:** Dictionary with p-values, statistics, effect sizes, and interpretations
- **Enhanced Features:** 
  - Robust handling of missing values
  - Effect size interpretation (negligible, small, medium, large)
  - Fallback to basic statistics if enhanced methods unavailable

#### 0.4.1c: Layout Optimization

**`calculate_heatmap_layout()` Function:**
- **Purpose:** Dynamic figure sizing for heatmaps with long labels
- **Parameters:**
  - `min_width`, `min_height`: Minimum figure dimensions (inches)
  - `char_width_factor`: Character-to-inches conversion (default: 0.08)
  - `row_height_factor`: Row-to-inches conversion (default: 0.4)
  - `base_left_margin`: Base left margin fraction (default: 0.15)
- **Returns:** Optimized (width, height, left_margin) tuple
- **Algorithm:** Accounts for label length and content requirements

### 0.4.2: Gene and Marker Selection

#### 0.4.2a: Gene Filtering

**`is_gene_to_exclude_util()` Function:**
- **Purpose:** Identify genes to exclude from marker analysis
- **Exclusion Patterns:** Ribosomal proteins, mitochondrial genes, heat shock proteins, housekeeping genes
- **Implementation:** Regex-based pattern matching

**`map_gene_names()` Function:**
- **Purpose:** Map between CD protein names and official gene symbols
- **Mapping Examples:**
  - CD16 → FCGR3A
  - CD56 → NCAM1  
  - NKG2C → KLRC2
- **Features:** Availability checking, missing gene reporting

#### 0.4.2b: Intelligent Marker Selection

**`select_optimal_subtype_markers()` Function:**
- **Purpose:** Advanced DEG selection with overlap resolution
- **Key Features:**
  - Composite scoring: -log10(p_adj) × logFC × (1 + scanpy_score)
  - Conflict resolution for overlapping genes between subtypes
  - Guaranteed exact number of markers per subtype
  - Robust handling of insufficient DEGs

**Algorithm Steps:**
1. **Score Calculation:** Multi-factor composite scoring
2. **Conflict Detection:** Identify genes appearing in multiple subtypes
3. **Conflict Resolution:** Assign genes to highest-scoring subtype
4. **Gap Filling:** Add unique genes to reach target marker count

**Selection Criteria:**
- Adjusted p-value < 0.05
- Log fold change > 0.25
- Excludes housekeeping/ribosomal genes
- Score ratio threshold: 1.5x for conflict resolution

### 0.4.3: Filename and Utility Helpers

**`create_filename()` Function:**
- **Purpose:** Generate descriptive, standardized filenames
- **Components:** base_name, context_name, tusc2_group, gene_set_name, plot_type_suffix, version
- **Sanitization:** Removes spaces, parentheses, slashes; replaces with underscores
- **Optional Extension:** Flexible extension handling

---

## Part 1: Data Ingestion, Preprocessing & Cohort Generation

### 1.1: Healthy Blood NK Cohort (adata_blood)

#### 1.1.1: Data Loading and Initial Setup

**Data Source:**
- **File:** `PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad`
- **Dataset:** Rebuffet et al. healthy donor peripheral blood NK cells
- **Original Format:** TPM-normalized expression data
- **Cell Count:** 35,578 NK cells
- **Gene Count:** 22,941 genes

#### 1.1.2: Enhanced Preprocessing Pipeline

**`enhanced_preprocessing_pipeline()` Function:**

**Enhanced QC Framework (if available):**
1. **Adaptive Quality Control:** 
   - Sample-aware filtering using donor information
   - Batch effect consideration
2. **Quality Metrics Calculation:**
   - Mitochondrial gene percentage
   - Ribosomal gene percentage  
   - Hemoglobin gene percentage
3. **Adaptive MT Filtering:** Dynamic thresholds based on sample characteristics
4. **Enhanced Doublet Detection:** Consensus-based doublet identification
5. **Comprehensive Filtering:** MT outliers and doublets removed

**Standard Processing (fallback):**
- Basic preprocessing when enhanced QC unavailable
- Maintains data integrity and structure

**Data Transformation Steps:**
1. **Source Data Handling:**
   - TPM data from .X matrix (log-normalized data from .raw if available)
   - Log(TPM+1) transformation applied to TPM data
   - Original TPM stored in `layers['tpm']`

2. **Metadata Standardization:**
   - Convert original `ident` column to standardized `Rebuffet_Subtype`
   - Apply ordered categorical encoding
   - Filter cells with undefined subtypes

3. **Data Structure Setup:**
   - Set `.raw` attribute before gene filtering (preserves all genes)
   - Apply gene filtering to main object (min_cells=10)
   - Final shape tracking and validation

#### 1.1.3: Dimensionality Reduction Pipeline

**State Management:**
- **Critical Fix:** Automatic detection and correction of scaled data
- **Validation:** Ensures unscaled, log-normalized data for HVG selection
- **Safety Checks:** Restores from .raw if data appears pre-scaled

**Highly Variable Genes (HVG) Selection:**
- **Method:** Seurat flavor (robust, widely-used method)
- **Parameters:** 
  - n_top_genes=1000
  - subset=False (flags genes without filtering)
- **Purpose:** Focus downstream analysis on most informative genes

**Data Scaling:**
- **Method:** Z-score normalization (mean=0, variance=1)
- **Parameters:** max_value=10 (clips extreme outliers)
- **Purpose:** Prepare data for PCA

**Principal Component Analysis (PCA):**
- **SVD Solver:** ARPACK (efficient for sparse data)
- **Random State:** 42 (reproducibility)
- **Features:** Uses highly variable genes only
- **Validation:** PCA variance ratio plotting with log scale (50 components)

**Nearest Neighbor Graph Construction:**
- **Method:** Scanpy neighbors function
- **Parameters:**
  - n_pcs=15 (number of principal components)
  - random_state=42 (reproducibility)
- **Purpose:** Foundation for clustering and UMAP

**UMAP (Uniform Manifold Approximation and Projection):**
- **Parameters:**
  - random_state=42 (reproducibility)
  - min_dist=0.3 (minimum distance between points)
- **Purpose:** 2D visualization embedding
- **Output:** Stored in `adata.obsm['X_umap']`

**Quality Control Visualizations:**
- **PCA Variance Plot:** 50 components, log scale
- **Output:** High-resolution PNG (300 DPI)
- **File Naming:** Standardized with context and version tags

### 1.2: Tang et al. Combined NK Dataset (adata_tang_full)

#### 1.2.1: Data Loading and Validation

**Data Source:**
- **File:** `comb_CD56_CD16_NK.h5ad`
- **Dataset:** Tang et al. pan-cancer NK cell atlas
- **Cell Count:** 142,304 NK cells  
- **Gene Count:** 13,493 genes
- **Context Composition:**
  - Blood: 67,202 cells (47.2%)
  - Tumor: 34,900 cells (24.5%)
  - Normal tissue: 22,792 cells (16.0%)
  - Other tissue: 17,410 cells (12.2%)

**Metadata Validation:**
- **Primary tissue context:** `meta_tissue_in_paper` (4 categories)
- **Major NK types:** `Majortype` (3 CD56/CD16 combinations)
- **Fine subtypes:** `celltype` (14 Tang subtypes)
- **Cancer types:** `meta_histology` (25 cancer types + healthy)
- **Dataset sources:** 64 total datasets, primarily 10X platform (93.9%)

#### 1.2.2: Preprocessing and Quality Control

**Data Processing Steps:**
1. **Raw Data Preservation:** Store original counts in `layers['counts']`
2. **Cell Filtering:** Minimum 200 genes per cell
3. **Normalization Assessment:** 
   - Auto-detect if normalization needed (max expression > 50)
   - Apply total normalization (target_sum=1e4) and log1p if needed
   - Skip if already normalized (max < 50)
4. **Raw Attribute Setup:** Store log-normalized, unscaled data in `.raw`

**Quality Assurance:**
- Gene availability validation (TUSC2 presence confirmed)
- Expression range verification
- Metadata completeness checking

#### 1.2.3: Master Dataset Finalization

**State Management:**
- **Robustness Fix:** Reset .X from .raw to ensure unscaled data
- **HVG Flagging:** Seurat method for highly variable gene identification
- **Validation:** Expression range verification (unscaled: min~0, max~10)

### 1.3: Context-Specific Cohort Generation

#### 1.3.1-1.3.2: Context Separation

**adata_normal_tissue Creation:**
- **Filter:** `meta_tissue_in_paper == "Normal"`
- **Cells:** Subset of 22,792 normal tissue NK cells
- **Raw Propagation:** Automatic subsetting of .raw attribute

**adata_tumor_tissue Creation:**
- **Filter:** `meta_tissue_in_paper == "Tumor"`  
- **Cells:** Subset of 34,900 tumor-infiltrating NK cells
- **Raw Propagation:** Automatic subsetting of .raw attribute

#### 1.3.3-1.3.4: Subtype Annotation Strategy

**Original Subtype Preservation:**
- **Rebuffet Data:** Maintain 6 original subtypes (NK2, NKint, NK1A-C, NK3)
- **Tang Data:** Maintain 14 original subtypes (CD56bright/dim combinations)
- **No Cross-Dataset Reassignment:** Each dataset uses its native subtypes

**Tang Subtype Processing:**
1. **Column Standardization:** Convert `celltype` to `Tang_Subtype`
2. **Categorical Encoding:** Ordered categories based on developmental progression
3. **Cell Filtering:** Remove cells with undefined subtypes
4. **Validation:** Confirm subtype distribution and counts

#### 1.3.5: Standardized Dimensionality Reduction Pipeline

**`run_dim_reduction_pipeline()` Function:**

**State-Safe Processing:**
1. **Data Restoration:** Auto-detect and correct scaled data (.X min < -0.001)
2. **Safety Checks:** Restore from .raw if needed
3. **Validation:** Ensure unscaled, log-normalized input

**Processing Steps:**
1. **Gene Filtering:** Remove genes expressed in <10 cells
2. **HVG Selection:**
   - Method: Seurat flavor
   - n_top_genes: 1000 (default)
   - subset: False (flag only, don't subset)
3. **Data Scaling:** Z-score normalization, max_value=10
4. **PCA:**
   - SVD solver: ARPACK
   - mask_var: 'highly_variable' (modern approach)
   - random_state: 42

**Enhanced PC Selection (Optional):**
- **Data-Driven Method:** Automatic PC number optimization
- **Criteria:** 
  - 80% variance threshold
  - Elbow method (differential analysis)
  - Conservative median selection
- **Constraints:** 10-50 PC range
- **Fallback:** 15 PCs if specified

**Graph Construction and UMAP:**
- **Neighbors:** k-NN graph using optimal PCs
- **UMAP Parameters:**
  - random_state: 42
  - min_dist: 0.3
  - Output: 2D embedding in `obsm['X_umap']`

**Pipeline Execution:**
- Applied to: adata_normal_tissue and adata_tumor_tissue
- Results: Ready-to-analyze objects with complete dimensionality reduction

#### 1.3.6: Data Persistence

**Processed Object Storage:**
- **Directory:** `1_Processed_Anndata/`
- **Format:** HDF5 (.h5ad) with gzip compression
- **Files:**
  - `adata_blood_processed.h5ad`
  - `adata_normal_tissue_processed.h5ad`
  - `adata_tumor_tissue_processed.h5ad`

**Validation Output:**
- Shape verification (main and .raw objects)
- Metadata column enumeration
- Subtype distribution confirmation

### 1.4: Dynamic Developmental Signature Generation

**Purpose:** Generate data-driven developmental signatures from actual Rebuffet blood NK DEGs rather than literature-curated gene lists

---

## Part 2: Baseline Characterization of NK Subtypes within Each Context

**Overview:** This section performs standardized characterization analyses for each biological context (Blood, Normal Tissue, Tumor Tissue) using a consistent analytical framework.

**Context Processing Structure:**
```python
cohorts_for_characterization = [
    ("Blood", adata_blood, OUTPUT_SUBDIRS["blood_nk_char"]),
    ("NormalTissue", adata_normal_tissue, OUTPUT_SUBDIRS["normal_tissue_nk_char"]),
    ("TumorTissue", adata_tumor_tissue, OUTPUT_SUBDIRS["tumor_tissue_nk_char"])
]
```

### 2.1: Composition and Visual Overview

#### 2.1.1: Subtype Composition Analysis

**Cell Filtering Strategy:**
- **Rebuffet Data (Blood):** Exclude "Unassigned" cells for composition analysis
- **Tang Data (Tissue):** Include all cells (no "Unassigned" category exists)

**Visualization Parameters:**
- **Figure Size:** 8×6 inches
- **Plot Type:** Seaborn barplot with custom color palette
- **Color Mapping:** `COMBINED_SUBTYPE_COLOR_PALETTE`
- **X-axis:** Subtype categories with 45° rotation
- **Y-axis:** Proportion of assigned cells (%)

**Statistical Outputs:**
- Cell counts per subtype
- Percentage proportions 
- Export format: CSV with columns `[Subtype, Cell_Count, Proportion_Pct]`

#### 2.1.2: UMAP Visualization by Subtypes

**Technical Requirements:**
- **Prerequisite:** `X_umap` coordinates in `adata.obsm`
- **Plot Parameters:**
  - Figure size: 10×7 inches
  - Legend: Right margin placement (fontsize=8)
  - Color palette: `COMBINED_SUBTYPE_COLOR_PALETTE`
- **Layout Adjustment:** `plt.subplots_adjust(right=0.75)` for legend space

**Data Export:**
- UMAP coordinates: `[UMAP1, UMAP2]` columns
- Cell metadata: Includes subtype assignments
- Index: Cell barcodes/IDs

#### 2.1.3: Cross-tabulations and Validation (Tang Data Only)

**Cross-tabulation with Original Tang Majortype:**
- **Column Used:** `METADATA_MAJORTYPE_COLUMN_GSE212890`
- **Method:** `pd.crosstab()` with `dropna=False`
- **Normalization:** Column-wise percentages for validation heatmap

**Heatmap Visualization:**
- **Colormap:** "viridis" 
- **Annotations:** Percentage values (format: ".1f")
- **Colorbar Label:** "% of Original Tang Majortype"
- **Layout:** 10×8 inch figure with linewidths=0.5

**Histology Distribution Analysis:**
- **Column Used:** `METADATA_HISTOLOGY_COLUMN_GSE212890`
- **Plot Type:** Stacked bar chart showing subtype proportions
- **Color Mapping:** Context-appropriate subtype colors
- **Dynamic Sizing:** Width based on number of histology categories

### 2.2: Transcriptional Definition (Context-Specific Markers)

#### 2.2.1: Differential Expression Analysis

**Statistical Method:**
- **Algorithm:** Wilcoxon rank-sum test (`method="wilcoxon"`)
- **Multiple Testing Correction:** Benjamini-Hochberg (`corr_method="benjamini-hochberg"`)
- **Data Source:** Uses `use_raw=True` for unscaled, log-normalized data
- **Additional Metrics:** Includes percentage of cells expressing (`pts=True`)

**Gene Selection Parameters:**
- **Number of Genes:** `TOP_N_MARKERS_CONTEXT + 150` extra for filtering
- **Adjusted P-value Threshold:** `ADJ_PVAL_THRESHOLD_DEG` (typically 0.05)
- **Log Fold Change Threshold:** `LOGFC_THRESHOLD_DEG` (typically 0.25)
- **Gene Exclusion:** Applied via `is_gene_to_exclude_util()` function

**Quality Control:**
- **Cell Filtering:** Minimum 2 subtypes required for analysis
- **Pattern Exclusion:** Filters ribosomal, mitochondrial, and other non-informative genes
- **Ranking Metric:** Sorted by "scores" (combined statistical measure)

#### 2.2.2: Marker Visualization and Export

**Dot Plot Generation:**
- **Marker Selection:** Top 4 unique markers per subtype
- **Standardization:** `standard_scale="var"` for cross-gene comparison
- **Data Source:** `use_raw=True` for consistent scaling
- **Deduplication:** Ensures unique markers across subtypes

**File Export Structure:**
```
context_markers/
├── figures/           # Dot plots (PNG, 300 DPI)
├── data_for_graphpad/ # Statistical analysis exports  
├── marker_lists/      # Text files with gene lists
└── stat_results_python/ # Python analysis outputs
```

### 2.3: Developmental and Functional Signature Profiling

#### 2.3.1: Signature Score Calculation

**Core Function:** `generate_signature_heatmap()`

**Gene Set Mapping:**
- **Function:** `map_gene_names()` matches gene symbols to available data
- **Minimum Threshold:** `MIN_GENES_FOR_SCORING` genes required
- **Missing Genes:** Sets to `np.nan` if insufficient genes available

**Scoring Methodology:**
- **Function:** `sc.tl.score_genes()`
- **Parameters:**
  - `use_raw=True` (log-normalized, unscaled data)
  - `random_state=RANDOM_SEED` (reproducibility)
- **Score Naming:** `{set_name}_Score` format

#### 2.3.2: Heatmap Visualization

**Layout Optimization:**
- **Function:** `calculate_heatmap_layout()` 
- **Parameters:**
  - `min_width=8, min_height=5`
  - `char_width_factor=0.1` (signature name length)
  - `row_height_factor=0.5` (row spacing)
- **Dynamic Sizing:** Adapts to signature name lengths

**Heatmap Specifications:**
- **Colormap:** "icefire" (diverging, centered at 0)
- **Annotations:** Mean scores with 3 decimal places (`fmt=".3f"`)
- **Data Aggregation:** Mean scores per subtype via `groupby()`
- **Label Cleaning:** Removes "_Score" suffix and formats for readability

**Tang Subtype Subset Analysis:**
- **Automatic Detection:** `should_split_tang_subtypes()` determines if splitting needed
- **Subset Generation:** `get_tang_subtype_subsets()` creates CD56+CD16- and CD56-CD16+ views
- **Parallel Processing:** Generates separate analyses for each subset
- **File Organization:** Subset-specific directories and filenames

#### 2.3.3: Blueprint Signature Analysis

**Core Function:** `create_blueprint_dotplot()`

**Blueprint Visualization:**
- **Plot Type:** Multi-signature dot plot with category grouping
- **Gene Selection:** Representative genes from each signature
- **Standardization:** `standard_scale="var"` for cross-gene comparison
- **Visual Parameters:**
  - Dynamic figure sizing based on gene count
  - Red color mapping ("Reds" colormap)
  - Dot size range: 0-0.8 max scale
  - Category boundaries marked with dashed lines

**Signature Categories:**
- **Developmental Blueprint:** Regulatory, Intermediate, Mature Cytotoxic, Adaptive NK
- **Functional Blueprint:** Activating/Inhibitory Receptors, Cytotoxicity, Cytokines, Exhaustion

### 2.4: Developmental Marker Profiling (Murine Orthologs)

**Marker Source:** `MURINE_DEV_MARKER_ORTHOLOGS` gene list
**Purpose:** Cross-species validation of developmental markers
**Analysis:** Expression profiling across NK subtypes using orthologous genes

### 2.5: Blueprint Dotplot Analysis

**Integration Function:** Comprehensive visualization combining developmental and functional signatures
**Output Location:** `6_Cross_Context_Synthesis/` directory
**Purpose:** Master blueprint plots for publication figures

### 2.6: Synthesis Analysis - Top Subtype-Defining Markers

**Enhanced Marker Selection:**
- **Function:** `select_optimal_subtype_markers()`
- **Parameters:**
  - `max_markers_per_subtype=4`
  - `pval_threshold=0.05`
  - `logfc_threshold=0.25`
- **Smart Deduplication:** Prevents marker overlap across subtypes

**Visual Grouping:**
- **Group Boundaries:** Dashed lines separating subtype-specific markers
- **Labels:** Subtype names positioned above gene groups
- **Layout:** Optimized for clarity with dynamic spacing

**Data Export:**
- Mean expression values per group
- Fraction of expressing cells
- GraphPad-compatible format

---

## Part 3: TUSC2 Analysis - A Layered Approach

**Overview:** This section implements a comprehensive, multi-layered investigation of TUSC2 expression and functional impact across NK cell subtypes and biological contexts.

**Analysis Structure:**
1. **Broad Context Analysis:** TUSC2 expression patterns across all contexts
2. **Within-Context Subtype Analysis:** Subtype-specific TUSC2 expression 
3. **Functional Impact Analysis:** TUSC2-associated functional changes
4. **Cross-Context Synthesis:** Comparative analysis across biological contexts

### 3.1: TUSC2 Expression Classification

**Binary Classification System:**
- **Gene of Interest:** TUSC2 (Tumor Suppressor Candidate 2)
- **Expression Threshold:** `TUSC2_EXPRESSION_THRESHOLD_BINARY` 
- **Binary Categories:**
  - `TUSC2_Not_Expressed` (#aec7e8 - light blue)
  - `TUSC2_Expressed` (#ff9896 - light red)

**Data Extraction Protocol:**
- **Source:** `.raw.X` matrix (unscaled, log-normalized data)
- **Storage:** Expression values in `{TUSC2_GENE_NAME}_Expression_Raw` column
- **Binary Groups:** Stored in `TUSC2_BINARY_GROUP_COL` as ordered categorical

### 3.2: Layer 1 - Broad Context Analysis

#### 3.2.1: Cross-Context Expression Comparison

**Statistical Method:**
- **Test:** Mann-Whitney U test (`stats.mannwhitneyu`)
- **Comparison:** Normal Tissue vs. Tumor Tissue TUSC2 expression
- **Alternative:** Two-sided test

**Visualization:**
- **Plot Type:** Violin plot with quartile indicators
- **Color Palette:** `CONTEXT_COLOR_PALETTE`
- **Statistical Annotation:** Significance stars positioned above comparison brackets
- **Layout:** Dynamic y-axis scaling (1.25× max expression)

#### 3.2.2: Binary Group Proportions

**Analysis Method:**
- **Function:** `groupby().value_counts(normalize=True)` 
- **Scaling:** Multiply by 100 for percentage values
- **Visualization:** Seaborn catplot with custom color mapping

**Plot Specifications:**
- **Type:** Grouped bar chart (`kind="bar"`)
- **Colors:** `TUSC2_BINARY_GROUP_COLORS`
- **Dimensions:** Height=6, aspect=1.2
- **Rotation:** 45° x-axis labels

### 3.3: Layer 2 - Within-Context Subtype Analysis

**Cell Filtering Strategy:**
- **Inclusion Criteria:** Cells with assigned subtypes only
- **Exclusion:** "Unassigned" category filtered out
- **Minimum Threshold:** At least 3 cells per group for statistical tests

**Subtype-Specific Analysis:**
- **Expression Patterns:** TUSC2 expression across NK subtypes within each context
- **Statistical Comparisons:** Between subtypes within contexts
- **Visualization:** Violin plots and categorical bar charts

### 3.4: Layer 3 - Functional Impact Analysis

#### 3.4.1: Signature Score Calculation

**On-Demand Scoring:**
- **Gene Sets:** `ALL_FUNCTIONAL_GENE_SETS` (developmental + functional signatures)
- **Method:** `sc.tl.score_genes()` with `use_raw=True`
- **Minimum Genes:** `MIN_GENES_FOR_SCORING` threshold
- **Missing Data:** Set to `np.nan` if insufficient genes

#### 3.4.2: TUSC2 Impact Quantification

**Statistical Framework:**
- **Test:** Mann-Whitney U test for each functional signature
- **Groups:** TUSC2 expressing vs. non-expressing cells
- **Multiple Testing:** FDR correction using Benjamini-Hochberg method
- **Effect Size:** Mean score difference calculation

**Quality Control:**
- **Minimum Sample Size:** 3 cells per group for valid comparisons
- **Data Validation:** NA handling and empty result checking

#### 3.4.3: Impact Visualization

**Heatmap Specifications:**
- **Colormap:** "RdBu_r" (diverging red-blue reversed)
- **Centering:** Centered at 0 for fold-change interpretation
- **Scaling:** Symmetric limits based on maximum absolute difference
- **Annotations:** Combined effect size and significance stars

**Data Export:**
- **Statistical Results:** P-values, Q-values, effect sizes, sample sizes
- **Format:** CSV with comprehensive statistical metadata

### 3.5: Layer 5 - Differential Expression Analysis

#### 3.5.1: DEG Analysis Framework

**Prerequisites:**
- **Minimum Cells:** 3 cells minimum in TUSC2-expressing group
- **Column Validation:** `TUSC2_BINARY_GROUP_COL` presence check
- **Data Source:** Uses `.raw` attribute for unscaled data

**Statistical Method:**
- **Algorithm:** Wilcoxon rank-sum test
- **Groups:** TUSC2 expressing vs. non-expressing cells
- **Correction:** Benjamini-Hochberg FDR correction
- **Output:** Comprehensive DEG tables with effect sizes

#### 3.5.2: Volcano Plot Generation

**Technical Dependencies:**
- **Label Optimization:** `adjustText` library (optional)
- **Fallback:** Manual positioning if adjustText unavailable
- **Gene Highlighting:** Significant genes with effect size thresholds

**Plot Parameters:**
- **X-axis:** Log fold change (TUSC2 expressing / non-expressing)
- **Y-axis:** -log10(adjusted p-value)
- **Significance Thresholds:** Vertical and horizontal lines for cutoffs
- **Point Colors:** Categorical coloring based on significance and direction

---

## Part 4: Cross-Context Synthesis & Comparative Insights

**Overview:** This section integrates findings across all biological contexts to provide comprehensive insights into TUSC2's role in NK cell biology and subtype-specific functions.

**Analysis Framework:** Cross-context comparison with statistical rigor and multi-dimensional visualization

### 4.1: Cross-Context TUSC2 Expression Patterns

**Comparative Analysis:**
- **Contexts:** Blood, Normal Tissue, Tumor Tissue
- **Statistical Method:** Context-specific comparisons using Mann-Whitney U tests
- **Visualization:** Integrated violin plots and proportion analyses

### 4.2: Functional Signature Integration

**Multi-Context Scoring:**
- **Signature Application:** All functional gene sets applied across contexts
- **Comparison Framework:** TUSC2 expressing vs. non-expressing cells per context
- **Statistical Correction:** FDR correction applied across all comparisons

### 4.3: TUSC2 Impact on NK Subtype Programs

#### 4.3.1: Subtype-Specific Analysis

**Analysis Strategy:**
- **Program Evaluation:** NK subtype signature scores calculated per context
- **TUSC2 Stratification:** Comparison within each subtype program
- **Statistical Framework:** Mann-Whitney U test with FDR correction

**Key Methodology:**
- **Minimum Sample Size:** 3 cells per group for valid comparisons
- **Score Calculation:** Mean difference (TUSC2+ vs. TUSC2-)
- **Multiple Testing:** Benjamini-Hochberg FDR correction

#### 4.3.2: Summary Visualization

**Heatmap Specifications:**
- **Pivot Structure:** Subtype Programs (rows) × Contexts (columns)
- **Color Mapping:** "RdBu_r" diverging colormap centered at 0
- **Annotations:** Combined effect size and significance indicators
- **Ordering:** `REBUFFET_SUBTYPES_ORDERED` for developmental progression

**Statistical Integration:**
- **Effect Size:** Mean score differences with precision formatting
- **Significance:** FDR-corrected p-values with star notation
- **Layout:** Dynamic sizing with proper label spacing

### 4.4: Developmental State Impact Analysis

#### 4.4.1: Developmental Marker Evaluation

**Marker Source:** `MURINE_DEV_MARKER_ORTHOLOGS` gene list
**Developmental Ordering:**
```
KIT → ID2 → GATA3 → TCF7 → IL2RB → SELL → CD27 → KLRC1 → 
NCAM1 → TBX21 → EOMES → ITGAM → FCGR3A → GZMB → PRF1 → KLRG1 → B3GAT1
```

**Analysis Methodology:**
- **Expression Extraction:** Direct gene expression from `.raw.X`
- **Group Comparison:** TUSC2+ vs. TUSC2- cells per marker
- **Statistical Test:** Mann-Whitney U test for each marker-context combination

#### 4.4.2: Developmental Impact Visualization

**Heatmap Design:**
- **Rows:** Developmental markers (ordered by progression)
- **Columns:** Biological contexts
- **Values:** Mean expression differences (TUSC2+ - TUSC2-)
- **Significance:** FDR-corrected q-values overlaid as stars

**Layout Optimization:**
- **Function:** `calculate_heatmap_layout()` for dynamic sizing
- **Parameters:** Optimized for long gene names and readability
- **Margins:** Calculated based on content length

---

## Final Analysis Summary

**Complete Documentation Status:**
- **Part 0:** ✅ Global setup, gene signatures, utility functions (41+ sections)
- **Part 1:** ✅ Data ingestion, preprocessing, cohort generation (4 sections)
- **Part 2:** ✅ NK subtype characterization (6 sections with Tang subset handling)
- **Part 3:** ✅ TUSC2 layered analysis (5 layers of investigation)
- **Part 4:** ✅ Cross-context synthesis and comparative insights (4 sections)

**Key Technical Achievements:**
1. **Comprehensive Gene Signature Library:** 84+ curated gene sets across multiple functional categories
2. **Robust Statistical Framework:** Multiple testing correction, effect size calculation, significance testing
3. **Advanced Visualization:** Dynamic layout optimization, publication-quality figures
4. **Cross-Dataset Integration:** Seamless handling of Rebuffet and Tang datasets with different annotation systems
5. **Tang Subtype Subset Analysis:** Automatic CD56+CD16- and CD56-CD16+ subset generation
6. **Reproducible Pipeline:** Complete parameter documentation and random seed control

**Methods Paper Readiness:**
- All statistical methods documented with specific parameters
- Software versions and dependencies captured
- Hardware specifications included for reproducibility
- Complete file structure and data organization documented
- Quality control and validation steps detailed
- Cross-context comparison methodology established

**Analysis Version:** NK Cell Transcriptomics and TUSC2 Function Analysis v3.2 (Cleaned)
**Documentation Date:** December 2024
**Total Documented Functions:** 25+ core analysis functions
**Total Analysis Sections:** 20+ major analytical workflows 