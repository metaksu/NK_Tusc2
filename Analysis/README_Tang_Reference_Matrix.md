# Tang NK Reference Matrix Generator for CIBERSORTx

## Overview

This script recreates the processing file that generates the `Tang_reference_matrix_700cells_per_phenotype.txt` file for CIBERSORTx deconvolution analysis. The script samples NK cells from **ALL tissue contexts** (tumor, normal, blood, etc.) to create a comprehensive reference matrix.

## Key Features

1. **Comprehensive Tissue Representation**: Uses ALL Tang dataset contexts (not just tumor)
2. **Equal Subtype Sampling**: Randomly samples up to 700 cells per NK subtype
3. **CIBERSORTx Compliance**: Outputs in proper format with correct normalization
4. **Detailed Logging**: Comprehensive analysis and validation reporting
5. **Main Project Integration**: Uses exact loading patterns from the main NK analysis scripts

## Script Workflow

### Step 1: Data Loading & Validation
- Loads Tang NK dataset using exact patterns from main project
- Validates metadata columns and data integrity
- Reports dataset overview and quality metrics

### Step 2: Data Preprocessing
- Stores raw counts for preservation
- Applies minimal cell filtering (>200 genes per cell)
- Handles normalization based on data state
- Sets .raw attribute for gene expression analysis

### Step 3: Tissue Distribution Analysis
- Analyzes tissue context distribution across dataset
- Reports NK subtype availability across all tissues
- Creates tissue × subtype crosstab for comprehensive overview

### Step 4: Randomized Subtype Sampling
- Samples up to 700 cells per NK subtype from ALL tissue contexts
- Uses random seed (42) for reproducibility
- Tracks tissue context representation in final sample
- Reports final cell counts and tissue breakdown

### Step 5: CIBERSORTx Format Preparation
- Extracts expression data in proper format (genes × cells)
- Handles log-space to linear conversion if needed
- Prepares cell labels with subtype annotations

### Step 6: Reference Matrix Creation
- Creates final matrix with genes as rows, cells as columns
- Adds gene names in first column (CIBERSORTx requirement)
- Validates for duplicate gene names
- Saves as tab-separated file

### Step 7: Analysis Summary
- Generates comprehensive summary report
- Documents configuration parameters
- Reports final cell distribution by subtype
- Lists CIBERSORTx format compliance checks

## Configuration

### Key Parameters
```python
TARGET_CELLS_PER_PHENOTYPE = 700    # Target cells per NK subtype
MIN_CELLS_PER_PHENOTYPE = 5         # Minimum cells required for inclusion
RANDOM_SEED = 42                     # For reproducibility
MIN_GENES_PER_CELL = 200             # Minimum genes per cell for filtering
```

### Input/Output
```python
INPUT_FILE = "data/processed/comb_CD56_CD16_NK.h5ad"
OUTPUT_DIR = "outputs/signature_matrices/CIBERSORTx_Input_Files"
OUTPUT_FILE = "Tang_reference_matrix_700cells_per_phenotype.txt"
SUMMARY_FILE = "Tang_reference_matrix_generation_summary.txt"
```

## Tang NK Subtypes (13 Total)

### CD56bright Subtypes (6)
- CD56brightCD16lo-c1-GZMH (GZMH+ cytotoxic)
- CD56brightCD16lo-c2-IL7R-RGS1lo (IL7R+ immature)
- CD56brightCD16lo-c3-CCL3 (CCL3+ chemokine)
- CD56brightCD16lo-c4-AREG (AREG+ tissue repair)
- CD56brightCD16lo-c5-CXCR4 (CXCR4+ trafficking)
- CD56brightCD16hi (Double-positive transitional)

### CD56dim Subtypes (7)
- CD56dimCD16hi-c1-IL32 (Cytokine-producing)
- CD56dimCD16hi-c2-CX3CR1 (Tissue-homing)
- CD56dimCD16hi-c3-ZNF90 (Mature)
- CD56dimCD16hi-c4-NFKBIA (Activated)
- CD56dimCD16hi-c5-MKI67 (Proliferating)
- CD56dimCD16hi-c6-DNAJB1 (Stress-response)
- CD56dimCD16hi-c7-NR4A3 (Stimulated)
- CD56dimCD16hi-c8-KLRC2 (Adaptive NKG2C+)

## CIBERSORTx Format Requirements

✅ **Genes in column 1; cell labels in row 1**
✅ **Data in non-log space (TPM recommended)**
✅ **Tab-separated format**
✅ **Gene symbol redundancy handled**
✅ **Maximum expression >50 (treated as linear)**

## Usage

### Basic Execution
```bash
python create_tang_reference_matrix.py
```

### Expected Output Structure
```
outputs/signature_matrices/CIBERSORTx_Input_Files/
├── Tang_reference_matrix_700cells_per_phenotype.txt      # Main output
└── Tang_reference_matrix_generation_summary.txt          # Analysis summary
```

### Expected File Size
- **Reference Matrix**: ~450-500 MB (13,493 genes × ~7,900 cells)
- **Summary File**: ~5-10 KB (text report)

## Output Validation

The script performs comprehensive validation:

1. **Format Compliance**: Ensures CIBERSORTx requirements met
2. **Cell Distribution**: Reports actual vs target cells per subtype
3. **Tissue Representation**: Shows tissue context breakdown
4. **Gene Coverage**: Validates gene count and uniqueness
5. **Expression Range**: Confirms proper normalization space

## Integration with Main Project

This script uses the **exact same** loading patterns and constants as the main NK analysis scripts:

- **File Paths**: Matches main project structure
- **Column Names**: Uses identical metadata column definitions
- **Preprocessing**: Follows same normalization workflow
- **Validation**: Mirrors main project validation logic

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure `data/processed/comb_CD56_CD16_NK.h5ad` exists
2. **Memory Issues**: Script handles large files efficiently with sparse matrices
3. **Insufficient Cells**: Some subtypes may have <700 cells (documented in summary)

### Expected Limitations

Based on previous analysis, 4 subtypes have fewer than 700 cells:
- CD56brightCD16lo-c1-GZMH: ~372 cells
- CD56brightCD16lo-c2-IL7R-RGS1lo: ~362 cells  
- CD56dimCD16hi-c1-IL32: ~570 cells
- CD56dimCD16hi-c5-MKI67: ~317 cells

## Technical Notes

- **Memory Efficiency**: Uses sparse matrices and memory-efficient processing
- **Reproducibility**: Fixed random seed ensures consistent results
- **Logging**: Comprehensive progress reporting and error handling
- **Compatibility**: Works with current scanpy/pandas versions

## Purpose

This reference matrix represents single-cell RNA-seq expression profiles from NK cells across all tissue contexts (tumor, normal, blood, etc.), suitable for CIBERSORTx signature matrix generation and subsequent deconvolution analysis of bulk RNA-seq data. 