# NK Subtype Signature Matrix Analysis Project

## Project Overview
This project contains a robust pipeline for creating signature matrices for NK cell subtypes using Rebuffet et al. 2024 scRNA-seq data. The signature matrices are designed for CIBERSORTx deconvolution analysis.

## Project Structure

```
Analysis/
├── data/
│   ├── raw/                    # Original data files
│   │   ├── PBMC_V2_VF1_AllGenes_NewNames.rds
│   │   ├── rebuffet_*.csv      # Exported Rebuffet data
│   │   └── rebuffet_*.mtx
│   └── processed/              # Processed data files
│       ├── PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad
│       └── PBMC_V2_VF1_AllGenes_NewNames_linear_fixed.h5ad
├── scripts/
│   ├── signature_matrix/       # Signature matrix generation scripts
│   │   └── robust_signature_matrix_pipeline.py
│   └── data_processing/        # Data processing and conversion scripts
│       ├── convert_to_tpm_corrected.py
│       ├── validate_tpm.py
│       ├── fix_anndata_creation.py
│       └── load_exported_data.py
├── outputs/
│   ├── signature_matrices/     # Generated signature matrices
│   │   ├── Robust_Signature_Matrix_Output_TPM/
│   │   └── Robust_Signature_Matrix_Output/
│   └── figures/                # Generated figures and plots
├── archive/                    # Archived old files
│   ├── old_notebooks/          # Previous notebook versions
│   └── old_scripts/            # Previous script versions
├── notebooks/                  # Current working notebooks
├── CIBERSORTx_Signature_Matrix_Output_v2/  # Previous signature matrices
├── CIBERSORTx_Signature_Matrix_Output_v3/  # Previous signature matrices
├── Combined_NK_TUSC2_Analysis_Output/      # Previous analysis outputs
├── TCGAdata/                   # TCGA data and analysis
├── TumorFractionOutput/        # Tumor fraction analysis outputs
└── README.md                   # This file
```

## Key Files

### Current Signature Matrix
- **Location**: `outputs/signature_matrices/Robust_Signature_Matrix_Output_TPM/NK_signature_matrix_linear.tsv`
- **Format**: CIBERSORTx-compatible TSV
- **Dimensions**: 814 genes × 6 NK subtypes
- **Normalization**: TPM (Transcripts Per Million)
- **Subtypes**: NK1A, NK1B, NK1C, NKint, NK2, NK3

### Main Pipeline Script
- **Location**: `scripts/signature_matrix/robust_signature_matrix_pipeline.py`
- **Purpose**: Complete signature matrix generation pipeline
- **Features**: 
  - Quality control
  - Marker gene identification
  - Signature matrix creation
  - Validation and visualization

## Quick Start

### 1. Run the Signature Matrix Pipeline
```bash
cd scripts/signature_matrix
python robust_signature_matrix_pipeline.py
```

### 2. Validate TPM Data
```bash
cd scripts/data_processing
python validate_tpm.py
```

### 3. Convert Raw Data to TPM (if needed)
```bash
cd scripts/data_processing
python convert_to_tpm_corrected.py
```

## Data Processing Pipeline

### Step 1: Raw Data Export
- Original Seurat object: `data/raw/PBMC_V2_VF1_AllGenes_NewNames.rds`
- Exported to: `data/raw/rebuffet_*.csv` and `data/raw/rebuffet_*.mtx`

### Step 2: AnnData Creation
- Script: `scripts/data_processing/fix_anndata_creation.py`
- Output: `data/processed/PBMC_V2_VF1_AllGenes_NewNames_linear_fixed.h5ad`

### Step 3: TPM Conversion
- Script: `scripts/data_processing/convert_to_tpm_corrected.py`
- Output: `data/processed/PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad`

### Step 4: Signature Matrix Generation
- Script: `scripts/signature_matrix/robust_signature_matrix_pipeline.py`
- Output: `outputs/signature_matrices/Robust_Signature_Matrix_Output_TPM/`

## NK Subtype Information

| Subtype | Cell Count | Description |
|---------|------------|-------------|
| NK1A    | 6,098      | Early NK cells |
| NK1B    | 3,959      | Intermediate NK cells |
| NK1C    | 8,381      | Mature NK cells |
| NKint   | 3,567      | Intermediate state |
| NK2     | 2,011      | Specialized NK cells |
| NK3     | 11,562     | Terminal NK cells |

## Quality Metrics

### TPM Validation
- **Expected sum per cell**: ~1,000,000
- **Data range**: 0.000 to 206,896.552
- **Data type**: float64

### Signature Matrix Quality
- **Total marker genes**: 814
- **Genes per subtype**: 115-200
- **Zero expression genes**: 1
- **High specificity markers**: Identified for each subtype

## Previous Versions

### Archive Location
- **Old notebooks**: `archive/old_notebooks/`
- **Old scripts**: `archive/old_scripts/`
- **Previous signature matrices**: `CIBERSORTx_Signature_Matrix_Output_v2/` and `CIBERSORTx_Signature_Matrix_Output_v3/`

### Version History
- **v1.0**: Initial signature matrix (notebook-based)
- **v2.0**: Improved signature matrix with better gene selection
- **v3.0**: Refined signature matrix with additional validation
- **Current**: Robust pipeline with TPM normalization and comprehensive quality control

## Dependencies

### Python Packages
- scanpy
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- rpy2 (for R integration)

### R Packages
- Seurat
- Matrix

## Usage Notes

1. **TPM vs Counts**: The current pipeline uses TPM normalization, which is appropriate for cross-sample comparisons
2. **Gene Selection**: Marker genes are selected based on differential expression analysis with quality filters
3. **Validation**: The pipeline includes comprehensive validation steps to ensure signature matrix quality
4. **CIBERSORTx Compatibility**: Output format is specifically designed for CIBERSORTx deconvolution

## Contact
For questions about this pipeline or the signature matrices, please refer to the analysis documentation in the respective output directories. 