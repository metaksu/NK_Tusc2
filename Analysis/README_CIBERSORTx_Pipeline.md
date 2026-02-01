# TCGA Raw Data to CIBERSORTx Mixture Pipeline

This pipeline processes raw TCGA data directly from XML clinical files and TSV RNA-seq files to create CIBERSORTx-compatible mixture files for NK cell deconvolution analysis.

## 🚀 Key Improvements

### **From Preprocessed to Raw Data Loading**
- **Before**: Loaded preprocessed CIBERSORTx result files
- **After**: Processes raw XML + TSV files using the same robust approach as `TCGA_STUDY_ANALYSIS.py`

### **Enhanced Data Quality**
- **Raw XML Parsing**: Full clinical metadata extraction with disease-specific configurations
- **Comprehensive Filtering**: CIBERSORTx-optimized gene filtering (expression, variance, zero fraction)
- **Tumor-Only Focus**: Automatic filtering to tumor samples only
- **Quality Control**: Extensive validation and error handling

### **Automated Processing**
- **Multi-Cancer Support**: Processes all detected cancer types automatically
- **Standardized Output**: Creates properly formatted CIBERSORTx mixture files
- **Rich Metadata**: Generates detailed summary files for each cancer type

## 📋 Requirements

### **Data Structure**
```
TCGAdata/
├── xml/                           # Clinical XML files
│   ├── nationwidechildrens.org_clinical.TCGA-XX-XXXX.xml
│   └── ...
├── rna/                           # RNA-seq TSV files  
│   ├── XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX.tsv
│   └── ...
└── gdc_sample_sheet.2025-06-26.tsv   # GDC sample sheet
```

### **Python Dependencies**
```bash
pip install pandas numpy scipy scanpy pathlib
```

## 🔧 Usage

### **Method 1: Simple Full Pipeline**
```python
from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor

# Run complete pipeline
processor = TCGACIBERSORTxProcessor(
    tcga_base_dir="TCGAdata",
    output_dir="outputs/tcga_cibersortx_mixtures"
)

results = processor.run_full_pipeline()
```

### **Method 2: Command Line**
```bash
python tcga_cibersortx_mixture_pipeline.py
```

### **Method 3: Testing**
```bash
# General testing
python test_cibersortx_pipeline.py

# BRCA-specific validation (tests 881 sample limit)
python test_brca_validation.py
```

## 📊 Output Files

### **Mixture Files**
- `{CANCER}_tumor_mixture_for_cibersortx_harmonized_{DATE}.txt`
- Format: Genes as rows, samples as columns, gene symbols in first column
- Ready for direct upload to CIBERSORTx platform

### **Metadata Summaries**
- `{CANCER}_mixture_metadata_summary_{DATE}.txt`
- Contains sample counts, clinical metadata distribution, expression statistics
- CIBERSORTx analysis instructions

### **Pipeline Summary**
- `pipeline_summary.json`
- Overall pipeline statistics and processing results

## ⚙️ Configuration

### **CIBERSORTx Filtering Thresholds**
```python
cibersortx_thresholds = {
    "min_expression_threshold": 0.1,     # Minimum mean expression
    "min_nonzero_samples": 10,           # Min samples with non-zero expression  
    "min_variance_threshold": 0.01,      # Minimum variance across samples
    "max_zero_fraction": 0.8,            # Max fraction of zero values per gene
    "preferred_rna_count_column": "tpm_unstranded"
}
```

### **Supported Cancer Types**
- BRCA, GBM, LUAD, LUSC, KIRC, HNSC, SKCM, OV, PRAD, THCA, COAD, BLCA, LIHC, STAD, LGG
- Automatically detected from sample sheet Project IDs

### **Ground Truth Sample Limits**
```python
expected_sample_limits = {
    'BRCA': 881,   # Ground truth: 1,098 cases, 1,097 clinical, 881 with proteome
    'LUAD': 500,   # Estimated based on typical TCGA sizes
    'GBM': 150,    # Smaller cohort
    'KIRC': 400,   # Medium cohort
    # ... (additional cancer types with estimated limits)
}
```
The pipeline validates that sample counts don't exceed these known limits.

## 🔬 Pipeline Steps

1. **Load Clinical Data**: Parse XML files using disease-specific configurations
2. **Load Sample Sheet**: Process GDC sample sheet metadata  
3. **Create Master Metadata**: Merge clinical and sample data
4. **Detect Cancer Types**: Identify available cancer types from Project IDs
5. **Process Each Cancer Type**:
   - Load RNA-seq data for cancer type samples
   - Apply CIBERSORTx-specific gene filtering
   - Create tumor-only subset
   - Generate CIBERSORTx mixture file
   - Create metadata summary

## 📈 Quality Control Features

### **Gene Filtering**
- Remove genes with >80% zeros across samples
- Filter genes below expression threshold (0.1 TPM)
- Remove low-variance genes (variance < 0.01)
- Require minimum samples with non-zero expression (10)

### **Sample Filtering**  
- Tumor samples only (based on Tissue_Type)
- Common samples between RNA-seq and metadata
- Valid expression data requirements

### **Ground Truth Validation**
- **BRCA Validation**: Maximum 881 samples (based on known proteome data)
- **Sample Count Limits**: Predefined limits for each cancer type
- **Real-time Validation**: Warns if sample counts exceed expected limits
- **Validation Reporting**: Detailed validation results in summary files

### **Data Validation**
- Duplicate gene handling (mean aggregation)
- Missing data checks
- File format validation
- Comprehensive error logging
- Sample count validation against ground truth

## 🎯 Next Steps for CIBERSORTx Analysis

1. **Upload mixture files** to CIBERSORTx online platform
2. **Use NK cell signature matrix** (e.g., Tang-derived reference)
3. **Run deconvolution analysis** with appropriate parameters
4. **Download results** for downstream analysis
5. **Correlate NK infiltration** with clinical outcomes using metadata

## 🔍 Comparison with Previous Approach

| Aspect | Previous Approach | New Approach |
|--------|------------------|--------------|
| **Data Source** | Preprocessed CIBERSORTx results | Raw XML + TSV files |
| **Clinical Data** | Limited metadata | Full XML parsing |
| **Gene Selection** | Tang reference genes only | CIBERSORTx-optimized filtering |
| **Sample Selection** | Manual tumor filtering | Automated tumor-only pipeline |
| **Output Format** | Analysis results | Raw mixture files for CIBERSORTx |
| **Quality Control** | Basic | Comprehensive filtering + validation |
| **Automation** | Manual per cancer type | Automated multi-cancer processing |

## 🐛 Troubleshooting

### **Common Issues**

1. **"No XML files found"**
   - Ensure XML files are in `TCGAdata/xml/` directory
   - Check file extensions are `.xml`

2. **"Sample sheet not found"**  
   - Verify sample sheet path and filename
   - Default: `gdc_sample_sheet.2025-06-26.tsv`

3. **"No RNA-seq files found"**
   - Check RNA-seq files are in `TCGAdata/rna/` directory
   - Supported extensions: `.tsv`, `.txt`, `.gz`

4. **"No common samples"**
   - Verify File_Name_Root matching between sample sheet and RNA-seq files
   - Check cancer type filtering in sample sheet

### **Performance Notes**
- Processing time: ~5-15 minutes per cancer type
- Memory usage: ~2-4 GB for large cancer types (BRCA, LUAD)
- Output file sizes: ~50-200 MB per cancer type

## 📚 Related Files

- `tcga_cibersortx_mixture_pipeline.py` - Main pipeline script
- `test_cibersortx_pipeline.py` - General testing and validation script  
- `test_brca_validation.py` - BRCA-specific validation test (881 sample limit)
- `TCGA_STUDY_ANALYSIS.py` - Reference implementation for data loading
- `tcga_scaden_raw_data_pipeline.py` - Original SCADEN pipeline

## 🤝 Contributing

The pipeline is designed to be modular and extensible. Key areas for enhancement:
- Additional cancer type support
- Custom gene filtering strategies  
- Integration with other deconvolution methods
- Performance optimizations for large datasets 