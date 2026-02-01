# NK Cell Analysis Pipeline - Modular Architecture

## Overview
This project provides a comprehensive, modular analysis pipeline for TUSC2 expression in human Natural Killer (NK) cells across different biological contexts (healthy blood, normal tissue, and tumor tissue). The pipeline features intelligent marker selection, GPU acceleration, and enhanced statistical analysis.

## Project Structure

```
Analysis/
├── 🚀 run_nk_analysis.py              # Main entry point - runs complete modular analysis
├── 📦 src/                            # Modular source code
│   ├── config/                        # Configuration modules
│   ├── core/                          # Core analysis functions (with enhanced marker selection)
│   ├── utils/                         # Utility functions
│   ├── workflows/                     # Analysis orchestration
│   └── validation/                    # Validation and testing
├── 🧪 tests/                          # Test files and validation
├── 📚 examples/                       # Example scripts and tutorials
├── 📄 docs/                           # Documentation
├── 📊 data/                           # Data files (processed & raw)
├── 📈 NK_Analysis_Results/            # Current analysis outputs
├── 🧬 TCGAdata/                       # TCGA bulk RNA-seq analysis
├── 🔧 scripts/                        # Organized analysis scripts
│   ├── 01_core_analysis/             # Main comprehensive analysis
│   ├── 02_specialized_analysis/      # Focused analysis scripts
│   ├── 03_visualization/             # Publication-quality plots
│   ├── 04_data_preparation/          # Data preprocessing utilities
│   ├── 05_research_investigation/    # Research hypothesis testing
│   ├── 06_testing_validation/        # Testing and validation
│   ├── 07_tcga_analysis/             # TCGA analysis scripts
│   ├── scaden_training/              # SCADEN deconvolution training
│   └── utilities/                    # Reusable utility functions
├── 🗂️ archive/                        # Legacy code and archived files
│   ├── legacy_monolithic_analysis/   # Original 3,746-line script
│   ├── legacy_logs/                  # Old log files
│   └── legacy_docs/                  # Archived documentation
└── 📝 notebooks/                      # Jupyter notebooks
```

> **📄 Detailed Structure**: See `PROJECT_STRUCTURE.md` for comprehensive architecture documentation

## Key Features

### 🎯 **Enhanced Marker Selection** ✅ **NEW**
- **Intelligent shared gene handling**: Assigns shared DEGs to most associated subtype
- **Max 3 markers per subtype**: Clean, interpretable subtype-specific markers
- **Composite scoring**: Combines p-value and effect size for optimal ranking
- **Location**: `src/core/subtype_assignment.py::select_optimal_subtype_markers()`

### ⚡ **GPU Acceleration** ✅ **ACTIVE**
- **Automatic detection**: GPU acceleration with CPU fallback
- **Enhanced PCA**: Data-driven principal component selection
- **Status**: Currently running with CPU fallback (GPU optional)

### 🧬 **Comprehensive Analysis**
- **11 Gene Signatures**: 4 developmental + 5 functional + 2 metabolic
- **Statistical Rigor**: Enhanced effect size calculations (Cohen's d, Cliff's delta)
- **Quality Control**: Adaptive QC with enhanced filtering

### 📦 **Modular Architecture**
- **Clean separation**: Configuration, core functions, utilities, workflows
- **Reusable components**: Import and use individual modules
- **Professional structure**: Modern Python project organization

## Getting Started

### 1. **Environment Setup**
```bash
# Install dependencies (GPU-enabled)
pip install -r requirements_gpu.txt
```

### 2. **Data Preparation**
- Place processed data files in `data/processed/` directory
- Supported formats: `.h5ad` (AnnData objects)

### 3. **Run Complete Analysis**
```bash
# Run the complete modular analysis pipeline
python run_nk_analysis.py
```

### 4. **Module Usage Examples**
```python
# Import specific modules for custom analysis
from src.core.subtype_assignment import select_optimal_subtype_markers
from src.config.gene_signatures import get_functional_signatures
from src.workflows.analysis_workflows import run_comprehensive_nk_analysis

# Use enhanced marker selection
optimal_markers = select_optimal_subtype_markers(
    adata, 'deg_key', subtypes, max_markers_per_subtype=3
)
```

## Scripts Organization

The project includes organized analysis scripts in the `scripts/` directory:

- **`01_core_analysis/`**: Main comprehensive NK cell analysis pipeline
- **`02_specialized_analysis/`**: Focused analysis scripts for specific research questions
- **`03_visualization/`**: Publication-quality plotting and visualization scripts
- **`04_data_preparation/`**: Data preprocessing and preparation utilities
- **`05_research_investigation/`**: Research hypothesis testing and investigation scripts
- **`06_testing_validation/`**: Testing, validation, and quality control scripts
- **`07_tcga_analysis/`**: TCGA bulk RNA-seq analysis scripts
- **`scaden_training/`**: SCADEN deconvolution training scripts
- **`utilities/`**: Reusable utility functions and helper modules

> **📄 Detailed Scripts Guide**: See `scripts/README_ORGANIZATION.md` for comprehensive script documentation

## Analysis Workflow

The modular analysis pipeline consists of:

1. **Data Processing**: Loading, preprocessing, and quality control
2. **Subtype Assignment**: NK cell subtype classification with confidence scoring
3. **Signature Analysis**: Gene signature scoring (developmental, functional, metabolic)
4. **Enhanced Marker Selection**: Intelligent DEG selection with shared gene handling
5. **TUSC2 Analysis**: Comprehensive TUSC2 expression analysis across contexts
6. **Cross-Context Synthesis**: Comparative analysis and visualization
7. **Statistical Analysis**: Enhanced statistical testing and effect size calculations

> **🔄 Workflow Details**: Each step is implemented as modular functions in `src/workflows/`

## Data Sources

- **Rebuffet et al. (2024)**: Healthy blood NK cells
- **Tang et al.**: Pan-cancer NK cell dataset
- **TCGA**: Bulk RNA-seq data for validation

## Key Findings

The enhanced analysis pipeline reveals:
- **Intelligent marker selection** provides cleaner, more interpretable subtype-specific markers
- **TUSC2 expression patterns** vary significantly across NK cell subtypes and biological contexts
- **Enhanced statistical analysis** provides robust effect size calculations and significance testing
- **Modular architecture** enables flexible, reusable analysis components

## Output Structure

The analysis generates a comprehensive output structure organized by analysis type:

```
Combined_NK_TUSC2_Analysis_Output/
├── 0_Setup_Figs/                    # Initial setup and validation plots
├── 1_Processed_Anndata/            # Processed AnnData objects
├── 2_Blood_NK_Char/                # Blood NK cell characterization
├── 3_NormalTissue_NK_Char/         # Normal tissue NK characterization  
├── 4_TumorTissue_NK_Char/          # Tumor tissue NK characterization
├── 5_TUSC2_Analysis/               # Multi-layered TUSC2 analysis
│   ├── 5A_Broad_Context/          # Cross-context comparison
│   ├── 5B_Within_Context_Subtypes/ # Context-specific analysis
│   └── 5C_DEG_TUSC2_Binary/       # Differential expression analysis
└── 6_Cross_Context_Synthesis/      # Comparative insights
    ├── TUSC2_Functional_Impact_Comparison/
    ├── TUSC2_Impact_on_Subtype_Programs/
    └── TUSC2_Impact_on_Dev_State/
```

Each analysis directory contains:
- `figures/` - Publication-ready plots organized by analysis type
- `data_for_graphpad/` - CSV files for external analysis  
- `stat_results_python/` - Statistical analysis results
- `marker_lists/` - Gene lists for each analysis

See `docs/OUTPUT_DIRECTORY_STRUCTURE.md` for detailed documentation.

## Documentation

- **📄 Quick Start**: This README
- **📄 Output Structure**: `docs/OUTPUT_DIRECTORY_STRUCTURE.md`
- **📄 Detailed Structure**: `PROJECT_STRUCTURE.md`
- **📄 Reorganization Summary**: `REORGANIZATION_SUMMARY.md`
- **📄 API Documentation**: In `docs/` directory
- **📄 GPU Setup**: `setup_gpu_acceleration.md`

## Citation

If you use this analysis pipeline, please cite the relevant papers and datasets used in this work.

## Status

**✅ Production Ready** | **✅ Enhanced Features Active** | **✅ Modular Architecture** | **✅ GPU Acceleration Available** 