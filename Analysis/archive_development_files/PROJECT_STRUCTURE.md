# NK Cell Analysis Pipeline - Project Structure

## 🏗️ **Modern Modular Architecture**

This project has been completely reorganized around a clean, modular architecture following Python best practices. The monolithic 3,746-line script has been replaced with focused, reusable modules.

## 📁 **Directory Structure**

```
NK_Cell_Analysis/
├── 🚀 run_nk_analysis.py          # Main entry point - runs complete analysis
├── 📋 requirements_gpu.txt         # GPU-accelerated dependencies
├── 📄 README.md                   # Project overview and quick start
├── 📄 PROJECT_STRUCTURE.md        # This file - detailed structure docs
│
├── 📦 src/                        # Main source code (modular architecture)
│   ├── ⚙️  config/                # Configuration modules
│   │   ├── constants.py           # Analysis constants and parameters
│   │   ├── gene_signatures.py     # Gene signature definitions
│   │   ├── plotting_config.py     # Plotting aesthetics and colors
│   │   ├── plotting.py           # Additional plotting utilities
│   │   ├── paths.py              # File paths and directory structure
│   │   ├── test_config.py        # Configuration testing utilities
│   │   └── __init__.py
│   │
│   ├── 🧬 core/                   # Core analysis functions
│   │   ├── data_processing.py     # Data loading and preprocessing
│   │   ├── dimensionality_reduction.py  # PCA, UMAP, clustering (enhanced)
│   │   ├── signature_analysis.py  # Gene signature scoring
│   │   ├── subtype_assignment.py  # NK subtype classification (with enhanced marker selection)
│   │   ├── enhanced_qc_functions.py     # Quality control functions
│   │   ├── signature_matrix_generator.py # Signature matrix generation
│   │   ├── phase3_demonstration.py      # Phase 3 demo functions
│   │   ├── test_core_operations.py      # Core operations testing
│   │   ├── data_processing/       # Data processing utilities
│   │   │   ├── convert_to_tpm_corrected.py
│   │   │   ├── validate_tpm.py
│   │   │   ├── fix_anndata_creation.py
│   │   │   └── load_exported_data.py
│   │   └── __init__.py
│   │
│   ├── 🔧 utils/                  # Utility functions
│   │   ├── data_utils.py          # Data manipulation utilities
│   │   ├── statistical_utils.py   # Statistical analysis functions
│   │   ├── file_utils.py          # File I/O and path management
│   │   └── __init__.py
│   │
│   ├── 🔄 workflows/              # Analysis orchestration
│   │   ├── data_workflows.py      # Data processing workflows
│   │   ├── analysis_workflows.py  # Analysis orchestration
│   │   ├── visualization_workflows.py  # Plotting workflows
│   │   ├── reporting_workflows.py # Report generation
│   │   ├── phase4_demonstration.py     # Phase 4 demo workflows
│   │   └── __init__.py
│   │
│   └── ✅ validation/             # Validation and testing
│       └── enhanced_pipeline_validation.py
│
├── 🧪 tests/                     # Test files and validation
│   ├── test_gpu_pca.py           # GPU acceleration tests
│   ├── test_utils.py             # Utility function tests
│   └── __init__.py
│
├── 📚 examples/                  # Example scripts and tutorials
│   └── (future example notebooks and scripts)
│
├── 📄 docs/                      # Documentation
│   ├── ANALYSIS_ADAPTATION_SUMMARY.md
│   ├── PROJECT_ORGANIZATION_SUMMARY.md
│   ├── PROJECT_ORGANIZATION.md
│   └── (other documentation files)
│
├── 📊 data/                      # Data files
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed data files
│
├── 📈 NK_Analysis_Results/       # Current analysis outputs
│   ├── cross_context_analysis/
│   ├── pipeline_metadata.json
│   └── (analysis results)
│
├── 📝 notebooks/                 # Jupyter notebooks
│   └── HumanNK_Transcriptomics_Function_Analysis_Final.ipynb
│
├── 🗃️  outputs/                  # Additional outputs
│   └── signature_matrices/
│
├── 🗂️  archive/                  # Legacy and archived files
│   ├── legacy_monolithic_analysis/  # Old 3,746-line monolithic script
│   │   ├── NK_analysis_main.py        # Original monolithic analysis
│   │   ├── NK_analysis_main_modular_demo.py
│   │   ├── NK_analysis_modular_demo_phase2.py
│   │   └── enhanced_histology_analysis.py
│   ├── legacy_outputs/           # Old analysis outputs
│   ├── legacy_logs/              # Old log files
│   ├── legacy_docs/              # Old documentation
│   ├── old_scripts/              # Other archived scripts
│   ├── old_notebooks/            # Archived notebooks
│   └── old_outputs/              # Archived outputs
│
└── 🧬 TCGAdata/                  # TCGA data (untouched)
    ├── Analysis_Python_Output_v3/
    ├── rna/
    ├── xml/
    └── (TCGA analysis files)
```

## 🚀 **Key Improvements**

### ✅ **Before (Monolithic)**
- ❌ Single 3,746-line file
- ❌ Mixed concerns (config + analysis + plotting)
- ❌ Hard to maintain and test
- ❌ Difficult to reuse components
- ❌ No clear separation of functionality

### 🟢 **After (Modular)**
- ✅ Clean separation of concerns
- ✅ Reusable, testable modules
- ✅ Modern Python project structure
- ✅ Easy to maintain and extend
- ✅ Clear dependency management
- ✅ Professional organization

## 🧬 **Enhanced Features**

### 🎯 **Intelligent Marker Selection** ✅ **IMPLEMENTED**
- **Location**: `src/core/subtype_assignment.py::select_optimal_subtype_markers()`
- **Features**: 
  - Handles shared DEGs intelligently
  - Assigns genes to most associated subtype
  - Max 3 markers per subtype (as requested)
  - Composite scoring (p-value × effect size)
  - Integrated in modular workflows

### ⚡ **GPU Acceleration** ✅ **ACTIVE**
- **Status**: CPU fallback active (GPU optional)
- **Configuration**: Automatic detection and graceful fallback
- **Enhanced PCA**: Data-driven PC selection in `src/core/dimensionality_reduction.py`
- **Testing**: GPU tests available in `tests/test_gpu_pca.py`

### 📊 **Comprehensive Analysis** ✅ **OPERATIONAL**
- **Signature Analysis**: 11 total signatures (4 developmental + 5 functional + 2 metabolic)
- **Statistical Rigor**: Enhanced statistical utilities in `src/utils/statistical_utils.py`
- **Effect Sizes**: Cohen's d, Cliff's delta, biological significance calculations
- **Quality Control**: Enhanced QC functions in `src/core/enhanced_qc_functions.py`

## 🔧 **Usage**

### **Simple Run**
```bash
python run_nk_analysis.py
```

### **Module Import Example**
```python
from src.core.subtype_assignment import select_optimal_subtype_markers
from src.config.gene_signatures import get_functional_signatures
from src.workflows.analysis_workflows import run_comprehensive_nk_analysis
```

## 📋 **Configuration**

All configuration is centralized in `src/config/`:
- **Constants**: Analysis parameters, thresholds
- **Gene Signatures**: All signature definitions
- **Plotting**: Colors, aesthetics, styling
- **Paths**: File locations, directory structure

## 🧪 **Testing**

Test files are organized in `tests/`:
- **GPU Tests**: `tests/test_gpu_pca.py`
- **Utility Tests**: `tests/test_utils.py`
- **Validation**: `src/validation/enhanced_pipeline_validation.py`

## 📚 **Documentation**

- **Quick Start**: `README.md`
- **Structure**: `PROJECT_STRUCTURE.md` (this file)
- **API Docs**: In `docs/` directory
- **Examples**: In `examples/` directory (future)

## 🗂️ **Archive Organization**

All legacy code has been properly archived:
- **Monolithic Analysis**: `archive/legacy_monolithic_analysis/`
- **Old Outputs**: `archive/legacy_outputs/`
- **Log Files**: `archive/legacy_logs/`
- **Documentation**: `archive/legacy_docs/`

---

**Last Updated**: December 22, 2024  
**Version**: Modular Architecture v1.0  
**Status**: Production Ready ✅ 