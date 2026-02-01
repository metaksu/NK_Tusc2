# Scripts Organization Guide

This directory contains all analysis scripts organized by functionality for better maintainability and clarity.

## 📁 Directory Structure

### **01_core_analysis/**
**Purpose**: Main comprehensive analysis scripts
- `NK_analysis_main.py` - Complete NK cell analysis pipeline (279KB, 7,265 lines)
  - Comprehensive TUSC2 analysis in NK cell context
  - Covers all major analysis sections and visualizations
  - Primary script for the complete workflow

### **02_specialized_analysis/**
**Purpose**: Focused analysis scripts for specific research questions
- `tusc2_subtype_focused_analysis.py` - Detailed TUSC2 subtype analysis (38KB, 994 lines)
- `tusc2_tang_analysis.py` - TUSC2 analysis specific to Tang dataset (12KB, 370 lines)
- `dataset_harmonization_analysis.py` - Dataset integration and harmonization (21KB, 517 lines)

### **03_visualization/**
**Purpose**: Publication-quality plotting and visualization scripts
- `tusc2_publication_quality_plots.py` - Publication-ready TUSC2 visualizations (22KB, 526 lines)
  - Modern, clean visualizations with statistical annotations
  - Publication-quality aesthetics and export formats

### **04_data_preparation/**
**Purpose**: Data preprocessing and preparation utilities
- `cibersort_create_sig_prep.py` - CIBERSORTx signature matrix preparation (12KB, 340 lines)
  - Creates signature matrices for deconvolution analysis
  - Handles data formatting for CIBERSORTx compatibility

### **05_research_investigation/**
**Purpose**: Research investigation and hypothesis testing scripts
- `investigate_mki67_tusc2_paradox.py` - MKI67-TUSC2 paradox investigation (9.6KB, 223 lines)
  - Explores Simpson's Paradox in NK cell biology
  - Investigates apparent contradictions in expression patterns

### **06_testing_validation/**
**Purpose**: Testing, validation, and quality control scripts
- `test_enhanced_classification.py` - Classification method testing (8.8KB, 250 lines)
  - Compares hybrid vs neural network approaches
  - Validates classification performance
- `check_tang_data_structure.py` - Data structure validation (6.8KB, 187 lines)
  - Validates Tang dataset structure and formatting

### **07_tcga_analysis/**
**Purpose**: TCGA bulk RNA-seq analysis scripts
- `TCGA_STUDY_ANALYSIS.py` - Complete TCGA data analysis pipeline (104KB, 2,963 lines)
  - Loads and preprocesses TCGA data
  - Integrates clinical and expression data
  - Survival analysis and immune deconvolution
- `tcga_xml_analyzer.py` - TCGA XML file analysis (18KB, 455 lines)
  - Parses TCGA clinical XML files
  - Extracts structured clinical data

### **scaden_training/**
**Purpose**: SCADEN deconvolution training scripts
- `tang_nk_scaden_training.py` - SCADEN training pipeline (26KB, 699 lines)
  - Trains SCADEN models for NK cell deconvolution
  - Handles training data preparation and model optimization

### **utilities/**
**Purpose**: Reusable utility functions and helper modules
- `enhanced_qc_functions.py` - Enhanced quality control functions (15KB, 376 lines)
  - Advanced QC metrics and filtering
  - Updated NK cell signatures (2024)
- `signature_matrix_generator.py` - Signature matrix generation utilities (17KB, 421 lines)
  - Creates and validates signature matrices
  - Handles gene mapping and normalization

## 🚀 Usage Guidelines

### **Getting Started**
1. **Main Analysis**: Start with `01_core_analysis/NK_analysis_main.py` for comprehensive analysis
2. **Specialized Questions**: Use scripts in `02_specialized_analysis/` for focused investigations
3. **Visualization**: Use `03_visualization/` scripts for publication-quality plots
4. **TCGA Analysis**: Use `07_tcga_analysis/` scripts for bulk RNA-seq analysis

### **Dependencies**
- Core analysis scripts may import from `utilities/`
- Specialized scripts are mostly self-contained
- Check import statements at the top of each script

### **Best Practices**
1. **Script Selection**: Choose the most specific script for your analysis need
2. **Documentation**: Each script has detailed docstrings explaining functionality
3. **Version Control**: Major changes should be documented in script headers
4. **Testing**: Use `06_testing_validation/` scripts to validate methods

## 🔧 Maintenance

### **Adding New Scripts**
- Place scripts in the appropriate directory based on their primary function
- Update this README when adding new categories
- Follow existing naming conventions

### **Script Dependencies**
- Keep utilities general and reusable
- Minimize cross-dependencies between specialized scripts
- Document any required imports or data dependencies

### **Archive Policy**
- Old/deprecated scripts should be moved to an `archive/` subdirectory
- Keep change logs for major script updates
- Remove intermediate/temporary scripts after validation

## 📊 Quick Reference

| Task | Script | Directory |
|------|--------|-----------|
| Full NK Analysis | `NK_analysis_main.py` | `01_core_analysis/` |
| TUSC2 Subtype Analysis | `tusc2_subtype_focused_analysis.py` | `02_specialized_analysis/` |
| Publication Plots | `tusc2_publication_quality_plots.py` | `03_visualization/` |
| TCGA Analysis | `TCGA_STUDY_ANALYSIS.py` | `07_tcga_analysis/` |
| Data Validation | `check_tang_data_structure.py` | `06_testing_validation/` |
| SCADEN Training | `tang_nk_scaden_training.py` | `scaden_training/` |

---

**Last Updated**: January 2025  
**Maintained By**: AI Assistant  
**Version**: 1.0 