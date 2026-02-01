# Project Organization Summary

## Reorganization Date: June 22, 2025

This document summarizes the reorganization of the NK Cell Analysis project for better maintainability and clarity.

## Key Changes Made

### 1. Main Analysis Script
- **Before**: `analysis.py` (in root directory)
- **After**: `scripts/main_analysis/NK_analysis_main.py`
- **Status**: ✅ This is the most up-to-date comprehensive analysis script (3,616 lines)

### 2. Signature Matrix Generation
- **Recommended**: `scripts/utilities/signature_matrix_generator.py`
- **Archived**: `archive/old_scripts/signature_matrix/robust_signature_matrix_pipeline.py`
- **Reason**: The generator script is more complete and handles TPM data properly

### 3. Directory Structure

#### Active Directories
```
scripts/
├── main_analysis/          # Main analysis scripts
│   └── NK_analysis_main.py
└── utilities/              # Utility scripts
    ├── signature_matrix_generator.py
    └── data_processing/    # Data processing utilities
```

#### Archived Directories
```
archive/
├── old_scripts/           # Outdated/diagnostic scripts
├── old_notebooks/         # Old Jupyter notebooks
├── test_outputs/          # Test output files
└── old_data/             # Old data files (GSE212890)
```

#### Documentation
```
docs/                      # All documentation files
├── README.md             # Main project documentation
├── PROJECT_ORGANIZATION.md
└── [other .md files]
```

### 4. Files Moved to Archive

#### Scripts
- `diagnostic_nk1b_tang_analysis.py` → `archive/old_scripts/`
- `extended_diagnostic_analysis.py` → `archive/old_scripts/`
- `investigate_dataset_differences.py` → `archive/old_scripts/`
- `scripts/signature_matrix/` → `archive/old_scripts/signature_matrix/`

#### Data Files
- `GSE212890_NK*` files → `archive/old_data/` (or removed as duplicates)

#### Documentation
- All `.md` files → `docs/`

### 5. Kept in Place
- `TCGAdata/` - Contains important TCGA analysis results
- `Combined_NK_TUSC2_Analysis_Output/` - Main analysis outputs
- `data/` - Current processed data files
- `outputs/` - Current analysis outputs
- `notebooks/` - Current Jupyter notebooks
- `TumorFractionOutput/` - Tumor fraction analysis results

## Best Practices Implemented

1. **Clear Separation**: Main analysis scripts separated from utilities
2. **Version Control**: Outdated files archived rather than deleted
3. **Documentation**: All documentation centralized in `docs/`
4. **Naming Convention**: Clear, descriptive file names
5. **Structure**: Logical hierarchy with clear purposes

## Usage Guidelines

### For Analysis
1. Use `scripts/main_analysis/NK_analysis_main.py` for comprehensive analysis
2. Use `scripts/utilities/signature_matrix_generator.py` for signature matrix generation
3. Use utilities in `scripts/utilities/data_processing/` for data preprocessing

### For Development
1. Keep new analysis scripts in `scripts/main_analysis/`
2. Keep utility functions in `scripts/utilities/`
3. Archive old versions rather than deleting them

### For Documentation
1. Update `README.md` for major changes
2. Add new documentation to `docs/`
3. Keep analysis notes and summaries in `docs/`

## File Recommendations

### Current Best Files
- **Main Analysis**: `scripts/main_analysis/NK_analysis_main.py`
- **Signature Matrix**: `scripts/utilities/signature_matrix_generator.py`
- **Data Processing**: Files in `scripts/utilities/data_processing/`

### Archived Files
- Check `archive/old_scripts/` for historical versions
- Old notebooks in `archive/old_notebooks/` for reference

## Next Steps
1. Test the reorganized structure
2. Update any scripts that reference old file paths
3. Consider creating a requirements.txt file for dependencies
4. Document any new analysis workflows

## Notes
- The project is now much cleaner and more maintainable
- All important files are preserved in the archive
- The structure follows best practices for scientific analysis projects 