#!/usr/bin/env Rscript
# Export Rebuffet Seurat Object to CSV Files for Python Pipeline
# =============================================================
# 
# This script exports the original Rebuffet Seurat object to CSV files
# that can be processed by the Python batch correction pipeline.
#
# Expected input: Seurat object (.rds file)
# Output: CSV files for counts, metadata, and gene information

library(Seurat)
library(dplyr)
# Using base R functions instead of readr

# Configuration
cat("=== REBUFFET SEURAT EXPORT SCRIPT ===\n")

# File paths - UPDATE THESE PATHS AS NEEDED
input_seurat_file <- "data/raw/PBMC_V2_VF1_AllGenes_NewNames.rds"  # Original Seurat object
output_dir <- "data/raw"

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Input Seurat file:", input_seurat_file, "\n")
cat("Output directory:", output_dir, "\n\n")

# Check if input file exists
if (!file.exists(input_seurat_file)) {
    cat("ERROR: Seurat file not found:", input_seurat_file, "\n")
    cat("Please update the path to your original Seurat object\n")
    quit(status = 1)
}

# Load Seurat object
cat("Loading Seurat object...\n")
seurat_obj <- readRDS(input_seurat_file)

# Display basic information
cat("Seurat object loaded successfully!\n")
cat("Dimensions:", dim(seurat_obj), "\n")
cat("Assays:", names(seurat_obj@assays), "\n")
cat("Default assay:", DefaultAssay(seurat_obj), "\n")

# Check available data slots and layers
cat("Available data slots in default assay:\n")
default_assay <- seurat_obj@assays[[DefaultAssay(seurat_obj)]]
available_slots <- slotNames(default_assay)

for (slot in c("counts", "data", "scale.data")) {
    if (slot %in% available_slots) {
        slot_data <- slot(default_assay, slot)
        if (length(slot_data) > 0) {
            cat("  ", slot, ": ", dim(slot_data), " (range: ", 
                round(min(slot_data), 3), " to ", round(max(slot_data), 3), ")\n", sep = "")
        }
    }
}

# Check for layers (newer Seurat versions may have TPM in layers)
cat("Checking for layers in assay:\n")
if ("layers" %in% slotNames(default_assay)) {
    layers_list <- slot(default_assay, "layers")
    if (length(layers_list) > 0) {
        for (layer_name in names(layers_list)) {
            layer_data <- layers_list[[layer_name]]
            if (length(layer_data) > 0) {
                cat("  Layer '", layer_name, "': ", dim(layer_data), " (range: ", 
                    round(min(layer_data), 3), " to ", round(max(layer_data), 3), ")\n", sep = "")
            }
        }
    } else {
        cat("  No layers found\n")
    }
} else {
    cat("  No layers slot available\n")
}

# Check metadata for any TPM-related information
cat("Checking metadata for TPM/normalization info:\n")
if ("misc" %in% slotNames(seurat_obj)) {
    misc_data <- slot(seurat_obj, "misc")
    if (length(misc_data) > 0) {
        cat("  Misc data available:", names(misc_data), "\n")
    }
}

# Determine which data to export (prioritize linear TPM data)
cat("\nDetermining data type to export...\n")

# Strategy: Look for linear TPM data in all available slots and layers
export_matrix <- NULL
data_type <- NULL

# Check all slots for TPM-like data (high values indicating linear scale)
cat("Searching for linear TPM data...\n")

# 1. Check 'data' slot
data_matrix <- GetAssayData(seurat_obj, slot = "data")
data_max <- max(data_matrix)
cat("  'data' slot max value: ", round(data_max, 1), "\n", sep = "")

# 2. Check 'counts' slot 
counts_matrix <- GetAssayData(seurat_obj, slot = "counts")
counts_max <- max(counts_matrix)
cat("  'counts' slot max value: ", round(counts_max, 1), "\n", sep = "")

# 3. Check layers if available
layer_matrices <- list()
if ("layers" %in% slotNames(default_assay)) {
    layers_list <- slot(default_assay, "layers")
    if (length(layers_list) > 0) {
        for (layer_name in names(layers_list)) {
            layer_data <- layers_list[[layer_name]]
            if (length(layer_data) > 0) {
                layer_max <- max(layer_data)
                cat("  Layer '", layer_name, "' max value: ", round(layer_max, 1), "\n", sep = "")
                layer_matrices[[layer_name]] <- layer_data
            }
        }
    }
}

# Decision logic: Prioritize linear TPM data (high values)
if (data_max > 50000) {
    # High values in 'data' slot - likely linear TPM
    cat("✅ Found linear TPM data in 'data' slot (max: ", round(data_max, 1), ")\n", sep = "")
    export_matrix <- data_matrix
    data_type <- "TPM"
} else if (counts_max > 50000) {
    # High values in 'counts' slot - could be TPM stored as counts
    cat("✅ Found high-value data in 'counts' slot, likely TPM (max: ", round(counts_max, 1), ")\n", sep = "")
    export_matrix <- counts_matrix  
    data_type <- "TPM"
} else {
    # Check layers for high values
    high_value_layer <- NULL
    for (layer_name in names(layer_matrices)) {
        layer_max <- max(layer_matrices[[layer_name]])
        if (layer_max > 50000) {
            high_value_layer <- layer_name
            break
        }
    }
    
    if (!is.null(high_value_layer)) {
        cat("✅ Found linear TPM data in layer '", high_value_layer, "' (max: ", 
            round(max(layer_matrices[[high_value_layer]]), 1), ")\n", sep = "")
        export_matrix <- layer_matrices[[high_value_layer]]
        data_type <- "TPM"
    } else if (counts_max > 100) {
        # Medium-high values in counts - could be TPM or raw counts
        cat("⚠️  Using 'counts' slot data (max: ", round(counts_max, 1), ") - checking if TPM...\n", sep = "")
        
        # Additional check: if counts are not integers, likely TPM
        sample_data <- as.vector(counts_matrix[1:100, 1:100])
        non_integer_count <- sum(sample_data != as.integer(sample_data), na.rm = TRUE)
        
        if (non_integer_count > 10) {
            cat("  Non-integer values detected - likely TPM data\n")
            data_type <- "TPM"
        } else {
            cat("  Integer values detected - likely raw counts\n") 
            data_type <- "counts"
        }
        export_matrix <- counts_matrix
    } else {
        # Low values - likely log-normalized, but use 'data' slot
        cat("⚠️  Only log-normalized data found (max: ", round(data_max, 1), ")\n", sep = "")
        cat("  Using 'data' slot - may need manual verification\n")
        export_matrix <- data_matrix
        data_type <- "log_normalized"
    }
}

# Final validation of selected data
cat("\n=== FINAL DATA SELECTION ===\n")
cat("Selected data type: ", data_type, "\n", sep = "")
cat("Matrix dimensions: ", dim(export_matrix), "\n", sep = "")
cat("Value range: ", round(min(export_matrix), 3), " to ", round(max(export_matrix), 3), "\n", sep = "")

# Additional TPM validation
if (data_type == "TPM") {
    cat("✅ Exporting LINEAR TPM data - suitable for log transformation in Python\n")
} else if (data_type == "counts") {
    cat("⚠️  Exporting count data - will be normalized in Python pipeline\n") 
} else {
    cat("⚠️  Exporting log-normalized data - may need verification\n")
}

# Convert sparse matrix to dense if needed and transpose for CSV export
cat("Converting expression matrix...\n")
if (class(export_matrix)[1] == "dgCMatrix") {
    cat("Converting sparse matrix to dense (this may take time)...\n")
    expr_dense <- as.matrix(export_matrix)
} else {
    expr_dense <- export_matrix
}

# Transpose so rows = cells, columns = genes (for easier Python processing)
expr_df <- as.data.frame(t(expr_dense))
cat("Expression matrix shape (cells x genes):", dim(expr_df), "\n")

# Export expression data
counts_file <- file.path(output_dir, "rebuffet_counts.csv")
cat("Exporting expression data to:", counts_file, "\n")
write.csv(expr_df, counts_file, row.names = TRUE)

# Export metadata
cat("Exporting metadata...\n")
metadata_df <- seurat_obj@meta.data

# Add rownames as a column for easier handling
metadata_df$cell_id <- rownames(metadata_df)

metadata_file <- file.path(output_dir, "rebuffet_metadata.csv")
cat("Metadata shape:", dim(metadata_df), "\n")
cat("Metadata columns:", colnames(metadata_df), "\n")
write.csv(metadata_df, metadata_file, row.names = TRUE)

# Export gene information
cat("Exporting gene information...\n")
gene_info <- data.frame(
    gene_id = rownames(seurat_obj),
    gene_name = rownames(seurat_obj),
    stringsAsFactors = FALSE
)

# Add any additional gene metadata if available
if (ncol(seurat_obj@assays[[DefaultAssay(seurat_obj)]]@meta.features) > 0) {
    gene_meta <- seurat_obj@assays[[DefaultAssay(seurat_obj)]]@meta.features
    gene_info <- cbind(gene_info, gene_meta)
}

genes_file <- file.path(output_dir, "rebuffet_genes.csv")
cat("Gene info shape:", dim(gene_info), "\n")
write.csv(gene_info, genes_file, row.names = TRUE)

# Create a summary file
summary_file <- file.path(output_dir, "export_summary.txt")
cat("Creating export summary...\n")

summary_lines <- c(
    "=== REBUFFET SEURAT EXPORT SUMMARY ===",
    "",
    paste("Export date:", Sys.time()),
    paste("Original Seurat file:", input_seurat_file),
    paste("Data type exported:", data_type),
    "",
    "DIMENSIONS:",
    paste("  Cells:", nrow(expr_df)),
    paste("  Genes:", ncol(expr_df)),
    paste("  Metadata columns:", ncol(metadata_df)),
    "",
    "FILES CREATED:",
    paste("  Expression data:", counts_file),
    paste("  Metadata:", metadata_file),
    paste("  Gene info:", genes_file),
    "",
    "METADATA COLUMNS:",
    paste("  ", colnames(metadata_df), collapse = "\n  "),
    "",
    "BATCH COLUMNS DETECTED:",
    paste("  ", grep("batch|donor|dataset|chemistry|platform", 
                    colnames(metadata_df), ignore.case = TRUE, value = TRUE), 
          collapse = "\n  "),
    "",
    "NEXT STEPS:",
    "1. Run the Python batch correction pipeline:",
    "   python create_rebuffet_h5ad_with_batch_correction.py",
    "2. Check the batch correction report for results",
    ""
)

writeLines(summary_lines, summary_file)

# Display completion message
cat("\n=== EXPORT COMPLETED SUCCESSFULLY ===\n")
cat("Files created:\n")
cat("  📊 Expression data:", counts_file, "\n")
cat("  📋 Metadata:", metadata_file, "\n")
cat("  🧬 Gene info:", genes_file, "\n")
cat("  📄 Summary:", summary_file, "\n")
cat("\nNext step: Run the Python batch correction pipeline\n")
cat("Command: python create_rebuffet_h5ad_with_batch_correction.py\n")

# Basic validation
cat("\nValidation checks:\n")

# Check for essential columns
essential_cols <- c("ident", "nCount_RNA", "nFeature_RNA")
missing_cols <- essential_cols[!essential_cols %in% colnames(metadata_df)]
if (length(missing_cols) > 0) {
    cat("⚠️  Missing essential columns:", paste(missing_cols, collapse = ", "), "\n")
} else {
    cat("✅ Essential metadata columns present\n")
}

# Check for batch columns
batch_cols <- grep("batch|donor|dataset|chemistry|platform", 
                   colnames(metadata_df), ignore.case = TRUE, value = TRUE)
if (length(batch_cols) > 0) {
    cat("✅ Batch columns found:", paste(batch_cols, collapse = ", "), "\n")
    
    # Show batch distribution for primary batch column
    primary_batch <- batch_cols[1]
    batch_table <- table(metadata_df[[primary_batch]])
    cat("  Distribution of", primary_batch, ":\n")
    for (i in seq_along(batch_table)) {
        cat("    ", names(batch_table)[i], ":", batch_table[i], "cells\n")
    }
} else {
    cat("⚠️  No obvious batch columns found\n")
}

# Check expression data
cat("✅ Expression data exported with range:", 
    round(min(expr_df[1:100, 1:100]), 3), "to", 
    round(max(expr_df[1:100, 1:100]), 3), "\n")

cat("\n🎉 Export completed! Ready for batch correction pipeline.\n") 