"""
Single-Subtype SCADEN Model Creation
====================================

This script creates binary training datasets for each NK cell subtype,
enabling single-subtype deconvolution instead of the problematic 
multi-subtype approach.

Author: Analysis Pipeline
Date: 2025-01-12
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('single_subtype_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SingleSubtypeModelCreator:
    """
    Creates binary training datasets for single-subtype SCADEN models.
    """
    
    def __init__(self, tang_data_path: str = "tang_tumor_training_data_dense.h5ad"):
        """
        Initialize the single-subtype model creator.
        
        Parameters:
        -----------
        tang_data_path : str
            Path to the Tang training data
        """
        self.tang_data_path = tang_data_path
        self.output_dir = Path("scaden_models/NK_binary_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NK cell subtype mapping
        self.nk_subtypes = [
            "CD56dimCD16hi-c5-MKI67",
            "CD56dimCD16hi-c7-NR4A3", 
            "CD56brightCD16lo-c1-GZMH",
            "CD56brightCD16lo-c3-CCL3",
            "CD56dimCD16hi-c6-DNAJB1",
            "CD56dimCD16hi-c3-ZNF90",
            "CD56dimCD16hi-c8-KLRC2",
            "CD56brightCD16lo-c5-CREM",
            "CD56dimCD16hi-c2-CX3CR1",
            "CD56dimCD16hi-c4-NFKBIA",
            "CD56brightCD16lo-c2-IL7R-RGS1lo",
            "CD56dimCD16hi-c1-IL32",
            "CD56brightCD16hi",
            "CD56brightCD16lo-c4-IL7R"
        ]
        
        logger.info(f"Initialized SingleSubtypeModelCreator")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"NK subtypes to process: {len(self.nk_subtypes)}")
    
    def load_tang_data(self) -> sc.AnnData:
        """
        Load Tang training data.
        
        Returns:
        --------
        sc.AnnData
            Loaded Tang data
        """
        logger.info(f"Loading Tang data from: {self.tang_data_path}")
        
        if not os.path.exists(self.tang_data_path):
            raise FileNotFoundError(f"Tang data not found: {self.tang_data_path}")
        
        adata = sc.read_h5ad(self.tang_data_path)
        logger.info(f"Loaded data shape: {adata.shape}")
        logger.info(f"Available cell types: {adata.obs['cell_type'].unique()}")
        
        return adata
    
    def create_binary_dataset(self, adata: sc.AnnData, target_subtype: str) -> sc.AnnData:
        """
        Create binary dataset for a specific NK subtype.
        
        Parameters:
        -----------
        adata : sc.AnnData
            Original Tang data
        target_subtype : str
            Target NK subtype
            
        Returns:
        --------
        sc.AnnData
            Binary dataset with target_subtype vs other_cells
        """
        logger.info(f"Creating binary dataset for: {target_subtype}")
        
        # Create binary labels
        binary_labels = adata.obs['cell_type'].copy()
        binary_labels = binary_labels.astype(str)
        
        # Set target subtype as positive class, all others as negative
        binary_labels[binary_labels == target_subtype] = target_subtype
        binary_labels[binary_labels != target_subtype] = "other_cells"
        
        # Create new AnnData object
        adata_binary = adata.copy()
        adata_binary.obs['binary_cell_type'] = binary_labels
        
        # Check class balance
        class_counts = binary_labels.value_counts()
        logger.info(f"  {target_subtype}: {class_counts[target_subtype]} cells")
        logger.info(f"  other_cells: {class_counts['other_cells']} cells")
        
        # Calculate class balance ratio
        target_ratio = class_counts[target_subtype] / len(binary_labels)
        logger.info(f"  Target subtype ratio: {target_ratio:.3f}")
        
        if target_ratio < 0.01:
            logger.warning(f"  Low target subtype ratio ({target_ratio:.3f}) - may affect model performance")
        
        return adata_binary
    
    def export_binary_scaden_format(self, adata_binary: sc.AnnData, target_subtype: str) -> str:
        """
        Export binary dataset to SCADEN format.
        
        Parameters:
        -----------
        adata_binary : sc.AnnData
            Binary dataset
        target_subtype : str
            Target NK subtype
            
        Returns:
        --------
        str
            Output directory path
        """
        # Create subtype-specific directory
        subtype_dir = self.output_dir / target_subtype.replace("-", "_").replace("+", "pos")
        training_dir = subtype_dir / "training_data"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting binary data to: {training_dir}")
        
        # Export counts matrix (genes x cells)
        counts_df = pd.DataFrame(
            adata_binary.X.T,
            index=adata_binary.var_names,
            columns=adata_binary.obs_names
        )
        
        counts_file = training_dir / "_counts.txt"
        counts_df.to_csv(counts_file, sep='\t')
        logger.info(f"  Counts matrix saved: {counts_file}")
        logger.info(f"  Shape: {counts_df.shape}")
        
        # Export cell type labels
        celltypes_df = pd.DataFrame({
            'cell_id': adata_binary.obs_names,
            'cell_type': adata_binary.obs['binary_cell_type']
        })
        
        celltypes_file = training_dir / "_celltypes.txt"
        celltypes_df.to_csv(celltypes_file, sep='\t', index=False)
        logger.info(f"  Cell types saved: {celltypes_file}")
        
        # Create model subdirectories
        for model_size in ['m256', 'm512', 'm1024']:
            model_dir = subtype_dir / model_size
            model_dir.mkdir(exist_ok=True)
            
            # Copy gene list
            genes_file = model_dir / "genes.txt"
            with open(genes_file, 'w') as f:
                for gene in adata_binary.var_names:
                    f.write(f"{gene}\n")
            
            # Create cell type mapping
            celltypes_mapping_file = model_dir / "celltypes.txt"
            with open(celltypes_mapping_file, 'w') as f:
                f.write(f"{target_subtype}\n")
                f.write("other_cells\n")
        
        # Create training summary
        summary = {
            'target_subtype': target_subtype,
            'total_cells': len(adata_binary),
            'target_cells': sum(adata_binary.obs['binary_cell_type'] == target_subtype),
            'other_cells': sum(adata_binary.obs['binary_cell_type'] == 'other_cells'),
            'target_ratio': sum(adata_binary.obs['binary_cell_type'] == target_subtype) / len(adata_binary),
            'genes_count': len(adata_binary.var_names),
            'training_data_path': str(training_dir),
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        summary_file = subtype_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"  Training summary saved: {summary_file}")
        
        return str(subtype_dir)
    
    def create_training_script(self, subtype_dir: str, target_subtype: str) -> str:
        """
        Create SCADEN training script for the subtype.
        
        Parameters:
        -----------
        subtype_dir : str
            Subtype directory path
        target_subtype : str
            Target NK subtype
            
        Returns:
        --------
        str
            Training script path
        """
        script_content = f"""#!/bin/bash
# SCADEN Training Script for {target_subtype}
# Generated automatically by create_single_subtype_models.py

echo "Starting SCADEN training for {target_subtype}"
echo "================================="

# Set up directories
SUBTYPE_DIR="{subtype_dir}"
TRAINING_DIR="$SUBTYPE_DIR/training_data"
MODEL_DIR="$SUBTYPE_DIR/model"

cd "$SUBTYPE_DIR"

# Step 1: Simulate synthetic bulk data
echo "Step 1: Simulating synthetic bulk data..."
scaden simulate --cells 100 --n_samples 1000 --data "$TRAINING_DIR" --pattern "_counts.txt"

# Step 2: Process training data
echo "Step 2: Processing training data..."
scaden process data.h5ad "$TRAINING_DIR/_counts.txt"

# Step 3: Train models
echo "Step 3: Training SCADEN models..."
mkdir -p "$MODEL_DIR"
scaden train processed.h5ad --model_dir "$MODEL_DIR" --steps 5000

echo "Training completed for {target_subtype}"
echo "Model saved in: $MODEL_DIR"
"""
        
        script_path = Path(subtype_dir) / "train_model.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Training script created: {script_path}")
        return str(script_path)
    
    def create_prediction_script(self, cancer_types: List[str]) -> str:
        """
        Create batch prediction script for all cancer types.
        
        Parameters:
        -----------
        cancer_types : List[str]
            List of cancer types to predict
            
        Returns:
        --------
        str
            Prediction script path
        """
        script_content = f"""#!/bin/bash
# Batch SCADEN Predictions for Single-Subtype Models
# Generated automatically by create_single_subtype_models.py

echo "Starting batch predictions for single-subtype models"
echo "==================================================="

# Cancer types to process
CANCER_TYPES=("{' '.join(cancer_types)}")

# NK subtypes
NK_SUBTYPES=("{' '.join([s.replace('-', '_').replace('+', 'pos') for s in self.nk_subtypes])}")

# Create predictions directory
PRED_DIR="tcga_predictions/binary_predictions"
mkdir -p "$PRED_DIR"

# Loop through each NK subtype
for subtype in "${{NK_SUBTYPES[@]}}"; do
    echo "Processing subtype: $subtype"
    
    MODEL_DIR="scaden_models/NK_binary_models/$subtype/model"
    
    # Check if model exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "  WARNING: Model not found for $subtype"
        continue
    fi
    
    # Loop through each cancer type
    for cancer in "${{CANCER_TYPES[@]}}"; do
        echo "  Predicting $cancer..."
        
        BULK_FILE="TCGA_SCADEN_Ready/$cancer/${{cancer}}_bulk_scaden_ready.txt"
        OUTPUT_FILE="$PRED_DIR/${{cancer}}_${{subtype}}_predictions.txt"
        
        # Check if bulk file exists
        if [ ! -f "$BULK_FILE" ]; then
            echo "    WARNING: Bulk file not found for $cancer"
            continue
        fi
        
        # Run prediction
        scaden predict --model_dir "$MODEL_DIR" --outname "$OUTPUT_FILE" "$BULK_FILE"
        
        if [ $? -eq 0 ]; then
            echo "    SUCCESS: $OUTPUT_FILE"
        else
            echo "    ERROR: Failed to predict $cancer for $subtype"
        fi
    done
done

echo "Batch predictions completed!"
echo "Results saved in: $PRED_DIR"
"""
        
        script_path = self.output_dir.parent / "batch_predict_single_subtypes.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Batch prediction script created: {script_path}")
        return str(script_path)
    
    def run_full_pipeline(self) -> Dict[str, str]:
        """
        Run the complete single-subtype model creation pipeline.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping subtypes to their directories
        """
        logger.info("🚀 Starting Single-Subtype Model Creation Pipeline")
        logger.info("=" * 70)
        
        # Load Tang data
        adata = self.load_tang_data()
        
        # Process each NK subtype
        results = {}
        failed_subtypes = []
        
        for subtype in self.nk_subtypes:
            try:
                logger.info(f"\n📊 Processing {subtype}...")
                
                # Create binary dataset
                adata_binary = self.create_binary_dataset(adata, subtype)
                
                # Export to SCADEN format
                subtype_dir = self.export_binary_scaden_format(adata_binary, subtype)
                
                # Create training script
                training_script = self.create_training_script(subtype_dir, subtype)
                
                results[subtype] = subtype_dir
                logger.info(f"✅ Successfully processed {subtype}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {subtype}: {e}")
                failed_subtypes.append(subtype)
        
        # Create batch prediction script
        cancer_types = [
            "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LIHC", 
            "LUAD", "OV", "PRAD", "SKCM", "STAD", "THCA"
        ]
        
        batch_script = self.create_prediction_script(cancer_types)
        
        # Create master summary
        master_summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_subtypes': len(self.nk_subtypes),
            'successful_subtypes': len(results),
            'failed_subtypes': len(failed_subtypes),
            'subtype_directories': results,
            'failed_subtypes_list': failed_subtypes,
            'batch_prediction_script': batch_script
        }
        
        summary_file = self.output_dir.parent / "single_subtype_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(master_summary, f, indent=2)
        
        # Final summary
        logger.info(f"\n🎯 Single-Subtype Pipeline Complete!")
        logger.info(f"✅ Successfully processed: {len(results)} subtypes")
        logger.info(f"❌ Failed: {len(failed_subtypes)} subtypes")
        
        if results:
            logger.info(f"\n📊 Ready for Model Training:")
            for subtype, directory in results.items():
                logger.info(f"  {subtype}: {directory}")
        
        logger.info(f"\n🔥 Next Steps:")
        logger.info(f"1. Train individual models: cd to each subtype directory and run train_model.sh")
        logger.info(f"2. Run batch predictions: {batch_script}")
        logger.info(f"3. Perform survival analysis with meaningful variance")
        
        return results

def main():
    """Main execution function."""
    try:
        # Create single-subtype model creator
        creator = SingleSubtypeModelCreator()
        
        # Run the pipeline
        results = creator.run_full_pipeline()
        
        if results:
            print("\n" + "="*70)
            print("🎉 SINGLE-SUBTYPE MODEL CREATION COMPLETED!")
            print("="*70)
            print(f"✅ Created binary training data for {len(results)} NK subtypes")
            print(f"📂 Output directory: {creator.output_dir}")
            print(f"📋 Summary: {creator.output_dir.parent}/single_subtype_pipeline_summary.json")
        else:
            print("❌ No single-subtype models were created successfully")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 