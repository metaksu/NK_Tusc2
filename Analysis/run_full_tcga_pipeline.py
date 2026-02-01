#!/usr/bin/env python3
"""
Run the complete TCGA CIBERSORTx pipeline for all available cancer types.
"""

import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tcga_pipeline_full.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete pipeline for all cancer types."""
    
    start_time = time.time()
    logger.info("Starting TCGA CIBERSORTx Pipeline for All Cancer Types")
    logger.info("=" * 80)
    
    try:
        # Import the pipeline
        logger.info("1. Importing pipeline...")
        from tcga_cibersortx_mixture_pipeline import TCGACIBERSORTxProcessor
        logger.info("Pipeline imported successfully")
        
        # Initialize processor
        logger.info("2. Initializing processor...")
        processor = TCGACIBERSORTxProcessor(
            tcga_base_dir="TCGAdata",
            output_dir="outputs/tcga_cibersortx_mixtures",
            sample_sheet_filename="gdc_sample_sheet.2025-06-26.tsv"
        )
        logger.info("Processor initialized")
        
        # Load data
        logger.info("3. Loading clinical data...")
        clinical_data = processor.load_clinical_data()
        processor.clinical_data = clinical_data
        logger.info(f"Clinical data loaded: {len(clinical_data)} records")
        
        logger.info("4. Loading sample sheet...")
        sample_sheet_data = processor.load_sample_sheet()
        processor.sample_sheet_data = sample_sheet_data
        logger.info(f"Sample sheet loaded: {len(sample_sheet_data)} records")
        
        # Detect cancer types
        logger.info("5. Detecting available cancer types...")
        cancer_types = processor.detect_available_cancer_types()
        processor.cancer_types = cancer_types
        logger.info(f"Detected {len(cancer_types)} cancer types: {cancer_types}")
        
        # Create master metadata
        logger.info("6. Creating master metadata...")
        master_metadata = processor.create_master_metadata()
        processor.master_metadata = master_metadata
        logger.info(f"Master metadata created: {len(master_metadata)} records")
        
        # Create output directory
        output_dir = Path(processor.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
        
        # Process each cancer type
        logger.info("7. Processing each cancer type...")
        logger.info("=" * 80)
        
        results = {}
        failed_types = []
        
        for i, cancer_type in enumerate(cancer_types, 1):
            logger.info(f"\nProcessing {cancer_type} ({i}/{len(cancer_types)})")
            logger.info("-" * 60)
            
            try:
                result = processor.process_cancer_type(cancer_type)
                if result:
                    results[cancer_type] = result
                    logger.info(f"{cancer_type} completed successfully!")
                    logger.info(f"   Final samples: {result.get('final_sample_count', 'Unknown')}")
                    logger.info(f"   Validation: {'PASSED' if result.get('validation_passed', False) else 'FAILED'}")
                    logger.info(f"   Output: {result.get('output_file', 'Unknown')}")
                else:
                    failed_types.append(cancer_type)
                    logger.error(f"{cancer_type} processing failed")
                    
            except Exception as e:
                failed_types.append(cancer_type)
                logger.error(f"{cancer_type} failed with error: {e}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("=" * 80)
        
        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time/60:.1f} minutes")
        logger.info(f"Successfully processed: {len(results)} cancer types")
        logger.info(f"Failed: {len(failed_types)} cancer types")
        
        if results:
            logger.info(f"\nSuccessful cancer types:")
            for cancer_type, result in results.items():
                samples = result.get('final_sample_count', 'Unknown')
                validation = 'PASSED' if result.get('validation_passed', False) else 'FAILED'
                logger.info(f"   {cancer_type}: {samples} samples {validation}")
        
        if failed_types:
            logger.info(f"\nFailed cancer types: {failed_types}")
        
        # Output locations
        logger.info(f"\nOutput directory: {output_dir}")
        logger.info(f"Log file: tcga_pipeline_full.log")
        
        # Next steps
        logger.info(f"\nNEXT STEPS:")
        logger.info(f"1. Review mixture files in: {output_dir}")
        logger.info(f"2. Upload to CIBERSORTx platform for deconvolution")
        logger.info(f"3. Use NK cell signature matrix for analysis")
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 