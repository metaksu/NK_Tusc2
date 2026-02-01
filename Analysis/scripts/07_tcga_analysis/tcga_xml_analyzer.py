#!/usr/bin/env python3
"""
TCGA XML Analyzer Script

This script analyzes TCGA XML clinical files to extract and summarize:
- All available cancer types (disease codes)
- Sample counts per cancer type
- File metadata and statistics
- Patient demographics summary
- Clinical data availability

Author: AI Assistant
Date: 2024
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict, Counter
import json
from pathlib import Path
import argparse
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TCGAXMLAnalyzer:
    """Analyzer for TCGA XML clinical files."""
    
    def __init__(self, xml_directory):
        """Initialize the analyzer with the XML directory path."""
        self.xml_directory = Path(xml_directory)
        self.results = {
            'cancer_types': defaultdict(int),
            'file_metadata': [],
            'patient_data': [],
            'errors': [],
            'summary_stats': {}
        }
        
        # TCGA cancer type mappings (common abbreviations to full names)
        self.cancer_type_mappings = {
            'ACC': 'Adrenocortical Carcinoma',
            'BLCA': 'Bladder Urothelial Carcinoma',
            'BRCA': 'Breast Invasive Carcinoma',
            'CESC': 'Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma',
            'CHOL': 'Cholangiocarcinoma',
            'COAD': 'Colon Adenocarcinoma',
            'DLBC': 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma',
            'ESCA': 'Esophageal Carcinoma',
            'GBM': 'Glioblastoma Multiforme',
            'HNSC': 'Head and Neck Squamous Cell Carcinoma',
            'KICH': 'Kidney Chromophobe',
            'KIRC': 'Kidney Renal Clear Cell Carcinoma',
            'KIRP': 'Kidney Renal Papillary Cell Carcinoma',
            'LAML': 'Acute Myeloid Leukemia',
            'LGG': 'Brain Lower Grade Glioma',
            'LIHC': 'Liver Hepatocellular Carcinoma',
            'LUAD': 'Lung Adenocarcinoma',
            'LUSC': 'Lung Squamous Cell Carcinoma',
            'MESO': 'Mesothelioma',
            'OV': 'Ovarian Serous Cystadenocarcinoma',
            'PAAD': 'Pancreatic Adenocarcinoma',
            'PCPG': 'Pheochromocytoma and Paraganglioma',
            'PRAD': 'Prostate Adenocarcinoma',
            'READ': 'Rectum Adenocarcinoma',
            'SARC': 'Sarcoma',
            'SKCM': 'Skin Cutaneous Melanoma',
            'STAD': 'Stomach Adenocarcinoma',
            'TGCT': 'Testicular Germ Cell Tumors',
            'THCA': 'Thyroid Carcinoma',
            'THYM': 'Thymoma',
            'UCEC': 'Uterine Corpus Endometrial Carcinoma',
            'UCS': 'Uterine Carcinosarcoma',
            'UVM': 'Uveal Melanoma'
        }
    
    def extract_disease_code(self, xml_file):
        """Extract disease code from a single XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Look for disease_code in admin section
            # Handle different namespace prefixes
            namespaces = {
                'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
                'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
                'shared': 'http://tcga.nci/bcr/xml/shared/2.7'
            }
            
            # Try different ways to find the disease code
            disease_code = None
            
            # Method 1: Direct admin:disease_code
            for prefix, uri in namespaces.items():
                try:
                    disease_elem = root.find(f'.//{{{uri}}}disease_code')
                    if disease_elem is not None:
                        disease_code = disease_elem.text
                        break
                except:
                    continue
            
            # Method 2: Look in admin section
            if disease_code is None:
                admin_section = root.find('.//admin:admin', namespaces)
                if admin_section is not None:
                    disease_elem = admin_section.find('admin:disease_code', namespaces)
                    if disease_elem is not None:
                        disease_code = disease_elem.text
            
            # Method 3: Extract from filename if XML parsing fails
            if disease_code is None:
                filename = xml_file.name
                # Look for TCGA-XX-XXXX pattern and extract the middle part
                import re
                match = re.search(r'TCGA-([A-Z0-9]+)-', filename)
                if match:
                    disease_code = match.group(1)
            
            return disease_code
            
        except ET.ParseError as e:
            logger.warning(f"XML parsing error in {xml_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {xml_file}: {e}")
            return None
    
    def extract_patient_data(self, xml_file, disease_code):
        """Extract basic patient data from XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            patient_data = {
                'file': xml_file.name,
                'disease_code': disease_code,
                'patient_id': None,
                'gender': None,
                'age_at_diagnosis': None,
                'vital_status': None,
                'tumor_stage': None,
                'histological_type': None
            }
            
            # Define namespaces
            namespaces = {
                'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
                'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
                'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
                'shared_stage': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7'
            }
            
            # Extract patient ID
            patient_elem = root.find('.//shared:patient_id', namespaces)
            if patient_elem is not None:
                patient_data['patient_id'] = patient_elem.text
            
            # Extract gender
            gender_elem = root.find('.//shared:gender', namespaces)
            if gender_elem is not None:
                patient_data['gender'] = gender_elem.text
            
            # Extract age at diagnosis
            age_elem = root.find('.//clin_shared:age_at_initial_pathologic_diagnosis', namespaces)
            if age_elem is not None:
                patient_data['age_at_diagnosis'] = age_elem.text
            
            # Extract vital status
            vital_elem = root.find('.//clin_shared:vital_status', namespaces)
            if vital_elem is not None:
                patient_data['vital_status'] = vital_elem.text
            
            # Extract tumor stage
            stage_elem = root.find('.//shared_stage:pathologic_stage', namespaces)
            if stage_elem is not None:
                patient_data['tumor_stage'] = stage_elem.text
            
            # Extract histological type
            hist_elem = root.find('.//shared:histological_type', namespaces)
            if hist_elem is not None:
                patient_data['histological_type'] = hist_elem.text
            
            return patient_data
            
        except Exception as e:
            logger.warning(f"Error extracting patient data from {xml_file}: {e}")
            return None
    
    def analyze_xml_files(self):
        """Analyze all XML files in the directory."""
        logger.info(f"Starting analysis of XML files in {self.xml_directory}")
        
        xml_files = list(self.xml_directory.glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files")
        
        for xml_file in xml_files:
            logger.info(f"Processing {xml_file.name}")
            
            # Extract disease code
            disease_code = self.extract_disease_code(xml_file)
            
            if disease_code:
                self.results['cancer_types'][disease_code] += 1
                
                # Extract patient data
                patient_data = self.extract_patient_data(xml_file, disease_code)
                if patient_data:
                    self.results['patient_data'].append(patient_data)
                
                # Store file metadata
                file_info = {
                    'filename': xml_file.name,
                    'disease_code': disease_code,
                    'file_size': xml_file.stat().st_size,
                    'full_name': self.cancer_type_mappings.get(disease_code, 'Unknown')
                }
                self.results['file_metadata'].append(file_info)
            else:
                self.results['errors'].append({
                    'file': xml_file.name,
                    'error': 'Could not extract disease code'
                })
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        logger.info("Analysis completed")
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics from the results."""
        total_files = len(self.results['file_metadata'])
        total_patients = len(self.results['patient_data'])
        
        # Cancer type statistics
        cancer_type_counts = dict(self.results['cancer_types'])
        unique_cancer_types = len(cancer_type_counts)
        
        # Patient demographics
        gender_counts = Counter()
        vital_status_counts = Counter()
        age_data = []
        
        for patient in self.results['patient_data']:
            if patient['gender']:
                gender_counts[patient['gender']] += 1
            if patient['vital_status']:
                vital_status_counts[patient['vital_status']] += 1
            if patient['age_at_diagnosis']:
                try:
                    age_data.append(int(patient['age_at_diagnosis']))
                except (ValueError, TypeError):
                    pass
        
        self.results['summary_stats'] = {
            'total_files': total_files,
            'total_patients': total_patients,
            'unique_cancer_types': unique_cancer_types,
            'cancer_type_counts': cancer_type_counts,
            'gender_distribution': dict(gender_counts),
            'vital_status_distribution': dict(vital_status_counts),
            'age_statistics': {
                'count': len(age_data),
                'mean': sum(age_data) / len(age_data) if age_data else None,
                'min': min(age_data) if age_data else None,
                'max': max(age_data) if age_data else None
            } if age_data else None
        }
    
    def generate_report(self, output_dir="tcga_analysis_output"):
        """Generate comprehensive analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create summary report
        self._create_summary_report(output_path)
        
        # Create detailed CSV files
        self._create_csv_files(output_path)
        
        # Create JSON export
        self._create_json_export(output_path)
        
        logger.info(f"Reports generated in {output_path}")
    
    def _create_summary_report(self, output_path):
        """Create a human-readable summary report."""
        report_file = output_path / "tcga_analysis_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TCGA XML CLINICAL DATA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"XML Directory: {self.xml_directory}\n\n")
            
            # Overall statistics
            stats = self.results['summary_stats']
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total XML files processed: {stats['total_files']}\n")
            f.write(f"Total patients: {stats['total_patients']}\n")
            f.write(f"Unique cancer types: {stats['unique_cancer_types']}\n\n")
            
            # Cancer type breakdown
            f.write("CANCER TYPE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            cancer_counts = sorted(stats['cancer_type_counts'].items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for cancer_code, count in cancer_counts:
                full_name = self.cancer_type_mappings.get(cancer_code, 'Unknown')
                f.write(f"{cancer_code:6} ({full_name:50}): {count:4d} patients\n")
            
            f.write("\n")
            
            # Demographics
            if stats['gender_distribution']:
                f.write("GENDER DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                for gender, count in stats['gender_distribution'].items():
                    f.write(f"{gender}: {count} patients\n")
                f.write("\n")
            
            if stats['vital_status_distribution']:
                f.write("VITAL STATUS DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                for status, count in stats['vital_status_distribution'].items():
                    f.write(f"{status}: {count} patients\n")
                f.write("\n")
            
            if stats['age_statistics']:
                age_stats = stats['age_statistics']
                f.write("AGE AT DIAGNOSIS STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Patients with age data: {age_stats['count']}\n")
                f.write(f"Mean age: {age_stats['mean']:.1f} years\n")
                f.write(f"Age range: {age_stats['min']} - {age_stats['max']} years\n\n")
            
            # Errors summary
            if self.results['errors']:
                f.write("ERRORS ENCOUNTERED:\n")
                f.write("-" * 40 + "\n")
                for error in self.results['errors']:
                    f.write(f"{error['file']}: {error['error']}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Analysis completed successfully.\n")
    
    def _create_csv_files(self, output_path):
        """Create CSV files for detailed data export."""
        # Cancer type summary
        cancer_summary = []
        for cancer_code, count in self.results['summary_stats']['cancer_type_counts'].items():
            cancer_summary.append({
                'Cancer_Code': cancer_code,
                'Full_Name': self.cancer_type_mappings.get(cancer_code, 'Unknown'),
                'Patient_Count': count
            })
        
        cancer_df = pd.DataFrame(cancer_summary)
        cancer_df.to_csv(output_path / "cancer_type_summary.csv", index=False)
        
        # Patient data
        if self.results['patient_data']:
            patient_df = pd.DataFrame(self.results['patient_data'])
            patient_df.to_csv(output_path / "patient_data.csv", index=False)
        
        # File metadata
        if self.results['file_metadata']:
            metadata_df = pd.DataFrame(self.results['file_metadata'])
            metadata_df.to_csv(output_path / "file_metadata.csv", index=False)
    
    def _create_json_export(self, output_path):
        """Create JSON export of all results."""
        json_file = output_path / "tcga_analysis_results.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        json_results = {
            'cancer_types': dict(self.results['cancer_types']),
            'file_metadata': self.results['file_metadata'],
            'patient_data': self.results['patient_data'],
            'errors': self.results['errors'],
            'summary_stats': self.results['summary_stats'],
            'cancer_type_mappings': self.cancer_type_mappings
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def print_summary(self):
        """Print a quick summary to console."""
        stats = self.results['summary_stats']
        
        print("\n" + "="*60)
        print("TCGA XML ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total files processed: {stats['total_files']}")
        print(f"Total patients: {stats['total_patients']}")
        print(f"Unique cancer types: {stats['unique_cancer_types']}")
        
        print("\nTop 10 Cancer Types:")
        print("-" * 40)
        cancer_counts = sorted(stats['cancer_type_counts'].items(), 
                             key=lambda x: x[1], reverse=True)
        
        for i, (cancer_code, count) in enumerate(cancer_counts[:10], 1):
            full_name = self.cancer_type_mappings.get(cancer_code, 'Unknown')
            print(f"{i:2d}. {cancer_code:6} ({full_name:40}): {count:4d} patients")
        
        if len(cancer_counts) > 10:
            print(f"   ... and {len(cancer_counts) - 10} more cancer types")
        
        print("="*60)


def main():
    """Main function to run the TCGA XML analyzer."""
    parser = argparse.ArgumentParser(description='Analyze TCGA XML clinical files')
    parser.add_argument('xml_directory', help='Directory containing TCGA XML files')
    parser.add_argument('--output', '-o', default='tcga_analysis_output',
                       help='Output directory for reports (default: tcga_analysis_output)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating detailed reports')
    
    args = parser.parse_args()
    
    # Check if XML directory exists
    xml_dir = Path(args.xml_directory)
    if not xml_dir.exists():
        logger.error(f"XML directory not found: {xml_dir}")
        return 1
    
    # Initialize and run analyzer
    analyzer = TCGAXMLAnalyzer(xml_dir)
    analyzer.analyze_xml_files()
    
    # Print summary to console
    analyzer.print_summary()
    
    # Generate detailed reports unless disabled
    if not args.no_report:
        analyzer.generate_report(args.output)
    
    return 0


if __name__ == "__main__":
    exit(main()) 