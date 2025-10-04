"""
Data Processing Module
Parses CSV and XML files, cleans data, and prepares documents for embedding
"""

import pandas as pd
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import re

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and cleans downloaded datasets"""

    def __init__(self, raw_data_dir: str = None, processed_data_dir: str = None):
        if raw_data_dir is None:
            self.raw_data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        else:
            self.raw_data_dir = Path(raw_data_dir)

        if processed_data_dir is None:
            self.processed_data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        else:
            self.processed_data_dir = Path(processed_data_dir)

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.documents = []

        logger.info(f"Initialized DataProcessor")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Processed data directory: {self.processed_data_dir}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            text = str(text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\-\:\(\)]', '', text)

        return text.strip()

    def process_csv_plantation(self) -> List[Dict[str, Any]]:
        """Process plantation data CSV"""
        logger.info("Processing plantation_data.csv...")

        csv_file = self.raw_data_dir / 'plantation_data.csv'
        if not csv_file.exists():
            logger.error(f"File not found: {csv_file}")
            return []

        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

            documents = []

            # Create document for each state
            for idx, row in df.iterrows():
                state = row['State']

                # Create a comprehensive text description
                text_parts = [
                    f"State: {state}",
                    f"Plantation Data from 2015-2024:",
                ]

                # Add cumulative data
                if 'Cumulative Plantation of 3 Years (2015-16 to 2017-18)' in row:
                    cumulative = row['Cumulative Plantation of 3 Years (2015-16 to 2017-18)']
                    text_parts.append(f"Cumulative plantation for 2015-16 to 2017-18: {cumulative} lakh trees")

                # Add yearly data
                for col in df.columns:
                    if 'Plantation in' in col and 'Total' in col:
                        year = col.split('in')[1].split('-')[0].strip()
                        value = row[col]
                        if pd.notna(value) and value != 0:
                            text_parts.append(f"Total plantation in {year}: {value} lakh trees")

                # Add total progress
                if 'Total Plantation Progress (2015-16 to 2023-24) - Total' in row:
                    total = row['Total Plantation Progress (2015-16 to 2023-24) - Total']
                    text_parts.append(f"Total plantation progress from 2015-16 to 2023-24: {total} lakh trees")

                text = '. '.join(text_parts) + '.'
                text = self.clean_text(text)

                doc = {
                    'content': text,
                    'metadata': {
                        'source': 'plantation_data.csv',
                        'state': state,
                        'type': 'plantation_statistics',
                        'processed_date': datetime.now().isoformat()
                    }
                }

                documents.append(doc)
                logger.debug(f"Processed document for state: {state}")

            logger.info(f"✓ Created {len(documents)} documents from CSV")
            return documents

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return []

    def process_xml_highway_funds(self) -> List[Dict[str, Any]]:
        """Process highway funds XML"""
        logger.info("Processing highway_funds.xml...")

        xml_file = self.raw_data_dir / 'highway_funds.xml'
        if not xml_file.exists():
            logger.error(f"File not found: {xml_file}")
            return []

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get title and description
            title = root.find('title')
            title_text = title.text if title is not None else "Highway Funds Data"

            documents = []

            # Find all records
            records = root.findall('.//record')
            logger.info(f"Found {len(records)} records in XML")

            for record in records:
                text_parts = [f"National Highway Funds Information: {title_text}"]
                metadata = {
                    'source': 'highway_funds.xml',
                    'type': 'highway_funds',
                    'processed_date': datetime.now().isoformat()
                }

                # Extract all fields from record
                for field in record:
                    field_name = field.tag
                    field_value = field.text if field.text else ""

                    if field_value and field_value.strip():
                        # Clean field name for readability
                        readable_name = field_name.replace('_', ' ').title()
                        text_parts.append(f"{readable_name}: {field_value}")

                        # Store state in metadata if present
                        if 'state' in field_name.lower() or 'ut' in field_name.lower():
                            metadata['state'] = field_value

                text = '. '.join(text_parts) + '.'
                text = self.clean_text(text)

                doc = {
                    'content': text,
                    'metadata': metadata
                }

                documents.append(doc)

            logger.info(f"✓ Created {len(documents)} documents from highway_funds.xml")
            return documents

        except Exception as e:
            logger.error(f"Error processing highway_funds.xml: {str(e)}")
            return []

    def process_xml_highway_length(self) -> List[Dict[str, Any]]:
        """Process highway length XML"""
        logger.info("Processing highway_length.xml...")

        xml_file = self.raw_data_dir / 'highway_length.xml'
        if not xml_file.exists():
            logger.error(f"File not found: {xml_file}")
            return []

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get title
            title = root.find('title')
            title_text = title.text if title is not None else "Highway Length Data"

            documents = []

            # Find all records
            records = root.findall('.//record')
            logger.info(f"Found {len(records)} records in XML")

            for record in records:
                text_parts = [f"National Highway Length Information: {title_text}"]
                metadata = {
                    'source': 'highway_length.xml',
                    'type': 'highway_length',
                    'processed_date': datetime.now().isoformat()
                }

                # Extract all fields
                for field in record:
                    field_name = field.tag
                    field_value = field.text if field.text else ""

                    if field_value and field_value.strip():
                        readable_name = field_name.replace('_', ' ').title()
                        text_parts.append(f"{readable_name}: {field_value}")

                        if 'state' in field_name.lower() or 'ut' in field_name.lower():
                            metadata['state'] = field_value

                text = '. '.join(text_parts) + '.'
                text = self.clean_text(text)

                doc = {
                    'content': text,
                    'metadata': metadata
                }

                documents.append(doc)

            logger.info(f"✓ Created {len(documents)} documents from highway_length.xml")
            return documents

        except Exception as e:
            logger.error(f"Error processing highway_length.xml: {str(e)}")
            return []

    def process_all(self) -> Dict[str, Any]:
        """Process all datasets"""
        logger.info("Starting data processing pipeline...")

        results = {
            'total_documents': 0,
            'by_source': {},
            'processing_time': None
        }

        start_time = datetime.now()

        # Process each dataset
        plantation_docs = self.process_csv_plantation()
        highway_funds_docs = self.process_xml_highway_funds()
        highway_length_docs = self.process_xml_highway_length()

        # Combine all documents
        self.documents = plantation_docs + highway_funds_docs + highway_length_docs

        # Update results
        results['total_documents'] = len(self.documents)
        results['by_source'] = {
            'plantation_data.csv': len(plantation_docs),
            'highway_funds.xml': len(highway_funds_docs),
            'highway_length.xml': len(highway_length_docs)
        }

        # Save processed documents
        if self.documents:
            output_file = self.processed_data_dir / 'processed_documents.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.documents)} documents to {output_file}")

            # Save summary
            summary_file = self.processed_data_dir / 'processing_summary.json'
            end_time = datetime.now()
            results['processing_time'] = str(end_time - start_time)
            results['timestamp'] = datetime.now().isoformat()

            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved processing summary to {summary_file}")

        return results


def main():
    """Main execution function"""

    print("="*70)
    print("LLM DOCUMENT Q&A PIPELINE - DATA PROCESSOR")
    print("="*70)
    print()

    # Initialize processor
    processor = DataProcessor()

    # Process all datasets
    results = processor.process_all()

    # Print summary
    print()
    print("="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total Documents Created: {results['total_documents']}")
    print()
    print("Documents by Source:")
    for source, count in results['by_source'].items():
        print(f"  ✓ {source:<30} {count:>3} documents")
    print()
    print(f"Processing Time: {results['processing_time']}")
    print("="*70)
    print()

    if results['total_documents'] > 0:
        print("🎉 Data processing completed successfully!")
        print(f"📁 Processed files saved to: {processor.processed_data_dir}")
        print("✅ Ready for Step 3: Embedding Generation")
    else:
        print("⚠️  No documents were processed. Check logs for details.")
        print(f"📝 Log file: {log_dir / 'data_processing.log'}")

    print()


if __name__ == "__main__":
    main()
