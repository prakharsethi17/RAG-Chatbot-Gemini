"""
Data Download Module
Downloads datasets from India Government Open Data API
"""

import requests
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import time
import os
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads datasets from various APIs with retry logic"""

    def __init__(self, save_dir: str = None):
        if save_dir is None:
            # Get the data/raw directory relative to this script
            self.save_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.download_metadata = []
        logger.info(f"Initialized DataDownloader. Save directory: {self.save_dir}")

    def download_dataset(
        self, 
        url: str, 
        filename: str, 
        max_retries: int = 3,
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        Download a dataset from URL with retry logic

        Args:
            url: API endpoint URL
            filename: Name to save the file
            max_retries: Number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            Tuple of (success: bool, message: str)
        """
        filepath = self.save_dir / filename

        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {filename} (Attempt {attempt + 1}/{max_retries})")

                response = requests.get(url, timeout=timeout)
                response.raise_for_status()

                # Save file
                with open(filepath, 'wb') as f:
                    f.write(response.content)

                # Store metadata
                metadata = {
                    'filename': filename,
                    'url': url,
                    'size_bytes': len(response.content),
                    'download_time': datetime.now().isoformat(),
                    'status': 'success'
                }
                self.download_metadata.append(metadata)

                logger.info(f"✓ Successfully downloaded {filename} ({len(response.content)} bytes)")
                return True, f"Downloaded successfully: {filename}"

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    error_msg = f"Failed to download {filename} after {max_retries} attempts"
                    logger.error(error_msg)

                    metadata = {
                        'filename': filename,
                        'url': url,
                        'size_bytes': 0,
                        'download_time': datetime.now().isoformat(),
                        'status': 'failed',
                        'error': str(e)
                    }
                    self.download_metadata.append(metadata)
                    return False, error_msg

    def download_all_datasets(self, datasets: List[Dict]) -> Dict:
        """
        Download multiple datasets

        Args:
            datasets: List of dicts with 'url' and 'filename' keys

        Returns:
            Summary dictionary with download results
        """
        results = {
            'total': len(datasets),
            'successful': 0,
            'failed': 0,
            'details': []
        }

        logger.info(f"Starting download of {len(datasets)} datasets")

        for dataset in datasets:
            success, message = self.download_dataset(
                dataset['url'], 
                dataset['filename']
            )

            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

            results['details'].append({
                'filename': dataset['filename'],
                'success': success,
                'message': message
            })

        # Save metadata
        self._save_metadata()

        return results

    def _save_metadata(self):
        """Save download metadata to JSON file"""
        metadata_file = self.save_dir / 'download_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.download_metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")


def main():
    """Main execution function"""

    print("="*70)
    print("LLM DOCUMENT Q&A PIPELINE - DATA DOWNLOADER")
    print("="*70)
    print()

    # Get API key from environment variable
    API_KEY = os.getenv('DATA_GOV_API_KEY')

    if not API_KEY:
        print("ERROR: DATA_GOV_API_KEY not found in .env file")
        print("Please add DATA_GOV_API_KEY to your .env file")
        return

    # Define datasets to download
    datasets = [
        {
            'url': f'https://api.data.gov.in/resource/c1ec0057-7de4-4669-8d97-f72b01846d83?api-key={API_KEY}&format=csv',
            'filename': 'plantation_data.csv',
            'description': 'State-wise Plantation Progress (2015-2024)'
        },
        {
            'url': f'https://api.data.gov.in/resource/7cdf1f2e-1027-4e4a-82c7-eade79b36f53?api-key={API_KEY}&format=xml',
            'filename': 'highway_funds.xml',
            'description': 'National Highway Funds Allocation (2023-2025)'
        },
        {
            'url': f'https://api.data.gov.in/resource/1e00ebf4-9008-4aef-b251-adc4b7537942?api-key={API_KEY}&format=xml',
            'filename': 'highway_length.xml',
            'description': 'National Highway Length Data (2024)'
        }
    ]

    # Print dataset information
    print("📊 Datasets to download:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['filename']:<25} - {dataset['description']}")
    print()

    # Initialize downloader
    downloader = DataDownloader()

    # Download all datasets
    results = downloader.download_all_datasets(datasets)

    # Print summary
    print()
    print("="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Total Datasets: {results['total']}")
    print(f"✓ Successful: {results['successful']}")
    print(f"✗ Failed: {results['failed']}")
    print()
    print("-"*70)

    for detail in results['details']:
        status = "✓" if detail['success'] else "✗"
        print(f"{status} {detail['filename']:<25} - {detail['message']}")

    print("="*70)
    print()

    if results['successful'] == results['total']:
        print("🎉 All datasets downloaded successfully!")
        print(f"📁 Files saved to: {downloader.save_dir}")
        print("✅ Ready for Step 2: Data Processing")
    else:
        print("⚠️  Some downloads failed. Check logs for details.")
        print(f"📝 Log file: {log_dir / 'data_download.log'}")


if __name__ == "__main__":
    main()
