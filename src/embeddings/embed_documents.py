"""
Embedding Generation Module
Generates embeddings using Google Gemini and stores in ChromaDB
"""

import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import time

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Create logs directory
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'embeddings.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class EmbeddingGenerator:
    """Generates embeddings using Google Gemini and stores in ChromaDB"""

    def __init__(self):
        # Get API key
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file. "
                "Please add your Gemini API key to the .env file."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        logger.info("Configured Google Gemini API")

        # Setup paths
        self.processed_data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        self.vectorstore_dir = Path(__file__).parent.parent.parent / 'vectorstore'
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vectorstore_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection_name = "documents"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Government data documents"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

        logger.info(f"Vector store directory: {self.vectorstore_dir}")

    def generate_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        """
        Generate embedding for text using Gemini

        Args:
            text: Text to embed
            retry_count: Number of retries on failure

        Returns:
            Embedding vector as list of floats
        """
        for attempt in range(retry_count):
            try:
                # Use Gemini embedding model
                result = genai.embed_content(
                    model="text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )

                return result['embedding']

            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {retry_count} attempts")
                    raise

    def load_documents(self) -> List[Dict[str, Any]]:
        """Load processed documents from JSON file"""
        doc_file = self.processed_data_dir / 'processed_documents.json'

        if not doc_file.exists():
            raise FileNotFoundError(
                f"Processed documents not found at {doc_file}. "
                "Please run the data processor first (Step 2)."
            )

        with open(doc_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        logger.info(f"Loaded {len(documents)} documents from {doc_file}")
        return documents

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlapping characters between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation within last 100 chars
                last_part = text[end-100:end]
                sentence_end = max(
                    last_part.rfind('.'),
                    last_part.rfind('?'),
                    last_part.rfind('!')
                )
                if sentence_end != -1:
                    end = end - 100 + sentence_end + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    def embed_documents(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
        """
        Generate embeddings for all documents and store in ChromaDB

        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process before saving

        Returns:
            Summary statistics
        """
        logger.info("Starting embedding generation...")

        stats = {
            'total_documents': len(documents),
            'total_chunks': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat()
        }

        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []

        for idx, doc in enumerate(documents):
            try:
                content = doc['content']
                metadata = doc['metadata']

                # Chunk the document
                chunks = self.chunk_text(content)
                logger.info(f"Document {idx+1}/{len(documents)}: {len(chunks)} chunks from {metadata.get('source', 'unknown')}")

                for chunk_idx, chunk in enumerate(chunks):
                    # Generate unique ID
                    doc_id = f"doc_{idx}_chunk_{chunk_idx}"

                    # Generate embedding
                    embedding = self.generate_embedding(chunk)

                    # Prepare metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = chunk_idx
                    chunk_metadata['total_chunks'] = len(chunks)
                    chunk_metadata['doc_id'] = idx

                    # Add to batch
                    all_ids.append(doc_id)
                    all_embeddings.append(embedding)
                    all_documents.append(chunk)
                    all_metadatas.append(chunk_metadata)

                    stats['total_chunks'] += 1

                    # Save batch to avoid memory issues
                    if len(all_ids) >= batch_size:
                        self.collection.add(
                            ids=all_ids,
                            embeddings=all_embeddings,
                            documents=all_documents,
                            metadatas=all_metadatas
                        )
                        logger.info(f"Saved batch of {len(all_ids)} chunks to vector store")

                        # Clear batch
                        all_ids = []
                        all_embeddings = []
                        all_documents = []
                        all_metadatas = []

                    # Rate limiting - small delay to respect API limits
                    time.sleep(0.1)

                stats['successful'] += 1

            except Exception as e:
                logger.error(f"Failed to process document {idx}: {str(e)}")
                stats['failed'] += 1

        # Save remaining documents
        if all_ids:
            self.collection.add(
                ids=all_ids,
                embeddings=all_embeddings,
                documents=all_documents,
                metadatas=all_metadatas
            )
            logger.info(f"Saved final batch of {len(all_ids)} chunks to vector store")

        stats['end_time'] = datetime.now().isoformat()

        # Save statistics
        stats_file = self.vectorstore_dir / 'embedding_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved embedding statistics to {stats_file}")

        return stats

    def test_retrieval(self, query: str = "plantation data", top_k: int = 3):
        """Test retrieval with a sample query"""
        logger.info(f"Testing retrieval with query: '{query}'")

        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            logger.info(f"Retrieved {len(results['documents'][0])} results")

            print("\n" + "="*70)
            print("SAMPLE RETRIEVAL TEST")
            print("="*70)
            print(f"Query: '{query}'")
            print()

            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"Result {i+1}:")
                print(f"  Source: {metadata.get('source', 'unknown')}")
                print(f"  Type: {metadata.get('type', 'unknown')}")
                print(f"  State: {metadata.get('state', 'N/A')}")
                print(f"  Content: {doc[:200]}...")
                print()

        except Exception as e:
            logger.error(f"Retrieval test failed: {str(e)}")


def main():
    """Main execution function"""

    print("="*70)
    print("LLM DOCUMENT Q&A PIPELINE - EMBEDDING GENERATION")
    print("="*70)
    print()

    try:
        # Initialize generator
        generator = EmbeddingGenerator()

        # Load documents
        documents = generator.load_documents()

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        print("This may take a few minutes depending on document size...")
        print()

        stats = generator.embed_documents(documents)

        # Print summary
        print()
        print("="*70)
        print("EMBEDDING GENERATION SUMMARY")
        print("="*70)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print("="*70)
        print()

        if stats['successful'] > 0:
            print("SUCCESS: Embeddings generated successfully!")
            print(f"Vector store saved to: {generator.vectorstore_dir}")
            print()

            # Test retrieval
            generator.test_retrieval("Tell me about plantation data")

            print("="*70)
            print("Ready for Step 4: RAG Implementation & UI")
            print("="*70)
        else:
            print("ERROR: No embeddings were generated. Check logs for details.")

    except ValueError as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease ensure you have:")
        print("  1. Created a .env file from .env.example")
        print("  2. Added your Gemini API key to the .env file")
        print("  3. Get a free API key from: https://ai.google.dev/")
    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease run Step 2 (data processor) first to create processed documents.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("Check logs/embeddings.log for details")


if __name__ == "__main__":
    main()
