"""
RAG Chain Implementation
Implements Retrieval Augmented Generation using Gemini and ChromaDB
"""

import os
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Setup logging
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'rag.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class RAGChain:
    """RAG implementation with Gemini LLM and ChromaDB vector store"""

    def __init__(self):
        # Get API key
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize LLM
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Initialized Gemini 2.5 Flash model")

        # Setup vector store
        vectorstore_dir = Path(__file__).parent.parent.parent / 'vectorstore'
        self.chroma_client = chromadb.PersistentClient(
            path=str(vectorstore_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Load collection
        try:
            self.collection = self.chroma_client.get_collection("documents")
            logger.info("Loaded vector store collection")
        except Exception as e:
            raise ValueError(
                "Vector store not found. Please run Step 3 (embedding generation) first."
            )

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for user query"""
        try:
            result = genai.embed_content(
                model="text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise

    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector store

        Args:
            query: User query
            top_k: Number of results to retrieve
            filter_metadata: Optional metadata filters (e.g., {'state': 'Delhi'})

        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)

            # Build query parameters
            query_params = {
                'query_embeddings': [query_embedding],
                'n_results': top_k
            }

            # Add metadata filter if provided
            if filter_metadata:
                query_params['where'] = filter_metadata

            # Search vector store
            results = self.collection.query(**query_params)

            # Format results
            retrieved_docs = []
            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0]
            )):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': metadata,
                    'rank': i + 1
                })

            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    def create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Create RAG prompt with context and query

        Args:
            query: User query
            context_docs: Retrieved context documents

        Returns:
            Formatted prompt string
        """
        # Build context section
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc['metadata'].get('source', 'unknown')
            state = doc['metadata'].get('state', 'N/A')
            doc_type = doc['metadata'].get('type', 'unknown')

            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"State: {state}\n"
                f"Type: {doc_type}\n"
                f"Content: {doc['content']}\n"
            )

        context_text = "\n".join(context_parts)

        # Create full prompt
        prompt = f"""You are a helpful AI assistant answering questions about Indian government data on plantations and national highways.

Context Information:
{context_text}

User Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context information
2. If the context doesn't contain enough information, clearly state that
3. Cite specific sources and states when providing statistics
4. Be concise but thorough
5. Use bullet points for multiple data points
6. Include relevant numbers and statistics from the context

Answer:"""

        return prompt

    def generate_answer(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using RAG pipeline

        Args:
            query: User query
            top_k: Number of context documents to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary with answer and source information
        """
        try:
            # Retrieve context
            context_docs = self.retrieve_context(query, top_k, filter_metadata)

            if not context_docs:
                return {
                    'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or ask about plantation data or national highways.",
                    'sources': [],
                    'error': 'No relevant documents found'
                }

            # Create prompt
            prompt = self.create_prompt(query, context_docs)

            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text

            # Extract sources
            sources = []
            for doc in context_docs:
                source_info = {
                    'source': doc['metadata'].get('source', 'unknown'),
                    'state': doc['metadata'].get('state', 'N/A'),
                    'type': doc['metadata'].get('type', 'unknown'),
                    'content_preview': doc['content'][:200] + "..."
                }
                if source_info not in sources:  # Avoid duplicates
                    sources.append(source_info)

            logger.info(f"Generated answer for query: {query[:50]}...")

            return {
                'answer': answer,
                'sources': sources[:3],  # Top 3 unique sources
                'num_sources': len(context_docs)
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return {
                'answer': f"An error occurred while generating the answer: {str(e)}",
                'sources': [],
                'error': str(e)
            }


def test_rag():
    """Test the RAG chain with sample queries"""
    print("="*70)
    print("RAG CHAIN TEST")
    print("="*70)
    print()

    try:
        rag = RAGChain()

        # Test queries
        test_queries = [
            "Which state has the highest plantation progress?",
            "Tell me about plantation data in Delhi",
            "What is the total plantation in Andhra Pradesh?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 70)

            result = rag.generate_answer(query, top_k=3)

            print(f"Answer:\n{result['answer']}")
            print()
            print(f"Sources used: {result['num_sources']}")
            print()

            if i < len(test_queries):
                print("="*70)
                print()

    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    test_rag()
