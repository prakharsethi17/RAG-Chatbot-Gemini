# LLM Document Q&A Pipeline
**End-to-End RAG System with Google Gemini API**

A production-ready Document Question-Answering system demonstrating complete ML pipeline from data acquisition to interactive Q&A using Retrieval Augmented Generation (RAG).

## Live Demo

**Try it now:** [https://rag-chatbot-gemini-ps.streamlit.app/](https://rag-chatbot-gemini-ps.streamlit.app/)

---

## Project Overview

This project showcases a complete LLM engineering pipeline with:
- **Data Acquisition**: Automated download from Indian Government Open Data APIs  
- **Multi-Format Processing**: CSV and XML parsing with robust error handling
- **Embedding Generation**: Google Gemini text-embedding-004 (768-dimensional vectors)
- **Vector Search**: ChromaDB for semantic similarity search
- **LLM Integration**: Gemini 2.5 Flash for answer generation
- **Interactive UI**: Streamlit-based web interface with chat functionality

### Data Sources
- **Plantation Statistics**: State-wise tree plantation data (2015-2024) - 10 documents
- **Highway Funds**: National highway budget allocation and expenditure (2023-2025) - 10 documents  
- **Highway Length**: State-wise national highway length data (2024) - 10 documents
- **Total**: 30 documents across 3 government datasets

---

## Quick Start (Windows)

### Prerequisites
- Python 3.9+ installed
- Google Gemini API key (free at https://ai.google.dev/)

### Installation

1. **Create project directory**
```bash
mkdir llm-doc-qa-pipeline
cd llm-doc-qa-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure API key**
```bash
copy .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_actual_api_key_here
```

### Running the Pipeline

**Step 1: Download Data**
```bash
cd src\data_download
python downloader.py
```

**Step 2: Process Data**
```bash
cd ..\data_processing
python processor.py
```

**Step 3: Generate Embeddings**
```bash
cd ..\embeddings
python embed_documents.py
```

**Step 4: Launch Web Interface**
```bash
cd ..\rag
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Project Structure

```
llm-doc-qa-pipeline/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies  
├── .env.example                        # Environment template
├── .env                               # API keys (create manually)
│
├── data/
│   ├── raw/                           # Downloaded datasets
│   │   ├── plantation_data.csv        # State plantation data
│   │   ├── highway_funds.xml          # Highway budget data
│   │   └── highway_length.xml         # Highway length data
│   └── processed/                     # Cleaned JSON documents
│
├── src/
│   ├── data_download/
│   │   └── downloader.py              # API data acquisition
│   ├── data_processing/
│   │   └── processor.py               # CSV/XML parsing & cleaning
│   ├── embeddings/
│   │   └── embed_documents.py         # Gemini embedding generation
│   └── rag/
│       ├── rag_chain.py               # RAG implementation  
│       └── app.py                     # Streamlit web interface
│
├── vectorstore/                       # ChromaDB vector database
├── logs/                             # Execution logs
└── requirements.txt                   # Dependencies
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Google Gemini 2.5 Flash | Answer generation |
| **Embeddings** | text-embedding-004 | Semantic search (768-dim) |
| **Vector DB** | ChromaDB | Document storage & retrieval |
| **Framework** | LangChain | RAG orchestration |
| **UI** | Streamlit | Interactive web interface |
| **Data Processing** | Pandas, BeautifulSoup | Multi-format parsing |
| **APIs** | Google AI, Data.gov.in | LLM access, data acquisition |

---

## Features

### Interactive Chat Interface
- **ChatGPT-style UI** with conversation history
- **Source attribution** with expandable citations  
- **Adjustable retrieval** (1-10 sources per query)
- **Smart filtering** by state or data type
- **Sample questions** for easy exploration

### Advanced RAG Pipeline  
- **Semantic search** with metadata filtering
- **Context-aware prompts** with source tracking
- **Intelligent chunking** with sentence boundaries
- **Error handling** with graceful fallbacks
- **Performance optimization** with caching

### Production Features
- **Comprehensive logging** across all components
- **Retry logic** with exponential backoff
- **Rate limiting** for API compliance
- **Metadata preservation** for traceability
- **Multi-format support** (CSV, XML, future PDF)

---

## Sample Queries

**Plantation Questions:**
- "Which state has the highest plantation progress?"
- "Compare plantation between Delhi and Haryana"
- "What are the plantation trends from 2015 to 2024?"

**Highway Budget Questions:**
- "Which state received the highest highway budget allocation?"
- "Compare highway expenditure between states"
- "Show me highway fund utilization for 2023-24"

**Highway Length Questions:**
- "Which state has the longest national highway network?"  
- "What is the total highway length in Gujarat?"
- "Compare highway infrastructure across states"

---

## Troubleshooting

### Common Issues

**API Key Error**
```
ERROR: GOOGLE_API_KEY not found
```
**Solution:** Add your Gemini API key to `.env` file

**Vector Store Not Found**
```
Vector store not found
```
**Solution:** Complete Step 3 (embedding generation) first

---

## Performance Metrics

- **Documents**: 30 processed documents across 3 datasets
- **Embedding Dimensions**: 768 (text-embedding-004)
- **Average Response Time**: 2-5 seconds per query
- **Vector Search**: Sub-second retrieval from ChromaDB
- **API Usage**: Optimized for Gemini free tier (1,500 requests/day)

---

## Development

### Requirements
All dependencies are specified in `requirements.txt`:
```
langchain
langchain-google-genai
chromadb
streamlit
pandas
google-generativeai
python-dotenv
```

### Environment Variables
```bash
# .env file
GOOGLE_API_KEY=your_gemini_api_key_here
DATA_GOV_API_KEY=your_data_gov_in_api_key_here
PROJECT_NAME=LLM Document Q&A Pipeline
LOG_LEVEL=INFO
```

### Testing
Test individual components:
```bash
# Test data download
python src/data_download/downloader.py

# Test data processing  
python src/data_processing/processor.py

# Test RAG chain
python src/rag/rag_chain.py
```

---

## Deployment

### Local Development
```bash
streamlit run src/rag/app.py
```

### Streamlit Community Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Community Cloud
3. Add `GOOGLE_API_KEY` to secrets
4. Deploy with one click

### Docker (Optional)
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "src/rag/app.py"]
```

---

## 🎓 Learning Outcomes

This project demonstrates:

**Technical Skills:**
- End-to-end ML pipeline development
- RAG architecture implementation  
- Vector database integration
- LLM API integration and prompt engineering
- Multi-format data processing
- Web application development
- Production error handling and logging

**Engineering Practices:**
- Modular code architecture
- Environment configuration management
- Comprehensive documentation
- Error handling and retry logic
- Performance optimization
- User experience design

---

## License

MIT License - Free for personal and commercial use

---

## Author

**Prakhar Sethi**
- GitHub: [@prakharsethi17](https://github.com/prakharsethi17)
- LinkedIn: [linkedin.com/in/prakhar17/](https://www.linkedin.com/in/prakhar17/)  
- Email: prakhar.robotics@gmail.com

---

## Acknowledgments

- **Google Gemini API** for free tier access
- **India Government Open Data Platform** for public datasets
- **LangChain Community** for RAG framework
- **Streamlit Team** for intuitive UI framework
- **ChromaDB** for vector database solution

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs in the `logs/` directory  
3. Open an issue on GitHub
4. Contact via email for urgent matters
