"""
Interactive Q&A Interface using Streamlit
Web UI for the LLM Document Q&A Pipeline
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from rag_chain import RAGChain

# Page configuration
st.set_page_config(
    page_title="LLM Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_chain():
    """Load RAG chain (cached for performance)"""
    try:
        return RAGChain()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


def process_query(query, rag, top_k, filter_metadata):
    """Process a query and display results"""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            if rag is None:
                st.error("RAG system not initialized. Please check the logs.")
                return

            # Generate answer
            result = rag.generate_answer(
                query, 
                top_k=top_k,
                filter_metadata=filter_metadata if filter_metadata else None
            )

            # Display answer
            st.markdown(result['answer'])

            # Display sources
            if result['sources']:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Source {i}:</strong><br>
                        <strong>File:</strong> {source['source']}<br>
                        <strong>State:</strong> {source['state']}<br>
                        <strong>Type:</strong> {source['type']}<br>
                        <strong>Preview:</strong> {source['content_preview']}
                        </div>
                        """, unsafe_allow_html=True)

            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "sources": result['sources']
            })


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 LLM Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about Indian government data on plantations and highways")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Number of sources
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More sources = more context but slower response"
        )

        # State filter
        st.subheader("📍 Filter by State")
        filter_state = st.selectbox(
            "Select a state (optional)",
            options=["All States", "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", 
                    "Delhi", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", 
                    "Jharkhand"],
            help="Filter results to a specific state"
        )

        # Data type filter
        st.subheader("📊 Filter by Data Type")
        filter_type = st.selectbox(
            "Select data type (optional)",
            options=["All Types", "plantation_statistics", "highway_funds", "highway_length"],
            help="Filter results by data category"
        )

        # About section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        This system uses:
        - **Google Gemini 1.5 Flash** for answers
        - **text-embedding-004** for search
        - **ChromaDB** for vector storage
        - **RAG** for accurate responses

        Data sources:
        - Plantation statistics (2015-2024)
        - Highway funds allocation
        - Highway length data
        """)

        # Stats
        st.markdown("---")
        st.subheader("📈 System Stats")
        try:
            rag = load_rag_chain()
            if rag:
                count = rag.collection.count()
                st.metric("Documents in Database", count)
        except:
            pass

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load RAG chain
    rag = load_rag_chain()

    # Build filter metadata
    filter_metadata = {}
    if filter_state != "All States":
        filter_metadata['state'] = filter_state
    if filter_type != "All Types":
        filter_metadata['type'] = filter_type

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        **Source {i}:**
                        - **File:** {source['source']}
                        - **State:** {source['state']}
                        - **Type:** {source['type']}
                        - **Preview:** {source['content_preview']}
                        """)

    # Chat input
    if query := st.chat_input("Ask a question about the data..."):
        process_query(query, rag, top_k, filter_metadata)

    # Sample questions (only show when no messages)
    if not st.session_state.messages:
        st.markdown("### 💡 Try these sample questions:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🌳 Which state has the highest plantation?", key="q1"):
                query = "Which state has the highest plantation progress from 2015 to 2024?"
                process_query(query, rag, top_k, filter_metadata)
                st.rerun()

            if st.button("📊 Compare Delhi and Haryana plantations", key="q2"):
                query = "Compare plantation progress between Delhi and Haryana"
                process_query(query, rag, top_k, filter_metadata)
                st.rerun()

        with col2:
            if st.button("🛣️ Tell me about highway funds", key="q3"):
                query = "What information is available about highway funds allocation?"
                process_query(query, rag, top_k, filter_metadata)
                st.rerun()

            if st.button("📈 Plantation trends over years", key="q4"):
                query = "What are the plantation trends from 2015 to 2024?"
                process_query(query, rag, top_k, filter_metadata)
                st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
