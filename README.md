# RAG Chat System: Intelligent PDF Document Interaction

## Overview
A sophisticated Retrieval-Augmented Generation (RAG) system enabling interactive chat with PDF documents using advanced language modeling and vector embeddings.

## 🌟 Features
- **Advanced Document Processing**
  - PDF document parsing and intelligent chunking
  - Metadata-preserved text extraction
  - Flexible document source management

- **Intelligent Retrieval**
  - Vector-based document retrieval with ChromaDB
  - Semantic search capabilities
  - Maximum Marginal Relevance (MMR) document ranking

- **Language Model Integration**
  - Configurable LLaMA model support
  - Context-aware response generation
  - GPU acceleration for enhanced performance

- **Robust Chat Interface**
  - Conversation history tracking
  - Dynamic context management
  - Query timeout protection

## 🔧 Prerequisites
- **System Requirements**
  - Python 3.8+
  - Minimum 16GB RAM recommended
  - Optional: CUDA-compatible GPU for accelerated processing

- **Software Dependencies**
  - PyTorch
  - Transformers
  - ChromaDB
  - SentenceTransformers
  - PyPDF libraries

## 📂 Project Structure
```
rag_experiment/
├── data/
│   └── tech/             # PDF document repository
├── models/               # LLaMA model storage
├── embeddings_cache/     # Generated embedding cache
├── requirements.txt
└── rag_experiment.py
```

## 🚀 Quick Setup

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ThammanaSrinivas/RAG_experiment
cd rag-chat-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Model and Document Preparation
- Download LLaMA model (Llama-3.2-3B-Instruct-Q4_K_M.gguf)
- Place model in `models/` directory
- Add PDF documents to `data/tech/` directory

## ⚙️ Configuration
Customize system behavior through the `Config` class:

```python
config = Config(
    pdf_dir=Path("data/tech"),
    chunk_size=900,           # Semantic chunk size
    chunk_overlap=50,          # Chunk overlap for context preservation
    temperature=0.2,           # Response creativity
    max_tokens=20000,          # Maximum response length
    response_timeout=300,      # Query timeout
    n_gpu_layers=1             # GPU acceleration layers
)
```

## 🖥️ Usage
```bash
python rag_experiment.py
```
- Interactive chat interface launches
- Ask questions about your documents
- Type 'quit' to exit

## 🏗️ System Architecture

### Document Processing
- **Loader**: PyPDFDirectoryLoader
- **Text Splitting**: RecursiveCharacterTextSplitter
- **Metadata Tracking**: Source document preservation

### Embedding System
- **Embedding Model**: SentenceTransformer (all-MiniLM-L12-v2)
- **Vector Storage**: ChromaDB
- **Caching**: Persistent embedding cache

### Language Model
- **Model**: LLaMA
- **Features**:
  - GPU acceleration
  - Timeout protection
  - Context-aware responses

## 🚀 Performance Optimization
- Leverage GPU for faster processing
- Utilize embedding cache
- Adjust chunk size for optimal retrieval
- Monitor GPU memory usage

## 🛠️ Troubleshooting
- **Common Issues**
  - Insufficient GPU memory
  - Model compatibility
  - Large document processing delays

- **Recommended Actions**
  - Update GPU drivers
  - Verify model and library versions
  - Reduce chunk size or document volume

## 📋 Future Roadmap
- Multi-language support
- Enhanced embedding models
- Improved context window management
- Advanced query preprocessing

## 🤝 Contributing
Contributions welcome! Please read our contributing guidelines before submitting pull requests.

## 📬 Contact
gmail : sreenivast84@gmail.com
