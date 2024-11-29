from pathlib import Path
import pickle
import os
import time
from typing import Any, Optional, Generator
from dataclasses import dataclass
import signal
from contextlib import contextmanager
import re  # Add this at the top with other imports
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds: int) -> Generator:
    """Context manager for adding timeout to operations."""
    def timeout_handler(signum, frame):
        raise TimeoutException("Operation timed out")

    # Set up timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

@dataclass
class Config:
    """Configuration settings for the RAG system.
    
    Parameters:
        pdf_dir: Directory containing PDF documents to process
        persist_directory: Directory for caching embeddings
        embeddings_model_name: Name of the model to use for embeddings
        llm_model_path: Path to the LLaMA model file
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks to maintain context
        temperature: Controls randomness in generation (0.0-1.0)
            0.0 = deterministic, 1.0 = most random
        max_tokens: Maximum number of tokens in generated response
        top_p: Nucleus sampling parameter (0.0-1.0)
            Controls the cumulative probability threshold for token sampling.
            For example, top_p=0.9 means only consider tokens whose cumulative
            probability mass is within the top 90% of all tokens.
            - Lower values (e.g., 0.1) = more focused/conservative responses
            - Higher values (e.g., 0.9) = more diverse/creative responses
            Default 1.0 considers all tokens.
        response_timeout: Maximum time to wait for response in seconds
        n_gpu_layers: Number of layers to offload to GPU
        context_window: Context window size
    """
    pdf_dir: Path = Path(os.getenv('PDF_DIRECTORY', 'data/tech'))
    persist_directory: str = "embeddings_cache"
    embeddings_model_name: str = "all-MiniLM-L12-v2"
    llm_model_path: str = os.getenv('LLM_MODEL_PATH', 'models/Llama-3.2-3B-Instruct-Q4_K_M.gguf')
    chunk_size: int = 900 # 15 minutes
    chunk_overlap: int = 50
    temperature: float = 0.2
    max_tokens: int = 20000
    top_p: float = 1.0
    response_timeout: int = 300  # Maximum time to wait for response in seconds
    n_gpu_layers: int = 1  # Number of layers to offload to GPU
    context_window: int = 10000  # Context window size

def get_or_create_embeddings(persist_directory: str = "embeddings_cache", model_name: str = "meta-llama/Llama-3.2-3B"):
    """Create or load cached embeddings."""
    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    cache_file = Path(persist_directory) / f"{safe_model_name}_embeddings.pkl"
    
    os.makedirs(persist_directory, exist_ok=True)
    
    if cache_file.exists():
        print(f"Loading embeddings from cache for model {model_name}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Creating new embeddings using model {model_name}...")
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    
    print(f"Saving embeddings to cache as {cache_file.name}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

class ChatHistory:
    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history
    
    def add_interaction(self, query: str, response: str):
        self.history.append({"query": query, "response": response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        return "\n".join([
            f"User: {h['query']}\nAssistant: {h['response']}"
            for h in self.history
        ])

class RAGSystem:
    """Main RAG (Retrieval-Augmented Generation) system implementation."""
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chat_history = ChatHistory()
        
    def setup(self) -> None:
        """Initialize the RAG system components."""
        print("\n=== Starting RAG System Setup ===")
        print("1. Validating directories...")
        self._validate_directory()
        
        print("2. Setting up embeddings...")
        self.embeddings = self._get_embeddings()
        
        print("3. Initializing vector store...")
        self._setup_vectorstore()
        
        print("4. Setting up LLM model...")
        self._setup_llm()
        print("=== Setup Complete ===\n")

    def _validate_directory(self) -> None:
        if not self.config.pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.config.pdf_dir}")
        if not self.config.pdf_dir.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {self.config.pdf_dir}")

    def _get_embeddings(self) -> Any:
        return get_or_create_embeddings(
            self.config.persist_directory,
            self.config.embeddings_model_name  # Use embeddings_model_name instead of model_name
        )

    def _setup_vectorstore(self) -> None:
        print("   - Loading PDF documents...")
        loader = PyPDFDirectoryLoader(str(self.config.pdf_dir))
        docs = loader.load()
        
        # Add metadata to track document sources
        for doc in docs:
            doc.metadata["source"] = str(doc.metadata.get("source", ""))
        
        print(f"   - Found {len(docs)} documents")
        
        print("   - Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)
        print(f"   - Created {len(chunks)} text chunks")
        
        print("   - Creating vector store...")
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings)
        print("   - Vector store creation complete")

    def _setup_llm(self) -> None:
        """Initialize the LLM with proper error handling."""
        try:
            print("   - Checking LLM model file...")
            if not os.path.exists(self.config.llm_model_path):
                raise FileNotFoundError(f"LLama model file not found at: {self.config.llm_model_path}")
            
            print("   - Initializing LLM...")
            self.llm = LlamaCpp(
                model_path=self.config.llm_model_path,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                n_ctx=self.config.context_window,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=True,
                n_threads=os.cpu_count(),  # Use all available CPU threads
                stream=False  # Disable streaming for better control
            )
            print("   - LLM initialization successful")
        except Exception as e:
            print("   - ERROR: LLM initialization failed!")
            raise RuntimeError(f"Failed to initialize LLama model: {str(e)}")

    def query(self, query_text: str) -> str:
        print("\n=== Processing Query ===")
        print(f"Query: {query_text}")
        start_time = time.time()
        
        try:
            with timeout(self.config.response_timeout):
                print("1. Setting up prompt template...")
                prompt_template = """
                You are a Technical Documentation Assistant engaged in an interactive conversation. 
                Previous conversation:
                {chat_history}

                Current context from documents:
                {context}
                
                Current question: {query}

                Answer: Please provide a comprehensive and detailed response based strictly on the provided context.
                Break down complex topics into clear explanations. Include relevant examples or specifications when available.
                Ensure thoroughness while maintaining clarity and relevance to the query.
                """
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_template)
                ])

                print("2. Configuring retriever...")
                # Configure retriever with MMR search
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance
                    search_kwargs={
                        'k': 4,
                        'fetch_k': 20,
                        'lambda_mult': 0.7  # Controls diversity (0.0-1.0)
                    }
                )

                print("3. Creating and executing RAG chain...")
                rag_chain = (
                    {
                        "context": retriever, 
                        "query": RunnablePassthrough(),
                        "chat_history": lambda _: self.chat_history.get_context()
                    }
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                print("4. Generating response...\n")
                response = rag_chain.invoke(query_text)
                
                # Clean up the response
                response = self._clean_response(response)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Add timing information to response
                response_with_timing = f"{response}\n\n[Response generated in {execution_time:.2f} seconds]"
                
                self.chat_history.add_interaction(query_text, response)
                print("\n=== Query Processing Complete ===")
                return response_with_timing

        except TimeoutException:
            end_time = time.time()
            error_msg = f"Query processing timed out after {self.config.response_timeout} seconds (elapsed: {end_time - start_time:.2f}s)"
            print(f"\nERROR: {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            end_time = time.time()
            error_msg = f"Error processing query: {str(e)} (elapsed: {end_time - start_time:.2f}s)"
            print(f"\nERROR: {error_msg}")
            return f"Error: {error_msg}"

    def _clean_response(self, response: str) -> str:
        """Clean up the response by removing unwanted text and formatting."""
        # Remove common template artifacts
        artifacts = [
            "Here is an example response:",
            "Best regards,",
            "Please note that",
            "If you need more assistance",
            "Let me know if you have any questions"
        ]
        
        cleaned = response
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, "")
            
        # Remove any remaining template instructions
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # Remove [Your Name] type text
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Replace multiple newlines
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)  # Trim whitespace
        
        return cleaned

def main():
    """Main entry point of the application."""
    print("\n=== Starting Interactive RAG Chat System ===")
    print("Setting up environment...")
    
    # Load environment variables from .env file
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        raise FileNotFoundError(
            "'.env' file not found. Please create one with required environment variables"
        )
    load_dotenv(env_path)
    
    # Validate required environment variables
    required_vars = ['HUGGINGFACEHUB_API_TOKEN', 'PDF_DIRECTORY', 'LLM_MODEL_PATH']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    print("Initializing configuration...")
    config = Config()
    rag_system = RAGSystem(config)
    
    # Remove the recursive main() call
    rag_system.setup()
    
    print("\nChat system ready! Type 'quit' to exit.")
    print("="*50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
            if not query:
                continue
                
            print("\nAssistant: ", end="")
            response = rag_system.query(query)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()