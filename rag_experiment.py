import os
import torch
from pathlib import Path
import pickle
import platform
import threading
import signal
import time
from typing import Any, Optional, Generator
from dataclasses import dataclass
from contextlib import contextmanager
import re
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
    """Cross-platform timeout context manager."""
    def timeout_handler():
        raise TimeoutException("Operation timed out")

    if platform.system() != 'Windows':
        # Unix-like systems can use SIGALRM
        def alarm_handler(signum, frame):
            raise TimeoutException("Operation timed out")
        
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows systems use threading
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

def get_gpu_config():
    """Determine GPU configuration based on available hardware."""
    print("\nGPU Detection:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        # Force CUDA computation mode
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.init()
        current_device = torch.cuda.current_device()
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        print(f"GPU detected: {torch.cuda.get_device_name(current_device)}")
        print(f"Available GPU memory: {gpu_memory:.2f} GB")
        print(f"Using CUDA device: {current_device}")
        
        # Generic configuration based on available memory
        if gpu_memory > 23:  # High-end GPUs (>24GB)
            return {
                "n_gpu_layers": -1,      # Use all layers
                "main_gpu": current_device,
                "tensor_split": None,
                "n_batch": 512,
                "f16_kv": True,
                "use_mmap": False,
                "use_mlock": True,
                "offload_kqv": True,
                "context_length": 4096,
                "gpu_memory_utilization": 0.9
            }
        elif gpu_memory > 7:  # Mid-range GPUs (8-24GB)
            return {
                "n_gpu_layers": 32,
                "main_gpu": current_device,
                "tensor_split": None,
                "n_batch": 256,
                "f16_kv": True,
                "use_mmap": False,
                "use_mlock": True,
                "offload_kqv": True,
                "context_length": 2048,
                "gpu_memory_utilization": 0.8
            }
        else:  # Low-memory GPUs (<8GB)
            return {
                "n_gpu_layers": 20,
                "main_gpu": current_device,
                "tensor_split": None,
                "n_batch": 128,
                "f16_kv": True,
                "use_mmap": False,
                "use_mlock": True,
                "offload_kqv": True,
                "context_length": 1024,
                "gpu_memory_utilization": 0.7
            }
    return {
        "n_gpu_layers": 0,  # CPU only
        "main_gpu": -1,
        "tensor_split": None,
        "n_batch": 64
    }

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
    pdf_dir: Path = Path(os.getenv('PDF_DIRECTORY', os.path.join('data', 'tech')))
    persist_directory: str = "embeddings_cache"
    embeddings_model_name: str = "all-MiniLM-L12-v2"
    llm_model_path: str = os.getenv('LLM_MODEL_PATH', os.path.join('models', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'))
    chunk_size: int = 900 # 15 minutes
    chunk_overlap: int = 50
    temperature: float = 0.2
    max_tokens: int = 20000
    top_p: float = 1.0
    response_timeout: int = 300  # Maximum time to wait for response in seconds
    n_gpu_layers: int = 1  # Number of layers to offload to GPU
    context_window: int = 10000  # Context window size
    gpu_config: dict = None
    
    def __post_init__(self):
        self.gpu_config = get_gpu_config()

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
        """Initialize the LLM with GPU support when available."""
        try:
            print("\nLLM GPU Setup:")
            if torch.cuda.is_available():
                print(f"CUDA Compute Mode: Enabled")
                print(f"CUDA Device Properties: {torch.cuda.get_device_properties(0)}")
                print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            print("   - Checking LLM model file...")
            model_path = Path(self.config.llm_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"LLama model file not found at: {model_path}")
            if not model_path.is_file():
                raise ValueError(f"Path exists but is not a file: {model_path}")
            
            print(f"   - Model file size: {model_path.stat().st_size / (1024*1024):.2f}MB")
            print("   - Initializing LLM with GPU configuration...")
            
            try:
                self.llm = LlamaCpp(
                    model_path=str(model_path),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    n_ctx=self.config.gpu_config.get("context_length", self.config.context_window),
                    n_gpu_layers=self.config.gpu_config["n_gpu_layers"],
                    main_gpu=self.config.gpu_config["main_gpu"],
                    tensor_split=self.config.gpu_config["tensor_split"],
                    n_batch=self.config.gpu_config["n_batch"],
                    f16_kv=self.config.gpu_config.get("f16_kv", True),
                    use_mmap=self.config.gpu_config.get("use_mmap", False),
                    use_mlock=self.config.gpu_config.get("use_mlock", True),
                    offload_kqv=self.config.gpu_config.get("offload_kqv", True),
                    verbose=True,
                    n_threads=os.cpu_count(),
                    gpu_memory_utilization=self.config.gpu_config.get("gpu_memory_utilization", 0.8)
                )
            except Exception as model_error:
                print(f"\nDetailed error loading model:")
                print(f"Error type: {type(model_error).__name__}")
                print(f"Error message: {str(model_error)}")
                raise RuntimeError(f"Failed to initialize LLama model: {str(model_error)}")
            
            if torch.cuda.is_available():
                print(f"GPU Memory After LLM Load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            print(f"   - LLM initialization successful with following GPU config:")
            print(f"     - GPU Layers: {self.config.gpu_config['n_gpu_layers']}")
            print(f"     - Batch Size: {self.config.gpu_config['n_batch']}")
            print(f"     - Context Length: {self.config.gpu_config.get('context_length', self.config.context_window)}")
            print(f"     - GPU Memory Utilization: {self.config.gpu_config.get('gpu_memory_utilization', 0)*100}%")
            
            # Add extra GPU verification
            if torch.cuda.is_available():
                print("\nGPU Verification:")
                print(f"GPU Memory After LLM Load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
                print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(0)/1024**2:.2f}MB")
                
                # Test CUDA computation
                test_tensor = torch.randn(1000, 1000).cuda()
                test_result = torch.matmul(test_tensor, test_tensor)
                print(f"GPU Test Computation Successful: {test_result.device}")
                del test_tensor, test_result
                torch.cuda.empty_cache()
        except Exception as e:
            print("   - ERROR: LLM initialization failed!")
            print(f"   - Error details: {str(e)}")
            raise RuntimeError(f"Failed to initialize LLama model: {str(e)}")

    def query(self, query_text: str) -> str:
        print("\n=== Processing Query ===")
        print(f"Query: {query_text}")
        start_time = time.time()
        
        if torch.cuda.is_available():
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
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
                if torch.cuda.is_available():
                    print(f"GPU Memory before inference: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                
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
                
                if torch.cuda.is_available():
                    print(f"GPU Memory after inference: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                    print(f"Max GPU Memory: {torch.cuda.max_memory_allocated()/1024**2:.2f}MB")

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
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
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
