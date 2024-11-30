import logging
import torch
import gc
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from utils import timeout, get_or_create_embeddings

logger = logging.getLogger(__name__)

class ChatHistory:
    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history
    
    def add(self, query: str, response: str):
        self.history.append({"query": query, "response": response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        return "\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in self.history])

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chat_history = ChatHistory()
        
    def setup(self):
        """Initialize RAG system components."""
        logger.info("Setting up RAG System")
        self._validate_directory()
        self.embeddings = get_or_create_embeddings(
            self.config.persist_directory,
            self.config.embeddings_model_name
        )
        self._setup_vectorstore()
        self._setup_llm()

    def _validate_directory(self):
        if not self.config.pdf_dir.exists() or not self.config.pdf_dir.is_dir():
            raise FileNotFoundError(f"Invalid PDF directory: {self.config.pdf_dir}")

    def _setup_vectorstore(self):
        loader = PyPDFDirectoryLoader(str(self.config.pdf_dir))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings)

    def _setup_llm(self):
        model_path = Path(self.config.llm_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        gpu_params = {
            "n_gpu_layers": 0,
            "n_batch": 64,
            "n_ctx": self.config.context_window  # Move n_ctx here
        }

        if torch.cuda.is_available():
            try:
                logger.info("GPU detected - configuring for GPU inference")
                # Clear CUDA cache
                torch.cuda.empty_cache()
                gc.collect()
                
                gpu_params = {
                    **self.config.gpu_config,  # This already includes n_ctx
                    "f16_kv": True,
                    "mmap": True
                }
                logger.info(f"GPU Configuration: {gpu_params}")
            except Exception as e:
                logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
                gpu_params = {
                    "n_gpu_layers": 0,
                    "n_batch": 64,
                    "n_ctx": self.config.context_window
                }

        self.llm = LlamaCpp(
            model_path=str(model_path),
            temperature=self.config.temperature,
            max_tokens=min(self.config.max_tokens, self.config.context_window - self.config.max_input_tokens),
            top_p=self.config.top_p,
            verbose=True,
            **gpu_params
        )

    def query(self, query_text: str) -> str:
        # Clear GPU memory before query
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        with timeout(self.config.response_timeout):
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={'k': 1, 'fetch_k': 5}  # reduced number of retrieved documents
                )

                # Get retrieval context
                retrieved_docs = retriever.get_relevant_documents(query_text)
                context = "\n".join(doc.page_content for doc in retrieved_docs)
                
                # Truncate context if too long (rough estimation)
                max_context_chars = self.config.max_input_tokens * 4  # rough char to token ratio
                if len(context) > max_context_chars:
                    context = context[:max_context_chars] + "..."

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer based on the following context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:")
                ])

                chat_history = self.chat_history.get_context()
                # Limit chat history length
                if chat_history:
                    chat_history = chat_history[-1000:]  # keep only last 1000 chars

                rag_chain = (
                    {"context": lambda _: context, 
                     "query": RunnablePassthrough(),
                     "chat_history": lambda _: chat_history}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(query_text)
                self.chat_history.add(query_text, response)
                return response
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error occurred: {e}")
                    # Clear memory and try again with smaller batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.config.gpu_config["n_batch"] //= 2
                    self._setup_llm()
                    return self.query(query_text)
                raise e