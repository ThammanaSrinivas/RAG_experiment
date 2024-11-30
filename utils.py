import platform
import signal
import threading
from contextlib import contextmanager
from typing import Generator
import torch
import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds: int) -> Generator:
    """Cross-platform timeout context manager."""
    def timeout_handler():
        raise TimeoutException("Operation timed out")
        
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, lambda *_: TimeoutException("Operation timed out"))
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

def get_or_create_embeddings(persist_directory: str, model_name: str):
    """Create or load cached embeddings."""
    cache_file = Path(persist_directory) / f"{model_name.replace('/', '_').replace('-', '_')}_embeddings.pkl"
    os.makedirs(persist_directory, exist_ok=True)
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    embeddings = SentenceTransformerEmbeddings(model_name)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def get_gpu_info():
    """Get GPU configuration and status."""
    if not torch.cuda.is_available():
        return None
        
    torch.cuda.init()
    current_device = torch.cuda.current_device()
    
    # Get memory information
    allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
    reserved = torch.cuda.memory_reserved(current_device) / (1024**3)
    
    return {
        "device": current_device,
        "name": torch.cuda.get_device_name(current_device),
        "memory": torch.cuda.get_device_properties(current_device).total_memory / (1024**3),
        "compute_mode": torch.cuda.get_device_capability(current_device),
        "memory_allocated": allocated,
        "memory_reserved": reserved
    }

def print_gpu_utilization():
    """Print current GPU utilization"""
    if not torch.cuda.is_available():
        return
        
    info = get_gpu_info()
    print(f"\nGPU Utilization:")
    print(f"Device: {info['name']}")
    print(f"Memory Allocated: {info['memory_allocated']:.2f} GB")
    print(f"Memory Reserved: {info['memory_reserved']:.2f} GB")
    print(f"Total Memory: {info['memory']:.2f} GB")