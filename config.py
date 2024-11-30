from pathlib import Path
from pydantic import BaseModel
from utils import get_gpu_info

class Config(BaseModel):
    """Application settings."""
    pdf_dir: Path = Path('data/tech')
    persist_directory: str = "embeddings_cache"
    embeddings_model_name: str = "all-MiniLM-L12-v2"
    # embeddings_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_model_path: Path = Path('models/Llama-3.2-3B-Instruct-Q4_K_M.gguf')
    chunk_size: int = 900
    chunk_overlap: int = 50
    temperature: float = 0.2
    max_tokens: int = 2048  # adjusted down from 4096
    top_p: float = 1.0
    response_timeout: int = 300
    context_window: int = 2048  # increased from 512
    max_input_tokens: int = 1536  # increased from 384
    gpu_config: dict = {}  # Add this field

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gpu_config = self._get_gpu_config()

    def _get_gpu_config(self) -> dict:
        gpu_info = get_gpu_info()
        if not gpu_info:
            return {
                "n_gpu_layers": 0,
                "main_gpu": -1,
                "n_batch": 64,
                "n_ctx": self.context_window
            }
            
        # More conservative GPU settings
        base_config = {
            "n_gpu_layers": -1,
            "main_gpu": gpu_info["device"],
            "n_ctx": self.context_window,
            "gpu_memory_utilization": 0.8,  # Reduced from 0.98
            "threads": -1,
            "use_mlock": False  # Changed to False
        }
        
        if gpu_info["memory"] > 23:  # For high-end GPUs (24GB+)
            return {**base_config, "n_batch": 512}  # Reduced from 4096
        return {**base_config, "n_batch": 128}  # Reduced from 1024
