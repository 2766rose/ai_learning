"""config.py - 全局配置中心（唯一环境变量入口）"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    # LLM
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # Embedding
    EMBED_MODEL_PATH: str = "./models/bge-small-zh-v1.5"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K: int = 5
    CHROMA_COLLECTION_NAME: str = "enterprise_knowledge"
    CHROMA_VECTOR_SPACE: str = "cosine"

    # Upload
    UPLOAD_DIR: Path = Path("./uploads")
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB

    # Memory
    MEMORY_BACKEND: str = "redis"  # redis | memory
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL: int = 3600
    MEMORY_MAX_TOKENS: int = 4000

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "extra": "ignore"}


rag_config = RAGConfig()