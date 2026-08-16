# src/ai_rag/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",  
        extra="ignore"
    )

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: float = 60.0

    EMBED_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5")

    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "knowledge_base"
    CHROMA_VECTOR_SPACE: str = "cosine"
    TOP_K: int = 5

    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/1"
    MEMORY_BACKEND: str = "redis"
    SESSION_TTL: int = 3600
    MEMORY_MAX_TOKENS: int = 4096

    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    MAX_FILE_SIZE: int = 20 * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".docx"}


rag_config = RAGConfig()