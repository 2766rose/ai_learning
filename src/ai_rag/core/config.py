# src/ai_rag/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore"
    )

    # LLM 配置
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: float = 60.0

    # Embedding 模型配置
    EMBED_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5")

    # ChromaDB 配置（RAG 知识库专用）
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "knowledge_base"
    CHROMA_VECTOR_SPACE: str = "cosine"
    TOP_K: int = 5

    # ✅ 新增：长期记忆专用 ChromaDB 配置（与 RAG 知识库物理隔离）
    MEMORY_CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "memory_db")
    MEMORY_COLLECTION_NAME: str = "user_memories"

    # Redis & Celery 配置
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/0"
    MEMORY_BACKEND: str = "redis"
    SESSION_TTL: int = 3600
    MEMORY_MAX_TOKENS: int = 4096

    # 文件上传配置
    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    MAX_FILE_SIZE: int = 20 * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".docx"}

    # Local Ollama 配置
    LOCAL_MODEL: str = "qwen3:8b"
    LOCAL_BASE_URL: str = "http://localhost:11434/v1"


rag_config = RAGConfig()

# ✅ 关键防呆：启动时打印实际解析的绝对路径，肉眼确认数据目录正确
print(f"[Config Init] PROJECT_ROOT      : {PROJECT_ROOT}")
print(f"[Config Init] ChromaDB Dir      : {rag_config.CHROMA_PERSIST_DIR}")
print(f"[Config Init] Memory DB Dir     : {rag_config.MEMORY_CHROMA_PERSIST_DIR}")  # ✅ 新增
print(f"[Config Init] Upload Dir        : {rag_config.UPLOAD_DIR}")