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
    API_KEY: str = ""  # RAG_API_KEY: API access key (set in .env for deployments)
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: float = 60.0

    # Embedding 模型配置
    EMBED_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5")

    # ChromaDB 配置（RAG 知识库专用）
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "knowledge_base"
    CHROMA_VECTOR_SPACE: str = "cosine"
    # Chunking (moved from root config.py)
    CHUNK_SIZE: int = 300  # 与 ETL 实际使用一致
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    # 检索质量参数（评估实测调优，可用 RAG_ 前缀环境变量覆盖）
    MIN_SIMILARITY: float = 0.30
    RERANK_MIN_SCORE: float = 0.20
    VEC_GATE_SCORE: float = 0.55
    MAX_CHUNK_CHARS: int = 450
    MAX_FORMATTED_CHUNKS: int = 3
    # Agent 参数
    MAX_AGENT_ITERATIONS: int = 5
    HISTORY_TOKEN_BUDGET: int = 3000

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
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".docx", ".xlsx"}

    # Local Ollama 配置
    LOCAL_MODEL: str = "qwen3:8b"
    LOCAL_BASE_URL: str = "http://localhost:11434/v1"


rag_config = RAGConfig()

# ✅ 关键防呆：启动时打印实际解析的绝对路径，肉眼确认数据目录正确
print(f"[Config Init] PROJECT_ROOT      : {PROJECT_ROOT}")
print(f"[Config Init] ChromaDB Dir      : {rag_config.CHROMA_PERSIST_DIR}")
print(f"[Config Init] Memory DB Dir     : {rag_config.MEMORY_CHROMA_PERSIST_DIR}")  # ✅ 新增
print(f"[Config Init] Upload Dir        : {rag_config.UPLOAD_DIR}")