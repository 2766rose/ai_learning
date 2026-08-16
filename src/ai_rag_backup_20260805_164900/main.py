# src/ai_rag/main.py
import logging
from fastapi import FastAPI

from src.ai_rag.core.lifespan import lifespan
from src.ai_rag.core.logging_config import configure_logging
from src.ai_rag.middleware.security import SecurityMiddleware
from src.ai_rag.api.rag_router import router as rag_router

configure_logging()

app = FastAPI(
    title="Enterprise RAG Knowledge Base",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(SecurityMiddleware)
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])

logger = logging.getLogger(__name__)


@app.get("/health")
async def health_check():
    logger.info("health_check_passed")
    return {"status": "ok"}