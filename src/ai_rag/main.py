# src/ai_rag/main.py
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ai_rag.core.lifespan import lifespan         
from ai_rag.core.logging_config import configure_logging 
from ai_rag.middleware.security import SecurityMiddleware       
from ai_rag.api.rag_router import router as rag_router
from ai_rag.api.conversation_router import router as conv_router      

configure_logging()

app = FastAPI(
    title="Enterprise RAG Knowledge Base",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(SecurityMiddleware)
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])
app.include_router(conv_router, prefix="/api/rag", tags=["Conversation"])

logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check():
    logger.info("health_check_passed")
    return {"status": "ok"}

# ===== 网页界面（放在最后挂载，不影响 /api 路由）=====
WEB_DIR = Path(__file__).resolve().parent / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")