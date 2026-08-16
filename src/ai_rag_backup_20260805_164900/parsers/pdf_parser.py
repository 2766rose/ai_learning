import io
import logging
from PyPDF2 import PdfReader
from src.ai_rag.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    def parse_bytes(self, content: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(content))
            parts = [p.extract_text() for p in reader.pages if p.extract_text()]
            return "\n".join(parts)
        except Exception as e:
            logger.exception("PDF parse failed | size=%d", len(content))
            raise RuntimeError(f"PDF parse failed: {e}") from e