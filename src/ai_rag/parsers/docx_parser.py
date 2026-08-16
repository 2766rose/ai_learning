import io
import logging
from docx import Document
from ai_rag.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class DocxParser(BaseParser):
    def parse_bytes(self, content: bytes) -> str:
        try:
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            logger.exception("DOCX parse failed | size=%d", len(content))
            raise RuntimeError(f"DOCX parse failed: {e}") from e