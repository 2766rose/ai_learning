# src/ai_rag/parsers/pdf_parser.py
"""PDF 解析器 v2：文本 + 复杂表格 → Markdown

- 使用 PyMuPDF 提取文本，并用 find_tables() 识别复杂表格
- 表格自动转换为 Markdown 表格（合并单元格前向填充，便于检索和 LLM 理解）
- 若 PyMuPDF 不可用，降级回 PyPDF2 纯文本提取
"""
import io
import logging

from ai_rag.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """PDF → Markdown（文本 + 表格）"""

    def parse_bytes(self, content: bytes) -> str:
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            logger.warning("pymupdf 不可用，降级为 PyPDF2 纯文本提取")
            return self._parse_with_pypdf(content)

        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            parts = [self._page_to_markdown(page) for page in doc]
            doc.close()
            return "\n\n".join(p for p in parts if p.strip())
        except Exception as e:
            logger.exception("PDF parse failed (pymupdf) | size=%d", len(content))
            raise RuntimeError(f"PDF parse failed: {e}") from e

    # ------------------------------------------------------------------
    #  PyMuPDF 主路径
    # ------------------------------------------------------------------
    def _page_to_markdown(self, page) -> str:
        """单页 → Markdown：按纵向位置合并文本块与表格，避免重复输出表格文字"""
        items = []  # (y0, kind, payload)

        # 1. 文本块
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text = (text or "").strip()
            if not text:
                continue
            # 跳过完全落在表格区域内的文本块（表格会以 Markdown 表格单独输出）
            if any(self._inside_bbox((x0, y0, x1, y1), tb) for tb in self._table_bboxes(page)):
                continue
            items.append((y0, "text", text))

        # 2. 表格
        for table in page.find_tables().tables:
            md = self._table_to_markdown(table.extract())
            if md:
                items.append((table.bbox[1], "table", md))

        items.sort(key=lambda x: x[0])
        return "\n\n".join(payload for _, _, payload in items)

    @staticmethod
    def _table_bboxes(page):
        return [t.bbox for t in page.find_tables().tables]

    @staticmethod
    def _inside_bbox(block, table_bbox, tolerance: float = 2.0) -> bool:
        bx0, by0, bx1, by1 = block
        tx0, ty0, tx1, ty1 = table_bbox
        return (
            bx0 >= tx0 - tolerance and bx1 <= tx1 + tolerance
            and by0 >= ty0 - tolerance and by1 <= ty1 + tolerance
        )

    @staticmethod
    def _table_to_markdown(rows) -> str:
        """二维表格数据 → Markdown 表格（合并单元格前向填充）"""
        if not rows:
            return ""

        # 规范化列数
        ncols = max(len(r) for r in rows)
        grid = [
            ["" if c is None else str(c).replace("\n", " ").strip() for c in r]
            + [""] * (ncols - len(r))
            for r in rows
        ]
        # 删除全空行
        grid = [r for r in grid if any(c for c in r)]
        if len(grid) < 2:  # 至少需要表头 + 一行
            return ""

        # 合并单元格前向填充（如"差旅费"跨多行时，每行都保留该值）
        prev = [""] * ncols
        for r in grid:
            for i in range(ncols):
                if r[i]:
                    prev[i] = r[i]
                else:
                    r[i] = prev[i]

        lines = [
            "| " + " | ".join(grid[0]) + " |",
            "| " + " | ".join(["---"] * ncols) + " |",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  降级路径（PyPDF2 纯文本）
    # ------------------------------------------------------------------
    def _parse_with_pypdf(self, content: bytes) -> str:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        parts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(parts)