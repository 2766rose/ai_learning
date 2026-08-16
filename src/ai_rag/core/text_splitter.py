# src/ai_rag/core/text_splitter.py
"""
Markdown-aware text splitter with header path inheritance and content type detection.
Backward compatible with original TextSplitter interface.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from ai_rag.core.config import rag_config

logger = logging.getLogger(__name__)


class TextSplitter:
    """Markdown-aware splitter that preserves section hierarchy and content semantics."""

    def __init__(
        self,
        chunk_size: int = rag_config.CHUNK_SIZE,
        chunk_overlap: int = rag_config.CHUNK_OVERLAP,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap({chunk_overlap}) must be < chunk_size({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Markdown header pattern: captures level and title text
        self._header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        # Content type detection patterns
        self._code_block_pattern = re.compile(r'^```', re.MULTILINE)
        self._table_pattern = re.compile(r'^\|.+\| $ ', re.MULTILINE)
        self._list_pattern = re.compile(r'^[\-\*\+]|\d+\.\s', re.MULTILINE)
        
        logger.info(
            "✅ MarkdownAware TextSplitter ready | size=%d | overlap=%d",
            chunk_size, chunk_overlap,
        )

    def _parse_headers(self, text: str) -> List[Tuple[int, str, int]]:
        """Extract all headers with their level, text, and position."""
        headers = []
        for match in self._header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            pos = match.start()
            headers.append((level, title, pos))
        return headers

    def _build_section_path(self, headers: List[Tuple[int, str, int]], pos: int) -> str:
        """Build full section path for a given position by inheriting parent headers."""
        path_parts = []
        current_levels = {}  # level -> title
        
        for level, title, header_pos in headers:
            if header_pos > pos:
                break
            current_levels[level] = title
            # Clear all child levels when a new parent appears
            for l in list(current_levels.keys()):
                if l > level:
                    del current_levels[l]
        
        # Build path in order
        for level in sorted(current_levels.keys()):
            prefix = "#" * level
            path_parts.append(f"{prefix} {current_levels[level]}")
            
        return " > ".join(path_parts) if path_parts else ""

    def _detect_content_type(self, text: str) -> str:
        """Detect semantic content type of a chunk."""
        stripped = text.strip()
        if not stripped:
            return "empty"
        if self._code_block_pattern.search(stripped):
            return "code_block"
        if self._table_pattern.search(stripped):
            return "table"
        if self._list_pattern.search(stripped):
            return "list"
        if stripped.startswith("#"):
            return "heading"
        return "paragraph"

    def _safe_split(self, text: str) -> List[str]:
        """Character-level split with semantic boundary awareness."""
        if not text or not text.strip():
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            
            # Avoid breaking inside code blocks or tables
            if end < text_len:
                # Prefer splitting at paragraph boundaries
                last_sep = max(
                    chunk.rfind(sep) for sep in ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
                )
                if last_sep > self.chunk_overlap:
                    chunk = chunk[:last_sep + 1]
                    end = start + last_sep + 1
                    
            chunks.append(chunk.strip())
            start = max(end - self.chunk_overlap, start + 1)  # Ensure progress
            
        return [c for c in chunks if c]  # Filter empty chunks

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks (backward compatible)."""
        return self._safe_split(text)

    def split_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> tuple[List[str], List[Dict]]:
        """
        Split texts with full Markdown awareness and enriched metadata.
        Backward compatible signature - ingestion_pipeline.py needs NO changes.
        """
        all_chunks: List[str] = []
        all_metadatas: List[Dict] = []

        for idx, text in enumerate(texts):
            parent_meta = (metadatas[idx] if metadatas else {}) or {}
            
            # Parse headers once per document
            headers = self._parse_headers(text)
            
            # Split into raw chunks first
            raw_chunks = self._safe_split(text)
            
            # Track approximate position for section path lookup
            char_pos = 0
            for chunk_idx, chunk in enumerate(raw_chunks):
                # Find section path based on chunk's starting position
                section_path = self._build_section_path(headers, char_pos)
                content_type = self._detect_content_type(chunk)
                
                all_chunks.append(chunk)
                all_metadatas.append({
                    **parent_meta,
                    "chunk_index": chunk_idx,
                    "parent_doc_index": idx,
                    "section_path": section_path,
                    "content_type": content_type,
                    "token_count": len(chunk) // 3,  # Rough estimate for CJK
                })
                
                char_pos += len(chunk) - self.chunk_overlap

        logger.info("✂️ Split %d docs → %d chunks (Markdown-aware)", len(texts), len(all_chunks))
        return all_chunks, all_metadatas


# Singleton instance - drop-in replacement
text_splitter = TextSplitter()