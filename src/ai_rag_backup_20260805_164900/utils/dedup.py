# src/ai_rag/utils/dedup.py
from __future__ import annotations

import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    key_field: str = "text",
) -> List[Dict[str, Any]]:
    """
    基于内容哈希的精确去重（入库前调用）

    Args:
        chunks: 待去重的 chunk 列表，每个元素为 dict
        key_field: 用于计算哈希的文本字段名

    Returns:
        去重后的 chunk 列表（保持原始顺序，保留首次出现的条目）
    """
    seen_hashes: set[str] = set()
    unique_chunks: List[Dict[str, Any]] = []
    dup_count = 0

    for chunk in chunks:
        content = chunk.get(key_field, "").strip()
        if not content:
            # 空内容直接跳过，不写入向量库
            dup_count += 1
            continue

        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
        else:
            dup_count += 1

    if dup_count > 0:
        logger.warning(
            "⚠️ Chunk dedup | removed=%d | kept=%d | total_input=%d",
            dup_count, len(unique_chunks), len(chunks),
        )
    else:
        logger.info("✅ Chunk dedup | no duplicates | kept=%d", len(unique_chunks))

    return unique_chunks