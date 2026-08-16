import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TxtParser:
    """解析纯文本文件，自动检测编码"""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".log"}

    def parse_file(self, file_path: str) -> str:
        path = Path(file_path)
        
        # 按优先级尝试多种编码
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        
        for encoding in encodings:
            try:
                text = path.read_text(encoding=encoding)
                logger.info(
                    "📄 TXT 解析完成: %s (%d chars, encoding=%s)",
                    file_path, len(text), encoding
                )
                return text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error("❌ TXT 解析失败: %s | error=%s", file_path, e, exc_info=True)
                raise ValueError(f"TXT 解析失败: {e}") from e
        
        raise ValueError(f"无法解码文件（已尝试编码: {encodings}）: {file_path}")