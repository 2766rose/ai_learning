from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    @abstractmethod
    def parse_bytes(self, content: bytes) -> str:
        ...

    def parse_file(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return self.parse_bytes(path.read_bytes())