from typing import TypedDict, Dict, Any

class RetrievedDoc(TypedDict):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
