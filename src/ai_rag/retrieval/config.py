from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    rrf_k: int = 60
    bm25_top_k: int = 50
    vector_top_k: int = 50
    final_top_k: int = 20
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
