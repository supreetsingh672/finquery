from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, TOP_K_RERANK


_reranker: CrossEncoder = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, chunk in scored[:top_k]:
        chunk = dict(chunk)
        chunk["rerank_score"] = float(score)
        results.append(chunk)

    return results
