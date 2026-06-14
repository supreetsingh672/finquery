from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.ingestion.indexer import get_chroma_collection, load_bm25, _load_parents
from src.config import TOP_K_RETRIEVAL


def _vector_search(query: str, embedder: SentenceTransformer, k: int) -> List[Dict[str, Any]]:
    collection = get_chroma_collection()
    query_embedding = embedder.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "score": 1 - dist})
    return hits


def _bm25_search(query: str, k: int) -> List[Dict[str, Any]]:
    data = load_bm25()
    bm25, all_ids, all_docs = data["bm25"], data["ids"], data["docs"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            hits.append({
                "chunk_id": all_ids[idx],
                "text": all_docs[idx],
                "score": float(scores[idx]),
            })
    return hits


def _reciprocal_rank_fusion(
    vector_hits: List[Dict],
    bm25_hits: List[Dict],
    k: int = 60,
) -> List[str]:
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}

    for rank, hit in enumerate(vector_hits):
        cid = hit["metadata"]["parent_id"]  # deduplicate on parent
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, hit in enumerate(bm25_hits):
        collection = get_chroma_collection()
        # Get child's parent_id from ChromaDB metadata
        try:
            result = collection.get(ids=[hit["chunk_id"]], include=["metadatas"])
            if result["metadatas"]:
                parent_id = result["metadatas"][0]["parent_id"]
            else:
                continue
        except Exception:
            continue
        scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (k + rank + 1)

    ranked_parent_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked_parent_ids


def hybrid_search(
    queries: List[str],
    embedder: SentenceTransformer,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    """Run hybrid search over multiple query variants and return parent chunks."""
    all_vector_hits: List[Dict] = []
    all_bm25_hits: List[Dict] = []

    for query in queries:
        all_vector_hits.extend(_vector_search(query, embedder, k=top_k))
        all_bm25_hits.extend(_bm25_search(query, k=top_k))

    ranked_parent_ids = _reciprocal_rank_fusion(all_vector_hits, all_bm25_hits)
    parents_store = _load_parents()

    results = []
    seen = set()
    for parent_id in ranked_parent_ids[:top_k]:
        if parent_id in seen or parent_id not in parents_store:
            continue
        seen.add(parent_id)
        p = parents_store[parent_id]
        results.append({
            "parent_id": parent_id,
            "text": p["text"],
            "doc_name": p["doc_name"],
            "source_file": p["source_file"],
            "page_number": p["page_number"],
            "section_hint": p["section_hint"],
        })

    return results
