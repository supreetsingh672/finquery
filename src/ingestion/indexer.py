import json
import pickle
import hashlib
import os
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import (
    CHROMA_DB_PATH, BM25_INDEX_PATH, CACHE_PATH,
    PARENTS_STORE_PATH, COLLECTION_NAME, EMBEDDING_MODEL, RAW_DOCS_PATH,
)
from src.ingestion.chunker import ParentChunk, ChildChunk


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def _load_cache() -> Dict[str, str]:
    cache_file = CACHE_PATH / "processed.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return {}


def _save_cache(cache: Dict[str, str]) -> None:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    (CACHE_PATH / "processed.json").write_text(json.dumps(cache, indent=2))


def _load_parents() -> Dict[str, dict]:
    if Path(PARENTS_STORE_PATH).exists():
        return json.loads(Path(PARENTS_STORE_PATH).read_text())
    return {}


def _save_parents(parents_store: Dict[str, dict]) -> None:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    Path(PARENTS_STORE_PATH).write_text(json.dumps(parents_store, indent=2))


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def build_index(
    parents: List[ParentChunk],
    children: List[ChildChunk],
    embedder: SentenceTransformer,
) -> None:
    collection = get_chroma_collection()
    BM25_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    # Save parents to JSON store
    parents_store = _load_parents()
    for p in parents:
        parents_store[p.chunk_id] = {
            "text": p.text,
            "doc_name": p.doc_name,
            "source_file": p.source_file,
            "page_number": p.page_number,
            "section_hint": p.section_hint,
            "child_ids": p.child_ids,
        }
    _save_parents(parents_store)

    # Embed and store child chunks in ChromaDB in batches
    BATCH = 64
    texts = [c.text for c in children]
    ids = [c.chunk_id for c in children]
    metadatas = [{
        "doc_name": c.doc_name,
        "source_file": c.source_file,
        "page_number": c.page_number,
        "parent_id": c.parent_id,
        "section_hint": c.section_hint,
    } for c in children]

    print(f"Embedding {len(children)} child chunks...")
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch_texts = texts[i:i + BATCH]
        batch_ids = ids[i:i + BATCH]
        batch_meta = metadatas[i:i + BATCH]
        embeddings = embedder.encode(batch_texts, normalize_embeddings=True).tolist()
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    # Build BM25 index over all child chunks
    _rebuild_bm25(collection)


def _rebuild_bm25(collection) -> None:
    print("Building BM25 index...")
    result = collection.get(include=["documents", "metadatas"])
    all_docs = result["documents"]
    all_ids = result["ids"]
    tokenized = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)

    BM25_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH / "bm25.pkl", 'wb') as f:
        pickle.dump({"bm25": bm25, "ids": all_ids, "docs": all_docs}, f)
    print(f"BM25 index built over {len(all_docs)} chunks.")


def load_bm25():
    bm25_path = BM25_INDEX_PATH / "bm25.pkl"
    if not bm25_path.exists():
        raise FileNotFoundError("BM25 index not found. Run ingest.py first.")
    with open(bm25_path, 'rb') as f:
        return pickle.load(f)


def needs_reindex(raw_docs_path: str) -> tuple[List[str], Dict[str, str]]:
    cache = _load_cache()
    to_process = []
    current_hashes = {}

    for fname in sorted(os.listdir(raw_docs_path)):
        if not fname.endswith('.pdf'):
            continue
        path = os.path.join(raw_docs_path, fname)
        h = _file_hash(path)
        current_hashes[fname] = h
        if cache.get(fname) != h:
            to_process.append(path)

    return to_process, current_hashes
