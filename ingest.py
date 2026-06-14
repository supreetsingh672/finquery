"""
Run this once to parse, chunk, embed and index all RBI PDFs.
Re-run after adding new documents — already-processed files are skipped.

Usage:
    python ingest.py
    python ingest.py --force   # re-index everything from scratch
"""

import sys
import shutil
import argparse
from pathlib import Path

from src.config import RAW_DOCS_PATH, CHROMA_DB_PATH, BM25_INDEX_PATH, CACHE_PATH
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import create_chunks
from src.ingestion.indexer import (
    get_embedding_model, build_index, needs_reindex,
    _save_cache, _load_cache,
)


def main(force: bool = False) -> None:
    print("\n=== FinQuery Ingestion Pipeline ===\n")

    if force:
        print("Force mode: clearing existing index...")
        for path in [CHROMA_DB_PATH, str(BM25_INDEX_PATH), str(CACHE_PATH)]:
            shutil.rmtree(path, ignore_errors=True)
        print("Index cleared.\n")

    to_process, current_hashes = needs_reindex(str(RAW_DOCS_PATH))

    if not to_process:
        print("All documents are already indexed. Nothing to do.")
        print("Use --force to re-index from scratch.")
        return

    print(f"Found {len(to_process)} new/modified document(s) to process:\n")
    for p in to_process:
        print(f"  • {Path(p).name}")

    print("\n[1/3] Parsing PDFs...")
    all_pages = []
    for path in to_process:
        pages = parse_pdf(path)
        all_pages.extend(pages)
        print(f"  ✓ {Path(path).name} — {len(pages)} pages extracted")

    print(f"\n[2/3] Chunking {len(all_pages)} pages...")
    parents, children = create_chunks(all_pages)
    print(f"  ✓ Created {len(parents)} parent chunks, {len(children)} child chunks")

    print("\n[3/3] Embedding and indexing...")
    embedder = get_embedding_model()
    build_index(parents, children, embedder)

    cache = _load_cache()
    cache.update(current_hashes)
    _save_cache(cache)

    print("\n=== Ingestion Complete ===")
    print(f"  Parents stored : {len(parents)}")
    print(f"  Children in DB : {len(children)}")
    print(f"  Ready to query : python -m streamlit run app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinQuery ingestion pipeline")
    parser.add_argument("--force", action="store_true", help="Re-index all documents from scratch")
    args = parser.parse_args()
    main(force=args.force)
