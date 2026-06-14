import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Suppress ChromaDB analytics noise
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

BASE_DIR = Path(__file__).parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
BM25_INDEX_PATH = BASE_DIR / "data" / "bm25_index"
CACHE_PATH = BASE_DIR / "data" / "cache"
RAW_DOCS_PATH = BASE_DIR / "data" / "raw"
PARENTS_STORE_PATH = BASE_DIR / "data" / "cache" / "parents.json"

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "4"))
CHUNK_SIZE_CHILD = int(os.getenv("CHUNK_SIZE_CHILD", "256"))
CHUNK_SIZE_PARENT = int(os.getenv("CHUNK_SIZE_PARENT", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

COLLECTION_NAME = "finquery_rbi"

DOCUMENT_NAMES = {
    "343MRC4FEBA35E0A8459587AC89804A80C936.pdf": "Tax Collection Scheme Circular 2016",
    "GUIDELINESDIGITALLENDINGD5C35A71D8124A0E92AEB940A7D25BB3.pdf": "Digital Lending Guidelines 2022",
    "MCIRACP535F1B4DE5494B4F82F69AB36B11538E.pdf": "Master Circular – IRACP Norms 2021",
    "MD18KYCF6E92C82E1E1419D87323E3869BC9F13.pdf": "KYC Master Direction 2016",
    "MD20D6FC6F31E8E5458F9E0411F433B7D40A.pdf": "Master Direction – Interest Rate on Advances",
    "RISK22042025.pdf": "Fraud Risk Management FAQs 2024",
}
