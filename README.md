# FinQuery — RAG over RBI Regulatory Documents

> Ask questions about Indian financial regulations in plain English and get precise answers with citations pointing to the exact source document and page.

---

## Why This Exists

RBI issues 100+ circulars annually. General LLMs have a knowledge cutoff and miss recent ones. Hallucination risk is high in regulatory contexts. FinQuery answers must cite the exact document and page so users can verify — making it audit-trail ready.

---

## Demo

```
Q: What are the key requirements for digital lending apps?

A: Under the RBI Digital Lending Guidelines 2022, lending apps must:
   - Ensure all loan disbursals go directly to the borrower's bank account [Source 1]
   - Collect only need-based data with explicit borrower consent [Source 1]
   - Comply with minimum cybersecurity baseline and end-to-end encryption [Source 2]
   - Disclose the Annual Percentage Rate (APR) to borrowers [Source 3]

Sources:
  Source 1 — Digital Lending Guidelines 2022, Page 3
  Source 2 — Digital Lending Guidelines 2022, Page 9
  Source 3 — Digital Lending Guidelines 2022, Page 4
```

---

## RAG Architecture

```
INGESTION (run once)
───────────────────────────────────────────────────
PDF → PyMuPDF extraction (text + page numbers)
    → Header-aware splitting (RBI section boundaries)
    → Parent-Child chunking:
         Child  = ~256 tokens  → ChromaDB (retrieval)
         Parent = ~1024 tokens → JSON store (context)
    → BGE-small embeddings (local, free)
    → BM25 index (keyword search)
    → MD5 cache (skip already-processed docs)

QUERY
───────────────────────────────────────────────────
User question
    → Multi-query expansion (Groq LLM → 3 variants)
    → Hybrid search:
         Vector search (ChromaDB cosine similarity)
         BM25 keyword search
         → Reciprocal Rank Fusion (merge rankings)
    → Fetch parent chunks for winning children
    → BGE reranker (cross-encoder precision pass)
    → Groq LLM → streaming answer with [Source N] citations
```

### Techniques Implemented

| Technique | Purpose |
|---|---|
| Header-aware chunking | Preserves RBI section structure |
| Parent-child retrieval | Small chunks for search, large for context |
| Hybrid search (BM25 + vector) | Catches exact regulatory terms BM25 misses |
| Reciprocal Rank Fusion | Merges two ranked lists without score normalization |
| Multi-query expansion | 3 query variants → higher recall |
| BGE cross-encoder reranking | Second-pass precision boost |
| Streaming responses | Real-time answer delivery |
| Citation extraction | Every answer cites doc name + page number |
| Embedding cache | Skip re-processing on restart |
| RAGAS evaluation | Faithfulness, relevancy, precision, recall metrics |

---

## Knowledge Base (6 RBI Documents)

| Document | Pages | Coverage |
|---|---|---|
| KYC Master Direction 2016 (updated Aug 2025) | 107 | Identity verification, AML, CDD |
| Digital Lending Guidelines 2022 | 12 | Lending apps, LSPs, APR disclosure |
| Master Circular – IRACP Norms 2021 | 77 | NPA classification, provisioning |
| Master Direction – Interest Rate on Advances | 20 | MCLR, base rate, lending rates |
| Fraud Risk Management FAQs 2024 | 2 | Fraud classification, SCBMF |
| Tax Collection Scheme Circular 2016 | 4 | Agency bank collection duties |

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq `llama-3.3-70b-versatile` | Free tier, 70B quality, fast inference |
| Embeddings | `BAAI/bge-small-en-v1.5` | Best free embedding model (MTEB leaderboard) |
| Vector DB | ChromaDB (local, persistent) | Zero cost, zero infra |
| Keyword search | BM25 (`rank_bm25`) | Catches exact regulatory terms |
| Reranker | `BAAI/bge-reranker-base` | Free cross-encoder, runs on CPU |
| PDF parsing | PyMuPDF | Reliable page number extraction |
| Framework | LangChain | Modular, portfolio-recognizable |
| UI | Streamlit | Chat interface, local + deployable |
| Evaluation | RAGAS | Synthetic testset + 4 metrics |

**Total cost to run: $0** — all models are local or free-tier API.

---

## Project Structure

```
finquery/
├── data/
│   ├── raw/              ← RBI PDFs
│   ├── chroma_db/        ← Vector store (auto-created)
│   ├── bm25_index/       ← BM25 index (auto-created)
│   └── cache/            ← Embedding cache + parent chunks
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── pdf_parser.py       ← PyMuPDF + page metadata
│   │   ├── chunker.py          ← Header-aware + parent-child
│   │   └── indexer.py          ← ChromaDB + BM25 + cache
│   ├── retrieval/
│   │   ├── hybrid_search.py    ← Vector + BM25 + RRF
│   │   └── reranker.py         ← BGE cross-encoder
│   ├── generation/
│   │   ├── query_expander.py   ← Multi-query via Groq
│   │   └── generator.py        ← Prompt + streaming + citations
│   └── evaluation/
│       └── evaluate.py         ← RAGAS synthetic eval
├── app.py                ← Streamlit chat UI
├── ingest.py             ← CLI ingestion script
└── requirements.txt
```

---

## Setup & Run

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/finquery.git
cd finquery
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your Groq API key (free at [console.groq.com](https://console.groq.com))

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### 3. Add RBI PDFs

Place your RBI PDF documents in `data/raw/`. The system auto-detects all PDFs in that folder.

### 4. Ingest documents

```bash
python ingest.py
# Re-run after adding new PDFs — already-processed files are skipped
# python ingest.py --force   # to re-index from scratch
```

### 5. Run the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 6. (Optional) Run RAGAS evaluation

```bash
python -m src.evaluation.evaluate --testset-size 10 --output eval_results.json
```

---

## Sample Questions to Try

- *What are the KYC requirements for opening a bank account?*
- *What is MCLR and how is it calculated?*
- *What actions can a bank take when a fraud is detected?*
- *What data can a digital lending app collect from a borrower?*
- *What are the NPA classification timelines for advances?*

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Get free at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Local reranker model |
| `TOP_K_RETRIEVAL` | `10` | Candidates before reranking |
| `TOP_K_RERANK` | `4` | Final chunks sent to LLM |
| `CHUNK_SIZE_CHILD` | `256` | Child chunk size (tokens approx) |
| `CHUNK_SIZE_PARENT` | `1024` | Parent chunk size (tokens approx) |

---

## How Evaluation Works

RAGAS auto-generates question-answer pairs from your documents (no manual labeling), runs the full pipeline on each question, then scores:

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer grounded in retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_recall` | Did retrieval find all relevant information? |
| `context_precision` | Were retrieved chunks ranked well? |
