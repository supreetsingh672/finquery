"""
RAGAS evaluation pipeline for FinQuery.

Generates a synthetic test set from your RBI PDFs, runs the full RAG pipeline
against each question, then scores on faithfulness, answer relevancy,
context precision and context recall.

Usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --testset-size 20 --output eval_results.json
"""

import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    LLMContextRecall,
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import pandas as pd

from src.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, RAW_DOCS_PATH
from src.ingestion.indexer import get_embedding_model
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank, get_reranker
from src.generation.query_expander import expand_query
from src.generation.generator import generate_answer, get_citations
from src.config import TOP_K_RERANK


def load_docs_for_ragas(raw_dir: str, max_pages_per_doc: int = 8) -> List:
    """Load a small representative sample from each PDF for testset generation.

    Keeping the chunk count low (< 80 total) avoids hitting Groq's free daily
    token limit during RAGAS SummaryExtractor passes.
    """
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for fname in sorted(Path(raw_dir).iterdir()):
        if fname.suffix != '.pdf':
            continue
        loader = PyMuPDFLoader(str(fname))
        raw = loader.load()[:max_pages_per_doc]
        chunks = splitter.split_documents(raw)
        all_docs.extend(chunks)
        print(f"  Loaded {fname.name}: {len(chunks)} chunks")

    return all_docs


def generate_testset(docs: List, size: int, llm: ChatGroq) -> List[Dict]:
    """Use RAGAS TestsetGenerator to auto-create Q&A pairs from docs."""
    print(f"\nGenerating {size} synthetic test questions...")

    generator_llm = LangchainLLMWrapper(llm)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=size,
    )
    return testset.to_pandas().to_dict(orient="records")


def run_pipeline(question: str, embedder: SentenceTransformer) -> Dict:
    """Run the full FinQuery pipeline for one question."""
    queries = expand_query(question)
    candidates = hybrid_search(queries, embedder)
    top_chunks = rerank(question, candidates, top_k=TOP_K_RERANK)

    contexts = [c["text"] for c in top_chunks]

    full_answer = ""
    for token in generate_answer(question, top_chunks):
        full_answer += token

    return {
        "question": question,
        "answer": full_answer,
        "contexts": contexts,
    }


def evaluate_pipeline(testset_size: int = 10, output_path: str = "eval_results.json") -> None:
    print("\n=== FinQuery RAGAS Evaluation ===\n")

    # Use a smaller model for RAGAS generation to save tokens on free tier
    ragas_model = "llama-3.1-8b-instant"
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=ragas_model, temperature=0)
    print(f"Using {ragas_model} for testset generation (token-efficient)")

    print("[1/4] Loading documents for testset generation...")
    docs = load_docs_for_ragas(str(RAW_DOCS_PATH))
    print(f"  Total chunks: {len(docs)}")

    print("\n[2/4] Generating synthetic testset...")
    testset = generate_testset(docs, size=testset_size, llm=llm)
    print(f"  Generated {len(testset)} questions")

    print("\n[3/4] Running RAG pipeline on each question...")
    embedder = get_embedding_model()
    get_reranker()

    results = []
    for i, item in enumerate(testset, 1):
        question = item.get("user_input", item.get("question", ""))
        reference = item.get("reference", item.get("ground_truth", ""))
        if not question:
            continue
        print(f"  [{i}/{len(testset)}] {question[:70]}...")
        pipeline_output = run_pipeline(question, embedder)
        pipeline_output["reference"] = reference
        results.append(pipeline_output)

    print("\n[4/4] Scoring with RAGAS metrics...")
    eval_llm = LangchainLLMWrapper(llm)
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    from ragas import SingleTurnSample
    samples = []
    for r in results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r.get("reference", ""),
        ))

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        LLMContextRecall(llm=eval_llm),
        LLMContextPrecisionWithoutReference(llm=eval_llm),
    ]

    score = evaluate(dataset=dataset, metrics=metrics)
    df = score.to_pandas()

    print("\n=== RAGAS Evaluation Results ===")
    summary = {
        "faithfulness": round(df["faithfulness"].mean(), 3),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
        "context_recall": round(df["llm_context_recall"].mean(), 3),
        "context_precision": round(df["llm_context_precision_without_reference"].mean(), 3),
    }
    for metric, val in summary.items():
        print(f"  {metric:<30} {val:.3f}")

    output = {
        "summary": summary,
        "per_question": results,
        "ragas_df": df.to_dict(orient="records"),
    }
    Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for FinQuery")
    parser.add_argument("--testset-size", type=int, default=10)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    evaluate_pipeline(testset_size=args.testset_size, output_path=args.output)
