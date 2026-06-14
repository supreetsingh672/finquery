from typing import List, Dict, Any, Generator
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

_client: Groq = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


_SYSTEM_PROMPT = """You are FinQuery, an expert assistant on Indian financial regulations issued by the Reserve Bank of India (RBI).

Your rules:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the context does not contain the answer, say: "I couldn't find this in the provided RBI documents. Please consult the official RBI website."
3. Be precise and cite your sources inline using [Source N] notation.
4. For regulatory requirements, quote exact thresholds, percentages, and timelines from the context.
5. Keep answers structured and concise."""


def _build_context_block(chunks: List[Dict[str, Any]]) -> tuple[str, List[Dict]]:
    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {chunk['doc_name']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Section: {chunk.get('section_hint', '')}\n\n"
            f"{chunk['text']}"
        )
        citations.append({
            "ref": f"Source {i}",
            "doc_name": chunk["doc_name"],
            "page_number": chunk["page_number"],
            "section_hint": chunk.get("section_hint", ""),
            "source_file": chunk.get("source_file", ""),
        })

    return "\n\n---\n\n".join(context_parts), citations


def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """Stream an answer grounded in the retrieved chunks."""
    context, citations = _build_context_block(chunks)

    user_message = f"""Context from RBI regulatory documents:

{context}

---

Question: {question}

Answer with inline citations [Source N] where applicable:"""

    client = _get_client()
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def get_citations(question: str, chunks: List[Dict[str, Any]]) -> List[Dict]:
    _, citations = _build_context_block(chunks)
    return citations
