from typing import List
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

_client: Groq = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


_EXPAND_PROMPT = """You are a search query optimizer for Indian financial regulation documents.

Given a user question, generate 3 alternative search queries that would help retrieve relevant regulatory content.
- Use different phrasings and regulatory terminology
- Include specific RBI terms, section references, or related concepts
- Keep each query concise (under 15 words)
- Output ONLY the 3 queries, one per line, no numbering or bullets

User question: {question}"""


def expand_query(question: str) -> List[str]:
    """Generate 3 query variants to improve retrieval recall."""
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _EXPAND_PROMPT.format(question=question)}],
            temperature=0.4,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        variants = [q.strip() for q in raw.split('\n') if q.strip()][:3]
    except Exception:
        variants = []

    # Always include the original question
    all_queries = [question] + variants
    return list(dict.fromkeys(all_queries))  # deduplicate while preserving order
