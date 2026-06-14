import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingestion.pdf_parser import PageDoc
from src.config import CHUNK_SIZE_CHILD, CHUNK_SIZE_PARENT, CHUNK_OVERLAP

# Section header patterns found in RBI documents
_SECTION_PATTERNS = [
    r'\nCHAPTER\s+[IVXLC]+\b',
    r'\n\d+\.\s+[A-Z][a-zA-Z]',
    r'\nPart\s+[IVXLC]+\s*[-–]',
    r'\nANNEXURE\s+[IVXLC\d]+',
    r'\nSCHEDULE\s+[IVXLC\d]+',
    r'\nAPPENDIX',
]
_HEADER_RE = re.compile('|'.join(_SECTION_PATTERNS))


@dataclass
class ParentChunk:
    chunk_id: str
    text: str
    doc_name: str
    source_file: str
    page_number: int
    section_hint: str = ""
    child_ids: List[str] = field(default_factory=list)


@dataclass
class ChildChunk:
    chunk_id: str
    text: str
    doc_name: str
    source_file: str
    page_number: int
    parent_id: str
    section_hint: str = ""


def _extract_section_hint(text: str) -> str:
    match = re.match(r'^((?:CHAPTER|Part|ANNEXURE|SCHEDULE|APPENDIX)[^\n]{0,80}|\d+\.[^\n]{0,80})', text.strip())
    if match:
        return match.group(1).strip()[:100]
    return text.strip()[:80]


def _split_into_sections(text: str) -> List[str]:
    splits = _HEADER_RE.split(text)
    headers = _HEADER_RE.findall(text)

    sections: List[str] = []
    if splits[0].strip():
        sections.append(splits[0])

    for header, body in zip(headers, splits[1:]):
        section_text = header + body
        if section_text.strip():
            sections.append(section_text)

    return sections if sections else [text]


def _tokenize_approx(text: str) -> int:
    return len(text.split())


def create_chunks(pages: List[PageDoc]) -> tuple[List[ParentChunk], List[ChildChunk]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_PARENT * 4,   # ~4 chars per token
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHILD * 4,
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " "],
    )

    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for page in pages:
        sections = _split_into_sections(page.text)

        for section_text in sections:
            if len(section_text.strip()) < 100:
                continue

            parent_texts = parent_splitter.split_text(section_text)
            section_hint = _extract_section_hint(section_text)

            for parent_text in parent_texts:
                if len(parent_text.strip()) < 80:
                    continue

                parent_id = str(uuid.uuid4())
                parent = ParentChunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    doc_name=page.doc_name,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    section_hint=section_hint,
                )

                child_texts = child_splitter.split_text(parent_text)
                for child_text in child_texts:
                    if len(child_text.strip()) < 40:
                        continue
                    child_id = str(uuid.uuid4())
                    child = ChildChunk(
                        chunk_id=child_id,
                        text=child_text,
                        doc_name=page.doc_name,
                        source_file=page.source_file,
                        page_number=page.page_number,
                        parent_id=parent_id,
                        section_hint=section_hint,
                    )
                    parent.child_ids.append(child_id)
                    children.append(child)

                if parent.child_ids:
                    parents.append(parent)

    return parents, children
