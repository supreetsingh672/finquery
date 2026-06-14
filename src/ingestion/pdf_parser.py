import re
import os
from dataclasses import dataclass
from typing import List
import fitz
from src.config import DOCUMENT_NAMES


@dataclass
class PageDoc:
    text: str
    page_number: int
    doc_name: str
    source_file: str


def _clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove lone page numbers (e.g. "\n 5 \n") common in RBI docs
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    return text.strip()


def _is_toc_page(text: str) -> bool:
    dotted_lines = len(re.findall(r'\.{5,}', text))
    return dotted_lines > 5


def parse_pdf(path: str) -> List[PageDoc]:
    filename = os.path.basename(path)
    doc_name = DOCUMENT_NAMES.get(filename, filename.replace('.pdf', ''))

    doc = fitz.open(path)
    pages: List[PageDoc] = []

    for page_idx in range(doc.page_count):
        text = doc[page_idx].get_text()
        text = _clean_text(text)

        if len(text) < 80:
            continue
        if _is_toc_page(text):
            continue

        pages.append(PageDoc(
            text=text,
            page_number=page_idx + 1,
            doc_name=doc_name,
            source_file=filename,
        ))

    return pages


def parse_all_pdfs(raw_dir: str) -> List[PageDoc]:
    all_pages: List[PageDoc] = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith('.pdf'):
            continue
        path = os.path.join(raw_dir, fname)
        pages = parse_pdf(path)
        all_pages.extend(pages)
        print(f"  Parsed: {DOCUMENT_NAMES.get(fname, fname)} — {len(pages)} pages")
    return all_pages
