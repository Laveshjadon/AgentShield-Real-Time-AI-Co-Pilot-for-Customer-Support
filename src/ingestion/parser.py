# --- FROM: ingestion/loaders/base_loader.py ---
"""
AgentShield - Base Document Loader
Abstract base class and models for the document ingestion framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import os


class DocumentLoadError(Exception):
    """Raised when a document fails to load or parse."""
    pass


class DocumentValidationError(Exception):
    """Raised when a document violates size limits or unsupported formats."""
    pass


@dataclass
class DocumentChunk:
    """Represents a chunk of text extracted from a document."""
    text: str
    source_file: str
    document_type: str
    page_number: Optional[int] = None


class BaseDocumentLoader(ABC):
    """
    Abstract base class for all document loaders.
    """
    
    def __init__(self, max_size_bytes: int = 50 * 1024 * 1024):
        self.max_size_bytes = max_size_bytes

    def validate(self, file_path: str) -> None:
        """
        Validates the file before loading.
        Raises DocumentValidationError if invalid.
        """
        if not os.path.exists(file_path):
            raise DocumentValidationError(f"File not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_bytes:
            raise DocumentValidationError(
                f"File size {file_size / (1024*1024):.1f}MB exceeds the maximum allowed "
                f"limit of {self.max_size_bytes / (1024*1024):.1f}MB."
            )

    @abstractmethod
    def load(self, file_path: str) -> List[DocumentChunk]:
        """
        Extracts text from the file and returns a list of DocumentChunks.
        Implementations should NOT chunk by token size (e.g., 500 tokens),
        but rather by logical blocks (e.g., pages for PDFs, or the entire text).
        The chunking into fixed sizes is handled by the ingestion pipeline later.
        """
        pass


# --- FROM: ingestion/loaders/pdf_loader.py ---
import os
from typing import List

import fitz  




class PdfLoader(BaseDocumentLoader):
    """basically goes through the pdf and gets the text one page at a time"""

    def load(self, file_path: str) -> List[DocumentChunk]:
        self.validate(file_path)
        filename = os.path.basename(file_path)
        chunks = []
        
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    chunks.append(
                        DocumentChunk(
                            text=text,
                            source_file=filename,
                            document_type="pdf",
                            page_number=page_num
                        )
                    )
            doc.close()
            return chunks
        except Exception as e:
            raise DocumentLoadError(f"Failed to load PDF file {filename}: {e}")


# --- FROM: ingestion/loaders/docx_loader.py ---
import os
from typing import Iterator, List, Union

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph




class DocxLoader(BaseDocumentLoader):
    """Load DOCX content as ordered, heading-aware sections."""

    @staticmethod
    def _iter_blocks(document: DocumentObject) -> Iterator[Union[Paragraph, Table]]:
        """Yield paragraphs and tables in their original document order."""
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, document)
            elif child.tag.endswith("}tbl"):
                yield Table(child, document)

    @staticmethod
    def _table_to_text(table: Table) -> str:
        rows = []
        for row in table.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def load(self, file_path: str) -> List[DocumentChunk]:
        self.validate(file_path)
        filename = os.path.basename(file_path)

        try:
            doc = Document(file_path)
            chunks: List[DocumentChunk] = []
            section_lines: List[str] = []
            heading_path: List[str] = []
            section_has_content = False

            def flush_section() -> None:
                nonlocal section_has_content
                text = "\n".join(section_lines).strip()
                if text and section_has_content:
                    chunks.append(
                        DocumentChunk(
                            text=text,
                            source_file=filename,
                            document_type="docx",
                            page_number=None,
                        )
                    )
                section_lines.clear()
                section_has_content = False

            for block in self._iter_blocks(doc):
                if isinstance(block, Paragraph):
                    text = " ".join(block.text.split())
                    if not text:
                        continue

                    style_name = getattr(block.style, "name", "") or ""
                    if style_name.startswith("Heading"):
                        flush_section()
                        try:
                            level = int(style_name.split()[-1])
                        except (ValueError, IndexError):
                            level = 1

                        heading_path[:] = heading_path[: level - 1]
                        heading_path.append(text)
                        section_lines.extend(heading_path)
                    else:
                        if not section_lines and heading_path:
                            section_lines.extend(heading_path)
                        section_lines.append(text)
                        section_has_content = True
                else:
                    table_text = self._table_to_text(block)
                    if table_text:
                        if not section_lines and heading_path:
                            section_lines.extend(heading_path)
                        section_lines.append(table_text)
                        section_has_content = True

            flush_section()
            return chunks
        except Exception as e:
            raise DocumentLoadError(f"Failed to load DOCX file {filename}: {e}")


# --- FROM: ingestion/loaders/html_loader.py ---
import os
from typing import List

from bs4 import BeautifulSoup




class HtmlLoader(BaseDocumentLoader):
    """just grab the text from html files, gonna ignore all the messy css and js"""

    def load(self, file_path: str) -> List[DocumentChunk]:
        self.validate(file_path)
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
            
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
                
            
            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return [
                DocumentChunk(
                    text=text,
                    source_file=filename,
                    document_type="html",
                    page_number=1
                )
            ]
        except Exception as e:
            raise DocumentLoadError(f"Failed to load HTML file {filename}: {e}")


# --- FROM: ingestion/loaders/txt_loader.py ---
import os
from typing import List




class TxtLoader(BaseDocumentLoader):
    """Loads plain text files."""

    def load(self, file_path: str) -> List[DocumentChunk]:
        self.validate(file_path)
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            document_type = os.path.splitext(filename)[1].lstrip(".").lower()
            return [
                DocumentChunk(
                    text=text,
                    source_file=filename,
                    document_type=document_type,
                    page_number=1
                )
            ]
        except Exception as e:
            raise DocumentLoadError(f"Failed to load text file {filename}: {e}")


def get_loader_for_file(file_path: str) -> BaseDocumentLoader:
    """
    Factory method to return the appropriate loader based on the file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return PdfLoader()
    elif ext == '.docx':
        return DocxLoader()
    elif ext in ['.html', '.htm']:
        return HtmlLoader()
    elif ext in ['.txt', '.md']:
        return TxtLoader()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
