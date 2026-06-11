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
