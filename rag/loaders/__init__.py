import os

from .base_loader import BaseDocumentLoader, DocumentChunk, DocumentLoadError, DocumentValidationError
from .txt_loader import TxtLoader
from .pdf_loader import PdfLoader
from .docx_loader import DocxLoader
from .html_loader import HtmlLoader

def get_loader_for_file(file_path: str) -> BaseDocumentLoader:
    """
    Factory function to return the correct loader instance based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in {".txt", ".md"}:
        return TxtLoader()
    elif ext == ".pdf":
        return PdfLoader()
    elif ext == ".docx":
        return DocxLoader()
    elif ext in [".html", ".htm"]:
        return HtmlLoader()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
