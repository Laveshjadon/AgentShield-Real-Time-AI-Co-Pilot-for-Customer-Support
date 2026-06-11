import os
from typing import List

from .base_loader import BaseDocumentLoader, DocumentChunk, DocumentLoadError


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
