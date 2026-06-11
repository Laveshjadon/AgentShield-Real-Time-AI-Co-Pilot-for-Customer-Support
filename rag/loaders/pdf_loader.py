import os
from typing import List

import fitz  

from .base_loader import BaseDocumentLoader, DocumentChunk, DocumentLoadError


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
