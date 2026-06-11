import os
from typing import List

from bs4 import BeautifulSoup

from .base_loader import BaseDocumentLoader, DocumentChunk, DocumentLoadError


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
