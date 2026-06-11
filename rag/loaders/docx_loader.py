import os
from typing import Iterator, List, Union

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base_loader import BaseDocumentLoader, DocumentChunk, DocumentLoadError


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
