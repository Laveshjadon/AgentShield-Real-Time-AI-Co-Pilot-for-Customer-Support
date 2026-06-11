import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from rag.loaders.base_loader import DocumentValidationError, DocumentLoadError
from rag.loaders.txt_loader import TxtLoader
from rag.loaders.html_loader import HtmlLoader
from rag.loaders import get_loader_for_file


@pytest.fixture
def temp_txt():
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write("Hello world!")
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def temp_html():
    fd, path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write("<html><head><style>.css{}</style></head><body><h1>Title</h1><p>Hello HTML</p><script>alert(1)</script></body></html>")
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_factory():
    assert isinstance(get_loader_for_file("doc.txt"), TxtLoader)
    assert isinstance(get_loader_for_file("policy.md"), TxtLoader)
    assert isinstance(get_loader_for_file("index.html"), HtmlLoader)
    with pytest.raises(ValueError):
        get_loader_for_file("doc.unsupported")


def test_txt_loader(temp_txt):
    loader = TxtLoader()
    chunks = loader.load(temp_txt)
    
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world!"
    assert chunks[0].document_type == "txt"
    assert chunks[0].page_number == 1


def test_html_loader(temp_html):
    loader = HtmlLoader()
    chunks = loader.load(temp_html)
    
    assert len(chunks) == 1
    
    assert "alert" not in chunks[0].text
    assert ".css" not in chunks[0].text
    assert "Title" in chunks[0].text
    assert "Hello HTML" in chunks[0].text
    assert chunks[0].document_type == "html"
    assert chunks[0].page_number == 1


def test_base_loader_size_validation(temp_txt):
    loader = TxtLoader(max_size_bytes=1) 
    with pytest.raises(DocumentValidationError, match="exceeds the maximum allowed limit"):
        loader.load(temp_txt)


@patch('rag.loaders.pdf_loader.fitz.open')
def test_pdf_loader_mocked(mock_fitz_open):
    from rag.loaders.pdf_loader import PdfLoader
    
    
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 Text"
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 Text"
    
    
    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_fitz_open.return_value = mock_doc
    
    loader = PdfLoader()
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"dummy")
        tmp_path = tmp.name
        
    chunks = loader.load(tmp_path)
    os.remove(tmp_path)
    
    assert len(chunks) == 2
    assert chunks[0].text == "Page 1 Text"
    assert chunks[0].page_number == 1
    assert chunks[0].document_type == "pdf"
    
    assert chunks[1].text == "Page 2 Text"
    assert chunks[1].page_number == 2
