import pytest
from rag.query_builder import QueryBuilder

def test_query_builder_terminology_normalization():
    qb = QueryBuilder()
    
    
    transcript = "I want my money back because the router broke."
    query = qb.build_query(transcript)
    
    assert "refund" in query
    assert "defective" in query
    
def test_query_builder_keyword_extraction():
    qb = QueryBuilder()
    
    
    transcript = "The internet router is completely defective and I need a replacement immediately."
    query = qb.build_query(transcript)
    
    assert "internet" in query or "router" in query
    assert "replacement" in query
