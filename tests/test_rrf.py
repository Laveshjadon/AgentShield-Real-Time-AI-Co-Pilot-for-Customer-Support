import pytest
from rag.rrf import reciprocal_rank_fusion

def test_rrf_scoring():
    
    dense_results = [
        {"chunk_index": 1, "source_file": "doc1.txt", "similarity": 0.9, "content": "A"},
        {"chunk_index": 2, "source_file": "doc1.txt", "similarity": 0.8, "content": "B"},
        {"chunk_index": 1, "source_file": "doc2.txt", "similarity": 0.7, "content": "C"}
    ]
    
    
    sparse_results = [
        {"chunk_index": 2, "source_file": "doc1.txt", "bm25_score": 15.0, "content": "B"},
        {"chunk_index": 3, "source_file": "doc3.txt", "bm25_score": 10.0, "content": "D"},
        {"chunk_index": 1, "source_file": "doc1.txt", "bm25_score": 5.0, "content": "A"}
    ]
    
    
    
    
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=4)
    
    assert len(fused) == 4
    assert fused[0]["source_file"] == "doc1.txt" and fused[0]["chunk_index"] == 2
    assert fused[1]["source_file"] == "doc1.txt" and fused[1]["chunk_index"] == 1
    assert fused[2]["source_file"] == "doc3.txt" and fused[2]["chunk_index"] == 3
    assert fused[3]["source_file"] == "doc2.txt" and fused[3]["chunk_index"] == 1
    
    
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
