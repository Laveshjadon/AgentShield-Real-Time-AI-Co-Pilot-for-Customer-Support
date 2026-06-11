"""
rrf scoring script
this fuses the lists from bm25 and the vector search together.
googled the formula for this: score = sum(1 / (k + rank)).
"""

from typing import List, Dict, Any, Tuple
from config.logger import get_logger

logger = get_logger("rag.rrf")


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    does the actual fusion math on the two lists.
    needs the chunks to have a source_file and chunk_index so i can tell what's what.
    returns the top ones after combining.
    """
    rrf_scores: Dict[Tuple[str, int], float] = {}
    chunk_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    
    
    def _process_ranked_list(ranked_list: List[Dict[str, Any]]):
        for rank, chunk in enumerate(ranked_list, start=1):
            unique_id = (chunk["source_file"], chunk["chunk_index"])
            
            if unique_id not in chunk_map:
                
                chunk_map[unique_id] = chunk
                rrf_scores[unique_id] = 0.0
                
            rrf_scores[unique_id] += 1.0 / (k + rank)

    
    _process_ranked_list(dense_results)
    _process_ranked_list(sparse_results)
    
    
    sorted_fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    
    final_results = []
    for unique_id, score in sorted_fused[:top_n]:
        chunk = chunk_map[unique_id]
        chunk["rrf_score"] = score
        final_results.append(chunk)
        
    logger.debug(f"[RRF] Fused {len(rrf_scores)} unique chunks down to top {top_n}")
    return final_results
