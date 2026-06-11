"""
AgentShield - Bias-Aware Performance Scoring
"""
from typing import List, Tuple

def compute_adjusted_score(
    clean_scores: List[float], 
    aggressive_scores: List[float]
) -> Tuple[float, float, int, int]:
    """
    Computes the raw and bias-adjusted performance score for an agent.
    
    Formula:
    adjusted = (clean_avg * clean_count + agg_avg * 0.5 * agg_count) 
               / (clean_count + 0.5 * agg_count)
               
    Args:
        clean_scores: List of scores for clean (non-aggressive) calls.
        aggressive_scores: List of scores for aggressive calls.
        
    Returns:
        Tuple containing (raw_average, adjusted_average, clean_count, aggressive_count)
    """
    clean_count = len(clean_scores)
    agg_count = len(aggressive_scores)
    
    clean_avg = sum(clean_scores) / clean_count if clean_count > 0 else 0.0
    agg_avg = sum(aggressive_scores) / agg_count if agg_count > 0 else 0.0
    
    total_count = clean_count + agg_count
    
    if total_count == 0:
        return 0.0, 0.0, 0, 0
        
    raw_average = (sum(clean_scores) + sum(aggressive_scores)) / total_count
    
    denominator = clean_count + (0.5 * agg_count)
    if denominator == 0:
        adjusted_average = 0.0
    else:
        adjusted_average = (clean_avg * clean_count + agg_avg * 0.5 * agg_count) / denominator
        
    return raw_average, adjusted_average, clean_count, agg_count
