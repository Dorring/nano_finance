"""Retrieval pipeline modules."""
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.candidate_fusion import (
    normalize_scores,
    dedupe_chunks,
    chunk_doc_name,
    ensure_multi_doc_coverage,
    boost_front_matter_chunks,
    summarize_retrieved_chunks,
    source_from_chunk,
)
from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator, SufficiencyResult
