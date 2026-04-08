from src.hybrid_retriever import HybridRetriever


def test_reciprocal_rank_fusion():
    """Test RRF merging logic without external dependencies."""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.RRF_K = 60
    retriever.alpha = 0.6

    vector_results = [
        {"id": "doc_a", "score": 0.95, "text": "A"},
        {"id": "doc_b", "score": 0.85, "text": "B"},
        {"id": "doc_c", "score": 0.75, "text": "C"},
    ]

    bm25_results = [
        {"id": "doc_b", "score": 5.2, "text": "B"},
        {"id": "doc_d", "score": 4.1, "text": "D"},
        {"id": "doc_a", "score": 3.0, "text": "A"},
    ]

    fused = retriever._reciprocal_rank_fusion(vector_results, bm25_results)

    # doc_a and doc_b should be at the top (appear in both lists)
    ids = [r["id"] for r in fused]
    assert "doc_a" in ids[:3]
    assert "doc_b" in ids[:3]

    # All unique documents should be present
    assert len(fused) == 4
    assert set(ids) == {"doc_a", "doc_b", "doc_c", "doc_d"}

    # RRF scores should be positive
    assert all(r["rrf_score"] > 0 for r in fused)

    # Results should be sorted by rrf_score descending
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)
