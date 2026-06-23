import pytest
from evaluation import EvaluationMetrics, ResponseAccuracy, PerformanceTracker

def test_evaluation_metrics():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc5", "doc6"]

    recall = EvaluationMetrics.recall_at_k(retrieved, relevant, k=3)
    precision = EvaluationMetrics.precision_at_k(retrieved, relevant, k=3)
    mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved, relevant)
    f1 = EvaluationMetrics.f1_score(precision, recall)

    assert 0 <= recall <= 1
    assert 0 <= precision <= 1
    assert mrr > 0
    assert f1 > 0

def test_response_accuracy():
    response = "The parking is open 24/7 with a technical break from 3:00 to 3:15 AM"
    keywords = ["24/7", "3:00"]
    is_accurate, matches = ResponseAccuracy.keyword_match(response, keywords, min_matches=2)

    assert is_accurate is True
    assert matches >= 2

def test_performance_tracker():
    tracker = PerformanceTracker()
    tracker.log_query_metric(
        query_text="Test query",
        response_text="Test response",
        response_time_ms=150.5,
        recall_k3=0.85,
        precision_k3=0.90
    )

    summary = tracker.get_performance_summary()
    assert summary['total_queries'] > 0
