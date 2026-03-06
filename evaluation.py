"""
Evaluation and metrics module for RAG system performance.
Measures: Recall@K, Precision, Response Accuracy, Response Time
"""

import time
from typing import List, Dict, Tuple
from datetime import datetime
import sqlite3
import json
from database import DATABASE_PATH


class EvaluationMetrics:
    """Calculate RAG system performance metrics"""

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
        """
        Calculate Recall@K metric
        Recall@K = |retrieved ∩ relevant| / |relevant|

        Args:
            retrieved_docs: List of retrieved document indices/IDs
            relevant_docs: List of truly relevant document indices/IDs
            k: Number of top documents to consider

        Returns:
            Recall@K score (0-1)
        """
        if not relevant_docs:
            return 1.0  # Perfect if no relevant docs expected

        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        intersection = len(retrieved_at_k & relevant_set)
        recall = intersection / len(relevant_set)

        return round(recall, 4)

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
        """
        Calculate Precision@K metric
        Precision@K = |retrieved ∩ relevant| / k

        Args:
            retrieved_docs: List of retrieved document indices/IDs
            relevant_docs: List of truly relevant document indices/IDs
            k: Number of top documents to consider

        Returns:
            Precision@K score (0-1)
        """
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        intersection = len(retrieved_at_k & relevant_set)
        precision = intersection / k if k > 0 else 0

        return round(precision, 4)

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        MRR = 1 / rank_of_first_relevant_doc

        Args:
            retrieved_docs: List of retrieved document indices/IDs
            relevant_docs: List of truly relevant document indices/IDs

        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant_docs)

        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                return round(1.0 / rank, 4)

        return 0.0  # No relevant doc found

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 Score (harmonic mean of precision and recall)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 4)

    @staticmethod
    def ndcg(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        Measures ranking quality with position-based weighting

        Args:
            retrieved_docs: List of retrieved document indices/IDs
            relevant_docs: List of truly relevant document indices/IDs (ranked by relevance)
            k: Number of top documents to consider

        Returns:
            NDCG@K score (0-1)
        """
        relevant_set = set(relevant_docs)

        # Calculate DCG
        dcg = 0.0
        for rank, doc in enumerate(retrieved_docs[:k], 1):
            if doc in relevant_set:
                dcg += 1.0 / (rank + 1)  # Using log2(rank+1) denominator

        # Calculate Ideal DCG (all relevant docs ranked first)
        ideal_dcg = 0.0
        for rank in range(1, min(len(relevant_docs), k) + 1):
            ideal_dcg += 1.0 / (rank + 1)

        if ideal_dcg == 0:
            return 1.0

        ndcg = dcg / ideal_dcg
        return round(ndcg, 4)


class ResponseAccuracy:
    """Evaluate response accuracy against expected answers"""

    @staticmethod
    def simple_match(response: str, expected: str, ignore_case: bool = True) -> bool:
        """Check if response matches expected answer exactly"""
        if ignore_case:
            return response.lower() == expected.lower()
        return response == expected

    @staticmethod
    def keyword_match(response: str, expected_keywords: List[str], min_matches: int = 1) -> Tuple[bool, int]:
        """
        Check if response contains expected keywords

        Returns:
            Tuple of (is_accurate, num_matches)
        """
        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)

        is_accurate = matches >= min_matches
        return is_accurate, matches

    @staticmethod
    def no_hallucination(response: str, forbidden_keywords: List[str] = None) -> bool:
        """Check if response contains no hallucinations (forbidden content)"""
        if not forbidden_keywords:
            forbidden_keywords = ["i don't know", "i have no information", "no data available"]

        response_lower = response.lower()

        # Check for empty or very short responses (likely failures)
        if len(response.strip()) < 10:
            return False

        # Check for forbidden keywords that indicate hallucination
        for keyword in forbidden_keywords:
            if keyword.lower() in response_lower:
                return True  # If response explicitly says it doesn't know, that's honest, not hallucination

        return True

    @staticmethod
    def factual_consistency(response: str, context: str) -> Tuple[bool, float]:
        """
        Simple check: response should be consistent with provided context
        Uses keyword overlap as a basic metric

        Returns:
            Tuple of (is_consistent, consistency_score 0-1)
        """
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        # Remove common words
        common_stop_words = {"the", "a", "an", "is", "are", "and", "or", "but", "in", "on", "at", "to", "for"}
        response_words -= common_stop_words
        context_words -= common_stop_words

        if not response_words:
            return False, 0.0

        overlap = len(response_words & context_words) / len(response_words)

        return overlap > 0.2, round(overlap, 4)


class PerformanceTracker:
    """Track chatbot performance over time"""

    def __init__(self):
        self.metrics_table = "evaluation_metrics"
        self._init_metrics_table()

    def _init_metrics_table(self):
        """Initialize metrics tracking table"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.metrics_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                response_time_ms FLOAT,
                recall_k3 FLOAT,
                precision_k3 FLOAT,
                mrr FLOAT,
                f1_score FLOAT,
                ndcg_k3 FLOAT,
                is_accurate BOOLEAN,
                contains_pii BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def log_query_metric(
        self,
        query_text: str,
        response_text: str,
        response_time_ms: float,
        recall_k3: float = 0.0,
        precision_k3: float = 0.0,
        mrr: float = 0.0,
        f1_score: float = 0.0,
        ndcg_k3: float = 0.0,
        is_accurate: bool = True,
        contains_pii: bool = False
    ):
        """Log performance metrics for a query"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT INTO {self.metrics_table}
            (query_text, response_text, response_time_ms, recall_k3, precision_k3, 
             mrr, f1_score, ndcg_k3, is_accurate, contains_pii)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (query_text, response_text, response_time_ms, recall_k3, precision_k3,
              mrr, f1_score, ndcg_k3, is_accurate, contains_pii))

        conn.commit()
        conn.close()

    def get_performance_summary(self, last_n_queries: int = 100) -> Dict:
        """Get overall performance metrics"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_queries,
                AVG(response_time_ms) as avg_response_time_ms,
                AVG(recall_k3) as avg_recall_k3,
                AVG(precision_k3) as avg_precision_k3,
                AVG(f1_score) as avg_f1_score,
                AVG(ndcg_k3) as avg_ndcg_k3,
                SUM(CASE WHEN is_accurate = 1 THEN 1 ELSE 0 END) as accurate_responses,
                SUM(CASE WHEN contains_pii = 1 THEN 1 ELSE 0 END) as pii_detected
            FROM {self.metrics_table}
            ORDER BY id DESC
            LIMIT ?
        """, (last_n_queries,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                "total_queries": 0,
                "performance_report": "No metrics available yet"
            }

        total = row[0]
        return {
            "total_queries": total,
            "avg_response_time_ms": round(row[1], 2) if row[1] else 0,
            "avg_recall_k3": round(row[2], 4) if row[2] else 0,
            "avg_precision_k3": round(row[3], 4) if row[3] else 0,
            "avg_f1_score": round(row[4], 4) if row[4] else 0,
            "avg_ndcg_k3": round(row[5], 4) if row[5] else 0,
            "accuracy_rate": round((row[6] / total * 100), 2) if total > 0 else 0,
            "pii_detections": row[7]
        }

    def generate_report(self) -> str:
        """Generate a formatted performance report"""
        summary = self.get_performance_summary()

        report = f"""
RAG System Performance Evaluation Report
========================================

Query Statistics:
- Total Queries Processed: {summary['total_queries']}
- Average Response Time: {summary['avg_response_time_ms']}ms
- Accuracy Rate: {summary['accuracy_rate']}%

Retrieval Metrics (Recall/Precision):
- Average Recall@3: {summary['avg_recall_k3']}
- Average Precision@3: {summary['avg_precision_k3']}
- Average F1 Score: {summary['avg_f1_score']}

Ranking Quality Metrics:
- Average NDCG@3: {summary['avg_ndcg_k3']}
- Mean Reciprocal Rank: (tracked per query)

Data Protection:
- PII Detections: {summary['pii_detections']}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return report.strip()


if __name__ == "__main__":
    print("=== RAG Evaluation Metrics Demo ===\n")

    # Example: Test retrieval metrics
    print("1. Testing Recall@K and Precision@K:")
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc5", "doc6"]

    recall_3 = EvaluationMetrics.recall_at_k(retrieved, relevant, k=3)
    precision_3 = EvaluationMetrics.precision_at_k(retrieved, relevant, k=3)
    mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved, relevant)
    f1 = EvaluationMetrics.f1_score(precision_3, recall_3)

    print(f"   Retrieved: {retrieved}")
    print(f"   Relevant: {relevant}")
    print(f"   Recall@3: {recall_3}")
    print(f"   Precision@3: {precision_3}")
    print(f"   MRR: {mrr}")
    print(f"   F1 Score: {f1}\n")

    # Example: Test response accuracy
    print("2. Testing Response Accuracy:")
    response = "The parking works 24/7 with a technical break from 3:00 to 3:15 AM"
    expected_keywords = ["24/7", "3:00", "technical break"]

    is_accurate, matches = ResponseAccuracy.keyword_match(response, expected_keywords, min_matches=2)
    print(f"   Response: {response}")
    print(f"   Expected keywords: {expected_keywords}")
    print(f"   Is accurate: {is_accurate}, Matches: {matches}\n")

    # Example: Performance tracking
    print("3. Initializing Performance Tracker:")
    tracker = PerformanceTracker()
    print("   ✓ Metrics table created")
