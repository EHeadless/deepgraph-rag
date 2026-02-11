"""
Evaluation metrics for Graph RAG systems.

This module provides metrics for measuring the quality and performance
of Graph RAG pipelines.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import re


@dataclass
class EvaluationResult:
    """Result of evaluating a single query.

    Example:
        result = EvaluationResult(
            query="What papers discuss transformers?",
            expected_answer="Papers include 'Attention Is All You Need'...",
            actual_answer="The paper 'Attention Is All You Need'...",
            latency_ms=1250.5,
            tokens_used=1523
        )
    """

    # Query information
    query: str
    expected_answer: Optional[str] = None
    actual_answer: Optional[str] = None

    # Ground truth (if available)
    expected_entities: Set[str] = field(default_factory=set)
    expected_papers: Set[str] = field(default_factory=set)
    expected_authors: Set[str] = field(default_factory=set)

    # Retrieved information
    retrieved_entities: Set[str] = field(default_factory=set)
    retrieved_papers: Set[str] = field(default_factory=set)
    retrieved_authors: Set[str] = field(default_factory=set)

    # Performance metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    nodes_retrieved: int = 0
    edges_traversed: int = 0

    # Quality scores (0-1)
    entity_precision: Optional[float] = None
    entity_recall: Optional[float] = None
    entity_f1: Optional[float] = None
    coverage: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationMetrics:
    """Collection of evaluation metrics for Graph RAG.

    Example:
        metrics = EvaluationMetrics()

        result = metrics.evaluate_result(
            query="What papers discuss transformers?",
            actual_answer="...",
            expected_entities={"Attention Is All You Need", "BERT"},
            retrieved_entities={"Attention Is All You Need", "GPT-2"},
            latency_ms=1250,
            tokens_used=1500
        )

        print(f"Precision: {result.entity_precision:.2%}")
        print(f"Recall: {result.entity_recall:.2%}")
        print(f"F1: {result.entity_f1:.2%}")
    """

    @staticmethod
    def calculate_precision(
        predicted: Set[str],
        ground_truth: Set[str]
    ) -> float:
        """Calculate precision.

        Precision = |predicted ∩ ground_truth| / |predicted|

        Args:
            predicted: Set of predicted items
            ground_truth: Set of ground truth items

        Returns:
            Precision score (0-1)
        """
        if not predicted:
            return 0.0

        correct = len(predicted & ground_truth)
        return correct / len(predicted)

    @staticmethod
    def calculate_recall(
        predicted: Set[str],
        ground_truth: Set[str]
    ) -> float:
        """Calculate recall.

        Recall = |predicted ∩ ground_truth| / |ground_truth|

        Args:
            predicted: Set of predicted items
            ground_truth: Set of ground truth items

        Returns:
            Recall score (0-1)
        """
        if not ground_truth:
            return 0.0

        correct = len(predicted & ground_truth)
        return correct / len(ground_truth)

    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """Calculate F1 score.

        F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_coverage(
        retrieved_count: int,
        total_count: int
    ) -> float:
        """Calculate coverage.

        Coverage = retrieved_count / total_count

        Args:
            retrieved_count: Number of items retrieved
            total_count: Total number of relevant items

        Returns:
            Coverage score (0-1)
        """
        if total_count == 0:
            return 0.0

        return min(retrieved_count / total_count, 1.0)

    def evaluate_result(
        self,
        query: str,
        actual_answer: str,
        expected_entities: Optional[Set[str]] = None,
        retrieved_entities: Optional[Set[str]] = None,
        expected_papers: Optional[Set[str]] = None,
        retrieved_papers: Optional[Set[str]] = None,
        expected_authors: Optional[Set[str]] = None,
        retrieved_authors: Optional[Set[str]] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        nodes_retrieved: int = 0,
        edges_traversed: int = 0,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate a single query result.

        Args:
            query: The query string
            actual_answer: Generated answer
            expected_entities: Ground truth entities
            retrieved_entities: Retrieved entities
            expected_papers: Ground truth papers
            retrieved_papers: Retrieved papers
            expected_authors: Ground truth authors
            retrieved_authors: Retrieved authors
            latency_ms: Query latency in milliseconds
            tokens_used: Number of tokens used
            nodes_retrieved: Number of nodes retrieved
            edges_traversed: Number of edges traversed
            **kwargs: Additional metadata

        Returns:
            EvaluationResult with computed metrics
        """
        result = EvaluationResult(
            query=query,
            actual_answer=actual_answer,
            expected_entities=expected_entities or set(),
            retrieved_entities=retrieved_entities or set(),
            expected_papers=expected_papers or set(),
            retrieved_papers=retrieved_papers or set(),
            expected_authors=expected_authors or set(),
            retrieved_authors=retrieved_authors or set(),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            nodes_retrieved=nodes_retrieved,
            edges_traversed=edges_traversed,
            metadata=kwargs
        )

        # Calculate entity metrics
        if expected_entities and retrieved_entities:
            result.entity_precision = self.calculate_precision(
                retrieved_entities, expected_entities
            )
            result.entity_recall = self.calculate_recall(
                retrieved_entities, expected_entities
            )
            result.entity_f1 = self.calculate_f1(
                result.entity_precision, result.entity_recall
            )

        # Calculate coverage
        if expected_papers:
            result.coverage = self.calculate_coverage(
                len(retrieved_papers or set()),
                len(expected_papers)
            )

        return result

    def extract_entities_from_answer(
        self,
        answer: str,
        entity_list: Optional[List[str]] = None
    ) -> Set[str]:
        """Extract entities mentioned in answer text.

        Args:
            answer: Answer text
            entity_list: Optional list of known entities to look for

        Returns:
            Set of entities found in answer
        """
        if not entity_list:
            # Simple extraction: look for quoted text and capitalized phrases
            entities = set()

            # Extract quoted text
            quoted = re.findall(r'"([^"]*)"', answer)
            entities.update(quoted)

            # Extract capitalized phrases (simplified)
            # This would be better with NER in production
            words = answer.split()
            current_phrase = []
            for word in words:
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word and clean_word[0].isupper() and clean_word.isalpha():
                    current_phrase.append(clean_word)
                else:
                    if len(current_phrase) >= 2:  # Multi-word entities
                        entities.add(" ".join(current_phrase))
                    current_phrase = []

            if len(current_phrase) >= 2:
                entities.add(" ".join(current_phrase))

            return entities
        else:
            # Check which known entities are mentioned
            answer_lower = answer.lower()
            return {
                entity for entity in entity_list
                if entity.lower() in answer_lower
            }


@dataclass
class BenchmarkSummary:
    """Summary statistics from a benchmark run.

    Example:
        summary = BenchmarkSummary(
            total_queries=50,
            avg_latency_ms=1250.5,
            avg_precision=0.85,
            avg_recall=0.78,
            avg_f1=0.81
        )
    """

    # Query statistics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # Performance metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Quality metrics
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0
    avg_coverage: float = 0.0

    # Resource usage
    total_tokens_used: int = 0
    avg_tokens_per_query: float = 0.0
    total_nodes_retrieved: int = 0
    total_edges_traversed: int = 0

    # Breakdown by query type (optional)
    by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_precision": self.avg_precision,
            "avg_recall": self.avg_recall,
            "avg_f1": self.avg_f1,
            "avg_coverage": self.avg_coverage,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_query": self.avg_tokens_per_query,
            "total_nodes_retrieved": self.total_nodes_retrieved,
            "total_edges_traversed": self.total_edges_traversed,
            "by_query_type": self.by_query_type
        }


def compute_summary(results: List[EvaluationResult]) -> BenchmarkSummary:
    """Compute summary statistics from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        BenchmarkSummary with aggregated statistics
    """
    if not results:
        return BenchmarkSummary()

    # Count queries
    total_queries = len(results)
    successful = len([r for r in results if r.actual_answer])
    failed = total_queries - successful

    # Latency statistics
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    # Percentiles
    sorted_latencies = sorted(latencies) if latencies else []
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0.0
    p99_idx = int(len(sorted_latencies) * 0.99)
    p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0.0

    # Quality metrics
    precisions = [r.entity_precision for r in results if r.entity_precision is not None]
    recalls = [r.entity_recall for r in results if r.entity_recall is not None]
    f1s = [r.entity_f1 for r in results if r.entity_f1 is not None]
    coverages = [r.coverage for r in results if r.coverage is not None]

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    # Resource usage
    total_tokens = sum(r.tokens_used for r in results)
    avg_tokens = total_tokens / total_queries if total_queries else 0
    total_nodes = sum(r.nodes_retrieved for r in results)
    total_edges = sum(r.edges_traversed for r in results)

    return BenchmarkSummary(
        total_queries=total_queries,
        successful_queries=successful,
        failed_queries=failed,
        avg_latency_ms=avg_latency,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_f1=avg_f1,
        avg_coverage=avg_coverage,
        total_tokens_used=total_tokens,
        avg_tokens_per_query=avg_tokens,
        total_nodes_retrieved=total_nodes,
        total_edges_traversed=total_edges
    )
