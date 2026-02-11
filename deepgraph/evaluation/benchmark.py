"""
Benchmarking utilities for Graph RAG systems.

This module provides tools for running benchmarks and comparing
different pipeline configurations.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from deepgraph.pipeline.base import GraphRAGPipeline
from deepgraph.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationResult,
    BenchmarkSummary,
    compute_summary
)
from deepgraph.evaluation.datasets import TestCase


class Benchmark:
    """Benchmark runner for Graph RAG pipelines.

    Example:
        from deepgraph.evaluation import Benchmark, create_test_dataset

        benchmark = Benchmark(pipeline, metrics=EvaluationMetrics())

        # Create test dataset
        test_cases = create_test_dataset()

        # Run benchmark
        results = benchmark.run(
            test_cases=test_cases,
            output_path="results/benchmark.json"
        )

        # Print summary
        summary = benchmark.summary()
        print(f"Avg Latency: {summary.avg_latency_ms:.0f}ms")
        print(f"Avg F1: {summary.avg_f1:.2%}")
    """

    def __init__(
        self,
        pipeline: GraphRAGPipeline,
        metrics: Optional[EvaluationMetrics] = None
    ):
        """Initialize benchmark.

        Args:
            pipeline: GraphRAGPipeline instance to benchmark
            metrics: Optional EvaluationMetrics instance
        """
        self._pipeline = pipeline
        self._metrics = metrics or EvaluationMetrics()
        self._results: List[EvaluationResult] = []

    def run(
        self,
        test_cases: List[TestCase],
        output_path: Optional[str] = None,
        verbose: bool = False
    ) -> List[EvaluationResult]:
        """Run benchmark on test cases.

        Args:
            test_cases: List of test cases to evaluate
            output_path: Optional path to save results JSON
            verbose: Print progress

        Returns:
            List of EvaluationResult objects

        Example:
            results = benchmark.run(test_cases, output_path="results.json")
        """
        self._results = []

        if verbose:
            print(f"Running benchmark with {len(test_cases)} test cases...")
            print()

        for i, test_case in enumerate(test_cases, 1):
            if verbose:
                print(f"[{i}/{len(test_cases)}] {test_case.query}")

            try:
                # Run query
                start_time = time.time()
                result = self._pipeline.run(
                    query=test_case.query,
                    top_k=test_case.top_k or 10,
                    use_reasoning=test_case.use_reasoning
                )
                latency_ms = (time.time() - start_time) * 1000

                # Extract entities from result
                retrieved_entities = self._extract_entities_from_result(result)

                # Evaluate
                eval_result = self._metrics.evaluate_result(
                    query=test_case.query,
                    actual_answer=result.answer,
                    expected_entities=set(test_case.expected_entities or []),
                    retrieved_entities=retrieved_entities,
                    expected_papers=set(test_case.expected_papers or []),
                    retrieved_papers=self._extract_papers_from_result(result),
                    expected_authors=set(test_case.expected_authors or []),
                    retrieved_authors=self._extract_authors_from_result(result),
                    latency_ms=latency_ms,
                    tokens_used=result.tokens_used or 0,
                    nodes_retrieved=result.nodes_retrieved,
                    edges_traversed=result.edges_traversed,
                    query_type=test_case.query_type
                )

                self._results.append(eval_result)

                if verbose:
                    print(f"  Latency: {latency_ms:.0f}ms")
                    if eval_result.entity_f1 is not None:
                        print(f"  F1: {eval_result.entity_f1:.2%}")
                    print()

            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                    print()

                # Record failed query
                error_result = EvaluationResult(
                    query=test_case.query,
                    actual_answer=None,
                    metadata={"error": str(e)}
                )
                self._results.append(error_result)

        # Save results if path provided
        if output_path:
            self.save_results(output_path)

        if verbose:
            print(f"\nCompleted {len(self._results)} queries")
            print(f"Success rate: {self.success_rate():.1%}")

        return self._results

    def _extract_entities_from_result(self, result) -> set:
        """Extract entities from query result."""
        entities = set()

        # Extract from answer
        if result.answer:
            entities.update(
                self._metrics.extract_entities_from_answer(result.answer)
            )

        return entities

    def _extract_papers_from_result(self, result) -> set:
        """Extract paper IDs/titles from result context."""
        papers = set()

        if hasattr(result, 'context') and result.context:
            for paper in result.context.get("papers", []):
                paper_id = paper.get("arxiv_id") or paper.get("title")
                if paper_id:
                    papers.add(paper_id)

        return papers

    def _extract_authors_from_result(self, result) -> set:
        """Extract author names from result context."""
        authors = set()

        if hasattr(result, 'context') and result.context:
            for author in result.context.get("authors", []):
                author_name = author.get("name")
                if author_name:
                    authors.add(author_name)

        return authors

    def summary(self) -> BenchmarkSummary:
        """Get summary statistics from benchmark results.

        Returns:
            BenchmarkSummary with aggregated metrics
        """
        return compute_summary(self._results)

    def success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Fraction of successful queries (0-1)
        """
        if not self._results:
            return 0.0

        successful = len([r for r in self._results if r.actual_answer])
        return successful / len(self._results)

    def save_results(self, output_path: str) -> None:
        """Save benchmark results to JSON file.

        Args:
            output_path: Path to save results
        """
        # Convert results to dictionaries
        results_data = []
        for result in self._results:
            results_data.append({
                "query": result.query,
                "actual_answer": result.actual_answer,
                "latency_ms": result.latency_ms,
                "tokens_used": result.tokens_used,
                "nodes_retrieved": result.nodes_retrieved,
                "edges_traversed": result.edges_traversed,
                "entity_precision": result.entity_precision,
                "entity_recall": result.entity_recall,
                "entity_f1": result.entity_f1,
                "coverage": result.coverage,
                "metadata": result.metadata
            })

        # Compute summary
        summary = self.summary()

        # Save to JSON
        output = {
            "summary": summary.to_dict(),
            "results": results_data
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    def load_results(self, input_path: str) -> None:
        """Load benchmark results from JSON file.

        Args:
            input_path: Path to results JSON
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Reconstruct results
        self._results = []
        for result_data in data.get("results", []):
            result = EvaluationResult(
                query=result_data["query"],
                actual_answer=result_data.get("actual_answer"),
                latency_ms=result_data.get("latency_ms", 0),
                tokens_used=result_data.get("tokens_used", 0),
                nodes_retrieved=result_data.get("nodes_retrieved", 0),
                edges_traversed=result_data.get("edges_traversed", 0),
                entity_precision=result_data.get("entity_precision"),
                entity_recall=result_data.get("entity_recall"),
                entity_f1=result_data.get("entity_f1"),
                coverage=result_data.get("coverage"),
                metadata=result_data.get("metadata", {})
            )
            self._results.append(result)

    def compare(
        self,
        other: "Benchmark",
        metric: str = "avg_f1"
    ) -> Dict[str, Any]:
        """Compare this benchmark with another.

        Args:
            other: Another Benchmark instance
            metric: Metric to compare (default: avg_f1)

        Returns:
            Dictionary with comparison results
        """
        self_summary = self.summary()
        other_summary = other.summary()

        self_value = getattr(self_summary, metric, 0)
        other_value = getattr(other_summary, metric, 0)

        improvement = ((self_value - other_value) / other_value * 100
                      if other_value > 0 else 0)

        return {
            "metric": metric,
            "baseline": other_value,
            "current": self_value,
            "improvement_percent": improvement,
            "better": self_value > other_value
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.summary()

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"\nQueries:")
        print(f"  Total:      {summary.total_queries}")
        print(f"  Successful: {summary.successful_queries}")
        print(f"  Failed:     {summary.failed_queries}")
        print(f"  Success Rate: {self.success_rate():.1%}")

        print(f"\nLatency:")
        print(f"  Average: {summary.avg_latency_ms:.0f}ms")
        print(f"  Min:     {summary.min_latency_ms:.0f}ms")
        print(f"  Max:     {summary.max_latency_ms:.0f}ms")
        print(f"  P50:     {summary.p50_latency_ms:.0f}ms")
        print(f"  P95:     {summary.p95_latency_ms:.0f}ms")
        print(f"  P99:     {summary.p99_latency_ms:.0f}ms")

        if summary.avg_f1 > 0:
            print(f"\nQuality:")
            print(f"  Precision: {summary.avg_precision:.2%}")
            print(f"  Recall:    {summary.avg_recall:.2%}")
            print(f"  F1 Score:  {summary.avg_f1:.2%}")
            print(f"  Coverage:  {summary.avg_coverage:.2%}")

        print(f"\nResources:")
        print(f"  Total Tokens: {summary.total_tokens_used:,}")
        print(f"  Avg Tokens/Query: {summary.avg_tokens_per_query:.0f}")
        print(f"  Total Nodes: {summary.total_nodes_retrieved}")
        print(f"  Total Edges: {summary.total_edges_traversed}")

        print("="*60 + "\n")

    def get_results(self) -> List[EvaluationResult]:
        """Get all evaluation results.

        Returns:
            List of EvaluationResult objects
        """
        return self._results
