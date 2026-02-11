"""
Evaluation and benchmarking for Graph RAG systems.

This package provides metrics, benchmarking utilities, and test datasets
for measuring and improving Graph RAG performance.
"""

from deepgraph.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationResult,
    BenchmarkSummary,
    compute_summary
)
from deepgraph.evaluation.benchmark import Benchmark
from deepgraph.evaluation.datasets import (
    TestCase,
    TestDataset,
    create_arxiv_test_dataset,
    create_test_dataset
)

__all__ = [
    "EvaluationMetrics",
    "EvaluationResult",
    "BenchmarkSummary",
    "compute_summary",
    "Benchmark",
    "TestCase",
    "TestDataset",
    "create_arxiv_test_dataset",
    "create_test_dataset",
]
