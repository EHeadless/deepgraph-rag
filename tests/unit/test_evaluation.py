"""
Unit tests for evaluation components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from deepgraph.evaluation import (
    EvaluationMetrics,
    EvaluationResult,
    BenchmarkSummary,
    compute_summary,
    Benchmark,
    TestCase,
    TestDataset,
    create_arxiv_test_dataset,
    create_test_dataset
)
from deepgraph.pipeline import GraphRAGPipeline
from deepgraph.planning.dsl import QueryResult


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics."""

    def test_calculate_precision(self):
        """Test precision calculation."""
        predicted = {"A", "B", "C"}
        ground_truth = {"A", "B", "D"}

        precision = EvaluationMetrics.calculate_precision(predicted, ground_truth)

        # 2 correct out of 3 predicted
        assert precision == 2/3

    def test_calculate_precision_empty(self):
        """Test precision with empty prediction."""
        predicted = set()
        ground_truth = {"A", "B"}

        precision = EvaluationMetrics.calculate_precision(predicted, ground_truth)

        assert precision == 0.0

    def test_calculate_recall(self):
        """Test recall calculation."""
        predicted = {"A", "B", "C"}
        ground_truth = {"A", "B", "D", "E"}

        recall = EvaluationMetrics.calculate_recall(predicted, ground_truth)

        # 2 correct out of 4 in ground truth
        assert recall == 2/4

    def test_calculate_recall_empty(self):
        """Test recall with empty ground truth."""
        predicted = {"A", "B"}
        ground_truth = set()

        recall = EvaluationMetrics.calculate_recall(predicted, ground_truth)

        assert recall == 0.0

    def test_calculate_f1(self):
        """Test F1 score calculation."""
        precision = 0.8
        recall = 0.6

        f1 = EvaluationMetrics.calculate_f1(precision, recall)

        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(f1 - expected_f1) < 0.001

    def test_calculate_f1_zero(self):
        """Test F1 with zero precision and recall."""
        f1 = EvaluationMetrics.calculate_f1(0.0, 0.0)
        assert f1 == 0.0

    def test_calculate_coverage(self):
        """Test coverage calculation."""
        coverage = EvaluationMetrics.calculate_coverage(8, 10)
        assert coverage == 0.8

    def test_calculate_coverage_over_100(self):
        """Test coverage caps at 100%."""
        coverage = EvaluationMetrics.calculate_coverage(12, 10)
        assert coverage == 1.0

    def test_evaluate_result(self):
        """Test evaluating a single result."""
        metrics = EvaluationMetrics()

        result = metrics.evaluate_result(
            query="Test query",
            actual_answer="Test answer",
            expected_entities={"A", "B", "C"},
            retrieved_entities={"A", "B", "D"},
            latency_ms=1250.5,
            tokens_used=1500
        )

        assert result.query == "Test query"
        assert result.actual_answer == "Test answer"
        assert result.latency_ms == 1250.5
        assert result.tokens_used == 1500
        assert result.entity_precision == 2/3  # 2 correct out of 3 retrieved
        assert result.entity_recall == 2/3     # 2 correct out of 3 expected
        assert result.entity_f1 is not None

    def test_extract_entities_from_answer(self):
        """Test entity extraction from answer text."""
        metrics = EvaluationMetrics()

        answer = 'The paper "Attention Is All You Need" discusses transformers.'
        entities = metrics.extract_entities_from_answer(answer)

        assert "Attention Is All You Need" in entities

    def test_extract_entities_with_list(self):
        """Test entity extraction with known entity list."""
        metrics = EvaluationMetrics()

        answer = "BERT and GPT-2 are transformer models."
        entity_list = ["BERT", "GPT-2", "GPT-3"]

        entities = metrics.extract_entities_from_answer(answer, entity_list)

        assert "BERT" in entities
        assert "GPT-2" in entities
        assert "GPT-3" not in entities


class TestBenchmarkSummary:
    """Tests for BenchmarkSummary."""

    def test_create_summary(self):
        """Test creating benchmark summary."""
        summary = BenchmarkSummary(
            total_queries=50,
            avg_latency_ms=1250.5,
            avg_f1=0.85
        )

        assert summary.total_queries == 50
        assert summary.avg_latency_ms == 1250.5
        assert summary.avg_f1 == 0.85

    def test_summary_to_dict(self):
        """Test converting summary to dictionary."""
        summary = BenchmarkSummary(
            total_queries=10,
            avg_f1=0.8
        )

        summary_dict = summary.to_dict()

        assert summary_dict["total_queries"] == 10
        assert summary_dict["avg_f1"] == 0.8

    def test_compute_summary(self):
        """Test computing summary from results."""
        results = [
            EvaluationResult(
                query="Q1",
                actual_answer="A1",
                latency_ms=1000,
                tokens_used=100,
                entity_precision=0.8,
                entity_recall=0.7,
                entity_f1=0.75
            ),
            EvaluationResult(
                query="Q2",
                actual_answer="A2",
                latency_ms=1500,
                tokens_used=150,
                entity_precision=0.9,
                entity_recall=0.8,
                entity_f1=0.85
            )
        ]

        summary = compute_summary(results)

        assert summary.total_queries == 2
        assert summary.successful_queries == 2
        assert summary.avg_latency_ms == 1250  # (1000 + 1500) / 2
        assert abs(summary.avg_precision - 0.85) < 0.001   # (0.8 + 0.9) / 2
        assert abs(summary.avg_recall - 0.75) < 0.001      # (0.7 + 0.8) / 2
        assert abs(summary.avg_f1 - 0.8) < 0.001           # (0.75 + 0.85) / 2
        assert summary.total_tokens_used == 250

    def test_compute_summary_empty(self):
        """Test computing summary with empty results."""
        summary = compute_summary([])

        assert summary.total_queries == 0
        assert summary.avg_latency_ms == 0.0


class TestDatasets:
    """Tests for test dataset management."""

    def test_create_test_case(self):
        """Test creating a test case."""
        test_case = TestCase(
            query="What papers discuss transformers?",
            expected_entities=["BERT", "GPT-2"],
            query_type="similarity"
        )

        assert test_case.query == "What papers discuss transformers?"
        assert len(test_case.expected_entities) == 2
        assert test_case.query_type == "similarity"

    def test_test_case_to_dict(self):
        """Test converting test case to dictionary."""
        test_case = TestCase(
            query="Test query",
            expected_entities=["A", "B"]
        )

        test_dict = test_case.to_dict()

        assert test_dict["query"] == "Test query"
        assert test_dict["expected_entities"] == ["A", "B"]

    def test_test_case_from_dict(self):
        """Test creating test case from dictionary."""
        data = {
            "query": "Test query",
            "query_type": "similarity",
            "expected_entities": ["A", "B"],
            "expected_papers": [],
            "expected_authors": [],
            "expected_answer": None,
            "top_k": 10,
            "use_reasoning": True,
            "tags": [],
            "difficulty": "medium",
            "metadata": {}
        }

        test_case = TestCase.from_dict(data)

        assert test_case.query == "Test query"
        assert test_case.expected_entities == ["A", "B"]

    def test_test_dataset(self):
        """Test creating and using test dataset."""
        dataset = TestDataset(name="test_dataset")

        dataset.add_test_case(TestCase(
            query="Query 1",
            query_type="similarity"
        ))
        dataset.add_test_case(TestCase(
            query="Query 2",
            query_type="author_papers"
        ))

        assert len(dataset) == 2

        # Filter by type
        similarity_cases = dataset.get_cases(query_type="similarity")
        assert len(similarity_cases) == 1
        assert similarity_cases[0].query == "Query 1"

    def test_dataset_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = TestDataset(name="test")
        dataset.add_test_case(TestCase(
            query="Test query",
            expected_entities=["A", "B"]
        ))

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            dataset.save(temp_path)

            # Load back
            loaded_dataset = TestDataset.load(temp_path)

            assert loaded_dataset.name == "test"
            assert len(loaded_dataset) == 1
            cases = loaded_dataset.get_cases()
            assert cases[0].query == "Test query"

        finally:
            Path(temp_path).unlink()

    def test_create_arxiv_test_dataset(self):
        """Test creating arXiv test dataset."""
        dataset = create_arxiv_test_dataset()

        assert len(dataset) > 0
        assert dataset.name == "arxiv_sample"

        # Check some cases exist
        cases = dataset.get_cases()
        queries = [c.query for c in cases]
        assert any("transformer" in q.lower() for q in queries)

    def test_create_test_dataset(self):
        """Test convenience function."""
        test_cases = create_test_dataset()

        assert isinstance(test_cases, list)
        assert len(test_cases) > 0
        assert isinstance(test_cases[0], TestCase)


class TestBenchmark:
    """Tests for Benchmark."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock pipeline
        self.mock_pipeline = Mock(spec=GraphRAGPipeline)

        # Mock query result
        mock_result = Mock(spec=QueryResult)
        mock_result.answer = "Test answer"
        mock_result.tokens_used = 100
        mock_result.nodes_retrieved = 5
        mock_result.edges_traversed = 3
        mock_result.context = {
            "papers": [{"arxiv_id": "p1", "title": "Paper 1"}],
            "authors": [{"name": "Alice"}]
        }

        self.mock_pipeline.run.return_value = mock_result

    def test_create_benchmark(self):
        """Test creating benchmark."""
        benchmark = Benchmark(self.mock_pipeline)

        assert benchmark._pipeline == self.mock_pipeline
        assert benchmark._metrics is not None

    def test_run_benchmark(self):
        """Test running benchmark."""
        benchmark = Benchmark(self.mock_pipeline)

        test_cases = [
            TestCase(query="Query 1"),
            TestCase(query="Query 2")
        ]

        results = benchmark.run(test_cases, verbose=False)

        assert len(results) == 2
        assert self.mock_pipeline.run.call_count == 2

    def test_benchmark_summary(self):
        """Test getting benchmark summary."""
        benchmark = Benchmark(self.mock_pipeline)

        test_cases = [TestCase(query="Query 1")]
        benchmark.run(test_cases)

        summary = benchmark.summary()

        assert summary.total_queries == 1
        assert summary.successful_queries == 1

    def test_benchmark_success_rate(self):
        """Test success rate calculation."""
        benchmark = Benchmark(self.mock_pipeline)

        test_cases = [TestCase(query="Q1"), TestCase(query="Q2")]
        benchmark.run(test_cases)

        success_rate = benchmark.success_rate()

        assert success_rate == 1.0  # All successful

    def test_benchmark_save_and_load(self):
        """Test saving and loading benchmark results."""
        benchmark = Benchmark(self.mock_pipeline)

        test_cases = [TestCase(query="Test query")]
        benchmark.run(test_cases)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            benchmark.save_results(temp_path)

            # Load back
            new_benchmark = Benchmark(self.mock_pipeline)
            new_benchmark.load_results(temp_path)

            assert len(new_benchmark.get_results()) == 1

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
