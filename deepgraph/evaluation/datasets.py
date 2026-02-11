"""
Test dataset management for evaluation.

This module provides utilities for creating and managing test datasets
for benchmarking Graph RAG systems.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class TestCase:
    """A single test case for evaluation.

    Example:
        test_case = TestCase(
            query="What papers discuss transformers?",
            expected_entities=["Attention Is All You Need", "BERT", "GPT-2"],
            expected_papers=["2301.12345", "2301.12346"],
            query_type="similarity",
            top_k=10
        )
    """

    # Query information
    query: str
    query_type: str = "similarity"  # similarity, author_papers, etc.

    # Expected results (ground truth)
    expected_answer: Optional[str] = None
    expected_entities: List[str] = field(default_factory=list)
    expected_papers: List[str] = field(default_factory=list)
    expected_authors: List[str] = field(default_factory=list)

    # Query parameters
    top_k: int = 10
    use_reasoning: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "query_type": self.query_type,
            "expected_answer": self.expected_answer,
            "expected_entities": self.expected_entities,
            "expected_papers": self.expected_papers,
            "expected_authors": self.expected_authors,
            "top_k": self.top_k,
            "use_reasoning": self.use_reasoning,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary."""
        return cls(**data)


class TestDataset:
    """Collection of test cases.

    Example:
        dataset = TestDataset(name="arxiv_benchmark")

        dataset.add_test_case(TestCase(
            query="What papers discuss transformers?",
            expected_entities=["Attention Is All You Need"]
        ))

        dataset.save("datasets/arxiv_benchmark.json")
    """

    def __init__(self, name: str = "default"):
        """Initialize test dataset.

        Args:
            name: Dataset name
        """
        self.name = name
        self._test_cases: List[TestCase] = []

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the dataset.

        Args:
            test_case: TestCase instance
        """
        self._test_cases.append(test_case)

    def get_cases(self, query_type: Optional[str] = None) -> List[TestCase]:
        """Get test cases, optionally filtered by type.

        Args:
            query_type: Optional query type filter

        Returns:
            List of TestCase objects
        """
        if query_type:
            return [tc for tc in self._test_cases if tc.query_type == query_type]
        return self._test_cases

    def save(self, output_path: str) -> None:
        """Save dataset to JSON file.

        Args:
            output_path: Path to save dataset
        """
        data = {
            "name": self.name,
            "num_cases": len(self._test_cases),
            "test_cases": [tc.to_dict() for tc in self._test_cases]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "TestDataset":
        """Load dataset from JSON file.

        Args:
            input_path: Path to dataset JSON

        Returns:
            TestDataset instance
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", "loaded"))

        for tc_data in data.get("test_cases", []):
            dataset.add_test_case(TestCase.from_dict(tc_data))

        return dataset

    def __len__(self) -> int:
        """Get number of test cases."""
        return len(self._test_cases)

    def __repr__(self) -> str:
        """String representation."""
        return f"TestDataset(name='{self.name}', cases={len(self._test_cases)})"


def create_arxiv_test_dataset() -> TestDataset:
    """Create a sample test dataset for arXiv papers.

    Returns:
        TestDataset with sample test cases

    Example:
        dataset = create_arxiv_test_dataset()
        test_cases = dataset.get_cases()
    """
    dataset = TestDataset(name="arxiv_sample")

    # Similarity search queries
    dataset.add_test_case(TestCase(
        query="What papers discuss transformers in natural language processing?",
        query_type="similarity",
        expected_entities=["Attention Is All You Need", "BERT", "transformer"],
        tags=["similarity", "nlp"],
        difficulty="easy"
    ))

    dataset.add_test_case(TestCase(
        query="Papers about deep learning for computer vision",
        query_type="similarity",
        expected_entities=["ResNet", "ImageNet", "convolutional neural network"],
        tags=["similarity", "cv"],
        difficulty="easy"
    ))

    # Author papers queries
    dataset.add_test_case(TestCase(
        query="What papers has Geoffrey Hinton written?",
        query_type="author_papers",
        expected_authors=["Geoffrey Hinton"],
        tags=["author", "single-author"],
        difficulty="easy"
    ))

    dataset.add_test_case(TestCase(
        query="Show me work by Yann LeCun and Yoshua Bengio",
        query_type="author_papers",
        expected_authors=["Yann LeCun", "Yoshua Bengio"],
        tags=["author", "multi-author"],
        difficulty="medium"
    ))

    # Collaborator queries
    dataset.add_test_case(TestCase(
        query="Who has Geoffrey Hinton collaborated with?",
        query_type="collaborator_papers",
        expected_entities=["Geoffrey Hinton"],
        tags=["collaborator", "network"],
        difficulty="medium"
    ))

    # Connection queries
    dataset.add_test_case(TestCase(
        query="How are Geoffrey Hinton and Yann LeCun connected?",
        query_type="connection_path",
        expected_entities=["Geoffrey Hinton", "Yann LeCun"],
        tags=["connection", "path"],
        difficulty="medium"
    ))

    # Comparison queries
    dataset.add_test_case(TestCase(
        query="Compare BERT and GPT-2",
        query_type="comparison",
        expected_entities=["BERT", "GPT-2"],
        tags=["comparison", "models"],
        difficulty="medium"
    ))

    dataset.add_test_case(TestCase(
        query="What's the difference between ResNet and VGG?",
        query_type="comparison",
        expected_entities=["ResNet", "VGG"],
        tags=["comparison", "architectures"],
        difficulty="medium"
    ))

    # Summary queries
    dataset.add_test_case(TestCase(
        query="Tell me about attention mechanisms in deep learning",
        query_type="summary",
        expected_entities=["attention", "transformer"],
        tags=["summary", "concept"],
        difficulty="easy"
    ))

    dataset.add_test_case(TestCase(
        query="What is transfer learning?",
        query_type="summary",
        expected_entities=["transfer learning", "pre-training"],
        tags=["summary", "concept"],
        difficulty="easy"
    ))

    # Complex multi-hop queries
    dataset.add_test_case(TestCase(
        query="Find papers about reinforcement learning by authors who have "
              "collaborated with researchers from DeepMind",
        query_type="collaborator_papers",
        expected_entities=["reinforcement learning", "DeepMind"],
        tags=["complex", "multi-hop"],
        difficulty="hard"
    ))

    dataset.add_test_case(TestCase(
        query="What recent work connects computer vision and natural language processing?",
        query_type="similarity",
        expected_entities=["vision", "language", "multimodal"],
        tags=["complex", "cross-domain"],
        difficulty="hard"
    ))

    return dataset


def create_test_dataset() -> List[TestCase]:
    """Convenience function to get test cases.

    Returns:
        List of TestCase objects

    Example:
        test_cases = create_test_dataset()
        benchmark.run(test_cases)
    """
    dataset = create_arxiv_test_dataset()
    return dataset.get_cases()
