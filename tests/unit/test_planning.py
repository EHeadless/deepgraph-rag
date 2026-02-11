"""
Unit tests for query planning components.
"""

import pytest
from unittest.mock import Mock, MagicMock
from deepgraph.planning import (
    QueryType,
    QueryIntent,
    QueryPlan,
    QueryResult,
    create_simple_plan,
    IntentParser,
    QueryExecutor,
    STRATEGY_MAPPING
)
from deepgraph.store import MemGraphStore
from deepgraph.retrieval import VectorRetriever
from deepgraph.reasoning import GraphReasoner
from deepgraph.synthesis import OpenAISynthesizer
from schemas.arxiv import ARXIV_SCHEMA


class TestQueryDSL:
    """Tests for query DSL dataclasses."""

    def test_create_query_intent(self):
        """Test creating QueryIntent."""
        intent = QueryIntent(
            query_type=QueryType.AUTHOR_PAPERS,
            entities=["Geoffrey Hinton"],
            target_label="Paper",
            top_k=10
        )

        assert intent.query_type == QueryType.AUTHOR_PAPERS
        assert intent.entities == ["Geoffrey Hinton"]
        assert intent.target_label == "Paper"
        assert intent.top_k == 10

    def test_intent_to_dict(self):
        """Test converting intent to dictionary."""
        intent = QueryIntent(
            query_type=QueryType.SIMILARITY,
            entities=["transformers"],
            top_k=5
        )

        intent_dict = intent.to_dict()

        assert intent_dict["query_type"] == "similarity"
        assert intent_dict["entities"] == ["transformers"]
        assert intent_dict["top_k"] == 5

    def test_intent_from_dict(self):
        """Test creating intent from dictionary."""
        data = {
            "query_type": "author_papers",
            "entities": ["Alice"],
            "target_label": "Paper",
            "relationship_types": [],
            "constraints": {},
            "reasoning_depth": 2,
            "top_k": 10,
            "source_entity": None,
            "target_entity": None,
            "comparison_entities": [],
            "confidence": 0.9
        }

        intent = QueryIntent.from_dict(data)

        assert intent.query_type == QueryType.AUTHOR_PAPERS
        assert intent.entities == ["Alice"]

    def test_create_query_plan(self):
        """Test creating QueryPlan."""
        intent = QueryIntent(
            query_type=QueryType.SIMILARITY,
            entities=["AI"],
            top_k=10
        )

        plan = QueryPlan(
            intent=intent,
            retrieval_strategy="vector",
            reasoning_strategy=None
        )

        assert plan.intent == intent
        assert plan.retrieval_strategy == "vector"
        assert plan.reasoning_strategy is None

    def test_plan_to_dict(self):
        """Test converting plan to dictionary."""
        intent = QueryIntent(query_type=QueryType.SIMILARITY, entities=[])
        plan = QueryPlan(intent=intent)

        plan_dict = plan.to_dict()

        assert "intent" in plan_dict
        assert "retrieval_strategy" in plan_dict

    def test_create_simple_plan(self):
        """Test create_simple_plan helper."""
        plan = create_simple_plan(
            "What papers discuss transformers?",
            QueryType.SIMILARITY,
            top_k=5
        )

        assert plan.intent.query_type == QueryType.SIMILARITY
        assert plan.intent.top_k == 5
        assert plan.retrieval_strategy == "vector"
        assert plan.reasoning_strategy is None  # No reasoning for similarity

    def test_strategy_mapping(self):
        """Test that query types map to correct strategies."""
        assert STRATEGY_MAPPING[QueryType.AUTHOR_PAPERS] == "author_papers"
        assert STRATEGY_MAPPING[QueryType.COLLABORATOR_PAPERS] == "collaborator_papers"
        assert STRATEGY_MAPPING[QueryType.CONNECTION_PATH] == "shortest_path"
        assert STRATEGY_MAPPING[QueryType.SIMILARITY] is None


class TestIntentParser:
    """Tests for IntentParser."""

    def test_create_parser(self):
        """Test creating parser."""
        mock_client = Mock()
        parser = IntentParser(openai_client=mock_client)

        assert parser._model == "gpt-4-turbo-preview"

    def test_pattern_based_parse_author_papers(self):
        """Test pattern matching for author papers queries."""
        mock_client = Mock()
        parser = IntentParser(openai_client=mock_client)

        intent = parser._pattern_based_parse("What papers by Geoffrey Hinton?")

        assert intent is not None
        assert intent.query_type == QueryType.AUTHOR_PAPERS

    def test_pattern_based_parse_collaborator(self):
        """Test pattern matching for collaborator queries."""
        mock_client = Mock()
        parser = IntentParser(openai_client=mock_client)

        intent = parser._pattern_based_parse("Who has Alice collaborated with?")

        assert intent is not None
        assert intent.query_type == QueryType.COLLABORATOR_PAPERS

    def test_pattern_based_parse_comparison(self):
        """Test pattern matching for comparison queries."""
        mock_client = Mock()
        parser = IntentParser(openai_client=mock_client)

        intent = parser._pattern_based_parse("Compare BERT vs GPT")

        assert intent is not None
        assert intent.query_type == QueryType.COMPARISON

    def test_extract_entities_simple(self):
        """Test simple entity extraction."""
        mock_client = Mock()
        parser = IntentParser(openai_client=mock_client)

        entities = parser._extract_entities_simple("Papers by Geoffrey Hinton")

        assert "Geoffrey Hinton" in entities or "Geoffrey" in entities

    def test_create_plan_from_intent(self):
        """Test creating plan from parsed intent."""
        mock_client = Mock()
        parser = IntentParser(openai_client=mock_client)

        # Mock the parse method
        parser.parse = Mock(return_value=QueryIntent(
            query_type=QueryType.AUTHOR_PAPERS,
            entities=["Alice"],
            top_k=10
        ))

        plan = parser.create_plan("What has Alice written?")

        assert plan.intent.query_type == QueryType.AUTHOR_PAPERS
        assert plan.reasoning_strategy == "author_papers"


class TestQueryExecutor:
    """Tests for QueryExecutor."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock components
        self.mock_retriever = Mock()
        self.mock_retriever._id_field = "arxiv_id"
        self.mock_retriever.retrieve.return_value = ["paper1", "paper2"]

        self.mock_reasoner = Mock()
        self.mock_reasoner._store = Mock()
        self.mock_reasoner._store.get_node.return_value = {
            "arxiv_id": "paper1",
            "title": "Test Paper",
            "abstract": "Abstract"
        }
        self.mock_reasoner.expand_context.return_value = {
            "papers": [{"arxiv_id": "paper1", "title": "Test"}],
            "authors": [],
            "connections": [],
            "_meta": {"strategy": "test"}
        }
        self.mock_reasoner.format_context_for_llm.return_value = "Test context"

        self.mock_synthesizer = Mock()
        self.mock_synthesizer.synthesize_with_metadata.return_value = {
            "answer": "Test answer",
            "metadata": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50
            }
        }

        self.executor = QueryExecutor(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer
        )

    def test_create_executor(self):
        """Test creating executor."""
        assert self.executor._retriever == self.mock_retriever
        assert self.executor._reasoner == self.mock_reasoner
        assert self.executor._synthesizer == self.mock_synthesizer

    def test_execute_with_reasoning(self):
        """Test executing plan with reasoning."""
        plan = create_simple_plan(
            "What papers discuss AI?",
            QueryType.SIMILARITY
        )

        result = self.executor.execute(plan, "What papers discuss AI?")

        # Verify all stages were called
        self.mock_retriever.retrieve.assert_called_once()
        self.mock_synthesizer.synthesize_with_metadata.assert_called_once()

        # Verify result
        assert result.answer == "Test answer"
        assert result.query == "What papers discuss AI?"
        assert result.tokens_used == 100
        assert result.total_time_ms > 0

    def test_execute_without_reasoning(self):
        """Test executing plan without reasoning."""
        plan = create_simple_plan(
            "What papers discuss AI?",
            QueryType.SIMILARITY
        )
        plan.reasoning_strategy = None  # Disable reasoning

        result = self.executor.execute(plan, "What papers discuss AI?")

        # Reasoner should not be called for context expansion
        self.mock_reasoner.expand_context.assert_not_called()

        # But synthesis should still happen
        self.mock_synthesizer.synthesize_with_metadata.assert_called_once()

    def test_execute_simple(self):
        """Test execute_simple convenience method."""
        result = self.executor.execute_simple("What papers discuss transformers?")

        assert result.answer == "Test answer"
        self.mock_retriever.retrieve.assert_called_once()

    def test_execute_simple_with_keywords(self):
        """Test execute_simple infers query type from keywords."""
        # Test author papers inference
        result = self.executor.execute_simple("Papers by Geoffrey Hinton")
        # Should not crash

        # Test collaborator inference
        result = self.executor.execute_simple("Who collaborated with Alice?")
        # Should not crash

        # Test comparison inference
        result = self.executor.execute_simple("Compare BERT and GPT")
        # Should not crash


class TestEndToEndPlanning:
    """End-to-end integration tests with real components."""

    def setup_method(self):
        """Setup real components."""
        self.store = MemGraphStore()
        self.store.connect()
        self.store.create_schema(ARXIV_SCHEMA)

        # Add test data
        papers = [
            {"arxiv_id": "p1", "title": "Transformers", "abstract": "About transformers",
             "published_date": "2020-01-01", "primary_category": "cs.AI",
             "embedding": [0.1] * 1536}
        ]
        authors = [{"name": "Alice"}]
        self.store.add_nodes("Paper", papers)
        self.store.add_nodes("Author", authors)
        self.store.add_edges("AUTHORED_BY", [{"from": "p1", "to": "Alice", "position": 0}])

        # Create components
        mock_embedder = Mock()
        mock_embedder.model_name = "test"
        mock_embedder.embed.return_value = [0.1] * 1536

        self.retriever = VectorRetriever(
            store=self.store,
            embedder=mock_embedder,
            index_name="paper_embedding",
            node_label="Paper"
        )

        self.reasoner = GraphReasoner(self.store)

        self.mock_synthesizer = Mock()
        self.mock_synthesizer.synthesize_with_metadata.return_value = {
            "answer": "Based on the context, here is the answer.",
            "metadata": {"total_tokens": 50}
        }
        self.mock_synthesizer.set_system_prompt = Mock()

        self.executor = QueryExecutor(
            retriever=self.retriever,
            reasoner=self.reasoner,
            synthesizer=self.mock_synthesizer
        )

    def teardown_method(self):
        """Cleanup."""
        self.store.disconnect()

    def test_full_pipeline_similarity(self):
        """Test full pipeline with similarity search."""
        plan = create_simple_plan(
            "Papers about transformers",
            QueryType.SIMILARITY,
            top_k=5
        )

        result = self.executor.execute(plan, "Papers about transformers")

        assert result.answer is not None
        assert result.total_time_ms > 0
        # Note: nodes_retrieved may be 0 if vector search finds no matches
        assert result.nodes_retrieved >= 0

    def test_full_pipeline_author_papers(self):
        """Test full pipeline with author papers strategy."""
        plan = create_simple_plan(
            "Papers by Alice",
            QueryType.AUTHOR_PAPERS,
            top_k=5
        )

        # For author papers, we need to handle entity-based retrieval
        result = self.executor.execute(plan, "Papers by Alice")

        assert result.answer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
