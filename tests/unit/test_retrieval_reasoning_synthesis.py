"""
Unit tests for retrieval, reasoning, and synthesis components.
"""

import pytest
from unittest.mock import Mock, MagicMock
from deepgraph.adapters.embedders import OpenAIEmbedder
from deepgraph.retrieval import VectorRetriever
from deepgraph.reasoning import (
    AuthorPapersStrategy,
    CollaboratorPapersStrategy,
    ShortestPathStrategy,
    SubgraphStrategy,
    GraphReasoner
)
from deepgraph.synthesis import (
    OpenAISynthesizer,
    PromptTemplates,
    detect_query_type,
    get_prompt_for_query
)
from deepgraph.store import MemGraphStore
from schemas.arxiv import ARXIV_SCHEMA


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    def test_create_embedder(self):
        """Test creating embedder instance."""
        # Mock OpenAI client
        mock_client = Mock()
        embedder = OpenAIEmbedder(client=mock_client)

        assert embedder.model_name == "text-embedding-ada-002"
        assert embedder.dimensions == 1536

    def test_embed_dimensions(self):
        """Test embedding dimensions for different models."""
        mock_client = Mock()

        # Ada-002
        embedder_ada = OpenAIEmbedder(client=mock_client, model="text-embedding-ada-002")
        assert embedder_ada.dimensions == 1536

        # 3-large
        embedder_large = OpenAIEmbedder(client=mock_client, model="text-embedding-3-large")
        assert embedder_large.dimensions == 3072

    def test_unsupported_model(self):
        """Test error on unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            OpenAIEmbedder(model="invalid-model")


class TestVectorRetriever:
    """Tests for VectorRetriever."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock store
        self.mock_store = Mock()
        self.mock_embedder = Mock()
        self.mock_embedder.model_name = "text-embedding-ada-002"
        self.mock_embedder.embed.return_value = [0.1] * 1536

        self.retriever = VectorRetriever(
            store=self.mock_store,
            embedder=self.mock_embedder,
            index_name="paper_embedding",
            node_label="Paper",
            id_field="arxiv_id"
        )

    def test_create_retriever(self):
        """Test creating retriever instance."""
        assert self.retriever.index_name == "paper_embedding"
        assert self.retriever.node_label == "Paper"

    def test_retrieve_calls_embedder(self):
        """Test that retrieve calls embedder and store."""
        # Setup mock responses
        self.mock_store.vector_search.return_value = [
            {"node": {"arxiv_id": "paper1", "title": "Paper 1"}, "score": 0.9},
            {"node": {"arxiv_id": "paper2", "title": "Paper 2"}, "score": 0.8}
        ]

        # Retrieve
        candidate_ids = self.retriever.retrieve("test query", top_k=2)

        # Verify embedder was called
        self.mock_embedder.embed.assert_called_once_with("test query")

        # Verify store was called
        self.mock_store.vector_search.assert_called_once()

        # Verify results
        assert candidate_ids == ["paper1", "paper2"]

    def test_retrieve_with_scores(self):
        """Test retrieve_with_scores returns scores."""
        self.mock_store.vector_search.return_value = [
            {"node": {"arxiv_id": "paper1", "title": "Test"}, "score": 0.95}
        ]

        results = self.retriever.retrieve_with_scores("query", top_k=1)

        assert len(results) == 1
        assert results[0]["id"] == "paper1"
        assert results[0]["score"] == 0.95
        assert "node" in results[0]


class TestReasoningStrategies:
    """Tests for reasoning strategies."""

    def setup_method(self):
        """Setup test fixtures."""
        self.store = MemGraphStore()
        self.store.connect()
        self.store.create_schema(ARXIV_SCHEMA)

        # Add test data
        papers = [
            {"arxiv_id": "paper1", "title": "Paper 1", "abstract": "...",
             "published_date": "2020-01-01", "primary_category": "cs.AI"},
            {"arxiv_id": "paper2", "title": "Paper 2", "abstract": "...",
             "published_date": "2020-02-01", "primary_category": "cs.AI"}
        ]
        authors = [
            {"name": "Alice"},
            {"name": "Bob"}
        ]
        self.store.add_nodes("Paper", papers)
        self.store.add_nodes("Author", authors)

        # Add edges
        edges = [
            {"from": "paper1", "to": "Alice", "position": 0},
            {"from": "paper2", "to": "Alice", "position": 0},
            {"from": "paper2", "to": "Bob", "position": 1}
        ]
        self.store.add_edges("AUTHORED_BY", edges)

    def teardown_method(self):
        """Cleanup."""
        self.store.disconnect()

    def test_author_papers_strategy(self):
        """Test AuthorPapersStrategy execution."""
        strategy = AuthorPapersStrategy()

        assert strategy.name == "author_papers"

        result = strategy.execute(
            store=self.store,
            candidate_ids=["paper1"]
        )

        # Should find paper1 and paper2 (both by Alice)
        assert "papers" in result
        assert "authors" in result
        assert "connections" in result
        assert len(result["authors"]) >= 1  # At least Alice

    def test_shortest_path_strategy(self):
        """Test ShortestPathStrategy execution."""
        strategy = ShortestPathStrategy()

        result = strategy.execute(
            store=self.store,
            candidate_ids=["Alice"],
            target_id="Bob",
            from_label="Author",
            to_label="Author"
        )

        # Should find a path Alice → paper2 → Bob
        assert "path" in result
        if result["path"]:
            assert result["length"] > 0

    def test_subgraph_strategy(self):
        """Test SubgraphStrategy execution."""
        strategy = SubgraphStrategy()

        result = strategy.execute(
            store=self.store,
            candidate_ids=["paper1"],
            node_label="Paper",
            id_field="arxiv_id"
        )

        assert "nodes" in result
        assert "edges" in result
        assert result["node_count"] >= 1


class TestGraphReasoner:
    """Tests for GraphReasoner."""

    def setup_method(self):
        """Setup test fixtures."""
        self.store = MemGraphStore()
        self.store.connect()
        self.store.create_schema(ARXIV_SCHEMA)

        # Add test data
        self.store.add_nodes("Paper", [
            {"arxiv_id": "p1", "title": "P1", "abstract": "...",
             "published_date": "2020-01-01", "primary_category": "cs.AI"}
        ])
        self.store.add_nodes("Author", [{"name": "Alice"}])
        self.store.add_edges("AUTHORED_BY", [{"from": "p1", "to": "Alice", "position": 0}])

        self.reasoner = GraphReasoner(self.store)

    def teardown_method(self):
        """Cleanup."""
        self.store.disconnect()

    def test_create_reasoner(self):
        """Test creating reasoner."""
        assert len(self.reasoner.available_strategies) >= 4

    def test_expand_context_with_strategy_name(self):
        """Test expand_context using strategy name."""
        context = self.reasoner.expand_context(
            candidate_ids=["p1"],
            strategy_name="author_papers"
        )

        assert "_meta" in context
        assert context["_meta"]["strategy"] == "author_papers"

    def test_expand_context_with_strategy_instance(self):
        """Test expand_context using strategy instance."""
        strategy = AuthorPapersStrategy()
        context = self.reasoner.expand_context(
            candidate_ids=["p1"],
            strategy=strategy
        )

        assert "_meta" in context
        assert context["_meta"]["strategy"] == "author_papers"

    def test_expand_context_requires_strategy(self):
        """Test that expand_context requires a strategy."""
        with pytest.raises(ValueError, match="Must provide either"):
            self.reasoner.expand_context(candidate_ids=["p1"])

    def test_format_context_for_llm(self):
        """Test formatting context for LLM."""
        context = self.reasoner.expand_context(
            candidate_ids=["p1"],
            strategy_name="author_papers"
        )

        text = self.reasoner.format_context_for_llm(context)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "author_papers" in text

    def test_register_custom_strategy(self):
        """Test registering custom strategy."""
        class CustomStrategy:
            @property
            def name(self):
                return "custom"

            @property
            def description(self):
                return "Custom strategy"

            def execute(self, store, candidate_ids, **kwargs):
                return {"result": "custom"}

        self.reasoner.register_strategy("custom", CustomStrategy())

        assert "custom" in self.reasoner.available_strategies

        context = self.reasoner.expand_context(
            candidate_ids=["p1"],
            strategy_name="custom"
        )

        assert context["result"] == "custom"


class TestOpenAISynthesizer:
    """Tests for OpenAISynthesizer."""

    def test_create_synthesizer(self):
        """Test creating synthesizer."""
        mock_client = Mock()
        synthesizer = OpenAISynthesizer(client=mock_client)

        assert synthesizer.model_name == "gpt-4-turbo-preview"
        assert len(synthesizer.system_prompt) > 0

    def test_set_system_prompt(self):
        """Test updating system prompt."""
        mock_client = Mock()
        synthesizer = OpenAISynthesizer(client=mock_client)

        new_prompt = "Custom prompt"
        synthesizer.set_system_prompt(new_prompt)

        assert synthesizer.system_prompt == new_prompt


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_get_template(self):
        """Test getting template by name."""
        prompt = PromptTemplates.get("paper_recommendation")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_invalid_template(self):
        """Test error on invalid template."""
        with pytest.raises(ValueError, match="Unknown template"):
            PromptTemplates.get("nonexistent")

    def test_available_templates(self):
        """Test getting available templates."""
        templates = PromptTemplates.available_templates()
        assert isinstance(templates, list)
        assert "default" in templates
        assert "paper_recommendation" in templates

    def test_format_context(self):
        """Test context formatting."""
        formatted = PromptTemplates.format_context(
            query="Test question?",
            context="Test context",
            include_instructions=True
        )

        assert "Test question?" in formatted
        assert "Test context" in formatted
        assert "Instructions" in formatted

    def test_detect_query_type(self):
        """Test query type detection."""
        assert detect_query_type("Recommend papers about AI") == "paper_recommendation"
        assert detect_query_type("Who is Geoffrey Hinton?") == "author_expertise"
        assert detect_query_type("Compare BERT and GPT") == "comparison"
        assert detect_query_type("Tell me about transformers") == "summary"
        assert detect_query_type("Random query") == "default"

    def test_get_prompt_for_query(self):
        """Test getting appropriate prompt for query."""
        prompt = get_prompt_for_query("Recommend similar papers")
        assert "recommendation" in prompt.lower()

        prompt_default = get_prompt_for_query("Random question")
        assert prompt_default == PromptTemplates.DEFAULT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
