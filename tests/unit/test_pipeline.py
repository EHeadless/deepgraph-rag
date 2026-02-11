"""
Unit tests for pipeline components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from deepgraph.pipeline import (
    GraphRAGPipeline,
    PipelineConfig,
    PipelineContext,
    PipelineStage,
    StageType,
    create_simple_pipeline,
    create_multi_hop_pipeline
)
from deepgraph.store import MemGraphStore
from deepgraph.retrieval import VectorRetriever
from deepgraph.reasoning import GraphReasoner
from deepgraph.synthesis import OpenAISynthesizer
from deepgraph.planning import IntentParser, QueryResult
from schemas.arxiv import ARXIV_SCHEMA


class TestPipelineStages:
    """Tests for pipeline stage definitions."""

    def test_create_pipeline_stage(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage(
            name="retrieval",
            stage_type=StageType.RETRIEVAL,
            enabled=True
        )

        assert stage.name == "retrieval"
        assert stage.stage_type == StageType.RETRIEVAL
        assert stage.enabled is True

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = PipelineConfig(
            use_planning=True,
            verbose=False,
            max_retries=3
        )

        assert config.use_planning is True
        assert config.verbose is False
        assert config.max_retries == 3

    def test_pipeline_context(self):
        """Test pipeline context."""
        context = PipelineContext(query="Test query")

        assert context.query == "Test query"
        assert context.completed is False
        assert context.error_occurred is False
        assert len(context.errors) == 0


class TestGraphRAGPipeline:
    """Tests for GraphRAGPipeline."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mocks
        self.mock_retriever = Mock()
        self.mock_retriever._id_field = "arxiv_id"
        self.mock_retriever.retrieve.return_value = ["p1", "p2"]

        self.mock_reasoner = Mock()
        self.mock_reasoner._store = Mock()
        self.mock_reasoner._store.get_node.return_value = {
            "arxiv_id": "p1",
            "title": "Test Paper"
        }
        self.mock_reasoner.expand_context.return_value = {
            "papers": [{"arxiv_id": "p1", "title": "Test"}],
            "authors": [],
            "connections": [],
            "_meta": {"strategy": "test", "description": "test"}
        }
        self.mock_reasoner.format_context_for_llm.return_value = "Test context"

        self.mock_synthesizer = Mock()
        self.mock_synthesizer.synthesize_with_metadata.return_value = {
            "answer": "Test answer",
            "metadata": {"total_tokens": 100}
        }
        self.mock_synthesizer.set_system_prompt = Mock()

        self.mock_parser = Mock()
        self.mock_parser.create_plan = Mock()

    def test_create_pipeline_without_parser(self):
        """Test creating pipeline without intent parser."""
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer,
            intent_parser=None
        )

        assert pipeline.retriever == self.mock_retriever
        assert pipeline.reasoner == self.mock_reasoner
        assert pipeline.synthesizer == self.mock_synthesizer
        assert pipeline.intent_parser is None

    def test_create_pipeline_with_parser(self):
        """Test creating pipeline with intent parser."""
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer,
            intent_parser=self.mock_parser
        )

        assert pipeline.intent_parser == self.mock_parser

    def test_run_simple_pipeline(self):
        """Test running pipeline without planning."""
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer,
            intent_parser=None,
            config=PipelineConfig(use_planning=False)
        )

        result = pipeline.run("Test query", top_k=5)

        assert isinstance(result, QueryResult)
        assert result.answer == "Test answer"

    def test_disable_stage(self):
        """Test disabling a pipeline stage."""
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer
        )

        pipeline.disable_stage("reasoning")

        stages = pipeline.get_stages()
        assert stages["reasoning"].enabled is False

    def test_enable_stage(self):
        """Test enabling a pipeline stage."""
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer
        )

        pipeline.disable_stage("reasoning")
        pipeline.enable_stage("reasoning")

        stages = pipeline.get_stages()
        assert stages["reasoning"].enabled is True

    def test_set_stage_hook(self):
        """Test setting stage hooks."""
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer
        )

        hook_called = []

        def test_hook(context):
            hook_called.append(True)

        pipeline.set_stage_hook("retrieval", "pre", test_hook)

        stages = pipeline.get_stages()
        assert stages["retrieval"].pre_hook is not None

    def test_pipeline_config_access(self):
        """Test accessing pipeline configuration."""
        config = PipelineConfig(verbose=True)
        pipeline = GraphRAGPipeline(
            retriever=self.mock_retriever,
            reasoner=self.mock_reasoner,
            synthesizer=self.mock_synthesizer,
            config=config
        )

        retrieved_config = pipeline.get_config()
        assert retrieved_config.verbose is True


class TestPrebuiltPipelines:
    """Tests for pre-built pipeline factory functions."""

    @patch('deepgraph.pipeline.prebuilt.OpenAI')
    def test_create_simple_pipeline(self, mock_openai_class):
        """Test creating simple pipeline."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        store = MemGraphStore()
        store.connect()
        store.create_schema(ARXIV_SCHEMA)

        pipeline = create_simple_pipeline(
            store=store,
            openai_api_key="sk-test"
        )

        assert isinstance(pipeline, GraphRAGPipeline)
        assert pipeline.intent_parser is None
        assert pipeline._config.use_planning is False

        # Reasoning should be disabled
        stages = pipeline.get_stages()
        assert stages["reasoning"].enabled is False

        store.disconnect()

    @patch('deepgraph.pipeline.prebuilt.OpenAI')
    def test_create_multi_hop_pipeline(self, mock_openai_class):
        """Test creating multi-hop pipeline."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        store = MemGraphStore()
        store.connect()
        store.create_schema(ARXIV_SCHEMA)

        pipeline = create_multi_hop_pipeline(
            store=store,
            openai_api_key="sk-test",
            use_planning=True
        )

        assert isinstance(pipeline, GraphRAGPipeline)
        assert pipeline.intent_parser is not None
        assert pipeline._config.use_planning is True

        store.disconnect()

    @patch('deepgraph.pipeline.prebuilt.OpenAI')
    def test_create_multi_hop_without_planning(self, mock_openai_class):
        """Test creating multi-hop pipeline without planning."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        store = MemGraphStore()
        store.connect()
        store.create_schema(ARXIV_SCHEMA)

        pipeline = create_multi_hop_pipeline(
            store=store,
            openai_api_key="sk-test",
            use_planning=False
        )

        assert pipeline.intent_parser is None
        assert pipeline._config.use_planning is False

        store.disconnect()


class TestEndToEndPipeline:
    """End-to-end tests with real components."""

    def setup_method(self):
        """Setup real components."""
        self.store = MemGraphStore()
        self.store.connect()
        self.store.create_schema(ARXIV_SCHEMA)

        # Add test data
        papers = [
            {
                "arxiv_id": "p1",
                "title": "Transformers in NLP",
                "abstract": "About transformers",
                "published_date": "2020-01-01",
                "primary_category": "cs.AI",
                "embedding": [0.1] * 1536
            }
        ]
        authors = [{"name": "Alice"}]
        self.store.add_nodes("Paper", papers)
        self.store.add_nodes("Author", authors)
        self.store.add_edges("AUTHORED_BY", [
            {"from": "p1", "to": "Alice", "position": 0}
        ])

    def teardown_method(self):
        """Cleanup."""
        self.store.disconnect()

    @patch('deepgraph.pipeline.prebuilt.OpenAI')
    def test_simple_pipeline_execution(self, mock_openai_class):
        """Test executing simple pipeline."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_embeddings.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_embeddings

        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test answer"))]
        mock_chat_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100
        )
        mock_chat_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_chat_response

        mock_openai_class.return_value = mock_client

        # Create pipeline
        pipeline = create_simple_pipeline(
            store=self.store,
            openai_api_key="sk-test"
        )

        # Run query
        result = pipeline.run("Papers about transformers", top_k=5)

        assert result is not None
        assert result.answer == "Test answer"
        assert result.total_time_ms > 0

    @patch('deepgraph.pipeline.prebuilt.OpenAI')
    def test_multi_hop_pipeline_execution(self, mock_openai_class):
        """Test executing multi-hop pipeline."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_embeddings.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_embeddings

        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test answer"))]
        mock_chat_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100
        )
        mock_chat_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_chat_response

        mock_openai_class.return_value = mock_client

        # Create pipeline without planning (to avoid LLM call for intent parsing)
        pipeline = create_multi_hop_pipeline(
            store=self.store,
            openai_api_key="sk-test",
            use_planning=False
        )

        # Run query
        result = pipeline.run("Papers by Alice", top_k=5, use_reasoning=True)

        assert result is not None
        assert result.answer == "Test answer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
