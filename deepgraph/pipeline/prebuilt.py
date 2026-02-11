"""
Pre-built pipeline configurations.

This module provides factory functions for creating common pipeline configurations.
"""

from typing import Optional, Dict, Any
from openai import OpenAI
from deepgraph.pipeline.base import GraphRAGPipeline
from deepgraph.pipeline.stages import PipelineConfig
from deepgraph.store.base import GraphStore, create_graph_store
from deepgraph.core.schema import GraphSchema
from deepgraph.adapters.embedders import OpenAIEmbedder
from deepgraph.retrieval import VectorRetriever
from deepgraph.reasoning import GraphReasoner
from deepgraph.synthesis import OpenAISynthesizer
from deepgraph.planning import IntentParser


def create_simple_pipeline(
    store: GraphStore,
    openai_api_key: str,
    index_name: str = "paper_embedding",
    node_label: str = "Paper",
    id_field: str = "id",
    embedding_model: str = "text-embedding-ada-002",
    chat_model: str = "gpt-4-turbo-preview",
    **kwargs
) -> GraphRAGPipeline:
    """Create a simple pipeline without intent parsing.

    This is the minimal setup - just retrieval and synthesis,
    no multi-hop reasoning or automatic planning.

    Args:
        store: Graph store instance
        openai_api_key: OpenAI API key
        index_name: Vector index name
        node_label: Node label to retrieve
        id_field: ID field name
        embedding_model: OpenAI embedding model
        chat_model: OpenAI chat model
        **kwargs: Additional configuration

    Returns:
        GraphRAGPipeline instance

    Example:
        from deepgraph.store import create_graph_store
        from deepgraph.pipeline import create_simple_pipeline

        store = create_graph_store("neo4j", uri="bolt://localhost:7687", ...)
        pipeline = create_simple_pipeline(store, "sk-...")

        result = pipeline.run("What papers discuss transformers?")
    """
    # Create OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)

    # Create components
    embedder = OpenAIEmbedder(client=openai_client, model=embedding_model)
    retriever = VectorRetriever(store, embedder, index_name, node_label, id_field)
    reasoner = GraphReasoner(store)
    synthesizer = OpenAISynthesizer(client=openai_client, model=chat_model)

    # Create config
    config = PipelineConfig(
        use_planning=False,  # No planning
        **kwargs
    )

    # Create pipeline
    pipeline = GraphRAGPipeline(
        retriever=retriever,
        reasoner=reasoner,
        synthesizer=synthesizer,
        intent_parser=None,
        config=config
    )

    # Disable reasoning by default for simple pipeline
    pipeline.disable_stage("reasoning")

    return pipeline


def create_multi_hop_pipeline(
    store: GraphStore,
    openai_api_key: str,
    index_name: str = "paper_embedding",
    node_label: str = "Paper",
    id_field: str = "id",
    embedding_model: str = "text-embedding-ada-002",
    chat_model: str = "gpt-4-turbo-preview",
    use_planning: bool = True,
    **kwargs
) -> GraphRAGPipeline:
    """Create a full multi-hop pipeline with intent parsing.

    This is the complete setup with all features:
    - Vector retrieval
    - Multi-hop reasoning
    - LLM-based intent parsing
    - Automatic strategy selection

    Args:
        store: Graph store instance
        openai_api_key: OpenAI API key
        index_name: Vector index name
        node_label: Node label to retrieve
        id_field: ID field name
        embedding_model: OpenAI embedding model
        chat_model: OpenAI chat model
        use_planning: Whether to use intent parsing
        **kwargs: Additional configuration

    Returns:
        GraphRAGPipeline instance

    Example:
        pipeline = create_multi_hop_pipeline(store, "sk-...")

        # Automatic planning and execution
        result = pipeline.run("What papers has Geoffrey Hinton written?")
    """
    # Create OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)

    # Create components
    embedder = OpenAIEmbedder(client=openai_client, model=embedding_model)
    retriever = VectorRetriever(store, embedder, index_name, node_label, id_field)
    reasoner = GraphReasoner(store)
    synthesizer = OpenAISynthesizer(client=openai_client, model=chat_model)

    # Create intent parser if planning enabled
    intent_parser = None
    if use_planning:
        intent_parser = IntentParser(openai_client=openai_client, model=chat_model)

    # Create config
    config = PipelineConfig(
        use_planning=use_planning,
        fallback_to_simple=True,
        **kwargs
    )

    # Create pipeline
    pipeline = GraphRAGPipeline(
        retriever=retriever,
        reasoner=reasoner,
        synthesizer=synthesizer,
        intent_parser=intent_parser,
        config=config
    )

    return pipeline


def create_pipeline_from_config(
    config_dict: Dict[str, Any],
    schema: Optional[GraphSchema] = None
) -> GraphRAGPipeline:
    """Create pipeline from configuration dictionary.

    This allows loading pipeline configuration from YAML/JSON files.

    Args:
        config_dict: Configuration dictionary
        schema: Optional graph schema

    Returns:
        GraphRAGPipeline instance

    Example:
        import yaml

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        pipeline = create_pipeline_from_config(config)
    """
    # Extract database config
    db_config = config_dict.get("database", {})
    backend = db_config.get("backend", "neo4j")

    # Create store
    store = create_graph_store(
        backend=backend,
        schema=schema,
        uri=db_config.get("uri"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )

    # Extract LLM config
    llm_config = config_dict.get("llm", {})
    openai_api_key = llm_config.get("api_key")
    if not openai_api_key:
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")

    # Extract retrieval config
    retrieval_config = config_dict.get("retrieval", {})
    index_name = retrieval_config.get("vector_index_name", "paper_embedding")

    # Extract pipeline config
    pipeline_config = config_dict.get("pipeline", {})
    use_planning = pipeline_config.get("use_planning", True)
    pipeline_type = pipeline_config.get("type", "multi_hop")

    # Create appropriate pipeline
    if pipeline_type == "simple":
        return create_simple_pipeline(
            store=store,
            openai_api_key=openai_api_key,
            index_name=index_name,
            embedding_model=llm_config.get("embedding_model", "text-embedding-ada-002"),
            chat_model=llm_config.get("chat_model", "gpt-4-turbo-preview")
        )
    else:
        return create_multi_hop_pipeline(
            store=store,
            openai_api_key=openai_api_key,
            index_name=index_name,
            embedding_model=llm_config.get("embedding_model", "text-embedding-ada-002"),
            chat_model=llm_config.get("chat_model", "gpt-4-turbo-preview"),
            use_planning=use_planning
        )


def create_pipeline_from_yaml(
    yaml_path: str,
    schema: Optional[GraphSchema] = None
) -> GraphRAGPipeline:
    """Create pipeline from YAML configuration file.

    Args:
        yaml_path: Path to YAML config file
        schema: Optional graph schema

    Returns:
        GraphRAGPipeline instance

    Example:
        from schemas.arxiv import ARXIV_SCHEMA

        pipeline = create_pipeline_from_yaml(
            "examples/arxiv/config.yaml",
            schema=ARXIV_SCHEMA
        )

        result = pipeline.run("What papers discuss transformers?")
    """
    import yaml

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    return create_pipeline_from_config(config, schema)
