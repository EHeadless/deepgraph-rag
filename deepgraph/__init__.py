"""
DeepGraph RAG - A Domain-Agnostic Graph RAG Framework

Find connections that text search can't see. Vector search finds papers with
similar words. Graph RAG finds papers with similar ideas.

Quick Start:
    from deepgraph import GraphRAG

    # Simple usage
    rag = GraphRAG(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="password",
        openai_api_key="sk-..."
    )

    # Vector + Graph search
    results = rag.search("transformers in NLP")

    # Graph-enhanced results (finds connected papers)
    results = rag.graph_search("transformers in NLP")

    # Generate answer with RAG
    answer = rag.answer("What are the key innovations in transformer architectures?")
"""

__version__ = "1.0.0"

# Core imports
from deepgraph.core.schema import GraphSchema, NodeSchema, EdgeSchema

# Store imports
from deepgraph.store.base import GraphStore, BaseGraphStore, create_graph_store
from deepgraph.store.neo4j import Neo4jGraphStore

# Pipeline imports
from deepgraph.pipeline.base import GraphRAGPipeline
from deepgraph.pipeline.prebuilt import (
    create_simple_pipeline,
    create_multi_hop_pipeline,
    create_pipeline_from_config,
    create_pipeline_from_yaml
)

# Component imports
from deepgraph.retrieval.vector import VectorRetriever
from deepgraph.reasoning.traversal import GraphReasoner
from deepgraph.synthesis.openai import OpenAISynthesizer
from deepgraph.adapters.embedders.openai import OpenAIEmbedder


class GraphRAG:
    """High-level Graph RAG interface.

    This is the main entry point for using DeepGraph RAG. It provides a simple
    interface for graph-enhanced retrieval and answer generation.

    Example:
        from deepgraph import GraphRAG

        rag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="password",
            openai_api_key="sk-..."
        )

        # Vector search only
        results = rag.search("machine learning")

        # Vector + graph-enhanced search
        results = rag.graph_search("machine learning")

        # Generate answer with context
        answer = rag.answer("What are the key papers on transformers?")
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = None,
        openai_api_key: str = None,
        vector_index: str = "paper_embedding",
        node_label: str = "Paper",
        id_field: str = "arxiv_id",
        embedding_model: str = "text-embedding-ada-002",
        chat_model: str = "gpt-4-turbo-preview"
    ):
        """Initialize GraphRAG.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            vector_index: Name of vector index in Neo4j
            node_label: Label for main nodes (e.g., "Paper")
            id_field: Field name for node IDs
            embedding_model: OpenAI embedding model
            chat_model: OpenAI chat model
        """
        import os
        from openai import OpenAI

        # Get API key
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._openai_api_key:
            raise ValueError(
                "OpenAI API key required. Pass openai_api_key or set OPENAI_API_KEY env var."
            )

        # Get Neo4j password
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            raise ValueError(
                "Neo4j password required. Pass neo4j_password or set NEO4J_PASSWORD env var."
            )

        # Create store
        self._store = create_graph_store(
            backend="neo4j",
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )

        # Create OpenAI client
        self._openai_client = OpenAI(api_key=self._openai_api_key)

        # Create components
        self._embedder = OpenAIEmbedder(
            client=self._openai_client,
            model=embedding_model
        )
        self._retriever = VectorRetriever(
            store=self._store,
            embedder=self._embedder,
            index_name=vector_index,
            node_label=node_label,
            id_field=id_field
        )
        self._reasoner = GraphReasoner(store=self._store)
        self._synthesizer = OpenAISynthesizer(
            client=self._openai_client,
            model=chat_model
        )

        # Store config
        self._node_label = node_label
        self._id_field = id_field

    @property
    def store(self) -> Neo4jGraphStore:
        """Access the underlying graph store."""
        return self._store

    def search(self, query: str, top_k: int = 5) -> list:
        """Vector search only.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching nodes
        """
        candidate_ids = self._retriever.retrieve(query, top_k=top_k)
        return candidate_ids

    def graph_search(self, query: str, top_k: int = 5) -> dict:
        """Vector search + graph-enhanced discovery.

        Returns both vector results and papers found through graph
        relationships (same authors, collaborators, cross-field).

        Args:
            query: Search query
            top_k: Number of results per category

        Returns:
            Dict with 'vector_results' and 'graph_discoveries' keys
        """
        # Get vector results
        vector_results = self._retriever.retrieve(query, top_k=top_k)

        # Get graph-enhanced results
        if vector_results:
            context = self._reasoner.expand_context(
                candidate_ids=vector_results,
                strategy_name="author_papers",
                max_depth=2
            )
            return {
                "vector_results": vector_results,
                "graph_discoveries": context.get("expanded_nodes", [])
            }
        return {
            "vector_results": vector_results,
            "graph_discoveries": []
        }

    def answer(self, question: str, top_k: int = 5, use_graph: bool = True) -> str:
        """Generate answer using RAG.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            use_graph: Whether to use graph-enhanced retrieval

        Returns:
            Generated answer string
        """
        # Retrieve context
        if use_graph:
            results = self.graph_search(question, top_k=top_k)
            all_nodes = results["vector_results"] + results.get("graph_discoveries", [])
        else:
            all_nodes = self.search(question, top_k=top_k)

        # Build context string
        context_parts = []
        for node_id in all_nodes[:top_k * 2]:  # Take more for diversity
            node = self._store.get_node(
                label=self._node_label,
                node_id=node_id,
                id_field=self._id_field
            )
            if node:
                title = node.get("title", "Unknown")
                authors = node.get("authors", [])
                if isinstance(authors, list):
                    authors = ", ".join(authors[:3])
                context_parts.append(f"- {title} ({authors})")

        context = "\n".join(context_parts)

        # Generate answer
        answer = self._synthesizer.synthesize(
            query=question,
            context=context
        )

        return answer

    def find_path(
        self,
        from_id: str,
        to_id: str,
        from_label: str = "Author",
        to_label: str = "Author",
        max_depth: int = 6
    ) -> dict:
        """Find shortest path between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            from_label: Source node label
            to_label: Target node label
            max_depth: Maximum path length

        Returns:
            Path info dict or None
        """
        return self._store.find_path(
            from_id=from_id,
            to_id=to_id,
            from_label=from_label,
            to_label=to_label,
            max_depth=max_depth
        )

    def get_stats(self) -> dict:
        """Get graph statistics.

        Returns:
            Dict with node/edge counts
        """
        results = self._store.query("""
            MATCH (n)
            WITH labels(n) as labels
            UNWIND labels as label
            RETURN label, count(*) as count
        """)

        stats = {r["label"]: r["count"] for r in results}
        return stats

    def close(self):
        """Close connections."""
        self._store.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        return f"GraphRAG(connected={self._store._connected})"


# Convenience exports
__all__ = [
    # Main class
    "GraphRAG",
    # Version
    "__version__",
    # Schema
    "GraphSchema",
    "NodeSchema",
    "EdgeSchema",
    # Store
    "GraphStore",
    "BaseGraphStore",
    "Neo4jGraphStore",
    "create_graph_store",
    # Pipeline
    "GraphRAGPipeline",
    "create_simple_pipeline",
    "create_multi_hop_pipeline",
    "create_pipeline_from_config",
    "create_pipeline_from_yaml",
    # Components
    "VectorRetriever",
    "GraphReasoner",
    "OpenAISynthesizer",
    "OpenAIEmbedder",
]
