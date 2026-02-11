"""
Vector-based retrieval implementation.

This module provides a retriever that uses vector similarity search
to find relevant nodes in the graph.
"""

from typing import List, Dict, Any, Optional
from deepgraph.store.base import GraphStore
from deepgraph.adapters.embedders.base import Embedder


class VectorRetriever:
    """Vector similarity-based candidate retriever.

    This retriever uses embeddings and vector search to find nodes
    that are semantically similar to the query.

    Example:
        from deepgraph.store import Neo4jGraphStore
        from deepgraph.adapters.embedders import OpenAIEmbedder

        store = Neo4jGraphStore()
        store.connect("bolt://localhost:7687", user="neo4j", password="...")

        embedder = OpenAIEmbedder(api_key="sk-...")
        retriever = VectorRetriever(
            store=store,
            embedder=embedder,
            index_name="paper_embedding",
            node_label="Paper"
        )

        # Retrieve candidates
        candidate_ids = retriever.retrieve("What papers discuss transformers?", top_k=10)
    """

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        index_name: str,
        node_label: str,
        id_field: str = "id"
    ):
        """Initialize vector retriever.

        Args:
            store: Graph store instance
            embedder: Embedder for generating query embeddings
            index_name: Name of vector index in the store
            node_label: Label of nodes to retrieve (e.g., "Paper")
            id_field: Property name for node ID field
        """
        self._store = store
        self._embedder = embedder
        self._index_name = index_name
        self._node_label = node_label
        self._id_field = id_field

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Retrieve candidate node IDs relevant to the query.

        Args:
            query: User query text
            top_k: Number of candidates to retrieve
            filters: Optional filters on node properties (not yet implemented)

        Returns:
            List of node IDs (in ranked order by similarity)

        Example:
            candidate_ids = retriever.retrieve(
                "What papers discuss transformers?",
                top_k=10
            )
            # Returns: ["2301.12345", "2301.12346", ...]
        """
        results = self.retrieve_with_scores(query, top_k, filters)
        return [r["id"] for r in results]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates with relevance scores.

        Args:
            query: User query text
            top_k: Number of candidates to retrieve
            filters: Optional filters on node properties (not yet implemented)

        Returns:
            List of dicts with 'id', 'score', and 'node' keys

        Example:
            results = retriever.retrieve_with_scores(
                "What papers discuss transformers?",
                top_k=10
            )
            # Returns: [
            #     {"id": "2301.12345", "score": 0.95, "node": {...}},
            #     ...
            # ]
        """
        # Generate query embedding
        query_embedding = self._embedder.embed(query)

        # Perform vector search
        search_results = self._store.vector_search(
            index_name=self._index_name,
            embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        # Extract node IDs and format results
        formatted_results = []
        for result in search_results:
            node = result["node"]
            score = result["score"]

            # Get node ID
            node_id = node.get(self._id_field)
            if node_id:
                formatted_results.append({
                    "id": node_id,
                    "score": score,
                    "node": node
                })

        return formatted_results

    def retrieve_nodes(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve full node dictionaries (convenience method).

        Args:
            query: User query text
            top_k: Number of candidates to retrieve
            filters: Optional filters on node properties

        Returns:
            List of node property dictionaries

        Example:
            nodes = retriever.retrieve_nodes("transformers", top_k=5)
            for node in nodes:
                print(node["title"])
        """
        results = self.retrieve_with_scores(query, top_k, filters)
        return [r["node"] for r in results]

    @property
    def index_name(self) -> str:
        """Get the vector index name."""
        return self._index_name

    @property
    def node_label(self) -> str:
        """Get the node label being retrieved."""
        return self._node_label

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VectorRetriever("
            f"index={self._index_name}, "
            f"label={self._node_label}, "
            f"embedder={self._embedder.model_name})"
        )
