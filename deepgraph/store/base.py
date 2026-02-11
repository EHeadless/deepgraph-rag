"""
Graph store protocol and base classes.

This module defines the interface that all graph database backends must implement,
enabling backend swapping (Neo4j, NetworkX, MemGraph, etc.).
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from deepgraph.core.schema import GraphSchema


@runtime_checkable
class GraphStore(Protocol):
    """Protocol defining the interface for graph database backends.

    All graph store implementations (Neo4j, NetworkX, MemGraph) must implement
    these methods to be compatible with the framework.
    """

    def connect(self, uri: str, **kwargs) -> None:
        """Establish connection to graph database.

        Args:
            uri: Connection URI (e.g., "bolt://localhost:7687")
            **kwargs: Backend-specific connection parameters
                     (user, password, database, etc.)

        Raises:
            ConnectionError: If connection fails
        """
        ...

    def disconnect(self) -> None:
        """Close connection to graph database."""
        ...

    def create_schema(self, schema: GraphSchema) -> None:
        """Create indexes and constraints from schema definition.

        Args:
            schema: GraphSchema instance with node/edge definitions

        Note:
            This creates constraints, indexes, and vector indexes.
            Implementation should be idempotent (safe to call multiple times).
        """
        ...

    def add_nodes(self, label: str, nodes: List[Dict[str, Any]]) -> None:
        """Batch insert or update nodes.

        Args:
            label: Node label (e.g., "Paper", "Author")
            nodes: List of node property dictionaries

        Note:
            Implementation should use MERGE semantics (create if not exists,
            update if exists) based on the schema's id_field.
        """
        ...

    def add_edges(
        self,
        edge_type: str,
        edges: List[Dict[str, Any]],
        from_field: str = "from",
        to_field: str = "to"
    ) -> None:
        """Batch create or update relationships.

        Args:
            edge_type: Relationship type (e.g., "AUTHORED_BY")
            edges: List of edge dictionaries with source/target IDs
            from_field: Key in edge dict for source node ID
            to_field: Key in edge dict for target node ID

        Example:
            edges = [
                {"from": "paper1", "to": "author1", "position": 0},
                {"from": "paper1", "to": "author2", "position": 1}
            ]
        """
        ...

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw query in backend's native language.

        Args:
            query: Query string (Cypher for Neo4j, etc.)
            params: Query parameters

        Returns:
            List of result dictionaries

        Note:
            For portability, prefer using specific methods (vector_search,
            find_path, etc.) over raw queries when possible.
        """
        ...

    def vector_search(
        self,
        index_name: str,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            index_name: Name of vector index to search
            embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional property filters

        Returns:
            List of dicts with 'node' and 'score' keys

        Example:
            results = store.vector_search(
                index_name="paper_embedding",
                embedding=[0.1, 0.2, ...],  # 1536-dim vector
                top_k=10
            )
            # Returns: [{"node": {...}, "score": 0.95}, ...]
        """
        ...

    def find_path(
        self,
        from_id: str,
        to_id: str,
        from_label: str,
        to_label: str,
        max_depth: int = 6,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            from_label: Source node label
            to_label: Target node label
            max_depth: Maximum path length
            relationship_types: Relationship types to traverse (None = all)

        Returns:
            Dict with 'nodes' and 'relationships' keys, or None if no path

        Example:
            path = store.find_path(
                from_id="author1",
                to_id="author2",
                from_label="Author",
                to_label="Author",
                max_depth=4
            )
            # Returns: {"nodes": [...], "relationships": [...], "length": 3}
        """
        ...

    def get_node(
        self,
        label: str,
        node_id: str,
        id_field: str = "id"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a single node by ID.

        Args:
            label: Node label
            node_id: Node identifier value
            id_field: Property name for ID field

        Returns:
            Node properties dict, or None if not found
        """
        ...

    def get_neighbors(
        self,
        node_id: str,
        label: str,
        id_field: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes.

        Args:
            node_id: Source node ID
            label: Source node label
            id_field: ID field name
            relationship_type: Relationship type to traverse (None = all)
            direction: "outgoing", "incoming", or "both"
            limit: Max number of neighbors to return

        Returns:
            List of neighbor node dictionaries
        """
        ...

    def delete_all(self) -> None:
        """Delete all nodes and relationships.

        Warning:
            This is destructive! Use with caution.
            Primarily for testing/development.
        """
        ...


class BaseGraphStore:
    """Base class with common functionality for graph stores.

    Concrete implementations can inherit from this to get utility methods.
    """

    def __init__(self):
        self._schema: Optional[GraphSchema] = None
        self._connected = False

    def set_schema(self, schema: GraphSchema) -> None:
        """Store reference to schema for validation and queries.

        Args:
            schema: GraphSchema instance
        """
        self._schema = schema

    def validate_connection(self) -> None:
        """Check if store is connected.

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected:
            raise RuntimeError("Graph store not connected. Call connect() first.")

    def validate_node_data(self, label: str, data: Dict[str, Any]) -> None:
        """Validate node data against schema.

        Args:
            label: Node label
            data: Node properties

        Raises:
            ValueError: If validation fails
        """
        if self._schema and not self._schema.validate_node(label, data):
            raise ValueError(f"Invalid node data for label '{label}': {data}")

    def validate_edge_data(self, edge_type: str, data: Dict[str, Any]) -> None:
        """Validate edge data against schema.

        Args:
            edge_type: Relationship type
            data: Edge properties

        Raises:
            ValueError: If validation fails
        """
        if self._schema and not self._schema.validate_edge(edge_type, data):
            raise ValueError(f"Invalid edge data for type '{edge_type}': {data}")

    def get_id_field(self, label: str) -> str:
        """Get ID field name for a node label.

        Args:
            label: Node label

        Returns:
            ID field name from schema

        Raises:
            ValueError: If label not in schema
        """
        if not self._schema:
            return "id"  # Default fallback

        if label not in self._schema.nodes:
            raise ValueError(f"Unknown node label: {label}")

        return self._schema.nodes[label].id_field

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect on exit."""
        self.disconnect()
        return False


def create_graph_store(
    backend: str,
    schema: Optional[GraphSchema] = None,
    **kwargs
) -> GraphStore:
    """Factory function for creating graph store instances.

    Args:
        backend: Backend type ("neo4j", "memgraph", "networkx")
        schema: Optional GraphSchema to associate with store
        **kwargs: Backend-specific connection parameters

    Returns:
        GraphStore implementation instance

    Example:
        store = create_graph_store(
            backend="neo4j",
            schema=ARXIV_SCHEMA,
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
    """
    from deepgraph.store.neo4j import Neo4jGraphStore
    from deepgraph.store.memgraph import MemGraphStore

    backends = {
        "neo4j": Neo4jGraphStore,
        "memgraph": MemGraphStore,
        "memory": MemGraphStore,  # Alias
    }

    if backend not in backends:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Choose from {list(backends.keys())}"
        )

    store_class = backends[backend]
    store = store_class()

    if schema:
        store.set_schema(schema)

    # Auto-connect if URI provided
    if "uri" in kwargs:
        store.connect(**kwargs)

    return store
