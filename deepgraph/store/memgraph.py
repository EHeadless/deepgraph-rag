"""
In-memory graph store implementation.

This module provides a lightweight in-memory graph store for testing and development,
eliminating the need for a real database connection during unit tests.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import math
from deepgraph.core.schema import GraphSchema
from deepgraph.store.base import BaseGraphStore


class MemGraphStore(BaseGraphStore):
    """In-memory implementation of GraphStore protocol.

    This class provides a simple in-memory graph database for testing without
    requiring Neo4j. It supports basic graph operations and vector search via
    cosine similarity.

    Note:
        This is NOT production-ready. Use Neo4j for real applications.
        This is purely for unit testing and development.

    Example:
        store = MemGraphStore()
        store.connect("memory://")
        store.add_nodes("Paper", [{"arxiv_id": "123", "title": "Test"}])
    """

    def __init__(self):
        """Initialize in-memory store."""
        super().__init__()
        # Storage: label -> id_value -> properties
        self._nodes: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        # Storage: edge_type -> list of (from_id, to_id, properties)
        self._edges: Dict[str, List[tuple]] = defaultdict(list)
        # Vector indexes: index_name -> [(node_id, embedding)]
        self._vector_indexes: Dict[str, List[tuple]] = defaultdict(list)

    def connect(self, uri: str = "memory://", **kwargs) -> None:
        """Establish connection (no-op for in-memory store).

        Args:
            uri: Connection URI (ignored, always "memory://")
            **kwargs: Ignored for in-memory store
        """
        self._connected = True

    def disconnect(self) -> None:
        """Close connection (no-op for in-memory store)."""
        self._connected = False

    def create_schema(self, schema: GraphSchema) -> None:
        """Store schema reference (no actual DDL needed for in-memory).

        Args:
            schema: GraphSchema instance
        """
        self.set_schema(schema)

    def add_nodes(self, label: str, nodes: List[Dict[str, Any]]) -> None:
        """Batch insert or update nodes.

        Args:
            label: Node label
            nodes: List of node property dictionaries
        """
        self.validate_connection()

        if not nodes:
            return

        # Get ID field
        id_field = self.get_id_field(label)

        for node in nodes:
            if id_field not in node:
                raise ValueError(f"Node missing required ID field '{id_field}': {node}")

            node_id = node[id_field]
            # Store node (MERGE semantics - update if exists)
            self._nodes[label][node_id] = node.copy()

            # Index vector embeddings if present
            if self._schema and label in self._schema.nodes:
                node_schema = self._schema.nodes[label]
                if node_schema.vector_config:
                    vector_field = node_schema.vector_config["field"]
                    index_name = node_schema.vector_config["index_name"]
                    if vector_field in node:
                        embedding = node[vector_field]
                        # Remove old entry if exists
                        self._vector_indexes[index_name] = [
                            (nid, emb) for nid, emb in self._vector_indexes[index_name]
                            if nid != node_id
                        ]
                        # Add new entry
                        self._vector_indexes[index_name].append((node_id, embedding))

    def add_edges(
        self,
        edge_type: str,
        edges: List[Dict[str, Any]],
        from_field: str = "from",
        to_field: str = "to"
    ) -> None:
        """Batch create or update relationships.

        Args:
            edge_type: Relationship type
            edges: List of edge dictionaries
            from_field: Key in edge dict for source node ID
            to_field: Key in edge dict for target node ID
        """
        self.validate_connection()

        if not edges:
            return

        for edge in edges:
            from_id = edge[from_field]
            to_id = edge[to_field]

            # Extract properties (exclude from/to)
            props = {k: v for k, v in edge.items() if k not in (from_field, to_field)}

            # Simple MERGE: remove existing edge, add new one
            self._edges[edge_type] = [
                (f, t, p) for f, t, p in self._edges[edge_type]
                if not (f == from_id and t == to_id)
            ]
            self._edges[edge_type].append((from_id, to_id, props))

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute query (limited support for in-memory store).

        Args:
            query: Query string (only basic patterns supported)
            params: Query parameters

        Returns:
            List of result dictionaries

        Note:
            This is a stub implementation. For real queries, use Neo4j.
            Only supports very basic MATCH patterns for testing.
        """
        self.validate_connection()
        # Minimal implementation for testing
        # Real implementation would parse Cypher and execute
        raise NotImplementedError(
            "MemGraphStore does not support arbitrary queries. "
            "Use specific methods (vector_search, get_node, etc.)"
        )

    def vector_search(
        self,
        index_name: str,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using cosine similarity.

        Args:
            index_name: Name of vector index to search
            embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional property filters (not implemented)

        Returns:
            List of dicts with 'node' and 'score' keys
        """
        self.validate_connection()

        if index_name not in self._vector_indexes:
            return []

        # Compute cosine similarity for all indexed vectors
        scores = []
        for node_id, stored_embedding in self._vector_indexes[index_name]:
            score = self._cosine_similarity(embedding, stored_embedding)
            scores.append((node_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Get top_k results
        results = []
        for node_id, score in scores[:top_k]:
            # Find the node
            node = self._find_node_by_id(node_id)
            if node:
                results.append({
                    "node": node,
                    "score": score
                })

        return results

    def find_path(
        self,
        from_id: str,
        to_id: str,
        from_label: str,
        to_label: str,
        max_depth: int = 6,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path between two nodes using BFS.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            from_label: Source node label
            to_label: Target node label
            max_depth: Maximum path length
            relationship_types: Relationship types to traverse

        Returns:
            Dict with 'nodes' and 'relationships' keys, or None if no path
        """
        self.validate_connection()

        # Check nodes exist
        if from_id not in self._nodes.get(from_label, {}):
            return None
        if to_id not in self._nodes.get(to_label, {}):
            return None

        # BFS to find shortest path
        queue = [(from_id, [from_id], [])]  # (current_id, path_nodes, path_rels)
        visited = {from_id}

        while queue:
            current_id, path_nodes, path_rels = queue.pop(0)

            if len(path_nodes) > max_depth:
                continue

            # Check if reached target
            if current_id == to_id:
                # Build result
                nodes = [self._find_node_by_id(nid) for nid in path_nodes]
                return {
                    "nodes": nodes,
                    "relationships": path_rels,
                    "length": len(path_rels)
                }

            # Explore neighbors
            for edge_type, edges in self._edges.items():
                # Filter by relationship type if specified
                if relationship_types and edge_type not in relationship_types:
                    continue

                for from_node, to_node, props in edges:
                    # Check outgoing edges
                    if from_node == current_id and to_node not in visited:
                        visited.add(to_node)
                        new_path_nodes = path_nodes + [to_node]
                        new_path_rels = path_rels + [{"type": edge_type, "properties": props}]
                        queue.append((to_node, new_path_nodes, new_path_rels))

                    # Check incoming edges (undirected search)
                    if to_node == current_id and from_node not in visited:
                        visited.add(from_node)
                        new_path_nodes = path_nodes + [from_node]
                        new_path_rels = path_rels + [{"type": edge_type, "properties": props}]
                        queue.append((from_node, new_path_nodes, new_path_rels))

        return None  # No path found

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
            id_field: Property name for ID field (not used, kept for API compatibility)

        Returns:
            Node properties dict, or None if not found
        """
        self.validate_connection()

        if label in self._nodes and node_id in self._nodes[label]:
            return self._nodes[label][node_id].copy()
        return None

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
            relationship_type: Relationship type to traverse
            direction: "outgoing", "incoming", or "both"
            limit: Max number of neighbors to return

        Returns:
            List of neighbor node dictionaries
        """
        self.validate_connection()

        neighbors = []

        for edge_type, edges in self._edges.items():
            # Filter by relationship type
            if relationship_type and edge_type != relationship_type:
                continue

            for from_node, to_node, props in edges:
                # Outgoing edges
                if direction in ("outgoing", "both") and from_node == node_id:
                    neighbor = self._find_node_by_id(to_node)
                    if neighbor:
                        neighbors.append(neighbor)

                # Incoming edges
                if direction in ("incoming", "both") and to_node == node_id:
                    neighbor = self._find_node_by_id(from_node)
                    if neighbor:
                        neighbors.append(neighbor)

                if len(neighbors) >= limit:
                    break

            if len(neighbors) >= limit:
                break

        return neighbors[:limit]

    def delete_all(self) -> None:
        """Delete all nodes and relationships."""
        self.validate_connection()
        self._nodes.clear()
        self._edges.clear()
        self._vector_indexes.clear()

    # ============== HELPER METHODS ==============

    def _find_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Find a node by ID across all labels."""
        for label_nodes in self._nodes.values():
            if node_id in label_nodes:
                return label_nodes[node_id].copy()
        return None

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        num_nodes = sum(len(nodes) for nodes in self._nodes.values())
        num_edges = sum(len(edges) for edges in self._edges.values())
        return f"MemGraphStore(status={status}, nodes={num_nodes}, edges={num_edges})"
