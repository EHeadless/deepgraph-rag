"""
Neo4j graph store implementation.

This module provides a Neo4j-backed implementation of the GraphStore protocol,
enabling connection to Neo4j databases and execution of graph operations.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Session, Driver
from deepgraph.core.schema import GraphSchema
from deepgraph.store.base import BaseGraphStore


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j implementation of GraphStore protocol.

    This class wraps the Neo4j Python driver and provides a clean interface
    for graph operations including vector search, path finding, and CRUD.

    Example:
        store = Neo4jGraphStore()
        store.connect("bolt://localhost:7687", user="neo4j", password="password")
        store.create_schema(ARXIV_SCHEMA)
        store.add_nodes("Paper", [{"arxiv_id": "2301.12345", "title": "..."}])
    """

    def __init__(self):
        """Initialize Neo4j store."""
        super().__init__()
        self._driver: Optional[Driver] = None

    def connect(self, uri: str, **kwargs) -> None:
        """Establish connection to Neo4j database.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            **kwargs: Connection parameters (user, password, database, etc.)

        Raises:
            ConnectionError: If connection fails

        Example:
            store.connect(
                "bolt://localhost:7687",
                user="neo4j",
                password="password",
                database="neo4j"  # Optional
            )
        """
        try:
            # Extract auth credentials
            user = kwargs.pop("user", "neo4j")
            password = kwargs.pop("password", None)

            if not password:
                raise ValueError("Password is required for Neo4j connection")

            # Create driver
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                **kwargs
            )

            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")

            self._connected = True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def disconnect(self) -> None:
        """Close connection to Neo4j database."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False

    def create_schema(self, schema: GraphSchema) -> None:
        """Create indexes and constraints from schema definition.

        Args:
            schema: GraphSchema instance with node/edge definitions

        Note:
            This is idempotent - safe to call multiple times.
            Existing indexes/constraints won't be recreated.
        """
        self.validate_connection()
        self.set_schema(schema)

        # Generate DDL statements
        ddl_statements = schema.to_cypher_ddl()

        with self._driver.session() as session:
            for statement in ddl_statements:
                try:
                    session.run(statement)
                except Exception as e:
                    # Ignore "already exists" errors
                    if "already exists" not in str(e).lower():
                        raise

    def add_nodes(self, label: str, nodes: List[Dict[str, Any]]) -> None:
        """Batch insert or update nodes.

        Args:
            label: Node label (e.g., "Paper", "Author")
            nodes: List of node property dictionaries

        Note:
            Uses MERGE semantics - creates if not exists, updates if exists
            based on the schema's id_field.

        Example:
            store.add_nodes("Paper", [
                {"arxiv_id": "2301.12345", "title": "Attention Is All You Need"},
                {"arxiv_id": "2301.12346", "title": "BERT"}
            ])
        """
        self.validate_connection()

        if not nodes:
            return

        # Get ID field from schema
        id_field = self.get_id_field(label)

        # Build MERGE query with all properties
        # We use MERGE on ID field, then SET all other properties
        query = f"""
            UNWIND $nodes AS node
            MERGE (n:{label} {{{id_field}: node.{id_field}}})
            SET n = node
        """

        with self._driver.session() as session:
            session.run(query, nodes=nodes)

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
            store.add_edges("AUTHORED_BY", [
                {"from": "paper1", "to": "author1", "position": 0},
                {"from": "paper1", "to": "author2", "position": 1}
            ])
        """
        self.validate_connection()

        if not edges:
            return

        # Get edge schema to determine node labels
        if self._schema and edge_type in self._schema.edges:
            edge_schema = self._schema.edges[edge_type]
            from_label = edge_schema.from_label
            to_label = edge_schema.to_label
            from_id_field = self.get_id_field(from_label)
            to_id_field = self.get_id_field(to_label)
        else:
            # Fallback - assume generic labels and id field
            from_label = "Node"
            to_label = "Node"
            from_id_field = "id"
            to_id_field = "id"

        # Extract relationship properties (exclude from/to fields)
        sample_edge = edges[0] if edges else {}
        prop_keys = [k for k in sample_edge.keys() if k not in (from_field, to_field)]

        # Build MERGE query
        if prop_keys:
            # Set properties on relationship
            set_clause = "SET r = properties({" + ", ".join(f"{k}: edge.{k}" for k in prop_keys) + "})"
        else:
            set_clause = ""

        query = f"""
            UNWIND $edges AS edge
            MATCH (from:{from_label} {{{from_id_field}: edge.{from_field}}})
            MATCH (to:{to_label} {{{to_id_field}: edge.{to_field}}})
            MERGE (from)-[r:{edge_type}]->(to)
            {set_clause}
        """

        with self._driver.session() as session:
            session.run(query, edges=edges)

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw Cypher query.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            List of result dictionaries

        Example:
            results = store.query(
                "MATCH (p:Paper) WHERE p.title CONTAINS $keyword RETURN p",
                {"keyword": "transformer"}
            )
        """
        self.validate_connection()

        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

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
            filters: Optional property filters (not yet implemented)

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
        self.validate_connection()

        query = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            RETURN node, score
        """

        with self._driver.session() as session:
            result = session.run(
                query,
                index_name=index_name,
                top_k=top_k,
                embedding=embedding
            )
            return [
                {
                    "node": dict(record["node"]),
                    "score": record["score"]
                }
                for record in result
            ]

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
        """
        self.validate_connection()

        # Get ID fields
        from_id_field = self.get_id_field(from_label)
        to_id_field = self.get_id_field(to_label)

        # Build relationship filter
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[r:{rel_filter}*1..{max_depth}]"
        else:
            rel_pattern = f"[r*1..{max_depth}]"

        query = f"""
            MATCH (from:{from_label} {{{from_id_field}: $from_id}})
            MATCH (to:{to_label} {{{to_id_field}: $to_id}})
            MATCH path = shortestPath((from)-{rel_pattern}-(to))
            RETURN
                [node in nodes(path) | properties(node)] as nodes,
                [rel in relationships(path) | {{type: type(rel), properties: properties(rel)}}] as relationships,
                length(path) as length
            LIMIT 1
        """

        with self._driver.session() as session:
            result = session.run(
                query,
                from_id=from_id,
                to_id=to_id
            )
            record = result.single()

            if not record:
                return None

            return {
                "nodes": record["nodes"],
                "relationships": record["relationships"],
                "length": record["length"]
            }

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

        Example:
            paper = store.get_node("Paper", "2301.12345", "arxiv_id")
        """
        self.validate_connection()

        query = f"""
            MATCH (n:{label} {{{id_field}: $node_id}})
            RETURN properties(n) as node
        """

        with self._driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            return record["node"] if record else None

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

        Example:
            neighbors = store.get_neighbors(
                node_id="author1",
                label="Author",
                id_field="name",
                relationship_type="CO_AUTHORED",
                direction="both"
            )
        """
        self.validate_connection()

        # Build relationship pattern
        if relationship_type:
            rel = f"[:{relationship_type}]"
        else:
            rel = ""

        if direction == "outgoing":
            pattern = f"(n)-{rel}->(neighbor)"
        elif direction == "incoming":
            pattern = f"(n)<-{rel}-(neighbor)"
        else:  # both
            pattern = f"(n)-{rel}-(neighbor)"

        query = f"""
            MATCH (n:{label} {{{id_field}: $node_id}})
            MATCH {pattern}
            RETURN DISTINCT properties(neighbor) as neighbor
            LIMIT $limit
        """

        with self._driver.session() as session:
            result = session.run(
                query,
                node_id=node_id,
                limit=limit
            )
            return [record["neighbor"] for record in result]

    def delete_all(self) -> None:
        """Delete all nodes and relationships.

        Warning:
            This is destructive! Use with caution.
            Primarily for testing/development.
        """
        self.validate_connection()

        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        schema_name = self._schema.name if self._schema else "none"
        return f"Neo4jGraphStore(status={status}, schema={schema_name})"
