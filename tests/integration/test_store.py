"""
Integration tests for graph store implementations.

These tests verify that both Neo4j and MemGraph backends produce consistent
results for the same operations, ensuring backend compatibility.
"""

import pytest
from deepgraph.store import Neo4jGraphStore, MemGraphStore, create_graph_store
from schemas.arxiv import ARXIV_SCHEMA, PAPER_SCHEMA, AUTHOR_SCHEMA


class TestMemGraphStore:
    """Tests for MemGraphStore implementation."""

    def setup_method(self):
        """Create fresh store for each test."""
        self.store = MemGraphStore()
        self.store.connect()
        self.store.create_schema(ARXIV_SCHEMA)

    def teardown_method(self):
        """Clean up after test."""
        if self.store._connected:
            self.store.delete_all()
            self.store.disconnect()

    def test_connect_disconnect(self):
        """Test connection lifecycle."""
        store = MemGraphStore()
        assert not store._connected

        store.connect()
        assert store._connected

        store.disconnect()
        assert not store._connected

    def test_add_and_get_nodes(self):
        """Test adding and retrieving nodes."""
        papers = [
            {
                "arxiv_id": "2301.12345",
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models...",
                "published_date": "2017-06-12",
                "primary_category": "cs.CL"
            },
            {
                "arxiv_id": "2301.12346",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": "We introduce a new language representation model...",
                "published_date": "2018-10-11",
                "primary_category": "cs.CL"
            }
        ]

        self.store.add_nodes("Paper", papers)

        # Retrieve first paper
        paper = self.store.get_node("Paper", "2301.12345", "arxiv_id")
        assert paper is not None
        assert paper["title"] == "Attention Is All You Need"

        # Retrieve second paper
        paper = self.store.get_node("Paper", "2301.12346", "arxiv_id")
        assert paper is not None
        assert paper["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"

        # Non-existent paper
        paper = self.store.get_node("Paper", "9999.99999", "arxiv_id")
        assert paper is None

    def test_add_and_get_edges(self):
        """Test adding relationships."""
        # Add nodes first
        papers = [{"arxiv_id": "paper1", "title": "Paper 1", "abstract": "...",
                   "published_date": "2020-01-01", "primary_category": "cs.AI"}]
        authors = [{"name": "Alice"}, {"name": "Bob"}]

        self.store.add_nodes("Paper", papers)
        self.store.add_nodes("Author", authors)

        # Add edges
        edges = [
            {"from": "paper1", "to": "Alice", "position": 0},
            {"from": "paper1", "to": "Bob", "position": 1}
        ]
        self.store.add_edges("AUTHORED_BY", edges)

        # Get neighbors
        neighbors = self.store.get_neighbors(
            node_id="paper1",
            label="Paper",
            id_field="arxiv_id",
            relationship_type="AUTHORED_BY",
            direction="outgoing"
        )

        assert len(neighbors) == 2
        author_names = {n["name"] for n in neighbors}
        assert author_names == {"Alice", "Bob"}

    def test_vector_search(self):
        """Test vector similarity search."""
        # Add papers with embeddings
        papers = [
            {
                "arxiv_id": "paper1",
                "title": "Transformers in NLP",
                "abstract": "...",
                "published_date": "2020-01-01",
                "primary_category": "cs.CL",
                "embedding": [0.1, 0.2, 0.3, 0.4]
            },
            {
                "arxiv_id": "paper2",
                "title": "Vision Transformers",
                "abstract": "...",
                "published_date": "2020-06-01",
                "primary_category": "cs.CV",
                "embedding": [0.1, 0.2, 0.3, 0.5]  # Very similar
            },
            {
                "arxiv_id": "paper3",
                "title": "Graph Neural Networks",
                "abstract": "...",
                "published_date": "2020-12-01",
                "primary_category": "cs.LG",
                "embedding": [0.9, 0.8, 0.1, 0.0]  # Very different
            }
        ]

        self.store.add_nodes("Paper", papers)

        # Search with query similar to paper1/paper2
        query_embedding = [0.1, 0.2, 0.3, 0.45]
        results = self.store.vector_search("paper_embedding", query_embedding, top_k=2)

        assert len(results) == 2
        # First two results should be paper1 and paper2
        top_ids = {r["node"]["arxiv_id"] for r in results}
        assert "paper1" in top_ids
        assert "paper2" in top_ids
        # Scores should be sorted descending
        assert results[0]["score"] >= results[1]["score"]

    def test_find_path(self):
        """Test shortest path finding."""
        # Create a small graph: Alice -> Paper1 <- Bob -> Paper2 <- Carol
        authors = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        papers = [
            {"arxiv_id": "paper1", "title": "P1", "abstract": "...",
             "published_date": "2020-01-01", "primary_category": "cs.AI"},
            {"arxiv_id": "paper2", "title": "P2", "abstract": "...",
             "published_date": "2020-01-01", "primary_category": "cs.AI"}
        ]

        self.store.add_nodes("Author", authors)
        self.store.add_nodes("Paper", papers)

        # Add edges
        edges = [
            {"from": "paper1", "to": "Alice", "position": 0},
            {"from": "paper1", "to": "Bob", "position": 1},
            {"from": "paper2", "to": "Bob", "position": 0},
            {"from": "paper2", "to": "Carol", "position": 1}
        ]
        self.store.add_edges("AUTHORED_BY", edges)

        # Find path Alice -> Carol (should go through Bob and papers)
        # Path: Alice -> paper1 -> Bob -> paper2 -> Carol
        path = self.store.find_path(
            from_id="Alice",
            to_id="Carol",
            from_label="Author",
            to_label="Author",
            max_depth=5
        )

        assert path is not None
        assert len(path["nodes"]) == 5  # Alice, paper1, Bob, paper2, Carol
        assert path["length"] == 4  # 4 relationships

        # Find path with no connection
        self.store.add_nodes("Author", [{"name": "Isolated"}])
        path = self.store.find_path(
            from_id="Alice",
            to_id="Isolated",
            from_label="Author",
            to_label="Author",
            max_depth=5
        )
        assert path is None

    def test_merge_semantics(self):
        """Test that add_nodes uses MERGE semantics (update if exists)."""
        # Add initial node
        self.store.add_nodes("Paper", [
            {
                "arxiv_id": "paper1",
                "title": "Original Title",
                "abstract": "Original abstract",
                "published_date": "2020-01-01",
                "primary_category": "cs.AI"
            }
        ])

        # Update same node
        self.store.add_nodes("Paper", [
            {
                "arxiv_id": "paper1",
                "title": "Updated Title",
                "abstract": "Updated abstract",
                "published_date": "2020-01-01",
                "primary_category": "cs.AI"
            }
        ])

        # Verify update
        paper = self.store.get_node("Paper", "paper1", "arxiv_id")
        assert paper["title"] == "Updated Title"
        assert paper["abstract"] == "Updated abstract"

    def test_delete_all(self):
        """Test delete_all removes everything."""
        # Add some data
        self.store.add_nodes("Paper", [
            {"arxiv_id": "p1", "title": "P1", "abstract": "...",
             "published_date": "2020-01-01", "primary_category": "cs.AI"}
        ])
        self.store.add_nodes("Author", [{"name": "Alice"}])

        # Verify data exists
        assert len(self.store._nodes["Paper"]) == 1
        assert len(self.store._nodes["Author"]) == 1

        # Delete all
        self.store.delete_all()

        # Verify empty
        assert len(self.store._nodes["Paper"]) == 0
        assert len(self.store._nodes["Author"]) == 0


class TestCreateGraphStore:
    """Tests for create_graph_store factory function."""

    def test_create_memgraph_store(self):
        """Test creating MemGraph store via factory."""
        store = create_graph_store(
            backend="memgraph",
            schema=ARXIV_SCHEMA,
            uri="memory://"
        )

        assert isinstance(store, MemGraphStore)
        assert store._connected
        assert store._schema == ARXIV_SCHEMA

        store.disconnect()

    def test_create_memgraph_store_with_alias(self):
        """Test creating MemGraph store using 'memory' alias."""
        store = create_graph_store(
            backend="memory",
            schema=ARXIV_SCHEMA,
            uri="memory://"
        )

        assert isinstance(store, MemGraphStore)
        store.disconnect()

    def test_create_unknown_backend(self):
        """Test error on unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_graph_store(backend="unknown")

    def test_create_neo4j_store(self):
        """Test creating Neo4j store via factory (without actual connection)."""
        # Don't connect - just test instantiation
        store = create_graph_store(backend="neo4j", schema=ARXIV_SCHEMA)

        assert isinstance(store, Neo4jGraphStore)
        assert not store._connected
        assert store._schema == ARXIV_SCHEMA


class TestBackendCompatibility:
    """Tests comparing Neo4j and MemGraph backends for consistency.

    Note: These tests require a running Neo4j instance.
    They will be skipped if connection fails.
    """

    @pytest.fixture
    def memgraph_store(self):
        """Create MemGraph store."""
        store = MemGraphStore()
        store.connect()
        store.create_schema(ARXIV_SCHEMA)
        yield store
        store.delete_all()
        store.disconnect()

    @pytest.fixture
    def neo4j_store(self):
        """Create Neo4j store (skip if unavailable)."""
        store = Neo4jGraphStore()
        try:
            store.connect(
                "bolt://localhost:7687",
                user="neo4j",
                password="deepgraph2025"
            )
            store.create_schema(ARXIV_SCHEMA)
            yield store
            store.delete_all()
            store.disconnect()
        except Exception:
            pytest.skip("Neo4j not available")

    def test_both_stores_add_nodes(self, memgraph_store, neo4j_store):
        """Test that both stores handle node addition identically."""
        papers = [
            {
                "arxiv_id": "2301.12345",
                "title": "Test Paper",
                "abstract": "Abstract text",
                "published_date": "2020-01-01",
                "primary_category": "cs.AI"
            }
        ]

        # Add to both stores
        memgraph_store.add_nodes("Paper", papers)
        neo4j_store.add_nodes("Paper", papers)

        # Retrieve from both
        mem_paper = memgraph_store.get_node("Paper", "2301.12345", "arxiv_id")
        neo_paper = neo4j_store.get_node("Paper", "2301.12345", "arxiv_id")

        # Compare results
        assert mem_paper["arxiv_id"] == neo_paper["arxiv_id"]
        assert mem_paper["title"] == neo_paper["title"]

    def test_both_stores_vector_search(self, memgraph_store, neo4j_store):
        """Test that both stores return similar vector search results."""
        papers = [
            {
                "arxiv_id": "paper1",
                "title": "Paper 1",
                "abstract": "Abstract",
                "published_date": "2020-01-01",
                "primary_category": "cs.AI",
                "embedding": [0.1, 0.2, 0.3] + [0.0] * 1533  # 1536-dim
            },
            {
                "arxiv_id": "paper2",
                "title": "Paper 2",
                "abstract": "Abstract",
                "published_date": "2020-01-01",
                "primary_category": "cs.AI",
                "embedding": [0.9, 0.8, 0.7] + [0.0] * 1533
            }
        ]

        # Add to both stores
        memgraph_store.add_nodes("Paper", papers)
        neo4j_store.add_nodes("Paper", papers)

        # Search both
        query = [0.1, 0.2, 0.3] + [0.0] * 1533
        mem_results = memgraph_store.vector_search("paper_embedding", query, top_k=2)
        neo_results = neo4j_store.vector_search("paper_embedding", query, top_k=2)

        # Both should return same top result
        assert len(mem_results) == 2
        assert len(neo_results) == 2
        assert mem_results[0]["node"]["arxiv_id"] == neo_results[0]["node"]["arxiv_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
