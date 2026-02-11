"""
Unit tests for core schema module.
"""

import pytest
import tempfile
import os
from deepgraph.core.schema import NodeSchema, EdgeSchema, GraphSchema
from schemas.arxiv import ARXIV_SCHEMA, PAPER_SCHEMA, AUTHOR_SCHEMA


class TestNodeSchema:
    """Tests for NodeSchema class."""

    def test_create_node_schema(self):
        """Test creating a basic node schema."""
        schema = NodeSchema(
            label="TestNode",
            id_field="id",
            properties={"id": str, "name": str, "count": int},
            indexes=["name"],
            constraints=["id"]
        )

        assert schema.label == "TestNode"
        assert schema.id_field == "id"
        assert len(schema.properties) == 3
        assert "name" in schema.indexes
        assert "id" in schema.constraints

    def test_validate_valid_node(self):
        """Test validation with valid node data."""
        schema = NodeSchema(
            label="Paper",
            id_field="arxiv_id",
            properties={"arxiv_id": str, "title": str}
        )

        valid_data = {"arxiv_id": "2301.12345", "title": "Test Paper"}
        assert schema.validate(valid_data) is True

    def test_validate_missing_id_field(self):
        """Test validation fails when ID field is missing."""
        schema = NodeSchema(
            label="Paper",
            id_field="arxiv_id",
            properties={"arxiv_id": str, "title": str}
        )

        invalid_data = {"title": "Test Paper"}  # Missing arxiv_id
        assert schema.validate(invalid_data) is False

    def test_validate_wrong_type(self):
        """Test validation fails with wrong property type."""
        schema = NodeSchema(
            label="Paper",
            id_field="id",
            properties={"id": str, "count": int}
        )

        invalid_data = {"id": "123", "count": "not an int"}
        assert schema.validate(invalid_data) is False

    def test_validate_extra_properties(self):
        """Test validation allows extra properties not in schema."""
        schema = NodeSchema(
            label="Paper",
            id_field="id",
            properties={"id": str, "title": str}
        )

        data_with_extra = {"id": "123", "title": "Test", "extra_field": "allowed"}
        assert schema.validate(data_with_extra) is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        schema = NodeSchema(
            label="Paper",
            id_field="id",
            properties={"id": str, "count": int}
        )

        schema_dict = schema.to_dict()
        assert schema_dict["label"] == "Paper"
        assert schema_dict["id_field"] == "id"
        assert schema_dict["properties"]["id"] == "str"
        assert schema_dict["properties"]["count"] == "int"


class TestEdgeSchema:
    """Tests for EdgeSchema class."""

    def test_create_edge_schema(self):
        """Test creating a basic edge schema."""
        schema = EdgeSchema(
            type="AUTHORED_BY",
            from_label="Paper",
            to_label="Author",
            properties={"position": int}
        )

        assert schema.type == "AUTHORED_BY"
        assert schema.from_label == "Paper"
        assert schema.to_label == "Author"
        assert "position" in schema.properties

    def test_validate_valid_edge(self):
        """Test validation with valid edge data."""
        schema = EdgeSchema(
            type="AUTHORED_BY",
            from_label="Paper",
            to_label="Author",
            properties={"position": int}
        )

        valid_data = {"position": 0}
        assert schema.validate(valid_data) is True

    def test_validate_empty_properties(self):
        """Test validation with no properties."""
        schema = EdgeSchema(
            type="CITES",
            from_label="Paper",
            to_label="Paper"
        )

        assert schema.validate({}) is True
        assert schema.validate({"extra": "allowed"}) is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        schema = EdgeSchema(
            type="AUTHORED_BY",
            from_label="Paper",
            to_label="Author",
            properties={"position": int}
        )

        schema_dict = schema.to_dict()
        assert schema_dict["type"] == "AUTHORED_BY"
        assert schema_dict["from_label"] == "Paper"
        assert schema_dict["to_label"] == "Author"
        assert schema_dict["properties"]["position"] == "int"


class TestGraphSchema:
    """Tests for GraphSchema class."""

    def test_create_graph_schema(self):
        """Test creating a complete graph schema."""
        node_schema = NodeSchema(
            label="Paper",
            id_field="id",
            properties={"id": str, "title": str}
        )

        edge_schema = EdgeSchema(
            type="AUTHORED_BY",
            from_label="Paper",
            to_label="Author"
        )

        graph_schema = GraphSchema(
            name="test_schema",
            nodes={"Paper": node_schema},
            edges={"AUTHORED_BY": edge_schema}
        )

        assert graph_schema.name == "test_schema"
        assert len(graph_schema.nodes) == 1
        assert len(graph_schema.edges) == 1
        assert "Paper" in graph_schema.nodes
        assert "AUTHORED_BY" in graph_schema.edges

    def test_validate_node(self):
        """Test node validation through graph schema."""
        schema = GraphSchema(
            name="test",
            nodes={"Paper": PAPER_SCHEMA},
            edges={}
        )

        valid_paper = {
            "arxiv_id": "2301.12345",
            "title": "Test",
            "abstract": "Abstract",
            "published_date": "2023-01-01",
            "primary_category": "cs.AI"
        }

        assert schema.validate_node("Paper", valid_paper) is True

        # Invalid: missing required field
        invalid_paper = {"title": "Test"}
        assert schema.validate_node("Paper", invalid_paper) is False

        # Invalid: unknown label
        assert schema.validate_node("Unknown", valid_paper) is False

    def test_validate_edge(self):
        """Test edge validation through graph schema."""
        schema = GraphSchema(
            name="test",
            nodes={},
            edges={"AUTHORED_BY": AUTHORED_BY}
        )

        valid_edge = {"position": 0}
        assert schema.validate_edge("AUTHORED_BY", valid_edge) is True

        invalid_edge = {"position": "not an int"}
        assert schema.validate_edge("AUTHORED_BY", invalid_edge) is False

    def test_to_cypher_ddl(self):
        """Test Cypher DDL generation."""
        ddl_statements = ARXIV_SCHEMA.to_cypher_ddl()

        assert isinstance(ddl_statements, list)
        assert len(ddl_statements) > 0

        # Check for constraint creation
        constraint_statements = [s for s in ddl_statements if "CONSTRAINT" in s]
        assert len(constraint_statements) > 0

        # Check for index creation
        index_statements = [s for s in ddl_statements if "INDEX" in s and "VECTOR" not in s]
        assert len(index_statements) > 0

        # Check for vector index creation
        vector_statements = [s for s in ddl_statements if "VECTOR INDEX" in s]
        assert len(vector_statements) > 0

        # Verify specific constraints
        arxiv_id_constraint = any("paper_arxiv_id" in s for s in ddl_statements)
        assert arxiv_id_constraint is True

    def test_from_yaml(self):
        """Test loading schema from YAML file."""
        # Create temporary YAML file
        yaml_content = """
schema:
  name: test_schema
  nodes:
    TestNode:
      id_field: id
      properties:
        id: string
        count: integer
      indexes:
        - count
      constraints:
        - id
  edges:
    TEST_EDGE:
      from: TestNode
      to: TestNode
      properties:
        weight: float
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            schema = GraphSchema.from_yaml(temp_path)

            assert schema.name == "test_schema"
            assert "TestNode" in schema.nodes
            assert "TEST_EDGE" in schema.edges

            # Check node schema
            node = schema.nodes["TestNode"]
            assert node.id_field == "id"
            assert node.properties["id"] == str
            assert node.properties["count"] == int
            assert "count" in node.indexes

            # Check edge schema
            edge = schema.edges["TEST_EDGE"]
            assert edge.from_label == "TestNode"
            assert edge.to_label == "TestNode"
            assert edge.properties["weight"] == float

        finally:
            os.unlink(temp_path)

    def test_type_parsing(self):
        """Test type string parsing."""
        assert GraphSchema._parse_type("string") == str
        assert GraphSchema._parse_type("str") == str
        assert GraphSchema._parse_type("integer") == int
        assert GraphSchema._parse_type("int") == int
        assert GraphSchema._parse_type("float") == float
        assert GraphSchema._parse_type("boolean") == bool
        assert GraphSchema._parse_type("vector") == list
        assert GraphSchema._parse_type("unknown") == str  # Default

    def test_to_dict(self):
        """Test serialization to dictionary."""
        schema_dict = ARXIV_SCHEMA.to_dict()

        assert schema_dict["name"] == "arxiv_research"
        assert "Paper" in schema_dict["nodes"]
        assert "AUTHORED_BY" in schema_dict["edges"]
        assert "metadata" in schema_dict

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(ARXIV_SCHEMA)

        assert "GraphSchema" in repr_str
        assert "arxiv_research" in repr_str
        assert "nodes=4" in repr_str  # Paper, Author, Concept, Method
        assert "edges=5" in repr_str  # AUTHORED_BY, CO_AUTHORED, etc.


class TestArxivSchema:
    """Tests specific to arXiv schema."""

    def test_arxiv_schema_structure(self):
        """Test arXiv schema has expected structure."""
        assert ARXIV_SCHEMA.name == "arxiv_research"
        assert len(ARXIV_SCHEMA.nodes) == 4
        assert len(ARXIV_SCHEMA.edges) == 5

        # Check all expected nodes exist
        expected_nodes = ["Paper", "Author", "Concept", "Method"]
        for node_label in expected_nodes:
            assert node_label in ARXIV_SCHEMA.nodes

        # Check all expected edges exist
        expected_edges = ["AUTHORED_BY", "CO_AUTHORED", "ABOUT_CONCEPT", "USES_METHOD", "CITES"]
        for edge_type in expected_edges:
            assert edge_type in ARXIV_SCHEMA.edges

    def test_paper_schema_vector_config(self):
        """Test Paper schema has vector index configuration."""
        assert PAPER_SCHEMA.vector_config is not None
        assert PAPER_SCHEMA.vector_config["field"] == "embedding"
        assert PAPER_SCHEMA.vector_config["dimensions"] == 1536
        assert PAPER_SCHEMA.vector_config["similarity"] == "cosine"

    def test_author_schema_constraints(self):
        """Test Author schema has unique name constraint."""
        assert "name" in AUTHOR_SCHEMA.constraints
        assert AUTHOR_SCHEMA.id_field == "name"

    def test_arxiv_metadata(self):
        """Test arXiv schema metadata."""
        assert "description" in ARXIV_SCHEMA.metadata
        assert "version" in ARXIV_SCHEMA.metadata
        assert ARXIV_SCHEMA.metadata["embedding_dimensions"] == 1536


# Import AUTHORED_BY for tests
from schemas.arxiv import AUTHORED_BY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
