"""
Domain-agnostic graph schema definitions.

This module provides base classes for defining graph schemas that work across
any domain (research papers, CRM, medical records, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import yaml


@dataclass
class NodeSchema:
    """Schema definition for a graph node type.

    Attributes:
        label: Node label (e.g., "Paper", "Author", "Company")
        id_field: Primary identifier field name
        properties: Property name → type mapping
        indexes: Fields to create indexes on
        constraints: Fields with unique constraints
        vector_config: Optional vector index configuration
    """
    label: str
    id_field: str
    properties: Dict[str, type]
    indexes: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    vector_config: Optional[Dict[str, Any]] = None

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate node data against this schema.

        Args:
            data: Node property dictionary

        Returns:
            True if valid, False otherwise
        """
        # Check required field exists
        if self.id_field not in data:
            return False

        # Validate property types
        for prop, prop_type in self.properties.items():
            if prop in data:
                # Handle special types
                if prop_type == list:
                    if not isinstance(data[prop], (list, tuple)):
                        return False
                elif not isinstance(data[prop], prop_type):
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            "label": self.label,
            "id_field": self.id_field,
            "properties": {k: v.__name__ for k, v in self.properties.items()},
            "indexes": self.indexes,
            "constraints": self.constraints,
            "vector_config": self.vector_config
        }


@dataclass
class EdgeSchema:
    """Schema definition for a graph relationship type.

    Attributes:
        type: Relationship type name (e.g., "AUTHORED_BY", "WORKS_AT")
        from_label: Source node label
        to_label: Target node label
        properties: Optional relationship properties
    """
    type: str
    from_label: str
    to_label: str
    properties: Dict[str, type] = field(default_factory=dict)

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate relationship data against this schema.

        Args:
            data: Relationship property dictionary

        Returns:
            True if valid, False otherwise
        """
        for prop, prop_type in self.properties.items():
            if prop in data and not isinstance(data[prop], prop_type):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            "type": self.type,
            "from_label": self.from_label,
            "to_label": self.to_label,
            "properties": {k: v.__name__ for k, v in self.properties.items()}
        }


@dataclass
class GraphSchema:
    """Complete graph schema definition for a domain.

    This represents the full schema for a knowledge graph in any domain,
    including all node types, relationship types, and constraints.

    Attributes:
        name: Schema name (e.g., "arxiv", "crm", "medical")
        nodes: Dictionary of node label → NodeSchema
        edges: Dictionary of relationship type → EdgeSchema
        metadata: Optional schema metadata
    """
    name: str
    nodes: Dict[str, NodeSchema]
    edges: Dict[str, EdgeSchema]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "GraphSchema":
        """Load schema from YAML configuration file.

        Args:
            path: Path to YAML schema file

        Returns:
            GraphSchema instance

        Example YAML:
            schema:
              name: arxiv_research
              nodes:
                Paper:
                  id_field: arxiv_id
                  properties:
                    arxiv_id: string
                    title: string
                    embedding: vector
                  indexes:
                    - published_date
                  vector_config:
                    field: embedding
                    dimensions: 1536
        """
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        schema_config = config.get('schema', config)

        # Parse node schemas
        nodes = {}
        for label, node_config in schema_config.get('nodes', {}).items():
            # Map YAML type strings to Python types
            properties = {}
            for prop_name, prop_type_str in node_config.get('properties', {}).items():
                properties[prop_name] = cls._parse_type(prop_type_str)

            nodes[label] = NodeSchema(
                label=label,
                id_field=node_config['id_field'],
                properties=properties,
                indexes=node_config.get('indexes', []),
                constraints=node_config.get('constraints', []),
                vector_config=node_config.get('vector_config')
            )

        # Parse edge schemas
        edges = {}
        for edge_type, edge_config in schema_config.get('edges', {}).items():
            properties = {}
            for prop_name, prop_type_str in edge_config.get('properties', {}).items():
                properties[prop_name] = cls._parse_type(prop_type_str)

            edges[edge_type] = EdgeSchema(
                type=edge_type,
                from_label=edge_config['from'],
                to_label=edge_config['to'],
                properties=properties
            )

        return cls(
            name=schema_config['name'],
            nodes=nodes,
            edges=edges,
            metadata=schema_config.get('metadata', {})
        )

    @staticmethod
    def _parse_type(type_str: str) -> type:
        """Parse YAML type string to Python type.

        Args:
            type_str: Type name like 'string', 'integer', 'float', 'vector', etc.

        Returns:
            Python type object
        """
        type_map = {
            'string': str,
            'str': str,
            'integer': int,
            'int': int,
            'float': float,
            'boolean': bool,
            'bool': bool,
            'datetime': datetime,
            'vector': list,  # Vector embeddings are lists of floats
            'list': list,
            'dict': dict,
        }
        return type_map.get(type_str.lower(), str)

    def to_cypher_ddl(self) -> List[str]:
        """Generate Neo4j DDL statements for schema creation.

        Returns:
            List of Cypher DDL statements
        """
        statements = []

        # Create node constraints
        for node in self.nodes.values():
            # Unique constraint on ID field
            if node.id_field:
                statements.append(
                    f"CREATE CONSTRAINT {node.label.lower()}_{node.id_field} IF NOT EXISTS "
                    f"FOR (n:{node.label}) REQUIRE n.{node.id_field} IS UNIQUE"
                )

            # Additional unique constraints
            for constraint_field in node.constraints:
                if constraint_field != node.id_field:
                    statements.append(
                        f"CREATE CONSTRAINT {node.label.lower()}_{constraint_field} IF NOT EXISTS "
                        f"FOR (n:{node.label}) REQUIRE n.{constraint_field} IS UNIQUE"
                    )

            # Create indexes
            for index_field in node.indexes:
                statements.append(
                    f"CREATE INDEX {node.label.lower()}_{index_field} IF NOT EXISTS "
                    f"FOR (n:{node.label}) ON (n.{index_field})"
                )

            # Create vector index if configured
            if node.vector_config:
                field = node.vector_config['field']
                dims = node.vector_config.get('dimensions', 1536)
                similarity = node.vector_config.get('similarity', 'cosine')
                index_name = node.vector_config.get('index_name', f"{node.label.lower()}_embedding")

                statements.append(
                    f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                    f"FOR (n:{node.label}) ON (n.{field}) "
                    f"OPTIONS {{"
                    f"  indexConfig: {{"
                    f"    `vector.dimensions`: {dims}, "
                    f"    `vector.similarity_function`: '{similarity}'"
                    f"  }}"
                    f"}}"
                )

        return statements

    def validate_node(self, label: str, data: Dict[str, Any]) -> bool:
        """Validate node data against schema.

        Args:
            label: Node label
            data: Node properties

        Returns:
            True if valid
        """
        if label not in self.nodes:
            return False
        return self.nodes[label].validate(data)

    def validate_edge(self, edge_type: str, data: Dict[str, Any]) -> bool:
        """Validate edge data against schema.

        Args:
            edge_type: Relationship type
            data: Relationship properties

        Returns:
            True if valid
        """
        if edge_type not in self.edges:
            return False
        return self.edges[edge_type].validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            "name": self.name,
            "nodes": {label: node.to_dict() for label, node in self.nodes.items()},
            "edges": {etype: edge.to_dict() for etype, edge in self.edges.items()},
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        """String representation of schema."""
        return (
            f"GraphSchema(name='{self.name}', "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )
