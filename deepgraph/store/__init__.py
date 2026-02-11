"""
Graph store implementations.

This package provides different graph database backends that implement
the GraphStore protocol, enabling backend swapping without code changes.
"""

from deepgraph.store.base import (
    GraphStore,
    BaseGraphStore,
    create_graph_store
)
from deepgraph.store.neo4j import Neo4jGraphStore
from deepgraph.store.memgraph import MemGraphStore

__all__ = [
    "GraphStore",
    "BaseGraphStore",
    "create_graph_store",
    "Neo4jGraphStore",
    "MemGraphStore",
]
