"""
Graph reasoning implementations.

This package provides multi-hop reasoning capabilities for extracting
context from the graph through various traversal strategies.
"""

from deepgraph.reasoning.strategies import (
    ReasoningStrategy,
    AuthorPapersStrategy,
    CollaboratorPapersStrategy,
    ShortestPathStrategy,
    SubgraphStrategy
)
from deepgraph.reasoning.traversal import GraphReasoner

__all__ = [
    "ReasoningStrategy",
    "AuthorPapersStrategy",
    "CollaboratorPapersStrategy",
    "ShortestPathStrategy",
    "SubgraphStrategy",
    "GraphReasoner",
]
