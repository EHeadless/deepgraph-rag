"""
Query planning and execution.

This package provides intelligent query parsing, intent detection,
and execution planning for graph RAG queries.
"""

from deepgraph.planning.dsl import (
    QueryType,
    QueryIntent,
    QueryPlan,
    QueryResult,
    create_simple_plan,
    QUERY_PATTERNS,
    STRATEGY_MAPPING,
    SYNTHESIS_PROMPT_MAPPING
)
from deepgraph.planning.intent import IntentParser
from deepgraph.planning.executor import QueryExecutor

__all__ = [
    "QueryType",
    "QueryIntent",
    "QueryPlan",
    "QueryResult",
    "create_simple_plan",
    "IntentParser",
    "QueryExecutor",
    "QUERY_PATTERNS",
    "STRATEGY_MAPPING",
    "SYNTHESIS_PROMPT_MAPPING",
]
