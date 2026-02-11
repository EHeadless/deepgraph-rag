"""
Query DSL (Domain-Specific Language).

This module defines the data structures for representing parsed queries,
intents, and execution plans.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class QueryType(Enum):
    """Types of graph RAG queries."""

    # Basic retrieval
    SIMILARITY = "similarity"                    # Vector search only
    KEYWORD = "keyword"                          # Keyword/filter search

    # Multi-hop reasoning
    AUTHOR_PAPERS = "author_papers"              # Paper → Authors → Papers
    COLLABORATOR_PAPERS = "collaborator_papers"  # Author → Papers → Co-authors → Papers
    RELATED_BY_CONCEPT = "related_by_concept"    # Paper → Concepts → Papers

    # Path finding
    CONNECTION_PATH = "connection_path"          # Find path between entities
    SHORTEST_PATH = "shortest_path"              # Shortest connection

    # Network analysis
    SUBGRAPH = "subgraph"                        # Extract local subgraph
    AUTHOR_NETWORK = "author_network"            # Author collaboration network

    # Comparison
    COMPARISON = "comparison"                    # Compare entities

    # Aggregation
    SUMMARY = "summary"                          # Summarize information
    TREND_ANALYSIS = "trend_analysis"            # Analyze trends over time


@dataclass
class QueryIntent:
    """Structured representation of query intent.

    This captures what the user is trying to find and how to find it.

    Example:
        intent = QueryIntent(
            query_type=QueryType.AUTHOR_PAPERS,
            entities=["Geoffrey Hinton"],
            target_label="Author",
            reasoning_depth=2,
            top_k=10
        )
    """

    # Query classification
    query_type: QueryType

    # Entities mentioned in query
    entities: List[str] = field(default_factory=list)

    # Target entity types
    target_label: Optional[str] = None  # e.g., "Paper", "Author"

    # Relationships to traverse
    relationship_types: List[str] = field(default_factory=list)

    # Constraints/filters
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Reasoning parameters
    reasoning_depth: int = 2  # Max hops for multi-hop queries
    top_k: int = 10          # Number of results

    # Special parameters
    source_entity: Optional[str] = None   # For path queries
    target_entity: Optional[str] = None   # For path queries
    comparison_entities: List[str] = field(default_factory=list)  # For comparisons

    # Metadata
    confidence: float = 1.0  # Parser confidence (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query_type": self.query_type.value,
            "entities": self.entities,
            "target_label": self.target_label,
            "relationship_types": self.relationship_types,
            "constraints": self.constraints,
            "reasoning_depth": self.reasoning_depth,
            "top_k": self.top_k,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "comparison_entities": self.comparison_entities,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryIntent":
        """Create from dictionary representation."""
        data = data.copy()
        data["query_type"] = QueryType(data["query_type"])
        return cls(**data)


@dataclass
class QueryPlan:
    """Execution plan for a graph RAG query.

    This specifies how to execute the query through retrieval,
    reasoning, and synthesis stages.

    Example:
        plan = QueryPlan(
            intent=intent,
            retrieval_strategy="vector",
            reasoning_strategy="author_papers",
            synthesis_config={"temperature": 0.3},
            use_planning=True
        )
    """

    # The parsed intent
    intent: QueryIntent

    # Retrieval stage
    retrieval_strategy: str = "vector"  # "vector", "keyword", "hybrid"
    retrieval_params: Dict[str, Any] = field(default_factory=dict)

    # Reasoning stage
    reasoning_strategy: Optional[str] = None  # Strategy name or None for no reasoning
    reasoning_params: Dict[str, Any] = field(default_factory=dict)

    # Synthesis stage
    synthesis_config: Dict[str, Any] = field(default_factory=dict)
    synthesis_prompt_template: Optional[str] = None  # Custom prompt template

    # Execution metadata
    use_planning: bool = True
    estimated_cost: Optional[float] = None  # Estimated API cost
    estimated_latency: Optional[float] = None  # Estimated latency (ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "intent": self.intent.to_dict(),
            "retrieval_strategy": self.retrieval_strategy,
            "retrieval_params": self.retrieval_params,
            "reasoning_strategy": self.reasoning_strategy,
            "reasoning_params": self.reasoning_params,
            "synthesis_config": self.synthesis_config,
            "synthesis_prompt_template": self.synthesis_prompt_template,
            "use_planning": self.use_planning,
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPlan":
        """Create from dictionary representation."""
        data = data.copy()
        data["intent"] = QueryIntent.from_dict(data["intent"])
        return cls(**data)


@dataclass
class QueryResult:
    """Result of executing a query plan.

    Contains the answer, context, and execution metadata.
    """

    # The generated answer
    answer: str

    # Retrieved context
    context: Dict[str, Any]
    context_text: str

    # Execution metadata
    query: str
    plan: QueryPlan

    # Performance metrics
    retrieval_time_ms: float = 0
    reasoning_time_ms: float = 0
    synthesis_time_ms: float = 0
    total_time_ms: float = 0

    # Token usage (if available)
    tokens_used: Optional[Dict[str, int]] = None

    # Number of nodes/edges involved
    nodes_retrieved: int = 0
    edges_traversed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "answer": self.answer,
            "context": self.context,
            "context_text": self.context_text,
            "query": self.query,
            "plan": self.plan.to_dict(),
            "retrieval_time_ms": self.retrieval_time_ms,
            "reasoning_time_ms": self.reasoning_time_ms,
            "synthesis_time_ms": self.synthesis_time_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_used": self.tokens_used,
            "nodes_retrieved": self.nodes_retrieved,
            "edges_traversed": self.edges_traversed
        }


# Pre-defined query patterns for quick intent detection
QUERY_PATTERNS = {
    QueryType.SIMILARITY: [
        r"papers? (about|on|discussing)",
        r"research on",
        r"find papers",
        r"what papers"
    ],
    QueryType.AUTHOR_PAPERS: [
        r"papers? by",
        r"what has .* written",
        r"works? by",
        r"publications? by",
        r"authored by"
    ],
    QueryType.COLLABORATOR_PAPERS: [
        r"collaborat(ed|ors|ion)",
        r"co-author",
        r"worked with",
        r"papers? .* with"
    ],
    QueryType.CONNECTION_PATH: [
        r"how (are|is) .* (connected|related)",
        r"connection between",
        r"link between",
        r"relationship between"
    ],
    QueryType.COMPARISON: [
        r"compare",
        r"difference between",
        r"vs\.?",
        r"versus",
        r"contrast"
    ],
    QueryType.SUMMARY: [
        r"summarize",
        r"overview",
        r"tell me about",
        r"what is",
        r"who is"
    ],
    QueryType.AUTHOR_NETWORK: [
        r"network",
        r"collaboration network",
        r"research community",
        r"who .* work with"
    ]
}


# Strategy mapping: QueryType → Reasoning Strategy
STRATEGY_MAPPING = {
    QueryType.SIMILARITY: None,  # No reasoning, just retrieval
    QueryType.AUTHOR_PAPERS: "author_papers",
    QueryType.COLLABORATOR_PAPERS: "collaborator_papers",
    QueryType.CONNECTION_PATH: "shortest_path",
    QueryType.SHORTEST_PATH: "shortest_path",
    QueryType.SUBGRAPH: "subgraph",
    QueryType.AUTHOR_NETWORK: "subgraph",
    QueryType.COMPARISON: "author_papers",  # Get papers for comparison
    QueryType.SUMMARY: None,  # Just retrieval + synthesis
    QueryType.TREND_ANALYSIS: "author_papers"
}


# Synthesis prompt mapping: QueryType → Prompt Template
SYNTHESIS_PROMPT_MAPPING = {
    QueryType.SIMILARITY: "default",
    QueryType.AUTHOR_PAPERS: "paper_recommendation",
    QueryType.COLLABORATOR_PAPERS: "paper_recommendation",
    QueryType.CONNECTION_PATH: "connection_explanation",
    QueryType.COMPARISON: "comparison",
    QueryType.SUMMARY: "summary",
    QueryType.AUTHOR_NETWORK: "author_expertise",
    QueryType.TREND_ANALYSIS: "trend_analysis"
}


def create_simple_plan(query: str, query_type: QueryType, **kwargs) -> QueryPlan:
    """Create a simple query plan without full intent parsing.

    Args:
        query: User query text
        query_type: Type of query
        **kwargs: Additional parameters for intent

    Returns:
        QueryPlan instance

    Example:
        plan = create_simple_plan(
            "What papers discuss transformers?",
            QueryType.SIMILARITY,
            top_k=10
        )
    """
    intent = QueryIntent(
        query_type=query_type,
        **kwargs
    )

    plan = QueryPlan(
        intent=intent,
        retrieval_strategy="vector",
        reasoning_strategy=STRATEGY_MAPPING.get(query_type),
        synthesis_prompt_template=SYNTHESIS_PROMPT_MAPPING.get(query_type, "default")
    )

    return plan
