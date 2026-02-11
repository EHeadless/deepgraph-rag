"""
Prompt templates for answer synthesis.

This module provides reusable prompt templates for different types
of graph RAG queries.
"""

from typing import Dict


class PromptTemplates:
    """Collection of prompt templates for answer synthesis."""

    # Default research assistant prompt
    DEFAULT = (
        "You are a helpful research assistant. Answer questions based on the "
        "provided graph context from academic papers. Be concise and factual. "
        "If the context doesn't contain relevant information, say so clearly."
    )

    # Paper recommendation prompt
    PAPER_RECOMMENDATION = (
        "You are a research paper recommendation assistant. Based on the provided "
        "graph context showing related papers, authors, and connections, provide "
        "relevant paper recommendations. Explain why each paper is relevant and "
        "how it connects to the query."
    )

    # Author expertise prompt
    AUTHOR_EXPERTISE = (
        "You are an academic researcher analyst. Based on the provided graph context "
        "showing an author's papers, collaborators, and research topics, summarize "
        "their expertise and research contributions. Be specific and cite their work."
    )

    # Research trend analysis
    TREND_ANALYSIS = (
        "You are a research trend analyst. Based on the provided graph context "
        "showing papers, their relationships, and temporal information, identify "
        "and explain research trends. Focus on how ideas evolved and connected."
    )

    # Connection explanation
    CONNECTION_EXPLANATION = (
        "You are a research network analyst. Based on the provided graph context "
        "showing connections between authors or papers, explain how they are "
        "related. Highlight the shortest path and key connecting elements."
    )

    # Comparison prompt
    COMPARISON = (
        "You are a research comparison specialist. Based on the provided graph "
        "context, compare and contrast the papers, methods, or authors mentioned. "
        "Highlight similarities, differences, and relationships."
    )

    # Summary prompt
    SUMMARY = (
        "You are a research summarization assistant. Based on the provided graph "
        "context, create a concise summary that captures the key information. "
        "Focus on the most important papers, authors, and relationships."
    )

    @classmethod
    def get(cls, template_name: str) -> str:
        """Get a prompt template by name.

        Args:
            template_name: Name of the template

        Returns:
            Prompt template string

        Raises:
            ValueError: If template name not found

        Example:
            prompt = PromptTemplates.get("paper_recommendation")
        """
        template_name_upper = template_name.upper()
        if hasattr(cls, template_name_upper):
            return getattr(cls, template_name_upper)
        else:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available: {cls.available_templates()}"
            )

    @classmethod
    def available_templates(cls) -> list:
        """Get list of available template names.

        Returns:
            List of template names
        """
        return [
            name.lower()
            for name in dir(cls)
            if not name.startswith("_") and name.isupper()
        ]

    @classmethod
    def format_context(
        cls,
        query: str,
        context: str,
        include_instructions: bool = True
    ) -> str:
        """Format query and context for LLM consumption.

        Args:
            query: User's question
            context: Graph context string
            include_instructions: Whether to include answer instructions

        Returns:
            Formatted prompt string

        Example:
            user_message = PromptTemplates.format_context(
                query="What papers discuss transformers?",
                context="1. Attention Is All You Need..."
            )
        """
        parts = []

        # Add context
        parts.append("# Graph Context\n")
        parts.append(context)
        parts.append("\n")

        # Add instructions (optional)
        if include_instructions:
            parts.append("# Instructions\n")
            parts.append(
                "Answer the question based on the graph context above. "
                "Be specific and cite relevant papers or authors. "
                "If the context doesn't contain enough information, "
                "acknowledge what you can and cannot answer.\n\n"
            )

        # Add query
        parts.append(f"# Question\n{query}")

        return "".join(parts)


# Pre-defined query type detection patterns
QUERY_TYPE_PATTERNS = {
    "paper_recommendation": [
        "recommend", "suggest", "similar papers", "related work"
    ],
    "author_expertise": [
        "who is", "what has", "worked on", "expertise", "research focus"
    ],
    "trend_analysis": [
        "trend", "evolution", "over time", "how has", "changed"
    ],
    "connection_explanation": [
        "how are", "connected", "relationship between", "link between"
    ],
    "comparison": [
        "compare", "difference between", "vs", "versus", "contrast"
    ],
    "summary": [
        "summarize", "overview", "tell me about", "what is"
    ]
}


def detect_query_type(query: str) -> str:
    """Detect the type of query based on keywords.

    Args:
        query: User's question

    Returns:
        Query type name (or "default" if no match)

    Example:
        query_type = detect_query_type("Recommend papers about transformers")
        # Returns: "paper_recommendation"
    """
    query_lower = query.lower()

    for query_type, patterns in QUERY_TYPE_PATTERNS.items():
        if any(pattern in query_lower for pattern in patterns):
            return query_type

    return "default"


def get_prompt_for_query(query: str) -> str:
    """Get the best prompt template for a query.

    Args:
        query: User's question

    Returns:
        Appropriate prompt template string

    Example:
        prompt = get_prompt_for_query("Compare BERT and GPT")
        # Returns comparison prompt
    """
    query_type = detect_query_type(query)

    if query_type == "default":
        return PromptTemplates.DEFAULT
    else:
        return PromptTemplates.get(query_type)
