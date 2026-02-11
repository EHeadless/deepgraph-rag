"""
Base synthesis protocol.

This module defines the interface for answer synthesis from graph context,
enabling different LLM providers (OpenAI, Anthropic, local models).
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class AnswerSynthesizer(Protocol):
    """Protocol defining the interface for answer synthesis.

    Synthesizers take graph context and generate natural language answers
    using LLMs.
    """

    def synthesize(
        self,
        query: str,
        context: str,
        **kwargs
    ) -> str:
        """Generate answer from query and context.

        Args:
            query: User's question
            context: Retrieved graph context as text
            **kwargs: Model-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Generated answer as string

        Example:
            answer = synthesizer.synthesize(
                query="What papers discuss transformers?",
                context="1. Attention Is All You Need...",
                temperature=0.3
            )
        """
        ...

    def synthesize_with_metadata(
        self,
        query: str,
        context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate answer with additional metadata.

        Args:
            query: User's question
            context: Retrieved graph context as text
            **kwargs: Model-specific parameters

        Returns:
            Dictionary with 'answer', 'metadata', and usage information

        Example:
            result = synthesizer.synthesize_with_metadata(
                query="...",
                context="..."
            )
            print(result["answer"])
            print(f"Tokens used: {result['metadata']['total_tokens']}")
        """
        ...
