"""
OpenAI answer synthesis implementation.

This module provides an answer synthesizer using OpenAI's chat models.
"""

from typing import Dict, Any, Optional
from openai import OpenAI


class OpenAISynthesizer:
    """OpenAI implementation of the AnswerSynthesizer protocol.

    This class uses OpenAI's chat completion API to generate answers
    from graph context.

    Example:
        synthesizer = OpenAISynthesizer(
            api_key="sk-...",
            model="gpt-4-turbo-preview"
        )

        answer = synthesizer.synthesize(
            query="What papers discuss transformers?",
            context="1. Attention Is All You Need...",
            temperature=0.3
        )
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful research assistant. Answer questions based on the "
        "provided graph context from academic papers. Be concise and factual. "
        "If the context doesn't contain relevant information, say so clearly."
    )

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4-turbo-preview",
        system_prompt: Optional[str] = None,
        client: OpenAI = None
    ):
        """Initialize OpenAI synthesizer.

        Args:
            api_key: OpenAI API key (not needed if client provided)
            model: Chat model name
            system_prompt: Custom system prompt (optional)
            client: Optional pre-configured OpenAI client
        """
        self._model = model
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = client or OpenAI(api_key=api_key)

    def synthesize(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate answer from query and context.

        Args:
            query: User's question
            context: Retrieved graph context as text
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI API parameters

        Returns:
            Generated answer as string

        Example:
            answer = synthesizer.synthesize(
                query="What papers discuss transformers?",
                context="1. Attention Is All You Need...",
                temperature=0.3
            )
        """
        result = self.synthesize_with_metadata(
            query=query,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return result["answer"]

    def synthesize_with_metadata(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate answer with additional metadata.

        Args:
            query: User's question
            context: Retrieved graph context as text
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI API parameters

        Returns:
            Dictionary with 'answer', 'metadata', and usage information

        Example:
            result = synthesizer.synthesize_with_metadata(
                query="...",
                context="..."
            )
            print(result["answer"])
            print(f"Tokens: {result['metadata']['total_tokens']}")
        """
        # Build messages
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"Context from graph:\n\n{context}\n\nQuestion: {query}"
            }
        ]

        # Call OpenAI API
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Extract answer
        answer = response.choices[0].message.content

        # Build metadata
        metadata = {
            "model": self._model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "finish_reason": response.choices[0].finish_reason
        }

        return {
            "answer": answer,
            "metadata": metadata,
            "raw_response": response
        }

    def stream_synthesize(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate answer with streaming (for real-time display).

        Args:
            query: User's question
            context: Retrieved graph context as text
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI API parameters

        Yields:
            Answer chunks as they're generated

        Example:
            for chunk in synthesizer.stream_synthesize(query, context):
                print(chunk, end="", flush=True)
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"Context from graph:\n\n{context}\n\nQuestion: {query}"
            }
        ]

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt.

        Args:
            prompt: New system prompt

        Example:
            synthesizer.set_system_prompt(
                "You are an expert in machine learning. "
                "Answer questions about research papers."
            )
        """
        self._system_prompt = prompt

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._system_prompt

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAISynthesizer(model={self._model})"
