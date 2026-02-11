"""
Base embedder protocol.

This module defines the interface for embedding generation, enabling
swapping between different embedding providers (OpenAI, HuggingFace, etc.).
"""

from typing import Protocol, List, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol defining the interface for embedding generation.

    All embedder implementations must implement these methods to be
    compatible with the framework.
    """

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Example:
            embedding = embedder.embed("What is a transformer?")
            # Returns: [0.1, 0.2, ..., 0.5]  # 1536-dim vector
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Example:
            embeddings = embedder.embed_batch([
                "First text",
                "Second text"
            ])
            # Returns: [[0.1, ...], [0.2, ...]]
        """
        ...

    @property
    def dimensions(self) -> int:
        """Get embedding dimension size.

        Returns:
            Number of dimensions in the embedding vector
        """
        ...

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model.

        Returns:
            Model name (e.g., "text-embedding-ada-002")
        """
        ...
