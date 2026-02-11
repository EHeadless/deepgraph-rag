"""
OpenAI embedder implementation.

This module provides an embedder using OpenAI's text-embedding models.
"""

from typing import List
from openai import OpenAI


class OpenAIEmbedder:
    """OpenAI implementation of the Embedder protocol.

    This class wraps OpenAI's embedding API to generate vector representations
    of text using models like text-embedding-ada-002.

    Example:
        embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-ada-002")
        embedding = embedder.embed("What is a transformer?")
        print(len(embedding))  # 1536
    """

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002", client: OpenAI = None):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key (not needed if client provided)
            model: Embedding model name
            client: Optional pre-configured OpenAI client

        Raises:
            ValueError: If model is not supported
        """
        if model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Choose from {list(self.MODEL_DIMENSIONS.keys())}"
            )

        self._model = model
        self._client = client or OpenAI(api_key=api_key)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Example:
            embedding = embedder.embed("What is a transformer?")
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Note:
            OpenAI supports batching up to 2048 texts per request.
            For larger batches, this method will chunk automatically.

        Example:
            embeddings = embedder.embed_batch([
                "First text",
                "Second text"
            ])
        """
        if not texts:
            return []

        # OpenAI supports up to 2048 texts per request
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    @property
    def dimensions(self) -> int:
        """Get embedding dimension size.

        Returns:
            Number of dimensions in the embedding vector
        """
        return self.MODEL_DIMENSIONS[self._model]

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model.

        Returns:
            Model name (e.g., "text-embedding-ada-002")
        """
        return self._model

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIEmbedder(model={self._model}, dimensions={self.dimensions})"
