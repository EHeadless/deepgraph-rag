"""
Embedder implementations.

This package provides embedders for generating vector representations of text.
"""

from deepgraph.adapters.embedders.base import Embedder
from deepgraph.adapters.embedders.openai import OpenAIEmbedder

__all__ = [
    "Embedder",
    "OpenAIEmbedder",
]
