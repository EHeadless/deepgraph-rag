"""
Answer synthesis implementations.

This package provides synthesizers for generating natural language answers
from graph context using LLMs.
"""

from deepgraph.synthesis.base import AnswerSynthesizer
from deepgraph.synthesis.openai import OpenAISynthesizer
from deepgraph.synthesis.prompts import (
    PromptTemplates,
    detect_query_type,
    get_prompt_for_query
)

__all__ = [
    "AnswerSynthesizer",
    "OpenAISynthesizer",
    "PromptTemplates",
    "detect_query_type",
    "get_prompt_for_query",
]
