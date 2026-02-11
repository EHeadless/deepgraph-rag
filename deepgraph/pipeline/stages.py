"""
Pipeline stage definitions.

This module defines the stages that make up the Graph RAG pipeline
and their configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from enum import Enum


class StageType(Enum):
    """Types of pipeline stages."""
    INTENT_PARSING = "intent_parsing"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"


@dataclass
class PipelineStage:
    """Definition of a pipeline stage.

    Example:
        stage = PipelineStage(
            name="retrieval",
            stage_type=StageType.RETRIEVAL,
            enabled=True,
            config={"top_k": 10}
        )
    """

    # Stage identification
    name: str
    stage_type: StageType

    # Stage control
    enabled: bool = True
    skip_on_error: bool = False

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Hooks
    pre_hook: Optional[Callable] = None   # Called before stage
    post_hook: Optional[Callable] = None  # Called after stage

    # Caching
    cache_enabled: bool = False
    cache_key_fn: Optional[Callable] = None


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline.

    Example:
        config = PipelineConfig(
            use_planning=True,
            enable_caching=False,
            max_retries=3,
            timeout_ms=30000
        )
    """

    # Planning
    use_planning: bool = True  # Use intent parser
    fallback_to_simple: bool = True  # Fall back to simple execution on parse error

    # Performance
    enable_caching: bool = False
    cache_ttl_seconds: int = 3600
    max_retries: int = 3
    timeout_ms: int = 30000

    # Logging
    verbose: bool = False
    log_level: str = "INFO"

    # Stage-specific configs
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    reasoning_config: Dict[str, Any] = field(default_factory=dict)
    synthesis_config: Dict[str, Any] = field(default_factory=dict)

    # Hooks
    on_start: Optional[Callable] = None
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None


@dataclass
class PipelineContext:
    """Context passed through pipeline stages.

    This holds intermediate results as the query flows through stages.
    """

    # Original query
    query: str

    # Stage results
    intent: Optional[Any] = None
    plan: Optional[Any] = None
    candidate_ids: Optional[list] = None
    reasoning_context: Optional[Dict[str, Any]] = None
    context_text: Optional[str] = None
    answer: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    errors: list = field(default_factory=list)

    # Flags
    completed: bool = False
    error_occurred: bool = False


def create_default_stages() -> Dict[str, PipelineStage]:
    """Create default pipeline stages.

    Returns:
        Dictionary of stage_name -> PipelineStage
    """
    return {
        "intent_parsing": PipelineStage(
            name="intent_parsing",
            stage_type=StageType.INTENT_PARSING,
            enabled=True
        ),
        "retrieval": PipelineStage(
            name="retrieval",
            stage_type=StageType.RETRIEVAL,
            enabled=True
        ),
        "reasoning": PipelineStage(
            name="reasoning",
            stage_type=StageType.REASONING,
            enabled=True
        ),
        "synthesis": PipelineStage(
            name="synthesis",
            stage_type=StageType.SYNTHESIS,
            enabled=True
        )
    }
