"""
Graph RAG pipeline orchestration.

This package provides the main pipeline abstraction for orchestrating
the complete Graph RAG workflow.
"""

from deepgraph.pipeline.base import GraphRAGPipeline
from deepgraph.pipeline.stages import (
    PipelineStage,
    PipelineConfig,
    PipelineContext,
    StageType,
    create_default_stages
)
from deepgraph.pipeline.prebuilt import (
    create_simple_pipeline,
    create_multi_hop_pipeline,
    create_pipeline_from_config,
    create_pipeline_from_yaml
)

__all__ = [
    "GraphRAGPipeline",
    "PipelineStage",
    "PipelineConfig",
    "PipelineContext",
    "StageType",
    "create_default_stages",
    "create_simple_pipeline",
    "create_multi_hop_pipeline",
    "create_pipeline_from_config",
    "create_pipeline_from_yaml",
]
