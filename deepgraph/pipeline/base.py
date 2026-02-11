"""
Graph RAG Pipeline base class.

This module provides the main GraphRAGPipeline class that orchestrates
the entire retrieval, reasoning, and synthesis workflow.
"""

import time
from typing import Optional, Dict, Any
from deepgraph.pipeline.stages import (
    PipelineConfig,
    PipelineContext,
    PipelineStage,
    StageType,
    create_default_stages
)
from deepgraph.retrieval.base import CandidateRetriever
from deepgraph.reasoning.traversal import GraphReasoner
from deepgraph.synthesis.base import AnswerSynthesizer
from deepgraph.planning.intent import IntentParser
from deepgraph.planning.executor import QueryExecutor
from deepgraph.planning.dsl import QueryResult


class GraphRAGPipeline:
    """Complete Graph RAG pipeline.

    This class orchestrates the full workflow:
    1. Intent parsing (optional)
    2. Retrieval
    3. Reasoning (optional)
    4. Synthesis

    Example:
        from deepgraph.pipeline import GraphRAGPipeline
        from deepgraph.retrieval import VectorRetriever
        from deepgraph.reasoning import GraphReasoner
        from deepgraph.synthesis import OpenAISynthesizer
        from deepgraph.planning import IntentParser

        pipeline = GraphRAGPipeline(
            retriever=VectorRetriever(...),
            reasoner=GraphReasoner(...),
            synthesizer=OpenAISynthesizer(...),
            intent_parser=IntentParser(...),
            config=PipelineConfig(use_planning=True)
        )

        result = pipeline.run("What papers discuss transformers?")
        print(result.answer)
    """

    def __init__(
        self,
        retriever: CandidateRetriever,
        reasoner: GraphReasoner,
        synthesizer: AnswerSynthesizer,
        intent_parser: Optional[IntentParser] = None,
        config: Optional[PipelineConfig] = None
    ):
        """Initialize pipeline.

        Args:
            retriever: Candidate retriever instance
            reasoner: Graph reasoner instance
            synthesizer: Answer synthesizer instance
            intent_parser: Optional intent parser for automatic planning
            config: Optional pipeline configuration
        """
        self._retriever = retriever
        self._reasoner = reasoner
        self._synthesizer = synthesizer
        self._intent_parser = intent_parser
        self._config = config or PipelineConfig()

        # Create executor
        self._executor = QueryExecutor(retriever, reasoner, synthesizer)

        # Initialize stages
        self._stages = create_default_stages()

        # Disable intent parsing if no parser provided
        if not self._intent_parser:
            self._stages["intent_parsing"].enabled = False
            self._config.use_planning = False

    def run(
        self,
        query: str,
        top_k: int = 10,
        use_reasoning: bool = True,
        **kwargs
    ) -> QueryResult:
        """Run the complete pipeline.

        Args:
            query: User query
            top_k: Number of results to retrieve
            use_reasoning: Whether to use multi-hop reasoning
            **kwargs: Additional parameters

        Returns:
            QueryResult with answer and metadata

        Example:
            result = pipeline.run("What papers discuss transformers?", top_k=5)
            print(result.answer)
            print(f"Took {result.total_time_ms:.0f}ms")
        """
        start_time = time.time()

        # Create context
        context = PipelineContext(query=query)

        # Execute hooks
        if self._config.on_start:
            self._config.on_start(context)

        try:
            # Execute pipeline
            if self._config.use_planning and self._stages["intent_parsing"].enabled:
                result = self._run_with_planning(query, context, **kwargs)
            else:
                result = self._run_simple(query, top_k, use_reasoning, context, **kwargs)

            # Mark as completed
            context.completed = True

            # Execute completion hook
            if self._config.on_complete:
                self._config.on_complete(context, result)

            return result

        except Exception as e:
            context.error_occurred = True
            context.errors.append(str(e))

            # Execute error hook
            if self._config.on_error:
                self._config.on_error(context, e)

            # Re-raise or fallback
            if self._config.fallback_to_simple and self._config.use_planning:
                # Try simple execution as fallback
                return self._run_simple(query, top_k, use_reasoning, context, **kwargs)
            else:
                raise

    def _run_with_planning(
        self,
        query: str,
        context: PipelineContext,
        **kwargs
    ) -> QueryResult:
        """Run pipeline with intent parsing and planning.

        Args:
            query: User query
            context: Pipeline context
            **kwargs: Additional parameters

        Returns:
            QueryResult
        """
        # Stage 1: Intent Parsing
        if self._execute_stage_hook("intent_parsing", "pre", context):
            parse_start = time.time()
            plan = self._intent_parser.create_plan(query)
            context.plan = plan
            context.intent = plan.intent
            context.timing["intent_parsing"] = (time.time() - parse_start) * 1000
            self._execute_stage_hook("intent_parsing", "post", context)

        # Execute plan
        result = self._executor.execute(
            plan=context.plan,
            query=query,
            verbose=self._config.verbose
        )

        return result

    def _run_simple(
        self,
        query: str,
        top_k: int,
        use_reasoning: bool,
        context: PipelineContext,
        **kwargs
    ) -> QueryResult:
        """Run pipeline with simple execution (no planning).

        Args:
            query: User query
            top_k: Number of results
            use_reasoning: Whether to use reasoning
            context: Pipeline context
            **kwargs: Additional parameters

        Returns:
            QueryResult
        """
        result = self._executor.execute_simple(
            query=query,
            top_k=top_k,
            use_reasoning=use_reasoning,
            verbose=self._config.verbose
        )

        return result

    def _execute_stage_hook(
        self,
        stage_name: str,
        hook_type: str,
        context: PipelineContext
    ) -> bool:
        """Execute stage hook if defined.

        Args:
            stage_name: Name of the stage
            hook_type: "pre" or "post"
            context: Pipeline context

        Returns:
            True if stage should continue, False to skip
        """
        if stage_name not in self._stages:
            return True

        stage = self._stages[stage_name]

        # Check if stage is enabled
        if not stage.enabled:
            return False

        # Execute hook
        hook = stage.pre_hook if hook_type == "pre" else stage.post_hook
        if hook:
            try:
                result = hook(context)
                return result if result is not None else True
            except Exception as e:
                if stage.skip_on_error:
                    context.errors.append(f"Hook error in {stage_name}: {e}")
                    return False
                else:
                    raise

        return True

    def disable_stage(self, stage_name: str) -> None:
        """Disable a pipeline stage.

        Args:
            stage_name: Name of stage to disable

        Example:
            pipeline.disable_stage("reasoning")  # Skip reasoning
            result = pipeline.run("What papers discuss transformers?")
        """
        if stage_name in self._stages:
            self._stages[stage_name].enabled = False

            # Special handling for reasoning
            if stage_name == "reasoning":
                # Reasoning is controlled by the executor
                pass

    def enable_stage(self, stage_name: str) -> None:
        """Enable a pipeline stage.

        Args:
            stage_name: Name of stage to enable
        """
        if stage_name in self._stages:
            self._stages[stage_name].enabled = True

    def set_stage_hook(
        self,
        stage_name: str,
        hook_type: str,
        hook_fn: callable
    ) -> None:
        """Set a hook for a stage.

        Args:
            stage_name: Name of stage
            hook_type: "pre" or "post"
            hook_fn: Hook function

        Example:
            def log_retrieval(context):
                print(f"Starting retrieval for: {context.query}")

            pipeline.set_stage_hook("retrieval", "pre", log_retrieval)
        """
        if stage_name in self._stages:
            if hook_type == "pre":
                self._stages[stage_name].pre_hook = hook_fn
            elif hook_type == "post":
                self._stages[stage_name].post_hook = hook_fn

    def get_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self._config

    def get_stages(self) -> Dict[str, PipelineStage]:
        """Get pipeline stages."""
        return self._stages

    @property
    def retriever(self) -> CandidateRetriever:
        """Get the retriever."""
        return self._retriever

    @property
    def reasoner(self) -> GraphReasoner:
        """Get the reasoner."""
        return self._reasoner

    @property
    def synthesizer(self) -> AnswerSynthesizer:
        """Get the synthesizer."""
        return self._synthesizer

    @property
    def intent_parser(self) -> Optional[IntentParser]:
        """Get the intent parser."""
        return self._intent_parser

    def __repr__(self) -> str:
        """String representation."""
        enabled_stages = [name for name, stage in self._stages.items() if stage.enabled]
        return (
            f"GraphRAGPipeline("
            f"planning={self._config.use_planning}, "
            f"enabled_stages={enabled_stages})"
        )
