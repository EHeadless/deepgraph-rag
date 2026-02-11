"""
Query execution orchestration.

This module provides the QueryExecutor that executes QueryPlans through
the retrieval, reasoning, and synthesis pipeline.
"""

import time
from typing import Optional, Dict, Any
from deepgraph.planning.dsl import QueryPlan, QueryResult, QueryType
from deepgraph.retrieval.base import CandidateRetriever
from deepgraph.reasoning.traversal import GraphReasoner
from deepgraph.synthesis.base import AnswerSynthesizer
from deepgraph.synthesis.prompts import PromptTemplates


class QueryExecutor:
    """Executes query plans through the full RAG pipeline.

    This orchestrator takes a QueryPlan and executes it through:
    1. Retrieval - Find relevant candidate nodes
    2. Reasoning - Expand context via multi-hop traversal
    3. Synthesis - Generate answer from context

    Example:
        from deepgraph.retrieval import VectorRetriever
        from deepgraph.reasoning import GraphReasoner
        from deepgraph.synthesis import OpenAISynthesizer

        executor = QueryExecutor(
            retriever=VectorRetriever(...),
            reasoner=GraphReasoner(...),
            synthesizer=OpenAISynthesizer(...)
        )

        result = executor.execute(plan, query="What papers discuss transformers?")
        print(result.answer)
        print(f"Took {result.total_time_ms:.0f}ms")
    """

    def __init__(
        self,
        retriever: CandidateRetriever,
        reasoner: GraphReasoner,
        synthesizer: AnswerSynthesizer
    ):
        """Initialize query executor.

        Args:
            retriever: Candidate retriever instance
            reasoner: Graph reasoner instance
            synthesizer: Answer synthesizer instance
        """
        self._retriever = retriever
        self._reasoner = reasoner
        self._synthesizer = synthesizer

    def execute(
        self,
        plan: QueryPlan,
        query: str,
        verbose: bool = False
    ) -> QueryResult:
        """Execute a query plan.

        Args:
            plan: QueryPlan to execute
            query: Original user query
            verbose: Print execution progress

        Returns:
            QueryResult with answer and metadata

        Example:
            result = executor.execute(plan, "What papers discuss transformers?")
        """
        start_time = time.time()

        if verbose:
            print(f"Executing query: {query}")
            print(f"Query type: {plan.intent.query_type.value}")
            print(f"Retrieval: {plan.retrieval_strategy}")
            print(f"Reasoning: {plan.reasoning_strategy or 'None'}")
            print()

        # Stage 1: Retrieval
        retrieval_start = time.time()
        candidate_ids = self._execute_retrieval(plan, query, verbose)
        retrieval_time = (time.time() - retrieval_start) * 1000

        if verbose:
            print(f"✓ Retrieved {len(candidate_ids)} candidates in {retrieval_time:.0f}ms")

        # Stage 2: Reasoning (optional)
        reasoning_start = time.time()
        if plan.reasoning_strategy:
            context = self._execute_reasoning(plan, candidate_ids, verbose)
        else:
            # No reasoning - just return candidates as context
            context = {
                "papers": [],
                "authors": [],
                "connections": [],
                "_meta": {
                    "strategy": "none",
                    "num_candidates": len(candidate_ids)
                }
            }
            # Fetch candidate nodes
            for cid in candidate_ids[:plan.intent.top_k]:
                node = self._reasoner._store.get_node(
                    plan.intent.target_label or "Paper",
                    cid,
                    self._retriever._id_field
                )
                if node:
                    context["papers"].append(node)

        reasoning_time = (time.time() - reasoning_start) * 1000

        if verbose:
            print(f"✓ Expanded context in {reasoning_time:.0f}ms")

        # Stage 3: Synthesis
        synthesis_start = time.time()
        answer, synthesis_metadata = self._execute_synthesis(
            plan, query, context, verbose
        )
        synthesis_time = (time.time() - synthesis_start) * 1000

        if verbose:
            print(f"✓ Generated answer in {synthesis_time:.0f}ms")

        # Build result
        total_time = (time.time() - start_time) * 1000

        # Format context as text
        context_text = self._reasoner.format_context_for_llm(context)

        # Count nodes and edges
        nodes_retrieved = len(context.get("papers", [])) + len(context.get("authors", []))
        edges_traversed = len(context.get("connections", []))

        result = QueryResult(
            answer=answer,
            context=context,
            context_text=context_text,
            query=query,
            plan=plan,
            retrieval_time_ms=retrieval_time,
            reasoning_time_ms=reasoning_time,
            synthesis_time_ms=synthesis_time,
            total_time_ms=total_time,
            tokens_used=synthesis_metadata.get("total_tokens"),
            nodes_retrieved=nodes_retrieved,
            edges_traversed=edges_traversed
        )

        if verbose:
            print(f"\n✓ Total time: {total_time:.0f}ms")
            print(f"  - Retrieval: {retrieval_time:.0f}ms")
            print(f"  - Reasoning: {reasoning_time:.0f}ms")
            print(f"  - Synthesis: {synthesis_time:.0f}ms")
            if result.tokens_used:
                print(f"  - Tokens: {result.tokens_used}")

        return result

    def _execute_retrieval(
        self,
        plan: QueryPlan,
        query: str,
        verbose: bool
    ) -> list:
        """Execute retrieval stage.

        Args:
            plan: Query plan
            query: User query
            verbose: Print progress

        Returns:
            List of candidate node IDs
        """
        # For connection path queries, use entities directly
        if plan.intent.query_type == QueryType.CONNECTION_PATH:
            if plan.intent.source_entity:
                return [plan.intent.source_entity]

        # For queries with explicit entities, try to use them directly
        if plan.intent.entities and plan.intent.query_type in [
            QueryType.AUTHOR_PAPERS,
            QueryType.COLLABORATOR_PAPERS,
            QueryType.AUTHOR_NETWORK
        ]:
            # Return entity names directly (they're likely author names)
            return plan.intent.entities[:plan.intent.top_k]

        # Otherwise, use vector retrieval
        candidate_ids = self._retriever.retrieve(
            query=query,
            top_k=plan.intent.top_k,
            filters=plan.retrieval_params.get("filters")
        )

        return candidate_ids

    def _execute_reasoning(
        self,
        plan: QueryPlan,
        candidate_ids: list,
        verbose: bool
    ) -> Dict[str, Any]:
        """Execute reasoning stage.

        Args:
            plan: Query plan
            candidate_ids: Initial candidate IDs
            verbose: Print progress

        Returns:
            Context dictionary
        """
        context = self._reasoner.expand_context(
            candidate_ids=candidate_ids,
            strategy_name=plan.reasoning_strategy,
            **plan.reasoning_params
        )

        return context

    def _execute_synthesis(
        self,
        plan: QueryPlan,
        query: str,
        context: Dict[str, Any],
        verbose: bool
    ) -> tuple:
        """Execute synthesis stage.

        Args:
            plan: Query plan
            query: User query
            context: Retrieved context
            verbose: Print progress

        Returns:
            Tuple of (answer, metadata)
        """
        # Format context for LLM
        context_text = self._reasoner.format_context_for_llm(context)

        # Get appropriate prompt template
        if plan.synthesis_prompt_template:
            try:
                prompt = PromptTemplates.get(plan.synthesis_prompt_template)
                self._synthesizer.set_system_prompt(prompt)
            except (ValueError, AttributeError):
                pass  # Use default prompt

        # Generate answer
        result = self._synthesizer.synthesize_with_metadata(
            query=query,
            context=context_text,
            **plan.synthesis_config
        )

        return result["answer"], result["metadata"]

    def execute_simple(
        self,
        query: str,
        top_k: int = 10,
        use_reasoning: bool = True,
        verbose: bool = False
    ) -> QueryResult:
        """Execute query with default settings (no explicit plan).

        Args:
            query: User query
            top_k: Number of results
            use_reasoning: Whether to use multi-hop reasoning
            verbose: Print progress

        Returns:
            QueryResult

        Example:
            result = executor.execute_simple(
                "What papers discuss transformers?",
                top_k=5
            )
        """
        from deepgraph.planning.dsl import create_simple_plan, QueryType

        # Infer query type from keywords
        query_lower = query.lower()
        if "collaborat" in query_lower or "co-author" in query_lower:
            query_type = QueryType.COLLABORATOR_PAPERS
        elif "by" in query_lower or "written" in query_lower or "authored" in query_lower:
            query_type = QueryType.AUTHOR_PAPERS
        elif "connect" in query_lower or "relationship" in query_lower:
            query_type = QueryType.CONNECTION_PATH
        elif "compare" in query_lower:
            query_type = QueryType.COMPARISON
        else:
            query_type = QueryType.SIMILARITY

        # Create simple plan
        plan = create_simple_plan(query, query_type, top_k=top_k)

        # Disable reasoning if requested
        if not use_reasoning:
            plan.reasoning_strategy = None

        return self.execute(plan, query, verbose)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QueryExecutor("
            f"retriever={self._retriever}, "
            f"reasoner={self._reasoner}, "
            f"synthesizer={self._synthesizer})"
        )
