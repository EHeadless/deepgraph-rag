"""
Graph reasoning and traversal.

This module provides the main GraphReasoner class that orchestrates
multi-hop graph traversal using different reasoning strategies.
"""

from typing import List, Dict, Any, Optional
from deepgraph.store.base import GraphStore
from deepgraph.reasoning.strategies import (
    ReasoningStrategy,
    AuthorPapersStrategy,
    CollaboratorPapersStrategy,
    ShortestPathStrategy,
    SubgraphStrategy
)


class GraphReasoner:
    """Multi-hop graph reasoning orchestrator.

    This class executes reasoning strategies to expand context from
    initial candidate nodes through multi-hop graph traversal.

    Example:
        from deepgraph.store import Neo4jGraphStore
        from deepgraph.reasoning import GraphReasoner, AuthorPapersStrategy

        store = Neo4jGraphStore()
        store.connect("bolt://localhost:7687", user="neo4j", password="...")

        reasoner = GraphReasoner(store)

        # Expand context using author papers strategy
        context = reasoner.expand_context(
            candidate_ids=["2301.12345", "2301.12346"],
            strategy=AuthorPapersStrategy()
        )

        print(f"Found {len(context['papers'])} related papers")
        print(f"Through {len(context['authors'])} authors")
    """

    # Built-in strategies
    STRATEGIES = {
        "author_papers": AuthorPapersStrategy(),
        "collaborator_papers": CollaboratorPapersStrategy(),
        "shortest_path": ShortestPathStrategy(),
        "subgraph": SubgraphStrategy()
    }

    def __init__(self, store: GraphStore):
        """Initialize graph reasoner.

        Args:
            store: Graph store instance
        """
        self._store = store

    def expand_context(
        self,
        candidate_ids: List[str],
        strategy: Optional[ReasoningStrategy] = None,
        strategy_name: Optional[str] = None,
        **strategy_kwargs
    ) -> Dict[str, Any]:
        """Expand context using a reasoning strategy.

        Args:
            candidate_ids: Initial candidate node IDs
            strategy: Strategy instance (takes precedence over strategy_name)
            strategy_name: Name of built-in strategy (e.g., "author_papers")
            **strategy_kwargs: Additional arguments for the strategy

        Returns:
            Context dictionary with reasoning results

        Raises:
            ValueError: If neither strategy nor strategy_name is provided

        Example:
            # Using strategy instance
            context = reasoner.expand_context(
                candidate_ids=["paper1", "paper2"],
                strategy=AuthorPapersStrategy(),
                max_papers_per_author=5
            )

            # Using built-in strategy name
            context = reasoner.expand_context(
                candidate_ids=["paper1"],
                strategy_name="author_papers"
            )
        """
        # Determine which strategy to use
        if strategy is not None:
            active_strategy = strategy
        elif strategy_name is not None:
            if strategy_name not in self.STRATEGIES:
                raise ValueError(
                    f"Unknown strategy: {strategy_name}. "
                    f"Choose from {list(self.STRATEGIES.keys())}"
                )
            active_strategy = self.STRATEGIES[strategy_name]
        else:
            raise ValueError("Must provide either strategy or strategy_name")

        # Execute strategy
        result = active_strategy.execute(
            store=self._store,
            candidate_ids=candidate_ids,
            **strategy_kwargs
        )

        # Add metadata
        result["_meta"] = {
            "strategy": active_strategy.name,
            "description": active_strategy.description,
            "num_candidates": len(candidate_ids),
            "candidate_ids": candidate_ids
        }

        return result

    def multi_hop(
        self,
        candidate_ids: List[str],
        strategies: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute multiple strategies sequentially.

        Args:
            candidate_ids: Initial candidate node IDs
            strategies: List of strategy names to execute
            **kwargs: Arguments for strategies

        Returns:
            Combined context from all strategies

        Example:
            # Execute both author_papers and collaborator_papers
            context = reasoner.multi_hop(
                candidate_ids=["paper1"],
                strategies=["author_papers", "collaborator_papers"]
            )
        """
        combined_result = {
            "strategies_executed": [],
            "papers": [],
            "authors": [],
            "connections": []
        }

        for strategy_name in strategies:
            result = self.expand_context(
                candidate_ids=candidate_ids,
                strategy_name=strategy_name,
                **kwargs
            )

            combined_result["strategies_executed"].append(strategy_name)

            # Merge results (avoiding duplicates)
            if "papers" in result:
                for paper in result["papers"]:
                    if paper not in combined_result["papers"]:
                        combined_result["papers"].append(paper)

            if "authors" in result:
                for author in result["authors"]:
                    if author not in combined_result["authors"]:
                        combined_result["authors"].append(author)

            if "connections" in result:
                combined_result["connections"].extend(result["connections"])

        return combined_result

    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format reasoning context as text for LLM consumption.

        Args:
            context: Context dictionary from expand_context()

        Returns:
            Formatted text string

        Example:
            context = reasoner.expand_context(...)
            text = reasoner.format_context_for_llm(context)
            # Pass text to LLM for answer generation
        """
        lines = []

        # Add metadata
        if "_meta" in context:
            meta = context["_meta"]
            lines.append(f"Reasoning Strategy: {meta['strategy']}")
            if "description" in meta:
                lines.append(f"Description: {meta['description']}")
            lines.append("")

        # Add papers
        if "papers" in context and context["papers"]:
            lines.append(f"## Relevant Papers ({len(context['papers'])} found)")
            lines.append("")
            for i, paper in enumerate(context["papers"][:20], 1):  # Limit to 20
                title = paper.get("title", "Unknown")
                arxiv_id = paper.get("arxiv_id", "")
                abstract = paper.get("abstract", "")[:200]  # First 200 chars
                lines.append(f"{i}. **{title}** ({arxiv_id})")
                if abstract:
                    lines.append(f"   {abstract}...")
                lines.append("")

        # Add authors
        if "authors" in context and context["authors"]:
            lines.append(f"## Authors ({len(context['authors'])} found)")
            lines.append("")
            for author in context["authors"][:10]:  # Limit to 10
                name = author.get("name", "Unknown")
                institution = author.get("institution", "")
                if institution:
                    lines.append(f"- {name} ({institution})")
                else:
                    lines.append(f"- {name}")
            lines.append("")

        # Add connections/relationships
        if "connections" in context and context["connections"]:
            lines.append(f"## Relationships ({len(context['connections'])} found)")
            lines.append("")
            for conn in context["connections"][:15]:  # Limit to 15
                if "source_paper" in conn:
                    lines.append(
                        f"- Paper {conn['source_paper']} → "
                        f"via author {conn['via_author']} → "
                        f"Paper {conn['related_paper']}"
                    )
                elif "author" in conn:
                    lines.append(
                        f"- {conn['author']} → "
                        f"via collaborator {conn['via_collaborator']} → "
                        f"Paper {conn['recommended_paper']}"
                    )
            lines.append("")

        # Add path (if present)
        if "path" in context and context["path"]:
            path_info = context["path"]
            lines.append(f"## Connection Path")
            lines.append(f"Length: {path_info.get('length', 0)} hops")
            lines.append("")

        return "\n".join(lines)

    def register_strategy(self, name: str, strategy: ReasoningStrategy) -> None:
        """Register a custom reasoning strategy.

        Args:
            name: Strategy name
            strategy: Strategy instance

        Example:
            class MyCustomStrategy:
                def execute(self, store, candidate_ids, **kwargs):
                    # Custom logic...
                    return {...}

            reasoner.register_strategy("my_strategy", MyCustomStrategy())
            context = reasoner.expand_context(
                candidate_ids=["..."],
                strategy_name="my_strategy"
            )
        """
        self.STRATEGIES[name] = strategy

    @property
    def available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.STRATEGIES.keys())

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GraphReasoner("
            f"strategies={len(self.STRATEGIES)}, "
            f"available={self.available_strategies})"
        )
