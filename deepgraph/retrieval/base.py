"""
Base retrieval protocol.

This module defines the interface for candidate retrieval from the graph,
enabling different retrieval strategies (vector, keyword, hybrid).
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class CandidateRetriever(Protocol):
    """Protocol defining the interface for candidate retrieval.

    Retrievers are responsible for finding the initial set of relevant
    nodes from the graph based on a query.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Retrieve candidate node IDs relevant to the query.

        Args:
            query: User query text
            top_k: Number of candidates to retrieve
            filters: Optional filters on node properties

        Returns:
            List of node IDs (in ranked order)

        Example:
            candidate_ids = retriever.retrieve(
                "What papers discuss transformers?",
                top_k=10
            )
            # Returns: ["2301.12345", "2301.12346", ...]
        """
        ...

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates with relevance scores.

        Args:
            query: User query text
            top_k: Number of candidates to retrieve
            filters: Optional filters on node properties

        Returns:
            List of dicts with 'id' and 'score' keys

        Example:
            results = retriever.retrieve_with_scores(
                "What papers discuss transformers?",
                top_k=10
            )
            # Returns: [{"id": "2301.12345", "score": 0.95}, ...]
        """
        ...
