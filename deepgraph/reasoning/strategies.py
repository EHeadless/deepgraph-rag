"""
Graph reasoning strategies.

This module defines specific multi-hop traversal strategies for extracting
context from the graph. Each strategy represents a different reasoning pattern.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from deepgraph.store.base import GraphStore


@runtime_checkable
class ReasoningStrategy(Protocol):
    """Protocol defining the interface for reasoning strategies.

    Strategies encapsulate specific multi-hop patterns for graph traversal.
    """

    def execute(
        self,
        store: GraphStore,
        candidate_ids: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the reasoning strategy.

        Args:
            store: Graph store instance
            candidate_ids: Initial candidate node IDs
            **kwargs: Strategy-specific parameters

        Returns:
            Context dictionary with reasoning results
        """
        ...

    @property
    def name(self) -> str:
        """Get strategy name."""
        ...

    @property
    def description(self) -> str:
        """Get strategy description."""
        ...


class AuthorPapersStrategy:
    """2-hop strategy: Paper → Authors → Their other papers.

    This strategy takes papers as input and expands to find all other
    papers written by the same authors.

    Example:
        Given: ["paper1"]
        Finds: paper1 → [author1, author2] → [paper2, paper3, paper4]
        Returns: Context with related papers and shared authors
    """

    @property
    def name(self) -> str:
        return "author_papers"

    @property
    def description(self) -> str:
        return "Find papers by the same authors (2-hop: Paper → Authors → Papers)"

    def execute(
        self,
        store: GraphStore,
        candidate_ids: List[str],
        max_papers_per_author: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute author papers strategy.

        Args:
            store: Graph store instance
            candidate_ids: Paper IDs to start from
            max_papers_per_author: Max papers to return per author

        Returns:
            Dictionary with 'papers', 'authors', and 'connections'
        """
        result = {
            "papers": [],
            "authors": [],
            "connections": []
        }

        for paper_id in candidate_ids:
            # Get paper
            paper = store.get_node("Paper", paper_id, "arxiv_id")
            if not paper:
                continue

            result["papers"].append(paper)

            # Get authors (1st hop)
            authors = store.get_neighbors(
                node_id=paper_id,
                label="Paper",
                id_field="arxiv_id",
                relationship_type="AUTHORED_BY",
                direction="outgoing"
            )

            # For each author, get their other papers (2nd hop)
            for author in authors:
                author_name = author.get("name")
                if not author_name:
                    continue

                if author not in result["authors"]:
                    result["authors"].append(author)

                # Get author's papers
                author_papers = store.get_neighbors(
                    node_id=author_name,
                    label="Author",
                    id_field="name",
                    relationship_type="AUTHORED_BY",
                    direction="incoming",
                    limit=max_papers_per_author
                )

                # Add papers we don't already have
                for other_paper in author_papers:
                    if other_paper.get("arxiv_id") != paper_id:
                        if other_paper not in result["papers"]:
                            result["papers"].append(other_paper)

                        result["connections"].append({
                            "source_paper": paper_id,
                            "via_author": author_name,
                            "related_paper": other_paper.get("arxiv_id")
                        })

        return result


class CollaboratorPapersStrategy:
    """3-hop strategy: Author → Papers → Co-authors → Their papers.

    This strategy takes authors as input and finds papers by their collaborators.

    Example:
        Given: ["Alice"]
        Finds: Alice → [paper1] → [Bob, Carol] → [paper2, paper3]
        Returns: Papers by Alice's collaborators
    """

    @property
    def name(self) -> str:
        return "collaborator_papers"

    @property
    def description(self) -> str:
        return "Find papers by collaborators (3-hop: Author → Papers → Co-authors → Papers)"

    def execute(
        self,
        store: GraphStore,
        candidate_ids: List[str],
        max_collaborators: int = 10,
        max_papers_per_collaborator: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute collaborator papers strategy.

        Args:
            store: Graph store instance
            candidate_ids: Author names to start from
            max_collaborators: Max collaborators to consider
            max_papers_per_collaborator: Max papers per collaborator

        Returns:
            Dictionary with 'collaborators', 'shared_papers', 'recommended_papers'
        """
        result = {
            "collaborators": [],
            "shared_papers": [],
            "recommended_papers": [],
            "connections": []
        }

        for author_name in candidate_ids:
            # Get author's papers (1st hop)
            author_papers = store.get_neighbors(
                node_id=author_name,
                label="Author",
                id_field="name",
                relationship_type="AUTHORED_BY",
                direction="incoming",
                limit=20
            )

            # For each paper, find co-authors (2nd hop)
            collaborators_seen = set()
            for paper in author_papers:
                paper_id = paper.get("arxiv_id")
                if not paper_id:
                    continue

                result["shared_papers"].append(paper)

                # Get co-authors
                coauthors = store.get_neighbors(
                    node_id=paper_id,
                    label="Paper",
                    id_field="arxiv_id",
                    relationship_type="AUTHORED_BY",
                    direction="outgoing",
                    limit=10
                )

                # For each co-author (excluding self), get their papers (3rd hop)
                for coauthor in coauthors:
                    coauthor_name = coauthor.get("name")
                    if not coauthor_name or coauthor_name == author_name:
                        continue

                    if coauthor_name not in collaborators_seen:
                        collaborators_seen.add(coauthor_name)
                        result["collaborators"].append(coauthor)

                        if len(result["collaborators"]) >= max_collaborators:
                            break

                        # Get collaborator's papers
                        collaborator_papers = store.get_neighbors(
                            node_id=coauthor_name,
                            label="Author",
                            id_field="name",
                            relationship_type="AUTHORED_BY",
                            direction="incoming",
                            limit=max_papers_per_collaborator
                        )

                        for collab_paper in collaborator_papers:
                            collab_paper_id = collab_paper.get("arxiv_id")
                            if collab_paper_id != paper_id:
                                result["recommended_papers"].append(collab_paper)
                                result["connections"].append({
                                    "author": author_name,
                                    "via_collaborator": coauthor_name,
                                    "shared_paper": paper_id,
                                    "recommended_paper": collab_paper_id
                                })

        return result


class ShortestPathStrategy:
    """N-hop strategy: Find shortest path between two entities.

    This strategy finds the shortest connection path between two nodes,
    useful for understanding relationships.

    Example:
        Given: author1="Alice", author2="Carol"
        Finds: Alice → paper1 → Bob → paper2 → Carol
        Returns: Path with nodes and relationships
    """

    @property
    def name(self) -> str:
        return "shortest_path"

    @property
    def description(self) -> str:
        return "Find shortest connection path between two entities"

    def execute(
        self,
        store: GraphStore,
        candidate_ids: List[str],
        target_id: str,
        from_label: str = "Author",
        to_label: str = "Author",
        max_depth: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute shortest path strategy.

        Args:
            store: Graph store instance
            candidate_ids: Source node IDs
            target_id: Target node ID
            from_label: Source node label
            to_label: Target node label
            max_depth: Maximum path length

        Returns:
            Dictionary with 'path', 'nodes', 'relationships', 'length'
        """
        if not candidate_ids:
            return {"path": None}

        source_id = candidate_ids[0]  # Use first candidate

        path = store.find_path(
            from_id=source_id,
            to_id=target_id,
            from_label=from_label,
            to_label=to_label,
            max_depth=max_depth
        )

        if path:
            return {
                "path": path,
                "source": source_id,
                "target": target_id,
                "nodes": path["nodes"],
                "relationships": path["relationships"],
                "length": path["length"]
            }
        else:
            return {
                "path": None,
                "message": f"No path found between {source_id} and {target_id}"
            }


class SubgraphStrategy:
    """Subgraph extraction strategy.

    This strategy extracts a local subgraph around the candidate nodes,
    including neighbors and relationships.

    Example:
        Given: ["paper1"]
        Finds: All directly connected nodes and edges
        Returns: Complete subgraph for visualization or analysis
    """

    @property
    def name(self) -> str:
        return "subgraph"

    @property
    def description(self) -> str:
        return "Extract local subgraph around candidates"

    def execute(
        self,
        store: GraphStore,
        candidate_ids: List[str],
        node_label: str = "Paper",
        id_field: str = "arxiv_id",
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute subgraph extraction.

        Args:
            store: Graph store instance
            candidate_ids: Center node IDs
            node_label: Label of center nodes
            id_field: ID field name
            depth: How many hops to include
            relationship_types: Relationship types to traverse

        Returns:
            Dictionary with 'nodes' and 'edges'
        """
        nodes = {}
        edges = []

        for node_id in candidate_ids:
            # Get center node
            node = store.get_node(node_label, node_id, id_field)
            if node:
                nodes[node_id] = node

                # Get neighbors
                neighbors = store.get_neighbors(
                    node_id=node_id,
                    label=node_label,
                    id_field=id_field,
                    relationship_type=relationship_types[0] if relationship_types else None,
                    direction="both",
                    limit=50
                )

                for neighbor in neighbors:
                    neighbor_id = neighbor.get("name") or neighbor.get("arxiv_id") or neighbor.get("id")
                    if neighbor_id and neighbor_id not in nodes:
                        nodes[neighbor_id] = neighbor

                    # Record edge
                    edges.append({
                        "from": node_id,
                        "to": neighbor_id
                    })

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
