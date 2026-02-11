"""
ArXiv research paper domain schema.

This module defines the graph schema for academic research papers from arXiv,
including papers, authors, concepts, and their relationships.
"""

from deepgraph.core.schema import NodeSchema, EdgeSchema, GraphSchema


# ============= NODE SCHEMAS =============

PAPER_SCHEMA = NodeSchema(
    label="Paper",
    id_field="arxiv_id",
    properties={
        "arxiv_id": str,
        "title": str,
        "abstract": str,
        "published_date": str,
        "primary_category": str,
        "embedding": list  # 1536-dim vector from text-embedding-ada-002
    },
    indexes=["published_date", "primary_category"],
    constraints=["arxiv_id"],
    vector_config={
        "field": "embedding",
        "dimensions": 1536,
        "similarity": "cosine",
        "index_name": "paper_embedding"
    }
)


AUTHOR_SCHEMA = NodeSchema(
    label="Author",
    id_field="name",
    properties={
        "name": str,
        "institution": str,  # Optional, may not always be available
    },
    constraints=["name"]
)


CONCEPT_SCHEMA = NodeSchema(
    label="Concept",
    id_field="name",
    properties={
        "name": str,
        "field": str  # AI, ML, CV, NLP, etc.
    },
    constraints=["name"]
)


METHOD_SCHEMA = NodeSchema(
    label="Method",
    id_field="name",
    properties={
        "name": str,
        "category": str  # Architecture, Training, Optimization, etc.
    },
    constraints=["name"]
)


# ============= EDGE SCHEMAS =============

AUTHORED_BY = EdgeSchema(
    type="AUTHORED_BY",
    from_label="Paper",
    to_label="Author",
    properties={
        "position": int  # Author position in paper (0-indexed)
    }
)


CO_AUTHORED = EdgeSchema(
    type="CO_AUTHORED",
    from_label="Author",
    to_label="Author",
    properties={
        "shared_papers": int  # Number of papers co-authored
    }
)


ABOUT_CONCEPT = EdgeSchema(
    type="ABOUT_CONCEPT",
    from_label="Paper",
    to_label="Concept"
)


USES_METHOD = EdgeSchema(
    type="USES_METHOD",
    from_label="Paper",
    to_label="Method"
)


CITES = EdgeSchema(
    type="CITES",
    from_label="Paper",
    to_label="Paper",
    properties={
        "citation_context": str  # Optional context where citation appears
    }
)


# ============= COMPLETE SCHEMA =============

ARXIV_SCHEMA = GraphSchema(
    name="arxiv_research",
    nodes={
        "Paper": PAPER_SCHEMA,
        "Author": AUTHOR_SCHEMA,
        "Concept": CONCEPT_SCHEMA,
        "Method": METHOD_SCHEMA
    },
    edges={
        "AUTHORED_BY": AUTHORED_BY,
        "CO_AUTHORED": CO_AUTHORED,
        "ABOUT_CONCEPT": ABOUT_CONCEPT,
        "USES_METHOD": USES_METHOD,
        "CITES": CITES
    },
    metadata={
        "description": "Graph schema for academic research papers from arXiv",
        "version": "1.0",
        "domains": ["AI", "ML", "CS"],
        "embedding_model": "text-embedding-ada-002",
        "embedding_dimensions": 1536
    }
)


# ============= HELPER FUNCTIONS =============

def create_paper_node(arxiv_id: str, title: str, published_date: str,
                     primary_category: str, abstract: str = "",
                     embedding: list = None) -> dict:
    """Create a paper node dictionary.

    Args:
        arxiv_id: arXiv ID (e.g., "2301.12345")
        title: Paper title
        published_date: Publication date (YYYY-MM-DD format)
        primary_category: Primary arXiv category (e.g., "cs.AI")
        abstract: Paper abstract (optional)
        embedding: 1536-dim embedding vector (optional)

    Returns:
        Dictionary with paper properties
    """
    node = {
        "arxiv_id": arxiv_id,
        "title": title,
        "published_date": published_date,
        "primary_category": primary_category
    }

    if abstract:
        node["abstract"] = abstract
    if embedding:
        node["embedding"] = embedding

    return node


def create_author_node(name: str, institution: str = None) -> dict:
    """Create an author node dictionary.

    Args:
        name: Author name
        institution: Institutional affiliation (optional)

    Returns:
        Dictionary with author properties
    """
    node = {"name": name}
    if institution:
        node["institution"] = institution

    return node


def create_concept_node(name: str, field: str = "CS") -> dict:
    """Create a concept node dictionary.

    Args:
        name: Concept name (e.g., "Transformer", "Attention Mechanism")
        field: Research field (default: "CS")

    Returns:
        Dictionary with concept properties
    """
    return {
        "name": name,
        "field": field
    }


def create_method_node(name: str, category: str = "Architecture") -> dict:
    """Create a method node dictionary.

    Args:
        name: Method name (e.g., "BERT", "ResNet")
        category: Method category (default: "Architecture")

    Returns:
        Dictionary with method properties
    """
    return {
        "name": name,
        "category": category
    }


# Example usage:
if __name__ == "__main__":
    # Print schema summary
    print(ARXIV_SCHEMA)
    print("\nNode types:", list(ARXIV_SCHEMA.nodes.keys()))
    print("Edge types:", list(ARXIV_SCHEMA.edges.keys()))

    # Generate Neo4j DDL
    print("\nNeo4j DDL statements:")
    for statement in ARXIV_SCHEMA.to_cypher_ddl():
        print(f"  {statement}")

    # Validate sample data
    paper = create_paper_node(
        arxiv_id="2301.12345",
        title="Attention Is All You Need",
        published_date="2017-06-12",
        primary_category="cs.CL"
    )

    is_valid = ARXIV_SCHEMA.validate_node("Paper", paper)
    print(f"\nSample paper node valid: {is_valid}")
