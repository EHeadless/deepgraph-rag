"""
DeepGraph RAG - Quickstart Example

This minimal example shows how to use DeepGraph RAG with your own data.
Just 50 lines to get graph-enhanced retrieval!

Requirements:
    pip install deepgraph-rag
    # Neo4j running on localhost:7687

Usage:
    export OPENAI_API_KEY=sk-...
    export NEO4J_PASSWORD=password
    python simple_rag.py
"""

from deepgraph import GraphRAG

# Initialize with defaults
rag = GraphRAG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",  # Or set NEO4J_PASSWORD env var
    # openai_api_key="sk-..."   # Or set OPENAI_API_KEY env var
)

# Check graph stats
stats = rag.get_stats()
print(f"Graph contains: {stats}")

# Simple vector search
print("\n--- Vector Search ---")
results = rag.search("machine learning transformers", top_k=3)
for r in results:
    print(f"  - {r}")

# Graph-enhanced search (finds connected papers too!)
print("\n--- Graph-Enhanced Search ---")
results = rag.graph_search("machine learning transformers", top_k=3)
print(f"  Vector results: {len(results['vector_results'])}")
print(f"  Graph discoveries: {len(results['graph_discoveries'])}")

# Generate answer with RAG
print("\n--- RAG Answer ---")
answer = rag.answer(
    "What are the key innovations in transformer architectures?",
    top_k=5,
    use_graph=True  # Include graph-discovered papers in context
)
print(f"Answer: {answer}")

# Find path between authors
print("\n--- Path Finding ---")
path = rag.find_path(
    from_id="Yann LeCun",
    to_id="Geoffrey Hinton",
    from_label="Author",
    to_label="Author"
)
if path:
    print(f"  Path length: {path['length']}")
    print(f"  Nodes: {len(path['nodes'])}")

# Clean up
rag.close()
print("\nDone!")
