# CLAUDE.md

Instructions for Claude Code when working with this repository.

## What This Project Does

DeepGraph RAG demonstrates **intersection queries** using knowledge graphs. The core value:

- Vector search: "Find things about X"
- Graph RAG: "Find things that HAVE X FOR Y"

Three demos apply the same pattern to different domains:
- **Research** (arXiv): Method x Concept
- **Products** (electronics): Feature x UseCase x Price
- **Neurology** (medical): Research vs Patient symptom gaps

## Quick Commands

```bash
# Start Neo4j
docker-compose up -d

# Run demos (each on its own port)
streamlit run landing.py --server.port 8500
streamlit run examples/arxiv/app.py --server.port 8505
streamlit run examples/products/app.py --server.port 8506
streamlit run examples/neurology/app.py --server.port 8507

# Neo4j Browser
open http://localhost:7474
# Credentials: neo4j / deepgraph2025
```

## Key Files

| File | Purpose |
|------|---------|
| `landing.py` | Main landing page |
| `examples/arxiv/app.py` | Research Navigator (Method x Concept) |
| `examples/products/app.py` | Product Navigator (Feature x UseCase) |
| `examples/neurology/app.py` | Neurology Navigator (Research vs Patient) |
| `deepgraph/core/schema.py` | Domain-agnostic schema definitions |

## Graph Schema Pattern

All demos follow the same structure:

```
Entity ──→ HAS_CAPABILITY ──→ Capability
  └──→ FOR_INTENT ──→ Intent
```

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Research | Paper | Method | Concept |
| Products | Product | Feature | UseCase |
| Neurology | Paper/RedditPost | Symptom | Disease |

## Neo4j Connection

All scripts use:
- URI: `bolt://localhost:7687`
- User: `neo4j`
- Password: `deepgraph2025`

These are set in `docker-compose.yml`.

## Common Tasks

### Adding a new query to a demo

1. Write and test the Cypher query in Neo4j Browser first
2. Add a function in the relevant `app.py`
3. Add UI in the appropriate tab

### Rebuilding data

```bash
# Research data
python scripts/01_download_arxiv.py --max-papers 500
python scripts/02_convert_arxiv_simple.py
python scripts/03_build_graph.py
python scripts/04_create_indexes.py
python scripts/06_extract_concepts.py

# Products data
python examples/products/scripts/build_product_graph.py

# Neurology data
python examples/neurology/scripts/build_neurology_graph.py
```

## Architecture Notes

- Streamlit apps query Neo4j directly (no separate backend needed for demos)
- Each demo is self-contained in its `examples/` subdirectory
- The `deepgraph/` library provides reusable schema definitions
- Pyvis is used for graph visualizations
