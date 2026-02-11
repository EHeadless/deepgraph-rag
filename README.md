# DeepGraph RAG

**Graph RAG for intersection queries. Find things that HAVE X FOR Y.**

Vector search answers: *"Find things about X"*
Graph RAG answers: *"Find things that have X for Y"*

---

## The Value

Graph RAG solves one specific problem: **intersection queries across structured relationships**.

| Query Type | Vector Search | Graph RAG |
|------------|---------------|-----------|
| "Papers about transformers" | Works | Works |
| "Papers USING transformers FOR reasoning" | Fails | Works |
| "Wireless headphones" | Works | Works |
| "Wireless headphones FOR running under $100" | Fails | Works |
| "Symptoms patients report but research ignores" | Impossible | Works |

The pattern is portable. The value depends on the domain.

---

## Three Demos

| Demo | Port | Killer Query | Data |
|------|------|--------------|------|
| **Research** | [8505](http://localhost:8505) | Method x Concept | 1,000 arXiv papers |
| **Products** | [8506](http://localhost:8506) | Feature x UseCase x Price | 500 electronics |
| **Neurology** | [8507](http://localhost:8507) | Research vs Patient gaps | 495 papers + 31K patient reports |

```bash
# Start all demos
streamlit run landing.py --server.port 8500 &
streamlit run examples/arxiv/app.py --server.port 8505 &
streamlit run examples/products/app.py --server.port 8506 &
streamlit run examples/neurology/app.py --server.port 8507 &
```

---

## The Pattern

All three demos use the same graph structure:

```
Entity ──→ HAS_CAPABILITY ──→ Capability
  └──→ FOR_INTENT ──→ Intent
```

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Research | Paper | Method | Concept |
| Products | Product | Feature | UseCase |
| Neurology | Paper/Post | Symptom | Disease |

The intersection query `Capability x Intent` is what vector search cannot do.

---

## Quick Start

```bash
# Prerequisites
# - Python 3.11+
# - Docker (for Neo4j)
# - OpenAI API key

# Clone and setup
git clone https://github.com/yourusername/deepgraph-rag.git
cd deepgraph-rag
pip install -r requirements.txt

# Configure
cp env.example .env
# Edit .env and add OPENAI_API_KEY

# Start Neo4j
docker-compose up -d
# Wait 30 seconds for Neo4j to initialize

# Build the graph (research demo)
python scripts/01_download_arxiv.py --max-papers 500
python scripts/02_convert_arxiv_simple.py
python scripts/03_build_graph.py
python scripts/04_create_indexes.py
python scripts/06_extract_concepts.py

# Run
streamlit run examples/arxiv/app.py --server.port 8505
```

---

## Project Structure

```
deepgraph-rag/
├── landing.py                 # Landing page
├── examples/
│   ├── arxiv/                 # Research Navigator
│   │   └── app.py             # Method x Concept queries
│   ├── products/              # Product Navigator
│   │   └── app.py             # Feature x UseCase queries
│   └── neurology/             # Neurology Navigator
│       └── app.py             # Research vs Patient comparison
├── deepgraph/                 # Core library
│   ├── core/schema.py         # Domain-agnostic schemas
│   ├── store/neo4j.py         # Neo4j abstraction
│   └── retrieval/vector.py    # Vector search
├── scripts/                   # Data pipelines
└── data/                      # Generated data (gitignored)
```

---

## Honest Assessment

| Demo | Differentiation | Notes |
|------|-----------------|-------|
| **Products** | High | Clear product-market fit. People shop by feature + use case + price. |
| **Neurology** | High | Unique dual-source capability. Research gap analysis is impossible otherwise. |
| **Research** | Medium | Researchers already have good tools (Semantic Scholar, Connected Papers). |

---

## Requirements

- Python 3.11+
- Neo4j 5.x (via Docker)
- OpenAI API key (for embeddings and entity extraction)

---

## License

MIT
