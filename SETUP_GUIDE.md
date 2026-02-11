# Setup Guide

## Prerequisites

- Python 3.11+
- Docker
- OpenAI API key

## Quick Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/deepgraph-rag.git
cd deepgraph-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Start Neo4j
docker-compose up -d
# Wait 30 seconds for initialization

# 5. Verify Neo4j is running
open http://localhost:7474
# Login: neo4j / deepgraph2025
```

## Building Demo Data

### Research Demo (arXiv)

```bash
python scripts/01_download_arxiv.py --max-papers 500
python scripts/02_convert_arxiv_simple.py
python scripts/03_build_graph.py
python scripts/04_create_indexes.py
python scripts/06_extract_concepts.py
```

### Products Demo

```bash
python examples/products/scripts/build_product_graph.py
```

### Neurology Demo

```bash
python examples/neurology/scripts/build_neurology_graph.py
```

## Running the Demos

```bash
# Landing page
streamlit run landing.py --server.port 8500

# Research Navigator
streamlit run examples/arxiv/app.py --server.port 8505

# Product Navigator
streamlit run examples/products/app.py --server.port 8506

# Neurology Navigator
streamlit run examples/neurology/app.py --server.port 8507
```

## Ports

| Service | Port | URL |
|---------|------|-----|
| Landing | 8500 | http://localhost:8500 |
| Research | 8505 | http://localhost:8505 |
| Products | 8506 | http://localhost:8506 |
| Neurology | 8507 | http://localhost:8507 |
| Neo4j Browser | 7474 | http://localhost:7474 |
| Neo4j Bolt | 7687 | bolt://localhost:7687 |

## Troubleshooting

### Neo4j won't start
```bash
docker logs deepgraph-neo4j
# Check if ports 7474/7687 are in use
lsof -i :7474
```

### No data in demos
```bash
# Verify data exists in Neo4j Browser
MATCH (n) RETURN labels(n), count(*)
```

### OpenAI errors
- Verify OPENAI_API_KEY in .env
- Check API quota at platform.openai.com
