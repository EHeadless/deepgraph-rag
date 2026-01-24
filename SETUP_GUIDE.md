# DeepGraph RAG - Complete Setup Guide

## 🚀 Quick Start (15 minutes)

### Prerequisites
```bash
# Required
- Python 3.10+
- Docker & Docker Compose
- 16GB RAM minimum
- OpenAI or Anthropic API key

# Optional but recommended
- GPU for faster embeddings
- 50GB disk space for full dataset
```

### Installation

**Step 1: Clone and Setup**
```bash
# Navigate to project
cd deepgraph-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor

# Required settings:
# - OPENAI_API_KEY or ANTHROPIC_API_KEY
# - NEO4J_URI (default: bolt://localhost:7687)
# - NEO4J_PASSWORD (default: deepgraph2025)
```

**Step 3: Start Neo4j**
```bash
# Start Neo4j with Docker
docker-compose up -d

# Wait for Neo4j to be ready (~30 seconds)
docker logs deepgraph-neo4j

# Access Neo4j Browser: http://localhost:7474
# Username: neo4j
# Password: deepgraph2025
```

**Step 4: Test Installation**
```bash
# Quick test
python -c "from deepgraph_rag import DeepGraphRAG; print('✓ Import successful!')"
```

## 📊 Building the Knowledge Graph

### Option A: Quick Demo (1K papers, ~30 minutes)

```bash
# 1. Download papers
python scripts/01_download_arxiv.py \
    --categories cs.AI \
    --max-papers 1000 \
    --output data/raw

# 2. Extract entities (requires API key)
python scripts/02_extract_entities.py \
    --input data/raw \
    --output data/processed \
    --max-papers 1000 \
    --workers 4

# 3. Build graph
python scripts/03_build_graph.py \
    --input data/processed \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password deepgraph2025

# 4. Create indexes and embeddings
python scripts/04_create_indexes.py \
    --neo4j-uri bolt://localhost:7687 \
    --embedding-model text-embedding-3-large

# 5. Test!
python demo.py
```

### Option B: Full Dataset (100K papers, ~2 days)

```bash
# Download full dataset
python scripts/01_download_arxiv.py \
    --categories cs.AI,cs.LG,cs.CL,cs.CV \
    --max-papers 100000 \
    --date-from 2020-01-01 \
    --output data/raw

# Process in batches (this will take ~40 hours with GPT-4)
# Consider running overnight
python scripts/02_extract_entities.py \
    --input data/raw \
    --output data/processed \
    --workers 4 \
    --llm-model gpt-4-turbo-preview

# Build graph (takes ~2 hours)
python scripts/03_build_graph.py \
    --input data/processed \
    --batch-size 500

# Create indexes (takes ~3 hours)
python scripts/04_create_indexes.py \
    --batch-size 100
```

## 💡 Usage Examples

### Python API

```python
from deepgraph_rag import DeepGraphRAG

# Initialize
rag = DeepGraphRAG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="deepgraph2025"
)

# Simple query
response = rag.query("What are transformer architectures?")
print(response.answer)

# Complex multi-hop query with explanation
response = rag.query(
    "Papers by PhD students of Turing Award winners on deep learning",
    explain=True
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2%}")
print(f"Sources: {len(response.sources)}")

# Show reasoning path
for step in response.reasoning_path:
    print(f"  Step {step['iteration']}: {step['action']}")

# Access sources
for source in response.sources[:3]:
    print(f"- {source['title']}")
    print(f"  Authors: {', '.join(source['authors'])}")
    print(f"  Method: {source['retrieval_method']}")
```

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/04_demo.ipynb
```

### Command Line Demo

```bash
# Run interactive demo
python demo.py
```

## 🎯 Use Cases

### 1. Literature Review
```python
# Find papers on specific topic with relationships
response = rag.query(
    "Recent papers on self-attention mechanisms"
    "by researchers from top 10 universities"
)
```

### 2. Research Lineage
```python
# Trace academic lineage
response = rag.query(
    "Papers by students of Geoffrey Hinton on deep learning"
)
```

### 3. Cross-Institution Research
```python
# Find collaborative research
response = rag.query(
    "Papers where MIT researchers cited Stanford work on transformers"
)
```

### 4. Method Propagation
```python
# Track how methods spread
response = rag.query(
    "Which recent papers use ResNet variants from original authors?"
)
```

## 🔧 Advanced Configuration

### Custom Retrieval Strategy

```python
from deepgraph_rag import DeepGraphConfig

# Configure for vector-only retrieval
config = DeepGraphConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="deepgraph2025",
    retrieval_strategy="vector",  # or "graph" or "hybrid"
    vector_top_k=20,
    enable_verification=True
)

rag = DeepGraphRAG(config=config)
```

### Using Different LLM

```python
# Use Claude instead of GPT-4
config = DeepGraphConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="deepgraph2025",
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514"
)

rag = DeepGraphRAG(config=config)
```

### Custom Graph Queries

```python
# Execute custom Cypher
from deepgraph_rag.retrieval import GraphRetriever

retriever = GraphRetriever(driver=rag.driver)

results = retriever.execute_custom_query(
    cypher="""
    MATCH (a:Author)-[:AUTHORED_BY]-(p:Paper)-[:CITES]->(cited:Paper)
    WHERE a.name = $author_name
    RETURN cited.title as title, count(*) as citations
    ORDER BY citations DESC
    LIMIT 10
    """,
    parameters={"author_name": "Geoffrey Hinton"}
)
```

## 🐛 Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# View logs
docker logs deepgraph-neo4j

# Restart Neo4j
docker-compose restart

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Out of Memory

```python
# Reduce batch sizes
# In .env:
VECTOR_TOP_K=5  # Reduce from 10
GRAPH_MAX_HOPS=5  # Reduce from 10
```

### Slow Queries

```bash
# Check indexes exist
# In Neo4j Browser:
SHOW INDEXES

# If missing, recreate:
python scripts/04_create_indexes.py
```

### API Rate Limits

```python
# Add delays in extraction
# Edit scripts/02_extract_entities.py
# Add: time.sleep(1) between API calls
```

## 📈 Performance Optimization

### For Faster Queries
1. Reduce `vector_top_k` to 5
2. Limit `graph_max_hops` to 5
3. Enable caching in config
4. Use smaller embedding model

### For Better Accuracy
1. Increase `vector_top_k` to 20
2. Enable verification
3. Use GPT-4 over GPT-3.5
4. Add more data sources

### For Lower Costs
1. Use GPT-3.5-turbo for extraction
2. Cache embeddings
3. Use sentence-transformers locally
4. Reduce max_tokens in config

## 🔐 Security Notes

- **Never commit** `.env` file with API keys
- Use **read-only** Neo4j user for production
- Implement **rate limiting** for API endpoints
- **Validate inputs** before graph queries

## 📚 Additional Resources

- **Neo4j Documentation**: https://neo4j.com/docs/
- **LangChain Docs**: https://python.langchain.com/
- **arXiv API**: https://arxiv.org/help/api/
- **Project Issues**: [GitHub Issues]

## 🆘 Getting Help

1. Check logs in `logs/` directory
2. Review error messages
3. Search GitHub Issues
4. Join Discord community
5. Open new issue with:
   - Error message
   - Steps to reproduce
   - Environment details

## 🚀 What's Next?

After setup, try:
1. Run `notebooks/04_demo.ipynb`
2. Experiment with different queries
3. Try custom graph patterns
4. Contribute improvements!

---

**Setup Time Estimate:**
- Installation: 5 minutes
- Demo dataset: 30 minutes
- Full dataset: 2 days

**Ready to build?** Start with the Quick Demo! 🎉
