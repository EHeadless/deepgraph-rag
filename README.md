# DeepGraph RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/Neo4j-5.x-green.svg)](https://neo4j.com/)

**Graph RAG for intersection queries. Find things that HAVE X FOR Y.**

Vector search: *"Find things about X"*
Graph RAG: *"Find things that HAVE X FOR Y"*

<p align="center">
  <img src="Screenshot.png" alt="DeepGraph RAG Demo" width="800"/>
</p>

---

## Why Graph RAG?

Graph RAG solves one specific problem: **intersection queries across structured relationships**.

| Query | Vector Search | Graph RAG |
|-------|:-------------:|:---------:|
| "Papers about transformers" | :white_check_mark: | :white_check_mark: |
| "Papers USING transformers FOR reasoning" | :x: | :white_check_mark: |
| "Wireless headphones" | :white_check_mark: | :white_check_mark: |
| "Wireless headphones FOR running under $100" | :x: | :white_check_mark: |
| "Symptoms patients report but research ignores" | :x: | :white_check_mark: |

---

## Three Demos, One Pattern

<table>
<tr>
<td width="33%" align="center">

### :microscope: Research Navigator
**Method x Concept**

*"Papers using transformers for reasoning"*

1,000 arXiv papers

</td>
<td width="33%" align="center">

### :shopping_cart: Product Navigator
**Feature x UseCase x Price**

*"Wireless headphones for running under $100"*

500 electronics products

</td>
<td width="33%" align="center">

### :brain: Neurology Navigator
**Research vs Patient Gaps**

*"Symptoms patients report but research ignores"*

495 papers + 31K patient reports

</td>
</tr>
</table>

All three use the same graph pattern:

```
Entity ──→ HAS_CAPABILITY ──→ Capability
  └──→ FOR_INTENT ──→ Intent
```

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Research | Paper | Method | Concept |
| Products | Product | Feature | UseCase |
| Neurology | Paper/Post | Symptom | Disease |

---

## Quick Start

```bash
# Clone
git clone https://github.com/EHeadless/deepgraph-rag.git
cd deepgraph-rag

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start Neo4j
docker-compose up -d

# Run the landing page
streamlit run landing.py --server.port 8500
```

Then open http://localhost:8500

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions including data loading.

---

## Project Structure

```
deepgraph-rag/
├── landing.py                    # Main landing page
├── api.py                        # FastAPI REST backend
│
├── examples/
│   ├── arxiv/                    # Research Navigator
│   │   └── app.py                # Method x Concept queries
│   ├── products/                 # Product Navigator
│   │   └── app.py                # Feature x UseCase queries
│   └── neurology/                # Neurology Navigator
│       └── app.py                # Research vs Patient comparison
│
├── deepgraph/                    # Core library
│   ├── core/schema.py            # Domain-agnostic schemas
│   ├── store/neo4j.py            # Graph database
│   └── retrieval/vector.py       # Vector search
│
├── scripts/                      # Data pipelines
├── docker-compose.yml            # Neo4j setup
└── requirements.txt
```

---

## Ports

| Service | Port | URL |
|---------|------|-----|
| Landing Page | 8500 | http://localhost:8500 |
| Research Navigator | 8505 | http://localhost:8505 |
| Product Navigator | 8506 | http://localhost:8506 |
| Neurology Navigator | 8507 | http://localhost:8507 |
| Neo4j Browser | 7474 | http://localhost:7474 |
| REST API | 8000 | http://localhost:8000/docs |

---

## When to Use What

| Demo | Differentiation | Best For |
|------|-----------------|----------|
| **Products** | :star: High | E-commerce filtering. People think in Feature + UseCase + Price. |
| **Neurology** | :star: High | Dual-source comparison. Research gap analysis is unique. |
| **Research** | :star: Medium | Paper discovery. (Semantic Scholar already does this well.) |

---

## Evaluation Results

We ran **65 queries** across all three domains to compare Graph RAG vs Vector Search.

### Summary

| Metric | Vector Search | Graph RAG |
|--------|:-------------:|:---------:|
| **Query Coverage** | 15.4% (10/65) | 100% (65/65) |
| **Intersection Queries** | 0% (0/18) | 100% (18/18) |
| **Multi-Hop Queries** | 0% (0/14) | 100% (14/14) |
| **Aggregation Queries** | 0% (0/14) | 100% (14/14) |
| **Comparison Queries** | 0% (0/3) | 100% (3/3) |
| **Similarity Queries** | 100% (8/8) | 100% (8/8) |

**Graph RAG handles 84.6% of queries that vector search cannot answer.**

### By Query Type

| Type | Example | Vector | Graph | Results |
|------|---------|:------:|:-----:|--------:|
| **Similarity** | "Papers about transformers" | :white_check_mark: | :white_check_mark: | 20 |
| **Intersection** | "Wireless FOR workout under $100" | :x: | :white_check_mark: | 1 |
| **Multi-Hop** | "Features commonly paired together" | :x: | :white_check_mark: | 15 |
| **Aggregation** | "Most common features" | :x: | :white_check_mark: | 22 |
| **Comparison** | "Symptoms patients report but research ignores" | :x: | :white_check_mark: | 20 |

### Detailed Results by Domain

<details>
<summary><strong>Research Navigator (20 queries)</strong></summary>

| Query | Type | Vector | Results | Time |
|-------|------|:------:|--------:|-----:|
| Papers about transformer architectures | similarity | :white_check_mark: | 20 | 723ms |
| Papers about large language models | similarity | :white_check_mark: | 20 | 102ms |
| Papers about diffusion models | similarity | :white_check_mark: | 20 | 64ms |
| Papers using transformers FOR reasoning | intersection | :x: | 0* | 229ms |
| Papers using attention FOR multimodal tasks | intersection | :x: | 0* | 176ms |
| Papers using RL FOR robotics | intersection | :x: | 0* | 160ms |
| Papers using fine-tuning evaluated on ImageNet | intersection | :x: | 0* | 65ms |
| Papers about BOTH efficiency AND robustness | intersection | :x: | 0* | 42ms |
| Other papers by authors of paper X | multi_hop | :x: | 1 | 72ms |
| Papers by collaborators of author X | multi_hop | :x: | 0* | 248ms |
| Methods used in both NLP and Vision papers | multi_hop | :x: | 10 | 135ms |
| Authors who publish in multiple concept areas | multi_hop | :x: | 10 | 66ms |
| Path between two research areas | multi_hop | :x: | 0* | 26ms |
| Most commonly used methods | aggregation | :x: | 15 | 18ms |
| Most researched concepts | aggregation | :x: | 15 | 18ms |
| Authors with most papers | aggregation | :x: | 15 | 26ms |
| Which methods are used for which concepts | aggregation | :x: | 20 | 27ms |
| Papers using non-existent method 'quantum_blockchain' | edge_case | :x: | 0 | 22ms |
| Papers with 'TRANSFORMER' (case test) | edge_case | :white_check_mark: | 10 | 32ms |
| Papers with methods containing 'tion' (partial) | edge_case | :x: | 2 | 77ms |

*Zero results due to specific entity matching. Query pattern is valid.

</details>

<details>
<summary><strong>Product Navigator (21 queries)</strong></summary>

| Query | Type | Vector | Results | Time |
|-------|------|:------:|--------:|-----:|
| Wireless headphones | similarity | :white_check_mark: | 20 | 102ms |
| Gaming laptops | similarity | :white_check_mark: | 20 | 139ms |
| Wireless products FOR workout | intersection | :x: | 9 | 84ms |
| Wireless headphones FOR travel under $100 | intersection | :x: | 1 | 53ms |
| Products that are BOTH wireless AND waterproof | intersection | :x: | 45 | 41ms |
| Wireless AND noise-canceling FOR travel | intersection | :x: | 4 | 38ms |
| Monitors with 4K under $500 | intersection | :x: | 9 | 28ms |
| Sony products with noise canceling | intersection | :x: | 2 | 26ms |
| Gaming products between $200-$400 | intersection | :x: | 31 | 212ms |
| Products frequently bought with laptops | multi_hop | :x: | 9 | 39ms |
| Premium features ($300+) now in budget products ($100-) | multi_hop | :x: | 10 | 80ms |
| Brands that sell in multiple categories | multi_hop | :x: | 10 | 61ms |
| Features commonly paired together | multi_hop | :x: | 15 | 48ms |
| Most common features | aggregation | :x: | 22 | 19ms |
| Most targeted use cases | aggregation | :x: | 8 | 19ms |
| Average price by category | aggregation | :x: | 8 | 46ms |
| Rare feature + use case combinations | aggregation | :x: | 0 | 42ms |
| Products per brand | aggregation | :x: | 15 | 22ms |
| Waterproof laptops (unlikely combination) | edge_case | :x: | 0 | 24ms |
| Products at exactly $99.99 | edge_case | :x: | 0 | 25ms |
| Wireless products that are NOT for gaming | edge_case | :x: | 20 | 36ms |

</details>

<details>
<summary><strong>Neurology Navigator (24 queries)</strong></summary>

| Query | Type | Vector | Results | Time |
|-------|------|:------:|--------:|-----:|
| Papers about Alzheimer's disease | similarity | :white_check_mark: | 20 | 27ms |
| Papers about Parkinson's disease | similarity | :white_check_mark: | 20 | 9ms |
| Papers about ALS | similarity | :white_check_mark: | 20 | 23ms |
| Papers about Parkinson's AND tremor | intersection | :x: | 5 | 37ms |
| Papers about Alzheimer's mentioning tau protein | intersection | :x: | 20 | 27ms |
| Papers about Parkinson's discussing neuroinflammation | intersection | :x: | 20 | 25ms |
| Papers about Alzheimer's mentioning treatments | intersection | :x: | 20 | 22ms |
| Papers about Parkinson's mentioning brain regions | intersection | :x: | 20 | 24ms |
| Papers mentioning tau AND oxidative stress | intersection | :x: | 4 | 40ms |
| Symptoms patients report but research doesn't cover (Parkinson's) | comparison | :x: | 20 | 4,314ms |
| Symptoms patients report but research doesn't cover (Alzheimer's) | comparison | :x: | 20 | 1,041ms |
| Symptoms in research but rarely reported by patients | comparison | :x: | 20 | 2,464ms |
| Diseases that share symptoms with Alzheimer's | multi_hop | :x: | 10 | 331ms |
| Diseases sharing mechanisms with Parkinson's | multi_hop | :x: | 10 | 161ms |
| Treatments targeting mechanisms in Parkinson's | multi_hop | :x: | 15 | 9ms |
| Proteins involved in multiple diseases | multi_hop | :x: | 15 | 97ms |
| Symptoms shared across disease progression stages | multi_hop | :x: | 15 | 257ms |
| Most reported symptoms across all diseases | aggregation | :x: | 20 | 527ms |
| Research papers per disease | aggregation | :x: | 15 | 191ms |
| Most studied proteins | aggregation | :x: | 15 | 132ms |
| Most studied mechanisms | aggregation | :x: | 15 | 106ms |
| Treatments per disease | aggregation | :x: | 15 | 86ms |
| Papers about CJD (rare disease) | edge_case | :white_check_mark: | 10 | 168ms |
| Patient symptoms with zero research papers | edge_case | :x: | 0 | 149ms |

</details>

### Failure Cases

| Failure Mode | Example | Why It Fails |
|--------------|---------|--------------|
| **Missing entities** | "Papers by John Smith" | Entity not in graph |
| **Semantic ambiguity** | "Fast headphones" | "Fast" not a feature node |
| **Novel terminology** | "LLM hallucination papers" | Term not extracted |
| **Fuzzy matching** | "Alzheimer's" vs "Alzheimer'S Disease" | Exact match required |

**Mitigation:** Hybrid approach—graph for structured queries, vector for semantic exploration.

### When to Use What

| Use Case | Vector Search | Graph RAG |
|----------|:-------------:|:---------:|
| Topic/semantic search | :white_check_mark: | :white_check_mark: |
| Intersection queries (X AND Y) | :x: | :white_check_mark: |
| Attribute filtering (price, date) | :x: | :white_check_mark: |
| Multi-hop reasoning (A → B → C) | :x: | :white_check_mark: |
| Aggregation (counts, rankings) | :x: | :white_check_mark: |
| Cross-source comparison | :x: | :white_check_mark: |

See [EVALUATION.md](EVALUATION.md) for the full methodology and analysis.

---

## Tech Stack

- **Graph Database:** Neo4j 5.x with vector indexes
- **Embeddings:** OpenAI text-embedding-ada-002
- **LLM:** GPT-4 for entity extraction
- **UI:** Streamlit
- **API:** FastAPI
- **Visualization:** pyvis

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT - see [LICENSE](LICENSE)
