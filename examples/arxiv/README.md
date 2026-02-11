# Research Navigator

Graph RAG for research paper discovery. Find papers by Method x Concept.

## Run

```bash
streamlit run examples/arxiv/app.py --server.port 8505
```

## The Value

| Query | Vector Search | Graph RAG |
|-------|---------------|-----------|
| "Papers about transformers" | Works | Works |
| "Papers USING transformers FOR reasoning" | Fails | Works |
| "Methods from NLP now used in Vision" | Impossible | Works |

## Tabs

1. **When to Use What** - Vector vs Graph explained
2. **Topic Search** - Standard semantic search
3. **Explore from Paper** - Author networks, collaborators
4. **Cross-Field Discovery** - Bridge researchers
5. **Method & Concept** - Papers by what they DO
6. **Research Navigator** - Method x Concept filter, Reading Paths

## Graph Schema

```
Paper
  ├── USES_METHOD ──→ Method (transformer, LoRA, diffusion...)
  ├── DISCUSSES ──→ Concept (reasoning, multimodal, robustness...)
  ├── USES_DATASET ──→ Dataset (ImageNet, COCO, MMLU...)
  └── AUTHORED_BY ──→ Author
```

## Data Pipeline

```bash
python scripts/01_download_arxiv.py --max-papers 500
python scripts/02_convert_arxiv_simple.py
python scripts/03_build_graph.py
python scripts/04_create_indexes.py
python scripts/06_extract_concepts.py
```

## Honest Assessment

Medium differentiation. Researchers already have good tools (Semantic Scholar, Connected Papers). The Method x Concept intersection is useful but not transformative.
