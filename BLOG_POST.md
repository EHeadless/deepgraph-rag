# I Built 3 Graph RAG Demos in a Weekend with Claude Code. Here's What I Learned About When Vector Search Fails.

*A deep dive into intersection queries, knowledge graphs, and why "wireless headphones for running under $100" breaks traditional RAG.*

---

## The Confession

I built this entire project—three working demos, a reusable library, and a REST API—in a weekend. With Claude Code.

I'm not saying this to brag. I'm saying it because it changed how I think about AI-assisted development. Claude didn't just autocomplete my code. It helped me think through architecture, challenged my assumptions, and at one point told me (correctly) that one of my demos wasn't as differentiated as the others.

More on that later. First, let me explain what I built and why it matters.

---

## The Problem: Vector Search Has a Blind Spot

Everyone's building RAG systems. Embed your documents, store them in a vector database, retrieve the most similar chunks, feed them to an LLM. It works great for questions like:

> "What papers discuss transformers?"

But it fails for questions like:

> "What papers USE transformers FOR reasoning tasks?"

See the difference? The first question is about **similarity**. The second is about **structure**—the intersection of two properties.

Here's a table that made the problem click for me:

| Query | Vector Search | Graph RAG |
|-------|:-------------:|:---------:|
| "Papers about transformers" | ✅ | ✅ |
| "Papers USING transformers FOR reasoning" | ❌ | ✅ |
| "Wireless headphones" | ✅ | ✅ |
| "Wireless headphones FOR running under $100" | ❌ | ✅ |
| "Symptoms patients report but research ignores" | ❌ | ✅ |

Vector search finds things **about** X. Graph RAG finds things that **have** X **for** Y.

---

## The Solution: One Pattern, Three Domains

I built three demos to prove this pattern is portable:

### 1. Research Navigator (arXiv Papers)

**The killer query:** Method × Concept

```
Paper ──→ USES_METHOD ──→ Method (transformer, LoRA, diffusion...)
  └──→ DISCUSSES ──→ Concept (reasoning, multimodal, robustness...)
```

Instead of searching for papers "about transformers and reasoning," I can query:

```cypher
MATCH (p:Paper)-[:USES_METHOD]->(:Method {name: "transformer"})
MATCH (p)-[:DISCUSSES]->(:Concept {name: "reasoning"})
RETURN p.title
```

This returns papers that **actually use** transformers **for** reasoning—not just papers that mention both words.

![Research Navigator Screenshot]

**Data:** 1,000 arXiv papers, 36 methods, 34 concepts, 5,000+ authors.

---

### 2. Product Navigator (Electronics)

**The killer query:** Feature × UseCase × Price

```
Product ──→ HAS_FEATURE ──→ Feature (wireless, waterproof, noise_canceling...)
  └──→ FOR_USE_CASE ──→ UseCase (travel, workout, gaming...)
```

The query "wireless headphones for running under $100" becomes:

```cypher
MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: "wireless"})
MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: "workout"})
WHERE p.price < 100
RETURN p.title, p.price
```

Try doing that with embeddings.

![Product Navigator Screenshot]

**Data:** 500 products, 22 features, 8 use cases, 51 brands.

**Bonus features:**
- Bundle Builder (co-purchase patterns)
- Niche Finder (rare feature + use case combos)
- Feature Migration (premium features in budget products)

---

### 3. Neurology Navigator (Research + Patient Data)

**The killer query:** Research vs Patient Gaps

This one's different. It combines two data sources:
- **PubMed papers** (what scientists study)
- **Reddit posts** (what patients actually experience)

```
Paper ──→ MENTIONS_SYMPTOM ──→ Symptom (clinical terms)
RedditPost ──→ REPORTS_SYMPTOM ──→ ReportedSymptom (patient language)

Disease
  ├── HAS_SYMPTOM ──→ Symptom (paper_count)
  └── HAS_REPORTED_SYMPTOM ──→ ReportedSymptom (report_count)
```

The unique query: **"What symptoms do patients report that research doesn't cover?"**

```cypher
MATCH (d:Disease {name: "Parkinson's"})-[:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
WHERE NOT EXISTS {
    MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name CONTAINS rs.name
}
RETURN rs.name as research_gap, rs.report_count
ORDER BY rs.report_count DESC
```

This is impossible with vector search. You need the graph structure to compare across data sources.

![Neurology Navigator Screenshot]

**Data:** 495 papers, 31,077 patient symptom extractions, 15 neurodegenerative diseases.

---

## The Universal Pattern

All three demos use the same structure:

```
Entity ──→ HAS_CAPABILITY ──→ Capability
  └──→ FOR_INTENT ──→ Intent
```

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Research | Paper | Method | Concept |
| Products | Product | Feature | UseCase |
| Neurology | Paper/Post | Symptom | Disease |

**The intersection query `Capability × Intent` is what vector search cannot do.**

---

## The Tech Stack

- **Graph Database:** Neo4j 5.x (with native vector indexes!)
- **Embeddings:** OpenAI text-embedding-ada-002
- **Entity Extraction:** GPT-4 (for methods, concepts, symptoms)
- **UI:** Streamlit
- **API:** FastAPI
- **Visualization:** pyvis

Neo4j 5.x is key here—it supports vector indexes natively, so you can combine vector similarity with graph traversal in a single query:

```cypher
// Find similar papers, then traverse to their methods
CALL db.index.vector.queryNodes('paper_embedding', 10, $embedding)
YIELD node as paper, score
MATCH (paper)-[:USES_METHOD]->(m:Method)
RETURN paper.title, collect(m.name) as methods, score
```

---

## The Honest Assessment

Here's where Claude Code earned its keep. When I asked for an objective assessment, it didn't just validate my work. It told me:

| Demo | Differentiation | Reality Check |
|------|-----------------|---------------|
| **Products** | High | Clear product-market fit. People actually think in Feature + UseCase + Price. |
| **Neurology** | High | Unique dual-source capability. Research gap analysis is genuinely novel. |
| **Research** | Medium | Researchers already have good tools (Semantic Scholar, Connected Papers). |

That last one stung. But it's true. The research demo is useful, but it's not as differentiated as I thought.

---

## What This Means for Enterprise

The pattern is portable. If your domain has:

1. **Entities** (documents, products, patients, contracts...)
2. **Capabilities** (features, methods, symptoms, clauses...)
3. **Intents** (use cases, goals, diagnoses, obligations...)

...then you can build intersection queries that vector search can't touch.

**Enterprise examples:**

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Legal | Contract | Clause | Obligation |
| Healthcare | Patient | Symptom | Diagnosis |
| HR | Candidate | Skill | Role |
| Supply Chain | Supplier | Capability | Requirement |

The query "suppliers WITH ISO-9001 certification FOR automotive components IN Europe" is a graph query, not a vector query.

---

## The Numbers: Evaluation Results

I ran a formal evaluation across **65 test queries** covering 6 query types. The results were stark:

| Metric | Vector Search | Graph RAG |
|--------|:-------------:|:---------:|
| **Overall Coverage** | 15.4% (10/65) | 100% (65/65) |
| **Intersection Queries** | 0% (0/18) | 100% |
| **Multi-Hop Queries** | 0% (0/14) | 100% |
| **Aggregation Queries** | 0% (0/14) | 100% |
| **Comparison Queries** | 0% (0/3) | 100% |
| **Similarity Queries** | 100% (8/8) | 100% |

### By Query Type

| Type | Example | Vector | Graph |
|------|---------|:------:|:-----:|
| Similarity | "Papers about transformers" | :white_check_mark: | :white_check_mark: |
| Intersection | "Wireless FOR workout under $100" | :x: | :white_check_mark: |
| Multi-Hop | "Collaborators of author X" | :x: | :white_check_mark: |
| Aggregation | "Most popular methods" | :x: | :white_check_mark: |
| Comparison | "Research vs patient symptoms" | :x: | :white_check_mark: |
| Edge Cases | "Wireless NOT for gaming" (negation) | :x: | :white_check_mark: |

**84.6% of test queries (55/65) were Graph-RAG-only.**

### Precision Test: "Wireless headphones FOR workout under $100"

| Approach | Results | Meets ALL Criteria |
|----------|---------|-------------------|
| Vector Search | 20 | ~10% |
| Graph RAG | 1 | 100% |

Vector search returns 20 results, but only ~2 actually meet all three criteria. Graph RAG returns exactly what you asked for.

### When Graph RAG Fails

It's not perfect. Graph RAG fails when:
- Entity not in graph (fallback to vector)
- Novel terminology not extracted
- Fuzzy matching needed ("Alzheimer's" vs "Alzheimer'S")

The solution? **Hybrid approach:** Graph for structured queries, vector as fallback.

Full evaluation: [EVALUATION.md](https://github.com/EHeadless/deepgraph-rag/blob/main/EVALUATION.md)

---

## Working with Claude Code

A few observations from building this with AI assistance:

**What worked:**
- Architecture discussions. Claude helped me see the universal pattern across domains.
- Honest feedback. When I asked if the neurology demo was as good as the others, it told me what was missing (and then helped fix it).
- Boilerplate acceleration. GitHub templates, issue templates, setup guides—done in minutes.

**What required human judgment:**
- Deciding which features actually mattered
- Knowing when to stop adding features
- The "is this actually useful?" gut check

**The meta-lesson:** AI-assisted development isn't about generating code. It's about having a collaborator who can engage with your ideas, push back when needed, and execute quickly when the direction is clear.

---

## Try It Yourself

The code is open source:

**GitHub:** https://github.com/EHeadless/deepgraph-rag

```bash
git clone https://github.com/EHeadless/deepgraph-rag.git
cd deepgraph-rag
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
docker-compose up -d
streamlit run landing.py --server.port 8500
```

---

## The Takeaway

Vector search is great for similarity. But the real world is full of intersection queries:

- "Products WITH feature X FOR use case Y under price Z"
- "Papers USING method X FOR concept Y"
- "Candidates WITH skill X FOR role Y in location Z"

If you're building RAG systems and your users are frustrated that search "doesn't understand what they really want," consider whether they're asking intersection queries that vector search can't answer.

The solution isn't better embeddings. It's adding structure.

---

*Built with Claude Code. Graph database by Neo4j. Opinions my own.*

---

## Appendix: Key Code Snippets

### The Intersection Query (Cypher)

```cypher
// Products: Feature × UseCase × Price
MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: $feature})
MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: $use_case})
WHERE p.price <= $max_price
MATCH (p)-[:IN_CATEGORY]->(c:Category)
MATCH (p)-[:MADE_BY]->(b:Brand)
RETURN p.title, p.price, p.rating, c.name as category, b.name as brand
ORDER BY p.rating DESC
LIMIT 20
```

### The Research Gap Query (Cypher)

```cypher
// Find symptoms patients report but research doesn't cover
MATCH (d:Disease {name: $disease})-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
WHERE NOT EXISTS {
    MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE toLower(s.name) CONTAINS toLower(rs.name)
       OR toLower(rs.name) CONTAINS toLower(s.name)
}
RETURN rs.name as symptom, r.report_count as patient_reports
ORDER BY r.report_count DESC
LIMIT 25
```

### The Schema Pattern (Python)

```python
# All domains follow this pattern
SCHEMA_PATTERN = {
    "entity": {
        "research": "Paper",
        "products": "Product",
        "neurology": "Paper | RedditPost"
    },
    "capability": {
        "research": "Method",
        "products": "Feature",
        "neurology": "Symptom"
    },
    "intent": {
        "research": "Concept",
        "products": "UseCase",
        "neurology": "Disease"
    },
    "killer_query": "Capability × Intent"
}
```

---

**Tags:** #RAG #GraphRAG #Neo4j #LLM #VectorSearch #KnowledgeGraphs #AI #MachineLearning #ClaudeCode
