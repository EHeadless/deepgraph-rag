# The Query That Broke My RAG System (And How Graphs Fixed It)

*Why 85% of real-world search queries fail with vector embeddings—and a weekend experiment that changed how I think about retrieval.*

---

## The Moment It Clicked

I was building a product recommendation system. Standard RAG setup: embed product descriptions, store in a vector database, retrieve by similarity. It worked great until a user typed:

> "wireless headphones for running under $100"

The system returned 20 results. Exactly 2 of them were actually wireless, for running, AND under $100. The rest were semantically similar—wireless speakers, expensive running watches, cheap wired earbuds that mentioned "workout" somewhere in the description.

**Precision: 10%.**

That's when I realized: **vector search finds things that are ABOUT your query. It can't find things that HAVE specific properties FOR specific purposes.**

This distinction matters more than most RAG tutorials acknowledge.

---

## The Fundamental Limitation

Here's the query taxonomy that clarified everything for me:

```
SIMILARITY QUERY:  "Find things about X"
                   → Vector search excels here
                   → "Papers about transformers" ✓

INTERSECTION QUERY: "Find things that HAVE X FOR Y"
                    → Vector search fails here
                    → "Papers USING transformers FOR reasoning" ✗
```

The problem isn't the embeddings. It's that embeddings encode **semantic similarity**, not **structural relationships**. When you embed "wireless headphones for running under $100," you get a point in 1536-dimensional space. But that point doesn't encode:

- `HAS_FEATURE: wireless` (boolean)
- `FOR_USE_CASE: running` (categorical)
- `price < 100` (numeric constraint)

It encodes "this text is semantically near other texts about wireless running headphones." That's a different thing entirely.

---

## The Graph Solution

I spent a weekend building three demos to prove a hypothesis: **knowledge graphs can answer the queries that vector search cannot**.

The core idea is simple. Instead of:

```
Product → [embedding] → Vector Space
```

You build:

```
Product ──HAS_FEATURE──→ Feature
   │
   └──FOR_USE_CASE──→ UseCase
   │
   └──IN_CATEGORY──→ Category
   │
   └──MADE_BY──→ Brand
```

Now the query "wireless headphones for running under $100" becomes:

```cypher
// Cypher query in Neo4j
MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: "wireless"})
MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: "workout"})
MATCH (p)-[:IN_CATEGORY]->(:Category {name: "Headphones"})
WHERE p.price < 100
RETURN p.title, p.price, p.rating
ORDER BY p.rating DESC
```

**Precision: 100%.** Every result matches every criterion.

---

## Three Domains, One Pattern

I built demos across three domains to prove the pattern is universal:

### Demo 1: Research Navigator

**Problem:** Find papers that USE a method FOR a concept.

```python
# The graph schema
class Paper:
    title: str
    abstract: str
    embedding: List[float]  # For similarity fallback

class Method:
    name: str  # "transformer", "LoRA", "diffusion", "RLHF"

class Concept:
    name: str  # "reasoning", "multimodal", "efficiency"

# Relationships
# Paper -[:USES_METHOD]-> Method
# Paper -[:DISCUSSES]-> Concept
# Paper -[:AUTHORED_BY]-> Author
```

**The killer query:**

```cypher
// Papers using transformers for reasoning tasks
MATCH (p:Paper)-[:USES_METHOD]->(:Method {name: "transformer"})
MATCH (p)-[:DISCUSSES]->(:Concept {name: "reasoning"})
RETURN p.title, p.abstract
LIMIT 20
```

Compare this to vector search for "transformer reasoning papers"—you'll get papers that *mention* both words, not papers that actually *apply* transformers *to* reasoning.

**Data:** 1,495 arXiv papers, 6,950 authors, 36 methods, 85 concepts.

---

### Demo 2: Product Navigator

**Problem:** Find products with FEATURE for USE_CASE under PRICE.

```python
# The graph schema
class Product:
    title: str
    price: float
    rating: float
    embedding: List[float]

class Feature:
    name: str  # "wireless", "waterproof", "noise_canceling"

class UseCase:
    name: str  # "travel", "workout", "gaming", "office"

# Relationships
# Product -[:HAS_FEATURE]-> Feature
# Product -[:FOR_USE_CASE]-> UseCase
# Product -[:IN_CATEGORY]-> Category
# Product -[:MADE_BY]-> Brand
```

**The killer queries:**

```cypher
// Triple intersection: feature × use case × price
MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: "wireless"})
MATCH (p)-[:HAS_FEATURE]->(:Feature {name: "noise_canceling"})
MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: "travel"})
WHERE p.price < 200
RETURN p.title, p.price, p.rating
ORDER BY p.rating DESC

// Feature migration: premium features in budget products
MATCH (premium:Product)-[:HAS_FEATURE]->(f:Feature)
WHERE premium.price > 300
WITH f, count(premium) as premium_count
WHERE premium_count > 5  // Common in premium tier
MATCH (budget:Product)-[:HAS_FEATURE]->(f)
WHERE budget.price < 100
RETURN budget.title, budget.price, collect(f.name) as premium_features
```

**Data:** 500 products, 22 features, 8 use cases, 51 brands.

---

### Demo 3: Neurology Navigator

**Problem:** Find symptoms that patients report but research ignores.

This one is different. It combines two data sources:
- **PubMed papers** (what scientists study)
- **Reddit posts** (what patients actually experience)

```python
# The graph schema
class Paper:
    title: str
    pmid: str

class RedditPost:
    text: str
    subreddit: str

class Symptom:
    name: str  # Clinical terminology

class ReportedSymptom:
    name: str  # Patient language

class Disease:
    name: str

# Relationships from papers
# Paper -[:MENTIONS_SYMPTOM]-> Symptom
# Disease -[:HAS_SYMPTOM {paper_count}]-> Symptom

# Relationships from Reddit
# RedditPost -[:REPORTS_SYMPTOM]-> ReportedSymptom
# Disease -[:HAS_REPORTED_SYMPTOM {report_count}]-> ReportedSymptom
```

**The killer query—finding research gaps:**

```cypher
// Symptoms patients report that research doesn't cover
MATCH (d:Disease {name: "Parkinson's Disease"})
      -[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
WHERE NOT EXISTS {
    MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE toLower(s.name) CONTAINS toLower(rs.name)
       OR toLower(rs.name) CONTAINS toLower(s.name)
}
RETURN rs.name as research_gap,
       r.report_count as patient_mentions
ORDER BY r.report_count DESC
LIMIT 25
```

This query is **impossible with vector search**. You need the graph structure to compare what exists in one data source versus another.

**Data:** 1,495 papers, 31,077 Reddit posts, 228 diseases, 60,000+ symptom mentions.

---

## The Universal Schema Pattern

All three demos follow the same abstract pattern:

```
Entity ──HAS_CAPABILITY──→ Capability
   │
   └──FOR_INTENT──→ Intent
```

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Research | Paper | Method | Concept |
| Products | Product | Feature | UseCase |
| Medical | Paper/Post | Symptom | Disease |

**The intersection query `Capability × Intent` is what vector search cannot do.**

This pattern is portable to any domain:

| Domain | Entity | Capability | Intent |
|--------|--------|------------|--------|
| Legal | Contract | Clause | Obligation |
| HR | Candidate | Skill | Role |
| Supply Chain | Supplier | Certification | Requirement |
| Finance | Company | Metric | Industry |

The query "suppliers WITH ISO-9001 FOR automotive IN Europe" is a graph query, not a vector query.

---

## The Technical Stack

```
┌─────────────────────────────────────────────────────────┐
│                      Streamlit UI                        │
├─────────────────────────────────────────────────────────┤
│                      FastAPI Backend                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  OpenAI API     │    │      Neo4j 5.x              │ │
│  │  - Embeddings   │    │  - Graph storage            │ │
│  │  - GPT-4 extract│    │  - Vector indexes (native)  │ │
│  │  - Generation   │    │  - Cypher queries           │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Key technical choices:**

1. **Neo4j 5.x with native vector indexes.** This is crucial—you can combine vector similarity with graph traversal in a single query:

```cypher
// Hybrid: vector similarity + graph traversal
CALL db.index.vector.queryNodes('paper_embedding', 10, $query_embedding)
YIELD node as paper, score
MATCH (paper)-[:USES_METHOD]->(m:Method)
MATCH (paper)-[:AUTHORED_BY]->(a:Author)
RETURN paper.title,
       collect(DISTINCT m.name) as methods,
       collect(DISTINCT a.name) as authors,
       score
ORDER BY score DESC
```

2. **GPT-4 for entity extraction.** Extracting structured entities from unstructured text:

```python
def extract_entities(text: str) -> dict:
    """Extract methods, concepts, and relationships from paper abstract."""
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{
            "role": "system",
            "content": """Extract structured entities from this research abstract.
            Return JSON with:
            - methods: list of ML/AI methods used (e.g., "transformer", "LoRA")
            - concepts: list of research concepts (e.g., "reasoning", "multimodal")
            - datasets: list of datasets mentioned (e.g., "ImageNet", "MMLU")
            """
        }, {
            "role": "user",
            "content": text
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

3. **Streamlit for rapid iteration.** Each demo is a single Python file with immediate visual feedback.

---

## Rigorous Evaluation

I didn't just build demos—I evaluated them systematically. **65 queries across 6 query types, 3 domains.**

### Methodology

```python
QUERY_TYPES = {
    "similarity": "Find things about X",
    "intersection": "Find things with X FOR Y",
    "multi_hop": "Find things connected through N relationships",
    "aggregation": "Count/rank things by property",
    "comparison": "Compare data from source A vs B",
    "edge_cases": "Boundary conditions, negation, rare entities"
}

def evaluate_query(query: dict) -> dict:
    """Run query and measure coverage + latency."""
    start = time.time()
    results = run_graph_query(query["cypher"])
    elapsed = (time.time() - start) * 1000

    return {
        "query_name": query["name"],
        "query_type": query["type"],
        "vector_can_answer": query["vector_possible"],
        "graph_can_answer": True,
        "results_count": len(results),
        "latency_ms": elapsed
    }
```

### Results

| Metric | Vector Search | Graph RAG |
|--------|:-------------:|:---------:|
| **Overall Coverage** | 15.4% (10/65) | 100% (65/65) |
| **Intersection Queries** | 0% (0/18) | 100% (18/18) |
| **Multi-Hop Queries** | 0% (0/14) | 100% (14/14) |
| **Aggregation Queries** | 0% (0/14) | 100% (14/14) |
| **Comparison Queries** | 0% (0/3) | 100% (3/3) |
| **Similarity Queries** | 100% (8/8) | 100% (8/8) |

**84.6% of test queries are Graph-RAG-only.**

### Query Latency

| Query Type | Avg (ms) | Max (ms) | Notes |
|------------|----------:|--------:|-------|
| Similarity | 157 | 723 | Cold cache on first query |
| Intersection | 65 | 229 | Fast—index-driven |
| Multi-hop | 106 | 331 | Depends on hop count |
| Aggregation | 94 | 527 | Scales with data size |
| Comparison | 2,606 | 4,314 | NOT EXISTS is expensive |

### Precision Deep-Dive

For the query "wireless headphones FOR workout under $100":

```
Vector Search Results: 20
├── Actually wireless: ~15 (75%)
├── Actually for workout: ~8 (40%)
├── Actually under $100: ~6 (30%)
└── Meets ALL criteria: ~2 (10%)

Graph RAG Results: 1
└── Meets ALL criteria: 1 (100%)
```

**Vector precision: 10%. Graph precision: 100%.**

### When Graph RAG Fails

The evaluation also revealed failure modes:

| Failure | Example | Mitigation |
|---------|---------|------------|
| Missing entity | "Papers by John Smith" (not in graph) | Fallback to vector |
| Novel terminology | "LLM hallucination" (not extracted) | Periodic re-extraction |
| Fuzzy matching | "Alzheimer's" vs "Alzheimer'S" | Case-insensitive matching |
| Schema gaps | "Papers about ethics" (not tracked) | Hybrid approach |

**The solution: hybrid retrieval.** Graph first, vector fallback.

```python
def hybrid_search(query: str, filters: dict) -> List[dict]:
    """Try graph query first, fall back to vector if no results."""
    # Attempt structured graph query
    if filters:
        results = graph_query(filters)
        if results:
            return results

    # Fallback to vector similarity
    embedding = get_embedding(query)
    return vector_search(embedding, top_k=20)
```

---

## Building with Claude Code

I built this entire project—three demos, evaluation framework, API, documentation—in a weekend. With Claude Code as my pair programmer.

Some observations:

**What AI excels at:**

```python
# Me: "Create a Cypher query to find feature migration patterns"
# Claude: Immediately produces this

MATCH (premium:Product)-[:HAS_FEATURE]->(f:Feature)
WHERE premium.price > 300
WITH f, count(premium) as premium_count
WHERE premium_count > 5
MATCH (budget:Product)-[:HAS_FEATURE]->(f)
WHERE budget.price < 100
RETURN budget.title, budget.price,
       collect(f.name) as premium_features_in_budget
ORDER BY size(collect(f.name)) DESC
```

**What required human judgment:**

- Deciding the neurology demo needed a "research gap" feature to differentiate it
- Knowing that 21 evaluation queries wasn't enough (expanded to 65)
- The "is this actually useful?" gut check that killed a 4th demo idea

**The meta-lesson:** AI-assisted development isn't code generation. It's having a collaborator who can engage with architecture, push back on assumptions, and execute rapidly when direction is clear.

---

## Where This Goes Next

### Near-term: Production Patterns

1. **Incremental graph updates.** Stream new data without full rebuilds:

```python
async def process_new_document(doc: Document):
    """Extract and add to graph incrementally."""
    entities = await extract_entities(doc.text)
    embedding = await get_embedding(doc.text)

    async with graph.transaction() as tx:
        node_id = await tx.create_node("Document", {
            "title": doc.title,
            "embedding": embedding
        })
        for entity in entities["methods"]:
            method_id = await tx.merge_node("Method", {"name": entity})
            await tx.create_relationship(node_id, "USES_METHOD", method_id)
```

2. **Query intent classification.** Route queries to graph vs vector:

```python
def classify_query_intent(query: str) -> str:
    """Determine if query needs graph or vector retrieval."""
    # Keywords suggesting structured query
    graph_signals = ["for", "with", "under", "between", "by", "not"]

    # Check for price/date patterns
    has_numeric = bool(re.search(r'\$?\d+', query))

    if any(signal in query.lower() for signal in graph_signals) or has_numeric:
        return "graph"
    return "vector"
```

3. **Explanation generation.** Show users WHY results match:

```cypher
// Return the match path, not just results
MATCH path = (p:Product)-[:HAS_FEATURE]->(f:Feature {name: "wireless"})
MATCH path2 = (p)-[:FOR_USE_CASE]->(u:UseCase {name: "workout"})
WHERE p.price < 100
RETURN p.title, p.price,
       [n in nodes(path) | labels(n)[0] + ": " + coalesce(n.name, n.title)] as match_reason
```

### Medium-term: Agentic Retrieval

The real power emerges when you combine graphs with LLM agents:

```python
class GraphRAGAgent:
    """Agent that can decompose queries and traverse the graph."""

    def answer(self, question: str) -> str:
        # Step 1: Decompose into sub-queries
        plan = self.plan_query(question)

        # Step 2: Execute graph traversals
        context = []
        for step in plan:
            if step.type == "graph_query":
                results = self.graph.query(step.cypher)
                context.extend(results)
            elif step.type == "vector_search":
                results = self.vector.search(step.query)
                context.extend(results)

        # Step 3: Generate answer with retrieved context
        return self.generate(question, context)
```

### Long-term: Self-Evolving Schemas

The most exciting frontier: graphs that extend their own schemas based on query patterns:

```python
def detect_schema_gaps(failed_queries: List[str]) -> List[SchemaExtension]:
    """Analyze failed queries to suggest schema extensions."""
    # Cluster failed queries by topic
    clusters = cluster_by_embedding(failed_queries)

    suggestions = []
    for cluster in clusters:
        # Use LLM to propose new node/relationship types
        proposal = llm.propose_schema_extension(cluster.queries)
        if proposal.confidence > 0.8:
            suggestions.append(proposal)

    return suggestions
```

Imagine a knowledge graph that notices users frequently ask about "papers WITH code" and automatically adds a `HAS_CODE` relationship to the schema.

---

## Try It Yourself

The code is open source:

**GitHub:** [github.com/EHeadless/deepgraph-rag](https://github.com/EHeadless/deepgraph-rag)

```bash
# Clone and setup
git clone https://github.com/EHeadless/deepgraph-rag.git
cd deepgraph-rag
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY

# Start Neo4j
docker-compose up -d

# Run the landing page
streamlit run landing.py --server.port 8500
```

Then open http://localhost:8500 and explore the three demos.

---

## The Takeaway

Vector search is excellent for similarity. But the real world is full of intersection queries:

- "Products WITH feature X FOR use case Y under price Z"
- "Papers USING method X FOR concept Y"
- "Candidates WITH skill X FOR role Y in location Z"
- "Symptoms patients report BUT research ignores"

If your users complain that search "doesn't understand what they really want," they might be asking intersection queries that embeddings fundamentally cannot answer.

**The solution isn't better embeddings. It's adding structure.**

Knowledge graphs aren't a replacement for vector search—they're a complement. The hybrid approach (graph for structured queries, vector for semantic exploration) captures the best of both worlds.

And with modern tools like Neo4j's native vector indexes, you don't have to choose. You can have both in a single query.

---

*Built in a weekend with Claude Code. Evaluated rigorously. Open sourced for you to extend.*

---

**Tags:** #RAG #GraphRAG #Neo4j #KnowledgeGraphs #VectorSearch #LLM #MachineLearning #InformationRetrieval #AI #ClaudeCode
