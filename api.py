"""
DeepGraph RAG API - FastAPI Backend

Exposes Graph RAG capabilities as REST endpoints for both research papers and products.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from examples.arxiv.arxiv_adapter import get_adapter, graph_enhanced_search

# Initialize FastAPI
app = FastAPI(
    title="DeepGraph RAG API",
    description="""
**Graph RAG API** - Intersection queries across structured relationships.

Vector search: "Find things about X"
Graph RAG: "Find things that HAVE X FOR Y"

Three domains, same pattern:
- **Research**: Method × Concept
- **Products**: Feature × UseCase × Price
- **Neurology**: Research vs Patient gaps
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize adapter (lazy loading)
_adapter = None

def get_rag_adapter():
    global _adapter
    if _adapter is None:
        _adapter = get_adapter()
    return _adapter


# ============== Request/Response Models ==============

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class GraphSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    include_vector_results: bool = True

class DiscoverRequest(BaseModel):
    paper_id: str
    max_hops: int = 2

class AnswerRequest(BaseModel):
    question: str
    context: Optional[str] = None
    use_graph: bool = True

class PaperResult(BaseModel):
    id: str
    title: str
    authors: List[str]
    category: Optional[str] = None
    score: Optional[float] = None
    connection_type: Optional[str] = None
    connection_via: Optional[str] = None

class GraphSearchResponse(BaseModel):
    vector_results: List[PaperResult]
    graph_discoveries: Dict[str, List[PaperResult]]
    total_unique_papers: int

class StatsResponse(BaseModel):
    papers: int
    authors: int
    concepts: int
    concept_connections: int


# ============== Endpoints ==============

@app.get("/")
async def root():
    """API health check and info."""
    return {
        "name": "DeepGraph RAG API",
        "version": "2.0.0",
        "status": "healthy",
        "demos": {
            "landing": "http://localhost:8500",
            "research": "http://localhost:8505",
            "products": "http://localhost:8506",
            "neurology": "http://localhost:8507",
        },
        "endpoints": {
            "research": {
                "search": "POST /search - Vector search",
                "graph_search": "POST /graph-search - Vector + graph-enhanced",
                "discover": "GET /discover/{paper_id} - Discovery journey",
                "answer": "POST /answer - Generate answer with context",
                "stats": "GET /stats - Graph statistics",
                "concepts": "GET /concepts - All concepts",
            },
            "products": {
                "search": "POST /products/search - Feature × UseCase search",
                "bundle": "GET /products/bundle/{asin} - Build product bundle",
                "stats": "GET /products/stats - Product graph stats",
                "features": "GET /products/features - All features",
                "use_cases": "GET /products/use-cases - All use cases",
            }
        }
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get graph statistics."""
    adapter = get_rag_adapter()
    stats = adapter.get_graph_stats()
    concepts = adapter.get_all_concepts()
    connections = adapter.get_concept_cooccurrence()

    return StatsResponse(
        papers=stats.get("Paper", 0),
        authors=stats.get("Author", 0),
        concepts=len(concepts),
        concept_connections=len(connections)
    )


@app.post("/search")
async def vector_search(request: SearchRequest):
    """
    Standard vector search.

    Returns papers with similar text/embeddings to the query.
    """
    adapter = get_rag_adapter()

    try:
        results = adapter.vector_search(request.query, top_k=request.top_k)

        return {
            "query": request.query,
            "results": [
                PaperResult(
                    id=p["id"],
                    title=p["title"],
                    authors=p.get("authors", []),
                    category=p.get("category"),
                    score=p.get("score")
                ).dict()
                for p in results
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph-search", response_model=GraphSearchResponse)
async def graph_search(request: GraphSearchRequest):
    """
    Graph-enhanced search.

    Returns both vector search results AND papers discovered through
    graph relationships (same authors, collaborators, cross-field).

    This is the key differentiator from standard RAG.
    """
    adapter = get_rag_adapter()

    try:
        # Run vector search
        vector_results = adapter.vector_search(request.query, top_k=request.top_k)

        # Run graph-enhanced search
        graph_results = graph_enhanced_search(adapter, vector_results, top_k=request.top_k)

        # Format vector results
        formatted_vector = [
            PaperResult(
                id=p["id"],
                title=p["title"],
                authors=p.get("authors", []),
                category=p.get("category"),
                score=p.get("score"),
                connection_type="vector_similarity"
            )
            for p in vector_results
        ]

        # Format graph discoveries
        formatted_graph = {}

        if graph_results.get("through_authors"):
            formatted_graph["same_authors_different_topics"] = [
                PaperResult(
                    id=p["id"],
                    title=p["title"],
                    authors=p.get("all_authors", []),
                    category=p.get("category"),
                    connection_type="same_author",
                    connection_via=p.get("connection_authors", ["Unknown"])[0]
                )
                for p in graph_results["through_authors"]
            ]

        if graph_results.get("through_coauthors"):
            formatted_graph["through_collaborator_network"] = [
                PaperResult(
                    id=p["id"],
                    title=p["title"],
                    authors=p.get("all_authors", []),
                    category=p.get("category"),
                    connection_type="collaborator_network",
                    connection_via=f"{p.get('source_author', '')} → {p.get('bridge_author', '')}"
                )
                for p in graph_results["through_coauthors"]
            ]

        if graph_results.get("through_concepts"):
            formatted_graph["cross_field_discoveries"] = [
                PaperResult(
                    id=p["id"],
                    title=p["title"],
                    authors=p.get("all_authors", []),
                    category=p.get("category"),
                    connection_type="cross_field",
                    connection_via=f"shared researchers in {p.get('source_category', '')}"
                )
                for p in graph_results["through_concepts"]
            ]

        # Count unique papers
        all_ids = set(p.id for p in formatted_vector)
        for papers in formatted_graph.values():
            all_ids.update(p.id for p in papers)

        return GraphSearchResponse(
            vector_results=formatted_vector if request.include_vector_results else [],
            graph_discoveries=formatted_graph,
            total_unique_papers=len(all_ids)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/discover/{paper_id}")
async def discover_journey(paper_id: str, max_hops: int = 2):
    """
    Discovery journey starting from a paper.

    Traverses the graph to find:
    1. Other papers by the same authors (different topics)
    2. Papers by collaborators (2 hops)
    3. Cross-field connections
    """
    adapter = get_rag_adapter()

    try:
        # Get paper info
        papers = adapter.get_all_papers()
        paper_info = next((p for p in papers if p['id'] == paper_id), None)

        if not paper_info:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

        journey = {
            "starting_paper": {
                "id": paper_info["id"],
                "title": paper_info["title"],
                "authors": paper_info.get("authors", []),
                "category": paper_info.get("category")
            },
            "discoveries": {}
        }

        # Step 1: Other papers by same authors
        authors = paper_info.get("authors", [])[:3]
        author_papers = []
        for author in authors:
            for ap in adapter.find_author_papers(author):
                if ap["id"] != paper_id:
                    ap["via_author"] = author
                    author_papers.append(ap)

        journey["discoveries"]["same_authors"] = [
            {
                "id": p["id"],
                "title": p["title"],
                "via": p.get("via_author"),
                "why": "Same author, potentially different topic"
            }
            for p in author_papers[:5]
        ]

        # Step 2: Collaborator papers
        collab_papers = []
        for author in authors[:2]:
            collabs = adapter.find_collaborators(author)
            for c in collabs[:3]:
                c_papers = adapter.find_author_papers(c["collaborator"])
                for cp in c_papers[:1]:
                    cp["via_author"] = author
                    cp["via_collaborator"] = c["collaborator"]
                    collab_papers.append(cp)

        journey["discoveries"]["through_collaborators"] = [
            {
                "id": p["id"],
                "title": p["title"],
                "via": f"{p.get('via_author')} → {p.get('via_collaborator')}",
                "why": "2 hops away in collaboration network"
            }
            for p in collab_papers[:5]
        ]

        # Step 3: Cross-field
        source_concept = paper_info.get("category", "")
        related_concepts = adapter.find_related_concepts(source_concept, top_k=3)

        journey["discoveries"]["cross_field"] = [
            {
                "from_concept": source_concept,
                "to_concept": rc["related_concept"],
                "shared_researchers": rc["shared_authors"],
                "why": "Researchers bridge both fields"
            }
            for rc in related_concepts
        ]

        return journey

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
async def generate_answer(request: AnswerRequest):
    """
    Generate an answer using RAG.

    If use_graph is True (default), includes graph-discovered papers in context.
    """
    adapter = get_rag_adapter()

    try:
        if request.context:
            # Use provided context
            context = request.context
        else:
            # Build context from search
            vector_results = adapter.vector_search(request.question, top_k=5)
            context_parts = [f"- {p['title']} by {', '.join(p.get('authors', [])[:2])}"
                          for p in vector_results]

            if request.use_graph:
                graph_results = graph_enhanced_search(adapter, vector_results, top_k=3)
                for p in graph_results.get("through_authors", [])[:2]:
                    context_parts.append(f"- [Graph] {p['title']} by {', '.join(p.get('all_authors', [])[:2])}")

            context = "\n".join(context_parts)

        answer = adapter.generate_answer(request.question, context)

        return {
            "question": request.question,
            "answer": answer,
            "context_used": context,
            "used_graph_enhancement": request.use_graph
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/concepts")
async def get_concepts():
    """Get all concepts and their connections."""
    adapter = get_rag_adapter()

    concepts = adapter.get_all_concepts()
    connections = adapter.get_concept_cooccurrence()

    return {
        "concepts": [
            {"code": c["concept"], "paper_count": c["paper_count"]}
            for c in concepts
        ],
        "connections": [
            {
                "concept1": c["concept1"],
                "concept2": c["concept2"],
                "shared_authors": c["shared_authors"],
                "papers_involved": c["papers"]
            }
            for c in connections
        ]
    }


@app.get("/authors/{author_name}/network")
async def get_author_network(author_name: str):
    """Get an author's collaboration network."""
    adapter = get_rag_adapter()

    try:
        papers = adapter.find_author_papers(author_name)
        collaborators = adapter.find_collaborators(author_name)
        network = adapter.get_author_network(author_name)

        if not papers and not collaborators:
            raise HTTPException(status_code=404, detail=f"Author '{author_name}' not found")

        return {
            "author": author_name,
            "papers": [{"id": p["id"], "title": p["title"]} for p in papers],
            "collaborators": [
                {"name": c["collaborator"], "shared_papers": c["shared_papers"]}
                for c in collaborators
            ],
            "paper_count": len(papers),
            "collaborator_count": len(collaborators)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers")
async def list_papers(limit: int = 100, offset: int = 0):
    """List all papers with pagination."""
    adapter = get_rag_adapter()

    papers = adapter.get_all_papers()

    return {
        "total": len(papers),
        "limit": limit,
        "offset": offset,
        "papers": [
            {
                "id": p["id"],
                "title": p["title"],
                "authors": p.get("authors", []),
                "category": p.get("category")
            }
            for p in papers[offset:offset+limit]
        ]
    }


# ============== Product Endpoints ==============

# Product database connection (lazy loading)
_product_driver = None

def get_product_driver():
    """Get Neo4j driver for product queries."""
    global _product_driver
    if _product_driver is None:
        from neo4j import GraphDatabase
        _product_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "deepgraph2025")
            )
        )
    return _product_driver


def product_query(cypher: str, params: dict = None):
    """Execute a Cypher query for products."""
    driver = get_product_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


class ProductSearchRequest(BaseModel):
    feature: Optional[str] = None
    use_case: Optional[str] = None
    price_tier: Optional[str] = None
    category: Optional[str] = None
    limit: int = 15


class ProductResult(BaseModel):
    asin: str
    title: str
    price: float
    rating: float
    category: str
    brand: str
    features: List[str]
    use_cases: List[str]


@app.get("/products/stats")
async def get_product_stats():
    """Get product graph statistics."""
    try:
        stats = product_query("""
            MATCH (n)
            WHERE n:Product OR n:Feature OR n:UseCase OR n:Category OR n:Brand
            RETURN labels(n)[0] as label, count(*) as count
        """)
        return {label: count for r in stats for label, count in [(r['label'], r['count'])]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/features")
async def get_product_features():
    """Get all features with product counts."""
    try:
        features = product_query("""
            MATCH (f:Feature)<-[:HAS_FEATURE]-(p:Product)
            RETURN f.name as name, f.display_name as display_name,
                   f.feature_type as feature_type, count(p) as product_count
            ORDER BY product_count DESC
        """)
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/use-cases")
async def get_product_use_cases():
    """Get all use cases with product counts."""
    try:
        use_cases = product_query("""
            MATCH (u:UseCase)<-[:FOR_USE_CASE]-(p:Product)
            RETURN u.name as name, u.display_name as display_name,
                   count(p) as product_count
            ORDER BY product_count DESC
        """)
        return {"use_cases": use_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products/search")
async def search_products(request: ProductSearchRequest):
    """
    Search products by Feature × UseCase.

    The killer query that vector search can't do:
    "Wireless products FOR travel under $100"
    """
    try:
        conditions = []
        params = {"limit": request.limit}

        if request.feature:
            conditions.append("(p)-[:HAS_FEATURE]->(:Feature {name: $feature})")
            params['feature'] = request.feature

        if request.use_case:
            conditions.append("(p)-[:FOR_USE_CASE]->(:UseCase {name: $use_case})")
            params['use_case'] = request.use_case

        if request.price_tier:
            conditions.append("p.price_tier = $price_tier")
            params['price_tier'] = request.price_tier

        if request.category:
            conditions.append("(p)-[:IN_CATEGORY]->(:Category {name: $category})")
            params['category'] = request.category

        where_clause = " AND ".join(conditions) if conditions else "true"

        results = product_query(f"""
            MATCH (p:Product)
            WHERE {where_clause}
            MATCH (p)-[:IN_CATEGORY]->(c:Category)
            MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (p)-[:FOR_USE_CASE]->(u:UseCase)
            RETURN p.asin as asin, p.title as title, p.price as price,
                   p.rating as rating, p.review_count as review_count,
                   c.name as category, b.name as brand,
                   collect(DISTINCT f.display_name) as features,
                   collect(DISTINCT u.display_name) as use_cases
            ORDER BY rating DESC, review_count DESC
            LIMIT $limit
        """, params)

        return {
            "query": {
                "feature": request.feature,
                "use_case": request.use_case,
                "price_tier": request.price_tier,
                "category": request.category
            },
            "count": len(results),
            "products": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/bundle/{asin}")
async def build_product_bundle(asin: str):
    """
    Build a product bundle starting from a product.

    Returns:
    - The selected product
    - Frequently bought together (BOUGHT_WITH)
    - Compatible products (shared features, different category)
    """
    try:
        # Get product info
        product_info = product_query("""
            MATCH (p:Product {asin: $asin})
            MATCH (p)-[:IN_CATEGORY]->(c:Category)
            MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            RETURN p.title as title, p.price as price, p.rating as rating,
                   c.name as category, b.name as brand,
                   collect(DISTINCT f.name) as features
        """, {"asin": asin})

        if not product_info:
            raise HTTPException(status_code=404, detail=f"Product {asin} not found")

        product = product_info[0]

        # Get frequently bought together
        bought_with = product_query("""
            MATCH (p:Product {asin: $asin})-[:BOUGHT_WITH]->(other:Product)
            MATCH (other)-[:IN_CATEGORY]->(c:Category)
            RETURN other.asin as asin, other.title as title, other.price as price,
                   other.rating as rating, c.name as category
            ORDER BY other.rating DESC
            LIMIT 5
        """, {"asin": asin})

        # Get compatible products (shared features, different category)
        compatible = product_query("""
            MATCH (p:Product {asin: $asin})-[:HAS_FEATURE]->(f:Feature)
                   <-[:HAS_FEATURE]-(other:Product)
            WHERE other.asin <> $asin
            AND other.category_name <> p.category_name
            WITH other, collect(f.display_name) as shared_features, count(f) as overlap
            WHERE overlap >= 2
            MATCH (other)-[:IN_CATEGORY]->(c:Category)
            RETURN other.asin as asin, other.title as title, other.price as price,
                   c.name as category, shared_features, overlap
            ORDER BY overlap DESC
            LIMIT 5
        """, {"asin": asin})

        # Calculate bundle total
        bundle_total = product['price']
        for item in bought_with:
            bundle_total += item['price']

        return {
            "selected_product": {
                "asin": asin,
                **product
            },
            "bought_together": bought_with,
            "compatible_products": compatible,
            "bundle_total": round(bundle_total, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/niches")
async def find_product_niches(min_products: int = 1, max_products: int = 10):
    """
    Find niche opportunities - rare feature + use case combinations.

    These represent underserved markets or product opportunities.
    """
    try:
        niches = product_query("""
            MATCH (f:Feature)<-[:HAS_FEATURE]-(p:Product)-[:FOR_USE_CASE]->(u:UseCase)
            WITH f, u, count(p) as product_count, collect(p.title)[0..3] as examples
            WHERE product_count >= $min AND product_count <= $max
            RETURN f.display_name as feature, u.display_name as use_case,
                   product_count, examples
            ORDER BY product_count ASC
            LIMIT 20
        """, {"min": min_products, "max": max_products})

        return {
            "niche_opportunities": niches,
            "count": len(niches)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
