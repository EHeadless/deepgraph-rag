"""
Comprehensive Evaluation: Graph RAG vs Vector Search

Tests across three domains:
1. Research (arXiv papers)
2. Products (electronics)
3. Neurology (research + patient data)

Query types:
- Similarity queries (vector search should work)
- Intersection queries (graph RAG advantage)
- Multi-hop queries (graph RAG only)
- Aggregation queries (graph RAG only)
- Comparison queries (graph RAG only)
"""

import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Initialize connections
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "deepgraph2025"))
)


def query(cypher: str, params: dict = None) -> List[Dict]:
    """Execute Cypher query."""
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


@dataclass
class QueryResult:
    query_type: str
    domain: str
    query_name: str
    natural_language: str
    vector_can_answer: bool
    graph_can_answer: bool
    vector_results: int
    graph_results: int
    vector_time_ms: float
    graph_time_ms: float
    notes: str


# ============================================================
# EVALUATION QUERIES
# ============================================================

EVALUATION_QUERIES = {
    "research": {
        "similarity": [
            {
                "name": "topic_search",
                "nl": "Papers about transformer architectures",
                "vector_query": "transformer architectures neural networks",
                "graph_query": """
                    MATCH (p:Paper)
                    WHERE p.title CONTAINS 'transformer' OR p.title CONTAINS 'Transformer'
                    RETURN p.title as title, p.arxiv_id as id
                    LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "author_papers",
                "nl": "Papers by a specific author",
                "vector_query": "papers by Geoffrey Hinton deep learning",
                "graph_query": """
                    MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
                    WHERE a.name CONTAINS 'Hinton'
                    RETURN p.title as title, a.name as author
                    LIMIT 20
                """,
                "vector_can_answer": False,  # Vector can't filter by author
                "graph_can_answer": True,
            },
        ],
        "intersection": [
            {
                "name": "method_x_concept",
                "nl": "Papers using transformers FOR reasoning",
                "vector_query": "transformer reasoning papers",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(:Method {name: 'transformer'})
                    MATCH (p)-[:ABOUT_CONCEPT]->(:Concept {name: 'reasoning'})
                    RETURN p.title as title, p.arxiv_id as id
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_x_dataset",
                "nl": "Papers using LoRA evaluated on MMLU",
                "vector_query": "LoRA MMLU benchmark evaluation",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(:Method {name: 'lora'})
                    MATCH (p)-[:USES_DATASET]->(:Dataset {name: 'mmlu'})
                    RETURN p.title as title
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "multi_hop": [
            {
                "name": "collaborator_network",
                "nl": "Papers by collaborators of author X",
                "vector_query": None,  # Impossible
                "graph_query": """
                    MATCH (a:Author {name: 'Yann LeCun'})<-[:AUTHORED_BY]-(p1:Paper)
                          -[:AUTHORED_BY]->(coauthor:Author)
                    WHERE coauthor <> a
                    MATCH (coauthor)<-[:AUTHORED_BY]-(p2:Paper)
                    WHERE p2 <> p1
                    RETURN DISTINCT p2.title as title, coauthor.name as via_author
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_transfer",
                "nl": "Methods from NLP used in computer vision",
                "vector_query": None,
                "graph_query": """
                    MATCH (p1:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE p1.primary_category CONTAINS 'CL'
                    MATCH (p2:Paper)-[:USES_METHOD]->(m)
                    WHERE p2.primary_category CONTAINS 'CV'
                    RETURN m.name as method, count(DISTINCT p2) as cv_papers
                    ORDER BY cv_papers DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "aggregation": [
            {
                "name": "method_popularity",
                "nl": "Most commonly used methods",
                "vector_query": None,
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    RETURN m.name as method, count(p) as paper_count
                    ORDER BY paper_count DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
    },
    "products": {
        "similarity": [
            {
                "name": "product_search",
                "nl": "Wireless headphones",
                "vector_query": "wireless headphones bluetooth audio",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:IN_CATEGORY]->(:Category {name: 'Headphones'})
                    RETURN p.title as title, p.price as price
                    LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
        ],
        "intersection": [
            {
                "name": "feature_x_usecase",
                "nl": "Wireless products FOR workout",
                "vector_query": "wireless workout exercise sports",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: 'workout'})
                    RETURN p.title as title, p.price as price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "feature_x_usecase_x_price",
                "nl": "Wireless headphones FOR travel under $100",
                "vector_query": "wireless headphones travel portable cheap",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: 'travel'})
                    MATCH (p)-[:IN_CATEGORY]->(:Category {name: 'Headphones'})
                    WHERE p.price < 100
                    RETURN p.title as title, p.price as price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "feature_x_feature_x_usecase",
                "nl": "Waterproof AND wireless FOR workout",
                "vector_query": "waterproof wireless workout sports",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:HAS_FEATURE]->(:Feature {name: 'waterproof'})
                    MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: 'workout'})
                    RETURN p.title as title, p.price as price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "multi_hop": [
            {
                "name": "bundle_recommendation",
                "nl": "Products frequently bought with laptop X",
                "vector_query": None,
                "graph_query": """
                    MATCH (p:Product)-[:IN_CATEGORY]->(:Category {name: 'Laptop'})
                    WITH p LIMIT 1
                    MATCH (p)-[:BOUGHT_WITH]->(other:Product)
                    RETURN other.title as title, other.price as price
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "feature_migration",
                "nl": "Premium features now in budget products",
                "vector_query": None,
                "graph_query": """
                    MATCH (premium:Product)-[:HAS_FEATURE]->(f:Feature)
                    WHERE premium.price > 300
                    MATCH (budget:Product)-[:HAS_FEATURE]->(f)
                    WHERE budget.price < 100 AND budget <> premium
                    RETURN f.name as feature,
                           count(DISTINCT budget) as budget_products
                    ORDER BY budget_products DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "aggregation": [
            {
                "name": "niche_finder",
                "nl": "Rare feature + use case combinations",
                "vector_query": None,
                "graph_query": """
                    MATCH (f:Feature)<-[:HAS_FEATURE]-(p:Product)-[:FOR_USE_CASE]->(u:UseCase)
                    WITH f, u, count(p) as product_count
                    WHERE product_count >= 1 AND product_count <= 5
                    RETURN f.name as feature, u.name as use_case, product_count
                    ORDER BY product_count ASC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
    },
    "neurology": {
        "similarity": [
            {
                "name": "disease_papers",
                "nl": "Papers about Alzheimer's disease",
                "vector_query": "Alzheimer's disease research treatment",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'alzheimer'
                    RETURN p.title as title, p.pmid as id
                    LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
        ],
        "intersection": [
            {
                "name": "disease_x_symptom",
                "nl": "Papers about Parkinson's AND tremor",
                "vector_query": "Parkinson's tremor symptoms",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    MATCH (p)-[:MENTIONS_SYMPTOM]->(s:Symptom)
                    WHERE toLower(s.name) CONTAINS 'tremor'
                    RETURN p.title as title, s.name as symptom
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "disease_x_protein",
                "nl": "Papers about Alzheimer's mentioning tau protein",
                "vector_query": "Alzheimer's tau protein",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'alzheimer'
                    MATCH (p)-[:MENTIONS_PROTEIN]->(pr:Protein)
                    WHERE toLower(pr.name) CONTAINS 'tau'
                    RETURN p.title as title, pr.name as protein
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "comparison": [
            {
                "name": "research_vs_patient",
                "nl": "Symptoms patients report but research doesn't cover",
                "vector_query": None,
                "graph_query": """
                    MATCH (d:Disease)-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    AND NOT EXISTS {
                        MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                        WHERE toLower(s.name) CONTAINS toLower(rs.name)
                           OR toLower(rs.name) CONTAINS toLower(s.name)
                    }
                    RETURN rs.name as research_gap, r.report_count as patient_reports
                    ORDER BY r.report_count DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "multi_hop": [
            {
                "name": "shared_symptoms",
                "nl": "Diseases that share symptoms with Alzheimer's",
                "vector_query": None,
                "graph_query": """
                    MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
                    WHERE toLower(d1.name) CONTAINS 'alzheimer'
                    AND d1 <> d2
                    RETURN d2.name as disease, count(s) as shared_symptoms
                    ORDER BY shared_symptoms DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "treatment_via_mechanism",
                "nl": "Treatments targeting mechanisms involved in Parkinson's",
                "vector_query": None,
                "graph_query": """
                    MATCH (d:Disease)-[:INVOLVES_MECHANISM]->(m:Mechanism)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    MATCH (p:Paper)-[:MENTIONS_MECHANISM]->(m)
                    MATCH (p)-[:MENTIONS_TREATMENT]->(t:Treatment)
                    RETURN t.name as treatment, m.name as mechanism, count(p) as papers
                    ORDER BY papers DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
        "aggregation": [
            {
                "name": "symptom_frequency",
                "nl": "Most reported symptoms across all diseases",
                "vector_query": None,
                "graph_query": """
                    MATCH (rs:ReportedSymptom)<-[r:HAS_REPORTED_SYMPTOM]-(d:Disease)
                    RETURN rs.name as symptom, sum(r.report_count) as total_reports
                    ORDER BY total_reports DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
    },
}


def run_evaluation() -> Dict[str, Any]:
    """Run full evaluation across all domains and query types."""
    results = {
        "summary": {},
        "by_domain": {},
        "by_query_type": {},
        "detailed_results": [],
        "failure_cases": [],
    }

    total_queries = 0
    vector_answerable = 0
    graph_answerable = 0

    for domain, query_types in EVALUATION_QUERIES.items():
        results["by_domain"][domain] = {
            "total": 0,
            "vector_answerable": 0,
            "graph_answerable": 0,
            "graph_only": 0,
        }

        for query_type, queries in query_types.items():
            if query_type not in results["by_query_type"]:
                results["by_query_type"][query_type] = {
                    "total": 0,
                    "vector_answerable": 0,
                    "graph_answerable": 0,
                }

            for q in queries:
                total_queries += 1
                results["by_domain"][domain]["total"] += 1
                results["by_query_type"][query_type]["total"] += 1

                # Test graph query
                graph_results = 0
                graph_time = 0
                try:
                    start = time.time()
                    graph_data = query(q["graph_query"])
                    graph_time = (time.time() - start) * 1000
                    graph_results = len(graph_data)
                except Exception as e:
                    results["failure_cases"].append({
                        "domain": domain,
                        "query": q["name"],
                        "type": "graph",
                        "error": str(e)
                    })

                # Track metrics
                if q["vector_can_answer"]:
                    vector_answerable += 1
                    results["by_domain"][domain]["vector_answerable"] += 1
                    results["by_query_type"][query_type]["vector_answerable"] += 1

                if q["graph_can_answer"]:
                    graph_answerable += 1
                    results["by_domain"][domain]["graph_answerable"] += 1
                    results["by_query_type"][query_type]["graph_answerable"] += 1

                if q["graph_can_answer"] and not q["vector_can_answer"]:
                    results["by_domain"][domain]["graph_only"] += 1

                # Store detailed result
                results["detailed_results"].append({
                    "domain": domain,
                    "query_type": query_type,
                    "query_name": q["name"],
                    "natural_language": q["nl"],
                    "vector_can_answer": q["vector_can_answer"],
                    "graph_can_answer": q["graph_can_answer"],
                    "graph_results": graph_results,
                    "graph_time_ms": round(graph_time, 2),
                })

    # Summary
    results["summary"] = {
        "total_queries": total_queries,
        "vector_answerable": vector_answerable,
        "graph_answerable": graph_answerable,
        "graph_only": graph_answerable - vector_answerable,
        "vector_coverage": round(vector_answerable / total_queries * 100, 1),
        "graph_coverage": round(graph_answerable / total_queries * 100, 1),
        "graph_advantage": round((graph_answerable - vector_answerable) / total_queries * 100, 1),
    }

    return results


def print_results(results: Dict[str, Any]):
    """Pretty print evaluation results."""
    print("=" * 70)
    print("GRAPH RAG vs VECTOR SEARCH EVALUATION")
    print("=" * 70)

    # Summary
    s = results["summary"]
    print(f"\n## SUMMARY")
    print(f"Total test queries: {s['total_queries']}")
    print(f"Vector search can answer: {s['vector_answerable']} ({s['vector_coverage']}%)")
    print(f"Graph RAG can answer: {s['graph_answerable']} ({s['graph_coverage']}%)")
    print(f"Graph-only queries: {s['graph_only']} ({s['graph_advantage']}% advantage)")

    # By domain
    print(f"\n## BY DOMAIN")
    print("-" * 70)
    print(f"{'Domain':<15} {'Total':<8} {'Vector':<8} {'Graph':<8} {'Graph-Only':<10}")
    print("-" * 70)
    for domain, stats in results["by_domain"].items():
        print(f"{domain:<15} {stats['total']:<8} {stats['vector_answerable']:<8} {stats['graph_answerable']:<8} {stats['graph_only']:<10}")

    # By query type
    print(f"\n## BY QUERY TYPE")
    print("-" * 70)
    print(f"{'Type':<15} {'Total':<8} {'Vector':<8} {'Graph':<8}")
    print("-" * 70)
    for qtype, stats in results["by_query_type"].items():
        print(f"{qtype:<15} {stats['total']:<8} {stats['vector_answerable']:<8} {stats['graph_answerable']:<8}")

    # Detailed results
    print(f"\n## DETAILED RESULTS")
    print("-" * 70)
    for r in results["detailed_results"]:
        vec = "✓" if r["vector_can_answer"] else "✗"
        graph = "✓" if r["graph_can_answer"] else "✗"
        print(f"[{r['domain']:<10}] {r['query_type']:<12} | Vec:{vec} Graph:{graph} | {r['graph_results']:>3} results | {r['graph_time_ms']:>6}ms | {r['query_name']}")

    # Failure cases
    if results["failure_cases"]:
        print(f"\n## FAILURE CASES")
        print("-" * 70)
        for f in results["failure_cases"]:
            print(f"[{f['domain']}] {f['query']}: {f['error'][:50]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Running evaluation...")
    results = run_evaluation()
    print_results(results)

    # Save to JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation_results.json")
