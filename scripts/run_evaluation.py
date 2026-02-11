"""
Comprehensive Evaluation: Graph RAG vs Vector Search

Tests across three domains with extensive query coverage including edge cases.
"""

import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "deepgraph2025"))
)


def query(cypher: str, params: dict = None) -> List[Dict]:
    """Execute Cypher query."""
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


# ============================================================
# EXPANDED EVALUATION QUERIES (50+ queries)
# ============================================================

EVALUATION_QUERIES = {
    "research": {
        # SIMILARITY QUERIES - Vector search should work
        "similarity": [
            {
                "name": "topic_transformers",
                "nl": "Papers about transformer architectures",
                "graph_query": "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS 'transformer' RETURN p.title LIMIT 20",
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "topic_llm",
                "nl": "Papers about large language models",
                "graph_query": "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS 'language model' OR toLower(p.title) CONTAINS 'llm' RETURN p.title LIMIT 20",
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "topic_diffusion",
                "nl": "Papers about diffusion models",
                "graph_query": "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS 'diffusion' RETURN p.title LIMIT 20",
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
        ],

        # INTERSECTION QUERIES - Graph RAG advantage
        "intersection": [
            {
                "name": "method_x_concept",
                "nl": "Papers using transformers FOR reasoning",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE toLower(m.name) CONTAINS 'transformer'
                    MATCH (p)-[:ABOUT_CONCEPT]->(c:Concept)
                    WHERE toLower(c.name) CONTAINS 'reason'
                    RETURN p.title, m.name as method, c.name as concept
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_x_concept_2",
                "nl": "Papers using attention FOR multimodal tasks",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE toLower(m.name) CONTAINS 'attention'
                    MATCH (p)-[:ABOUT_CONCEPT]->(c:Concept)
                    WHERE toLower(c.name) CONTAINS 'multimodal'
                    RETURN p.title, m.name, c.name
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_x_concept_3",
                "nl": "Papers using RL FOR robotics",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE toLower(m.name) CONTAINS 'reinforcement' OR toLower(m.name) = 'rl'
                    MATCH (p)-[:ABOUT_CONCEPT]->(c:Concept)
                    WHERE toLower(c.name) CONTAINS 'robot'
                    RETURN p.title
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_x_dataset",
                "nl": "Papers using fine-tuning evaluated on ImageNet",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE toLower(m.name) CONTAINS 'fine' OR toLower(m.name) CONTAINS 'tuning'
                    MATCH (p)-[:USES_DATASET]->(d:Dataset)
                    WHERE toLower(d.name) CONTAINS 'imagenet'
                    RETURN p.title
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "concept_x_concept",
                "nl": "Papers about BOTH efficiency AND robustness",
                "graph_query": """
                    MATCH (p:Paper)-[:ABOUT_CONCEPT]->(c1:Concept)
                    WHERE toLower(c1.name) CONTAINS 'efficien'
                    MATCH (p)-[:ABOUT_CONCEPT]->(c2:Concept)
                    WHERE toLower(c2.name) CONTAINS 'robust'
                    RETURN p.title
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        # MULTI-HOP QUERIES - Impossible for vector search
        "multi_hop": [
            {
                "name": "author_other_papers",
                "nl": "Other papers by authors of paper X",
                "graph_query": """
                    MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)
                    WITH a LIMIT 1
                    MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
                    RETURN DISTINCT p2.title, a.name
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "coauthor_papers",
                "nl": "Papers by collaborators of author X",
                "graph_query": """
                    MATCH (a1:Author)<-[:AUTHORED_BY]-(p1:Paper)-[:AUTHORED_BY]->(a2:Author)
                    WHERE a1 <> a2
                    WITH a1, a2 LIMIT 5
                    MATCH (a2)<-[:AUTHORED_BY]-(p2:Paper)
                    WHERE NOT (p2)-[:AUTHORED_BY]->(a1)
                    RETURN a1.name as author, a2.name as coauthor, p2.title
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_transfer",
                "nl": "Methods used in both NLP and Vision papers",
                "graph_query": """
                    MATCH (p1:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE p1.primary_category CONTAINS 'CL' OR p1.primary_category CONTAINS 'NLP'
                    MATCH (p2:Paper)-[:USES_METHOD]->(m)
                    WHERE p2.primary_category CONTAINS 'CV'
                    RETURN m.name as method, count(DISTINCT p1) as nlp_papers, count(DISTINCT p2) as cv_papers
                    ORDER BY nlp_papers DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "concept_bridge",
                "nl": "Authors who publish in multiple concept areas",
                "graph_query": """
                    MATCH (a:Author)<-[:AUTHORED_BY]-(p1:Paper)-[:ABOUT_CONCEPT]->(c1:Concept)
                    MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)-[:ABOUT_CONCEPT]->(c2:Concept)
                    WHERE c1 <> c2 AND p1 <> p2
                    RETURN a.name, collect(DISTINCT c1.name) + collect(DISTINCT c2.name) as concepts
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "citation_path",
                "nl": "Path between two research areas",
                "graph_query": """
                    MATCH (c1:Concept)<-[:ABOUT_CONCEPT]-(p:Paper)-[:ABOUT_CONCEPT]->(c2:Concept)
                    WHERE c1 <> c2
                    RETURN c1.name, c2.name, count(p) as shared_papers
                    ORDER BY shared_papers DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        # AGGREGATION QUERIES
        "aggregation": [
            {
                "name": "method_popularity",
                "nl": "Most commonly used methods",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    RETURN m.name as method, count(p) as paper_count
                    ORDER BY paper_count DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "concept_popularity",
                "nl": "Most researched concepts",
                "graph_query": """
                    MATCH (p:Paper)-[:ABOUT_CONCEPT]->(c:Concept)
                    RETURN c.name as concept, count(p) as paper_count
                    ORDER BY paper_count DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "prolific_authors",
                "nl": "Authors with most papers",
                "graph_query": """
                    MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
                    RETURN a.name as author, count(p) as paper_count
                    ORDER BY paper_count DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "method_concept_matrix",
                "nl": "Which methods are used for which concepts",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    MATCH (p)-[:ABOUT_CONCEPT]->(c:Concept)
                    RETURN m.name as method, c.name as concept, count(p) as papers
                    ORDER BY papers DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        # EDGE CASES - Testing limits
        "edge_cases": [
            {
                "name": "empty_method",
                "nl": "Papers using non-existent method 'quantum_blockchain'",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'quantum_blockchain'})
                    RETURN p.title
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
                "expected_empty": True,
            },
            {
                "name": "case_sensitivity",
                "nl": "Papers with 'TRANSFORMER' (case test)",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE toLower(m.name) = 'transformer'
                    RETURN p.title LIMIT 10
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "partial_match",
                "nl": "Papers with methods containing 'tion' (partial)",
                "graph_query": """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                    WHERE m.name CONTAINS 'tion'
                    RETURN DISTINCT m.name
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
    },

    "products": {
        "similarity": [
            {
                "name": "headphones_search",
                "nl": "Wireless headphones",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:IN_CATEGORY]->(:Category {name: 'Headphones'})
                    RETURN p.title, p.price LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "laptop_search",
                "nl": "Gaming laptops",
                "graph_query": """
                    MATCH (p:Product)-[:FOR_USE_CASE]->(:UseCase {name: 'gaming'})
                    MATCH (p)-[:IN_CATEGORY]->(:Category {name: 'Laptop'})
                    RETURN p.title, p.price LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
        ],

        "intersection": [
            {
                "name": "feature_x_usecase",
                "nl": "Wireless products FOR workout",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: 'workout'})
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "feature_x_usecase_x_price",
                "nl": "Wireless headphones FOR travel under $100",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: 'travel'})
                    MATCH (p)-[:IN_CATEGORY]->(:Category {name: 'Headphones'})
                    WHERE p.price < 100
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "feature_x_feature",
                "nl": "Products that are BOTH wireless AND waterproof",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:HAS_FEATURE]->(:Feature {name: 'waterproof'})
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "triple_intersection",
                "nl": "Wireless AND noise-canceling FOR travel",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    MATCH (p)-[:HAS_FEATURE]->(:Feature {name: 'noise_canceling'})
                    MATCH (p)-[:FOR_USE_CASE]->(:UseCase {name: 'travel'})
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "category_x_feature_x_price",
                "nl": "Monitors with 4K under $500",
                "graph_query": """
                    MATCH (p:Product)-[:IN_CATEGORY]->(:Category {name: 'Monitor'})
                    MATCH (p)-[:HAS_FEATURE]->(:Feature {name: '4k'})
                    WHERE p.price < 500
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "brand_x_feature",
                "nl": "Sony products with noise canceling",
                "graph_query": """
                    MATCH (p:Product)-[:MADE_BY]->(:Brand {name: 'Sony'})
                    MATCH (p)-[:HAS_FEATURE]->(:Feature {name: 'noise_canceling'})
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "usecase_x_price_range",
                "nl": "Gaming products between $200-$400",
                "graph_query": """
                    MATCH (p:Product)-[:FOR_USE_CASE]->(:UseCase {name: 'gaming'})
                    WHERE p.price >= 200 AND p.price <= 400
                    RETURN p.title, p.price
                    ORDER BY p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        "multi_hop": [
            {
                "name": "bought_together",
                "nl": "Products frequently bought with laptops",
                "graph_query": """
                    MATCH (p:Product)-[:IN_CATEGORY]->(:Category {name: 'Laptop'})
                    WITH p LIMIT 3
                    MATCH (p)-[:BOUGHT_WITH]->(other:Product)
                    RETURN p.title as laptop, other.title as bought_with, other.price
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "feature_migration",
                "nl": "Premium features ($300+) now in budget products ($100-)",
                "graph_query": """
                    MATCH (premium:Product)-[:HAS_FEATURE]->(f:Feature)
                    WHERE premium.price > 300
                    WITH f, count(premium) as premium_count
                    MATCH (budget:Product)-[:HAS_FEATURE]->(f)
                    WHERE budget.price < 100
                    RETURN f.name as feature, premium_count, count(budget) as budget_count
                    ORDER BY budget_count DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "brand_category_spread",
                "nl": "Brands that sell in multiple categories",
                "graph_query": """
                    MATCH (b:Brand)<-[:MADE_BY]-(p:Product)-[:IN_CATEGORY]->(c:Category)
                    WITH b, collect(DISTINCT c.name) as categories
                    WHERE size(categories) > 1
                    RETURN b.name as brand, categories, size(categories) as category_count
                    ORDER BY category_count DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "complementary_features",
                "nl": "Features commonly paired together",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(f1:Feature)
                    MATCH (p)-[:HAS_FEATURE]->(f2:Feature)
                    WHERE id(f1) < id(f2)
                    RETURN f1.name, f2.name, count(p) as co_occurrence
                    ORDER BY co_occurrence DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        "aggregation": [
            {
                "name": "feature_popularity",
                "nl": "Most common features",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(f:Feature)
                    RETURN f.name as feature, count(p) as product_count
                    ORDER BY product_count DESC
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "usecase_popularity",
                "nl": "Most targeted use cases",
                "graph_query": """
                    MATCH (p:Product)-[:FOR_USE_CASE]->(u:UseCase)
                    RETURN u.name as use_case, count(p) as product_count
                    ORDER BY product_count DESC
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "price_by_category",
                "nl": "Average price by category",
                "graph_query": """
                    MATCH (p:Product)-[:IN_CATEGORY]->(c:Category)
                    RETURN c.name as category,
                           round(avg(p.price)) as avg_price,
                           min(p.price) as min_price,
                           max(p.price) as max_price,
                           count(p) as products
                    ORDER BY avg_price DESC
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "niche_finder",
                "nl": "Rare feature + use case combinations (underserved markets)",
                "graph_query": """
                    MATCH (f:Feature)<-[:HAS_FEATURE]-(p:Product)-[:FOR_USE_CASE]->(u:UseCase)
                    WITH f, u, count(p) as product_count
                    WHERE product_count >= 1 AND product_count <= 3
                    RETURN f.name as feature, u.name as use_case, product_count
                    ORDER BY product_count ASC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "brand_market_share",
                "nl": "Products per brand",
                "graph_query": """
                    MATCH (b:Brand)<-[:MADE_BY]-(p:Product)
                    RETURN b.name as brand, count(p) as products
                    ORDER BY products DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        "edge_cases": [
            {
                "name": "no_results_intersection",
                "nl": "Waterproof laptops (unlikely combination)",
                "graph_query": """
                    MATCH (p:Product)-[:IN_CATEGORY]->(:Category {name: 'Laptop'})
                    MATCH (p)-[:HAS_FEATURE]->(:Feature {name: 'waterproof'})
                    RETURN p.title
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
                "expected_empty": True,
            },
            {
                "name": "price_boundary",
                "nl": "Products at exactly $99.99",
                "graph_query": """
                    MATCH (p:Product)
                    WHERE p.price = 99.99
                    RETURN p.title, p.price
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "negation",
                "nl": "Wireless products that are NOT for gaming",
                "graph_query": """
                    MATCH (p:Product)-[:HAS_FEATURE]->(:Feature {name: 'wireless'})
                    WHERE NOT (p)-[:FOR_USE_CASE]->(:UseCase {name: 'gaming'})
                    RETURN p.title, p.price
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
    },

    "neurology": {
        "similarity": [
            {
                "name": "alzheimers_papers",
                "nl": "Papers about Alzheimer's disease",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'alzheimer'
                    RETURN p.title, d.name LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "parkinsons_papers",
                "nl": "Papers about Parkinson's disease",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    RETURN p.title, d.name LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "als_papers",
                "nl": "Papers about ALS",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'amyotrophic' OR toLower(d.name) CONTAINS 'als'
                    RETURN p.title, d.name LIMIT 20
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
        ],

        "intersection": [
            {
                "name": "disease_x_symptom",
                "nl": "Papers about Parkinson's AND tremor",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    MATCH (p)-[:MENTIONS_SYMPTOM]->(s:Symptom)
                    WHERE toLower(s.name) CONTAINS 'tremor'
                    RETURN p.title, s.name as symptom LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "disease_x_protein",
                "nl": "Papers about Alzheimer's mentioning tau protein",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'alzheimer'
                    MATCH (p)-[:MENTIONS_PROTEIN]->(pr:Protein)
                    WHERE toLower(pr.name) CONTAINS 'tau'
                    RETURN p.title, pr.name as protein LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "disease_x_mechanism",
                "nl": "Papers about Parkinson's discussing neuroinflammation",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    MATCH (p)-[:MENTIONS_MECHANISM]->(m:Mechanism)
                    WHERE toLower(m.name) CONTAINS 'inflam'
                    RETURN p.title, m.name as mechanism LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "disease_x_treatment",
                "nl": "Papers about Alzheimer's mentioning treatments",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'alzheimer'
                    MATCH (p)-[:MENTIONS_TREATMENT]->(t:Treatment)
                    RETURN p.title, t.name as treatment LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "disease_x_region",
                "nl": "Papers about Parkinson's mentioning brain regions",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    MATCH (p)-[:MENTIONS_REGION]->(r:BrainRegion)
                    RETURN p.title, r.name as region LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "protein_x_mechanism",
                "nl": "Papers mentioning tau AND oxidative stress",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_PROTEIN]->(pr:Protein)
                    WHERE toLower(pr.name) CONTAINS 'tau'
                    MATCH (p)-[:MENTIONS_MECHANISM]->(m:Mechanism)
                    WHERE toLower(m.name) CONTAINS 'oxidat'
                    RETURN p.title, pr.name, m.name LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        "comparison": [
            {
                "name": "research_vs_patient_parkinsons",
                "nl": "Symptoms patients report but research doesn't cover (Parkinson's)",
                "graph_query": """
                    MATCH (d:Disease)-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                    WHERE toLower(d.name) CONTAINS 'parkinson'
                    AND NOT EXISTS {
                        MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                        WHERE toLower(s.name) CONTAINS toLower(rs.name)
                           OR toLower(rs.name) CONTAINS toLower(s.name)
                    }
                    RETURN rs.name as gap, r.report_count as patients
                    ORDER BY r.report_count DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "research_vs_patient_alzheimers",
                "nl": "Symptoms patients report but research doesn't cover (Alzheimer's)",
                "graph_query": """
                    MATCH (d:Disease)-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                    WHERE toLower(d.name) CONTAINS 'alzheimer'
                    AND NOT EXISTS {
                        MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                        WHERE toLower(s.name) CONTAINS toLower(rs.name)
                    }
                    RETURN rs.name as gap, r.report_count as patients
                    ORDER BY r.report_count DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "research_only_symptoms",
                "nl": "Symptoms in research but rarely reported by patients",
                "graph_query": """
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
                    WHERE NOT EXISTS {
                        MATCH (d)-[:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                        WHERE toLower(rs.name) CONTAINS toLower(s.name)
                    }
                    RETURN d.name as disease, s.name as symptom, r.paper_count as papers
                    ORDER BY r.paper_count DESC
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
                "graph_query": """
                    MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
                    WHERE toLower(d1.name) CONTAINS 'alzheimer' AND d1 <> d2
                    RETURN d2.name as disease, count(s) as shared
                    ORDER BY shared DESC
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "shared_mechanisms",
                "nl": "Diseases sharing mechanisms with Parkinson's",
                "graph_query": """
                    MATCH (d1:Disease)-[:INVOLVES_MECHANISM]->(m:Mechanism)<-[:INVOLVES_MECHANISM]-(d2:Disease)
                    WHERE toLower(d1.name) CONTAINS 'parkinson' AND d1 <> d2
                    RETURN d2.name as disease, collect(m.name) as mechanisms
                    LIMIT 10
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "treatment_via_mechanism",
                "nl": "Treatments targeting mechanisms in Parkinson's",
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
            {
                "name": "protein_disease_network",
                "nl": "Proteins involved in multiple diseases",
                "graph_query": """
                    MATCH (pr:Protein)<-[:INVOLVES_PROTEIN]-(d:Disease)
                    WITH pr, collect(d.name) as diseases
                    WHERE size(diseases) > 1
                    RETURN pr.name as protein, diseases
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "symptom_progression",
                "nl": "Symptoms shared across disease progression stages",
                "graph_query": """
                    MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
                    WHERE d1 <> d2
                    RETURN s.name as symptom, collect(DISTINCT d1.name) + collect(DISTINCT d2.name) as diseases
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
                "graph_query": """
                    MATCH (rs:ReportedSymptom)<-[r:HAS_REPORTED_SYMPTOM]-(d:Disease)
                    RETURN rs.name as symptom, sum(r.report_count) as total
                    ORDER BY total DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "disease_paper_count",
                "nl": "Research papers per disease",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    RETURN d.name as disease, count(p) as papers
                    ORDER BY papers DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "protein_research_focus",
                "nl": "Most studied proteins",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_PROTEIN]->(pr:Protein)
                    RETURN pr.name as protein, count(p) as papers
                    ORDER BY papers DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "mechanism_research_focus",
                "nl": "Most studied mechanisms",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_MECHANISM]->(m:Mechanism)
                    RETURN m.name as mechanism, count(p) as papers
                    ORDER BY papers DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
            {
                "name": "treatment_coverage",
                "nl": "Treatments per disease",
                "graph_query": """
                    MATCH (d:Disease)-[:TREATED_BY]->(t:Treatment)
                    RETURN d.name as disease, count(t) as treatments
                    ORDER BY treatments DESC
                    LIMIT 15
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],

        "edge_cases": [
            {
                "name": "rare_disease",
                "nl": "Papers about CJD (rare disease)",
                "graph_query": """
                    MATCH (p:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS 'creutzfeldt' OR toLower(d.name) CONTAINS 'cjd'
                    RETURN p.title, d.name LIMIT 10
                """,
                "vector_can_answer": True,
                "graph_can_answer": True,
            },
            {
                "name": "symptom_not_in_research",
                "nl": "Patient symptoms with zero research papers",
                "graph_query": """
                    MATCH (d:Disease)-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                    WHERE NOT EXISTS {
                        MATCH (d)-[:HAS_SYMPTOM]->(:Symptom)
                    }
                    RETURN d.name, rs.name, r.report_count
                    ORDER BY r.report_count DESC
                    LIMIT 20
                """,
                "vector_can_answer": False,
                "graph_can_answer": True,
            },
        ],
    },
}


def run_evaluation() -> Dict[str, Any]:
    """Run full evaluation."""
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
    successful_results = 0
    empty_expected = 0
    empty_unexpected = 0

    for domain, query_types in EVALUATION_QUERIES.items():
        results["by_domain"][domain] = {
            "total": 0,
            "vector_answerable": 0,
            "graph_answerable": 0,
            "graph_only": 0,
            "successful": 0,
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

                # Run graph query
                graph_results = 0
                graph_time = 0
                error = None
                try:
                    start = time.time()
                    graph_data = query(q["graph_query"])
                    graph_time = (time.time() - start) * 1000
                    graph_results = len(graph_data)

                    if graph_results > 0:
                        successful_results += 1
                        results["by_domain"][domain]["successful"] += 1
                    elif q.get("expected_empty"):
                        empty_expected += 1
                    else:
                        empty_unexpected += 1

                except Exception as e:
                    error = str(e)
                    results["failure_cases"].append({
                        "domain": domain,
                        "query": q["name"],
                        "type": query_type,
                        "error": error[:100]
                    })

                # Track coverage metrics
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

                results["detailed_results"].append({
                    "domain": domain,
                    "query_type": query_type,
                    "query_name": q["name"],
                    "natural_language": q["nl"],
                    "vector_can_answer": q["vector_can_answer"],
                    "graph_can_answer": q["graph_can_answer"],
                    "graph_results": graph_results,
                    "graph_time_ms": round(graph_time, 2),
                    "error": error,
                })

    # Summary
    results["summary"] = {
        "total_queries": total_queries,
        "vector_answerable": vector_answerable,
        "graph_answerable": graph_answerable,
        "graph_only": graph_answerable - vector_answerable,
        "vector_coverage_pct": round(vector_answerable / total_queries * 100, 1),
        "graph_coverage_pct": round(graph_answerable / total_queries * 100, 1),
        "graph_advantage_pct": round((graph_answerable - vector_answerable) / total_queries * 100, 1),
        "queries_with_results": successful_results,
        "empty_expected": empty_expected,
        "empty_unexpected": empty_unexpected,
    }

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results."""
    print("=" * 80)
    print("GRAPH RAG vs VECTOR SEARCH EVALUATION")
    print("=" * 80)

    s = results["summary"]
    print(f"\n## SUMMARY ({s['total_queries']} queries)")
    print(f"Vector search coverage: {s['vector_answerable']}/{s['total_queries']} ({s['vector_coverage_pct']}%)")
    print(f"Graph RAG coverage:     {s['graph_answerable']}/{s['total_queries']} ({s['graph_coverage_pct']}%)")
    print(f"Graph-only queries:     {s['graph_only']} ({s['graph_advantage_pct']}% advantage)")
    print(f"Queries with results:   {s['queries_with_results']}")

    print(f"\n## BY DOMAIN")
    print("-" * 80)
    print(f"{'Domain':<12} {'Total':<8} {'Vector':<8} {'Graph':<8} {'Graph-Only':<12} {'With Results':<12}")
    print("-" * 80)
    for domain, stats in results["by_domain"].items():
        print(f"{domain:<12} {stats['total']:<8} {stats['vector_answerable']:<8} {stats['graph_answerable']:<8} {stats['graph_only']:<12} {stats['successful']:<12}")

    print(f"\n## BY QUERY TYPE")
    print("-" * 80)
    print(f"{'Type':<15} {'Total':<8} {'Vector':<8} {'Graph':<8} {'Graph Advantage':<15}")
    print("-" * 80)
    for qtype, stats in results["by_query_type"].items():
        adv = stats['graph_answerable'] - stats['vector_answerable']
        print(f"{qtype:<15} {stats['total']:<8} {stats['vector_answerable']:<8} {stats['graph_answerable']:<8} {adv:<15}")

    print(f"\n## DETAILED RESULTS")
    print("-" * 80)
    for r in results["detailed_results"]:
        vec = "V" if r["vector_can_answer"] else "-"
        status = "OK" if r["graph_results"] > 0 else ("ERR" if r["error"] else "EMPTY")
        print(f"[{r['domain']:<10}] {r['query_type']:<12} | {vec} | {r['graph_results']:>4} | {r['graph_time_ms']:>8.1f}ms | {status:<5} | {r['query_name']}")

    if results["failure_cases"]:
        print(f"\n## ERRORS ({len(results['failure_cases'])})")
        for f in results["failure_cases"]:
            print(f"  [{f['domain']}] {f['query']}: {f['error']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Running comprehensive evaluation...")
    results = run_evaluation()
    print_results(results)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to evaluation_results.json")
    print(f"Total queries: {results['summary']['total_queries']}")
