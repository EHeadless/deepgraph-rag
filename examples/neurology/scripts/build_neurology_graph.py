"""
Build the complete Neurology Knowledge Graph in Neo4j

Combines:
- Research data (PubMed papers with extracted entities)
- Patient data (Reddit posts with reported symptoms)

Run from the repository root:
    python examples/neurology/scripts/build_neurology_graph.py
"""

import json
import os
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "neurology"

# Neo4j connection (same as other examples)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "deepgraph2025")
    )
)


def normalize(text: str) -> str:
    """Normalize text for consistent node names."""
    if isinstance(text, dict):
        text = str(text)
    return text.strip().lower().title()[:200]


def create_constraints():
    """Create uniqueness constraints for all node types."""
    constraints = [
        # Research nodes
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BrainRegion) REQUIRE b.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Protein) REQUIRE pr.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        # Patient nodes
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RedditPost) REQUIRE r.post_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (rs:ReportedSymptom) REQUIRE rs.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (li:LifeImpact) REQUIRE li.name IS UNIQUE",
    ]

    print("Creating constraints...")
    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                pass  # Constraint may already exist
    print("  ✓ Constraints created")


def build_research_graph(papers: list):
    """Build graph from PubMed papers with extracted entities."""
    print(f"\nBuilding research graph from {len(papers)} papers...")

    with driver.session() as session:
        for paper in tqdm(papers, desc="  Papers"):
            pmid = paper["pmid"]
            entities = paper.get("entities", {})

            # Create Paper node
            session.run("""
                MERGE (p:Paper {pmid: $pmid})
                SET p.title = $title,
                    p.year = $year,
                    p.abstract = $abstract,
                    p.disease_query = $disease_query
            """, pmid=pmid, title=paper["title"], year=paper.get("year", ""),
                abstract=paper.get("abstract", "")[:500],
                disease_query=paper.get("disease_query", ""))

            # Authors
            for author in paper.get("authors", []):
                if author:
                    session.run("""
                        MERGE (a:Author {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:AUTHORED_BY]->(a)
                    """, name=author, pmid=pmid)

            # Diseases
            for disease in entities.get("diseases", []):
                if disease:
                    session.run("""
                        MERGE (d:Disease {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_DISEASE]->(d)
                    """, name=normalize(disease), pmid=pmid)

            # Symptoms
            for symptom in entities.get("symptoms", []):
                if symptom:
                    session.run("""
                        MERGE (s:Symptom {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_SYMPTOM]->(s)
                    """, name=normalize(symptom), pmid=pmid)

            # Brain regions
            for region in entities.get("brain_regions", []):
                if region:
                    session.run("""
                        MERGE (b:BrainRegion {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_REGION]->(b)
                    """, name=normalize(region), pmid=pmid)

            # Proteins
            for protein in entities.get("proteins", []):
                if protein:
                    session.run("""
                        MERGE (pr:Protein {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_PROTEIN]->(pr)
                    """, name=normalize(protein), pmid=pmid)

            # Genes
            for gene in entities.get("genes", []):
                if gene:
                    session.run("""
                        MERGE (g:Gene {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_GENE]->(g)
                    """, name=normalize(gene), pmid=pmid)

            # Mechanisms
            for mechanism in entities.get("mechanisms", []):
                if mechanism:
                    session.run("""
                        MERGE (m:Mechanism {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_MECHANISM]->(m)
                    """, name=normalize(mechanism), pmid=pmid)

            # Treatments
            for treatment in entities.get("treatments", []):
                if treatment:
                    session.run("""
                        MERGE (t:Treatment {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_TREATMENT]->(t)
                    """, name=normalize(treatment), pmid=pmid)

    print("  ✓ Research graph built")


def build_patient_graph(symptoms_data: list):
    """Build graph from Reddit patient reports."""
    print(f"\nBuilding patient graph from {len(symptoms_data)} items...")

    with driver.session() as session:
        for item in tqdm(symptoms_data, desc="  Reddit items"):
            post_id = item.get("post_id")
            disease = item.get("disease", "")

            if not post_id:
                continue

            # Create RedditPost node
            session.run("""
                MERGE (r:RedditPost {post_id: $post_id})
                SET r.title = $title,
                    r.url = $url,
                    r.subreddit = $subreddit,
                    r.disease = $disease,
                    r.source = $source
            """, post_id=post_id, title=item.get("title", "")[:200],
                url=item.get("url", ""), subreddit=item.get("subreddit", ""),
                disease=disease, source=item.get("source", "post"))

            # Link to disease (fuzzy match)
            if disease:
                session.run("""
                    MATCH (d:Disease)
                    WHERE toLower(d.name) CONTAINS toLower($disease_part)
                    WITH d LIMIT 1
                    MATCH (r:RedditPost {post_id: $post_id})
                    MERGE (r)-[:DISCUSSES]->(d)
                """, disease_part=disease.split()[0], post_id=post_id)

            # Reported symptoms
            symptoms = item.get("symptoms", {})
            all_symptoms = (
                symptoms.get("physical_symptoms", []) +
                symptoms.get("cognitive_symptoms", []) +
                symptoms.get("behavioral_symptoms", []) +
                symptoms.get("psychiatric_symptoms", [])
            )

            for sym in all_symptoms:
                if sym and isinstance(sym, str) and len(sym) > 2:
                    session.run("""
                        MERGE (rs:ReportedSymptom {name: $name})
                        WITH rs
                        MATCH (r:RedditPost {post_id: $post_id})
                        MERGE (r)-[:REPORTS_SYMPTOM]->(rs)
                    """, name=normalize(sym), post_id=post_id)

            # Daily impacts
            for impact in item.get("daily_impacts", []):
                if impact and isinstance(impact, str) and len(impact) > 2:
                    session.run("""
                        MERGE (li:LifeImpact {name: $name})
                        WITH li
                        MATCH (r:RedditPost {post_id: $post_id})
                        MERGE (r)-[:REPORTS_IMPACT]->(li)
                    """, name=normalize(impact), post_id=post_id)

    print("  ✓ Patient graph built")


def create_aggregated_relationships():
    """Create direct Disease → Entity relationships with counts."""
    print("\nCreating aggregated relationships...")

    with driver.session() as session:
        # Disease → Symptom (from research)
        print("  Disease → Symptom...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_SYMPTOM]->(s:Symptom)
            WITH d, s, count(p) as papers
            MERGE (d)-[r:HAS_SYMPTOM]->(s)
            SET r.paper_count = papers
        """)

        # Disease → BrainRegion
        print("  Disease → BrainRegion...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_REGION]->(b:BrainRegion)
            WITH d, b, count(p) as papers
            MERGE (d)-[r:AFFECTS_REGION]->(b)
            SET r.paper_count = papers
        """)

        # Disease → Protein
        print("  Disease → Protein...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_PROTEIN]->(pr:Protein)
            WITH d, pr, count(p) as papers
            MERGE (d)-[r:INVOLVES_PROTEIN]->(pr)
            SET r.paper_count = papers
        """)

        # Disease → Gene
        print("  Disease → Gene...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_GENE]->(g:Gene)
            WITH d, g, count(p) as papers
            MERGE (d)-[r:LINKED_TO_GENE]->(g)
            SET r.paper_count = papers
        """)

        # Disease → Mechanism
        print("  Disease → Mechanism...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_MECHANISM]->(m:Mechanism)
            WITH d, m, count(p) as papers
            MERGE (d)-[r:INVOLVES_MECHANISM]->(m)
            SET r.paper_count = papers
        """)

        # Disease → Treatment
        print("  Disease → Treatment...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_TREATMENT]->(t:Treatment)
            WITH d, t, count(p) as papers
            MERGE (d)-[r:TREATED_BY]->(t)
            SET r.paper_count = papers
        """)

        # Disease ↔ Disease (shared symptoms)
        print("  Disease ↔ Disease (shared symptoms)...")
        session.run("""
            MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
            WHERE d1 <> d2 AND id(d1) < id(d2)
            WITH d1, d2, count(s) as shared_symptoms
            MERGE (d1)-[r:SHARES_SYMPTOMS_WITH]->(d2)
            SET r.shared_count = shared_symptoms
        """)

        # Disease → ReportedSymptom (from patients)
        print("  Disease → ReportedSymptom (patient data)...")
        session.run("""
            MATCH (d:Disease)<-[:DISCUSSES]-(r:RedditPost)-[:REPORTS_SYMPTOM]->(rs:ReportedSymptom)
            WITH d, rs, count(r) as report_count
            MERGE (d)-[rel:HAS_REPORTED_SYMPTOM]->(rs)
            SET rel.report_count = report_count
        """)

        # Disease → LifeImpact
        print("  Disease → LifeImpact...")
        session.run("""
            MATCH (d:Disease)<-[:DISCUSSES]-(r:RedditPost)-[:REPORTS_IMPACT]->(li:LifeImpact)
            WITH d, li, count(r) as report_count
            MERGE (d)-[rel:HAS_LIFE_IMPACT]->(li)
            SET rel.report_count = report_count
        """)

    print("  ✓ Aggregated relationships created")


def print_stats():
    """Print graph statistics."""
    print("\n" + "=" * 60)
    print("NEUROLOGY KNOWLEDGE GRAPH STATISTICS")
    print("=" * 60)

    with driver.session() as session:
        stats = {}

        # Node counts
        print("\n📊 Nodes:")
        node_labels = [
            "Paper", "Disease", "Symptom", "BrainRegion", "Protein",
            "Gene", "Mechanism", "Treatment", "Author",
            "RedditPost", "ReportedSymptom", "LifeImpact"
        ]

        for label in node_labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as c")
            count = result.single()["c"]
            stats[label] = count
            print(f"  {label:20} {count:,}")

        # Relationship counts
        print("\n🔗 Relationships:")
        rel_types = [
            "MENTIONS_DISEASE", "MENTIONS_SYMPTOM", "MENTIONS_PROTEIN",
            "HAS_SYMPTOM", "INVOLVES_PROTEIN", "TREATED_BY",
            "SHARES_SYMPTOMS_WITH",
            "DISCUSSES", "REPORTS_SYMPTOM", "HAS_REPORTED_SYMPTOM"
        ]

        for rel in rel_types:
            result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) as c")
            count = result.single()["c"]
            print(f"  {rel:25} {count:,}")

        # Total
        result = session.run("MATCH ()-[r]->() RETURN count(r) as c")
        total = result.single()["c"]
        print(f"\n  {'TOTAL':25} {total:,}")

    print("=" * 60)


def main():
    print("=" * 60)
    print("NEUROLOGY KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)

    # Check for data files
    research_file = DATA_DIR / "extracted_entities.json"
    patient_file = DATA_DIR / "reddit_symptoms_complete.json"

    if not research_file.exists():
        print(f"\n❌ Research data not found: {research_file}")
        print("   Run the extraction scripts first.")
        return

    if not patient_file.exists():
        print(f"\n⚠️  Patient data not found: {patient_file}")
        print("   Continuing with research data only...")
        patient_data = []
    else:
        with open(patient_file, "r") as f:
            patient_data = json.load(f)
        print(f"\n✓ Loaded {len(patient_data):,} patient reports")

    # Load research data
    with open(research_file, "r") as f:
        research_data = json.load(f)
    print(f"✓ Loaded {len(research_data):,} research papers")

    # Build graph
    print("\n" + "-" * 60)

    create_constraints()
    build_research_graph(research_data)

    if patient_data:
        build_patient_graph(patient_data)

    create_aggregated_relationships()
    print_stats()

    driver.close()
    print("\n✅ Neurology knowledge graph built successfully!")
    print("\nRun the app with:")
    print("  streamlit run examples/neurology/app.py --server.port 8507")


if __name__ == "__main__":
    main()
