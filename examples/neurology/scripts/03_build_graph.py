"""
Build the NeuroGraph knowledge graph in Neo4j
"""
import json
from neo4j import GraphDatabase
from tqdm import tqdm

driver = GraphDatabase.driver(
    "bolt://localhost:7688",
    auth=("neo4j", "neurograph2025")
)

def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared")

def create_constraints():
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BrainRegion) REQUIRE b.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Protein) REQUIRE pr.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE"
    ]
    
    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
            except:
                pass
    print("Constraints created")

def normalize(text: str) -> str:
    return text.strip().lower().title()

def build_graph(papers: list):
    with driver.session() as session:
        for paper in tqdm(papers, desc="Building graph"):
            pmid = paper["pmid"]
            entities = paper["entities"]
            
            session.run("""
                MERGE (p:Paper {pmid: $pmid})
                SET p.title = $title, p.year = $year, p.abstract = $abstract
            """, pmid=pmid, title=paper["title"], year=paper.get("year", ""), 
                abstract=paper["abstract"][:500])
            
            for author in paper.get("authors", []):
                if author:
                    session.run("""
                        MERGE (a:Author {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:AUTHORED_BY]->(a)
                    """, name=author, pmid=pmid)
            
            for disease in entities.get("diseases", []):
                if disease:
                    session.run("""
                        MERGE (d:Disease {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_DISEASE]->(d)
                    """, name=normalize(disease), pmid=pmid)
            
            for symptom in entities.get("symptoms", []):
                if symptom:
                    session.run("""
                        MERGE (s:Symptom {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_SYMPTOM]->(s)
                    """, name=normalize(symptom), pmid=pmid)
            
            for region in entities.get("brain_regions", []):
                if region:
                    session.run("""
                        MERGE (b:BrainRegion {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_REGION]->(b)
                    """, name=normalize(region), pmid=pmid)
            
            for protein in entities.get("proteins", []):
                if protein:
                    session.run("""
                        MERGE (pr:Protein {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_PROTEIN]->(pr)
                    """, name=normalize(protein), pmid=pmid)
            
            for gene in entities.get("genes", []):
                if gene:
                    session.run("""
                        MERGE (g:Gene {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_GENE]->(g)
                    """, name=normalize(gene), pmid=pmid)
            
            for mechanism in entities.get("mechanisms", []):
                if mechanism:
                    session.run("""
                        MERGE (m:Mechanism {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_MECHANISM]->(m)
                    """, name=normalize(mechanism), pmid=pmid)
            
            for treatment in entities.get("treatments", []):
                if treatment:
                    session.run("""
                        MERGE (t:Treatment {name: $name})
                        MERGE (p:Paper {pmid: $pmid})
                        MERGE (p)-[:MENTIONS_TREATMENT]->(t)
                    """, name=normalize(treatment), pmid=pmid)

def create_disease_relationships():
    """Create direct relationships between diseases and their attributes.
    NO THRESHOLD - all connections from papers are included."""
    
    print("\nCreating disease relationships (no threshold)...")
    
    with driver.session() as session:
        print("  Disease -> Symptom...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_SYMPTOM]->(s:Symptom)
            WITH d, s, count(p) as papers
            MERGE (d)-[r:HAS_SYMPTOM]->(s)
            SET r.paper_count = papers
        """)
        
        print("  Disease -> BrainRegion...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_REGION]->(b:BrainRegion)
            WITH d, b, count(p) as papers
            MERGE (d)-[r:AFFECTS_REGION]->(b)
            SET r.paper_count = papers
        """)
        
        print("  Disease -> Protein...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_PROTEIN]->(pr:Protein)
            WITH d, pr, count(p) as papers
            MERGE (d)-[r:INVOLVES_PROTEIN]->(pr)
            SET r.paper_count = papers
        """)
        
        print("  Disease -> Gene...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_GENE]->(g:Gene)
            WITH d, g, count(p) as papers
            MERGE (d)-[r:LINKED_TO_GENE]->(g)
            SET r.paper_count = papers
        """)
        
        print("  Disease -> Mechanism...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_MECHANISM]->(m:Mechanism)
            WITH d, m, count(p) as papers
            MERGE (d)-[r:INVOLVES_MECHANISM]->(m)
            SET r.paper_count = papers
        """)
        
        print("  Disease -> Treatment...")
        session.run("""
            MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)-[:MENTIONS_TREATMENT]->(t:Treatment)
            WITH d, t, count(p) as papers
            MERGE (d)-[r:TREATED_BY]->(t)
            SET r.paper_count = papers
        """)
        
        print("  Disease <-> Disease (shared symptoms)...")
        session.run("""
            MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
            WHERE d1 <> d2 AND id(d1) < id(d2)
            WITH d1, d2, count(s) as shared_symptoms
            MERGE (d1)-[r:SHARES_SYMPTOMS_WITH]->(d2)
            SET r.shared_count = shared_symptoms
        """)
    
    print("Disease relationships created")

def print_stats():
    with driver.session() as session:
        stats = {}
        for label in ["Paper", "Disease", "Symptom", "BrainRegion", "Protein", "Gene", "Mechanism", "Treatment", "Author"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as c")
            stats[label] = result.single()["c"]
        
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        stats["Total Relationships"] = rel_count
        
        # Count specific relationship types
        for rel in ["HAS_SYMPTOM", "AFFECTS_REGION", "INVOLVES_PROTEIN", "LINKED_TO_GENE", "INVOLVES_MECHANISM", "TREATED_BY"]:
            result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) as c")
            stats[f"rel_{rel}"] = result.single()["c"]
    
    print("\n" + "=" * 50)
    print("NEUROGRAPH STATISTICS")
    print("=" * 50)
    print("\nNodes:")
    for label in ["Paper", "Disease", "Symptom", "BrainRegion", "Protein", "Gene", "Mechanism", "Treatment", "Author"]:
        print(f"  {label:15} {stats[label]:,}")
    
    print("\nRelationships:")
    for rel in ["HAS_SYMPTOM", "AFFECTS_REGION", "INVOLVES_PROTEIN", "LINKED_TO_GENE", "INVOLVES_MECHANISM", "TREATED_BY"]:
        print(f"  {rel:20} {stats[f'rel_{rel}']:,}")
    
    print(f"\n  {'TOTAL':20} {stats['Total Relationships']:,}")
    print("=" * 50)

def main():
    with open("data/extracted_entities.json", "r") as f:
        papers = json.load(f)
    
    print(f"Building graph from {len(papers)} papers...")
    
    clear_database()
    create_constraints()
    build_graph(papers)
    create_disease_relationships()
    print_stats()
    
    driver.close()

if __name__ == "__main__":
    main()
