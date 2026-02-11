"""
Add Reddit-reported symptoms to NeuroGraph
"""
import json
from neo4j import GraphDatabase
from tqdm import tqdm
from collections import defaultdict

driver = GraphDatabase.driver(
    "bolt://localhost:7688",
    auth=("neo4j", "neurograph2025")
)

def normalize(text):
    if isinstance(text, dict):
        text = str(text)
    return text.strip().lower().title()[:200]  # Limit length

def main():
    with open("data/reddit_symptoms_complete.json", "r") as f:
        data = json.load(f)
    
    print(f"Adding {len(data)} Reddit items to graph...")
    
    with driver.session() as session:
        # Create constraints
        try:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:RedditPost) REQUIRE r.post_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (rs:ReportedSymptom) REQUIRE rs.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (li:LifeImpact) REQUIRE li.name IS UNIQUE")
        except:
            pass
        
        for item in tqdm(data):
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
            
            # Link to disease
            if disease:
                session.run("""
                    MATCH (d:Disease)
                    WHERE toLower(d.name) CONTAINS toLower($disease_part)
                    WITH d LIMIT 1
                    MATCH (r:RedditPost {post_id: $post_id})
                    MERGE (r)-[:DISCUSSES]->(d)
                """, disease_part=disease.split()[0], post_id=post_id)
            
            # Add symptoms
            symptoms = item.get("symptoms", {})
            all_symptoms = (
                symptoms.get("physical_symptoms", []) +
                symptoms.get("cognitive_symptoms", []) +
                symptoms.get("behavioral_symptoms", []) +
                symptoms.get("psychiatric_symptoms", [])
            )
            
            for sym in all_symptoms:
                if sym and (isinstance(sym, str) and len(sym) > 2):
                    session.run("""
                        MERGE (rs:ReportedSymptom {name: $name})
                        WITH rs
                        MATCH (r:RedditPost {post_id: $post_id})
                        MERGE (r)-[:REPORTS_SYMPTOM]->(rs)
                    """, name=normalize(sym), post_id=post_id)
            
            # Add daily impacts
            for impact in item.get("daily_impacts", []):
                if impact and (isinstance(impact, str) and len(impact) > 2):
                    session.run("""
                        MERGE (li:LifeImpact {name: $name})
                        WITH li
                        MATCH (r:RedditPost {post_id: $post_id})
                        MERGE (r)-[:REPORTS_IMPACT]->(li)
                    """, name=normalize(impact), post_id=post_id)
    
    # Create disease-symptom links
    print("\nLinking diseases to reported symptoms...")
    with driver.session() as session:
        session.run("""
            MATCH (d:Disease)<-[:DISCUSSES]-(r:RedditPost)-[:REPORTS_SYMPTOM]->(rs:ReportedSymptom)
            WITH d, rs, count(r) as report_count
            MERGE (d)-[rel:HAS_REPORTED_SYMPTOM]->(rs)
            SET rel.report_count = report_count
        """)
        
        session.run("""
            MATCH (d:Disease)<-[:DISCUSSES]-(r:RedditPost)-[:REPORTS_IMPACT]->(li:LifeImpact)
            WITH d, li, count(r) as report_count
            MERGE (d)-[rel:HAS_LIFE_IMPACT]->(li)
            SET rel.report_count = report_count
        """)
    
    # Print stats
    print("\n=== GRAPH STATISTICS ===")
    with driver.session() as session:
        for label in ["RedditPost", "ReportedSymptom", "LifeImpact"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as c")
            print(f"{label}: {result.single()['c']:,}")
        
        result = session.run("MATCH ()-[r:HAS_REPORTED_SYMPTOM]->() RETURN count(r) as c")
        print(f"Disease-Symptom links: {result.single()['c']:,}")
        
        result = session.run("MATCH ()-[r:HAS_LIFE_IMPACT]->() RETURN count(r) as c")
        print(f"Disease-Impact links: {result.single()['c']:,}")
        
        # Top reported symptoms
        print("\n=== TOP REPORTED SYMPTOMS ===")
        result = session.run("""
            MATCH (rs:ReportedSymptom)<-[:REPORTS_SYMPTOM]-(r:RedditPost)
            RETURN rs.name as symptom, count(r) as mentions
            ORDER BY mentions DESC LIMIT 20
        """)
        for r in result:
            print(f"  {r['symptom'][:50]}: {r['mentions']}")
    
    driver.close()
    print("\n✅ Reddit data added to graph!")

if __name__ == "__main__":
    main()
