"""
Build knowledge graph from extracted entities.
"""

import json
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase


class GraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✓ Connected to Neo4j")
    
    def build_from_extractions(self, extractions):
        print(f"Building graph from {len(extractions)} papers...")
        
        # Create schema
        self._create_schema()
        
        # Process each paper
        for extraction in tqdm(extractions, desc="Building graph"):
            self._create_paper(extraction)
            self._create_authors(extraction)
            self._create_concepts(extraction)
            self._create_methods(extraction)
        
        print("✅ Graph construction complete!")
    
    def _create_schema(self):
        """Create constraints and indexes."""
        print("Creating schema...")
        
        queries = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX paper_date IF NOT EXISTS FOR (p:Paper) ON (p.published_date)",
        ]
        
        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                except:
                    pass  # Already exists
        
        print("✓ Schema created")
    
    def _create_paper(self, extraction):
        """Create paper node."""
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Paper {arxiv_id: $arxiv_id})
                SET p.title = $title,
                    p.published_date = $published_date,
                    p.primary_category = $primary_category
            """, 
                arxiv_id=extraction.get("paper_id"),
                title=extraction.get("paper_title"),
                published_date=extraction.get("published_date"),
                primary_category=extraction.get("primary_category")
            )
    
    def _create_authors(self, extraction):
        """Create author nodes and relationships."""
        paper_id = extraction.get("paper_id")
        authors = extraction.get("authors", [])
        
        with self.driver.session() as session:
            for idx, author in enumerate(authors):
                name = author.get("name")
                if not name:
                    continue
                
                # Create author
                session.run("""
                    MERGE (a:Author {name: $name})
                """, name=name)
                
                # Link to paper
                session.run("""
                    MATCH (p:Paper {arxiv_id: $paper_id})
                    MATCH (a:Author {name: $author_name})
                    MERGE (p)-[:AUTHORED_BY {position: $position}]->(a)
                """, 
                    paper_id=paper_id,
                    author_name=name,
                    position=idx
                )
    
    def _create_concepts(self, extraction):
        """Create concept nodes."""
        paper_id = extraction.get("paper_id")
        concepts = extraction.get("concepts", [])
        
        with self.driver.session() as session:
            for concept in concepts:
                name = concept.get("name")
                if not name:
                    continue
                
                session.run("""
                    MERGE (c:Concept {name: $name})
                    SET c.field = $field
                """, name=name, field=concept.get("field", "CS"))
                
                session.run("""
                    MATCH (p:Paper {arxiv_id: $paper_id})
                    MATCH (c:Concept {name: $concept_name})
                    MERGE (p)-[:ABOUT_CONCEPT]->(c)
                """, paper_id=paper_id, concept_name=name)
    
    def _create_methods(self, extraction):
        """Create method nodes."""
        paper_id = extraction.get("paper_id")
        methods = extraction.get("methods", [])
        
        with self.driver.session() as session:
            for method in methods:
                name = method.get("name")
                if not name:
                    continue
                
                session.run("""
                    MERGE (m:Method {name: $name})
                """, name=name)
                
                session.run("""
                    MATCH (p:Paper {arxiv_id: $paper_id})
                    MATCH (m:Method {name: $method_name})
                    MERGE (p)-[:USES_METHOD]->(m)
                """, paper_id=paper_id, method_name=name)
    
    def get_stats(self):
        """Get graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
            """)
            
            stats = {}
            for record in result:
                stats[record["label"]] = record["count"]
            
            return stats
    
    def close(self):
        self.driver.close()


# Load and build
print("Loading extractions...")
extractions = []
with open('data/processed/extracted_entities.jsonl') as f:
    for line in f:
        extractions.append(json.loads(line))

print(f"✓ Loaded {len(extractions)} extractions")

# Build graph
builder = GraphBuilder(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="deepgraph2025"
)

builder.build_from_extractions(extractions)

# Show stats
stats = builder.get_stats()
print("\n📊 Graph Statistics:")
for label, count in stats.items():
    print(f"  {label}: {count}")

builder.close()
print("\n✅ Done! Check Neo4j Browser: http://localhost:7474")