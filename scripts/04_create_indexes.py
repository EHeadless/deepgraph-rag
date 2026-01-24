"""Create vector indexes and embeddings."""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Connect
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "deepgraph2025")
)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("✓ Connected to Neo4j")

# Create vector index
print("Creating vector index...")
with driver.session() as session:
    try:
        session.run("""
            CREATE VECTOR INDEX paper_embedding IF NOT EXISTS
            FOR (p:Paper) ON (p.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
        """)
        print("✓ Vector index created")
    except Exception as e:
        print(f"Index already exists or error: {e}")

# Get papers without embeddings
print("Fetching papers...")
with driver.session() as session:
    result = session.run("""
        MATCH (p:Paper)
        WHERE p.embedding IS NULL
        RETURN p.arxiv_id as id, p.title as title
    """)
    papers = [(r["id"], r["title"]) for r in result]

print(f"Generating embeddings for {len(papers)} papers...")

# Generate embeddings
for paper_id, title in tqdm(papers, desc="Embedding papers"):
    try:
        # Generate embedding
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=title[:500]
        )
        embedding = response.data[0].embedding
        
        # Save to graph
        with driver.session() as session:
            session.run("""
                MATCH (p:Paper {arxiv_id: $paper_id})
                SET p.embedding = $embedding
            """, paper_id=paper_id, embedding=embedding)
    except Exception as e:
        print(f"Error on {paper_id}: {e}")

print("✅ Embeddings created!")

# Verify
with driver.session() as session:
    result = session.run("""
        MATCH (p:Paper)
        RETURN count(p) as total, count(p.embedding) as with_embedding
    """)
    stats = result.single()
    print(f"\n📊 Papers: {stats['total']}, With embeddings: {stats['with_embedding']}")

driver.close()
print("\n✅ All done! Your graph is ready to query!")