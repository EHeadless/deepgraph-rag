"""Simple RAG demo that actually works."""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

# Connect
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "deepgraph2025")
)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("=" * 60)
print("🔍 DeepGraph RAG - Simple Demo")
print("=" * 60)

def search_papers(query: str, top_k: int = 5):
    """Search papers using vector similarity."""
    
    # Generate query embedding
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Search in Neo4j
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('paper_embedding', $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:AUTHORED_BY]->(a:Author)
            RETURN node.title as title, node.arxiv_id as id, 
                   score, collect(a.name) as authors
        """, embedding=query_embedding, top_k=top_k)
        
        papers = []
        for record in result:
            papers.append({
                "title": record["title"],
                "id": record["id"],
                "score": record["score"],
                "authors": record["authors"]
            })
        
        return papers

def answer_question(query: str):
    """Answer a question using RAG."""
    
    print(f"\n💬 Question: {query}")
    print("-" * 40)
    
    # 1. Retrieve relevant papers
    print("🔍 Searching knowledge graph...")
    papers = search_papers(query, top_k=5)
    
    if not papers:
        print("No relevant papers found!")
        return
    
    print(f"📚 Found {len(papers)} relevant papers:")
    for i, p in enumerate(papers, 1):
        print(f"   {i}. {p['title'][:50]}... (score: {p['score']:.3f})")
    
    # 2. Create context
    context = "\n\n".join([
        f"Paper: {p['title']}\nAuthors: {', '.join(p['authors'][:3])}"
        for p in papers
    ])
    
    # 3. Generate answer
    print("\n🤖 Generating answer...")
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant. Answer based on the provided papers."},
            {"role": "user", "content": f"Based on these papers:\n\n{context}\n\nQuestion: {query}\n\nProvide a concise answer:"}
        ],
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    
    print(f"\n💡 Answer:\n{answer}")
    print("\n" + "=" * 60)


# Test queries
queries = [
    "What are recent advances in AI?",
    "Papers about neural networks",
    "Machine learning techniques"
]

for query in queries:
    answer_question(query)

driver.close()
print("\n✅ Demo complete!")