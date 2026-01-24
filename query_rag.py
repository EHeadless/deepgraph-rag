"""
Interactive Graph RAG with Multi-Hop Reasoning
"""
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


def vector_search(query: str, top_k: int = 5):
    """Search papers by semantic similarity."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('paper_embedding', $top_k, $embedding)
            YIELD node, score
            RETURN node.title as title, node.arxiv_id as id, score
        """, embedding=query_embedding, top_k=top_k)
        
        return [{"title": r["title"], "id": r["id"], "score": r["score"]} for r in result]


def multi_hop_author_search(author_name: str):
    """
    Multi-hop: Find papers by collaborators of an author.
    Path: Author -> Paper -> Co-Author -> Their Papers
    """
    with driver.session() as session:
        result = session.run("""
            // Find the author
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            
            // Find papers they wrote
            MATCH (a)<-[:AUTHORED_BY]-(p1:Paper)
            
            // Find co-authors on those papers
            MATCH (p1)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            
            // Find OTHER papers by those co-authors
            MATCH (coauthor)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            
            RETURN DISTINCT 
                a.name as original_author,
                coauthor.name as collaborator,
                p2.title as recommended_paper
            LIMIT 10
        """, name=author_name)
        
        return [dict(r) for r in result]


def multi_hop_paper_network(paper_title_fragment: str):
    """
    Multi-hop: Find related papers through shared authors.
    Path: Paper -> Author -> Other Papers by same author
    """
    with driver.session() as session:
        result = session.run("""
            // Find the paper
            MATCH (p1:Paper)
            WHERE toLower(p1.title) CONTAINS toLower($title)
            
            // Find its authors
            MATCH (p1)-[:AUTHORED_BY]->(a:Author)
            
            // Find other papers by those authors
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            
            RETURN 
                p1.title as source_paper,
                a.name as shared_author,
                p2.title as related_paper
            LIMIT 10
        """, title=paper_title_fragment)
        
        return [dict(r) for r in result]


def find_author_network(author_name: str):
    """
    Multi-hop: Map an author's collaboration network.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            
            // Papers by this author
            MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
            
            // Co-authors
            MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            
            RETURN 
                a.name as author,
                count(DISTINCT p) as paper_count,
                collect(DISTINCT coauthor.name) as collaborators
        """, name=author_name)
        
        return [dict(r) for r in result]


def answer_with_rag(question: str):
    """Full RAG pipeline with context."""
    print(f"\n🔍 Searching for: {question}")
    
    # Vector search
    papers = vector_search(question, top_k=5)
    
    if not papers:
        return "No relevant papers found."
    
    print(f"📚 Found {len(papers)} papers")
    
    # Build context
    context = "\n".join([f"- {p['title']}" for p in papers])
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Answer based on the research papers provided. Be concise."},
            {"role": "user", "content": f"Papers:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content


def interactive_menu():
    """Main interactive loop."""
    print("=" * 60)
    print("🧠 DeepGraph RAG - Interactive Query System")
    print("=" * 60)
    print("\nQuery Types:")
    print("  1. Ask a question (semantic search + RAG)")
    print("  2. Find papers by author's collaborators (multi-hop)")
    print("  3. Find related papers through shared authors (multi-hop)")
    print("  4. Map author's collaboration network (multi-hop)")
    print("  5. Exit")
    print("-" * 60)
    
    while True:
        print("\n")
        choice = input("Choose query type (1-5): ").strip()
        
        if choice == "1":
            question = input("Enter your question: ").strip()
            if question:
                answer = answer_with_rag(question)
                print(f"\n💡 Answer:\n{answer}")
        
        elif choice == "2":
            author = input("Enter author name: ").strip()
            if author:
                print(f"\n🔗 Multi-hop: {author} → Their Papers → Co-authors → Their Papers")
                results = multi_hop_author_search(author)
                if results:
                    print(f"\n📚 Papers by {author}'s collaborators:")
                    for r in results:
                        print(f"  • Via {r['collaborator']}: {r['recommended_paper'][:50]}...")
                else:
                    print("No results found. Try a partial name.")
        
        elif choice == "3":
            title = input("Enter paper title (partial): ").strip()
            if title:
                print(f"\n🔗 Multi-hop: Paper → Authors → Their Other Papers")
                results = multi_hop_paper_network(title)
                if results:
                    print(f"\n📚 Related papers through shared authors:")
                    for r in results:
                        print(f"  • Via {r['shared_author']}: {r['related_paper'][:50]}...")
                else:
                    print("No results found. Try different keywords.")
        
        elif choice == "4":
            author = input("Enter author name: ").strip()
            if author:
                print(f"\n🔗 Mapping collaboration network for: {author}")
                results = find_author_network(author)
                if results:
                    for r in results:
                        print(f"\n👤 {r['author']}")
                        print(f"   Papers: {r['paper_count']}")
                        print(f"   Collaborators: {', '.join(r['collaborators'][:5])}")
                        if len(r['collaborators']) > 5:
                            print(f"   ... and {len(r['collaborators']) - 5} more")
                else:
                    print("No results found.")
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("Invalid choice. Enter 1-5.")


if __name__ == "__main__":
    try:
        interactive_menu()
    finally:
        driver.close()