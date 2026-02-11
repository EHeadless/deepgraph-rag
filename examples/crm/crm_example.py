"""
CRM Graph RAG Example

This example demonstrates using DeepGraph RAG for a CRM domain:
- Find similar deals based on description
- Discover deals through relationship networks
- Find warm introductions to target companies

Requirements:
    pip install deepgraph-rag
    export OPENAI_API_KEY=sk-...
    export NEO4J_PASSWORD=password
"""

import os
from openai import OpenAI
from deepgraph import create_graph_store, GraphSchema
from deepgraph.adapters.embedders.openai import OpenAIEmbedder
from examples.crm.crm_schema import CRM_SCHEMA


def create_sample_data(store, embedder):
    """Create sample CRM data for demonstration."""
    print("Creating sample CRM data...")

    # Sample companies
    companies = [
        {
            "company_id": "acme-corp",
            "name": "Acme Corporation",
            "industry": "Manufacturing",
            "size": "enterprise",
            "revenue": 500000000,
            "description": "Global manufacturing company specializing in industrial equipment and automation solutions"
        },
        {
            "company_id": "techstart-io",
            "name": "TechStart.io",
            "industry": "Technology",
            "size": "small",
            "revenue": 5000000,
            "description": "AI-powered startup focusing on machine learning solutions for e-commerce personalization"
        },
        {
            "company_id": "finserv-global",
            "name": "FinServ Global",
            "industry": "Finance",
            "size": "large",
            "revenue": 200000000,
            "description": "Financial services firm providing investment banking and wealth management"
        },
        {
            "company_id": "healthplus",
            "name": "HealthPlus Medical",
            "industry": "Healthcare",
            "size": "medium",
            "revenue": 50000000,
            "description": "Healthcare technology company building patient management and telemedicine platforms"
        },
    ]

    # Add embeddings
    for company in companies:
        embedding = embedder.embed(company["description"])
        company["embedding"] = embedding

    store.add_nodes("Company", companies)
    print(f"  Added {len(companies)} companies")

    # Sample people
    people = [
        {"email": "john@acme.com", "name": "John Smith", "title": "VP of Operations", "department": "Operations", "seniority": "executive"},
        {"email": "sarah@acme.com", "name": "Sarah Johnson", "title": "CTO", "department": "Engineering", "seniority": "executive"},
        {"email": "mike@techstart.io", "name": "Mike Chen", "title": "CEO", "department": "Executive", "seniority": "executive"},
        {"email": "lisa@finserv.com", "name": "Lisa Wang", "title": "Managing Director", "department": "Investment Banking", "seniority": "executive"},
        {"email": "bob@healthplus.com", "name": "Bob Anderson", "title": "Head of Partnerships", "department": "Business Development", "seniority": "senior"},
        {"email": "alice@sales.com", "name": "Alice Brown", "title": "Account Executive", "department": "Sales", "seniority": "mid"},
    ]
    store.add_nodes("Person", people)
    print(f"  Added {len(people)} people")

    # Sample deals
    deals = [
        {
            "deal_id": "deal-001",
            "name": "Acme Automation Platform",
            "value": 250000,
            "stage": "negotiation",
            "close_date": "2024-03-15",
            "description": "Enterprise automation platform for manufacturing processes and quality control"
        },
        {
            "deal_id": "deal-002",
            "name": "TechStart ML Integration",
            "value": 75000,
            "stage": "proposal",
            "close_date": "2024-02-28",
            "description": "Machine learning recommendation engine for e-commerce personalization"
        },
        {
            "deal_id": "deal-003",
            "name": "FinServ Data Analytics",
            "value": 500000,
            "stage": "qualified",
            "close_date": "2024-06-30",
            "description": "Advanced analytics and reporting platform for investment portfolio management"
        },
    ]

    for deal in deals:
        embedding = embedder.embed(deal["description"])
        deal["embedding"] = embedding

    store.add_nodes("Deal", deals)
    print(f"  Added {len(deals)} deals")

    # Relationships: People work at companies
    works_at = [
        {"from": "john@acme.com", "to": "acme-corp"},
        {"from": "sarah@acme.com", "to": "acme-corp"},
        {"from": "mike@techstart.io", "to": "techstart-io"},
        {"from": "lisa@finserv.com", "to": "finserv-global"},
        {"from": "bob@healthplus.com", "to": "healthplus"},
    ]
    store.add_edges("WORKS_AT", works_at)

    # Relationships: People know each other
    knows = [
        {"from": "john@acme.com", "to": "mike@techstart.io", "relationship": "former_colleague", "strength": 4},
        {"from": "sarah@acme.com", "to": "lisa@finserv.com", "relationship": "conference", "strength": 2},
        {"from": "mike@techstart.io", "to": "bob@healthplus.com", "relationship": "investor_connection", "strength": 3},
        {"from": "alice@sales.com", "to": "john@acme.com", "relationship": "customer", "strength": 5},
    ]
    store.add_edges("KNOWS", knows)

    # Relationships: Deals with companies
    deal_with = [
        {"from": "deal-001", "to": "acme-corp"},
        {"from": "deal-002", "to": "techstart-io"},
        {"from": "deal-003", "to": "finserv-global"},
    ]
    store.add_edges("DEAL_WITH", deal_with)

    # Relationships: People own deals
    owns_deal = [
        {"from": "alice@sales.com", "to": "deal-001"},
        {"from": "alice@sales.com", "to": "deal-002"},
        {"from": "alice@sales.com", "to": "deal-003"},
    ]
    store.add_edges("OWNS_DEAL", owns_deal)

    print("Sample data created!")


def demo_graph_rag_queries(store, embedder):
    """Demonstrate graph-enhanced queries for CRM."""
    print("\n" + "=" * 60)
    print("CRM Graph RAG Queries")
    print("=" * 60)

    # Query 1: Vector search for similar deals
    print("\n1. Vector Search: Find deals similar to 'AI automation'")
    print("-" * 50)

    query_embedding = embedder.embed("AI automation for manufacturing")
    results = store.vector_search(
        index_name="deal_embedding",
        embedding=query_embedding,
        top_k=3
    )

    for r in results:
        node = r["node"]
        print(f"  {node['name']}: ${node['value']:,.0f} ({node['stage']})")
        print(f"    Score: {r['score']:.3f}")

    # Query 2: Find warm introductions through network
    print("\n2. Graph Query: Find path to FinServ Global")
    print("-" * 50)

    path = store.find_path(
        from_id="alice@sales.com",
        to_id="lisa@finserv.com",
        from_label="Person",
        to_label="Person",
        max_depth=4
    )

    if path:
        print(f"  Path found! Length: {path['length']} hops")
        print(f"  Path: ", end="")
        names = [n.get("name", n.get("email", "?")) for n in path["nodes"]]
        print(" -> ".join(names))
    else:
        print("  No path found")

    # Query 3: Find deals through company relationships
    print("\n3. Graph Query: Find all deals connected to a person's network")
    print("-" * 50)

    # Find all companies connected to John's network (2 hops)
    query = """
        MATCH (p:Person {email: $email})-[:KNOWS*1..2]-(connected:Person)-[:WORKS_AT]->(c:Company)
        MATCH (d:Deal)-[:DEAL_WITH]->(c)
        WHERE p.email <> connected.email
        RETURN DISTINCT d.name as deal_name, d.value as value, d.stage as stage,
               c.name as company, connected.name as connection
    """
    results = store.query(query, {"email": "alice@sales.com"})

    for r in results:
        print(f"  {r['deal_name']}: ${r['value']:,.0f}")
        print(f"    Company: {r['company']} (via {r['connection']})")


def main():
    """Run the CRM example."""
    # Get API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Set OPENAI_API_KEY environment variable")
        return

    # Create OpenAI client
    client = OpenAI(api_key=openai_api_key)
    embedder = OpenAIEmbedder(client=client)

    # Create graph store
    print("Connecting to Neo4j...")
    store = create_graph_store(
        backend="neo4j",
        schema=CRM_SCHEMA,
        uri="bolt://localhost:7687",
        user="neo4j",
        password=os.getenv("NEO4J_PASSWORD", "password")
    )

    # Create schema (indexes, constraints)
    print("Creating schema...")
    store.create_schema(CRM_SCHEMA)

    # Check if data exists
    existing = store.query("MATCH (c:Company) RETURN count(c) as count")
    if existing[0]["count"] == 0:
        create_sample_data(store, embedder)
    else:
        print(f"Using existing data ({existing[0]['count']} companies)")

    # Run demo queries
    demo_graph_rag_queries(store, embedder)

    # Clean up
    store.disconnect()
    print("\n\nDone!")


if __name__ == "__main__":
    main()
