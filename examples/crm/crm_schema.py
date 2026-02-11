"""
CRM Domain Schema for DeepGraph RAG

This demonstrates how to define a custom domain schema for any use case.
The CRM domain tracks Companies, People, Deals, and their relationships.
"""

from deepgraph import GraphSchema, NodeSchema, EdgeSchema

# Define node types
COMPANY_NODE = NodeSchema(
    label="Company",
    id_field="company_id",
    properties={
        "company_id": str,
        "name": str,
        "industry": str,
        "size": str,           # small, medium, large, enterprise
        "revenue": float,
        "description": str,
        "embedding": list,     # For vector search on company descriptions
    },
    indexes=["industry", "size"],
    constraints=["company_id"],
    vector_config={
        "field": "embedding",
        "dimensions": 1536,
        "index_name": "company_embedding"
    }
)

PERSON_NODE = NodeSchema(
    label="Person",
    id_field="email",
    properties={
        "email": str,
        "name": str,
        "title": str,
        "department": str,
        "seniority": str,      # junior, mid, senior, executive
    },
    indexes=["title", "department"],
    constraints=["email"]
)

DEAL_NODE = NodeSchema(
    label="Deal",
    id_field="deal_id",
    properties={
        "deal_id": str,
        "name": str,
        "value": float,
        "stage": str,          # prospect, qualified, proposal, negotiation, closed_won, closed_lost
        "close_date": str,
        "description": str,
        "embedding": list,
    },
    indexes=["stage", "close_date"],
    constraints=["deal_id"],
    vector_config={
        "field": "embedding",
        "dimensions": 1536,
        "index_name": "deal_embedding"
    }
)

PRODUCT_NODE = NodeSchema(
    label="Product",
    id_field="sku",
    properties={
        "sku": str,
        "name": str,
        "category": str,
        "price": float,
    },
    indexes=["category"],
    constraints=["sku"]
)

# Define relationship types
WORKS_AT = EdgeSchema(
    type="WORKS_AT",
    from_label="Person",
    to_label="Company",
    properties={
        "start_date": str,
        "role": str,
    }
)

OWNS_DEAL = EdgeSchema(
    type="OWNS_DEAL",
    from_label="Person",
    to_label="Deal",
    properties={}
)

DEAL_WITH = EdgeSchema(
    type="DEAL_WITH",
    from_label="Deal",
    to_label="Company",
    properties={}
)

KNOWS = EdgeSchema(
    type="KNOWS",
    from_label="Person",
    to_label="Person",
    properties={
        "relationship": str,  # colleague, referral, partner, etc.
        "strength": int,      # 1-5
    }
)

INCLUDES_PRODUCT = EdgeSchema(
    type="INCLUDES_PRODUCT",
    from_label="Deal",
    to_label="Product",
    properties={
        "quantity": int,
        "discount": float,
    }
)

# Complete CRM schema
CRM_SCHEMA = GraphSchema(
    name="crm",
    nodes={
        "Company": COMPANY_NODE,
        "Person": PERSON_NODE,
        "Deal": DEAL_NODE,
        "Product": PRODUCT_NODE,
    },
    edges={
        "WORKS_AT": WORKS_AT,
        "OWNS_DEAL": OWNS_DEAL,
        "DEAL_WITH": DEAL_WITH,
        "KNOWS": KNOWS,
        "INCLUDES_PRODUCT": INCLUDES_PRODUCT,
    },
    metadata={
        "domain": "crm",
        "version": "1.0.0",
        "description": "Sales CRM with companies, contacts, and deals"
    }
)


# Example usage
if __name__ == "__main__":
    print("CRM Schema Definition")
    print("=" * 50)
    print(f"Schema: {CRM_SCHEMA}")
    print(f"\nNodes: {list(CRM_SCHEMA.nodes.keys())}")
    print(f"Edges: {list(CRM_SCHEMA.edges.keys())}")

    print("\n\nGenerated Cypher DDL:")
    print("-" * 50)
    for stmt in CRM_SCHEMA.to_cypher_ddl():
        print(stmt)
