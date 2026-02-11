"""
Build Product Knowledge Graph

Reads products.jsonl and creates:
- Product nodes
- Feature nodes
- UseCase nodes
- Category nodes
- Brand nodes
- Relationships: HAS_FEATURE, FOR_USE_CASE, IN_CATEGORY, MADE_BY, BOUGHT_WITH
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.products.product_schema import FEATURE_PATTERNS, USE_CASE_PATTERNS, get_price_tier


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "products"


class ProductGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✓ Connected to Neo4j")

    def create_schema(self):
        """Create constraints and indexes for product graph."""
        print("Creating schema...")

        queries = [
            # Constraints (unique IDs)
            "CREATE CONSTRAINT product_asin IF NOT EXISTS FOR (p:Product) REQUIRE p.asin IS UNIQUE",
            "CREATE CONSTRAINT feature_name IF NOT EXISTS FOR (f:Feature) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT usecase_name IF NOT EXISTS FOR (u:UseCase) REQUIRE u.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",

            # Indexes for faster queries
            "CREATE INDEX product_price IF NOT EXISTS FOR (p:Product) ON (p.price)",
            "CREATE INDEX product_rating IF NOT EXISTS FOR (p:Product) ON (p.rating)",
            "CREATE INDEX product_category IF NOT EXISTS FOR (p:Product) ON (p.category_name)",
            "CREATE INDEX product_price_tier IF NOT EXISTS FOR (p:Product) ON (p.price_tier)",
            "CREATE INDEX feature_type IF NOT EXISTS FOR (f:Feature) ON (f.feature_type)",
        ]

        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    pass  # Already exists

        print("✓ Schema created")

    def clear_product_data(self):
        """Clear existing product graph data."""
        print("Clearing existing product data...")
        with self.driver.session() as session:
            session.run("MATCH (n:Product) DETACH DELETE n")
            session.run("MATCH (n:Feature) DETACH DELETE n")
            session.run("MATCH (n:UseCase) DETACH DELETE n")
            session.run("MATCH (n:Category) DETACH DELETE n")
            session.run("MATCH (n:Brand) DETACH DELETE n")
        print("✓ Cleared existing data")

    def extract_features(self, text: str) -> list:
        """Extract features from product text using patterns."""
        text_lower = text.lower()
        found = []

        for feature_name, patterns in FEATURE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(feature_name)
                    break

        return list(set(found))

    def extract_use_cases(self, text: str) -> list:
        """Extract use cases from product text using patterns."""
        text_lower = text.lower()
        found = []

        for use_case, patterns in USE_CASE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(use_case)
                    break

        return list(set(found))

    def build_from_products(self, products: list):
        """Build the graph from product list."""
        print(f"Building graph from {len(products)} products...")

        # Collect all unique features, use cases, categories, brands
        all_features = set()
        all_use_cases = set()
        all_categories = set()
        all_brands = set()

        # First pass: collect unique entities
        for product in products:
            all_categories.add(product.get("category", "Unknown"))
            all_brands.add(product.get("brand", "Unknown"))

            # Use pre-extracted features if available, otherwise extract
            features = product.get("features", [])
            if not features:
                features = self.extract_features(
                    product.get("title", "") + " " + product.get("description", "")
                )
            all_features.update(features)

            use_cases = product.get("use_cases", [])
            if not use_cases:
                use_cases = self.extract_use_cases(
                    product.get("title", "") + " " + product.get("description", "")
                )
            all_use_cases.update(use_cases)

        # Create Feature nodes
        print(f"Creating {len(all_features)} Feature nodes...")
        with self.driver.session() as session:
            for feature in all_features:
                display_name = feature.replace("_", " ").title()
                # Determine feature type
                feature_type = "general"
                if feature in ["wireless", "multi_device", "low_latency", "usb_c"]:
                    feature_type = "connectivity"
                elif feature in ["waterproof", "dustproof", "shockproof", "sweatproof"]:
                    feature_type = "durability"
                elif feature in ["noise_canceling", "bass_boost", "surround_sound", "hi_res_audio"]:
                    feature_type = "audio"
                elif feature in ["long_battery", "fast_charging", "wireless_charging"]:
                    feature_type = "power"
                elif feature in ["4k", "hdr", "oled", "high_refresh"]:
                    feature_type = "display"

                session.run("""
                    MERGE (f:Feature {name: $name})
                    SET f.display_name = $display_name,
                        f.feature_type = $feature_type
                """, name=feature, display_name=display_name, feature_type=feature_type)

        # Create UseCase nodes
        print(f"Creating {len(all_use_cases)} UseCase nodes...")
        with self.driver.session() as session:
            for use_case in all_use_cases:
                display_name = use_case.replace("_", " ").title()
                session.run("""
                    MERGE (u:UseCase {name: $name})
                    SET u.display_name = $display_name
                """, name=use_case, display_name=display_name)

        # Create Category nodes
        print(f"Creating {len(all_categories)} Category nodes...")
        with self.driver.session() as session:
            for category in all_categories:
                session.run("""
                    MERGE (c:Category {name: $name})
                    SET c.display_name = $name
                """, name=category)

        # Create Brand nodes
        print(f"Creating {len(all_brands)} Brand nodes...")
        with self.driver.session() as session:
            for brand in all_brands:
                session.run("""
                    MERGE (b:Brand {name: $name})
                    SET b.display_name = $name
                """, name=brand)

        # Create Product nodes and relationships
        print("Creating Product nodes and relationships...")
        for product in tqdm(products, desc="Products"):
            with self.driver.session() as session:
                # Create product node
                price = product.get("price", 0)
                price_tier = get_price_tier(price)

                session.run("""
                    MERGE (p:Product {asin: $asin})
                    SET p.title = $title,
                        p.description = $description,
                        p.price = $price,
                        p.price_tier = $price_tier,
                        p.rating = $rating,
                        p.review_count = $review_count,
                        p.category_name = $category
                """,
                    asin=product.get("asin"),
                    title=product.get("title"),
                    description=product.get("description", ""),
                    price=price,
                    price_tier=price_tier,
                    rating=product.get("rating", 0),
                    review_count=product.get("review_count", 0),
                    category=product.get("category", "Unknown")
                )

                # Link to category
                session.run("""
                    MATCH (p:Product {asin: $asin})
                    MATCH (c:Category {name: $category})
                    MERGE (p)-[:IN_CATEGORY]->(c)
                """, asin=product.get("asin"), category=product.get("category", "Unknown"))

                # Link to brand
                session.run("""
                    MATCH (p:Product {asin: $asin})
                    MATCH (b:Brand {name: $brand})
                    MERGE (p)-[:MADE_BY]->(b)
                """, asin=product.get("asin"), brand=product.get("brand", "Unknown"))

                # Link to features
                features = product.get("features", [])
                if not features:
                    features = self.extract_features(
                        product.get("title", "") + " " + product.get("description", "")
                    )

                for feature in features:
                    session.run("""
                        MATCH (p:Product {asin: $asin})
                        MATCH (f:Feature {name: $feature})
                        MERGE (p)-[:HAS_FEATURE]->(f)
                    """, asin=product.get("asin"), feature=feature)

                # Link to use cases
                use_cases = product.get("use_cases", [])
                if not use_cases:
                    use_cases = self.extract_use_cases(
                        product.get("title", "") + " " + product.get("description", "")
                    )

                for use_case in use_cases:
                    session.run("""
                        MATCH (p:Product {asin: $asin})
                        MATCH (u:UseCase {name: $use_case})
                        MERGE (p)-[:FOR_USE_CASE]->(u)
                    """, asin=product.get("asin"), use_case=use_case)

        # Create BOUGHT_WITH relationships
        print("Creating BOUGHT_WITH relationships...")
        for product in tqdm(products, desc="Co-purchases"):
            also_bought = product.get("also_bought", [])
            if also_bought:
                with self.driver.session() as session:
                    for other_asin in also_bought:
                        session.run("""
                            MATCH (p1:Product {asin: $asin1})
                            MATCH (p2:Product {asin: $asin2})
                            MERGE (p1)-[:BOUGHT_WITH]->(p2)
                        """, asin1=product.get("asin"), asin2=other_asin)

        # Create Feature → UseCase relationships based on co-occurrence
        print("Creating Feature → UseCase relationships...")
        with self.driver.session() as session:
            session.run("""
                MATCH (p:Product)-[:HAS_FEATURE]->(f:Feature)
                MATCH (p)-[:FOR_USE_CASE]->(u:UseCase)
                WITH f, u, count(p) as co_occurrences
                WHERE co_occurrences >= 3
                MERGE (f)-[r:USEFUL_FOR]->(u)
                SET r.strength = co_occurrences
            """)

        print("✅ Graph construction complete!")

    def get_stats(self):
        """Get graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            node_stats = {r["label"]: r["count"] for r in result}

            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """)
            edge_stats = {r["rel_type"]: r["count"] for r in result}

            return node_stats, edge_stats

    def close(self):
        self.driver.close()


def main():
    # Load products
    products_file = DATA_DIR / "products.jsonl"

    if not products_file.exists():
        print(f"Products file not found: {products_file}")
        print("Run 01_download_amazon.py first")
        return

    print(f"Loading products from {products_file}")
    products = []
    with open(products_file) as f:
        for line in f:
            products.append(json.loads(line))
    print(f"✓ Loaded {len(products)} products")

    # Build graph
    builder = ProductGraphBuilder(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "deepgraph2025")
    )

    builder.create_schema()
    builder.clear_product_data()
    builder.build_from_products(products)

    # Show stats
    node_stats, edge_stats = builder.get_stats()

    print("\n📊 Graph Statistics:")
    print("\nNodes:")
    for label, count in node_stats.items():
        print(f"  {label}: {count}")

    print("\nRelationships:")
    for rel_type, count in edge_stats.items():
        print(f"  {rel_type}: {count}")

    builder.close()
    print("\n✅ Done! Check Neo4j Browser: http://localhost:7474")


if __name__ == "__main__":
    main()
