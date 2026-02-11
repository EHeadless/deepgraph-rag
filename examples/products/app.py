"""
Product Navigator - Graph RAG for Product Recommendations

Demonstrates the same Graph RAG patterns as the research demo, applied to products:
- Feature × UseCase filtering (like Method × Concept)
- Bundle Builder (like Reading Path)
- Niche Finder (like Research Frontier)
- Feature Migration (like Method Transfer)
"""

import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from examples.products.product_schema import PRICE_TIERS, get_price_tier

# Page config
st.set_page_config(
    page_title="Product Navigator - Graph RAG",
    page_icon="🛒",
    layout="wide"
)


# ============== DATABASE CONNECTION ==============

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "deepgraph2025")
        )
    )


def query(cypher: str, params: dict = None):
    """Execute a Cypher query and return results."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


# ============== HELPER FUNCTIONS ==============

def get_stats():
    """Get graph statistics."""
    nodes = query("""
        MATCH (n)
        RETURN labels(n)[0] as label, count(*) as count
    """)
    return {r['label']: r['count'] for r in nodes}


def get_all_features():
    """Get all features with product counts."""
    return query("""
        MATCH (f:Feature)<-[:HAS_FEATURE]-(p:Product)
        RETURN f.name as name, f.display_name as display_name,
               f.feature_type as feature_type, count(p) as product_count
        ORDER BY product_count DESC
    """)


def get_all_use_cases():
    """Get all use cases with product counts."""
    return query("""
        MATCH (u:UseCase)<-[:FOR_USE_CASE]-(p:Product)
        RETURN u.name as name, u.display_name as display_name,
               count(p) as product_count
        ORDER BY product_count DESC
    """)


def get_all_categories():
    """Get all categories with product counts."""
    return query("""
        MATCH (c:Category)<-[:IN_CATEGORY]-(p:Product)
        RETURN c.name as name, count(p) as product_count
        ORDER BY product_count DESC
    """)


def get_all_brands():
    """Get all brands with product counts."""
    return query("""
        MATCH (b:Brand)<-[:MADE_BY]-(p:Product)
        RETURN b.name as name, count(p) as product_count
        ORDER BY product_count DESC
        LIMIT 20
    """)


# Category to image mapping using Unsplash
CATEGORY_IMAGES = {
    "Headphones": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=150&h=150&fit=crop",
    "Speakers": "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=150&h=150&fit=crop",
    "Smartwatch": "https://images.unsplash.com/photo-1579586337278-3befd40fd17a?w=150&h=150&fit=crop",
    "Camera": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=150&h=150&fit=crop",
    "Laptop": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=150&h=150&fit=crop",
    "Keyboard": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=150&h=150&fit=crop",
    "Mouse": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=150&h=150&fit=crop",
    "Monitor": "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=150&h=150&fit=crop",
}


def get_product_image(category: str) -> str:
    """Get product image URL based on category."""
    return CATEGORY_IMAGES.get(category, "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=150&h=150&fit=crop")


def create_graph_visualization(nodes_data: list, edges_data: list, height: str = "600px"):
    """Create an interactive graph visualization using pyvis."""
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Color scheme for node types
    colors = {
        "Product": "#ff6b6b",
        "Feature": "#4ecdc4",
        "UseCase": "#ffe66d",
        "Category": "#95e1d3",
        "Brand": "#dda0dd"
    }

    # Add nodes
    for node in nodes_data:
        net.add_node(
            node['id'],
            label=node['label'][:30],
            title=node.get('title', node['label']),
            color=colors.get(node['type'], "#888888"),
            size=node.get('size', 20)
        )

    # Add edges
    for edge in edges_data:
        net.add_edge(
            edge['from'],
            edge['to'],
            title=edge.get('label', ''),
            color="#666666"
        )

    # Generate HTML
    html = net.generate_html()
    return html


# ============== HEADER ==============

st.title("🛒 Product Navigator")
st.markdown("*Graph RAG for smarter product discovery. Same pattern as research papers, different domain.*")

# Quick stats
stats = get_stats()
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Products", stats.get('Product', 0))
with col2:
    st.metric("Features", stats.get('Feature', 0))
with col3:
    st.metric("Use Cases", stats.get('UseCase', 0))
with col4:
    st.metric("Categories", stats.get('Category', 0))
with col5:
    st.metric("Brands", stats.get('Brand', 0))

st.divider()

# ============== TABS ==============

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 How It Works",
    "🔀 Feature × UseCase Filter",
    "📦 Bundle Builder",
    "🔮 Niche Finder",
    "🔄 Feature Migration",
    "🕸️ Graph Explorer"
])


# ============== TAB 1: HOW IT WORKS ==============

with tab1:
    st.header("📊 Graph RAG for Products")

    st.markdown("""
    ### The Same Pattern, Different Domain

    | Research Papers | Product Recommendations |
    |-----------------|-------------------------|
    | Paper → USES_METHOD → Method | Product → HAS_FEATURE → Feature |
    | Paper → DISCUSSES → Concept | Product → FOR_USE_CASE → UseCase |
    | Paper → AUTHORED_BY → Author | Product → MADE_BY → Brand |
    | Paper → IN_CATEGORY → Field | Product → IN_CATEGORY → Category |
    | Paper → CITES → Paper | Product → BOUGHT_WITH → Product |
    """)

    st.markdown("### The Knowledge Graph")

    st.code("""
Product
  ├── HAS_FEATURE ──→ Feature (wireless, waterproof, noise_canceling...)
  ├── FOR_USE_CASE ──→ UseCase (travel, workout, gaming, work_from_home...)
  ├── IN_CATEGORY ──→ Category (Headphones, Speakers, Laptop...)
  ├── MADE_BY ──→ Brand (Sony, Apple, Bose...)
  └── BOUGHT_WITH ──→ Product (co-purchase patterns)
    """)

    st.markdown("""
    ### Why Graph Beats Vector for Products

    | Query | Vector Search | Graph RAG |
    |-------|---------------|-----------|
    | "Wireless headphones" | Good | Good |
    | "Wireless headphones FOR running under $100" | Poor | Excellent |
    | "What premium features are now in budget products?" | Impossible | Works |
    | "Build me a home office bundle" | Impossible | Works |
    | "Underserved niches (feature gaps)" | Impossible | Works |

    **Vector search finds products with similar descriptions.**
    **Graph RAG finds products that HAVE specific features FOR specific uses.**
    """)


# ============== TAB 2: FEATURE × USE CASE FILTER ==============

with tab2:
    st.header("🔀 Feature × UseCase Filter")
    st.markdown("""
    **The killer feature.** Find products with specific FEATURES for specific USE CASES.

    *Like "Method × Concept" for research papers.*
    """)

    features = get_all_features()
    use_cases = get_all_use_cases()
    categories = get_all_categories()

    col1, col2, col3 = st.columns(3)

    with col1:
        feature_opts = ["Any feature"] + [f"{f['display_name']} ({f['product_count']})" for f in features[:15]]
        selected_feature = st.selectbox("Feature:", feature_opts)

    with col2:
        use_case_opts = ["Any use case"] + [f"{u['display_name']} ({u['product_count']})" for u in use_cases[:15]]
        selected_use_case = st.selectbox("Use Case:", use_case_opts)

    with col3:
        price_tier_opts = ["Any price"] + list(PRICE_TIERS.keys())
        selected_price = st.selectbox("Price Tier:", price_tier_opts)

    if st.button("🔍 Find Products", type="primary", key="filter_btn"):
        # Build query
        conditions = []
        params = {}

        if selected_feature != "Any feature":
            feature_name = features[[f['display_name'] for f in features].index(selected_feature.split(" (")[0])]['name']
            conditions.append("(p)-[:HAS_FEATURE]->(:Feature {name: $feature})")
            params['feature'] = feature_name

        if selected_use_case != "Any use case":
            use_case_name = use_cases[[u['display_name'] for u in use_cases].index(selected_use_case.split(" (")[0])]['name']
            conditions.append("(p)-[:FOR_USE_CASE]->(:UseCase {name: $use_case})")
            params['use_case'] = use_case_name

        if selected_price != "Any price":
            conditions.append("p.price_tier = $price_tier")
            params['price_tier'] = selected_price

        where_clause = " AND ".join(conditions) if conditions else "true"

        results = query(f"""
            MATCH (p:Product)
            WHERE {where_clause}
            MATCH (p)-[:IN_CATEGORY]->(c:Category)
            MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (p)-[:FOR_USE_CASE]->(u:UseCase)
            RETURN p.asin as asin, p.title as title, p.price as price,
                   p.rating as rating, p.review_count as review_count,
                   p.price_tier as price_tier,
                   c.name as category, b.name as brand,
                   collect(DISTINCT f.display_name) as features,
                   collect(DISTINCT u.display_name) as use_cases
            ORDER BY rating DESC, review_count DESC
            LIMIT 15
        """, params)

        if results:
            st.success(f"**Found {len(results)} products** matching your criteria!")

            for i, product in enumerate(results, 1):
                col_img, col_info, col_price = st.columns([1, 3, 1])

                with col_img:
                    st.image(get_product_image(product['category']), width=100)

                with col_info:
                    st.markdown(f"**{i}. {product['title']}**")
                    st.caption(f"📁 {product['category']} | 🏷️ {product['brand']} | ⭐ {product['rating']}")

                    if product['features']:
                        st.markdown(f"🔧 Features: {', '.join(product['features'][:5])}")
                    if product['use_cases']:
                        st.markdown(f"🎯 Use Cases: {', '.join(product['use_cases'][:4])}")

                with col_price:
                    st.metric("Price", f"${product['price']:.2f}")
                    st.caption(f"Tier: {product['price_tier']}")

                st.markdown("---")

            st.info("""
            💡 **Why this is powerful:**

            Vector search for "wireless travel" finds products mentioning these words.
            This finds products that **HAVE wireless** AND are **FOR travel** — structural query, not text matching.
            """)
        else:
            st.warning("No products found. Try different criteria.")


# ============== TAB 3: BUNDLE BUILDER ==============

with tab3:
    st.header("📦 Bundle Builder")
    st.markdown("""
    **Pick a product. Get complementary items.**

    *Based on co-purchase patterns and feature compatibility.*
    """)

    # Get some products to choose from
    sample_products = query("""
        MATCH (p:Product)
        WHERE p.review_count > 100
        RETURN p.asin as asin, p.title as title, p.category_name as category
        ORDER BY p.review_count DESC
        LIMIT 50
    """)

    product_options = {f"{p['title'][:60]}... ({p['category']})": p['asin'] for p in sample_products}

    selected_product = st.selectbox("Select a product:", list(product_options.keys()))

    if st.button("📦 Build Bundle", type="primary", key="bundle_btn"):
        asin = product_options[selected_product]

        # Get the product details
        product_info = query("""
            MATCH (p:Product {asin: $asin})
            MATCH (p)-[:IN_CATEGORY]->(c:Category)
            MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            RETURN p.title as title, p.price as price, p.rating as rating,
                   c.name as category, b.name as brand,
                   collect(DISTINCT f.name) as features
        """, {"asin": asin})[0]

        st.markdown("### 🎯 Your Selected Product")
        sel_img, sel_info = st.columns([1, 4])
        with sel_img:
            st.image(get_product_image(product_info['category']), width=100)
        with sel_info:
            st.markdown(f"**{product_info['title']}**")
            st.markdown(f"💰 ${product_info['price']:.2f} | ⭐ {product_info['rating']} | 🏷️ {product_info['brand']}")

        st.divider()

        col1, col2 = st.columns(2)

        # Frequently bought together
        with col1:
            st.markdown("### 🛒 Frequently Bought Together")

            bought_with = query("""
                MATCH (p:Product {asin: $asin})-[:BOUGHT_WITH]->(other:Product)
                MATCH (other)-[:IN_CATEGORY]->(c:Category)
                RETURN other.title as title, other.price as price,
                       other.rating as rating, c.name as category
                ORDER BY other.rating DESC
                LIMIT 5
            """, {"asin": asin})

            if bought_with:
                total = product_info['price']
                for item in bought_with:
                    bw_img, bw_info = st.columns([1, 3])
                    with bw_img:
                        st.image(get_product_image(item['category']), width=60)
                    with bw_info:
                        st.markdown(f"➕ **{item['title'][:40]}...**")
                        st.caption(f"${item['price']:.2f} | {item['category']}")
                    total += item['price']

                st.markdown(f"**Bundle Total: ${total:.2f}**")
            else:
                st.info("No co-purchase data available.")

        # Same features, different category
        with col2:
            st.markdown("### 🔗 Compatible Products")
            st.caption("Same features, different category")

            compatible = query("""
                MATCH (p:Product {asin: $asin})-[:HAS_FEATURE]->(f:Feature)
                       <-[:HAS_FEATURE]-(other:Product)
                WHERE other.asin <> $asin
                AND other.category_name <> p.category_name
                WITH other, collect(f.display_name) as shared_features, count(f) as feature_overlap
                WHERE feature_overlap >= 2
                MATCH (other)-[:IN_CATEGORY]->(c:Category)
                RETURN other.title as title, other.price as price,
                       c.name as category, shared_features, feature_overlap
                ORDER BY feature_overlap DESC
                LIMIT 5
            """, {"asin": asin})

            if compatible:
                for item in compatible:
                    cp_img, cp_info = st.columns([1, 3])
                    with cp_img:
                        st.image(get_product_image(item['category']), width=60)
                    with cp_info:
                        st.markdown(f"🔧 **{item['title'][:40]}...**")
                        st.caption(f"${item['price']:.2f} | {item['category']}")
                        st.caption(f"Shared: {', '.join(item['shared_features'][:3])}")
            else:
                st.info("No compatible products found.")

        st.divider()
        st.success("""
        ✅ **Bundle built using graph traversal:**
        - Co-purchase patterns (BOUGHT_WITH relationships)
        - Feature compatibility (shared HAS_FEATURE edges)

        Vector search can't do this — it would need to understand relationships.
        """)


# ============== TAB 4: NICHE FINDER ==============

with tab4:
    st.header("🔮 Niche Finder")
    st.markdown("""
    **Find underserved niches.** Rare feature + use case combinations = market opportunities.

    *Like "Research Frontier" for papers.*
    """)

    if st.button("🔮 Discover Niches", type="primary", key="niche_btn"):
        # Find rare feature + use case combinations
        niches = query("""
            MATCH (p:Product)-[:HAS_FEATURE]->(f:Feature)
            MATCH (p)-[:FOR_USE_CASE]->(u:UseCase)
            WITH f.display_name as feature, u.display_name as use_case, count(p) as products
            WHERE products >= 1 AND products <= 5
            RETURN feature, use_case, products
            ORDER BY products ASC
            LIMIT 20
        """)

        if niches:
            st.success(f"**Found {len(niches)} underserved niches!**")

            st.markdown("### 🟢 Rare Combinations (Market Opportunities)")

            for niche in niches:
                icon = "🟢" if niche['products'] == 1 else "🟡" if niche['products'] <= 3 else "🟠"
                st.markdown(f"{icon} **{niche['feature']}** × **{niche['use_case']}** — only {niche['products']} product(s)")

            st.divider()

            # High-potential niches (popular features, popular use cases, few products)
            st.markdown("### 🚀 High-Potential Gaps")
            st.caption("Popular features + popular use cases with few products")

            gaps = query("""
                MATCH (f:Feature)<-[:HAS_FEATURE]-(p1:Product)
                WITH f, count(p1) as feature_popularity
                WHERE feature_popularity >= 20

                MATCH (u:UseCase)<-[:FOR_USE_CASE]-(p2:Product)
                WITH f, feature_popularity, u, count(p2) as use_case_popularity
                WHERE use_case_popularity >= 20

                OPTIONAL MATCH (p3:Product)-[:HAS_FEATURE]->(f)
                WHERE (p3)-[:FOR_USE_CASE]->(u)
                WITH f, u, feature_popularity, use_case_popularity, count(p3) as combined
                WHERE combined <= 3

                RETURN f.display_name as feature, u.display_name as use_case,
                       feature_popularity, use_case_popularity, combined
                ORDER BY (feature_popularity + use_case_popularity) DESC
                LIMIT 10
            """)

            if gaps:
                for gap in gaps:
                    if gap['combined'] == 0:
                        st.markdown(f"🔥 **{gap['feature']}** + **{gap['use_case']}** = **UNEXPLORED!**")
                        st.caption(f"   Feature in {gap['feature_popularity']} products, use case in {gap['use_case_popularity']} — but never combined!")
                    else:
                        st.markdown(f"⚡ **{gap['feature']}** + **{gap['use_case']}** = only {gap['combined']} products")

            st.info("""
            💡 **Business Insight:**

            🟢 = 1 product (first-mover opportunity)
            🟡 = 2-3 products (emerging niche)
            🔥 = 0 products (unexplored market)

            These are structural queries impossible with vector search.
            """)


# ============== TAB 5: FEATURE MIGRATION ==============

with tab5:
    st.header("🔄 Feature Migration")
    st.markdown("""
    **What premium features are now in budget products?**

    *Track how features migrate from luxury to mass market.*
    """)

    col1, col2 = st.columns(2)

    with col1:
        source_tier = st.selectbox("Features FROM:", ["premium", "luxury"], key="source_tier")

    with col2:
        target_tier = st.selectbox("Now IN:", ["budget", "mid_range"], key="target_tier")

    if st.button("🔄 Find Feature Migration", type="primary", key="migration_btn"):
        # Find features that exist in both tiers
        migration = query("""
            MATCH (p1:Product {price_tier: $source})-[:HAS_FEATURE]->(f:Feature)
                  <-[:HAS_FEATURE]-(p2:Product {price_tier: $target})
            WITH f, count(DISTINCT p1) as source_count, count(DISTINCT p2) as target_count,
                 collect(DISTINCT p2.title)[0..2] as example_products,
                 avg(p2.price) as avg_target_price
            WHERE source_count > 0 AND target_count > 0
            RETURN f.display_name as feature,
                   source_count, target_count,
                   example_products,
                   avg_target_price
            ORDER BY target_count DESC
            LIMIT 15
        """, {"source": source_tier, "target": target_tier})

        if migration:
            st.success(f"**Found {len(migration)} features migrating from {source_tier} to {target_tier}!**")

            for item in migration:
                st.markdown(f"### 🔧 {item['feature']}")
                st.markdown(f"- **{source_tier.title()}:** {item['source_count']} products")
                st.markdown(f"- **{target_tier.title()}:** {item['target_count']} products (avg ${item['avg_target_price']:.2f})")

                st.markdown("**Example products:**")
                for ex in item['example_products']:
                    st.markdown(f"  - {ex[:60]}...")

                st.markdown("---")

            st.success("""
            ✅ **Feature Migration Analysis:**

            This shows which premium features have become accessible in budget products.

            **Business applications:**
            - Identify democratizing technology trends
            - Find value opportunities (premium features, budget prices)
            - Track market maturation

            Impossible with vector search — requires structural comparison across price tiers.
            """)
        else:
            st.info(f"No feature migration found from {source_tier} to {target_tier}.")


# ============== TAB 6: GRAPH EXPLORER ==============

with tab6:
    st.header("🕸️ Graph Explorer")
    st.markdown("""
    **Visualize the knowledge graph.** See how products, features, use cases, and brands connect.

    *Filter to explore specific areas of the graph.*
    """)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        categories = get_all_categories()
        category_options = ["All Categories"] + [c['name'] for c in categories]
        selected_category = st.selectbox("Category:", category_options, key="graph_category")

    with col2:
        features = get_all_features()
        feature_options = ["All Features"] + [f['display_name'] for f in features[:15]]
        selected_feature = st.selectbox("Feature:", feature_options, key="graph_feature")

    with col3:
        use_cases = get_all_use_cases()
        use_case_options = ["All Use Cases"] + [u['display_name'] for u in use_cases]
        selected_use_case = st.selectbox("Use Case:", use_case_options, key="graph_use_case")

    # Limit for performance
    max_products = st.slider("Max products to show:", 5, 30, 15, key="graph_limit")

    if st.button("🔍 Explore Graph", type="primary", key="graph_btn"):
        # Build query based on filters
        conditions = []
        params = {"limit": max_products}

        if selected_category != "All Categories":
            conditions.append("(p)-[:IN_CATEGORY]->(:Category {name: $category})")
            params['category'] = selected_category

        if selected_feature != "All Features":
            # Find the feature name from display name
            feat_name = next((f['name'] for f in features if f['display_name'] == selected_feature), None)
            if feat_name:
                conditions.append("(p)-[:HAS_FEATURE]->(:Feature {name: $feature})")
                params['feature'] = feat_name

        if selected_use_case != "All Use Cases":
            uc_name = next((u['name'] for u in use_cases if u['display_name'] == selected_use_case), None)
            if uc_name:
                conditions.append("(p)-[:FOR_USE_CASE]->(:UseCase {name: $use_case})")
                params['use_case'] = uc_name

        where_clause = " AND ".join(conditions) if conditions else "true"

        # Get products and their connections
        graph_data = query(f"""
            MATCH (p:Product)
            WHERE {where_clause}
            WITH p LIMIT $limit
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (p)-[:FOR_USE_CASE]->(u:UseCase)
            OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
            OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
            RETURN p.asin as product_id, p.title as product_title, p.price as price,
                   collect(DISTINCT {{id: f.name, name: f.display_name, type: 'Feature'}}) as features,
                   collect(DISTINCT {{id: u.name, name: u.display_name, type: 'UseCase'}}) as use_cases,
                   c.name as category,
                   b.name as brand
        """, params)

        if graph_data:
            # Build nodes and edges for visualization
            nodes = []
            edges = []
            seen_nodes = set()

            for item in graph_data:
                # Product node
                product_id = f"product_{item['product_id']}"
                if product_id not in seen_nodes:
                    nodes.append({
                        'id': product_id,
                        'label': item['product_title'][:25] + "...",
                        'title': f"{item['product_title']}\n${item['price']:.2f}",
                        'type': 'Product',
                        'size': 25
                    })
                    seen_nodes.add(product_id)

                # Category node
                if item['category']:
                    cat_id = f"category_{item['category']}"
                    if cat_id not in seen_nodes:
                        nodes.append({
                            'id': cat_id,
                            'label': item['category'],
                            'title': f"Category: {item['category']}",
                            'type': 'Category',
                            'size': 30
                        })
                        seen_nodes.add(cat_id)
                    edges.append({'from': product_id, 'to': cat_id, 'label': 'IN_CATEGORY'})

                # Brand node
                if item['brand']:
                    brand_id = f"brand_{item['brand']}"
                    if brand_id not in seen_nodes:
                        nodes.append({
                            'id': brand_id,
                            'label': item['brand'],
                            'title': f"Brand: {item['brand']}",
                            'type': 'Brand',
                            'size': 20
                        })
                        seen_nodes.add(brand_id)
                    edges.append({'from': product_id, 'to': brand_id, 'label': 'MADE_BY'})

                # Feature nodes
                for feat in item['features']:
                    if feat['id']:
                        feat_id = f"feature_{feat['id']}"
                        if feat_id not in seen_nodes:
                            nodes.append({
                                'id': feat_id,
                                'label': feat['name'] or feat['id'],
                                'title': f"Feature: {feat['name'] or feat['id']}",
                                'type': 'Feature',
                                'size': 18
                            })
                            seen_nodes.add(feat_id)
                        edges.append({'from': product_id, 'to': feat_id, 'label': 'HAS_FEATURE'})

                # UseCase nodes
                for uc in item['use_cases']:
                    if uc['id']:
                        uc_id = f"usecase_{uc['id']}"
                        if uc_id not in seen_nodes:
                            nodes.append({
                                'id': uc_id,
                                'label': uc['name'] or uc['id'],
                                'title': f"Use Case: {uc['name'] or uc['id']}",
                                'type': 'UseCase',
                                'size': 18
                            })
                            seen_nodes.add(uc_id)
                        edges.append({'from': product_id, 'to': uc_id, 'label': 'FOR_USE_CASE'})

            # Create visualization
            st.markdown(f"### Graph: {len(nodes)} nodes, {len(edges)} relationships")

            # Legend
            st.markdown("""
            **Legend:**
            🔴 Products | 🟢 Features | 🟡 Use Cases | 🟣 Brands | 🩵 Categories
            """)

            # Generate and display the graph
            html = create_graph_visualization(nodes, edges, height="550px")
            components.html(html, height=600, scrolling=True)

            # Stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Products", len([n for n in nodes if n['type'] == 'Product']))
            with col2:
                st.metric("Features", len([n for n in nodes if n['type'] == 'Feature']))
            with col3:
                st.metric("Use Cases", len([n for n in nodes if n['type'] == 'UseCase']))

            st.info("""
            💡 **Graph visualization shows:**
            - Red nodes = Products
            - Cyan nodes = Features
            - Yellow nodes = Use Cases
            - Purple nodes = Brands
            - Green nodes = Categories

            **Drag nodes** to explore. **Zoom** with scroll. **Click** for details.
            """)
        else:
            st.warning("No products found matching your filters.")


# ============== FOOTER ==============

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<strong>Product Navigator</strong> — Graph RAG for smarter product discovery<br>
<small>Same pattern as research papers: Feature × UseCase, Bundle Builder, Niche Finder</small>
</div>
""", unsafe_allow_html=True)
