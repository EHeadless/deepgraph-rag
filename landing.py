"""
DeepGraph RAG - Landing Page

A clean landing page that explains Graph RAG and provides access to both demos:
- Research Navigator (arXiv papers)
- Product Navigator (product recommendations)
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="DeepGraph RAG - Graph-Powered Discovery",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .main-header h1 {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .tagline {
        font-size: 1.5rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .concept-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #333;
    }
    .demo-card {
        background: #1e1e2e;
        border-radius: 15px;
        padding: 1.5rem;
        height: 100%;
        border: 1px solid #333;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .demo-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .demo-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .comparison-table {
        width: 100%;
        margin: 1rem 0;
    }
    .comparison-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #333;
    }
    .check { color: #4ecdc4; }
    .cross { color: #ff6b6b; }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>DeepGraph RAG</h1>
    <p class="tagline">When Vector Search Isn't Enough</p>
</div>
""", unsafe_allow_html=True)

# Main concept explanation
st.markdown("""
<div class="concept-box">
<h3 style="color: #4ecdc4; margin-top: 0;">The Problem</h3>
<p style="font-size: 1.1rem; line-height: 1.8;">
You're drowning in information. <strong>Too many papers to read. Too many products to compare.</strong>
Vector search helps you find things that <em>mention</em> what you want. But what if you need to find things
based on <em>relationships</em>?
</p>

<h3 style="color: #ff6b6b;">The Solution: Graph RAG</h3>
<p style="font-size: 1.1rem; line-height: 1.8;">
Graph RAG combines the semantic power of embeddings with the structural intelligence of knowledge graphs.
Instead of just matching text, it understands <strong>connections</strong>: what methods a paper uses,
what features a product has, who collaborates with whom, what gets bought together.
</p>

<p style="font-size: 1.2rem; color: #ffe66d; margin-top: 1.5rem;">
<strong>The result?</strong> Queries that were impossible become trivial:<br>
<em>"Papers using transformers FOR reasoning"</em> · <em>"Wireless headphones FOR running under $100"</em> · <em>"Symptoms shared by Alzheimer's AND Parkinson's"</em>
</p>
</div>
""", unsafe_allow_html=True)

# Quick comparison
st.markdown("### Vector Search vs Graph RAG")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Vector Search** finds by similarity:
    - "Papers about transformers" ✅
    - "Products mentioning wireless" ✅
    - Good for topic discovery
    """)

with col2:
    st.markdown("""
    **Graph RAG** finds by structure:
    - "Papers USING transformers FOR reasoning" ✅
    - "Wireless products FOR running" ✅
    - Enables multi-hop reasoning
    """)

st.divider()

# Demo cards
st.markdown("## Try It Yourself")
st.markdown("*Three domains, same powerful pattern*")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="demo-card">
        <div class="demo-icon">📚</div>
        <h2>Research Navigator</h2>
        <p style="color: #aaa; font-size: 1rem;">
        Explore 1,000 AI/ML papers from arXiv. Find papers by <strong>method × concept</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    **What you can do:**
    - 🔀 **Method × Concept Filter**
    - 📚 **Reading Path Builder**
    - 🔮 **Research Frontier**
    - 👥 **Author Network**
    """)

    st.markdown("")
    if st.button("🚀 Launch Research", type="primary", use_container_width=True, key="research_btn"):
        st.markdown("""
        <meta http-equiv="refresh" content="0; url=http://localhost:8505">
        """, unsafe_allow_html=True)
        st.info("Opening at http://localhost:8505")
        st.markdown("[Click here →](http://localhost:8505)")

with col2:
    st.markdown("""
    <div class="demo-card">
        <div class="demo-icon">🛒</div>
        <h2>Product Navigator</h2>
        <p style="color: #aaa; font-size: 1rem;">
        Explore 500 electronics products. Find products by <strong>feature × use case</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    **What you can do:**
    - 🔀 **Feature × UseCase Filter**
    - 📦 **Bundle Builder**
    - 🔮 **Niche Finder**
    - 🕸️ **Graph Explorer**
    """)

    st.markdown("")
    if st.button("🚀 Launch Products", type="primary", use_container_width=True, key="product_btn"):
        st.markdown("""
        <meta http-equiv="refresh" content="0; url=http://localhost:8506">
        """, unsafe_allow_html=True)
        st.info("Opening at http://localhost:8506")
        st.markdown("[Click here →](http://localhost:8506)")

with col3:
    st.markdown("""
    <div class="demo-card">
        <div class="demo-icon">🧠</div>
        <h2>Neurology Navigator</h2>
        <p style="color: #aaa; font-size: 1rem;">
        Explore 495 papers + 31K patient reports. Compare <strong>research vs patient</strong> data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    **What you can do:**
    - 🔀 **Disease × Symptom Filter**
    - ⚖️ **Research vs Patient**
    - 🔗 **Shared Symptoms**
    - 💊 **Treatment Finder**
    """)

    st.markdown("")
    if st.button("🚀 Launch Neurology", type="primary", use_container_width=True, key="neuro_btn"):
        st.markdown("""
        <meta http-equiv="refresh" content="0; url=http://localhost:8507">
        """, unsafe_allow_html=True)
        st.info("Opening at http://localhost:8507")
        st.markdown("[Click here →](http://localhost:8507)")

st.divider()

# The pattern explanation
st.markdown("## The Universal Pattern")

st.markdown("""
All three demos use the **exact same Graph RAG pattern**, just with different schemas:

| Research | Products | Neurology | Pattern |
|----------|----------|-----------|---------|
| Paper | Product | Paper / RedditPost | Entity node |
| USES_METHOD → Method | HAS_FEATURE → Feature | MENTIONS_SYMPTOM → Symptom | Capability edge |
| DISCUSSES → Concept | FOR_USE_CASE → UseCase | MENTIONS_DISEASE → Disease | Intent edge |
| AUTHORED_BY → Author | MADE_BY → Brand | INVOLVES_PROTEIN → Protein | Attribution edge |
| **Method × Concept** | **Feature × UseCase** | **Research vs Patient** | **The killer query** |

This means you can apply Graph RAG to **any domain**: CRM, healthcare, legal documents,
supply chain — anywhere you have entities and relationships.
""")

# Technical details (collapsed)
with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Stack:**
    - **Graph Database:** Neo4j with vector indexes
    - **Embeddings:** OpenAI text-embedding-ada-002
    - **LLM:** GPT-4 for synthesis
    - **Frontend:** Streamlit
    - **Visualization:** pyvis for graph rendering

    **Key Cypher Pattern:**
    ```cypher
    // The intersection query that vector search can't do
    MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: "transformer"})
    MATCH (p)-[:DISCUSSES]->(c:Concept {name: "reasoning"})
    RETURN p.title, p.abstract
    ```

    **Framework:** Built on DeepGraph RAG framework with:
    - Domain-agnostic schema definitions
    - Pluggable graph backends
    - Separated retrieval, reasoning, and synthesis stages
    """)

# Footer
st.markdown("""
<div class="footer">
    <strong>DeepGraph RAG</strong> — Graph-powered discovery for the AI age<br>
    <small>Built with Neo4j, OpenAI, and Streamlit</small>
</div>
""", unsafe_allow_html=True)
