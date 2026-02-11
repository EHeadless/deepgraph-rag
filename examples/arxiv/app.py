"""
DeepGraph RAG - Right Tool for the Job

Shows the CORRECT scenarios where each approach excels:
- Vector Search: Topic queries, semantic similarity
- Graph RAG: Exploration from a starting point, relationship queries, cross-field discovery
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.arxiv.arxiv_adapter import get_adapter

load_dotenv()

# Must be first Streamlit command
st.set_page_config(
    page_title="Graph RAG: Right Tool for the Job",
    page_icon="🔗",
    layout="wide"
)

@st.cache_resource
def init_adapter():
    return get_adapter()

adapter = init_adapter()

# ============== CONCEPT NAME MAPPING ==============
CONCEPT_NAMES = {
    "cs.AI": "AI",
    "cs.CL": "NLP",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.MA": "Multi-Agent",
    "cs.RO": "Robotics",
    "cs.IR": "Info Retrieval",
    "cs.HC": "Human-Computer",
    "cs.CR": "Cryptography",
    "cs.SE": "Software Eng",
    "cs.NE": "Neural/Evolutionary",
    "cs.SI": "Social Networks",
    "cs.ET": "Emerging Tech",
    "stat.ML": "Statistical ML",
}

def get_concept_name(code):
    return CONCEPT_NAMES.get(code, code)

# ============== HEADER ==============
st.title("🔗 Graph RAG: Right Tool for the Job")
st.markdown("*Vector search for topics. Graph traversal for exploration & relationships.*")

# Quick stats
stats = adapter.get_graph_stats()
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Papers", f"{stats.get('Paper', 0):,}")
with col2:
    st.metric("Authors", f"{stats.get('Author', 0):,}")
with col3:
    st.metric("Methods", f"{stats.get('Method', 0):,}")
with col4:
    st.metric("Concepts", f"{stats.get('Concept', 0):,}")
with col5:
    st.metric("Datasets", f"{stats.get('Dataset', 0):,}")
with col6:
    st.metric("Relationships", f"{stats.get('rel_AUTHORED_BY', 0) + stats.get('rel_USES_METHOD', 0) + stats.get('rel_DISCUSSES', 0):,}")

st.divider()

# ============== TABS ==============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 When to Use What",
    "🔍 Topic Search (Vector)",
    "🗺️ Explore from Paper (Graph)",
    "🌉 Cross-Field Discovery (Graph)",
    "🔬 Method & Concept Discovery",
    "🎯 Research Navigator"
])

# ============== TAB 1: WHEN TO USE WHAT ==============
with tab1:
    st.header("Choose the Right Tool")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Vector Search")
        st.success("""
        **Best for: "Find papers about X"**

        ✅ Topic queries
        ✅ Semantic similarity
        ✅ Keyword matching

        **Example queries:**
        - "attention mechanisms"
        - "reinforcement learning for robotics"
        - "transformer architectures"

        **How it works:**
        Your query → embedding → find similar embeddings

        **Limitation:** Only finds papers with similar *text*.
        """)

    with col2:
        st.subheader("🔗 Graph RAG")
        st.info("""
        **Best for: Exploration & Deep Discovery**

        ✅ "What papers use transformer + reinforcement learning?"
        ✅ "What NLP methods are now used in Robotics?"
        ✅ "Who bridges Field A and Field B?"
        ✅ "What methods do people use for reasoning research?"

        **Example queries:**
        - Find papers by METHOD (not keywords)
        - Discover method TRANSFER across fields
        - Explore author networks & collaborations
        - Find concept connections via shared techniques

        **How it works:**
        Papers → Methods/Concepts/Authors → traverse relationships

        **Superpower:** Understands paper CONTENT, not just text.
        """)

    st.divider()

    st.subheader("❌ The Wrong Way to Use Graph RAG")

    st.error("""
    **Don't:** Search "agentic AI" and expect useful graph discoveries.

    **What happens:**
    - Vector finds: Papers about agentic AI ✅
    - Graph adds: "Same author's unrelated computer vision papers" 🤷

    The graph connection is *real* (same author) but not *useful* for your topic query.
    Graph RAG adds noise, not signal.
    """)

    st.subheader("✅ The Right Way to Use Graph RAG")

    st.success("""
    **Do:** Start from a specific paper and explore outward.

    **What happens:**
    - You pick: A paper you already know is relevant
    - Graph finds: What else these authors work on, their collaborators' work, adjacent fields

    Now the "same author, different field" discovery IS useful - you're exploring
    a researcher's expertise, not searching for a topic.
    """)

# ============== TAB 2: TOPIC SEARCH ==============
with tab2:
    st.header("🔍 Topic Search")
    st.markdown("**Vector search excels here.** Find papers by semantic similarity.")

    query = st.text_input(
        "Search for papers about:",
        placeholder="e.g., transformer architectures, reinforcement learning, GANs"
    )

    if st.button("🔍 Search", type="primary", key="topic_search") and query:
        with st.spinner("Searching..."):
            results = adapter.vector_search(query, top_k=10)

        st.success(f"Found {len(results)} papers about '{query}'")

        for i, paper in enumerate(results, 1):
            score = paper.get('score', 0)
            arxiv_url = f"https://arxiv.org/abs/{paper['id']}"
            category = get_concept_name(paper.get('category', ''))
            authors = ', '.join(paper.get('authors', [])[:2])

            st.markdown(f"""
            **{i}. [{paper.get('title', 'Unknown')}]({arxiv_url})**

            📊 Similarity: {score:.2f} | 📁 {category} | 👤 {authors}
            """)
            st.markdown("---")

        st.info("""
        💡 **Why Vector Search Works Here:**

        You searched for a *topic*. Vector embeddings found papers with semantically similar content.
        This is exactly what vector search is designed for.

        **Graph RAG wouldn't help** - finding "same author's other papers" doesn't get you more papers about this topic.
        """)

# ============== TAB 3: PAPER EXPLORATION ==============
with tab3:
    st.header("🗺️ Explore from a Paper")
    st.markdown("""
    **Graph RAG excels here.** Start from a paper you know, discover connections you'd never find through search.

    *This is where "same author, different field" actually makes sense - you're exploring a researcher's work, not searching for a topic.*
    """)

    papers = adapter.get_all_papers()
    paper_options = {f"{p['title'][:70]}...": p['id'] for p in papers[:100]}

    selected_title = st.selectbox("Select a starting paper:", list(paper_options.keys()))

    if st.button("🗺️ Explore from this Paper", type="primary", key="explore"):
        paper_id = paper_options[selected_title]
        paper_info = next((p for p in papers if p['id'] == paper_id), None)

        if paper_info:
            # Show starting point
            arxiv_url = f"https://arxiv.org/abs/{paper_id}"
            st.markdown(f"""
            ### 📍 Starting Point

            **[{paper_info['title']}]({arxiv_url})**

            📁 {get_concept_name(paper_info.get('category', ''))} | 👤 {', '.join(paper_info.get('authors', [])[:3])}
            """)

            st.divider()

            col1, col2 = st.columns(2)

            # Same author's OTHER work
            with col1:
                st.markdown("### 👤 Same Authors, Different Work")
                st.markdown("*What else do these researchers study?*")

                authors = paper_info.get('authors', [])[:2]
                other_papers = []

                for author in authors:
                    author_papers = adapter.find_author_papers(author)
                    for ap in author_papers:
                        if ap['id'] != paper_id:
                            ap['via_author'] = author
                            other_papers.append(ap)

                # Prioritize different categories
                diff_field = [p for p in other_papers if p.get('category') != paper_info.get('category')]
                same_field = [p for p in other_papers if p.get('category') == paper_info.get('category')]

                if diff_field:
                    st.success("**Cross-field work by these authors:**")
                    for op in diff_field[:3]:
                        arxiv_url = f"https://arxiv.org/abs/{op['id']}"
                        cat = get_concept_name(op.get('category', ''))
                        st.markdown(f"""
                        📗 **[{op['title'][:60]}...]({arxiv_url})**

                        📁 {cat} | via *{op['via_author']}*

                        💡 *This author works across fields - might use similar techniques!*
                        """)
                        st.markdown("")

                if same_field:
                    st.info("**More work in the same field:**")
                    for op in same_field[:2]:
                        arxiv_url = f"https://arxiv.org/abs/{op['id']}"
                        st.markdown(f"- [{op['title'][:50]}...]({arxiv_url})")

                if not other_papers:
                    st.info("No other papers by these authors in the database.")

            # Collaborator network
            with col2:
                st.markdown("### 🤝 Collaborator Network")
                st.markdown("*Papers by people who worked with these authors*")

                collab_papers = []
                for author in authors[:1]:
                    collaborators = adapter.find_collaborators(author)
                    for c in collaborators[:4]:
                        c_papers = adapter.find_author_papers(c['collaborator'])
                        for cp in c_papers[:1]:
                            if cp['id'] != paper_id:
                                cp['via'] = f"{author} → {c['collaborator']}"
                                cp['collaborator'] = c['collaborator']
                                collab_papers.append(cp)

                if collab_papers:
                    for cp in collab_papers[:4]:
                        arxiv_url = f"https://arxiv.org/abs/{cp['id']}"
                        cat = get_concept_name(cp.get('category', ''))
                        st.markdown(f"""
                        📘 **[{cp['title'][:60]}...]({arxiv_url})**

                        📁 {cat} | 🔗 via *{cp['collaborator']}*

                        💡 *2 hops away in the collaboration graph!*
                        """)
                        st.markdown("")
                else:
                    st.info("No collaborator papers found.")

            st.divider()

            st.success("""
            ✅ **Why Graph RAG Works Here:**

            You started with a *specific paper* and explored *relationships*.

            - "Same author, different field" → now makes sense! You're exploring what else they know.
            - "Collaborator network" → expands your research horizon to related work.

            **This is impossible with vector search.** It can only find papers with similar text.
            """)

# ============== TAB 4: CROSS-FIELD DISCOVERY ==============
with tab4:
    st.header("🌉 Cross-Field Discovery")
    st.markdown("""
    **Only Graph RAG can do this.** Find researchers and papers that bridge different fields.

    *Vector search has no concept of "author" or "field" - it only sees text.*
    """)

    concepts = adapter.get_all_concepts()
    concept_options = [c['concept'] for c in concepts]

    col1, col2 = st.columns(2)
    with col1:
        field1 = st.selectbox("Field 1:", concept_options, index=0, format_func=get_concept_name)
    with col2:
        default_idx = min(1, len(concept_options) - 1)
        field2 = st.selectbox("Field 2:", concept_options, index=default_idx, format_func=get_concept_name)

    if st.button("🌉 Find Bridge Researchers", type="primary", key="bridge"):
        if field1 == field2:
            st.warning("Please select two different fields.")
        else:
            with st.spinner("Finding researchers who publish in both fields..."):
                bridge_query = """
                MATCH (p1:Paper {primary_category: $field1})-[:AUTHORED_BY]->(a:Author)
                      <-[:AUTHORED_BY]-(p2:Paper {primary_category: $field2})
                WHERE p1 <> p2
                WITH a, collect(DISTINCT p1.title) as f1_papers,
                     collect(DISTINCT p2.title) as f2_papers
                WHERE size(f1_papers) > 0 AND size(f2_papers) > 0
                RETURN a.name as author,
                       f1_papers[0..3] as papers_in_field1,
                       f2_papers[0..3] as papers_in_field2,
                       size(f1_papers) as count1,
                       size(f2_papers) as count2
                ORDER BY count1 + count2 DESC
                LIMIT 10
                """

                try:
                    results = adapter.store.query(bridge_query, {
                        "field1": field1,
                        "field2": field2
                    })

                    f1_name = get_concept_name(field1)
                    f2_name = get_concept_name(field2)

                    if results:
                        st.success(f"Found {len(results)} researchers bridging **{f1_name}** and **{f2_name}**!")

                        for r in results:
                            with st.expander(f"👤 {r['author']} ({r['count1']} + {r['count2']} papers)"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(f"**📁 {f1_name}:**")
                                    for p in r['papers_in_field1']:
                                        st.markdown(f"- {p[:50]}...")

                                with col2:
                                    st.markdown(f"**📁 {f2_name}:**")
                                    for p in r['papers_in_field2']:
                                        st.markdown(f"- {p[:50]}...")

                        st.divider()

                        st.success("""
                        ✅ **This query is IMPOSSIBLE with vector search.**

                        We asked: "Who publishes in BOTH fields?"

                        This requires traversing the graph:
                        ```
                        Field1 Papers → Authors → Field2 Papers
                        ```

                        No amount of text similarity can answer this question.
                        You need to traverse relationships.
                        """)
                    else:
                        st.info(f"No researchers found bridging {f1_name} and {f2_name}.")

                except Exception as e:
                    st.error(f"Query error: {e}")

    st.divider()

    # Show existing connections
    st.subheader("🔗 Field Connections in the Graph")
    st.markdown("*Which fields share researchers?*")

    connections = adapter.get_concept_cooccurrence()
    if connections:
        for conn in connections[:8]:
            c1_name = get_concept_name(conn['concept1'])
            c2_name = get_concept_name(conn['concept2'])
            shared = conn.get('shared_authors', 0)

            if shared > 0:
                st.markdown(f"**{c1_name}** ↔ **{c2_name}**: {shared} shared researchers")

# ============== TAB 5: METHOD & CONCEPT DISCOVERY ==============
with tab5:
    st.header("🔬 Method & Concept Discovery")
    st.markdown("""
    **Deep Graph RAG power.** Query based on *what methods papers use* and *what concepts they discuss*.

    *This goes beyond keywords - the graph knows which papers use transformers, which discuss reasoning, etc.*
    """)

    # Quick stats on methods/concepts
    methods = adapter.get_all_methods()
    topics = adapter.get_all_topics()
    datasets = adapter.get_all_datasets()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Methods Extracted", len(methods))
    with col2:
        st.metric("Research Topics", len(topics))
    with col3:
        st.metric("Datasets Referenced", len(datasets))

    st.divider()

    # Sub-tabs for different discovery modes
    subtab1, subtab2, subtab3 = st.tabs([
        "🔧 Find by Method",
        "💡 Find by Concept",
        "🔄 Method Transfer Across Fields"
    ])

    # ---- SUBTAB 1: Find by Method ----
    with subtab1:
        st.subheader("🔧 Find Papers by Method")
        st.markdown("*What papers use transformers? LoRA? Diffusion models?*")

        if methods:
            # Method selector with counts
            method_options = {f"{m.get('display_name', m['method'])} ({m['paper_count']} papers)": m['method']
                            for m in methods[:20]}
            selected_method_display = st.selectbox("Select a method:", list(method_options.keys()))

            if st.button("🔍 Find Papers Using This Method", type="primary", key="method_search"):
                method_name = method_options[selected_method_display]

                with st.spinner("Searching..."):
                    papers = adapter.find_papers_by_method(method_name, limit=15)

                if papers:
                    st.success(f"Found {len(papers)} papers using **{method_name.replace('_', ' ').title()}**")

                    for paper in papers:
                        arxiv_url = f"https://arxiv.org/abs/{paper['id']}"
                        cat = get_concept_name(paper.get('category', ''))
                        authors = ', '.join(paper.get('authors', [])[:2])
                        st.markdown(f"""
                        📄 **[{paper['title'][:70]}...]({arxiv_url})**

                        📁 {cat} | 👤 {authors}
                        """)
                        st.markdown("---")

                    st.info("""
                    💡 **Why This is Powerful:**

                    Vector search for "transformer" finds papers that *mention* the word.
                    This finds papers that actually *use transformer architecture* - even if the abstract
                    doesn't explicitly say "transformer" (e.g., "we use attention-based encoding...").
                    """)
                else:
                    st.info("No papers found using this method.")

    # ---- SUBTAB 2: Find by Concept ----
    with subtab2:
        st.subheader("💡 Find Papers by Research Concept")
        st.markdown("*What papers discuss reasoning? Multimodal learning? Robustness?*")

        if topics:
            # Concept selector with counts
            topic_options = {f"{t.get('display_name', t['concept'])} ({t['paper_count']} papers)": t['concept']
                           for t in topics[:20]}
            selected_topic_display = st.selectbox("Select a research concept:", list(topic_options.keys()))

            # Also show which methods are commonly used for this concept
            if st.button("🔍 Find Papers & Methods for This Concept", type="primary", key="concept_search"):
                topic_name = topic_options[selected_topic_display]

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("### Papers Discussing This Concept")
                    with st.spinner("Searching..."):
                        papers = adapter.find_papers_by_topic(topic_name, limit=10)

                    if papers:
                        for paper in papers:
                            arxiv_url = f"https://arxiv.org/abs/{paper['id']}"
                            cat = get_concept_name(paper.get('category', ''))
                            st.markdown(f"📄 **[{paper['title'][:60]}...]({arxiv_url})** | {cat}")
                    else:
                        st.info("No papers found.")

                with col2:
                    st.markdown("### Methods Used")
                    with st.spinner("Finding methods..."):
                        concept_methods = adapter.find_methods_for_concept(topic_name, limit=8)

                    if concept_methods:
                        for cm in concept_methods:
                            st.markdown(f"🔧 **{cm['display_name']}** ({cm['usage_count']} papers)")
                    else:
                        st.info("No methods data.")

                st.divider()
                st.success("""
                ✅ **Graph Power:** We found papers that *discuss* this concept AND the methods they use.

                This enables queries like: "I want to work on reasoning - what methods do people use?"
                """)

    # ---- SUBTAB 3: Method Transfer ----
    with subtab3:
        st.subheader("🔄 Method Transfer Across Fields")
        st.markdown("""
        **The most powerful graph query.** Find methods that started in one field and are now being used in another.

        *Example: "What NLP methods are now being used in Computer Vision?"*
        """)

        concepts = adapter.get_all_concepts()
        concept_list = [c['concept'] for c in concepts]

        col1, col2 = st.columns(2)
        with col1:
            source_field = st.selectbox("Methods FROM:", concept_list, index=0, format_func=get_concept_name, key="source")
        with col2:
            target_idx = min(1, len(concept_list) - 1)
            target_field = st.selectbox("Used IN:", concept_list, index=target_idx, format_func=get_concept_name, key="target")

        if st.button("🔄 Find Method Transfer", type="primary", key="transfer_search"):
            if source_field == target_field:
                st.warning("Please select two different fields.")
            else:
                source_name = get_concept_name(source_field)
                target_name = get_concept_name(target_field)

                with st.spinner(f"Finding methods that transferred from {source_name} to {target_name}..."):
                    transfers = adapter.find_method_transfer(source_field, target_field, limit=10)

                if transfers:
                    st.success(f"Found {len(transfers)} methods transferring from **{source_name}** → **{target_name}**!")

                    for t in transfers:
                        st.markdown(f"""
                        ### 🔧 {t['display_name']}

                        - **{source_name}:** {t['source_count']} papers
                        - **{target_name}:** {t['target_count']} papers

                        **Example papers in {target_name}:**
                        """)
                        for ex in t.get('example_papers', []):
                            st.markdown(f"  - {ex[:60]}...")

                        st.markdown("---")

                    st.success("""
                    ✅ **This query is IMPOSSIBLE with vector search.**

                    We asked: "What methods originated in field A and are now used in field B?"

                    This requires:
                    1. Knowing what methods each paper uses (extracted from content)
                    2. Knowing each paper's field
                    3. Finding overlap across fields

                    Pure graph reasoning!
                    """)
                else:
                    st.info(f"No method transfer found from {source_name} to {target_name}.")

    st.divider()

    # Show connected concepts (bonus)
    st.subheader("🔗 Concepts Connected by Shared Methods")
    st.markdown("*Research areas that use similar techniques*")

    bridges = adapter.find_concept_method_bridge(limit=6)
    if bridges:
        for b in bridges:
            methods_str = ', '.join(b['shared_methods'][:3])
            st.markdown(f"**{b['concept1']}** ↔ **{b['concept2']}** via: {methods_str}")
    else:
        st.info("Not enough data for method-concept bridges yet.")

# ============== TAB 6: RESEARCH NAVIGATOR ==============
with tab6:
    st.header("🎯 Research Navigator")
    st.markdown("""
    **Too many papers? Narrow down intelligently.**

    *Combine method + concept filters, find reading paths, discover research gaps.*
    """)

    with st.expander("💡 How This Works (Graph RAG in Action)", expanded=False):
        st.markdown("""
        ### The Problem
        You search "transformers" and get 500 papers. Which ones matter? Where do you start?

        ### The Graph RAG Solution
        We built a **knowledge graph** that understands paper *content*, not just text:

        ```
        Paper ──→ USES_METHOD ──→ [transformer, attention, LoRA...]
          ├───→ DISCUSSES ────→ [reasoning, multimodal, efficiency...]
          ├───→ USES_DATASET ─→ [ImageNet, COCO, MMLU...]
          └───→ AUTHORED_BY ──→ Author ──→ Collaborator Network
        ```

        ### What You Can Do Here

        | Feature | What It Does | Why It's Impossible with Vector Search |
        |---------|--------------|----------------------------------------|
        | **Method × Concept Filter** | "Transformers FOR reasoning" | Vector finds keywords. We find papers that *actually use* the method *for* the concept. |
        | **Reading Path** | Ordered list: foundations → target | Based on method overlap, not citations. Shows *what techniques* to learn first. |
        | **Research Frontier** | Rare method+concept combos | Requires counting across graph relationships. Pure structure, no text. |

        ### The Key Insight
        **Vector search** answers: "What papers mention these words?"
        **Graph RAG** answers: "What papers use this technique for this problem?"

        That's the difference between 500 results and 5 perfect ones.
        """)

    nav_tab1, nav_tab2, nav_tab3 = st.tabs([
        "🔀 Method × Concept Filter",
        "📚 Build Reading Path",
        "🔮 Research Frontier"
    ])

    # ---- NAV TAB 1: Method × Concept Intersection ----
    with nav_tab1:
        st.subheader("🔀 Find Papers at the Intersection")
        st.markdown("""
        **The killer feature.** Find papers that use a specific METHOD for a specific CONCEPT.

        *Example: "Papers using **transformers** for **reasoning**" or "**reinforcement learning** in **robotics**"*
        """)

        col1, col2 = st.columns(2)

        methods_list = adapter.get_all_methods()
        topics_list = adapter.get_all_topics()

        with col1:
            method_opts = ["Any method"] + [m.get('display_name', m['method']) for m in methods_list[:15]]
            selected_method = st.selectbox("Method:", method_opts, key="nav_method")

        with col2:
            topic_opts = ["Any concept"] + [t.get('display_name', t['concept']) for t in topics_list[:15]]
            selected_topic = st.selectbox("Concept:", topic_opts, key="nav_topic")

        if st.button("🎯 Find Intersection", type="primary", key="intersection_btn"):
            # Build the query based on selections
            method_name = None
            topic_name = None

            if selected_method != "Any method":
                method_name = next((m['method'] for m in methods_list if m.get('display_name', m['method']) == selected_method), None)
            if selected_topic != "Any concept":
                topic_name = next((t['concept'] for t in topics_list if t.get('display_name', t['concept']) == selected_topic), None)

            # Query for intersection
            if method_name and topic_name:
                query = """
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: $method})
                    MATCH (p)-[:DISCUSSES]->(c:Concept {name: $topic})
                    MATCH (p)-[:AUTHORED_BY]->(a:Author)
                    OPTIONAL MATCH (p)-[:USES_METHOD]->(other_m:Method)
                    OPTIONAL MATCH (p)-[:DISCUSSES]->(other_c:Concept {type: 'topic'})
                    RETURN p.title as title, p.arxiv_id as id, p.primary_category as category,
                           collect(DISTINCT a.name)[0..2] as authors,
                           collect(DISTINCT other_m.display_name) as all_methods,
                           collect(DISTINCT other_c.display_name) as all_concepts
                    LIMIT 15
                """
                results = adapter.store.query(query, {"method": method_name, "topic": topic_name})
            elif method_name:
                results = adapter.find_papers_by_method(method_name, limit=15)
            elif topic_name:
                results = adapter.find_papers_by_topic(topic_name, limit=15)
            else:
                results = []

            if results:
                st.success(f"**Found {len(results)} papers** matching your criteria!")
                st.markdown("---")

                for i, paper in enumerate(results, 1):
                    arxiv_url = f"https://arxiv.org/abs/{paper['id']}"
                    cat = get_concept_name(paper.get('category', ''))
                    authors = ', '.join(paper.get('authors', [])[:2]) if paper.get('authors') else ''

                    # Show methods and concepts if available
                    methods_str = ', '.join(paper.get('all_methods', [])[:4]) if paper.get('all_methods') else ''
                    concepts_str = ', '.join(paper.get('all_concepts', [])[:4]) if paper.get('all_concepts') else ''

                    st.markdown(f"""
                    **{i}. [{paper['title'][:80]}...]({arxiv_url})**

                    📁 {cat} | 👤 {authors}
                    """)

                    if methods_str or concepts_str:
                        col1, col2 = st.columns(2)
                        with col1:
                            if methods_str:
                                st.caption(f"🔧 Methods: {methods_str}")
                        with col2:
                            if concepts_str:
                                st.caption(f"💡 Concepts: {concepts_str}")

                    st.markdown("---")

                st.info(f"""
                💡 **Why this is powerful:**

                Vector search for "{selected_method} {selected_topic}" would find papers mentioning these words.
                This finds papers that **actually use {selected_method}** to **address {selected_topic}** — based on content analysis, not keyword matching.
                """)
            else:
                st.warning("No papers found at this intersection. Try broadening your criteria.")

    # ---- NAV TAB 2: Reading Path ----
    with nav_tab2:
        st.subheader("📚 Build Your Reading Path")
        st.markdown("""
        **Pick a target paper. We'll find what to read first.**

        *Shows papers with simpler/foundational methods that lead to understanding your target.*
        """)

        papers = adapter.get_all_papers()
        paper_options = {f"{p['title'][:70]}...": p['id'] for p in papers[:100]}

        target_paper = st.selectbox("Select your target paper:", list(paper_options.keys()), key="reading_target")

        if st.button("📚 Generate Reading Path", type="primary", key="reading_path_btn"):
            paper_id = paper_options[target_paper]

            # Get target paper's full context
            target_context = adapter.get_paper_full_context(paper_id)

            if target_context:
                st.markdown("### 🎯 Your Target Paper")
                arxiv_url = f"https://arxiv.org/abs/{paper_id}"
                st.markdown(f"**[{target_context['title']}]({arxiv_url})**")

                target_methods = target_context.get('methods', [])
                target_concepts = target_context.get('concepts', [])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Methods used:**")
                    for m in target_methods[:5]:
                        st.markdown(f"- 🔧 {m}")
                with col2:
                    st.markdown("**Concepts discussed:**")
                    for c in target_concepts[:5]:
                        st.markdown(f"- 💡 {c}")

                st.markdown("---")
                st.markdown("### 📖 Suggested Reading Path")
                st.markdown("*Papers that share methods/concepts — read these to build up to your target:*")

                # Find papers with overlapping methods (simpler combinations)
                similar = adapter.find_similar_papers_by_methods(paper_id, limit=8)

                if similar:
                    # Group by overlap level
                    high_overlap = [p for p in similar if p.get('method_overlap', 0) >= 2]
                    low_overlap = [p for p in similar if p.get('method_overlap', 0) == 1]

                    if low_overlap:
                        st.markdown("#### 1️⃣ Foundational (1 shared method)")
                        st.caption("Start here — introduces individual techniques")
                        for p in low_overlap[:3]:
                            arxiv_url = f"https://arxiv.org/abs/{p['id']}"
                            shared = ', '.join(p.get('shared_methods', []))
                            st.markdown(f"📄 [{p['title'][:60]}...]({arxiv_url})")
                            st.caption(f"   Shares: {shared}")

                    if high_overlap:
                        st.markdown("#### 2️⃣ Intermediate (2+ shared methods)")
                        st.caption("Then these — combines multiple techniques")
                        for p in high_overlap[:4]:
                            arxiv_url = f"https://arxiv.org/abs/{p['id']}"
                            shared = ', '.join(p.get('shared_methods', []))
                            cat = get_concept_name(p.get('category', ''))
                            st.markdown(f"📄 [{p['title'][:60]}...]({arxiv_url}) | {cat}")
                            st.caption(f"   Shares: {shared}")

                    st.markdown("#### 3️⃣ Target")
                    st.markdown(f"📄 **{target_context['title'][:60]}...** ← *You're ready for this!*")

                    st.success("""
                    ✅ **Reading path generated based on METHOD overlap.**

                    Unlike "related papers" from citations, this path is built on *what techniques* each paper uses.
                    You'll build up the prerequisite knowledge to understand your target paper.
                    """)
                else:
                    st.info("Couldn't find related papers with method overlap. Try a different target paper.")

    # ---- NAV TAB 3: Research Frontier ----
    with nav_tab3:
        st.subheader("🔮 Research Frontier")
        st.markdown("""
        **Find the gaps.** What method + concept combinations are RARE?

        *These are potential research opportunities — methods not yet widely applied to certain problems.*
        """)

        if st.button("🔮 Discover Research Gaps", type="primary", key="frontier_btn"):
            # Find rare method-concept combinations
            frontier_query = """
                // Find all method-concept pairs with their counts
                MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                MATCH (p)-[:DISCUSSES]->(c:Concept {type: 'topic'})
                WITH m.display_name as method, c.display_name as concept, count(p) as papers
                WHERE papers >= 1 AND papers <= 3
                RETURN method, concept, papers
                ORDER BY papers ASC, method
                LIMIT 20
            """

            rare_combos = adapter.store.query(frontier_query, {})

            if rare_combos:
                st.success(f"**Found {len(rare_combos)} under-explored combinations!**")
                st.markdown("*These method + concept pairs have very few papers — potential research opportunities:*")
                st.markdown("---")

                # Group by paper count
                for combo in rare_combos:
                    papers = combo['papers']
                    icon = "🟢" if papers == 1 else "🟡" if papers == 2 else "🟠"
                    st.markdown(f"{icon} **{combo['method']}** × **{combo['concept']}** — only {papers} paper(s)")

                st.markdown("---")
                st.info("""
                💡 **Research Opportunity Analysis:**

                🟢 = 1 paper (highly novel combination)
                🟡 = 2 papers (emerging area)
                🟠 = 3 papers (growing but still early)

                **Try combining these!** A method that works well in one area might be transformative in another.
                """)

                # Also show popular methods not yet applied to certain concepts
                st.markdown("### 🚀 High-Impact Opportunities")
                st.markdown("*Popular methods that could be applied to trending concepts:*")

                opportunity_query = """
                    // Find popular methods
                    MATCH (m:Method)<-[:USES_METHOD]-(p1:Paper)
                    WITH m, count(p1) as method_popularity
                    WHERE method_popularity >= 10
                    // Find popular concepts
                    MATCH (c:Concept {type: 'topic'})<-[:DISCUSSES]-(p2:Paper)
                    WITH m, method_popularity, c, count(p2) as concept_popularity
                    WHERE concept_popularity >= 10
                    // Check how many papers combine them
                    OPTIONAL MATCH (p3:Paper)-[:USES_METHOD]->(m)
                    WHERE (p3)-[:DISCUSSES]->(c)
                    WITH m, c, method_popularity, concept_popularity, count(p3) as combined
                    WHERE combined <= 2
                    RETURN m.display_name as method, c.display_name as concept,
                           method_popularity, concept_popularity, combined
                    ORDER BY (method_popularity + concept_popularity) DESC
                    LIMIT 8
                """

                opportunities = adapter.store.query(opportunity_query, {})

                if opportunities:
                    for opp in opportunities:
                        combined = opp['combined']
                        if combined == 0:
                            st.markdown(f"🔥 **{opp['method']}** + **{opp['concept']}** = **UNEXPLORED!**")
                            st.caption(f"   Method used in {opp['method_popularity']} papers, concept in {opp['concept_popularity']} — but never combined!")
                        else:
                            st.markdown(f"⚡ **{opp['method']}** + **{opp['concept']}** = only {combined} paper(s)")
                            st.caption(f"   Both popular ({opp['method_popularity']} and {opp['concept_popularity']} papers) but rarely combined")
            else:
                st.info("Run the concept extraction script first to populate method-concept relationships.")

        st.markdown("---")
        st.markdown("""
        ### 🧭 How to Use This

        1. **Find a gap** that interests you
        2. **Go to Method × Concept Filter** to see existing work (if any)
        3. **Build a Reading Path** from foundational papers
        4. **You now have a research direction + reading list!**
        """)

# ============== FOOTER ==============
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<strong>Graph RAG</strong> — Right tool for the job<br>
<small>Vector for topics | Graph for exploration & relationships</small>
</div>
""", unsafe_allow_html=True)
