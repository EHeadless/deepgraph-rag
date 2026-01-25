"""
Research Explorer - Discover ideas, not just papers
"""
import os
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import json
import re

load_dotenv()

@st.cache_resource
def init_connections():
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "deepgraph2025")
    )
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return driver, client

driver, client = init_connections()

# ============== DATA FUNCTIONS ==============

@st.cache_data
def get_graph_stats():
    with driver.session() as session:
        papers = session.run("MATCH (p:Paper) RETURN count(p) as c").single()['c']
        authors = session.run("MATCH (a:Author) RETURN count(a) as c").single()['c']
        connections = session.run("MATCH ()-[r:CO_AUTHORED]->() RETURN count(r) as c").single()['c']
        return {"papers": papers, "authors": authors, "connections": connections}

@st.cache_data
def get_all_papers():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.arxiv_id as id, p.title as title, p.published_date as date,
                   collect(a.name) as authors
            ORDER BY p.published_date DESC
        """)
        return [dict(r) for r in result]

@st.cache_data
def get_all_authors():
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            RETURN a.name as name, count(p) as paper_count
            ORDER BY paper_count DESC
        """)
        return [dict(r) for r in result]

@st.cache_data
def get_papers_by_theme(theme: str):
    response = client.embeddings.create(model="text-embedding-ada-002", input=theme)
    embedding = response.data[0].embedding
    
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('paper_embedding', 15, $embedding)
            YIELD node, score
            MATCH (node)-[:AUTHORED_BY]->(a:Author)
            RETURN node.title as title, node.arxiv_id as id, score,
                   collect(a.name) as authors
        """, embedding=embedding)
        return [dict(r) for r in result]

@st.cache_data
def get_author_details(author_name: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            RETURN a.name as name, 
                   collect(DISTINCT {title: p.title, id: p.arxiv_id}) as papers,
                   collect(DISTINCT coauthor.name) as collaborators
            LIMIT 1
        """, name=author_name)
        return [dict(r) for r in result]

@st.cache_data
def get_paper_details(arxiv_id: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper {arxiv_id: $id})
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(other:Paper)
            WHERE other <> p
            RETURN p.title as title, p.arxiv_id as id,
                   collect(DISTINCT a.name) as authors,
                   collect(DISTINCT {title: other.title, id: other.arxiv_id})[0..5] as related
        """, id=arxiv_id)
        return [dict(r) for r in result]

def explore_author_ecosystem(author_name: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p1:Paper)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            MATCH (coauthor)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN DISTINCT 
                a.name as author,
                coauthor.name as collaborator,
                p2.title as paper,
                p2.arxiv_id as id
            LIMIT 20
        """, name=author_name)
        return [dict(r) for r in result]

def explore_paper_connections(arxiv_id: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper {arxiv_id: $id})-[:AUTHORED_BY]->(a:Author)
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p
            RETURN DISTINCT
                p.title as source,
                a.name as via_author,
                p2.title as related_paper,
                p2.arxiv_id as related_id
            LIMIT 15
        """, id=arxiv_id)
        return [dict(r) for r in result]

# ============== AI FUNCTIONS ==============

@st.cache_data(ttl=1)
def generate_themes():
    papers = get_all_papers()
    titles = [p['title'] for p in papers[:100]]
    titles_text = "\n".join(titles)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": """Analyze these AI research paper titles and extract 8-10 main themes/topics.
For each theme provide:
- name: Short name (2-3 words)
- count: Estimated number of papers about this
- description: One sentence description

Return as JSON array: [{"name": "...", "count": N, "description": "..."}, ...]"""},
            {"role": "user", "content": titles_text}
        ],
        temperature=0.3
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return [
            {"name": "Large Language Models", "count": 80, "description": "Research on LLMs"},
            {"name": "Computer Vision", "count": 50, "description": "Image and video understanding"},
            {"name": "Reinforcement Learning", "count": 40, "description": "RL methods and applications"},
            {"name": "Transformers", "count": 60, "description": "Attention-based architectures"}
        ]

@st.cache_data(ttl=86400)
def generate_digest():
    papers = get_all_papers()
    titles = [f"- {p['title']} (by {', '.join(p['authors'][:2])})" for p in papers[:50]]
    titles_text = "\n".join(titles)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": """You are a research journalist writing a digest of recent AI research.

Write a 3-4 paragraph narrative summary that:
1. Groups papers by theme
2. Highlights the most interesting/novel ideas
3. Names specific authors when relevant
4. Is engaging and easy to scan

IMPORTANT: Wrap key ideas/concepts in [[double brackets]] like this: "Researchers are exploring [[model compression techniques]] to make AI more efficient."

These bracketed terms will become clickable links. Use 8-12 bracketed terms throughout."""},
            {"role": "user", "content": f"Recent papers:\n{titles_text}"}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

@st.cache_data(ttl=3600)
def extract_paper_insight(title: str):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": """Given this paper title, provide:
1. one_liner: Plain English summary (what a journalist would write)
2. key_idea: The main contribution (2-4 words)
3. why_care: Why should researchers care? (1 sentence)

Return as JSON: {"one_liner": "...", "key_idea": "...", "why_care": "..."}"""},
            {"role": "user", "content": title}
        ],
        temperature=0.3,
        max_tokens=200
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"one_liner": title, "key_idea": "Research", "why_care": "Novel contribution to the field."}

# ============== GRAPH FUNCTIONS ==============

def build_idea_graph(idea: str):
    papers = get_papers_by_theme(idea)
    
    net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="#ffffff")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
    
    net.add_node("idea", label=idea, color="#FF6B6B", size=40, shape="ellipse", font={'size': 16, 'color': '#ffffff'})
    
    added_authors = set()
    
    for i, p in enumerate(papers[:10]):
        paper_label = p['title'][:30] + "..."
        net.add_node(f"paper_{i}", label=paper_label, color="#4ECDC4", size=20, 
                    shape="box", title=p['title'], font={'color': '#ffffff'})
        net.add_edge("idea", f"paper_{i}", color="#666666")
        
        for author in p['authors'][:2]:
            if author not in added_authors:
                net.add_node(author, label=author, color="#45B7D1", size=15, shape="ellipse", font={'color': '#ffffff'})
                added_authors.add(author)
            net.add_edge(f"paper_{i}", author, color="#444444")
    
    return net, papers

def build_author_graph(author_name: str):
    details = get_author_details(author_name)
    if not details:
        return None, None
    
    d = details[0]
    
    net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="#ffffff")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
    
    net.add_node("author", label=d['name'], color="#FF6B6B", size=40, shape="ellipse", font={'size': 16, 'color': '#ffffff'})
    
    for i, p in enumerate(d['papers'][:10]):
        paper_label = p['title'][:25] + "..."
        net.add_node(f"paper_{i}", label=paper_label, color="#4ECDC4", size=20, 
                    shape="box", title=p['title'], font={'color': '#ffffff'})
        net.add_edge("author", f"paper_{i}", color="#666666")
    
    for i, collab in enumerate(d['collaborators'][:10]):
        net.add_node(f"collab_{i}", label=collab, color="#45B7D1", size=15, shape="ellipse", font={'color': '#ffffff'})
        net.add_edge("author", f"collab_{i}", color="#444444", dashes=True)
    
    return net, d

def build_paper_graph(arxiv_id: str):
    details = get_paper_details(arxiv_id)
    if not details:
        return None, None
    
    d = details[0]
    
    net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="#ffffff")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
    
    paper_label = d['title'][:35] + "..."
    net.add_node("paper", label=paper_label, color="#FF6B6B", size=35, 
                shape="box", title=d['title'], font={'size': 14, 'color': '#ffffff'})
    
    for i, author in enumerate(d['authors']):
        net.add_node(f"author_{i}", label=author, color="#45B7D1", size=20, shape="ellipse", font={'color': '#ffffff'})
        net.add_edge("paper", f"author_{i}", color="#666666")
    
    for i, rel in enumerate(d['related']):
        if rel and rel.get('title'):
            rel_label = rel['title'][:20] + "..."
            net.add_node(f"rel_{i}", label=rel_label, color="#4ECDC4", size=15, 
                        shape="box", title=rel['title'], font={'color': '#ffffff'})
            net.add_edge(f"author_0", f"rel_{i}", color="#444444", dashes=True)
    
    return net, d

def show_network(net):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as hf:
            components.html(hf.read(), height=520)

# ============== STREAMLIT UI ==============

st.set_page_config(page_title="Research Explorer", page_icon="🔭", layout="wide")

# Custom CSS - FIXED COLORS
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    
    .digest-box {
    background: #2d3748;
    padding: 1.5rem;
    border-radius: 12px;
    line-height: 1.9;
    font-size: 1.05rem;
    color: #f7fafc !important;
    border: 1px solid #4a5568;
            }

.digest-box strong, .digest-box b {
    color: #4ECDC4 !important;
}
    
    .stat-box {
        background: #2d3748;
        border: 1px solid #4a5568;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    
    .stat-label {
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    
    .paper-item {
        background: #1e2a3a;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #4ECDC4;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .paper-item:hover {
        background: #2a3a4a;
        cursor: pointer;
    }
    
    .author-item {
        background: #1e2a3a;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #45B7D1;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .author-item:hover {
        background: #2a3a4a;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ============== SESSION STATE ==============
if 'selected_idea' not in st.session_state:
    st.session_state.selected_idea = None
if 'selected_author' not in st.session_state:
    st.session_state.selected_author = None
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None
if 'show_idea_detail' not in st.session_state:
    st.session_state.show_idea_detail = False

# ============== SIDEBAR ==============
with st.sidebar:
    st.title("📊 Graph Stats")
    
    stats = get_graph_stats()
    
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{stats['papers']}</div>
        <div class="stat-label">Papers</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{stats['authors']}</div>
        <div class="stat-label">Authors</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{stats['connections']:,}</div>
        <div class="stat-label">Connections</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Jump - FIXED
    st.markdown("### 🎯 Quick Jump")
    themes_for_jump = ["", "Large Language Models", "Transformers", "Reinforcement Learning", 
                       "Computer Vision", "Diffusion Models", "Neural Networks"]
    quick_theme = st.selectbox("Jump to theme:", themes_for_jump, key="sidebar_theme_jump")
    
    if quick_theme and quick_theme != "":
        if st.button("Go →", key="jump_btn"):
            st.session_state.selected_idea = quick_theme
            st.session_state.show_idea_detail = True
            st.rerun()

# ============== MAIN CONTENT ==============

st.title("🔭 Research Explorer")
st.caption("Discover ideas, not just papers")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💡 Ideas & Digest", 
    "🕸️ Graph View", 
    "📚 Papers", 
    "👥 Authors",
    "🔗 Explore"
])

# ============== TAB 1: IDEAS & DIGEST ==============
with tab1:
    
    # Check if we should show idea detail
    if st.session_state.show_idea_detail and st.session_state.selected_idea:
        idea = st.session_state.selected_idea
        
        st.subheader(f"💡 Exploring: {idea}")
        
        if st.button("← Back to Digest", key="back_to_digest"):
            st.session_state.show_idea_detail = False
            st.session_state.selected_idea = None
            st.rerun()
        
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🕸️ Connection Graph")
            with st.spinner("Building graph..."):
                net, papers = build_idea_graph(idea)
            show_network(net)
            st.caption("🔴 Idea | 🟢 Papers | 🔵 Authors")
        
        with col2:
            st.markdown("### 📄 Related Papers")
            if papers:
                for idx, p in enumerate(papers[:8]):
                    with st.container():
                        st.markdown(f"**{p['title'][:60]}...**")
                        st.caption(f"👥 {', '.join(p['authors'][:2])} | Match: {p['score']:.0%}")
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            if st.button("View Details", key=f"idea_paper_btn_{idx}"):
                                st.session_state.selected_paper = p['id']
                        with col_b:
                            st.markdown(f"[arXiv](https://arxiv.org/abs/{p['id']})")
                        st.divider()
    
    else:
        # Theme Clouds
        st.subheader("🏷️ Research Themes")
        st.caption("Click a theme to explore papers")
        
        with st.spinner("Analyzing themes..."):
            themes = generate_themes()
        
        cols = st.columns(4)
        for idx, theme in enumerate(themes[:8]):
            with cols[idx % 4]:
                if st.button(f"{theme['name']} ({theme['count']})", key=f"theme_btn_{idx}", use_container_width=True):
                    st.session_state.selected_idea = theme['name']
                    st.session_state.show_idea_detail = True
                    st.rerun()
        
        st.divider()
        
        # Digest
        st.subheader("📰 Research Digest")
        st.caption("A narrative summary — click the idea buttons below to explore")
        
        with st.spinner("Generating digest..."):
            digest = generate_digest()
        
        # Extract ideas from digest
        ideas_in_digest = re.findall(r'\[\[(.*?)\]\]', digest)
        
        # Clean up digest for display - replace [[idea]] with bold colored text
        digest_display = digest
        for idea in ideas_in_digest:
            digest_display = digest_display.replace(f"[[{idea}]]", f'<span style="color: #4ECDC4; font-weight: 600;">{idea}</span>')
        
        # Display digest in styled box
        st.markdown(f'<div class="digest-box">{digest_display}</div>', unsafe_allow_html=True)
        
        # Clickable idea buttons - FIXED
        if ideas_in_digest:
            st.markdown("")
            st.markdown("**🔗 Click to explore these ideas:**")
            
            # Create unique buttons for each idea
            cols = st.columns(4)
            unique_ideas = list(dict.fromkeys(ideas_in_digest))  # Remove duplicates while preserving order
            
            for idx, idea in enumerate(unique_ideas[:12]):
                with cols[idx % 4]:
                    if st.button(f"→ {idea[:25]}", key=f"digest_idea_btn_{idx}"):
                        st.session_state.selected_idea = idea
                        st.session_state.show_idea_detail = True
                        st.rerun()

# ============== TAB 2: GRAPH VIEW ==============
with tab2:
    st.subheader("🕸️ Knowledge Graph")
    
    graph_type = st.radio(
        "View:",
        ["Overview", "By Author", "By Paper"],
        horizontal=True,
        key="graph_view_type"
    )
    
    if graph_type == "Overview":
        st.caption("Overall structure of the research landscape")
        
        if st.button("Generate Overview Graph", key="gen_overview"):
            with st.spinner("Building graph..."):
                with driver.session() as session:
                    result = session.run("""
                        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
                        WITH a, count(p) as papers
                        WHERE papers >= 3
                        MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
                        RETURN a.name as author, p.title as paper, p.arxiv_id as id
                        LIMIT 100
                    """)
                    data = [dict(r) for r in result]
                
                if data:
                    net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="#ffffff")
                    net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=100)
                    
                    added = set()
                    for row in data:
                        author = row['author']
                        paper_id = row['id']
                        
                        if author not in added:
                            net.add_node(author, label=author, color="#45B7D1", size=20, shape="ellipse", font={'color': '#ffffff'})
                            added.add(author)
                        
                        if paper_id not in added:
                            net.add_node(paper_id, label=row['paper'][:20]+"...", color="#4ECDC4", 
                                        size=10, shape="box", title=row['paper'], font={'color': '#ffffff'})
                            added.add(paper_id)
                            net.add_edge(author, paper_id, color="#444444")
                    
                    show_network(net)
                    st.caption("🔵 Authors | 🟢 Papers — Showing authors with 3+ papers")
    
    elif graph_type == "By Author":
        authors = get_all_authors()
        selected = st.selectbox("Select author:", [a['name'] for a in authors[:100]], key="graph_author_select")
        
        if st.button("Show Network", type="primary", key="show_author_network"):
            with st.spinner("Building graph..."):
                net, details = build_author_graph(selected)
            if net:
                show_network(net)
                st.caption("🔴 Selected author | 🟢 Their papers | 🔵 Collaborators")
                
                if details:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Papers", len(details['papers']))
                    with col2:
                        st.metric("Collaborators", len(details['collaborators']))
    
    elif graph_type == "By Paper":
        papers = get_all_papers()
        paper_options = [(p['title'][:60], p['id']) for p in papers[:100]]
        selected_idx = st.selectbox("Select paper:", range(len(paper_options)), 
                                   format_func=lambda x: paper_options[x][0], key="graph_paper_select")
        
        if st.button("Show Network", type="primary", key="show_paper_network"):
            selected_id = paper_options[selected_idx][1]
            with st.spinner("Building graph..."):
                net, details = build_paper_graph(selected_id)
            if net:
                show_network(net)
                st.caption("🔴 Selected paper | 🔵 Authors | 🟢 Related papers")

# ============== TAB 3: PAPERS ==============
with tab3:
    st.subheader("📚 All Papers")
    
    papers = get_all_papers()
    
    search = st.text_input("🔍 Search papers:", placeholder="Filter by title...", key="papers_search")
    
    if search:
        papers = [p for p in papers if search.lower() in p['title'].lower()]
    
    st.caption(f"Showing {min(len(papers), 50)} of {len(papers)} papers")
    
    for idx, p in enumerate(papers[:50]):
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Clickable paper title
                if st.button(f"📄 {p['title'][:70]}...", key=f"paper_click_{idx}"):
                    st.session_state.selected_paper = p['id']
                st.caption(f"👥 {', '.join(p['authors'][:3]) if p['authors'] else 'Unknown'}")
            
            with col2:
                st.markdown(f"[arXiv ↗](https://arxiv.org/abs/{p['id']})")
        
        # Show detail if selected
        if st.session_state.selected_paper == p['id']:
            with st.expander("📊 Paper Details", expanded=True):
                with st.spinner("Analyzing..."):
                    insight = extract_paper_insight(p['title'])
                
                st.markdown(f"**💡 Key Idea:** {insight['key_idea']}")
                st.markdown(f"**📝 Summary:** {insight['one_liner']}")
                st.markdown(f"**❓ Why care:** {insight['why_care']}")
                
                st.divider()
                
                # Related papers - CLICKABLE
                connections = explore_paper_connections(p['id'])
                if connections:
                    st.markdown("**🔗 Related Papers (click to explore):**")
                    for conn_idx, conn in enumerate(connections[:5]):
                        if st.button(f"→ {conn['related_paper'][:50]}... (via {conn['via_author']})", 
                                    key=f"related_paper_{idx}_{conn_idx}"):
                            st.session_state.selected_paper = conn['related_id']
                            st.rerun()
                
                if st.button("Close", key=f"close_paper_{idx}"):
                    st.session_state.selected_paper = None
                    st.rerun()
        
        st.divider()

# ============== TAB 4: AUTHORS ==============
with tab4:
    st.subheader("👥 Researchers")
    
    authors = get_all_authors()
    
    search = st.text_input("🔍 Search authors:", placeholder="Filter by name...", key="authors_search")
    
    if search:
        authors = [a for a in authors if search.lower() in a['name'].lower()]
    
    st.caption(f"Showing {min(len(authors), 50)} of {len(authors)} authors")
    
    for idx, a in enumerate(authors[:50]):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Clickable author name
                if st.button(f"👤 {a['name']}", key=f"author_click_{idx}"):
                    st.session_state.selected_author = a['name']
            
            with col2:
                st.caption(f"{a['paper_count']} papers")
        
        # Show detail if selected
        if st.session_state.selected_author == a['name']:
            with st.expander("📊 Author Details", expanded=True):
                details = get_author_details(a['name'])
                
                if details:
                    d = details[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📄 Papers (click to view):**")
                        for paper_idx, paper in enumerate(d['papers'][:5]):
                            if st.button(f"→ {paper['title'][:40]}...", key=f"auth_paper_{idx}_{paper_idx}"):
                                st.session_state.selected_paper = paper['id']
                                st.rerun()
                    
                    with col2:
                        st.markdown("**🤝 Collaborators (click to view):**")
                        for collab_idx, collab in enumerate(d['collaborators'][:5]):
                            if st.button(f"→ {collab}", key=f"auth_collab_{idx}_{collab_idx}"):
                                st.session_state.selected_author = collab
                                st.rerun()
                
                if st.button("Close", key=f"close_author_{idx}"):
                    st.session_state.selected_author = None
                    st.rerun()
        
        st.divider()

# ============== TAB 5: EXPLORE CONNECTIONS ==============
with tab5:
    st.subheader("🔗 Explore Connections")
    st.caption("Find hidden relationships between ideas, authors, and papers")
    
    explore_type = st.radio(
        "I want to explore:",
        [
            "💡 What papers relate to this idea?",
            "👤 What is this researcher's ecosystem?",
            "🔥 Who's pushing this idea forward?"
        ],
        key="explore_type_radio"
    )
    
    if explore_type == "💡 What papers relate to this idea?":
        idea = st.text_input("Enter an idea or concept:", 
                            placeholder="e.g., 'attention mechanisms' or 'model compression'",
                            key="explore_idea_input")
        
        if idea and st.button("Explore →", type="primary", key="explore_idea_btn"):
            with st.spinner("Finding connections..."):
                papers = get_papers_by_theme(idea)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🕸️ Connection Graph")
                net, _ = build_idea_graph(idea)
                show_network(net)
                st.caption("🔴 Idea | 🟢 Papers | 🔵 Authors")
            
            with col2:
                st.markdown("### 📄 Related Papers")
                for paper_idx, p in enumerate(papers[:8]):
                    if st.button(f"📄 {p['title'][:45]}... ({p['score']:.0%})", key=f"explore_paper_{paper_idx}"):
                        st.session_state.selected_paper = p['id']
                    st.caption(f"👥 {', '.join(p['authors'][:2])}")
                    st.divider()
    
    elif explore_type == "👤 What is this researcher's ecosystem?":
        authors = get_all_authors()
        selected = st.selectbox("Select researcher:", [a['name'] for a in authors[:100]], key="explore_author_select")
        
        if st.button("Explore →", type="primary", key="explore_author_btn"):
            with st.spinner("Mapping ecosystem..."):
                ecosystem = explore_author_ecosystem(selected)
                net, details = build_author_graph(selected)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🕸️ Network")
                if net:
                    show_network(net)
                    st.caption("🔴 Author | 🟢 Papers | 🔵 Collaborators")
            
            with col2:
                st.markdown("### 📄 Collaborators' Work")
                if ecosystem:
                    # Remove duplicates
                    seen = set()
                    for eco_idx, item in enumerate(ecosystem[:10]):
                        if item['paper'] not in seen:
                            seen.add(item['paper'])
                            if st.button(f"📄 {item['paper'][:40]}...", key=f"eco_paper_{eco_idx}"):
                                st.session_state.selected_paper = item['id']
                            st.caption(f"via {item['collaborator']}")
                            st.divider()
                else:
                    st.info("No extended network found")
    
    elif explore_type == "🔥 Who's pushing this idea forward?":
        idea = st.text_input("Enter an idea:", 
                            placeholder="e.g., 'diffusion models' or 'in-context learning'",
                            key="explore_champions_input")
        
        if idea and st.button("Find Champions →", type="primary", key="explore_champions_btn"):
            with st.spinner("Finding key researchers..."):
                papers = get_papers_by_theme(idea)
                
                # Extract top authors from these papers
                author_counts = {}
                for p in papers[:15]:
                    for author in p['authors'][:2]:
                        author_counts[author] = author_counts.get(author, 0) + 1
                
                top_authors = sorted(author_counts.items(), key=lambda x: -x[1])[:5]
            
            if top_authors:
                st.markdown(f"### 🏆 Key Researchers in '{idea}'")
                
                for auth_idx, (author, count) in enumerate(top_authors):
                    with st.expander(f"👤 {author} ({count} relevant papers)"):
                        details = get_author_details(author)
                        if details:
                            d = details[0]
                            
                            st.markdown("**Their papers:**")
                            for p_idx, paper in enumerate(d['papers'][:3]):
                                if st.button(f"→ {paper['title'][:50]}...", key=f"champ_paper_{auth_idx}_{p_idx}"):
                                    st.session_state.selected_paper = paper['id']
                            
                            if d['collaborators']:
                                st.markdown(f"**Collaborates with:** {', '.join(d['collaborators'][:5])}")
            else:
                st.info("No prominent researchers found for this idea")

# ============== FOOTER ==============
st.divider()
st.caption("🔭 Research Explorer | Powered by Graph RAG")