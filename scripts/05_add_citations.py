"""
DeepGraph RAG - Interactive Web Interface with Graph & Citation Visualization
"""
import os
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile

load_dotenv()

# Initialize connections
@st.cache_resource
def init_connections():
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "deepgraph2025")
    )
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return driver, client

driver, client = init_connections()


# ============== DATA LOADING ==============

@st.cache_data
def get_all_papers():
    """Get all papers from the graph."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.arxiv_id as id, 
                   p.title as title, 
                   p.published_date as date,
                   p.primary_category as category,
                   p.citation_count as citations,
                   collect(a.name) as authors
            ORDER BY p.citation_count DESC
        """)
        return [dict(r) for r in result]


@st.cache_data
def get_all_authors():
    """Get all authors with paper counts."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            RETURN a.name as name, count(p) as paper_count
            ORDER BY paper_count DESC
        """)
        return [dict(r) for r in result]


@st.cache_data
def get_graph_stats():
    """Get graph statistics."""
    with driver.session() as session:
        # Node counts
        node_result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
        """)
        stats = {r["label"]: r["count"] for r in node_result}
        
        # Relationship counts
        rel_result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
        """)
        for r in rel_result:
            stats[f"rel_{r['type']}"] = r["count"]
        
        return stats


# ============== CITATION FUNCTIONS ==============

def get_citation_network(paper_title: str):
    """Get papers that cite or are cited by a paper."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE toLower(p.title) CONTAINS toLower($title)
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
            OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
            RETURN p.title as paper,
                   p.arxiv_id as paper_id,
                   p.citation_count as citation_count,
                   collect(DISTINCT {title: cited.title, id: cited.arxiv_id}) as cites,
                   collect(DISTINCT {title: citing.title, id: citing.arxiv_id}) as cited_by
            LIMIT 1
        """, title=paper_title)
        return [dict(r) for r in result]


def multi_hop_citation_chain(paper_title: str, hops: int = 2):
    """
    Multi-hop: Paper → Cites → Paper → Cites → Paper
    Find papers 2+ hops away in citation chain.
    """
    with driver.session() as session:
        query = """
            MATCH (p:Paper)
            WHERE toLower(p.title) CONTAINS toLower($title)
            MATCH path = (p)-[:CITES*1..%d]->(cited:Paper)
            RETURN p.title as source,
                   cited.title as cited_paper,
                   cited.arxiv_id as cited_id,
                   length(path) as hops
            ORDER BY hops, cited.citation_count DESC
            LIMIT 30
        """ % hops
        result = session.run(query, title=paper_title)
        return [dict(r) for r in result]


def get_citation_path(from_paper: str, to_paper: str):
    """Find citation path between two papers."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p1:Paper), (p2:Paper)
            WHERE toLower(p1.title) CONTAINS toLower($from_title)
              AND toLower(p2.title) CONTAINS toLower($to_title)
            MATCH path = shortestPath((p1)-[:CITES*..5]->(p2))
            RETURN [n IN nodes(path) | n.title] as path_titles,
                   length(path) as path_length
        """, from_title=from_paper, to_title=to_paper)
        return [dict(r) for r in result]


def get_most_cited_papers(limit: int = 10):
    """Get most cited papers."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.citation_count > 0
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.title as title, 
                   p.arxiv_id as id,
                   p.citation_count as citations,
                   collect(a.name) as authors
            ORDER BY p.citation_count DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(r) for r in result]


def get_papers_citing_author(author_name: str):
    """Find papers that cite an author's work."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p1:Paper)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (p1)<-[:CITES]-(p2:Paper)
            MATCH (p2)-[:AUTHORED_BY]->(a2:Author)
            RETURN DISTINCT
                a.name as original_author,
                p1.title as original_paper,
                p2.title as citing_paper,
                collect(DISTINCT a2.name) as citing_authors
            LIMIT 20
        """, name=author_name)
        return [dict(r) for r in result]


# ============== GRAPH VISUALIZATION ==============

def get_subgraph_for_visualization(limit=50):
    """Get a subgraph for visualization."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            WITH p, a
            LIMIT $limit
            RETURN p.arxiv_id as paper_id, 
                   p.title as paper_title,
                   a.name as author_name
        """, limit=limit)
        return [dict(r) for r in result]


def get_citation_graph_data(paper_title: str):
    """Get citation graph centered on a paper."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE toLower(p.title) CONTAINS toLower($title)
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
            OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
            WITH p, collect(DISTINCT cited) as cited_papers, collect(DISTINCT citing) as citing_papers
            RETURN p.title as center_title,
                   p.arxiv_id as center_id,
                   [c IN cited_papers | {title: c.title, id: c.arxiv_id}] as cites,
                   [c IN citing_papers | {title: c.title, id: c.arxiv_id}] as cited_by
        """, title=paper_title)
        return [dict(r) for r in result]


def get_author_graph(author_name: str):
    """Get graph centered on an author."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            RETURN a.name as author,
                   p.arxiv_id as paper_id,
                   p.title as paper_title,
                   coauthor.name as coauthor
            LIMIT 100
        """, name=author_name)
        return [dict(r) for r in result]


def create_network_graph(data, graph_type="general"):
    """Create a pyvis network visualization."""
    net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=200)
    
    added_nodes = set()
    
    if graph_type == "general":
        for row in data:
            paper_id = row['paper_id']
            paper_title = row['paper_title'][:40] + "..."
            author = row['author_name']
            
            if paper_id not in added_nodes:
                net.add_node(paper_id, label=paper_title, color="#4CAF50", 
                           title=row['paper_title'], shape="box")
                added_nodes.add(paper_id)
            
            if author and author not in added_nodes:
                net.add_node(author, label=author, color="#2196F3", 
                           title=author, shape="ellipse")
                added_nodes.add(author)
            
            if author:
                net.add_edge(paper_id, author, color="#666666")
    
    elif graph_type == "author":
        for row in data:
            author = row['author']
            paper_id = row['paper_id']
            paper_title = row['paper_title'][:40] + "..."
            coauthor = row['coauthor']
            
            if author not in added_nodes:
                net.add_node(author, label=author, color="#FF5722", 
                           title=author, shape="ellipse", size=30)
                added_nodes.add(author)
            
            if paper_id not in added_nodes:
                net.add_node(paper_id, label=paper_title, color="#4CAF50", 
                           title=row['paper_title'], shape="box")
                added_nodes.add(paper_id)
                net.add_edge(author, paper_id, color="#666666")
            
            if coauthor and coauthor != author and coauthor not in added_nodes:
                net.add_node(coauthor, label=coauthor, color="#2196F3", 
                           title=coauthor, shape="ellipse")
                added_nodes.add(coauthor)
                net.add_edge(paper_id, coauthor, color="#666666")
    
    elif graph_type == "citation":
        data = data[0] if data else {}
        center_title = data.get('center_title', '')[:40] + "..."
        center_id = data.get('center_id', 'center')
        
        # Add center node
        net.add_node(center_id, label=center_title, color="#FF5722",
                    title=data.get('center_title', ''), shape="box", size=30)
        added_nodes.add(center_id)
        
        # Add cited papers (papers this one cites)
        for cited in data.get('cites', []):
            if cited.get('id') and cited['id'] not in added_nodes:
                label = cited['title'][:30] + "..." if cited.get('title') else cited['id']
                net.add_node(cited['id'], label=label, color="#4CAF50",
                           title=cited.get('title', ''), shape="box")
                added_nodes.add(cited['id'])
                net.add_edge(center_id, cited['id'], color="#00FF00", 
                           arrows="to", title="cites")
        
        # Add citing papers (papers that cite this one)
        for citing in data.get('cited_by', []):
            if citing.get('id') and citing['id'] not in added_nodes:
                label = citing['title'][:30] + "..." if citing.get('title') else citing['id']
                net.add_node(citing['id'], label=label, color="#2196F3",
                           title=citing.get('title', ''), shape="box")
                added_nodes.add(citing['id'])
                net.add_edge(citing['id'], center_id, color="#2196F3", 
                           arrows="to", title="cites")
    
    return net


def display_network(net):
    """Display pyvis network in Streamlit."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as html_file:
            html_content = html_file.read()
        components.html(html_content, height=620, scrolling=True)


# ============== SEARCH FUNCTIONS ==============

def vector_search(query: str, top_k: int = 5):
    """Semantic search using embeddings."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('paper_embedding', $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:AUTHORED_BY]->(a:Author)
            RETURN node.title as title, 
                   node.arxiv_id as id,
                   node.citation_count as citations,
                   score,
                   collect(a.name) as authors
        """, embedding=query_embedding, top_k=top_k)
        
        return [dict(r) for r in result]


def multi_hop_collaborator_papers(author_name: str):
    """Find papers by an author's collaborators."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p1:Paper)
            MATCH (p1)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            MATCH (coauthor)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN DISTINCT 
                a.name as original_author,
                p1.title as original_paper,
                coauthor.name as collaborator,
                p2.title as recommended_paper
            LIMIT 20
        """, name=author_name)
        return [dict(r) for r in result]


def multi_hop_related_papers(paper_title: str):
    """Find related papers through shared authors."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p1:Paper)
            WHERE toLower(p1.title) CONTAINS toLower($title)
            MATCH (p1)-[:AUTHORED_BY]->(a:Author)
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN 
                p1.title as source_paper,
                a.name as shared_author,
                p2.title as related_paper
            LIMIT 20
        """, title=paper_title)
        return [dict(r) for r in result]


def get_author_network(author_name: str):
    """Get author's collaboration network."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            RETURN 
                a.name as author,
                collect(DISTINCT p.title) as papers,
                collect(DISTINCT coauthor.name) as collaborators
        """, name=author_name)
        return [dict(r) for r in result]


def answer_with_rag(question: str, papers: list):
    """Generate answer using RAG."""
    context = "\n".join([f"- {p['title']}" for p in papers])
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Answer based on the research papers provided. Be concise and informative."},
            {"role": "user", "content": f"Research Papers:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content


# ============== STREAMLIT UI ==============

st.set_page_config(
    page_title="DeepGraph RAG",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DeepGraph RAG")
st.markdown("**Graph-Powered Research Assistant with Multi-Hop Reasoning**")

# Sidebar with stats
with st.sidebar:
    st.header("📊 Knowledge Graph Stats")
    stats = get_graph_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", stats.get("Paper", 0))
        st.metric("Authors", stats.get("Author", 0))
    with col2:
        st.metric("Citations", stats.get("rel_CITES", 0))
        st.metric("Co-authors", stats.get("rel_CO_AUTHORED", 0))
    
    st.divider()
    st.markdown("### 🔗 Relationships")
    st.markdown(f"""
    - AUTHORED_BY: {stats.get('rel_AUTHORED_BY', 0)}
    - CITES: {stats.get('rel_CITES', 0)}
    - CO_AUTHORED: {stats.get('rel_CO_AUTHORED', 0)}
    """)
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. **Vector Search**: Semantic similarity
    2. **Graph Traversal**: Follow relationships
    3. **Multi-Hop**: Discover connections
    4. **Citations**: Track influence
    """)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Ask Questions", 
    "📑 Citations",
    "🕸️ Graph View",
    "📚 All Papers", 
    "👥 All Authors",
    "🔗 Multi-Hop",
    "👤 Author Network"
])

# ============== TAB 1: ASK QUESTIONS ==============
with tab1:
    st.header("Ask a Research Question")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What are recent advances in neural networks?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.slider("Results", 3, 10, 5)
    
    if st.button("🔍 Search & Answer", type="primary"):
        if question:
            with st.spinner("Searching knowledge graph..."):
                papers = vector_search(question, top_k)
            
            if papers:
                st.subheader(f"📚 Found {len(papers)} Relevant Papers")
                
                for p in papers:
                    citations = p.get('citations') or 0
                    with st.expander(f"**{p['title'][:80]}...** (score: {p['score']:.3f}, cited: {citations})"):
                        st.write(f"**Authors:** {', '.join(p['authors'][:5])}")
                        st.write(f"**ID:** {p['id']}")
                        st.write(f"**Citations:** {citations}")
                
                st.divider()
                
                with st.spinner("Generating answer with GPT-4..."):
                    answer = answer_with_rag(question, papers)
                
                st.subheader("💡 Answer")
                st.write(answer)
            else:
                st.warning("No relevant papers found.")

# ============== TAB 2: CITATIONS ==============
with tab2:
    st.header("📑 Citation Analysis")
    
    citation_view = st.radio(
        "View:",
        ["Most Cited Papers", "Citation Network", "Citation Chain (Multi-Hop)", "Who Cites Author?"],
        horizontal=True
    )
    
    if citation_view == "Most Cited Papers":
        st.subheader("🏆 Most Cited Papers")
        
        most_cited = get_most_cited_papers(20)
        
        if most_cited:
            for i, p in enumerate(most_cited, 1):
                with st.expander(f"**#{i} [{p['citations']} citations]** {p['title'][:70]}..."):
                    st.write(f"**Authors:** {', '.join(p['authors'][:5])}")
                    st.write(f"**arXiv ID:** {p['id']}")
        else:
            st.info("No citation data yet. Run `python scripts/05_add_citations.py` first.")
    
    elif citation_view == "Citation Network":
        st.subheader("🕸️ Paper Citation Network")
        st.markdown("*See what a paper cites and what cites it*")
        
        all_papers = get_all_papers()
        paper_titles = [p['title'][:80] for p in all_papers[:100]]
        
        selected_paper = st.selectbox("Select a paper:", paper_titles, key="citation_paper")
        
        if st.button("📊 Show Citation Network", type="primary"):
            with st.spinner("Loading citations..."):
                results = get_citation_network(selected_paper[:40])
            
            if results and results[0]:
                r = results[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📤 This Paper Cites:")
                    cites = [c for c in r.get('cites', []) if c.get('title')]
                    if cites:
                        for c in cites:
                            st.write(f"• {c['title'][:60]}...")
                    else:
                        st.write("*No outgoing citations found*")
                
                with col2:
                    st.markdown("### 📥 Cited By:")
                    cited_by = [c for c in r.get('cited_by', []) if c.get('title')]
                    if cited_by:
                        for c in cited_by:
                            st.write(f"• {c['title'][:60]}...")
                    else:
                        st.write("*No incoming citations found*")
                
                # Visualize
                st.divider()
                st.markdown("### 🕸️ Citation Graph")
                st.markdown("🟠 Selected paper | 🟢 Papers it cites | 🔵 Papers citing it")
                
                graph_data = get_citation_graph_data(selected_paper[:40])
                if graph_data:
                    net = create_network_graph(graph_data, "citation")
                    display_network(net)
            else:
                st.warning("No citation data found for this paper.")
    
    elif citation_view == "Citation Chain (Multi-Hop)":
        st.subheader("🔗 Citation Chain (Multi-Hop)")
        st.markdown("**Path:** Paper → Cites → Paper → Cites → Paper")
        st.markdown("*Follow the citation chain to discover influential foundational papers*")
        
        all_papers = get_all_papers()
        paper_titles = [p['title'][:80] for p in all_papers[:100]]
        
        selected_paper = st.selectbox("Select starting paper:", paper_titles, key="chain_paper")
        hops = st.slider("Number of hops:", 1, 3, 2)
        
        if st.button("🔗 Trace Citation Chain", type="primary"):
            with st.spinner("Traversing citation graph..."):
                results = multi_hop_citation_chain(selected_paper[:40], hops)
            
            if results:
                st.success(f"Found {len(results)} papers in citation chain!")
                
                # Group by hops
                by_hops = {}
                for r in results:
                    h = r['hops']
                    if h not in by_hops:
                        by_hops[h] = []
                    by_hops[h].append(r)
                
                for hop_num in sorted(by_hops.keys()):
                    st.markdown(f"### Hop {hop_num}")
                    for r in by_hops[hop_num]:
                        st.write(f"• {r['cited_paper'][:70]}...")
            else:
                st.warning("No citation chain found.")
    
    else:  # Who Cites Author
        st.subheader("👤 Who Cites This Author?")
        st.markdown("*Find papers that cite an author's work*")
        
        all_authors = get_all_authors()
        author_names = [a['name'] for a in all_authors[:100]]
        
        selected_author = st.selectbox("Select an author:", author_names, key="cite_author")
        
        if st.button("🔍 Find Citing Papers", type="primary"):
            with st.spinner("Searching citations..."):
                results = get_papers_citing_author(selected_author)
            
            if results:
                st.success(f"Found {len(results)} citations to {selected_author}'s work!")
                
                for r in results:
                    with st.expander(f"**{r['citing_paper'][:60]}...**"):
                        st.write(f"**Cites:** {r['original_paper'][:60]}...")
                        st.write(f"**Authors:** {', '.join(r['citing_authors'][:3])}")
            else:
                st.warning("No citations found for this author's papers.")

# ============== TAB 3: GRAPH VIEW ==============
with tab3:
    st.header("🕸️ Knowledge Graph Visualization")
    
    view_type = st.radio(
        "View:",
        ["Overview (Papers & Authors)", "Author-Centered View"],
        horizontal=True
    )
    
    if view_type == "Overview (Papers & Authors)":
        st.markdown("*Showing papers (green boxes) and authors (blue circles)*")
        
        node_limit = st.slider("Connections to show:", 20, 100, 50)
        
        if st.button("🔄 Generate Graph", type="primary"):
            with st.spinner("Building visualization..."):
                data = get_subgraph_for_visualization(node_limit)
                if data:
                    net = create_network_graph(data, "general")
                    display_network(net)
                else:
                    st.warning("No data found.")
    
    else:
        st.markdown("*See an author's papers and collaborators*")
        
        all_authors = get_all_authors()
        author_names = [a['name'] for a in all_authors[:100]]
        
        selected_author = st.selectbox("Select an author:", author_names, key="graph_author")
        
        if st.button("🔄 Show Author Graph", type="primary"):
            with st.spinner("Building visualization..."):
                data = get_author_graph(selected_author)
                if data:
                    net = create_network_graph(data, "author")
                    display_network(net)
                    
                    st.markdown("""
                    **Legend:** 🟠 Selected author | 🟢 Papers | 🔵 Co-authors
                    """)
                else:
                    st.warning("No data found.")

# ============== TAB 4: ALL PAPERS ==============
with tab4:
    st.header("📚 All Papers")
    
    papers = get_all_papers()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Filter papers:", placeholder="Search by title...")
    with col2:
        sort_by = st.selectbox("Sort by:", ["Citations", "Date", "Title"])
    
    if search:
        papers = [p for p in papers if search.lower() in p['title'].lower()]
    
    if sort_by == "Date":
        papers = sorted(papers, key=lambda x: x.get('date') or '', reverse=True)
    elif sort_by == "Title":
        papers = sorted(papers, key=lambda x: x.get('title') or '')
    
    st.write(f"**Showing {len(papers)} papers**")
    
    for p in papers[:100]:
        citations = p.get('citations') or 0
        with st.expander(f"**[{citations} cites]** {p['title'][:80]}"):
            st.write(f"**arXiv ID:** {p['id']}")
            st.write(f"**Date:** {p['date'][:10] if p.get('date') else 'N/A'}")
            st.write(f"**Category:** {p.get('category', 'N/A')}")
            st.write(f"**Authors:** {', '.join(p['authors'])}")

# ============== TAB 5: ALL AUTHORS ==============
with tab5:
    st.header("👥 All Authors")
    
    authors = get_all_authors()
    
    search = st.text_input("Filter authors:", placeholder="Search by name...", key="author_search")
    
    if search:
        authors = [a for a in authors if search.lower() in a['name'].lower()]
    
    st.write(f"**Showing {len(authors)} authors**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Author Name**")
    with col2:
        st.write("**Papers**")
    
    st.divider()
    
    for a in authors[:100]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(a['name'])
        with col2:
            st.write(a['paper_count'])

# ============== TAB 6: MULTI-HOP ==============
with tab6:
    st.header("🔗 Multi-Hop Reasoning")
    
    st.markdown("**Multi-hop** queries traverse the graph to find non-obvious connections.")
    
    search_type = st.radio(
        "Search Type:",
        ["Collaborator Papers", "Related Papers (Shared Authors)"]
    )
    
    if search_type == "Collaborator Papers":
        st.markdown("**Path:** Author → Paper → Co-Author → Their Papers")
        
        all_authors = get_all_authors()
        author_names = [a['name'] for a in all_authors]
        
        selected_author = st.selectbox("Select author:", author_names, key="mh_author")
        
        if st.button("🔗 Find Collaborator Papers", type="primary"):
            with st.spinner("Traversing graph..."):
                results = multi_hop_collaborator_papers(selected_author)
            
            if results:
                st.success(f"Found {len(results)} papers!")
                
                for r in results:
                    with st.expander(f"Via **{r['collaborator']}**"):
                        st.write(f"**Original:** {r['original_paper'][:60]}...")
                        st.write(f"**Recommended:** {r['recommended_paper']}")
            else:
                st.warning("No connections found.")
    
    else:
        st.markdown("**Path:** Paper → Authors → Their Other Papers")
        
        all_papers = get_all_papers()
        paper_titles = [p['title'][:80] for p in all_papers]
        
        selected_paper = st.selectbox("Select paper:", paper_titles, key="mh_paper")
        
        if st.button("🔗 Find Related Papers", type="primary"):
            with st.spinner("Traversing graph..."):
                results = multi_hop_related_papers(selected_paper[:40])
            
            if results:
                st.success(f"Found {len(results)} related papers!")
                
                for r in results:
                    with st.expander(f"Via **{r['shared_author']}**"):
                        st.write(f"**Related:** {r['related_paper']}")
            else:
                st.warning("No related papers found.")

# ============== TAB 7: AUTHOR NETWORK ==============
with tab7:
    st.header("👤 Author Collaboration Network")
    
    all_authors = get_all_authors()
    author_names = [a['name'] for a in all_authors]
    
    selected_author = st.selectbox("Select author:", author_names, key="network_author")
    
    if st.button("🕸️ Show Network", type="primary"):
        with st.spinner("Building network..."):
            results = get_author_network(selected_author)
        
        if results:
            for r in results:
                st.subheader(f"👤 {r['author']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📚 Papers:**")
                    for paper in r['papers'][:10]:
                        st.write(f"• {paper[:60]}...")
                    if len(r['papers']) > 10:
                        st.write(f"*+{len(r['papers']) - 10} more*")
                
                with col2:
                    st.markdown("**🤝 Collaborators:**")
                    for collab in r['collaborators'][:10]:
                        st.write(f"• {collab}")
                    if len(r['collaborators']) > 10:
                        st.write(f"*+{len(r['collaborators']) - 10} more*")
        else:
            st.warning("Author not found.")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Neo4j • OpenAI • Streamlit • PyVis</p>
    <p>🧠 Graph RAG with Multi-Hop Reasoning & Citation Analysis</p>
</div>
""", unsafe_allow_html=True)