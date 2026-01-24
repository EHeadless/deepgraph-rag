"""
DeepGraph RAG - Honest Demonstration
"""
import os
import time
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile

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
def get_all_papers():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.arxiv_id as id, p.title as title, 
                   p.published_date as date, p.primary_category as category,
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
def get_graph_stats():
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN labels(n)[0] as l, count(*) as c")
        stats = {r["l"]: r["c"] for r in nodes}
        rels = session.run("MATCH ()-[r]->() RETURN type(r) as t, count(*) as c")
        for r in rels:
            stats[f"rel_{r['t']}"] = r["c"]
        return stats


# ============== SEARCH FUNCTIONS ==============

def vector_search(query: str, top_k: int = 5):
    """Standard vector search."""
    response = client.embeddings.create(model="text-embedding-ada-002", input=query)
    embedding = response.data[0].embedding
    
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('paper_embedding', $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:AUTHORED_BY]->(a:Author)
            RETURN node.title as title, node.arxiv_id as id, score,
                   collect(a.name) as authors
        """, embedding=embedding, top_k=top_k)
        return [dict(r) for r in result]


def find_author_papers(author_name: str):
    """Find all papers by an author."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            WHERE toLower(a.name) CONTAINS toLower($name)
            RETURN a.name as author, p.title as title, p.arxiv_id as id
            LIMIT 20
        """, name=author_name)
        return [dict(r) for r in result]


def find_collaborators(author_name: str):
    """Find who an author has worked with."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE toLower(a.name) CONTAINS toLower($name) AND a <> coauthor
            RETURN DISTINCT a.name as author, coauthor.name as collaborator,
                   count(p) as shared_papers
            ORDER BY shared_papers DESC
            LIMIT 20
        """, name=author_name)
        return [dict(r) for r in result]


def find_collaborator_papers(author_name: str):
    """Multi-hop: Author → Paper → Co-author → Their other papers."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p1:Paper)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE toLower(a.name) CONTAINS toLower($name) AND a <> coauthor
            MATCH (coauthor)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN DISTINCT a.name as author, coauthor.name as via,
                   p1.title as shared_paper, p2.title as recommended_paper, p2.arxiv_id as id
            LIMIT 20
        """, name=author_name)
        return [dict(r) for r in result]


def find_related_papers_by_author(paper_title: str):
    """Multi-hop: Paper → Authors → Their other papers."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)
            WHERE toLower(p1.title) CONTAINS toLower($title)
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN DISTINCT p1.title as source_paper, a.name as shared_author,
                   p2.title as related_paper
            LIMIT 20
        """, title=paper_title)
        return [dict(r) for r in result]


def find_connection_path(author1: str, author2: str):
    """Find how two authors are connected."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a1:Author), (a2:Author)
            WHERE toLower(a1.name) CONTAINS toLower($name1) 
              AND toLower(a2.name) CONTAINS toLower($name2)
            MATCH path = shortestPath((a1)-[:AUTHORED_BY|CO_AUTHORED*..8]-(a2))
            RETURN [n IN nodes(path) | 
                CASE WHEN 'Author' IN labels(n) THEN n.name 
                     ELSE n.title END] as path,
                   length(path) as hops
            LIMIT 1
        """, name1=author1, name2=author2)
        return [dict(r) for r in result]


def get_author_network(author_name: str):
    """Get author's full network."""
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


def generate_answer(question: str, context: str):
    """Generate answer from context."""
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Answer based on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# ============== VISUALIZATION ==============

def get_author_subgraph(author_name: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            WHERE toLower(a.name) CONTAINS toLower($name)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            RETURN a.name as author, p.arxiv_id as paper_id, 
                   p.title as paper_title, coauthor.name as coauthor
            LIMIT 100
        """, name=author_name)
        return [dict(r) for r in result]

def get_general_subgraph(limit: int = 50):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            RETURN p.arxiv_id as paper_id, p.title as paper_title, a.name as author_name
            LIMIT $limit
        """, limit=limit)
        return [dict(r) for r in result]

def create_network(data, graph_type="author"):
    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=200)
    
    added = set()
    
    if graph_type == "author":
        for row in data:
            author = row['author']
            paper_id = row['paper_id']
            coauthor = row.get('coauthor')
            
            if author not in added:
                net.add_node(author, label=author, color="#FF5722", shape="ellipse", size=25)
                added.add(author)
            
            if paper_id not in added:
                net.add_node(paper_id, label=row['paper_title'][:30]+"...", 
                            color="#4CAF50", shape="box", title=row['paper_title'])
                added.add(paper_id)
                net.add_edge(author, paper_id, color="#666")
            
            if coauthor and coauthor != author and coauthor not in added:
                net.add_node(coauthor, label=coauthor, color="#2196F3", shape="ellipse")
                added.add(coauthor)
            
            if coauthor and coauthor != author:
                net.add_edge(paper_id, coauthor, color="#666")
    
    elif graph_type == "general":
        for row in data:
            paper_id = row['paper_id']
            paper_title = row['paper_title'][:30] + "..."
            author = row['author_name']
            
            if paper_id not in added:
                net.add_node(paper_id, label=paper_title, color="#4CAF50", 
                           title=row['paper_title'], shape="box")
                added.add(paper_id)
            
            if author and author not in added:
                net.add_node(author, label=author, color="#2196F3", shape="ellipse")
                added.add(author)
            
            if author:
                net.add_edge(paper_id, author, color="#666")
    
    return net

def show_network(net):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as hf:
            components.html(hf.read(), height=520)


# ============== STREAMLIT UI ==============

st.set_page_config(page_title="DeepGraph RAG", page_icon="🧠", layout="wide")

st.title("🧠 DeepGraph RAG")
st.markdown("**Graph-Powered Research Assistant**")

with st.sidebar:
    st.header("📊 Graph Stats")
    stats = get_graph_stats()
    st.metric("Papers", stats.get("Paper", 0))
    st.metric("Authors", stats.get("Author", 0))
    st.metric("Co-authorships", stats.get("rel_CO_AUTHORED", 0))
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. **Vector Search**: Find similar papers
    2. **Graph Queries**: Traverse relationships
    3. **Multi-Hop**: Discover hidden connections
    4. **RAG**: Generate answers
    """)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Ask Questions", 
    "📊 Why Graph RAG?",
    "🔗 Multi-Hop",
    "🕸️ Graph View",
    "👤 Author Network",
    "📚 Papers", 
    "👥 Authors"
])

# ============== TAB 1: ASK QUESTIONS ==============
with tab1:
    st.header("Ask a Research Question")
    
    query = st.text_input("Question:", placeholder="What are recent advances in transformers?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.slider("Results", 3, 10, 5)
    
    if st.button("🔍 Search & Answer", type="primary") and query:
        papers = vector_search(query, top_k)
        
        if papers:
            st.subheader(f"📚 Found {len(papers)} papers")
            
            context_parts = []
            for p in papers:
                with st.expander(f"**{p['title'][:70]}...** ({p['score']:.3f})"):
                    st.write(f"Authors: {', '.join(p['authors'][:5])}")
                    st.write(f"ID: {p['id']}")
                context_parts.append(f"- {p['title']} by {', '.join(p['authors'][:3])}")
            
            st.divider()
            
            with st.spinner("Generating answer..."):
                context = "\n".join(context_parts)
                answer = generate_answer(query, context)
            
            st.subheader("💡 Answer")
            st.write(answer)
        else:
            st.warning("No papers found")

# ============== TAB 2: WHY GRAPH RAG ==============
with tab2:
    st.header("📊 What Can Graphs Do That Vector Search Can't?")
    
    st.markdown("""
    Vector search finds documents by **text similarity**.  
    Graphs answer questions about **relationships**.
    
    This isn't "better vs worse" - they solve **different problems**.
    """)
    
    st.divider()
    
    # Query Type 1
    st.subheader("✅ Type 1: Text Similarity")
    st.markdown("*Both can do this*")
    st.code('"Find papers about transformer architectures"', language=None)
    st.info("Vector search handles this well. Graph adds nothing special here.")
    
    st.divider()
    
    # Query Type 2
    st.subheader("🔗 Type 2: Relationship Questions")
    st.markdown("*Only graphs can answer these*")
    st.code('"What papers has Author X written?"', language=None)
    st.code('"Who has Author X collaborated with?"', language=None)
    
    authors = get_all_authors()
    author_names = [a['name'] for a in authors[:100]]
    
    selected = st.selectbox("Try it - select an author:", author_names, key="rel_author")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Their Papers", type="primary"):
            results = find_author_papers(selected)
            if results:
                st.success(f"Found {len(results)} papers")
                for r in results[:10]:
                    st.write(f"• {r['title'][:60]}...")
            else:
                st.warning("No papers found")
    
    with col2:
        if st.button("🤝 Their Collaborators", type="primary"):
            results = find_collaborators(selected)
            if results:
                st.success(f"Found {len(results)} collaborators")
                for r in results[:10]:
                    st.write(f"• {r['collaborator']} ({r['shared_papers']} shared)")
            else:
                st.warning("No collaborators found")
    
    st.divider()
    
    # Query Type 3
    st.subheader("🔗🔗 Type 3: Multi-Hop Questions")
    st.markdown("*Requires traversing connections*")
    st.code('"Find papers by Author X\'s collaborators"', language=None)
    st.markdown("**Path:** Author → Papers → Co-authors → Their other papers")
    st.info("See the **Multi-Hop** tab to try this!")
    
    st.divider()
    
    # Query Type 4
    st.subheader("🔗🔗🔗 Type 4: Path Finding")
    st.markdown("*How are two researchers connected?*")
    
    col1, col2 = st.columns(2)
    with col1:
        author1 = st.selectbox("Author 1:", author_names, key="path1")
    with col2:
        author2 = st.selectbox("Author 2:", author_names, index=min(1, len(author_names)-1), key="path2")
    
    if st.button("🔍 Find Connection", type="primary"):
        with st.spinner("Finding path..."):
            results = find_connection_path(author1, author2)
        
        if results:
            r = results[0]
            st.success(f"Connected in {r['hops']} hops!")
            st.markdown("**Path:**")
            path_display = " → ".join([str(x)[:40] + "..." if len(str(x)) > 40 else str(x) for x in r['path']])
            st.write(path_display)
        else:
            st.warning("No connection found within 6 hops")
    
    st.divider()
    
    # Summary Table
    st.subheader("📋 Summary")
    st.markdown("""
    | Query Type | Vector Search | Graph |
    |------------|---------------|-------|
    | "Papers about X topic" | ✅ Yes | ✅ Yes |
    | "Papers by Author X" | ❌ No* | ✅ Yes |
    | "Who collaborates with X?" | ❌ No | ✅ Yes |
    | "Papers by X's collaborators" | ❌ No | ✅ Yes |
    | "How are A and B connected?" | ❌ No | ✅ Yes |
    
    *Vector search could match if author name is in text, but can't reliably find "all papers by X"*
    """)

# ============== TAB 3: MULTI-HOP ==============
with tab3:
    st.header("🔗 Multi-Hop Queries")
    
    st.markdown("""
    Multi-hop queries traverse multiple relationships to find results that 
    **text search cannot discover** because there's no direct textual connection.
    """)
    
    st.divider()
    
    query_type = st.radio(
        "Query type:",
        [
            "Author → Collaborators → Their Papers (3 hops)",
            "Paper → Authors → Their Other Papers (2 hops)",
            "Author → Co-authors → Their Co-authors (3 hops)",
            "Author → Author Connection Path (N hops)",
            "Most Connected Authors (Network Hubs)"
        ]
    )
    
    authors = get_all_authors()
    author_names = [a['name'] for a in authors]
    
    if query_type == "Author → Collaborators → Their Papers (3 hops)":
        st.markdown("""
        **Path:** Author → Papers → Co-authors → Co-authors' other papers
        
        *Find papers by people who have worked with this author*
        """)
        
        selected = st.selectbox("Select author:", author_names, key="mh_author1")
        
        if st.button("🔗 Find Collaborators' Papers", type="primary"):
            with st.spinner("Traversing 3 hops..."):
                results = find_collaborator_papers(selected)
            
            if results:
                st.success(f"Found {len(results)} papers via collaborators!")
                for r in results:
                    with st.expander(f"**{r['recommended_paper'][:60]}...**"):
                        st.write(f"**Via:** {r['via']}")
                        st.write(f"**Shared paper:** {r['shared_paper'][:50]}...")
            else:
                st.warning("No results found")
    
    elif query_type == "Paper → Authors → Their Other Papers (2 hops)":
        st.markdown("""
        **Path:** Paper → Authors → Their other papers
        
        *Find other work by the authors of this paper*
        """)
        
        papers = get_all_papers()
        paper_titles = [p['title'][:80] for p in papers[:100]]
        selected = st.selectbox("Select paper:", paper_titles, key="mh_paper")
        
        if st.button("🔗 Find Authors' Other Papers", type="primary"):
            with st.spinner("Traversing 2 hops..."):
                results = find_related_papers_by_author(selected[:40])
            
            if results:
                st.success(f"Found {len(results)} related papers!")
                for r in results:
                    with st.expander(f"**{r['related_paper'][:60]}...**"):
                        st.write(f"**Via author:** {r['shared_author']}")
            else:
                st.warning("No results found")
    
    elif query_type == "Author → Co-authors → Their Co-authors (3 hops)":
        st.markdown("""
        **Path:** Author → Co-authors → Co-authors' co-authors
        
        *Find the extended research network - people 2 degrees away*
        """)
        
        selected = st.selectbox("Select author:", author_names, key="mh_author2")
        
        if st.button("🔗 Find Extended Network", type="primary"):
            with st.spinner("Traversing 3 hops..."):
                with driver.session() as session:
                    result = session.run("""
                        MATCH (a:Author)
                        WHERE toLower(a.name) CONTAINS toLower($name)
                        
                        // 1st hop: their co-authors
                        MATCH (a)<-[:AUTHORED_BY]-(p1:Paper)-[:AUTHORED_BY]->(coauthor1:Author)
                        WHERE coauthor1 <> a
                        
                        // 2nd hop: co-authors' co-authors
                        MATCH (coauthor1)<-[:AUTHORED_BY]-(p2:Paper)-[:AUTHORED_BY]->(coauthor2:Author)
                        WHERE coauthor2 <> a AND coauthor2 <> coauthor1
                        
                        RETURN DISTINCT 
                            a.name as author,
                            coauthor1.name as first_degree,
                            coauthor2.name as second_degree,
                            count(DISTINCT p2) as shared_papers
                        ORDER BY shared_papers DESC
                        LIMIT 30
                    """, name=selected)
                    results = [dict(r) for r in result]
            
            if results:
                st.success(f"Found {len(results)} people in extended network!")
                
                # Group by first degree
                st.markdown(f"**{selected}'s extended network:**")
                
                first_degrees = {}
                for r in results:
                    fd = r['first_degree']
                    if fd not in first_degrees:
                        first_degrees[fd] = []
                    first_degrees[fd].append(r['second_degree'])
                
                for fd, sds in list(first_degrees.items())[:10]:
                    with st.expander(f"**Via {fd}** → {len(sds)} connections"):
                        for sd in sds[:10]:
                            st.write(f"• {sd}")
                        if len(sds) > 10:
                            st.write(f"*...and {len(sds) - 10} more*")
            else:
                st.warning("No extended network found")
    
    elif query_type == "Author → Author Connection Path (N hops)":
        st.markdown("""
        **Path:** Author A → ... → Author B
        
        *How are two researchers connected? Find the shortest path.*
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            author1 = st.selectbox("From author:", author_names, key="path_from")
        with col2:
            author2 = st.selectbox("To author:", author_names, index=min(5, len(author_names)-1), key="path_to")
        
        if st.button("🔗 Find Connection Path", type="primary"):
            if author1 == author2:
                st.warning("Select two different authors")
            else:
                with st.spinner("Finding shortest path..."):
                    results = find_connection_path(author1, author2)
                
                if results:
                    r = results[0]
                    st.success(f"Connected in {r['hops']} hops!")
                    
                    st.markdown("**Connection path:**")
                    
                    # Display path nicely
                    path_items = r['path']
                    for i, item in enumerate(path_items):
                        if i % 2 == 0:  # Author
                            st.markdown(f"👤 **{item}**")
                        else:  # Paper
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↓ 📄 *{item[:50]}...*")
                else:
                    st.warning("No connection found within 6 hops")
    
    elif query_type == "Most Connected Authors (Network Hubs)":
        st.markdown("""
        **Query:** Find authors who connect the most different researchers
        
        *These are the "hubs" of the research network*
        """)
        
        if st.button("🔗 Find Network Hubs", type="primary"):
            with st.spinner("Analyzing network..."):
                with driver.session() as session:
                    result = session.run("""
                        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(coauthor:Author)
                        WHERE a <> coauthor
                        WITH a, count(DISTINCT coauthor) as collaborator_count, 
                             count(DISTINCT p) as paper_count
                        RETURN a.name as author, collaborator_count, paper_count
                        ORDER BY collaborator_count DESC
                        LIMIT 20
                    """)
                    results = [dict(r) for r in result]
            
            if results:
                st.success(f"Top {len(results)} most connected authors:")
                
                st.markdown("| Rank | Author | Collaborators | Papers |")
                st.markdown("|------|--------|---------------|--------|")
                
                for i, r in enumerate(results, 1):
                    st.markdown(f"| {i} | {r['author']} | {r['collaborator_count']} | {r['paper_count']} |")
                
                st.divider()
                st.info("These authors bridge multiple research communities and could be good starting points for exploring the graph.")
            else:
                st.warning("No results found")

# ============== TAB 4: GRAPH VIEW ==============
with tab4:
    st.header("🕸️ Graph Visualization")
    
    view_type = st.radio(
        "View:",
        ["Overview (Papers & Authors)", "Author-Centered"],
        horizontal=True
    )
    
    if view_type == "Overview (Papers & Authors)":
        st.markdown("*Sample of papers (green) and authors (blue)*")
        
        limit = st.slider("Connections to show:", 20, 100, 50)
        
        if st.button("🔄 Generate Graph", type="primary"):
            with st.spinner("Building..."):
                data = get_general_subgraph(limit)
                if data:
                    net = create_network(data, "general")
                    show_network(net)
                    st.markdown("🟢 Papers | 🔵 Authors")
                else:
                    st.warning("No data")
    
    else:
        st.markdown("*Author's papers and co-authors*")
        
        authors = get_all_authors()
        selected = st.selectbox("Select author:", [a['name'] for a in authors[:100]], key="viz_author")
        
        if st.button("🔄 Show Network", type="primary"):
            with st.spinner("Building..."):
                data = get_author_subgraph(selected)
                if data:
                    net = create_network(data, "author")
                    show_network(net)
                    st.markdown("🟠 Selected author | 🟢 Papers | 🔵 Co-authors")
                else:
                    st.warning("No data")

# ============== TAB 5: AUTHOR NETWORK ==============
with tab5:
    st.header("👤 Author Network Details")
    
    authors = get_all_authors()
    selected = st.selectbox("Select author:", [a['name'] for a in authors], key="net_author")
    
    if st.button("🕸️ Show Full Network", type="primary"):
        with st.spinner("Loading..."):
            results = get_author_network(selected)
        
        if results:
            r = results[0]
            st.subheader(f"👤 {r['author']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**📚 Papers ({len(r['papers'])}):**")
                for p in r['papers'][:15]:
                    st.write(f"• {p[:55]}...")
                if len(r['papers']) > 15:
                    st.write(f"*...and {len(r['papers']) - 15} more*")
            
            with col2:
                st.markdown(f"**🤝 Collaborators ({len(r['collaborators'])}):**")
                for c in r['collaborators'][:15]:
                    st.write(f"• {c}")
                if len(r['collaborators']) > 15:
                    st.write(f"*...and {len(r['collaborators']) - 15} more*")
        else:
            st.warning("Author not found")

# ============== TAB 6: PAPERS ==============
with tab6:
    st.header("📚 All Papers")
    
    papers = get_all_papers()
    search = st.text_input("Filter by title:", key="paper_filter")
    
    if search:
        papers = [p for p in papers if search.lower() in p['title'].lower()]
    
    st.write(f"**{len(papers)} papers**")
    
    for p in papers[:50]:
        with st.expander(p['title'][:80]):
            st.write(f"**ID:** {p['id']}")
            st.write(f"**Date:** {p['date'][:10] if p.get('date') else 'N/A'}")
            st.write(f"**Category:** {p.get('category', 'N/A')}")
            st.write(f"**Authors:** {', '.join(p['authors'])}")

# ============== TAB 7: AUTHORS ==============
with tab7:
    st.header("👥 All Authors")
    
    authors = get_all_authors()
    search = st.text_input("Filter by name:", key="author_filter")
    
    if search:
        authors = [a for a in authors if search.lower() in a['name'].lower()]
    
    st.write(f"**{len(authors)} authors**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Name**")
    with col2:
        st.markdown("**Papers**")
    
    st.divider()
    
    for a in authors[:100]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(a['name'])
        with col2:
            st.write(a['paper_count'])

# Footer
st.divider()
st.caption("Built with Neo4j • OpenAI • Streamlit | Graph RAG Demo")