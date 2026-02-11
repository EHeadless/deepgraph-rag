"""
Research Discovery Tool - Find papers, explore connections, get insights

MIGRATED VERSION using deepgraph framework.

This app has been refactored to use the new deepgraph-rag framework
while maintaining identical functionality to the original version.

Changes:
- Uses ArxivRAGAdapter instead of direct Neo4j/OpenAI calls
- All graph queries now go through the GraphStore abstraction
- LLM calls use the AnswerSynthesizer component
- Backend and LLM providers can be swapped via config.yaml

Original: /app_user.py
Migration date: 2026-02-08
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

# Import the adapter
from examples.arxiv.arxiv_adapter import get_adapter

load_dotenv()

@st.cache_resource
def init_adapter():
    """Initialize the ArxivRAGAdapter (replaces init_connections)."""
    return get_adapter()

# Initialize adapter instead of driver and client
adapter = init_adapter()


# ============== SEARCH FUNCTIONS ==============
# Now using adapter methods instead of direct Neo4j/OpenAI calls

def search_papers(query: str, top_k: int = 10):
    """Search papers by semantic similarity."""
    return adapter.vector_search(query, top_k=top_k)


def get_paper_details(arxiv_id: str):
    """Get full paper details with connections."""
    return adapter.get_paper_details(arxiv_id)


def find_expert(topic: str):
    """Find top authors on a topic."""
    # Get papers on the topic via vector search
    papers = adapter.vector_search(topic, top_k=20)

    # Aggregate by author
    author_papers = {}
    author_scores = {}

    for p in papers:
        for author in p.get('authors', []):
            if author not in author_papers:
                author_papers[author] = 0
                author_scores[author] = []
            author_papers[author] += 1
            author_scores[author].append(p['score'])

    # Calculate average scores and total papers
    results = []
    for author, count in author_papers.items():
        avg_score = sum(author_scores[author]) / len(author_scores[author])

        # Get total paper count for this author
        author_details = adapter.find_author_papers(author)
        total_papers = len(author_details)

        results.append({
            "author": author,
            "relevance": count,
            "avg_score": avg_score,
            "total_papers": total_papers
        })

    # Sort by relevance, then by average score
    results.sort(key=lambda x: (-x['relevance'], -x['avg_score']))

    return results[:10]


def get_reading_list(start_paper_title: str):
    """Generate a reading list starting from a paper."""
    # Use the existing adapter method
    related = adapter.find_related_papers_by_author(start_paper_title)

    if related:
        # Convert to the expected format
        reading_list = []
        for item in related:
            # Get paper details
            paper_title = item.get('related_paper')
            author = item.get('shared_author')

            # Search for arxiv_id
            paper_search = adapter.store.query("""
                MATCH (p:Paper)
                WHERE p.title = $title
                RETURN p.arxiv_id as id, p.primary_category as category
                LIMIT 1
            """, {"title": paper_title})

            if paper_search:
                reading_list.append({
                    "title": paper_title,
                    "id": paper_search[0].get('id', 'unknown'),
                    "author": author,
                    "category": paper_search[0].get('category', 'unknown')
                })

        return [{
            "start_paper": start_paper_title,
            "reading_list": reading_list[:10]
        }]
    return []


def explore_research_area(author_name: str):
    """See what research areas an author's network covers."""
    # Get collaborator papers
    collab_papers = adapter.find_collaborator_papers(author_name)

    if collab_papers:
        # Extract unique research areas and collaborators
        research_areas = set()
        collaborators = set()

        for item in collab_papers:
            collaborators.add(item.get('via', ''))

            # Get category for the recommended paper
            paper_id = item.get('id')
            if paper_id:
                details = adapter.get_paper_details(paper_id)
                if details and details[0].get('category'):
                    research_areas.add(details[0]['category'])

        return [{
            "author": author_name,
            "research_areas": list(research_areas),
            "collaborators": list(collaborators)[:10],
            "papers_in_network": len(collab_papers)
        }]

    return []


def answer_question(question: str, context_papers: list):
    """Generate answer from papers."""
    context = "\n".join([f"- {p['title']} by {', '.join(p['authors'][:2])}"
                        for p in context_papers])
    return adapter.generate_answer(question, context)


def get_author_profile(author_name: str):
    """Get detailed author profile."""
    return adapter.get_author_profile(author_name)


def visualize_author_network(author_name: str):
    """Create network visualization for an author."""
    # Get author details
    profile = adapter.get_author_profile(author_name)

    if not profile:
        return []

    # Extract papers and collaborators
    papers = profile[0].get('papers', [])
    collaborators = profile[0].get('collaborators', [])

    # Convert to expected format
    results = []
    for paper in papers[:50]:
        for collab in paper.get('coauthors', []):
            results.append({
                "center": author_name,
                "paper": paper['title'],
                "collaborator": collab
            })

    return results


def create_author_network(data, center_name):
    """Build pyvis network."""
    net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    added = set()

    # Add center author
    net.add_node(center_name, label=center_name, color="#FF6B6B", size=30, shape="ellipse")
    added.add(center_name)

    for row in data:
        paper = row['paper'][:25] + "..."
        collaborator = row['collaborator']

        if paper not in added:
            net.add_node(paper, label=paper, color="#4ECDC4", size=15, shape="box", title=row['paper'])
            added.add(paper)
            net.add_edge(center_name, paper, color="#666")

        if collaborator not in added:
            net.add_node(collaborator, label=collaborator, color="#45B7D1", size=20, shape="ellipse")
            added.add(collaborator)

        net.add_edge(paper, collaborator, color="#666")

    return net


def show_network(net):
    """Display network in Streamlit."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as hf:
            components.html(hf.read(), height=420)


# ============== STREAMLIT UI ==============
# UI code remains identical to original

st.set_page_config(
    page_title="Research Discovery",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        margin-top: 0;
    }
    .paper-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 Research Discovery</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find papers, discover experts, explore connections</p>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search Papers",
    "👤 Find Experts",
    "📚 Reading Lists",
    "🗺️ Explore Networks"
])

# ============== TAB 1: SEARCH PAPERS ==============
with tab1:
    st.header("What are you researching?")

    query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g., 'attention mechanisms in transformers' or 'reinforcement learning for robotics'"
    )

    if query:
        with st.spinner("Searching..."):
            papers = search_papers(query, 10)

        if papers:
            st.subheader(f"Found {len(papers)} relevant papers")

            # Quick answer
            with st.expander("💡 Quick Summary", expanded=True):
                with st.spinner("Generating summary..."):
                    answer = answer_question(f"Summarize the main themes and findings in these papers about: {query}", papers[:5])
                st.write(answer)

            st.divider()

            # Paper list
            for p in papers:
                col1, col2 = st.columns([4, 1])

                with col1:
                    with st.expander(f"**{p['title'][:80]}...**"):
                        st.write(f"**Authors:** {', '.join(p['authors'][:5])}")
                        st.write(f"**Category:** {p.get('category', 'N/A')}")
                        st.write(f"**arXiv:** [{p['id']}](https://arxiv.org/abs/{p['id']})")

                        # Related papers by same authors
                        details = get_paper_details(p['id'])
                        if details and details[0].get('related_papers'):
                            st.write("**More from these authors:**")
                            for rp in details[0]['related_papers'][:3]:
                                if rp:
                                    st.write(f"  • {rp[:60]}...")

                with col2:
                    st.metric("Match", f"{p['score']:.0%}")

# ============== TAB 2: FIND EXPERTS ==============
with tab2:
    st.header("Find experts in a topic")

    topic = st.text_input(
        "What topic do you need an expert in?",
        placeholder="e.g., 'large language models' or 'computer vision for medical imaging'"
    )

    if topic:
        with st.spinner("Finding experts..."):
            experts = find_expert(topic)

        if experts:
            st.subheader(f"Top researchers in '{topic}'")

            for i, e in enumerate(experts, 1):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    if st.button(f"👤 {e['author']}", key=f"expert_{i}"):
                        st.session_state.selected_expert = e['author']

                with col2:
                    st.metric("Relevant Papers", e['relevance'])

                with col3:
                    st.metric("Total Papers", e['total_papers'])

            # Show expert details if selected
            if 'selected_expert' in st.session_state:
                st.divider()
                st.subheader(f"📋 Profile: {st.session_state.selected_expert}")

                profile = get_author_profile(st.session_state.selected_expert)
                if profile:
                    p = profile[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Recent Papers:**")
                        for paper in p['papers'][:5]:
                            st.write(f"• {paper['title'][:60]}...")
                            st.caption(f"  with {', '.join(paper['coauthors'][:3])}")

                    with col2:
                        st.write("**Research Network:**")
                        network_data = visualize_author_network(st.session_state.selected_expert)
                        if network_data:
                            net = create_author_network(network_data, p['name'])
                            show_network(net)

# ============== TAB 3: READING LISTS ==============
with tab3:
    st.header("Build a reading list")

    st.markdown("Start with a paper you like, and we'll find related work by the same authors and their collaborators.")

    # Search for starting paper
    start_query = st.text_input(
        "Search for a starting paper:",
        placeholder="e.g., 'attention is all you need' or 'BERT'"
    )

    if start_query:
        with st.spinner("Searching..."):
            papers = search_papers(start_query, 5)

        if papers:
            st.write("**Select a starting paper:**")

            for p in papers:
                if st.button(f"📄 {p['title'][:70]}...", key=f"start_{p['id']}"):
                    st.session_state.reading_start = p['title']

    if 'reading_start' in st.session_state:
        st.divider()
        st.subheader(f"📚 Reading list starting from:")
        st.info(st.session_state.reading_start[:80] + "...")

        with st.spinner("Building reading list..."):
            reading_list = get_reading_list(st.session_state.reading_start[:40])

        if reading_list and reading_list[0].get('reading_list'):
            items = reading_list[0]['reading_list']

            st.write(f"**{len(items)} related papers to read next:**")

            for i, item in enumerate(items, 1):
                with st.expander(f"{i}. {item['title'][:70]}..."):
                    st.write(f"**Connected via:** {item['author']}")
                    st.write(f"**Category:** {item['category']}")
                    st.write(f"**arXiv:** [{item['id']}](https://arxiv.org/abs/{item['id']})")
        else:
            st.warning("No related papers found. Try a different starting paper.")

# ============== TAB 4: EXPLORE NETWORKS ==============
with tab4:
    st.header("Explore research networks")

    st.markdown("See how researchers are connected and what areas their network covers.")

    author_search = st.text_input(
        "Search for a researcher:",
        placeholder="e.g., 'Yann LeCun' or 'Fei-Fei Li'"
    )

    if author_search:
        with st.spinner("Loading network..."):
            network_info = explore_research_area(author_search)

        if network_info:
            info = network_info[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Papers in Network", info['papers_in_network'])
            with col2:
                st.metric("Collaborators", len(info['collaborators']))
            with col3:
                st.metric("Research Areas", len(info['research_areas']))

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Research Areas Covered:**")
                for area in info['research_areas'][:10]:
                    st.write(f"• {area}")

                st.write("")
                st.write("**Key Collaborators:**")
                for collab in info['collaborators'][:10]:
                    st.write(f"• {collab}")

            with col2:
                st.write("**Collaboration Network:**")
                network_data = visualize_author_network(author_search)
                if network_data:
                    net = create_author_network(network_data, info['author'])
                    show_network(net)
        else:
            st.warning("Researcher not found. Try a different name.")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Powered by deepgraph-rag framework | Refactored to use modular components</p>
    <p>Backend: Neo4j | LLM: OpenAI GPT-4</p>
</div>
""", unsafe_allow_html=True)
