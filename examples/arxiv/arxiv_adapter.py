"""
ArXiv Graph RAG Adapter

This module provides a backward-compatible API for migrating Streamlit apps
to use the new deepgraph framework while maintaining the same function signatures.
"""
import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import deepgraph framework components
from deepgraph.core.schema import GraphSchema
from deepgraph.store.neo4j import Neo4jGraphStore
from deepgraph.adapters.embedders.openai import OpenAIEmbedder
from deepgraph.retrieval.vector import VectorRetriever
from deepgraph.reasoning.traversal import GraphReasoner
from deepgraph.synthesis.openai import OpenAISynthesizer
from deepgraph.planning.intent import IntentParser
from deepgraph.planning.executor import QueryExecutor
from deepgraph.pipeline.base import GraphRAGPipeline
from deepgraph.pipeline.stages import PipelineConfig


class ArxivRAGAdapter:
    """
    Adapter for ArXiv Graph RAG application.

    Wraps the new deepgraph framework to provide backward-compatible
    functions for legacy Streamlit apps.

    Example:
        adapter = ArxivRAGAdapter()
        results = adapter.vector_search("transformers in NLP", top_k=5)
        answer = adapter.generate_answer(query, context)
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize adapter with configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in examples/arxiv/
            config_path = Path(__file__).parent / "config.yaml"

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all framework components."""
        # Initialize graph store
        db_config = self.config['database']
        self.store = Neo4jGraphStore()
        self.store.connect(
            uri=db_config['uri'],
            user=db_config['user'],
            password=db_config['password']
        )

        # Initialize embedder
        llm_config = self.config['llm']
        self.embedder = OpenAIEmbedder(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=llm_config['embedding_model']
        )

        # Initialize retriever
        retrieval_config = self.config['retrieval']
        self.retriever = VectorRetriever(
            store=self.store,
            embedder=self.embedder,
            index_name=retrieval_config['vector_index_name'],
            node_label="Paper",
            id_field="arxiv_id"
        )

        # Initialize reasoner
        self.reasoner = GraphReasoner(store=self.store)

        # Initialize synthesizer
        self.synthesizer = OpenAISynthesizer(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=llm_config['chat_model']
        )

        # Store default temperature for use in synthesize calls
        self._default_temperature = llm_config.get('temperature', 0.3)

        # Initialize intent parser
        self.intent_parser = IntentParser(
            api_key=os.getenv('OPENAI_API_KEY'),
            schema=None  # ArXiv schema is implicit in patterns
        )

        # Initialize executor
        self.executor = QueryExecutor(
            retriever=self.retriever,
            reasoner=self.reasoner,
            synthesizer=self.synthesizer
        )

        # Initialize pipeline
        self.pipeline = GraphRAGPipeline(
            retriever=self.retriever,
            reasoner=self.reasoner,
            synthesizer=self.synthesizer,
            intent_parser=self.intent_parser,
            config=PipelineConfig(
                use_planning=False,  # Disable for legacy compatibility
                enable_caching=False,
                verbose=False
            )
        )

    # ============== BACKWARD-COMPATIBLE FUNCTIONS ==============

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Standard vector search for papers.

        Compatible with: app.py:66, app_user.py:29, app_user_clouds.py:61

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of paper dictionaries with title, id, score, authors
        """
        # Use retriever with scores
        candidates_with_scores = self.retriever.retrieve_with_scores(query, top_k=top_k)

        # Get full paper details with authors
        results = []
        for candidate in candidates_with_scores:
            paper_id = candidate['id']
            score = candidate['score']

            paper_data = self.store.query("""
                MATCH (p:Paper {arxiv_id: $id})
                MATCH (p)-[:AUTHORED_BY]->(a:Author)
                RETURN p.title as title, p.arxiv_id as id,
                       p.primary_category as category,
                       collect(a.name) as authors
            """, {"id": paper_id})

            if paper_data:
                result = paper_data[0]
                result['score'] = score
                results.append(result)

        return results

    def find_author_papers(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Find all papers by an author.

        Compatible with: app.py:82

        Args:
            author_name: Author name to search for

        Returns:
            List of paper dictionaries
        """
        results = self.store.query("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            WHERE toLower(a.name) CONTAINS toLower($name)
            RETURN a.name as author, p.title as title, p.arxiv_id as id
            LIMIT 20
        """, {"name": author_name})

        return results

    def find_collaborators(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Find who an author has worked with.

        Compatible with: app.py:94

        Args:
            author_name: Author name to search for

        Returns:
            List of collaborator dictionaries with shared paper counts
        """
        results = self.store.query("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE toLower(a.name) CONTAINS toLower($name) AND a <> coauthor
            RETURN DISTINCT a.name as author, coauthor.name as collaborator,
                   count(p) as shared_papers
            ORDER BY shared_papers DESC
            LIMIT 20
        """, {"name": author_name})

        return results

    def find_collaborator_papers(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Multi-hop: Author → Paper → Co-author → Their other papers.

        Compatible with: app.py:108

        Args:
            author_name: Starting author name

        Returns:
            List of recommended papers through collaborators
        """
        results = self.store.query("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p1:Paper)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE toLower(a.name) CONTAINS toLower($name) AND a <> coauthor
            MATCH (coauthor)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN DISTINCT a.name as author, coauthor.name as via,
                   p1.title as shared_paper, p2.title as recommended_paper, p2.arxiv_id as id
            LIMIT 20
        """, {"name": author_name})

        return results

    def find_related_papers_by_author(self, paper_title: str) -> List[Dict[str, Any]]:
        """
        Multi-hop: Paper → Authors → Their other papers.

        Compatible with: app.py:123

        Args:
            paper_title: Starting paper title

        Returns:
            List of related papers
        """
        results = self.store.query("""
            MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)
            WHERE toLower(p1.title) CONTAINS toLower($title)
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p1
            RETURN DISTINCT p1.title as source_paper, a.name as shared_author,
                   p2.title as related_paper, p2.arxiv_id as id
            LIMIT 20
        """, {"title": paper_title})

        return results

    def find_connection_path(self, author1: str, author2: str) -> List[Dict[str, Any]]:
        """
        Find how two authors are connected.

        Compatible with: app.py:138

        Args:
            author1: First author name
            author2: Second author name

        Returns:
            List with path information
        """
        results = self.store.query("""
            MATCH (a1:Author), (a2:Author)
            WHERE toLower(a1.name) CONTAINS toLower($name1)
              AND toLower(a2.name) CONTAINS toLower($name2)
            MATCH path = shortestPath((a1)-[:AUTHORED_BY|CO_AUTHORED*..8]-(a2))
            RETURN [n IN nodes(path) |
                CASE WHEN 'Author' IN labels(n) THEN n.name
                     ELSE n.title END] as path,
                   length(path) as hops
            LIMIT 1
        """, {"name1": author1, "name2": author2})

        return results

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer from context using LLM.

        Compatible with: app.py:172, app_user.py:118

        Args:
            question: User question
            context: Context string (papers, etc.)

        Returns:
            Generated answer text
        """
        # Use synthesizer directly - context should be a string
        answer = self.synthesizer.synthesize(
            query=question,
            context=context,
            temperature=self._default_temperature
        )

        return answer

    def get_paper_details(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Get full paper details with connections.

        Compatible with: app_user.py:45, app_user_clouds.py:92

        Args:
            arxiv_id: ArXiv paper ID

        Returns:
            List with paper details
        """
        results = self.store.query("""
            MATCH (p:Paper {arxiv_id: $id})
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(other:Paper)
            WHERE other <> p
            RETURN p.title as title, p.arxiv_id as id,
                   p.primary_category as category,
                   collect(DISTINCT a.name) as authors,
                   collect(DISTINCT other.title)[0..5] as related_papers,
                   collect(DISTINCT {title: other.title, id: other.arxiv_id})[0..5] as related
        """, {"id": arxiv_id})

        return results

    def get_author_profile(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Get detailed author profile.

        Compatible with: app_user.py:135, app_user_clouds.py:76

        Args:
            author_name: Author name

        Returns:
            List with author profile data
        """
        results = self.store.query("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            WITH a, p, collect(DISTINCT coauthor.name) as paper_coauthors
            RETURN a.name as name,
                   collect({
                       title: p.title,
                       id: p.arxiv_id,
                       category: p.primary_category,
                       coauthors: paper_coauthors
                   }) as papers,
                   collect(DISTINCT paper_coauthors) as all_collaborators
        """, {"name": author_name})

        # Flatten collaborators list
        if results:
            flattened = []
            for collab_list in results[0].get('all_collaborators', []):
                if isinstance(collab_list, list):
                    flattened.extend(collab_list)
            results[0]['collaborators'] = list(set(c for c in flattened if c))
            del results[0]['all_collaborators']

        return results

    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get graph statistics (node and relationship counts).

        Compatible with: app.py:54

        Returns:
            Dictionary with counts by node label and relationship type
        """
        stats = {}

        # Get node counts
        node_counts = self.store.query("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
        """)

        for row in node_counts:
            stats[row['label']] = row['count']

        # Get relationship counts
        rel_counts = self.store.query("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as count
        """)

        for row in rel_counts:
            stats[f"rel_{row['rel_type']}"] = row['count']

        return stats

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get all papers with basic info.

        Compatible with: app.py:31

        Returns:
            List of all papers
        """
        results = self.store.query("""
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.arxiv_id as id, p.title as title,
                   p.published_date as date, p.primary_category as category,
                   collect(a.name) as authors
            ORDER BY p.published_date DESC
        """)

        return results

    def get_all_authors(self) -> List[Dict[str, Any]]:
        """
        Get all authors with paper counts.

        Compatible with: app.py:44

        Returns:
            List of all authors sorted by paper count
        """
        results = self.store.query("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            RETURN a.name as name, count(p) as paper_count
            ORDER BY paper_count DESC
        """)

        return results

    def get_author_network(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Get author's full network (papers and collaborators).

        Compatible with: app.py:155

        Args:
            author_name: Author name to search for

        Returns:
            List with author network data
        """
        results = self.store.query("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor <> a
            RETURN
                a.name as author,
                collect(DISTINCT p.title) as papers,
                collect(DISTINCT coauthor.name) as collaborators
        """, {"name": author_name})

        return results

    def get_author_subgraph(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Get author subgraph for visualization.

        Compatible with: app.py:187

        Args:
            author_name: Author name

        Returns:
            List of nodes and relationships for visualization
        """
        results = self.store.query("""
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
            WHERE toLower(a.name) CONTAINS toLower($name)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            RETURN a.name as author, p.arxiv_id as paper_id,
                   p.title as paper_title, coauthor.name as coauthor
            LIMIT 100
        """, {"name": author_name})

        return results

    def get_general_subgraph(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get general graph structure for visualization.

        Compatible with: app.py:199

        Args:
            limit: Maximum number of papers to include

        Returns:
            List of papers and authors for visualization
        """
        results = self.store.query("""
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            RETURN p.arxiv_id as paper_id, p.title as paper_title, a.name as author_name
            LIMIT $limit
        """, {"limit": limit})

        return results

    def run_full_query(self, query: str, top_k: int = 10,
                       use_reasoning: bool = True) -> Dict[str, Any]:
        """
        Run full RAG pipeline query.

        Args:
            query: Natural language query
            top_k: Number of results to retrieve
            use_reasoning: Whether to use multi-hop reasoning

        Returns:
            Dictionary with answer, context, and metadata
        """
        result = self.pipeline.run(
            query=query,
            top_k=top_k,
            use_reasoning=use_reasoning
        )

        return {
            "answer": result.answer,
            "context": result.context,
            "tokens_used": result.tokens_used,
            "latency_ms": result.total_time_ms,
            "nodes_retrieved": result.nodes_retrieved,
            "edges_traversed": result.edges_traversed
        }

    def generate_themes_from_papers(self, paper_titles: List[str]) -> List[Dict[str, Any]]:
        """
        Generate research themes from paper titles using LLM.

        Compatible with: app_user_clouds.py:141

        Args:
            paper_titles: List of paper title strings

        Returns:
            List of theme dictionaries with name, count, description
        """
        import json

        titles_text = "\n".join(paper_titles)

        # Use synthesizer's OpenAI client directly for this specialized task
        response = self.synthesizer._client.chat.completions.create(
            model=self.synthesizer._model,
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
            # Fallback themes
            return [
                {"name": "Large Language Models", "count": 80, "description": "Research on LLMs"},
                {"name": "Computer Vision", "count": 50, "description": "Image and video understanding"},
                {"name": "Reinforcement Learning", "count": 40, "description": "RL methods and applications"},
                {"name": "Transformers", "count": 60, "description": "Attention-based architectures"}
            ]

    def generate_digest_from_papers(self, paper_summaries: List[str]) -> str:
        """
        Generate a narrative digest from paper summaries using LLM.

        Compatible with: app_user_clouds.py:171

        Args:
            paper_summaries: List of paper summary strings

        Returns:
            Narrative digest text with [[bracketed]] key terms
        """
        titles_text = "\n".join(paper_summaries)

        response = self.synthesizer._client.chat.completions.create(
            model=self.synthesizer._model,
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

    def extract_paper_insight(self, title: str) -> Dict[str, str]:
        """
        Extract key insights from a paper title using LLM.

        Compatible with: app_user_clouds.py:197

        Args:
            title: Paper title

        Returns:
            Dictionary with one_liner, key_idea, why_care
        """
        import json

        response = self.synthesizer._client.chat.completions.create(
            model=self.synthesizer._model,
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
            return {
                "one_liner": title,
                "key_idea": "Research",
                "why_care": "Novel contribution to the field."
            }

    def explore_author_ecosystem(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Explore author's research ecosystem (collaborators' papers).

        Compatible with: app_user_clouds.py:105

        Args:
            author_name: Author name to explore

        Returns:
            List of papers in author's extended network
        """
        results = self.store.query("""
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
        """, {"name": author_name})

        return results

    def explore_paper_connections(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Explore paper connections through authors.

        Compatible with: app_user_clouds.py:123

        Args:
            arxiv_id: ArXiv paper ID

        Returns:
            List of related papers through shared authors
        """
        results = self.store.query("""
            MATCH (p:Paper {arxiv_id: $id})-[:AUTHORED_BY]->(a:Author)
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2 <> p
            RETURN DISTINCT
                p.title as source,
                a.name as via_author,
                p2.title as related_paper,
                p2.arxiv_id as related_id
            LIMIT 15
        """, {"id": arxiv_id})

        return results

    # ============== CONCEPT-BASED METHODS ==============

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """Get all concepts with paper counts."""
        results = self.store.query("""
            MATCH (p:Paper)
            RETURN p.primary_category as concept, count(p) as paper_count
            ORDER BY paper_count DESC
        """)
        return results

    def find_papers_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        """Find all papers for a specific concept."""
        results = self.store.query("""
            MATCH (p:Paper {primary_category: $concept})
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.title as title, p.arxiv_id as id,
                   p.published_date as date,
                   collect(a.name) as authors
            ORDER BY p.published_date DESC
        """, {"concept": concept})
        return results

    def get_concept_cooccurrence(self) -> List[Dict[str, Any]]:
        """
        Find concept co-occurrence via shared authors.
        Shows which concepts are connected through researchers who work across multiple areas.
        """
        results = self.store.query("""
            MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p1.primary_category < p2.primary_category
            RETURN p1.primary_category as concept1,
                   p2.primary_category as concept2,
                   count(DISTINCT a) as shared_authors,
                   count(DISTINCT p1) + count(DISTINCT p2) as papers
            ORDER BY shared_authors DESC
            LIMIT 20
        """)
        return results

    def find_bridge_papers(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find papers whose authors work across multiple concepts.
        These are "bridge papers" connecting different research areas.
        """
        results = self.store.query("""
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            MATCH (a)<-[:AUTHORED_BY]-(other:Paper)
            WHERE p <> other AND p.primary_category <> other.primary_category
            WITH p, collect(DISTINCT other.primary_category) as other_concepts,
                 count(DISTINCT other.primary_category) as concept_diversity
            WHERE concept_diversity > 0
            MATCH (p)-[:AUTHORED_BY]->(author:Author)
            RETURN p.title as title, p.arxiv_id as id,
                   p.primary_category as primary_concept,
                   other_concepts,
                   concept_diversity,
                   collect(author.name) as authors
            ORDER BY concept_diversity DESC
            LIMIT $limit
        """, {"limit": limit})
        return results

    def find_related_concepts(self, concept: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find concepts related to a given concept via shared authors.
        """
        results = self.store.query("""
            MATCH (p1:Paper {primary_category: $concept})-[:AUTHORED_BY]->(a:Author)
            MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2.primary_category <> $concept
            RETURN p2.primary_category as related_concept,
                   count(DISTINCT a) as shared_authors,
                   count(DISTINCT p2) as paper_count
            ORDER BY shared_authors DESC, paper_count DESC
            LIMIT $top_k
        """, {"concept": concept, "top_k": top_k})
        return results

    def discover_research_path(self, start_paper_id: str) -> Dict[str, Any]:
        """
        Given a starting paper, discover a research path through related concepts.
        Returns a journey through the knowledge graph.
        """
        # Get starting paper's concept
        paper_info = self.store.query("""
            MATCH (p:Paper {arxiv_id: $id})
            RETURN p.title as title, p.primary_category as concept
        """, {"id": start_paper_id})

        if not paper_info:
            return {}

        start_concept = paper_info[0]['concept']

        # Find related concepts and papers
        related_concepts = self.find_related_concepts(start_concept, top_k=3)

        # Build the journey
        journey = {
            "start_paper": paper_info[0]['title'],
            "start_concept": start_concept,
            "paths": []
        }

        for rc in related_concepts:
            related_concept = rc['related_concept']
            # Find bridge papers
            bridge_papers = self.store.query("""
                MATCH (p1:Paper {primary_category: $concept1})-[:AUTHORED_BY]->(a:Author)
                MATCH (a)<-[:AUTHORED_BY]-(p2:Paper {primary_category: $concept2})
                RETURN DISTINCT p2.title as title, p2.arxiv_id as id,
                       a.name as bridge_author
                LIMIT 3
            """, {"concept1": start_concept, "concept2": related_concept})

            if bridge_papers:
                journey["paths"].append({
                    "to_concept": related_concept,
                    "shared_authors": rc['shared_authors'],
                    "bridge_papers": bridge_papers
                })

        return journey

    # ============== RICH CONCEPT/METHOD/DATASET METHODS ==============

    def get_all_methods(self) -> List[Dict[str, Any]]:
        """Get all methods with paper counts."""
        results = self.store.query("""
            MATCH (m:Method)<-[:USES_METHOD]-(p:Paper)
            RETURN m.name as method, m.display_name as display_name,
                   count(p) as paper_count
            ORDER BY paper_count DESC
        """)
        return results

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """Get all topic concepts (not categories) with paper counts."""
        results = self.store.query("""
            MATCH (c:Concept {type: 'topic'})<-[:DISCUSSES]-(p:Paper)
            RETURN c.name as concept, c.display_name as display_name,
                   count(p) as paper_count
            ORDER BY paper_count DESC
        """)
        return results

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Get all datasets with paper counts."""
        results = self.store.query("""
            MATCH (d:Dataset)<-[:USES_DATASET]-(p:Paper)
            RETURN d.name as dataset, d.display_name as display_name,
                   count(p) as paper_count
            ORDER BY paper_count DESC
        """)
        return results

    def find_papers_by_method(self, method: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Find papers using a specific method."""
        results = self.store.query("""
            MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: $method})
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.title as title, p.arxiv_id as id,
                   p.primary_category as category,
                   collect(DISTINCT a.name) as authors
            LIMIT $limit
        """, {"method": method, "limit": limit})
        return results

    def find_papers_by_topic(self, topic: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Find papers discussing a specific topic concept."""
        results = self.store.query("""
            MATCH (p:Paper)-[:DISCUSSES]->(c:Concept {name: $topic})
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.title as title, p.arxiv_id as id,
                   p.primary_category as category,
                   collect(DISTINCT a.name) as authors
            LIMIT $limit
        """, {"topic": topic, "limit": limit})
        return results

    def find_method_in_field(self, method: str, field: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Find papers using a specific method in a specific field (category)."""
        results = self.store.query("""
            MATCH (p:Paper {primary_category: $field})-[:USES_METHOD]->(m:Method {name: $method})
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.title as title, p.arxiv_id as id,
                   collect(DISTINCT a.name) as authors
            LIMIT $limit
        """, {"method": method, "field": field, "limit": limit})
        return results

    def find_method_transfer(self, source_field: str, target_field: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find methods that started in one field and are now used in another.
        This is powerful for discovering technique transfer across domains.
        """
        results = self.store.query("""
            // Find methods used in both fields
            MATCH (p1:Paper {primary_category: $source})-[:USES_METHOD]->(m:Method)
                  <-[:USES_METHOD]-(p2:Paper {primary_category: $target})
            WITH m, count(DISTINCT p1) as source_count, count(DISTINCT p2) as target_count
            WHERE source_count > 0 AND target_count > 0

            // Get example papers from target field
            MATCH (example:Paper {primary_category: $target})-[:USES_METHOD]->(m)
            WITH m, source_count, target_count, collect(DISTINCT example.title)[0..2] as example_papers

            RETURN m.name as method, m.display_name as display_name,
                   source_count, target_count,
                   example_papers
            ORDER BY target_count DESC
            LIMIT $limit
        """, {"source": source_field, "target": target_field, "limit": limit})
        return results

    def find_methods_for_concept(self, concept: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find which methods are commonly used for a specific concept/topic."""
        results = self.store.query("""
            MATCH (p:Paper)-[:DISCUSSES]->(c:Concept {name: $concept})
            MATCH (p)-[:USES_METHOD]->(m:Method)
            WITH m, count(p) as usage_count
            ORDER BY usage_count DESC
            LIMIT $limit
            RETURN m.name as method, m.display_name as display_name, usage_count
        """, {"concept": concept, "limit": limit})
        return results

    def find_concept_method_bridge(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find which concepts share similar method usage.
        This reveals unexpected connections between research areas.
        """
        results = self.store.query("""
            MATCH (c1:Concept {type: 'topic'})<-[:DISCUSSES]-(p1:Paper)-[:USES_METHOD]->(m:Method)
                  <-[:USES_METHOD]-(p2:Paper)-[:DISCUSSES]->(c2:Concept {type: 'topic'})
            WHERE c1.name < c2.name  // Avoid duplicates
            WITH c1, c2, collect(DISTINCT m.name) as shared_methods, count(DISTINCT m) as method_count
            WHERE method_count >= 2
            RETURN c1.display_name as concept1, c2.display_name as concept2,
                   shared_methods, method_count
            ORDER BY method_count DESC
            LIMIT $limit
        """, {"limit": limit})
        return results

    def get_paper_full_context(self, arxiv_id: str) -> Dict[str, Any]:
        """Get full paper context including methods, concepts, datasets."""
        results = self.store.query("""
            MATCH (p:Paper {arxiv_id: $id})
            OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
            OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
            OPTIONAL MATCH (p)-[:DISCUSSES]->(c:Concept {type: 'topic'})
            OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
            RETURN p.title as title, p.arxiv_id as id,
                   p.primary_category as category,
                   collect(DISTINCT a.name) as authors,
                   collect(DISTINCT m.display_name) as methods,
                   collect(DISTINCT c.display_name) as concepts,
                   collect(DISTINCT d.display_name) as datasets
        """, {"id": arxiv_id})
        return results[0] if results else {}

    def find_similar_papers_by_methods(self, arxiv_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers using similar methods (regardless of field)."""
        results = self.store.query("""
            MATCH (p1:Paper {arxiv_id: $id})-[:USES_METHOD]->(m:Method)
                  <-[:USES_METHOD]-(p2:Paper)
            WHERE p2 <> p1
            WITH p2, collect(DISTINCT m.display_name) as shared_methods, count(DISTINCT m) as method_overlap
            WHERE method_overlap > 0
            MATCH (p2)-[:AUTHORED_BY]->(a:Author)
            RETURN p2.title as title, p2.arxiv_id as id,
                   p2.primary_category as category,
                   shared_methods,
                   method_overlap,
                   collect(DISTINCT a.name)[0..2] as authors
            ORDER BY method_overlap DESC
            LIMIT $limit
        """, {"id": arxiv_id, "limit": limit})
        return results

    def close(self):
        """Close all connections."""
        self.store.close()


# Global adapter instance (will be cached by Streamlit)
_adapter: Optional[ArxivRAGAdapter] = None


def get_adapter(config_path: Optional[str] = None) -> ArxivRAGAdapter:
    """
    Get or create the global adapter instance.

    This function should be wrapped with @st.cache_resource in Streamlit apps.

    Args:
        config_path: Optional path to config file

    Returns:
        ArxivRAGAdapter instance
    """
    global _adapter
    if _adapter is None:
        _adapter = ArxivRAGAdapter(config_path)
    return _adapter


def graph_enhanced_search(adapter, vector_results: List[Dict], top_k: int = 5) -> Dict[str, Any]:
    """
    Find papers through graph connections that vector search would miss.

    This is the "aha moment" - papers with low text similarity but high graph relevance.

    Returns:
        Dict with 'through_authors', 'through_concepts', 'hidden_connections'
    """
    if not vector_results:
        return {"through_authors": [], "through_concepts": [], "hidden_connections": []}

    # Get IDs and authors from vector results
    vector_ids = [p['id'] for p in vector_results]
    vector_authors = []
    for p in vector_results:
        vector_authors.extend(p.get('authors', [])[:3])  # Top 3 authors per paper
    vector_authors = list(set(vector_authors))[:10]  # Unique, max 10

    # 1. Find papers by the same authors (but not in vector results)
    through_authors = adapter.store.query("""
        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
        WHERE a.name IN $authors
        AND NOT p.arxiv_id IN $exclude_ids
        WITH p, collect(DISTINCT a.name) as matching_authors
        MATCH (p)-[:AUTHORED_BY]->(all_authors:Author)
        RETURN DISTINCT p.title as title,
               p.arxiv_id as id,
               p.primary_category as category,
               matching_authors as connection_authors,
               collect(DISTINCT all_authors.name) as all_authors,
               'Same author, different topic' as why_relevant
        LIMIT $limit
    """, {"authors": vector_authors, "exclude_ids": vector_ids, "limit": top_k})

    # 2. Find papers through co-author network (2 hops)
    through_coauthors = adapter.store.query("""
        MATCH (p1:Paper)-[:AUTHORED_BY]->(a1:Author)<-[:AUTHORED_BY]-(shared:Paper)
              -[:AUTHORED_BY]->(a2:Author)<-[:AUTHORED_BY]-(p2:Paper)
        WHERE p1.arxiv_id IN $seed_ids
        AND NOT p2.arxiv_id IN $exclude_ids
        AND p2.primary_category <> p1.primary_category
        AND a1 <> a2
        WITH p2, a1, a2, p1, count(DISTINCT shared) as connection_strength
        WHERE connection_strength > 0
        MATCH (p2)-[:AUTHORED_BY]->(authors:Author)
        WITH p2, a1, a2, p1, connection_strength, collect(DISTINCT authors.name) as all_authors
        RETURN p2.title as title,
               p2.arxiv_id as id,
               p2.primary_category as category,
               a1.name as source_author,
               a2.name as bridge_author,
               p1.primary_category as source_category,
               all_authors,
               connection_strength,
               'Connected through collaborator network' as why_relevant
        ORDER BY connection_strength DESC
        LIMIT $limit
    """, {"seed_ids": vector_ids[:3], "exclude_ids": vector_ids, "limit": top_k})

    # 3. Find papers in adjacent concepts (different field, shared researchers)
    if vector_results:
        seed_category = vector_results[0].get('category', '')
        through_concepts = adapter.store.query("""
            MATCH (p1:Paper {primary_category: $category})-[:AUTHORED_BY]->(a:Author)
                  <-[:AUTHORED_BY]-(p2:Paper)
            WHERE p2.primary_category <> $category
            AND NOT p2.arxiv_id IN $exclude_ids
            WITH p2, p2.primary_category as other_category,
                 collect(DISTINCT a.name) as bridge_authors
            WHERE size(bridge_authors) > 0
            MATCH (p2)-[:AUTHORED_BY]->(all_auth:Author)
            RETURN DISTINCT p2.title as title,
                   p2.arxiv_id as id,
                   other_category as category,
                   bridge_authors,
                   collect(DISTINCT all_auth.name) as all_authors,
                   $category as source_category,
                   'Cross-field by shared researchers' as why_relevant
            ORDER BY size(bridge_authors) DESC
            LIMIT $limit
        """, {"category": seed_category, "exclude_ids": vector_ids, "limit": top_k})
    else:
        through_concepts = []

    return {
        "through_authors": through_authors,
        "through_coauthors": through_coauthors,
        "through_concepts": through_concepts
    }
