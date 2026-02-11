"""
Neurology Navigator - Graph RAG for Medical Research & Patient Data

Demonstrates Graph RAG patterns applied to neurology:
- Disease x Symptom filtering (like Feature x UseCase)
- Research vs Patient comparison (UNIQUE: dual data sources!)
- Research Gap Finder (what patients report but research ignores)
- Cross-Disease Discovery (like Bridge Researchers)
- Treatment Mechanism Paths (like Reading Path)
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

# Page config
st.set_page_config(
    page_title="Neurology Navigator - Graph RAG",
    page_icon="🧠",
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


def get_all_diseases():
    """Get all diseases with paper counts."""
    return query("""
        MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(p:Paper)
        RETURN d.name as name, count(p) as paper_count
        ORDER BY paper_count DESC
    """)


def get_all_symptoms():
    """Get all research symptoms with mention counts."""
    return query("""
        MATCH (s:Symptom)<-[:MENTIONS_SYMPTOM]-(p:Paper)
        RETURN s.name as name, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT 50
    """)


def get_all_reported_symptoms():
    """Get patient-reported symptoms with counts."""
    return query("""
        MATCH (rs:ReportedSymptom)<-[:REPORTS_SYMPTOM]-(r:RedditPost)
        RETURN rs.name as name, count(r) as report_count
        ORDER BY report_count DESC
        LIMIT 50
    """)


def get_all_proteins():
    """Get proteins with mention counts."""
    return query("""
        MATCH (pr:Protein)<-[:MENTIONS_PROTEIN]-(p:Paper)
        RETURN pr.name as name, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT 30
    """)


def get_all_mechanisms():
    """Get mechanisms with mention counts."""
    return query("""
        MATCH (m:Mechanism)<-[:MENTIONS_MECHANISM]-(p:Paper)
        RETURN m.name as name, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT 30
    """)


def get_all_treatments():
    """Get treatments with mention counts."""
    return query("""
        MATCH (t:Treatment)<-[:MENTIONS_TREATMENT]-(p:Paper)
        RETURN t.name as name, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT 30
    """)


# Disease category icons
DISEASE_ICONS = {
    "Alzheimer": "🧓",
    "Parkinson": "🤲",
    "Multiple Sclerosis": "🔗",
    "ALS": "💪",
    "Huntington": "🧬",
    "Dementia": "🧠",
    "default": "🏥"
}


def get_disease_icon(disease_name: str) -> str:
    """Get icon for disease."""
    for key, icon in DISEASE_ICONS.items():
        if key.lower() in disease_name.lower():
            return icon
    return DISEASE_ICONS["default"]


def create_graph_visualization(nodes_data: list, edges_data: list, height: str = "600px"):
    """Create an interactive graph visualization using pyvis."""
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Color scheme for node types
    colors = {
        "Paper": "#ff6b6b",
        "Disease": "#4ecdc4",
        "Symptom": "#ffe66d",
        "ReportedSymptom": "#ffa500",
        "BrainRegion": "#95e1d3",
        "Protein": "#dda0dd",
        "Gene": "#87ceeb",
        "Mechanism": "#f0e68c",
        "Treatment": "#98fb98",
        "RedditPost": "#ff7f50"
    }

    # Add nodes
    for node in nodes_data:
        net.add_node(
            node['id'],
            label=node['label'][:25],
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

st.title("🧠 Neurology Navigator")
st.markdown("*Graph RAG combining research literature + patient experiences. What does research miss?*")

# Quick stats
stats = get_stats()
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Papers", stats.get('Paper', 0))
with col2:
    st.metric("Diseases", stats.get('Disease', 0))
with col3:
    st.metric("Symptoms", stats.get('Symptom', 0))
with col4:
    st.metric("Patient Posts", stats.get('RedditPost', 0))
with col5:
    st.metric("Reported Symptoms", stats.get('ReportedSymptom', 0))
with col6:
    st.metric("Treatments", stats.get('Treatment', 0))

st.divider()

# ============== TABS ==============

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 How It Works",
    "🔀 Disease x Symptom",
    "⚖️ Research vs Patient",
    "🔍 Research Gaps",
    "🌉 Cross-Disease",
    "💊 Treatment Paths",
    "🕸️ Graph Explorer"
])


# ============== TAB 1: HOW IT WORKS ==============

with tab1:
    st.header("📊 Graph RAG for Neurology")

    st.markdown("""
    ### The Same Pattern, Medical Domain

    | Research Papers | Products | **Neurology** |
    |-----------------|----------|---------------|
    | Paper → USES_METHOD → Method | Product → HAS_FEATURE → Feature | Paper → MENTIONS_SYMPTOM → Symptom |
    | Paper → DISCUSSES → Concept | Product → FOR_USE_CASE → UseCase | Paper → MENTIONS_DISEASE → Disease |
    | Paper → AUTHORED_BY → Author | Product → MADE_BY → Brand | Paper → MENTIONS_PROTEIN → Protein |
    | **Method x Concept** | **Feature x UseCase** | **Disease x Symptom** |
    """)

    st.markdown("### Our Unique Advantage: Dual Data Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **📚 Research Data (PubMed)**

        What scientists study:
        - Clinical symptoms
        - Biomarkers
        - Mechanisms
        - Treatments

        *Rigorous but may miss lived experience*
        """)

    with col2:
        st.warning("""
        **👥 Patient Data (Reddit)**

        What patients actually experience:
        - Day-to-day symptoms
        - Quality of life impacts
        - Emotional challenges
        - What helps (or doesn't)

        *Authentic but may lack clinical precision*
        """)

    st.success("""
    ### 🔥 The Killer Feature: Research Gap Finder

    **No other demo has this.** We can find:
    - Symptoms patients report that research ignores
    - Treatments that work in practice but aren't studied
    - Quality of life impacts that deserve more research

    **Vector search can't do this.** You need graph structure to compare across data sources.
    """)

    st.markdown("### The Knowledge Graph")

    st.code("""
Research Data (PubMed):
  Paper
    ├── MENTIONS_DISEASE ──→ Disease (Alzheimer's, Parkinson's...)
    ├── MENTIONS_SYMPTOM ──→ Symptom (memory loss, tremor...)
    ├── MENTIONS_PROTEIN ──→ Protein (tau, amyloid-beta...)
    ├── MENTIONS_MECHANISM ──→ Mechanism (neuroinflammation...)
    └── MENTIONS_TREATMENT ──→ Treatment (levodopa, memantine...)

Patient Data (Reddit):
  RedditPost
    ├── DISCUSSES ──→ Disease
    ├── REPORTS_SYMPTOM ──→ ReportedSymptom (detailed patient experiences)
    └── REPORTS_IMPACT ──→ LifeImpact (daily life effects)

Aggregated:
  Disease
    ├── HAS_SYMPTOM ──→ Symptom (from research, with paper_count)
    ├── HAS_REPORTED_SYMPTOM ──→ ReportedSymptom (from patients, with report_count)
    └── SHARES_SYMPTOMS_WITH ──→ Disease (symptom overlap)
    """)


# ============== TAB 2: DISEASE x SYMPTOM FILTER ==============

with tab2:
    st.header("🔀 Disease x Symptom Filter")
    st.markdown("""
    **The intersection query.** Find papers about specific DISEASES with specific SYMPTOMS.

    *Like "Feature x UseCase" for products, or "Method x Concept" for research papers.*
    """)

    diseases = get_all_diseases()
    symptoms = get_all_symptoms()

    col1, col2 = st.columns(2)

    with col1:
        disease_opts = ["Any disease"] + [f"{d['name']} ({d['paper_count']} papers)" for d in diseases]
        selected_disease = st.selectbox("Disease:", disease_opts)

    with col2:
        symptom_opts = ["Any symptom"] + [f"{s['name']} ({s['paper_count']} papers)" for s in symptoms[:20]]
        selected_symptom = st.selectbox("Symptom:", symptom_opts)

    if st.button("🔍 Find Research", type="primary", key="filter_btn"):
        conditions = []
        params = {}

        if selected_disease != "Any disease":
            disease_name = diseases[[d['name'] for d in diseases].index(selected_disease.split(" (")[0])]['name']
            conditions.append("(p)-[:MENTIONS_DISEASE]->(:Disease {name: $disease})")
            params['disease'] = disease_name

        if selected_symptom != "Any symptom":
            symptom_name = symptoms[[s['name'] for s in symptoms].index(selected_symptom.split(" (")[0])]['name']
            conditions.append("(p)-[:MENTIONS_SYMPTOM]->(:Symptom {name: $symptom})")
            params['symptom'] = symptom_name

        where_clause = " AND ".join(conditions) if conditions else "true"

        results = query(f"""
            MATCH (p:Paper)
            WHERE {where_clause}
            OPTIONAL MATCH (p)-[:MENTIONS_DISEASE]->(d:Disease)
            OPTIONAL MATCH (p)-[:MENTIONS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (p)-[:MENTIONS_TREATMENT]->(t:Treatment)
            RETURN p.pmid as pmid, p.title as title, p.year as year,
                   p.abstract as abstract,
                   collect(DISTINCT d.name) as diseases,
                   collect(DISTINCT s.name) as symptoms,
                   collect(DISTINCT t.name) as treatments
            ORDER BY p.year DESC
            LIMIT 15
        """, params)

        if results:
            st.success(f"**Found {len(results)} papers** matching your criteria!")

            for i, paper in enumerate(results, 1):
                icon = get_disease_icon(paper['diseases'][0] if paper['diseases'] else "")

                st.markdown(f"### {icon} {i}. {paper['title']}")
                st.caption(f"PMID: {paper['pmid']} | Year: {paper['year']}")

                if paper['diseases']:
                    st.markdown(f"🏥 **Diseases:** {', '.join(paper['diseases'][:5])}")
                if paper['symptoms']:
                    st.markdown(f"🩺 **Symptoms:** {', '.join(paper['symptoms'][:5])}")
                if paper['treatments']:
                    st.markdown(f"💊 **Treatments:** {', '.join(paper['treatments'][:3])}")

                with st.expander("View Abstract"):
                    st.write(paper['abstract'][:500] + "..." if paper['abstract'] else "No abstract available")

                st.markdown("---")

            st.info("""
            💡 **Why this is powerful:**

            Vector search for "Alzheimer's memory" finds papers mentioning these words.
            This finds papers that **study Alzheimer's** AND **discuss memory symptoms** — structural query.
            """)
        else:
            st.warning("No papers found. Try different criteria.")


# ============== TAB 3: RESEARCH VS PATIENT ==============

with tab3:
    st.header("⚖️ Research vs Patient Experience")
    st.markdown("""
    **Side-by-side comparison.** What does research say vs what patients actually report?

    *This is impossible with vector search — it requires comparing across two different data sources.*
    """)

    diseases = get_all_diseases()

    selected_disease = st.selectbox(
        "Select a disease to compare:",
        [d['name'] for d in diseases],
        key="compare_disease"
    )

    if st.button("⚖️ Compare Perspectives", type="primary", key="compare_btn"):
        col1, col2 = st.columns(2)

        # Research symptoms
        with col1:
            st.markdown("### 📚 Research Symptoms")
            st.caption("What PubMed papers mention")

            research = query("""
                MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom)
                RETURN s.name as symptom, r.paper_count as count
                ORDER BY r.paper_count DESC
                LIMIT 15
            """, {"disease": selected_disease})

            if research:
                max_count = max(r['count'] for r in research) if research else 1
                for item in research:
                    pct = int((item['count'] / max_count) * 100)
                    st.progress(pct / 100, text=f"**{item['symptom']}** ({item['count']} papers)")
            else:
                st.info("No research symptoms found for this disease.")

        # Patient-reported symptoms
        with col2:
            st.markdown("### 👥 Patient-Reported Symptoms")
            st.caption("What patients on Reddit mention")

            patient = query("""
                MATCH (d:Disease {name: $disease})-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                RETURN rs.name as symptom, r.report_count as count
                ORDER BY r.report_count DESC
                LIMIT 15
            """, {"disease": selected_disease})

            if patient:
                max_count = max(p['count'] for p in patient) if patient else 1
                for item in patient:
                    pct = int((item['count'] / max_count) * 100)
                    # Truncate long symptom names
                    symptom_display = item['symptom'][:50] + "..." if len(item['symptom']) > 50 else item['symptom']
                    st.progress(pct / 100, text=f"**{symptom_display}** ({item['count']} reports)")
            else:
                st.info("No patient reports found for this disease.")

        st.divider()

        # Key insights
        st.markdown("### 💡 Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **Research** tends to focus on:
            - Clinical symptoms with diagnostic value
            - Biomarkers and pathological findings
            - Measurable outcomes for trials
            """)

        with col2:
            st.warning("""
            **Patients** often report:
            - Day-to-day quality of life impacts
            - Emotional and social challenges
            - "Invisible" symptoms
            """)

        st.success("""
        🎯 **The gap between these lists represents research opportunities!**

        Go to the **Research Gaps** tab to find symptoms patients report that research hasn't covered.
        """)


# ============== TAB 4: RESEARCH GAPS ==============

with tab4:
    st.header("🔍 Research Gap Finder")
    st.markdown("""
    **Our unique advantage.** Find what patients report that research doesn't cover.

    *This is ONLY possible because we have both data sources in the same graph.*
    """)

    diseases = get_all_diseases()

    selected_disease = st.selectbox(
        "Select a disease:",
        [d['name'] for d in diseases],
        key="gap_disease"
    )

    if st.button("🔍 Find Research Gaps", type="primary", key="gap_btn"):
        # Find patient-reported symptoms that have NO matching research symptom
        gaps = query("""
            MATCH (d:Disease {name: $disease})-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
            WHERE NOT EXISTS {
                MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                WHERE toLower(s.name) CONTAINS toLower(rs.name)
                   OR toLower(rs.name) CONTAINS toLower(s.name)
            }
            RETURN rs.name as symptom, r.report_count as patient_reports
            ORDER BY r.report_count DESC
            LIMIT 25
        """, {"disease": selected_disease})

        if gaps:
            st.success(f"**Found {len(gaps)} potential research gaps!**")
            st.markdown("*These symptoms are frequently reported by patients but not found in research literature:*")

            st.markdown("---")

            # Group by report count
            high_priority = [g for g in gaps if g['patient_reports'] >= 5]
            medium_priority = [g for g in gaps if 2 <= g['patient_reports'] < 5]
            low_priority = [g for g in gaps if g['patient_reports'] < 2]

            if high_priority:
                st.markdown("### 🔴 High Priority (5+ patient reports)")
                for gap in high_priority:
                    st.markdown(f"**{gap['symptom']}** — {gap['patient_reports']} patients report this")

            if medium_priority:
                st.markdown("### 🟡 Medium Priority (2-4 patient reports)")
                for gap in medium_priority:
                    st.markdown(f"**{gap['symptom']}** — {gap['patient_reports']} reports")

            if low_priority:
                with st.expander(f"🟢 Lower Priority ({len(low_priority)} symptoms with 1 report)"):
                    for gap in low_priority:
                        st.markdown(f"- {gap['symptom']}")

            st.divider()

            st.success("""
            ✅ **Why This Matters:**

            These are symptoms that real patients experience but that may not be getting research attention.

            **Potential reasons:**
            - Symptoms too subjective to study
            - Quality of life impacts undervalued
            - Emerging symptoms not yet recognized

            **Research opportunity:** These gaps could inform patient-centered research priorities.
            """)
        else:
            st.info("No clear research gaps found. Research and patient reports align well for this disease.")

        st.divider()

        # Also show the reverse: what research covers that patients don't mention
        st.markdown("### 📚 Research-Only Symptoms")
        st.caption("Symptoms in literature that patients rarely report (may indicate clinical vs lived experience gap)")

        research_only = query("""
            MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom)
            WHERE NOT EXISTS {
                MATCH (d)-[:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                WHERE toLower(rs.name) CONTAINS toLower(s.name)
                   OR toLower(s.name) CONTAINS toLower(rs.name)
            }
            RETURN s.name as symptom, r.paper_count as papers
            ORDER BY r.paper_count DESC
            LIMIT 10
        """, {"disease": selected_disease})

        if research_only:
            for item in research_only:
                st.markdown(f"- **{item['symptom']}** ({item['papers']} papers)")

            st.info("""
            💡 These may be:
            - Technical/pathological terms patients don't use
            - Symptoms detected only through testing
            - Early-stage indicators not yet experienced
            """)


# ============== TAB 5: CROSS-DISEASE DISCOVERY ==============

with tab5:
    st.header("🌉 Cross-Disease Discovery")
    st.markdown("""
    **Find connections between diseases.** Shared symptoms, mechanisms, or treatments could indicate
    common pathways or research opportunities.

    *Like "Bridge Researchers" in the research demo.*
    """)

    subtab1, subtab2, subtab3 = st.tabs([
        "🔗 Shared Symptoms",
        "⚙️ Shared Mechanisms",
        "💊 Shared Treatments"
    ])

    diseases = get_all_diseases()

    # ---- Shared Symptoms ----
    with subtab1:
        st.subheader("🔗 Diseases with Overlapping Symptoms")

        col1, col2 = st.columns(2)

        with col1:
            disease1 = st.selectbox("First disease:", [d['name'] for d in diseases], key="disease1")

        with col2:
            disease2 = st.selectbox("Second disease:", [d['name'] for d in diseases], key="disease2", index=1)

        if st.button("🔗 Find Shared Symptoms", type="primary", key="shared_btn"):
            if disease1 == disease2:
                st.warning("Please select two different diseases.")
            else:
                # Find shared symptoms
                shared = query("""
                    MATCH (d1:Disease {name: $disease1})-[r1:HAS_SYMPTOM]->(s:Symptom)
                          <-[r2:HAS_SYMPTOM]-(d2:Disease {name: $disease2})
                    RETURN s.name as symptom,
                           r1.paper_count as disease1_papers,
                           r2.paper_count as disease2_papers
                    ORDER BY (r1.paper_count + r2.paper_count) DESC
                    LIMIT 20
                """, {"disease1": disease1, "disease2": disease2})

                if shared:
                    st.success(f"**Found {len(shared)} shared symptoms!**")

                    icon1 = get_disease_icon(disease1)
                    icon2 = get_disease_icon(disease2)

                    st.markdown(f"### {icon1} {disease1} ↔️ {icon2} {disease2}")

                    for item in shared:
                        col_sym, col_d1, col_d2 = st.columns([3, 1, 1])
                        with col_sym:
                            st.markdown(f"🔸 **{item['symptom']}**")
                        with col_d1:
                            st.caption(f"{item['disease1_papers']} papers")
                        with col_d2:
                            st.caption(f"{item['disease2_papers']} papers")

                    st.divider()

                    # Unique to each
                    st.markdown("### 🎯 Differentiating Symptoms")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Only in {disease1}:**")
                        unique1 = query("""
                            MATCH (d1:Disease {name: $disease1})-[:HAS_SYMPTOM]->(s:Symptom)
                            WHERE NOT (s)<-[:HAS_SYMPTOM]-(:Disease {name: $disease2})
                            RETURN s.name as symptom
                            LIMIT 8
                        """, {"disease1": disease1, "disease2": disease2})
                        for item in unique1:
                            st.markdown(f"• {item['symptom']}")

                    with col2:
                        st.markdown(f"**Only in {disease2}:**")
                        unique2 = query("""
                            MATCH (d2:Disease {name: $disease2})-[:HAS_SYMPTOM]->(s:Symptom)
                            WHERE NOT (s)<-[:HAS_SYMPTOM]-(:Disease {name: $disease1})
                            RETURN s.name as symptom
                            LIMIT 8
                        """, {"disease1": disease1, "disease2": disease2})
                        for item in unique2:
                            st.markdown(f"• {item['symptom']}")

                    st.success("""
                    ✅ **Clinical Insight:**

                    Shared symptoms → possible misdiagnosis, common mechanisms
                    Unique symptoms → differential diagnosis markers
                    """)
                else:
                    st.info("No shared symptoms found in the research literature.")

    # ---- Shared Mechanisms ----
    with subtab2:
        st.subheader("⚙️ Diseases with Shared Mechanisms")
        st.markdown("*Common biological pathways may indicate treatment opportunities*")

        if st.button("🔍 Find Mechanism Connections", type="primary", key="mech_btn"):
            mechanism_links = query("""
                MATCH (d1:Disease)-[:INVOLVES_MECHANISM]->(m:Mechanism)<-[:INVOLVES_MECHANISM]-(d2:Disease)
                WHERE d1 <> d2 AND id(d1) < id(d2)
                WITH d1, d2, collect(m.name) as shared_mechanisms, count(m) as mech_count
                WHERE mech_count >= 1
                RETURN d1.name as disease1, d2.name as disease2,
                       shared_mechanisms, mech_count
                ORDER BY mech_count DESC
                LIMIT 15
            """)

            if mechanism_links:
                st.success(f"**Found {len(mechanism_links)} disease pairs sharing mechanisms!**")

                for link in mechanism_links:
                    icon1 = get_disease_icon(link['disease1'])
                    icon2 = get_disease_icon(link['disease2'])
                    mechs = ', '.join(link['shared_mechanisms'][:3])

                    st.markdown(f"{icon1} **{link['disease1']}** ↔️ {icon2} **{link['disease2']}**")
                    st.caption(f"Shared mechanisms: {mechs}")
                    st.markdown("---")

                st.info("""
                💡 **Research Opportunity:**

                Diseases sharing mechanisms may respond to similar treatments.
                This enables drug repurposing research.
                """)
            else:
                st.info("No mechanism connections found.")

    # ---- Shared Treatments ----
    with subtab3:
        st.subheader("💊 Treatments Used Across Diseases")
        st.markdown("*Treatments that work for multiple conditions*")

        if st.button("🔍 Find Shared Treatments", type="primary", key="shared_treat_btn"):
            treatment_links = query("""
                MATCH (d1:Disease)-[:TREATED_BY]->(t:Treatment)<-[:TREATED_BY]-(d2:Disease)
                WHERE d1 <> d2 AND id(d1) < id(d2)
                WITH t, collect(DISTINCT d1.name) + collect(DISTINCT d2.name) as diseases
                WITH t, apoc.coll.toSet(diseases) as unique_diseases
                WHERE size(unique_diseases) >= 2
                RETURN t.name as treatment, unique_diseases as diseases, size(unique_diseases) as disease_count
                ORDER BY disease_count DESC
                LIMIT 15
            """)

            if treatment_links:
                st.success(f"**Found {len(treatment_links)} cross-disease treatments!**")

                for t in treatment_links:
                    disease_list = ', '.join(t['diseases'][:4])
                    st.markdown(f"💊 **{t['treatment']}** — used in {t['disease_count']} diseases")
                    st.caption(f"Diseases: {disease_list}")
                    st.markdown("---")
            else:
                # Fallback query without APOC
                treatment_links = query("""
                    MATCH (d:Disease)-[r:TREATED_BY]->(t:Treatment)
                    WITH t, collect(d.name) as diseases, count(d) as disease_count
                    WHERE disease_count >= 2
                    RETURN t.name as treatment, diseases, disease_count
                    ORDER BY disease_count DESC
                    LIMIT 15
                """)

                if treatment_links:
                    st.success(f"**Found {len(treatment_links)} cross-disease treatments!**")
                    for t in treatment_links:
                        disease_list = ', '.join(t['diseases'][:4])
                        st.markdown(f"💊 **{t['treatment']}** — used in {t['disease_count']} diseases")
                        st.caption(f"Diseases: {disease_list}")
                        st.markdown("---")
                else:
                    st.info("No cross-disease treatments found.")


# ============== TAB 6: TREATMENT PATHS ==============

with tab6:
    st.header("💊 Treatment Path Finder")
    st.markdown("""
    **Find treatments through mechanisms.** Select a target (protein, mechanism, or disease)
    and discover treatment options.

    *Like "Reading Path" for research papers — navigate the knowledge graph to find connections.*
    """)

    col1, col2 = st.columns(2)

    proteins = get_all_proteins()
    mechanisms = get_all_mechanisms()
    diseases = get_all_diseases()

    with col1:
        target_type = st.radio("Search by:", ["Disease", "Mechanism", "Protein"], key="target_type")

    with col2:
        if target_type == "Protein":
            targets = proteins
            selected_target = st.selectbox("Select protein:", [p['name'] for p in targets[:20]])
        elif target_type == "Mechanism":
            targets = mechanisms
            selected_target = st.selectbox("Select mechanism:", [m['name'] for m in targets[:20]])
        else:
            selected_target = st.selectbox("Select disease:", [d['name'] for d in diseases])

    if st.button("💊 Find Treatment Paths", type="primary", key="treatment_btn"):
        st.markdown(f"### Treatment paths for: **{selected_target}**")

        if target_type == "Disease":
            # Direct treatments
            direct = query("""
                MATCH (d:Disease {name: $target})-[r:TREATED_BY]->(t:Treatment)
                RETURN t.name as treatment, r.paper_count as papers
                ORDER BY r.paper_count DESC
                LIMIT 10
            """, {"target": selected_target})

            if direct:
                st.markdown("#### 💊 Direct Treatments")
                for t in direct:
                    st.markdown(f"- **{t['treatment']}** ({t['papers']} papers)")

            # Via mechanism
            via_mechanism = query("""
                MATCH (d:Disease {name: $target})-[:INVOLVES_MECHANISM]->(m:Mechanism)
                      <-[:MENTIONS_MECHANISM]-(p:Paper)-[:MENTIONS_TREATMENT]->(t:Treatment)
                WITH t, m, count(p) as papers
                RETURN t.name as treatment, m.name as mechanism, papers
                ORDER BY papers DESC
                LIMIT 10
            """, {"target": selected_target})

            if via_mechanism:
                st.markdown("#### ⚙️ Treatments via Shared Mechanisms")
                for t in via_mechanism:
                    st.markdown(f"- **{t['treatment']}** ← targets *{t['mechanism']}* ({t['papers']} papers)")

            # Via protein
            via_protein = query("""
                MATCH (d:Disease {name: $target})-[:INVOLVES_PROTEIN]->(pr:Protein)
                      <-[:MENTIONS_PROTEIN]-(p:Paper)-[:MENTIONS_TREATMENT]->(t:Treatment)
                WITH t, pr, count(p) as papers
                RETURN t.name as treatment, pr.name as protein, papers
                ORDER BY papers DESC
                LIMIT 10
            """, {"target": selected_target})

            if via_protein:
                st.markdown("#### 🧬 Treatments via Protein Targets")
                for t in via_protein:
                    st.markdown(f"- **{t['treatment']}** ← targets *{t['protein']}* ({t['papers']} papers)")

        elif target_type == "Mechanism":
            results = query("""
                MATCH (m:Mechanism {name: $target})<-[:MENTIONS_MECHANISM]-(p:Paper)
                      -[:MENTIONS_TREATMENT]->(t:Treatment)
                WITH t, count(p) as papers, collect(DISTINCT p.title)[0..2] as sample_papers
                MATCH (t)<-[:MENTIONS_TREATMENT]-(p2:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                RETURN t.name as treatment, papers,
                       collect(DISTINCT d.name) as diseases,
                       sample_papers
                ORDER BY papers DESC
                LIMIT 15
            """, {"target": selected_target})

            if results:
                st.success(f"**Found {len(results)} treatments** targeting {selected_target}!")
                for item in results:
                    st.markdown(f"### 💊 {item['treatment']}")
                    st.caption(f"Mentioned in {item['papers']} papers about {selected_target}")
                    if item['diseases']:
                        st.markdown(f"🏥 **Related diseases:** {', '.join(item['diseases'][:5])}")
                    st.markdown("---")

        else:  # Protein
            results = query("""
                MATCH (pr:Protein {name: $target})<-[:MENTIONS_PROTEIN]-(p:Paper)
                      -[:MENTIONS_TREATMENT]->(t:Treatment)
                WITH t, count(p) as papers
                MATCH (t)<-[:MENTIONS_TREATMENT]-(p2:Paper)-[:MENTIONS_DISEASE]->(d:Disease)
                RETURN t.name as treatment, papers,
                       collect(DISTINCT d.name) as diseases
                ORDER BY papers DESC
                LIMIT 15
            """, {"target": selected_target})

            if results:
                st.success(f"**Found {len(results)} treatments** targeting {selected_target}!")
                for item in results:
                    st.markdown(f"### 💊 {item['treatment']}")
                    st.caption(f"Mentioned in {item['papers']} papers about {selected_target}")
                    if item['diseases']:
                        st.markdown(f"🏥 **Related diseases:** {', '.join(item['diseases'][:5])}")
                    st.markdown("---")

        st.info("""
        💡 **Graph Path Analysis:**

        We traversed the graph: Target → Papers → Treatments → Diseases

        This finds treatments that research connects to your target,
        even if the connection is indirect (via mechanisms or proteins).
        """)


# ============== TAB 7: GRAPH EXPLORER ==============

with tab7:
    st.header("🕸️ Graph Explorer")
    st.markdown("""
    **Visualize the knowledge graph.** See how diseases, symptoms, proteins, and treatments connect.
    """)

    # Filters
    col1, col2, col3 = st.columns(3)

    diseases = get_all_diseases()

    with col1:
        disease_options = ["All Diseases"] + [d['name'] for d in diseases]
        selected_disease = st.selectbox("Disease:", disease_options, key="graph_disease")

    with col2:
        node_types = st.multiselect(
            "Show node types:",
            ["Symptom", "ReportedSymptom", "Protein", "Mechanism", "Treatment"],
            default=["Symptom", "Treatment"],
            key="graph_types"
        )

    with col3:
        max_nodes = st.slider("Max nodes per type:", 5, 20, 10, key="graph_limit")

    if st.button("🔍 Explore Graph", type="primary", key="graph_btn"):
        nodes = []
        edges = []
        seen_nodes = set()

        # Get disease node
        if selected_disease != "All Diseases":
            disease_id = f"disease_{selected_disease}"
            nodes.append({
                'id': disease_id,
                'label': selected_disease,
                'title': f"Disease: {selected_disease}",
                'type': 'Disease',
                'size': 35
            })
            seen_nodes.add(disease_id)

            # Get connected entities
            for node_type in node_types:
                if node_type == "Symptom":
                    data = query("""
                        MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom)
                        RETURN s.name as name, r.paper_count as count
                        ORDER BY r.paper_count DESC
                        LIMIT $limit
                    """, {"disease": selected_disease, "limit": max_nodes})
                    rel_type = "HAS_SYMPTOM"

                elif node_type == "ReportedSymptom":
                    data = query("""
                        MATCH (d:Disease {name: $disease})-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
                        RETURN rs.name as name, r.report_count as count
                        ORDER BY r.report_count DESC
                        LIMIT $limit
                    """, {"disease": selected_disease, "limit": max_nodes})
                    rel_type = "HAS_REPORTED_SYMPTOM"

                elif node_type == "Protein":
                    data = query("""
                        MATCH (d:Disease {name: $disease})-[r:INVOLVES_PROTEIN]->(pr:Protein)
                        RETURN pr.name as name, r.paper_count as count
                        ORDER BY r.paper_count DESC
                        LIMIT $limit
                    """, {"disease": selected_disease, "limit": max_nodes})
                    rel_type = "INVOLVES_PROTEIN"

                elif node_type == "Mechanism":
                    data = query("""
                        MATCH (d:Disease {name: $disease})-[r:INVOLVES_MECHANISM]->(m:Mechanism)
                        RETURN m.name as name, r.paper_count as count
                        ORDER BY r.paper_count DESC
                        LIMIT $limit
                    """, {"disease": selected_disease, "limit": max_nodes})
                    rel_type = "INVOLVES_MECHANISM"

                elif node_type == "Treatment":
                    data = query("""
                        MATCH (d:Disease {name: $disease})-[r:TREATED_BY]->(t:Treatment)
                        RETURN t.name as name, r.paper_count as count
                        ORDER BY r.paper_count DESC
                        LIMIT $limit
                    """, {"disease": selected_disease, "limit": max_nodes})
                    rel_type = "TREATED_BY"

                else:
                    data = []

                for item in data:
                    node_id = f"{node_type.lower()}_{item['name']}"
                    if node_id not in seen_nodes:
                        nodes.append({
                            'id': node_id,
                            'label': item['name'][:20],
                            'title': f"{node_type}: {item['name']}\n{item['count']} {'reports' if node_type == 'ReportedSymptom' else 'papers'}",
                            'type': node_type,
                            'size': 15 + min(item['count'] or 0, 20)
                        })
                        seen_nodes.add(node_id)
                    edges.append({
                        'from': disease_id,
                        'to': node_id,
                        'label': rel_type
                    })

        else:
            # First get ALL diseases
            all_diseases = query("""
                MATCH (d:Disease)
                OPTIONAL MATCH (d)<-[:MENTIONS_DISEASE]-(p:Paper)
                RETURN d.name as name, count(p) as paper_count
                ORDER BY paper_count DESC
            """)

            # Add all disease nodes
            for d in all_diseases:
                disease_id = f"disease_{d['name']}"
                if disease_id not in seen_nodes:
                    nodes.append({
                        'id': disease_id,
                        'label': d['name'][:20],
                        'title': f"Disease: {d['name']}\n{d['paper_count']} papers",
                        'type': 'Disease',
                        'size': 20 + min(d['paper_count'] or 0, 15)
                    })
                    seen_nodes.add(disease_id)

            # Then get shared symptom relationships
            disease_data = query("""
                MATCH (d1:Disease)-[r:SHARES_SYMPTOMS_WITH]->(d2:Disease)
                RETURN d1.name as from_disease, d2.name as to_disease, r.shared_count as count
                ORDER BY r.shared_count DESC
                LIMIT 50
            """)

            for item in disease_data:
                edges.append({
                    'from': f"disease_{item['from_disease']}",
                    'to': f"disease_{item['to_disease']}",
                    'label': f"{item['count']} shared"
                })

        if nodes:
            st.markdown(f"### Graph: {len(nodes)} nodes, {len(edges)} relationships")

            # Legend
            st.markdown("""
            **Legend:**
            🟢 Disease | 🟡 Symptom | 🟠 Patient Symptom | 🟣 Protein | 🟢 Treatment | 🟡 Mechanism
            """)

            html = create_graph_visualization(nodes, edges, height="550px")
            components.html(html, height=600, scrolling=True)

            # Stats
            st.markdown("---")
            type_counts = {}
            for n in nodes:
                type_counts[n['type']] = type_counts.get(n['type'], 0) + 1

            cols = st.columns(len(type_counts))
            for i, (ntype, count) in enumerate(type_counts.items()):
                with cols[i]:
                    st.metric(ntype, count)

            st.info("""
            💡 **Visualization colors:**
            - Cyan = Disease
            - Yellow = Research Symptom
            - Orange = Patient-Reported Symptom
            - Purple = Protein
            - Green = Treatment

            **Drag nodes** to explore. **Zoom** with scroll. **Click** for details.
            """)
        else:
            st.warning("No data to visualize. Select a disease or check that data is loaded.")


# ============== FOOTER ==============

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<strong>Neurology Navigator</strong> — Graph RAG for medical research & patient experiences<br>
<small>The only demo that compares research literature with real patient reports</small>
</div>
""", unsafe_allow_html=True)
