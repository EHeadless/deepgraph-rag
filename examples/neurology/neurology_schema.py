"""
Neurology Knowledge Graph Schema

Knowledge Graph Structure:
    Paper (PubMed)
      ├── MENTIONS_DISEASE ──→ Disease
      ├── MENTIONS_SYMPTOM ──→ Symptom (clinical)
      ├── MENTIONS_REGION ──→ BrainRegion
      ├── MENTIONS_PROTEIN ──→ Protein
      ├── MENTIONS_GENE ──→ Gene
      ├── MENTIONS_MECHANISM ──→ Mechanism
      ├── MENTIONS_TREATMENT ──→ Treatment
      └── AUTHORED_BY ──→ Author

    RedditPost (Patient data)
      ├── DISCUSSES ──→ Disease
      ├── REPORTS_SYMPTOM ──→ ReportedSymptom
      └── REPORTS_IMPACT ──→ LifeImpact

    Disease (aggregated)
      ├── HAS_SYMPTOM ──→ Symptom (with paper_count)
      ├── AFFECTS_REGION ──→ BrainRegion
      ├── INVOLVES_PROTEIN ──→ Protein
      ├── LINKED_TO_GENE ──→ Gene
      ├── INVOLVES_MECHANISM ──→ Mechanism
      ├── TREATED_BY ──→ Treatment
      ├── HAS_REPORTED_SYMPTOM ──→ ReportedSymptom (with report_count)
      ├── HAS_LIFE_IMPACT ──→ LifeImpact
      ├── HAS_PATIENT_EXPERIENCE ──→ PatientExperience
      └── SHARES_SYMPTOMS_WITH ──→ Disease

This enables queries like:
    - "What symptoms do Alzheimer's and Parkinson's share?"
    - "What treatments target tau protein?"
    - "What do patients report vs what research says?"
    - "What brain regions are affected by ALS?"
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepgraph.core.schema import GraphSchema, NodeSchema, EdgeSchema


# ============== NODE SCHEMAS ==============

PAPER_NODE = NodeSchema(
    label="Paper",
    id_field="pmid",
    properties={
        "pmid": str,
        "title": str,
        "abstract": str,
        "year": str,
        "disease_query": str,  # Original search query
        "embedding": list,
    },
    indexes=["title", "year"],
    constraints=["pmid"],
    vector_config={
        "field": "embedding",
        "dimensions": 1536,
        "similarity": "cosine"
    }
)

DISEASE_NODE = NodeSchema(
    label="Disease",
    id_field="name",
    properties={
        "name": str,
        "category": str,  # e.g., "neurodegenerative"
        "embedding": list,
    },
    indexes=["category"],
    constraints=["name"]
)

SYMPTOM_NODE = NodeSchema(
    label="Symptom",
    id_field="name",
    properties={
        "name": str,
        "category": str,  # clinical, pathology, imaging, biomarker, mechanism
    },
    indexes=["category"],
    constraints=["name"]
)

BRAIN_REGION_NODE = NodeSchema(
    label="BrainRegion",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

PROTEIN_NODE = NodeSchema(
    label="Protein",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

GENE_NODE = NodeSchema(
    label="Gene",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

MECHANISM_NODE = NodeSchema(
    label="Mechanism",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

TREATMENT_NODE = NodeSchema(
    label="Treatment",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

AUTHOR_NODE = NodeSchema(
    label="Author",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

# Patient data nodes
REDDIT_POST_NODE = NodeSchema(
    label="RedditPost",
    id_field="post_id",
    properties={
        "post_id": str,
        "title": str,
        "url": str,
        "subreddit": str,
        "disease": str,
        "source": str,  # "post" or "comment"
    },
    indexes=["disease", "subreddit"],
    constraints=["post_id"]
)

REPORTED_SYMPTOM_NODE = NodeSchema(
    label="ReportedSymptom",
    id_field="name",
    properties={
        "name": str,
        "category": str,  # physical, cognitive, behavioral, psychiatric
    },
    indexes=["category"],
    constraints=["name"]
)

LIFE_IMPACT_NODE = NodeSchema(
    label="LifeImpact",
    id_field="name",
    properties={
        "name": str,
    },
    indexes=[],
    constraints=["name"]
)

PATIENT_EXPERIENCE_NODE = NodeSchema(
    label="PatientExperience",
    id_field="theme_id",
    properties={
        "theme_id": str,
        "name": str,
        "description": str,
        "clinical_insight": str,
        "sample_quotes": list,
        "symptoms_count": int,
    },
    indexes=["name"],
    constraints=["theme_id"]
)


# ============== EDGE SCHEMAS ==============

# Paper relationships
AUTHORED_BY = EdgeSchema(
    type="AUTHORED_BY",
    from_label="Paper",
    to_label="Author",
    properties={}
)

MENTIONS_DISEASE = EdgeSchema(
    type="MENTIONS_DISEASE",
    from_label="Paper",
    to_label="Disease",
    properties={}
)

MENTIONS_SYMPTOM = EdgeSchema(
    type="MENTIONS_SYMPTOM",
    from_label="Paper",
    to_label="Symptom",
    properties={}
)

MENTIONS_REGION = EdgeSchema(
    type="MENTIONS_REGION",
    from_label="Paper",
    to_label="BrainRegion",
    properties={}
)

MENTIONS_PROTEIN = EdgeSchema(
    type="MENTIONS_PROTEIN",
    from_label="Paper",
    to_label="Protein",
    properties={}
)

MENTIONS_GENE = EdgeSchema(
    type="MENTIONS_GENE",
    from_label="Paper",
    to_label="Gene",
    properties={}
)

MENTIONS_MECHANISM = EdgeSchema(
    type="MENTIONS_MECHANISM",
    from_label="Paper",
    to_label="Mechanism",
    properties={}
)

MENTIONS_TREATMENT = EdgeSchema(
    type="MENTIONS_TREATMENT",
    from_label="Paper",
    to_label="Treatment",
    properties={}
)

# Disease aggregated relationships
HAS_SYMPTOM = EdgeSchema(
    type="HAS_SYMPTOM",
    from_label="Disease",
    to_label="Symptom",
    properties={
        "paper_count": int,
    }
)

AFFECTS_REGION = EdgeSchema(
    type="AFFECTS_REGION",
    from_label="Disease",
    to_label="BrainRegion",
    properties={
        "paper_count": int,
    }
)

INVOLVES_PROTEIN = EdgeSchema(
    type="INVOLVES_PROTEIN",
    from_label="Disease",
    to_label="Protein",
    properties={
        "paper_count": int,
    }
)

LINKED_TO_GENE = EdgeSchema(
    type="LINKED_TO_GENE",
    from_label="Disease",
    to_label="Gene",
    properties={
        "paper_count": int,
    }
)

INVOLVES_MECHANISM = EdgeSchema(
    type="INVOLVES_MECHANISM",
    from_label="Disease",
    to_label="Mechanism",
    properties={
        "paper_count": int,
    }
)

TREATED_BY = EdgeSchema(
    type="TREATED_BY",
    from_label="Disease",
    to_label="Treatment",
    properties={
        "paper_count": int,
    }
)

SHARES_SYMPTOMS_WITH = EdgeSchema(
    type="SHARES_SYMPTOMS_WITH",
    from_label="Disease",
    to_label="Disease",
    properties={
        "shared_count": int,
    }
)

# Reddit/Patient relationships
DISCUSSES = EdgeSchema(
    type="DISCUSSES",
    from_label="RedditPost",
    to_label="Disease",
    properties={}
)

REPORTS_SYMPTOM = EdgeSchema(
    type="REPORTS_SYMPTOM",
    from_label="RedditPost",
    to_label="ReportedSymptom",
    properties={}
)

REPORTS_IMPACT = EdgeSchema(
    type="REPORTS_IMPACT",
    from_label="RedditPost",
    to_label="LifeImpact",
    properties={}
)

HAS_REPORTED_SYMPTOM = EdgeSchema(
    type="HAS_REPORTED_SYMPTOM",
    from_label="Disease",
    to_label="ReportedSymptom",
    properties={
        "report_count": int,
    }
)

HAS_LIFE_IMPACT = EdgeSchema(
    type="HAS_LIFE_IMPACT",
    from_label="Disease",
    to_label="LifeImpact",
    properties={
        "report_count": int,
    }
)

HAS_PATIENT_EXPERIENCE = EdgeSchema(
    type="HAS_PATIENT_EXPERIENCE",
    from_label="Disease",
    to_label="PatientExperience",
    properties={}
)


# ============== ENTITY EXTRACTION PATTERNS ==============

ENTITY_TYPES = [
    "diseases",
    "symptoms",
    "brain_regions",
    "proteins",
    "genes",
    "mechanisms",
    "treatments"
]

SYMPTOM_CATEGORIES = [
    "physical_symptoms",
    "cognitive_symptoms",
    "behavioral_symptoms",
    "psychiatric_symptoms",
    "daily_impacts"
]

# Target diseases for PubMed search
TARGET_DISEASES = [
    "Alzheimer's disease",
    "Parkinson's disease",
    "Amyotrophic Lateral Sclerosis",
    "Multiple Sclerosis",
    "Huntington's disease",
    "Frontotemporal dementia",
    "Lewy body dementia",
    "Progressive supranuclear palsy",
    "Corticobasal degeneration",
    "Multiple system atrophy",
    "Spinocerebellar ataxia",
    "Creutzfeldt-Jakob disease",
    "Primary lateral sclerosis",
    "Spinal muscular atrophy",
    "Friedreich's ataxia",
]

# Reddit subreddits by disease
REDDIT_SUBREDDITS = {
    "Alzheimer's Disease": ["Alzheimers", "dementia", "CaregiverSupport"],
    "Parkinson's Disease": ["Parkinsons", "parkinsons"],
    "Multiple Sclerosis": ["MultipleSclerosis", "MS_WARRIORS"],
    "ALS": ["ALS", "MND"],
    "Huntington's Disease": ["huntingtonsdisease"],
    "Lewy Body Dementia": ["LewyBodyDementia", "dementia"],
    # ... more mappings
}


# ============== COMPLETE SCHEMA ==============

NEUROLOGY_SCHEMA = GraphSchema(
    name="neurology",
    nodes={
        "Paper": PAPER_NODE,
        "Disease": DISEASE_NODE,
        "Symptom": SYMPTOM_NODE,
        "BrainRegion": BRAIN_REGION_NODE,
        "Protein": PROTEIN_NODE,
        "Gene": GENE_NODE,
        "Mechanism": MECHANISM_NODE,
        "Treatment": TREATMENT_NODE,
        "Author": AUTHOR_NODE,
        "RedditPost": REDDIT_POST_NODE,
        "ReportedSymptom": REPORTED_SYMPTOM_NODE,
        "LifeImpact": LIFE_IMPACT_NODE,
        "PatientExperience": PATIENT_EXPERIENCE_NODE,
    },
    edges={
        "AUTHORED_BY": AUTHORED_BY,
        "MENTIONS_DISEASE": MENTIONS_DISEASE,
        "MENTIONS_SYMPTOM": MENTIONS_SYMPTOM,
        "MENTIONS_REGION": MENTIONS_REGION,
        "MENTIONS_PROTEIN": MENTIONS_PROTEIN,
        "MENTIONS_GENE": MENTIONS_GENE,
        "MENTIONS_MECHANISM": MENTIONS_MECHANISM,
        "MENTIONS_TREATMENT": MENTIONS_TREATMENT,
        "HAS_SYMPTOM": HAS_SYMPTOM,
        "AFFECTS_REGION": AFFECTS_REGION,
        "INVOLVES_PROTEIN": INVOLVES_PROTEIN,
        "LINKED_TO_GENE": LINKED_TO_GENE,
        "INVOLVES_MECHANISM": INVOLVES_MECHANISM,
        "TREATED_BY": TREATED_BY,
        "SHARES_SYMPTOMS_WITH": SHARES_SYMPTOMS_WITH,
        "DISCUSSES": DISCUSSES,
        "REPORTS_SYMPTOM": REPORTS_SYMPTOM,
        "REPORTS_IMPACT": REPORTS_IMPACT,
        "HAS_REPORTED_SYMPTOM": HAS_REPORTED_SYMPTOM,
        "HAS_LIFE_IMPACT": HAS_LIFE_IMPACT,
        "HAS_PATIENT_EXPERIENCE": HAS_PATIENT_EXPERIENCE,
    }
)


if __name__ == "__main__":
    print("Neurology Knowledge Graph Schema")
    print(f"  Nodes: {list(NEUROLOGY_SCHEMA.nodes.keys())}")
    print(f"  Edges: {list(NEUROLOGY_SCHEMA.edges.keys())}")
    print(f"\nTarget Diseases: {len(TARGET_DISEASES)}")
    print(f"Entity Types: {ENTITY_TYPES}")
    print(f"Symptom Categories: {SYMPTOM_CATEGORIES}")
