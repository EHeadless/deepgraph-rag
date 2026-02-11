# Neurology Navigator

Graph RAG combining research literature with patient experiences. Find gaps between what scientists study and what patients report.

## Run

```bash
streamlit run examples/neurology/app.py --server.port 8507
```

## The Value

| Query | Vector Search | Graph RAG |
|-------|---------------|-----------|
| "Alzheimer's symptoms" | Works | Works |
| "Symptoms patients report but research ignores" | Impossible | Works |
| "Diseases sharing mechanisms" | Impossible | Works |

**The unique capability:** Structural comparison across two data sources (PubMed + Reddit).

## Tabs

1. **How It Works** - Dual data source explanation
2. **Disease x Symptom** - Intersection query
3. **Research vs Patient** - Side-by-side comparison
4. **Research Gaps** - Patient symptoms missing from literature
5. **Cross-Disease** - Shared symptoms, mechanisms, treatments
6. **Treatment Paths** - Find treatments via mechanisms/proteins
7. **Graph Explorer** - Interactive visualization

## Graph Schema

```
Research Data (PubMed):
  Paper
    ├── MENTIONS_DISEASE ──→ Disease
    ├── MENTIONS_SYMPTOM ──→ Symptom
    ├── MENTIONS_PROTEIN ──→ Protein
    ├── MENTIONS_MECHANISM ──→ Mechanism
    └── MENTIONS_TREATMENT ──→ Treatment

Patient Data (Reddit):
  RedditPost
    ├── DISCUSSES ──→ Disease
    └── REPORTS_SYMPTOM ──→ ReportedSymptom

Aggregated:
  Disease
    ├── HAS_SYMPTOM ──→ Symptom (paper_count)
    └── HAS_REPORTED_SYMPTOM ──→ ReportedSymptom (report_count)
```

## Data

- 495 PubMed papers across 15 neurodegenerative diseases
- 4,000+ Reddit posts, 40,000+ comments
- 31,077 patient symptom extractions

## Target Diseases

Alzheimer's, Parkinson's, ALS, Multiple Sclerosis, Huntington's, Frontotemporal dementia, Lewy body dementia, Progressive supranuclear palsy, Corticobasal degeneration, Multiple system atrophy, Spinocerebellar ataxia, Creutzfeldt-Jakob disease, Primary lateral sclerosis, Spinal muscular atrophy, Friedreich's ataxia

## Data Pipeline

```bash
python examples/neurology/scripts/build_neurology_graph.py
```

## Honest Assessment

High differentiation. The dual-source comparison (research vs patient) is genuinely unique. Research gap analysis is impossible without graph structure connecting both sources. Niche but defensible.
