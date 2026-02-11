"""
Step 2: Enrich patient data
- Group similar ReportedSymptoms into themes
- Extract actual patient quotes from post titles
- Create PatientExperience nodes with rich context

This will give us better patient insights than the current generic labels.
"""
import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "neurograph2025"))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_patient_data_for_disease(disease_name: str) -> list:
    """Get all patient data for a specific disease."""
    with driver.session() as session:
        # Get ReportedSymptoms with their post context
        result = session.run("""
            MATCH (d:Disease)-[r:HAS_REPORTED_SYMPTOM]->(rs:ReportedSymptom)
            WHERE d.name = $disease
            OPTIONAL MATCH (post:RedditPost)-[:REPORTS_SYMPTOM]->(rs)
            RETURN rs.name as symptom, 
                   r.report_count as count,
                   collect(DISTINCT post.title) as post_titles,
                   collect(DISTINCT post.subreddit) as subreddits
            ORDER BY r.report_count DESC
        """, disease=disease_name)
        return [dict(r) for r in result]


def group_patient_experiences(symptoms_data: list, disease_name: str) -> list:
    """Group similar patient experiences into themes."""
    
    # Prepare symptoms for AI analysis
    symptoms_for_ai = []
    for s in symptoms_data[:30]:  # Top 30 to avoid token limits
        symptoms_for_ai.append({
            'symptom': s['symptom'],
            'count': s['count'],
            'sample_posts': s['post_titles'][:3] if s['post_titles'] else []
        })
    
    prompt = f"""You are analyzing patient-reported experiences for {disease_name}.

Here are specific patient experiences with their frequency and sample post titles:

{json.dumps(symptoms_for_ai, indent=2)}

Group these into 5-7 meaningful THEMES that would be valuable for clinicians to understand. Each theme should:
1. Represent a category of patient experience
2. Include the most relevant specific symptoms from the list
3. Have an actionable insight for clinicians

Examples of good themes:
- "Behavioral Changes & Agitation" (violence, property damage, personality shifts)
- "Daily Living Challenges" (specific tasks patients struggle with)
- "Early Warning Signs" (subtle changes patients noticed first)
- "Communication Difficulties" (specific language/speech issues)
- "Sleep & Routine Disruption" (specific patterns)

Return JSON:
{{
  "themes": [
    {{
      "name": "Theme Name",
      "description": "What this represents for clinicians",
      "symptoms": ["symptom1", "symptom2", ...],
      "clinical_insight": "Why this matters for diagnosis/care",
      "sample_quotes": ["relevant post title 1", "post title 2"]
    }}
  ]
}}

Focus on themes that reveal specific patient experiences, not generic emotions."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)['themes']
    except Exception as e:
        print(f"AI grouping failed: {e}")
        return []


def create_patient_experience_nodes(disease_name: str, themes: list):
    """Create PatientExperience nodes in the database."""
    with driver.session() as session:
        # First, clear existing PatientExperience nodes for this disease
        session.run("""
            MATCH (d:Disease {name: $disease})-[:HAS_PATIENT_EXPERIENCE]->(pe:PatientExperience)
            DETACH DELETE pe
        """, disease=disease_name)
        
        # Create new PatientExperience nodes
        for i, theme in enumerate(themes):
            session.run("""
                MATCH (d:Disease {name: $disease})
                CREATE (pe:PatientExperience {
                    name: $name,
                    description: $description,
                    clinical_insight: $insight,
                    sample_quotes: $quotes,
                    symptoms_count: $count,
                    theme_id: $theme_id
                })
                CREATE (d)-[:HAS_PATIENT_EXPERIENCE]->(pe)
            """, 
            disease=disease_name,
            name=theme['name'],
            description=theme['description'],
            insight=theme['clinical_insight'],
            quotes=theme['sample_quotes'],
            count=len(theme['symptoms']),
            theme_id=i
            )
        
        print(f"Created {len(themes)} PatientExperience nodes for {disease_name}")


def process_disease(disease_name: str):
    """Process one disease completely."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {disease_name}")
    print(f"{'='*60}")
    
    # Get patient data
    print("Getting patient-reported symptoms...")
    symptoms_data = get_patient_data_for_disease(disease_name)
    print(f"Found {len(symptoms_data)} patient-reported symptoms")
    
    if len(symptoms_data) < 3:
        print("Not enough patient data to analyze")
        return
    
    # Show top symptoms
    print("\nTop patient experiences:")
    for s in symptoms_data[:10]:
        print(f"  {s['count']:2d} mentions: {s['symptom'][:80]}")
    
    # Group into themes
    print("\nGrouping into themes...")
    themes = group_patient_experiences(symptoms_data, disease_name)
    
    if not themes:
        print("Failed to group themes")
        return
    
    print(f"\nIdentified {len(themes)} themes:")
    for theme in themes:
        print(f"  • {theme['name']}: {len(theme['symptoms'])} symptoms")
        print(f"    {theme['clinical_insight'][:100]}...")
    
    # Create nodes in database
    print("\nCreating PatientExperience nodes...")
    create_patient_experience_nodes(disease_name, themes)
    
    print(f"✅ {disease_name} complete!")


def main():
    print("=" * 60)
    print("STEP 2: ENRICHING PATIENT DATA")
    print("=" * 60)
    
    # Process key diseases
    diseases_to_process = [
        "Alzheimer'S Diseases",
        "Parkinson'S Diseases", 
        "Amyotrophic Lateral Sclerosis",
        "Creutzfeldt-Jakob Disease",
        "Huntington'S Diseases"
    ]
    
    for disease in diseases_to_process:
        try:
            process_disease(disease)
        except Exception as e:
            print(f"ERROR processing {disease}: {e}")
    
    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE!")
    print("=" * 60)
    
    # Verify what we created
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Disease)-[:HAS_PATIENT_EXPERIENCE]->(pe:PatientExperience)
            RETURN d.name as disease, count(pe) as theme_count
            ORDER BY theme_count DESC
        """)
        
        print("\nPatientExperience nodes created:")
        total = 0
        for r in result:
            print(f"  {r['disease']}: {r['theme_count']} themes")
            total += r['theme_count']
        print(f"\nTotal: {total} patient experience themes")
    
    print("\n✅ Step 2 complete!")
    print("\nNext: Update the app to use the new structured data")


if __name__ == "__main__":
    main()