"""
Step 1: Classify clinical symptoms
- Real clinical symptoms (what a doctor observes/patient experiences)
- Pathological findings (what you see under a microscope / in imaging)
- Other/noise

Adds a 'category' property to each Symptom node.
"""
import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "neurograph2025"))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_all_symptoms():
    """Get all 470 symptoms."""
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Symptom)
            RETURN s.name as name
            ORDER BY s.name
        """)
        return [r['name'] for r in result]


def classify_batch(symptoms: list) -> dict:
    """Classify a batch of symptoms using AI."""
    prompt = f"""Classify each of these medical terms into exactly one category:

1. "clinical_symptom" — Something a patient experiences or a doctor can observe in a clinical exam.
   Examples: tremor, memory loss, difficulty walking, seizures, insomnia, confusion, muscle stiffness

2. "pathology_finding" — A microscopic, histological, or laboratory finding. Not something a patient feels.
   Examples: neurofibrillary tangles, amyloid plaques, white matter degeneration, neuronal loss, fibrillar accumulations

3. "imaging_finding" — Something seen on MRI/CT/PET scan, not directly experienced by patient.
   Examples: cortical atrophy, ventricular enlargement, signal hyperintensity

4. "biomarker" — A measurable biological indicator.
   Examples: elevated tau protein, CSF markers, genetic mutation

5. "mechanism" — A biological process or mechanism, not a symptom.
   Examples: oxidative stress, mitochondrial dysfunction, protein aggregation

Terms to classify:
{json.dumps(symptoms)}

Return ONLY a JSON object mapping each term to its category:
{{"term1": "clinical_symptom", "term2": "pathology_finding", ...}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content.strip()
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    return json.loads(content)


def update_symptoms_in_db(classifications: dict):
    """Update symptom nodes with their category."""
    with driver.session() as session:
        for name, category in classifications.items():
            session.run("""
                MATCH (s:Symptom {name: $name})
                SET s.category = $category
            """, name=name, category=category)


def main():
    print("=" * 60)
    print("STEP 1: CLASSIFYING CLINICAL SYMPTOMS")
    print("=" * 60)
    
    symptoms = get_all_symptoms()
    print(f"\nFound {len(symptoms)} symptom nodes")
    
    # Process in batches of 40
    batch_size = 40
    all_classifications = {}
    
    for i in range(0, len(symptoms), batch_size):
        batch = symptoms[i:i + batch_size]
        print(f"\nClassifying batch {i // batch_size + 1}/{(len(symptoms) - 1) // batch_size + 1} ({len(batch)} symptoms)...")
        
        try:
            classifications = classify_batch(batch)
            all_classifications.update(classifications)
            
            # Show progress
            cats = {}
            for v in classifications.values():
                cats[v] = cats.get(v, 0) + 1
            print(f"  Results: {cats}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            # Try one by one as fallback
            for symptom in batch:
                try:
                    c = classify_batch([symptom])
                    all_classifications.update(c)
                except:
                    all_classifications[symptom] = "unknown"
    
    # Summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    summary = {}
    for name, cat in all_classifications.items():
        summary[cat] = summary.get(cat, 0) + 1
    
    for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Show examples of each category
    print("\nExamples:")
    for cat in summary:
        examples = [n for n, c in all_classifications.items() if c == cat][:5]
        print(f"\n  {cat}:")
        for e in examples:
            print(f"    - {e}")
    
    # Update database
    print("\n\nUpdating database...")
    update_symptoms_in_db(all_classifications)
    print(f"Updated {len(all_classifications)} symptom nodes with category property")
    
    # Verify
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Symptom)
            RETURN s.category as category, count(*) as count
            ORDER BY count DESC
        """)
        print("\nVerification from database:")
        for r in result:
            print(f"  {r['category']}: {r['count']}")
    
    print("\n✅ Step 1 complete!")
    print("\nNext: Run step2_enrich_patient_data.py")


if __name__ == "__main__":
    main()