"""
Extract structured medical entities from PubMed abstracts using GPT-4
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

EXTRACTION_PROMPT = """Analyze this medical research abstract and extract structured information.

Title: {title}
Abstract: {abstract}

Extract the following (leave empty array [] if not mentioned):

1. diseases: List of neurodegenerative diseases mentioned
2. symptoms: Clinical symptoms or signs described
3. brain_regions: Brain areas or structures mentioned
4. proteins: Proteins, biomarkers, or molecular markers
5. genes: Genes or genetic factors
6. mechanisms: Biological mechanisms (e.g., "protein aggregation", "neuroinflammation", "oxidative stress")
7. treatments: Drugs, therapies, or interventions mentioned

Return ONLY valid JSON in this exact format:
{{
    "diseases": ["disease1", "disease2"],
    "symptoms": ["symptom1", "symptom2"],
    "brain_regions": ["region1", "region2"],
    "proteins": ["protein1", "protein2"],
    "genes": ["gene1", "gene2"],
    "mechanisms": ["mechanism1", "mechanism2"],
    "treatments": ["treatment1", "treatment2"]
}}
"""

def extract_entities(paper: dict) -> dict:
    """Extract entities from a single paper."""
    prompt = EXTRACTION_PROMPT.format(
        title=paper['title'],
        abstract=paper['abstract'][:2000]  # Limit length
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a medical research analyst. Extract structured information from research abstracts. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        entities = json.loads(content)
        return entities
        
    except Exception as e:
        print(f"Error extracting from {paper['pmid']}: {e}")
        return {
            "diseases": [],
            "symptoms": [],
            "brain_regions": [],
            "proteins": [],
            "genes": [],
            "mechanisms": [],
            "treatments": []
        }

def main():
    # Load papers
    with open("data/pubmed_papers.json", "r") as f:
        papers = json.load(f)
    
    print(f"Extracting entities from {len(papers)} papers...")
    print("Estimated cost: ~$8-12")
    print("Estimated time: ~20-30 minutes")
    print("=" * 50)
    
    input("Press Enter to continue (Ctrl+C to cancel)...")
    
    extracted = []
    
    for paper in tqdm(papers):
        entities = extract_entities(paper)
        
        extracted.append({
            "pmid": paper["pmid"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": paper["authors"],
            "year": paper.get("year", ""),
            "disease_query": paper.get("disease_query", ""),
            "entities": entities
        })
        
        # Rate limiting
        time.sleep(0.2)
        
        # Save progress every 50 papers
        if len(extracted) % 50 == 0:
            with open("data/extracted_entities.json", "w") as f:
                json.dump(extracted, f, indent=2)
            print(f"\n  Saved progress: {len(extracted)} papers")
    
    # Final save
    with open("data/extracted_entities.json", "w") as f:
        json.dump(extracted, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Extraction complete: {len(extracted)} papers")
    print("Saved to: data/extracted_entities.json")

if __name__ == "__main__":
    main()
