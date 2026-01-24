import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import re

load_dotenv()

# Load papers
papers = []
with open('data/raw/papers_complete.jsonl') as f:
    for line in f:
        papers.append(json.loads(line))

papers = papers[:500]
print(f"Processing {len(papers)} papers...")

# Setup
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
extractions = []

# Process each
for paper in tqdm(papers):
    prompt = f"""Extract from this paper. Return JSON only.

Title: {paper['title']}
Abstract: {paper['abstract'][:1500]}

{{
  "concepts": [{{"name": "concept", "field": "AI"}}],
  "methods": [{{"name": "method", "category": "Architecture"}}],
  "institutions": [{{"name": "MIT", "country": "USA"}}]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(json_match.group(0)) if json_match else {}
        
    except:
        result = {"concepts": [], "methods": [], "institutions": []}
    
    result.update({
        "paper_id": paper['arxiv_id'],
        "paper_title": paper['title'],
        "authors": paper['authors'],
        "published_date": paper['published_date'],
        "primary_category": paper['primary_category']
    })
    
    extractions.append(result)

# Save
Path('data/processed').mkdir(exist_ok=True)
with open('data/processed/extracted_entities.jsonl', 'w') as f:
    for e in extractions:
        f.write(json.dumps(e) + '\n')

print(f"✅ Saved {len(extractions)} extractions!")