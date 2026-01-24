"""
Script to extract entities and relationships from arXiv papers using LLMs.

Usage:
    python scripts/02_extract_entities.py --input data/raw --output data/processed
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import re

from openai import OpenAI


class EntityExtractor:
    """Extract entities and relationships from papers using LLM."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def extract_from_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from a single paper."""
        try:
            prompt = self._create_prompt(paper)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000
            )
            
            result = self._parse_response(response.choices[0].message.content)
            
            # Add paper metadata
            result["paper_id"] = paper.get("arxiv_id")
            result["paper_title"] = paper.get("title")
            result["published_date"] = paper.get("published_date")
            result["primary_category"] = paper.get("primary_category")
            result["abstract"] = paper.get("abstract", "")
            
            # Merge authors
            result["authors"] = self._merge_authors(
                paper.get("authors", []),
                result.get("authors", [])
            )
            
            return result
            
        except Exception as e:
            print(f"Error extracting from {paper.get('arxiv_id')}: {e}")
            return self._create_fallback(paper)
    
    def _create_prompt(self, paper: Dict[str, Any]) -> str:
        """Create extraction prompt."""
        return f"""Extract structured information from this research paper. Return ONLY valid JSON.

Title: {paper.get('title', '')}

Abstract: {paper.get('abstract', '')[:2000]}

Extract:
1. authors: List of {{name, inferred_institution}} 
2. institutions: List of {{name, country}}
3. concepts: List of {{name, field}} - main research topics
4. methods: List of {{name, category}} - techniques/algorithms used

Return format (JSON only, no markdown):
{{
  "authors": [{{"name": "Author Name", "inferred_institution": "University"}}],
  "institutions": [{{"name": "MIT", "country": "USA"}}],
  "concepts": [{{"name": "deep learning", "field": "AI"}}],
  "methods": [{{"name": "transformer", "category": "Architecture"}}]
}}

Only extract explicitly mentioned information. Return empty lists if none found."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return {
                "authors": [],
                "institutions": [],
                "concepts": [],
                "methods": []
            }
    
    def _merge_authors(self, original: List, extracted: List) -> List:
        """Merge original and extracted author info."""
        merged = []
        extracted_dict = {a.get("name"): a for a in extracted if a.get("name")}
        
        for orig in original:
            name = orig.get("name")
            if name in extracted_dict:
                merged.append({**orig, **extracted_dict[name]})
            else:
                merged.append(orig)
        
        return merged
    
    def _create_fallback(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback when extraction fails."""
        return {
            "paper_id": paper.get("arxiv_id"),
            "paper_title": paper.get("title"),
            "published_date": paper.get("published_date"),
            "primary_category": paper.get("primary_category"),
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", []),
            "institutions": [],
            "concepts": [{"name": paper.get("primary_category"), "field": "CS"}],
            "methods": []
        }


def load_papers(input_dir: Path) -> List[Dict[str, Any]]:
    """Load papers from input directory."""
    papers = []
    complete_file = input_dir / "papers_complete.jsonl"
    
    if complete_file.exists():
        print(f"Loading from {complete_file}")
        with open(complete_file, "r") as f:
            for line in f:
                papers.append(json.loads(line))
    else:
        print(f"Error: {complete_file} not found")
        return []
    
    print(f"Loaded {len(papers)} papers")
    return papers


def save_extractions(extractions: List[Dict[str, Any]], output_dir: Path):
    """Save extracted entities."""
    output_dir.mkdir(parents=True, exist_ok=True)