"""
Simple converter for arXiv papers - extracts entities from metadata (no LLM needed).
"""

import json
from pathlib import Path
from tqdm import tqdm

def convert_paper(paper):
    """Convert paper to extracted format using existing metadata."""
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

def main():
    input_file = Path("data/raw/papers_complete.jsonl")
    output_file = Path("data/processed/extracted_entities.jsonl")

    print(f"Converting {input_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    papers = []
    with open(input_file, "r") as f:
        for line in f:
            papers.append(json.loads(line))

    print(f"Loaded {len(papers)} papers")

    with open(output_file, "w") as f:
        for paper in tqdm(papers, desc="Converting"):
            extracted = convert_paper(paper)
            f.write(json.dumps(extracted) + "\n")

    print(f"✅ Saved {len(papers)} extractions to {output_file}")

if __name__ == "__main__":
    main()
