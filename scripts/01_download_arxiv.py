"""
Download papers from arXiv.
"""

import argparse
import arxiv
import json
from pathlib import Path
from tqdm import tqdm
import time


def main():
    parser = argparse.ArgumentParser(description="Download arXiv papers")
    parser.add_argument("--categories", type=str, default="cs.AI")
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/raw")
    
    args = parser.parse_args()
    
    # Parse categories
    categories = [c.strip() for c in args.categories.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {args.max_papers} papers from {categories}")
    
    # Build query
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    # Search
    client = arxiv.Client()
    search = arxiv.Search(
        query=category_query,
        max_results=args.max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for paper in tqdm(client.results(search), total=args.max_papers, desc="Downloading"):
        papers.append({
            "arxiv_id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "abstract": paper.summary,
            "authors": [{"name": a.name} for a in paper.authors],
            "published_date": paper.published.isoformat(),
            "primary_category": paper.primary_category,
            "categories": paper.categories,
        })
        time.sleep(0.1)
    
    # Save
    output_file = output_dir / "papers_complete.jsonl"
    with open(output_file, "w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    
    print(f"✅ Saved {len(papers)} papers to {output_file}")


if __name__ == "__main__":
    main()