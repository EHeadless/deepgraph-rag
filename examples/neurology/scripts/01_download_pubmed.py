"""
Download papers from PubMed for neurodegenerative diseases
"""
import requests
import json
import time
import os
from tqdm import tqdm

# PubMed E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Diseases to search
DISEASES = [
    "Alzheimer's disease",
    "Parkinson's disease", 
    "Lewy body dementia",
    "Creutzfeldt-Jakob disease",
    "Amyotrophic lateral sclerosis",
    "Huntington's disease",
    "Multiple sclerosis",
    "Frontotemporal dementia",
    "Progressive supranuclear palsy",
    "Corticobasal degeneration",
    "Prion disease",
    "Motor neuron disease",
    "Spinocerebellar ataxia",
    "Multiple system atrophy",
    "Vascular dementia"
]

def search_pubmed(query: str, max_results: int = 50) -> list:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": f"{query}[Title/Abstract] AND neurodegeneration[MeSH]",
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    
    response = requests.get(ESEARCH_URL, params=params)
    data = response.json()
    
    return data.get("esearchresult", {}).get("idlist", [])

def fetch_abstracts(pmids: list) -> list:
    """Fetch abstracts for given PMIDs."""
    if not pmids:
        return []
    
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract"
    }
    
    response = requests.get(EFETCH_URL, params=params)
    
    # Parse XML response
    from xml.etree import ElementTree as ET
    root = ET.fromstring(response.content)
    
    papers = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.find(".//PMID").text
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Get authors
            authors = []
            for author in article.findall(".//Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            
            # Get publication year
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""
            
            if title and abstract:
                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors[:5],  # First 5 authors
                    "year": year
                })
        except Exception as e:
            continue
    
    return papers

def main():
    os.makedirs("data", exist_ok=True)
    
    all_papers = []
    seen_pmids = set()
    
    print("Downloading papers from PubMed...")
    print("=" * 50)
    
    for disease in tqdm(DISEASES):
        print(f"\nSearching: {disease}")
        
        # Search for papers
        pmids = search_pubmed(disease, max_results=40)
        print(f"  Found {len(pmids)} papers")
        
        # Filter out duplicates
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)
        
        # Fetch abstracts
        if new_pmids:
            papers = fetch_abstracts(new_pmids)
            for paper in papers:
                paper["disease_query"] = disease
            all_papers.extend(papers)
            print(f"  Downloaded {len(papers)} abstracts")
        
        # Be nice to PubMed servers
        time.sleep(0.5)
    
    # Save to file
    output_file = "data/pubmed_papers.json"
    with open(output_file, "w") as f:
        json.dump(all_papers, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Total papers downloaded: {len(all_papers)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
