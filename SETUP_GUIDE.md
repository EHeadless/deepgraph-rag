# 🚀 DeepGraph RAG - Complete Setup Guide

## Prerequisites
- Python 3.11+
- Docker Desktop
- OpenAI API key (~$15-20 for full setup)

---

## Step 1: Clone the repo
```bash
git clone https://github.com/EHeadless/deepgraph-rag.git
cd deepgraph-rag
```

## Step 2: Create Python environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

## Step 5: Start Neo4j database
```bash
docker-compose up -d
```
*Wait 30 seconds for initialization.*

## Step 6: Download papers from arXiv
```bash
python scripts/01_download_arxiv.py --categories cs.AI,cs.LG --max-papers 500
```

## Step 7: Extract entities with GPT-4
```bash
python extract_simple.py
```
*Uses GPT-4 to extract authors and concepts. Costs ~$10-15.*

## Step 8: Build the knowledge graph
```bash
python scripts/03_build_graph.py
```

## Step 9: Create vector embeddings
```bash
python scripts/04_create_indexes.py
```
*Costs ~$2.*

## Step 10: Add co-authorship relationships
```bash
python scripts/05_add_citations.py
```

## Step 11: Run the app
```bash
python -m streamlit run app.py
```
Open http://localhost:8501

**Or the user-friendly research explorer:**
```bash
python -m streamlit run app_user_clouds.py
```

---

## 🛑 Stopping the App
```bash
# Stop Streamlit: Ctrl+C

# Stop Neo4j:
docker-compose down

# Deactivate environment:
deactivate
```

---

## 🔄 Quick Start (after initial setup)
```bash
cd deepgraph-rag
docker-compose up -d
source venv/bin/activate
python -m streamlit run app_user_clouds.py
```

---

## 💰 Cost Summary

| Step | Cost | Time |
|------|------|------|
| Download papers | Free | 5 min |
| Extract entities | ~$10-15 | 30 min |
| Create embeddings | ~$2 | 10 min |
| **Total** | **~$12-17** | **~45 min** |

---

## 📁 Project Structure
```
deepgraph-rag/
├── app.py                    # Technical demo
├── app_user_clouds.py        # User-friendly explorer
├── docker-compose.yml        # Neo4j setup
├── scripts/
│   ├── 01_download_arxiv.py  # Download papers
│   ├── 03_build_graph.py     # Build Neo4j graph
│   ├── 04_create_indexes.py  # Create embeddings
│   └── 05_add_citations.py   # Add relationships
├── extract_simple.py         # GPT-4 entity extraction
└── data/                     # Downloaded papers
```
