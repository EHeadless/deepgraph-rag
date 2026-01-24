# 🧠 DeepGraph RAG

**Graph-Powered Research Assistant with Multi-Hop Reasoning**

A demonstration of how knowledge graphs enhance RAG (Retrieval-Augmented Generation) by enabling relationship-based queries that traditional vector search cannot answer.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-5.16-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)

## 🎯 What This Project Demonstrates

| Query Type | Vector Search | Graph RAG |
|------------|---------------|-----------|
| "Papers about transformers" | ✅ | ✅ |
| "Papers by Author X" | ❌ | ✅ |
| "Who collaborates with X?" | ❌ | ✅ |
| "Papers by X's collaborators" | ❌ | ✅ |
| "How are Author A and B connected?" | ❌ | ✅ |

**Key insight:** Graphs don't replace vector search—they enable a different class of questions that require traversing relationships.

## 🏗️ Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Vector Search  │────▶│   Retrieved     │
│                 │     │  (Embeddings)   │     │   Papers        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│   GPT-4         │◀────│  Graph Context  │◀─────────────┘
│   Answer        │     │  (Multi-hop)    │
└─────────────────┘     └─────────────────┘
```

## 📊 Knowledge Graph Stats

- **550 Papers** from arXiv (AI/ML categories)
- **2,792 Authors** extracted
- **13,779 Co-authorship** relationships
- **5 Multi-hop query types**

## 🔗 Multi-Hop Query Types

1. **Collaborators' Papers** (3 hops)
   - Author → Papers → Co-authors → Their papers
   
2. **Authors' Other Papers** (2 hops)
   - Paper → Authors → Their other papers
   
3. **Extended Network** (3 hops)
   - Author → Co-authors → Their co-authors
   
4. **Connection Path** (N hops)
   - Find shortest path between two authors
   
5. **Network Hubs**
   - Most connected authors in the graph

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Neo4j)
- OpenAI API key

### Installation
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/deepgraph-rag.git
cd deepgraph-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your OpenAI API key
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start Neo4j
docker-compose up -d

# Wait 30 seconds for Neo4j to start, then run the app
streamlit run app.py
```

### Building the Knowledge Graph (Optional)

If you want to build the graph from scratch:
```bash
# 1. Download papers from arXiv
python scripts/01_download_arxiv.py --categories cs.AI,cs.LG --max-papers 500

# 2. Extract entities using GPT-4 (~$15 for 500 papers)
python extract_simple.py

# 3. Build the graph in Neo4j
python scripts/03_build_graph.py

# 4. Create vector embeddings (~$2)
python scripts/04_create_indexes.py

# 5. Add co-authorship relationships
python scripts/05_add_citations.py
```

## 📁 Project Structure
```
deepgraph-rag/
├── app.py                 # Streamlit web interface
├── docker-compose.yml     # Neo4j container config
├── requirements.txt       # Python dependencies
├── env.example           # Environment template
├── scripts/
│   ├── 01_download_arxiv.py    # Download papers
│   ├── 03_build_graph.py       # Build Neo4j graph
│   ├── 04_create_indexes.py    # Create embeddings
│   └── 05_add_citations.py     # Add relationships
├── extract_simple.py     # Entity extraction with GPT-4
└── data/                 # Downloaded papers (gitignored)
```

## 🛠️ Tech Stack

- **Neo4j** - Graph database for storing papers, authors, relationships
- **OpenAI** - Embeddings (ada-002) and generation (GPT-4)
- **Streamlit** - Interactive web interface
- **PyVis** - Graph visualization
- **arXiv API** - Paper data source

## 📈 Why Graph RAG?

Traditional RAG finds documents by **text similarity**.
Graph RAG answers questions about **relationships**.

This project honestly demonstrates what graphs add:
- They don't make text search "better"
- They enable queries that require traversing connections
- They're valuable when relationships matter (research networks, org charts, knowledge bases)

## 📝 License

MIT

## 🙏 Acknowledgments

- arXiv for the open paper data
- Neo4j for the graph database
- OpenAI for embeddings and GPT-4
