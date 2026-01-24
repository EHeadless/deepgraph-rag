"""
Simple demo script to test DeepGraph RAG.

Usage:
    python demo.py
"""

import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from deepgraph_rag import DeepGraphRAG, DeepGraphConfig


def main():
    # Load environment
    load_dotenv()
    
    print("=" * 80)
    print("DeepGraph RAG Demo")
    print("=" * 80)
    print()
    
    # Initialize
    print("Initializing DeepGraph RAG...")
    rag = DeepGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "deepgraph2025")
    )
    print("✓ Initialized!\n")
    
    # Example queries
    queries = [
        "What are the latest developments in transformer architectures?",
        "Papers on attention mechanisms by researchers from top institutions",
        "Which methods from Stanford are being cited by recent papers?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print("=" * 80)
        
        try:
            # Run query
            response = rag.query(query, explain=True)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"  - Sources: {len(response.sources)}")
            print(f"  - Confidence: {response.confidence:.2%}")
            print(f"  - Iterations: {response.retrieval_stats.get('iterations', 0)}")
            
            print(f"\n💡 Answer:")
            print(f"  {response.answer[:500]}...")
            
            if response.reasoning_path:
                print(f"\n🔍 Reasoning Path:")
                for step in response.reasoning_path[:3]:  # Show first 3 steps
                    print(f"  Step {step['iteration']}: {step['action']}")
            
            print(f"\n📚 Top Sources:")
            for j, source in enumerate(response.sources[:3], 1):
                print(f"  [{j}] {source.get('title', 'N/A')}")
                print(f"      Method: {source.get('retrieval_method', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Close
    rag.close()
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
