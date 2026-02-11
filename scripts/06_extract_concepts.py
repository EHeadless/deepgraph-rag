"""
Extract rich concepts, methods, and datasets from paper abstracts.

This transforms a shallow graph (Paper → Author → Category) into a rich graph:
  Paper → USES_METHOD → Method (transformer, attention, CNN, ...)
  Paper → DISCUSSES → Concept (self-supervision, few-shot learning, ...)
  Paper → USES_DATASET → Dataset (ImageNet, COCO, ...)

This enables powerful queries like:
  - "Papers using transformers in robotics"
  - "What methods from NLP are now used in Vision?"
  - "Find papers combining attention AND reinforcement learning"
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# ============== METHOD PATTERNS ==============
# Common ML/AI methods - will match case-insensitively
METHODS = {
    # Architectures
    "transformer": ["transformer", "transformers", "attention mechanism"],
    "attention": ["self-attention", "cross-attention", "multi-head attention"],
    "cnn": ["cnn", "convolutional neural network", "convnet", "conv network"],
    "rnn": ["rnn", "recurrent neural network", "lstm", "gru"],
    "gnn": ["gnn", "graph neural network", "graph network", "message passing"],
    "vae": ["vae", "variational autoencoder", "variational encoder"],
    "gan": ["gan", "generative adversarial", "discriminator"],
    "diffusion": ["diffusion model", "denoising diffusion", "ddpm", "score-based"],
    "autoencoder": ["autoencoder", "encoder-decoder"],
    "mlp": ["mlp", "multi-layer perceptron", "feedforward network"],
    "resnet": ["resnet", "residual network", "skip connection"],
    "bert": ["bert", "masked language model"],
    "gpt": ["gpt", "autoregressive language model", "causal language model"],
    "vit": ["vit", "vision transformer"],
    "unet": ["u-net", "unet"],
    "nerf": ["nerf", "neural radiance field"],

    # Training methods
    "reinforcement_learning": ["reinforcement learning", "rl ", "reward", "policy gradient", "q-learning", "dqn"],
    "contrastive_learning": ["contrastive learning", "contrastive loss", "simclr", "moco"],
    "self_supervised": ["self-supervised", "self-supervision", "pretext task"],
    "supervised_learning": ["supervised learning", "labeled data", "classification"],
    "unsupervised": ["unsupervised", "clustering", "k-means"],
    "transfer_learning": ["transfer learning", "pretrained", "fine-tuning", "finetuning"],
    "meta_learning": ["meta-learning", "learning to learn", "few-shot", "maml"],
    "curriculum_learning": ["curriculum learning", "curriculum"],
    "knowledge_distillation": ["knowledge distillation", "teacher-student", "distillation"],
    "multitask": ["multitask", "multi-task learning"],

    # Optimization
    "adam": ["adam optimizer", "adam "],
    "sgd": ["stochastic gradient descent", "sgd "],
    "dropout": ["dropout"],
    "batch_norm": ["batch normalization", "batchnorm"],
    "layer_norm": ["layer normalization", "layernorm"],

    # Specific techniques
    "lora": ["lora", "low-rank adaptation"],
    "prompt_tuning": ["prompt tuning", "soft prompt", "prompt learning"],
    "rag": ["retrieval-augmented", "rag ", "retrieval augmented generation"],
    "chain_of_thought": ["chain-of-thought", "chain of thought", "cot prompting"],
    "in_context_learning": ["in-context learning", "icl ", "few-shot prompting"],
}

# ============== CONCEPT PATTERNS ==============
# Research concepts/areas
CONCEPTS = {
    # Core ML concepts
    "representation_learning": ["representation learning", "learned representations", "feature learning"],
    "generalization": ["generalization", "out-of-distribution", "ood ", "domain shift"],
    "optimization": ["optimization", "convergence", "loss landscape"],
    "scalability": ["scalability", "scaling laws", "parameter efficient"],
    "interpretability": ["interpretability", "explainability", "explainable ai", "xai"],
    "fairness": ["fairness", "bias", "debiasing"],
    "robustness": ["robustness", "adversarial", "perturbation"],
    "efficiency": ["efficiency", "computation", "latency", "memory"],
    "multimodal": ["multimodal", "multi-modal", "vision-language", "cross-modal"],
    "embodied_ai": ["embodied ai", "embodied agent", "physical interaction"],

    # NLP concepts
    "language_understanding": ["language understanding", "nlu", "semantic understanding"],
    "language_generation": ["language generation", "text generation", "nlg"],
    "question_answering": ["question answering", "qa ", "reading comprehension"],
    "summarization": ["summarization", "summarize", "abstractive"],
    "translation": ["translation", "machine translation", "nmt"],
    "sentiment": ["sentiment analysis", "opinion mining"],
    "named_entity": ["named entity", "ner ", "entity recognition"],
    "dialogue": ["dialogue", "conversation", "chatbot"],

    # Vision concepts
    "object_detection": ["object detection", "detector", "bounding box"],
    "segmentation": ["segmentation", "semantic segmentation", "instance segmentation"],
    "image_classification": ["image classification", "visual recognition"],
    "image_generation": ["image generation", "image synthesis", "text-to-image"],
    "video_understanding": ["video understanding", "video analysis", "temporal"],
    "3d_vision": ["3d ", "3d vision", "point cloud", "depth estimation"],
    "pose_estimation": ["pose estimation", "human pose", "skeleton"],

    # Agents & reasoning
    "agentic": ["agentic", "agent", "autonomous agent", "tool use"],
    "reasoning": ["reasoning", "logical reasoning", "commonsense"],
    "planning": ["planning", "task planning", "motion planning"],
    "decision_making": ["decision making", "decision-making"],

    # Other domains
    "medical_ai": ["medical", "clinical", "healthcare", "diagnosis"],
    "robotics": ["robotics", "robot", "manipulation", "locomotion"],
    "autonomous_driving": ["autonomous driving", "self-driving", "vehicle"],
    "speech": ["speech recognition", "speech synthesis", "asr", "tts"],
    "recommendation": ["recommendation", "recommender system", "collaborative filtering"],
    "graph_learning": ["graph learning", "knowledge graph", "graph representation"],
}

# ============== DATASET PATTERNS ==============
DATASETS = {
    # Vision
    "imagenet": ["imagenet"],
    "coco": ["coco", "ms coco"],
    "cifar": ["cifar-10", "cifar-100", "cifar10", "cifar100"],
    "mnist": ["mnist"],
    "celeba": ["celeba", "celebahq"],
    "laion": ["laion"],

    # NLP
    "glue": ["glue benchmark", "glue "],
    "squad": ["squad"],
    "wikitext": ["wikitext"],
    "commoncrawl": ["common crawl", "commoncrawl"],
    "pile": ["the pile"],
    "mmlu": ["mmlu"],
    "gsm8k": ["gsm8k", "gsm-8k"],

    # Multimodal
    "vqa": ["vqa ", "visual question"],
    "flickr": ["flickr30k", "flickr8k"],
    "conceptual_captions": ["conceptual captions"],
}


def extract_from_abstract(abstract: str) -> dict:
    """Extract methods, concepts, and datasets from an abstract."""
    abstract_lower = abstract.lower()

    found_methods = []
    found_concepts = []
    found_datasets = []

    # Extract methods
    for method_name, patterns in METHODS.items():
        for pattern in patterns:
            if pattern.lower() in abstract_lower:
                found_methods.append(method_name)
                break

    # Extract concepts
    for concept_name, patterns in CONCEPTS.items():
        for pattern in patterns:
            if pattern.lower() in abstract_lower:
                found_concepts.append(concept_name)
                break

    # Extract datasets
    for dataset_name, patterns in DATASETS.items():
        for pattern in patterns:
            if pattern.lower() in abstract_lower:
                found_datasets.append(dataset_name)
                break

    return {
        "methods": list(set(found_methods)),
        "concepts": list(set(found_concepts)),
        "datasets": list(set(found_datasets))
    }


class ConceptGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✓ Connected to Neo4j")

    def create_schema(self):
        """Create additional constraints for new node types."""
        queries = [
            "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
            "CREATE INDEX concept_type IF NOT EXISTS FOR (c:Concept) ON (c.type)",
        ]

        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                except:
                    pass
        print("✓ Extended schema created")

    def add_paper_extractions(self, paper_id: str, extractions: dict):
        """Add extracted methods, concepts, and datasets to a paper."""
        with self.driver.session() as session:
            # Add methods
            for method in extractions["methods"]:
                display_name = method.replace("_", " ").title()
                session.run("""
                    MERGE (m:Method {name: $name})
                    SET m.display_name = $display_name
                """, name=method, display_name=display_name)

                session.run("""
                    MATCH (p:Paper {arxiv_id: $paper_id})
                    MATCH (m:Method {name: $method})
                    MERGE (p)-[:USES_METHOD]->(m)
                """, paper_id=paper_id, method=method)

            # Add concepts (as new Concept nodes, not categories)
            for concept in extractions["concepts"]:
                display_name = concept.replace("_", " ").title()
                session.run("""
                    MERGE (c:Concept {name: $name})
                    SET c.display_name = $display_name,
                        c.type = 'topic'
                """, name=concept, display_name=display_name)

                session.run("""
                    MATCH (p:Paper {arxiv_id: $paper_id})
                    MATCH (c:Concept {name: $concept})
                    MERGE (p)-[:DISCUSSES]->(c)
                """, paper_id=paper_id, concept=concept)

            # Add datasets
            for dataset in extractions["datasets"]:
                display_name = dataset.replace("_", " ").upper() if len(dataset) <= 6 else dataset.replace("_", " ").title()
                session.run("""
                    MERGE (d:Dataset {name: $name})
                    SET d.display_name = $display_name
                """, name=dataset, display_name=display_name)

                session.run("""
                    MATCH (p:Paper {arxiv_id: $paper_id})
                    MATCH (d:Dataset {name: $dataset})
                    MERGE (p)-[:USES_DATASET]->(d)
                """, paper_id=paper_id, dataset=dataset)

    def create_method_concept_links(self):
        """Create METHOD_FOR relationships between Methods and Concepts based on co-occurrence."""
        print("Creating method-concept links...")

        with self.driver.session() as session:
            # Find methods and concepts that frequently appear together
            result = session.run("""
                MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
                MATCH (p)-[:DISCUSSES]->(c:Concept)
                WITH m, c, count(p) as co_occurrences
                WHERE co_occurrences >= 3
                MERGE (m)-[r:METHOD_FOR]->(c)
                SET r.strength = co_occurrences
                RETURN count(r) as links_created
            """)
            count = result.single()["links_created"]
            print(f"✓ Created {count} method-concept links")

    def get_stats(self):
        """Get updated graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            return {r["label"]: r["count"] for r in result}

    def get_edge_stats(self):
        """Get relationship statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """)
            return {r["rel_type"]: r["count"] for r in result}

    def close(self):
        self.driver.close()


def main():
    print("=" * 60)
    print("EXTRACTING RICH CONCEPTS FROM PAPER ABSTRACTS")
    print("=" * 60)

    # Load papers
    data_path = Path("data/processed/extracted_entities.jsonl")
    papers = []
    with open(data_path) as f:
        for line in f:
            papers.append(json.loads(line))
    print(f"✓ Loaded {len(papers)} papers")

    # Extract from all papers
    print("\nExtracting methods, concepts, and datasets...")
    extractions = {}
    method_counts = defaultdict(int)
    concept_counts = defaultdict(int)
    dataset_counts = defaultdict(int)

    for paper in tqdm(papers, desc="Extracting"):
        abstract = paper.get("abstract", "")
        extracted = extract_from_abstract(abstract)
        extractions[paper["paper_id"]] = extracted

        for m in extracted["methods"]:
            method_counts[m] += 1
        for c in extracted["concepts"]:
            concept_counts[c] += 1
        for d in extracted["datasets"]:
            dataset_counts[d] += 1

    # Show extraction summary
    print(f"\n📊 Extraction Summary:")
    print(f"  Methods found: {len(method_counts)} types, {sum(method_counts.values())} occurrences")
    print(f"  Concepts found: {len(concept_counts)} types, {sum(concept_counts.values())} occurrences")
    print(f"  Datasets found: {len(dataset_counts)} types, {sum(dataset_counts.values())} occurrences")

    print(f"\n🔧 Top Methods:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {method.replace('_', ' ').title()}: {count} papers")

    print(f"\n💡 Top Concepts:")
    for concept, count in sorted(concept_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {concept.replace('_', ' ').title()}: {count} papers")

    # Add to Neo4j
    print("\n" + "=" * 60)
    print("ADDING TO NEO4J GRAPH")
    print("=" * 60)

    builder = ConceptGraphBuilder(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "deepgraph2025")
    )

    builder.create_schema()

    print("\nAdding extractions to graph...")
    for paper_id, extracted in tqdm(extractions.items(), desc="Adding to Neo4j"):
        if extracted["methods"] or extracted["concepts"] or extracted["datasets"]:
            builder.add_paper_extractions(paper_id, extracted)

    # Create cross-links
    builder.create_method_concept_links()

    # Show final stats
    print("\n📊 Updated Graph Statistics:")
    print("\nNodes:")
    for label, count in builder.get_stats().items():
        print(f"  {label}: {count}")

    print("\nRelationships:")
    for rel_type, count in builder.get_edge_stats().items():
        print(f"  {rel_type}: {count}")

    builder.close()
    print("\n✅ Done! Your graph now has rich method/concept connections.")
    print("\nTry queries like:")
    print('  MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: "transformer"})')
    print('  MATCH (p:Paper)-[:DISCUSSES]->(c:Concept {name: "reasoning"})')
    print('  MATCH (m:Method)-[:METHOD_FOR]->(c:Concept) RETURN m, c')


if __name__ == "__main__":
    main()
