"""
DeepGraph RAG - Graph RAG for intersection queries.

Find things that HAVE X FOR Y.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="deepgraph-rag",
    version="2.0.0",
    author="DeepGraph Team",
    author_email="",
    description="Graph RAG for intersection queries. Find things that HAVE X FOR Y.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepgraph-rag",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/deepgraph-rag/issues",
        "Documentation": "https://github.com/yourusername/deepgraph-rag#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "neo4j>=5.16.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
        ],
        "ui": [
            "streamlit>=1.31.0",
            "pyvis>=0.3.2",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
            "streamlit>=1.31.0",
            "pyvis>=0.3.2",
            "pytest>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepgraph=deepgraph.cli:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "rag",
        "graph-rag",
        "retrieval-augmented-generation",
        "knowledge-graph",
        "neo4j",
        "llm",
        "openai",
        "nlp",
        "machine-learning",
    ],
)
