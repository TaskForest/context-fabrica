from pathlib import Path

from setuptools import setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="context-fabrica",
    version="1.0.0",
    description="Governed agent memory with hybrid retrieval, temporal recall, and graph reasoning",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    author="TaskForest",
    author_email="human@taskforest.xyz",
    url="https://github.com/TaskForest/context-fabrica",
    project_urls={
        "Documentation": "https://github.com/TaskForest/context-fabrica/tree/main/docs",
        "Issues": "https://github.com/TaskForest/context-fabrica/issues",
        "Changelog": "https://github.com/TaskForest/context-fabrica/blob/main/CHANGELOG.md",
    },
    license="MIT",
    license_files=("LICENSE",),
    keywords=["ai", "agents", "memory", "rag", "knowledge-graph", "retrieval"],
    package_dir={"": "src"},
    include_package_data=True,
    packages=[
        "context_fabrica",
        "context_fabrica.extractors",
        "context_fabrica.storage",
    ],
    extras_require={
        "postgres": ["psycopg[binary]>=3.2", "pgvector>=0.3"],
        "kuzu": ["kuzu>=0.8"],
        "fastembed": ["fastembed"],
        "transformers": ["sentence-transformers>=2.0"],
        "all": [
            "psycopg[binary]>=3.2",
            "pgvector>=0.3",
            "kuzu>=0.8",
            "fastembed",
            "sentence-transformers>=2.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "context-fabrica=context_fabrica.cli:main",
            "context-fabrica-demo=context_fabrica.demo_cli:main",
            "context-fabrica-bootstrap=context_fabrica.bootstrap_cli:main",
            "context-fabrica-doctor=context_fabrica.doctor_cli:main",
            "context-fabrica-projector=context_fabrica.projector_cli:main",
            "context-fabrica-project-memory=context_fabrica.project_memory_cli:main",
            "context-fabrica-mcp=context_fabrica.mcp_server:main",
            "context-fabrica-extract=context_fabrica.extract_cli:main",
            "context-fabrica-install=context_fabrica.install_cli:main",
        ]
    },
)
