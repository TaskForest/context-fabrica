from setuptools import setup


setup(
    name="context-fabrica",
    version="0.3.0",
    description="Hybrid graph + semantic memory substrate for AI agents",
    long_description=open("README.md", encoding="utf-8").read(),
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
    package_dir={"": "src"},
    packages=[
        "context_fabrica",
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
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
        ]
    },
)
