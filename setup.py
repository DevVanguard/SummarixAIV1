from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="summarixai",
    version="1.0.0",
    author="SummarixAI Team",
    description="Standalone offline-capable desktop application for document summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "PyQt6>=6.6.0",
        "PyMuPDF>=1.23.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "networkx>=3.2",
        "nltk>=3.8",
        "scikit-learn>=1.3.0",
        "bitsandbytes>=0.41.0",
        "python-docx>=1.1.0",
        "reportlab>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

