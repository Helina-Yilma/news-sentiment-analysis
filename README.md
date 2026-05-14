## News Sentiment Analysis

1. Project Structure
The repository follows a standard data science project structure:

├── .vscode/ # Editor configurations ├── .github/ # CI/CD workflows (e.g., unittests.yml) ├── README.md # This file ├── src/ # Helper classes and core logic (e.g., NMF helper functions) ├── notebooks/ # Exploratory notebooks and documentation of the pipeline execution ├── tests/ # Unit tests for core functions ├── scripts/ # Production-ready, sequential analytical scripts ├── data/ # Storage for processed data files (e.g., .pkl files) └── requirements.txt # Project dependencies

Setup and Execution
2.1. Prerequisites

You will need Python 3.8+ installed. All required dependencies are listed in requirements.txt.

To install dependencies:

pip install -r requirements.txt

2.2. Data Input

This pipeline assumes a raw CSV file named financial_news.csv (or similar, as configured in the scripts) containing at least three columns:

A timestamp column (timestamp).

The full headline text (headline).

The source/publisher name (publisher).

2.3. Analysis Pipeline Execution

The entire analytical workflow is split into four sequential scripts in the scripts/ directory, ensuring modularity and reproducibility.

python scripts/data_cleaning.py python scripts/topic_modeling.py python scripts/time_series_analysis.py python scripts/publisher_specialization.py