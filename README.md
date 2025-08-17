# Residential-Treatment-Incident-Prediction

This repository contains a series of proof-of-concept machine learning models for predicting high-risk behavioral incidents in clinical settings, using both structured and unstructured (narrative) data. The goal is to explore how predictive modeling of behavior and natural language processing can support proactive intervention and improve care outcomes.

**All datasets in this repository are synthetic and do not contain any PHI. They were generated to reflect the statistical properties of real-world data.**

## Project Overview

This project consists of two parallel pipelines:

### Structured Data (Tabular)

*Goal: Predict next-day behavioral incidents using compliance scores and engineered trends.*

Models:

- Random Forest

- XGBoost


Key features:

- Rolling 3-day compliance averages

- Behavioral deltas (e.g., score escalation)

- SMOTE oversampling for class imbalance

- Confusion matrices and feature importance plots


Notebooks:

- random_forest_notebook.ipynb
- xgboost_notebook.ipynb

### Narrative Data (NLP)

*Goal: Leverage clinical narratives to identify latent risk and summarize clusters of concerning behavior.*

Contents:

- Hybrid BERT x BigBird classifier for sequence classification on short or long text (case logs)

- Unsupervised narrative clustering using:
    - SentenceTransformer embeddings
    - PCA + K-Means for topic discovery
    - BART-large summarization for cluster interpretation

Scripts:

- NLP/BERT and Bigbird Narrative Classification/bert_bigbird_trainer.py

- NLP/Narrative PCA with Transformers/PCA with Sentence-Transormers.ipynb

### Tech Stack
- Python: Core scripting
- scikit-learn, XGBoost: Model training
- Transformers, SentenceTransformers, BART: NLP modeling
- pandas, matplotlib, seaborn: Data handling & visualization
- SMOTE: Oversampling
- Jupyter Notebooks: Interactive analysis

## Disclaimer
This repository is for demonstration and educational purposes only. All data has been artificially generated to simulate production-like characteristics, and no identifiable or protected health information is used

## Contact
Built by Will Hallgren.  Feel free to reach out on [Linkedin](https://www.linkedin.com/in/william-hallgren/) or open an issue to discuss this work!