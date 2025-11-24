# Overview
This is part of the Machine Learning solo project at BeCode Data Science &amp; AI Bootcamp 2025. 

This repository has clean separation of concerns that makes it easy for deployment. It allows OOP design and is recognisable by Machine Learning (ML) operators and engineers alike. As this project grows in the future (hopefully!), this project design is very flexible and scalable.

# Timeline
--- Day 1 (24 Nov 2025) ---
- Introduction to the project
- Deeper understanding of regression
- Understanding machine learning
- Designing project structure
- Drawing preprocessing pipeline (with scikit-learn)
- Conducting data splitting and preprocessing

--- Day 2 (25 Nov 2025) ---

--- Day 3 (26 Nov 2025) ---

--- Day 4 (27 Nov 2025 - FINAL DAY) ---


# Project Structure
```markdown
immo-ML-project/
│
├── data/
│   ├── raw/
│   │   └── immo_raw.csv
│   ├── processed/
│   │   └── train.csv
│   │   └── test.csv
│   └── README.md
│
├── models/
│   ├── trained_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py        # functions to load CSVs
│   │   └── preprocess_data.py  # cleaning BEFORE train-test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py   # feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py          # training class (MyModelTrainer)
│   │   ├── evaluator.py        # evaluation functions
│   │   └── predict.py          # predict new data with saved model
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── price_pipeline.py  # scikit-learn Pipeline object
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # misc utilities
│
├── notebooks/
│   ├── EDA.ipynb               # exploratory analysis
│   └── experiments.ipynb       # trying models
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_pipeline.py
│   └── test_model_training.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

