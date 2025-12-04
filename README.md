# About Me
Given my intensive background in cognitive psychology, I like to use human's behavioural learning and modelling as analogy for me to learn Machine Learning. I like to think that Machine Learning is the mathematical model of human's behaviour imitated by machines. Or maybe, more aptly phrased: mathematical models of human's behaviour created by humans such that machines can imitate us :wink:

# Project Overview
This project is part of the Machine Learning solo project at BeCode Data Science &amp; AI Bootcamp 2025. 

This repository has clean separation of concerns that makes it easy for deployment. It allows OOP design and is recognisable by Machine Learning (ML) operators and engineers alike. As this project grows in the future (hopefully!), this project design is very flexible and scalable.

# Timeline
--- Day 1 (24 Nov 2025) ---
- Introduction to the project
- Deeper understanding of regression
- Understanding of machine learning
- Design project structure 
- Create `notebooks/EDA.ipynb` and `notebooks/experiments.ipynb`
- Plan basic cleaning pipeline
- Create `src/data/load_data.py`, `src/data/main.py`, and `src/data/preprocess_data.py`

--- Day 2 (25 Nov 2025) ---
- Update files in the `tests` folder for unit-testing
- Unit-test the functions made the previous day
- Plan ML preprocessing pipeline and hyperparameters tuning
- Create `src/models/trainer.py` and `src/models/tuner.py`
- Test trainer functions in `notebooks/experiments.ipynb`

--- Day 3 (26 Nov 2025) ---
- Update files in the `tests` folder for unit-testing
- Unit-test the functions made the previous day
- Revisit EDA and models' trainer
- Improve `src/models/trainer.py`

--- Day 4 (27 Nov 2025 - FINAL DAY) ---
- Create `predict.py` (but not run)
- Skip hyperparameter tuning (was too heavy and time-consuming)
- Conduct cross validation for all models
- Run the final model trainers on `notebooks/experiments.ipynb`
- Save all the preprocessors and models
- Evaluate and compare models

# Package Usage
## Preprocessing pipeline (src/models/trainer.py)
The preprocessing pipeline includes imputation, data scaling (only for Ridge linear model), one-hot encoding, logarithmic transformation for variable living_area, and Winsor capping. The pipelines will also be saved for each model in models/ as models/*_pipeline.pkl.

## Exploratory Data Analyses (notebooks/EDA.ipynb)
Used to visualise and inspect both raw and clean data.

## Training the models (notebooks/experiments.ipynb)
This notebook contains the step-by-step process of training the 3 models: Ridge, RandomForest, and XGBoost. It also has the interpretations of the results, both analytically (??) and in business terms. 

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
├── ml_components/
│   └── transformers.py
│
├── models/
│   ├── RandomForest_pipeline.pkl
│   ├── Ridge_pipeline.pkl
│   └── XGBoost_pipeline.pkl
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py        # functions to load CSVs
│   │   ├── main.py
│   │   └── preprocess_data.py  # cleaning BEFORE train-test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # feature engineering (CURRENTLY UNUSED)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluator.py        # evaluation functions
│   │   ├── trainer.py          # training class
│   │   ├── tuner.py            # hyperparameter tuning (CURRENTLY UNUSED)
│   │   └── predict.py          # predict new data with saved model
│   │
│   └── pipelines/
│       ├── __init__.py
│       └── price_pipeline.py   # scikit-learn Pipeline object         
│
├── notebooks/
│   ├── EDA.ipynb               
│   └── experiments.ipynb       
│
├── tests/                      # unit testing (CURRENTLY UNUSED)
│   ├── test_data_loading.py
│   ├── test_pipeline.py
│   ├── test_preprocess_data.py
│   └── test_model_training.py
│
├── tools/                      
│   └── resave_models.py
│
├── Makefile
├── requirements.txt
├── pyproject.toml              # used for unit-testing
├── pytest.ini                  # used for unit-testing
├── README.md
└── LICENSE
```

