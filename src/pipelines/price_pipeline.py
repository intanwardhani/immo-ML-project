# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 01-12-2025

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ============================================================
# 1. Custom Transformers
# ============================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply safe log1p transform to selected numeric columns."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if hasattr(X, "iloc"):
            X = X.values

        for idx in self.columns:
            X[:, idx] = np.log1p(X[:, idx])
        return X


class CustomOutlierCappingTransformer(BaseEstimator, TransformerMixin):
    """Winsorise numerical columns using IQR-based capping."""

    def __init__(self, columns, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        X = X.copy()
        if hasattr(X, "iloc"):
            X = X.values

        for idx in self.columns:
            col_data = X[:, idx]
            q1 = np.quantile(col_data, 0.25)
            q3 = np.quantile(col_data, 0.75)
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            self.bounds[idx] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        if hasattr(X, "iloc"):
            X = X.values
        for idx, (lower, upper) in self.bounds.items():
            X[:, idx] = np.clip(X[:, idx], lower, upper)
        return X


# ============================================================
# 2. Pipeline Builders
# ============================================================

def build_ridge_pipeline(
        numerical_cols,
        categorical_cols,
        continuous_index):
    """Pipeline for Ridge (scaled + log-transform)."""

    num_impute = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe_scaled = Pipeline([
        ("imputer", num_impute),
        ("log_transform", LogTransformer(columns=[continuous_index])),
        ("winsor", CustomOutlierCappingTransformer(columns=[continuous_index])),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe_scaled, numerical_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    model = Ridge(alpha=1.0)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])


def build_tree_pipeline(
        numerical_cols,
        categorical_cols,
        continuous_index,
        model_type="RandomForest"):
    """Pipeline for RandomForest or XGBoost (no scaling, no log)."""

    num_impute = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = Pipeline([
        ("imputer", num_impute),
        ("winsor", CustomOutlierCappingTransformer(columns=[continuous_index]))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numerical_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])


# ============================================================
# 3. Pipeline Factory
# ============================================================

def get_pipeline(model_name, numerical_cols, categorical_cols):
    """
    Returns the correct pipeline for training,
    based on the model name.
    """
    continuous_index = numerical_cols.index("living_area")

    if model_name == "Ridge":
        return build_ridge_pipeline(
            numerical_cols,
            categorical_cols,
            continuous_index
        )
    else:
        return build_tree_pipeline(
            numerical_cols,
            categorical_cols,
            continuous_index,
            model_type=model_name
        )

