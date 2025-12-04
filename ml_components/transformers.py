# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 03-12-2025

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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

        for idx in self.columns: # type: ignore
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


