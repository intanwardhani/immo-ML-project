# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 26-11-2025

import numpy as np
import joblib 
import os
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
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
# 2. ModelTrainer
# ============================================================

class ModelTrainer:
    """
    Class to train and evaluate regression models for house price prediction.

    Updates:
    - Uses log(price) as target (log1p)
    - Applies log(living_area) ONLY for Ridge
    - Applies winsorisation (IQR capping) on price_log + living_area_log
    - Uses 2 different preprocessors (scaled, non-scaled)
    """

    def __init__(self, X_train, y_train, X_test, y_test, save_dir="models/"):
        self.y_train = y_train
        self.y_test = y_test

        # Keep raw X
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        
        # Create save directory
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Define models
        self.models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        # Column splitting
        self.numeric_continuous = ["living_area"]
        self.binary_and_ordinal = ["swimming_pool", "garden", "terrace", "facades"]
        self.numeric_as_categorical = ["postal_code", "number_bedrooms", "build_year"]
        self.categorical_cols = (["build_year_cat",
                                  "building_state",
                                  "locality_name",
                                  "property_type",
                                  "province"]
                                 + self.binary_and_ordinal
                                 + self.numeric_as_categorical
                                 )
        self.numerical_cols = self.numeric_continuous
        
        # ---- FIX: convert categorical columns to string ----
        for col in self.categorical_cols:
            self.X_train[col] = self.X_train[col].astype(str)
            self.X_test[col] = self.X_test[col].astype(str)

        # Living area and price require special transformations
        self.log_cols_for_models = {
            "Ridge": ["living_area"],   # ONLY Ridge gets log
            "RandomForest": [],         # No log
            "XGBoost": []
        }

        # Columns to cap for outliers (log versions)
        self.outlier_cols = ["living_area", "price"]

        self._build_preprocessors()

    def _build_preprocessors(self):
        """Construct 2 separate preprocessors: scaled & non-scaled."""

        # Shared categorical pipeline
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Shared numerical imputing
        num_impute = SimpleImputer(strategy="median")

        # ==============================================================
        # Ridge PREPROCESSOR (with scaling + log living_area + capping)
        # ==============================================================

        living_area_index = self.numerical_cols.index("living_area")
        num_pipe_scaled = Pipeline([
            ("imputer", num_impute),
            ("log_transform", LogTransformer(columns=[living_area_index])),   # <--- only Ridge
            ("winsor", CustomOutlierCappingTransformer(columns=[living_area_index])),
            ("scaler", StandardScaler())
        ])

        self.preprocessor_scaled = ColumnTransformer([
            ("num", num_pipe_scaled, self.numerical_cols),
            ("cat", cat_pipe, self.categorical_cols)
        ])

        # ==============================================================
        # Tree-based PREPROCESSOR (no scaling, no log living_area)
        # ==============================================================

        num_pipe_noscale = Pipeline([
            ("imputer", num_impute),
            ("winsor", CustomOutlierCappingTransformer(columns=[living_area_index]))
        ])

        self.preprocessor_noscale = ColumnTransformer([
            ("num", num_pipe_noscale, self.numerical_cols),
            ("cat", cat_pipe, self.categorical_cols)
        ])

    # Pick correct pipeline
    def _get_preprocessor(self, model_name):
        return (
            self.preprocessor_scaled
            if model_name == "Ridge"
            else self.preprocessor_noscale
        )

    # ==============================================================
    # Cross-validation
    # ==============================================================

    def cross_validate_models(self, cv=5):
        results = {}

        for name, model in self.models.items():
            preprocessor = self._get_preprocessor(name)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])

            # Local y variable, log-transform only for Ridge
            y_cv = np.log1p(self.y_train) if name == "Ridge" else self.y_train
            scores = cross_val_score(
                pipeline,
                self.X_train,
                y_cv,
                cv=cv,
                scoring="neg_root_mean_squared_error"
            )

            results[name] = {
                "cv_scores": -scores,
                "mean_cv_rmse": -scores.mean()
            }

        return results

    # ==============================================================
    # Train + Evaluate (returns RMSE in price scale)
    # ==============================================================

    def train_and_evaluate(self, model_name=None):
        """
        Train and evaluate models.
        
        Parameters
        ----------
        model_name : str, optional
            If specified, only runs this model. Must be a key in self.models.
        
        Returns
        -------
        dict
            model_name → dict with rmse and r2_test
        """
        
        
        results = {}
        models_to_run = (
                        {model_name: self.models[model_name]} 
                        if model_name else self.models
                        )

        for name, model in models_to_run.items():
            preprocessor = self._get_preprocessor(name)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])
            
            # Local y variable for training
            y_train_model = np.log1p(self.y_train) if name == "Ridge" else self.y_train
            y_test_model = np.log1p(self.y_test) if name == "Ridge" else self.y_test

            # Fit model
            pipeline.fit(self.X_train, y_train_model)
            
            # Predictions
            y_pred_train = pipeline.predict(self.X_train)
            y_pred_test = pipeline.predict(self.X_test)
            
            # Convert back to original scale if Ridge
            if name == "Ridge":
                y_pred_train = np.expm1(y_pred_train)
                y_pred_test = np.expm1(y_pred_test)
                y_train_model_orig = np.expm1(y_train_model)
                y_test_model_orig = np.expm1(y_test_model)
            else:
                y_train_model_orig = y_train_model
                y_test_model_orig = y_test_model

            # Evaluation metrics
            rmse_train = root_mean_squared_error(y_train_model_orig, y_pred_train)
            r2_train = r2_score(y_train_model_orig, y_pred_train)
            rmse_test = root_mean_squared_error(y_test_model_orig, y_pred_test)
            r2_test = r2_score(y_test_model_orig, y_pred_test)
            
            results[name] = {
                            "rmse_train": rmse_train,
                            "r2_train": r2_train,
                            "rmse_test": rmse_test,
                            "r2_test": r2_test
                            }
            
            # ---------- SAVE THE FULL PIPELINE ---------- 
            model_path = os.path.join(self.save_dir, f"{name}_pipeline.pkl")
            joblib.dump(pipeline, model_path)
            print(f"Saved {name} model to: {model_path}")
            # ---------------------------------------------

        return results



