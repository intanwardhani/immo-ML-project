# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 26-11-2025


import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score

from src.pipelines.price_pipeline import get_pipeline


class ModelTrainer:
    """
    Trains & evaluates regression models for house price prediction.
    Uses pipelines imported from src/pipelines/price_pipeline.py.
    """

    def __init__(self, X_train, y_train, X_test, y_test, save_dir="models/"):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.y_test = y_test

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.models = {
            "Ridge": None,          # regressor is inside pipeline_factory
            "RandomForest": None,
            "XGBoost": None
        }

        # Column groups
        self.numeric_continuous = ["living_area"]
        self.binary_and_ordinal = ["swimming_pool", "garden", "terrace", "facades"]
        self.numeric_as_categorical = ["postal_code", "number_bedrooms", "build_year"]
        self.categorical_cols = (
            ["build_year_cat", "building_state", "locality_name",
             "property_type", "province"]
            + self.binary_and_ordinal
            + self.numeric_as_categorical
        )
        self.numerical_cols = self.numeric_continuous

        # Ensure categorical cols are string
        for col in self.categorical_cols:
            self.X_train[col] = self.X_train[col].astype(str)
            self.X_test[col] = self.X_test[col].astype(str)

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------
    def cross_validate_models(self, cv=5):
        results = {}
        for name in self.models.keys():
            pipeline = get_pipeline(
                name,
                self.numerical_cols,
                self.categorical_cols
            )

            y_train_model = np.log1p(self.y_train) if name == "Ridge" else self.y_train

            scores = cross_val_score(
                pipeline,
                self.X_train,
                y_train_model,
                cv=cv,
                scoring="neg_root_mean_squared_error"
            )

            results[name] = {
                "cv_scores": -scores,
                "mean_cv_rmse": -scores.mean()
            }

        return results

    # --------------------------------------------------------
    # Train & Evaluate
    # --------------------------------------------------------
    def train_and_evaluate(self, model_name=None):
        results = {}
        models_to_run = (
            {model_name: None} if model_name else self.models
        )

        for name in models_to_run.keys():
            pipeline = get_pipeline(
                name,
                self.numerical_cols,
                self.categorical_cols
            )

            # Training data
            y_train_model = np.log1p(self.y_train) if name == "Ridge" else self.y_train
            y_test_model = np.log1p(self.y_test) if name == "Ridge" else self.y_test

            pipeline.fit(self.X_train, y_train_model)

            # Predictions
            train_pred = pipeline.predict(self.X_train)
            test_pred = pipeline.predict(self.X_test)

            if name == "Ridge":
                train_pred = np.expm1(train_pred)
                test_pred = np.expm1(test_pred)
                y_train_orig = np.expm1(y_train_model)
                y_test_orig = np.expm1(y_test_model)
            else:
                y_train_orig = y_train_model
                y_test_orig = y_test_model

            # Metrics
            rmse_train = root_mean_squared_error(y_train_orig, train_pred)
            r2_train = r2_score(y_train_orig, train_pred)

            rmse_test = root_mean_squared_error(y_test_orig, test_pred)
            r2_test = r2_score(y_test_orig, test_pred)

            results[name] = {
                "rmse_train": rmse_train,
                "r2_train": r2_train,
                "rmse_test": rmse_test,
                "r2_test": r2_test
            }

            # Save full pipeline
            model_path = os.path.join(self.save_dir, f"{name}_pipeline.pkl")
            joblib.dump(pipeline, model_path)
            print(f"Saved {name} model to: {model_path}")

        return results




