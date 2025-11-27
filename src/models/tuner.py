# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 27-11-2025

import os
import joblib
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .trainer import ModelTrainer   # import my trainer class

logger = logging.getLogger(__name__)


class ModelTuner:
    """
    (Hyperparameter tuning is OPTIONAL)
    Hyperparameter tuning class using GridSearchCV.
    
    This class reuses the preprocessing pipeline built inside ModelTrainer
    (imputation, scaling, one-hot encoding) and applies hyperparameter
    tuning on top of that.

    Steps:
    -------
    1. Inherit preprocessing from ModelTrainer
    2. Define parameter grids for selected models
    3. Run GridSearchCV
    4. Save the best tuned model
    """

    def __init__(self, trainer: ModelTrainer, model_name: str, output_dir="models/"):
        """
        Parameters
        ----------
        trainer : ModelTrainer
            An already initialized ModelTrainer instance
            (containing X_train, X_test, preprocessing pipeline).
        model_name : str
            Name of the model to tune (must be one of the keys in trainer.models).
        output_dir : str
            Directory where tuned models should be saved.
        """
        
        self.trainer = trainer
        self.model_name = model_name
        self.preprocessor = trainer._get_preprocessor(model_name)
        self.X_train = trainer.X_train
        self.y_train = trainer.y_train
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Parameter grids for each model
    # ------------------------------------------------------------
    def get_param_grids(self):

        ridge_params = {
            "regressor__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
        }

        rf_params = {
            "regressor__n_estimators": [200, 400],
            "regressor__max_depth": [None, 20, 40],
            "regressor__min_samples_split": [2, 10],
        }

        xgb_params = {
            "regressor__n_estimators": [200, 400],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [4, 6],
            "regressor__subsample": [0.8, 1.0],
        }

        return {
            "Ridge": ridge_params,
            "RandomForest": rf_params,
            "XGBoost": xgb_params
        }

    # ------------------------------------------------------------
    # Tune a single model
    # ------------------------------------------------------------
    def tune_model(self, model_name):
        """
        Perform GridSearchCV for a single model.

        Parameters
        ----------
        model_name : str
            Name of the model from ModelTrainer.models

        Returns
        -------
        GridSearchCV object
        """

        if model_name not in self.trainer.models:
            raise ValueError(f"Unknown model: {model_name}")

        regressor = self.trainer.models[model_name]
        param_grid = self.get_param_grids()[model_name]

        logger.info(f"Starting tuning for: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", regressor)
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f"Best params for {model_name}: {grid.best_params_}")
        logger.info(f"Best CV RMSE: {-grid.best_score_:.4f}")

        # Save model
        model_path = os.path.join(self.output_dir, f"{model_name}_best.pkl")
        joblib.dump(grid.best_estimator_, model_path)
        logger.info(f"Saved best {model_name} to: {model_path}")
        
        # -------- SAVE BEST FULL PIPELINE --------
        model_path = os.path.join(self.output_dir, f"{model_name}_best.pkl")
        joblib.dump(grid.best_estimator_, model_path)
        logger.info(f"Saved best {model_name} to: {model_path}")
        # ------------------------------------------

        return grid

    # ------------------------------------------------------------
    # Tune all models
    # ------------------------------------------------------------
    def tune_all(self):
        """
        Run GridSearchCV for all models defined in ModelTrainer.

        Returns
        -------
        dict
            Mapping of model_name → best GridSearchCV object
        """
        results = {}
        for name in self.trainer.models.keys():
            results[name] = self.tune_model(name)
        return results
