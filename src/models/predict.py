# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 01-12-2025


import joblib
import numpy as np
import pandas as pd
import os


def load_model(model_path: str):
    """Load a trained pipeline (.pkl)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def prepare_input(data):
    """
    Convert input data (dict or DataFrame) into a DataFrame
    without altering column types (trainer already handled casts).
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, dict):
        return pd.DataFrame([data])
    else:
        raise ValueError("Input data must be a pandas DataFrame or a dict.")


def predict(data, model_path: str):
    """
    Predict prices using a trained pipeline.

    Automatically applies inverse log-transform if the model is Ridge.
    """
    pipeline = load_model(model_path)
    df = prepare_input(data)

    # detect model type (Ridge or tree)
    model_name = os.path.basename(model_path).split("_")[0]

    preds = pipeline.predict(df)

    if model_name == "Ridge":
        preds = np.expm1(preds)  # invert log1p

    return preds


if __name__ == "__main__":
    # Example manual usage
    sample = {
        "living_area": 120,
        "postal_code": "9000",
        "number_bedrooms": "3",
        "build_year": "2022",
        "build_year_cat": "2020s",
        "building_state": "Excellent",
        "locality_name": "Gent",
        "property_type": "House",
        "province": "East Flanders",
        "swimming_pool": "0",
        "garden": "1",
        "terrace": "1",
        "facades": "2",
    }

    model_path = "models/Ridge_pipeline.pkl" # FILEPATH AND FILENAME MUST BE ADJUSTED
    prediction = predict(sample, model_path)
    print(f"Predicted price: € {round(prediction[0], 3)} ± € 169k") # MUST BE ADJUSTED

