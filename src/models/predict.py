# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 27-11-2025


import joblib
import pandas as pd
import os


def load_model(model_path: str):
    """Load the full saved pipeline (preprocessing + model)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict(model_path: str, new_data: pd.DataFrame):
    """
    Predict price using a saved pipeline.

    Parameters
    ----------
    model_path : str
        Path to saved pipeline file (.pkl)
    new_data : pd.DataFrame
        Raw new data (same columns as training data)

    Returns
    -------
    np.ndarray
        Predicted prices
    """
    pipeline = load_model(model_path)
    return pipeline.predict(new_data)


if __name__ == "__main__":
    example = pd.DataFrame({
        "living_area": [80],
        "postal_code": [1000],
        "number_bedrooms": [2],
        "property_type": ["APARTMENT"],
        "province": ["Bruxelles"],
        "building_state": ["GOOD"],
        "swimming_pool": [0],
        "garden": [1],
        "terrace": [1],
        "facades": [2],
        "build_year": [1990],
        "build_year_cat": ["1990s"],
        "locality_name": ["Ixelles"]
    })

    model_file = "models/Ridge_pipeline.pkl"
    y_pred = predict(model_file, example)
    print("Predicted price:", y_pred)

