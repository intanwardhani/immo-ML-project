# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 03-12-2025


import os
import joblib
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Which models to version
MODELS = [
    "RandomForest_pipeline.pkl",
    "Ridge_pipeline.pkl",
    "XGBoost_pipeline.pkl"
]

def create_version_tag():
    """Create a version tag based on date."""
    return datetime.now().strftime("%Y%m%d")

def version_model(model_name: str):
    """Load a model and resave it as a versioned file."""
    original_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(original_path):
        print(f"Model not found: {original_path}")
        return

    version_tag = create_version_tag()
    base_name = model_name.replace(".pkl", "")
    new_name = f"{base_name}_v{version_tag}.pkl"
    new_path = os.path.join(MODEL_DIR, new_name)

    print(f"Loading {original_path}...")
    pipeline = joblib.load(original_path)

    print(f"Saving versioned model → {new_path}")
    joblib.dump(pipeline, new_path)

    # Also update LATEST alias
    latest_alias = os.path.join(MODEL_DIR, f"{base_name}_latest.pkl")
    if os.path.exists(latest_alias):
        os.remove(latest_alias)
    os.symlink(new_name, latest_alias)

    print(f"Updated symlink: {latest_alias} → {new_name}")


if __name__ == "__main__":
    print("Versioning all models...")
    for m in MODELS:
        version_model(m)
    print("Done creating versioned pipelines!")
