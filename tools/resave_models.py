# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 03-12-2025


import os
import joblib
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODELS = [
    "RandomForest_pipeline.pkl",
    "Ridge_pipeline.pkl",
    "XGBoost_pipeline.pkl"
]

def create_version_tag():
    return datetime.now().strftime("%Y%m%d")

def version_model(model_name: str):
    original_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(original_path):
        print(f"❌ Model not found: {original_path}")
        return

    version_tag = create_version_tag()
    base = model_name.replace(".pkl", "")
    new_name = f"{base}_v{version_tag}.pkl"
    new_path = os.path.join(MODEL_DIR, new_name)

    print(f"📦 Loading  {original_path}")
    pipeline = joblib.load(original_path)

    print(f"💾 Saving versioned → {new_path}")
    joblib.dump(pipeline, new_path)

    # Save a REAL file, not symlink
    latest_path = os.path.join(MODEL_DIR, f"{base}_latest.pkl")
    print(f"🔄 Updating latest → {latest_path}")
    joblib.dump(pipeline, latest_path)

    print(f"✔ Updated: {latest_path}\n")


if __name__ == "__main__":
    print("🔥 Versioning all models ...\n")
    for m in MODELS:
        version_model(m)
    print("🎉 Done! All latest models saved as real files.")
