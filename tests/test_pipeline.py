# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 01-12-2025


import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.pipelines.price_pipeline import (
    get_pipeline,
    LogTransformer,
    CustomOutlierCappingTransformer
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "living_area": [100, 150, 200],
        "build_year_cat": ["80-90", "90-00", "00-10"],
        "building_state": ["GOOD", "GOOD", "RENOVATE"],
        "locality_name": ["Brussels", "Ghent", "Antwerp"],
        "property_type": ["HOUSE", "APARTMENT", "HOUSE"],
        "province": ["Brussels", "Flanders", "Flanders"],
        "swimming_pool": ["0", "1", "0"],
        "garden": ["1", "0", "1"],
        "terrace": ["1", "1", "0"],
        "facades": ["2", "4", "3"],
        "postal_code": ["1000", "9000", "2000"],
        "number_bedrooms": ["2", "3", "4"],
        "build_year": ["1990", "2005", "1980"]
    })


@pytest.fixture
def columns():
    numerical = ["living_area"]
    categorical = [
        "build_year_cat", "building_state", "locality_name",
        "property_type", "province",
        "swimming_pool", "garden", "terrace", "facades",
        "postal_code", "number_bedrooms", "build_year"
    ]
    return numerical, categorical


# -----------------------------
# Test pipeline creation
# -----------------------------
def test_get_pipeline_ridge(columns):
    numerical, categorical = columns
    pipe = get_pipeline("Ridge", numerical, categorical)
    assert isinstance(pipe, Pipeline)


def test_get_pipeline_randomforest(columns):
    numerical, categorical = columns
    pipe = get_pipeline("RandomForest", numerical, categorical)
    assert isinstance(pipe, Pipeline)


# -----------------------------
# Test transformers independently
# -----------------------------
def test_log_transformer():
    arr = np.array([[10.0], [100.0], [1000.0]])
    tr = LogTransformer(columns=[0])
    out = tr.transform(arr.copy())
    assert np.allclose(out[:, 0], np.log1p(arr[:, 0]))


def test_outlier_capping():
    arr = np.array([[1], [2], [100]])  # 100 is an outlier
    tr = CustomOutlierCappingTransformer(columns=[0])
    tr.fit(arr)
    out = tr.transform(arr.copy())
    assert out[2, 0] <= tr.bounds[0][1]  # capped at upper bound


# -----------------------------
# End-to-end test
# -----------------------------
def test_pipeline_fit_predict(sample_data, columns):
    numerical, categorical = columns

    pipeline = get_pipeline("RandomForest", numerical, categorical)

    # Fake target
    y = np.array([200000, 250000, 300000])

    pipeline.fit(sample_data, y)
    preds = pipeline.predict(sample_data)

    assert len(preds) == 3
    assert np.all(np.isfinite(preds))

