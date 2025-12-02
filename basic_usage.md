# Basic Usage of predict.py
## 1. Predict one sample

Running script or notebook must be in the level of project root or have the `sys.path`setup. The Ridge model uses log-transformed prices.

```python
from src.models.predict import predict

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

model_path = "models/Ridge_pipeline.pkl"
prediction = predict(sample, model_path)
print("Predicted price:", prediction[0])
```

## 2. Use RandomForest or XGBoost

The tree model uses raw price unit (in Euro).

```python
from src.models.predict import predict

pred = predict(sample, model_path="models/RandomForest_pipeline.pkl") # --> RandomForest or XGBoost
print(pred)
```

## 3. Batch prediction
If there are multiple listings to be price-predicted.

```python
import pandas as pd
from src.models.predict import predict

df = pd.read_csv("new_property_data.csv") # --> adjust filepath and filename
preds = predict(df, "models/XGBoost_pipeline.pkl") # --> RandomForest or XGBoost

df["predicted_price"] = preds
df.head()
```

## 4. From a script
If there is a standalone script to run the prediction, for example `models/run_prediction.py`.

```python
import sys, os
sys.path.append(os.path.abspath("."))

from src.models.predict import predict
import pandas as pd

data = pd.read_csv("new_data.csv") # --> adjust filepath and filename
predictions = predict(data, "models/Ridge_pipeline.pkl") # --> RandomForest or XGBoost
print(predictions)
```

Then from the CLI, run the following command:

```bash
python scripts/run_prediction.py
```

## 5. Directly from CLI
There is a `__main__` block inside of `predict.py`, so it can be run directly in a CLI with the following command:

```bash
python predict.py
```



