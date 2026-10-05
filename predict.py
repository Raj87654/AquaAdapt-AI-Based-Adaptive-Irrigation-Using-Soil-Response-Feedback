"""
AquaAdapt - CatBoost Prediction Script
========================================
Loads the trained CatBoost model and asks the user for field conditions
through the terminal, then predicts the required irrigation water (litres).
"""

import json
import pandas as pd
from catboost import CatBoostRegressor


# ----------------------------------------------
# 1. Configuration
# ----------------------------------------------
MODEL_PATH = "aquaadapt_catboost_model.cbm"
META_PATH = "model_metadata.json"


# ----------------------------------------------
# 2. Load model + metadata
# ----------------------------------------------
with open(META_PATH, "r") as f:
    meta = json.load(f)

model = CatBoostRegressor()
model.load_model(MODEL_PATH)

feature_names   = meta["feature_names"]
cat_features    = meta["cat_features"]
num_features    = meta["num_features"]
cat_unique_vals = meta["cat_unique_values"]


# ----------------------------------------------
# 3. Friendly display names  (snake_case -> Title)
# ----------------------------------------------
def pretty(col_name: str) -> str:
    """Convert 'soil_moisture_before' -> 'Soil Moisture Before'."""
    return col_name.replace("_", " ").title()


# ----------------------------------------------
# 4. Collect user input
# ----------------------------------------------
print("=" * 50)
print("  AQUAADAPT - WATER REQUIREMENT PREDICTOR")
print("=" * 50)
print("\nPlease enter the field conditions below.\n")

user_input = {}

for feat in feature_names:
    if feat in cat_features:
        # Show valid options for categorical features
        options = cat_unique_vals[feat]
        options_str = ", ".join(options)
        while True:
            val = input(f"  {pretty(feat)} [{options_str}]: ").strip()
            if val in options:
                user_input[feat] = val
                break
            else:
                print(f"    [!] Invalid. Choose from: {options_str}")
    else:
        # Numerical feature
        while True:
            val = input(f"  {pretty(feat)}: ").strip()
            try:
                user_input[feat] = float(val)
                break
            except ValueError:
                print("    [!] Please enter a valid number.")


# ----------------------------------------------
# 5. Build DataFrame and predict
# ----------------------------------------------
input_df = pd.DataFrame([user_input], columns=feature_names)
prediction = model.predict(input_df)[0]


# ----------------------------------------------
# 6. Display results
# ----------------------------------------------
print("\n" + "=" * 50)
print("  AQUAADAPT WATER PREDICTION")
print("=" * 50)

print("\n  Input:")
for feat in feature_names:
    print(f"    {pretty(feat)}: {user_input[feat]}")

print("\n" + "-" * 50)
print(f"\n  >> Predicted Water Required: {prediction:.2f} L")
print("\n" + "=" * 50 + "\n")
