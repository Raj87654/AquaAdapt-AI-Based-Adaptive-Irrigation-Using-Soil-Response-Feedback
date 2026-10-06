"""
AquaAdapt - CatBoost Regression Training Script
=================================================
Trains a CatBoostRegressor to predict irrigation water requirement (litres).

The script automatically:
  - Loads the Excel dataset
  - Identifies the target column ('water_required')
  - Drops non-feature columns (IDs)
  - Detects categorical vs numerical features
  - Splits data 80/20
  - Trains CatBoost with native categorical feature support
  - Evaluates with MAE, RMSE, R2
  - Saves the trained model + feature metadata for the prediction script
"""

import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------
# 1. Configuration
# ----------------------------------------------
DATASET_PATH = "Tomato_Sandy_Clay_Loam_3000_Dataset.xlsx"
MODEL_PATH = "aquaadapt_catboost_model.cbm"
META_PATH = "model_metadata.json"
TARGET_COLUMN = "water_required"
DROP_COLUMNS = ["irrigation_episode_id"]   # ID column, not a feature
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ----------------------------------------------
# 2. Load dataset
# ----------------------------------------------
print("=" * 50)
print("  AQUAADAPT - MODEL TRAINING")
print("=" * 50)

df = pd.read_excel(DATASET_PATH)
print(f"\n[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ----------------------------------------------
# 3. Separate features and target
# ----------------------------------------------
df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"[OK] Target column : {TARGET_COLUMN}")
print(f"[OK] Feature count  : {X.shape[1]}")


# ----------------------------------------------
# 4. Detect categorical vs numerical features
# ----------------------------------------------
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_features = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\n  Categorical features ({len(cat_features)}): {cat_features}")
print(f"  Numerical features  ({len(num_features)}): {num_features}")

# CatBoost needs cat feature indices
cat_feature_indices = [X.columns.get_loc(c) for c in cat_features]

# Store unique values for each categorical feature (needed by predict script)
cat_unique_values = {col: sorted(X[col].unique().tolist()) for col in cat_features}


# ----------------------------------------------
# 5. Train / test split
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n[OK] Train set: {X_train.shape[0]} rows")
print(f"[OK] Test  set: {X_test.shape[0]} rows")


# ----------------------------------------------
# 6. Train CatBoostRegressor
# ----------------------------------------------
print("\n>>> Training CatBoost model ...\n")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    cat_features=cat_feature_indices,
    verbose=200,                       # print every 200 iterations
    random_seed=RANDOM_STATE,
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
print("\n[OK] Training complete!")


# ----------------------------------------------
# 7. Evaluate on test set
# ----------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("  EVALUATION RESULTS (Test Set)")
print("=" * 50)
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R2   : {r2:.4f}")
print("=" * 50)


# ----------------------------------------------
# 8. Save model + metadata
# ----------------------------------------------
model.save_model(MODEL_PATH)
print(f"\n[OK] Model saved -> {MODEL_PATH}")

metadata = {
    "feature_names": X.columns.tolist(),
    "cat_features": cat_features,
    "cat_feature_indices": cat_feature_indices,
    "num_features": num_features,
    "cat_unique_values": cat_unique_values,
    "target_column": TARGET_COLUMN,
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Metadata saved -> {META_PATH}")
print("\n[DONE] All done! You can now run predict.py to make predictions.\n")
