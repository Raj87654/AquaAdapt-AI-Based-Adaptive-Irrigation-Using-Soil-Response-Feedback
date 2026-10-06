"""
AquaAdapt - Stage 2 Extra Trees Regression Training Script
=================================================
Trains an ExtraTreesRegressor to predict adaptive irrigation water requirement (litres).
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ----------------------------------------------
# 1. Configuration
# ----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2_DIR = os.path.join(PROJECT_ROOT, "Stage 2 Prediction Model")

DATASET_PATH = os.path.join(STAGE2_DIR, "Tomato_Sandy_Clay_Loam_3000_Dataset.xlsx")
MODEL_PATH = os.path.join(STAGE2_DIR, "stage2_extra_trees_model.joblib")
META_PATH = os.path.join(STAGE2_DIR, "model_metadata.json")
TARGET_COLUMN = "water_required"
DROP_COLUMNS = ["irrigation_episode_id"]   # ID column, not a feature
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ----------------------------------------------
# 2. Load dataset
# ----------------------------------------------
print("=" * 50)
print("  AQUAADAPT - STAGE 2 MODEL TRAINING (Extra Trees)")
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

# Store unique values for each categorical feature
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
# 6. Build Pipeline and Train ExtraTreesRegressor
# ----------------------------------------------
print("\n>>> Training Stage 2 Extra Trees model ...\n")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=1))
])

model.fit(X_train, y_train)
print("\n[OK] Training complete!")


# ----------------------------------------------
# 7. Evaluate on test set
# ----------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("  EVALUATION RESULTS (Test Set)")
print("=" * 50)
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R2   : {r2:.4f}")
print("=" * 50)


# ----------------------------------------------
# 8. Save model + metadata
# ----------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"\n[OK] Model saved -> {MODEL_PATH}")

metadata = {
    "feature_names": X.columns.tolist(),
    "cat_features": cat_features,
    "num_features": num_features,
    "cat_unique_values": cat_unique_values,
    "target_column": TARGET_COLUMN,
    "evaluation": {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Metadata saved -> {META_PATH}")
print("\n[DONE] All done!\n")
