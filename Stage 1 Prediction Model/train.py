"""
AquaAdapt — Stage 1 Extra Trees Regression Training
===================================================
Trains an ExtraTreesRegressor to predict the INITIAL irrigation water
requirement (litres) using ONLY pre-irrigation field conditions.

Uses the 'Stage1_Initial_Model' sheet from the Excel dataset.

Outputs:
  - stage1_extra_trees_model.joblib   (trained model)
  - model_metadata.json        (features + evaluation metrics)
  - stage1_results.json     (detailed evaluation results)
"""

import os
import sys
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

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_ROOT, "Stage 1 Prediction Model",
    "Tomato_Sandy_Clay_Loam_3000_Dataset.xlsx"
)
SHEET_NAME = "Stage1_Initial_Model"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Stage 1 Prediction Model")
MODEL_PATH = os.path.join(MODEL_DIR, "stage1_extra_trees_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

EVAL_DIR = os.path.join(PROJECT_ROOT, "Stage 1 Prediction Model")
EVAL_PATH = os.path.join(EVAL_DIR, "stage1_results.json")

TARGET_COLUMN = "water_required"
DROP_COLUMNS = ["irrigation_episode_id"]  # ID column, not a feature

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 2. Ensure output directories exist
# ---------------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Load dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("  AQUAADAPT — STAGE 1 MODEL TRAINING (Extra Trees)")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print(f"\n[ERROR] Dataset not found: {DATASET_PATH}")
    sys.exit(1)

df = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
print(f"\n[OK] Dataset loaded from sheet '{SHEET_NAME}': "
      f"{df.shape[0]} rows, {df.shape[1]} columns")
print(f"[OK] Columns: {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 4. Separate features and target
# ---------------------------------------------------------------------------
df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"\n[OK] Target column : {TARGET_COLUMN}")
print(f"[OK] Feature count : {X.shape[1]}")

# ---------------------------------------------------------------------------
# 5. Detect categorical vs numerical features
# ---------------------------------------------------------------------------
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_features = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\n  Categorical features ({len(cat_features)}): {cat_features}")
print(f"  Numerical features  ({len(num_features)}): {num_features}")

# ---------------------------------------------------------------------------
# 6. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n[OK] Train set: {X_train.shape[0]} rows")
print(f"[OK] Test  set: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# 7. Build Pipeline and Train ExtraTreesRegressor
# ---------------------------------------------------------------------------
print("\n>>> Training Stage 1 Extra Trees model ...\n")

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

# ---------------------------------------------------------------------------
# 8. Evaluate on test set
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  STAGE 1 EVALUATION RESULTS (Test Set)")
print("=" * 60)
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"\n[OK] Model saved -> {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 10. Save metadata (features + evaluation metrics)
# ---------------------------------------------------------------------------
metadata = {
    "feature_names": X.columns.tolist(),
    "cat_features": cat_features,
    "num_features": num_features,
    "target_column": TARGET_COLUMN,
    "evaluation": {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    },
    "dataset_info": {
        "sheet": SHEET_NAME,
        "total_records": int(df.shape[0]),
        "train_records": int(X_train.shape[0]),
        "test_records": int(X_test.shape[0]),
        "crop": "Tomato",
        "soil": "Sandy clay loam",
    },
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Metadata saved -> {META_PATH}")

# ---------------------------------------------------------------------------
# 11. Save detailed evaluation results
# ---------------------------------------------------------------------------
comparison_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "residual": y_test.values - y_pred,
})

eval_results = {
    "metrics": metadata["evaluation"],
    "residual_stats": {
        "mean": round(comparison_df["residual"].mean(), 4),
        "std": round(comparison_df["residual"].std(), 4),
        "max_abs": round(comparison_df["residual"].abs().max(), 4),
        "min": round(comparison_df["residual"].min(), 4),
        "max": round(comparison_df["residual"].max(), 4),
    },
    "dataset_info": metadata["dataset_info"],
}

with open(EVAL_PATH, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"[OK] Evaluation results saved -> {EVAL_PATH}")

print("\n[DONE] Stage 1 training complete!")
print("You can now run the Streamlit app:  streamlit run app.py\n")
