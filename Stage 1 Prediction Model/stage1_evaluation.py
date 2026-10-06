"""
AquaAdapt — Stage 1 Evaluation Script
=======================================
Loads the saved evaluation results and displays them.
Can also re-evaluate the model on the test set if needed.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(PROJECT_ROOT, "Stage 1 Prediction Model", "stage1_results.json")
META_PATH = os.path.join(PROJECT_ROOT, "Stage 1 Prediction Model", "model_metadata.json")


def load_evaluation() -> dict:
    """Load saved Stage 1 evaluation results."""
    if not os.path.exists(EVAL_PATH):
        raise FileNotFoundError(
            f"Evaluation results not found at {EVAL_PATH}. "
            "Please train the model first: python 'Stage 1 Prediction Model/train.py'"
        )
    with open(EVAL_PATH, "r") as f:
        return json.load(f)


def display_evaluation():
    """Print Stage 1 evaluation results."""
    results = load_evaluation()

    print("=" * 60)
    print("  AQUAADAPT — STAGE 1 EVALUATION RESULTS")
    print("=" * 60)

    metrics = results.get("metrics", {})
    print(f"\n  MAE  : {metrics.get('mae', 'N/A')}")
    print(f"  MSE  : {metrics.get('mse', 'N/A')}")
    print(f"  RMSE : {metrics.get('rmse', 'N/A')}")
    print(f"  R²   : {metrics.get('r2', 'N/A')}")

    print("\n  Feature Importance:")
    for feat, imp in sorted(
        results.get("feature_importance", {}).items(),
        key=lambda x: -x[1],
    ):
        print(f"    {feat:30s} : {imp:.2f}")

    residuals = results.get("residual_stats", {})
    print(f"\n  Residual Statistics:")
    print(f"    Mean         : {residuals.get('mean', 'N/A')}")
    print(f"    Std          : {residuals.get('std', 'N/A')}")
    print(f"    Max Absolute : {residuals.get('max_abs', 'N/A')}")

    info = results.get("dataset_info", {})
    print(f"\n  Dataset: {info.get('crop', '')} — {info.get('soil', '')}")
    print(f"  Records: {info.get('total_records', '')}")
    print("=" * 60)


if __name__ == "__main__":
    display_evaluation()
