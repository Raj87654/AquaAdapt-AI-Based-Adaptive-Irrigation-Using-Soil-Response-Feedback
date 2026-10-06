"""
AquaAdapt — Stage 1 Initial Extra Trees Model
============================================
Provides a reusable prediction function for the Stage 1 model.
Stage 1 predicts the initial water requirement based on pre-irrigation
field conditions ONLY (before any test dose is applied).

Input features:
  - soil_type, crop_type, crop_stage          (categorical)
  - soil_moisture_before, temperature_before,
    humidity_before, rainfall_forecast_before,
    wind_speed_before, solar_radiation_before  (numerical)

Output:
  - initial_water_required (litres)
"""

import os
import json
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE_DIR, "Stage 1 Prediction Model", "stage1_extra_trees_model.joblib")
META_PATH = os.path.join(_BASE_DIR, "Stage 1 Prediction Model", "model_metadata.json")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model = None
_meta = None


def _load():
    """Load model + metadata once."""
    global _model, _meta

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Stage 1 model not found at {MODEL_PATH}. "
            "Please train it first by running:  python 'Stage 1 Prediction Model/train.py'"
        )
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Stage 1 metadata not found at {META_PATH}. "
            "Please train it first by running:  python 'Stage 1 Prediction Model/train.py'"
        )

    _model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r") as f:
        _meta = json.load(f)


def get_metadata() -> dict:
    """Return Stage 1 metadata (feature names, evaluation metrics, etc.)."""
    _load()
    return _meta


def predict_stage1(
    soil_type: str,
    crop_type: str,
    crop_stage: str,
    soil_moisture_before: float,
    temperature_before: float,
    humidity_before: float,
    rainfall_forecast_before: float,
    wind_speed_before: float,
    solar_radiation_before: float,
) -> float:
    """
    Predict the initial water requirement (litres) using the Stage 1
    Extra Trees model.

    Parameters
    ----------
    soil_type : str               e.g. "Sandy clay loam"
    crop_type : str               e.g. "Tomato"
    crop_stage : str              one of "Seedling", "Vegetative", "Flowering", "Maturity"
    soil_moisture_before : float  soil moisture reading before irrigation (%)
    temperature_before : float    ambient temperature (°C)
    humidity_before : float       relative humidity (%)
    rainfall_forecast_before : float  forecasted rainfall (mm)
    wind_speed_before : float     wind speed (km/h)
    solar_radiation_before : float  solar radiation (W/m²)

    Returns
    -------
    float
        Predicted initial water requirement in litres.
    """
    _load()

    feature_names = _meta["feature_names"]

    row = {
        "soil_type": soil_type,
        "crop_type": crop_type,
        "crop_stage": crop_stage,
        "soil_moisture_before": soil_moisture_before,
        "temperature_before": temperature_before,
        "humidity_before": humidity_before,
        "rainfall_forecast_before": rainfall_forecast_before,
        "wind_speed_before": wind_speed_before,
        "solar_radiation_before": solar_radiation_before,
    }

    input_df = pd.DataFrame([row], columns=feature_names)
    prediction = _model.predict(input_df)[0]
    return float(prediction)
