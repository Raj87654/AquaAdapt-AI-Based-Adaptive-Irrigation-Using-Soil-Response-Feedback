"""
AquaAdapt — Stage 2 Adaptive Prediction Model
===============================================
Wraps the EXISTING Stage 2 Extra Trees model as a reusable prediction function.

Stage 2 takes the SAME pre-irrigation conditions PLUS post-test-dose
observations and predicts the final adaptive water requirement.

Additional input features (on top of Stage 1 inputs):
  - test_dose_applied
  - soil_moisture_after, temperature_after, humidity_after
  - rainfall_forecast_after, wind_speed_after, solar_radiation_after
  - moisture_gain

Output:
  - final_adaptive_water_required (litres)
"""

import os
import json
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths  —  reuse the existing Stage 2 model in its original location
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE2_DIR = os.path.join(_BASE_DIR, "Stage 2 Prediction Model")

MODEL_PATH = os.path.join(_STAGE2_DIR, "stage2_extra_trees_model.joblib")
META_PATH = os.path.join(_STAGE2_DIR, "model_metadata.json")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model = None
_meta = None


def _load():
    """Load Stage 2 model + metadata once."""
    global _model, _meta

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Stage 2 model not found at {MODEL_PATH}. "
            "The existing Stage 2 model file may have been moved."
        )
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Stage 2 metadata not found at {META_PATH}. "
            "The existing Stage 2 metadata file may have been moved."
        )

    _model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r") as f:
        _meta = json.load(f)


def get_metadata() -> dict:
    """Return Stage 2 metadata (feature names, etc.)."""
    _load()
    return _meta


def predict_stage2(
    # --- Same pre-irrigation inputs as Stage 1 ---
    soil_type: str,
    crop_type: str,
    crop_stage: str,
    soil_moisture_before: float,
    temperature_before: float,
    humidity_before: float,
    rainfall_forecast_before: float,
    wind_speed_before: float,
    solar_radiation_before: float,
    # --- Post-test-dose observations ---
    test_dose_applied: float,
    soil_moisture_after: float,
    temperature_after: float,
    humidity_after: float,
    rainfall_forecast_after: float,
    wind_speed_after: float,
    solar_radiation_after: float,
    moisture_gain: float,
) -> float:
    """
    Predict the final adaptive water requirement (litres) using the
    existing Stage 2 Extra Trees model.

    Parameters
    ----------
    (pre-irrigation — same as Stage 1)
    soil_type, crop_type, crop_stage,
    soil_moisture_before, temperature_before, humidity_before,
    rainfall_forecast_before, wind_speed_before, solar_radiation_before

    (post-test-dose observations)
    test_dose_applied : float    litres of test dose applied
    soil_moisture_after : float  soil moisture after test dose (%)
    temperature_after : float    temperature after test dose (°C)
    humidity_after : float       humidity after test dose (%)
    rainfall_forecast_after : float  forecast after observation window (mm)
    wind_speed_after : float     wind speed after test dose (km/h)
    solar_radiation_after : float  solar radiation after test dose (W/m²)
    moisture_gain : float        change in soil moisture (after − before) (%)

    Returns
    -------
    float
        Predicted final adaptive water requirement in litres.
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
        "test_dose_applied": test_dose_applied,
        "soil_moisture_after": soil_moisture_after,
        "temperature_after": temperature_after,
        "humidity_after": humidity_after,
        "rainfall_forecast_after": rainfall_forecast_after,
        "wind_speed_after": wind_speed_after,
        "solar_radiation_after": solar_radiation_after,
        "moisture_gain": moisture_gain,
    }

    input_df = pd.DataFrame([row], columns=feature_names)
    prediction = _model.predict(input_df)[0]
    return float(prediction)
