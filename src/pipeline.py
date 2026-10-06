"""
AquaAdapt — Adaptive Irrigation Pipeline
==========================================
Connects Stage 1 (initial prediction) with Stage 2 (adaptive prediction)
into a single end-to-end workflow.

Workflow:
  1. Collect initial field conditions
  2. Stage 1 → predict initial water requirement
  3. Apply a small test dose
  4. Observe soil response after the test dose
  5. Stage 2 → predict final adaptive water requirement
  6. Calculate adjustment
"""

from src.stage1_model import predict_stage1
from src.stage2_model import predict_stage2


def run_pipeline(
    # --- Initial field conditions (Stage 1 inputs) ---
    soil_type: str,
    crop_type: str,
    crop_stage: str,
    soil_moisture_before: float,
    temperature_before: float,
    humidity_before: float,
    rainfall_forecast_before: float,
    wind_speed_before: float,
    solar_radiation_before: float,
    # --- Post-test-dose observations (Stage 2 extra inputs) ---
    test_dose_applied: float,
    soil_moisture_after: float,
    temperature_after: float,
    humidity_after: float,
    rainfall_forecast_after: float,
    wind_speed_after: float,
    solar_radiation_after: float,
    moisture_gain: float,
) -> dict:
    """
    Execute the complete AquaAdapt adaptive irrigation pipeline.

    Returns
    -------
    dict with keys:
        initial_water_required  : float   Stage 1 prediction (L)
        test_dose_applied       : float   test dose used (L)
        moisture_gain           : float   observed moisture change (%)
        final_water_required    : float   Stage 2 adaptive prediction (L)
        adjustment_litres       : float   difference (initial − final) (L)
        adjustment_percent      : float   percentage adjustment (%)
    """

    # ---- STEP 1: Stage 1 initial prediction ----
    initial_water = predict_stage1(
        soil_type=soil_type,
        crop_type=crop_type,
        crop_stage=crop_stage,
        soil_moisture_before=soil_moisture_before,
        temperature_before=temperature_before,
        humidity_before=humidity_before,
        rainfall_forecast_before=rainfall_forecast_before,
        wind_speed_before=wind_speed_before,
        solar_radiation_before=solar_radiation_before,
    )

    # ---- STEP 2: Stage 2 adaptive prediction ----
    final_water = predict_stage2(
        soil_type=soil_type,
        crop_type=crop_type,
        crop_stage=crop_stage,
        soil_moisture_before=soil_moisture_before,
        temperature_before=temperature_before,
        humidity_before=humidity_before,
        rainfall_forecast_before=rainfall_forecast_before,
        wind_speed_before=wind_speed_before,
        solar_radiation_before=solar_radiation_before,
        test_dose_applied=test_dose_applied,
        soil_moisture_after=soil_moisture_after,
        temperature_after=temperature_after,
        humidity_after=humidity_after,
        rainfall_forecast_after=rainfall_forecast_after,
        wind_speed_after=wind_speed_after,
        solar_radiation_after=solar_radiation_after,
        moisture_gain=moisture_gain,
    )

    # ---- STEP 3: Compute adjustment ----
    adjustment = initial_water - final_water
    if initial_water != 0:
        adjustment_pct = (adjustment / initial_water) * 100
    else:
        adjustment_pct = 0.0

    return {
        "initial_water_required": round(initial_water, 2),
        "test_dose_applied": round(test_dose_applied, 2),
        "moisture_gain": round(moisture_gain, 2),
        "final_water_required": round(final_water, 2),
        "adjustment_litres": round(adjustment, 2),
        "adjustment_percent": round(adjustment_pct, 2),
    }
