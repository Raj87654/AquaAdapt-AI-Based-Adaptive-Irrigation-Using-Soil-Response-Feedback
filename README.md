# AquaAdapt 🌱💧

### AI-Based Adaptive Irrigation Using Soil Response Feedback

AquaAdapt is a smart irrigation system that uses **AI, sensor data, and real-time soil feedback** to make better irrigation decisions and reduce water wastage. 
---

## 💡 Problem Statement

Traditional irrigation systems apply a fixed amount of water based on static estimates, ignoring the actual soil response. This leads to **water wastage** and **suboptimal crop growth**. There is a need for an intelligent system that can **observe, predict, test, and adapt** irrigation decisions using real-time feedback.

## 🎯 Objective

* Reduce unnecessary water usage
* Learn from actual field conditions in real time
* Improve irrigation predictions through adaptive feedback
* Combine AI predictions with real-world sensor feedback
* Provide reliable and data-driven irrigation decisions

> **AquaAdapt learns from the field instead of relying only on assumptions.**

---

## 🔄 System Workflow

```
Observe → Predict → Test → Observe Response → Adapt → Irrigate
```

### How It Works

1. **Observe** — Collect soil moisture and weather conditions using sensors.
2. **Predict (Stage 1)** — AI predicts the approximate initial water requirement using CatBoost.
3. **Test** — A small controlled amount of water (test dose) is applied to the field.
4. **Observe Response** — Sensors measure how the soil absorbs and responds to the test dose.
5. **Adapt (Stage 2)** — The adaptive model uses the observed soil response to correct the initial prediction.
6. **Irrigate** — The system provides the final, more accurate irrigation quantity.

### Visual Flow

```
🌱 Initial Soil Conditions
    ↓
🤖 Stage 1 — Extra Trees Prediction
    ↓
💧 Initial Water Requirement
    ↓
🧪 Small Test Dose Applied
    ↓
📊 Observe Soil Response
    ↓
🧠 Stage 2 — Adaptive Prediction
    ↓
💧 Final Adaptive Water Requirement
```

### Example

Instead of immediately applying **300 litres**, the system performs a controlled test and observes the soil response. After learning from the feedback, the system may determine:

> **"Approximately 220 litres is sufficient under the current conditions."**

The goal is to make irrigation **adaptive, data-driven, and water-efficient**.

---

## 🧠 Stage 1 — Extra Trees Initial Model

* **Algorithm:** Extra Trees Regression
* **Input:** Pre-irrigation field conditions (soil type, crop type, crop stage, soil moisture, temperature, humidity, rainfall forecast, wind speed, solar radiation)
* **Output:** Initial water requirement (litres)
* **Dataset:** Tomato — Sandy clay loam (3,000 records)
* **Sheet:** `Stage1_Initial_Model`

Stage 1 uses only the **initial field conditions** to make a prediction, before any water is applied.

## 🧠 Stage 2 — Adaptive Prediction Model

* **Algorithm:** Extra Trees Regression
* **Input:** All Stage 1 inputs + post-test-dose observations (test dose applied, soil moisture after, temperature after, humidity after, rainfall forecast after, wind speed after, solar radiation after, moisture gain)
* **Output:** Final adaptive water requirement (litres)
* **Sheet:** `Stage2_Adaptive_Model`

Stage 2 takes the actual soil response **after the test dose** and adapts the prediction accordingly.

## 🔗 Stage 1 → Stage 2 Integration

The system runs both models as a pipeline:
1. User enters initial field conditions → Stage 1 predicts initial water
2. A test dose is applied and soil response is observed
3. All data (before + after) is fed to Stage 2 → Stage 2 predicts final adaptive water
4. The adjustment (difference between initial and final) is displayed

---

## 🛠️ Technology Stack

| Component    | Technology               |
|-------------|--------------------------|
| ML Model    | Extra Trees Regression     |
| Language    | Python 3.11              |
| Data        | Pandas, NumPy, OpenPyXL  |
| Evaluation  | Scikit-learn             |
| Frontend    | Streamlit                |
| Visualization | Matplotlib             |

---

## 📁 Project Structure

```
AquaAdapt/
│
├── app.py                        # Streamlit frontend
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore
│
├── data/                          # (reserved for future datasets)
│
├── src/
│   ├── __init__.py
│   ├── stage1_model.py            # Stage 1 prediction function
│   ├── stage2_model.py            # Stage 2 prediction function (wraps existing model)
│   └── pipeline.py                # Full Stage 1 → Stage 2 pipeline
│
├── Stage 1 Prediction Model/     # Stage 1 dataset, scripts, and model
│   ├── Tomato_Sandy_Clay_Loam_3000_Dataset.xlsx
│   ├── stage1_extra_trees_model.joblib  # Trained Stage 1 model
│   ├── model_metadata.json        # Stage 1 features + metrics
│   ├── stage1_results.json        # Saved evaluation metrics
│   ├── train.py                   # Stage 1 training script
│   └── stage1_evaluation.py       # Evaluation display script
│
└── Stage 2 Prediction Model/     # Stage 2 dataset, scripts, and model
    ├── stage2_extra_trees_model.joblib
    ├── model_metadata.json
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    └── Tomato_Sandy_Clay_Loam_3000_Dataset.xlsx
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Train the Stage 1 Model

```bash
python "Stage 1 Prediction Model/train.py"
```

This will:
- Load the dataset from the `Stage1_Initial_Model` sheet
- Train an Extra Trees model
- Evaluate with MAE, MSE, RMSE, R²
- Save the model to `Stage 1 Prediction Model/stage1_extra_trees_model.joblib`
- Save metadata + metrics to `Stage 1 Prediction Model/model_metadata.json`
- Save evaluation results to `Stage 1 Prediction Model/stage1_results.json`

### 3. Run the Frontend

```bash
streamlit run app.py
```

This will launch the Streamlit dashboard where you can:
1. Enter initial field conditions
2. Enter post-test-dose observations
3. Click **"Run AquaAdapt"**
4. See the complete Stage 1 → Stage 2 adaptive prediction results

---

## 📊 Evaluation

Stage 1 model evaluation metrics are saved in `Stage 1 Prediction Model/stage1_results.json` and displayed in the Streamlit sidebar. To view them in the terminal:

```bash
python "Stage 1 Prediction Model/stage1_evaluation.py"
```

---

## 📝 Notes

* The dataset is specific to **Tomato** crops on **Sandy clay loam** soil.
* The UI is designed to support additional crops and soil types in the future.
* Extra Trees Regression was selected after comparative evaluation against other regression models on the current synthetic dataset. It achieved the best performance among the evaluated candidate models.
* All predictions are made by the actual trained Extra Trees models — no hardcoded or random values.

---

> **AquaAdapt — Because every drop counts.** 💧
