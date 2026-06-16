# 🌾 Food Import Dependency Predictor

A machine learning-powered system designed to analyze historical Food and Agriculture Organization (FAO) food balance datasets and predict future food import volumes and import dependency percentages. The project is split into a robust preprocessing and training pipeline (`preprocess_and_train.py`) and an interactive Streamlit dashboard (`streamlit_app.py`) for live forecasting and model diagnostics.

---

## 📂 Project Directory Structure

```text
FoodImportPrediction/
├── dataset/
│   ├── food_data.csv            # Raw FAO dataset (1961 - 2013)
│   └── processed_data.csv       # Cleaned long-form dataset with computed dependencies
├── model/
│   ├── rf_model_import.pkl      # Random Forest - Import Quantity Model
│   ├── rf_model_dependency.pkl  # Random Forest - Dependency Pct Model
│   ├── xgb_model_import.pkl     # XGBoost - Import Quantity Model
│   ├── xgb_model_dependency.pkl # XGBoost - Dependency Pct Model
│   ├── ridge_model_import.pkl   # Ridge Regression - Import Quantity Model
│   ├── ridge_model_dependency.pkl # Ridge Regression - Dependency Pct Model
│   ├── area_encoder.pkl         # Label encoder for Country/Area
│   ├── item_encoder.pkl         # Label encoder for Food Items
│   └── metrics.json             # Diagnostic metrics, hyperparameters, and best models
├── preprocess_and_train.py      # ML data prep, feature engineering, and model training script
├── streamlit_app.py             # Premium multi-page Streamlit web app
└── requirements.txt             # Pinned python package dependencies
```

---

## ⚡ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Set Up Virtual Environment (Recommended)
Create and activate a virtual environment to avoid package conflicts:
```bash
# Create environment
python -m venv venv

# Activate on Windows (cmd)
venv\Scripts\activate
# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages using the pinned `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Phase 1: Preprocess Data and Train Models
Run the training pipeline to parse the raw data, build lag features, train the regression models, and export model binaries:
```bash
python preprocess_and_train.py
```
*This will output step-by-step progress, plot feature importance histograms to the console, save files into `model/` and `dataset/`, and record overall model diagnostics into `metrics.json`.*

### Phase 2: Launch the Web Application
Launch the interactive Streamlit dashboard to explore predictions and metrics:
```bash
streamlit run streamlit_app.py
```

---

## 🔄 Data Pipeline & Feature Engineering

The raw FAO dataset contains years in wide columns (`Y1961` to `Y2013`). To make the dataset model-ready:

1. **Filtering**: Retains only records where the element is `Import Quantity` (tonnes) or `Food supply quantity (kg/capita/yr)`.
2. **Reshaping**: Melts the table from a wide format to a long format (`Area`, `Item`, `Element`, `Year`, `Value`).
3. **Interpolation**: Grouped by Country + Food Item, missing values are linearly interpolated (both directions) to prevent data loss.
4. **Dependency Calculation**: Computes the dependency metric:
   $$\text{Dependency Pct} = \left( \frac{\text{Import Quantity}}{\text{Food Supply}} \right) \times 100$$
   *To handle cases where supply is extremely low (resulting in large outliers), the dependency metric is clipped to a maximum of **200%**.*
5. **Autoregressive Lag Features**: To prevent data leakage and allow future predictions, features are constructed using past years' actual figures:
   * `Import_Lag1`, `Import_Lag2`, `Import_Lag3`: Import volumes from $t-1$, $t-2$, and $t-3$.
   * `Dep_Lag1`, `Dep_Lag2`: Dependency percentages from $t-1$ and $t-2$.
   * `Import_RollingMean3`: The average import volume over the last 3 years (shifted by 1 year).
   * `FoodSupply_Lag1`: Food supply volume from the previous year.

---

## 📈 Model Performance & Evaluation

The training script trains three algorithms (**Random Forest**, **XGBoost**, and **Ridge Regression**) on both target variables. Evaluation is done using an 80/20 train/test split:

### Target: Import Quantity (tonnes)
* **Ridge Regression** ★: $R^2 = 0.992663$ | $\text{RMSE} = 286.23$
* **Random Forest**: $R^2 = 0.957065$ | $\text{RMSE} = 692.40$
* **XGBoost**: $R^2 = 0.930589$ | $\text{RMSE} = 880.38$

### Target: Dependency %
* **Random Forest** ★: $R^2 = 0.882435$ | $\text{RMSE} = 31.97$
* **XGBoost**: $R^2 = 0.882365$ | $\text{RMSE} = 31.98$
* **Ridge Regression**: $R^2 = 0.877552$ | $\text{RMSE} = 32.62$

*★ indicates the final selected best model for live deployment in the app.*

---

## 🖥️ Streamlit App Features

The Streamlit web application utilizes modern CSS (glassmorphism cards, custom font stack `Inter`, smooth animations, and dark sidebar styling) and features two main panels:

### 1. 🔮 Predictor Page
* Allows selection of Country/Area, Food Item, Model Algorithm, and Target Year (up to 2018).
* **Multi-Year Iterative Inference**: For years beyond the historical dataset (e.g. 2014–2018), predictions for year $t$ are fed recursively as the lag features for year $t+1$.
* Visualizes interactive trends of historical data and forecast projections side-by-side using Plotly.

### 2. 📊 Dashboard Page
* **Dataset Diagnostics**: Displays records shape, category counts, features list, and year ranges.
* **Algorithm Configs**: Displays the exact hyperparameters used for RF, XGBoost, and Ridge.
* **Evaluation Comparison**: Features interactive performance bar charts and a dynamic color-coded error metrics table.
* **Feature Importance**: Plots the relative importance (or coefficients) of the engineered features.
* **Future Forecast Comparison**: Renders side-by-side forecasts for the next 3 years across all three ML algorithms.
* **Data Distributions**: Plots interactive histograms showing the data distribution of imports and dependency ratios.
