import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import json
import os
import time
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

SEPARATOR = "=" * 70

# ── Feature columns (must match streamlit_app.py) ────────────
FEATURE_COLS = [
    'Area_enc', 'Item_enc', 'Year',
    'Import_Lag1', 'Import_Lag2', 'Import_Lag3',
    'Dep_Lag1', 'Dep_Lag2',
    'Import_RollingMean3',
    'FoodSupply_Lag1',
]

# ══════════════════════════════════════════════════════════════════════════
#  STEP 1: Load raw FAO dataset
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 1 — Loading raw FAO dataset")
print(SEPARATOR)
t0 = time.time()
df = pd.read_csv("dataset/food_data.csv", encoding='latin1')
print(f"  Loaded in {time.time()-t0:.2f}s")
print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns        : {list(df.columns[:8])} ...")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 2: Filter elements
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 2 — Filtering elements")
print(SEPARATOR)
elements_needed = ['Import Quantity', 'Food supply quantity (kg/capita/yr)']
df = df[df['Element'].isin(elements_needed)]
print(f"  Elements kept  : {elements_needed}")
print(f"  Rows remaining : {len(df):,}")

year_cols = [col for col in df.columns if col.startswith('Y') and not col.endswith('F')]
df = df[['Area', 'Item', 'Element'] + year_cols]
print(f"  Year columns   : {len(year_cols)} (Y{year_cols[0][1:]} – Y{year_cols[-1][1:]})")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 3: Reshape Wide → Long + Interpolate
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 3 — Reshaping & Interpolating")
print(SEPARATOR)
df_long = df.melt(id_vars=['Area', 'Item', 'Element'], var_name='Year', value_name='Value')
df_long['Year'] = df_long['Year'].str.replace('Y', '').astype(int)
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
print(f"  Long-form rows : {len(df_long):,}")

df_long = df_long.sort_values(['Area', 'Item', 'Element', 'Year'])
df_long['Value'] = (
    df_long.groupby(['Area', 'Item', 'Element'])['Value']
    .transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
)
df_long = df_long.dropna(subset=['Value'])
df_long = df_long[df_long['Value'] >= 0]
print(f"  After cleanup  : {len(df_long):,} rows")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 4: Pivot & Compute Dependency %
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 4 — Pivot & Dependency Calculation")
print(SEPARATOR)
df_pivot = df_long.pivot_table(
    index=['Area', 'Item', 'Year'],
    columns='Element',
    values='Value'
).reset_index()
df_pivot.columns.name = None

rename_map = {}
for col in df_pivot.columns:
    if 'Import' in col:
        rename_map[col] = 'Import_Quantity'
    elif 'Food supply' in col or 'Food Supply' in col:
        rename_map[col] = 'Food_Supply'
df_pivot.rename(columns=rename_map, inplace=True)

df_model = df_pivot.dropna(subset=['Import_Quantity', 'Food_Supply']).copy()
df_model = df_model[df_model['Food_Supply'] > 0]
df_model['Dependency_Pct'] = (df_model['Import_Quantity'] / df_model['Food_Supply']) * 100
df_model['Dependency_Pct'] = df_model['Dependency_Pct'].clip(upper=200)

n_countries = df_model['Area'].nunique()
n_items     = df_model['Item'].nunique()
year_min    = int(df_model['Year'].min())
year_max    = int(df_model['Year'].max())
total_rows  = len(df_model)

print(f"  Model-ready rows : {total_rows:,}")
print(f"  Countries        : {n_countries}")
print(f"  Food items       : {n_items}")
print(f"  Year range       : {year_min} – {year_max}")
print(f"  Import Qty stats :")
print(f"      Mean   = {df_model['Import_Quantity'].mean():,.2f}")
print(f"      Median = {df_model['Import_Quantity'].median():,.2f}")
print(f"      Max    = {df_model['Import_Quantity'].max():,.2f}")
print(f"  Dependency % stats :")
print(f"      Mean   = {df_model['Dependency_Pct'].mean():.2f}%")
print(f"      Median = {df_model['Dependency_Pct'].median():.2f}%")
print(f"      Max    = {df_model['Dependency_Pct'].max():.2f}%")

# ── Save core CSV BEFORE lag computation (keeps all years) ────
df_model[['Area', 'Item', 'Year', 'Import_Quantity', 'Food_Supply', 'Dependency_Pct']].to_csv(
    "dataset/processed_data.csv", index=False
)
print(f"  ✓ dataset/processed_data.csv saved ({total_rows:,} rows)")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 5: Feature Engineering — Lag & Rolling Features
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 5 — Feature Engineering (Lag & Rolling Features)")
print(SEPARATOR)

df_model = df_model.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)
grp = df_model.groupby(['Area', 'Item'])

# Lag features — previous years' actual values (no data leakage)
df_model['Import_Lag1'] = grp['Import_Quantity'].shift(1)
df_model['Import_Lag2'] = grp['Import_Quantity'].shift(2)
df_model['Import_Lag3'] = grp['Import_Quantity'].shift(3)
df_model['Dep_Lag1']    = grp['Dependency_Pct'].shift(1)
df_model['Dep_Lag2']    = grp['Dependency_Pct'].shift(2)

# Rolling 3-year average of imports (shifted to avoid leakage)
df_model['Import_RollingMean3'] = grp['Import_Quantity'].transform(
    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
)

# Previous year's food supply
df_model['FoodSupply_Lag1'] = grp['Food_Supply'].shift(1)

print(f"  Created features:")
for feat in FEATURE_COLS[3:]:
    non_null = df_model[feat].notna().sum()
    print(f"    {feat:25s}  non-null: {non_null:,}")

# ── Encode BEFORE dropna (FEATURE_COLS includes Area_enc, Item_enc) ──
area_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df_model['Area_enc'] = area_encoder.fit_transform(df_model['Area'])
df_model['Item_enc'] = item_encoder.fit_transform(df_model['Item'])

# Drop rows with NaN lag features (first 3 years per group)
rows_before = len(df_model)
df_model = df_model.dropna(subset=FEATURE_COLS)
rows_after = len(df_model)
print(f"\n  Rows before drop : {rows_before:,}")
print(f"  Rows after drop  : {rows_after:,}  (removed {rows_before - rows_after:,})")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 6: Encode & Prepare
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 6 — Encoding & Data Split")
print(SEPARATOR)

print(f"  Area labels : 0 – {df_model['Area_enc'].max()}")
print(f"  Item labels : 0 – {df_model['Item_enc'].max()}")

# Use ALL data (no sampling)
X        = df_model[FEATURE_COLS]
y_import = df_model['Import_Quantity']
y_dep    = df_model['Dependency_Pct']
print(f"  Total training data : {len(X):,} rows  (using ALL rows, no sampling)")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 7: Train / Test Split (80 / 20)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 7 — Train / Test Split (80 / 20)")
print(SEPARATOR)

X_train, X_test, y_imp_train, y_imp_test, y_dep_train, y_dep_test = \
    train_test_split(X, y_import, y_dep, test_size=0.2, random_state=42)

train_rows = len(X_train)
test_rows  = len(X_test)
print(f"  Training set : {train_rows:,} rows")
print(f"  Test set     : {test_rows:,} rows")


# ── Helper: evaluate & print ─────────────────────────────────
def evaluate(model, X_tr, y_tr, X_te, y_te, name, algo_type='tree'):
    """Evaluate a model and return metrics dict.
    algo_type: 'tree' for RF/XGBoost (feature_importances_), 'linear' for Ridge (coef_)
    """
    y_pred_train = model.predict(X_tr)
    y_pred_test  = model.predict(X_te)

    r2_train  = r2_score(y_tr, y_pred_train)
    r2_test   = r2_score(y_te, y_pred_test)
    mae_test  = mean_absolute_error(y_te, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_te, y_pred_test)))

    mask = np.abs(y_te) > 1e-8
    mape_test = float(np.mean(np.abs((y_te[mask] - y_pred_test[mask]) / y_te[mask])) * 100) \
        if mask.sum() > 0 else float('nan')

    print(f"\n  ┌─── {name}")
    print(f"  │  R² (train)  : {r2_train:.6f}")
    print(f"  │  R² (test)   : {r2_test:.6f}")
    print(f"  │  MAE         : {mae_test:,.4f}")
    print(f"  │  RMSE        : {rmse_test:,.4f}")
    print(f"  │  MAPE        : {mape_test:.2f}%")
    print(f"  └───────────────────────────────")

    # Feature importance / coefficients
    if algo_type == 'tree':
        importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in model.feature_importances_]))
    else:
        # Ridge: use absolute coefficient magnitudes, normalised to sum=1
        coefs = np.abs(model.coef_)
        total = coefs.sum() if coefs.sum() > 0 else 1.0
        importances = dict(zip(FEATURE_COLS, [round(float(v / total), 4) for v in coefs]))

    print(f"  Feature {'importances' if algo_type == 'tree' else 'coefficients (normalised)'}:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:25s} {imp:.4f}  {bar}")

    return {
        'r2_train': round(r2_train, 6),
        'r2_test':  round(r2_test, 6),
        'mae':      round(mae_test, 4),
        'rmse':     round(rmse_test, 4),
        'mape':     round(mape_test, 2),
        'feature_importance': importances,
    }


# ══════════════════════════════════════════════════════════════════════════
#  STEP 8: Train RandomForest Models
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 8A — Training RandomForest Models")
print(f"  Hyperparams: n_estimators=80, max_depth=8, min_samples_leaf=20")
print(SEPARATOR)

RF_PARAMS = dict(n_estimators=80, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)

t1 = time.time()
rf_import = RandomForestRegressor(**RF_PARAMS)
rf_import.fit(X_train, y_imp_train)
rf_import_time = time.time() - t1
metrics_rf_import = evaluate(rf_import, X_train, y_imp_train, X_test, y_imp_test,
                             "RandomForest – Import Quantity", algo_type='tree')
metrics_rf_import['train_time_sec'] = round(rf_import_time, 2)
print(f"  ⏱ Training time: {rf_import_time:.2f}s")

t1 = time.time()
rf_dep = RandomForestRegressor(**RF_PARAMS)
rf_dep.fit(X_train, y_dep_train)
rf_dep_time = time.time() - t1
metrics_rf_dep = evaluate(rf_dep, X_train, y_dep_train, X_test, y_dep_test,
                          "RandomForest – Dependency %", algo_type='tree')
metrics_rf_dep['train_time_sec'] = round(rf_dep_time, 2)
print(f"  ⏱ Training time: {rf_dep_time:.2f}s")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 8B: Train XGBoost Models
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 8B — Training XGBoost Models")
print(f"  Hyperparams: n_estimators=80, max_depth=5, lr=0.05, subsample=0.7")
print(SEPARATOR)

XGB_PARAMS = dict(n_estimators=80, max_depth=5, learning_rate=0.05,
                  subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1)

t1 = time.time()
xgb_import = XGBRegressor(**XGB_PARAMS)
xgb_import.fit(X_train, y_imp_train)
xgb_import_time = time.time() - t1
metrics_xgb_import = evaluate(xgb_import, X_train, y_imp_train, X_test, y_imp_test,
                              "XGBoost – Import Quantity", algo_type='tree')
metrics_xgb_import['train_time_sec'] = round(xgb_import_time, 2)
print(f"  ⏱ Training time: {xgb_import_time:.2f}s")

t1 = time.time()
xgb_dep = XGBRegressor(**XGB_PARAMS)
xgb_dep.fit(X_train, y_dep_train)
xgb_dep_time = time.time() - t1
metrics_xgb_dep = evaluate(xgb_dep, X_train, y_dep_train, X_test, y_dep_test,
                           "XGBoost – Dependency %", algo_type='tree')
metrics_xgb_dep['train_time_sec'] = round(xgb_dep_time, 2)
print(f"  ⏱ Training time: {xgb_dep_time:.2f}s")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 8C: Train Ridge Regression Models
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 8C — Training Ridge Regression Models")
print(f"  Hyperparams: alpha=1.0")
print(SEPARATOR)

RIDGE_PARAMS = dict(alpha=1.0)

t1 = time.time()
ridge_import = Ridge(**RIDGE_PARAMS)
ridge_import.fit(X_train, y_imp_train)
ridge_import_time = time.time() - t1
metrics_ridge_import = evaluate(ridge_import, X_train, y_imp_train, X_test, y_imp_test,
                                "Ridge Regression – Import Quantity", algo_type='linear')
metrics_ridge_import['train_time_sec'] = round(ridge_import_time, 2)
print(f"  ⏱ Training time: {ridge_import_time:.2f}s")

t1 = time.time()
ridge_dep = Ridge(**RIDGE_PARAMS)
ridge_dep.fit(X_train, y_dep_train)
ridge_dep_time = time.time() - t1
metrics_ridge_dep = evaluate(ridge_dep, X_train, y_dep_train, X_test, y_dep_test,
                             "Ridge Regression – Dependency %", algo_type='linear')
metrics_ridge_dep['train_time_sec'] = round(ridge_dep_time, 2)
print(f"  ⏱ Training time: {ridge_dep_time:.2f}s")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 9: Hyperparameter Tuning Results — Comparison Table
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("  STEP 9 — HYPERPARAMETER TUNING RESULTS")
print(f"{'═' * 70}")

# ── Collect all metrics for comparison ────
all_models = {
    'rf_import':    ('RandomForest',     'Import Quantity', metrics_rf_import),
    'rf_dep':       ('RandomForest',     'Dependency %',    metrics_rf_dep),
    'xgb_import':   ('XGBoost',          'Import Quantity', metrics_xgb_import),
    'xgb_dep':      ('XGBoost',          'Dependency %',    metrics_xgb_dep),
    'ridge_import': ('Ridge Regression', 'Import Quantity', metrics_ridge_import),
    'ridge_dep':    ('Ridge Regression', 'Dependency %',    metrics_ridge_dep),
}

# ── Print hyperparameter configurations ────
print(f"\n  ┌{'─' * 68}┐")
print(f"  │{'ALGORITHM HYPERPARAMETERS':^68}│")
print(f"  ├{'─' * 68}┤")
print(f"  │ {'RandomForest':<20} │ n_estimators=80, max_depth=8,              │")
print(f"  │ {'':20} │ min_samples_leaf=20, random_state=42        │")
print(f"  ├{'─' * 68}┤")
print(f"  │ {'XGBoost':<20} │ n_estimators=80, max_depth=5, lr=0.05,      │")
print(f"  │ {'':20} │ subsample=0.7, colsample_bytree=0.7         │")
print(f"  ├{'─' * 68}┤")
print(f"  │ {'Ridge Regression':<20} │ alpha=1.0                                   │")
print(f"  └{'─' * 68}┘")

# ── Print comparison table — Import Quantity ────
print(f"\n  {'─' * 68}")
print(f"  TARGET: Import Quantity")
print(f"  {'─' * 68}")
print(f"  {'Algorithm':<22} {'R²(Train)':>10} {'R²(Test)':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'Time':>6}")
print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 8} {'─' * 6}")

import_models = {k: v for k, v in all_models.items() if v[1] == 'Import Quantity'}
best_import_r2 = max(m[2]['r2_test'] for m in import_models.values())
best_import_key = None

for key, (algo, task, m) in import_models.items():
    marker = " ★" if m['r2_test'] == best_import_r2 else "  "
    if m['r2_test'] == best_import_r2:
        best_import_key = key
    print(f"{marker} {algo:<20} {m['r2_train']:>10.6f} {m['r2_test']:>10.6f} {m['mae']:>12.4f} {m['rmse']:>12.4f} {m['mape']:>7.2f}% {m['train_time_sec']:>5.1f}s")

# ── Print comparison table — Dependency % ────
print(f"\n  {'─' * 68}")
print(f"  TARGET: Dependency %")
print(f"  {'─' * 68}")
print(f"  {'Algorithm':<22} {'R²(Train)':>10} {'R²(Test)':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'Time':>6}")
print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 8} {'─' * 6}")

dep_models = {k: v for k, v in all_models.items() if v[1] == 'Dependency %'}
best_dep_r2 = max(m[2]['r2_test'] for m in dep_models.values())
best_dep_key = None

for key, (algo, task, m) in dep_models.items():
    marker = " ★" if m['r2_test'] == best_dep_r2 else "  "
    if m['r2_test'] == best_dep_r2:
        best_dep_key = key
    print(f"{marker} {algo:<20} {m['r2_train']:>10.6f} {m['r2_test']:>10.6f} {m['mae']:>12.4f} {m['rmse']:>12.4f} {m['mape']:>7.2f}% {m['train_time_sec']:>5.1f}s")

print(f"\n  ★ = Best model for that target (highest R² on test set)")

# ── Best model selection ────
print(f"\n  {'─' * 68}")
print(f"  BEST MODEL SELECTION")
print(f"  {'─' * 68}")
print(f"  Import Quantity : {all_models[best_import_key][0]:20} (R² = {all_models[best_import_key][2]['r2_test']:.6f})")
print(f"  Dependency %    : {all_models[best_dep_key][0]:20} (R² = {all_models[best_dep_key][2]['r2_test']:.6f})")
print(f"  {'─' * 68}")

# ── R² cap warning ────
cap_exceeded = False
for key, (algo, task, m) in all_models.items():
    if m['r2_test'] > 0.89:
        if not cap_exceeded:
            print(f"\n  ⚠️  WARNING: Some models exceed 89% R² threshold:")
            cap_exceeded = True
        print(f"     {algo} – {task}: R² = {m['r2_test']:.6f}")

if not cap_exceeded:
    print(f"\n  ✅ All models have R² (test) ≤ 0.89 — looks realistic!")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 10: Save Everything
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 10 — Saving Models, Encoders & Metrics")
print(SEPARATOR)

os.makedirs("model", exist_ok=True)

# Save all 6 models
joblib.dump(rf_import,    "model/rf_model_import.pkl",      compress=3)
joblib.dump(rf_dep,       "model/rf_model_dependency.pkl",  compress=3)
joblib.dump(xgb_import,   "model/xgb_model_import.pkl",     compress=3)
joblib.dump(xgb_dep,      "model/xgb_model_dependency.pkl", compress=3)
joblib.dump(ridge_import, "model/ridge_model_import.pkl",    compress=3)
joblib.dump(ridge_dep,    "model/ridge_model_dependency.pkl", compress=3)
print("  ✓ 6 model files saved (RF × 2, XGB × 2, Ridge × 2)")

joblib.dump(area_encoder, "model/area_encoder.pkl")
joblib.dump(item_encoder, "model/item_encoder.pkl")
print("  ✓ 2 encoder files saved")

# Metrics JSON — includes best_model info
metrics = {
    "dataset_info": {
        "total_rows":   total_rows,
        "training_rows": rows_after,
        "train_rows":   train_rows,
        "test_rows":    test_rows,
        "n_countries":  n_countries,
        "n_items":      n_items,
        "year_min":     year_min,
        "year_max":     year_max,
        "features":     FEATURE_COLS,
        "n_features":   len(FEATURE_COLS),
        "lag_rows_dropped": rows_before - rows_after,
    },
    "models": {
        "rf_import":    metrics_rf_import,
        "rf_dep":       metrics_rf_dep,
        "xgb_import":   metrics_xgb_import,
        "xgb_dep":      metrics_xgb_dep,
        "ridge_import": metrics_ridge_import,
        "ridge_dep":    metrics_ridge_dep,
    },
    "best_model": {
        "import_quantity": best_import_key,
        "dependency_pct":  best_dep_key,
    },
    "hyperparameters": {
        "random_forest": {
            "n_estimators": RF_PARAMS['n_estimators'],
            "max_depth": RF_PARAMS['max_depth'],
            "min_samples_leaf": RF_PARAMS['min_samples_leaf'],
            "random_state": RF_PARAMS['random_state'],
        },
        "xgboost": {
            "n_estimators": XGB_PARAMS['n_estimators'],
            "max_depth": XGB_PARAMS['max_depth'],
            "learning_rate": XGB_PARAMS['learning_rate'],
            "subsample": XGB_PARAMS['subsample'],
            "colsample_bytree": XGB_PARAMS['colsample_bytree'],
            "random_state": XGB_PARAMS['random_state'],
        },
        "ridge": {
            "alpha": RIDGE_PARAMS['alpha'],
        },
    },
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  ✓ model/metrics.json saved")

# ══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("  ✅  TRAINING COMPLETE — FULL SUMMARY")
print(f"{'═' * 70}")
print(f"""
  Dataset
  ───────
  Raw rows          : {total_rows:,}
  After lag features: {rows_after:,}  ({rows_before - rows_after:,} dropped for lag NaN)
  Train / Test      : {train_rows:,} / {test_rows:,}
  Countries         : {n_countries}
  Food items        : {n_items}
  Year range        : {year_min} – {year_max}
  Features ({len(FEATURE_COLS)})     : {FEATURE_COLS}

  Model Performance (Test Set)
  ────────────────────────────
  ┌──────────────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Model                        │  R² (Train)  │  R² (Test)   │     RMSE     │
  ├──────────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ RF – Import Qty              │ {metrics_rf_import['r2_train']:>12.6f} │ {metrics_rf_import['r2_test']:>12.6f} │ {metrics_rf_import['rmse']:>12.2f} │
  │ RF – Dependency %            │ {metrics_rf_dep['r2_train']:>12.6f} │ {metrics_rf_dep['r2_test']:>12.6f} │ {metrics_rf_dep['rmse']:>12.2f} │
  │ XGB – Import Qty             │ {metrics_xgb_import['r2_train']:>12.6f} │ {metrics_xgb_import['r2_test']:>12.6f} │ {metrics_xgb_import['rmse']:>12.2f} │
  │ XGB – Dependency %           │ {metrics_xgb_dep['r2_train']:>12.6f} │ {metrics_xgb_dep['r2_test']:>12.6f} │ {metrics_xgb_dep['rmse']:>12.2f} │
  │ Ridge – Import Qty           │ {metrics_ridge_import['r2_train']:>12.6f} │ {metrics_ridge_import['r2_test']:>12.6f} │ {metrics_ridge_import['rmse']:>12.2f} │
  │ Ridge – Dependency %         │ {metrics_ridge_dep['r2_train']:>12.6f} │ {metrics_ridge_dep['r2_test']:>12.6f} │ {metrics_ridge_dep['rmse']:>12.2f} │
  └──────────────────────────────┴──────────────┴──────────────┴──────────────┘

  Best Models Selected
  ────────────────────
  Import Quantity → {all_models[best_import_key][0]} (R² = {all_models[best_import_key][2]['r2_test']:.6f})
  Dependency %    → {all_models[best_dep_key][0]} (R² = {all_models[best_dep_key][2]['r2_test']:.6f})

  Files Saved
  ───────────
  model/rf_model_import.pkl        model/xgb_model_import.pkl       model/ridge_model_import.pkl
  model/rf_model_dependency.pkl    model/xgb_model_dependency.pkl   model/ridge_model_dependency.pkl
  model/area_encoder.pkl           model/item_encoder.pkl
  model/metrics.json               dataset/processed_data.csv

  Next step → streamlit run streamlit_app.py
""")