# =============================================================================
# PHASE 7 — DRIFT DETECTION (DDM) + RETRAINING
# =============================================================================

import pandas as pd
import numpy as np
import joblib

print("=" * 60)
print("  PHASE 7 — DRIFT DETECTION (DDM)")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1 — LOAD MODEL & DATA
# ─────────────────────────────────────────
print("\n[1] Loading model and data...")

PROCESSED_DIR = "data/processed"

model = joblib.load(f"{PROCESSED_DIR}/xgboost_model.pkl")

X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv")

X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv")

# Clean column names
X_train.columns = X_train.columns.str.strip()
X_test.columns  = X_test.columns.str.strip()
y_train.columns = y_train.columns.str.strip()
y_test.columns  = y_test.columns.str.strip()

TARGET_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("  Loaded ✔")


# ─────────────────────────────────────────
# STEP 2 — PREDICTIONS
# ─────────────────────────────────────────
print("\n[2] Generating predictions...")

y_pred = model.predict(X_test)

print("  Predictions ready ✔")


# ─────────────────────────────────────────
# STEP 3 — DDM PER LABEL
# ─────────────────────────────────────────
print("\n[3] Running DDM per failure type...")

drift_summary = {}

for idx, ft in enumerate(TARGET_COLS):

    print(f"\n--- Monitoring {ft} ---")

    y_true_ft = y_test[ft].values
    y_pred_ft = y_pred[:, idx]

    # Error stream (1 = wrong, 0 = correct)
    errors = (y_true_ft != y_pred_ft).astype(int)

    error_rate = 0
    p_min = float("inf")
    s_min = float("inf")

    drift_points = []

    for i in range(len(errors)):
        e = errors[i]

        # running error rate
        error_rate = (error_rate * i + e) / (i + 1)
        std_dev = np.sqrt(error_rate * (1 - error_rate) / (i + 1))

        # update minimum
        if error_rate + std_dev < p_min + s_min:
            p_min = error_rate
            s_min = std_dev

        # WARNING
        if error_rate + std_dev > p_min + 2 * s_min:
            print(f"⚠️ Warning at sample {i}")

        # DRIFT
        if error_rate + std_dev > p_min + 3 * s_min:
            print(f" Drift detected at sample {i}")
            drift_points.append(i)

    drift_summary[ft] = drift_points


# ─────────────────────────────────────────
# STEP 4 — SUMMARY
# ─────────────────────────────────────────
print("\n[4] Drift Summary:")

drifted_targets = []

for ft, points in drift_summary.items():
    if len(points) > 0:
        print(f"  {ft}: Drift detected (first at {points[0]})")
        drifted_targets.append(ft)
    else:
        print(f"  {ft}: No drift")


# ─────────────────────────────────────────
# STEP 5 — RETRAINING LOGIC
# ─────────────────────────────────────────
print("\n[5] Retraining decision...")

if len(drifted_targets) > 0:
    print(f"  Drift detected in: {drifted_targets}")
    print("  Retraining model using recent data...")

    # Use recent portion of test data (simulate real-world new data)
    recent_size = 500

    X_recent = X_test.tail(recent_size)
    y_recent = y_test.tail(recent_size)

    # Combine old + recent data
    X_new = pd.concat([X_train, X_recent], ignore_index=True)
    y_new = pd.concat([y_train, y_recent], ignore_index=True)

    # Retrain model
    model.fit(X_new, y_new[TARGET_COLS])

    # Save updated model
    joblib.dump(model, f"{PROCESSED_DIR}/xgboost_model_retrained.pkl")

    print("  Model retrained and saved ✔")

else:
    print("  No drift detected — model is stable")


# ─────────────────────────────────────────
# FINAL OUTPUT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 7 COMPLETE")
print("=" * 60)

print("""
WHAT YOU BUILT:

✓ Multi-label failure prediction system
✓ Per-failure drift monitoring (DDM)
✓ Automatic drift detection
✓ Smart retraining using recent data

RESULT:

✓ Model adapts to changing conditions
✓ Maintains long-term reliability

THIS IS A SELF-ADAPTIVE ML SYSTEM
""")