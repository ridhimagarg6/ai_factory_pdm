import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


target ="Machine failure"
Failure_types=["TWF", "HDF", "PWF", "OSF", "RNF"]
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

df=pd.read_csv("ai4i2020.csv", encoding="utf-8-sig")
df_eng=df.copy()
#temp delta
df_eng["temp_delta"] = (
    df_eng["Process temperature [K]"] - df_eng["Air temperature [K]"]
)
#power
df_eng["power"] = (
    df_eng["Rotational speed [rpm]"] * df_eng["Torque [Nm]"]
)
#wear_torque
df_eng["wear_torque"] = (
    df_eng["Tool wear [min]"] * df_eng["Torque [Nm]"]
)

new_features = ["temp_delta", "power", "wear_torque"]

# print("\n  Verification — mean value during failure vs normal:")
# print(f"  {'Feature':<20} {'Normal':>10} {'Failure':>10} {'Better?':>10}")
# print(f"  {'-'*55}")
# for feat in new_features:
#     nm = df_eng[df_eng[target] == 0][feat].mean()
#     fm = df_eng[df_eng[target] == 1][feat].mean()
#     diff_pct = abs(fm - nm) / abs(nm) * 100 if nm != 0 else 0
#     better = "✓ YES" if diff_pct > 5 else "— small"
#     print(f"  {feat:<20} {nm:>10.2f} {fm:>10.2f} {better:>10}")


#encoding
type_map = {"L": 0, "M": 1, "H": 2}
df_eng["Type_encoded"] = df_eng["Type"].map(type_map)
print(f"  Type distribution:")
for t, v in type_map.items():
    count = (df_eng["Type"] == t).sum()
    print(f"    {t} ({v}): {count:,} rows")


DROP_COLS = ["UDI", "Product ID", "Type"]
FEATURE_COLS = [
    # Original sensor features (keep all 5 raw)
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    # Encoded type
    "Type_encoded",
    # Engineered features
    "temp_delta",
    "power",
    "wear_torque"

]
TARGET_COLS = [target] + Failure_types
df_model = df_eng[FEATURE_COLS + TARGET_COLS].copy()

#splitting
X = df_model[FEATURE_COLS]
y = df_model[TARGET_COLS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y[target] )

#feature scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),  # fit + transform on TRAIN only
    columns=FEATURE_COLS,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),  # transform only on TEST (no fitting)
    columns=FEATURE_COLS,
    index=X_test.index
)

print(f"  Scaler fitted on training set ({X_train.shape[0]:,} rows)")
print(f"  Applied to test set (no data leakage)")

# Save scaler — needed in Phase 7 (drift)
PROCESSED_DIR  = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)
scaler_path = f"{PROCESSED_DIR}/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"\n  Scaler saved: {scaler_path}")



# STEP 8 — SAVE ALL OUTPUTS
# Everything saved here is what Phase 3, 5, 6, 7, 9 will load.

# =============================================================================
# STEP 8 — SAVE PROCESSED DATA (CLEAN VERSION)
# =============================================================================

print("\n[8] Saving processed data...")

import os
import json

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =============================================================================
# CLEAN COLUMN NAMES BEFORE SAVING (IMPORTANT)
# =============================================================================
def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    df.columns = df.columns.str.replace("_+", "_", regex=True)
    return df


# Apply cleaning
df_eng        = clean_columns(df_eng)
X_train_scaled = clean_columns(X_train_scaled)
X_test_scaled  = clean_columns(X_test_scaled)
y_train        = clean_columns(y_train)
y_test         = clean_columns(y_test)


# =============================================================================
# SAVE ENGINEERED DATASET (FOR PHASE 3)
# =============================================================================
df_eng_path = f"{PROCESSED_DIR}/ai4i2020_engineered.csv"
df_eng.to_csv(df_eng_path, index=False)
print(f"  Full engineered dataset : {df_eng_path}")




# =============================================================================
# SAVE TRAIN-TEST DATA (FOR MODEL)
# =============================================================================
X_train_scaled.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
X_test_scaled.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

print(f"  Train/Test datasets saved in : {PROCESSED_DIR}/")



# SAVE FEATURE LIST (FOR CONSISTENCY ACROSS PHASES)

with open(f"{PROCESSED_DIR}/feature_cols.json", "w") as f:
    json.dump({
        "features": FEATURE_COLS,
        "targets": TARGET_COLS
    }, f, indent=2)

print(f"  Feature list saved : {PROCESSED_DIR}/feature_cols.json")






# DEBUG (OPTIONAL)

print("\nColumns (X_train):")
print(X_train_scaled.columns.tolist())



print("PHASE 2 COMPLETE — CLEAN PIPELINE")


# Show new engineered features
print("\n[Feature Engineering Output]")
cols = [
    "Air_temperature_K_",
    "Process_temperature_K_",
    "temp_delta",
    "Rotational_speed_rpm_",
    "Torque_Nm_",
    "power",
    "Tool_wear_min_",
    "wear_torque"
]

print(df_eng[cols].head(10))

print("\n[Feature Relationships]")

print("\nPower = Speed × Torque")
print((df_eng["Rotational_speed_rpm_"] * df_eng["Torque_Nm_"]).head())

print("\nTemp Delta = Process - Air")
print((df_eng["Process_temperature_K_"] - df_eng["Air_temperature_K_"]).head())

print("\n[Target Labels]")
print(TARGET_COLS)

print("\n[Final Dataset Sample]")
print(df_eng.head(10))
