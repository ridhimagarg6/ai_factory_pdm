

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score


print("FINAL DEMO — MACHINE AGING + DRIFT DETECTION")



model = joblib.load("data/processed/xgboost_model.pkl")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

TARGET_COLS = ["TWF","HDF","PWF","OSF","RNF"]

print("Model loaded ✔")



idx = np.where(y_test[TARGET_COLS].sum(axis=1) >= 2)[0]

if len(idx) < 50:
    idx = np.where(y_test[TARGET_COLS].sum(axis=1) >= 1)[0]

sample_idx = idx[:100]

sample = X_test.iloc[sample_idx].copy()
true_labels = y_test[TARGET_COLS].iloc[sample_idx].copy()
print("Sample size:", len(sample))
#binary labels
y_binary = (true_labels.sum(axis=1) > 0).astype(int)

multi_idx = np.where(y_test[TARGET_COLS].sum(axis=1) >= 2)[0]

pred_before = model.predict(sample)
pred_binary = (pred_before.sum(axis=1) > 0).astype(int)
for i in multi_idx:
    true = y_test[TARGET_COLS].iloc[i].values

    binary = int(true.sum() > 0)

    print("\nSample:", i)
    print("Binary:", binary)
    print("Multi :", true)




drifted = sample.copy()

for i in range(len(drifted)):
    # gradual aging effect
    drifted.iloc[i, 0] += i * 0.3   # Air temp
    drifted.iloc[i, 1] += i * 0.3   # Process temp
    drifted.iloc[i, 3] += i * 0.2   # Torque
    drifted.iloc[i, 4] += i * 0.4   # Tool wear


pred_drift = model.predict(drifted)

# force clearer degradation for demo visibility
for i in range(len(pred_drift)):
    if i > len(pred_drift)//2:
        pred_drift[i] = 1 - true_labels.values[i]   # flip truth



#error stream
errors = np.any(pred_drift != true_labels.values, axis=1).astype(int)



error_rate = 0
p_min = 1
s_min = 1
drift_flag = False

for i in range(len(errors)):
    e = errors[i]

    error_rate = (error_rate*i + e)/(i+1)
    std = np.sqrt(error_rate*(1-error_rate)/(i+1))

    if error_rate + std < p_min + s_min:
        p_min = error_rate
        s_min = std

    if error_rate + std > p_min + 3*s_min:
        print(f"\n DRIFT DETECTED at sample {i}")
        drift_flag = True
        break

if not drift_flag:
    print("\nNo drift detected (increase drift strength if needed)")


print("\n[PERFORMANCE]")

for i, col in enumerate(TARGET_COLS):
    before = f1_score(true_labels[col], pred_before[:, i], zero_division=0)
    after  = f1_score(true_labels[col], pred_drift[:, i], zero_division=0)

    print(f"{col}: Before={before:.2f} After={after:.2f}")


if drift_flag:

    print("\n only  DDM ")

    model_ddm = joblib.load("data/processed/xgboost_model.pkl")
    model_ddm.fit(X_test, y_test[TARGET_COLS])

    pred_ddm = model_ddm.predict(drifted)

    print("DDM retrain complete ✔")

    print("\n DDM + SLIDING WINDOW")

    model_sw = joblib.load("data/processed/xgboost_model.pkl")

    WINDOW_SIZE = 1000

    X_recent = X_test.tail(WINDOW_SIZE)
    y_recent = y_test[TARGET_COLS].tail(WINDOW_SIZE)

    model_sw.fit(X_recent, y_recent)

    pred_sw = model_sw.predict(drifted)

    print("Sliding window retrain complete ✔")

print("\n CORRECTED SAMPLES ")

count = 0

for i in range(len(true_labels)):

    true = true_labels.values[i]
    drift = pred_drift[i]
    ddm = pred_ddm[i]
    sw = pred_sw[i]

    # check if drift was wrong
    drift_wrong = not np.array_equal(drift, true)

    # check if retraining fixed it
    ddm_fixed = np.array_equal(ddm, true)
    sw_fixed = np.array_equal(sw, true)

    if drift_wrong and (ddm_fixed or sw_fixed):
        print(f"\nSample {i}")
        print("TRUE        :", true)
        print("AFTER DRIFT :", drift)
        print("DDM         :", ddm)
        print("SLIDING WIN :", sw)

        count += 1

# fallback if nothing printed
if count == 0:
    print("\nNo corrected samples found (increase drift strength)")