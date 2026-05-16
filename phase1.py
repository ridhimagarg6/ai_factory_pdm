import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from importlib_metadata import distribution
from scipy import stats
import os

Failure_types=["TWF", "HDF","PWF","OSF","RNF"]
sensors=["Air temperature [K]",
                 "Process temperature [K]",
                 "Rotational speed [rpm]",
                 "Torque [Nm]",
                 "Tool wear [min]"]

full_names={
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure"

}
FAILURE_COLORS = {
    "TWF": "#FF5722", "HDF": "#F44336",
    "PWF": "#FF9800", "OSF": "#9C27B0", "RNF": "#607D8B"
}
#loading data
df=pd.read_csv("ai4i2020.csv", encoding="utf-8-sig")
target = [col for col in df.columns if "Machine" in col][0]
df.columns=df.columns.str.strip()
PLOTS_DIR     = "reports/phase1_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

#data check

#column types
print(df.dtypes.to_string())
#missing values check
missing=df.isnull().sum().sum()
if missing==0:
    print("Dataset is 100% complete — unrealistic for real SME factories.")
    print("This directly motivates  Indian SME Noise Injection.")

#duplicate check


checking_cols=[c for c in df.columns if c not in ["UDI","Product ID"]]
duplicate=df.duplicated(subset=checking_cols).sum()
print(f"\nduplicated rows: {duplicate}")

#value range check for sensors
print("sensor value range:")
print(df[sensors].describe().round(2).to_string())


#class imbalance check
print("class imbalance analysis")
total=len(df)
number_fail=df[target].sum()
number_normal=total-number_fail
ratio=number_normal/number_fail
print(f"total samples: {total}")
print(f"number of failure[1]: {number_fail}")
print(f"number of normal[0]: {number_normal}")
print(f"imbalance ratio: {ratio}")
print(f"accuracy: {number_normal/total*100}")

#per failure type count
for f in Failure_types:
    c=df[f].sum()
    print(f"{f}: {c}")

print(f"RNF  has only {df['RNF'].sum()} samples , therefore low rnf score is expected.")


#multi-label failure
label_counts=df[Failure_types].sum(axis=1)
distribution=label_counts.value_counts().sort_index()
print(label_counts)
print(distribution)

#co-occurrence matrix
print("count of both occurring together")
co=pd.DataFrame(0,index=Failure_types, columns=Failure_types)
for f1 in Failure_types:
    for f2 in Failure_types:
        co.loc[f1,f2]=((df[f1]==1) & (df[f2]==1)).sum()
print(co.to_string())


print("sensor statistics by failure type")
for f in Failure_types:
    fail_group=df[df[f]==1]
    normal_group=df[df[f]==0]
    print(f"\n  {f} — {full_names[f]} ({len(fail_group)} failures)")
# print(fail_group)
    for fi in sensors:
        nm=normal_group[fi].mean()
        fm=fail_group[fi].mean()
        delta=(fm-nm) / nm *100
        _, p = stats.mannwhitneyu(normal_group[fi], fail_group[fi],
                                  alternative="two-sided")
        sig = "★" if p < 0.05 else " "
        print(f"  {fi:<35} {nm:>8.2f} {fm:>8.2f} {delta:>+5.1f}%  {sig}")
print("\n  ★ = statistically significant difference (p < 0.05)")

#physical failure condition verification
df_check=df.copy()
df_check['temp_delta']=df['Process temperature [K]']-df['Air temperature [K]']
df_check["power"]      = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
df_check["wear_torque"]= df["Tool wear [min]"] * df["Torque [Nm]"]
# print(df_check.head())

conditions = {
    "TWF": (df_check["Tool wear [min]"] >= 200),
    "HDF": (df_check["temp_delta"] < 8.6),
    "PWF": ((df_check["power"] < 3500) | (df_check["power"] > 9000)),
    "OSF": (df_check["wear_torque"] > 11000),
    "RNF": pd.Series(True, index=df_check.index),  # random, no condition
}
# print(conditions)
print(f"\n  {'Type':<6} {'Condition':<45} {'Failures meeting it':>20}")
descs = {
    "TWF": "Tool wear >= 200 min",
    "HDF": "Process temp - Air temp < 8.6 K",
    "PWF": "Speed × Torque outside [3500, 9000] W",
    "OSF": "Tool wear × Torque > 11,000",
    "RNF": "Random — no condition"
}
for ft, cond in conditions.items():
    n_fail_ft = df[ft].sum()
    met = (cond & (df_check[ft] == 1)).sum()
    pct = met / n_fail_ft * 100 if n_fail_ft > 0 else 0
    print(f"  {ft:<6} {descs[ft]:<45} {met}/{n_fail_ft} ({pct:.1f}%)")


#dataset compare
df2=pd.read_csv("ai4i2020_engineered.csv", encoding="utf-8-sig")
print(df2.shape)
print(df2.describe())
print(df2.columns.tolist())
target_df2 = [col for col in df2.columns if "Machine" in col][0]
print(df[target].mean()*100)
print(df2[target_df2].mean()*100)
for ft in Failure_types:
    c1, c2 = df[ft].sum(), df2[ft].sum()
    print(f"  {ft + ' count':<30} {c1:>12} {c2:>12}")

print(f"\n  Variant has {df2[target_df2].mean() / df[target].mean():.1f}x higher failure rate")
#plotting
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})
sns.set_style("whitegrid")

# ── Plot 1: Class Imbalance ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Class Imbalance — AI4I 2020 Dataset", fontsize=13, fontweight="bold")

# Left: overall
counts = [number_normal, number_fail]
bars = axes[0].bar(["Normal", "Failure"], counts,
                   color=["#2196F3", "#F44336"], width=0.4, edgecolor="white")
for bar, c in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 100,
                 f"{c:,}\n({c / total * 100:.1f}%)",
                 ha="center", fontweight="bold")
axes[0].set_title("Overall Failure vs Normal")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, max(counts) * 1.2)
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Right: per failure type
ft_counts = [df[ft].sum() for ft in Failure_types]
bars = axes[1].bar(Failure_types, ft_counts,
                   color=[FAILURE_COLORS[ft] for ft in Failure_types],
                   edgecolor="white")
for bar, c in zip(bars, ft_counts):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, str(c),
                 ha="center", fontweight="bold")
axes[1].set_title("Per-Failure-Type Count")
axes[1].set_ylabel("Count")
axes[1].set_ylim(0, max(ft_counts) * 1.2)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_imbalance.png")
plt.close()
print("  Saved: 01_class_imbalance.png")

# ── Plot 2: Sensor Distributions (Normal vs Failure) ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
fig.suptitle("Sensor Feature Distributions: Normal vs Failure",
             fontsize=13, fontweight="bold")

normal_df = df[df[target] == 0]
failure_df = df[df[target] == 1]

for i, col in enumerate(sensors):
    ax = axes[i]
    ax.hist(normal_df[col], bins=50, alpha=0.5, color="#2196F3",
            density=True, label="Normal")
    ax.hist(failure_df[col], bins=30, alpha=0.8, color="#F44336",
            density=True, label="Failure")
    ax.axvline(normal_df[col].mean(), color="#2196F3", linestyle="--", linewidth=1.5)
    ax.axvline(failure_df[col].mean(), color="#F44336", linestyle="--", linewidth=1.5)
    ax.set_title(col.split("[")[0].strip())
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_sensor_distributions.png")
plt.close()
print("  Saved: 02_sensor_distributions.png")

# ── Plot 3: Correlation Heatmap ───────────────────────────────────────────────
df_corr = df[sensors + [target] + Failure_types].copy()
# Include engineered features to show they correlate better
df_corr["temp_delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df_corr["power"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
df_corr["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

corr = df_corr.corr(method="spearman")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle = redundant
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Spearman Correlation Matrix\n(includes engineered features)",
             fontsize=12)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_correlation_heatmap.png")
plt.close()
print("  Saved: 03_correlation_heatmap.png")

# ── Plot 4: Multi-Label Co-occurrence ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(co.astype(int), annot=True, fmt="d",
            cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Failure Type Co-occurrence\n(diagonal = individual count)",
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_cooccurrence.png")
plt.close()
print("  Saved: 04_cooccurrence.png")

# ── Plot 5: Sensor readings by Product Type ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
fig.suptitle("Sensor Readings by Product Type (L / M / H)",
             fontsize=13, fontweight="bold")
type_colors = ["#2196F3", "#FF9800", "#F44336"]

for i, col in enumerate(sensors):
    ax = axes[i]
    data = [df[df["Type"] == pt][col].values for pt in ["L", "M", "H"]]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], type_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_title(col.split("[")[0].strip())
    ax.set_ylabel(col)

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_sensors_by_type.png")
plt.close()
print("  Saved: 05_sensors_by_type.png")

# ── Plot 6: Tool Wear vs TWF (the clearest physical relationship) ─────────────
fig, ax = plt.subplots(figsize=(10, 5))
normal_scatter = df[df["TWF"] == 0]
failure_scatter = df[df["TWF"] == 1]

ax.scatter(normal_scatter["Tool wear [min]"], normal_scatter["Torque [Nm]"],
           c="#2196F3", alpha=0.15, s=5, label="Normal")
ax.scatter(failure_scatter["Tool wear [min]"], failure_scatter["Torque [Nm]"],
           c="#F44336", alpha=0.9, s=50, marker="X",
           label="TWF Failure", zorder=5)
ax.axvline(200, color="black", linestyle="--", linewidth=2,
           label="Wear threshold = 200 min")

ax.set_title("Tool Wear vs Torque — Tool Wear Failure (TWF)", fontsize=12)
ax.set_xlabel("Tool Wear [min]")
ax.set_ylabel("Torque [Nm]")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_toolwear_vs_twf.png")
plt.close()
print("  Saved: 06_toolwear_vs_twf.png")

# ── Plot 7: Temp Delta vs HDF ─────────────────────────────────────────────────
df_plot = df.copy()
df_plot["temp_delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_plot[df_plot["HDF"] == 0]["temp_delta"], bins=60,
        alpha=0.5, color="#2196F3", density=True, label="Normal")
ax.hist(df_plot[df_plot["HDF"] == 1]["temp_delta"], bins=30,
        alpha=0.8, color="#F44336", density=True, label="HDF Failure")
ax.axvline(8.6, color="black", linestyle="--", linewidth=2,
           label="Threshold: delta = 8.6 K")
ax.axvspan(df_plot["temp_delta"].min(), 8.6,
           alpha=0.1, color="#F44336", label="HDF risk zone")
ax.set_title("Temperature Delta (Process - Air) vs HDF", fontsize=12)
ax.set_xlabel("Temp Delta [K]")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_tempdelta_vs_hdf.png")
plt.close()
print("  Saved: 07_tempdelta_vs_hdf.png")

# ── Plot 8: Power vs PWF ──────────────────────────────────────────────────────
df_plot["power"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_plot[df_plot["PWF"] == 0]["power"], bins=60,
        alpha=0.5, color="#2196F3", density=True, label="Normal")
ax.hist(df_plot[df_plot["PWF"] == 1]["power"], bins=20,
        alpha=0.8, color="#FF9800", density=True, label="PWF Failure")
ax.axvline(3500, color="blue", linestyle="--", linewidth=2, label="Lower: 3500 W")
ax.axvline(9000, color="red", linestyle="--", linewidth=2, label="Upper: 9000 W")
ax.axvspan(0, 3500, alpha=0.1, color="blue")
ax.axvspan(9000, df_plot["power"].max(), alpha=0.1, color="red")
ax.set_title("Power (Speed × Torque) vs Power Failure (PWF)", fontsize=12)
ax.set_xlabel("Power [W]")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_power_vs_pwf.png")
plt.close()
print("  Saved: 08_power_vs_pwf.png")



