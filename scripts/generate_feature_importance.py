import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# Real IEEE-CIS feature names -> (group, base relative importance)
FEATURES = {
    "TransactionAmt": ("Transaction", 0.075),
    "ProductCD":      ("Transaction", 0.030),
    "TransactionDT":  ("Transaction", 0.018),
    "card1": ("Card", 0.090), "card2": ("Card", 0.052),
    "card5": ("Card", 0.034), "card6": ("Card", 0.020),
    "card3": ("Card", 0.014),
    "addr1": ("Address", 0.080), "addr2": ("Address", 0.022),
    "dist1": ("Address", 0.045), "dist2": ("Address", 0.016),
    "P_emaildomain": ("Email", 0.040), "R_emaildomain": ("Email", 0.021),
    "C13": ("Counting", 0.048), "C1": ("Counting", 0.039),
    "C14": ("Counting", 0.031), "C2": ("Counting", 0.024),
    "C5": ("Counting", 0.017), "C9": ("Counting", 0.012),
    "D15": ("Timedelta", 0.050), "D10": ("Timedelta", 0.036),
    "D4": ("Timedelta", 0.028), "D2": ("Timedelta", 0.022),
    "D1": ("Timedelta", 0.019), "D3": ("Timedelta", 0.013),
    "D8": ("Timedelta", 0.011),
    "M4": ("Match", 0.030), "M5": ("Match", 0.018),
    "M6": ("Match", 0.014), "M3": ("Match", 0.009),
    "V317": ("Vesta", 0.033), "V307": ("Vesta", 0.027),
    "V310": ("Vesta", 0.021), "V130": ("Vesta", 0.018),
    "V313": ("Vesta", 0.015), "V279": ("Vesta", 0.013),
    "V294": ("Vesta", 0.012), "V62": ("Vesta", 0.010),
    "V187": ("Vesta", 0.009), "V201": ("Vesta", 0.008),
    "V258": ("Vesta", 0.007), "V45": ("Vesta", 0.006),
}

rows = []
for feat, (group, base) in FEATURES.items():
    imp = max(base * rng.lognormal(0, 0.12), 0.0005)
    rows.append({"Feature": feat, "FeatureGroup": group, "Importance": imp})

df = pd.DataFrame(rows)
df["Importance"] = df["Importance"] / df["Importance"].sum()   # normalize to sum = 1
df["Importance"] = df["Importance"].round(5)
df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
df.to_csv("feature_importances.csv", index=False)

print(f"Wrote feature_importances.csv with {len(df)} features")
print(df.head(10).to_string(index=False))
print("\nImportance by group:")
print(df.groupby("FeatureGroup")["Importance"].sum()
        .sort_values(ascending=False).round(3).to_string())