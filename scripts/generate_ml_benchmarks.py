import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

MODELS = {
    "Logistic Regression": {"family": "Linear",   "auc": 0.860, "train": 2.1,   "infer": 0.4},
    "Decision Tree":       {"family": "Tree",     "auc": 0.842, "train": 4.8,   "infer": 0.6},
    "Random Forest":       {"family": "Ensemble", "auc": 0.931, "train": 38.0,  "infer": 9.5},
    "XGBoost":             {"family": "Ensemble", "auc": 0.951, "train": 21.0,  "infer": 3.2},
    "LightGBM":            {"family": "Ensemble", "auc": 0.948, "train": 9.5,   "infer": 1.8},
    "SVM (RBF)":           {"family": "Kernel",   "auc": 0.890, "train": 145.0, "infer": 22.0},
    "KNN":                 {"family": "Instance", "auc": 0.851, "train": 0.8,   "infer": 31.0},
    "Neural Net (MLP)":    {"family": "Neural",   "auc": 0.912, "train": 86.0,  "infer": 2.4},
}

DATASETS = {
    "Fraud Detection":  {"difficulty":  0.00, "size": 285000},
    "Loan Default":     {"difficulty": -0.02, "size": 45000},
    "Churn Prediction": {"difficulty": -0.04, "size": 7000},
}

rows = []
for ds_name, ds in DATASETS.items():
    size_factor = ds["size"] / 285000
    for model_name, m in MODELS.items():
        for fold in range(1, 6):
            auc = np.clip(m["auc"] + ds["difficulty"] + rng.normal(0, 0.008), 0.5, 0.999)
            recall = np.clip(auc - 0.10 + rng.normal(0, 0.02), 0.30, 0.99)
            precision = np.clip(auc - 0.06 + rng.normal(0, 0.02), 0.30, 0.99)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = np.clip(auc + 0.02 + rng.normal(0, 0.005), 0.5, 0.999)
            train_time = max(m["train"] * size_factor * rng.lognormal(0, 0.10), 0.05)
            infer_ms = max(m["infer"] * rng.lognormal(0, 0.08), 0.05)
            rows.append({
                "Dataset": ds_name,
                "Model": model_name,
                "ModelFamily": m["family"],
                "Fold": fold,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "ROC_AUC": round(auc, 4),
                "TrainingTimeSec": round(train_time, 2),
                "InferenceMsPer1k": round(infer_ms, 2),
            })

df = pd.DataFrame(rows)
df.to_csv("model_benchmarks.csv", index=False)
print(f"Wrote model_benchmarks.csv with {len(df)} rows")
print(df.head(8))