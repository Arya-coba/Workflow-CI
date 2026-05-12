import os
import pickle
import argparse
import dagshub
import mlflow
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",      type=int, default=100)
parser.add_argument("--max_depth",         type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=5)
parser.add_argument("--min_samples_leaf",  type=int, default=1)
parser.add_argument("--max_features",      type=str, default="log2")
args = parser.parse_args()

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
DAGSHUB_USERNAME  = os.environ.get("DAGSHUB_USERNAME", "Arya-coba")
DAGSHUB_REPO_NAME = "Workflow-CI"

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "creditcard_preprocessing")
TRAIN_PATH   = os.path.join(DATA_DIR, "train.csv")
TEST_PATH    = os.path.join(DATA_DIR, "test.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_COL      = "Class"
EXPERIMENT_NAME = "CreditCard-Fraud-CI"

# ──────────────────────────────────────────────
# INIT DAGSHUB + MLFLOW
# ──────────────────────────────────────────────
dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True
)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(disable=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
print("[INFO] Memuat dataset...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=[TARGET_COL])
y_train = train[TARGET_COL]
X_test  = test.drop(columns=[TARGET_COL])
y_test  = test[TARGET_COL]

print(f"      Train : {X_train.shape}")
print(f"      Test  : {X_test.shape}")

# ──────────────────────────────────────────────
# TRAINING + LOGGING
# ──────────────────────────────────────────────
print("[INFO] Training model...")

model = RandomForestClassifier(
    n_estimators      = args.n_estimators,
    max_depth         = args.max_depth,
    min_samples_split = args.min_samples_split,
    min_samples_leaf  = args.min_samples_leaf,
    max_features      = args.max_features,
    random_state      = 42,
    n_jobs            = -1
)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"      Accuracy : {acc:.4f} | F1 : {f1:.4f} | ROC-AUC : {roc_auc:.4f}")

# ──────────────────────────────────────────────
# BUAT ARTEFAK
# ──────────────────────────────────────────────
# Confusion Matrix
cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Fraud"])
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()

# ROC Curve
roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={roc_auc:.4f}")
ax.plot([0,1],[0,1],"k--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curve"); ax.legend()
plt.tight_layout()
plt.savefig(roc_path, dpi=150)
plt.close()

# Model pickle
model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# ──────────────────────────────────────────────
# LOG KE DAGSHUB — run baru (bukan dari MLProject)
# ──────────────────────────────────────────────
print("[INFO] Logging ke DagsHub...")

# Nonaktifkan MLFLOW_RUN_ID dari environment agar tidak konflik
os.environ.pop("MLFLOW_RUN_ID", None)

with mlflow.start_run(run_name="RandomForest-CI") as run:
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
    mlflow.log_param("max_features",      args.max_features)

    mlflow.log_metric("accuracy",         acc)
    mlflow.log_metric("precision",        precision)
    mlflow.log_metric("recall",           recall)
    mlflow.log_metric("f1_score",         f1)
    mlflow.log_metric("roc_auc",          roc_auc)
    mlflow.log_metric("true_positives",   int(tp))
    mlflow.log_metric("false_positives",  int(fp))
    mlflow.log_metric("true_negatives",   int(tn))
    mlflow.log_metric("false_negatives",  int(fn))
    mlflow.log_metric("fraud_recall",     float(tp/(tp+fn)) if (tp+fn)>0 else 0)

    mlflow.log_artifact(cm_path,    artifact_path="plots")
    mlflow.log_artifact(roc_path,   artifact_path="plots")
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"\n{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"{'='*50}")
    print(f"\n✅ Run ID: {run.info.run_id}")
