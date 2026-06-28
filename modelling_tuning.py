import os
import time
import json
import mlflow
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import estimator_html_repr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score, f1_score, accuracy_score)

# Inisialisasi Dagshub (otomatis set up MLflow URI & Token)
dagshub.init(
    repo_owner="Rizki-Kidut", 
    repo_name="Sistem_ML_Heart_Disease", 
    mlflow=True
)

# Create a new MLflow Experiment
mlflow.set_experiment("Modelling_SML_Heart_Disesase_Tuning")

target_col = 'num'
n_classes = 5

df_train = pd.read_csv("heart_disease_train.csv")

X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col].astype(int)

df_test = pd.read_csv("heart_disease_test.csv")

X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col].astype(int)

input_example = X_train[0:5]

print(f"\n✅ Train : {X_train.shape}")
print(f"✅ Test  : {X_test.shape}")
print(f"📊 Distribusi kelas (train): {y_train.value_counts().sort_index().to_dict()}")
print(f"📊 Distribusi kelas (test) : {y_test.value_counts().sort_index().to_dict()}")

Class_names = [f"Kelas {i}" for i in range(n_classes)]

models_params = {
 
    "Logistic Regression": {
        "model": LogisticRegression(
            random_state=42,
            max_iter=5000,
            solver="saga"
        ),
        "params": {
            "C"         : [0.01, 0.1, 1, 10, 100],
            "l1_ratio"  : [0, 0.5, 1]
        }
    },
 
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators"     : [50, 100, 200],
            "max_depth"        : [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf" : [1, 2, 4]
        }
    },
 
    "XGBoost": {
        "model": XGBClassifier(
            random_state=42,
            objective='multi:softprob',   # multi-class
            num_class=n_classes,
            eval_metric='mlogloss'
        ),
        "params": {
            "n_estimators" : [50, 100, 200],
            "max_depth"    : [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample"    : [0.7, 0.8, 1.0]
        }
    },
 
    "SVM": {
        "model": CalibratedClassifierCV(
            estimator=SVC(
                random_state=42,
                decision_function_shape='ovr'
                # probability=True dihapus dari sini
            ),
            ensemble=False
        ),
        "params": {
            # Tambahkan awalan 'estimator__' agar GridSearchCV tahu
            # parameter ini milik SVC yang ada di dalam bungkusannya
            "estimator__C"     : [0.1, 1, 10, 100],
            "estimator__kernel": ["linear", "rbf"],
            "estimator__gamma" : ["scale", "auto"]
        }
    }
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results= {}
os.makedirs("modelling/artifacts", exist_ok=True)

for model_name, mp in models_params.items():
    print(f"\n{'─'*60}")
    print(f"  🔍 Tuning: {model_name}")
    print(f"{'─'*60}")

    with mlflow.start_run(run_name=model_name):

        mlflow.set_tags({
            "model"     : model_name,
            "dataset"   : "Heart Disease UCI",
            "task"      : "multiclass_classification",
            "n_classes" : n_classes,
            "developer" : "Rizki Hidayat"
        })

        start = time.time()
        grid_search = GridSearchCV(
            estimator   = mp["model"],
            param_grid  = mp["params"],
            cv          = cv,
            scoring     = "roc_auc_ovr_weighted",
            n_jobs      = -1,
            verbose     = 1,
            refit       = True
        )

        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start

        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring_metric", "roc_auc_ovr_weighted")
        mlflow.log_param("tuning_time_s", round(elapsed, 2))

        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        if y_proba.ndim == 1 or y_proba.shape[1] < n_classes:
            classes    = best_model.classes_.astype(int)
            full_proba = np.zeros((len(X_test), n_classes))
            for col_idx, class_val in enumerate(classes):
                if y_proba.ndim == 1:
                    full_proba[:, class_val] = y_proba
                else:
                    full_proba[:, class_val] = y_proba[:, col_idx]
            y_proba = full_proba

        # ✅ Handle NaN/Inf akibat model tidak konvergen
        # (saga yang belum konvergen bisa menghasilkan NaN di probabilitas)
        y_proba = np.nan_to_num(
            y_proba,
            nan=1.0 / n_classes,   # ganti NaN dengan probabilitas uniform
            posinf=1.0,
            neginf=0.0
        )

        # ✅ Normalisasi agar tiap baris sum = 1 (wajib untuk roc_auc_score)
        row_sums = y_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # hindari division by zero
        y_proba = y_proba / row_sums

        acc      = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_wt    = f1_score(y_test, y_pred, average='weighted')
        roc_auc  = roc_auc_score(
            y_test, y_proba,
            multi_class='ovr',
            average='weighted'
        )
 
        # ── Log semua metrics ke MLflow
        mlflow.log_metrics({
            "cv_roc_auc_weighted"  : round(grid_search.best_score_, 4),
            "test_roc_auc_weighted": round(roc_auc, 4),
            "test_accuracy"        : round(acc, 4),
            "test_f1_macro"        : round(f1_macro, 4),
            "test_f1_weighted"     : round(f1_wt, 4),
            "tuning_time_seconds"  : round(elapsed, 2)
        })

        # ── 1. Buat & Log estimator.html
        html_content = estimator_html_repr(best_model)
        html_path = f"modelling/artifacts/estimator.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        mlflow.log_artifact(html_path)

        # ── 2. Buat & Log metric_info.json
        # Kita kumpulkan metrik yang sudah Anda hitung sebelumnya
        metrics_dict = {
            "cv_roc_auc_weighted": round(grid_search.best_score_, 4),
            "test_roc_auc_weighted": round(roc_auc, 4),
            "test_accuracy": round(acc, 4),
            "test_f1_macro": round(f1_macro, 4),
            "test_f1_weighted": round(f1_wt, 4)
        }
        json_path = f"modelling/artifacts/metric_info.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact(json_path)

        # ── 3. Buat & Log training_confusion_matrix.png
        # Karena namanya 'training_', kita prediksi menggunakan X_train
        y_train_pred = best_model.predict(X_train)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_train, 
            y_train_pred, 
            display_labels=Class_names, 
            cmap="Blues", 
            ax=ax
        )
        plt.title(f"Training Confusion Matrix - {model_name}")
        cm_path = f"modelling/artifacts/training_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig) # Tutup figure agar tidak menumpuk di memori
        mlflow.log_artifact(cm_path)
 
        # Log F1 per kelas
        f1_per_class = f1_score(y_test, y_pred, average=None)
        for i, score in enumerate(f1_per_class):
            mlflow.log_metric(f"test_f1_class_{i}", round(score, 4))
        
        # ── Simpan & Log Classification Report sebagai artifact
        report     = classification_report(y_test, y_pred, target_names=Class_names)
        report_path = f"modelling/artifacts/report_{model_name.replace(' ', '_')}.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Params: {grid_search.best_params_}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="classification_report")

        # ── Log Model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, artifact_path="model", input_example=input_example)
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=input_example)
 
        # ── Print ringkasan
        print(f"\n  ⏱️  Waktu         : {elapsed:.1f}s")
        print(f"  🏆 Best Params   : {grid_search.best_params_}")
        print(f"  📊 CV ROC-AUC    : {grid_search.best_score_:.4f}")
        print(f"  📊 Test ROC-AUC  : {roc_auc:.4f}")
        print(f"  📊 Test Accuracy : {acc:.4f}")
        print(f"  📊 Test F1 Macro : {f1_macro:.4f}")
        print(f"\n{report}")
 
        results[model_name] = {
            "cv_roc_auc" : grid_search.best_score_,
            "test_roc_auc": roc_auc,
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "best_params" : grid_search.best_params_,
            "elapsed"     : elapsed
        }

        # ── Simpan ringkasan ke CSV
        summary_df = pd.DataFrame([
            {"Model": n, **{k: v for k, v in r.items() if k != "best_params"},
            **{f"param_{k}": v for k, v in r["best_params"].items()}}
            for n, r in results.items()
        ])


        summary_path = "modelling/tuning_results.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Hasil disimpan ke : {summary_path}")
        print(f"💾 Artifacts di      : modelling/artifacts/")
        print(f"\n🚀 Lihat MLflow UI  : mlflow ui  →  http://127.0.0.1:5000")
        print("\n" + "="*60)
        print("  ✅ SELESAI")
        print("="*60 + "\n")