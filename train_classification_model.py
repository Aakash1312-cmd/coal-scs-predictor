"""
Multimodal Fusion Classification Training Pipeline
===================================================
Merges Ash, Fixed Carbon, and Ignition Temperature feature matrices.
Scales features and trains a RandomForestClassifier (n_estimators=500, random_state=42).
Evaluates on Train (70%), Test (20%), Validation (10%) splits and 5-Fold StratifiedKFold CV.
Saves model bundle to saved_models/classification_pipeline.joblib and confusion matrix plot.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

SAVED_MODELS_DIR = "saved_models"
DATA_DIR = "data"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def resolve_file(filename):
    if os.path.exists(os.path.join(DATA_DIR, filename)):
        return os.path.join(DATA_DIR, filename)
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Missing required file: {filename}")


# ==============================================================================
# 1. MERGING FEATURE MATRICES (MULTIMODAL FUSION)
# ==============================================================================
def merge_feature_matrices():
    print("=" * 80)
    print("STEP 1: MULTIMODAL FEATURE MERGING")
    print("=" * 80)

    ash_path = resolve_file("features_ash_content.csv")
    carbon_path = resolve_file("features_carbon_content.csv")
    ignition_path = resolve_file("features_ignition_temp.csv")

    ash = pd.read_csv(ash_path)
    carbon = pd.read_csv(carbon_path)
    ignition = pd.read_csv(ignition_path)

    # Carbon: keep instance_id and non-duplicate feature columns
    carbon_keep = ["instance_id"]
    for col in carbon.columns:
        if col not in ash.columns:
            carbon_keep.append(col)
    carbon_sub = carbon[carbon_keep]

    # Ignition: keep instance_id and non-duplicate feature columns
    existing = set(list(ash.columns) + list(carbon_sub.columns))
    ignition_keep = ["instance_id"]
    for col in ignition.columns:
        if col not in existing:
            ignition_keep.append(col)
    ignition_sub = ignition[ignition_keep]

    # Inner merge on instance_id
    merged = ash.merge(carbon_sub, on="instance_id", how="inner")
    merged = merged.merge(ignition_sub, on="instance_id", how="inner")

    print("\nMerged Shape:")
    print(merged.shape)

    output_path = "new_merged_features_with_class.csv"
    merged.to_csv(output_path, index=False)
    if os.path.exists(DATA_DIR):
        merged.to_csv(os.path.join(DATA_DIR, output_path), index=False)
    print(f"\nSaved {output_path}")

    return merged


# ==============================================================================
# 2. FEATURE PREPROCESSING & SCALING
# ==============================================================================
def preprocess_features(df):
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE PREPROCESSING & SCALING")
    print("=" * 80)

    print("Original Shape:")
    print(df.shape)

    meta_cols = ["instance_id", "coal_sample", "pellet", "subsample", "scs_class"]
    meta = df[meta_cols]

    X = df.drop(
        columns=["instance_id", "coal_sample", "pellet", "subsample", "scs_class", "target_value"],
        errors="ignore"
    )
    print("\nFeature Shape:")
    print(X.shape)

    # Impute missing values with median
    imputer_median = X.median(numeric_only=True)
    X = X.fillna(imputer_median)

    print("\nRemaining NaN Values:")
    print(X.isnull().sum().sum())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    final_df = pd.concat([meta.reset_index(drop=True), X_scaled.reset_index(drop=True)], axis=1)

    output_path = "new_scaled_dataset.csv"
    final_df.to_csv(output_path, index=False)
    if os.path.exists(DATA_DIR):
        final_df.to_csv(os.path.join(DATA_DIR, output_path), index=False)

    print("\nScaled Shape:")
    print(final_df.shape)
    print(f"\nSaved: {output_path}")

    return final_df, imputer_median, scaler, X.columns.tolist()


# ==============================================================================
# 3. RANDOM FOREST CLASSIFICATION TRAINING & EVALUATION
# ==============================================================================
def train_and_evaluate(df, imputer_median, scaler, feature_names):
    print("\n" + "=" * 80)
    print("STEP 3: CLASSIFICATION (RandomForest)")
    print("=" * 80)

    label_map = {
        "LOW": 0, "MEDIUM": 1, "HIGH": 2,
        "Low": 0, "Medium": 1, "High": 2,
        "low": 0, "medium": 1, "high": 2
    }
    inv_label_map = {0: "LOW RISK", 1: "MODERATE RISK", 2: "HIGH RISK"}

    X = df.drop(columns=["instance_id", "coal_sample", "pellet", "subsample", "scs_class"], errors="ignore")

    all_nan_cols = [col for col in X.columns if X[col].isnull().all()]
    if all_nan_cols:
        print("\nAll NaN Columns:")
        print(all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    y = df["scs_class"].map(label_map)

    print("\nDataset Shape:")
    print(df.shape)
    print("\nFeature Shape:")
    print(X.shape)
    print("\nTotal NaN Values:")
    print(X.isnull().sum().sum())

    # TRAIN : TEST : VALIDATION = 70 : 20 : 10
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape :", X_test.shape)
    print("Validation Shape :", X_val.shape)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        # Train
        model.fit(X_train, y_train)

        # Accuracies
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        # 5-Fold Stratified Cross Validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        print(f"\nTrain Accuracy : {train_acc:.4f}")
        print(f"Validation Accuracy : {val_acc:.4f}")
        print(f"Test Accuracy : {test_acc:.4f}")
        print(f"CV Mean Accuracy : {cv_mean:.4f}")
        print(f"CV Std : {cv_std:.4f}")

        print("\nClassification Report\n")
        print(classification_report(y_test, test_pred, target_names=["LOW", "MEDIUM", "HIGH"]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["LOW", "MEDIUM", "HIGH"],
            yticklabels=["LOW", "MEDIUM", "HIGH"],
            ax=ax
        )
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()

        cm_filename = f"{name}_confusion_matrix.png"
        fig.savefig(cm_filename, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {cm_filename}")
        plt.close(fig)

        results.append([name, train_acc, val_acc, test_acc, cv_mean, cv_std])

        # Save deployment pipeline bundle
        bundle = {
            "model": model,
            "scaler": scaler,
            "imputer_median": imputer_median,
            "feature_names": feature_names,
            "classes": ["LOW", "MEDIUM", "HIGH"],
            "label_map": label_map,
            "inv_label_map": inv_label_map,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "confusion_matrix_path": cm_filename
        }
        bundle_path = os.path.join(SAVED_MODELS_DIR, "classification_pipeline.joblib")
        joblib.dump(bundle, bundle_path)
        print(f"\nSaved Deployment Bundle to: {bundle_path}")

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Train Accuracy", "Validation Accuracy", "Test Accuracy", "CV Mean", "CV Std"]
    )
    print("\nResults Summary:")
    print(results_df)

    return results_df


def main():
    merged_df = merge_feature_matrices()
    scaled_df, imputer_median, scaler, feature_names = preprocess_features(merged_df)
    train_and_evaluate(scaled_df, imputer_median, scaler, feature_names)


if __name__ == "__main__":
    main()
