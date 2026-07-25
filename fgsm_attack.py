#!/usr/bin/env python3
"""
FGSM attack on Random Forest using the trained/source MLP model, with optional attack scope.

This version is meant to work with the normal MLP output folder, for example:
    mlp_results_full/
        mlp_model.pt
        mlp_preprocessor.pkl
        mlp_summary.json

Key idea:
- The MLP is NOT trained to imitate Random Forest.
- The MLP is the source/gradient model trained on true labels.
- FGSM examples are generated through the MLP gradients.
- The adversarial raw examples are evaluated against the Random Forest.
"""

import os
import json
import random
import argparse
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_RF_RESULTS_DIR = "rf_results"
DEFAULT_MLP_DIR = "mlp_results"
DEFAULT_OUTPUT_DIR = "fgsm_rf_full_results"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

EPSILON_LIST = [1.5]

# This is only a fallback. If mlp_model.pt has hidden_dims/dropout saved,
# the checkpoint values will be used instead.
DEFAULT_HIDDEN_DIMS = (128, 64, 32)


# ============================================================
# FEATURE SETS
# ============================================================

NON_FEATURE_COLS = ["student_id", "academic_year", "target"]

# This script only runs the full-feature model, matching the RF and MLP scripts.
FULL_FEATURES = [
    # static / enrolment
    "degree_size",
    "seniority",
    "highest_course_year_enrolled",
    "adapted_studies_flag",
    "credits_enrolled_semester_a",
    "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",

    # prior-year history
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
    # semester-A performance
    "pass_ratio_sem_a",
    "pass_ratio_sem_a_and_zero_flag",
    # semester-A activity
    "lms_events_sem_a",
    "lms_assignment_submissions_sem_a",
    "lms_test_submissions_sem_a",
    "lms_total_minutes_sem_a",
    "attendance_days_sem_a",
    # full-year course aggregates
    "n_courses",
    "courses_mean",
    # full-year / semester-B outcomes
    "total_credits_passed_academic_year",
    "pass_ratio_sem_b",
    "pass_ratio_sem_b_and_zero_flag",
    "total_performance",
    "total_performance_and_zero_flag",
    "obtained_new_degree_flag",
    # semester-B activity
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_b",
    "attendance_days_sem_b",
]


# ============================================================
# ATTACKABLE FEATURES
# ============================================================

# Only these raw features are allowed to move. Flags and categorical one-hot columns
# are frozen by default.
ATTACKABLE_FEATURES = [
    "credits_enrolled_semester_a",
    "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "courses_mean",
    "total_credits_passed_academic_year",
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
    "lms_events_sem_a",
    "lms_assignment_submissions_sem_a",
    "lms_test_submissions_sem_a",
    "lms_total_minutes_sem_a",
    "attendance_days_sem_a",
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_b",
    "attendance_days_sem_b",
]

ROUND_TO_INT_FEATURES = [
    "lms_events_sem_a",
    "lms_assignment_submissions_sem_a",
    "lms_test_submissions_sem_a",
    "attendance_days_sem_a",
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_b",
    "attendance_days_sem_b",
]

PERCENTAGE_FEATURES = [
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
]

NONNEGATIVE_FEATURES = sorted(set(ATTACKABLE_FEATURES))


# ============================================================
# MODEL: must match the training MLP exactly
# ============================================================

class DropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=DEFAULT_HIDDEN_DIMS, dropout=0.30):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# BASIC HELPERS
# ============================================================

def set_global_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_python(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def one_hot(labels, n_classes=2):
    labels = np.asarray(labels).astype(int)
    return np.eye(n_classes, dtype=np.float32)[labels]


def validate_feature_set_definitions():
    duplicate_full = sorted({c for c in FULL_FEATURES if FULL_FEATURES.count(c) > 1})
    if duplicate_full:
        raise ValueError(f"Invalid FULL_FEATURES definition. Duplicated columns: {duplicate_full}")


def get_selected_features(feature_set):
    if feature_set != "full":
        raise ValueError("Unsupported feature_set. This script only supports feature_set='full'.")
    return FULL_FEATURES.copy()


def normalize_target(series):
    target_normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "non dropout": "non-dropout",
            "non_dropout": "non-dropout",
            "drop out": "dropout",
        })
    )

    target_map = {
        "non-dropout": 0,
        "dropout": 1,
        "b": 0,
        "a": 1,
        "0": 0,
        "1": 1,
    }

    y = target_normalized.map(target_map)
    if y.isna().any():
        bad_values = series.loc[y.isna()].unique().tolist()
        raise ValueError(f"Ada target yang tidak bisa dipetakan: {bad_values}")

    return y.astype(int)


def prepare_features_and_label(df, feature_set):
    validate_feature_set_definitions()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan.")

    selected_features = get_selected_features(feature_set)
    missing_features = [c for c in selected_features if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Feature set {feature_set} butuh kolom yang tidak ada:\n{missing_features}"
        )

    X = df[selected_features].copy()
    y = normalize_target(df[TARGET_COL])

    if "student_id" in df.columns:
        student_ids = df["student_id"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")

    return X, y, student_ids, selected_features


def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    """
    Stratified random split with no grouping. This is identical to the split used
    by the Random Forest and MLP training scripts:
    - approximately 70% train
    - approximately 20% test
    - approximately 10% validation

    The split is stratified by label so the dropout proportion is preserved across
    all splits. Each row is treated as an independent observation. Using the same
    split logic, the same input file, and the same random_state as the RF/MLP
    training scripts guarantees that the test/validation rows here are exactly the
    rows those models were evaluated on (and are disjoint from their training rows).
    """
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")

    # Step 1: 70% train / 30% temp
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X,
        y,
        student_ids,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: split the 30% temp into 20% test and 10% validation.
    # test_size = 1/3 of temp -> validation = 10% of total, test = 20% of total.
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=1.0 / 3.0,
        stratify=y_temp,
        random_state=random_state,
    )

    split_report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "validation_rows": len(X_val),
        "train_pct": len(X_train) / len(X),
        "test_pct": len(X_test) / len(X),
        "validation_pct": len(X_val) / len(X),
        "train_label_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_label_distribution": y_test.value_counts().sort_index().to_dict(),
        "validation_label_distribution": y_val.value_counts().sort_index().to_dict(),
    }

    return (
        X_train, X_test, X_val,
        y_train, y_test, y_val,
        id_train, id_test, id_val,
        split_report,
    )


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_rf_model_path(rf_results_dir, feature_set):
    candidates = [
        os.path.join(rf_results_dir, feature_set, "rf_model.pkl"),
        os.path.join(rf_results_dir, "rf_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("RF model tidak ditemukan. Dicek:\n" + "\n".join(candidates))


def resolve_rf_summary_path(rf_results_dir, feature_set):
    candidates = [
        os.path.join(rf_results_dir, feature_set, "rf_summary.json"),
        os.path.join(rf_results_dir, "rf_summary.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_rf_threshold(rf_summary_path, fallback=0.5):
    if rf_summary_path and os.path.exists(rf_summary_path):
        with open(rf_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        threshold = summary.get("selection", {}).get("best_threshold", None)
        if threshold is not None:
            return float(threshold), rf_summary_path
    return float(fallback), "fallback_0.5"


def resolve_mlp_model_path(mlp_dir, feature_set):
    candidates = [
        os.path.join(mlp_dir, "mlp_model.pt"),
        os.path.join(mlp_dir, feature_set, "mlp_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("MLP model tidak ditemukan. Dicek:\n" + "\n".join(candidates))


def resolve_mlp_preprocessor_path(mlp_dir, feature_set):
    candidates = [
        os.path.join(mlp_dir, "mlp_preprocessor.pkl"),
        os.path.join(mlp_dir, feature_set, "mlp_preprocessor.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("MLP preprocessor tidak ditemukan. Dicek:\n" + "\n".join(candidates))


def load_torch_checkpoint(path, device):
    # PyTorch versions differ on weights_only support/default.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mlp_model(model_path, input_dim, device):
    obj = load_torch_checkpoint(model_path, device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        hidden_dims = tuple(obj.get("hidden_dims", DEFAULT_HIDDEN_DIMS))
        dropout = float(obj.get("dropout", 0.30))
        state_dict = obj["model_state_dict"]
        checkpoint_features = obj.get("selected_features", None)
        checkpoint_feature_set = obj.get("feature_set", None)
        checkpoint_threshold = obj.get("best_threshold", None)
    else:
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        state_dict = obj
        checkpoint_features = None
        checkpoint_feature_set = None
        checkpoint_threshold = None

    model = DropoutMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, {
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "selected_features": checkpoint_features,
        "feature_set": checkpoint_feature_set,
        "best_threshold": checkpoint_threshold,
    }


def wrap_mlp_for_art(model, input_dim):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=2,
    )


# ============================================================
# PREPROCESSOR HELPERS AND MASKS
# ============================================================

def transform_to_float32(preprocessor, X):
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)


def get_cols_from_preprocessor(preprocessor, transformer_name):
    for name, transformer, cols in preprocessor.transformers_:
        if name == transformer_name:
            return list(cols)
    return []


def get_numeric_cols_from_preprocessor(preprocessor):
    cols = get_cols_from_preprocessor(preprocessor, "num")
    if not cols:
        raise ValueError("Transformer 'num' tidak ditemukan di preprocessor.")
    return cols


def get_scaler_from_preprocessor(preprocessor):
    num_pipeline = preprocessor.named_transformers_["num"]
    if "scaler" not in num_pipeline.named_steps:
        raise ValueError("Scaler tidak ditemukan di numeric pipeline.")
    return num_pipeline.named_steps["scaler"]


def make_preprocessed_attack_mask(preprocessor, selected_features, feature_set, input_dim, output_dir):
    """
    Build an ART mask in transformed/preprocessed space.

    ColumnTransformer output order in the MLP preprocessor is:
    numeric block, flag block, categorical one-hot block.
    We only allow numeric transformed columns corresponding to attackable raw features.
    """
    os.makedirs(output_dir, exist_ok=True)

    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)

    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(
            f"Attackable features ini tidak ada di selected_features untuk {feature_set}:\n"
            f"{missing_attackable}"
        )

    mask = np.zeros(input_dim, dtype=np.float32)

    # Numeric block starts at column 0 and has len(numeric_cols) columns.
    rows = []
    for idx, col in enumerate(numeric_cols):
        is_attackable = col in attackable_raw
        mask[idx] = 1.0 if is_attackable else 0.0
        rows.append({
            "preprocessed_index": idx,
            "raw_feature": col,
            "block": "num",
            "perturbable_mask": int(is_attackable),
            "attack_status": "attackable" if is_attackable else "frozen",
        })

    # Save raw-level mask too, because it is easier to read.
    raw_mask_df = pd.DataFrame({
        "feature": selected_features,
        "raw_attackable": [int(f in attackable_raw) for f in selected_features],
        "round_to_int": [int(f in ROUND_TO_INT_FEATURES) for f in selected_features],
        "attack_status": ["attackable" if f in attackable_raw else "frozen" for f in selected_features],
    })

    preprocessed_mask_df = pd.DataFrame(rows)

    raw_mask_df.to_csv(os.path.join(output_dir, "attack_mask_raw_features.csv"), index=False)
    preprocessed_mask_df.to_csv(os.path.join(output_dir, "attack_mask_preprocessed_numeric_block.csv"), index=False)
    np.save(os.path.join(output_dir, "perturbable_mask_preprocessed.npy"), mask)

    if int(mask.sum()) == 0:
        raise ValueError("Tidak ada preprocessed feature yang bisa diserang. Cek ATTACKABLE_FEATURES.")

    return mask, raw_mask_df, preprocessed_mask_df


def preprocessed_adv_to_raw_and_scaled(
    X_adv_preprocessed,
    X_clean_preprocessed,
    X_clean_raw,
    selected_features,
    preprocessor,
    feature_set,
    train_min,
    train_max,
    perturbable_mask_preprocessed,
):
    """
    Convert adversarial transformed data back to raw space.

    Important: the transformed array contains numeric + flag + one-hot categorical blocks.
    Only the numeric block can be inverse-transformed with StandardScaler.
    Frozen raw features are restored from the clean raw data.
    """
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    scaler = get_scaler_from_preprocessor(preprocessor)

    n_num = len(numeric_cols)
    adv_numeric_scaled = X_adv_preprocessed[:, :n_num]
    adv_numeric_raw = scaler.inverse_transform(adv_numeric_scaled)

    X_adv_raw = X_clean_raw[selected_features].reset_index(drop=True).copy()

    # Update numeric columns from adversarial numeric block.
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]

    # Restore non-attackable raw features.
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    # Clip numeric values to train min/max when bounds exist.
    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])

    # Domain constraints.
    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)

    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)

    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)

    # Restore frozen features again after clipping/rounding.
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[selected_features])

    # Hard-restore frozen transformed dimensions too.
    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]

    return X_adv_raw[selected_features], X_projected_preprocessed


# ============================================================
# EVALUATION AND SAVING
# ============================================================

def evaluate_rf_with_threshold(rf_model, X_raw, y_true, threshold, condition, epsilon, split):
    y_true = np.asarray(y_true).astype(int)
    y_proba = rf_model.predict_proba(X_raw)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    result = {
        "model": "Random Forest",
        "condition": condition,
        "epsilon": float(epsilon),
        "split": split,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dropout": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    return result, cm, y_pred, y_proba


def perturbation_stats(X_clean_preprocessed, X_adv_preprocessed, perturbable_mask):
    diff = X_adv_preprocessed - X_clean_preprocessed
    pm = perturbable_mask.astype(bool)
    frozen = ~pm

    return {
        "mean_abs_perturbation_all_preprocessed": float(np.abs(diff).mean()),
        "mean_abs_perturbation_attacked_preprocessed": float(np.abs(diff[:, pm]).mean()),
        "max_abs_perturbation_attacked_preprocessed": float(np.abs(diff[:, pm]).max()),
        "max_abs_perturbation_frozen_preprocessed": float(np.abs(diff[:, frozen]).max()) if frozen.any() else 0.0,
    }


def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)


def save_attack_predictions(
    student_ids,
    y_true,
    clean_pred,
    adv_pred,
    clean_proba,
    adv_proba,
    epsilon,
    split,
    path,
):
    y_true = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        "student_id": student_ids.values,
        "split": split,
        "epsilon": float(epsilon),
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "rf_clean_pred_label": clean_pred,
        "rf_clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "rf_adv_pred_label": adv_pred,
        "rf_adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "rf_clean_prob_dropout": clean_proba,
        "rf_adv_prob_dropout": adv_proba,
        "prob_dropout_change": adv_proba - clean_proba,
        "prediction_changed": clean_pred != adv_pred,
        "dropout_hidden_tp_to_fn": ((y_true == 1) & (clean_pred == 1) & (adv_pred == 0)),
        "false_alarm_tn_to_fp": ((y_true == 0) & (clean_pred == 0) & (adv_pred == 1)),
    })
    pred_df.to_csv(path, index=False)


def save_clean_vs_adv_sample(
    X_clean_raw,
    X_adv_raw,
    y_true,
    clean_pred,
    adv_pred,
    clean_proba,
    adv_proba,
    selected_features,
    epsilon,
    split,
    path,
    max_samples=50,
):
    n = min(max_samples, len(X_clean_raw))
    rows = []

    X_clean_raw = X_clean_raw.reset_index(drop=True)
    X_adv_raw = X_adv_raw.reset_index(drop=True)
    y_true = np.asarray(y_true).astype(int)

    for i in range(n):
        row = {
            "sample_id": i,
            "split": split,
            "epsilon": float(epsilon),
            "true_label": int(y_true[i]),
            "true_target": "dropout" if y_true[i] == 1 else "non-dropout",
            "rf_clean_probability_dropout": float(clean_proba[i]),
            "rf_adversarial_probability_dropout": float(adv_proba[i]),
            "rf_clean_prediction": int(clean_pred[i]),
            "rf_adversarial_prediction": int(adv_pred[i]),
            "prediction_changed": bool(clean_pred[i] != adv_pred[i]),
            "hidden_dropout": bool(y_true[i] == 1 and clean_pred[i] == 1 and adv_pred[i] == 0),
        }

        for feature in selected_features:
            row[f"clean_{feature}"] = X_clean_raw.loc[i, feature]
            row[f"adv_{feature}"] = X_adv_raw.loc[i, feature]
            row[f"diff_{feature}"] = X_adv_raw.loc[i, feature] - X_clean_raw.loc[i, feature]

        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# ATTACK LABELS
# ============================================================

def make_attack_labels(attack_mode, y_true):
    if attack_mode == "untargeted":
        # Paper-style source-model attack: maximize loss against TRUE labels.
        return one_hot(y_true), False

    if attack_mode == "target_non_dropout":
        # Target selected examples toward class 0 = non-dropout.
        target_labels = np.zeros(len(y_true), dtype=int)
        return one_hot(target_labels), True

    raise ValueError("attack_mode harus 'untargeted' atau 'target_non_dropout'.")


def make_attack_scope_mask(attack_scope, y_true, clean_pred):
    """
    Decide which rows should actually be perturbed.

    all:
        Perturb every row. Useful for general robustness tests, but for
        target_non_dropout it can also improve non-dropout false positives.
    dropout_only:
        Perturb all true dropout rows only.
    true_positive_only:
        Perturb only true dropout rows that RF correctly detected as dropout
        before the attack. This is the cleanest dropout-evasion setting.
    """
    y_true = np.asarray(y_true).astype(int)
    clean_pred = np.asarray(clean_pred).astype(int)

    if attack_scope == "all":
        return np.ones(len(y_true), dtype=bool)

    if attack_scope == "dropout_only":
        return y_true == POSITIVE_LABEL

    if attack_scope == "true_positive_only":
        return (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL)

    raise ValueError("attack_scope harus 'all', 'dropout_only', atau 'true_positive_only'.")


def generate_fgsm_for_scope(
    fgsm,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    attack_mode,
    attack_scope,
    perturbable_mask,
):
    """Generate FGSM only on selected rows; untouched rows remain clean."""
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()

    if int(attack_idx.sum()) == 0:
        return X_adv_preprocessed, attack_idx

    attack_y, _ = make_attack_labels(attack_mode, y_true=np.asarray(y_true)[attack_idx])

    X_adv_subset = fgsm.generate(
        x=X_clean_preprocessed[attack_idx],
        y=attack_y,
        mask=perturbable_mask,
    )

    # Hard restore frozen transformed dimensions for attacked rows.
    frozen = perturbable_mask == 0
    X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]

    X_adv_preprocessed[attack_idx] = X_adv_subset

    return X_adv_preprocessed, attack_idx


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_single_fgsm_experiment(
    df,
    csv_path,
    feature_set,
    output_dir,
    rf_results_dir,
    mlp_dir,
    epsilon_list,
    random_state,
    attack_mode,
    attack_scope,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FGSM ATTACK ON RANDOM FOREST VIA SOURCE MLP")
    print("=" * 80)
    print(f"CSV path       : {csv_path}")
    print(f"Feature set    : {feature_set}")
    print(f"Output dir     : {output_dir}")
    print(f"RF results dir : {rf_results_dir}")
    print(f"MLP dir        : {mlp_dir}")
    print(f"Attack mode    : {attack_mode}")
    print(f"Attack scope   : {attack_scope}")
    print(f"Device         : {device}")
    print(f"Dataset shape  : {df.shape}")
    print()

    X, y, student_ids, selected_features = prepare_features_and_label(df, feature_set)

    (
        X_train, X_test, X_val,
        y_train, y_test, y_val,
        id_train, id_test, id_val,
        split_report,
    ) = split_70_20_10_stratified(X, y, student_ids, random_state=random_state)

    print("Split report:")
    print(json.dumps(to_python(split_report), indent=4))
    print()

    rf_model_path = resolve_rf_model_path(rf_results_dir, feature_set)
    rf_summary_path = resolve_rf_summary_path(rf_results_dir, feature_set)
    rf_threshold, rf_threshold_source = load_rf_threshold(rf_summary_path)
    rf_model = joblib.load(rf_model_path)

    mlp_model_path = resolve_mlp_model_path(mlp_dir, feature_set)
    mlp_preprocessor_path = resolve_mlp_preprocessor_path(mlp_dir, feature_set)
    preprocessor = joblib.load(mlp_preprocessor_path)

    X_train_preprocessed = transform_to_float32(preprocessor, X_train)
    X_test_preprocessed = transform_to_float32(preprocessor, X_test)
    X_val_preprocessed = transform_to_float32(preprocessor, X_val)

    input_dim = X_test_preprocessed.shape[1]
    mlp_model, mlp_info = load_mlp_model(mlp_model_path, input_dim=input_dim, device=device)
    art_mlp = wrap_mlp_for_art(mlp_model, input_dim=input_dim)

    if mlp_info["selected_features"] is not None and list(mlp_info["selected_features"]) != list(selected_features):
        raise ValueError(
            "selected_features di checkpoint MLP tidak sama dengan selected_features script FGSM.\n"
            "Pastikan --csv-path dan --feature-set sama dengan training MLP."
        )

    perturbable_mask, raw_mask_df, preprocessed_mask_df = make_preprocessed_attack_mask(
        preprocessor=preprocessor,
        selected_features=selected_features,
        feature_set=feature_set,
        input_dim=input_dim,
        output_dir=output_dir,
    )

    print("Loaded RF model:")
    print(rf_model_path)
    print(f"RF threshold: {rf_threshold} | source: {rf_threshold_source}")
    print()

    print("Loaded source MLP model:")
    print(mlp_model_path)
    print("Loaded MLP preprocessor:")
    print(mlp_preprocessor_path)
    print(f"MLP hidden dims: {mlp_info['hidden_dims']} | dropout: {mlp_info['dropout']}")
    print()

    print("Attackable raw features:")
    print(raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist())
    print()

    print("Preprocessed attackable dimensions:")
    print(int(perturbable_mask.sum()), "of", len(perturbable_mask))
    print()

    # Clean RF baseline.
    clean_test_result, clean_test_cm, clean_test_pred, clean_test_proba = evaluate_rf_with_threshold(
        rf_model=rf_model,
        X_raw=X_test,
        y_true=y_test.values,
        threshold=rf_threshold,
        condition="Clean",
        epsilon=0.0,
        split="test",
    )

    clean_val_result, clean_val_cm, clean_val_pred, clean_val_proba = evaluate_rf_with_threshold(
        rf_model=rf_model,
        X_raw=X_val,
        y_true=y_val.values,
        threshold=rf_threshold,
        condition="Clean",
        epsilon=0.0,
        split="validation",
    )

    save_confusion_matrix(clean_test_cm, os.path.join(output_dir, "rf_clean_test_confusion_matrix.csv"))
    save_confusion_matrix(clean_val_cm, os.path.join(output_dir, "rf_clean_validation_confusion_matrix.csv"))

    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    all_results = [clean_test_result, clean_val_result]
    all_stats = []

    print("Clean test result:")
    print(clean_test_result)
    print("Clean test confusion matrix:")
    print(clean_test_cm)
    print()
    print("Clean validation result:")
    print(clean_val_result)
    print("Clean validation confusion matrix:")
    print(clean_val_cm)
    print()

    for epsilon in epsilon_list:
        print("-" * 80)
        print(f"FGSM epsilon = {epsilon}")
        print("-" * 80)

        _, targeted = make_attack_labels(attack_mode, y_true=y_test.values[:1])

        fgsm = FastGradientMethod(
            estimator=art_mlp,
            eps=float(epsilon),
            targeted=targeted,
            batch_size=128,
        )

        X_test_adv_pre_raw, test_attack_idx = generate_fgsm_for_scope(
            fgsm=fgsm,
            X_clean_preprocessed=X_test_preprocessed,
            y_true=y_test.values,
            clean_pred=clean_test_pred,
            attack_mode=attack_mode,
            attack_scope=attack_scope,
            perturbable_mask=perturbable_mask,
        )

        X_val_adv_pre_raw, val_attack_idx = generate_fgsm_for_scope(
            fgsm=fgsm,
            X_clean_preprocessed=X_val_preprocessed,
            y_true=y_val.values,
            clean_pred=clean_val_pred,
            attack_mode=attack_mode,
            attack_scope=attack_scope,
            perturbable_mask=perturbable_mask,
        )

        X_test_adv_raw, X_test_adv_preprocessed = preprocessed_adv_to_raw_and_scaled(
            X_adv_preprocessed=X_test_adv_pre_raw,
            X_clean_preprocessed=X_test_preprocessed,
            X_clean_raw=X_test,
            selected_features=selected_features,
            preprocessor=preprocessor,
            feature_set=feature_set,
            train_min=train_min,
            train_max=train_max,
            perturbable_mask_preprocessed=perturbable_mask,
        )

        X_val_adv_raw, X_val_adv_preprocessed = preprocessed_adv_to_raw_and_scaled(
            X_adv_preprocessed=X_val_adv_pre_raw,
            X_clean_preprocessed=X_val_preprocessed,
            X_clean_raw=X_val,
            selected_features=selected_features,
            preprocessor=preprocessor,
            feature_set=feature_set,
            train_min=train_min,
            train_max=train_max,
            perturbable_mask_preprocessed=perturbable_mask,
        )

        test_result, test_cm, test_adv_pred, test_adv_proba = evaluate_rf_with_threshold(
            rf_model=rf_model,
            X_raw=X_test_adv_raw,
            y_true=y_test.values,
            threshold=rf_threshold,
            condition=f"FGSM {attack_mode}",
            epsilon=epsilon,
            split="test",
        )

        val_result, val_cm, val_adv_pred, val_adv_proba = evaluate_rf_with_threshold(
            rf_model=rf_model,
            X_raw=X_val_adv_raw,
            y_true=y_val.values,
            threshold=rf_threshold,
            condition=f"FGSM {attack_mode}",
            epsilon=epsilon,
            split="validation",
        )

        hidden_test = int(np.sum((y_test.values == 1) & (clean_test_pred == 1) & (test_adv_pred == 0)))
        false_alarm_test = int(np.sum((y_test.values == 0) & (clean_test_pred == 0) & (test_adv_pred == 1)))
        changed_test = int(np.sum(clean_test_pred != test_adv_pred))

        test_result["attack_scope"] = attack_scope
        test_result["attacked_rows_count"] = int(test_attack_idx.sum())
        test_result["attacked_dropout_rows_count"] = int(np.sum(test_attack_idx & (y_test.values == 1)))
        test_result["prediction_changed_count"] = changed_test
        test_result["dropout_to_nondropout_flips"] = hidden_test
        test_result["nondropout_to_dropout_flips"] = false_alarm_test

        hidden_val = int(np.sum((y_val.values == 1) & (clean_val_pred == 1) & (val_adv_pred == 0)))
        false_alarm_val = int(np.sum((y_val.values == 0) & (clean_val_pred == 0) & (val_adv_pred == 1)))
        changed_val = int(np.sum(clean_val_pred != val_adv_pred))

        val_result["attack_scope"] = attack_scope
        val_result["attacked_rows_count"] = int(val_attack_idx.sum())
        val_result["attacked_dropout_rows_count"] = int(np.sum(val_attack_idx & (y_val.values == 1)))
        val_result["prediction_changed_count"] = changed_val
        val_result["dropout_to_nondropout_flips"] = hidden_val
        val_result["nondropout_to_dropout_flips"] = false_alarm_val

        all_results.extend([test_result, val_result])

        test_stats = perturbation_stats(X_test_preprocessed, X_test_adv_preprocessed, perturbable_mask)
        test_stats.update({
            "epsilon": float(epsilon),
            "split": "test",
            "attack_scope": attack_scope,
            "attacked_rows_count": int(test_attack_idx.sum()),
        })
        val_stats = perturbation_stats(X_val_preprocessed, X_val_adv_preprocessed, perturbable_mask)
        val_stats.update({
            "epsilon": float(epsilon),
            "split": "validation",
            "attack_scope": attack_scope,
            "attacked_rows_count": int(val_attack_idx.sum()),
        })
        all_stats.extend([test_stats, val_stats])

        if test_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen transformed features changed in test set.")
        if val_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen transformed features changed in validation set.")

        print("Test result:")
        print(test_result)
        print("Test confusion matrix:")
        print(test_cm)
        print()
        print("Validation result:")
        print(val_result)
        print("Validation confusion matrix:")
        print(val_cm)
        print()

        # Save arrays and raw CSVs.
        eps_tag = str(epsilon).replace(".", "p")

        np.save(os.path.join(output_dir, f"X_test_fgsm_preprocessed_eps_{eps_tag}.npy"), X_test_adv_preprocessed)
        np.save(os.path.join(output_dir, f"X_validation_fgsm_preprocessed_eps_{eps_tag}.npy"), X_val_adv_preprocessed)

        X_test_adv_raw.to_csv(os.path.join(output_dir, f"X_test_fgsm_raw_eps_{eps_tag}.csv"), index=False)
        X_val_adv_raw.to_csv(os.path.join(output_dir, f"X_validation_fgsm_raw_eps_{eps_tag}.csv"), index=False)

        save_confusion_matrix(test_cm, os.path.join(output_dir, f"rf_fgsm_test_confusion_matrix_eps_{eps_tag}.csv"))
        save_confusion_matrix(val_cm, os.path.join(output_dir, f"rf_fgsm_validation_confusion_matrix_eps_{eps_tag}.csv"))

        save_attack_predictions(
            student_ids=id_test,
            y_true=y_test.values,
            clean_pred=clean_test_pred,
            adv_pred=test_adv_pred,
            clean_proba=clean_test_proba,
            adv_proba=test_adv_proba,
            epsilon=epsilon,
            split="test",
            path=os.path.join(output_dir, f"rf_fgsm_test_predictions_eps_{eps_tag}.csv"),
        )

        save_attack_predictions(
            student_ids=id_val,
            y_true=y_val.values,
            clean_pred=clean_val_pred,
            adv_pred=val_adv_pred,
            clean_proba=clean_val_proba,
            adv_proba=val_adv_proba,
            epsilon=epsilon,
            split="validation",
            path=os.path.join(output_dir, f"rf_fgsm_validation_predictions_eps_{eps_tag}.csv"),
        )

        save_clean_vs_adv_sample(
            X_clean_raw=X_test,
            X_adv_raw=X_test_adv_raw,
            y_true=y_test.values,
            clean_pred=clean_test_pred,
            adv_pred=test_adv_pred,
            clean_proba=clean_test_proba,
            adv_proba=test_adv_proba,
            selected_features=selected_features,
            epsilon=epsilon,
            split="test",
            path=os.path.join(output_dir, f"clean_vs_fgsm_test_sample_eps_{eps_tag}.csv"),
            max_samples=50,
        )

        save_clean_vs_adv_sample(
            X_clean_raw=X_val,
            X_adv_raw=X_val_adv_raw,
            y_true=y_val.values,
            clean_pred=clean_val_pred,
            adv_pred=val_adv_pred,
            clean_proba=clean_val_proba,
            adv_proba=val_adv_proba,
            selected_features=selected_features,
            epsilon=epsilon,
            split="validation",
            path=os.path.join(output_dir, f"clean_vs_fgsm_validation_sample_eps_{eps_tag}.csv"),
            max_samples=50,
        )

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)

    results_path = os.path.join(output_dir, "rf_fgsm_attack_results.csv")
    stats_path = os.path.join(output_dir, "rf_fgsm_perturbation_stats.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "dataset": csv_path,
        "feature_set": feature_set,
        "output_dir": output_dir,
        "attack": "FGSM generated using source MLP gradients, evaluated on Random Forest",
        "attack_mode": attack_mode,
        "attack_scope": attack_scope,
        "epsilons": epsilon_list,
        "target_model": "Random Forest",
        "rf_model_path": rf_model_path,
        "rf_threshold": rf_threshold,
        "rf_threshold_source": rf_threshold_source,
        "gradient_model": "Source MLP trained on true labels",
        "mlp_model_path": mlp_model_path,
        "mlp_preprocessor_path": mlp_preprocessor_path,
        "mlp_hidden_dims": list(mlp_info["hidden_dims"]),
        "mlp_dropout": mlp_info["dropout"],
        "n_raw_features": len(selected_features),
        "n_preprocessed_features": int(input_dim),
        "n_attackable_preprocessed_features": int(perturbable_mask.sum()),
        "n_frozen_preprocessed_features": int(len(perturbable_mask) - perturbable_mask.sum()),
        "attackable_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist(),
        "frozen_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 0, "feature"].tolist(),
        "constraints": {
            "frozen_features": "Restored to clean values after FGSM.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
            "percentage_features": "Clipped to [0, 100].",
            "integer_count_features": "Rounded after inverse transform.",
            "nonnegative_features": "Clipped to lower bound 0.",
        },
        "split": split_report,
        "results_file": results_path,
        "perturbation_stats_file": stats_path,
    }

    summary_path = os.path.join(output_dir, "rf_fgsm_attack_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Stats saved to  : {stats_path}")
    print(f"Summary saved to: {summary_path}")

    return summary


# ============================================================
# ARGUMENTS
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="FGSM attack on Random Forest using the source MLP model."
    )

    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}",
    )

    parser.add_argument(
        "--feature-set",
        default="full",
        choices=["full"],
        help="Feature set. This script only supports 'full'.",
    )

    parser.add_argument(
        "--rf-results-dir",
        default=DEFAULT_RF_RESULTS_DIR,
        help=f"Folder hasil RF. Default: {DEFAULT_RF_RESULTS_DIR}",
    )

    parser.add_argument(
        "--mlp-dir",
        default=DEFAULT_MLP_DIR,
        help=f"Folder hasil MLP source model. Default: {DEFAULT_MLP_DIR}",
    )

    # Backward-compatible alias, in case old command uses --surrogate-dir.
    parser.add_argument(
        "--surrogate-dir",
        dest="mlp_dir",
        help="Alias lama untuk --mlp-dir.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder output FGSM. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random state. Default: {DEFAULT_RANDOM_STATE}",
    )

    parser.add_argument(
        "--attack-mode",
        default="target_non_dropout",
        choices=["untargeted", "target_non_dropout"],
        help=(
            "untargeted = maximize loss against true labels. "
            "target_non_dropout = targeted attack untuk mendorong prediksi ke non-dropout."
        ),
    )

    parser.add_argument(
        "--attack-scope",
        default="true_positive_only",
        choices=["all", "dropout_only", "true_positive_only"],
        help=(
            "all = attack semua row. "
            "dropout_only = attack hanya true dropout rows. "
            "true_positive_only = attack hanya dropout yang clean RF-nya benar terdeteksi dropout."
        ),
    )

    parser.add_argument(
        "--epsilons",
        default="1.5",
        help="Comma-separated epsilon list. Default: 1.5",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.csv_path}")

    epsilon_list = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    df = pd.read_csv(args.csv_path)

    run_single_fgsm_experiment(
        df=df,
        csv_path=args.csv_path,
        feature_set="full",
        output_dir=args.output_dir,
        rf_results_dir=args.rf_results_dir,
        mlp_dir=args.mlp_dir,
        epsilon_list=epsilon_list,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        attack_scope=args.attack_scope,
    )


if __name__ == "__main__":
    main()
