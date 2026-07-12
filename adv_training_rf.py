#!/usr/bin/env python3
"""
Adversarial training (defence) for Random Forest using FGSM samples generated
from the trained/source MLP model. FULL feature set only.

Config is aligned with the FGSM RF attack we actually ran:
- Dataset default: zenodo.csv
- RF baseline folder default: rf_results
- Source MLP folder default: mlp_results
- FGSM uses the source MLP gradients (mlp_model.pt + mlp_preprocessor.pkl).
- Attack mode: target_non_dropout.
- Training attack scope: dropout_only (augment adversarial dropout rows).
- Evaluation attack scope: true_positive_only (same as the attack flow).
- PGD/FGSM epsilon (train and eval): 1.0, matching the final FGSM RF attack.
- Split: stratified 70/20/10, same as the baseline RF.
- The defended RF is a clone of the baseline RF pipeline refit on clean + adversarial data.
- Threshold for the defended model is selected on the clean TEST set.

Steps:
1. Generate adversarial dropout samples with source MLP gradients.
2. Add adversarial training samples to the clean RF training set.
3. Train a defended Random Forest.
4. Evaluate defended RF on clean and FGSM adversarial data.
"""

import os
import json
import ast
import copy
import random
import argparse
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
# CONFIG
# ============================================================

DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_RF_RESULTS_DIR = "rf_results"
DEFAULT_MLP_DIR = "mlp_results"
DEFAULT_OUTPUT_DIR = "rf_adversarial_training_full_results"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# Match your final FGSM RF attack setting.
TRAIN_EPSILON_LIST = [1.0]
EVAL_EPSILON_LIST = [1.0]

THRESHOLDS = np.arange(0.05, 0.96, 0.01)
DEFAULT_HIDDEN_DIMS = (128, 64, 32)


# ============================================================
# FEATURE SETS - MUST MATCH MLP/RF TRAINING
# ============================================================

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
# SOURCE MLP MODEL - MUST MATCH TRAINING
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
        raise ValueError(f"Feature set {feature_set} butuh kolom yang tidak ada:\n{missing_features}")

    X = df[selected_features].copy()
    y = normalize_target(df[TARGET_COL])

    if "student_id" in df.columns:
        student_ids = df["student_id"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")

    return X, y, student_ids, selected_features


def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    """
    Stratified random split with no grouping. Identical to the split used by the
    Random Forest and MLP training scripts: ~70% train, ~20% test, ~10% validation,
    stratified by label, same random_state. This guarantees the test/validation rows
    here are exactly the rows the baseline models were evaluated on.
    """
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")

    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids, test_size=0.30, stratify=y, random_state=random_state,
    )
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=1.0 / 3.0, stratify=y_temp, random_state=random_state,
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
# PATHS / LOADING
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


def parse_class_weight(value):
    if value is None:
        return None
    if value in ["balanced", "balanced_subsample"]:
        return value
    if isinstance(value, dict):
        return {int(k): v for k, v in value.items()}
    if isinstance(value, str):
        value_strip = value.strip()
        if value_strip in ["None", "none", "null"]:
            return None
        if value_strip in ["balanced", "balanced_subsample"]:
            return value_strip
        try:
            parsed = ast.literal_eval(value_strip)
            if isinstance(parsed, dict):
                return {int(k): v for k, v in parsed.items()}
        except Exception:
            pass
    return value


def load_rf_best_params(rf_summary_path):
    default_params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 50,
        "min_samples_leaf": 20,
        "max_features": 0.5,
        "class_weight": {0: 1, 1: 4},
        "bootstrap": True,
        "max_samples": 0.7,
    }

    if rf_summary_path is None or not os.path.exists(rf_summary_path):
        return default_params, "default_params"

    with open(rf_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    candidates = [
        summary.get("selection", {}).get("best_params"),
        summary.get("best_params"),
        summary.get("tuning", {}).get("best_params"),
    ]

    best_params = None
    for candidate in candidates:
        if isinstance(candidate, dict):
            best_params = candidate
            break

    if best_params is None:
        return default_params, "default_params"

    allowed_keys = set(default_params.keys())
    cleaned = {k: v for k, v in best_params.items() if k in allowed_keys}

    params = copy.deepcopy(default_params)
    params.update(cleaned)
    params["class_weight"] = parse_class_weight(params.get("class_weight"))
    if isinstance(params.get("bootstrap"), str):
        params["bootstrap"] = params["bootstrap"].lower() == "true"

    return params, rf_summary_path


def load_rf_threshold(rf_summary_path, fallback=0.5):
    if rf_summary_path and os.path.exists(rf_summary_path):
        with open(rf_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        candidates = [
            summary.get("selection", {}).get("best_threshold"),
            summary.get("best_threshold"),
            summary.get("tuning", {}).get("best_threshold"),
        ]
        for threshold in candidates:
            if threshold is not None:
                return float(threshold), rf_summary_path
    return float(fallback), "fallback_0.5"


def load_torch_checkpoint(path, device):
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
    else:
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        state_dict = obj
        checkpoint_features = None
        checkpoint_feature_set = None

    model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, {
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "selected_features": checkpoint_features,
        "feature_set": checkpoint_feature_set,
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
# PREPROCESSOR / MASKS / PROJECTION
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
    os.makedirs(output_dir, exist_ok=True)

    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)

    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(
            f"Attackable features ini tidak ada di selected_features untuk {feature_set}:\n{missing_attackable}"
        )

    mask = np.zeros(input_dim, dtype=np.float32)
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
    attacked_row_mask=None,
):
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    scaler = get_scaler_from_preprocessor(preprocessor)

    n_num = len(numeric_cols)
    adv_numeric_scaled = X_adv_preprocessed[:, :n_num]
    adv_numeric_raw = scaler.inverse_transform(adv_numeric_scaled)

    X_adv_raw = X_clean_raw[selected_features].reset_index(drop=True).copy()
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)

    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]

    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])

    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)

    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)

    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)

    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        not_attacked = ~attacked_row_mask
        if not_attacked.any():
            # Cast target dataframe to object temporarily to avoid pandas dtype warning.
            X_adv_raw = X_adv_raw.astype(object)
            X_adv_raw.loc[not_attacked, selected_features] = clean_reset.loc[not_attacked, selected_features].values

    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[selected_features])

    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]

    if attacked_row_mask is not None:
        not_attacked = ~np.asarray(attacked_row_mask).astype(bool)
        if not_attacked.any():
            X_projected_preprocessed[not_attacked, :] = X_clean_preprocessed[not_attacked, :]

    return X_adv_raw[selected_features], X_projected_preprocessed


# ============================================================
# ATTACK HELPERS
# ============================================================

def make_attack_labels(attack_mode, y_true):
    if attack_mode == "untargeted":
        return one_hot(y_true), False
    if attack_mode == "target_non_dropout":
        target_labels = np.zeros(len(y_true), dtype=int)
        return one_hot(target_labels), True
    raise ValueError("attack_mode harus 'untargeted' atau 'target_non_dropout'.")


def make_attack_scope_mask(attack_scope, y_true, clean_pred):
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
    art_mlp,
    X_clean_raw,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    epsilon,
    attack_mode,
    attack_scope,
    perturbable_mask,
    preprocessor,
    selected_features,
    feature_set,
    train_min,
    train_max,
    batch_size=128,
):
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()

    if int(attack_idx.sum()) > 0:
        attack_y, targeted = make_attack_labels(attack_mode, np.asarray(y_true)[attack_idx])
        fgsm = FastGradientMethod(
            estimator=art_mlp,
            eps=float(epsilon),
            targeted=targeted,
            batch_size=batch_size,
        )
        X_adv_subset = fgsm.generate(
            x=X_clean_preprocessed[attack_idx],
            y=attack_y,
            mask=perturbable_mask,
        )
        frozen = perturbable_mask == 0
        X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
        X_adv_preprocessed[attack_idx] = X_adv_subset

    X_adv_raw, X_adv_projected = preprocessed_adv_to_raw_and_scaled(
        X_adv_preprocessed=X_adv_preprocessed,
        X_clean_preprocessed=X_clean_preprocessed,
        X_clean_raw=X_clean_raw,
        selected_features=selected_features,
        preprocessor=preprocessor,
        feature_set=feature_set,
        train_min=train_min,
        train_max=train_max,
        perturbable_mask_preprocessed=perturbable_mask,
        attacked_row_mask=attack_idx,
    )

    return X_adv_raw, X_adv_projected, attack_idx


# ============================================================
# EVALUATION / SAVING
# ============================================================

def tune_threshold_from_proba(y_true, y_proba):
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_dropout": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "recall_dropout": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "f1_dropout": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        })
    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        ["f1_dropout", "recall_dropout", "precision_dropout"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_df, best_row.to_dict()


def evaluate_rf_with_threshold(model, X, y, condition, train_epsilon, eval_epsilon, split, threshold):
    y_true = np.asarray(y).astype(int)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    result = {
        "model": "Random Forest",
        "condition": condition,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "epsilon": eval_epsilon,
        "split": split,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dropout": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    return result, cm, y_pred, y_proba


def perturbation_stats(X_clean_preprocessed, X_adv_preprocessed, perturbable_mask, attacked_row_mask=None):
    diff = X_adv_preprocessed - X_clean_preprocessed
    pm = perturbable_mask.astype(bool)
    frozen = ~pm

    stats = {
        "mean_abs_perturbation_all_preprocessed": float(np.abs(diff).mean()),
        "mean_abs_perturbation_attackable_preprocessed": float(np.abs(diff[:, pm]).mean()),
        "max_abs_perturbation_attackable_preprocessed": float(np.abs(diff[:, pm]).max()),
        "max_abs_perturbation_frozen_preprocessed": float(np.abs(diff[:, frozen]).max()) if frozen.any() else 0.0,
    }

    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        stats["attacked_rows_count"] = int(attacked_row_mask.sum())
        if attacked_row_mask.any():
            attacked_diff = diff[attacked_row_mask]
            stats["mean_abs_perturbation_attacked_rows_attackable_preprocessed"] = float(np.abs(attacked_diff[:, pm]).mean())
            stats["max_abs_perturbation_attacked_rows_attackable_preprocessed"] = float(np.abs(attacked_diff[:, pm]).max())
        else:
            stats["mean_abs_perturbation_attacked_rows_attackable_preprocessed"] = 0.0
            stats["max_abs_perturbation_attacked_rows_attackable_preprocessed"] = 0.0

    return stats


def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)


def save_predictions(student_ids, y_true, y_pred, y_proba, threshold, condition, train_epsilon, eval_epsilon, path):
    y_true = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        "student_id": student_ids.values,
        "condition": condition,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "predicted_label": y_pred,
        "predicted_target": np.where(y_pred == 1, "dropout", "non-dropout"),
        "prob_dropout": y_proba,
        "threshold": threshold,
    })
    pred_df.to_csv(path, index=False)


def save_training_sample_csv(X_clean_raw, X_adv_raw, y, selected_features, attack_idx, train_epsilon, path, max_samples=50):
    attack_idx = np.asarray(attack_idx).astype(bool)
    chosen = np.where(attack_idx)[0][:max_samples]
    rows = []

    X_clean_raw = X_clean_raw.reset_index(drop=True)
    X_adv_raw = X_adv_raw.reset_index(drop=True)
    y = np.asarray(y).astype(int)

    for i in chosen:
        row = {
            "sample_id": int(i),
            "label": int(y[i]),
            "label_text": "dropout" if y[i] == 1 else "non-dropout",
            "train_epsilon": float(train_epsilon),
        }
        for feature in selected_features:
            row[f"clean_{feature}"] = X_clean_raw.loc[i, feature]
            row[f"adv_{feature}"] = X_adv_raw.loc[i, feature]
            row[f"diff_{feature}"] = X_adv_raw.loc[i, feature] - X_clean_raw.loc[i, feature]
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# RF TRAINING
# ============================================================

def train_defended_rf(rf_baseline, X_train_mixed, y_train_mixed, random_state):
    # Clone the baseline RF pipeline so the defended model uses the exact same
    # preprocessing (median imputation, flag handling, one-hot of seniority) and the
    # same tuned hyperparameters as the baseline, then refit it on the clean +
    # adversarial training data. rf_params is kept only for logging in the summary.
    rf_defended = clone(rf_baseline)
    rf_defended.fit(X_train_mixed, y_train_mixed)
    return rf_defended


# ============================================================
# SINGLE FEATURE SET EXPERIMENT
# ============================================================

def run_single_adversarial_training_experiment(
    df,
    csv_path,
    feature_set,
    output_dir,
    rf_results_dir,
    mlp_dir,
    train_epsilon_list,
    eval_epsilon_list,
    random_state,
    attack_mode,
    train_attack_scope,
    eval_attack_scope,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("RF ADVERSARIAL TRAINING WITH FGSM VIA SOURCE MLP")
    print("=" * 80)
    print(f"CSV path       : {csv_path}")
    print(f"Feature set    : {feature_set}")
    print(f"Output dir     : {output_dir}")
    print(f"RF results dir : {rf_results_dir}")
    print(f"MLP dir        : {mlp_dir}")
    print(f"Attack mode    : {attack_mode}")
    print(f"Train attack scope : {train_attack_scope}")
    print(f"Eval attack scope  : {eval_attack_scope}")
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

    rf_baseline_model_path = resolve_rf_model_path(rf_results_dir, feature_set)
    rf_summary_path = resolve_rf_summary_path(rf_results_dir, feature_set)
    rf_baseline = joblib.load(rf_baseline_model_path)
    rf_baseline_threshold, rf_threshold_source = load_rf_threshold(rf_summary_path)
    rf_params, rf_params_source = load_rf_best_params(rf_summary_path)

    mlp_model_path = resolve_mlp_model_path(mlp_dir, feature_set)
    mlp_preprocessor_path = resolve_mlp_preprocessor_path(mlp_dir, feature_set)
    preprocessor = joblib.load(mlp_preprocessor_path)

    X_train_preprocessed = transform_to_float32(preprocessor, X_train)
    X_test_preprocessed = transform_to_float32(preprocessor, X_test)
    X_val_preprocessed = transform_to_float32(preprocessor, X_val)
    input_dim = X_train_preprocessed.shape[1]

    mlp_model, mlp_info = load_mlp_model(mlp_model_path, input_dim=input_dim, device=device)

    if mlp_info["selected_features"] is not None and list(mlp_info["selected_features"]) != list(selected_features):
        raise ValueError(
            "selected_features di checkpoint MLP tidak sama dengan selected_features script adversarial training.\n"
            "Pastikan --csv-path dan --feature-set sama dengan training MLP."
        )

    art_mlp = wrap_mlp_for_art(mlp_model, input_dim=input_dim)

    perturbable_mask, raw_mask_df, preprocessed_mask_df = make_preprocessed_attack_mask(
        preprocessor=preprocessor,
        selected_features=selected_features,
        feature_set=feature_set,
        input_dim=input_dim,
        output_dir=output_dir,
    )

    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    print("Loaded baseline RF:")
    print(rf_baseline_model_path)
    print(f"Baseline RF threshold: {rf_baseline_threshold} | source: {rf_threshold_source}")
    print(f"RF params source     : {rf_params_source}")
    print(f"RF params            : {rf_params}")
    print()

    print("Loaded source MLP:")
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

    # Baseline clean predictions used to generate training attack samples.
    baseline_train_clean, baseline_train_cm, baseline_train_pred, baseline_train_proba = evaluate_rf_with_threshold(
        rf_baseline, X_train, y_train.values, "Baseline Clean", np.nan, 0.0, "train", rf_baseline_threshold
    )
    baseline_test_clean, baseline_test_cm, baseline_test_pred, baseline_test_proba = evaluate_rf_with_threshold(
        rf_baseline, X_test, y_test.values, "Baseline Clean", np.nan, 0.0, "test", rf_baseline_threshold
    )
    baseline_val_clean, baseline_val_cm, baseline_val_pred, baseline_val_proba = evaluate_rf_with_threshold(
        rf_baseline, X_val, y_val.values, "Baseline Clean", np.nan, 0.0, "validation", rf_baseline_threshold
    )

    all_results = [baseline_train_clean, baseline_test_clean, baseline_val_clean]
    all_stats = []
    all_summary = []

    print("Baseline clean test result:")
    print(baseline_test_clean)
    print(baseline_test_cm)
    print()

    # Save baseline (before-defence) confusion matrices for test and validation,
    # so this stage is consistent with the defended stages and with the MLP script.
    save_confusion_matrix(baseline_test_cm, os.path.join(output_dir, "rf_baseline_clean_test_cm.csv"))
    save_confusion_matrix(baseline_val_cm, os.path.join(output_dir, "rf_baseline_clean_validation_cm.csv"))

    for train_epsilon in train_epsilon_list:
        print("=" * 80)
        print(f"TRAIN DEFENDED RF | train epsilon = {train_epsilon}")
        print("=" * 80)

        X_train_adv_raw, X_train_adv_preprocessed, train_attack_idx = generate_fgsm_for_scope(
            art_mlp=art_mlp,
            X_clean_raw=X_train,
            X_clean_preprocessed=X_train_preprocessed,
            y_true=y_train.values,
            clean_pred=baseline_train_pred,
            epsilon=train_epsilon,
            attack_mode=attack_mode,
            attack_scope=train_attack_scope,
            perturbable_mask=perturbable_mask,
            preprocessor=preprocessor,
            selected_features=selected_features,
            feature_set=feature_set,
            train_min=train_min,
            train_max=train_max,
        )

        train_stats = perturbation_stats(
            X_clean_preprocessed=X_train_preprocessed,
            X_adv_preprocessed=X_train_adv_preprocessed,
            perturbable_mask=perturbable_mask,
            attacked_row_mask=train_attack_idx,
        )
        train_stats.update({
            "feature_set": feature_set,
            "train_epsilon": float(train_epsilon),
            "eval_epsilon": float(train_epsilon),
            "split": "train_adversarial_generation",
            "attack_scope": train_attack_scope,
            "train_attack_scope": train_attack_scope,
        })
        all_stats.append(train_stats)

        if train_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen features changed in adversarial training data.")

        eps_tag = str(train_epsilon).replace(".", "p")
        X_train_adv_raw.to_csv(os.path.join(output_dir, f"X_train_fgsm_raw_train_eps_{eps_tag}.csv"), index=False)
        np.save(os.path.join(output_dir, f"X_train_fgsm_preprocessed_train_eps_{eps_tag}.npy"), X_train_adv_preprocessed)

        save_training_sample_csv(
            X_clean_raw=X_train,
            X_adv_raw=X_train_adv_raw,
            y=y_train.values,
            selected_features=selected_features,
            attack_idx=train_attack_idx,
            train_epsilon=train_epsilon,
            path=os.path.join(output_dir, f"clean_vs_fgsm_train_sample_train_eps_{eps_tag}.csv"),
            max_samples=50,
        )

        # Use clean training data + adversarial samples only for attacked rows.
        X_adv_train_only = X_train_adv_raw.loc[train_attack_idx, selected_features].reset_index(drop=True)
        y_adv_train_only = y_train.values.astype(int)[train_attack_idx]

        X_train_mixed = pd.concat([
            X_train[selected_features].reset_index(drop=True),
            X_adv_train_only,
        ], axis=0, ignore_index=True)
        y_train_mixed = np.concatenate([y_train.values.astype(int), y_adv_train_only])

        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X_train_mixed))
        X_train_mixed = X_train_mixed.iloc[idx].reset_index(drop=True)
        y_train_mixed = y_train_mixed[idx]

        print(f"Clean train rows              : {len(X_train)}")
        print(f"Adversarial train rows added  : {len(X_adv_train_only)}")
        print(f"Mixed train rows              : {len(X_train_mixed)}")
        print()

        rf_defended = train_defended_rf(
            rf_baseline=rf_baseline,
            X_train_mixed=X_train_mixed,
            y_train_mixed=y_train_mixed,
            random_state=random_state,
        )

        test_clean_proba_for_threshold = rf_defended.predict_proba(X_test)[:, 1]
        defended_threshold, threshold_df, best_threshold_row = tune_threshold_from_proba(
            y_true=y_test.values,
            y_proba=test_clean_proba_for_threshold,
        )
        threshold_df.to_csv(os.path.join(output_dir, f"rf_defended_threshold_tuning_train_eps_{eps_tag}.csv"), index=False)

        model_path = os.path.join(output_dir, f"rf_defended_model_train_eps_{eps_tag}.pkl")
        joblib.dump(rf_defended, model_path)

        print(f"Saved defended RF model: {model_path}")
        print(f"Defended RF threshold : {defended_threshold}")
        print()

        defended_clean_preds = {}
        defended_clean_probas = {}
        for split_name, X_split, y_split, id_split in [
            ("train", X_train, y_train, id_train),
            ("test", X_test, y_test, id_test),
            ("validation", X_val, y_val, id_val),
        ]:
            result, cm, pred, proba = evaluate_rf_with_threshold(
                model=rf_defended,
                X=X_split,
                y=y_split.values,
                condition="Defended Clean",
                train_epsilon=float(train_epsilon),
                eval_epsilon=0.0,
                split=split_name,
                threshold=defended_threshold,
            )
            all_results.append(result)
            defended_clean_preds[split_name] = pred
            defended_clean_probas[split_name] = proba

            save_confusion_matrix(cm, os.path.join(output_dir, f"rf_defended_clean_{split_name}_cm_train_eps_{eps_tag}.csv"))
            save_predictions(
                student_ids=id_split,
                y_true=y_split.values,
                y_pred=pred,
                y_proba=proba,
                threshold=defended_threshold,
                condition="Defended Clean",
                train_epsilon=float(train_epsilon),
                eval_epsilon=0.0,
                path=os.path.join(output_dir, f"rf_defended_clean_{split_name}_predictions_train_eps_{eps_tag}.csv"),
            )

            print(f"Defended clean {split_name}:")
            print(result)
            print(cm)
            print()

        for eval_epsilon in eval_epsilon_list:
            eval_tag = str(eval_epsilon).replace(".", "p")
            print("-" * 80)
            print(f"EVALUATE DEFENDED RF | train eps = {train_epsilon} | eval eps = {eval_epsilon}")
            print("-" * 80)

            for split_name, X_split, X_split_pre, y_split, id_split in [
                ("test", X_test, X_test_preprocessed, y_test, id_test),
                ("validation", X_val, X_val_preprocessed, y_val, id_val),
            ]:
                clean_pred_for_scope = defended_clean_preds[split_name]

                X_adv_raw, X_adv_preprocessed, eval_attack_idx = generate_fgsm_for_scope(
                    art_mlp=art_mlp,
                    X_clean_raw=X_split,
                    X_clean_preprocessed=X_split_pre,
                    y_true=y_split.values,
                    clean_pred=clean_pred_for_scope,
                    epsilon=eval_epsilon,
                    attack_mode=attack_mode,
                    attack_scope=eval_attack_scope,
                    perturbable_mask=perturbable_mask,
                    preprocessor=preprocessor,
                    selected_features=selected_features,
                    feature_set=feature_set,
                    train_min=train_min,
                    train_max=train_max,
                )

                stats = perturbation_stats(
                    X_clean_preprocessed=X_split_pre,
                    X_adv_preprocessed=X_adv_preprocessed,
                    perturbable_mask=perturbable_mask,
                    attacked_row_mask=eval_attack_idx,
                )
                stats.update({
                    "feature_set": feature_set,
                    "train_epsilon": float(train_epsilon),
                    "eval_epsilon": float(eval_epsilon),
                    "split": split_name,
                    "attack_scope": eval_attack_scope,
                    "train_attack_scope": train_attack_scope,
                    "eval_attack_scope": eval_attack_scope,
                })
                all_stats.append(stats)

                if stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
                    raise AssertionError(f"Frozen features changed in {split_name} eval data.")

                X_adv_raw.to_csv(
                    os.path.join(output_dir, f"X_{split_name}_fgsm_raw_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                    index=False,
                )

                result, cm, pred, proba = evaluate_rf_with_threshold(
                    model=rf_defended,
                    X=X_adv_raw,
                    y=y_split.values,
                    condition=f"Defended FGSM {attack_mode}",
                    train_epsilon=float(train_epsilon),
                    eval_epsilon=float(eval_epsilon),
                    split=split_name,
                    threshold=defended_threshold,
                )
                result["attack_scope"] = eval_attack_scope
                result["train_attack_scope"] = train_attack_scope
                result["eval_attack_scope"] = eval_attack_scope
                result["attacked_rows_count"] = int(eval_attack_idx.sum())
                result["attacked_dropout_rows_count"] = int(np.sum(eval_attack_idx & (y_split.values == POSITIVE_LABEL)))
                result["dropout_to_nondropout_flips"] = int(np.sum((y_split.values == 1) & (clean_pred_for_scope == 1) & (pred == 0)))
                result["nondropout_to_dropout_flips"] = int(np.sum((y_split.values == 0) & (clean_pred_for_scope == 0) & (pred == 1)))
                result["prediction_changed_count"] = int(np.sum(clean_pred_for_scope != pred))

                all_results.append(result)

                save_confusion_matrix(
                    cm,
                    os.path.join(output_dir, f"rf_defended_fgsm_{split_name}_cm_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                )
                save_predictions(
                    student_ids=id_split,
                    y_true=y_split.values,
                    y_pred=pred,
                    y_proba=proba,
                    threshold=defended_threshold,
                    condition=f"Defended FGSM {attack_mode}",
                    train_epsilon=float(train_epsilon),
                    eval_epsilon=float(eval_epsilon),
                    path=os.path.join(output_dir, f"rf_defended_fgsm_{split_name}_predictions_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                )

                print(f"Defended FGSM {split_name}:")
                print(result)
                print(cm)
                print()

        all_summary.append({
            "feature_set": feature_set,
            "model": "Random Forest Defended",
            "defence": "Adversarial training with FGSM samples generated via source MLP",
            "attack_mode": attack_mode,
            "train_attack_scope": train_attack_scope,
            "eval_attack_scope": eval_attack_scope,
            "train_epsilon": float(train_epsilon),
            "defended_threshold": float(defended_threshold),
            "best_threshold_row": best_threshold_row,
            "clean_train_rows": int(len(X_train)),
            "fgsm_train_rows_added": int(len(X_adv_train_only)),
            "mixed_train_rows": int(len(X_train_mixed)),
            "model_path": model_path,
        })

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)
    summary_df = pd.DataFrame(all_summary)

    results_path = os.path.join(output_dir, "rf_adversarial_training_results.csv")
    stats_path = os.path.join(output_dir, "rf_adversarial_training_perturbation_stats.csv")
    summary_csv_path = os.path.join(output_dir, "rf_adversarial_training_summary_by_train_epsilon.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    summary = {
        "dataset": csv_path,
        "feature_set": feature_set,
        "output_dir": output_dir,
        "defence": "Random Forest adversarial training",
        "training_method": f"Clean training data plus FGSM adversarial samples generated from train_attack_scope={train_attack_scope} (attacked rows only).",
        "target_model": "Random Forest Defended",
        "gradient_model": "Source MLP trained on true labels",
        "attack_used_for_training": "FGSM",
        "attack_mode": attack_mode,
        "train_attack_scope": train_attack_scope,
        "eval_attack_scope": eval_attack_scope,
        "train_epsilon_list": train_epsilon_list,
        "eval_epsilon_list": eval_epsilon_list,
        "rf_baseline_model_path": rf_baseline_model_path,
        "rf_baseline_threshold": rf_baseline_threshold,
        "rf_params": rf_params,
        "rf_params_source": rf_params_source,
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
            "non_attacked_rows": "Restored completely to clean values.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
            "percentage_features": "Clipped to [0, 100].",
            "integer_count_features": "Rounded after inverse transform.",
            "nonnegative_features": "Clipped to lower bound 0.",
        },
        "split": split_report,
        "results_file": results_path,
        "stats_file": stats_path,
        "summary_by_train_epsilon_file": summary_csv_path,
    }

    summary_path = os.path.join(output_dir, "rf_adversarial_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("RF ADVERSARIAL TRAINING DONE")
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
        description="Adversarial training for Random Forest using FGSM via source MLP."
    )

    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=["full"],
        help="Feature set. This script only supports 'full'.",
    )
    parser.add_argument("--rf-results-dir", default=DEFAULT_RF_RESULTS_DIR, help=f"Folder hasil RF baseline. Default: {DEFAULT_RF_RESULTS_DIR}")
    parser.add_argument("--mlp-dir", default=DEFAULT_MLP_DIR, help=f"Folder hasil source MLP. Default: {DEFAULT_MLP_DIR}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Folder output defended RF. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random state. Default: {DEFAULT_RANDOM_STATE}")
    parser.add_argument(
        "--attack-mode",
        default="target_non_dropout",
        choices=["untargeted", "target_non_dropout"],
        help="untargeted = general degradation. target_non_dropout = targeted attack ke non-dropout.",
    )
    parser.add_argument(
        "--train-attack-scope",
        default="dropout_only",
        choices=["all", "dropout_only", "true_positive_only"],
        help="Scope used to generate FGSM samples for adversarial training. Default: dropout_only.",
    )
    parser.add_argument(
        "--eval-attack-scope",
        default="true_positive_only",
        choices=["all", "dropout_only", "true_positive_only"],
        help="Scope used for final FGSM evaluation. Default: true_positive_only.",
    )
    parser.add_argument(
        "--attack-scope",
        default=None,
        choices=["all", "dropout_only", "true_positive_only"],
        help="Backward-compatible alias. If provided, it sets both train and eval attack scope.",
    )
    parser.add_argument("--train-epsilons", default="1.0", help="Comma-separated train epsilon list. Default: 1.0")
    parser.add_argument("--eval-epsilons", default="1.0", help="Comma-separated eval epsilon list. Default: 1.0")

    return parser


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    train_epsilon_list = parse_float_list(args.train_epsilons)
    eval_epsilon_list = parse_float_list(args.eval_epsilons)

    if args.attack_scope is not None:
        args.train_attack_scope = args.attack_scope
        args.eval_attack_scope = args.attack_scope

    run_single_adversarial_training_experiment(
        df=df,
        csv_path=args.csv_path,
        feature_set="full",
        output_dir=args.output_dir,
        rf_results_dir=args.rf_results_dir,
        mlp_dir=args.mlp_dir,
        train_epsilon_list=train_epsilon_list,
        eval_epsilon_list=eval_epsilon_list,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        train_attack_scope=args.train_attack_scope,
        eval_attack_scope=args.eval_attack_scope,
    )


if __name__ == "__main__":
    main()