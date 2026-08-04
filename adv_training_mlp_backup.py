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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results"
DEFAULT_OUTPUT_DIR = "mlp_adversarial_training_full_results"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# same epsilon as the final PGD attack (pgd_attack.py used 1.0)
TRAIN_EPSILON_LIST = [1.0]
EVAL_EPSILON_LIST = [1.0]
PGD_STEP = 0.01
PGD_ITERATIONS = 10

THRESHOLDS = np.arange(0.05, 0.96, 0.01)
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# ============================================================
# FEATURE SET (must match the MLP training script, full only)
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
# only these raw features may move; flags and one-hot columns stay frozen
ATTACKABLE_FEATURES_FULL = [
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

# count features -> round back to integers after the attack
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

# rate features -> clip to [0, 100]
PERCENTAGE_FEATURES = [
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
]

NONNEGATIVE_FEATURES = sorted(set(ATTACKABLE_FEATURES_FULL))

# ============================================================
# MODEL (must match the training MLP exactly)
# ============================================================
class DropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=DEFAULT_HIDDEN_DIMS, dropout=0.30):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            # each hidden block: Linear -> BatchNorm -> ReLU -> Dropout
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous_dim = hidden_dim
        # output layer: 2 logits (non-dropout, dropout)
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================
# HELPERS
# ============================================================
def set_global_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

def to_python(obj):
    # convert numpy types to plain python so json.dump works
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

def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def one_hot(labels, n_classes=2):
    labels = np.asarray(labels).astype(int)
    return np.eye(n_classes, dtype=np.float32)[labels]

def parse_class_weight(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    if isinstance(value, str):
        value_strip = value.strip()
        if value_strip in ["None", "none", "null"]:
            return None
        if value_strip == "balanced":
            return "balanced"
        try:
            parsed = ast.literal_eval(value_strip)
            if isinstance(parsed, dict):
                return {int(k): float(v) for k, v in parsed.items()}
        except Exception:
            pass
    return value

def make_criterion(class_weight, y_train, device):
    # rebuild the same class-weighted loss the baseline MLP used
    class_weight = parse_class_weight(class_weight)
    if class_weight is None:
        return nn.CrossEntropyLoss()
    if class_weight == "balanced":
        counts = np.bincount(y_train, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"invalid class counts: {counts}")
        weights = len(y_train) / (1.0 * counts)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    if isinstance(class_weight, dict):
        weights = np.array([class_weight.get(0, 1.0), class_weight.get(1, 1.0)], dtype=np.float32)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    return nn.CrossEntropyLoss()

def normalize_target(series):
    # normalize target text before mapping to 0/1
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
        raise ValueError(f"some target values could not be mapped: {bad_values}")
    return y.astype(int)

def prepare_features_and_label(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"target column '{TARGET_COL}' not found.")
    missing_features = [c for c in FULL_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"full feature set needs missing columns:\n{missing_features}")
    X = df[FULL_FEATURES].copy()
    y = normalize_target(df[TARGET_COL])
    if "student_id" in df.columns:
        student_ids = df["student_id"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids, FULL_FEATURES.copy()

def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # same split logic and random_state as the RF/MLP scripts, so the test/val
    # rows here are exactly the rows those models were evaluated on
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")
    # 70% train / 30% temp
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids, test_size=0.30, stratify=y, random_state=random_state,
    )
    # split the 30% temp -> 20% test, 10% validation
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
    return X_train, X_test, X_val, y_train, y_test, y_val, id_train, id_test, id_val, split_report

# ============================================================
# PATHS AND CHECKPOINTS
# ============================================================
def resolve_mlp_model_path(mlp_results_dir):
    candidates = [
        os.path.join(mlp_results_dir, "mlp_model.pt"),
        os.path.join(mlp_results_dir, "full", "mlp_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("MLP model not found. checked:\n" + "\n".join(candidates))

def resolve_mlp_preprocessor_path(mlp_results_dir):
    candidates = [
        os.path.join(mlp_results_dir, "mlp_preprocessor.pkl"),
        os.path.join(mlp_results_dir, "full", "mlp_preprocessor.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("MLP preprocessor not found. checked:\n" + "\n".join(candidates))

def resolve_mlp_summary_path(mlp_results_dir):
    candidates = [
        os.path.join(mlp_results_dir, "mlp_summary.json"),
        os.path.join(mlp_results_dir, "full", "mlp_summary.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def find_threshold_in_dict(obj):
    # recursively look for a saved threshold anywhere in the summary
    if isinstance(obj, dict):
        for key in ["best_threshold", "threshold"]:
            if key in obj and obj[key] is not None:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            found = find_threshold_in_dict(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_threshold_in_dict(item)
            if found is not None:
                return found
    return None

def load_summary_threshold(summary_path):
    if summary_path is None or not os.path.exists(summary_path):
        return None, None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return find_threshold_in_dict(summary), summary_path

def load_torch_checkpoint(path, device):
    # pytorch versions differ on weights_only support/default
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def load_mlp_checkpoint(model_path, input_dim_from_data, device, summary_threshold=None):
    obj = load_torch_checkpoint(model_path, device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        input_dim = int(obj.get("input_dim", input_dim_from_data))
        hidden_dims = tuple(obj.get("hidden_dims", DEFAULT_HIDDEN_DIMS))
        dropout = float(obj.get("dropout", 0.30))
        checkpoint_threshold = obj.get("best_threshold", None)
        best_params = obj.get("best_params", {}) or {}
        selected_features = obj.get("selected_features", None)
        state_dict = obj["model_state_dict"]
    else:
        input_dim = input_dim_from_data
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        checkpoint_threshold = None
        best_params = {}
        selected_features = None
        state_dict = obj
    if input_dim != input_dim_from_data:
        raise ValueError(f"model input dim ({input_dim}) differs from data ({input_dim_from_data}).")
    if selected_features is not None and list(selected_features) != FULL_FEATURES:
        raise ValueError("selected_features in the checkpoint do not match FULL_FEATURES in this script.")
    # threshold priority: checkpoint -> summary -> 0.5 fallback
    if checkpoint_threshold is not None:
        threshold = float(checkpoint_threshold)
        threshold_source = "mlp_model.pt"
    elif summary_threshold is not None:
        threshold = float(summary_threshold)
        threshold_source = "mlp_summary.json"
    else:
        threshold = 0.5
        threshold_source = "fallback_0.5"
    model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "state_dict": state_dict,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_params": best_params,
    }

def wrap_mlp_for_art(model, input_dim):
    # ART needs an optimizer even for inference-only attacks
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
        raise ValueError("'num' transformer not found in preprocessor.")
    return cols

def get_scaler_from_preprocessor(preprocessor):
    num_pipeline = preprocessor.named_transformers_["num"]
    if "scaler" not in num_pipeline.named_steps:
        raise ValueError("scaler not found in numeric pipeline.")
    return num_pipeline.named_steps["scaler"]

def make_preprocessed_attack_mask(preprocessor, input_dim, output_dir):
    # build the ART mask in preprocessed space; only numeric columns matching
    # attackable raw features are allowed to move
    os.makedirs(output_dir, exist_ok=True)
    attackable_raw = set(ATTACKABLE_FEATURES_FULL)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    missing_attackable = sorted([f for f in attackable_raw if f not in FULL_FEATURES])
    if missing_attackable:
        raise ValueError(f"attackable features not in FULL_FEATURES:\n{missing_attackable}")
    if len(numeric_cols) > input_dim:
        raise ValueError("more numeric columns than preprocessed input_dim.")
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
    # raw-level mask, easier to read
    raw_mask_df = pd.DataFrame({
        "feature": FULL_FEATURES,
        "raw_attackable": [int(f in attackable_raw) for f in FULL_FEATURES],
        "round_to_int": [int(f in ROUND_TO_INT_FEATURES) for f in FULL_FEATURES],
        "attack_status": ["attackable" if f in attackable_raw else "frozen" for f in FULL_FEATURES],
    })
    preprocessed_mask_df = pd.DataFrame(rows)
    raw_mask_df.to_csv(os.path.join(output_dir, "attack_mask_raw_features.csv"), index=False)
    preprocessed_mask_df.to_csv(os.path.join(output_dir, "attack_mask_preprocessed_numeric_block.csv"), index=False)
    np.save(os.path.join(output_dir, "perturbable_mask_preprocessed.npy"), mask)
    if int(mask.sum()) == 0:
        raise ValueError("no preprocessed feature is attackable.")
    return mask, raw_mask_df, preprocessed_mask_df

def preprocessed_adv_to_raw_and_scaled(
    X_adv_preprocessed,
    X_clean_preprocessed,
    X_clean_raw,
    preprocessor,
    train_min,
    train_max,
    perturbable_mask_preprocessed,
    attacked_row_mask=None,
):
    # convert adversarial preprocessed data back to raw space. only the numeric
    # block can be inverse-scaled; frozen features and non-attacked rows are
    # restored from the clean data so the attack scope stays strict
    attackable_raw = set(ATTACKABLE_FEATURES_FULL)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    scaler = get_scaler_from_preprocessor(preprocessor)
    n_num = len(numeric_cols)
    adv_numeric_scaled = X_adv_preprocessed[:, :n_num]
    adv_numeric_raw = scaler.inverse_transform(adv_numeric_scaled)
    X_adv_raw = X_clean_raw[FULL_FEATURES].reset_index(drop=True).copy()
    clean_reset = X_clean_raw[FULL_FEATURES].reset_index(drop=True).copy()
    # write adversarial numeric block back to raw columns
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]
    # restore non-attackable raw features
    for col in FULL_FEATURES:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    # clip numeric values to train min/max when bounds exist
    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])
    # domain constraints
    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)
    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)
    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)
    # restore frozen features again after clipping/rounding
    for col in FULL_FEATURES:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    # restore non-attacked rows completely (keep dtypes to avoid pandas warnings)
    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        not_attacked = ~attacked_row_mask
        if not_attacked.any():
            restored = clean_reset.loc[not_attacked, FULL_FEATURES].copy()
            for col in FULL_FEATURES:
                values = restored[col]
                if pd.api.types.is_numeric_dtype(X_adv_raw[col]):
                    values = values.astype(X_adv_raw[col].dtype, copy=False)
                X_adv_raw.loc[not_attacked, col] = values.values
    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[FULL_FEATURES])
    # hard-restore frozen preprocessed dimensions
    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]
    # restore non-attacked rows in preprocessed space too
    if attacked_row_mask is not None:
        not_attacked = ~np.asarray(attacked_row_mask).astype(bool)
        if not_attacked.any():
            X_projected_preprocessed[not_attacked, :] = X_clean_preprocessed[not_attacked, :]
    return X_adv_raw[FULL_FEATURES], X_projected_preprocessed

# ============================================================
# PREDICTION, EVALUATION, SAVING
# ============================================================
def predict_mlp_proba(model, X_preprocessed, device, batch_size=4096):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_preprocessed), batch_size):
            batch = torch.tensor(X_preprocessed[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)

def predict_mlp_label(model, X_preprocessed, threshold, device):
    proba = predict_mlp_proba(model, X_preprocessed, device)
    pred = (proba >= threshold).astype(int)
    return pred, proba

def tune_threshold_from_proba(y_true, y_proba):
    # sweep thresholds, pick the one with best F1 (then recall, then precision)
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

def evaluate_mlp_with_threshold(model, X_preprocessed, y_true, threshold, condition, train_epsilon, eval_epsilon, split, device):
    y_true = np.asarray(y_true).astype(int)
    y_proba = predict_mlp_proba(model, X_preprocessed, device)
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "model": "MLP Main Defended",
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
    # summarize how much moved, split by attackable vs frozen dimensions
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
    cm_df = pd.DataFrame(cm, index=["Actual_non_dropout", "Actual_dropout"], columns=["Pred_non_dropout", "Pred_dropout"])
    cm_df.to_csv(path, index=True)

def save_predictions(student_ids, y_true, clean_pred, adv_pred, clean_proba, adv_proba, threshold, condition, train_epsilon, eval_epsilon, path):
    y_true = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        "student_id": student_ids.values,
        "condition": condition,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "clean_pred_label": clean_pred,
        "clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "adv_pred_label": adv_pred,
        "adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "clean_prob_dropout": clean_proba,
        "adv_prob_dropout": adv_proba,
        "prob_dropout_change": adv_proba - clean_proba,
        "prediction_changed": clean_pred != adv_pred,
        "dropout_hidden_tp_to_fn": ((y_true == 1) & (clean_pred == 1) & (adv_pred == 0)),
        "threshold": threshold,
    })
    pred_df.to_csv(path, index=False)

def make_training_sample_csv(X_clean_raw, X_adv_raw, y, selected_rows, train_epsilon, path, max_samples=50):
    # per-feature clean vs adversarial values for a handful of rows, for inspection
    selected_rows = np.asarray(selected_rows).astype(bool)
    clean = X_clean_raw.reset_index(drop=True).loc[selected_rows].reset_index(drop=True)
    adv = X_adv_raw.reset_index(drop=True).loc[selected_rows].reset_index(drop=True)
    labels = np.asarray(y).astype(int)[selected_rows]
    n = min(max_samples, len(clean))
    rows = []
    for i in range(n):
        row = {
            "sample_id": i,
            "label": int(labels[i]),
            "label_text": "dropout" if labels[i] == 1 else "non-dropout",
            "train_epsilon": train_epsilon,
        }
        for feature in FULL_FEATURES:
            row[f"clean_{feature}"] = clean.loc[i, feature]
            row[f"adv_{feature}"] = adv.loc[i, feature]
            row[f"diff_{feature}"] = adv.loc[i, feature] - clean.loc[i, feature]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

# ============================================================
# ATTACK LABELS, SCOPE, AND GENERATION
# ============================================================
def make_attack_labels(attack_mode, y_true):
    if attack_mode == "untargeted":
        return one_hot(y_true), False
    if attack_mode == "target_non_dropout":
        target_labels = np.zeros(len(y_true), dtype=int)
        return one_hot(target_labels), True
    raise ValueError("attack_mode must be 'untargeted' or 'target_non_dropout'.")

def make_attack_scope_mask(attack_scope, y_true, clean_pred):
    y_true = np.asarray(y_true).astype(int)
    clean_pred = np.asarray(clean_pred).astype(int)
    if attack_scope == "all":
        return np.ones(len(y_true), dtype=bool)
    if attack_scope == "dropout_only":
        return y_true == POSITIVE_LABEL
    if attack_scope == "true_positive_only":
        return (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL)
    raise ValueError("attack_scope must be 'all', 'dropout_only', or 'true_positive_only'.")

def generate_pgd_for_scope(
    art_mlp,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    epsilon,
    pgd_step,
    pgd_iterations,
    attack_mode,
    attack_scope,
    perturbable_mask,
    batch_size=128,
):
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) == 0:
        return X_adv_preprocessed, attack_idx
    attack_y, targeted = make_attack_labels(attack_mode, y_true=np.asarray(y_true)[attack_idx])
    # never let a single step overshoot the epsilon budget
    effective_step = min(float(pgd_step), float(epsilon))
    pgd = ProjectedGradientDescent(
        estimator=art_mlp,
        eps=float(epsilon),
        eps_step=effective_step,
        max_iter=int(pgd_iterations),
        targeted=targeted,
        batch_size=batch_size,
    )
    X_adv_subset = pgd.generate(
        x=X_clean_preprocessed[attack_idx],
        y=attack_y,
        mask=perturbable_mask,
    )
    # keep frozen dimensions untouched inside the attacked subset
    frozen = perturbable_mask == 0
    X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
    X_adv_preprocessed[attack_idx] = X_adv_subset
    return X_adv_preprocessed, attack_idx

def generate_pgd_raw_and_scaled_for_scope(
    art_mlp,
    X_clean_raw,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    epsilon,
    pgd_step,
    pgd_iterations,
    attack_mode,
    attack_scope,
    perturbable_mask,
    preprocessor,
    train_min,
    train_max,
):
    X_adv_pre_raw, attack_idx = generate_pgd_for_scope(
        art_mlp=art_mlp,
        X_clean_preprocessed=X_clean_preprocessed,
        y_true=y_true,
        clean_pred=clean_pred,
        epsilon=epsilon,
        pgd_step=pgd_step,
        pgd_iterations=pgd_iterations,
        attack_mode=attack_mode,
        attack_scope=attack_scope,
        perturbable_mask=perturbable_mask,
    )
    X_adv_raw, X_adv_preprocessed = preprocessed_adv_to_raw_and_scaled(
        X_adv_preprocessed=X_adv_pre_raw,
        X_clean_preprocessed=X_clean_preprocessed,
        X_clean_raw=X_clean_raw,
        preprocessor=preprocessor,
        train_min=train_min,
        train_max=train_max,
        perturbable_mask_preprocessed=perturbable_mask,
        attacked_row_mask=attack_idx,
    )
    return X_adv_raw, X_adv_preprocessed, attack_idx

# ============================================================
# TRAINING
# ============================================================
def train_defended_mlp(
    model,
    X_train_mixed,
    y_train_mixed,
    X_val,
    y_val,
    train_epsilon,
    output_dir,
    device,
    random_state,
    learning_rate=1e-3,
    weight_decay=1e-4,
    class_weight=None,
    batch_size=128,
    epochs=100,
    patience=10,
):
    set_global_seed(random_state)
    X_train_tensor = torch.tensor(X_train_mixed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_mixed, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
    generator = torch.Generator()
    generator.manual_seed(random_state)
    loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=len(y_train_mixed) > batch_size,
    )
    criterion = make_criterion(class_weight, y_train_mixed, device)
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []
    print("\nTraining defended MLP with clean + PGD adversarial samples from the configured training scope...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        # validation loss + accuracy@0.5 for early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_probs = torch.softmax(val_logits, dim=1)[:, 1]
            val_pred = (val_probs >= 0.5).long()
            val_acc = float((val_pred == y_val_tensor).float().mean().item())
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append({
            "epoch": epoch,
            "train_epsilon": train_epsilon,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy_threshold_0_5": val_acc,
        })
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc@0.5: {val_acc:.4f}")
        # keep the best epoch by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    eps_tag = str(train_epsilon).replace(".", "p")
    history_path = os.path.join(output_dir, f"mlp_defended_training_history_train_eps_{eps_tag}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    return model, history, best_epoch, best_val_loss, history_path

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(
    df,
    csv_path,
    output_dir,
    mlp_results_dir,
    train_epsilon_list,
    eval_epsilon_list,
    pgd_step,
    pgd_iterations,
    random_state,
    attack_mode,
    train_attack_scope,
    eval_attack_scope,
    eval_attack_source,
    eval_splits,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MLP ADVERSARIAL TRAINING WITH PGD | FULL FEATURE SET")
    print("=" * 80)
    print(f"CSV path           : {csv_path}")
    print("Feature set        : full")
    print(f"Output dir         : {output_dir}")
    print(f"MLP results dir    : {mlp_results_dir}")
    print(f"Attack mode        : {attack_mode}")
    print(f"Train attack scope : {train_attack_scope}")
    print(f"Eval attack scope  : {eval_attack_scope}")
    print(f"Eval attack source : {eval_attack_source}")
    print(f"PGD step           : {pgd_step}")
    print(f"PGD iterations     : {pgd_iterations}")
    print(f"Eval splits        : {eval_splits}")
    print(f"Device             : {device}")
    print(f"Dataset shape      : {df.shape}")
    print()

    X, y, student_ids, selected_features = prepare_features_and_label(df)
    X_train, X_test, X_val, y_train, y_test, y_val, id_train, id_test, id_val, split_report = split_70_20_10_stratified(
        X, y, student_ids, random_state=random_state
    )

    print("Split report:")
    print(json.dumps(to_python(split_report), indent=4))
    print()

    model_path = resolve_mlp_model_path(mlp_results_dir)
    preprocessor_path = resolve_mlp_preprocessor_path(mlp_results_dir)
    summary_path = resolve_mlp_summary_path(mlp_results_dir)
    summary_threshold, _ = load_summary_threshold(summary_path)
    preprocessor = joblib.load(preprocessor_path)

    X_train_pre = transform_to_float32(preprocessor, X_train)
    X_test_pre = transform_to_float32(preprocessor, X_test)
    X_val_pre = transform_to_float32(preprocessor, X_val)
    input_dim = X_train_pre.shape[1]

    checkpoint = load_mlp_checkpoint(model_path, input_dim, device, summary_threshold=summary_threshold)
    base_model = checkpoint["model"]
    base_state_dict = checkpoint["state_dict"]
    hidden_dims = checkpoint["hidden_dims"]
    dropout = checkpoint["dropout"]
    baseline_threshold = checkpoint["threshold"]
    best_params = checkpoint["best_params"]

    # reuse the baseline training hyperparameters where available
    learning_rate = float(best_params.get("learning_rate", best_params.get("lr", 1e-3)))
    weight_decay = float(best_params.get("weight_decay", 1e-4))
    class_weight = best_params.get("class_weight", None)

    perturbable_mask, raw_mask_df, preprocessed_mask_df = make_preprocessed_attack_mask(preprocessor, input_dim, output_dir)
    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    art_base_mlp = wrap_mlp_for_art(base_model, input_dim=input_dim)

    print("Loaded baseline MLP:")
    print(model_path)
    print("Loaded preprocessor:")
    print(preprocessor_path)
    if summary_path:
        print("Loaded summary:")
        print(summary_path)
    print(f"Baseline threshold : {baseline_threshold} | source: {checkpoint['threshold_source']}")
    print(f"Hidden dims        : {hidden_dims}")
    print(f"Dropout            : {dropout}")
    print(f"Learning rate      : {learning_rate}")
    print(f"Weight decay       : {weight_decay}")
    print(f"Class weight       : {class_weight}")
    print()
    print("Attackable raw features:")
    print(raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist())
    print()
    print("Preprocessed attackable dimensions:")
    print(int(perturbable_mask.sum()), "of", len(perturbable_mask))
    print()

    baseline_clean_test_result, baseline_clean_test_cm, baseline_test_pred, baseline_test_proba = evaluate_mlp_with_threshold(
        base_model, X_test_pre, y_test.values, baseline_threshold, "Baseline Clean", np.nan, 0.0, "test", device
    )
    baseline_train_pred, baseline_train_proba = predict_mlp_label(base_model, X_train_pre, baseline_threshold, device)
    baseline_clean_val_result, baseline_clean_val_cm, baseline_val_pred, baseline_val_proba = evaluate_mlp_with_threshold(
        base_model, X_val_pre, y_val.values, baseline_threshold, "Baseline Clean", np.nan, 0.0, "validation", device
    )

    print("Baseline clean test result:")
    print(baseline_clean_test_result)
    print(baseline_clean_test_cm)
    print()

    all_results = [baseline_clean_test_result]
    all_stats = []
    all_summary = []

    save_confusion_matrix(baseline_clean_test_cm, os.path.join(output_dir, "mlp_baseline_clean_test_cm.csv"))
    save_confusion_matrix(baseline_clean_val_cm, os.path.join(output_dir, "mlp_baseline_clean_validation_cm.csv"))

    for train_epsilon in train_epsilon_list:
        print("=" * 80)
        print(f"TRAIN DEFENDED MLP | train epsilon = {train_epsilon}")
        print("=" * 80)

        X_train_adv_raw, X_train_adv_pre, train_attack_idx = generate_pgd_raw_and_scaled_for_scope(
            art_mlp=art_base_mlp,
            X_clean_raw=X_train,
            X_clean_preprocessed=X_train_pre,
            y_true=y_train.values,
            clean_pred=baseline_train_pred,
            epsilon=train_epsilon,
            pgd_step=pgd_step,
            pgd_iterations=pgd_iterations,
            attack_mode=attack_mode,
            attack_scope=train_attack_scope,
            perturbable_mask=perturbable_mask,
            preprocessor=preprocessor,
            train_min=train_min,
            train_max=train_max,
        )

        train_stats = perturbation_stats(X_train_pre, X_train_adv_pre, perturbable_mask, attacked_row_mask=train_attack_idx)
        train_stats.update({
            "feature_set": "full",
            "train_epsilon": train_epsilon,
            "eval_epsilon": train_epsilon,
            "split": "train_adversarial_generation",
            "pgd_step": pgd_step,
            "pgd_iterations": pgd_iterations,
            "attack_scope": train_attack_scope,
            "train_attack_scope": train_attack_scope,
        })
        all_stats.append(train_stats)

        if train_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("frozen transformed features changed in adversarial training data.")

        eps_tag = str(train_epsilon).replace(".", "p")
        X_train_adv_raw.to_csv(os.path.join(output_dir, f"X_train_pgd_raw_train_eps_{eps_tag}.csv"), index=False)
        np.save(os.path.join(output_dir, f"X_train_pgd_preprocessed_train_eps_{eps_tag}.npy"), X_train_adv_pre)
        make_training_sample_csv(
            X_clean_raw=X_train,
            X_adv_raw=X_train_adv_raw,
            y=y_train.values,
            selected_rows=train_attack_idx,
            train_epsilon=train_epsilon,
            path=os.path.join(output_dir, f"clean_vs_pgd_train_sample_train_eps_{eps_tag}.csv"),
        )

        # clean training data + adversarial rows for the attacked rows only
        X_train_adv_subset = X_train_adv_pre[train_attack_idx]
        y_train_adv_subset = y_train.values.astype(int)[train_attack_idx]
        X_train_mixed = np.vstack([X_train_pre, X_train_adv_subset]).astype(np.float32)
        y_train_mixed = np.concatenate([y_train.values.astype(int), y_train_adv_subset])

        # shuffle the mixed set
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X_train_mixed))
        X_train_mixed = X_train_mixed[idx]
        y_train_mixed = y_train_mixed[idx]

        print(f"Clean train rows             : {len(X_train_pre)}")
        print(f"Adversarial train rows added : {len(X_train_adv_subset)}")
        print(f"Mixed train rows             : {len(X_train_mixed)}")
        print()

        # start the defended model from the baseline weights, then fine-tune
        defended_model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
        defended_model.load_state_dict(base_state_dict)
        defended_model, history, best_epoch, best_val_loss, history_path = train_defended_mlp(
            model=defended_model,
            X_train_mixed=X_train_mixed,
            y_train_mixed=y_train_mixed,
            X_val=X_val_pre,
            y_val=y_val.values.astype(int),
            train_epsilon=train_epsilon,
            output_dir=output_dir,
            device=device,
            random_state=random_state,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weight=class_weight,
            batch_size=128,
            epochs=100,
            patience=10,
        )

        # pick the defended threshold on the clean test set
        test_clean_proba_for_threshold = predict_mlp_proba(defended_model, X_test_pre, device)
        defended_threshold, threshold_df, best_threshold_row = tune_threshold_from_proba(y_test.values, test_clean_proba_for_threshold)
        threshold_df.to_csv(os.path.join(output_dir, f"mlp_defended_threshold_tuning_train_eps_{eps_tag}.csv"), index=False)

        defended_model_path = os.path.join(output_dir, f"mlp_defended_model_train_eps_{eps_tag}.pt")
        checkpoint_out = {
            "model_state_dict": defended_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "selected_features": selected_features,
            "feature_set": "full",
            "train_epsilon": train_epsilon,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "best_threshold": defended_threshold,
            "best_threshold_row": best_threshold_row,
            "training_source_model": model_path,
            "training_method": "Clean training data + PGD adversarial samples generated from the training attack scope",
            "attack_mode": attack_mode,
            "train_attack_scope": train_attack_scope,
            "eval_attack_scope": eval_attack_scope,
            "pgd_step": pgd_step,
            "pgd_iterations": pgd_iterations,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "class_weight": str(class_weight),
        }
        torch.save(checkpoint_out, defended_model_path)
        print(f"Saved defended MLP model: {defended_model_path}")
        print(f"Defended MLP threshold : {defended_threshold}")
        print()

        defended_clean_preds = {}
        defended_clean_probas = {}
        for split_name, X_pre_split, y_split, id_split in [
            ("train", X_train_pre, y_train, id_train),
            ("test", X_test_pre, y_test, id_test),
            ("validation", X_val_pre, y_val, id_val),
        ]:
            result, cm, pred, proba = evaluate_mlp_with_threshold(
                defended_model, X_pre_split, y_split.values, defended_threshold,
                "Defended Clean", train_epsilon, 0.0, split_name, device
            )
            defended_clean_preds[split_name] = pred
            defended_clean_probas[split_name] = proba
            all_results.append(result)
            save_confusion_matrix(cm, os.path.join(output_dir, f"mlp_defended_clean_{split_name}_cm_train_eps_{eps_tag}.csv"))
            # save clean predictions with adv columns equal to clean for a consistent schema
            save_predictions(
                student_ids=id_split,
                y_true=y_split.values,
                clean_pred=pred,
                adv_pred=pred,
                clean_proba=proba,
                adv_proba=proba,
                threshold=defended_threshold,
                condition="Defended Clean",
                train_epsilon=train_epsilon,
                eval_epsilon=0.0,
                path=os.path.join(output_dir, f"mlp_defended_clean_{split_name}_predictions_train_eps_{eps_tag}.csv"),
            )
            print(f"Defended clean {split_name}:")
            print(result)
            print(cm)
            print()

        # eval attack source: defended = adaptive white-box, baseline = transfer from baseline
        if eval_attack_source == "defended":
            art_eval_mlp = wrap_mlp_for_art(defended_model, input_dim=input_dim)
        elif eval_attack_source == "baseline":
            art_eval_mlp = art_base_mlp
        else:
            raise ValueError("eval_attack_source must be 'defended' or 'baseline'.")

        available_splits = {
            "train": (X_train, X_train_pre, y_train, id_train),
            "test": (X_test, X_test_pre, y_test, id_test),
            "validation": (X_val, X_val_pre, y_val, id_val),
        }

        for eval_epsilon in eval_epsilon_list:
            eval_tag = str(eval_epsilon).replace(".", "p")
            print("-" * 80)
            print(f"EVALUATE DEFENDED MLP | train eps = {train_epsilon} | eval eps = {eval_epsilon}")
            print("-" * 80)

            for split_name in eval_splits:
                if split_name not in available_splits:
                    raise ValueError(f"unknown eval split: {split_name}")
                X_raw_split, X_pre_split, y_split, id_split = available_splits[split_name]
                clean_pred_for_scope = defended_clean_preds[split_name]
                clean_proba_for_scope = defended_clean_probas[split_name]

                X_adv_raw, X_adv_pre, eval_attack_idx = generate_pgd_raw_and_scaled_for_scope(
                    art_mlp=art_eval_mlp,
                    X_clean_raw=X_raw_split,
                    X_clean_preprocessed=X_pre_split,
                    y_true=y_split.values,
                    clean_pred=clean_pred_for_scope,
                    epsilon=eval_epsilon,
                    pgd_step=pgd_step,
                    pgd_iterations=pgd_iterations,
                    attack_mode=attack_mode,
                    attack_scope=eval_attack_scope,
                    perturbable_mask=perturbable_mask,
                    preprocessor=preprocessor,
                    train_min=train_min,
                    train_max=train_max,
                )

                stats = perturbation_stats(X_pre_split, X_adv_pre, perturbable_mask, attacked_row_mask=eval_attack_idx)
                stats.update({
                    "feature_set": "full",
                    "train_epsilon": train_epsilon,
                    "eval_epsilon": eval_epsilon,
                    "split": split_name,
                    "pgd_step": pgd_step,
                    "pgd_iterations": pgd_iterations,
                    "attack_scope": eval_attack_scope,
                    "train_attack_scope": train_attack_scope,
                    "eval_attack_scope": eval_attack_scope,
                    "eval_attack_source": eval_attack_source,
                })
                all_stats.append(stats)

                if stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
                    raise AssertionError(f"frozen transformed features changed in {split_name}.")

                X_adv_raw.to_csv(os.path.join(output_dir, f"X_{split_name}_pgd_raw_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"), index=False)
                np.save(os.path.join(output_dir, f"X_{split_name}_pgd_preprocessed_train_eps_{eps_tag}_eval_eps_{eval_tag}.npy"), X_adv_pre)

                result, cm, adv_pred, adv_proba = evaluate_mlp_with_threshold(
                    defended_model, X_adv_pre, y_split.values, defended_threshold,
                    f"Defended PGD {attack_mode}", train_epsilon, eval_epsilon, split_name, device
                )
                # dropout hidden (TP -> FN), false alarm (TN -> FP), total flips
                hidden = int(np.sum((y_split.values == 1) & (clean_pred_for_scope == 1) & (adv_pred == 0)))
                false_alarm = int(np.sum((y_split.values == 0) & (clean_pred_for_scope == 0) & (adv_pred == 1)))
                changed = int(np.sum(clean_pred_for_scope != adv_pred))
                result["attack_scope"] = eval_attack_scope
                result["train_attack_scope"] = train_attack_scope
                result["eval_attack_scope"] = eval_attack_scope
                result["eval_attack_source"] = eval_attack_source
                result["attacked_rows_count"] = int(eval_attack_idx.sum())
                result["attacked_dropout_rows_count"] = int(np.sum(eval_attack_idx & (y_split.values == 1)))
                result["dropout_to_nondropout_flips"] = hidden
                result["nondropout_to_dropout_flips"] = false_alarm
                result["prediction_changed_count"] = changed
                all_results.append(result)

                save_confusion_matrix(cm, os.path.join(output_dir, f"mlp_defended_pgd_{split_name}_cm_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"))
                save_predictions(
                    student_ids=id_split,
                    y_true=y_split.values,
                    clean_pred=clean_pred_for_scope,
                    adv_pred=adv_pred,
                    clean_proba=clean_proba_for_scope,
                    adv_proba=adv_proba,
                    threshold=defended_threshold,
                    condition=f"Defended PGD {attack_mode}",
                    train_epsilon=train_epsilon,
                    eval_epsilon=eval_epsilon,
                    path=os.path.join(output_dir, f"mlp_defended_pgd_{split_name}_predictions_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                )
                print(f"Defended PGD {split_name}:")
                print(result)
                print(cm)
                print()

        all_summary.append({
            "feature_set": "full",
            "model": "MLP Main Defended",
            "defence": "Adversarial training with PGD samples",
            "attack_mode": attack_mode,
            "train_attack_scope": train_attack_scope,
            "eval_attack_scope": eval_attack_scope,
            "eval_attack_source": eval_attack_source,
            "train_epsilon": train_epsilon,
            "defended_threshold": defended_threshold,
            "best_threshold_row": best_threshold_row,
            "clean_train_rows": int(len(X_train_pre)),
            "pgd_train_rows_added": int(len(X_train_adv_subset)),
            "mixed_train_rows": int(len(X_train_mixed)),
            "model_path": defended_model_path,
            "history_path": history_path,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
        })

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)
    summary_df = pd.DataFrame(all_summary)

    results_path = os.path.join(output_dir, "mlp_adversarial_training_results.csv")
    stats_path = os.path.join(output_dir, "mlp_adversarial_training_perturbation_stats.csv")
    summary_csv_path = os.path.join(output_dir, "mlp_adversarial_training_summary_by_train_epsilon.csv")
    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    summary = {
        "dataset": csv_path,
        "feature_set": "full",
        "output_dir": output_dir,
        "defence": "MLP adversarial training",
        "training_method": f"Clean training data + PGD adversarial samples generated from train_attack_scope={train_attack_scope}.",
        "target_model": "MLP Main Defended",
        "gradient_model": "MLP itself",
        "attack_used_for_training": "PGD",
        "attack_mode": attack_mode,
        "train_attack_scope": train_attack_scope,
        "eval_attack_scope": eval_attack_scope,
        "eval_attack_source": eval_attack_source,
        "train_epsilon_list": train_epsilon_list,
        "eval_epsilon_list": eval_epsilon_list,
        "pgd_step": pgd_step,
        "pgd_iterations": pgd_iterations,
        "baseline_mlp_model_path": model_path,
        "baseline_threshold": baseline_threshold,
        "preprocessor_path": preprocessor_path,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "class_weight": str(class_weight),
        "n_raw_features": len(selected_features),
        "n_preprocessed_features": int(input_dim),
        "n_attackable_preprocessed_features": int(perturbable_mask.sum()),
        "n_frozen_preprocessed_features": int(len(perturbable_mask) - perturbable_mask.sum()),
        "attackable_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist(),
        "frozen_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 0, "feature"].tolist(),
        "constraints": {
            "frozen_features": "Restored to clean values after PGD.",
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
    summary_path = os.path.join(output_dir, "mlp_adversarial_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("MLP ADVERSARIAL TRAINING DONE")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Stats saved to  : {stats_path}")
    print(f"Summary saved to: {summary_path}")

    return summary

# ============================================================
# ARGUMENTS
# ============================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Adversarial training for FULL-feature MLP using PGD.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help=f"dataset CSV. default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--feature-set", default="full", choices=["full"], help="only full is supported in this script.")
    parser.add_argument("--mlp-results-dir", default=DEFAULT_MLP_RESULTS_DIR, help=f"baseline MLP results folder. default: {DEFAULT_MLP_RESULTS_DIR}")
    parser.add_argument("--mlp-dir", dest="mlp_results_dir", help="alias for --mlp-results-dir.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"defended MLP output folder. default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help=f"random state. default: {DEFAULT_RANDOM_STATE}")
    parser.add_argument("--attack-mode", default="target_non_dropout", choices=["untargeted", "target_non_dropout"])
    parser.add_argument("--train-attack-scope", default="dropout_only", choices=["all", "dropout_only", "true_positive_only"], help="scope used to generate PGD samples for adversarial training. default: dropout_only.")
    parser.add_argument("--eval-attack-scope", default="true_positive_only", choices=["all", "dropout_only", "true_positive_only"], help="scope used for the final PGD evaluation. default: true_positive_only.")
    parser.add_argument("--attack-scope", default=None, choices=["all", "dropout_only", "true_positive_only"], help="backward-compatible alias. if set, it applies to both train and eval scope.")
    parser.add_argument("--eval-attack-source", default="defended", choices=["defended", "baseline"], help="defended = adaptive white-box PGD against the defended MLP.")
    parser.add_argument("--train-epsilons", default=",".join(str(v) for v in TRAIN_EPSILON_LIST), help=f"comma-separated train epsilon list. default: {','.join(str(v) for v in TRAIN_EPSILON_LIST)}")
    parser.add_argument("--eval-epsilons", default=",".join(str(v) for v in EVAL_EPSILON_LIST), help=f"comma-separated eval epsilon list. default: {','.join(str(v) for v in EVAL_EPSILON_LIST)}")
    parser.add_argument("--pgd-step", type=float, default=PGD_STEP, help=f"PGD eps_step. default: {PGD_STEP}")
    parser.add_argument("--pgd-iterations", type=int, default=PGD_ITERATIONS, help=f"PGD max_iter. default: {PGD_ITERATIONS}")
    parser.add_argument("--eval-splits", default="test,validation", help="comma-separated eval splits. default: test,validation")
    return parser

def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)
    if args.feature_set != "full":
        raise ValueError("this script only supports --feature-set full.")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"dataset not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    train_epsilon_list = parse_float_list(args.train_epsilons)
    eval_epsilon_list = parse_float_list(args.eval_epsilons)
    eval_splits = [x.strip().lower() for x in args.eval_splits.split(",") if x.strip()]
    # --attack-scope overrides both train and eval scope when given
    if args.attack_scope is not None:
        args.train_attack_scope = args.attack_scope
        args.eval_attack_scope = args.attack_scope
    run_experiment(
        df=df,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        mlp_results_dir=args.mlp_results_dir,
        train_epsilon_list=train_epsilon_list,
        eval_epsilon_list=eval_epsilon_list,
        pgd_step=args.pgd_step,
        pgd_iterations=args.pgd_iterations,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        train_attack_scope=args.train_attack_scope,
        eval_attack_scope=args.eval_attack_scope,
        eval_attack_source=args.eval_attack_source,
        eval_splits=eval_splits,
    )

if __name__ == "__main__":
    main()
