import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results"
DEFAULT_OUTPUT_DIR = "proximity_sensitivity_final_results"
DEFAULT_RF_DEFENCE_DIR = "rf_adversarial_training_full_results"
DEFAULT_MLP_DEFENCE_DIR = "mlp_adversarial_training_full_results"

# RF uses FGSM at epsilon 1.5, MLP uses PGD at epsilon 1.0. the budgets differ on
# purpose (FGSM transfer needed the larger budget to hurt RF F1, PGD reaches the
# defended MLP directly at 1.0), so RF and MLP rows are not a like-for-like compare
RF_TRAIN_EPSILON = 1.5
RF_EVAL_EPSILON = 1.5
MLP_TRAIN_EPSILON = 1.0
MLP_EVAL_EPSILON = 1.0

TARGET_COL = "target"
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
EPS = 1e-8


# ============================================================
# FEATURE SET (must match the training scripts, full only)
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


# ============================================================
# DATA HELPERS
# ============================================================

def eps_tag(value: float) -> str:
    return str(value).replace(".", "p")


def candidate_eps_strings(value: float) -> List[str]:
    # supports both the new p-format and the old decimal format in filenames
    raw = str(value)
    return list(dict.fromkeys([eps_tag(value), raw, raw.rstrip("0").rstrip(".")]))


def find_existing_path(patterns: List[str]) -> Optional[str]:
    for path in patterns:
        if os.path.exists(path):
            return path
    return None


def normalize_target(series: pd.Series) -> pd.Series:
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


def load_and_split_dataset(csv_path: str, random_state: int):
    df = pd.read_csv(csv_path)
    missing = [c for c in FULL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing FULL_FEATURES columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"missing target column: {TARGET_COL}")

    X = df[FULL_FEATURES].copy()
    y = normalize_target(df[TARGET_COL])

    if "student_id" in df.columns:
        student_ids = df["student_id"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")

    # same stratified 70/20/10 split and random_state as the training/defence scripts,
    # so the test/val rows and their order match the adversarial arrays saved there
    # (clean and adversarial samples line up row by row)
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids, test_size=0.30, stratify=y, random_state=random_state,
    )
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=1.0 / 3.0, stratify=y_temp, random_state=random_state,
    )

    splits = {
        "train": (X_train, y_train, id_train),
        "test": (X_test, y_test, id_test),
        "validation": (X_val, y_val, id_val),
    }
    split_data = {}
    for split, (Xs, ys, ids) in splits.items():
        split_data[split] = {
            "X_raw": Xs.reset_index(drop=True).copy(),
            "y": ys.reset_index(drop=True).values.astype(int),
            "student_ids": ids.reset_index(drop=True),
        }
    return split_data


def load_preprocessor(mlp_results_dir: str):
    candidates = [
        os.path.join(mlp_results_dir, "mlp_preprocessor.pkl"),
        os.path.join(mlp_results_dir, "full", "mlp_preprocessor.pkl"),
    ]
    path = find_existing_path(candidates)
    if path is None:
        raise FileNotFoundError("MLP preprocessor not found. checked:\n" + "\n".join(candidates))
    return joblib.load(path), path


def transform_to_float32(preprocessor, X: pd.DataFrame) -> np.ndarray:
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)


# ============================================================
# PREPROCESSOR MASK HELPERS
# ============================================================

def get_transformer_columns(preprocessor, transformer_name: str) -> List[str]:
    for name, transformer, cols in preprocessor.transformers_:
        if name == transformer_name:
            return list(cols)
    return []


def get_transformer_slice(preprocessor, transformer_name: str, input_dim: int, fallback_start: int = 0):
    if hasattr(preprocessor, "output_indices_") and transformer_name in preprocessor.output_indices_:
        return preprocessor.output_indices_[transformer_name]
    # fallback: assume the numeric block comes first
    if transformer_name == "num":
        n_cols = len(get_transformer_columns(preprocessor, "num"))
        return slice(0, min(n_cols, input_dim))
    return slice(0, 0)


def make_preprocessed_masks(preprocessor, input_dim: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(input_dim)]

    perturbable_mask = np.zeros(input_dim, dtype=bool)
    numerical_mask = np.zeros(input_dim, dtype=bool)
    categorical_ohe_mask = np.zeros(input_dim, dtype=bool)

    numeric_cols = get_transformer_columns(preprocessor, "num")
    num_slice = get_transformer_slice(preprocessor, "num", input_dim)
    num_indices = list(range(num_slice.start or 0, num_slice.stop or 0))

    # numeric block: mark attackable raw numeric features as perturbable
    for local_idx, raw_col in enumerate(numeric_cols):
        if local_idx >= len(num_indices):
            continue
        idx = num_indices[local_idx]
        numerical_mask[idx] = True
        if raw_col in set(ATTACKABLE_FEATURES_FULL):
            perturbable_mask[idx] = True

    # other blocks are categorical/OHE for sensitivity purposes; frozen in the attacks
    for name, _, _ in preprocessor.transformers_:
        if name in ["num", "remainder"]:
            continue
        if hasattr(preprocessor, "output_indices_") and name in preprocessor.output_indices_:
            block_slice = preprocessor.output_indices_[name]
            categorical_ohe_mask[block_slice] = True

    # if no categorical mask was found, mark all non-numeric dims as categorical/frozen
    if categorical_ohe_mask.sum() == 0:
        categorical_ohe_mask[~numerical_mask] = True

    mask_df = pd.DataFrame({
        "preprocessed_index": np.arange(input_dim),
        "feature_name": feature_names,
        "is_attackable_feature": perturbable_mask,
        "is_numerical_feature": numerical_mask,
        "is_categorical_or_other_feature": categorical_ohe_mask,
    })
    mask_df.to_csv(os.path.join(output_dir, "preprocessed_feature_masks.csv"), index=False)
    return perturbable_mask, numerical_mask, categorical_ohe_mask, feature_names, mask_df


# ============================================================
# PREDICTION CSV HELPERS
# ============================================================

def extract_predictions_from_csv(pred_path: str, clean_pred_path: Optional[str] = None):
    pred_df = pd.read_csv(pred_path)
    if "actual_label" not in pred_df.columns:
        raise ValueError(f"actual_label not found in {pred_path}")
    y_true = pred_df["actual_label"].values.astype(int)

    if "adv_pred_label" in pred_df.columns:
        adv_pred = pred_df["adv_pred_label"].values.astype(int)
    elif "predicted_label" in pred_df.columns:
        adv_pred = pred_df["predicted_label"].values.astype(int)
    else:
        raise ValueError(f"no adv/predicted label column found in {pred_path}")

    if "clean_pred_label" in pred_df.columns:
        clean_pred = pred_df["clean_pred_label"].values.astype(int)
    elif clean_pred_path is not None and os.path.exists(clean_pred_path):
        clean_df = pd.read_csv(clean_pred_path)
        if "predicted_label" in clean_df.columns:
            clean_pred = clean_df["predicted_label"].values.astype(int)
        elif "adv_pred_label" in clean_df.columns:
            clean_pred = clean_df["adv_pred_label"].values.astype(int)
        else:
            raise ValueError(f"no predicted_label in clean prediction file: {clean_pred_path}")
    else:
        raise ValueError(
            f"clean_pred_label not found in {pred_path} and clean_pred_path was not available."
        )
    return y_true, clean_pred, adv_pred, pred_df


def true_positive_attack_mask(y_true: np.ndarray, clean_pred: np.ndarray) -> np.ndarray:
    return (np.asarray(y_true) == POSITIVE_LABEL) & (np.asarray(clean_pred) == POSITIVE_LABEL)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def safe_subset(X: np.ndarray, row_mask: Optional[np.ndarray]) -> np.ndarray:
    if row_mask is None:
        return X
    row_mask = np.asarray(row_mask).astype(bool)
    if row_mask.sum() == 0:
        return X[:0]
    return X[row_mask]


def compute_proximity_metrics(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    perturbable_mask: np.ndarray,
    row_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    # proximity (Eq. 3/4): L1/L2/Linf and mean/max absolute change, clean vs adversarial
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return {"row_count_used_for_distance": 0}

    diff = Xa - Xc
    abs_diff = np.abs(diff)
    pm = perturbable_mask.astype(bool)
    frozen = ~pm
    diff_attackable = diff[:, pm]
    abs_attackable = np.abs(diff_attackable)
    abs_frozen = np.abs(diff[:, frozen]) if frozen.any() else np.zeros((len(Xc), 0), dtype=np.float32)

    l1_all = np.linalg.norm(diff, ord=1, axis=1)
    l2_all = np.linalg.norm(diff, ord=2, axis=1)
    linf_all = np.linalg.norm(diff, ord=np.inf, axis=1)
    l1_attackable = np.linalg.norm(diff_attackable, ord=1, axis=1)
    l2_attackable = np.linalg.norm(diff_attackable, ord=2, axis=1)
    linf_attackable = np.linalg.norm(diff_attackable, ord=np.inf, axis=1)

    return {
        "row_count_used_for_distance": int(len(Xc)),
        "mean_l1_all": float(np.mean(l1_all)),
        "median_l1_all": float(np.median(l1_all)),
        "mean_l2_all": float(np.mean(l2_all)),
        "median_l2_all": float(np.median(l2_all)),
        "mean_linf_all": float(np.mean(linf_all)),
        "median_linf_all": float(np.median(linf_all)),
        "mean_l1_attackable": float(np.mean(l1_attackable)),
        "median_l1_attackable": float(np.median(l1_attackable)),
        "mean_l2_attackable": float(np.mean(l2_attackable)),
        "median_l2_attackable": float(np.median(l2_attackable)),
        "mean_linf_attackable": float(np.mean(linf_attackable)),
        "median_linf_attackable": float(np.median(linf_attackable)),
        "mean_abs_change_all": float(np.mean(abs_diff)),
        "mean_abs_change_attackable": float(np.mean(abs_attackable)),
        "max_abs_change_all": float(np.max(abs_diff)),
        "max_abs_change_attackable": float(np.max(abs_attackable)),
        "mean_abs_change_frozen": float(np.mean(abs_frozen)) if abs_frozen.size else 0.0,
        "max_abs_change_frozen": float(np.max(abs_frozen)) if abs_frozen.size else 0.0,
        "frozen_changed_total": int(np.sum(abs_frozen > EPS)) if abs_frozen.size else 0,
        "frozen_changed_samples": int(np.sum(np.any(abs_frozen > EPS, axis=1))) if abs_frozen.size else 0,
    }


def compute_sensitivity_metrics(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    numerical_mask: np.ndarray,
    categorical_ohe_mask: np.ndarray,
    row_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return {"row_count_used_for_sensitivity": 0}

    abs_diff = np.abs(Xa - Xc)
    num_mask = numerical_mask.astype(bool)
    cat_mask = categorical_ohe_mask.astype(bool)

    # sensitivity (Eq. 5/6): sum over numeric features of |adv - clean| / sigma_j, then
    # average over samples. the numeric block is StandardScaler output, so the absolute
    # numeric difference already equals |adv - clean| / sigma_j (no extra divide by std)
    if num_mask.sum() > 0:
        per_sample_sen = abs_diff[:, num_mask].sum(axis=1)
    else:
        per_sample_sen = np.zeros(len(Xc), dtype=np.float32)

    # diagnostic only (not part of the formula): fraction of categorical/OHE dims that
    # changed. should be ~0 because those features are frozen in the attacks
    if cat_mask.sum() > 0:
        cat_change_rate = (abs_diff[:, cat_mask] > EPS).astype(np.float32).mean(axis=1)
    else:
        cat_change_rate = np.zeros(len(Xc), dtype=np.float32)

    return {
        "row_count_used_for_sensitivity": int(len(Xc)),
        "mean_sensitivity": float(np.mean(per_sample_sen)),
        "median_sensitivity": float(np.median(per_sample_sen)),
        "max_sensitivity": float(np.max(per_sample_sen)),
        "mean_categorical_change_rate": float(np.mean(cat_change_rate)),
    }


def compute_attack_effect(y_true: np.ndarray, clean_pred: np.ndarray, adv_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    clean_pred = np.asarray(clean_pred).astype(int)
    adv_pred = np.asarray(adv_pred).astype(int)

    attack_idx = true_positive_attack_mask(y_true, clean_pred)
    hidden = (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred == NEGATIVE_LABEL)
    non_dropout_to_dropout = (y_true == NEGATIVE_LABEL) & (clean_pred == NEGATIVE_LABEL) & (adv_pred == POSITIVE_LABEL)
    changed = clean_pred != adv_pred

    attacked_count = int(attack_idx.sum())
    hidden_count = int(hidden.sum())
    return {
        "attacked_rows_count": attacked_count,
        "attacked_dropout_rows_count": int(np.sum(attack_idx & (y_true == POSITIVE_LABEL))),
        "dropout_to_nondropout_flips": hidden_count,
        "dropout_hiding_rate": float(hidden_count / attacked_count) if attacked_count else 0.0,
        "nondropout_to_dropout_flips": int(non_dropout_to_dropout.sum()),
        "prediction_changed_count": int(changed.sum()),
    }


def feature_sensitivity_summary(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    feature_names: List[str],
    perturbable_mask: np.ndarray,
    numerical_mask: np.ndarray,
    categorical_ohe_mask: np.ndarray,
    model: str,
    attack: str,
    split: str,
    train_epsilon: float,
    eval_epsilon: float,
    row_scope: str,
    row_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return pd.DataFrame()

    diff = Xa - Xc
    abs_diff = np.abs(diff)
    num_mask = numerical_mask.astype(bool)
    cat_mask = categorical_ohe_mask.astype(bool)

    # per-feature sensitivity term (Eq. 5): |adv - clean| / sigma_j. numeric block is
    # StandardScaler output so the absolute numeric diff already equals that; categorical
    # features are not part of the formula and stay at zero
    sens_matrix = np.zeros_like(abs_diff, dtype=np.float32)
    if num_mask.sum() > 0:
        sens_matrix[:, num_mask] = abs_diff[:, num_mask]

    df = pd.DataFrame({
        "model": model,
        "attack": attack,
        "split": split,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "row_scope": row_scope,
        "feature_name": feature_names,
        "is_attackable_feature": perturbable_mask.astype(bool),
        "is_numerical_feature": num_mask,
        "is_categorical_or_other_feature": cat_mask,
        "mean_abs_change": abs_diff.mean(axis=0),
        "max_abs_change": abs_diff.max(axis=0),
        "mean_signed_change": diff.mean(axis=0),
        "changed_sample_count": (abs_diff > EPS).sum(axis=0),
        "changed_sample_rate": (abs_diff > EPS).mean(axis=0),
        "mean_sensitivity": sens_matrix.mean(axis=0),
        "max_sensitivity": sens_matrix.max(axis=0),
    })
    df["sensitivity_rank"] = df["mean_sensitivity"].rank(ascending=False, method="dense")
    return df.sort_values("mean_sensitivity", ascending=False)


# ============================================================
# FILE PATH BUILDERS
# ============================================================

def defended_raw_adv_candidates(defence_dir: str, model_key: str, split: str, train_eps: float, eval_eps: float) -> List[str]:
    split = split.lower()
    te_tags = candidate_eps_strings(train_eps)
    ee_tags = candidate_eps_strings(eval_eps)
    out = []
    if model_key == "RF":
        for te in te_tags:
            for ee in ee_tags:
                out.append(os.path.join(defence_dir, f"X_{split}_fgsm_raw_train_eps_{te}_eval_eps_{ee}.csv"))
    elif model_key == "MLP":
        for te in te_tags:
            for ee in ee_tags:
                out.append(os.path.join(defence_dir, f"X_{split}_pgd_raw_train_eps_{te}_eval_eps_{ee}.csv"))
    return out


def defended_pre_adv_candidates(defence_dir: str, model_key: str, split: str, train_eps: float, eval_eps: float) -> List[str]:
    split = split.lower()
    te_tags = candidate_eps_strings(train_eps)
    ee_tags = candidate_eps_strings(eval_eps)
    out = []
    if model_key == "MLP":
        for te in te_tags:
            for ee in ee_tags:
                out.append(os.path.join(defence_dir, f"X_{split}_pgd_preprocessed_train_eps_{te}_eval_eps_{ee}.npy"))
                out.append(os.path.join(defence_dir, f"X_{split}_pgd_scaled_train_eps_{te}_eval_eps_{ee}.npy"))
    elif model_key == "RF":
        for te in te_tags:
            for ee in ee_tags:
                out.append(os.path.join(defence_dir, f"X_{split}_fgsm_preprocessed_train_eps_{te}_eval_eps_{ee}.npy"))
                out.append(os.path.join(defence_dir, f"X_{split}_fgsm_scaled_train_eps_{te}_eval_eps_{ee}.npy"))
    return out


def defended_pred_candidates(defence_dir: str, model_key: str, split: str, train_eps: float, eval_eps: float) -> List[str]:
    split = split.lower()
    te_tags = candidate_eps_strings(train_eps)
    ee_tags = candidate_eps_strings(eval_eps)
    out = []
    if model_key == "RF":
        for te in te_tags:
            for ee in ee_tags:
                out.append(os.path.join(defence_dir, f"rf_defended_fgsm_{split}_predictions_train_eps_{te}_eval_eps_{ee}.csv"))
                out.append(os.path.join(defence_dir, f"rf_defended_{split}_predictions_fgsm_train_eps_{te}_eval_eps_{ee}.csv"))
    elif model_key == "MLP":
        for te in te_tags:
            for ee in ee_tags:
                out.append(os.path.join(defence_dir, f"mlp_defended_pgd_{split}_predictions_train_eps_{te}_eval_eps_{ee}.csv"))
                out.append(os.path.join(defence_dir, f"mlp_defended_{split}_predictions_pgd_train_eps_{te}_eval_eps_{ee}.csv"))
    return out


def defended_clean_pred_candidates(defence_dir: str, model_key: str, split: str, train_eps: float) -> List[str]:
    split = split.lower()
    te_tags = candidate_eps_strings(train_eps)
    out = []
    if model_key == "RF":
        for te in te_tags:
            out.append(os.path.join(defence_dir, f"rf_defended_clean_{split}_predictions_train_eps_{te}.csv"))
    elif model_key == "MLP":
        for te in te_tags:
            out.append(os.path.join(defence_dir, f"mlp_defended_clean_{split}_predictions_train_eps_{te}.csv"))
    return out


# ============================================================
# CASE PROCESSOR
# ============================================================

def process_final_case(
    model_key: str,
    model_name: str,
    attack_name: str,
    defence_dir: str,
    train_eps: float,
    eval_eps: float,
    split_name: str,
    split_data: Dict,
    preprocessor,
    perturbable_mask: np.ndarray,
    numerical_mask: np.ndarray,
    categorical_ohe_mask: np.ndarray,
    feature_names: List[str],
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    split_lower = split_name.lower()
    X_clean_raw = split_data[split_lower]["X_raw"]
    X_clean_pre = transform_to_float32(preprocessor, X_clean_raw)

    raw_path = find_existing_path(defended_raw_adv_candidates(defence_dir, model_key, split_lower, train_eps, eval_eps))
    pre_path = find_existing_path(defended_pre_adv_candidates(defence_dir, model_key, split_lower, train_eps, eval_eps))
    pred_path = find_existing_path(defended_pred_candidates(defence_dir, model_key, split_lower, train_eps, eval_eps))
    clean_pred_path = find_existing_path(defended_clean_pred_candidates(defence_dir, model_key, split_lower, train_eps))

    if raw_path is None and pre_path is None:
        print(f"SKIP {model_name} {split_name}: adversarial sample file not found in {defence_dir}")
        return [], []
    if pred_path is None:
        print(f"SKIP {model_name} {split_name}: prediction CSV not found in {defence_dir}")
        return [], []

    if pre_path is not None:
        X_adv_pre = np.load(pre_path).astype(np.float32)
    else:
        X_adv_raw = pd.read_csv(raw_path)
        # keep column order aligned with FULL_FEATURES
        X_adv_pre = transform_to_float32(preprocessor, X_adv_raw[FULL_FEATURES])

    y_true, clean_pred, adv_pred, pred_df = extract_predictions_from_csv(pred_path, clean_pred_path)
    if len(y_true) != len(X_clean_pre):
        raise ValueError(f"length mismatch for {model_name} {split_name}: predictions vs clean data")
    if X_adv_pre.shape != X_clean_pre.shape:
        raise ValueError(f"shape mismatch for {model_name} {split_name}: clean {X_clean_pre.shape}, adv {X_adv_pre.shape}")

    attack_idx = true_positive_attack_mask(y_true, clean_pred)
    effect = compute_attack_effect(y_true, clean_pred, adv_pred)

    rows = []
    feature_dfs = []
    # report metrics for all rows and for attacked rows only
    for row_scope, row_mask in [("all_rows", None), ("attacked_rows_only", attack_idx)]:
        row = {
            "model": model_name,
            "attack": attack_name,
            "condition": "After Defence",
            "split": split_name,
            "train_epsilon": train_eps,
            "eval_epsilon": eval_eps,
            "epsilon": eval_eps,
            "row_scope": row_scope,
            "adversarial_sample_source": pre_path if pre_path is not None else raw_path,
            "prediction_source": pred_path,
            "clean_prediction_source": clean_pred_path if clean_pred_path is not None else "included_in_prediction_source",
        }
        row.update(effect)
        row.update(compute_proximity_metrics(X_clean_pre, X_adv_pre, perturbable_mask, row_mask=row_mask))
        row.update(compute_sensitivity_metrics(X_clean_pre, X_adv_pre, numerical_mask, categorical_ohe_mask, row_mask=row_mask))
        rows.append(row)

        fdf = feature_sensitivity_summary(
            X_clean=X_clean_pre,
            X_adv=X_adv_pre,
            feature_names=feature_names,
            perturbable_mask=perturbable_mask,
            numerical_mask=numerical_mask,
            categorical_ohe_mask=categorical_ohe_mask,
            model=model_name,
            attack=attack_name,
            split=split_name,
            train_epsilon=train_eps,
            eval_epsilon=eval_eps,
            row_scope=row_scope,
            row_mask=row_mask,
        )
        feature_dfs.append(fdf)

    print(
        f"OK {model_name} {split_name}: train_eps={train_eps}, eval_eps={eval_eps}, "
        f"attacked={effect['attacked_rows_count']}, hidden={effect['dropout_to_nondropout_flips']}"
    )
    return rows, feature_dfs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Final Zenodo proximity and sensitivity analysis.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--mlp-results-dir", default=DEFAULT_MLP_RESULTS_DIR)
    parser.add_argument("--rf-defence-dir", default=DEFAULT_RF_DEFENCE_DIR)
    parser.add_argument("--mlp-defence-dir", default=DEFAULT_MLP_DEFENCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--splits", default="test,validation", help="comma-separated: test,validation,train")
    parser.add_argument("--rf-train-epsilon", type=float, default=RF_TRAIN_EPSILON)
    parser.add_argument("--rf-eval-epsilon", type=float, default=RF_EVAL_EPSILON)
    parser.add_argument("--mlp-train-epsilon", type=float, default=MLP_TRAIN_EPSILON)
    parser.add_argument("--mlp-eval-epsilon", type=float, default=MLP_EVAL_EPSILON)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split_names = [s.strip().lower() for s in args.splits.split(",") if s.strip()]

    print("=" * 80)
    print("FINAL ZENODO PROXIMITY AND SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"CSV path       : {args.csv_path}")
    print(f"MLP preprocessor: {args.mlp_results_dir}")
    print(f"RF defence dir : {args.rf_defence_dir}")
    print(f"MLP defence dir: {args.mlp_defence_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Splits         : {split_names}")
    print()

    split_data = load_and_split_dataset(args.csv_path, args.random_state)
    preprocessor, preprocessor_path = load_preprocessor(args.mlp_results_dir)

    # build masks from the test-split dimension
    X_test_pre = transform_to_float32(preprocessor, split_data["test"]["X_raw"])
    input_dim = X_test_pre.shape[1]
    perturbable_mask, numerical_mask, categorical_ohe_mask, feature_names, mask_df = make_preprocessed_masks(
        preprocessor, input_dim, args.output_dir
    )

    print(f"Loaded preprocessor: {preprocessor_path}")
    print(f"Preprocessed dimension: {input_dim}")
    print(f"Attackable preprocessed dimensions: {int(perturbable_mask.sum())} of {input_dim}")
    print()

    all_rows: List[Dict] = []
    all_feature_dfs: List[pd.DataFrame] = []

    final_cases = [
        {
            "model_key": "RF",
            "model_name": "Random Forest Defended",
            "attack_name": "FGSM via source MLP",
            "defence_dir": args.rf_defence_dir,
            "train_eps": args.rf_train_epsilon,
            "eval_eps": args.rf_eval_epsilon,
        },
        {
            "model_key": "MLP",
            "model_name": "MLP Defended",
            "attack_name": "PGD defended-source",
            "defence_dir": args.mlp_defence_dir,
            "train_eps": args.mlp_train_epsilon,
            "eval_eps": args.mlp_eval_epsilon,
        },
    ]

    for case in final_cases:
        for split_name in split_names:
            rows, feature_dfs = process_final_case(
                model_key=case["model_key"],
                model_name=case["model_name"],
                attack_name=case["attack_name"],
                defence_dir=case["defence_dir"],
                train_eps=case["train_eps"],
                eval_eps=case["eval_eps"],
                split_name=split_name,
                split_data=split_data,
                preprocessor=preprocessor,
                perturbable_mask=perturbable_mask,
                numerical_mask=numerical_mask,
                categorical_ohe_mask=categorical_ohe_mask,
                feature_names=feature_names,
            )
            all_rows.extend(rows)
            all_feature_dfs.extend(feature_dfs)

    if not all_rows:
        raise FileNotFoundError(
            "no proximity/sensitivity rows were created. check that the RF/MLP defence output folders exist "
            "and contain adversarial sample files plus prediction CSVs."
        )

    summary_df = pd.DataFrame(all_rows)
    feature_df = pd.concat(all_feature_dfs, ignore_index=True) if all_feature_dfs else pd.DataFrame()

    summary_path = os.path.join(args.output_dir, "proximity_sensitivity_final_summary.csv")
    feature_path = os.path.join(args.output_dir, "feature_sensitivity_final_summary.csv")
    top15_path = os.path.join(args.output_dir, "top15_sensitive_features_final.csv")
    excel_path = os.path.join(args.output_dir, "proximity_sensitivity_final_analysis.xlsx")
    json_path = os.path.join(args.output_dir, "proximity_sensitivity_final_summary.json")

    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)

    if not feature_df.empty:
        top15_df = (
            feature_df[feature_df["is_attackable_feature"] == True]
            .sort_values(["model", "split", "row_scope", "mean_sensitivity"], ascending=[True, True, True, False])
            .groupby(["model", "split", "row_scope"])
            .head(15)
            .reset_index(drop=True)
        )
    else:
        top15_df = pd.DataFrame()
    top15_df.to_csv(top15_path, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not feature_df.empty:
            feature_df.to_excel(writer, sheet_name="Feature Sensitivity", index=False)
        if not top15_df.empty:
            top15_df.to_excel(writer, sheet_name="Top 15 Sensitive", index=False)
        mask_df.to_excel(writer, sheet_name="Preprocessed Masks", index=False)

    json_summary = {
        "dataset": args.csv_path,
        "feature_set": "full",
        "analysis": "Final proximity and sensitivity analysis for defended RF and defended MLP outputs.",
        "important_note": "Metrics are reported for all_rows and attacked_rows_only. The attacked_rows_only rows are more informative for true_positive_only attacks because non-attacked rows are restored to clean values.",
        "rf_final_setting": {
            "train_epsilon": args.rf_train_epsilon,
            "eval_epsilon": args.rf_eval_epsilon,
            "attack": "FGSM target_non_dropout via source MLP",
        },
        "mlp_final_setting": {
            "train_epsilon": args.mlp_train_epsilon,
            "eval_epsilon": args.mlp_eval_epsilon,
            "attack": "PGD target_non_dropout defended-source",
        },
        "proximity_definition": "L1, L2, Linf distances between clean and adversarial preprocessed samples.",
        "sensitivity_definition": "Sensitivity (Eq. 5/6) = sum over numeric features of |adv - clean| / sigma_j, averaged over samples. Numeric features are StandardScaler-scaled, so the absolute preprocessed difference already equals |adv - clean| / sigma_j.",
        "dropout_hiding_rate_definition": "dropout_to_nondropout_flips / attacked_rows_count for clean true-positive dropout cases.",
        "files_created": {
            "summary_csv": summary_path,
            "feature_sensitivity_csv": feature_path,
            "top15_sensitive_csv": top15_path,
            "excel": excel_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)

    print("\nSummary:")
    keep_cols = [
        "model", "split", "row_scope", "train_epsilon", "eval_epsilon",
        "attacked_rows_count", "dropout_to_nondropout_flips", "dropout_hiding_rate",
        "mean_l2_attackable", "mean_linf_attackable", "mean_sensitivity",
        "max_abs_change_frozen", "frozen_changed_samples",
    ]
    print(summary_df[keep_cols].to_string(index=False))

    print("\nSaved outputs:")
    print(summary_path)
    print(feature_path)
    print(top15_path)
    print(excel_path)
    print(json_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
