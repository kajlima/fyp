import os
import json
import itertools
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_OUTPUT_DIR = "rf_results"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# thresholds to sweep when tuning the decision cutoff
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# regularized grid: shallow trees + large leaf sizes to reduce overfitting,
# class_weight to handle the imbalanced dropout class
PARAM_GRID = {
    "n_estimators": [500],
    "max_depth": [4, 6, 8, 10],
    "min_samples_split": [50, 100, 200],
    "min_samples_leaf": [20, 50, 100],
    "max_features": ["sqrt", 0.5],
    "class_weight": [
        "balanced_subsample",
        {0: 1, 1: 4},
        {0: 1, 1: 6},
        {0: 1, 1: 8},
        {0: 1, 1: 10},
    ],
    "bootstrap": [True],
    "max_samples": [0.7],
}

# ============================================================
# FEATURE SET
# ============================================================
NON_FEATURE_COLS = ["student_id", "academic_year", "target"]

# seniority -> one-hot (not treated as a numeric scale)
# *_flag columns -> kept as 0/1, not scaled
ONEHOT_CATEGORICAL_FEATURES = ["seniority"]
FLAG_COLUMN_SUFFIX = "_flag"

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

FEATURE_SET_DESCRIPTIONS = {
    "full": "FULL_FEATURES only: end-of-year time point.",
}


def validate_feature_set_definitions() -> dict:
    # make sure no column is listed twice
    duplicate_full = sorted({c for c in FULL_FEATURES if FULL_FEATURES.count(c) > 1})
    if duplicate_full:
        raise ValueError(f"Duplicated columns in FULL_FEATURES: {duplicate_full}")
    return {
        "n_full_features_defined": len(FULL_FEATURES),
        "full_features_definition": "FULL_FEATURES only",
    }


def get_selected_features_for_feature_set(feature_set: str) -> list:
    if feature_set != "full":
        raise ValueError("Only feature_set='full' is supported.")
    return FULL_FEATURES.copy()


def save_feature_set_metadata(output_dir: str, feature_set: str, selected_features: list, df_columns: list) -> str:
    # write a small JSON logging which features were actually used
    feature_meta = validate_feature_set_definitions()
    present_full = [c for c in FULL_FEATURES if c in df_columns]
    missing_full = [c for c in FULL_FEATURES if c not in df_columns]
    meta = {
        **feature_meta,
        "feature_set_used": feature_set,
        "feature_set_description": FEATURE_SET_DESCRIPTIONS.get(feature_set, "full feature set"),
        "n_features_selected": len(selected_features),
        "selected_features": selected_features,
        "present_full_features_count": len(present_full),
        "missing_full_features": missing_full,
        "non_feature_columns_excluded": NON_FEATURE_COLS,
        "leakage_note": "student_id, academic_year, and target are excluded from predictors.",
    }
    path = os.path.join(output_dir, "rf_feature_set_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(meta), f, indent=4, ensure_ascii=False)
    return path


# ============================================================
# HELPERS
# ============================================================
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


def make_onehot_encoder():
    # sparse_output vs sparse depending on sklearn version
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_features_and_label(df, feature_set):
    validate_feature_set_definitions()
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # normalize target text before mapping to 0/1
    target_normalized = (
        df[TARGET_COL]
        .astype(str)
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
    df["label"] = target_normalized.map(target_map)
    if df["label"].isna().any():
        bad_values = df.loc[df["label"].isna(), TARGET_COL].unique().tolist()
        raise ValueError(f"Some target values could not be mapped: {bad_values}")
    df["label"] = df["label"].astype(int)

    print("Target distribution:")
    print(df[TARGET_COL].value_counts())
    print()
    print("Numeric label distribution:")
    print(df["label"].value_counts().sort_index())
    print(df["label"].value_counts(normalize=True).sort_index())
    print()

    # only the full feature set is used
    feature_set = "full"
    selected_features = get_selected_features_for_feature_set(feature_set)

    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing columns for feature set '{feature_set}': {missing_features}")

    X = df[selected_features].copy()
    y = df["label"]

    student_ids = None
    if "student_id" in df.columns:
        student_ids = df["student_id"].copy()

    print(f"Feature set used: {feature_set}")
    print(f"Number of features used: {X.shape[1]}")
    print("Features used:")
    print(selected_features)
    print()

    return X, y, student_ids, selected_features


def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # stratified split: ~70% train, ~20% test, ~10% validation
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")

    # 70% train / 30% temp
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X,
        y,
        student_ids,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    # split the 30% temp -> 20% test, 10% validation
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=1.0 / 3.0,
        stratify=y_temp,
        random_state=random_state,
    )

    n_total = len(X)
    split_report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "validation_rows": len(X_val),
        "train_pct": len(X_train) / n_total,
        "test_pct": len(X_test) / n_total,
        "validation_pct": len(X_val) / n_total,
    }

    print("Stratified split check:")
    print(f"Train      : {split_report['train_rows']} rows ({split_report['train_pct']:.2%})")
    print(f"Test       : {split_report['test_rows']} rows ({split_report['test_pct']:.2%})")
    print(f"Validation : {split_report['validation_rows']} rows ({split_report['validation_pct']:.2%})")
    print()
    print("Train distribution:")
    print(y_train.value_counts().sort_index())
    print(y_train.value_counts(normalize=True).sort_index())
    print()
    print("Test distribution:")
    print(y_test.value_counts().sort_index())
    print(y_test.value_counts(normalize=True).sort_index())
    print()
    print("Validation distribution:")
    print(y_val.value_counts().sort_index())
    print(y_val.value_counts(normalize=True).sort_index())
    print()

    return X_train, X_test, X_val, y_train, y_test, y_val, id_train, id_test, id_val, split_report


def infer_feature_groups(X):
    # split columns into: numeric (median impute), binary flags (kept 0/1),
    # categorical (one-hot). seniority is forced into the categorical group.
    all_cols = X.columns.tolist()

    explicit_onehot_cols = [col for col in ONEHOT_CATEGORICAL_FEATURES if col in X.columns]
    object_categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    categorical_cols = []
    for col in explicit_onehot_cols + object_categorical_cols:
        if col not in categorical_cols:
            categorical_cols.append(col)

    flag_cols = [
        col for col in all_cols
        if col.endswith(FLAG_COLUMN_SUFFIX) and col not in categorical_cols
    ]
    numeric_cols = [
        col for col in X.select_dtypes(include=["number", "bool"]).columns.tolist()
        if col not in flag_cols and col not in categorical_cols
    ]

    # every column must land in exactly one group
    assigned_cols = set(numeric_cols) | set(flag_cols) | set(categorical_cols)
    unassigned_cols = [col for col in all_cols if col not in assigned_cols]
    if unassigned_cols:
        raise ValueError(f"Some features were not assigned to a group: {unassigned_cols}")

    return numeric_cols, flag_cols, categorical_cols


def build_preprocessor(X):
    numeric_cols, flag_cols, categorical_cols = infer_feature_groups(X)

    transformers = []
    if numeric_cols:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", numeric_transformer, numeric_cols))
    if flag_cols:
        flag_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
        transformers.append(("flag", flag_transformer, flag_cols))
    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No columns available for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, flag_cols, categorical_cols


def tune_threshold_from_proba(y_true, y_proba):
    # sweep thresholds, pick the one with best F1 (then recall, then precision)
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        })
    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        ["f1_score", "recall", "precision"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_df, best_row.to_dict()


def compute_metrics_with_threshold(model, X, y, split_name, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, y_pred),
        "precision_dropout": precision_score(y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }
    return result, y_pred, y_proba


def save_predictions(student_ids, y_true, y_pred, y_proba, threshold, path):
    pred_df = pd.DataFrame({
        "student_id": student_ids.values,
        "actual_label": y_true.values,
        "actual_target": np.where(y_true.values == 1, "dropout", "non-dropout"),
        "predicted_label": y_pred,
        "predicted_target": np.where(y_pred == 1, "dropout", "non-dropout"),
        "prob_dropout": y_proba,
        "threshold": threshold,
    })
    pred_df.to_csv(path, index=False)


def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)


def get_feature_names_from_pipeline(pipeline, numeric_cols, flag_cols, categorical_cols):
    # rebuild the column order after preprocessing (for feature importance)
    feature_names = []
    feature_names.extend(numeric_cols)
    feature_names.extend(flag_cols)
    if categorical_cols:
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_pipeline = preprocessor.named_transformers_["cat"]
        onehot = cat_pipeline.named_steps["onehot"]
        try:
            cat_feature_names = onehot.get_feature_names_out(categorical_cols).tolist()
        except AttributeError:
            cat_feature_names = onehot.get_feature_names(categorical_cols).tolist()
        feature_names.extend(cat_feature_names)
    return feature_names


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Regularized Random Forest for student dropout prediction with stratified split."
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the dataset CSV. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=["full"],
        help="Feature set to use (only 'full').",
    )
    parser.add_argument(
        "--random-state",
        default=DEFAULT_RANDOM_STATE,
        type=int,
        help=f"Random state. Default: {DEFAULT_RANDOM_STATE}",
    )
    return parser


# ============================================================
# MAIN
# ============================================================
def run_single_rf_experiment(df, csv_path, output_dir, feature_set, random_state):
    # train on train set, pick params + threshold on test set,
    # keep validation set for the final held-out evaluation
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("RANDOM FOREST - STRATIFIED SPLIT + REGULARIZED TUNING")
    print("=" * 80)
    print(f"CSV path     : {csv_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Feature set  : {feature_set}")
    print(f"Random state : {random_state}")
    print(f"Shape dataset: {df.shape}")
    print()

    X, y, student_ids, selected_features = prepare_features_and_label(df, feature_set)

    (
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        id_train,
        id_test,
        id_val,
        split_report,
    ) = split_70_20_10_stratified(X, y, student_ids, random_state=random_state)

    preprocessor, numeric_cols, flag_cols, categorical_cols = build_preprocessor(X_train)
    print("Preprocessing groups:")
    print(f"Numeric imputed columns   : {len(numeric_cols)}")
    print(f"Binary flag columns       : {len(flag_cols)}")
    print(f"Categorical one-hot cols  : {len(categorical_cols)}")
    print("Categorical one-hot cols:", categorical_cols)
    print()

    feature_list_path = os.path.join(output_dir, "rf_features_used.csv")
    pd.DataFrame({"feature": selected_features}).to_csv(feature_list_path, index=False)

    feature_metadata_path = save_feature_set_metadata(
        output_dir=output_dir,
        feature_set=feature_set,
        selected_features=selected_features,
        df_columns=df.columns.tolist(),
    )

    # build all hyperparameter combinations
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    tuning_rows = []
    best_score = -np.inf
    best_params = None
    best_model = None
    best_threshold = None
    best_threshold_row = None

    print(f"Tuning {len(combos)} Random Forest combinations.")
    print("Threshold is selected on the test set; validation is kept for final evaluation.")
    print("selection_score = test_f1 - 0.25 * max(0, train_f1 - test_f1)")
    print()

    for combo_idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
        model = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", rf),
            ]
        )
        model.fit(X_train, y_train)

        # pick threshold on the test set
        test_proba_tuning = model.predict_proba(X_test)[:, 1]
        threshold, threshold_df, threshold_best_row = tune_threshold_from_proba(y_test, test_proba_tuning)
        test_pred_tuning = (test_proba_tuning >= threshold).astype(int)

        # measure the train-test F1 gap as an overfitting penalty
        train_proba_tmp = model.predict_proba(X_train)[:, 1]
        train_pred_tmp = (train_proba_tmp >= threshold).astype(int)
        train_f1_tmp = f1_score(y_train, train_pred_tmp, pos_label=POSITIVE_LABEL, zero_division=0)
        test_f1_tmp = f1_score(y_test, test_pred_tuning, pos_label=POSITIVE_LABEL, zero_division=0)

        overfit_gap = train_f1_tmp - test_f1_tmp
        selection_score = test_f1_tmp - 0.25 * max(0, overfit_gap)

        row = {
            **params,
            "best_threshold": threshold,
            "test_accuracy": accuracy_score(y_test, test_pred_tuning),
            "test_precision_dropout": precision_score(y_test, test_pred_tuning, pos_label=POSITIVE_LABEL, zero_division=0),
            "test_recall_dropout": recall_score(y_test, test_pred_tuning, pos_label=POSITIVE_LABEL, zero_division=0),
            "test_f1_dropout": test_f1_tmp,
            "train_f1_dropout_tmp": train_f1_tmp,
            "overfit_gap": overfit_gap,
            "selection_score": selection_score,
        }
        tuning_rows.append(row)

        score = row["selection_score"]
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            best_threshold = threshold
            best_threshold_row = threshold_best_row

        if combo_idx % 10 == 0 or combo_idx == len(combos):
            print(f"Tuning progress: {combo_idx}/{len(combos)} completed")

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["selection_score", "test_f1_dropout", "test_recall_dropout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    tuning_path = os.path.join(output_dir, "rf_tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print("Top 10 tuning results:")
    print(tuning_df.head(10).to_string(index=False))
    print()
    print(f"Best params          : {best_params}")
    print(f"Best threshold       : {best_threshold:.2f}")
    print(f"Best selection score : {best_score:.4f}")
    print(f"Best test F1         : {best_threshold_row['f1_score']:.4f}")
    print()

    rf_model = best_model

    # save the full threshold sweep for the best model on the test set
    best_test_proba = rf_model.predict_proba(X_test)[:, 1]
    _, best_threshold_df, _ = tune_threshold_from_proba(y_test, best_test_proba)
    best_threshold_df.to_csv(os.path.join(output_dir, "rf_threshold_tuning_test.csv"), index=False)

    train_results, train_pred, train_proba = compute_metrics_with_threshold(
        rf_model, X_train, y_train, "train", best_threshold
    )
    test_results, test_pred, test_proba = compute_metrics_with_threshold(
        rf_model, X_test, y_test, "test_selection", best_threshold
    )
    val_results, val_pred, val_proba = compute_metrics_with_threshold(
        rf_model, X_val, y_val, "validation_final", best_threshold
    )

    results_df = pd.DataFrame([train_results, test_results, val_results])
    results_path = os.path.join(output_dir, "rf_results.csv")
    results_df.to_csv(results_path, index=False)
    print("Performance results:")
    print(results_df.to_string(index=False))
    print()
    print("Final held-out evaluation split: validation")
    print()

    save_predictions(id_train, y_train, train_pred, train_proba, best_threshold,
                     os.path.join(output_dir, "rf_train_predictions.csv"))
    save_predictions(id_test, y_test, test_pred, test_proba, best_threshold,
                     os.path.join(output_dir, "rf_test_selection_predictions.csv"))
    save_predictions(id_val, y_val, val_pred, val_proba, best_threshold,
                     os.path.join(output_dir, "rf_validation_final_predictions.csv"))

    # confusion matrices for test (selection) and validation (final),
    # rows = actual, cols = predicted, order [non-dropout, dropout]
    test_cm = confusion_matrix(y_test, test_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    val_cm = confusion_matrix(y_val, val_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    save_confusion_matrix(test_cm, os.path.join(output_dir, "rf_test_confusion_matrix.csv"))
    save_confusion_matrix(val_cm, os.path.join(output_dir, "rf_validation_confusion_matrix.csv"))

    print("Test confusion matrix:")
    print(pd.DataFrame(
        test_cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    ).to_string())
    print()
    print("Validation confusion matrix:")
    print(pd.DataFrame(
        val_cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    ).to_string())
    print()

    model_path = os.path.join(output_dir, "rf_model.pkl")
    joblib.dump(rf_model, model_path)

    # feature importance from the final fitted forest
    try:
        final_rf = rf_model.named_steps["model"]
        feature_names = get_feature_names_from_pipeline(rf_model, numeric_cols, flag_cols, categorical_cols)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": final_rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(os.path.join(output_dir, "rf_feature_importance.csv"), index=False)
    except Exception as e:
        print(f"Failed to save feature importance: {e}")

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "feature_set": feature_set,
        "n_features": len(selected_features),
        "features_used": selected_features,
        "feature_set_metadata_file": feature_metadata_path,
        "feature_set_logic": {
            "full": "Uses FULL_FEATURES only.",
            "n_full_features_defined": len(FULL_FEATURES),
        },
        "label_mapping": {
            "0": "non-dropout",
            "1": "dropout",
        },
        "split": {
            "method": "Stratified random split via train_test_split",
            "train": "approximately 70%",
            "test": "approximately 20%",
            "validation": "approximately 10%",
            "stratified": True,
            "group_aware": False,
            "random_state": random_state,
            "split_report": split_report,
            "important_note": "Split 70/20/10 stratified by label. Each row is an independent observation; no grouping by student_id.",
        },
        "model": "RandomForestClassifier",
        "preprocessing": {
            "policy": "Numeric median-imputed, *_flag kept as 0/1, seniority/categorical one-hot encoded.",
            "numeric": "SimpleImputer(strategy='median') fitted on train only; no scaling (RF does not need it)",
            "binary_flags": "SimpleImputer(strategy='most_frequent'); kept as 0/1",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore') fitted on train only",
            "numeric_columns": numeric_cols,
            "binary_flag_columns": flag_cols,
            "categorical_columns": categorical_cols,
            "onehot_categorical_policy": ONEHOT_CATEGORICAL_FEATURES,
        },
        "balancing_strategy": "class_weight tuning, no resampling",
        "selection": {
            "selection_split": "test",
            "selection_metric": "selection_score = test_f1_dropout - 0.25 * max(0, train_f1_dropout - test_f1_dropout)",
            "best_params": best_params,
            "best_threshold": best_threshold,
            "best_selection_score": best_score,
            "best_test_f1": best_threshold_row["f1_score"],
            "best_threshold_row": best_threshold_row,
        },
        "final_evaluation": {
            "final_evaluation_split": "validation",
            "note": "Validation is the held-out set, evaluated after selection on the test set.",
            "metrics_used": [
                "accuracy_score",
                "precision_score",
                "recall_score",
                "f1_score",
            ],
        },
        "results": {
            "train": train_results,
            "test_selection": test_results,
            "validation_final": val_results,
        },
        "decision_rule": f"Predict dropout if prob_dropout >= {best_threshold:.2f}",
    }
    with open(os.path.join(output_dir, "rf_summary.json"), "w") as f:
        json.dump(to_python(summary), f, indent=4)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output saved to folder: {output_dir}")
    print(f"Model saved to: {model_path}")
    print(f"Results CSV: {results_path}")
    print(f"Tuning CSV : {tuning_path}")
    print(f"Features CSV: {feature_list_path}")
    print(f"Feature metadata JSON: {feature_metadata_path}")

    return summary


def main():
    args = build_arg_parser().parse_args()
    csv_path = args.csv_path
    output_dir = args.output_dir
    feature_set = args.feature_set
    random_state = args.random_state

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File not found: {csv_path}\n"
            f"Put the CSV next to this script, or pass --csv-path path/file.csv"
        )

    df = pd.read_csv(csv_path)
    validate_feature_set_definitions()
    feature_set = "full"

    run_single_rf_experiment(
        df=df,
        csv_path=csv_path,
        output_dir=output_dir,
        feature_set=feature_set,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
