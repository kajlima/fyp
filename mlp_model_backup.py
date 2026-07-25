import os
import json
import copy
import random
import argparse
import itertools
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_OUTPUT_DIR = "mlp_results"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# network shape (funnel) + training hyperparameters
HIDDEN_DIMS = (128, 64, 32)
PARAM_GRID = {
    "class_weight": [
        None,
        "balanced",
        {0: 1, 1: 4},
        {0: 1, 1: 6},
    ],
    "learning_rate": [1e-3, 5e-4],
    "dropout": [0.20, 0.30],
    "weight_decay": [1e-4, 1e-3],
}
BATCH_SIZE = 128
EPOCHS = 100      # max epochs (early stopping usually stops earlier)
PATIENCE = 10     # stop after 10 epochs without improvement

# ============================================================
# FEATURE SET
# ============================================================
NON_FEATURE_COLS = ["student_id", "academic_year", "target"]

# preprocessing policy:
# - seniority -> one-hot (not a numeric scale)
# - *_flag columns -> kept as 0/1, not standardized
# - other numeric columns -> median-imputed + standardized
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


# ============================================================
# MODEL
# ============================================================
class DropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=0.30):
        super().__init__()
        layers = []
        previous_dim = input_dim
        # each hidden block: Linear -> BatchNorm -> ReLU -> Dropout
        for hidden_dim in hidden_dims:
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
def set_global_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    # note: benchmark=True favors speed over bit-exact reproducibility on GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def save_feature_set_metadata(output_dir, feature_set, selected_features, df_columns):
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
    path = os.path.join(output_dir, "mlp_feature_set_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(meta), f, indent=4, ensure_ascii=False)
    return path


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

    feature_set = "full"
    selected_features = get_selected_features_for_feature_set(feature_set)

    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing columns for feature set '{feature_set}': {missing_features}")

    X = df[selected_features].copy()
    y = df["label"].copy()

    if "student_id" in df.columns:
        student_ids = df["student_id"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")

    print("Target distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))
    print()
    print("Numeric label distribution:")
    print(y.value_counts().sort_index())
    print(y.value_counts(normalize=True).sort_index())
    print()
    print(f"Feature set used: {feature_set}")
    print(f"Number of features used: {X.shape[1]}")
    print("Features used:")
    print(selected_features)
    print()

    return X, y, student_ids, selected_features


def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # stratified split ~70/20/10; same logic and random_state as the RF script
    # so both models use identical partitions. each row is one unique student_id.
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
    # split columns into: numeric (scaled), binary flags (kept 0/1),
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
        # MLP needs scaling (unlike RF)
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
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


def transform_to_float32(preprocessor, X):
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)


def make_criterion(class_weight, y_train, device):
    # build the loss, applying class weights to handle imbalance
    if class_weight is None:
        return nn.CrossEntropyLoss()
    if class_weight == "balanced":
        counts = np.bincount(y_train, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"Invalid class count for balanced weights: {counts}")
        weights = len(y_train) / (2.0 * counts)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    if isinstance(class_weight, dict):
        weights = np.array(
            [
                float(class_weight.get(0, 1.0)),
                float(class_weight.get(1, 1.0)),
            ],
            dtype=np.float32,
        )
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Unknown class_weight: {class_weight}")


def train_one_mlp(
    params,
    X_train_np,
    y_train_np,
    X_selection_np,
    y_selection_np,
    input_dim,
    device,
    random_state,
    record_history=False,
):
    # same seed per combo so comparisons differ only by hyperparameters
    set_global_seed(random_state)

    model = DropoutMLP(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=params["dropout"],
    ).to(device)

    criterion = make_criterion(params["class_weight"], y_train_np, device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    X_tr_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train_np, dtype=torch.long)
    X_selection_tensor = torch.tensor(X_selection_np, dtype=torch.float32, device=device)
    y_selection_tensor = torch.tensor(y_selection_np, dtype=torch.long, device=device)

    generator = torch.Generator()
    generator.manual_seed(random_state)
    # drop last partial batch (BatchNorm breaks on a batch of size 1)
    drop_last = len(y_train_np) > BATCH_SIZE
    loader = DataLoader(
        TensorDataset(X_tr_tensor, y_tr_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
    )

    best_selection_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        # standard training step: zero_grad -> forward -> loss -> backward -> step
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        # evaluate on the selection (test) set for early stopping
        model.eval()
        with torch.no_grad():
            selection_logits = model(X_selection_tensor)
            selection_loss = float(criterion(selection_logits, y_selection_tensor).item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        if record_history:
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_selection_loss": selection_loss,
            })

        # keep the best weights; stop if no improvement for PATIENCE epochs
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # restore the best weights, not the last epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch, best_selection_loss


def predict_proba_mlp(model, X_np, device, batch_size=4096):
    model.eval()
    probs = []
    dataset = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)


def tune_threshold_from_proba(y_true, y_proba):
    # sweep thresholds, pick best F1 (then recall, then precision)
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


def compute_metrics_with_threshold(model, X_np, y, split_name, threshold, device):
    y_np = np.asarray(y).astype(int)
    y_proba = predict_proba_mlp(model, X_np, device)
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_np, y_pred),
        "precision_dropout": precision_score(y_np, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y_np, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y_np, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }
    return result, y_pred, y_proba


def save_predictions(student_ids, y_true, y_pred, y_proba, threshold, path):
    y_true_np = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        "student_id": student_ids.values,
        "actual_label": y_true_np,
        "actual_target": np.where(y_true_np == 1, "dropout", "non-dropout"),
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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="MLP for student dropout prediction with a stratified split and full features only."
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
# MAIN EXPERIMENT
# ============================================================
def run_single_mlp_experiment(df, csv_path, output_dir, feature_set, random_state):
    # train on train set; select params/early-stopping/threshold on test set;
    # keep validation set for the final held-out evaluation
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MLP - STRATIFIED SPLIT + FLAG/SENIORITY PREPROCESSING + REGULARIZED TUNING")
    print("=" * 80)
    print(f"CSV path     : {csv_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Feature set  : {feature_set}")
    print(f"Random state : {random_state}")
    print(f"Device       : {device}")
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
    print(f"Numeric scaled columns   : {len(numeric_cols)}")
    print(f"Binary flag columns      : {len(flag_cols)}")
    print(f"Categorical one-hot cols : {len(categorical_cols)}")
    print("Categorical one-hot cols:", categorical_cols)
    print()

    # fit preprocessing on train only, then apply to test/val
    X_train_np = preprocessor.fit_transform(X_train)
    if hasattr(X_train_np, "toarray"):
        X_train_np = X_train_np.toarray()
    X_train_np = X_train_np.astype(np.float32)
    X_test_np = transform_to_float32(preprocessor, X_test)
    X_val_np = transform_to_float32(preprocessor, X_val)

    y_train_np = y_train.values.astype(int)
    y_test_np = y_test.values.astype(int)
    y_val_np = y_val.values.astype(int)
    input_dim = X_train_np.shape[1]

    feature_list_path = os.path.join(output_dir, "mlp_features_used.csv")
    pd.DataFrame({"feature": selected_features}).to_csv(feature_list_path, index=False)

    feature_metadata_path = save_feature_set_metadata(
        output_dir=output_dir,
        feature_set=feature_set,
        selected_features=selected_features,
        df_columns=df.columns.tolist(),
    )

    preprocessor_path = os.path.join(output_dir, "mlp_preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)

    # build all hyperparameter combinations
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    tuning_rows = []
    best_score = -np.inf
    best_params = None
    best_model = None
    best_threshold = None
    best_threshold_row = None
    best_history = None
    best_epoch = None
    best_test_selection_loss = None

    print(f"Tuning {len(combos)} MLP combinations.")
    print("Early stopping uses test-selection loss; threshold selected on the test set.")
    print("Validation is kept for final evaluation.")
    print("selection_score = test_f1 - 0.25 * max(0, train_f1 - test_f1)")
    print()

    for combo_idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        model, history, best_epoch_tmp, best_test_selection_loss_tmp = train_one_mlp(
            params=params,
            X_train_np=X_train_np,
            y_train_np=y_train_np,
            X_selection_np=X_test_np,
            y_selection_np=y_test_np,
            input_dim=input_dim,
            device=device,
            random_state=random_state,
            record_history=True,
        )

        # pick threshold on the test set
        test_proba_tuning = predict_proba_mlp(model, X_test_np, device)
        threshold, threshold_df, threshold_best_row = tune_threshold_from_proba(y_test_np, test_proba_tuning)
        test_pred_tuning = (test_proba_tuning >= threshold).astype(int)

        # train-test F1 gap as an overfitting penalty
        train_proba_tmp = predict_proba_mlp(model, X_train_np, device)
        train_pred_tmp = (train_proba_tmp >= threshold).astype(int)
        train_f1_tmp = f1_score(y_train_np, train_pred_tmp, pos_label=POSITIVE_LABEL, zero_division=0)
        test_f1_tmp = f1_score(y_test_np, test_pred_tuning, pos_label=POSITIVE_LABEL, zero_division=0)

        overfit_gap = train_f1_tmp - test_f1_tmp
        selection_score = test_f1_tmp - 0.25 * max(0, overfit_gap)

        row = {
            "class_weight": str(params["class_weight"]),
            "learning_rate": params["learning_rate"],
            "dropout": params["dropout"],
            "weight_decay": params["weight_decay"],
            "best_epoch": best_epoch_tmp,
            "best_test_selection_loss": best_test_selection_loss_tmp,
            "best_threshold": threshold,
            "test_accuracy": accuracy_score(y_test_np, test_pred_tuning),
            "test_precision_dropout": precision_score(y_test_np, test_pred_tuning, pos_label=POSITIVE_LABEL, zero_division=0),
            "test_recall_dropout": recall_score(y_test_np, test_pred_tuning, pos_label=POSITIVE_LABEL, zero_division=0),
            "test_f1_dropout": test_f1_tmp,
            "train_f1_dropout_tmp": train_f1_tmp,
            "overfit_gap": overfit_gap,
            "selection_score": selection_score,
        }
        tuning_rows.append(row)

        if selection_score > best_score:
            best_score = selection_score
            best_params = copy.deepcopy(params)
            best_model = copy.deepcopy(model)
            best_threshold = threshold
            best_threshold_row = threshold_best_row
            best_history = copy.deepcopy(history)
            best_epoch = best_epoch_tmp
            best_test_selection_loss = best_test_selection_loss_tmp

        print(
            f"Tuning progress: {combo_idx}/{len(combos)} | "
            f"test_f1={test_f1_tmp:.4f} | "
            f"selection_score={selection_score:.4f} | "
            f"threshold={threshold:.2f}"
        )

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["selection_score", "test_f1_dropout", "test_recall_dropout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    tuning_path = os.path.join(output_dir, "mlp_tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print("Top 10 tuning results:")
    print(tuning_df.head(10).to_string(index=False))
    print()
    print(f"Best params              : {best_params}")
    print(f"Best threshold           : {best_threshold:.2f}")
    print(f"Best epoch               : {best_epoch}")
    print(f"Best test-selection loss : {best_test_selection_loss:.6f}")
    print(f"Best selection score     : {best_score:.4f}")
    print(f"Best test F1             : {best_threshold_row['f1_score']:.4f}")
    print()

    pd.DataFrame(best_history).to_csv(os.path.join(output_dir, "mlp_training_history.csv"), index=False)

    # save the full threshold sweep for the best model on the test set
    best_test_proba = predict_proba_mlp(best_model, X_test_np, device)
    _, best_threshold_df, _ = tune_threshold_from_proba(y_test_np, best_test_proba)
    threshold_tuning_path = os.path.join(output_dir, "mlp_threshold_tuning_test.csv")
    best_threshold_df.to_csv(threshold_tuning_path, index=False)

    train_results, train_pred, train_proba = compute_metrics_with_threshold(
        best_model, X_train_np, y_train_np, "train", best_threshold, device
    )
    test_results, test_pred, test_proba = compute_metrics_with_threshold(
        best_model, X_test_np, y_test_np, "test_selection", best_threshold, device
    )
    val_results, val_pred, val_proba = compute_metrics_with_threshold(
        best_model, X_val_np, y_val_np, "validation_final", best_threshold, device
    )

    results_df = pd.DataFrame([train_results, test_results, val_results])
    results_path = os.path.join(output_dir, "mlp_results.csv")
    results_df.to_csv(results_path, index=False)
    print("Performance results:")
    print(results_df.to_string(index=False))
    print()
    print("Final held-out validation results:")
    print(pd.DataFrame([val_results]).to_string(index=False))
    print()

    save_predictions(id_train, y_train_np, train_pred, train_proba, best_threshold,
                     os.path.join(output_dir, "mlp_train_predictions.csv"))
    save_predictions(id_test, y_test_np, test_pred, test_proba, best_threshold,
                     os.path.join(output_dir, "mlp_test_selection_predictions.csv"))
    save_predictions(id_val, y_val_np, val_pred, val_proba, best_threshold,
                     os.path.join(output_dir, "mlp_validation_final_predictions.csv"))

    # confusion matrices for test (selection) and validation (final),
    # rows = actual, cols = predicted, order [non-dropout, dropout]
    test_cm = confusion_matrix(y_test_np, test_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    val_cm = confusion_matrix(y_val_np, val_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    save_confusion_matrix(test_cm, os.path.join(output_dir, "mlp_test_confusion_matrix.csv"))
    save_confusion_matrix(val_cm, os.path.join(output_dir, "mlp_validation_confusion_matrix.csv"))

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

    model_path = os.path.join(output_dir, "mlp_model.pt")
    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": best_params["dropout"],
        "selected_features": selected_features,
        "feature_set": feature_set,
        "best_params": best_params,
        "best_threshold": best_threshold,
        "label_mapping": {
            "0": "non-dropout",
            "1": "dropout",
        },
    }
    torch.save(checkpoint, model_path)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "feature_set": feature_set,
        "n_features": len(selected_features),
        "n_features_after_preprocessing": input_dim,
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
            "important_note": "Split 70/20/10 stratified by label. Each row is an independent observation; no grouping by student_id. Same split logic and random_state as the RF script.",
            "split_usage": {
                "train": "model fitting",
                "test": "hyperparameter selection, early stopping, and threshold selection",
                "validation": "final held-out evaluation",
            },
        },
        "model": "DropoutMLP",
        "architecture": {
            "input_dim": input_dim,
            "hidden_dims": list(HIDDEN_DIMS),
            "output_dim": 2,
            "activation": "ReLU",
            "normalization": "BatchNorm1d",
            "dropout": best_params["dropout"],
        },
        "preprocessing": {
            "policy": "Numeric scaled, *_flag kept as 0/1, seniority/categorical one-hot encoded.",
            "numeric": "SimpleImputer(strategy='median') + StandardScaler fitted on train only",
            "binary_flags": "SimpleImputer(strategy='most_frequent'); no scaling, kept as 0/1",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore') fitted on train only",
            "numeric_columns": numeric_cols,
            "binary_flag_columns": flag_cols,
            "categorical_columns": categorical_cols,
            "onehot_categorical_policy": ONEHOT_CATEGORICAL_FEATURES,
            "preprocessor_file": preprocessor_path,
        },
        "training": {
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "batch_size": BATCH_SIZE,
            "epochs_max": EPOCHS,
            "early_stopping_patience": PATIENCE,
            "early_stopping_split": "test",
            "early_stopping_metric": "test_selection_loss",
            "best_epoch": best_epoch,
            "best_test_selection_loss": best_test_selection_loss,
            "device": str(device),
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
        "metrics_used": [
            "accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
        ],
        "results": {
            "train": train_results,
            "test_selection": test_results,
            "validation_final": val_results,
        },
        "decision_rule": f"Predict dropout if prob_dropout >= {best_threshold:.2f}",
    }
    summary_path = os.path.join(output_dir, "mlp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output saved to folder: {output_dir}")
    print(f"Model checkpoint saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Results CSV: {results_path}")
    print(f"Tuning CSV : {tuning_path}")
    print(f"Threshold tuning CSV: {threshold_tuning_path}")
    print(f"Features CSV: {feature_list_path}")
    print(f"Feature metadata JSON: {feature_metadata_path}")
    print(f"Summary JSON: {summary_path}")

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

    run_single_mlp_experiment(
        df=df,
        csv_path=csv_path,
        output_dir=output_dir,
        feature_set=feature_set,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
