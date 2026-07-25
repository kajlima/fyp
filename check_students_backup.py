from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = Path("zenodo.csv")
OUTPUT_DIR = Path("zenodo_attack_recovery_students_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EXCEL = OUTPUT_DIR / "zenodo_full_successfully_attacked_and_recovered_students.xlsx"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "zenodo_full_attack_recovery_summary.csv"
OUTPUT_JSON = OUTPUT_DIR / "zenodo_full_attack_recovery_file_map.json"

TARGET_COL = "target"
STUDENT_COL = "student_id"
POSITIVE_LABEL = 1  # dropout
SPLITS = ("test", "validation")

# Real FULL feature set (33 columns)
FULL_FEATURES = [
    "degree_size", "seniority", "highest_course_year_enrolled", "adapted_studies_flag",
    "credits_enrolled_semester_a", "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year", 
    "completion_rate_one_year_before", "completion_rate_two_years_before",
    "completion_rate_three_years_before", "completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag", "completion_rate_three_years_before_missing_flag",
    "pass_ratio_sem_a", "pass_ratio_sem_a_and_zero_flag",
    "lms_events_sem_a", "lms_assignment_submissions_sem_a", "lms_test_submissions_sem_a",
    "lms_total_minutes_sem_a", "attendance_days_sem_a",
    "n_courses", "courses_mean", "total_credits_passed_academic_year",
    "pass_ratio_sem_b", "pass_ratio_sem_b_and_zero_flag", "total_performance",
    "total_performance_and_zero_flag", "obtained_new_degree_flag",
    "lms_events_sem_b", "lms_assignment_submissions_sem_b", "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_b", "attendance_days_sem_b",
]

# Real attackable feature set (21 columns)
ATTACKABLE_FEATURES = [
    "credits_enrolled_semester_a", "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",
    "completion_rate_one_year_before", "completion_rate_two_years_before",
    "completion_rate_three_years_before", "courses_mean",
    "total_credits_passed_academic_year", "pass_ratio_sem_a", "pass_ratio_sem_b",
    "total_performance",
    "lms_events_sem_a", "lms_assignment_submissions_sem_a", "lms_test_submissions_sem_a",
    "lms_total_minutes_sem_a", "attendance_days_sem_a",
    "lms_events_sem_b", "lms_assignment_submissions_sem_b", "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_b", "attendance_days_sem_b",
]

# Output folders produced by the attack / defence scripts.
RF_ATTACK_DIR = "fgsm_rf_full_results"
RF_DEFENCE_DIR = "rf_adversarial_training_full_results"
MLP_ATTACK_DIR = "pgd_mlp_full_results"
MLP_DEFENCE_DIR = "mlp_adversarial_training_full_results"

# ============================================================
# EPSILONS USED IN THE FINAL RUNS
# ============================================================
RF_ATTACK_EPS = 1.5
RF_DEFENCE_TRAIN_EPS = 1.5
RF_DEFENCE_EVAL_EPS = 1.5

MLP_ATTACK_EPS = 1.0
MLP_DEFENCE_TRAIN_EPS = 1.0
MLP_DEFENCE_EVAL_EPS = 1.0


def eps_tag(value: float) -> str:
    """1.5 -> '1p5'. Same tag the attack and defence scripts write."""
    return str(float(value)).replace(".", "p")


# Per-model file paths. {split} is filled in; epsilons come from the constants
MODEL_FILES = {
    "Random Forest": {
        "attack": "FGSM",
        "attack_pred": [f"{RF_ATTACK_DIR}/rf_fgsm_{{split}}_predictions_eps_{eps_tag(RF_ATTACK_EPS)}.csv"],
        "attack_raw": [f"{RF_ATTACK_DIR}/X_{{split}}_fgsm_raw_eps_{eps_tag(RF_ATTACK_EPS)}.csv"],
        "defence_pred": [
            f"{RF_DEFENCE_DIR}/rf_defended_fgsm_{{split}}_predictions"
            f"_train_eps_{eps_tag(RF_DEFENCE_TRAIN_EPS)}_eval_eps_{eps_tag(RF_DEFENCE_EVAL_EPS)}.csv"
        ],
        "clean_prefix": "rf",
    },
    "MLP": {
        "attack": "PGD",
        "attack_pred": [f"{MLP_ATTACK_DIR}/mlp_pgd_{{split}}_predictions_eps_{eps_tag(MLP_ATTACK_EPS)}.csv"],
        "attack_raw": [f"{MLP_ATTACK_DIR}/X_{{split}}_pgd_raw_eps_{eps_tag(MLP_ATTACK_EPS)}.csv"],
        "defence_pred": [
            f"{MLP_DEFENCE_DIR}/mlp_defended_pgd_{{split}}_predictions"
            f"_train_eps_{eps_tag(MLP_DEFENCE_TRAIN_EPS)}_eval_eps_{eps_tag(MLP_DEFENCE_EVAL_EPS)}.csv"
        ],
        "clean_prefix": "mlp",
    },
}


# ============================================================
# HELPERS
# ============================================================

def label_text(value) -> str:
    try:
        return "Dropout" if int(value) == POSITIVE_LABEL else "Non-dropout"
    except Exception:
        return "Unknown"


def eps_from_name(path: str) -> Dict[str, Optional[float]]:
    """Pull eval / train epsilon out of a filename tag like eps_1p5 or eval_eps_1p0."""
    import re
    out = {"epsilon": None, "train_epsilon": None, "eval_epsilon": None}
    for key, pat in [("train_epsilon", r"train_eps_([0-9]+p[0-9]+)"),
                     ("eval_epsilon", r"eval_eps_([0-9]+p[0-9]+)"),
                     ("epsilon", r"(?<!train_)(?<!eval_)eps_([0-9]+p[0-9]+)")]:
        m = re.search(pat, path)
        if m:
            out[key] = float(m.group(1).replace("p", "."))
    return out


def resolve_one(patterns: List[str], split: str) -> Optional[Path]:
    
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat.format(split=split)))
    hits = sorted(set(hits))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            "More than one file matches a pinned epsilon:\n  "
            + "\n  ".join(hits)
            + "\nRemove or archive the duplicates before running."
        )
    return Path(hits[0])


def normalize_target(series: pd.Series) -> pd.Series:
    norm = (series.astype(str).str.strip().str.lower()
            .replace({"non dropout": "non-dropout", "non_dropout": "non-dropout", "drop out": "dropout"}))
    mapping = {"non-dropout": 0, "dropout": 1, "b": 0, "a": 1, "0": 0, "1": 1}
    y = norm.map(mapping)
    if y.isna().any():
        bad = series.loc[y.isna()].unique().tolist()
        raise ValueError(f"Target values could not be mapped: {bad}")
    return y.astype(int)


def _first_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None


# ============================================================
# READ + STANDARDIZE PREDICTION FILES (keyed by student_id)
# ============================================================

def read_attack_predictions(path: Path, clean_prefix: str) -> pd.DataFrame:
    """Undefended attack file -> tidy per-student frame."""
    df = pd.read_csv(path)
    clean_pred = _first_col(df, [f"{clean_prefix}_clean_pred_label", "clean_pred_label"])
    adv_pred = _first_col(df, [f"{clean_prefix}_adv_pred_label", "adv_pred_label"])
    clean_prob = _first_col(df, [f"{clean_prefix}_clean_prob_dropout", "clean_prob_dropout"])
    adv_prob = _first_col(df, [f"{clean_prefix}_adv_prob_dropout", "adv_prob_dropout"])
    if STUDENT_COL not in df.columns or clean_pred is None or adv_pred is None:
        raise KeyError(f"{path} is missing student_id / clean / adv prediction columns.")
    out = pd.DataFrame({
        STUDENT_COL: df[STUDENT_COL].values,
        "actual_label": df["actual_label"].astype(int).values,
        "clean_pred_label": df[clean_pred].astype(int).values,
        "attack_adv_pred_label": df[adv_pred].astype(int).values,
    })
    if clean_prob:
        out["clean_prob_dropout"] = pd.to_numeric(df[clean_prob], errors="coerce").values
    if adv_prob:
        out["attack_adv_prob_dropout"] = pd.to_numeric(df[adv_prob], errors="coerce").values
    return out


def read_defence_predictions(path: Path) -> pd.DataFrame:
    """
    The two defence scripts use different schemas:
      * MLP writes both clean and adversarial columns in one row
        (clean_pred_label / adv_pred_label, clean_prob_dropout / adv_prob_dropout).
      * RF writes a single defended prediction per file
        (predicted_label / prob_dropout); because this file is the "Defended FGSM"
        run, predicted_label IS the defended prediction under attack.
    """
    df = pd.read_csv(path)
    if STUDENT_COL not in df.columns:
        raise KeyError(f"{path} has no '{STUDENT_COL}' column.")

    adv_pred_col = _first_col(df, ["adv_pred_label", "predicted_label", "y_pred", "prediction"])
    if adv_pred_col is None:
        raise KeyError(f"{path} has no defended prediction column "
                       f"(looked for adv_pred_label / predicted_label). Columns: {list(df.columns)}")
    adv_prob_col = _first_col(df, ["adv_prob_dropout", "prob_dropout", "predicted_prob_dropout"])
    clean_pred_col = _first_col(df, ["clean_pred_label"])
    clean_prob_col = _first_col(df, ["clean_prob_dropout"])

    out = pd.DataFrame({
        STUDENT_COL: df[STUDENT_COL].values,
        "defended_adv_pred_label": df[adv_pred_col].astype(int).values,
    })
    out["defended_clean_pred_label"] = (
        df[clean_pred_col].astype("Int64").values if clean_pred_col else pd.array([pd.NA] * len(df), dtype="Int64"))
    if adv_prob_col:
        out["defended_adv_prob_dropout"] = pd.to_numeric(df[adv_prob_col], errors="coerce").values
    if clean_prob_col:
        out["defended_clean_prob_dropout"] = pd.to_numeric(df[clean_prob_col], errors="coerce").values
    return out


# ============================================================
# ADVERSARIAL RAW FEATURE DIFFERENCES
# ============================================================

def attach_adv_diffs(table: pd.DataFrame, attack_pred_path: Optional[Path],
                     raw_adv_path: Optional[Path], clean_by_id: Optional[pd.DataFrame]) -> pd.DataFrame:
    
    if attack_pred_path is None or raw_adv_path is None or clean_by_id is None or table.empty:
        return table
    pred_ids = pd.read_csv(attack_pred_path)[STUDENT_COL].values
    raw = pd.read_csv(raw_adv_path)
    if len(pred_ids) != len(raw):
        print(f"  Skipped adv diffs: row count mismatch "
              f"({len(pred_ids)} ids vs {len(raw)} raw rows) for {raw_adv_path.name}.")
        return table
    raw = raw.copy()
    raw[STUDENT_COL] = pred_ids
    attackable = [c for c in ATTACKABLE_FEATURES if c in raw.columns and c in clean_by_id.columns]
    adv = raw[[STUDENT_COL] + attackable].rename(columns={c: f"adv_{c}" for c in attackable})
    merged = table.merge(adv, on=STUDENT_COL, how="left")
    for c in attackable:
        clean_map = clean_by_id.set_index(STUDENT_COL)[c]
        merged[f"diff_{c}"] = merged[f"adv_{c}"] - merged[STUDENT_COL].map(clean_map)
    return merged


# ============================================================
# BUILD ONE MODEL x SPLIT TABLE
# ============================================================

def build_model_split(model: str, split: str, clean_by_id: pd.DataFrame,
                      file_map: dict) -> Tuple[Optional[pd.DataFrame], dict]:
    spec = MODEL_FILES[model]
    stats = {"model": model, "split": split, "attack": spec["attack"]}

    attack_pred_path = resolve_one(spec["attack_pred"], split)
    defence_pred_path = resolve_one(spec["defence_pred"], split)
    raw_adv_path = resolve_one(spec["attack_raw"], split)

    file_map[f"{model}|{split}"] = {
        "attack_predictions": str(attack_pred_path) if attack_pred_path else None,
        "defence_predictions": str(defence_pred_path) if defence_pred_path else None,
        "attack_raw_features": str(raw_adv_path) if raw_adv_path else None,
    }
    if attack_pred_path is None or defence_pred_path is None:
        print(f"  {model} / {split}: missing attack or defence prediction file, skipped.")
        return None, stats

    atk = read_attack_predictions(attack_pred_path, spec["clean_prefix"])
    dfn = read_defence_predictions(defence_pred_path)
    stats.update({k: v for k, v in eps_from_name(attack_pred_path.name).items() if v is not None})
    stats.update({f"defence_{k}": v for k, v in eps_from_name(defence_pred_path.name).items() if v is not None})

    # successfully attacked = real dropout, caught when clean, hidden under attack
    attacked = atk[(atk["actual_label"] == 1) &
                   (atk["clean_pred_label"] == 1) &
                   (atk["attack_adv_pred_label"] == 0)].copy()

    table = attacked.merge(dfn, on=STUDENT_COL, how="left")
    table["successfully_attacked"] = True
    table["successfully_recovered"] = (table["defended_adv_pred_label"] == 1)

    # readable label columns
    table["actual_text"] = table["actual_label"].map(label_text)
    table["clean_pred_text"] = table["clean_pred_label"].map(label_text)
    table["attack_adv_pred_text"] = table["attack_adv_pred_label"].map(label_text)
    table["defended_adv_pred_text"] = table["defended_adv_pred_label"].map(label_text)

    # clean raw FULL features by student_id
    clean_cols = [STUDENT_COL] + [c for c in FULL_FEATURES if c in clean_by_id.columns]
    clean_small = clean_by_id[clean_cols].rename(
        columns={c: f"clean_{c}" for c in FULL_FEATURES if c in clean_by_id.columns})
    table = table.merge(clean_small, on=STUDENT_COL, how="left")

    # adversarial raw diffs (attackable features only)
    table = attach_adv_diffs(table, attack_pred_path, raw_adv_path, clean_by_id)

    n_attacked = len(table)
    n_recovered = int(table["successfully_recovered"].sum())
    stats["successfully_attacked_count"] = n_attacked
    stats["successfully_recovered_count"] = n_recovered
    stats["recovery_rate"] = (n_recovered / n_attacked) if n_attacked else np.nan
    print(f"  {model} / {split}: attacked={n_attacked}, recovered={n_recovered} "
          f"({stats['recovery_rate']:.1%})" if n_attacked else
          f"  {model} / {split}: no successfully-attacked students.")

    # tidy column order
    lead = [STUDENT_COL, "actual_text", "clean_pred_text", "attack_adv_pred_text",
            "defended_adv_pred_text", "successfully_attacked", "successfully_recovered"]
    prob_cols = [c for c in ["clean_prob_dropout", "attack_adv_prob_dropout",
                             "defended_adv_prob_dropout"] if c in table.columns]
    diff_cols = [c for c in table.columns if c.startswith("diff_")]
    other = [c for c in table.columns if c not in lead + prob_cols + diff_cols]
    table = table[[c for c in lead if c in table.columns] + prob_cols + diff_cols +
                  [c for c in other if c not in lead + prob_cols + diff_cols]]
    return table, stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("ZENODO FULL — successfully attacked & recovered students")
    print("=" * 80)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    raw = pd.read_csv(CSV_PATH)
    if STUDENT_COL not in raw.columns:
        raise KeyError(f"Dataset has no '{STUDENT_COL}' column; cannot join predictions.")
    raw["actual_label"] = normalize_target(raw[TARGET_COL])
    clean_by_id = raw.drop_duplicates(subset=STUDENT_COL).reset_index(drop=True)

    file_map: dict = {}
    all_stats: List[dict] = []
    sheets: Dict[str, pd.DataFrame] = {}

    for model in MODEL_FILES:
        for split in SPLITS:
            table, stats = build_model_split(model, split, clean_by_id, file_map)
            all_stats.append(stats)
            if table is not None and not table.empty:
                sheet = f"{'RF' if model == 'Random Forest' else 'MLP'}_{split}"[:31]
                sheets[sheet] = table

    if not sheets:
        raise RuntimeError("No attacked/recovered tables were built. Check the result folders.")

    summary = pd.DataFrame(all_stats)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\nSaved summary: {OUTPUT_SUMMARY_CSV}")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    print(f"Saved workbook: {OUTPUT_EXCEL}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(file_map, f, indent=2)
    print(f"Saved file map: {OUTPUT_JSON}")

    print("\nSummary:")
    show = [c for c in ["model", "split", "epsilon", "successfully_attacked_count",
                        "successfully_recovered_count", "recovery_rate"] if c in summary.columns]
    print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
